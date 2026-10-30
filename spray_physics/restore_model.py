"""基于NAFNet结构改写的双图背景扰动恢复器；先恢复图像，再交给边界模块。

结构来源：https://github.com/megvii-research/NAFNet
原始结构版权：(c) 2022 megvii-model，MIT，见licenses/NAFNet-MIT.txt。
本项目改为四通道参考输入、有符号背景扰动输出，未引入BasicSR实现。
"""

import torch
from torch import nn
from torch.nn import functional as F

RESTORE_VERSION = "4.1.0-alpha1"


class ChannelNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        return F.layer_norm(x.permute(0, 2, 3, 1), (x.shape[1],), self.weight, self.bias, 1e-6).permute(0, 3, 1, 2)


class Gate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, 1)
        return a*b


class RestorationBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm1, self.norm2 = ChannelNorm(width), ChannelNorm(width)
        self.spatial = nn.Sequential(nn.Conv2d(width, width*2, 1),
            nn.Conv2d(width*2, width*2, 3, padding=1, groups=width*2), Gate())
        self.attention = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(width, width, 1))
        self.project = nn.Conv2d(width, width, 1)
        self.ffn = nn.Sequential(nn.Conv2d(width, width*2, 1), Gate(), nn.Conv2d(width, width, 1))
        self.beta = nn.Parameter(torch.zeros(1, width, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, width, 1, 1))

    def forward(self, x):
        z = self.spatial(self.norm1(x))
        y = x+self.project(z*self.attention(z))*self.beta
        return y+self.ffn(self.norm2(y))*self.gamma


class ReferenceRestorer(nn.Module):
    """网络预测差分中属于背景扰动的部分；输出不经过喷雾掩膜相乘。"""
    def __init__(self, width=16, depth=3, correction_limit=0.15):
        super().__init__()
        if width < 4 or not 1 <= depth <= 4 or correction_limit <= 0:
            raise ValueError("恢复模型的宽度、深度或修正限值无效。")
        self.spec = {"width": width, "depth": depth, "correction_limit": correction_limit}
        self.intro = nn.Conv2d(4, width, 3, padding=1)
        self.encoders, self.downs, self.ups, self.decoders = [nn.ModuleList() for _ in range(4)]
        ch = width
        for _ in range(depth):
            self.encoders.append(RestorationBlock(ch))
            self.downs.append(nn.Conv2d(ch, ch*2, 2, stride=2)); ch *= 2
        self.middle = nn.Sequential(RestorationBlock(ch), RestorationBlock(ch))
        for _ in range(depth):
            self.ups.append(nn.Sequential(nn.Conv2d(ch, ch*2, 1), nn.PixelShuffle(2)))
            ch //= 2
            self.decoders.append(RestorationBlock(ch))
        self.ending = nn.Conv2d(width, 1, 3, padding=1)
        nn.init.zeros_(self.ending.weight); nn.init.zeros_(self.ending.bias)

    def features(self, x):
        h, w = x.shape[-2:]
        factor = 2**self.spec["depth"]
        padded = F.pad(x, (0, (-w) % factor, 0, (-h) % factor), mode="replicate")
        z, skips = self.intro(padded), []
        for block, down in zip(self.encoders, self.downs):
            z = block(z); skips.append(z); z = down(z)
        z = self.middle(z)
        for up, block, skip in zip(self.ups, self.decoders, reversed(skips)):
            z = block(up(z)+skip)
        return z[..., :h, :w]

    def forward(self, x):
        nuisance = self.ending(self.features(x)).tanh()*self.spec["correction_limit"]
        restored = x[:, :1]-x[:, 1:2]-nuisance
        return restored, nuisance


class OpticalRestorer(ReferenceRestorer):
    """按I=(1-alpha)B+alpha F分层，前景层不受差分必为零的限制。"""
    def __init__(self, width=16, depth=3, correction_limit=0.15, difference_gain=8.0, physical_projection=False):
        super().__init__(width, depth, correction_limit)
        self.spec["difference_gain"] = difference_gain
        self.spec["physical_projection"] = physical_projection
        self.ending = nn.Conv2d(width, 3, 3, padding=1)
        nn.init.zeros_(self.ending.weight); nn.init.zeros_(self.ending.bias)
        with torch.no_grad():
            self.ending.bias[0] = -3

    def decompose(self, x):
        scaled = torch.cat([x[:, :2], x[:, 2:]*self.spec["difference_gain"]], 1)
        z = self.ending(self.features(scaled))
        alpha = z[:, :1].sigmoid()
        foreground = alpha*z[:, 1:2].sigmoid()
        offset = z[:, 2:3].tanh()*self.spec["correction_limit"]
        background = (x[:, 1:2]+offset).clamp(0, 1)
        background_contribution = (1-alpha)*background
        # 由测得的图像约束背景贡献，保留清晰的正变化与原图纹理。
        projected_background = torch.minimum(background_contribution, x[:, :1])
        projected_foreground = (x[:, :1]-projected_background).clamp(0, 1)
        return {"foreground": foreground, "alpha": alpha, "alpha_logits": z[:, :1],
                "offset": offset, "background_contribution": background_contribution,
                "reconstruction": foreground+background_contribution,
                "projected_foreground": projected_foreground, "projected_background": projected_background}

    def forward(self, x):
        layers = self.decompose(x)
        if self.spec["physical_projection"]:
            return layers["projected_foreground"], layers["projected_background"]-x[:, 1:2]
        return layers["foreground"], layers["background_contribution"]-x[:, 1:2]


def optical_loss(layers, target, inputs):
    foreground = target["alpha"] > 0.05
    def balanced(error):
        a = (error*foreground).sum()/foreground.sum().clamp_min(1)
        b = (error*~foreground).sum()/(~foreground).sum().clamp_min(1)
        return a+b
    alpha = F.binary_cross_entropy_with_logits(layers["alpha_logits"], target["alpha"], reduction="none")
    appearance = balanced((layers["foreground"]-target["foreground"]).abs())
    alpha_loss = balanced(alpha)
    reconstruction = F.l1_loss(layers["reconstruction"], inputs[:, :1])
    background = F.l1_loss(layers["offset"], target["background_offset"])
    return appearance*2+alpha_loss*0.5+reconstruction+background*0.5


def restoration_loss(restored, target, semantic):
    # 背景残留和前景失真同时计入；整幅抹黑无法通过前景损失。
    mask = semantic[:, :1]
    distance = ((restored-target).square()+1e-6).sqrt()-0.001
    photometric = (distance*(1-mask)).sum()/(1-mask).sum().clamp_min(1)
    photometric = photometric+2*(distance*mask).sum()/mask.sum().clamp_min(1)
    gradient = (F.l1_loss(restored[..., 1:, :]-restored[..., :-1, :], target[..., 1:, :]-target[..., :-1, :])+
                F.l1_loss(restored[..., :, 1:]-restored[..., :, :-1], target[..., :, 1:]-target[..., :, :-1]))
    return photometric+0.25*gradient
