"""背景条件轻量 U-Net 与共享参数细化；不依赖人像预训练权重。"""

import torch
from torch import nn
from torch.nn import functional as F

DEEP_VERSION = "3.0.0-alpha1"


class Block(nn.Module):
    def __init__(self, incoming, outgoing):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(4, outgoing), nn.SiLU(),
            nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(4, outgoing), nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class BackgroundUNet(nn.Module):
    """四通道输入；输出喷雾、边缘及不可见性提示的未归一化值。"""
    def __init__(self, width=16, iterations=2):
        super().__init__()
        if width < 4 or width % 4 or not 0 <= iterations <= 4:
            raise ValueError("通道宽度须为4的倍数，细化次数应在0至4之间。")
        self.spec = {"width": width, "iterations": iterations}
        self.down1 = Block(4, width)
        self.down2 = Block(width, width * 2)
        self.down3 = Block(width * 2, width * 4)
        self.bridge = Block(width * 4, width * 8)
        self.up3 = Block(width * 12, width * 4)
        self.up2 = Block(width * 6, width * 2)
        self.up1 = Block(width * 3, width)
        self.head = nn.Conv2d(width, 3, 1)
        self.refiner = nn.Sequential(Block(7, width), nn.Conv2d(width, 3, 1))

    def forward(self, x, iterations=None):
        if x.ndim != 4 or x.shape[1] != 4 or min(x.shape[2:]) < 16:
            raise ValueError("输入必须为N×4×H×W，且高宽至少16像素。")
        n = self.spec["iterations"] if iterations is None else iterations
        if not 0 <= n <= 4:
            raise ValueError("细化次数应在0至4之间。")
        a = self.down1(x)
        b = self.down2(F.avg_pool2d(a, 2))
        c = self.down3(F.avg_pool2d(b, 2))
        d = self.bridge(F.avg_pool2d(c, 2))
        def up(v, skip, block):
            return block(torch.cat([F.interpolate(v, size=skip.shape[-2:], mode="bilinear", align_corners=False), skip], 1))
        z = self.head(up(up(up(d, c, self.up3), b, self.up2), a, self.up1))
        outputs = [z]
        for _ in range(n):
            # 原图与背景始终可见，避免重复处理已经损坏的中间图。
            z = z + self.refiner(torch.cat([x, z.sigmoid()], 1)) * 0.5
            outputs.append(z)
        return outputs


def supervised_loss(outputs, targets, valid=None):
    """未知标注通过valid屏蔽；不可见性头仅作为未校准的提示。"""
    if valid is None:
        valid = torch.ones_like(targets[:, :1])
    total = targets.sum() * 0
    for i, z in enumerate(outputs):
        p = z[:, :1].sigmoid()
        target = targets[:, :1]
        ce = F.binary_cross_entropy_with_logits(z[:, :1], target, reduction="none")
        ce = (ce * valid).sum() / valid.sum().clamp_min(1)
        inter = (p * target * valid).sum((1, 2, 3))
        dice = 1 - ((2 * inter + 1) / ((p * valid).sum((1, 2, 3)) + (target * valid).sum((1, 2, 3)) + 1))
        aux = F.binary_cross_entropy_with_logits(z[:, 1:], targets[:, 1:], reduction="none")
        aux = (aux * valid).sum() / (2 * valid.sum().clamp_min(1))
        total = total + (ce + dice.mean() + 0.5 * aux) * (1 if i == len(outputs)-1 else 0.25)
    return total
