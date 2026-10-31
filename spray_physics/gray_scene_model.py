"""三帧背景参考分层模型：把透过的打光背景替换为灰底，保留原图细节。"""

import torch
from torch import nn
from torch.nn import functional as F

from restore_model import ReferenceRestorer

VERSION = "5.0.0-alpha1"


class TemporalGrayRestorer(ReferenceRestorer):
    """复用本项目NAF恢复骨干；输出透明度、前景亮度和有限背景漂移。"""
    def __init__(self, width=16, depth=3, correction_limit=0.04):
        super().__init__(width, depth, correction_limit)
        self.intro = nn.Conv2d(9, width, 3, padding=1)
        self.ending = nn.Conv2d(width, 3, 3, padding=1)
        nn.init.zeros_(self.ending.weight)
        nn.init.zeros_(self.ending.bias)
        with torch.no_grad():
            self.ending.bias[0] = -3

    def decompose(self, x, project=False):
        z = self.ending(self.features(x))
        alpha, appearance = z[:, :1].sigmoid(), z[:, 1:2].sigmoid()
        offset = z[:, 2:3].tanh()*self.spec["correction_limit"]
        background = (x[:, 3:4]+offset).clamp(0, 1)
        transmission = 1-alpha
        if project:
            # I=T·B+S，且0<=S<=1-T；噪声容差避免把细小噪声强制当成喷雾。
            eps = 3/255
            tmax = torch.minimum((x[:, 1:2]+eps)/(background+eps),
                                 (1-x[:, 1:2]+eps)/(1-background+eps)).clamp(0, 1)
            transmission = torch.minimum(transmission, tmax)
        gray = x[:, 1:2]-transmission*(background-x[:, 4:5])
        reconstruction = (1-alpha)*background+alpha*appearance
        return {"alpha": alpha, "transmission": transmission, "appearance": appearance,
                "offset": offset, "gray": gray, "reconstruction": reconstruction}

    def forward(self, x):
        return self.decompose(x)["gray"]


def gray_scene_loss(z, target, x):
    fg = target["alpha"] > 0.05

    def balanced(error):
        return (error*fg).sum()/fg.sum().clamp_min(1)+(error*~fg).sum()/(~fg).sum().clamp_min(1)

    opacity = balanced((z["alpha"]-target["alpha"]).abs())
    gray = balanced((z["gray"]-target["gray"]).abs())
    reconstruction = F.l1_loss(z["reconstruction"], x[:, 1:2])
    offset = F.l1_loss(z["offset"], target["offset"])
    gradient = (F.l1_loss(z["gray"][..., 1:, :]-z["gray"][..., :-1, :],
                          target["gray"][..., 1:, :]-target["gray"][..., :-1, :])+
                F.l1_loss(z["gray"][..., :, 1:]-z["gray"][..., :, :-1],
                          target["gray"][..., :, 1:]-target["gray"][..., :, :-1]))
    return opacity+3*gray+reconstruction+offset*0.5+gradient*0.2
