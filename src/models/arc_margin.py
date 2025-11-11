# ============================================================
# 📄 파일명: ai_modules/src/models/arc_margin.py
# 📘 목적: ArcFace 분류 헤드 정의
# ============================================================
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.3, easy_margin: bool = False):
        super().__init__()
        self.s, self.m = s, m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt(torch.clamp(1.0 - cosine**2, 0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s
