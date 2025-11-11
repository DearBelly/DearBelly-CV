# ============================================================
# 📄 파일명: ai_modules/src/models/backbone_timm.py
# 📘 목적: timm 백본 + ArcFace/FC 헤드 모델 정의
# ============================================================
from __future__ import annotations
import torch
import torch.nn as nn
import timm
from ai_modules.src.models.arc_margin import ArcMarginProduct

class BackboneWithHead(nn.Module):
    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True, use_arcface: bool = False, dropout: float = 0.2):
        super().__init__()
        self.use_arcface = use_arcface
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='avg')
        feat_dim = self.backbone.num_features
        self.bn = nn.BatchNorm1d(feat_dim)
        self.dp = nn.Dropout(dropout)
        if use_arcface:
            self.arc = ArcMarginProduct(feat_dim, num_classes, s=30.0, m=0.3)
        else:
            self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
        feats = self.backbone(x)
        feats = self.bn(feats)
        feats = self.dp(feats)
        if self.use_arcface:
            if target is None:
                raise ValueError("ArcFace 사용 시 forward에 target(label)을 반드시 전달해야 함")
            return self.arc(feats, target)
        return self.fc(feats)
