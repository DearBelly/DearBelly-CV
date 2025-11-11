# models/efficientnet_baseline.py
# ------------------------------------------------------------
# 목적: timm의 EfficientNet 백본을 사용한 분류 베이스라인 모델 정의 파일임.
# 특징:
#   - 사전학습(pretrained) 가중치 사용 가능
#   - 전이학습을 위한 백본 고정(freeze) 옵션 제공
#   - 분류기 헤드(BN + Dropout + Linear)로 단순/안정 구성
# 사용 예:
#   from models.efficientnet_baseline import EfficientNetBaseline
#   model = EfficientNetBaseline(num_classes=492, model_name="efficientnet_b3", pretrained=True)
# ------------------------------------------------------------

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import timm


class EfficientNetBaseline(nn.Module):
    """
    EfficientNetBaseline 모델 정의 클래스임.

    구성:
      - backbone: timm.create_model(model_name, pretrained, num_classes=0, global_pool="avg")
      - head: (BatchNorm1d → Dropout → Linear)

    Args:
        num_classes (int): 분류 클래스 수임.
        model_name (str): timm 모델 이름임. 기본값 "efficientnet_b3"임.
        pretrained (bool): 사전학습 가중치 사용 여부임.
        dropout (float): 분류기 헤드 드롭아웃 비율임.
        freeze_backbone (bool): 백본 파라미터를 고정해 전이학습 초기에 안정화할지 여부임.

    Methods:
        freeze_backbone_params(): 백본 가중치를 고정함.
        unfreeze_backbone_params(): 백본 가중치를 해제함.
        feature_dim: 백본 피처 차원을 반환함 (프로퍼티).
    """

    def __init__(
        self,
        num_classes: int,
        model_name: str = "efficientnet_b3",
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        # 1) 백본 생성: 분류기 제거(num_classes=0), 글로벌 풀 avg 고정임
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        # 2) 피처 차원 확인
        feat_dim = getattr(self.backbone, "num_features", None)
        if feat_dim is None:
            # timm 백본에 따라 속성명이 다를 가능성을 방어함
            # common fallback
            if hasattr(self.backbone, "classifier") and hasattr(self.backbone.classifier, "in_features"):
                feat_dim = self.backbone.classifier.in_features
            else:
                raise RuntimeError("백본의 특성 차원을 확인할 수 없음임. timm 백본 정의를 확인 바람.")

        # 3) 분류 헤드 구성(BN → Dropout → Linear)
        self.bn = nn.BatchNorm1d(feat_dim)
        self.dp = nn.Dropout(dropout)
        self.fc = nn.Linear(feat_dim, num_classes)

        # 4) 초기화 옵션: 백본 고정
        if freeze_backbone:
            self.freeze_backbone_params()

    @property
    def feature_dim(self) -> int:
        """백본 피처 차원을 반환함."""
        return self.bn.num_features

    @torch.no_grad()
    def freeze_backbone_params(self) -> None:
        """백본 파라미터를 고정함."""
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def unfreeze_backbone_params(self) -> None:
        """백본 파라미터 고정을 해제함."""
        for p in self.backbone.parameters():
            p.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파 정의임.

        입력:
            x: (B, C, H, W)

        출력:
            logits: (B, num_classes)
        """
        feats = self.backbone(x)         # (B, feat_dim)
        feats = self.bn(feats)
        feats = self.dp(feats)
        logits = self.fc(feats)
        return logits


def build_efficientnet_baseline(
    num_classes: int,
    model_name: str = "efficientnet_b3",
    pretrained: bool = True,
    dropout: float = 0.2,
    freeze_backbone: bool = False,
) -> EfficientNetBaseline:
    """
    팩토리 함수임. 설정값으로 모델 인스턴스를 생성해 반환함.
    """
    return EfficientNetBaseline(
        num_classes=num_classes,
        model_name=model_name,
        pretrained=pretrained,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    )
