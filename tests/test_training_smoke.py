# ============================================================
# 📄 파일명: ai_modules/tests/test_training_smoke.py
# 📘 목적: 최소 스모크 테스트임. 모델 포워드가 동작하는지 확인함.
# ============================================================

from __future__ import annotations
import torch
from ai_modules.src.models.simple_cnn import SimpleCNN

def test_forward_shape():
    """임의 텐서로 포워드가 동작하고 출력 shape이 맞는지 확인함."""
    num_classes = 7
    model = SimpleCNN(num_classes=num_classes)
    x = torch.randn(2, 3, 128, 128)   # 배치=2
    out = model(x)
    assert out.shape == (2, num_classes)
