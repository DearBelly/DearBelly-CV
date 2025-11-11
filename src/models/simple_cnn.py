# ============================================================
# 📄 파일명: simple_cnn.py
# 📁 위치: ai_modules/src/models/simple_cnn.py
# 📘 목적:
#   - 128×128 입력을 가정한 기본 CNN 분류 모델 정의임.
#   - 구조: Conv(3→16) → ReLU → MaxPool → Conv(16→32) → ReLU → MaxPool → FC → FC
#
# 🧪 사용 예시:
#   from ai_modules.src.models.simple_cnn import SimpleCNN
#   model = SimpleCNN(num_classes=492)
#
# ✅ 특징:
#   - 학습 파이프라인(train.py)과 호환되도록 설계됨.
#   - 입력 크기가 128×128일 때 FC 차원이 정확히 맞음(32*32*32).
# ============================================================

from __future__ import annotations
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    간단한 CNN 분류 모델 (입력: 128x128 이미지)
    구조:
        Conv(3→16) → ReLU → MaxPool
        Conv(16→32) → ReLU → MaxPool
        FC(32*32*32→128) → ReLU → FC(128→num_classes)
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # (B,16,128,128)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # (B,32,128,128)
        self.pool = nn.MaxPool2d(2, 2)                           # 다운샘플링 ×2
        # 128→64(1차 풀링), 64→32(2차 풀링) → (B,32,32,32)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # (B,3,128,128) → (B,16,64,64)
        x = self.pool(F.relu(self.conv1(x)))
        # (B,16,64,64) → (B,32,32,32)
        x = self.pool(F.relu(self.conv2(x)))
        # (B,32*32*32)
        x = x.view(x.size(0), -1)
        # (B,128) → (B,num_classes)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x