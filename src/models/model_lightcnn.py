# ============================================================
# 📄 파일명: model_lightcnn.py
# 📁 위치: ai_modules/src/models/model_lightcnn.py
# 📘 목적:
#   - 64×64 입력을 가정한 경량 CNN 분류 모델(LightCNN) 정의임.
#   - 구조: Conv(3→8) → ReLU → MaxPool → Conv(8→16) → ReLU → MaxPool
#           → GAP(4×4) → FC(256→64) → FC(64→num_classes)
#
# 🧪 사용 예시:
#   from ai_modules.src.models.model_lightcnn import LightCNN
#   model = LightCNN(num_classes=492)
#
# ✅ 특징:
#   - 파라미터 수가 적어 빠르게 추론 가능함.
#   - predict_and_advise.py 파이프라인에서 기본 모델로 사용됨.
# ============================================================

from __future__ import annotations
import torch.nn as nn
import torch.nn.functional as F

class LightCNN(nn.Module):
    """
    경량화된 이미지 분류 CNN 모델임.
    입력 크기: 64×64 RGB
    출력 크기: num_classes
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)   # (B,8,64,64)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # (B,16,64,64)
        self.pool = nn.MaxPool2d(2, 2)                           # 64→32→16
        self.gap  = nn.AdaptiveAvgPool2d((4, 4))                 # (B,16,4,4)
        self.fc1  = nn.Linear(16 * 4 * 4, 64)                    # 256→64
        self.fc2  = nn.Linear(64, num_classes)

    def forward(self, x):
        # (B,3,64,64) → (B,8,32,32)
        x = self.pool(F.relu(self.conv1(x)))
        # (B,8,32,32) → (B,16,16,16)
        x = self.pool(F.relu(self.conv2(x)))
        # (B,16,4,4)
        x = self.gap(x)
        # (B,256)
        x = x.view(x.size(0), -1)
        # (B,64) → (B,num_classes)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
