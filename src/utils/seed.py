# ============================================================
# 📄 파일명: seed.py
# 📁 위치: ai_modules/src/utils/seed.py
# 📘 목적:
#   - 재현성 확보를 위해 파이썬/넘파이/파이토치의 난수 시드를 고정하는 유틸임.
#
# 🧪 사용 예시:
#   from ai_modules.src.utils.seed import set_seed
#   set_seed(42)
#
# ✅ 특징:
#   - CUDA 사용 시에도 동일한 시퀀스를 보장하도록 cudnn 옵션을 설정함.
# ============================================================

import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """난수 시드를 고정하여 실험 재현성을 높이는 함수임."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CUDNN 결정적 동작 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[INFO] Seed fixed to {seed}")
