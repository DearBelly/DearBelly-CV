# ============================================================
# 📄 파일명: ai_modules/src/utils/early_stopping.py
# 📘 목적:
#   - 검증 점수가 일정 에폭 동안 개선되지 않으면 학습을 조기 종료하는 유틸임.
#
# 🧪 사용 예시:
#   from ai_modules.src.utils.early_stopping import EarlyStopping
#   stopper = EarlyStopping(patience=7, delta=1e-3, path="runs/exp/best.pt")
#   stopper(val_acc, model)
#   if stopper.early_stop: break
# ============================================================

from __future__ import annotations
import torch

class EarlyStopping:
    def __init__(self, patience: int = 7, delta: float = 1e-3, path: str = 'earlystop.pth'):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.path = path

    def __call__(self, val_score: float, model) -> None:
        if self.best_score is None:
            self.best_score = val_score
            self._save(model)
            return

        if val_score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self._save(model)
            self.counter = 0

    def _save(self, model) -> None:
        torch.save(model.state_dict(), self.path)
