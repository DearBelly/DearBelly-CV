# ============================================================
# 📄 파일명: ai_modules/src/utils/ema.py
# 📘 목적: Exponential Moving Average(EMA) 유틸
# ============================================================
from __future__ import annotations
import torch
import copy

class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for ema_p, p in zip(self.ema.parameters(), model.parameters()):
            ema_p.copy_(ema_p * self.decay + p * (1.0 - self.decay))
