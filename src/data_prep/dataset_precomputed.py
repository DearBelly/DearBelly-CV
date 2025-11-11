# ============================================================
# 📄 파일명: ai_modules/src/data_prep/dataset_precomputed.py
# 📘 목적:
#   - JSON 목록(사전 전처리된 이미지 경로 + 정수 라벨)을 직접 읽어
#     학습/평가에 사용하는 Dataset 클래스임.
#   - JSON 구조:
#       { "samples": [{"image_path": "...", "label": 123}, ...],
#         "label2idx": {...}, "idx2label": {...} }  또는
#       [ {"image_path": "...", "label": 123}, ... ]
#
# 🧪 사용 예시:
#   from ai_modules.src.data_prep.dataset_precomputed import PrecomputedPillDataset
#   ds = PrecomputedPillDataset("matched_train.json", transform=...)
# ============================================================

from __future__ import annotations
from typing import Any, Dict, List
from PIL import Image
import json
from torch.utils.data import Dataset

class PrecomputedPillDataset(Dataset):
    def __init__(self, json_path: str, transform=None, cache: bool = False):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict):
            self.samples: List[Dict[str, Any]] = data.get('samples', [])
            self.label2idx: Dict[str, int] = data.get('label2idx', {})
            self.idx2label: Dict[str, str] = data.get('idx2label', {})
        else:
            self.samples = data
            self.label2idx, self.idx2label = {}, {}

        self.transform = transform
        self.cache = cache
        self._cache_images: Dict[str, Any] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        rec = self.samples[idx]
        img_path = rec['image_path']
        label = rec['label']

        if self.cache and img_path in self._cache_images:
            image = self._cache_images[img_path]
        else:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            if self.cache:
                self._cache_images[img_path] = image

        return image, label
