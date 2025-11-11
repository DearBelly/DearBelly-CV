# 약 이미지 Dataset 모듈임# ============================================================
# 📄 파일명: dataset_pill.py
# 📁 위치: ai_modules/src/data_prep/dataset_pill.py
# 📘 설명:
#   - 경구약제(단일 약) 이미지와 라벨(JSON)을 매칭하여
#     PyTorch Dataset 형태로 로드하는 클래스임.
#   - 폴더 구조 예시:
#       ├── TS_81_단일crop128/
#       │    ├── K-001234/
#       │    │    ├── image_01.jpg
#       │    │    └── image_02.jpg
#       │    └── ...
#       └── TL_81_단일_json/
#            ├── K-001234.json
#            ├── K-001235.json
#            └── ...
#   - 학습 시, 이미지와 JSON의 "dl_name" (약 이름) 키를 라벨로 사용함.
# ============================================================

from __future__ import annotations
import os
import json
from typing import List, Tuple, Dict
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


class PillDataset(Dataset):
    """
    💊 PillDataset 클래스

    이미지와 JSON 라벨 파일을 매칭하여 PyTorch Dataset으로 구성하는 클래스임.
    - image_root: 이미지 폴더 루트 경로 (예: TS_81_단일crop128)
    - label_root: 라벨 JSON 폴더 루트 경로 (예: TL_81_단일)
    - label_key : JSON 내 라벨 키 이름 (기본값: 'dl_name')
    - transform : torchvision.transforms.Compose 형태의 변환기
    """

    def __init__(
        self,
        image_root: str,
        label_root: str,
        label_key: str = "dl_name",
        transform: transforms.Compose | None = None,
        extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
        use_tqdm: bool = True,
    ):
        self.samples: List[Tuple[str, int]] = []
        self.label2idx: Dict[str, int] = {}
        self.idx2label: List[str] = []
        self.transform = transform
        self.label_key = label_key
        self.extensions = tuple(e.lower() for e in extensions)

        # 이미지 폴더 목록 (예: TS_81_단일crop128/ 내부의 모든 하위 폴더)
        folders = [f for f in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, f))]
        iterator = tqdm(folders, desc="🔍 이미지 폴더 탐색 중") if use_tqdm else folders

        for folder_name in iterator:
            img_dir = os.path.join(image_root, folder_name)
            json_dir = os.path.join(label_root, f"{folder_name}_json")

            if not os.path.isdir(json_dir):
                continue  # 해당 폴더에 JSON 매칭 폴더 없으면 skip

            for file in os.listdir(img_dir):
                if not file.lower().endswith(self.extensions):
                    continue

                img_path = os.path.join(img_dir, file)
                base = os.path.splitext(file)[0]
                json_path = os.path.join(json_dir, base + ".json")

                if not os.path.exists(json_path):
                    continue

                # JSON 로딩 및 라벨 추출
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    images = data.get("images", [])
                    if not images or not isinstance(images, list):
                        continue

                    label = images[0].get(self.label_key)
                    if label is None:
                        continue

                except Exception as e:
                    print(f"[⚠️ JSON 로드 실패] {json_path}: {e}")
                    continue

                # 라벨 → 인덱스 매핑
                if label not in self.label2idx:
                    self.label2idx[label] = len(self.label2idx)

                self.samples.append((img_path, self.label2idx[label]))

        # 역매핑(idx2label) 구성
        self.idx2label = [None] * len(self.label2idx)
        for k, v in self.label2idx.items():
            self.idx2label[v] = k

        print(f"✅ 총 유효 샘플 수: {len(self.samples)}")
        print(f"✅ 총 클래스 수: {len(self.label2idx)}")

    def __len__(self) -> int:
        """전체 샘플 개수 반환"""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        """
        인덱스 기반으로 (이미지 텐서, 라벨 인덱스)를 반환함.
        - 이미지 변환(transform)이 지정되어 있으면 변환 후 반환.
        """
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
