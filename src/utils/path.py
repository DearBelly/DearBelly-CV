# ============================================================
# 📄 파일명: paths.py
# 📁 위치: ai_modules/src/utils/paths.py
# 📘 목적:
#   - 데이터/결과 경로를 한 곳에서 관리하는 경량 유틸임.
#   - 하드코딩을 피하고, 실행 인자나 설정파일(YAML)과 연동하기 쉽게 함.
#
# 🧪 사용 예시:
#   from ai_modules.src.utils.paths import DataPaths
#   p = DataPaths(image_root=".../TS_81_단일crop128",
#                 label_root=".../TL_81_단일",
#                 save_dir="runs/exp001")
#   print(p.image_root); print(p.label_root); p.ensure_save_dir()
#
# ✅ 특징:
#   - dataclass 기반으로 필드를 명확히 관리함.
#   - ensure_* 메서드로 디렉터리 자동 생성 지원함.
# ============================================================

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataPaths:
    """이미지/라벨/결과 경로를 묶어 관리하는 데이터 클래스임."""
    image_root: str
    label_root: str
    save_dir: str = "runs/exp001"
    label_key: str = "dl_name"

    def image_root_path(self) -> Path:
        return Path(self.image_root)

    def label_root_path(self) -> Path:
        return Path(self.label_root)

    def save_dir_path(self) -> Path:
        return Path(self.save_dir)

    def ensure_save_dir(self) -> None:
        """결과 저장 디렉터리를 생성함."""
        self.save_dir_path().mkdir(parents=True, exist_ok=True)
