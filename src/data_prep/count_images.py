# ============================================================
# 📄 파일명: count_images.py
# 📁 위치: ai_modules/src/data_prep/count_images.py
# 📘 목적:
#   - 특정 루트 디렉터리 하위의 이미지 파일 개수를 재귀적으로 집계함.
#   - 지원 확장자: .png, .jpg, .jpeg, .bmp, .gif
#
# 🔌 입력 인자:
#   --root : 탐색 시작 루트 디렉터리 경로
#
# 🧪 사용 예시:
#   python -m ai_modules.src.data_prep.count_images \
#       --root "/path/to/TS_57_단일"
#
# 📝 출력:
#   "총 이미지 파일 수: <N>"
# ============================================================

from __future__ import annotations
import argparse, os
from pathlib import Path

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}

def count_images(root: str) -> int:
    """루트 이하 모든 하위 폴더를 재귀적으로 순회하며 이미지 확장자를 카운트함."""
    root = os.path.abspath(root)
    total = 0
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if Path(fn).suffix.lower() in EXTS:
                total += 1
    return total

def main():
    ap = argparse.ArgumentParser(description="이미지 파일 개수를 재귀적으로 집계하는 유틸임.")
    ap.add_argument("--root", required=True, help="탐색 시작 루트 디렉터리")
    args = ap.parse_args()
    n = count_images(args.root)
    print(f"총 이미지 파일 수: {n}")

if __name__ == "__main__":
    main()
