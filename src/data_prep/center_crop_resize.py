# ============================================================
# 📄 파일명: center_crop_resize.py
# 📁 위치: ai_modules/src/data_prep/center_crop_resize.py
# 📘 목적:
#   - 입력 폴더의 모든 이미지에 대해 중앙 크롭 후 지정 크기로 리사이즈하여 출력 폴더에 저장함.
#   - 원본 폴더 구조를 그대로 보존하여 출력 경로에 반영함.
#   - 지원 확장자: .png, .jpg, .jpeg
#
# 🔌 입력 인자:
#   --input       : 입력 루트 경로
#   --output      : 출력 루트 경로(없으면 생성함)
#   --crop-size   : 중앙 크롭 정사각형 한 변 길이(기본 512)
#   --resize-size : 리사이즈 결과 한 변 길이(기본 128)
#
# 🧪 사용 예시:
#   python -m ai_modules.src.data_prep.center_crop_resize \
#       --input  "/path/TS_54_단일" \
#       --output "/path/TS_54_단일crop128" \
#       --crop-size 512 --resize-size 128
#
# ⚠️ 주의:
#   - 원본 이미지가 crop-size보다 작으면 스킵함.
#   - EXIF가 깨진 이미지 등은 경고만 출력하고 계속 진행함.
# ============================================================

from __future__ import annotations
import argparse, os
from pathlib import Path
from PIL import Image

EXTS = {".png", ".jpg", ".jpeg"}

def center_crop_resize(
    input_dir: str,
    output_dir: str,
    crop_size: int = 512,
    resize_size: int = 128,
) -> None:
    """입력 폴더 트리를 보존하면서 중앙 크롭 후 리사이즈된 이미지를 출력 폴더에 저장함."""
    in_p, out_p = Path(input_dir), Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    for root, _, files in os.walk(in_p):
        rel = Path(root).relative_to(in_p)
        save_dir = out_p / rel
        save_dir.mkdir(parents=True, exist_ok=True)

        for f in files:
            if Path(f).suffix.lower() not in EXTS:
                continue

            src = Path(root) / f
            dst = save_dir / f
            try:
                img = Image.open(src).convert("RGB")
                w, h = img.size
                if w < crop_size or h < crop_size:
                    # 크롭 영역이 원본보다 크면 스킵
                    print(f"[SKIP] {src} (원본 크기 {w}x{h} < crop {crop_size})")
                    continue

                cx, cy = w // 2, h // 2
                left   = cx - crop_size // 2
                top    = cy - crop_size // 2
                right  = left + crop_size
                bottom = top + crop_size

                cropped = img.crop((left, top, right, bottom))
                resized = cropped.resize((resize_size, resize_size), Image.BILINEAR)
                resized.save(dst)
            except Exception as e:
                print(f"[WARN] 처리 실패: {src} → {e}")

def main():
    p = argparse.ArgumentParser(description="중앙 크롭 후 리사이즈 파이프라인임.")
    p.add_argument("--input", required=True, help="입력 루트 경로")
    p.add_argument("--output", required=True, help="출력 루트 경로")
    p.add_argument("--crop-size", type=int, default=512, help="중앙 크롭 정사각형 크기")
    p.add_argument("--resize-size", type=int, default=128, help="리사이즈 크기")
    args = p.parse_args()

    center_crop_resize(
        input_dir=args.input,
        output_dir=args.output,
        crop_size=args.crop_size,
        resize_size=args.resize_size,
    )

if __name__ == "__main__":
    main()
