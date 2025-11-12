# ============================================================
# 📄 파일명: ai_modules/src/data_augment/adjust_brightness.py
# 📘 목적: 폴더 단위로 밝기 조절 이미지를 생성함.
# 사용 예:
#   python -m ai_modules.src.data_augmentation.adjust_brightness \
#     --src "/path/in" --dst "/path/out" --width 128 --height 128 --bright_min 1.0 --bright_max 2.0
# ============================================================
from __future__ import annotations
import argparse, os, random
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

def apply_brightness(img_tensor, bright_min=1.0, bright_max=2.0):
    factor = random.uniform(bright_min, bright_max)
    return TF.adjust_brightness(img_tensor, factor)

def process_folder(src: str, dst: str, size=(128,128), bright_min=1.0, bright_max=2.0) -> None:
    os.makedirs(dst, exist_ok=True)
    tf = T.Compose([T.Resize(size), T.ToTensor()])
    to_pil = T.ToPILImage()

    files = [f for f in os.listdir(src) if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp"))]
    total = len(files)
    print(f"총 {total}장 처리 시작 (brightness∈[{bright_min}, {bright_max}], size={size})")
    for i, name in enumerate(files, 1):
        in_path  = os.path.join(src, name)
        out_path = os.path.join(dst, name)
        try:
            img = Image.open(in_path).convert("RGB")
            tensor = tf(img)
            bright = apply_brightness(tensor, bright_min, bright_max)
            to_pil(bright).save(out_path)
            if i % 50 == 0 or i == total:
                print(f"[{i}/{total}] {name} → 저장 완료")
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
    print(f"완료: {dst} 에 저장됨")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--height", type=int, default=128)
    ap.add_argument("--bright_min", type=float, default=1.0)
    ap.add_argument("--bright_max", type=float, default=2.0)
    return ap.parse_args()

def main():
    args = parse_args()
    process_folder(args.src, args.dst, (args.width, args.height), args.bright_min, args.bright_max)

if __name__ == "__main__":
    main()
