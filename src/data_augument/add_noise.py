# ============================================================
# 📄 파일명: ai_modules/src/data_augment/add_noise.py
# 📘 목적: 폴더 단위로 가우시안 노이즈 이미지를 생성함.
# 사용 예:
#   python -m ai_modules.src.data_augmentation.add_noise \
#     --src "/path/in" --dst "/path/out" --width 128 --height 128 --mean 0.0 --std 0.05
# ============================================================
from __future__ import annotations
import argparse, os
from PIL import Image
import torch
import torchvision.transforms as T

def add_gaussian_noise(img_tensor, mean=0.0, std=0.05):
    noise = torch.randn_like(img_tensor) * std + mean
    noisy = img_tensor + noise
    return torch.clamp(noisy, 0.0, 1.0)

def process_folder(src: str, dst: str, size=(128,128), mean=0.0, std=0.05) -> None:
    os.makedirs(dst, exist_ok=True)
    tf = T.Compose([T.Resize(size), T.ToTensor()])
    to_pil = T.ToPILImage()

    files = [f for f in os.listdir(src) if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp"))]
    total = len(files)
    print(f"총 {total}장 처리 시작 (noise mean={mean}, std={std}, size={size})")
    for i, name in enumerate(files, 1):
        in_path  = os.path.join(src, name)
        out_path = os.path.join(dst, name)
        try:
            img = Image.open(in_path).convert("RGB")
            tensor = tf(img)
            noisy  = add_gaussian_noise(tensor, mean=mean, std=std)
            to_pil(noisy).save(out_path)
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
    ap.add_argument("--mean", type=float, default=0.0)
    ap.add_argument("--std",  type=float, default=0.05)
    return ap.parse_args()

def main():
    args = parse_args()
    process_folder(args.src, args.dst, (args.width, args.height), args.mean, args.std)

if __name__ == "__main__":
    main()
