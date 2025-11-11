from __future__ import annotations
import os, argparse, random
from typing import Tuple
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# -----------------------------
# 공통 유틸
# -----------------------------
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

def list_images(src_folder: str):
    return [f for f in os.listdir(src_folder) if f.lower().endswith(IMG_EXTS)]

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def to_tensor_transform(size: Tuple[int, int]):
    return T.Compose([T.Resize(size), T.ToTensor()])

def to_pil(t):
    return T.ToPILImage()(t)

# -----------------------------
# 증강 연산들
# -----------------------------
def add_gaussian_noise(tensor: torch.Tensor, mean=0.0, std=0.05):
    noise = torch.randn_like(tensor) * std + mean
    out = tensor + noise
    return torch.clamp(out, 0.0, 1.0)

def apply_shear(tensor: torch.Tensor, degrees=(-45, 45)):
    angle = 0.0
    translate = [0, 0]
    scale = 1.0
    shear = random.uniform(degrees[0], degrees[1])
    # torchvision >= 0.13: shear=[x, y] 형태 가능. 여기서는 x축만 사용.
    return TF.affine(tensor, angle=angle, translate=translate, scale=scale, shear=[shear, 0])

def apply_brightness(tensor: torch.Tensor, brightness_range=(1.0, 2.0)):
    factor = random.uniform(*brightness_range)
    return TF.adjust_brightness(tensor, factor)

# -----------------------------
# 파이프라인
# -----------------------------
def run_augment(op: str, src: str, dst: str, size=(128, 128),
                noise_mean=0.0, noise_std=0.05,
                shear_min=-45, shear_max=45,
                bright_min=1.0, bright_max=2.0):
    ensure_dir(dst)
    tf_in = to_tensor_transform(size)

    files = list_images(src)
    total = len(files)
    if total == 0:
        print(f"[WARN] 입력 폴더에 이미지가 없음: {src}")
        return

    if op == "noise":
        print(f"총 {total}장 → 가우시안 노이즈(mean={noise_mean}, std={noise_std}) 적용 중...")
    elif op == "shear":
        print(f"총 {total}장 → Shear 범위=({shear_min}, {shear_max}) 적용 중...")
    elif op == "brightness":
        print(f"총 {total}장 → 밝기 범위=({bright_min}, {bright_max}) 적용 중...")
    else:
        raise ValueError(f"알 수 없는 op: {op}")

    for i, fname in enumerate(files, 1):
        src_path = os.path.join(src, fname)
        dst_path = os.path.join(dst, fname)
        try:
            img = Image.open(src_path).convert("RGB")
            t = tf_in(img)

            if op == "noise":
                t = add_gaussian_noise(t, mean=noise_mean, std=noise_std)
            elif op == "shear":
                t = apply_shear(t, degrees=(shear_min, shear_max))
            elif op == "brightness":
                t = apply_brightness(t, brightness_range=(bright_min, bright_max))

            to_pil(t).save(dst_path)
            print(f"[{i}/{total}] 저장 완료: {fname}")
        except Exception as e:
            print(f"[ERROR] {fname}: {e}")

    print(f"\n✅ 완료: {dst} 에 {total}개 파일 생성됨")

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Folder-level image augmentation")
    ap.add_argument("--op", required=True, choices=["noise", "shear", "brightness"])
    ap.add_argument("--src", required=True, help="입력 이미지 폴더")
    ap.add_argument("--dst", required=True, help="출력 이미지 폴더")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--height", type=int, default=128)

    # noise
    ap.add_argument("--noise_mean", type=float, default=0.0)
    ap.add_argument("--noise_std",  type=float, default=0.05)
    # shear
    ap.add_argument("--shear_min", type=float, default=-45.0)
    ap.add_argument("--shear_max", type=float, default=45.0)
    # brightness
    ap.add_argument("--bright_min", type=float, default=1.0)
    ap.add_argument("--bright_max", type=float, default=2.0)
    return ap.parse_args()

def main():
    args = parse_args()
    run_augment(
        op=args.op,
        src=args.src,
        dst=args.dst,
        size=(args.width, args.height),
        noise_mean=args.noise_mean,
        noise_std=args.noise_std,
        shear_min=args.shear_min,
        shear_max=args.shear_max,
        bright_min=args.bright_min,
        bright_max=args.bright_max,
    )

if __name__ == "__main__":
    main()
