# add_noise.py
"""
이미지 데이터셋에 가우시안 노이즈를 추가하는 전처리 유틸 스크립트
"""

import os
from PIL import Image
import torch
import torchvision.transforms as transforms

def add_gaussian_noise(tensor, mean=0.0, std=0.05):
    noise = torch.randn_like(tensor) * std + mean
    noisy_tensor = tensor + noise
    return torch.clamp(noisy_tensor, 0.0, 1.0)

def save_noisy_images(src_folder, dst_folder, image_size=(128, 128), mean=0.0, std=0.05):
    os.makedirs(dst_folder, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    to_pil = transforms.ToPILImage()

    img_files = [f for f in os.listdir(src_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for fname in img_files:
        src_path = os.path.join(src_folder, fname)
        dst_path = os.path.join(dst_folder, fname)

        try:
            img = Image.open(src_path).convert("RGB")
            tensor = transform(img)
            noisy_tensor = add_gaussian_noise(tensor, mean=mean, std=std)
            noisy_img = to_pil(noisy_tensor)
            noisy_img.save(dst_path)
        except Exception as e:
            print(f"[ERROR] {fname}: {e}")

    print(f"✅ 완료: {len(img_files)}개의 노이즈 이미지 저장됨 → {dst_folder}")
