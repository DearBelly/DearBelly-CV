# ============================================================
# 📄 파일명: train.py
# 📁 위치: ai_modules/src/train.py
# 📘 목적:
#   - SimpleCNN(128x128) 기반 분류 모델 학습/검증 파이프라인임.
#   - baseline.yaml 또는 CLI 인자로 데이터 경로/하이퍼파라미터를 받아 수행함.
#
# 🧪 사용 예시:
#   python -m ai_modules.src.train --config ai_modules/configs/baseline.yaml
#   # 또는
#   python -m ai_modules.src.train --image_root ... --label_root ... --epochs 5
#
# ✅ 특징:
#   - 최고 검증 정확도 시 best.pt 자동 저장
#   - CUDA/CPU 자동 선택, 시드 고정 지원
# ============================================================

from __future__ import annotations
import argparse, os, yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from ai_modules.src.data_prep.dataset_pill import PillDataset
from ai_modules.src.models.simple_cnn import SimpleCNN
from ai_modules.src.utils.seed import set_seed

def build_loaders(image_root: str, label_root: str, label_key: str,
                  img_size: int, batch_size: int):
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    dataset = PillDataset(
        image_root=image_root,
        label_root=label_root,
        label_key=label_key,
        transform=tfm,
        use_tqdm=True,
    )
    if len(dataset) == 0:
        raise RuntimeError("유효한 이미지-라벨 쌍이 없음. 경로/label_key를 확인할 것.")

    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, len(dataset.label2idx)

@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total, correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        total += y.size(0)
        correct += (pred == y).sum().item()
    return correct / total if total else 0.0

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, num_classes = build_loaders(
        image_root=args.image_root,
        label_root=args.label_root,
        label_key=args.label_key,
        img_size=args.img_size,
        batch_size=args.batch_size,
    )

    model = SimpleCNN(num_classes=num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.save_dir, exist_ok=True)
    best_val = -1.0
    best_path = os.path.join(args.save_dir, "best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, correct = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            pred = logits.argmax(1)
            total += y.size(0)
            correct += (pred == y).sum().item()
        train_acc = correct / total if total else 0.0

        val_acc = evaluate(model, val_loader, device)
        print(f"[Epoch {epoch:03d}] train_acc={train_acc:.3f} | val_acc={val_acc:.3f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"✅ Saved best to {best_path} (val_acc={val_acc:.3f})")

    print(f"[DONE] best_val_acc={best_val:.3f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)

    # 데이터 경로
    p.add_argument("--image_root", type=str, default=None)
    p.add_argument("--label_root", type=str, default=None)
    p.add_argument("--label_key",  type=str, default="dl_name")

    # 하이퍼파라미터
    p.add_argument("--img_size",  type=int, default=128)
    p.add_argument("--batch_size",type=int, default=32)
    p.add_argument("--epochs",    type=int, default=5)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--seed",      type=int, default=42)

    # 결과 저장
    p.add_argument("--save_dir",  type=str, default="runs/exp001")

    args = p.parse_args()

    # YAML 설정 불러오기(있으면 CLI보다 우선 적용)
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        for k, v in cfg.items():
            setattr(args, k, v)

    # 필수 경로 체크
    if not args.image_root or not args.label_root:
        raise SystemExit("image_root/label_root가 필요함. --config 또는 CLI 인자로 지정할 것.")

    train(args)
