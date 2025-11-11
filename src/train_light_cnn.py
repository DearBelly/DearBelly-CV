# ============================================================
# 📄 파일명: ai_modules/src/train_light_cnn.py
# 📘 목적:
#   - 사전 생성된 JSON 목록(이미지 경로 + 정수 라벨)로 LightCNN(64x64 기준)을 학습·평가하는 스크립트임.
#   - 학습/검증 split, 조기 종료, best 모델 저장, 두 개의 테스트셋 평가, Acc/F1 플롯을 포함함.
#
# 🧪 사용 예시:
#   python -m ai_modules.src.train_light_cnn \
#     --train_json /path/matched_train_90.json \
#     --test_json1 /path/fortest.json \
#     --test_json2 /path/matched_test_18_deduped.json \
#     --save_dir runs/exp_lightcnn \
#     --epochs 100 --batch_size 32 --lr 1e-3
# ============================================================

from __future__ import annotations
import argparse
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt

# ✅ 기존처럼 모듈화된 경로로 임포트 유지
from ai_modules.src.data_prep.dataset_precomputed import PrecomputedPillDataset
from ai_modules.src.models.model_lightcnn import LightCNN
from ai_modules.src.utils.early_stopping import EarlyStopping
from ai_modules.src.utils.seed import set_seed


# ------------------------------
# 데이터로더 구성
# ------------------------------
def build_loaders(
    train_json: str,
    test_json1: Optional[str],
    test_json2: Optional[str],
    img_size: int,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Optional[DataLoader], int, dict]:
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    train_val_ds = PrecomputedPillDataset(train_json, transform=tfm)
    n_train = int(0.8 * len(train_val_ds))
    n_val = len(train_val_ds) - n_train
    train_ds, val_ds = random_split(
        train_val_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    test_loader_1 = None
    test_loader_2 = None
    if test_json1:
        test_loader_1 = DataLoader(
            PrecomputedPillDataset(test_json1, transform=tfm),
            batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
        )
    if test_json2:
        test_loader_2 = DataLoader(
            PrecomputedPillDataset(test_json2, transform=tfm),
            batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
        )

    # 클래스 수 추론
    if getattr(train_val_ds, "label2idx", None):
        num_classes = len(train_val_ds.label2idx)
    else:
        num_classes = max(s["label"] for s in train_val_ds.samples) + 1

    return train_loader, val_loader, test_loader_1, test_loader_2, num_classes, getattr(train_val_ds, "idx2label", {})


# ------------------------------
# 학습/평가 루프
# ------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_preds, total_labels = [], []
    loop = tqdm(loader, desc="🟢 Training", leave=False)

    for imgs, labels in loop:
        imgs = imgs.to(device)
        labels = torch.as_tensor(labels, dtype=torch.long, device=device)

        logits = model(imgs)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        total_preds.extend(preds.detach().cpu().numpy())
        total_labels.extend(labels.detach().cpu().numpy())
        loop.set_postfix(loss=float(loss.item()))

    acc = accuracy_score(total_labels, total_preds)
    f1  = f1_score(total_labels, total_preds, average='weighted')
    return acc, f1


@torch.no_grad()
def evaluate(model, loader, device, idx2label=None):
    model.eval()
    total_preds, total_labels = [], []
    loop = tqdm(loader, desc="🔵 Evaluating", leave=False)

    for imgs, labels in loop:
        imgs = imgs.to(device)
        labels = torch.as_tensor(labels, dtype=torch.long, device=device)

        logits = model(imgs)
        preds = torch.argmax(logits, dim=1)
        total_preds.extend(preds.detach().cpu().numpy())
        total_labels.extend(labels.detach().cpu().numpy())

    acc = accuracy_score(total_labels, total_preds)
    f1  = f1_score(total_labels, total_preds, average='weighted')

    if idx2label:
        try:
            idx_to_label = {int(k): v for k, v in idx2label.items()}
            target_names = [idx_to_label.get(i, str(i)) for i in range(len(idx_to_label))]
            print("\n[🔍 Classification Report]")
            print(classification_report(total_labels, total_preds, target_names=target_names, zero_division=0))
        except Exception:
            pass

    return acc, f1


def plot_metrics(train_acc, val_acc, train_f1, val_f1, save_dir: Optional[str] = None):
    plt.figure(figsize=(10, 5))
    # Acc
    plt.subplot(1, 2, 1)
    plt.plot(train_acc, label='Train Acc'); plt.plot(val_acc, label='Val Acc')
    plt.title("Accuracy"); plt.legend(); plt.grid(True)
    # F1
    plt.subplot(1, 2, 2)
    plt.plot(train_f1, label='Train F1'); plt.plot(val_f1, label='Val F1')
    plt.title("F1 Score"); plt.legend(); plt.grid(True)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, "metrics.png")
        plt.savefig(out, dpi=150)
        print(f"[FIG] saved: {out}")
    else:
        plt.show()


# ------------------------------
# 메인
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", required=True)
    ap.add_argument("--test_json1", default=None)
    ap.add_argument("--test_json2", default=None)
    ap.add_argument("--save_dir",   default="runs/exp_lightcnn")
    ap.add_argument("--img_size",   type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs",     type=int, default=100)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--seed",       type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    train_loader, val_loader, test_loader_1, test_loader_2, num_classes, idx2label = build_loaders(
        args.train_json, args.test_json1, args.test_json2, args.img_size, args.batch_size
    )

    model = LightCNN(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val = -1.0
    best_epoch = -1
    best_path = os.path.join(args.save_dir, "best_model.pth")
    stopper = EarlyStopping(patience=7, delta=1e-3, path=os.path.join(args.save_dir, "earlystop.pth"))

    tr_accs, va_accs, tr_f1s, va_f1s = [], [], [], []

    for epoch in range(1, args.epochs + 1):
        print(f"\n[Epoch {epoch}]")
        tr_acc, tr_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_acc, va_f1 = evaluate(model, val_loader, device, idx2label=idx2label)

        tr_accs.append(tr_acc); va_accs.append(va_acc)
        tr_f1s.append(tr_f1);   va_f1s.append(va_f1)

        if va_acc > best_val:
            best_val = va_acc
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "num_classes": num_classes,
                "img_size": args.img_size,
            }, best_path)
            print(f"⭐ Best updated: {best_path} (val_acc={va_acc:.4f})")

        print(f"Train Acc: {tr_acc:.2%} | F1: {tr_f1:.4f}")
        print(f"Val   Acc: {va_acc:.2%} | F1: {va_f1:.4f}")

        stopper(va_acc, model)
        if stopper.early_stop:
            print(f"\n⛔ Early stopping at epoch {epoch}")
            break

    print(f"\n🏁 Done. Best Val Acc: {best_val:.2%} @ epoch {best_epoch}")
    print(f"📦 Best saved: {best_path}")

    # 테스트 평가
    if test_loader_1:
        print("\n🧪 [Test 1]")
        acc1, f11 = evaluate(model, test_loader_1, device, idx2label=idx2label)
        print(f"Test1 Acc: {acc1:.2%} | F1: {f11:.4f}")

    if test_loader_2:
        print("\n🧪 [Test 2]")
        acc2, f12 = evaluate(model, test_loader_2, device, idx2label=idx2label)
        print(f"Test2 Acc: {acc2:.2%} | F1: {f12:.4f}")

    plot_metrics(tr_accs, va_accs, tr_f1s, va_f1s, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
