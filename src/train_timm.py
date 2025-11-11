# ============================================================
# 📄 파일명: ai_modules/src/train_timm.py
# 📘 목적: timm 백본(EfficientNet 등) + (선택)ArcFace로 학습
#       - PrecomputedPillDataset(JSON) 사용
#       - Mixup/라벨스무딩/AMP/클래스불균형 대응 포함
# 사용 예:
#   python -m ai_modules.src.train_timm \
#     --train_json /path/train.json \
#     --save_dir runs/exp_timm \
#     --model efficientnet_b3 --img_size 320 --epochs 200
# ============================================================
from __future__ import annotations
import argparse, os, math, json, random
from collections import Counter
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.data import Mixup

from ai_modules.src.data_prep.dataset_precomputed import PrecomputedPillDataset
from ai_modules.src.models.backbone_timm import BackboneWithHead
from ai_modules.src.utils.ema import EMA
from ai_modules.src.utils.seed import set_seed

def build_transforms(img_size: int):
    has_randaugment = hasattr(transforms, "RandAugment")
    train_tf_list = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30, fill=(128,128,128)),
        transforms.ColorJitter(0.25,0.25,0.25,0.15),
    ]
    if has_randaugment:
        train_tf_list.insert(1, transforms.RandAugment(num_ops=2, magnitude=9))
    train_tf_list += [
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ]
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])
    return transforms.Compose(train_tf_list), val_tf

def plot_metrics(train_acc, val_acc, train_f1, val_f1, save_dir: str | None = None):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1); plt.plot(train_acc, label='Train Acc'); plt.plot(val_acc, label='Val Acc'); plt.legend(); plt.grid(True)
    plt.subplot(1,2,2); plt.plot(train_f1, label='Train F1'); plt.plot(val_f1, label='Val F1'); plt.legend(); plt.grid(True)
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, "metrics.png")
        plt.savefig(out, dpi=150)
        print(f"[FIG] saved: {out}")
    else:
        plt.show()

def build_loaders(train_json: str, img_size: int, batch_size: int, seed: int):
    train_tf, val_tf = build_transforms(img_size)
    full_ds = PrecomputedPillDataset(train_json, transform=None)
    val_size = int(len(full_ds) * 0.2)
    train_size = len(full_ds) - val_size
    idx = list(range(len(full_ds)))
    g = torch.Generator().manual_seed(seed)
    train_idx, val_idx = random_split(idx, [train_size, val_size], generator=g)
    train_idx, val_idx = list(train_idx), list(val_idx)

    base_train = PrecomputedPillDataset(train_json, transform=train_tf)
    base_val   = PrecomputedPillDataset(train_json, transform=val_tf)

    class SubsetDS(torch.utils.data.Dataset):
        def __init__(self, base, idx): self.base, self.idx = base, idx
        def __len__(self): return len(self.idx)
        def __getitem__(self, i): return self.base[self.idx[i]]

    ds_train = SubsetDS(base_train, train_idx)
    ds_val   = SubsetDS(base_val,   val_idx)

    # 불균형 대응: 클래스 가중치 → WeightedRandomSampler
    labels_train = [int(base_train.samples[i]['label']) for i in train_idx]
    cnt = Counter(labels_train)
    classes = sorted(cnt.keys())
    num_classes = max(classes) + 1
    class_counts = torch.tensor([cnt.get(c, 0) for c in range(num_classes)], dtype=torch.float)
    class_weights = (class_counts.sum() / torch.clamp(class_counts, min=1.0))
    class_weights = class_weights / class_weights.mean()
    sample_weights = torch.tensor([class_weights[int(base_train.samples[i]['label'])] for i in train_idx], dtype=torch.float)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(ds_train, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False,   num_workers=2, pin_memory=True)
    return train_loader, val_loader, num_classes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", required=True)
    ap.add_argument("--save_dir",   default="runs/exp_timm")
    ap.add_argument("--model",      default="efficientnet_b3")
    ap.add_argument("--img_size",   type=int, default=320)
    ap.add_argument("--batch_size", type=int, default=48)
    ap.add_argument("--epochs",     type=int, default=200)
    ap.add_argument("--lr",         type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--label_smooth", type=float, default=0.05)
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--use_arcface", action="store_true")
    ap.add_argument("--use_mixup",   action="store_true")
    ap.add_argument("--amp",         action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, num_classes = build_loaders(args.train_json, args.img_size, args.batch_size, args.seed)

    # ArcFace + Mixup 동시에 사용 금지
    if args.use_arcface and args.use_mixup:
        raise ValueError("ArcFace는 hard label이 필요하므로 Mixup과 함께 사용할 수 없음")

    model = BackboneWithHead(args.model, num_classes, pretrained=True, use_arcface=args.use_arcface, dropout=0.2).to(device)

    # 손실함수 선택
    if args.use_mixup and not args.use_arcface:
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smooth)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6)
    ema = EMA(model, decay=0.999)

    mixup_fn = None
    if args.use_mixup and not args.use_arcface:
        mixup_fn = Mixup(mixup_alpha=0.2, cutmix_alpha=0.0, label_smoothing=args.label_smooth, num_classes=num_classes)

    best_val = -1.0
    best_path = os.path.join(args.save_dir, "best_model_timm.pth")
    tr_accs, va_accs, tr_f1s, va_f1s = [], [], [], []

    print("\n========== Training (timm) ==========")
    for epoch in range(1, args.epochs + 1):
        # ---- train ----
        model.train()
        total_preds, total_labels = [], []
        loop = tqdm(train_loader, desc=f"🟢 Epoch {epoch}", leave=False)
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device).long()
            if mixup_fn is not None:
                imgs, labels_mixed = mixup_fn(imgs, labels)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=args.amp):
                logits = model(imgs, labels if args.use_arcface else None) if args.use_arcface else model(imgs)
                loss = criterion(logits, labels if (args.use_arcface or mixup_fn is None) else labels_mixed)
            if args.amp:
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()
            ema.update(model)
            preds = torch.argmax(logits, dim=1)
            if not args.use_mixup:  # hard label일 때만 정확도 집계
                total_preds.extend(preds.detach().cpu().numpy())
                total_labels.extend(labels.detach().cpu().numpy())
            loop.set_postfix(loss=float(loss))

        tr_acc = accuracy_score(total_labels, total_preds) if total_labels else 0.0
        tr_f1  = f1_score(total_labels, total_preds, average='weighted') if total_labels else 0.0
        tr_accs.append(tr_acc); tr_f1s.append(tr_f1)

        # ---- val ----
        model.eval()
        ema_model = ema.ema  # EMA 가중치로 평가
        v_preds, v_labels = [], []
        val_loss = 0.0
        ce = nn.CrossEntropyLoss()
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device).long()
                logits = ema_model(imgs, labels) if args.use_arcface else ema_model(imgs)
                val_loss += float(ce(logits, labels)) * imgs.size(0)
                preds = torch.argmax(logits, dim=1)
                v_preds.extend(preds.detach().cpu().numpy())
                v_labels.extend(labels.detach().cpu().numpy())

        v_acc = accuracy_score(v_labels, v_preds)
        v_f1  = f1_score(v_labels, v_preds, average='weighted')
        va_accs.append(v_acc); va_f1s.append(v_f1)
        scheduler.step(v_acc)

        print(f"[E{epoch:03d}] Train {tr_acc*100:.2f}% | F1 {tr_f1:.4f}  ||  Val {v_acc*100:.2f}% | F1 {v_f1:.4f}")

        if v_acc > best_val:
            best_val = v_acc
            torch.save({"epoch": epoch, "model_state_dict": ema.ema.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, best_path)
            print(f"⭐ Best updated: {best_path} (val_acc={v_acc:.4f})")

    print(f"\n🏁 Done. Best Val Acc: {best_val*100:.2f}% → {best_path}")
    # (선택) 분류 리포트 한 번 출력
    try:
        idx2label = {}
        with open(args.train_json, 'r', encoding='utf-8') as f:
            d = json.load(f)
        if isinstance(d, dict):
            idx2label = {int(k): v for k, v in d.get("idx2label", {}).items()}
        if idx2label:
            print("\n[참고] idx2label 존재: 샘플 리포트는 검증 루프 내 classification_report 호출로 대체 가능")
    except Exception:
        pass

    plot_metrics(tr_accs, va_accs, tr_f1s, va_f1s, save_dir=args.save_dir)

if __name__ == "__main__":
    main()
