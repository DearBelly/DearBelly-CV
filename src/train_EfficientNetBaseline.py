# ============================================================
# 📄 파일명: ai_modules/src/train_efficientnet_baseline.py
# 📘 목적:
#   - 직접 정의한 EfficientNetBaseline 모델을 이용해
#     사전 전처리된 약 이미지(JSON) 기반으로 학습 및 평가 수행.
#   - LightCNN보다 큰 입력 크기(128x128)와 더 깊은 백본 사용.
#   - 학습/검증 분할, 모델 저장, 테스트셋 2개 평가, 성능 그래프 포함.
#
# 🧪 실행 예시:
#   python -m ai_modules.src.train_efficientnet_baseline \
#     --train_json /content/gdrive/MyDrive/Matched/matched_train_90_original_noisy_sheared_bright.json \
#     --test_json1 /content/gdrive/MyDrive/Matched/fortest.json \
#     --test_json2 /content/gdrive/MyDrive/Matched/matched_test_18_deduped.json \
#     --save_dir /content/gdrive/MyDrive/ModelCheckpoints_baseline \
#     --epochs 30 --batch_size 32
# ============================================================

from __future__ import annotations
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt

# ✅ 네가 직접 정의했던 모듈들
from ai_modules.src.data_prep.dataset_precomputed import PrecomputedPillDataset
from ai_modules.src.models.efficientnet_baseline import EfficientNetBaseline


# ------------------------------
# 학습 루프
# ------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_preds, total_labels = [], []
    loop = tqdm(loader, desc="🟢 Training", leave=False)

    for imgs, labels in loop:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, dim=1)
        total_preds.extend(preds.cpu().numpy())
        total_labels.extend(labels.cpu().numpy())
        loop.set_postfix(loss=loss.item())

    acc = accuracy_score(total_labels, total_preds)
    f1 = f1_score(total_labels, total_preds, average='weighted')
    return acc, f1


# ------------------------------
# 검증 루프
# ------------------------------
@torch.no_grad()
def evaluate(model, loader, device, idx2label=None):
    model.eval()
    total_preds, total_labels = [], []
    loop = tqdm(loader, desc="🔵 Evaluating", leave=False)

    for imgs, labels in loop:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)
        total_preds.extend(preds.cpu().numpy())
        total_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(total_labels, total_preds)
    f1 = f1_score(total_labels, total_preds, average='weighted')

    if idx2label:
        idx_to_label = {int(k): v for k, v in idx2label.items()}
        target_names = [idx_to_label.get(i, str(i)) for i in range(len(idx_to_label))]
        print("\n[🔍 Classification Report]")
        print(classification_report(total_labels, total_preds, target_names=target_names, zero_division=0))

    return acc, f1


# ------------------------------
# 학습 결과 시각화
# ------------------------------
def plot_metrics(train_acc, val_acc, train_f1, val_f1):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(train_acc, label='Train Acc')
    plt.plot(val_acc, label='Val Acc')
    plt.title("Accuracy")
    plt.legend(); plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(train_f1, label='Train F1')
    plt.plot(val_f1, label='Val F1')
    plt.title("F1 Score")
    plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.show()


# ------------------------------
# 메인 실행부
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", required=True)
    ap.add_argument("--test_json1", default=None)
    ap.add_argument("--test_json2", default=None)
    ap.add_argument("--save_dir", default="./runs/efficientnet_baseline")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------
    # Transform 정의
    # ------------------------------
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])

    # ------------------------------
    # Dataset & Dataloader
    # ------------------------------
    train_val_dataset = PrecomputedPillDataset(args.train_json, transform=transform)
    n_train = int(0.8 * len(train_val_dataset))
    n_val = len(train_val_dataset) - n_train
    train_ds, val_ds = random_split(train_val_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    test_loader_1, test_loader_2 = None, None
    if args.test_json1:
        test_loader_1 = DataLoader(PrecomputedPillDataset(args.test_json1, transform=transform), batch_size=args.batch_size)
    if args.test_json2:
        test_loader_2 = DataLoader(PrecomputedPillDataset(args.test_json2, transform=transform), batch_size=args.batch_size)

    num_classes = len(train_val_dataset.label2idx) if train_val_dataset.label2idx else (
        max(s['label'] for s in train_val_dataset.samples) + 1
    )

    # ------------------------------
    # 모델 초기화
    # ------------------------------
    model = EfficientNetBaseline(num_classes=num_classes, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc, best_epoch = 0.0, -1
    train_acc_list, val_acc_list, train_f1_list, val_f1_list = [], [], [], []

    # ------------------------------
    # 학습 루프
    # ------------------------------
    for epoch in range(args.epochs):
        print(f"\n[Epoch {epoch+1}/{args.epochs}]")
        train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, val_f1 = evaluate(model, val_loader, device, idx2label=train_val_dataset.idx2label)

        train_acc_list.append(train_acc); val_acc_list.append(val_acc)
        train_f1_list.append(train_f1);   val_f1_list.append(val_f1)

        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch + 1
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))
            print("⭐️ Best model updated!")

        print(f"Train Acc: {train_acc:.2%} | F1: {train_f1:.4f}")
        print(f"Val   Acc: {val_acc:.2%} | F1: {val_f1:.4f}")

    print(f"\n🏁 학습 완료! Best Val Acc: {best_val_acc:.2%} (Epoch {best_epoch})")

    # ------------------------------
    # 테스트셋 평가
    # ------------------------------
    if test_loader_1:
        print("\n🧪 [Test 1 - fortest.json]")
        test_acc1, test_f1_1 = evaluate(model, test_loader_1, device, idx2label=train_val_dataset.idx2label)
        print(f"Test 1 Acc: {test_acc1:.2%} | F1: {test_f1_1:.4f}")

    if test_loader_2:
        print("\n🧪 [Test 2 - matched_test_18_deduped.json]")
        test_acc2, test_f1_2 = evaluate(model, test_loader_2, device, idx2label=train_val_dataset.idx2label)
        print(f"Test 2 Acc: {test_acc2:.2%} | F1: {test_f1_2:.4f}")

    # ------------------------------
    # 학습 곡선 출력
    # ------------------------------
    plot_metrics(train_acc_list, val_acc_list, train_f1_list, val_f1_list)


if __name__ == "__main__":
    main()
