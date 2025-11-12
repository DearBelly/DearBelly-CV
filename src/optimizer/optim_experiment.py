from __future__ import annotations
import os, io, csv, sys, time, json, random
from dataclasses import dataclass
from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from PIL import Image, ImageFile
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 안전하게 손상 이미지 로딩 허용
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 프로젝트 내 공용 모듈 임포트 (기존 구조 유지)
from ai_modules.src.data_prep.dataset_precomputed import PrecomputedPillDataset
from ai_modules.src.models.model_lightcnn import LightCNN
from ai_modules.src.utils.early_stopping import EarlyStopping
from ai_modules.src.utils.seed import set_seed


# -----------------------------
# 안전 이미지 오픈(필요 시 사용)
# -----------------------------
def _safe_open_image(img_path: str, retry: int = 3, sleep: float = 0.2):
    for _ in range(retry):
        try:
            with Image.open(img_path) as im:
                return im.convert("RGB")
        except (OSError, FileNotFoundError):
            time.sleep(sleep)
    return None


# -----------------------------
# 안전 collate: None 샘플 제거
# -----------------------------
def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    imgs, labels = zip(*batch)
    return torch.stack(imgs, dim=0), torch.tensor(labels, dtype=torch.long)


# -----------------------------
# 학습/평가 루프
# -----------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_preds, total_labels = [], []
    loop = tqdm(loader, desc="🟢 Training", leave=False)
    for imgs, labels in loop:
        if imgs.numel() == 0:  # 빈 배치 방지
            continue
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        total_preds.extend(preds.detach().cpu().numpy())
        total_labels.extend(labels.detach().cpu().numpy())

        loop.set_postfix(loss=float(loss.item()))

    acc = accuracy_score(total_labels, total_preds) if total_labels else 0.0
    f1  = f1_score(total_labels, total_preds, average='weighted') if total_labels else 0.0
    return acc, f1


@torch.no_grad()
def evaluate(model, loader, device, idx2label=None):
    model.eval()
    total_preds, total_labels = [], []
    loop = tqdm(loader, desc="🔵 Evaluating", leave=False)
    for imgs, labels in loop:
        if imgs.numel() == 0:
            continue
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        preds = torch.argmax(logits, dim=1)
        total_preds.extend(preds.detach().cpu().numpy())
        total_labels.extend(labels.detach().cpu().numpy())

    acc = accuracy_score(total_labels, total_preds) if total_labels else 0.0
    f1  = f1_score(total_labels, total_preds, average='weighted') if total_labels else 0.0

    if idx2label and total_labels:
        try:
            idx_to_label = {int(k): v for k, v in idx2label.items()}
            target_names = [idx_to_label.get(i, str(i)) for i in range(len(idx_to_label))]
            print("\n[🔍 Classification Report]")
            print(classification_report(total_labels, total_preds, target_names=target_names, zero_division=0))
        except Exception:
            pass
    return acc, f1


# -----------------------------
# 데이터 로더 구성
# -----------------------------
def build_dataloaders(
    train_json: str,
    test_json_1: str,
    test_json_2: str,
    img_size: int = 64,
    batch_size: int = 32,
    num_workers: int = 0,          # Colab/Drive 안전값
    pin_memory: bool = False,
    persistent_workers: bool = False,
    seed: int = 42,
):
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    train_val_ds = PrecomputedPillDataset(train_json, transform=tfm)
    test_ds_1    = PrecomputedPillDataset(test_json_1, transform=tfm)
    test_ds_2    = PrecomputedPillDataset(test_json_2, transform=tfm)

    n_train = int(0.8 * len(train_val_ds))
    n_val   = len(train_val_ds) - n_train
    train_ds, val_ds = random_split(
        train_val_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )

    dl_args = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=safe_collate,
    )

    train_loader  = DataLoader(train_ds, shuffle=True,  **dl_args)
    val_loader    = DataLoader(val_ds,   shuffle=False, **dl_args)
    test_loader_1 = DataLoader(test_ds_1, shuffle=False, **dl_args)
    test_loader_2 = DataLoader(test_ds_2, shuffle=False, **dl_args)

    meta = {
        "num_classes": max(s['label'] for s in train_val_ds.samples) + 1,
        "idx2label": getattr(train_val_ds, "idx2label", {}),
    }
    return (train_loader, val_loader, test_loader_1, test_loader_2), meta


# -----------------------------
# 콘솔+파일 동시 로그
# -----------------------------
class _Tee(io.TextIOBase):
    def __init__(self, *streams): self.streams = streams
    def write(self, s):
        for st in self.streams: st.write(s)
        return len(s)
    def flush(self):
        for st in self.streams: st.flush()

class tee_stdout:
    def __init__(self, log_path: str):
        self.log_path = log_path
        self._orig = None
        self._fh = None
    def __enter__(self):
        self._orig = sys.stdout
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self._fh = open(self.log_path, "a", encoding="utf-8")
        sys.stdout = _Tee(self._orig, self._fh)
    def __exit__(self, exc_type, exc, tb):
        sys.stdout = self._orig
        if self._fh: self._fh.close()


# -----------------------------
# 실험 실행 엔진
# -----------------------------
@dataclass
class OptimConfig:
    name: str
    ctor: Any
    kwargs: Dict[str, Any]

def _tag(name: str) -> str:
    return name.lower().replace(" ", "_")

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _csv_header(csv_path: str):
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_acc", "train_f1", "val_acc", "val_f1", "best_val_acc_so_far"])

def _csv_append(csv_path: str, epoch: int, tr_acc: float, tr_f1: float, va_acc: float, va_f1: float, best: float):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([epoch, f"{tr_acc:.6f}", f"{tr_f1:.6f}", f"{va_acc:.6f}", f"{va_f1:.6f}", f"{best:.6f}"])


def run_experiment_for(
    opt_cfg: OptimConfig,
    train_json: str,
    test_json_1: str,
    test_json_2: str,
    save_dir: str,
    img_size: int = 64,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
    max_epochs: int = 100,
    patience: int = 7,
    delta: float = 0.001,
) -> Dict[str, Any]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ensure_dir(save_dir)

    tag = _tag(opt_cfg.name)
    csv_path     = os.path.join(save_dir, f"metrics_{tag}.csv")
    log_path     = os.path.join(save_dir, f"train_log_{tag}.txt")
    summary_path = os.path.join(save_dir, f"summary_{tag}.txt")
    best_pth     = os.path.join(save_dir, f"best_model_{tag}.pth")
    best_pt      = os.path.join(save_dir, f"best_model_{tag}.pt")
    early_pth    = os.path.join(save_dir, f"earlystop_{tag}.pth")

    _csv_header(csv_path)

    loaders, meta = build_dataloaders(
        train_json, test_json_1, test_json_2,
        img_size=img_size, batch_size=batch_size, num_workers=num_workers, seed=seed
    )
    train_loader, val_loader, test_loader_1, test_loader_2 = loaders
    num_classes = meta["num_classes"]
    idx2label   = meta["idx2label"]

    model = LightCNN(num_classes=num_classes).to(device)
    optimizer = opt_cfg.ctor(model.parameters(), **opt_cfg.kwargs)
    criterion = nn.CrossEntropyLoss()
    stopper = EarlyStopping(patience=patience, delta=delta, path=early_pth)

    best_val, best_epoch = 0.0, -1

    with tee_stdout(log_path):
        print("\n" + "=" * 80)
        print(f"🚀 Start training with optimizer: {opt_cfg.name}")
        print("=" * 80)

        for epoch in range(1, max_epochs + 1):
            print(f"\n[{opt_cfg.name}] Epoch {epoch}/{max_epochs}")
            tr_acc, tr_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
            va_acc, va_f1 = evaluate(model, val_loader, device, idx2label=idx2label)

            best_val = max(best_val, va_acc)
            _csv_append(csv_path, epoch, tr_acc, tr_f1, va_acc, va_f1, best_val)

            if va_acc >= best_val:
                best_epoch = epoch
                torch.save({
                    "epoch": best_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "optimizer_name": opt_cfg.name,
                }, best_pth)
                torch.save(model, best_pt)
                print(f"⭐️ [{opt_cfg.name}] Best updated @ epoch {best_epoch} (val_acc={va_acc:.4f})")

            print(f"Train Acc: {tr_acc:.2%} | F1: {tr_f1:.4f}")
            print(f"Val   Acc: {va_acc:.2%} | F1: {va_f1:.4f}")

            stopper(va_acc, model)
            if stopper.early_stop:
                print(f"\n⛔️ [{opt_cfg.name}] Early stopping at epoch {epoch}")
                break

        # EarlyStop 가중치로 테스트
        if os.path.exists(early_pth):
            model.load_state_dict(torch.load(early_pth, map_location=device))

        print(f"\n🧪 [{opt_cfg.name}] Test on fortest.json")
        t1_acc, t1_f1 = evaluate(model, test_loader_1, device, idx2label=idx2label)
        print(f"[{opt_cfg.name}] Test1 Acc: {t1_acc:.2%} | F1: {t1_f1:.4f}")

        print(f"\n🧪 [{opt_cfg.name}] Test on matched_test_18_deduped.json")
        t2_acc, t2_f1 = evaluate(model, test_loader_2, device, idx2label=idx2label)
        print(f"[{opt_cfg.name}] Test2 Acc: {t2_acc:.2%} | F1: {t2_f1:.4f}")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"[{opt_cfg.name}] Summary\n")
        f.write(f"- Best Val Acc : {best_val:.6f} @ epoch {best_epoch}\n")
        f.write(f"- CSV          : {os.path.basename(csv_path)}\n")
        f.write(f("- Log          : {os.path.basename(log_path)}\n"))
        f.write(f"- Best Model   : {os.path.basename(best_pth)} / {os.path.basename(best_pt)}\n")
        f.write(f("- EarlyStop    : {os.path.basename(early_pth)}\n"))
        f.write(f"- Test1 Acc/F1 : {t1_acc:.6f} / {t1_f1:.6f}\n")
        f.write(f"- Test2 Acc/F1 : {t2_acc:.6f} / {t2_f1:.6f}\n")

    return {
        "tag": tag,
        "best_val_acc": best_val,
        "best_epoch": best_epoch,
        "csv_path": csv_path,
        "log_path": log_path,
        "summary_path": summary_path,
        "best_model_pth": best_pth,
        "best_model_pt": best_pt,
        "early_path": early_pth,
        "test1": {"acc": t1_acc, "f1": t1_f1},
        "test2": {"acc": t2_acc, "f1": t2_f1},
    }


# -----------------------------
# CSV에서 곡선 그리기
# -----------------------------
def plot_from_csvs(csv_paths: List[str], out_png: str | None = None):
    histories = {}
    for p in csv_paths:
        name = os.path.splitext(os.path.basename(p))[0].replace("metrics_", "")
        val_acc, val_f1 = [], []
        with open(p, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                val_acc.append(float(row["val_acc"]))
                val_f1.append(float(row["val_f1"]))
        histories[name] = {"val_acc": val_acc, "val_f1": val_f1}

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for name, h in histories.items():
        plt.plot(h["val_acc"], label=name)
    plt.title("Validation Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Acc")
    plt.grid(True); plt.legend()

    plt.subplot(1, 2, 2)
    for name, h in histories.items():
        plt.plot(h["val_f1"], label=name)
    plt.title("Validation F1"); plt.xlabel("Epoch"); plt.ylabel("F1")
    plt.grid(True); plt.legend()

    plt.tight_layout()
    if out_png:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        print(f"[Saved] {out_png}")
    else:
        plt.show()
