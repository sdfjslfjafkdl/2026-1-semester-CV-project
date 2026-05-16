import csv
import json
import time
import random
import argparse
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights


# =========================
# Paths
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent

VIDEO_PROCESSED_DIR = PROJECT_ROOT / "classification" / "video_data" / "processed"

DEFAULT_TRAIN_JSON = VIDEO_PROCESSED_DIR / "train.json"
DEFAULT_VAL_JSON = VIDEO_PROCESSED_DIR / "val.json"
DEFAULT_TEST_JSON = VIDEO_PROCESSED_DIR / "test.json"

OUTPUT_DIR = PROJECT_ROOT / "classification" / "video_classification_outputs"


# =========================
# ImageNet normalization
# =========================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =========================
# Args
# =========================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_json", type=str, default=str(DEFAULT_TRAIN_JSON))
    parser.add_argument("--val_json", type=str, default=str(DEFAULT_VAL_JSON))
    parser.add_argument("--test_json", type=str, default=str(DEFAULT_TEST_JSON))
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))

    parser.add_argument(
        "--label_keys",
        nargs="+",
        default=[
            "accident_place_feature",
            "vehicle_a_progress_info",
            "vehicle_b_progress_info",
            "traffic_accident_type",
        ],
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--frame_size", type=int, default=224)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use_amp", action="store_true")

    parser.add_argument("--resume", type=str, default=None)

    return parser.parse_args()


# =========================
# Utils
# =========================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_csv_log(path: Path, history: List[Dict]):
    if len(history) == 0:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def build_label_mappings(train_data, val_data, test_data, label_keys):
    label_mappings = {}

    all_data = train_data + val_data + test_data

    for key in label_keys:
        values = []

        for item in all_data:
            if key not in item:
                raise KeyError(f"label key '{key}' not found in item: {item}")

            values.append(str(item[key]))

        unique_values = sorted(set(values))

        label_to_idx = {label: idx for idx, label in enumerate(unique_values)}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}

        label_mappings[key] = {
            "label_to_idx": label_to_idx,
            "idx_to_label": idx_to_label,
            "num_classes": len(unique_values),
        }

    return label_mappings


def attach_label_indices(data, label_keys, label_mappings):
    samples = []

    for item in data:
        video_path = Path(item["video_path"])

        if not video_path.exists():
            print(f"[Warning] video not found: {video_path}")
            continue

        labels = {}

        for key in label_keys:
            raw_label = str(item[key])
            labels[key] = label_mappings[key]["label_to_idx"][raw_label]

        sample = {
            "video_path": str(video_path),
            "labels": labels,
            "raw_labels": {key: str(item[key]) for key in label_keys},
        }

        samples.append(sample)

    return samples


# =========================
# Dataset
# =========================

class AccidentVideoDataset(Dataset):
    def __init__(
        self,
        samples,
        label_keys,
        frame_size=224,
    ):
        self.samples = samples
        self.label_keys = label_keys
        self.frame_size = frame_size

        self.black = torch.zeros(3, frame_size, frame_size)

    def __len__(self):
        return len(self.samples)

    def _read_frame_tensor(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(
            frame,
            (self.frame_size, self.frame_size),
            interpolation=cv2.INTER_LINEAR,
        )

        x = TF.to_tensor(frame)
        x = TF.normalize(x, IMAGENET_MEAN, IMAGENET_STD)

        return x

    def _load_full_video(self, video_path):
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"[Warning] failed to open video: {video_path}")
            return torch.stack([self.black.clone()], dim=0)

        frames = []

        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                break

            x = self._read_frame_tensor(frame)
            frames.append(x)

        cap.release()

        if len(frames) == 0:
            print(f"[Warning] no frames found in video: {video_path}")
            return torch.stack([self.black.clone()], dim=0)

        video = torch.stack(frames, dim=0)  # [T, C, H, W]

        return video

    def __getitem__(self, idx):
        sample = self.samples[idx]

        video_path = Path(sample["video_path"])
        video = self._load_full_video(video_path)

        labels = {}

        for key in self.label_keys:
            labels[key] = torch.tensor(sample["labels"][key], dtype=torch.long)

        return video, labels


def full_video_collate_fn(batch):
    expected_t = 150

    videos, labels_list = zip(*batch)

    for i, video in enumerate(videos):
        t = video.shape[0]

        if t != expected_t:
            raise ValueError(
                f"Invalid number of frames in batch item {i}: "
                f"expected {expected_t}, got {t}. "
                f"모든 영상은 반드시 {expected_t}프레임이어야 함."
            )

    videos = torch.stack(videos, dim=0)  # [B, 150, C, H, W]

    labels = {}
    label_keys = labels_list[0].keys()

    for key in label_keys:
        labels[key] = torch.stack(
            [label_dict[key] for label_dict in labels_list],
            dim=0,
        )

    return videos, labels


# =========================
# Model
# =========================

class R2Plus1DMultiHeadClassifier(nn.Module):
    def __init__(self, num_classes_dict: Dict[str, int], freeze_backbone=True):
        super().__init__()

        self.backbone = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.heads = nn.ModuleDict()

        for key, num_classes in num_classes_dict.items():
            self.heads[key] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # Dataset output: [B, T, C, H, W]
        # r2plus1d input: [B, C, T, H, W]
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        feat = self.backbone(x)

        outputs = {}

        for key, head in self.heads.items():
            outputs[key] = head(feat)

        return outputs


# =========================
# Train / Eval
# =========================

def move_labels_to_device(labels, device):
    return {key: value.to(device, non_blocking=True) for key, value in labels.items()}


def compute_multitask_loss_and_acc(outputs, labels, criterion, label_keys):
    total_loss = 0.0
    acc_dict = {}

    for key in label_keys:
        logits = outputs[key]
        target = labels[key]

        loss = criterion(logits, target)
        total_loss = total_loss + loss

        pred = logits.argmax(dim=1)
        correct = (pred == target).sum().item()
        acc_dict[key] = correct

    return total_loss, acc_dict


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp, label_keys):
    model.train()

    total_loss = 0.0
    total_correct = {key: 0 for key in label_keys}
    total_count = 0

    pbar = tqdm(loader, desc="train", leave=False)

    for videos, labels in pbar:
        videos = videos.to(device, non_blocking=True)
        labels = move_labels_to_device(labels, device)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(videos)
                loss, correct_dict = compute_multitask_loss_and_acc(
                    outputs=outputs,
                    labels=labels,
                    criterion=criterion,
                    label_keys=label_keys,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        else:
            outputs = model(videos)
            loss, correct_dict = compute_multitask_loss_and_acc(
                outputs=outputs,
                labels=labels,
                criterion=criterion,
                label_keys=label_keys,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        bs = videos.size(0)

        total_loss += loss.item() * bs
        total_count += bs

        for key in label_keys:
            total_correct[key] += correct_dict[key]

        avg_loss = total_loss / max(total_count, 1)
        avg_acc = {
            key: total_correct[key] / max(total_count, 1)
            for key in label_keys
        }
        mean_acc = sum(avg_acc.values()) / len(label_keys)

        postfix = {"loss": avg_loss, "mean_acc": mean_acc}

        for key in label_keys:
            postfix[f"{key}_acc"] = avg_acc[key]

        pbar.set_postfix(postfix)

    avg_loss = total_loss / total_count
    avg_acc = {
        key: total_correct[key] / total_count
        for key in label_keys
    }
    mean_acc = sum(avg_acc.values()) / len(label_keys)

    return avg_loss, avg_acc, mean_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device, label_keys, split_name="val"):
    model.eval()

    total_loss = 0.0
    total_correct = {key: 0 for key in label_keys}
    total_count = 0

    pbar = tqdm(loader, desc=split_name, leave=False)

    for videos, labels in pbar:
        videos = videos.to(device, non_blocking=True)
        labels = move_labels_to_device(labels, device)

        outputs = model(videos)

        loss, correct_dict = compute_multitask_loss_and_acc(
            outputs=outputs,
            labels=labels,
            criterion=criterion,
            label_keys=label_keys,
        )

        bs = videos.size(0)

        total_loss += loss.item() * bs
        total_count += bs

        for key in label_keys:
            total_correct[key] += correct_dict[key]

        avg_loss = total_loss / max(total_count, 1)
        avg_acc = {
            key: total_correct[key] / max(total_count, 1)
            for key in label_keys
        }
        mean_acc = sum(avg_acc.values()) / len(label_keys)

        postfix = {"loss": avg_loss, "mean_acc": mean_acc}

        for key in label_keys:
            postfix[f"{key}_acc"] = avg_acc[key]

        pbar.set_postfix(postfix)

    avg_loss = total_loss / total_count
    avg_acc = {
        key: total_correct[key] / total_count
        for key in label_keys
    }
    mean_acc = sum(avg_acc.values()) / len(label_keys)

    return avg_loss, avg_acc, mean_acc


# =========================
# Save / Load
# =========================

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    best_val_score,
    args,
    label_mappings,
):
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_val_score": best_val_score,
        "args": vars(args),
        "label_mappings": label_mappings,
    }

    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["model_state"])

    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])

    start_epoch = ckpt.get("epoch", 0) + 1
    best_val_score = ckpt.get("best_val_score", 0.0)

    return start_epoch, best_val_score


def save_curves(output_dir, history, label_keys):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(history) == 0:
        return

    epochs = [h["epoch"] for h in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [h["train_loss"] for h in history], label="train_loss")
    plt.plot(epochs, [h["val_loss"] for h in history], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [h["train_mean_acc"] for h in history], label="train_mean_acc")
    plt.plot(epochs, [h["val_mean_acc"] for h in history], label="val_mean_acc")

    for key in label_keys:
        plt.plot(epochs, [h[f"train_{key}_acc"] for h in history], label=f"train_{key}_acc")
        plt.plot(epochs, [h[f"val_{key}_acc"] for h in history], label=f"val_{key}_acc")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend(fontsize=7)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "acc_curve.png", dpi=200)
    plt.close()


# =========================
# Main
# =========================

def main():
    args = parse_args()
    set_seed(args.seed)

    train_json = Path(args.train_json)
    val_json = Path(args.val_json)
    test_json = Path(args.test_json)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not train_json.exists():
        raise FileNotFoundError(f"train_json not found: {train_json}")

    if not val_json.exists():
        raise FileNotFoundError(f"val_json not found: {val_json}")

    if not test_json.exists():
        raise FileNotFoundError(f"test_json not found: {test_json}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA를 사용할 수 없음. nvidia-smi와 torch CUDA 설치를 확인해야 함.")

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    label_keys = args.label_keys

    print("=" * 80)
    print("Accident Video Multi-Task Classification Training")
    print("=" * 80)
    print(f"train_json:  {train_json}")
    print(f"val_json:    {val_json}")
    print(f"test_json:   {test_json}")
    print(f"output_dir:  {output_dir}")
    print(f"device:      {device}")
    print(f"label_keys:  {label_keys}")
    print(f"frame_size:  {args.frame_size}")
    print(f"batch_size:  {args.batch_size}")
    print(f"epochs:      {args.epochs}")
    print(f"use_amp:     {args.use_amp}")
    print("frame mode:  full video frames")
    print("augment:     False")
    print("=" * 80)

    train_data = load_json(train_json)
    val_data = load_json(val_json)
    test_data = load_json(test_json)

    print(f"raw train samples: {len(train_data)}")
    print(f"raw val samples:   {len(val_data)}")
    print(f"raw test samples:  {len(test_data)}")

    label_mappings = build_label_mappings(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        label_keys=label_keys,
    )

    save_json(output_dir / "label_mappings.json", label_mappings)

    for key in label_keys:
        save_json(output_dir / f"{key}_to_idx.json", label_mappings[key]["label_to_idx"])
        save_json(output_dir / f"idx_to_{key}.json", label_mappings[key]["idx_to_label"])

    train_samples = attach_label_indices(
        train_data,
        label_keys=label_keys,
        label_mappings=label_mappings,
    )

    val_samples = attach_label_indices(
        val_data,
        label_keys=label_keys,
        label_mappings=label_mappings,
    )

    test_samples = attach_label_indices(
        test_data,
        label_keys=label_keys,
        label_mappings=label_mappings,
    )

    if len(train_samples) == 0:
        raise ValueError("train samples가 비어 있음.")

    if len(val_samples) == 0:
        raise ValueError("val samples가 비어 있음.")

    if len(test_samples) == 0:
        raise ValueError("test samples가 비어 있음.")

    num_classes_dict = {
        key: label_mappings[key]["num_classes"]
        for key in label_keys
    }

    print(f"valid train samples: {len(train_samples)}")
    print(f"valid val samples:   {len(val_samples)}")
    print(f"valid test samples:  {len(test_samples)}")

    for key in label_keys:
        print(f"num_classes[{key}]: {num_classes_dict[key]}")

    train_ds = AccidentVideoDataset(
        train_samples,
        label_keys=label_keys,
        frame_size=args.frame_size,
    )

    val_ds = AccidentVideoDataset(
        val_samples,
        label_keys=label_keys,
        frame_size=args.frame_size,
    )

    test_ds = AccidentVideoDataset(
        test_samples,
        label_keys=label_keys,
        frame_size=args.frame_size,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
        collate_fn=full_video_collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
        collate_fn=full_video_collate_fn,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
        collate_fn=full_video_collate_fn,
    )

    model = R2Plus1DMultiHeadClassifier(
        num_classes_dict=num_classes_dict,
        freeze_backbone=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )

    use_amp = bool(args.use_amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 1
    best_val_score = 0.0
    history = []

    if args.resume is not None:
        resume_path = Path(args.resume)

        if resume_path.exists():
            start_epoch, best_val_score = load_checkpoint(
                resume_path,
                model,
                optimizer,
                scheduler,
                device,
            )

            print(f"resume: {resume_path}")
            print(f"start_epoch: {start_epoch}")
            print(f"best_val_score: {best_val_score:.4f}")

        else:
            print(f"[Warning] resume checkpoint not found: {resume_path}")

    best_path = output_dir / "best.pth"
    last_path = output_dir / "last.pth"
    log_csv_path = output_dir / "train_log.csv"
    log_json_path = output_dir / "train_log.json"
    test_result_path = output_dir / "test_result.json"

    print("\n학습 시작\n")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"[Epoch {epoch}/{args.epochs}] lr={current_lr:.6e}")

        train_loss, train_acc_dict, train_mean_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            label_keys=label_keys,
        )

        val_loss, val_acc_dict, val_mean_acc = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            label_keys=label_keys,
            split_name="val",
        )

        scheduler.step()

        elapsed = time.time() - t0

        val_score = val_mean_acc

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_mean_acc": train_mean_acc,
            "val_mean_acc": val_mean_acc,
            "val_score": val_score,
            "elapsed_sec": elapsed,
        }

        for key in label_keys:
            row[f"train_{key}_acc"] = train_acc_dict[key]
            row[f"val_{key}_acc"] = val_acc_dict[key]

        history.append(row)

        is_best = val_score > best_val_score

        if is_best:
            best_val_score = val_score

            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_score=best_val_score,
                args=args,
                label_mappings=label_mappings,
            )

        save_checkpoint(
            path=last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val_score=best_val_score,
            args=args,
            label_mappings=label_mappings,
        )

        save_csv_log(log_csv_path, history)
        save_json(log_json_path, history)
        save_curves(output_dir, history, label_keys)

        best_mark = " <- best" if is_best else ""

        print(
            f"train_loss={train_loss:.4f}, "
            f"train_mean_acc={train_mean_acc:.4f} | "
            f"val_loss={val_loss:.4f}, "
            f"val_mean_acc={val_mean_acc:.4f}, "
            f"val_score={val_score:.4f}{best_mark} | "
            f"time={elapsed:.1f}s"
        )

        for key in label_keys:
            print(
                f"  {key}: "
                f"train_acc={train_acc_dict[key]:.4f}, "
                f"val_acc={val_acc_dict[key]:.4f}"
            )

        print(f"saved last: {last_path}")

        if is_best:
            print(f"saved best: {best_path}")

        print()

    print("=" * 80)
    print("학습 완료")
    print(f"best_val_score: {best_val_score:.4f}")
    print(f"best checkpoint: {best_path}")
    print(f"last checkpoint: {last_path}")
    print("=" * 80)

    print("\nbest checkpoint로 test 평가 시작\n")

    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
    else:
        print("[Warning] best.pth가 없어서 현재 모델로 test 평가를 진행함.")

    test_loss, test_acc_dict, test_mean_acc = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        label_keys=label_keys,
        split_name="test",
    )

    test_result = {
        "test_loss": test_loss,
        "test_mean_acc": test_mean_acc,
    }

    for key in label_keys:
        test_result[f"test_{key}_acc"] = test_acc_dict[key]

    save_json(test_result_path, test_result)

    print("=" * 80)
    print("Test 결과")
    print(f"test_loss:     {test_loss:.4f}")
    print(f"test_mean_acc: {test_mean_acc:.4f}")

    for key in label_keys:
        print(f"test_{key}_acc: {test_acc_dict[key]:.4f}")

    print("=" * 80)
    print(f"log csv:     {log_csv_path}")
    print(f"log json:    {log_json_path}")
    print(f"test result: {test_result_path}")
    print(f"loss curve:  {output_dir / 'loss_curve.png'}")
    print(f"acc curve:   {output_dir / 'acc_curve.png'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
