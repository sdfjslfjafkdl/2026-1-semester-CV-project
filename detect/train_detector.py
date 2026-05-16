import json
import random
from pathlib import Path
from collections import defaultdict

from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchmetrics.detection.mean_ap import MeanAveragePrecision


# =========================================================
# Config
# =========================================================

configs = {
    "experiment_name": "faster_rcnn_baseline",
    "batch_size": 8,
    "num_workers": 4,
    "epochs": 50,
    "lr": 0.005,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "step_size": 15,
    "gamma": 0.1,
    "seed": 42,
    "device": "cuda:0",
}


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DETECT_ROOT = PROJECT_ROOT / "detect"
PROCESSED_ROOT = DETECT_ROOT / "img_data" / "processed"

TRAIN_COCO_PATH = PROCESSED_ROOT / "train_coco.json"
VAL_COCO_PATH = PROCESSED_ROOT / "val_coco.json"
TEST_COCO_PATH = PROCESSED_ROOT / "test_coco.json"

OUTPUT_ROOT = DETECT_ROOT / "detection_outputs"
CHECKPOINT_ROOT = OUTPUT_ROOT / "checkpoints"
LOG_ROOT = OUTPUT_ROOT / "logs"


def ensure_dirs():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)


def print_paths():
    print("[경로 확인]")
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("DETECT_ROOT:", DETECT_ROOT)
    print("PROCESSED_ROOT:", PROCESSED_ROOT)
    print("TRAIN_COCO_PATH:", TRAIN_COCO_PATH)
    print("VAL_COCO_PATH:", VAL_COCO_PATH)
    print("TEST_COCO_PATH:", TEST_COCO_PATH)
    print("OUTPUT_ROOT:", OUTPUT_ROOT)
    print("CHECKPOINT_ROOT:", CHECKPOINT_ROOT)
    print("TRAIN_COCO exists?", TRAIN_COCO_PATH.exists())
    print("VAL_COCO exists?", VAL_COCO_PATH.exists())
    print("TEST_COCO exists?", TEST_COCO_PATH.exists())


# =========================================================
# Dataset
# =========================================================

class CocoDetectionDataset(Dataset):
    def __init__(self, coco_json_path, project_root, transforms=None):
        self.coco_json_path = Path(coco_json_path)
        self.project_root = Path(project_root)
        self.transforms = transforms

        if not self.coco_json_path.exists():
            raise FileNotFoundError(f"COCO json 파일이 없습니다: {self.coco_json_path}")

        with open(self.coco_json_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.annotations = coco["annotations"]
        self.categories = sorted(coco["categories"], key=lambda x: x["id"])

        self.category_id_to_name = {
            cat["id"]: cat["name"]
            for cat in self.categories
        }

        # Faster R-CNN은 label 0을 background로 사용하므로 실제 class는 1부터 시작
        self.category_id_to_contiguous = {
            cat["id"]: idx + 1
            for idx, cat in enumerate(self.categories)
        }

        self.contiguous_to_category_id = {
            v: k
            for k, v in self.category_id_to_contiguous.items()
        }

        self.image_id_to_annotations = defaultdict(list)

        for ann in self.annotations:
            if ann.get("iscrowd", 0) == 1:
                continue

            self.image_id_to_annotations[ann["image_id"]].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_info = self.images[idx]
        image_id = image_info["id"]

        image_path = self.project_root / image_info["file_name"]

        if not image_path.exists():
            raise FileNotFoundError(f"이미지 파일이 없습니다: {image_path}")

        image = Image.open(image_path).convert("RGB")
        anns = self.image_id_to_annotations.get(image_id, [])

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in anns:
            x, y, w, h = ann["bbox"]

            if w <= 0 or h <= 0:
                continue

            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h

            boxes.append([x1, y1, x2, y2])
            labels.append(self.category_id_to_contiguous[ann["category_id"]])
            areas.append(ann.get("area", w * h))
            iscrowd.append(ann.get("iscrowd", 0))

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            areas = torch.tensor(areas, dtype=torch.float32)
            iscrowd = torch.tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": areas,
            "iscrowd": iscrowd,
        }

        image = F.to_tensor(image)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


# =========================================================
# Model
# =========================================================

def get_model(num_classes):
    """
    num_classes는 background 포함 개수.
    예:
      실제 클래스가 2개면 num_classes = 3
      background 0, vehicle 1, traffic-light-etc 2
    """

    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes,
    )

    return model


# =========================================================
# Train / Eval
# =========================================================

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=100):
    model.train()

    running_loss = 0.0

    for step, (images, targets) in enumerate(data_loader):
        images = [
            img.to(device)
            for img in images
        ]

        targets = [
            {
                k: v.to(device)
                for k, v in t.items()
            }
            for t in targets
        ]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        loss_value = losses.item()
        running_loss += loss_value

        if step % print_freq == 0:
            loss_str = ", ".join(
                [
                    f"{k}: {v.item():.4f}"
                    for k, v in loss_dict.items()
                ]
            )

            print(
                f"[Epoch {epoch}][Step {step}/{len(data_loader)}] "
                f"total_loss={loss_value:.4f} | {loss_str}"
            )

    avg_loss = running_loss / max(len(data_loader), 1)

    return avg_loss


@torch.no_grad()
def evaluate_loss(model, data_loader, device):
    """
    torchvision detection model은 eval mode에서는 loss가 아니라 prediction을 반환함.
    그래서 validation/test loss를 계산하려면 model.train() 상태로 두고 no_grad()를 사용해야 함.
    """

    was_training = model.training
    model.train()

    total_loss = 0.0

    for images, targets in data_loader:
        images = [
            img.to(device)
            for img in images
        ]

        targets = [
            {
                k: v.to(device)
                for k, v in t.items()
            }
            for t in targets
        ]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        total_loss += losses.item()

    if not was_training:
        model.eval()

    avg_loss = total_loss / max(len(data_loader), 1)

    return avg_loss


@torch.no_grad()
def evaluate_map(model, data_loader, device):
    """
    COCO-style bbox mAP 계산.

    반환값 주요 항목:
      map     : AP@[IoU=0.50:0.95]
      map_50  : AP@IoU=0.50
      map_75  : AP@IoU=0.75
      mar_100 : AR@maxDets=100
    """

    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox")

    for images, targets in data_loader:
        images = [
            img.to(device)
            for img in images
        ]

        outputs = model(images)

        preds = []
        gts = []

        for output, target in zip(outputs, targets):
            preds.append(
                {
                    "boxes": output["boxes"].detach().cpu(),
                    "scores": output["scores"].detach().cpu(),
                    "labels": output["labels"].detach().cpu(),
                }
            )

            gts.append(
                {
                    "boxes": target["boxes"].detach().cpu(),
                    "labels": target["labels"].detach().cpu(),
                    "area": target["area"].detach().cpu(),
                    "iscrowd": target["iscrowd"].detach().cpu(),
                }
            )

        metric.update(preds, gts)

    result = metric.compute()
    result = tensor_dict_to_python(result)

    return result


# =========================================================
# Utils
# =========================================================

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor_to_python(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return float(value.item())
        return value.tolist()
    return value


def tensor_dict_to_python(data):
    return {
        k: tensor_to_python(v)
        for k, v in data.items()
    }


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def plot_metrics(history, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = history["epoch"]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path.parent / "loss_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["val_map"], label="val_mAP")
    plt.plot(epochs, history["val_map_50"], label="val_mAP@50")
    plt.plot(epochs, history["val_map_75"], label="val_mAP@75")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.title("Validation mAP Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path.parent / "map_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["lr"], label="lr")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path.parent / "lr_curve.png", dpi=200, bbox_inches="tight")
    plt.close()


# =========================================================
# Main
# =========================================================

def main():
    set_seed(configs["seed"])

    ensure_dirs()
    print_paths()

    if not TRAIN_COCO_PATH.exists():
        raise FileNotFoundError(
            f"train_coco.json이 없습니다: {TRAIN_COCO_PATH}\n"
            f"먼저 python detect/process.py 를 실행하세요."
        )

    if not VAL_COCO_PATH.exists():
        raise FileNotFoundError(
            f"val_coco.json이 없습니다: {VAL_COCO_PATH}\n"
            f"먼저 python detect/process.py 를 실행하세요."
        )

    if not TEST_COCO_PATH.exists():
        raise FileNotFoundError(
            f"test_coco.json이 없습니다: {TEST_COCO_PATH}\n"
            f"train/val/test split 생성 코드에서 test_coco.json도 만들었는지 확인하세요."
        )

    experiment_name = configs["experiment_name"]

    work_dir = CHECKPOINT_ROOT / experiment_name
    work_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = CocoDetectionDataset(
        coco_json_path=TRAIN_COCO_PATH,
        project_root=PROJECT_ROOT,
        transforms=None,
    )

    val_dataset = CocoDetectionDataset(
        coco_json_path=VAL_COCO_PATH,
        project_root=PROJECT_ROOT,
        transforms=None,
    )

    test_dataset = CocoDetectionDataset(
        coco_json_path=TEST_COCO_PATH,
        project_root=PROJECT_ROOT,
        transforms=None,
    )

    categories = train_dataset.categories
    classes = tuple(cat["name"] for cat in categories)
    num_classes = len(classes) + 1

    print("\n[클래스 정보]")
    print("classes:", classes)
    print("num_classes_without_background:", len(classes))
    print("num_classes_with_background:", num_classes)
    print("category_id_to_contiguous:", train_dataset.category_id_to_contiguous)
    print("contiguous_to_category_id:", train_dataset.contiguous_to_category_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=configs["batch_size"],
        shuffle=True,
        num_workers=configs["num_workers"],
        pin_memory=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=configs["batch_size"],
        shuffle=False,
        num_workers=configs["num_workers"],
        pin_memory=True,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=configs["batch_size"],
        shuffle=False,
        num_workers=configs["num_workers"],
        pin_memory=True,
        collate_fn=collate_fn,
    )

    device = configs["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    print(f"\n[Device] {device}")

    model = get_model(num_classes=num_classes)
    model.to(device)

    params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.SGD(
        params,
        lr=configs["lr"],
        momentum=configs["momentum"],
        weight_decay=configs["weight_decay"],
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=configs["step_size"],
        gamma=configs["gamma"],
    )

    num_epochs = configs["epochs"]

    best_val_map = -1.0
    best_epoch = -1

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_map": [],
        "val_map_50": [],
        "val_map_75": [],
        "val_mar_100": [],
        "lr": [],
    }

    class_info = {
        "classes": classes,
        "num_classes_without_background": len(classes),
        "num_classes_with_background": num_classes,
        "category_id_to_contiguous": train_dataset.category_id_to_contiguous,
        "contiguous_to_category_id": train_dataset.contiguous_to_category_id,
    }

    save_json(class_info, work_dir / "class_info.json")
    save_json(configs, work_dir / "configs.json")

    print("\n[학습 시작]")

    for epoch in range(1, num_epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            print_freq=100,
        )

        val_loss = evaluate_loss(
            model=model,
            data_loader=val_loader,
            device=device,
        )

        val_map_result = evaluate_map(
            model=model,
            data_loader=val_loader,
            device=device,
        )

        val_map = val_map_result["map"]
        val_map_50 = val_map_result["map_50"]
        val_map_75 = val_map_result["map_75"]
        val_mar_100 = val_map_result["mar_100"]

        lr_scheduler.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_map"].append(val_map)
        history["val_map_50"].append(val_map_50)
        history["val_map_75"].append(val_map_75)
        history["val_mar_100"].append(val_mar_100)
        history["lr"].append(current_lr)

        save_json(history, work_dir / "metrics_history.json")
        save_json(history, work_dir / "loss_history.json")  # 기존 그래프 코드 호환용
        plot_metrics(history, work_dir / "curves.png")

        print(
            f"[Epoch {epoch}] "
            f"lr={current_lr:.6f}, "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_mAP={val_map:.4f}, "
            f"val_mAP@50={val_map_50:.4f}, "
            f"val_mAP@75={val_map_75:.4f}, "
            f"val_mAR@100={val_mar_100:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_map": val_map,
            "val_map_50": val_map_50,
            "val_map_75": val_map_75,
            "val_mar_100": val_mar_100,
            "val_map_result": val_map_result,
            "classes": classes,
            "num_classes_without_background": len(classes),
            "num_classes_with_background": num_classes,
            "category_id_to_contiguous": train_dataset.category_id_to_contiguous,
            "contiguous_to_category_id": train_dataset.contiguous_to_category_id,
            "configs": configs,
        }

        torch.save(checkpoint, work_dir / "latest.pth")

        if val_map > best_val_map:
            best_val_map = val_map
            best_epoch = epoch

            torch.save(checkpoint, work_dir / "best.pth")

            print(
                f"[Best] best_val_mAP updated to {best_val_map:.4f} "
                f"at epoch {best_epoch}"
            )

    print("\n[Best checkpoint test 평가 시작]")

    best_checkpoint_path = work_dir / "best.pth"

    if not best_checkpoint_path.exists():
        print("[Warning] best.pth가 없어서 현재 모델로 test 평가를 진행함.")
    else:
        best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(best_checkpoint["model_state_dict"])
        model.to(device)

    test_loss = evaluate_loss(
        model=model,
        data_loader=test_loader,
        device=device,
    )

    test_map_result = evaluate_map(
        model=model,
        data_loader=test_loader,
        device=device,
    )

    best_ckpt_for_result = torch.load(best_checkpoint_path, map_location=device) if best_checkpoint_path.exists() else checkpoint

    final_results = {
        "best_epoch": best_ckpt_for_result["epoch"],
        "best_val_loss": best_ckpt_for_result["val_loss"],
        "best_val_map": best_ckpt_for_result["val_map"],
        "best_val_map_50": best_ckpt_for_result["val_map_50"],
        "best_val_map_75": best_ckpt_for_result["val_map_75"],
        "test_loss": test_loss,
        "test_map": test_map_result["map"],
        "test_map_50": test_map_result["map_50"],
        "test_map_75": test_map_result["map_75"],
        "test_mar_100": test_map_result["mar_100"],
        "test_map_result": test_map_result,
    }

    save_json(final_results, work_dir / "test_results.json")

    print("\n[최종 Test 결과]")
    print(f"best_epoch: {final_results['best_epoch']}")
    print(f"best_val_mAP: {final_results['best_val_map']:.4f}")
    print(f"test_loss: {final_results['test_loss']:.4f}")
    print(f"test_mAP: {final_results['test_map']:.4f}")
    print(f"test_mAP@50: {final_results['test_map_50']:.4f}")
    print(f"test_mAP@75: {final_results['test_map_75']:.4f}")
    print(f"test_mAR@100: {final_results['test_mar_100']:.4f}")

    print("\n[학습 완료]")
    print("work_dir:", work_dir)
    print("latest checkpoint:", work_dir / "latest.pth")
    print("best checkpoint:", work_dir / "best.pth")
    print("metrics history:", work_dir / "metrics_history.json")
    print("loss curve:", work_dir / "loss_curve.png")
    print("mAP curve:", work_dir / "map_curve.png")
    print("test results:", work_dir / "test_results.json")


if __name__ == "__main__":
    main()
