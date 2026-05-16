from __future__ import annotations

from pathlib import Path
from typing import Dict

import cv2
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.models.video import r2plus1d_18

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class R2Plus1DMultiHeadClassifier(nn.Module):
    def __init__(self, num_classes_dict: dict[str, int]):
        super().__init__()
        self.backbone = r2plus1d_18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.heads = nn.ModuleDict({
            key: nn.Linear(in_features, n)
            for key, n in num_classes_dict.items()
        })

    def forward(self, x):
        # [B, T, C, H, W] -> [B, C, T, H, W]
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        feat = self.backbone(x)
        return {key: head(feat) for key, head in self.heads.items()}


def _read_video(video_path: str | Path, frame_size: int, expected_frames: int) -> torch.Tensor:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (frame_size, frame_size), interpolation=cv2.INTER_LINEAR)
        frame = TF.to_tensor(frame)
        frame = TF.normalize(frame, IMAGENET_MEAN, IMAGENET_STD)
        frames.append(frame)
    cap.release()

    if not frames:
        frames = [torch.zeros(3, frame_size, frame_size)]

    if len(frames) > expected_frames:
        indices = torch.linspace(0, len(frames) - 1, expected_frames).long().tolist()
        frames = [frames[i] for i in indices]

    while len(frames) < expected_frames:
        frames.append(frames[-1].clone())

    return torch.stack(frames, dim=0).unsqueeze(0)  # [1, T, C, H, W]


class VideoClassifier:
    def __init__(
        self,
        weights_path: str | Path,
        device: str = "cpu",
        frame_size: int = 224,
        expected_frames: int = 150,
    ):
        weights_path = Path(weights_path)
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.frame_size = frame_size
        self.expected_frames = expected_frames

        ckpt = torch.load(weights_path, map_location=self.device)
        self.label_mappings: dict = ckpt["label_mappings"]
        self.label_keys: list[str] = list(self.label_mappings.keys())

        num_classes_dict = {
            key: self.label_mappings[key]["num_classes"]
            for key in self.label_keys
        }
        self.model = R2Plus1DMultiHeadClassifier(num_classes_dict)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, video_path: str | Path) -> Dict[str, Dict]:
        """각 레이블 키에 대해 class_name + score 반환."""
        video_tensor = _read_video(
            video_path, self.frame_size, self.expected_frames
        ).to(self.device)
        outputs = self.model(video_tensor)

        result = {}
        for key, logits in outputs.items():
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = int(probs.argmax().item())
            idx_to_label = self.label_mappings[key]["idx_to_label"]
            # JSON 직렬화 시 int key가 str로 바뀌므로 양쪽 모두 시도
            pred_label = idx_to_label.get(pred_idx, idx_to_label.get(str(pred_idx)))
            result[key] = {
                "class_name": pred_label,
                "score": round(float(probs[pred_idx].item()), 4),
            }
        return result
