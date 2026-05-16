from __future__ import annotations

from pathlib import Path

import cv2
import torch
import torchvision.transforms.functional as TF
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from accident_liability.schemas import Detection


COCO_TO_TRAFFIC_LABEL = {
    "car": "vehicle",
    "bus": "vehicle",
    "truck": "vehicle",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
    "person": "pedestrian",
}


class FasterRCNNDetector:
    def __init__(
        self,
        weights_path: str | Path,
        allowed_labels: set[str] | None = None,
        score_threshold: float = 0.5,
        device: str = "cuda:0",
    ):
        self.score_threshold = score_threshold
        self.allowed_labels = allowed_labels or {"car", "bus", "truck", "motorcycle", "bicycle", "person"}

        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        ckpt = torch.load(weights_path, map_location=self.device)

        num_classes = ckpt["num_classes_with_background"]
        self.classes: tuple[str, ...] = ckpt["classes"]

        self.model = fasterrcnn_resnet50_fpn(weights=None)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def _label_name(self, contiguous_idx: int) -> str:
        return self.classes[contiguous_idx - 1]

    def detect_frame(self, frame, frame_index: int) -> list[Detection]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = TF.to_tensor(rgb).to(self.device)

        with torch.no_grad():
            outputs = self.model([tensor])

        output = outputs[0]
        boxes = output["boxes"].cpu()
        scores = output["scores"].cpu()
        labels = output["labels"].cpu()

        detections: list[Detection] = []
        for box, score, label_idx in zip(boxes, scores, labels):
            if score.item() < self.score_threshold:
                continue
            raw_label = self._label_name(label_idx.item())
            if raw_label not in self.allowed_labels:
                continue
            x1, y1, x2, y2 = box.tolist()
            detections.append(
                Detection(
                    frame_index=frame_index,
                    bbox_xyxy=(x1, y1, x2, y2),
                    score=float(score.item()),
                    label=COCO_TO_TRAFFIC_LABEL.get(raw_label, raw_label),
                )
            )
        return detections

    def detect_video(self, video_path: Path, frame_stride: int = 1) -> list[list[Detection]]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        all_detections: list[list[Detection]] = []
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % frame_stride == 0:
                all_detections.append(self.detect_frame(frame, frame_index))
            frame_index += 1
        cap.release()
        return all_detections
