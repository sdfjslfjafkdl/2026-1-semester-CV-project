from __future__ import annotations

import json
from pathlib import Path

from accident_liability.schemas import Detection, TrackedObject


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom else 0.0


class AnnotationAssigner:
    """Assigns A/B using the first annotated boxes.

    Expected JSON:
    {
      "A": {"frame_index": 0, "bbox_xyxy": [x1, y1, x2, y2]},
      "B": {"frame_index": 0, "bbox_xyxy": [x1, y1, x2, y2]}
    }
    """

    def __init__(self, annotation_path: Path):
        with Path(annotation_path).open("r", encoding="utf-8") as f:
            self.annotations = json.load(f)

    def assign_detections(self, detections: list[Detection], min_iou: float = 0.2) -> dict[int, str]:
        assignments: dict[int, str] = {}
        for actor in ["A", "B"]:
            ann = self.annotations.get(actor)
            if not ann:
                continue
            ann_frame = int(ann.get("frame_index", 0))
            ann_box = tuple(float(v) for v in ann["bbox_xyxy"])
            best_idx = None
            best_iou = 0.0
            for idx, det in enumerate(detections):
                if det.frame_index != ann_frame:
                    continue
                score = iou(det.bbox_xyxy, ann_box)
                if score > best_iou:
                    best_idx = idx
                    best_iou = score
            if best_idx is not None and best_iou >= min_iou:
                assignments[best_idx] = actor
        return assignments


def propagate_actor_by_track(objects: list[TrackedObject]) -> list[TrackedObject]:
    track_to_actor: dict[int, str] = {}
    for obj in objects:
        if obj.actor is not None:
            track_to_actor[obj.track_id] = obj.actor
    return [
        TrackedObject(
            frame_index=o.frame_index,
            track_id=o.track_id,
            bbox_xyxy=o.bbox_xyxy,
            score=o.score,
            label=o.label,
            actor=o.actor or track_to_actor.get(o.track_id),
        )
        for o in objects
    ]
