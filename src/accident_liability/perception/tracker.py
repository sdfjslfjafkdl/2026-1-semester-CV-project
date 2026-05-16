from __future__ import annotations

from collections import defaultdict

from accident_liability.schemas import Detection, Track, TrackedObject


def _iou(box1: tuple, box2: tuple) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


class IoUTracker:
    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_missed: int = 5,
        min_track_length: int = 5,
    ):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.min_track_length = min_track_length

    def track(self, detections_by_frame: list[list[Detection]]) -> list[TrackedObject]:
        active: list[dict] = []
        finished: list[dict] = []
        next_id = 1

        for frame_idx, frame_dets in enumerate(detections_by_frame):
            used: set[int] = set()

            for track in active:
                best_id, best_score = None, 0.0
                for det_id, det in enumerate(frame_dets):
                    if det_id in used or det.label != track["label"]:
                        continue
                    score = _iou(track["bbox"], det.bbox_xyxy)
                    if score > best_score:
                        best_score = score
                        best_id = det_id

                if best_id is not None and best_score >= self.iou_threshold:
                    det = frame_dets[best_id]
                    track["bbox"] = det.bbox_xyxy
                    track["last_frame"] = frame_idx
                    track["missed"] = 0
                    track["history"].append(det)
                    used.add(best_id)
                else:
                    track["missed"] += 1

            alive, dead = [], []
            for t in active:
                (dead if t["missed"] > self.max_missed else alive).append(t)
            active = alive
            finished.extend(dead)

            for det_id, det in enumerate(frame_dets):
                if det_id in used:
                    continue
                active.append({
                    "track_id": next_id,
                    "label": det.label,
                    "bbox": det.bbox_xyxy,
                    "start_frame": frame_idx,
                    "last_frame": frame_idx,
                    "missed": 0,
                    "history": [det],
                })
                next_id += 1

        result: list[TrackedObject] = []
        for track in finished + active:
            if len(track["history"]) < self.min_track_length:
                continue
            for det in track["history"]:
                result.append(TrackedObject(
                    frame_index=det.frame_index,
                    track_id=track["track_id"],
                    bbox_xyxy=det.bbox_xyxy,
                    score=det.score,
                    label=det.label,
                ))
        return result


def tracks_from_objects(objects: list[TrackedObject]) -> list[Track]:
    grouped: dict[int, list[TrackedObject]] = defaultdict(list)
    for obj in objects:
        grouped[obj.track_id].append(obj)
    tracks = []
    for track_id, observations in grouped.items():
        observations = sorted(observations, key=lambda x: x.frame_index)
        actor = next((o.actor for o in observations if o.actor is not None), None)
        tracks.append(Track(
            track_id=track_id,
            label=observations[0].label,
            actor=actor,
            observations=observations,
        ))
    return tracks
