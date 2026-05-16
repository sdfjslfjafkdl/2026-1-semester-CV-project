"""
비디오 목록을 받아서:
  1. VideoClassifier   -> accident_place_feature, vehicle_a/b_progress_info
  2. FasterRCNNDetector -> detection
  3. IoUTracker        -> track
  4. TrajectoryReasoner -> Evidence
  5. BaseRatioLookup   -> base_ratio_a
  6. 위 결과 + true_ratio_a 합쳐서 adjustment 학습용 CSV 저장

입력 JSON 스키마 (항목당):
{
  "video_path": "...",
  "accident_place": "교차로",      // base ratio lookup용 (분류기가 예측 안 하는 필드)
  "true_ratio_a": 30,              // ground truth
  "annotation_path": "..."         // 선택: A/B 차량 annotation JSON
}

annotation JSON 없으면 첫 등장 프레임 기준 가장 큰 두 track을 A, B로 자동 할당.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from accident_liability.perception.detector import FasterRCNNDetector
from accident_liability.perception.tracker import IoUTracker, tracks_from_objects
from accident_liability.perception.assignment import AnnotationAssigner, propagate_actor_by_track
from accident_liability.rules.adjustment import evidence_to_flat_dict
from accident_liability.rules.base_ratio import BaseRatioLookup
from accident_liability.scene.video_classifier import VideoClassifier
from accident_liability.trajectory.reasoner import TrajectoryReasoner
from accident_liability.schemas import TrackedObject


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build adjustment training CSV from videos")
    parser.add_argument("--input_json", type=Path, required=True,
                        help="각 항목에 video_path / accident_place / true_ratio_a 포함한 JSON")
    parser.add_argument("--base_ratio_csv", type=Path, required=True)
    parser.add_argument("--classifier_weights", type=Path, required=True,
                        help="classification/.../best.pth")
    parser.add_argument("--detector_weights", type=Path, required=True,
                        help="detect/.../best.pth")
    parser.add_argument("--output_csv", type=Path, default=Path("outputs/adjustment_input.csv"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--iou_threshold", type=float, default=0.3)
    parser.add_argument("--max_missed", type=int, default=5)
    parser.add_argument("--min_track_length", type=int, default=5)
    parser.add_argument("--fps", type=float, default=15.0)
    return parser.parse_args()


def _box_area(bbox_xyxy: tuple) -> float:
    x1, y1, x2, y2 = bbox_xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _assign_ab_by_size(objects: list[TrackedObject]) -> list[TrackedObject]:
    """annotation 없을 때: 첫 등장 프레임 기준 가장 큰 두 track을 A, B로 할당."""
    first_obs: dict[int, TrackedObject] = {}
    for obj in sorted(objects, key=lambda o: o.frame_index):
        if obj.track_id not in first_obs:
            first_obs[obj.track_id] = obj

    ranked = sorted(first_obs.values(), key=lambda o: _box_area(o.bbox_xyxy), reverse=True)
    actor_map: dict[int, str] = {obs.track_id: actor for obs, actor in zip(ranked[:2], ["A", "B"])}

    return [
        TrackedObject(
            frame_index=o.frame_index,
            track_id=o.track_id,
            bbox_xyxy=o.bbox_xyxy,
            score=o.score,
            label=o.label,
            actor=actor_map.get(o.track_id, o.actor),
        )
        for o in objects
    ]


def process_video(
    item: dict,
    classifier: VideoClassifier,
    detector: FasterRCNNDetector,
    tracker: IoUTracker,
    base_lookup: BaseRatioLookup,
    fps: float,
) -> dict | None:
    video_path = Path(item["video_path"])
    true_ratio_a = float(item["true_ratio_a"])
    accident_place = str(item["accident_place"])

    print(f"  분류기 예측 중: {video_path.name}")
    clf_result = classifier.predict(video_path)

    accident_place_feature = clf_result["accident_place_feature"]["class_name"]
    vehicle_a_progress_info = clf_result["vehicle_a_progress_info"]["class_name"]
    vehicle_b_progress_info = clf_result["vehicle_b_progress_info"]["class_name"]
    clf_confidence = min(v["score"] for v in clf_result.values())

    print(f"  디텍터 실행 중: {video_path.name}")
    detections_by_frame = detector.detect_video(video_path)

    cap = cv2.VideoCapture(str(video_path))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    tracked_objects = tracker.track(detections_by_frame)

    annotation_path = item.get("annotation_path")
    if annotation_path and Path(annotation_path).exists():
        assigner = AnnotationAssigner(Path(annotation_path))
        all_detections = [det for frame in detections_by_frame for det in frame]
        assignments = assigner.assign_detections(all_detections)
        tracked_objects = [
            TrackedObject(
                frame_index=o.frame_index,
                track_id=o.track_id,
                bbox_xyxy=o.bbox_xyxy,
                score=o.score,
                label=o.label,
                actor=assignments.get(i, o.actor),
            )
            for i, o in enumerate(tracked_objects)
        ]
        tracked_objects = propagate_actor_by_track(tracked_objects)
    else:
        tracked_objects = _assign_ab_by_size(tracked_objects)

    tracks = tracks_from_objects(tracked_objects)
    reasoner = TrajectoryReasoner(fps=fps, frame_width=frame_width, frame_height=frame_height)
    evidence = reasoner.build_evidence(tracks)

    try:
        base = base_lookup.get(
            accident_place=accident_place,
            accident_place_feature=accident_place_feature,
            vehicle_a_progress_info=vehicle_a_progress_info,
            vehicle_b_progress_info=vehicle_b_progress_info,
        )
        base_ratio_a = base.ratio_a
    except KeyError as e:
        print(f"  [Warning] base ratio lookup 실패: {e} — 스킵")
        return None

    flat_evidence = evidence_to_flat_dict(evidence)
    row = {
        "video_path": str(video_path),
        "accident_place": accident_place,
        "accident_place_feature": accident_place_feature,
        "vehicle_a_progress_info": vehicle_a_progress_info,
        "vehicle_b_progress_info": vehicle_b_progress_info,
        "clf_confidence": round(clf_confidence, 4),
        "base_ratio_a": base_ratio_a,
        "true_ratio_a": true_ratio_a,
    }
    row.update(flat_evidence)
    return row


def main() -> None:
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    print("[모델 로드 중]")
    classifier = VideoClassifier(weights_path=args.classifier_weights, device=args.device)
    detector = FasterRCNNDetector(
        weights_path=args.detector_weights,
        score_threshold=args.score_threshold,
        device=args.device,
    )
    tracker = IoUTracker(
        iou_threshold=args.iou_threshold,
        max_missed=args.max_missed,
        min_track_length=args.min_track_length,
    )
    base_lookup = BaseRatioLookup.from_csv(args.base_ratio_csv)
    print("  완료")

    with open(args.input_json, "r", encoding="utf-8") as f:
        items = json.load(f)

    print(f"\n[총 {len(items)}개 영상 처리 시작]\n")
    rows = []
    for i, item in enumerate(items, 1):
        print(f"[{i}/{len(items)}] {item['video_path']}")
        try:
            row = process_video(item, classifier, detector, tracker, base_lookup, args.fps)
            if row is not None:
                rows.append(row)
                print(f"  -> base_ratio_a={row['base_ratio_a']}, true_ratio_a={row['true_ratio_a']}")
        except Exception as e:
            print(f"  [Error] {e} — 스킵")

    if not rows:
        print("\n[Warning] 유효한 결과가 없습니다.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False, encoding="utf-8")
    print(f"\n[완료] {len(rows)}/{len(items)}개 처리됨")
    print(f"저장: {args.output_csv}")


if __name__ == "__main__":
    main()
