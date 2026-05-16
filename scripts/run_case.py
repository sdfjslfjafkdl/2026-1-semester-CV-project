from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from accident_liability.pipeline.service import LiabilityPipeline
from accident_liability.rules.adjustment import AdjustmentModel
from accident_liability.rules.base_ratio import BaseRatioLookup
from accident_liability.schemas import CaseInput, Evidence, SceneAnchor

DEFAULT_CLASSIFIER_WEIGHTS = PROJECT_ROOT / "classification" / "video_classification_outputs" / "best.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one case through the full liability pipeline")
    parser.add_argument("--video_path", type=Path, required=True)
    parser.add_argument("--statement", type=str, default="", help="사용자 진술 (--use_llm 시 위반사항 추출에 사용)")
    parser.add_argument("--base_ratio_csv", type=Path, required=True)
    parser.add_argument("--adjustment_model", type=Path, required=True)

    # 분류기 가중치: 지정하면 영상에서 anchor 자동 예측
    parser.add_argument("--classifier_weights", type=Path, default=None,
                        help="video classifier 가중치 경로. 지정 시 anchor 자동 예측")
    parser.add_argument("--device", type=str, default="cpu")

    # accident_place 는 classifier가 예측하지 않으므로 항상 수동 입력
    parser.add_argument("--accident_place", type=str, required=True,
                        help="사고 장소 (예: 직선 도로, 사거리교차로(신호등 없음))")

    # classifier 미사용 시 수동 입력
    parser.add_argument("--accident_place_feature", type=str, default=None)
    parser.add_argument("--vehicle_a_progress_info", type=str, default=None)
    parser.add_argument("--vehicle_b_progress_info", type=str, default=None)

    parser.add_argument("--entry_order", type=str, default="unknown",
                        choices=["A_first", "B_first", "unknown"])
    parser.add_argument("--first_entry_strength", type=str, default=None,
                        choices=["weak", "medium", "strong"])
    parser.add_argument("--first_entry_conf", type=float, default=0.0)
    parser.add_argument("--use_llm", action="store_true", help="Groq LLM으로 진술 위반사항 파싱")
    return parser.parse_args()


def predict_anchor_from_video(video_path: Path, weights_path: Path, device: str) -> dict:
    from accident_liability.scene.video_classifier import VideoClassifier
    from accident_liability.scene.class_maps import decode

    print(f"[분류기] 가중치 로드 중: {weights_path}")
    clf = VideoClassifier(weights_path=weights_path, device=device)
    print(f"[분류기] 영상 추론 중: {video_path.name}")
    raw = clf.predict(video_path)

    # AI Hub 숫자 코드 → 사람이 읽을 수 있는 레이블로 변환
    predictions = {}
    print("[분류기] 예측 결과:")
    for key, pred in raw.items():
        code = str(pred["class_name"])
        label = decode(key, code) or code  # 맵에 없으면 코드 그대로 사용
        predictions[key] = {"class_name": label, "score": pred["score"]}
        print(f"  {key}: {label} (score={pred['score']:.3f})")
    return predictions


def main() -> None:
    args = parse_args()

    # ── anchor 결정 ──────────────────────────────────────────────────
    # classifier는 명시적으로 지정했을 때만 실행
    if args.classifier_weights is not None:
        weights = args.classifier_weights
    else:
        weights = None

    if weights is not None:
        predictions = predict_anchor_from_video(args.video_path, weights, args.device)
        accident_place_feature  = predictions["accident_place_feature"]["class_name"]
        vehicle_a_progress_info = predictions["vehicle_a_progress_info"]["class_name"]
        vehicle_b_progress_info = predictions["vehicle_b_progress_info"]["class_name"]
        clf_confidence = min(v["score"] for v in predictions.values())
    else:
        # 수동 입력 모드
        missing = [k for k in ["accident_place_feature", "vehicle_a_progress_info", "vehicle_b_progress_info"]
                   if getattr(args, k) is None]
        if missing:
            raise ValueError(
                f"--classifier_weights 미지정 시 다음 인수를 직접 입력해야 합니다: {missing}"
            )
        accident_place_feature  = args.accident_place_feature
        vehicle_a_progress_info = args.vehicle_a_progress_info
        vehicle_b_progress_info = args.vehicle_b_progress_info
        clf_confidence = 1.0

    anchor = SceneAnchor(
        accident_place=args.accident_place,
        accident_place_feature=accident_place_feature,
        vehicle_a_progress_info=vehicle_a_progress_info,
        vehicle_b_progress_info=vehicle_b_progress_info,
        confidence=clf_confidence,
    )

    # ── LLM 진술 파싱 ────────────────────────────────────────────────
    violation_parser = None
    if args.use_llm:
        from accident_liability.llm.violation_parser import ViolationParser
        violation_parser = ViolationParser()

    # ── 파이프라인 실행 ──────────────────────────────────────────────
    pipeline = LiabilityPipeline(
        base_lookup=BaseRatioLookup.from_csv(args.base_ratio_csv),
        adjustment_model=AdjustmentModel.load(args.adjustment_model),
        violation_parser=violation_parser,
    )
    case = CaseInput(video_path=args.video_path, statement=args.statement)
    evidence = Evidence(
        entry_order=args.entry_order,
        first_entry_strength=args.first_entry_strength,
        first_entry_conf=args.first_entry_conf,
    )
    result = pipeline.run_with_ready_inputs(case, anchor, evidence)
    print()
    print(result["report"])


if __name__ == "__main__":
    main()
