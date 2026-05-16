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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one case with ready scene/evidence inputs")
    parser.add_argument("--video_path", type=Path, required=True)
    parser.add_argument("--statement", type=str, default="")
    parser.add_argument("--base_ratio_csv", type=Path, required=True)
    parser.add_argument("--adjustment_model", type=Path, required=True)

    # MVP: pass anchor explicitly until VTN inference wrappers/checkpoints are ready.
    parser.add_argument("--accident_place", type=str, required=True)
    parser.add_argument("--accident_place_feature", type=str, required=True)
    parser.add_argument("--vehicle_a_progress_info", type=str, required=True)
    parser.add_argument("--vehicle_b_progress_info", type=str, required=True)
    parser.add_argument("--entry_order", type=str, default="unknown", choices=["A_first", "B_first", "unknown"])
    parser.add_argument("--first_entry_strength", type=str, default=None, choices=["weak", "medium", "strong"])
    parser.add_argument("--first_entry_conf", type=float, default=0.0)
    parser.add_argument("--use_llm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parser = None
    if args.use_llm:
        from accident_liability.llm.violation_parser import ViolationParser

        parser = ViolationParser()

    pipeline = LiabilityPipeline(
        base_lookup=BaseRatioLookup.from_csv(args.base_ratio_csv),
        adjustment_model=AdjustmentModel.load(args.adjustment_model),
        violation_parser=parser,
    )
    case = CaseInput(video_path=args.video_path, statement=args.statement)
    anchor = SceneAnchor(
        accident_place=args.accident_place,
        accident_place_feature=args.accident_place_feature,
        vehicle_a_progress_info=args.vehicle_a_progress_info,
        vehicle_b_progress_info=args.vehicle_b_progress_info,
        confidence=1.0,
    )
    evidence = Evidence(
        entry_order=args.entry_order,
        first_entry_strength=args.first_entry_strength,
        first_entry_conf=args.first_entry_conf,
    )
    result = pipeline.run_with_ready_inputs(case, anchor, evidence)
    print(result["report"])


if __name__ == "__main__":
    main()
