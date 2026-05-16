from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from accident_liability.rules.adjustment import AdjustmentModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train residual adjustment model")
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/adjustment"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, metrics = AdjustmentModel.train_from_csv(args.input_csv)
    model.save(args.output_dir / "adjustment_model.joblib")
    if model.table is not None:
        model.table.to_csv(args.output_dir / "learned_adjustment_table.csv", index=False)
    print({"metrics": metrics, "output": str(args.output_dir)})


if __name__ == "__main__":
    main()
