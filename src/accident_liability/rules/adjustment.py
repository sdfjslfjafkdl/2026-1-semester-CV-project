from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from accident_liability.schemas import AdjustmentResult, Evidence


STRENGTH_LEVELS = ["weak", "medium", "strong"]


def safe_strength(value: Any) -> str | None:
    return value if value in STRENGTH_LEVELS else None


def evidence_to_flat_dict(evidence: Evidence | dict[str, Any]) -> dict[str, Any]:
    if isinstance(evidence, Evidence):
        data = asdict(evidence)
        actor_events = data.pop("actor_events", {})
        flat = {
            "entry_order": data.get("entry_order"),
            "first_entry_strength": data.get("first_entry_strength"),
            "first_entry_conf": data.get("first_entry_conf", 0.0),
        }
        for actor in ["A", "B"]:
            events = actor_events.get(actor, {})
            flat[f"{actor}_no_deceleration"] = events.get("no_deceleration", False)
            flat[f"{actor}_no_deceleration_strength"] = events.get("no_deceleration_strength")
            flat[f"{actor}_no_deceleration_conf"] = events.get("no_deceleration_confidence", 0.0)
            flat[f"{actor}_evasive_action"] = events.get("evasive_action", False)
            flat[f"{actor}_evasive_action_strength"] = events.get("evasive_action_strength")
            flat[f"{actor}_evasive_action_conf"] = events.get("evasive_action_confidence", 0.0)
        return flat
    return dict(evidence)


def evidence_to_design_row(row: dict[str, Any]) -> dict[str, float]:
    feat: dict[str, float] = {}

    entry_order = row.get("entry_order")
    entry_strength = safe_strength(row.get("first_entry_strength"))
    entry_conf = float(row.get("first_entry_conf", 0.0) or 0.0)
    for who in ["A_first", "B_first"]:
        for strength in STRENGTH_LEVELS:
            feat[f"first_entry_{who}_{strength}"] = 0.0
    if entry_order in ["A_first", "B_first"] and entry_strength is not None:
        feat[f"first_entry_{entry_order}_{entry_strength}"] = entry_conf

    for actor in ["A", "B"]:
        for strength in STRENGTH_LEVELS:
            feat[f"{actor}_no_decel_{strength}"] = 0.0
            feat[f"{actor}_evasive_{strength}"] = 0.0

        no_decel_strength = safe_strength(row.get(f"{actor}_no_deceleration_strength"))
        no_decel_conf = float(row.get(f"{actor}_no_deceleration_conf", 0.0) or 0.0)
        if bool(row.get(f"{actor}_no_deceleration", False)) and no_decel_strength is not None:
            feat[f"{actor}_no_decel_{no_decel_strength}"] = no_decel_conf

        evasive_strength = safe_strength(row.get(f"{actor}_evasive_action_strength"))
        evasive_conf = float(row.get(f"{actor}_evasive_action_conf", 0.0) or 0.0)
        if bool(row.get(f"{actor}_evasive_action", False)) and evasive_strength is not None:
            feat[f"{actor}_evasive_{evasive_strength}"] = evasive_conf

    return feat


def build_design_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rows = [evidence_to_design_row(r.to_dict()) for _, r in df.iterrows()]
    x = pd.DataFrame(rows).fillna(0.0)
    return x, list(x.columns)


class AdjustmentModel:
    def __init__(self, model: Any, feature_cols: list[str], table: pd.DataFrame | None = None):
        self.model = model
        self.feature_cols = feature_cols
        self.table = table

    @classmethod
    def train_from_csv(cls, csv_path: Path) -> tuple["AdjustmentModel", dict[str, float]]:
        from sklearn.linear_model import RidgeCV
        from sklearn.metrics import mean_absolute_error
        from sklearn.model_selection import train_test_split

        df = pd.read_csv(csv_path)
        for col in ["base_ratio_a", "true_ratio_a"]:
            if col not in df.columns:
                raise ValueError(f"residual CSV missing required column: {col}")
        df = df.copy()
        df["target_adjustment"] = df["true_ratio_a"] - df["base_ratio_a"]
        x, feature_cols = build_design_matrix(df)
        y = df["target_adjustment"].astype(float)

        model = RidgeCV(alphas=[0.1, 1.0, 10.0, 30.0])
        metrics: dict[str, float] = {}
        if len(df) >= 5:
            x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
            model.fit(x_train, y_train)
            pred = model.predict(x_val)
            metrics["val_mae"] = float(mean_absolute_error(y_val, pred))
        else:
            model.fit(x, y)
            pred = model.predict(x)
            metrics["train_mae"] = float(mean_absolute_error(y, pred))

        coef_df = pd.DataFrame({"feature": feature_cols, "weight": model.coef_})
        table = build_learned_adjustment_table(coef_df)
        return cls(model, feature_cols, table), metrics

    @classmethod
    def load(cls, path: Path) -> "AdjustmentModel":
        import joblib

        payload = joblib.load(path)
        return cls(
            model=payload["model"],
            feature_cols=payload["feature_cols"],
            table=payload.get("table"),
        )

    def save(self, path: Path) -> None:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self.model, "feature_cols": self.feature_cols, "table": self.table},
            path,
        )

    def predict_adjustment(self, evidence: Evidence | dict[str, Any]) -> float:
        flat = evidence_to_flat_dict(evidence)
        design = evidence_to_design_row(flat)
        x = pd.DataFrame([{c: design.get(c, 0.0) for c in self.feature_cols}])
        return float(self.model.predict(x)[0])

    def apply(self, base_ratio_a: float, evidence: Evidence | dict[str, Any], round_to: int = 10) -> AdjustmentResult:
        adjustment = self.predict_adjustment(evidence)
        final_a = clip_ratio(base_ratio_a + adjustment, round_to=round_to)
        return AdjustmentResult(
            adjustment=adjustment,
            final_ratio_a=final_a,
            final_ratio_b=100 - final_a,
            reasons=self.explain(evidence),
        )

    def explain(self, evidence: Evidence | dict[str, Any], top_k: int = 5) -> list[dict[str, Any]]:
        if self.table is None:
            return []
        flat = evidence_to_flat_dict(evidence)
        design = evidence_to_design_row(flat)
        active = []
        weights = dict(zip(self.table["feature"], self.table["weight"], strict=False))
        for feature, value in design.items():
            if value:
                active.append(
                    {
                        "feature": feature,
                        "confidence": float(value),
                        "weight": float(weights.get(feature, 0.0)),
                        "contribution": float(value) * float(weights.get(feature, 0.0)),
                    }
                )
        return sorted(active, key=lambda r: abs(r["contribution"]), reverse=True)[:top_k]


def build_learned_adjustment_table(coef_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in coef_df.iterrows():
        feature = row["feature"]
        parts = feature.split("_")
        if feature.startswith("first_entry_"):
            group = "first_entry"
            key = "_".join(parts[2:-1])
            level = parts[-1]
        else:
            group = parts[1] if len(parts) >= 3 else "unknown"
            key = parts[0]
            level = parts[-1]
        rows.append(
            {
                "feature": feature,
                "group": group,
                "key": key,
                "level": level,
                "weight": float(row["weight"]),
                "abs_weight": abs(float(row["weight"])),
            }
        )
    return pd.DataFrame(rows).sort_values(["abs_weight", "feature"], ascending=[False, True])


def clip_ratio(value: float, round_to: int = 10) -> int:
    clipped = max(0.0, min(100.0, float(value)))
    return int(round(clipped / round_to) * round_to)
