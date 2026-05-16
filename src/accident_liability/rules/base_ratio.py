from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from accident_liability.schemas import BaseRatioResult, SceneAnchor


BASE_RATIO_KEY_COLS = [
    "accident_place",
    "accident_place_feature",
    "vehicle_a_progress_info",
    "vehicle_b_progress_info",
]


def _norm(value: Any) -> str:
    return str(value).strip()


class ClassMaps:
    def __init__(self, maps: dict[str, dict[int, str]]):
        self.maps = maps

    @classmethod
    def from_csv_dir(cls, class_map_dir: Path) -> "ClassMaps":
        maps: dict[str, dict[int, str]] = {}
        for name in BASE_RATIO_KEY_COLS:
            path = Path(class_map_dir) / f"{name}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            maps[name] = {int(r["class_id"]): str(r["label"]) for _, r in df.iterrows()}
        return cls(maps)

    def decode(self, **class_ids: int | None) -> dict[str, str | None]:
        decoded: dict[str, str | None] = {}
        for name, class_id in class_ids.items():
            if class_id is None:
                decoded[name] = None
            else:
                decoded[name] = self.maps.get(name, {}).get(int(class_id))
        return decoded


class BaseRatioLookup:
    def __init__(self, rows: pd.DataFrame):
        missing = [c for c in BASE_RATIO_KEY_COLS + ["ratio_a", "ratio_b"] if c not in rows.columns]
        if missing:
            raise ValueError(f"base ratio CSV missing columns: {missing}")
        self.rows = rows.copy()
        self.lookup = self._build_lookup(self.rows)

    @classmethod
    def from_csv(cls, path: Path) -> "BaseRatioLookup":
        return cls(pd.read_csv(path))

    @staticmethod
    def _build_lookup(rows: pd.DataFrame) -> dict[tuple[str, str, str, str], dict[str, Any]]:
        lookup: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        for _, row in rows.iterrows():
            key = tuple(_norm(row[c]) for c in BASE_RATIO_KEY_COLS)
            lookup[key] = row.to_dict()
        return lookup

    def get(
        self,
        accident_place: str,
        accident_place_feature: str,
        vehicle_a_progress_info: str,
        vehicle_b_progress_info: str,
        default: BaseRatioResult | None = None,
    ) -> BaseRatioResult:
        key = (
            _norm(accident_place),
            _norm(accident_place_feature),
            _norm(vehicle_a_progress_info),
            _norm(vehicle_b_progress_info),
        )
        item = self.lookup.get(key)
        if item is None:
            if default is not None:
                return default
            raise KeyError(f"base ratio lookup miss: {key}")
        return BaseRatioResult(
            ratio_a=float(item["ratio_a"]),
            ratio_b=float(item["ratio_b"]),
            ratio_class=int(item["ratio_class"]) if "ratio_class" in item and pd.notna(item["ratio_class"]) else None,
            matched_key=key,
        )

    def get_from_anchor(self, anchor: SceneAnchor) -> BaseRatioResult:
        required = {
            "accident_place": anchor.accident_place,
            "accident_place_feature": anchor.accident_place_feature,
            "vehicle_a_progress_info": anchor.vehicle_a_progress_info,
            "vehicle_b_progress_info": anchor.vehicle_b_progress_info,
        }
        missing = [k for k, v in required.items() if v is None]
        if missing:
            raise ValueError(f"SceneAnchor missing fields for base ratio lookup: {missing}")
        return self.get(**required)  # type: ignore[arg-type]
