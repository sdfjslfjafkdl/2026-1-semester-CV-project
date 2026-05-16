from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


Actor = Literal["A", "B"]


@dataclass(frozen=True)
class Detection:
    frame_index: int
    bbox_xyxy: tuple[float, float, float, float]
    score: float
    label: str

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass(frozen=True)
class TrackedObject:
    frame_index: int
    track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    score: float
    label: str
    actor: Actor | None = None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass
class Track:
    track_id: int
    label: str
    actor: Actor | None
    observations: list[TrackedObject] = field(default_factory=list)

    def sorted_observations(self) -> list[TrackedObject]:
        return sorted(self.observations, key=lambda x: x.frame_index)


@dataclass(frozen=True)
class SceneAnchor:
    accident_place: str | None = None
    accident_place_feature: str | None = None
    vehicle_a_progress_info: str | None = None
    vehicle_b_progress_info: str | None = None
    confidence: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Evidence:
    entry_order: Literal["A_first", "B_first", "unknown"] = "unknown"
    first_entry_strength: Literal["weak", "medium", "strong"] | None = None
    first_entry_conf: float = 0.0
    relative_speed: dict[str, float] = field(default_factory=dict)
    heading: dict[str, str] = field(default_factory=dict)
    side_approach: dict[str, str] = field(default_factory=dict)
    collision_relative_position: str | None = None
    actor_events: dict[str, dict[str, Any]] = field(default_factory=dict)
    raw_features: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BaseRatioResult:
    ratio_a: float
    ratio_b: float
    ratio_class: int | None = None
    matched_key: tuple[str, str, str, str] | None = None
    confidence: float = 1.0


@dataclass(frozen=True)
class AdjustmentResult:
    adjustment: float
    final_ratio_a: int
    final_ratio_b: int
    reasons: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class CaseInput:
    video_path: Path
    statement: str = ""
    annotations_path: Path | None = None
    scene_anchor: SceneAnchor | None = None
