from __future__ import annotations

import math
from statistics import median

from accident_liability.schemas import Evidence, Track


def _speed_samples(track: Track, fps: float) -> list[float]:
    obs = track.sorted_observations()
    speeds = []
    for prev, cur in zip(obs, obs[1:], strict=False):
        dt_frames = max(1, cur.frame_index - prev.frame_index)
        dx = cur.center[0] - prev.center[0]
        dy = cur.center[1] - prev.center[1]
        speeds.append(math.hypot(dx, dy) * fps / dt_frames)
    return speeds


def _heading(track: Track) -> str:
    obs = track.sorted_observations()
    if len(obs) < 2:
        return "unknown"
    dx = obs[-1].center[0] - obs[0].center[0]
    dy = obs[-1].center[1] - obs[0].center[1]
    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


def _entry_frame(track: Track) -> int | None:
    obs = track.sorted_observations()
    return obs[0].frame_index if obs else None


def _strength_from_frame_margin(margin_frames: int, fps: float) -> tuple[str | None, float]:
    margin_sec = abs(margin_frames) / fps
    if margin_sec >= 1.5:
        return "strong", 0.9
    if margin_sec >= 0.8:
        return "medium", 0.75
    if margin_sec >= 0.3:
        return "weak", 0.55
    return None, 0.0


def _no_deceleration_event(track: Track, fps: float) -> dict:
    speeds = _speed_samples(track, fps)
    if len(speeds) < 4:
        return {
            "no_deceleration": False,
            "no_deceleration_strength": None,
            "no_deceleration_confidence": 0.0,
        }
    early = median(speeds[: max(2, len(speeds) // 3)])
    late = median(speeds[-max(2, len(speeds) // 3) :])
    ratio = late / early if early > 1e-6 else 0.0
    if ratio >= 0.95:
        strength, conf = "strong", 0.85
    elif ratio >= 0.8:
        strength, conf = "medium", 0.7
    elif ratio >= 0.65:
        strength, conf = "weak", 0.55
    else:
        strength, conf = None, 0.0
    return {
        "no_deceleration": strength is not None,
        "no_deceleration_strength": strength,
        "no_deceleration_confidence": conf,
    }


def _side_approach(track: Track, frame_width: int | None, frame_height: int | None) -> str:
    obs = track.sorted_observations()
    if not obs or frame_width is None or frame_height is None:
        return "unknown"
    x, y = obs[0].center
    margins = {
        "left": x,
        "right": frame_width - x,
        "top": y,
        "bottom": frame_height - y,
    }
    return min(margins, key=margins.get)


class TrajectoryReasoner:
    def __init__(self, fps: float = 15.0, frame_width: int | None = None, frame_height: int | None = None):
        self.fps = fps
        self.frame_width = frame_width
        self.frame_height = frame_height

    def build_evidence(self, tracks: list[Track]) -> Evidence:
        actors = {t.actor: t for t in tracks if t.actor in {"A", "B"}}
        track_a = actors.get("A")
        track_b = actors.get("B")

        entry_order = "unknown"
        first_entry_strength = None
        first_entry_conf = 0.0
        if track_a is not None and track_b is not None:
            entry_a = _entry_frame(track_a)
            entry_b = _entry_frame(track_b)
            if entry_a is not None and entry_b is not None and entry_a != entry_b:
                entry_order = "A_first" if entry_a < entry_b else "B_first"
                first_entry_strength, first_entry_conf = _strength_from_frame_margin(
                    entry_a - entry_b,
                    self.fps,
                )

        actor_events = {}
        relative_speed = {}
        heading = {}
        side_approach = {}
        for actor, track in actors.items():
            speeds = _speed_samples(track, self.fps)
            relative_speed[actor] = float(median(speeds)) if speeds else 0.0
            heading[actor] = _heading(track)
            side_approach[actor] = _side_approach(track, self.frame_width, self.frame_height)
            actor_events[actor] = _no_deceleration_event(track, self.fps)

        collision_relative_position = None
        if track_a is not None and track_b is not None:
            a_last = track_a.sorted_observations()[-1]
            b_last = track_b.sorted_observations()[-1]
            dx = a_last.center[0] - b_last.center[0]
            dy = a_last.center[1] - b_last.center[1]
            if abs(dx) > abs(dy):
                collision_relative_position = "A_right_of_B" if dx > 0 else "A_left_of_B"
            else:
                collision_relative_position = "A_below_B" if dy > 0 else "A_above_B"

        return Evidence(
            entry_order=entry_order,  # type: ignore[arg-type]
            first_entry_strength=first_entry_strength,  # type: ignore[arg-type]
            first_entry_conf=first_entry_conf,
            relative_speed=relative_speed,
            heading=heading,
            side_approach=side_approach,
            collision_relative_position=collision_relative_position,
            actor_events=actor_events,
            raw_features={"fps": self.fps},
        )
