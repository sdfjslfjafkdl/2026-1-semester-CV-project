from __future__ import annotations

import pandas as pd

from accident_liability.rules.adjustment import clip_ratio, evidence_to_design_row
from accident_liability.rules.base_ratio import BaseRatioLookup


def test_base_ratio_lookup_exact_match():
    lookup = BaseRatioLookup(
        pd.DataFrame(
            [
                {
                    "accident_place": "사거리 교차로(신호등 없음)",
                    "accident_place_feature": "동일폭 도로",
                    "vehicle_a_progress_info": "오른쪽에서 직진",
                    "vehicle_b_progress_info": "왼쪽에서 직진",
                    "ratio_a": 40,
                    "ratio_b": 60,
                    "ratio_class": 21,
                }
            ]
        )
    )
    result = lookup.get(
        "사거리 교차로(신호등 없음)",
        "동일폭 도로",
        "오른쪽에서 직진",
        "왼쪽에서 직진",
    )
    assert result.ratio_a == 40
    assert result.ratio_b == 60
    assert result.ratio_class == 21


def test_evidence_design_row_confidence_weighted():
    row = {
        "entry_order": "A_first",
        "first_entry_strength": "strong",
        "first_entry_conf": 0.9,
        "A_no_deceleration": True,
        "A_no_deceleration_strength": "medium",
        "A_no_deceleration_conf": 0.7,
    }
    design = evidence_to_design_row(row)
    assert design["first_entry_A_first_strong"] == 0.9
    assert design["A_no_decel_medium"] == 0.7
    assert design["B_no_decel_medium"] == 0.0


def test_clip_ratio():
    assert clip_ratio(34) == 30
    assert clip_ratio(36) == 40
    assert clip_ratio(-10) == 0
    assert clip_ratio(105) == 100
