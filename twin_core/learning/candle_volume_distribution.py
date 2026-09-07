"""Stage 06: retain candle and volume observations without deciding."""

from __future__ import annotations

import math
from typing import Any, Mapping


FIELDS = (
    "candle_direction_for_side",
    "candle_body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "close_position_for_side",
    "volume_ratio20",
    "volume_ratio72",
    "range_ratio20",
)


def evaluate_candle_volume(observation: Mapping[str, Any]) -> dict[str, Any]:
    profile = {field: observation.get(field) for field in FIELDS}
    volume_values = (profile["volume_ratio20"], profile["volume_ratio72"])
    volume_available = all(
        value is not None and math.isfinite(float(value)) and float(value) != -9.0
        for value in volume_values
    )
    if not volume_available:
        profile["volume_ratio20"] = None
        profile["volume_ratio72"] = None
    return {
        "stage": "06",
        "volume_available": volume_available,
        "candle_volume_reaction_profile": profile,
        "trade_action": "WAIT",
        "handoff": "07",
    }
