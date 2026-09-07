"""Stage 05: decode relative clocks without adding a waiting threshold."""

from __future__ import annotations

import math
from typing import Any, Mapping


def _decode_optional(value: Any) -> int | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number) or number == -9.0:
        return None
    return int(round(math.expm1(number)))


def evaluate_time(observation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": "05",
        "wave_age_bars": _decode_optional(observation.get("wave_age_log")),
        "zone_age_bars": _decode_optional(observation.get("zone_age_log")),
        "time_ratio_prev1_available": observation.get("time_ratio_prev1") not in (None, -9, -9.0),
        "time_ratio_prev1": observation.get("time_ratio_prev1"),
        "time_ratio_prev2_available": observation.get("time_ratio_prev2") not in (None, -9, -9.0),
        "time_ratio_prev2": observation.get("time_ratio_prev2"),
        "hardcoded_minimum_wait": None,
        "trade_action": "WAIT",
        "handoff": "06",
    }
