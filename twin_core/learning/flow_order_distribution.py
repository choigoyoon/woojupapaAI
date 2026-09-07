"""Stage 08: decode the causal ordering of eight latest ZC positions."""

from __future__ import annotations

import math
from typing import Any, Mapping

from twin_core.authentic_contract import TIMEFRAMES


def evaluate_flow_order(observation: Mapping[str, Any], *, raw_position: int) -> dict[str, Any]:
    positions: dict[str, int | None] = {}
    for timeframe in TIMEFRAMES:
        value = observation.get(f"macd_{timeframe}_zc_age_log")
        if value is None or not math.isfinite(float(value)) or float(value) == -9.0:
            positions[timeframe] = None
        else:
            positions[timeframe] = int(raw_position - round(math.expm1(float(value))))
    available = [value for value in positions.values() if value is not None]
    groups: dict[int, list[str]] = {}
    for timeframe, position in positions.items():
        if position is not None:
            groups.setdefault(position, []).append(timeframe)
    order = [
        {"raw_position": position, "simultaneous_timeframes": groups[position]}
        for position in sorted(groups)
    ]
    return {
        "stage": "08",
        "last_zc_raw_positions": positions,
        "distinct_last_zc_position_count": len(set(available)),
        "signal_order_sequence": order,
        "tie_policy": "KEEP_SIMULTANEOUS",
        "trade_action": "WAIT",
        "handoff": "09",
    }
