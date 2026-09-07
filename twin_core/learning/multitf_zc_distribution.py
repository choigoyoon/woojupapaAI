"""Stage 07: expose all eight MACD witnesses without voting them together."""

from __future__ import annotations

from typing import Any, Mapping

from twin_core.authentic_contract import TIMEFRAMES


def evaluate_multitf(observation: Mapping[str, Any]) -> dict[str, Any]:
    witnesses: list[dict[str, Any]] = []
    for timeframe in TIMEFRAMES:
        prefix = f"macd_{timeframe}_"
        witnesses.append(
            {
                "timeframe": timeframe,
                "hist_for_side": observation.get(prefix + "hist_for_side"),
                "delta_for_side": observation.get(prefix + "delta_for_side"),
                "sign_for_side": observation.get(prefix + "sign_for_side"),
                "zc_count_so_far": observation.get(prefix + "zc_count_so_far"),
                "zc_age_log": observation.get(prefix + "zc_age_log"),
            }
        )
    return {
        "stage": "07",
        "timeframe_witnesses": witnesses,
        "timeframe_vote": None,
        "single_score": None,
        "trade_action": "WAIT",
        "handoff": "08",
    }
