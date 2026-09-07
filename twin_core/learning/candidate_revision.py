"""Stage 03: update the candidate axis; never emit a trade NOW."""

from __future__ import annotations

from typing import Any, Mapping

from twin_core import TwinContractError


def evaluate_candidate(
    observation: Mapping[str, Any],
    *,
    seek_side: str | None = None,
    current_extreme: float | None = None,
) -> dict[str, Any]:
    side = (seek_side or str(observation.get("seek_side", ""))).upper()
    if side not in {"H", "L"}:
        raise TwinContractError("Stage 03 requires seek_side H or L")

    declared = bool(int(float(observation.get("new_extreme_now", 0))))
    price_breach: bool | None = None
    next_extreme = current_extreme
    if current_extreme is not None and "high" in observation and "low" in observation:
        observed = float(observation["high"] if side == "H" else observation["low"])
        price_breach = observed > current_extreme if side == "H" else observed < current_extreme
        next_extreme = observed if price_breach else current_extreme
        if price_breach != declared:
            raise TwinContractError("Stage 03 new_extreme_now disagrees with the causal price breach")

    return {
        "stage": "03",
        "candidate_transition": "REARM" if declared else "HOLD",
        "new_extreme_now": declared,
        "next_candidate_extreme": next_extreme,
        "price_breach_checked": price_breach is not None,
        "trade_action": "WAIT",
        "handoff": "04",
    }
