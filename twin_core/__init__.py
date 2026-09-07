"""Shared contracts for the rebuilt TWIN learner."""

from __future__ import annotations


RUNTIME_ACTIONS = ("WAIT", "NOW", "HOLD", "REARM")
POSITION_SIDES = ("FLAT", "LONG", "SHORT")
FORBIDDEN_RUNTIME_FIELDS = frozenset(
    {
        "official_lh",
        "official_side",
        "actual_target_side",
        "actual_target_idx",
        "actual_target_time",
        "actual_target_price",
        "future_candle",
        "remaining_5m_bars",
        "posthoc_pnl",
        "posthoc_mae",
        "fixed_action",
        "is_final_survivor",
    }
)


class TwinContractError(RuntimeError):
    """Raised when a canonical input or causal-runtime contract is broken."""


def require_columns(columns, required: set[str], owner: str) -> None:
    missing = sorted(required.difference(columns))
    if missing:
        raise TwinContractError(f"{owner}: missing columns {missing}")
