"""Sequential replay: decide on a closed prefix and execute at next actual open."""

from __future__ import annotations

import numpy as np
import pandas as pd

from twin_core import TwinContractError, require_columns


_ACTION_PRIORITY = {"NOW": 3, "HOLD": 2, "REARM": 1, "WAIT": 0}


def _one_decision_per_bar(frontier: pd.DataFrame) -> pd.DataFrame:
    ranked = frontier.copy()
    ranked["_priority"] = ranked["relative_decision"].map({"NOW": 3, "WAIT": 0}).fillna(0)
    ranked["_priority"] += ranked["rearm_event"].astype(int)
    return (
        ranked.sort_values(["global_idx", "_priority", "case_id"], ascending=[True, False, True])
        .drop_duplicates("global_idx", keep="first")
        .drop(columns="_priority")
    )


def build_prefix_replay(
    frontier: pd.DataFrame,
    ohlcv: pd.DataFrame,
    *,
    initial_position: str = "FLAT",
) -> pd.DataFrame:
    require_columns(
        frontier.columns,
        {"global_idx", "timestamp", "relative_decision", "rearm_event", "desired_position"},
        "replay_builder.frontier",
    )
    require_columns(ohlcv.columns, {"timestamp", "open"}, "replay_builder.ohlcv")
    if initial_position not in {"FLAT", "LONG", "SHORT"}:
        raise TwinContractError(f"replay_builder: invalid initial position {initial_position}")
    bars = ohlcv.sort_values("timestamp").reset_index(drop=True)
    decisions = _one_decision_per_bar(frontier).sort_values("global_idx")
    position = initial_position
    entry_price: float | None = None
    records: list[dict[str, object]] = []

    for row in decisions.itertuples(index=False):
        desired = str(row.desired_position)
        if position == desired:
            action = "HOLD"
        elif row.relative_decision == "NOW":
            action = "NOW"
        elif bool(row.rearm_event):
            action = "REARM"
        else:
            action = "WAIT"
        execution_idx = int(row.global_idx) + 1 if action == "NOW" else None
        execution_price = None
        realized_return = None
        before = position
        if execution_idx is not None and execution_idx < len(bars):
            execution_price = float(bars.at[execution_idx, "open"])
            if position != "FLAT" and entry_price is not None:
                multiplier = 1.0 if position == "LONG" else -1.0
                realized_return = multiplier * (execution_price / entry_price - 1.0)
            position = desired
            entry_price = execution_price
        elif execution_idx is not None:
            action = "WAIT"
            execution_idx = None

        records.append(
            {
                "decision_idx": int(row.global_idx),
                "decision_time": row.timestamp,
                "position_before": before,
                "desired_position": desired,
                "action": action,
                "execution_idx": execution_idx,
                "execution_time": bars.at[execution_idx, "timestamp"]
                if execution_idx is not None
                else pd.NaT,
                "execution_open": execution_price,
                "position_after": position,
                "realized_return": realized_return,
                "decision_uses_future": False,
            }
        )
    return pd.DataFrame.from_records(records)
