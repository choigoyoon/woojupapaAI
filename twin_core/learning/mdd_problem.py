"""Execution-result audit; fees and drawdown never become signal inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from twin_core import require_columns


def analyze_mdd_and_failures(replay: pd.DataFrame, *, fee_bps: float = 8.0) -> dict[str, object]:
    require_columns(replay.columns, {"action", "realized_return"}, "mdd_problem.replay")
    trades = pd.to_numeric(
        replay.loc[replay["realized_return"].notna(), "realized_return"], errors="coerce"
    ).dropna()
    net = trades - (2.0 * float(fee_bps) / 10_000.0)
    capital = 1.0
    curve: list[float] = []
    insolvent = False
    for value in net:
        factor = 1.0 + float(value)
        if factor <= 0.0:
            capital = 0.0
            insolvent = True
        elif capital > 0.0:
            capital *= factor
        curve.append(capital)
    equity = pd.Series(curve, dtype=float)
    if equity.empty:
        max_drawdown = None
        total_return = None
    else:
        high_water = pd.concat([pd.Series([1.0]), equity]).cummax().iloc[1:].reset_index(drop=True)
        drawdown = equity / high_water - 1.0
        max_drawdown = float(drawdown.min())
        total_return = float(equity.iloc[-1] - 1.0)
    return {
        "switches": int(replay["action"].eq("NOW").sum()),
        "closed_trades": int(len(trades)),
        "fee_bps_per_execution": float(fee_bps),
        "total_return_after_fees": total_return,
        "max_drawdown": max_drawdown,
        "mean_trade_return": float(net.mean()) if len(net) else None,
        "loss_trades": int((net < 0).sum()),
        "insolvent_without_risk_controls": insolvent,
        "note": "posthoc execution audit; no field here is a runtime rule input",
    }

