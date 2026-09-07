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
    equity = (1.0 + net).cumprod()
    if equity.empty:
        max_drawdown = None
        total_return = None
    else:
        drawdown = equity / equity.cummax() - 1.0
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
        "note": "posthoc execution audit; no field here is a runtime rule input",
    }
