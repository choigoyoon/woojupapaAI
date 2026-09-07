"""Post-trade drawdown calculation, isolated from stages 03-11."""

from __future__ import annotations

from typing import Iterable


def maximum_drawdown(equity: Iterable[float]) -> float:
    peak: float | None = None
    worst = 0.0
    for value in equity:
        current = float(value)
        peak = current if peak is None else max(peak, current)
        if peak > 0:
            worst = min(worst, current / peak - 1.0)
    return worst
