"""Post-hoc action parity scoring; never called by the runtime decision path."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping, Any


def score_actions(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row["trade_action"]) for row in rows)
    return {"rows": sum(counts.values()), **dict(sorted(counts.items()))}
