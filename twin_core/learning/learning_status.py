"""Stage timing and receipts for one learning run."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter
from typing import Iterator


@dataclass
class LearningStatus:
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    state: str = "RUNNING"
    stages: list[dict[str, object]] = field(default_factory=list)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        receipt: dict[str, object] = {"name": name, "state": "RUNNING"}
        self.stages.append(receipt)
        started = perf_counter()
        try:
            yield
        except Exception as exc:
            receipt.update(
                {"state": "FAILED", "seconds": perf_counter() - started, "error": str(exc)}
            )
            self.state = "FAILED"
            raise
        else:
            receipt.update({"state": "COMPLETE", "seconds": perf_counter() - started})

    def finish(self, state: str = "COMPLETE") -> None:
        self.state = state


def get_learning_status(status: LearningStatus) -> dict[str, object]:
    return {
        "started_at": status.started_at,
        "state": status.state,
        "stages": [dict(stage) for stage in status.stages],
    }
