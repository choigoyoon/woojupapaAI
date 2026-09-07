"""Small status object for long sequential validation runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LearningStatus:
    state: str = "READY"
    completed_rows: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "completed_rows": self.completed_rows,
            "details": dict(self.details),
        }
