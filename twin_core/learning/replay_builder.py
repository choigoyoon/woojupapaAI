"""Sequential prefix replay helper for a future hash-locked feature ledger."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from twin_core.learning.learning_worker import AuthenticLearningWorker


def replay_prefix_rows(
    worker: AuthenticLearningWorker,
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for record in rows:
        metadata = record["runtime_metadata"]
        outputs.append(
            worker.step(
                record["observation"],
                event_key=str(metadata["event_key"]),
                seek_side=str(metadata["seek_side"]),
                raw_position=int(metadata["raw_position"]),
                observed_1h_zc_switch=metadata.get("observed_1h_zc_switch"),
                current_candidate_extreme=metadata.get("current_candidate_extreme"),
            )
        )
    return outputs
