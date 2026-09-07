"""Stage 11: the only module allowed to emit the fixed trade action NOW."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from twin_core.authentic_contract import PARALLEL_METHODS
from twin_core.learning.rule_induction import RuleCatalog


@dataclass
class EventReleaseState:
    event_key: str | None = None
    released: bool = False

    def begin(self, event_key: str) -> None:
        if self.event_key != event_key:
            self.event_key = event_key
            self.released = False


def _winner(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    method_order = {method: index for index, method in enumerate(PARALLEL_METHODS)}
    return min(
        candidates,
        key=lambda candidate: (
            -float(candidate["probability"]),
            -int(candidate["support"]),
            method_order[candidate["method"]],
            int(candidate["artifact_order"]),
            candidate["signature"],
        ),
    )


def decide_final_action(
    stage03: dict[str, Any],
    stage10: dict[str, Any],
    catalog: RuleCatalog,
    state: EventReleaseState,
    *,
    event_key: str,
    observed_1h_zc_switch: bool,
) -> dict[str, Any]:
    state.begin(event_key)
    selected = [
        candidate
        for candidate in stage10["ready_candidates"]
        if candidate["signature"] in catalog.selected_signatures
    ]
    winner = _winner(selected) if selected else None

    if not state.released and winner is not None:
        action, source = "NOW", "RULE_NOW"
        state.released = True
    elif not state.released and observed_1h_zc_switch:
        action, source = "NOW", "OBSERVED_1H_ZC_STATE_NOW"
        state.released = True
    else:
        action, source = "WAIT", "WAIT"

    return {
        "stage": "11",
        "candidate_transition": stage03["candidate_transition"],
        "trade_action": action,
        "action_source": source,
        "selected_signature": winner["signature"] if winner else None,
        "selected_method": winner["method"] if winner else None,
        "entry_fill": "NEXT_CLOSED_5M_OPEN" if action == "NOW" else None,
        "rearm_and_now": bool(stage03["candidate_transition"] == "REARM" and action == "NOW"),
    }
