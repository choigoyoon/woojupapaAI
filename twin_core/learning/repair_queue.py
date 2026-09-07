"""Stage 10: apply learned persistence and retain explicit wait reasons."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from twin_core.authentic_contract import PARALLEL_METHODS


@dataclass
class PersistenceState:
    signatures: dict[str, str | None] = field(default_factory=dict)
    runs: dict[str, int] = field(default_factory=dict)

    def reset(self) -> None:
        self.signatures.clear()
        self.runs.clear()

    def advance(self, key: str, signature: str | None) -> int:
        if signature is None:
            self.signatures[key] = None
            self.runs[key] = 0
            return 0
        if self.signatures.get(key) == signature:
            self.runs[key] = self.runs.get(key, 0) + 1
        else:
            self.signatures[key] = signature
            self.runs[key] = 1
        return self.runs[key]


def _source_state(
    candidate: dict[str, Any] | None,
    run: int,
    *,
    gate: float,
    tolerance: float,
    minimum_support: int,
    source: str,
) -> tuple[bool, str, dict[str, Any] | None]:
    if candidate is None:
        return False, f"NO_{source}_RULE_MATCH", None
    if int(candidate["support"]) < minimum_support:
        return False, f"{source}_SUPPORT_BELOW_MINIMUM", candidate
    if float(candidate["probability"]) + tolerance < gate:
        return False, f"{source}_PROBABILITY_BELOW_GATE", candidate
    if run < int(candidate["required"]):
        return False, f"{source}_PERSISTENCE_INCOMPLETE", candidate
    return True, f"{source}_READY", candidate


def apply_persistence(stage09: dict[str, Any], state: PersistenceState) -> dict[str, Any]:
    gate = float(stage09["action_gate"])
    tolerance = float(stage09["probability_tolerance"])
    method_results: dict[str, dict[str, Any]] = {}
    all_ready_candidates: list[dict[str, Any]] = []

    for method in PARALLEL_METHODS:
        view = stage09["methods"][method]
        base = view["base"]
        context = view["context"]
        base_run = state.advance(f"{method}:BASE", base["signature"] if base else None)
        context_run = state.advance(f"{method}:CONTEXT", context["signature"] if context else None)
        base_ready, base_reason, base_candidate = _source_state(
            base,
            base_run,
            gate=gate,
            tolerance=tolerance,
            minimum_support=int(stage09["minimum_event_support"]),
            source="BASE",
        )
        context_ready, context_reason, context_candidate = _source_state(
            context,
            context_run,
            gate=gate,
            tolerance=tolerance,
            minimum_support=int(stage09["minimum_context_event_support"]),
            source="CONTEXT",
        )
        ready = base_ready or context_ready
        candidates: list[dict[str, Any]] = []
        if base_ready and base_candidate:
            candidates.append({**base_candidate, "run": base_run})
        if context_ready and context_candidate:
            candidates.append({**context_candidate, "run": context_run})
        all_ready_candidates.extend(candidates)
        method_results[method] = {
            "base_run": base_run,
            "base_ready": base_ready,
            "base_reason": base_reason,
            "context_run": context_run,
            "context_ready": context_ready,
            "context_reason": context_reason,
            "ready": ready,
            "ready_sources": [candidate["source"] for candidate in candidates],
            "wait_state": "METHOD_READY" if ready else f"BASE:{base_reason}|CONTEXT:{context_reason}",
        }

    return {
        "stage": "10",
        "methods": method_results,
        "ready_method_count": sum(int(value["ready"]) for value in method_results.values()),
        "released": bool(all_ready_candidates),
        "ready_candidates": all_ready_candidates,
        "blocker_count": 0 if all_ready_candidates else 1,
        "trade_action": "WAIT",
        "handoff": "11",
    }
