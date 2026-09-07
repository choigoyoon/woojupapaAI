"""Stateful runner that preserves the authentic Stage 03 -> 11 order."""

from __future__ import annotations

from typing import Any, Mapping

from twin_core.learning.candidate_revision import evaluate_candidate
from twin_core.learning.candle_volume_distribution import evaluate_candle_volume
from twin_core.learning.context_partition import evaluate_context
from twin_core.learning.flow_order_distribution import evaluate_flow_order
from twin_core.learning.multitf_zc_distribution import evaluate_multitf
from twin_core.learning.repair_queue import PersistenceState, apply_persistence
from twin_core.learning.rule_induction import RuleCatalog
from twin_core.learning.threshold_frontier import EventReleaseState, decide_final_action
from twin_core.learning.time_distribution import evaluate_time


class AuthenticLearningWorker:
    def __init__(self, catalog: RuleCatalog):
        self.catalog = catalog
        self.persistence = PersistenceState()
        self.release = EventReleaseState()
        self.event_key: str | None = None
        self.history: list[dict[str, Any]] = []

    def _begin_event(self, event_key: str) -> None:
        if self.event_key != event_key:
            self.event_key = event_key
            self.history.clear()
            self.persistence.reset()
            self.release.begin(event_key)

    def step(
        self,
        observation: Mapping[str, Any],
        *,
        event_key: str,
        seek_side: str,
        raw_position: int,
        observed_1h_zc_switch: bool | None = None,
        current_candidate_extreme: float | None = None,
    ) -> dict[str, Any]:
        self._begin_event(event_key)
        row = dict(observation)
        self.catalog.validate_observation(row)

        if observed_1h_zc_switch is None:
            previous = self.history[-1] if self.history else None
            observed_1h_zc_switch = bool(
                previous
                and (
                    previous.get("macd_1h_zc_count_so_far") != row.get("macd_1h_zc_count_so_far")
                    or previous.get("macd_1h_sign_for_side") != row.get("macd_1h_sign_for_side")
                )
            )

        stage03 = evaluate_candidate(
            row, seek_side=seek_side, current_extreme=current_candidate_extreme
        )
        stage04 = evaluate_context(row)
        stage05 = evaluate_time(row)
        stage06 = evaluate_candle_volume(row)
        stage07 = evaluate_multitf(row)
        stage08 = evaluate_flow_order(row, raw_position=raw_position)
        stage09 = self.catalog.evaluate(
            row,
            self.history,
            background=stage04["context_background_name"],
            seek_side=seek_side,
        )
        stage10 = apply_persistence(stage09, self.persistence)
        stage11 = decide_final_action(
            stage03,
            stage10,
            self.catalog,
            self.release,
            event_key=event_key,
            observed_1h_zc_switch=bool(observed_1h_zc_switch),
        )
        self.history.append(row)
        return {
            "stage_order": ["03", "04", "05", "06", "07", "08", "09", "10", "11"],
            "stages": {
                "03": stage03,
                "04": stage04,
                "05": stage05,
                "06": stage06,
                "07": stage07,
                "08": stage08,
                "09": stage09,
                "10": stage10,
                "11": stage11,
            },
            "candidate_transition": stage11["candidate_transition"],
            "trade_action": stage11["trade_action"],
        }
