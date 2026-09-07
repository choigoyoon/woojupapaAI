"""Nine-stage runtime reasoning over learned relative boundaries."""

from __future__ import annotations

import numpy as np
import pandas as pd

from twin_core import RUNTIME_ACTIONS, TwinContractError, require_columns
from twin_core.learning.rule_induction import assert_runtime_safe_rules


THINKING_STAGES = (
    "POSITION_STATE",
    "TARGET_DIRECTION",
    "ANCHOR_SELECTION",
    "RELATIVE_MEASUREMENT",
    "SEQUENCE_DIVERGENCE",
    "HISTORICAL_RETRIEVAL",
    "COUNTEREXAMPLE_CHECK",
    "FIXED_ACTION",
    "POSITION_TRANSITION",
)


def _desired_position(target_side: pd.Series) -> pd.Series:
    # A completed low asks for LONG; a completed high asks for SHORT.
    return target_side.map({"L": "LONG", "H": "SHORT"})


def _transition(current: str, desired: str, action: str) -> str:
    if action == "HOLD":
        return f"KEEP_{current}"
    if action == "NOW":
        return f"OPEN_{desired}" if current == "FLAT" else f"SWITCH_TO_{desired}"
    if action == "REARM":
        return f"REARM_FOR_{desired}"
    return f"WAIT_FOR_{desired}"


def compute_threshold_frontier(
    observations: pd.DataFrame,
    rules: pd.DataFrame,
    *,
    current_position: str = "FLAT",
) -> pd.DataFrame:
    """Apply learned rules. Fixed actions are outputs, never numeric thresholds."""
    require_columns(
        observations.columns,
        {"target_side", "market_context", "rearm_event", "case_id", "step_id"},
        "threshold_frontier.observations",
    )
    if current_position not in {"FLAT", "LONG", "SHORT"}:
        raise TwinContractError(f"threshold_frontier: invalid position {current_position}")
    result = observations.copy()
    now_score = np.zeros(len(result), dtype=float)
    wait_score = np.zeros(len(result), dtype=float)
    supporting = np.zeros(len(result), dtype=np.int32)
    counterexamples = np.zeros(len(result), dtype=np.int32)

    if not rules.empty:
        assert_runtime_safe_rules(rules)
        for rule in rules.itertuples(index=False):
            if rule.feature not in result.columns:
                raise TwinContractError(f"threshold_frontier: missing learned feature {rule.feature}")
            value = pd.to_numeric(result[rule.feature], errors="coerce").to_numpy(dtype=float)
            applicable = (
                result["target_side"].eq(rule.target_side).to_numpy()
                & result["market_context"].eq(rule.market_context).to_numpy()
                & np.isfinite(value)
            )
            comparison = value <= rule.threshold if rule.operator == "<=" else value > rule.threshold
            predicts_now = applicable & comparison
            predicts_wait = applicable & ~comparison
            weight = max(float(rule.balanced_accuracy) - 0.5, np.finfo(float).eps)
            now_score[predicts_now] += weight
            wait_score[predicts_wait] += weight
            supporting[predicts_now] += 1
            counterexamples[predicts_wait] += 1

    result["now_evidence"] = now_score
    result["wait_evidence"] = wait_score
    result["supporting_rule_count"] = supporting
    result["counterexample_rule_count"] = counterexamples
    result["relative_decision"] = np.where(now_score > wait_score, "NOW", "WAIT")
    result["desired_position"] = _desired_position(result["target_side"])
    if result["desired_position"].isna().any():
        raise TwinContractError("threshold_frontier: target_side must be L or H")

    if "current_position" in result.columns:
        positions = result["current_position"].astype(str).str.upper()
    else:
        positions = pd.Series(current_position, index=result.index)
        result["current_position"] = positions
    same_direction = positions.eq(result["desired_position"])
    action = np.where(
        same_direction,
        "HOLD",
        np.where(
            result["relative_decision"].eq("NOW"),
            "NOW",
            np.where(result["rearm_event"].astype(bool), "REARM", "WAIT"),
        ),
    )
    result["runtime_action"] = action
    result["rearm_and_now"] = result["rearm_event"].astype(bool) & result[
        "relative_decision"
    ].eq("NOW")
    result["position_transition"] = [
        _transition(str(current), str(desired), str(selected))
        for current, desired, selected in zip(positions, result["desired_position"], action)
    ]
    if not set(result["runtime_action"]).issubset(RUNTIME_ACTIONS):
        raise TwinContractError("threshold_frontier: emitted unknown fixed action")

    result["stage_1_position_state"] = positions
    result["stage_2_target_direction"] = result["desired_position"]
    result["stage_3_anchor_selection"] = "previous_wave|current_candidate|previous_candidate"
    result["stage_4_relative_measurement"] = "time|move|speed|rebound"
    result["stage_5_sequence_divergence"] = "candidate|candle|volume|8tf_macd"
    result["stage_6_historical_retrieval"] = supporting + counterexamples
    result["stage_7_counterexample_check"] = counterexamples
    result["stage_8_fixed_action"] = result["runtime_action"]
    result["stage_9_position_transition"] = result["position_transition"]
    return result
