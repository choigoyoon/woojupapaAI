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
    log_odds = np.zeros(len(result), dtype=float)
    retrieved = np.zeros(len(result), dtype=np.int32)
    supporting = np.zeros(len(result), dtype=np.int32)
    counterexamples = np.zeros(len(result), dtype=np.int32)

    if not rules.empty:
        assert_runtime_safe_rules(rules)
        # Correlated features must not become duplicate votes. Use the strongest
        # learned relation from each of the six independent evidence families.
        selected = (
            rules.sort_values(["balanced_accuracy", "source_rows"], ascending=[False, False])
            .drop_duplicates(["target_side", "market_context", "entry_family"])
            .reset_index(drop=True)
        )
        for (side, context), group_rules in selected.groupby(
            ["target_side", "market_context"], sort=False
        ):
            group_mask = (
                result["target_side"].eq(side).to_numpy()
                & result["market_context"].eq(context).to_numpy()
            )
            if not group_mask.any():
                continue
            first = group_rules.iloc[0]
            group_now = float(first["now_support"])
            group_wait = float(first["wait_support"])
            prior = np.log((group_now + 0.5) / (group_wait + 0.5))
            log_odds[group_mask] = prior
            if prior < 0:
                wait_score[group_mask] += -prior
            else:
                now_score[group_mask] += prior

            for rule in group_rules.itertuples(index=False):
                if rule.feature not in result.columns:
                    raise TwinContractError(
                        f"threshold_frontier: missing learned feature {rule.feature}"
                    )
                value = pd.to_numeric(result[rule.feature], errors="coerce").to_numpy(dtype=float)
                applicable = group_mask & np.isfinite(value)
                comparison = (
                    value <= rule.threshold if rule.operator == "<=" else value > rule.threshold
                )
                sensitivity = (float(rule.true_now_support) + 0.5) / (
                    float(rule.now_support) + 1.0
                )
                false_positive = (float(rule.true_wait_support) + 0.5) / (
                    float(rule.wait_support) + 1.0
                )
                true_lr = np.log(sensitivity / false_positive)
                false_lr = np.log((1.0 - sensitivity) / (1.0 - false_positive))
                contribution = np.where(comparison, true_lr, false_lr)
                log_odds[applicable] += contribution[applicable]
                now_score[applicable] += np.maximum(contribution[applicable], 0.0)
                wait_score[applicable] += np.maximum(-contribution[applicable], 0.0)
                retrieved[applicable] += 1
                supporting[applicable & (contribution > 0)] += 1
                counterexamples[applicable & (contribution <= 0)] += 1

    result["now_evidence"] = now_score
    result["wait_evidence"] = wait_score
    result["historical_log_odds"] = log_odds
    result["historical_now_probability"] = 1.0 / (1.0 + np.exp(-np.clip(log_odds, -700, 700)))
    result["retrieved_rule_count"] = retrieved
    result["supporting_rule_count"] = supporting
    result["counterexample_rule_count"] = counterexamples
    result["relative_decision"] = np.where((retrieved > 0) & (log_odds > 0), "NOW", "WAIT")
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
    result["stage_6_historical_retrieval"] = retrieved
    result["stage_7_counterexample_check"] = counterexamples
    result["stage_8_fixed_action"] = result["runtime_action"]
    result["stage_9_position_transition"] = result["position_transition"]
    return result

