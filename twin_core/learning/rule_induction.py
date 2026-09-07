"""Learn market boundaries from fixed actions and relative observations."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from twin_core import FORBIDDEN_RUNTIME_FIELDS, TwinContractError, require_columns
from twin_core.learning.context_partition import classify_entry_family


RELATIVE_FEATURES = (
    "elapsed_ratio",
    "candidate_age_ratio",
    "relative_move",
    "relative_rebound",
    "relative_speed",
    "relative_move_change_1",
    "relative_move_change_2",
    "relative_move_change_6",
    "relative_speed_change_1",
    "candidate_age_change_1",
    "range_to_close",
    "directional_body",
    "directional_wick_rejection",
    "volume_relative_20",
    "volume_change",
    "extension_increment",
    "rebound_increment",
    "speed_increment",
    "rearm_interval_ratio",
    "candle_volume_force",
    "force_increment",
    "price_vs_volume_divergence",
    "price_vs_candle_force_divergence",
    "price_vs_macd_5m_divergence",
    "time_without_price_progress",
    "replacement_efficiency",
    "tf_alignment_fraction",
    "macd_5m_directional_slope",
    "macd_15m_directional_slope",
    "macd_30m_directional_slope",
    "macd_1h_directional_slope",
    "macd_2h_directional_slope",
    "macd_4h_directional_slope",
    "macd_1d_directional_slope",
    "macd_1w_directional_slope",
)


def _best_observed_split(values: np.ndarray, labels: np.ndarray) -> dict[str, object] | None:
    order = np.argsort(values, kind="mergesort")
    x = values[order]
    y = labels[order].astype(np.int64)
    valid_split = np.flatnonzero(x[:-1] < x[1:])
    positives, negatives = int(y.sum()), int(len(y) - y.sum())
    if not len(valid_split) or not positives or not negatives:
        return None
    positive_left = np.cumsum(y)[valid_split]
    count_left = valid_split + 1
    negative_left = count_left - positive_left
    positive_right = positives - positive_left
    negative_right = negatives - negative_left

    score_upper = 0.5 * (positive_right / positives + negative_left / negatives)
    score_lower = 0.5 * (positive_left / positives + negative_right / negatives)
    upper_index = int(np.argmax(score_upper))
    lower_index = int(np.argmax(score_lower))
    if score_upper[upper_index] >= score_lower[lower_index]:
        chosen, operator, score = upper_index, ">", float(score_upper[upper_index])
        true_now = int(positive_right[chosen])
        true_wait = int(negative_right[chosen])
    else:
        chosen, operator, score = lower_index, "<=", float(score_lower[lower_index])
        true_now = int(positive_left[chosen])
        true_wait = int(negative_left[chosen])
    split_at = int(valid_split[chosen])
    threshold = float(x[split_at] + (x[split_at + 1] - x[split_at]) / 2.0)
    return {
        "operator": operator,
        "threshold": threshold,
        "balanced_accuracy": score,
        "now_support": positives,
        "wait_support": negatives,
        "true_now_support": true_now,
        "true_wait_support": true_wait,
    }


def assert_runtime_safe_rules(rules: pd.DataFrame) -> None:
    require_columns(rules.columns, {"feature", "threshold", "operator"}, "rule_induction.rules")
    forbidden = sorted(set(rules["feature"]).intersection(FORBIDDEN_RUNTIME_FIELDS))
    if forbidden:
        raise TwinContractError(f"rule_induction: future/offline fields in rules {forbidden}")
    invalid = sorted(set(rules["feature"]).difference(RELATIVE_FEATURES))
    if invalid:
        raise TwinContractError(f"rule_induction: non-relative rule features {invalid}")


def fit_independent_evidence(
    observations: pd.DataFrame,
    *,
    feature_names: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    require_columns(
        observations.columns,
        {"fixed_action", "target_side", "market_context"},
        "rule_induction",
    )
    requested = tuple(feature_names or RELATIVE_FEATURES)
    forbidden = sorted(set(requested).intersection(FORBIDDEN_RUNTIME_FIELDS))
    if forbidden:
        raise TwinContractError(f"rule_induction: forbidden training inputs {forbidden}")
    features = [name for name in requested if name in observations.columns]
    records: list[dict[str, object]] = []
    for (side, context), group in observations.groupby(
        ["target_side", "market_context"], sort=True, dropna=False
    ):
        labels = group["fixed_action"].eq("NOW").to_numpy()
        if labels.all() or not labels.any():
            continue
        for feature in features:
            numeric = pd.to_numeric(group[feature], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(numeric)
            learned = _best_observed_split(numeric[finite], labels[finite])
            if learned is None or float(learned["balanced_accuracy"]) <= 0.5:
                continue
            identity = f"{side}|{context}|{feature}|{learned['operator']}|{learned['threshold']:.17g}"
            records.append(
                {
                    "rule_id": hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16],
                    "target_side": str(side),
                    "market_context": str(context),
                    "entry_family": classify_entry_family(feature),
                    "feature": feature,
                    **learned,
                    "source_rows": int(finite.sum()),
                    "boundary_source": "observed_fixed_action_frontier",
                }
            )
    columns = [
        "rule_id",
        "target_side",
        "market_context",
        "entry_family",
        "feature",
        "operator",
        "threshold",
        "balanced_accuracy",
        "now_support",
        "wait_support",
        "true_now_support",
        "true_wait_support",
        "source_rows",
        "boundary_source",
    ]
    rules = pd.DataFrame.from_records(records, columns=columns)
    if not rules.empty:
        assert_runtime_safe_rules(rules)
        rules = rules.sort_values(
            ["balanced_accuracy", "now_support", "rule_id"], ascending=[False, False, True]
        ).reset_index(drop=True)
    return rules
