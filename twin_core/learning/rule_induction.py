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
    positives, negatives = int(labels.sum()), int(len(labels) - labels.sum())
    if not positives or not negatives:
        return None
    now_values = values[labels]
    wait_values = values[~labels]
    candidates: list[tuple[int, str, float]] = []

    # Start at the most selective edge and widen until the first historical
    # counterexample would enter. The midpoint is therefore observed, not set
    # by an analyst (and is not an RSI-style universal number).
    wait_max = float(np.max(wait_values))
    upper_now = now_values[now_values > wait_max]
    if len(upper_now):
        nearest_now = float(np.min(upper_now))
        candidates.append((int(len(upper_now)), ">", wait_max + (nearest_now - wait_max) / 2.0))
    wait_min = float(np.min(wait_values))
    lower_now = now_values[now_values < wait_min]
    if len(lower_now):
        nearest_now = float(np.max(lower_now))
        candidates.append((int(len(lower_now)), "<=", nearest_now + (wait_min - nearest_now) / 2.0))
    if not candidates:
        return None
    true_now, operator, threshold = max(candidates, key=lambda item: (item[0], item[1] == ">"))
    recall = true_now / positives
    score = 0.5 * (recall + 1.0)  # specificity is one at the learned frontier.
    return {
        "operator": operator,
        "threshold": threshold,
        "balanced_accuracy": score,
        "now_support": positives,
        "wait_support": negatives,
        "true_now_support": true_now,
        "true_wait_support": 0,
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
                    "boundary_source": "first_counterexample_frontier",
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

