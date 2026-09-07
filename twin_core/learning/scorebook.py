"""Integrity and outcome scoring for learned relative-action rules."""

from __future__ import annotations

import pandas as pd

from twin_core import FORBIDDEN_RUNTIME_FIELDS, TwinContractError, require_columns
from twin_core.learning.rule_induction import assert_runtime_safe_rules


def score_candidate_outcomes(
    frontier: pd.DataFrame,
    rules: pd.DataFrame | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    require_columns(
        frontier.columns,
        {"case_id", "step_id", "fixed_action", "relative_decision"},
        "scorebook.frontier",
    )
    if rules is not None and not rules.empty:
        assert_runtime_safe_rules(rules)
    actual_now = frontier["fixed_action"].eq("NOW")
    predicted_now = frontier["relative_decision"].eq("NOW")
    exact = actual_now & predicted_now
    premature = ~actual_now & predicted_now
    missed = actual_now & ~predicted_now
    diagnostic = frontier.loc[
        exact | premature | missed,
        [
            "case_id",
            "step_id",
            "fixed_action",
            "relative_decision",
            "supporting_rule_count",
            "counterexample_rule_count",
        ],
    ].copy()
    diagnostic["outcome"] = "EXACT"
    diagnostic.loc[premature[diagnostic.index], "outcome"] = "PREMATURE"
    diagnostic.loc[missed[diagnostic.index], "outcome"] = "MISSED"
    event_rows = int(actual_now.sum())
    metrics = {
        "rows": int(len(frontier)),
        "event_rows": event_rows,
        "exact_now": int(exact.sum()),
        "premature_now": int(premature.sum()),
        "missed_now": int(missed.sum()),
        "event_recall": float(exact.sum() / event_rows) if event_rows else None,
        "predicted_now_precision": float(exact.sum() / predicted_now.sum())
        if predicted_now.any()
        else None,
        "cases": int(frontier["case_id"].nunique()),
        "future_fields_in_rules": sorted(
            set(rules["feature"]).intersection(FORBIDDEN_RUNTIME_FIELDS)
        )
        if rules is not None and not rules.empty
        else [],
    }
    if metrics["future_fields_in_rules"]:
        raise TwinContractError("scorebook: offline answer leaked into runtime rules")
    return metrics, diagnostic.reset_index(drop=True)
