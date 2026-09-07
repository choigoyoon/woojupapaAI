"""Keep failures as explicit cases instead of averaging them away."""

from __future__ import annotations

import numpy as np
import pandas as pd

from twin_core import require_columns


def build_repair_queue(frontier: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        frontier.columns,
        {
            "case_id",
            "step_id",
            "fixed_action",
            "relative_decision",
            "supporting_rule_count",
            "counterexample_rule_count",
            "market_context",
        },
        "repair_queue",
    )
    actual_now = frontier["fixed_action"].eq("NOW")
    predicted_now = frontier["relative_decision"].eq("NOW")
    conflict = frontier["supporting_rule_count"].gt(0) & frontier[
        "counterexample_rule_count"
    ].gt(0)
    mask = actual_now.ne(predicted_now) | conflict
    queue = frontier.loc[
        mask,
        [
            "case_id",
            "step_id",
            "target_side",
            "market_context",
            "fixed_action",
            "relative_decision",
            "supporting_rule_count",
            "counterexample_rule_count",
            "now_evidence",
            "wait_evidence",
        ],
    ].copy()
    queue["repair_reason"] = np.select(
        [actual_now[mask] & ~predicted_now[mask], ~actual_now[mask] & predicted_now[mask], conflict[mask]],
        ["MISSED_NOW", "PREMATURE_NOW", "CONFLICTING_HISTORY"],
        default="UNSEEN_PATH",
    )
    return queue.sort_values(["case_id", "step_id"]).reset_index(drop=True)
