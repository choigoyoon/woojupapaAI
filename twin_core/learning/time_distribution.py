"""Relative time and movement relationships; no market thresholds live here."""

from __future__ import annotations

import numpy as np
import pandas as pd

from twin_core import require_columns


def compute_time_distribution(prefix: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        prefix.columns,
        {"case_id", "step_id", "elapsed_ratio", "candidate_age", "previous_wave_bars", "relative_move"},
        "time_distribution",
    )
    result = prefix.copy()
    denominator = result["elapsed_ratio"].replace(0, np.nan)
    result["candidate_age_ratio"] = result["candidate_age"] / result["previous_wave_bars"].clip(lower=1)
    result["relative_speed"] = result["relative_move"] / denominator
    grouped = result.groupby("case_id", sort=False)
    for lag in (1, 2, 6, 12):
        result[f"relative_move_change_{lag}"] = grouped["relative_move"].diff(lag)
        result[f"relative_speed_change_{lag}"] = grouped["relative_speed"].diff(lag)
        result[f"candidate_age_change_{lag}"] = grouped["candidate_age_ratio"].diff(lag)
    return result
