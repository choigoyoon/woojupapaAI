"""Candidate creation, replacement and survival relationships."""

from __future__ import annotations

import numpy as np
import pandas as pd

from twin_core import require_columns


def compute_flow_order_distribution(prefix: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        prefix.columns,
        {
            "case_id",
            "step_id",
            "candidate_step",
            "candidate_age_ratio",
            "rearm_event",
            "relative_move",
            "relative_rebound",
            "relative_speed",
            "directional_body",
            "volume_relative_20",
            "macd_5m_directional_slope",
        },
        "flow_order_distribution",
    )
    result = prefix.copy()
    grouped = result.groupby("case_id", sort=False)

    result["extension_increment"] = grouped["relative_move"].diff()
    result["rebound_increment"] = grouped["relative_rebound"].diff()
    result["speed_increment"] = grouped["relative_speed"].diff()
    result["previous_candidate_step"] = grouped["candidate_step"].shift()

    interval = result["candidate_step"] - result["previous_candidate_step"]
    result["rearm_interval"] = interval.where(result["rearm_event"])
    result["previous_rearm_interval"] = result.groupby("case_id", sort=False)[
        "rearm_interval"
    ].shift()
    result["rearm_interval_ratio"] = result["rearm_interval"] / result[
        "previous_rearm_interval"
    ].replace(0, np.nan)

    force = result["directional_body"] * result["volume_relative_20"]
    result["candle_volume_force"] = force
    result["force_increment"] = result.groupby("case_id", sort=False)[
        "candle_volume_force"
    ].diff()
    price_sign = np.sign(result["extension_increment"])
    result["price_vs_volume_divergence"] = -price_sign * np.sign(
        result.groupby("case_id", sort=False)["volume_relative_20"].diff()
    )
    result["price_vs_candle_force_divergence"] = -price_sign * np.sign(
        result["force_increment"]
    )
    result["price_vs_macd_5m_divergence"] = -price_sign * np.sign(
        result["macd_5m_directional_slope"]
    )
    epsilon = np.finfo(float).eps
    result["time_without_price_progress"] = result["candidate_age_ratio"] / (
        result["extension_increment"].abs() + epsilon
    )
    result["replacement_efficiency"] = result["extension_increment"] / (
        result["rearm_interval"].abs() + epsilon
    )
    return result
