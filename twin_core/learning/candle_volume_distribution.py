"""Causal candle and volume relations on the shared five-minute clock."""

from __future__ import annotations

import numpy as np
import pandas as pd

from twin_core import require_columns


def compute_candle_volume_distribution(prefix: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        prefix.columns,
        {"global_idx", "case_id", "open", "high", "low", "close", "volume", "target_side"},
        "candle_volume_distribution",
    )
    result = prefix.copy()
    unique = result.drop_duplicates("global_idx").sort_values("global_idx").copy()
    candle_range = (unique["high"] - unique["low"]).replace(0, np.nan)
    body = unique["close"] - unique["open"]
    unique["range_to_close"] = candle_range / unique["close"].abs().replace(0, np.nan)
    unique["signed_body_ratio"] = body / candle_range
    unique["upper_wick_ratio"] = (unique["high"] - unique[["open", "close"]].max(axis=1)) / candle_range
    unique["lower_wick_ratio"] = (unique[["open", "close"]].min(axis=1) - unique["low"]) / candle_range
    unique["close_location"] = (unique["close"] - unique["low"]) / candle_range
    volume_base = unique["volume"].rolling(20, min_periods=1).median().replace(0, np.nan)
    unique["volume_relative_20"] = unique["volume"] / volume_base
    unique["volume_change"] = unique["volume"].pct_change(fill_method=None)
    feature_columns = [
        "range_to_close",
        "signed_body_ratio",
        "upper_wick_ratio",
        "lower_wick_ratio",
        "close_location",
        "volume_relative_20",
        "volume_change",
    ]
    mapped = unique.set_index("global_idx")[feature_columns]
    for column in feature_columns:
        result[column] = result["global_idx"].map(mapped[column])
    # A fall supports an approaching L; a rise supports an approaching H.
    direction = np.where(result["target_side"].eq("L"), -1.0, 1.0)
    result["directional_body"] = result["signed_body_ratio"] * direction
    result["directional_wick_rejection"] = np.where(
        result["target_side"].eq("L"), result["lower_wick_ratio"], result["upper_wick_ratio"]
    )
    grouped = result.groupby("case_id", sort=False)
    for column in ["range_to_close", "directional_body", "directional_wick_rejection", "volume_relative_20"]:
        result[f"{column}_change_1"] = grouped[column].diff()
        result[f"{column}_change_3"] = grouped[column].diff(3)
    return result
