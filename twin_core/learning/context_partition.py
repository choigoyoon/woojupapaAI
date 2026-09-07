"""Partition observations without discarding mixed or opposed cases."""

from __future__ import annotations

import numpy as np
import pandas as pd

from twin_core import require_columns
from twin_core.learning.multitf_zc_distribution import TIMEFRAMES


ENTRY_FAMILIES = (
    "RELATIVE_PATH",
    "CANDIDATE_FLOW",
    "CANDLE_FORCE",
    "VOLUME_FLOW",
    "MULTITF_MOMENTUM",
    "DIVERGENCE",
)


def classify_entry_family(feature: str) -> str:
    name = feature.lower()
    if "divergence" in name:
        return "DIVERGENCE"
    if "macd" in name:
        return "MULTITF_MOMENTUM"
    if "volume" in name:
        return "VOLUME_FLOW"
    if any(token in name for token in ("body", "wick", "candle", "close_location", "range_")):
        return "CANDLE_FORCE"
    if any(token in name for token in ("candidate", "rearm", "replacement")):
        return "CANDIDATE_FLOW"
    return "RELATIVE_PATH"


def partition_contexts(prefix: pd.DataFrame) -> pd.DataFrame:
    hist_columns = [f"macd_{name}_hist_relative" for name in TIMEFRAMES]
    require_columns(prefix.columns, {"target_side", *hist_columns}, "context_partition")
    result = prefix.copy()
    direction = np.where(result["target_side"].eq("L"), -1.0, 1.0)
    votes = np.column_stack(
        [np.sign(result[column].to_numpy(dtype=float) * direction) for column in hist_columns]
    )
    positive = (votes > 0).sum(axis=1)
    negative = (votes < 0).sum(axis=1)
    width = len(hist_columns)
    result["tf_support_count"] = positive
    result["tf_opposition_count"] = negative
    result["tf_alignment_fraction"] = (positive - negative) / width
    result["market_context"] = np.select(
        [positive == width, negative == width], ["ALIGNED", "OPPOSED"], default="MIXED"
    )
    return result
