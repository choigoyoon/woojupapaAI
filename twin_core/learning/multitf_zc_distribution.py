"""Causal multi-timeframe MACD prefixes.

Each higher-timeframe value is recalculated from the current forming candle
without committing that provisional candle more than once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from twin_core import require_columns


TIMEFRAMES = {
    "5m": 5 * 60,
    "15m": 15 * 60,
    "30m": 30 * 60,
    "1h": 60 * 60,
    "2h": 2 * 60 * 60,
    "4h": 4 * 60 * 60,
    "1d": 24 * 60 * 60,
    "1w": 7 * 24 * 60 * 60,
}
TIMEFRAME_OFFSETS = {"1w": 4 * 24 * 60 * 60}  # Monday 00:00 UTC from Unix Thursday.


def _ema_step(value: float, previous: float | None, span: int) -> float:
    if previous is None:
        return value
    alpha = 2.0 / (span + 1.0)
    return alpha * value + (1.0 - alpha) * previous


def _forming_macd(close: np.ndarray, buckets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hist = np.empty(len(close), dtype=np.float64)
    committed_fast = committed_slow = committed_signal = None
    provisional_fast = provisional_slow = provisional_signal = None
    previous_bucket = None
    for index, (value, bucket) in enumerate(zip(close, buckets)):
        if previous_bucket is not None and bucket != previous_bucket:
            committed_fast = provisional_fast
            committed_slow = provisional_slow
            committed_signal = provisional_signal
        provisional_fast = _ema_step(float(value), committed_fast, 12)
        provisional_slow = _ema_step(float(value), committed_slow, 26)
        line = provisional_fast - provisional_slow
        provisional_signal = _ema_step(line, committed_signal, 9)
        hist[index] = line - provisional_signal
        previous_bucket = bucket
    slope = np.diff(hist, prepend=hist[0])
    return hist, slope


def compute_all_8tf_zc_distributions(prefix: pd.DataFrame) -> pd.DataFrame:
    require_columns(prefix.columns, {"global_idx", "timestamp", "close", "target_side"}, "multitf_zc")
    result = prefix.copy()
    unique = result.drop_duplicates("global_idx").sort_values("global_idx").copy()
    timestamp_ns = pd.to_datetime(unique["timestamp"], utc=True).astype("int64").to_numpy()
    close = unique["close"].to_numpy(dtype=np.float64)
    feature_columns: list[str] = []
    for name, seconds in TIMEFRAMES.items():
        offset_ns = TIMEFRAME_OFFSETS.get(name, 0) * 1_000_000_000
        buckets = (timestamp_ns - offset_ns) // (seconds * 1_000_000_000)
        hist, slope = _forming_macd(close, buckets)
        scale = np.maximum(np.abs(close), np.finfo(float).eps)
        unique[f"macd_{name}_hist_relative"] = hist / scale
        unique[f"macd_{name}_slope_relative"] = slope / scale
        unique[f"macd_{name}_sign"] = np.sign(hist).astype(np.int8)
        feature_columns.extend(
            [f"macd_{name}_hist_relative", f"macd_{name}_slope_relative", f"macd_{name}_sign"]
        )
    mapped = unique.set_index("global_idx")[feature_columns]
    for column in feature_columns:
        result[column] = result["global_idx"].map(mapped[column])
    # Direction-adjusted positive values always mean progress toward the target.
    direction = np.where(result["target_side"].eq("L"), -1.0, 1.0)
    for name in TIMEFRAMES:
        result[f"macd_{name}_directional_slope"] = result[f"macd_{name}_slope_relative"] * direction
    return result

