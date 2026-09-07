"""Execute the full learned Stage 03-11 relation model on closed 5-minute bars.

The router evaluates structured operands, intervals, relative changes and
persistence.  The 2,775 historical outputs are audit evidence only and are
never loaded as a decision whitelist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SCHEMA = "twin.executable-thought-program.v3"
STAGE_ORDER = ("03", "04", "05", "06", "07", "08", "09", "10", "11")
EXPECTED_OPERATORS = (
    "NEW_EXTREME_CANDIDATE_AXIS",
    "EXACT_MACRO_SIGN_SIGNATURE",
    "DECODE_RELATIVE_TIME",
    "PRESERVE_CANDLE_VOLUME_PROFILE",
    "PRESERVE_EIGHT_TIMEFRAME_MACD",
    "DECODE_LAST_ZC_ORDER",
    "EVALUATE_FULL_LEARNED_RELATION_MODEL",
    "APPLY_LEARNED_PERSISTENCE",
    "FIRST_READY_OR_OBSERVED_1H_ZC",
)
TIMEFRAMES = ("5m", "15m", "30m", "1h", "2h", "4h", "1d", "1w")
TIMEFRAME_MULTIPLIERS = {
    "5m": 1,
    "15m": 3,
    "30m": 6,
    "1h": 12,
    "2h": 24,
    "4h": 48,
    "1d": 288,
    "1w": 2016,
}
COUNTED_ZC_TIMEFRAMES = ("5m", "15m", "30m", "1h")
MISSING_VALUE = -9.0
EPSILON = 1e-12
CANONICAL_FEATURES = (
    "side_code",
    "wave_age_log",
    "time_ratio_prev1",
    "time_ratio_prev2",
    "move_ratio_prev1",
    "move_ratio_prev2",
    "wave_move_pct_so_far",
    "zone_move_pct_so_far",
    "speed_recent3",
    "speed_ratio_3_to_12",
    "candidate_age_log",
    "rearm_count_log",
    "rearm_gap_bars_log",
    "rearm_extension_pct",
    "rearm_extension_ratio",
    "new_extreme_now",
    "candidate_rejection_pct",
    "candidate_rejection_fraction",
    "candidate_rejection_speed",
    "candidate_rejection_range_units",
    "candle_direction_for_side",
    "candle_body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "close_position_for_side",
    "volume_ratio20",
    "volume_ratio72",
    "range_ratio20",
    "macd_5m_hist_for_side",
    "macd_5m_delta_for_side",
    "macd_5m_sign_for_side",
    "macd_5m_zc_count_so_far",
    "macd_5m_zc_age_log",
    "macd_15m_hist_for_side",
    "macd_15m_delta_for_side",
    "macd_15m_sign_for_side",
    "macd_15m_zc_count_so_far",
    "macd_15m_zc_age_log",
    "macd_30m_hist_for_side",
    "macd_30m_delta_for_side",
    "macd_30m_sign_for_side",
    "macd_30m_zc_count_so_far",
    "macd_30m_zc_age_log",
    "macd_1h_hist_for_side",
    "macd_1h_delta_for_side",
    "macd_1h_sign_for_side",
    "macd_1h_zc_count_so_far",
    "macd_1h_zc_age_log",
    "macd_2h_hist_for_side",
    "macd_2h_delta_for_side",
    "macd_2h_sign_for_side",
    "macd_2h_zc_age_log",
    "macd_4h_hist_for_side",
    "macd_4h_delta_for_side",
    "macd_4h_sign_for_side",
    "macd_4h_zc_age_log",
    "macd_1d_hist_for_side",
    "macd_1d_delta_for_side",
    "macd_1d_sign_for_side",
    "macd_1d_zc_age_log",
    "macd_1w_hist_for_side",
    "macd_1w_delta_for_side",
    "macd_1w_sign_for_side",
    "macd_1w_zc_age_log",
)
METHODS = (
    "WAVE_REARM_AGE",
    "MOMENTUM_SPEED",
    "CANDLE_REVERSAL",
    "VOLUME_RANGE",
    "MACRO_TREND",
    "MID_MACD_TREND",
)
METHOD_ORDER = {method: index for index, method in enumerate(METHODS)}


class RouterContractError(ValueError):
    pass


def _macd_arrays(close: pd.Series, multiplier: int) -> dict[str, np.ndarray]:
    line = close.ewm(span=12 * multiplier, adjust=False).mean() - close.ewm(
        span=26 * multiplier, adjust=False
    ).mean()
    histogram_raw = line - line.ewm(span=9 * multiplier, adjust=False).mean()
    histogram_scale = histogram_raw.abs().ewm(
        span=26 * multiplier, adjust=False
    ).mean()
    histogram = (
        histogram_raw / histogram_scale.where(histogram_scale > EPSILON, EPSILON)
    ).clip(-20.0, 20.0)
    delta_raw = histogram_raw.diff().fillna(0.0)
    delta_scale = delta_raw.abs().ewm(span=9 * multiplier, adjust=False).mean()
    delta = (delta_raw / delta_scale.where(delta_scale > EPSILON, EPSILON)).clip(
        -20.0, 20.0
    )
    raw_values = histogram_raw.to_numpy(dtype=float)
    sign = np.where(raw_values >= 0.0, 1.0, -1.0)
    crossing = np.r_[True, sign[1:] != sign[:-1]]
    positions = np.arange(len(close), dtype=np.int64)
    last_crossing = np.maximum.accumulate(np.where(crossing, positions, 0))
    return {
        "histogram": histogram.to_numpy(dtype=float),
        "delta": delta.to_numpy(dtype=float),
        "sign": sign,
        "crossing": crossing,
        "zc_age_log": np.log1p(positions - last_crossing),
    }


def build_canonical_observations(
    ohlcv: pd.DataFrame,
    *,
    first_1h_zone_index: int = 2,
    last_raw_position: int | None = None,
    missing_value: float = MISSING_VALUE,
) -> pd.DataFrame:
    """Recreate the existing 64 causal observations from closed 5m OHLCV."""

    required = ("open", "high", "low", "close", "volume")
    missing_columns = [name for name in required if name not in ohlcv.columns]
    if missing_columns:
        raise RouterContractError(f"OHLCV is missing columns: {missing_columns}")
    if len(ohlcv) == 0:
        raise RouterContractError("OHLCV contains no closed bars")
    raw = ohlcv.reset_index(drop=True)
    count = len(raw)
    positions = np.arange(count, dtype=np.int64)
    open_values = raw["open"].to_numpy(dtype=float)
    high_values = raw["high"].to_numpy(dtype=float)
    low_values = raw["low"].to_numpy(dtype=float)
    close_values = raw["close"].to_numpy(dtype=float)
    volume_values = raw["volume"].to_numpy(dtype=float)
    if not all(
        np.isfinite(values).all()
        for values in (open_values, high_values, low_values, close_values)
    ):
        raise RouterContractError("OHLC prices must be finite")

    close_series = pd.Series(close_values)
    macd = {
        timeframe: _macd_arrays(close_series, multiplier)
        for timeframe, multiplier in TIMEFRAME_MULTIPLIERS.items()
    }
    side = macd["1h"]["sign"]
    zone_starts = np.r_[0, np.flatnonzero(side[1:] != side[:-1]) + 1].astype(
        np.int64
    )
    zone_ends = np.r_[zone_starts[1:] - 1, count - 1].astype(np.int64)
    if first_1h_zone_index >= len(zone_starts):
        raise RouterContractError("OHLCV has not reached the first runtime 1h zone")
    zone_ids = np.empty(count, dtype=np.int64)
    pivots = np.empty(len(zone_starts), dtype=np.int64)
    for zone_index, (start, end) in enumerate(zip(zone_starts, zone_ends)):
        start_int, end_int = int(start), int(end)
        zone_ids[start_int : end_int + 1] = zone_index
        if side[start_int] > 0.0:
            pivots[zone_index] = start_int + int(
                np.argmax(high_values[start_int : end_int + 1])
            )
        else:
            pivots[zone_index] = start_int + int(
                np.argmin(low_values[start_int : end_int + 1])
            )

    candidate_position = np.empty(count, dtype=np.int64)
    candidate_price = np.empty(count, dtype=float)
    new_extreme = np.zeros(count, dtype=float)
    rearm_count = np.empty(count, dtype=np.int64)
    rearm_gap = np.empty(count, dtype=np.int64)
    rearm_extension = np.empty(count, dtype=float)
    rearm_extension_ratio = np.full(count, np.nan, dtype=float)
    for start, end in zip(zone_starts, zone_ends):
        start_int, end_int = int(start), int(end)
        current_position = start_int
        current_price = high_values[start_int] if side[start_int] > 0.0 else low_values[start_int]
        current_count = 1
        current_gap = 1
        latest_extension = (
            abs(current_price - open_values[start_int])
            / max(abs(open_values[start_int]), EPSILON)
            * 100.0
        )
        prior_extension = math.nan
        for position in range(start_int, end_int + 1):
            if position == start_int:
                is_new = True
            else:
                side_price = high_values[position] if side[position] > 0.0 else low_values[position]
                is_new = side_price > current_price if side[position] > 0.0 else side_price < current_price
                if is_new:
                    prior_price = current_price
                    prior_extension = latest_extension
                    latest_extension = (
                        abs(side_price - prior_price)
                        / max(abs(prior_price), EPSILON)
                        * 100.0
                    )
                    current_gap = position - current_position
                    current_position = position
                    current_price = side_price
                    current_count += 1
            candidate_position[position] = current_position
            candidate_price[position] = current_price
            new_extreme[position] = float(is_new)
            rearm_count[position] = current_count
            rearm_gap[position] = current_gap
            rearm_extension[position] = latest_extension
            if math.isfinite(prior_extension):
                rearm_extension_ratio[position] = latest_extension / max(prior_extension, EPSILON)

    runtime_zone_indexes = np.arange(first_1h_zone_index, len(zone_starts), dtype=np.int64)
    wave_age = np.zeros(count, dtype=float)
    time_ratio_prev1 = np.full(count, np.nan, dtype=float)
    time_ratio_prev2 = np.full(count, np.nan, dtype=float)
    move_ratio_prev1 = np.full(count, np.nan, dtype=float)
    move_ratio_prev2 = np.full(count, np.nan, dtype=float)
    wave_move = np.zeros(count, dtype=float)
    for runtime_zone, global_zone in enumerate(runtime_zone_indexes):
        start = int(zone_starts[global_zone])
        end = int(zone_ends[global_zone])
        rows = np.arange(start, end + 1, dtype=np.int64)
        if runtime_zone == 0:
            begin = start
            base_price = low_values[start] if side[start] > 0.0 else high_values[start]
            ages = rows - start
        else:
            previous_global_zone = int(runtime_zone_indexes[runtime_zone - 1])
            begin = int(pivots[previous_global_zone])
            base_price = high_values[begin] if side[begin] > 0.0 else low_values[begin]
            ages = rows - begin
        path_prices = high_values[begin : end + 1] if side[start] > 0.0 else low_values[begin : end + 1]
        path_candidate = np.maximum.accumulate(path_prices) if side[start] > 0.0 else np.minimum.accumulate(path_prices)
        current_move = (
            np.abs(path_candidate[rows - begin] - base_price)
            / max(abs(base_price), EPSILON)
            * 100.0
        )
        wave_age[rows] = ages
        wave_move[rows] = current_move
        if runtime_zone >= 2:
            pivot1 = int(pivots[runtime_zone_indexes[runtime_zone - 1]])
            pivot2 = int(pivots[runtime_zone_indexes[runtime_zone - 2]])
            price1 = high_values[pivot1] if side[pivot1] > 0.0 else low_values[pivot1]
            price2 = high_values[pivot2] if side[pivot2] > 0.0 else low_values[pivot2]
            prior_duration = max(pivot1 - pivot2, 1)
            prior_move = abs(price1 - price2) / max(abs(price2), EPSILON) * 100.0
            time_ratio_prev1[rows] = ages / prior_duration
            move_ratio_prev1[rows] = current_move / max(prior_move, EPSILON)
        if runtime_zone >= 3:
            pivot2 = int(pivots[runtime_zone_indexes[runtime_zone - 2]])
            pivot3 = int(pivots[runtime_zone_indexes[runtime_zone - 3]])
            price2 = high_values[pivot2] if side[pivot2] > 0.0 else low_values[pivot2]
            price3 = high_values[pivot3] if side[pivot3] > 0.0 else low_values[pivot3]
            prior_duration = max(pivot2 - pivot3, 1)
            prior_move = abs(price2 - price3) / max(abs(price3), EPSILON) * 100.0
            time_ratio_prev2[rows] = ages / prior_duration
            move_ratio_prev2[rows] = current_move / max(prior_move, EPSILON)

    zone_start_for_row = zone_starts[zone_ids]
    zone_age = positions - zone_start_for_row
    speed_values: dict[int, np.ndarray] = {}
    for lookback in (3, 12):
        prior = np.maximum(positions - lookback, zone_start_for_row)
        elapsed = np.maximum(positions - prior, 1)
        speed_values[lookback] = np.maximum(wave_move - wave_move[prior], 0.0) / elapsed
    speed_ratio = np.zeros(count, dtype=float)
    has_speed = speed_values[12] > 0.0
    speed_ratio[has_speed] = np.minimum(speed_values[3][has_speed] / speed_values[12][has_speed], 4.0)

    raw_range = high_values - low_values
    safe_range = np.maximum(raw_range, EPSILON)
    range_pct = safe_range / np.maximum(np.abs(open_values), EPSILON) * 100.0
    rejection_pct = (
        np.where(side > 0.0, candidate_price - close_values, close_values - candidate_price)
        / np.maximum(np.abs(candidate_price), EPSILON)
        * 100.0
    )
    zone_move_pct = (
        np.abs(candidate_price - open_values[zone_start_for_row])
        / np.maximum(np.abs(open_values[zone_start_for_row]), EPSILON)
        * 100.0
    )
    candidate_age = positions - candidate_position
    volume_series = pd.Series(volume_values)
    volume_ratios: dict[int, np.ndarray] = {}
    for window in (20, 72):
        rolling_median = volume_series.shift(1).rolling(window, min_periods=1).median()
        volume_ratios[window] = (
            volume_series / rolling_median.where(rolling_median.abs() > EPSILON, EPSILON)
        ).clip(0.0, 100.0).to_numpy(dtype=float)
    range_series = pd.Series(range_pct)
    range_median = range_series.shift(1).rolling(20, min_periods=1).median()
    range_ratio = (
        range_series / range_median.where(range_median.abs() > EPSILON, EPSILON)
    ).clip(upper=100.0).to_numpy(dtype=float)

    features: dict[str, np.ndarray] = {
        "side_code": side,
        "wave_age_log": np.log1p(wave_age),
        "time_ratio_prev1": time_ratio_prev1,
        "time_ratio_prev2": time_ratio_prev2,
        "move_ratio_prev1": move_ratio_prev1,
        "move_ratio_prev2": move_ratio_prev2,
        "wave_move_pct_so_far": wave_move,
        "zone_move_pct_so_far": zone_move_pct,
        "speed_recent3": speed_values[3],
        "speed_ratio_3_to_12": speed_ratio,
        "candidate_age_log": np.log1p(candidate_age),
        "rearm_count_log": np.log1p(rearm_count),
        "rearm_gap_bars_log": np.log1p(rearm_gap),
        "rearm_extension_pct": rearm_extension,
        "rearm_extension_ratio": rearm_extension_ratio,
        "new_extreme_now": new_extreme,
        "candidate_rejection_pct": rejection_pct,
        "candidate_rejection_fraction": rejection_pct / np.maximum(zone_move_pct, EPSILON),
        "candidate_rejection_speed": rejection_pct / np.maximum(candidate_age, 1),
        "candidate_rejection_range_units": rejection_pct / np.maximum(range_pct, EPSILON),
        "candle_direction_for_side": np.sign(close_values - open_values) * side,
        "candle_body_ratio": np.abs(close_values - open_values) / safe_range,
        "upper_wick_ratio": (high_values - np.maximum(open_values, close_values)) / safe_range,
        "lower_wick_ratio": (np.minimum(open_values, close_values) - low_values) / safe_range,
        "close_position_for_side": np.where(
            side > 0.0,
            (close_values - low_values) / safe_range,
            1.0 - (close_values - low_values) / safe_range,
        ),
        "volume_ratio20": volume_ratios[20],
        "volume_ratio72": volume_ratios[72],
        "range_ratio20": range_ratio,
    }
    for timeframe in TIMEFRAMES:
        witness = macd[timeframe]
        prefix = f"macd_{timeframe}_"
        features[prefix + "hist_for_side"] = witness["histogram"] * side
        features[prefix + "delta_for_side"] = witness["delta"] * side
        features[prefix + "sign_for_side"] = witness["sign"] * side
        features[prefix + "zc_age_log"] = witness["zc_age_log"]
        if timeframe in COUNTED_ZC_TIMEFRAMES:
            zc_count = np.zeros(count, dtype=float)
            running = 0
            for position in range(count):
                if position == 0 or side[position] != side[position - 1]:
                    running = 0
                if witness["crossing"][position]:
                    running += 1
                zc_count[position] = running
            features[prefix + "zc_count_so_far"] = zc_count

    if set(features) != set(CANONICAL_FEATURES):
        raise RouterContractError("OHLCV builder did not produce the existing 64")
    end_position = count - 1 if last_raw_position is None else int(last_raw_position)
    start_position = int(zone_starts[first_1h_zone_index])
    if not start_position <= end_position < count:
        raise RouterContractError("last_raw_position is outside the runtime range")
    selected = slice(start_position, end_position + 1)
    output = pd.DataFrame(
        {
            feature: np.where(
                np.isfinite(features[feature][selected]),
                features[feature][selected],
                missing_value,
            )
            for feature in CANONICAL_FEATURES
        }
    )
    selected_positions = positions[selected]
    selected_zone_ids = zone_ids[selected]
    output.insert(0, "seek_side", np.where(side[selected] > 0.0, "H", "L"))
    output.insert(0, "event_key", [f"Z{int(zone-first_1h_zone_index+1):06d}" for zone in selected_zone_ids])
    output.insert(0, "raw_position", selected_positions)
    if "timestamp" in raw.columns:
        output.insert(1, "timestamp", raw["timestamp"].to_numpy()[selected])
    output["observed_1h_zc_switch"] = selected_positions == zone_starts[selected_zone_ids]
    output["current_candidate_extreme"] = candidate_price[selected]
    output["zone_start_position"] = zone_starts[selected_zone_ids]
    output["zone_end_position"] = zone_ends[selected_zone_ids]
    output["zone_pivot_position"] = pivots[selected_zone_ids]
    return output


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _decode_age(value: Any) -> int | None:
    number = _number(value)
    if number is None or number == -9.0:
        return None
    return int(round(math.expm1(number)))


def _side_required(output: Mapping[str, Any], seek_side: str) -> int:
    value = output["output_value"]
    per_side = value.get("minimum_persistence_bars_by_side", {})
    return int(per_side.get(seek_side, value.get("minimum_persistence_bars", 1)))


def _output_rank(output: Mapping[str, Any]) -> tuple[Any, ...]:
    value = output["output_value"]
    order = output.get("artifact_order", {})
    return (
        -float(value["state_probability"]),
        -int(value["event_support"]),
        METHOD_ORDER[str(output["method"])],
        int(order.get("channel", 1_000_000)),
        int(order.get("rule", 1_000_000)),
        int(order.get("bin", 1_000_000)),
        str(output["calculation_id"]),
    )


class RuleRouter:
    def __init__(self, program: Mapping[str, Any]):
        self.program = dict(program)
        self._validate_program()
        contract = self.program["runtime_input_contract"]
        self.runtime_inputs = tuple(contract["features"])
        self.forbidden = frozenset(contract["forbidden"])
        self.calculations = tuple(self.program["runtime_calculations"])
        self.calculations_by_background = {
            background: tuple(
                calculation
                for calculation in self.calculations
                if calculation["background"] == background
            )
            for background in ("ALIGNED", "MIXED", "OPPOSED")
        }
        self.ranked_indexes = {
            (background, source): tuple(
                sorted(
                    (
                        index
                        for index, calculation in enumerate(self.calculations)
                        if calculation["background"] == background
                        and calculation["output_source"] == source
                    ),
                    key=lambda index: _output_rank(self.calculations[index]),
                )
            )
            for background in ("ALIGNED", "MIXED", "OPPOSED")
            for source in ("BASE", "CONTEXT")
        }
        self.selector = self.program["existing_selector_values"]
        self.current_event_key: str | None = None
        self.history: list[dict[str, Any]] = []
        self.group_signatures: dict[str, str | None] = {}
        self.group_runs: dict[str, int] = {}
        self.event_released = False

    @classmethod
    def load(cls, path: str | Path) -> "RuleRouter":
        with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
            program = json.load(handle)
        if not isinstance(program, dict):
            raise RouterContractError("executable program must be a JSON object")
        return cls(program)

    def _validate_program(self) -> None:
        if self.program.get("schema") != SCHEMA:
            raise RouterContractError("wrong executable thought-program schema")
        calculation = self.program["calculation_program"]
        if tuple(calculation["stage_order"]) != STAGE_ORDER:
            raise RouterContractError("Stage 03-11 order changed")
        operators = tuple(stage["executable_operator"] for stage in calculation["stages"])
        if operators != EXPECTED_OPERATORS:
            raise RouterContractError("an existing Stage 03-11 calculation was replaced")
        for stage in calculation["stages"]:
            for required in (
                "observation_start",
                "fixed_observations",
                "relative_calculations",
                "filters",
                "handoff_fields",
            ):
                if required not in stage:
                    raise RouterContractError(
                        f"Stage {stage['stage']} is missing {required}"
                    )
        contract = self.program["runtime_input_contract"]
        if int(contract["count"]) != 64 or len(contract["features"]) != 64:
            raise RouterContractError("runtime input contract is not the existing 64")
        output = self.program["runtime_calculation_contract"]
        if (
            int(output["count"]) != 10_863
            or int(output["base_interval_calculations"]) != 5_363
            or int(output["context_relation_calculations"]) != 5_500
            or bool(output["selected_historical_output_filter_used"])
        ):
            raise RouterContractError("runtime model is not the full 10,863 calculations")
        calculations = self.program["runtime_calculations"]
        if len(calculations) != 10_863 or len(
            {item["calculation_id"] for item in calculations}
        ) != 10_863:
            raise RouterContractError("runtime calculations are missing or duplicated")
        if self.program["calculation_program"].get("historical_evidence_dependency"):
            raise RouterContractError("historical evidence cannot select a runtime action")

    def _begin_event(self, event_key: str) -> None:
        if self.current_event_key != event_key:
            self.current_event_key = event_key
            self.history.clear()
            self.group_signatures.clear()
            self.group_runs.clear()
            self.event_released = False

    def _reset_evidence_runs_if_required(
        self, observation: Mapping[str, Any], *, new_extreme_now: bool
    ) -> list[str]:
        reasons: list[str] = []
        if new_extreme_now:
            reasons.append("NEW_EXTREME")
        if self.history:
            previous = self.history[-1]
            if (
                previous.get("macd_1h_zc_count_so_far")
                != observation.get("macd_1h_zc_count_so_far")
                or previous.get("macd_1h_sign_for_side")
                != observation.get("macd_1h_sign_for_side")
            ):
                reasons.append("ONE_HOUR_WAVE_CHANGE")
        if reasons:
            self.group_signatures.clear()
            self.group_runs.clear()
        return reasons

    def _validate_observation(self, observation: Mapping[str, Any]) -> None:
        forbidden = sorted(set(observation).intersection(self.forbidden))
        if forbidden:
            raise RouterContractError(f"future/study fields reached the router: {forbidden}")
        missing = [feature for feature in self.runtime_inputs if feature not in observation]
        if missing:
            raise RouterContractError(f"closed-bar observation is missing {len(missing)} inputs")

    @staticmethod
    def _batch_background(observations: pd.DataFrame) -> np.ndarray:
        signs = []
        for timeframe in ("4h", "1d", "1w"):
            values = observations[f"macd_{timeframe}_sign_for_side"].to_numpy(
                dtype=float
            )
            signs.append(np.where(values > 0.0, "+", "-"))
        raw = np.char.add(np.char.add(signs[0], signs[1]), signs[2])
        return np.where(raw == "+++", "ALIGNED", np.where(raw == "---", "OPPOSED", "MIXED"))

    @staticmethod
    def _batch_operand_values(
        observations: pd.DataFrame,
        event_codes: np.ndarray,
        operand: Mapping[str, Any],
    ) -> np.ndarray:
        feature = str(operand["feature"])
        current = observations[feature].to_numpy(dtype=float)
        if operand["calculation"] == "CURRENT_VALUE":
            return current
        if operand["calculation"] != "CURRENT_MINUS_PRIOR":
            raise RouterContractError(
                f"unknown operand calculation: {operand['calculation']}"
            )
        lookback = int(operand["lookback_closed_5m_bars"])
        previous = np.full(len(current), np.nan, dtype=float)
        if lookback > 0:
            same_event = np.zeros(len(current), dtype=bool)
            same_event[lookback:] = event_codes[lookback:] == event_codes[:-lookback]
            previous[lookback:] = current[:-lookback]
            previous[~same_event] = np.nan
        return current - previous

    def _batch_winners(
        self,
        observations: pd.DataFrame,
        event_codes: np.ndarray,
        backgrounds: np.ndarray,
        source: str,
    ) -> np.ndarray:
        """Select the strongest satisfied existing relation on every row."""

        if source not in {"BASE", "CONTEXT"}:
            raise RouterContractError("batch source must be BASE or CONTEXT")
        winners = np.full(len(observations), -1, dtype=np.int32)
        operand_cache: dict[tuple[str, str, int], np.ndarray] = {}

        def values(operand: Mapping[str, Any]) -> np.ndarray:
            key = (
                str(operand["feature"]),
                str(operand["calculation"]),
                int(operand["lookback_closed_5m_bars"]),
            )
            if key not in operand_cache:
                operand_cache[key] = self._batch_operand_values(
                    observations, event_codes, operand
                )
            return operand_cache[key]

        for background in ("ALIGNED", "MIXED", "OPPOSED"):
            background_mask = backgrounds == background
            for calculation_index in self.ranked_indexes[(background, source)]:
                available = background_mask & (winners < 0)
                if not available.any():
                    break
                output = self.calculations[calculation_index]
                body = output["calculation"]
                if body["operator"] == "INTERVAL_CONTAINS":
                    current = values(body["operand"])
                    matched = np.isfinite(current)
                    lower = body.get("lower_exclusive")
                    upper = body.get("upper_inclusive")
                    if lower is not None:
                        matched &= current > float(lower)
                    if upper is not None:
                        matched &= current <= float(upper)
                elif body["operator"] == "ALL_CONDITIONS_TRUE":
                    matched = np.ones(len(observations), dtype=bool)
                    for condition in body.get("conditions", []):
                        current = values(condition["operand"])
                        if condition["operator"] == ">":
                            matched &= current > float(condition["learned_boundary"])
                        elif condition["operator"] == "<=":
                            matched &= current <= float(condition["learned_boundary"])
                        else:
                            raise RouterContractError("unknown learned comparison")
                        if not (matched & available).any():
                            break
                else:
                    raise RouterContractError("unknown batch calculation operator")
                winners[available & matched] = calculation_index
        return winners

    @staticmethod
    def _safe_ratio(numerator: Any, denominator: Any) -> float | None:
        top = _number(numerator)
        bottom = _number(denominator)
        if top is None or bottom is None or bottom == 0.0:
            return None
        return top / bottom

    def _materialize_observation(
        self,
        observation: Mapping[str, Any],
        *,
        seek_side: str,
        current_candidate_extreme: float | None,
    ) -> dict[str, Any]:
        """Execute supplied module formulas when their raw operands are present."""

        row = dict(observation)
        if current_candidate_extreme is not None and {"high", "low"} <= row.keys():
            current = float(row["high"] if seek_side == "H" else row["low"])
            row["new_extreme_now"] = float(
                current > current_candidate_extreme
                if seek_side == "H"
                else current < current_candidate_extreme
            )

        for target, raw in (
            ("wave_age_log", "wave_age_bars"),
            ("candidate_age_log", "candidate_age_bars"),
            ("rearm_count_log", "rearm_count"),
            ("rearm_gap_bars_log", "rearm_gap_bars"),
        ):
            if raw in row:
                value = _number(row[raw])
                if value is not None and value >= 0:
                    row[target] = math.log1p(value)

        if {"open", "high", "low", "close"} <= row.keys():
            open_value = float(row["open"])
            high_value = float(row["high"])
            low_value = float(row["low"])
            close_value = float(row["close"])
            candle_range = high_value - low_value
            if candle_range > 0.0:
                row["candle_body_ratio"] = (
                    abs(close_value - open_value) / candle_range
                )
                row["upper_wick_ratio"] = (
                    high_value - max(open_value, close_value)
                ) / candle_range
                row["lower_wick_ratio"] = (
                    min(open_value, close_value) - low_value
                ) / candle_range
                side_code = _number(row.get("side_code"))
                if side_code is not None:
                    direction = (
                        1.0
                        if close_value > open_value
                        else -1.0
                        if close_value < open_value
                        else 0.0
                    )
                    close_location = (close_value - low_value) / candle_range
                    row["candle_direction_for_side"] = direction * side_code
                    row["close_position_for_side"] = (
                        close_location if side_code > 0 else 1.0 - close_location
                    )
            if "volume" in row:
                for target, reference in (
                    ("volume_ratio20", "volume_sma20"),
                    ("volume_ratio72", "volume_sma72"),
                ):
                    ratio = self._safe_ratio(row["volume"], row.get(reference))
                    if ratio is not None:
                        row[target] = ratio
            range_ratio = self._safe_ratio(candle_range, row.get("range_sma20"))
            if range_ratio is not None:
                row["range_ratio20"] = range_ratio

        side_code = _number(row.get("side_code"))
        if side_code is not None:
            for timeframe in TIMEFRAMES:
                prefix = f"macd_{timeframe}_"
                for metric in ("hist", "delta", "sign"):
                    target = prefix + metric + "_for_side"
                    raw = prefix + metric
                    value = _number(row.get(raw))
                    if value is not None:
                        row[target] = value * side_code
                age_target = prefix + "zc_age_log"
                age_raw = prefix + "zc_age_bars"
                age = _number(row.get(age_raw))
                if age is not None and age >= 0:
                    row[age_target] = math.log1p(age)
        return row

    def _operand_value(
        self, operand: Mapping[str, Any], observation: Mapping[str, Any]
    ) -> float | None:
        feature = str(operand["feature"])
        current = _number(observation.get(feature))
        calculation = operand["calculation"]
        if calculation == "CURRENT_VALUE":
            return current
        if calculation != "CURRENT_MINUS_PRIOR":
            raise RouterContractError(f"unknown operand calculation: {calculation}")
        lookback = int(operand["lookback_closed_5m_bars"])
        if current is None or lookback <= 0 or len(self.history) < lookback:
            return None
        previous = _number(self.history[-lookback].get(feature))
        return None if previous is None else current - previous

    @staticmethod
    def _comparison(value: float | None, operator: str, boundary: Any) -> bool:
        if value is None:
            return False
        learned = float(boundary)
        if operator == ">":
            return value > learned
        if operator == "<=":
            return value <= learned
        raise RouterContractError(f"unknown learned comparison: {operator}")

    def _matches(self, output: Mapping[str, Any], observation: Mapping[str, Any]) -> bool:
        calculation = output["calculation"]
        operator = calculation["operator"]
        if operator == "INTERVAL_CONTAINS":
            value = self._operand_value(calculation["operand"], observation)
            if value is None:
                return False
            lower = calculation.get("lower_exclusive")
            upper = calculation.get("upper_inclusive")
            return (lower is None or value > float(lower)) and (
                upper is None or value <= float(upper)
            )
        if operator == "ALL_CONDITIONS_TRUE":
            return all(
                self._comparison(
                    self._operand_value(condition["operand"], observation),
                    str(condition["operator"]),
                    condition["learned_boundary"],
                )
                for condition in calculation["conditions"]
            )
        raise RouterContractError(f"unknown output calculation: {operator}")

    @staticmethod
    def _background(observation: Mapping[str, Any]) -> tuple[str, str]:
        symbols: list[str] = []
        for timeframe in ("4h", "1d", "1w"):
            value = _number(observation[f"macd_{timeframe}_sign_for_side"])
            if value is None or value == 0:
                raise RouterContractError("macro sign is missing or zero")
            symbols.append("+" if value > 0 else "-")
        raw = "".join(symbols)
        background = "ALIGNED" if raw == "+++" else "OPPOSED" if raw == "---" else "MIXED"
        return raw, background

    def _stage03(
        self,
        observation: Mapping[str, Any],
        *,
        seek_side: str,
        current_candidate_extreme: float | None,
    ) -> dict[str, Any]:
        new_extreme = bool(int(float(observation["new_extreme_now"])))
        compared_value = observation.get("high" if seek_side == "H" else "low")
        return {
            "stage": "03",
            "observation_start": "ram.current_candidate_extreme",
            "fixed_observation": {
                "seek_side": seek_side,
                "current_candidate_extreme": current_candidate_extreme,
                "current_side_price": compared_value,
            },
            "relative_calculation": {
                "operator": "SIDE_PRICE_BREACH",
                "result": new_extreme,
            },
            "candidate_transition": "REARM" if new_extreme else "HOLD",
            "new_extreme_now": new_extreme,
            "trade_action": "WAIT",
            "handoff": "04",
        }

    def _stage04(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        raw, background = self._background(observation)
        return {
            "stage": "04",
            "observation_start": "stage03.current_candidate",
            "relative_calculation": {
                "operator": "SIDE_RELATIVE_SIGN_SIGNATURE",
                "ordered_signs": raw,
            },
            "context_signature_raw": raw,
            "context_background_name": background,
            "trade_action": "WAIT",
            "handoff": "05",
        }

    @staticmethod
    def _stage05(observation: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "stage": "05",
            "observation_start": "current_wave_start",
            "wave_age_bars": _decode_age(observation["wave_age_log"]),
            "zone_age_bars": _decode_age(observation.get("zone_age_log")),
            "time_ratio_prev1": observation["time_ratio_prev1"],
            "time_ratio_prev2": observation["time_ratio_prev2"],
            "hardcoded_minimum_wait": None,
            "relative_calculation": "CURRENT_WAVE_VS_PREVIOUS_WAVES",
            "trade_action": "WAIT",
            "handoff": "06",
        }

    @staticmethod
    def _stage06(observation: Mapping[str, Any]) -> dict[str, Any]:
        fields = (
            "candle_direction_for_side",
            "candle_body_ratio",
            "upper_wick_ratio",
            "lower_wick_ratio",
            "close_position_for_side",
            "volume_ratio20",
            "volume_ratio72",
            "range_ratio20",
        )
        return {
            "stage": "06",
            "observation_start": "latest_closed_5m_candle",
            "profile": {field: observation[field] for field in fields},
            "relative_calculation": "CANDLE_RANGE_AND_ROLLING_VOLUME_RELATIONS",
            "trade_action": "WAIT",
            "handoff": "07",
        }

    @staticmethod
    def _stage07(observation: Mapping[str, Any]) -> dict[str, Any]:
        witnesses: list[dict[str, Any]] = []
        for timeframe in TIMEFRAMES:
            prefix = f"macd_{timeframe}_"
            witnesses.append(
                {
                    "timeframe": timeframe,
                    "hist_for_side": observation[prefix + "hist_for_side"],
                    "delta_for_side": observation[prefix + "delta_for_side"],
                    "sign_for_side": observation[prefix + "sign_for_side"],
                    "zc_count_so_far": observation.get(prefix + "zc_count_so_far"),
                    "zc_age_log": observation[prefix + "zc_age_log"],
                }
            )
        return {
            "stage": "07",
            "observation_start": "latest_closed_timeframe_witness",
            "witnesses": witnesses,
            "timeframe_vote": None,
            "single_score": None,
            "trade_action": "WAIT",
            "handoff": "08",
        }

    @staticmethod
    def _stage08(observation: Mapping[str, Any], raw_position: int) -> dict[str, Any]:
        positions: dict[str, int | None] = {}
        for timeframe in TIMEFRAMES:
            age = _decode_age(observation[f"macd_{timeframe}_zc_age_log"])
            positions[timeframe] = None if age is None else raw_position - age
        groups: dict[int, list[str]] = {}
        for timeframe, position in positions.items():
            if position is not None:
                groups.setdefault(position, []).append(timeframe)
        order = [
            {"raw_position": position, "simultaneous_timeframes": groups[position]}
            for position in sorted(groups)
        ]
        return {
            "stage": "08",
            "observation_start": "each_timeframe_last_zc",
            "last_zc_raw_positions": positions,
            "distinct_last_zc_position_count": len(groups),
            "signal_order_sequence": order,
            "trade_action": "WAIT",
            "handoff": "09",
        }

    def _stage09(
        self, observation: Mapping[str, Any], background: str
    ) -> dict[str, Any]:
        matches = [
            calculation
            for calculation in self.calculations_by_background[background]
            if self._matches(calculation, observation)
        ]
        method_views: dict[str, dict[str, Mapping[str, Any] | None]] = {}
        for method in METHODS:
            base = [
                calculation
                for calculation in matches
                if calculation["method"] == method
                and calculation["output_source"] == "BASE"
            ]
            context = [
                calculation
                for calculation in matches
                if calculation["method"] == method
                and calculation["output_source"] == "CONTEXT"
            ]
            method_views[method] = {
                "base": min(base, key=_output_rank) if base else None,
                "context": min(context, key=_output_rank) if context else None,
            }
        base_matches = [
            calculation
            for calculation in matches
            if calculation["output_source"] == "BASE"
        ]
        context_matches = [
            calculation
            for calculation in matches
            if calculation["output_source"] == "CONTEXT"
        ]
        return {
            "stage": "09",
            "observation_start": "stage03_to_stage08_handoffs",
            "runtime_model_calculation_count": len(
                self.calculations_by_background[background]
            ),
            "satisfied_relation_count": len(matches),
            "official_single_winner_candidates": {
                "base": min(base_matches, key=_output_rank)
                if base_matches
                else None,
                "context": min(context_matches, key=_output_rank)
                if context_matches
                else None,
            },
            "methods": method_views,
            "historical_2775_evidence_consulted": False,
            "ready_method_count": 0,
            "aggregation_across_groups": None,
            "single_total_score": None,
            "trade_action": "WAIT",
            "handoff": "10",
        }

    def _advance_group(self, group: str, calculation_id: str | None) -> int:
        if calculation_id is None:
            self.group_signatures[group] = None
            self.group_runs[group] = 0
            return 0
        if self.group_signatures.get(group) == calculation_id:
            self.group_runs[group] = self.group_runs.get(group, 0) + 1
        else:
            self.group_signatures[group] = calculation_id
            self.group_runs[group] = 1
        return self.group_runs[group]

    def _stage10(
        self,
        stage09: Mapping[str, Any],
        seek_side: str,
        evidence_reset_reasons: Sequence[str],
    ) -> dict[str, Any]:
        gate = float(self.selector["action_gate"])
        tolerance = float(self.selector["probability_tolerance"])

        def state_for(
            output: Mapping[str, Any] | None, *, run_key: str
        ) -> dict[str, Any]:
            run = self._advance_group(
                run_key, output["calculation_id"] if output else None
            )
            if output is None:
                return {
                    "ready": False,
                    "run": 0,
                    "reason": "NO_RELATION_SATISFIED",
                }
            value = output["output_value"]
            minimum_support = int(
                self.selector[
                    "minimum_event_support"
                    if output["output_source"] == "BASE"
                    else "minimum_context_event_support"
                ]
            )
            required = _side_required(output, seek_side)
            probability = float(value["state_probability"])
            support = int(value["event_support"])
            if support < minimum_support:
                reason = "SUPPORT_BELOW_MINIMUM"
            elif probability + tolerance < gate:
                reason = "PROBABILITY_BELOW_GATE"
            elif run < required:
                reason = "PERSISTENCE_INCOMPLETE"
            else:
                reason = "READY"
            return {
                "calculation_id": output["calculation_id"],
                "probability": probability,
                "support": support,
                "run": run,
                "required": required,
                "ready": reason == "READY",
                "reason": reason,
            }

        official_candidates = stage09["official_single_winner_candidates"]
        official_states = {
            source: state_for(
                official_candidates[source],
                run_key=f"OFFICIAL::{source.upper()}",
            )
            for source in ("base", "context")
        }
        ready = [
            official_candidates[source]
            for source in ("base", "context")
            if official_candidates[source] is not None
            and official_states[source]["ready"]
        ]

        method_states: dict[str, dict[str, Any]] = {}
        for method in METHODS:
            view = stage09["methods"][method]
            base = state_for(
                view["base"], run_key=f"{method}::BASE"
            )
            context = state_for(
                view["context"], run_key=f"{method}::CONTEXT"
            )
            method_ready = bool(base["ready"] or context["ready"])
            method_states[method] = {
                "base": base,
                "context": context,
                "ready": method_ready,
                "wait_reason": "METHOD_READY"
                if method_ready
                else f"BASE:{base['reason']}|CONTEXT:{context['reason']}",
            }
        return {
            "stage": "10",
            "observation_start": "stage09_method_candidates",
            "official_single_winner_states": official_states,
            "methods": method_states,
            "ready_method_count": sum(
                int(state["ready"]) for state in method_states.values()
            ),
            "ready_output_count": len(ready),
            "ready_outputs": ready,
            "released": bool(ready),
            "active_blocker_count": 0 if ready else 1,
            "persistence_reset_reasons": list(evidence_reset_reasons),
            "trade_action": "WAIT",
            "handoff": "11",
        }

    def _observed_1h_zc_switch(
        self, observation: Mapping[str, Any], declared: bool | None
    ) -> bool:
        if declared is not None:
            return bool(declared)
        if not self.history:
            return False
        previous = self.history[-1]
        return bool(
            previous.get("macd_1h_zc_count_so_far")
            != observation.get("macd_1h_zc_count_so_far")
            or previous.get("macd_1h_sign_for_side")
            != observation.get("macd_1h_sign_for_side")
        )

    def _stage11(
        self,
        stage03: Mapping[str, Any],
        stage10: Mapping[str, Any],
        observed_1h_zc_switch: bool,
    ) -> dict[str, Any]:
        ready = list(stage10["ready_outputs"])
        winner = min(ready, key=_output_rank) if ready else None
        if not self.event_released and winner is not None:
            action = "NOW"
            source = "RULE_NOW"
            self.event_released = True
        elif not self.event_released and observed_1h_zc_switch:
            action = "NOW"
            source = "OBSERVED_1H_ZC_STATE_NOW"
            self.event_released = True
        else:
            action = "WAIT"
            source = "WAIT"
        return {
            "stage": "11",
            "observation_start": "stage10_ready_candidates",
            "candidate_transition": stage03["candidate_transition"],
            "trade_action": action,
            "action_source": source,
            "runtime_calculation_id": winner["calculation_id"] if winner else None,
            "historical_2775_evidence_consulted": False,
            "entry_fill": "NEXT_CLOSED_5M_OPEN" if action == "NOW" else None,
            "rearm_and_now": bool(
                stage03["candidate_transition"] == "REARM" and action == "NOW"
            ),
        }

    def route_closed_bar(
        self,
        observation: Mapping[str, Any],
        *,
        event_key: str,
        seek_side: str,
        raw_position: int,
        observed_1h_zc_switch: bool | None = None,
        current_candidate_extreme: float | None = None,
    ) -> dict[str, Any]:
        side = seek_side.upper()
        if side not in {"H", "L"}:
            raise RouterContractError("seek_side must be H or L")
        self._begin_event(event_key)
        row = self._materialize_observation(
            observation,
            seek_side=side,
            current_candidate_extreme=current_candidate_extreme,
        )
        self._validate_observation(row)
        stage03 = self._stage03(
            row,
            seek_side=side,
            current_candidate_extreme=current_candidate_extreme,
        )
        reset_reasons = self._reset_evidence_runs_if_required(
            row, new_extreme_now=bool(stage03["new_extreme_now"])
        )
        stage04 = self._stage04(row)
        stage05 = self._stage05(row)
        stage06 = self._stage06(row)
        stage07 = self._stage07(row)
        stage08 = self._stage08(row, raw_position)
        stage09 = self._stage09(row, stage04["context_background_name"])
        stage10 = self._stage10(stage09, side, reset_reasons)
        fallback = self._observed_1h_zc_switch(row, observed_1h_zc_switch)
        stage11 = self._stage11(stage03, stage10, fallback)
        self.history.append(row)
        stages = {
            "03": stage03,
            "04": stage04,
            "05": stage05,
            "06": stage06,
            "07": stage07,
            "08": stage08,
            "09": stage09,
            "10": stage10,
            "11": stage11,
        }
        return {
            "stage_order": list(STAGE_ORDER),
            "stages": stages,
            "candidate_transition": stage11["candidate_transition"],
            "trade_action": stage11["trade_action"],
            "action_source": stage11["action_source"],
        }


def _candidate_state_arrays(
    router: RuleRouter,
    calculation_indexes: np.ndarray,
    *,
    source: str,
    sides: np.ndarray,
    event_codes: np.ndarray,
    new_extreme: np.ndarray,
) -> dict[str, np.ndarray]:
    count = len(router.calculations)
    probability_by_calculation = np.full(count, np.nan, dtype=np.float64)
    support_by_calculation = np.zeros(count, dtype=np.int32)
    required_h = np.ones(count, dtype=np.int32)
    required_l = np.ones(count, dtype=np.int32)
    for index, calculation in enumerate(router.calculations):
        if calculation["output_source"] != source:
            continue
        value = calculation["output_value"]
        probability_by_calculation[index] = float(value["state_probability"])
        support_by_calculation[index] = int(value["event_support"])
        required_h[index] = _side_required(calculation, "H")
        required_l[index] = _side_required(calculation, "L")

    valid = calculation_indexes >= 0
    safe_indexes = np.maximum(calculation_indexes, 0)
    probability = probability_by_calculation[safe_indexes]
    support = support_by_calculation[safe_indexes]
    required = np.where(sides == "H", required_h[safe_indexes], required_l[safe_indexes])
    positions = np.arange(len(calculation_indexes), dtype=np.int32)
    boundary = np.r_[True, event_codes[1:] != event_codes[:-1]]
    boundary[1:] |= calculation_indexes[1:] != calculation_indexes[:-1]
    boundary |= new_extreme
    starts = np.maximum.accumulate(np.where(boundary, positions, -1))
    run = positions - starts + 1
    run[~valid] = 0
    minimum_support = int(
        router.selector[
            "minimum_event_support"
            if source == "BASE"
            else "minimum_context_event_support"
        ]
    )
    gate = float(router.selector["action_gate"])
    tolerance = float(router.selector["probability_tolerance"])
    reason = np.full(len(calculation_indexes), "NO_RELATION_SATISFIED", dtype=object)
    reason[valid & (support < minimum_support)] = "SUPPORT_BELOW_MINIMUM"
    enough_support = valid & (support >= minimum_support)
    reason[enough_support & (probability + tolerance < gate)] = "PROBABILITY_BELOW_GATE"
    enough_probability = enough_support & (probability + tolerance >= gate)
    reason[enough_probability & (run < required)] = "PERSISTENCE_INCOMPLETE"
    ready = enough_probability & (run >= required)
    reason[ready] = "READY"
    probability[~valid] = np.nan
    support[~valid] = 0
    required[~valid] = 0
    return {
        "probability": probability,
        "support": support,
        "required": required.astype(np.int32),
        "run": run.astype(np.int32),
        "ready": ready,
        "reason": reason,
    }


def build_reconstructed_intermediate_ledger(
    router: RuleRouter,
    observations: pd.DataFrame,
    *,
    winner_arrays: Mapping[str, np.ndarray] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Rebuild Stage 03-11 handoffs from formulas, not the 2,775 list.

    The result is explicitly reconstructed.  It does not claim the hashes of
    the unavailable original intermediate parquet files.
    """

    missing = [feature for feature in router.runtime_inputs if feature not in observations]
    if missing:
        raise RouterContractError(
            f"intermediate ledger is missing {len(missing)} runtime inputs"
        )
    frame = observations.reset_index(drop=True)
    event_codes, event_names = pd.factorize(frame["event_key"], sort=False)
    event_codes = np.asarray(event_codes, dtype=np.int32)
    backgrounds = router._batch_background(frame)

    if winner_arrays is None:
        computed: dict[str, np.ndarray] = {}
        for source in ("BASE", "CONTEXT"):
            computed[f"OFFICIAL__{source}"] = router._batch_winners(
                frame, event_codes, backgrounds, source
            )
        original_ranked = router.ranked_indexes
        try:
            for method in METHODS:
                router.ranked_indexes = {
                    key: tuple(
                        index
                        for index in indexes
                        if router.calculations[index]["method"] == method
                    )
                    for key, indexes in original_ranked.items()
                }
                for source in ("BASE", "CONTEXT"):
                    computed[f"{method}__{source}"] = router._batch_winners(
                        frame, event_codes, backgrounds, source
                    )
        finally:
            router.ranked_indexes = original_ranked
        winner_arrays = computed
    required_winners = {
        *(f"OFFICIAL__{source}" for source in ("BASE", "CONTEXT")),
        *(
            f"{method}__{source}"
            for method in METHODS
            for source in ("BASE", "CONTEXT")
        ),
    }
    missing_winners = sorted(required_winners.difference(winner_arrays))
    if missing_winners:
        raise RouterContractError(f"winner arrays are missing {missing_winners}")
    if any(len(winner_arrays[key]) != len(frame) for key in required_winners):
        raise RouterContractError("winner arrays do not match observation rows")

    ledger = frame.copy()
    new_extreme = frame["new_extreme_now"].to_numpy(dtype=float) != 0.0
    sides = frame["seek_side"].astype(str).to_numpy()
    ledger["stage03_candidate_transition"] = np.where(new_extreme, "REARM", "HOLD")
    signs = [
        np.where(frame[f"macd_{timeframe}_sign_for_side"].to_numpy(float) > 0, "+", "-")
        for timeframe in ("4h", "1d", "1w")
    ]
    ledger["stage04_context_signature_for_side"] = np.char.add(
        np.char.add(signs[0], signs[1]), signs[2]
    )
    ledger["stage04_context_background_name"] = backgrounds

    last_zc_columns = []
    for timeframe in TIMEFRAMES:
        age = np.rint(
            np.expm1(frame[f"macd_{timeframe}_zc_age_log"].to_numpy(dtype=float))
        ).astype(np.int64)
        column = f"stage08_macd_{timeframe}_last_zc_raw_position"
        ledger[column] = frame["raw_position"].to_numpy(dtype=np.int64) - age
        last_zc_columns.append(column)
    position_matrix = ledger[last_zc_columns].to_numpy(dtype=np.int64)
    ordered_positions = np.sort(position_matrix, axis=1)
    ledger["stage08_distinct_last_zc_position_count"] = (
        1 + (ordered_positions[:, 1:] != ordered_positions[:, :-1]).sum(axis=1)
    ).astype(np.int8)

    states: dict[str, dict[str, np.ndarray]] = {}
    for owner in ("OFFICIAL", *METHODS):
        for source in ("BASE", "CONTEXT"):
            key = f"{owner}__{source}"
            indexes = np.asarray(winner_arrays[key], dtype=np.int32)
            ledger[f"stage09_{key.lower()}_calculation_index"] = indexes
            state = _candidate_state_arrays(
                router,
                indexes,
                source=source,
                sides=sides,
                event_codes=event_codes,
                new_extreme=new_extreme,
            )
            states[key] = state
            ledger[f"stage10_{key.lower()}_run"] = state["run"]
            ledger[f"stage10_{key.lower()}_required"] = state["required"]
            ledger[f"stage10_{key.lower()}_reason"] = state["reason"]
            ledger[f"stage10_{key.lower()}_ready"] = state["ready"]

    method_ready_arrays: dict[str, np.ndarray] = {}
    for method in METHODS:
        base = states[f"{method}__BASE"]
        context = states[f"{method}__CONTEXT"]
        method_ready = base["ready"] | context["ready"]
        method_ready_arrays[method] = method_ready
        ledger[f"stage10_{method.lower()}_ready"] = method_ready
        ledger[f"stage10_{method.lower()}_wait_reason"] = np.where(
            method_ready,
            "METHOD_READY",
            [
                f"BASE:{base_reason}|CONTEXT:{context_reason}"
                for base_reason, context_reason in zip(base["reason"], context["reason"])
            ],
        )

    official_release = (
        states["OFFICIAL__BASE"]["ready"]
        | states["OFFICIAL__CONTEXT"]["ready"]
    )
    parallel_any_ready = np.logical_or.reduce(
        [method_ready_arrays[method] for method in METHODS]
    )
    ledger["stage10_released"] = official_release
    ledger["stage10_active_blocker_count"] = (~official_release).astype(np.int8)
    ledger["stage11_candidate_transition"] = ledger["stage03_candidate_transition"]
    trade_action = np.full(len(ledger), "WAIT", dtype=object)
    action_source = np.full(len(ledger), "WAIT", dtype=object)
    action_calculation = np.full(len(ledger), -1, dtype=np.int32)
    released_so_far = np.zeros(len(ledger), dtype=bool)
    event_actions: list[dict[str, Any]] = []
    for event_code, event_name in enumerate(event_names):
        indexes = np.flatnonzero(event_codes == event_code)
        ready_indexes = indexes[official_release[indexes]]
        pivot = int(frame.at[indexes[0], "zone_pivot_position"])
        if len(ready_indexes):
            action_index = int(ready_indexes[0])
            candidates = [
                int(winner_arrays[f"OFFICIAL__{source}"][action_index])
                for source in ("BASE", "CONTEXT")
                if states[f"OFFICIAL__{source}"]["ready"][action_index]
            ]
            winner = min(
                candidates, key=lambda value: _output_rank(router.calculations[value])
            )
            action_position = int(frame.at[action_index, "raw_position"])
            source = "RULE_NOW"
            method = str(router.calculations[winner]["method"])
            trade_action[action_index] = "NOW"
            action_source[action_index] = source
            action_calculation[action_index] = winner
            released_so_far[indexes[indexes >= action_index]] = True
        else:
            winner = -1
            action_position = int(frame.at[indexes[-1], "zone_end_position"]) + 1
            source = "OBSERVED_1H_ZC_STATE_NOW"
            method = "ZC_FALLBACK"
        event_actions.append(
            {
                "event_key": str(event_name),
                "seek_side": str(frame.at[indexes[0], "seek_side"]),
                "pivot_position": pivot,
                "action_position": action_position,
                "distance_bars": action_position - pivot,
                "action_source": source,
                "method": method,
                "runtime_calculation_index": winner,
                "historical_2775_evidence_consulted": False,
            }
        )
    ledger["stage11_trade_action"] = trade_action
    ledger["stage11_action_source"] = action_source
    ledger["stage11_runtime_calculation_index"] = action_calculation
    ledger["stage11_event_released"] = released_so_far
    events = pd.DataFrame.from_records(event_actions)
    action_counts = events["action_source"].value_counts()
    summary = {
        "status": "RECONSTRUCTED_FROM_AVAILABLE_FORMULAS",
        "rows": int(len(ledger)),
        "events": int(len(events)),
        "stage_order": list(STAGE_ORDER),
        "rule_now": int(action_counts.get("RULE_NOW", 0)),
        "observed_1h_zc_state_now": int(
            action_counts.get("OBSERVED_1H_ZC_STATE_NOW", 0)
        ),
        "too_fast_relative_to_study_pivot": int((events["distance_bars"] < 0).sum()),
        "historical_2775_evidence_consulted": False,
        "original_missing_intermediate_hash_claimed": False,
    }
    source_audit = router.program.get("historical_learning_evidence", {}).get(
        "source_thought_audit", {}
    )
    if source_audit:
        source_stage09 = source_audit.get("09", {})
        source_stage10 = source_audit.get("10", {})
        source_stage11 = source_audit.get("11", {})
        source_method_reasons = source_stage10.get("reason_counts_by_method", {})
        reconstructed_method_ready = {
            method: int(method_ready_arrays[method].sum()) for method in METHODS
        }
        source_method_ready = {
            method: int(
                source_method_reasons.get(method, {}).get("METHOD_READY", 0)
            )
            for method in METHODS
        }
        summary["source_aggregate_parity"] = {
            "status": "MATCH"
            if (
                int(official_release.sum())
                == int(source_stage09.get("official_rule_ready_row_count", -1))
                and int(parallel_any_ready.sum())
                == int(source_stage09.get("parallel_any_ready_row_count", -1))
                and summary["rule_now"]
                == int(source_stage11.get("rule_now_count", -1))
                and summary["observed_1h_zc_state_now"]
                == int(source_stage11.get("fallback_now_count", -1))
            )
            else "MISMATCH",
            "source": {
                "official_ready_rows": int(
                    source_stage09.get("official_rule_ready_row_count", 0)
                ),
                "parallel_any_ready_rows": int(
                    source_stage09.get("parallel_any_ready_row_count", 0)
                ),
                "method_ready_rows": source_method_ready,
                "rule_now": int(source_stage11.get("rule_now_count", 0)),
                "fallback_now": int(source_stage11.get("fallback_now_count", 0)),
            },
            "reconstructed": {
                "official_ready_rows": int(official_release.sum()),
                "parallel_any_ready_rows": int(parallel_any_ready.sum()),
                "method_ready_rows": reconstructed_method_ready,
                "rule_now": summary["rule_now"],
                "fallback_now": summary["observed_1h_zc_state_now"],
            },
            "difference_reconstructed_minus_source": {
                "official_ready_rows": int(official_release.sum())
                - int(source_stage09.get("official_rule_ready_row_count", 0)),
                "parallel_any_ready_rows": int(parallel_any_ready.sum())
                - int(source_stage09.get("parallel_any_ready_row_count", 0)),
                "method_ready_rows": {
                    method: reconstructed_method_ready[method]
                    - source_method_ready[method]
                    for method in METHODS
                },
                "rule_now": summary["rule_now"]
                - int(source_stage11.get("rule_now_count", 0)),
                "fallback_now": summary["observed_1h_zc_state_now"]
                - int(source_stage11.get("fallback_now_count", 0)),
            },
        }
    return ledger, events, summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compact_reconstructed_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    """Reduce storage without changing any decision or relation identity."""

    compact = ledger.copy()
    for column in compact.columns:
        series = compact[column]
        if series.dtype == np.dtype("float64"):
            compact[column] = series.astype(np.float32)
        elif column != "event_key" and (
            isinstance(series.dtype, pd.StringDtype)
            or pd.api.types.is_object_dtype(series.dtype)
        ):
            compact[column] = series.astype("category")
    return compact


def export_reconstructed_intermediate_ledger(
    program_path: str | Path,
    ohlcv_path: str | Path,
    output_dir: str | Path,
    *,
    last_raw_position: int | None = None,
    winner_arrays: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Create the reproducible Stage 03-11 ledger from OHLCV and formulas.

    Historical 2,775 evidence is deliberately not accepted as an argument.
    """

    program_file = Path(program_path).expanduser().resolve()
    ohlcv_file = Path(ohlcv_path).expanduser().resolve()
    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(ohlcv_file)
    observations = build_canonical_observations(
        raw, last_raw_position=last_raw_position
    )
    router = RuleRouter.load(program_file)
    ledger, events, summary = build_reconstructed_intermediate_ledger(
        router, observations, winner_arrays=winner_arrays
    )
    stored_ledger = _compact_reconstructed_ledger(ledger)
    ledger_path = target / "professor_16_modules_intermediate_ledger_reconstructed_v1.parquet"
    events_path = target / "stage11_event_actions_reconstructed_v1.parquet"
    stored_ledger.to_parquet(
        ledger_path,
        index=False,
        compression="brotli",
        compression_level=5,
    )
    events.to_parquet(events_path, index=False, compression="brotli")
    manifest = {
        "schema": "twin.reconstructed-intermediate-ledger.v1",
        "status": "RECONSTRUCTED_FROM_AVAILABLE_FORMULAS_NOT_ORIGINAL_HASH",
        "source_ohlcv": {
            "file": ohlcv_file.name,
            "rows": int(len(raw)),
            "historical_prefix_last_raw_position": (
                int(last_raw_position) if last_raw_position is not None else None
            ),
        },
        "source_program": {
            "file": program_file.name,
            "sha256": _sha256(program_file),
        },
        "runtime_uses_historical_2775_evidence": False,
        "stage_order": list(STAGE_ORDER),
        "numeric_storage": "float32_after_float64_formula_calculation",
        "summary": summary,
        "files": [
            {
                "file": ledger_path.name,
                "rows": int(len(stored_ledger)),
                "columns": int(len(stored_ledger.columns)),
                "bytes": ledger_path.stat().st_size,
                "sha256": _sha256(ledger_path),
            },
            {
                "file": events_path.name,
                "rows": int(len(events)),
                "columns": int(len(events.columns)),
                "bytes": events_path.stat().st_size,
                "sha256": _sha256(events_path),
            },
        ],
        "original_missing_parquet_hash_claimed": False,
    }
    manifest_path = target / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def route_jsonl(program_path: str | Path, input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    router = RuleRouter.load(program_path)
    source = Path(input_path).expanduser().resolve()
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    actions = {"NOW": 0, "WAIT": 0}
    with source.open("r", encoding="utf-8") as reader, target.open("w", encoding="utf-8") as writer:
        for line_number, raw in enumerate(reader, 1):
            if not raw.strip():
                continue
            item = json.loads(raw)
            try:
                result = router.route_closed_bar(
                    item["observation"],
                    event_key=str(item["event_key"]),
                    seek_side=str(item["seek_side"]),
                    raw_position=int(item["raw_position"]),
                    observed_1h_zc_switch=item.get("observed_1h_zc_switch"),
                    current_candidate_extreme=item.get("current_candidate_extreme"),
                )
            except Exception as error:
                raise RouterContractError(f"input line {line_number}: {error}") from error
            writer.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
            rows += 1
            actions[result["trade_action"]] += 1
    return {"rows": rows, "actions": actions, "output": str(target)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("program")
    parser.add_argument("input_jsonl")
    parser.add_argument("output_jsonl")
    args = parser.parse_args()
    print(
        json.dumps(
            route_jsonl(args.program, args.input_jsonl, args.output_jsonl),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
