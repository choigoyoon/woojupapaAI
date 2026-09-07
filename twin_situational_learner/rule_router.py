"""Execute the full learned Stage 03-11 relation model on closed 5-minute bars.

The router evaluates structured operands, intervals, relative changes and
persistence.  The 2,775 historical outputs are audit evidence only and are
never loaded as a decision whitelist.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


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
