"""Execute one exported Stage 03-11 JSON on sequential closed 5-minute bars.

The router evaluates structured operands, intervals, relative changes and
persistence.  It never searches the 2,775 original signature strings.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "twin.executable-thought-program.v1"
STAGE_ORDER = ("03", "04", "05", "06", "07", "08", "09", "10", "11")
EXPECTED_OPERATORS = (
    "NEW_EXTREME_CANDIDATE_AXIS",
    "EXACT_MACRO_SIGN_SIGNATURE",
    "DECODE_RELATIVE_TIME",
    "PRESERVE_CANDLE_VOLUME_PROFILE",
    "PRESERVE_EIGHT_TIMEFRAME_MACD",
    "DECODE_LAST_ZC_ORDER",
    "EVALUATE_COMPILED_OUTPUTS",
    "APPLY_LEARNED_PERSISTENCE",
    "FIRST_READY_OR_OBSERVED_1H_ZC",
)
TIMEFRAMES = ("5m", "15m", "30m", "1h", "2h", "4h", "1d", "1w")


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
        int(order.get("channel", 1_000_000)),
        int(order.get("rule", 1_000_000)),
        int(order.get("bin", 1_000_000)),
        str(output["output_id"]),
    )


class RuleRouter:
    def __init__(self, program: Mapping[str, Any]):
        self.program = dict(program)
        self._validate_program()
        contract = self.program["runtime_input_contract"]
        self.runtime_inputs = tuple(contract["features"])
        self.forbidden = frozenset(contract["forbidden"])
        self.outputs = tuple(self.program["compiled_outputs"])
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
        contract = self.program["runtime_input_contract"]
        if int(contract["count"]) != 64 or len(contract["features"]) != 64:
            raise RouterContractError("runtime input contract is not the existing 64")
        output = self.program["output_contract"]
        if (
            int(output["count"]) != 2_775
            or int(output["context_calculation_outputs"]) != 2_499
            or int(output["base_interval_outputs"]) != 276
            or int(output["unmatched"]) != 0
        ):
            raise RouterContractError("compiled output contract is not 2,775 = 2,499 + 276")
        compiled = self.program["compiled_outputs"]
        if len(compiled) != 2_775 or len({item["output_id"] for item in compiled}) != 2_775:
            raise RouterContractError("compiled outputs are missing or duplicated")

    def _begin_event(self, event_key: str) -> None:
        if self.current_event_key != event_key:
            self.current_event_key = event_key
            self.history.clear()
            self.group_signatures.clear()
            self.group_runs.clear()
            self.event_released = False

    def _validate_observation(self, observation: Mapping[str, Any]) -> None:
        forbidden = sorted(set(observation).intersection(self.forbidden))
        if forbidden:
            raise RouterContractError(f"future/study fields reached the router: {forbidden}")
        missing = [feature for feature in self.runtime_inputs if feature not in observation]
        if missing:
            raise RouterContractError(f"closed-bar observation is missing {len(missing)} inputs")

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

    def _stage03(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        new_extreme = bool(int(float(observation["new_extreme_now"])))
        return {
            "stage": "03",
            "candidate_transition": "REARM" if new_extreme else "HOLD",
            "new_extreme_now": new_extreme,
            "trade_action": "WAIT",
            "handoff": "04",
        }

    def _stage04(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        raw, background = self._background(observation)
        return {
            "stage": "04",
            "context_signature_raw": raw,
            "context_background_name": background,
            "trade_action": "WAIT",
            "handoff": "05",
        }

    @staticmethod
    def _stage05(observation: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "stage": "05",
            "wave_age_bars": _decode_age(observation["wave_age_log"]),
            "time_ratio_prev1": observation["time_ratio_prev1"],
            "time_ratio_prev2": observation["time_ratio_prev2"],
            "hardcoded_minimum_wait": None,
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
            "profile": {field: observation[field] for field in fields},
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
            output
            for output in self.outputs
            if output["background"] == background and self._matches(output, observation)
        ]
        groups: dict[str, list[Mapping[str, Any]]] = {}
        for output in matches:
            groups.setdefault(str(output["calculation_group"]), []).append(output)
        strongest = {
            group: min(outputs, key=_output_rank) for group, outputs in groups.items()
        }
        return {
            "stage": "09",
            "matched_output_count": len(matches),
            "strongest_by_calculation_group": strongest,
            "ready_method_count": 0,
            "aggregation_across_groups": None,
            "single_total_score": None,
            "trade_action": "WAIT",
            "handoff": "10",
        }

    def _advance_group(self, group: str, output_id: str | None) -> int:
        if output_id is None:
            self.group_signatures[group] = None
            self.group_runs[group] = 0
            return 0
        if self.group_signatures.get(group) == output_id:
            self.group_runs[group] = self.group_runs.get(group, 0) + 1
        else:
            self.group_signatures[group] = output_id
            self.group_runs[group] = 1
        return self.group_runs[group]

    def _stage10(self, stage09: Mapping[str, Any], seek_side: str) -> dict[str, Any]:
        gate = float(self.selector["action_gate"])
        tolerance = float(self.selector["probability_tolerance"])
        active = stage09["strongest_by_calculation_group"]
        known_groups = {
            "BASE_SINGLE_STRONGEST_CHANNEL",
            "WAVE_REARM_AGE",
            "MOMENTUM_SPEED",
            "CANDLE_REVERSAL",
            "VOLUME_RANGE",
            "MACRO_TREND",
            "MID_MACD_TREND",
        }
        states: dict[str, dict[str, Any]] = {}
        ready: list[Mapping[str, Any]] = []
        for group in sorted(known_groups):
            output = active.get(group)
            run = self._advance_group(group, output["output_id"] if output else None)
            if output is None:
                states[group] = {"ready": False, "run": 0, "reason": "NO_OUTPUT_MATCH"}
                continue
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
                ready.append(output)
            states[group] = {
                "output_id": output["output_id"],
                "probability": probability,
                "support": support,
                "run": run,
                "required": required,
                "ready": reason == "READY",
                "reason": reason,
            }
        return {
            "stage": "10",
            "calculation_group_states": states,
            "ready_output_count": len(ready),
            "ready_outputs": ready,
            "released": bool(ready),
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
            "candidate_transition": stage03["candidate_transition"],
            "trade_action": action,
            "action_source": source,
            "output_id": winner["output_id"] if winner else None,
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
    ) -> dict[str, Any]:
        side = seek_side.upper()
        if side not in {"H", "L"}:
            raise RouterContractError("seek_side must be H or L")
        self._begin_event(event_key)
        self._validate_observation(observation)
        stage03 = self._stage03(observation)
        stage04 = self._stage04(observation)
        stage05 = self._stage05(observation)
        stage06 = self._stage06(observation)
        stage07 = self._stage07(observation)
        stage08 = self._stage08(observation, raw_position)
        stage09 = self._stage09(observation, stage04["context_background_name"])
        stage10 = self._stage10(stage09, side)
        fallback = self._observed_1h_zc_switch(observation, observed_1h_zc_switch)
        stage11 = self._stage11(stage03, stage10, fallback)
        self.history.append(dict(observation))
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
