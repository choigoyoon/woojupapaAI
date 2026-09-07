"""Export the existing Stage 03-11 relation model and its evidence as one JSON.

This is a serializer/compiler, not a learner.  It does not change a module,
fit a threshold, average evidence, or add an entry method.  The 2,775
historical release signatures are exported as audit evidence only; they never
select which calculations are executable at runtime.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping


SCHEMA = "twin.executable-thought-program.v3"
STAGE_ORDER = ("03", "04", "05", "06", "07", "08", "09", "10", "11")
STAGE_MODULES = {
    "03": "candidate_revision.py",
    "04": "context_partition.py",
    "05": "time_distribution.py",
    "06": "candle_volume_distribution.py",
    "07": "multitf_zc_distribution.py",
    "08": "flow_order_distribution.py",
    "09": "rule_induction.py",
    "10": "repair_queue.py",
    "11": "threshold_frontier.py",
}
EXECUTABLE_OPERATORS = {
    "03": "NEW_EXTREME_CANDIDATE_AXIS",
    "04": "EXACT_MACRO_SIGN_SIGNATURE",
    "05": "DECODE_RELATIVE_TIME",
    "06": "PRESERVE_CANDLE_VOLUME_PROFILE",
    "07": "PRESERVE_EIGHT_TIMEFRAME_MACD",
    "08": "DECODE_LAST_ZC_ORDER",
    "09": "EVALUATE_FULL_LEARNED_RELATION_MODEL",
    "10": "APPLY_LEARNED_PERSISTENCE",
    "11": "FIRST_READY_OR_OBSERVED_1H_ZC",
}
FORBIDDEN_RUNTIME_FIELDS = (
    "outcome_study_only",
    "survived_study_only",
    "future_answer_columns",
    "current_event_future",
    "official_action_position",
    "event_boundary_lookahead",
    "event_position_study_only",
    "official_rule_ready",
    "official_wait_state",
)
METHODS = (
    "WAVE_REARM_AGE",
    "MOMENTUM_SPEED",
    "CANDLE_REVERSAL",
    "VOLUME_RANGE",
    "MACRO_TREND",
    "MID_MACD_TREND",
)
BLUEPRINT_FILENAME = "stages_03_to_11_authentic_blueprint.json"
STAGE_HANDOFF_FIELDS = {
    "03": ("seek_side", "candidate_transition", "new_extreme_now", "candidate_age_log", "rearm_count_log", "rearm_gap_bars_log", "rearm_extension_pct", "rearm_extension_ratio"),
    "04": ("context_signature_raw", "context_signature_for_side", "context_background_name"),
    "05": ("wave_age_bars", "zone_age_bars", "time_ratio_prev1", "time_ratio_prev2", "move_ratio_prev1", "move_ratio_prev2", "wave_move_pct_so_far", "zone_move_pct_so_far", "speed_recent3", "speed_ratio_3_to_12"),
    "06": ("candle_direction_for_side", "candle_body_ratio", "upper_wick_ratio", "lower_wick_ratio", "close_position_for_side", "volume_ratio20", "volume_ratio72", "range_ratio20"),
    "07": ("eight_timeframe_macd_witnesses",),
    "08": ("last_zc_raw_positions", "distinct_last_zc_position_count", "signal_order_sequence"),
    "09": ("official_base_candidate", "official_context_candidate", "six_method_signal_claims", "six_method_contextualized_claims"),
    "10": ("official_base_run", "official_context_run", "per_method_verdict", "per_method_arguments", "active_blocker_count", "keep_candidates"),
    "11": ("candidate_transition", "trade_action", "action_source", "entry_fill"),
}
STAGE_SCOPE_CONTRACTS = {
    "03": {
        "view_when": "ACTIVE_SEEK_SIDE_CURRENT_CANDIDATE_ON_THE_LATEST_CLOSED_5M_BAR",
        "exclude_when": [
            "OTHER_SEEK_SIDE_CANDIDATE_PREFIX",
            "CURRENT_EVENT_SELF_MATCH_DURING_STUDY_AUDIT",
        ],
    },
    "04": {
        "view_when": "STAGE03_HANDOFF_EQUALS_04; BOTH_REARM_AND_WAIT_CANDIDATES_CONTINUE",
        "exclude_when": [],
    },
    "05": {
        "view_when": "EVERY_CURRENT_CANDIDATE_HANDED_OFF_BY_STAGE04",
        "exclude_when": [
            "RELATION_TO_A_PRIOR_WAVE_THAT_DOES_NOT_EXIST; KEEP_MISSING_INSTEAD"
        ],
    },
    "06": {
        "view_when": "EVERY_CURRENT_CANDIDATE_ROW_HANDED_OFF_BY_STAGE05",
        "exclude_when": [
            "VOLUME_RELATION_WHEN_VOLUME_IS_UNAVAILABLE; KEEP_NULL_INSTEAD"
        ],
    },
    "07": {
        "view_when": "EVERY_STAGE06_HANDOFF_WITH_ALL_EIGHT_CAUSAL_TIMEFRAME_WITNESSES",
        "exclude_when": [
            "FUTURE_OR_STUDY_ONLY_TIMEFRAME_VALUE",
            "TIMEFRAME_VOTE_OR_AGGREGATED_SINGLE_SCORE",
        ],
    },
    "08": {
        "view_when": "EVERY_STAGE07_HANDOFF_WITH_EIGHT_LAST_ZC_POSITIONS",
        "exclude_when": [
            "ANY_ORDER_CREATED_BY_BREAKING_SIMULTANEOUS_ZC_TIES",
            "AGGREGATED_SEQUENCE_SCORE",
        ],
    },
    "09": {
        "view_when": "EVERY_STAGE08_HANDOFF_IN_ITS_EXACT_CURRENT_BACKGROUND",
        "exclude_when": [
            "DIRECT_MATCH_AGAINST_THE_2775_HISTORICAL_OUTPUT_ADDRESSES",
            "CROSS_METHOD_AGGREGATION_OR_SINGLE_TOTAL_SCORE",
        ],
    },
    "10": {
        "view_when": "EVERY_STAGE09_BASE_AND_CONTEXT_CANDIDATE_WITH_ITS_CURRENT_RUN_STATE",
        "exclude_when": [
            "ANY_RELEASE_THAT_BYPASSES_THE_ORIGINAL_ELIGIBILITY_BLOCKERS"
        ],
    },
    "11": {
        "view_when": "CURRENT_STAGE10_RELEASE_STATE_AND_CURRENT_STAGE03_CANDIDATE_TRANSITION",
        "exclude_when": [
            "HISTORICAL_2775_OUTPUT_ADDRESS_AS_A_RUNTIME_SELECTOR",
            "OFFICIAL_ACTION_POSITION_OR_OTHER_FUTURE_EVENT_FIELD",
        ],
    },
}
STAGE_EXECUTION_CONTRACTS = {
    "03": {
        "observation_start": "ram.current_candidate_extreme",
        "fixed_observations": ["seek_side", "latest_closed_5m.high", "latest_closed_5m.low"],
        "relative_calculations": [
            {
                "output": "new_extreme_now",
                "operator": "SIDE_PRICE_BREACH",
                "formula": "high[t] > ram.current_candidate_extreme if seek_side == 'H' else low[t] < ram.current_candidate_extreme",
            }
        ],
        "filters": ["SAME_SEEK_SIDE", "CURRENT_CANDIDATE_PREFIX_ONLY"],
    },
    "04": {
        "observation_start": "stage03.current_candidate",
        "fixed_observations": ["side_code", "macd_4h_sign_causal", "macd_1d_sign_causal", "macd_1w_sign_causal"],
        "relative_calculations": [
            {
                "output": "context_signature_for_side",
                "operator": "SIDE_RELATIVE_SIGN_SIGNATURE",
                "formula": "macd_{tf}_sign_for_side = macd_{tf}_sign_causal * side_code",
            }
        ],
        "filters": ["PRESERVE_EXACT_THREE_SIGN_ORDER", "NO_CLUSTERING"],
    },
    "05": {
        "observation_start": "current_wave_start",
        "fixed_observations": ["wave_age_log", "zone_age_log", "previous_wave_1", "previous_wave_2"],
        "relative_calculations": [
            {"output": "wave_age_bars", "operator": "ROUND_EXPM1", "formula": "round(expm1(wave_age_log))"},
            {"output": "zone_age_bars", "operator": "ROUND_EXPM1", "formula": "round(expm1(zone_age_log))"},
            {"output": "time_ratio_prev1", "operator": "SAFE_RATIO", "formula": "current_wave_elapsed_bars / previous_wave_1_elapsed_bars"},
            {"output": "time_ratio_prev2", "operator": "SAFE_RATIO", "formula": "current_wave_elapsed_bars / previous_wave_2_elapsed_bars"},
        ],
        "filters": ["MISSING_PRIOR_WAVE_STAYS_MISSING", "NO_HARDCODED_MINIMUM_WAIT"],
    },
    "06": {
        "observation_start": "latest_closed_5m_candle",
        "fixed_observations": ["open", "high", "low", "close", "volume", "volume_sma20", "volume_sma72", "range_sma20", "side_code"],
        "relative_calculations": [
            {"output": "candle_body_ratio", "operator": "SAFE_RATIO", "formula": "abs(close-open)/(high-low)"},
            {"output": "upper_wick_ratio", "operator": "SAFE_RATIO", "formula": "(high-max(open,close))/(high-low)"},
            {"output": "lower_wick_ratio", "operator": "SAFE_RATIO", "formula": "(min(open,close)-low)/(high-low)"},
            {"output": "volume_ratio20", "operator": "SAFE_RATIO", "formula": "volume/volume_sma20"},
            {"output": "volume_ratio72", "operator": "SAFE_RATIO", "formula": "volume/volume_sma72"},
            {"output": "range_ratio20", "operator": "SAFE_RATIO", "formula": "(high-low)/range_sma20"},
        ],
        "filters": ["ZERO_RANGE_SAFE", "MISSING_VOLUME_STAYS_MISSING"],
    },
    "07": {
        "observation_start": "latest_closed_timeframe_witness",
        "fixed_observations": ["side_code", "macd_5m_through_1w_hist", "macd_5m_through_1w_delta", "macd_5m_through_1w_zc_state"],
        "relative_calculations": [
            {"output": "macd_{tf}_hist_for_side", "operator": "SIDE_MULTIPLY", "formula": "macd_{tf}_hist * side_code"},
            {"output": "macd_{tf}_delta_for_side", "operator": "SIDE_MULTIPLY", "formula": "macd_{tf}_delta * side_code"},
        ],
        "filters": ["KEEP_ALL_EIGHT_TIMEFRAMES", "NO_TIMEFRAME_VOTE", "NO_SINGLE_SCORE"],
    },
    "08": {
        "observation_start": "each_timeframe_last_zc",
        "fixed_observations": ["raw_position", "macd_5m_through_1w_zc_age_log"],
        "relative_calculations": [
            {"output": "last_zc_raw_position", "operator": "POSITION_MINUS_ROUND_EXPM1", "formula": "raw_position - round(expm1(zc_age_log))"},
            {"output": "signal_order_sequence", "operator": "ORDER_WITH_SIMULTANEOUS_TIES", "formula": "sort distinct last_zc_raw_position; keep equal positions tied"},
        ],
        "filters": ["NEVER_BREAK_SIMULTANEOUS_TIE", "NO_SEQUENCE_SCORE"],
    },
    "09": {
        "observation_start": "stage03_to_stage08_handoffs",
        "fixed_observations": ["canonical_64_features", "three_background_rulebooks", "six_method_ownership", "official_single_winner_path"],
        "relative_calculations": [
            {"output": "base_signal_claim", "operator": "SINGLE_STRONGEST_SATISFIED_BASE", "formula": "rank every satisfied signal interval inside the current background; a raw match is a claim, never an action"},
            {"output": "contextualized_signal_claim", "operator": "SINGLE_STRONGEST_SATISFIED_CONTEXT", "formula": "rank every relation that contains the same method's signal condition plus its current/prior relative environment conditions"},
            {"output": "six_method_dialogue", "operator": "PRESERVE_PARALLEL_METHOD_EXPLANATIONS", "formula": "preserve each method's signal claim, contextualized claim, support, counterargument and unresolved persistence without a total score"},
        ],
        "filters": ["RAW_FORMULA_MATCH_IS_NOT_AN_ACTION", "CONTEXT_IS_SIGNAL_PLUS_ENVIRONMENT_NOT_AN_INDEPENDENT_SIGNAL", "OFFICIAL_SINGLE_WINNER_PATH_FOR_RELEASE", "NO_CROSS_METHOD_AGGREGATION", "NO_SINGLE_TOTAL_SCORE", "NO_TRADE_ACTION"],
    },
    "10": {
        "observation_start": "stage09_method_candidates",
        "fixed_observations": ["action_gate", "probability_tolerance", "minimum_support", "learned_required_persistence"],
        "relative_calculations": [
            {"output": "run_length", "operator": "CONSECUTIVE_SAME_RELATION", "formula": "reset when relation signature changes"},
            {"output": "path_verdict", "operator": "SUPPORT_COUNTERARGUMENT_PERSISTENCE", "formula": "EXCLUDE when probability/support blocks; WAIT when a qualified relation has not persisted; KEEP when every existing blocker clears"},
            {"output": "official_source_ready", "operator": "ALL_BLOCKERS_CLEARED", "formula": "relation satisfied and support >= minimum and probability + tolerance >= gate and run >= required"},
            {"output": "release", "operator": "ANY_QUALIFIED_CURRENT_RELATION", "formula": "release only a KEEP verdict; neither a raw BASE match nor environment alone can release"},
        ],
        "filters": ["ONE_ORIGINAL_ELIGIBILITY_GATE", "RAW_MATCH_CANNOT_BYPASS_ENVIRONMENT_OR_PERSISTENCE", "PARALLEL_METHODS_DO_NOT_RELEASE", "WAIT_WHILE_THE_OFFICIAL_GATE_IS_BLOCKED"],
    },
    "11": {
        "observation_start": "stage10_ready_candidates",
        "fixed_observations": ["candidate_transition_axis", "trade_action_axis", "observed_1h_zc_switch", "event_release_state"],
        "relative_calculations": [
            {"output": "trade_action", "operator": "ORDERED_FINAL_GATE", "formula": "first ready official current relation -> NOW; else observed 1h ZC switch -> NOW; else WAIT"}
        ],
        "filters": ["ONE_RELEASE_PER_EVENT", "REARM_AND_NOW_ARE_INDEPENDENT", "NEXT_CLOSED_5M_OPEN"],
    },
}


class ExportContractError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ExportContractError("source rule artifact must be a JSON object")
    return value


def _blueprint_stage(blueprint: Mapping[str, Any], number: str) -> dict[str, Any]:
    matches = [
        value
        for key, value in blueprint.get("stages", {}).items()
        if str(key).startswith(f"{number}_")
    ]
    if len(matches) != 1 or not isinstance(matches[0], dict):
        raise ExportContractError(f"blueprint does not contain exactly one Stage {number}")
    return deepcopy(matches[0])


def _module_execution_program(
    decision_stages: Mapping[str, Any], blueprint: Mapping[str, Any]
) -> list[dict[str, Any]]:
    program: list[dict[str, Any]] = []
    for number in STAGE_ORDER:
        supplied = _blueprint_stage(blueprint, number)
        if supplied.get("module") != STAGE_MODULES[number]:
            raise ExportContractError(
                f"blueprint Stage {number} is not owned by {STAGE_MODULES[number]}"
            )
        fixed_action = "DYNAMIC_FINAL_GATE" if number == "11" else "WAIT"
        source_decision = _clean_stage(decision_stages[number])
        excluded = [
            *STAGE_SCOPE_CONTRACTS[number]["exclude_when"],
            *[
                f"RUNTIME_FORBIDDEN::{field}"
                for field in source_decision.get("runtime_forbidden_inputs", [])
            ],
        ]
        handoff_target = supplied.get(
            "handoff", source_decision.get("default_handoff", "EXECUTION")
        )
        five_part_contract = {
            "view_when": STAGE_SCOPE_CONTRACTS[number]["view_when"],
            "exclude_when": excluded,
            "fixed_observations": deepcopy(
                STAGE_EXECUTION_CONTRACTS[number]["fixed_observations"]
            ),
            "relative_comparisons": deepcopy(
                STAGE_EXECUTION_CONTRACTS[number]["relative_calculations"]
            ),
            "handoff_to_next_stage": {
                "target": handoff_target,
                "values": list(STAGE_HANDOFF_FIELDS[number]),
            },
        }
        program.append(
            {
                "stage": number,
                "module": STAGE_MODULES[number],
                "executable_operator": EXECUTABLE_OPERATORS[number],
                "observation_start": STAGE_EXECUTION_CONTRACTS[number][
                    "observation_start"
                ],
                "fixed_observations": deepcopy(
                    STAGE_EXECUTION_CONTRACTS[number]["fixed_observations"]
                ),
                "relative_calculations": deepcopy(
                    STAGE_EXECUTION_CONTRACTS[number]["relative_calculations"]
                ),
                "filters": deepcopy(STAGE_EXECUTION_CONTRACTS[number]["filters"]),
                "fixed_action": fixed_action,
                "handoff_fields": list(STAGE_HANDOFF_FIELDS[number]),
                "five_part_contract": five_part_contract,
                "source_blueprint_contract": supplied,
                "source_decision_contract": source_decision,
            }
        )
    return program


def _reverse_stage_program(
    forward_program: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Trace the supplied handoffs backward without redefining the stages."""

    by_stage = {str(item["stage"]): item for item in forward_program}
    reverse: list[dict[str, Any]] = []
    for number in reversed(STAGE_ORDER):
        stage = by_stage[number]
        reverse.append(
            {
                "stage": number,
                "module": stage["module"],
                "responsibility": stage["source_decision_contract"].get(
                    "responsibility"
                ),
                "reverse_question": (
                    "WHICH_UPSTREAM_HANDOFF_VALUES_PRODUCED_THIS_STAGE_OUTPUT"
                ),
                "five_part_contract": deepcopy(stage["five_part_contract"]),
                "source_contract_preserved": True,
            }
        )
    return reverse


def _feature_origin(feature: str) -> dict[str, str]:
    base = feature.split("::", 1)[-1]
    if base == "side_code" or base.startswith(("candidate_", "rearm_")) or base == "new_extreme_now":
        return {
            "stage": "03",
            "module": "candidate_revision.py",
            "observation_start": "CURRENT_CANDIDATE_PREFIX",
        }
    if base.startswith(("candle_", "volume_")) or base in {
        "upper_wick_ratio",
        "lower_wick_ratio",
        "close_position_for_side",
        "range_ratio20",
    }:
        return {
            "stage": "06",
            "module": "candle_volume_distribution.py",
            "observation_start": "LATEST_CLOSED_5M_CANDLE",
        }
    if base.startswith("macd_"):
        return {
            "stage": "07",
            "module": "multitf_zc_distribution.py",
            "observation_start": "LATEST_CLOSED_TIMEFRAME_WITNESS",
        }
    return {
        "stage": "05",
        "module": "time_distribution.py",
        "observation_start": "CURRENT_WAVE_PREFIX",
    }


def _base_feature_method(feature: str) -> str:
    """Use the supplied six-method module's observation ownership."""

    if feature.startswith("macd_"):
        if feature.startswith("macd_5m_"):
            return "MOMENTUM_SPEED"
        return (
            "MACRO_TREND"
            if any(f"macd_{timeframe}_" in feature for timeframe in ("4h", "1d", "1w"))
            else "MID_MACD_TREND"
        )
    if feature.startswith("volume_") or feature == "range_ratio20":
        return "VOLUME_RANGE"
    if feature.startswith("candle_") or feature in {
        "upper_wick_ratio",
        "lower_wick_ratio",
        "close_position_for_side",
    }:
        return "CANDLE_REVERSAL"
    if feature.startswith("speed_"):
        return "MOMENTUM_SPEED"
    return "WAVE_REARM_AGE"


def _relative_operand(feature: str) -> dict[str, Any]:
    if "::" not in feature:
        return {
            "feature": feature,
            "calculation": "CURRENT_VALUE",
            "lookback_closed_5m_bars": 0,
            "origin": _feature_origin(feature),
        }
    prefix, base = feature.split("::", 1)
    match = re.fullmatch(r"change_(\d+)", prefix)
    if not match:
        raise ExportContractError(f"unknown relative feature expression: {feature}")
    lookback = int(match.group(1))
    return {
        "feature": base,
        "calculation": "CURRENT_MINUS_PRIOR",
        "lookback_closed_5m_bars": lookback,
        "formula": f"{base}[t] - {base}[t-{lookback}]",
        "origin": _feature_origin(base),
    }


def _condition(condition: Mapping[str, Any]) -> dict[str, Any]:
    operator = str(condition["operator"])
    if operator not in {">", "<="}:
        raise ExportContractError(f"unsupported existing operator: {operator}")
    return {
        "operand": _relative_operand(str(condition["feature"])),
        "operator": operator,
        "learned_boundary": condition["threshold"],
    }


def _learned_result(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "state_probability": item["state_probability"],
        "event_support": item["event_support"],
        "survivor_event_support": item.get("survivor_event_support"),
        "replaced_event_support": item.get("replaced_event_support"),
        "observations": item.get("observations"),
        "minimum_persistence_bars": item.get("minimum_persistence_bars", 1),
        "minimum_persistence_bars_by_side": item.get(
            "minimum_persistence_bars_by_side", {}
        ),
    }


def _source_indexes(source: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    base: dict[str, dict[str, Any]] = {}
    context: dict[str, dict[str, Any]] = {}
    for background, behavior in source["behaviors"].items():
        for channel_order, channel in enumerate(behavior.get("channels", [])):
            feature = str(channel["feature"])
            for bin_order, item in enumerate(channel.get("bins", [])):
                signature = f"{background}|{feature}|{item['bin']}"
                if signature in base:
                    raise ExportContractError(f"duplicate base output address: {signature}")
                base[signature] = {
                    "background": background,
                    "feature": feature,
                    "channel_order": channel_order,
                    "bin_order": bin_order,
                    "item": item,
                }
        for rule_order, item in enumerate(behavior.get("family_context_rules", [])):
            family = str(item.get("entry_family", "UNKNOWN"))
            variant = str(item.get("rule_variant", "SNAPSHOT_V1"))
            bin_id = int(item.get("context_bin", -1))
            signature = f"{background}|CONTEXT::{family}::{variant}|{bin_id}"
            if signature in context:
                raise ExportContractError(f"duplicate context output address: {signature}")
            context[signature] = {
                "background": background,
                "family": family,
                "variant": variant,
                "rule_order": rule_order,
                "item": item,
            }
    return base, context


def _compile_base(calculation_id: str, source_address: str, record: Mapping[str, Any]) -> dict[str, Any]:
    item = record["item"]
    feature = str(record["feature"])
    method = _base_feature_method(feature)
    return {
        "calculation_id": calculation_id,
        "source_calculation_address": source_address,
        "background": record["background"],
        "output_source": "BASE",
        "method": method,
        "calculation_path": "BASE",
        "semantic_role": "SIGNAL_FORMULA_CLAIM_INSIDE_CURRENT_BACKGROUND",
        "raw_match_is_entry": False,
        "calculation_group": f"{method}::BASE",
        "calculation_link": {
            "origin": _feature_origin(feature),
            "recognition_stage": "09",
            "persistence_stage": "10",
            "decision_stage": "11",
        },
        "calculation": {
            "operator": "INTERVAL_CONTAINS",
            "operand": _relative_operand(feature),
            "lower_exclusive": item.get("lower_exclusive"),
            "upper_inclusive": item.get("upper_inclusive"),
            "formula": "(lower is null or value > lower) and (upper is null or value <= upper)",
        },
        "output_value": {
            "feature": feature,
            "bin": item["bin"],
            **_learned_result(item),
        },
        "artifact_order": {
            "channel": record["channel_order"],
            "bin": record["bin_order"],
        },
    }


def _compile_context(calculation_id: str, source_address: str, record: Mapping[str, Any]) -> dict[str, Any]:
    item = record["item"]
    family = str(record["family"])
    conditions = list(item.get("conditions", []))
    contains_own_family_signal = any(
        _base_feature_method(str(condition["feature"]).split("::", 1)[-1])
        == family
        for condition in conditions
    )
    if not contains_own_family_signal:
        raise ExportContractError(
            f"context relation has environment but no {family} signal condition: "
            f"{source_address}"
        )
    return {
        "calculation_id": calculation_id,
        "source_calculation_address": source_address,
        "background": record["background"],
        "output_source": "CONTEXT",
        "method": family,
        "calculation_path": "CONTEXT",
        "semantic_role": "SAME_METHOD_SIGNAL_FORMULA_WITH_RELATIVE_ENVIRONMENT",
        "contains_own_family_signal": contains_own_family_signal,
        "environment_only_signal": False,
        "raw_match_is_entry": False,
        "calculation_group": f"{family}::CONTEXT",
        "calculation_link": {
            "origins": sorted(
                {
                    (origin := _feature_origin(str(condition["feature"]))) ["stage"]: origin
                    for condition in item.get("conditions", [])
                }.values(),
                key=lambda value: value["stage"],
            ),
            "recognition_stage": "09",
            "persistence_stage": "10",
            "decision_stage": "11",
        },
        "calculation": {
            "operator": "ALL_CONDITIONS_TRUE",
            "conditions": [_condition(condition) for condition in conditions],
        },
        "output_value": {
            "entry_family": family,
            "rule_variant": record["variant"],
            "context_bin": item.get("context_bin", -1),
            "candidate_id": item.get("candidate_id"),
            **_learned_result(item),
        },
        "artifact_order": {"rule": record["rule_order"]},
    }


def _clean_stage(stage: Mapping[str, Any]) -> dict[str, Any]:
    cleaned = deepcopy(dict(stage))
    for key in (
        "learned_now_signatures",
        "learned_release_batons",
        "study_only_parity_reconstruction",
    ):
        cleaned.pop(key, None)
    return cleaned


def _all_runtime_calculations(
    source: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], int, int]:
    """Compile every learned relation, independently of the evidence list."""

    base_index, context_index = _source_indexes(source)
    calculations: list[dict[str, Any]] = []
    by_source_address: dict[str, dict[str, Any]] = {}
    base_count = 0
    context_count = 0

    # Dictionary insertion order is the artifact's original calculation order.
    for source_address, record in base_index.items():
        calculation_id = f"CALC-{len(calculations) + 1:05d}"
        calculation = _compile_base(calculation_id, source_address, record)
        calculations.append(calculation)
        by_source_address[source_address] = calculation
        base_count += 1
    for source_address, record in context_index.items():
        calculation_id = f"CALC-{len(calculations) + 1:05d}"
        calculation = _compile_context(calculation_id, source_address, record)
        calculations.append(calculation)
        by_source_address[source_address] = calculation
        context_count += 1

    return calculations, by_source_address, base_count, context_count


def _calculation_operands(calculation: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    body = calculation["calculation"]
    if calculation["output_source"] == "BASE":
        return [body["operand"]]
    return [condition["operand"] for condition in body.get("conditions", [])]


def _reverse_dependency_trace(
    source_address: str, calculation: Mapping[str, Any]
) -> dict[str, Any]:
    operands = _calculation_operands(calculation)
    features_by_stage: dict[str, list[str]] = {number: [] for number in STAGE_ORDER}
    for operand in operands:
        origin_stage = str(operand["origin"]["stage"])
        feature = str(operand["feature"])
        if feature not in features_by_stage[origin_stage]:
            features_by_stage[origin_stage].append(feature)
    learned = calculation["output_value"]
    return {
        "reverse_order": list(reversed(STAGE_ORDER)),
        "11": {
            "status": "RECOVERED_FROM_2775_EVIDENCE_SET",
            "historical_output_address": source_address,
            "runtime_selector": False,
        },
        "10": {
            "status": "RECOVERED_LEARNED_ELIGIBILITY_VALUES",
            "state_probability": learned["state_probability"],
            "event_support": learned["event_support"],
            "minimum_persistence_bars": learned.get(
                "minimum_persistence_bars", 1
            ),
            "minimum_persistence_bars_by_side": learned.get(
                "minimum_persistence_bars_by_side", {}
            ),
            "case_level_run_and_blocker_state": "NOT_ENCODED_IN_THIS_2775_ADDRESS",
        },
        "09": {
            "status": "RECOVERED_RELATION_REFERENCE",
            "runtime_calculation_id": calculation["calculation_id"],
            "output_source": calculation["output_source"],
            "method": calculation["method"],
            "background": calculation["background"],
        },
        "08": {
            "status": "STAGE_CONTRACT_RECOVERED_CASE_HANDOFF_NOT_PACKAGED",
            "features_referenced_by_selected_relation": features_by_stage["08"],
        },
        "07": {
            "status": "RELATION_OPERANDS_RECOVERED",
            "features_referenced_by_selected_relation": features_by_stage["07"],
        },
        "06": {
            "status": "RELATION_OPERANDS_RECOVERED",
            "features_referenced_by_selected_relation": features_by_stage["06"],
        },
        "05": {
            "status": "RELATION_OPERANDS_RECOVERED",
            "features_referenced_by_selected_relation": features_by_stage["05"],
        },
        "04": {
            "status": "BACKGROUND_RECOVERED",
            "context_background_name": calculation["background"],
        },
        "03": {
            "status": "RELATION_OPERANDS_RECOVERED",
            "features_referenced_by_selected_relation": features_by_stage["03"],
            "case_level_candidate_handoff": "NOT_ENCODED_IN_THIS_2775_ADDRESS",
        },
    }


def _historical_evidence(
    selected: list[str], by_source_address: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Link past release results to calculations without making them executable selectors."""

    evidence: list[dict[str, Any]] = []
    for index, source_address in enumerate(selected, 1):
        calculation = by_source_address.get(source_address)
        if calculation is None:
            raise ExportContractError(
                f"historical evidence has no existing calculation: {source_address}"
            )
        evidence.append(
            {
                "evidence_id": f"EVIDENCE-{index:04d}",
                "runtime_calculation_id": calculation["calculation_id"],
                "historical_output_address": source_address,
                "historical_output_value": deepcopy(calculation["output_value"]),
                "reverse_dependency_trace": _reverse_dependency_trace(
                    source_address, calculation
                ),
                "role": "PAST_RELEASE_EVIDENCE_ONLY",
            }
        )
    return evidence


def _evidence_breakdown(
    evidence: list[Mapping[str, Any]],
    by_source_address: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {
        "by_background": {},
        "by_method": {},
        "by_output_source": {},
    }
    for item in evidence:
        calculation = by_source_address[item["historical_output_address"]]
        for bucket, value in (
            ("by_background", calculation["background"]),
            ("by_method", calculation["method"]),
            ("by_output_source", calculation["output_source"]),
        ):
            name = str(value)
            result[bucket][name] = result[bucket].get(name, 0) + 1
    return result


def _release_baton_evidence(
    batons: list[Mapping[str, Any]],
    by_source_address: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Preserve Stage 10 opinions carried into Stage 11 as study-only evidence."""

    evidence: list[dict[str, Any]] = []
    dialogue_patterns: set[tuple[tuple[str, str], ...]] = set()
    for index, baton in enumerate(batons, 1):
        source_address = (
            f"{baton['background']}|{baton['feature']}|{int(baton['bin'])}"
        )
        calculation = by_source_address.get(source_address)
        if calculation is None:
            raise ExportContractError(
                f"release baton has no existing calculation: {source_address}"
            )
        method_states: dict[str, dict[str, Any]] = {}
        raw_states = baton.get("method_wait_states", {})
        for method in METHODS:
            raw_state = str(raw_states[method])
            if raw_state == "METHOD_READY":
                method_states[method] = {
                    "ready": True,
                    "base_reason": None,
                    "context_reason": None,
                }
                continue
            parts = raw_state.split("|", 1)
            if len(parts) != 2 or not parts[0].startswith("BASE:") or not parts[1].startswith("CONTEXT:"):
                raise ExportContractError(
                    f"unknown Stage 10 baton state: {raw_state}"
                )
            method_states[method] = {
                "ready": False,
                "base_reason": parts[0].removeprefix("BASE:"),
                "context_reason": parts[1].removeprefix("CONTEXT:"),
            }
        dialogue_patterns.add(
            tuple(
                (method, str(raw_states[method]))
                for method in METHODS
            )
        )
        evidence.append(
            {
                "baton_id": f"BATON-{index:04d}",
                "winner_output_address": source_address,
                "winner_runtime_calculation_id": calculation["calculation_id"],
                "winner_method": calculation["method"],
                "winner_source_recorded": baton["rule_source"],
                "background": baton["background"],
                "side_code": int(baton["side_code"]),
                "stage10_method_states": method_states,
                "role": "PAST_STAGE10_TO_STAGE11_HANDOFF_EVIDENCE_ONLY",
            }
        )
    return evidence, {
        "count": len(evidence),
        "unique_winner_output_count": len(
            {item["winner_output_address"] for item in evidence}
        ),
        "unique_dialogue_pattern_count": len(dialogue_patterns),
        "used_by_runtime_router": False,
    }


def _referenced_intermediate_ledgers(
    stages: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    for number in STAGE_ORDER:
        source = stages[number].get("historical_pattern_source", {})
        if not isinstance(source, Mapping) or "path" not in source:
            continue
        references.append(
            {
                "stage": number,
                "path_recorded_by_source": source.get("path"),
                "sha256_recorded_by_source": source.get("sha256"),
                "bytes_recorded_by_source": source.get("bytes"),
                "observation_count_recorded_by_source": source.get(
                    "observation_count"
                ),
                "content_embedded_in_the_supplied_rule_json": False,
            }
        )
    return references


def build_executable_artifact(
    source: Mapping[str, Any],
    blueprint: Mapping[str, Any],
    *,
    source_sha256: str,
    blueprint_sha256: str,
) -> dict[str, Any]:
    if "decision_program" not in source:
        raise ExportContractError("source JSON has no existing Stage 03-11 decision_program")
    decision_program = source["decision_program"]
    if tuple(decision_program["stage_order"]) != STAGE_ORDER:
        raise ExportContractError("source Stage 03-11 order changed")
    stages = decision_program["stages"]
    for number, module in STAGE_MODULES.items():
        if stages[number].get("module") != module:
            raise ExportContractError(f"Stage {number} is not owned by {module}")

    behaviors = source["behaviors"]
    behavior_names = tuple(behaviors)
    if behavior_names != ("ALIGNED", "MIXED", "OPPOSED"):
        raise ExportContractError("behavior names/order changed")
    runtime_inputs = tuple(behaviors["ALIGNED"]["inputs"])
    if len(runtime_inputs) != 64 or any(tuple(value["inputs"]) != runtime_inputs for value in behaviors.values()):
        raise ExportContractError("the three rulebooks do not share the existing 64 inputs")

    calculations, by_source_address, base_count, context_count = (
        _all_runtime_calculations(source)
    )
    if (base_count, context_count) != (5_363, 5_500):
        raise ExportContractError(
            "full learned relation model changed: "
            f"base={base_count}, context={context_count}"
        )

    selected = list(stages["11"]["learned_now_signatures"])
    if len(selected) != 2_775 or len(set(selected)) != 2_775:
        raise ExportContractError("source does not contain 2,775 unique evidence outputs")
    evidence = _historical_evidence(selected, by_source_address)
    evidence_base_count = sum(
        1
        for item in evidence
        if by_source_address[item["historical_output_address"]]["output_source"]
        == "BASE"
    )
    evidence_context_count = len(evidence) - evidence_base_count
    if (evidence_context_count, evidence_base_count) != (2_499, 276):
        raise ExportContractError(
            "historical evidence split changed: "
            f"context={evidence_context_count}, base={evidence_base_count}"
        )
    release_batons, release_baton_summary = _release_baton_evidence(
        list(stages["11"].get("learned_release_batons", [])),
        by_source_address,
    )
    if len(release_batons) != 3_972:
        raise ExportContractError(
            "source does not contain 3,972 Stage 10-to-11 release batons"
        )

    selector = deepcopy(source["selector"])
    module_program = _module_execution_program(stages, blueprint)
    reverse_program = _reverse_stage_program(module_program)
    return {
        "schema": SCHEMA,
        "source": {
            "schema": source.get("schema"),
            "sha256": source_sha256,
            "stage_blueprint_sha256": blueprint_sha256,
            "official_artifact": source.get("official_artifact"),
            "source_official_entry_rules_sha256": source.get(
                "source_official_entry_rules_sha256"
            ),
            "transformation": "FULL_RELATION_MODEL_EXPORT_NO_RELEARNING",
        },
        "runtime_input_contract": {
            "base_clock": source.get("base_clock", "closed_5m"),
            "count": len(runtime_inputs),
            "features": list(runtime_inputs),
            "missing_value": behaviors["ALIGNED"].get("missing_value", -9.0),
            "forbidden": list(FORBIDDEN_RUNTIME_FIELDS),
        },
        "calculation_program": {
            "stage_order": list(STAGE_ORDER),
            "stages": module_program,
            "candidate_transition_axis": "REARM_ON_NEW_EXTREME_ELSE_HOLD",
            "trade_action_axis": "INDEPENDENT_WAIT_OR_NOW",
            "aggregation_across_methods": None,
            "single_total_score": None,
            "runtime_decision_dependency": "FULL_RELATION_MODEL_ONLY",
            "historical_evidence_dependency": False,
        },
        "signal_environment_contract": {
            "purpose": "SEPARATE_RAW_SIGNAL_CLAIM_FROM_KEEP_EXCLUDE_WAIT_JUDGMENT",
            "base_role": (
                "A same-method formula claim evaluated inside the current Stage 04 "
                "background; a raw interval match never enters by itself"
            ),
            "context_role": (
                "A contextualized version of the same method's signal formula; every "
                "context relation also contains at least one condition owned by that method"
            ),
            "context_is_independent_entry_signal": False,
            "three_judgment_lenses": [
                {
                    "lens": "RELATION",
                    "question": "Did the current signal formula and its required relative relations match?",
                },
                {
                    "lens": "ENVIRONMENT",
                    "question": "Does historical survivor/replaced evidence support this relation in the current background?",
                },
                {
                    "lens": "PERSISTENCE",
                    "question": "Has the same qualified relation outlived the learned false-candidate run?",
                },
            ],
            "verdicts": {
                "KEEP": "Every existing blocker cleared; the claim may reach Stage 11",
                "WAIT": "The relation is relevant but required persistence is incomplete",
                "EXCLUDE": "Support or probability evidence rejects this thought path",
                "NOT_APPLICABLE": "No current relation belongs to this thought path",
            },
            "final_action_rule": "ONLY_KEEP_CAN_REACH_STAGE11_NOW",
            "new_formula_or_threshold_added": False,
        },
        "reverse_reconstruction": {
            "purpose": "RECOVER_EXISTING_THOUGHT_FROM_2775_EVIDENCE_OUTPUTS",
            "learning_reconstruction_direction": list(reversed(STAGE_ORDER)),
            "runtime_execution_direction": list(STAGE_ORDER),
            "stage_responsibilities_redefined": False,
            "new_conditions_or_values_added": False,
            "historical_2775_role": "EVIDENCE_LEDGER_ONLY",
            "historical_2775_count": len(evidence),
            "historical_2775_breakdown": _evidence_breakdown(
                evidence, by_source_address
            ),
            "historical_stage10_to_stage11_batons": release_baton_summary,
            "reverse_stage_program": reverse_program,
            "case_level_handoff_limit": {
                "status": "SOURCE_INTERMEDIATE_LEDGER_REQUIRED_FOR_EXACT_PER_CASE_TRACE",
                "meaning": (
                    "The supplied rule JSON preserves stage contracts, aggregate audits, "
                    "2775 selected output addresses and learned values, but does not embed "
                    "the per-observation intermediate ledgers referenced by each stage."
                ),
                "referenced_ledgers": _referenced_intermediate_ledgers(stages),
                "guessing_allowed": False,
            },
        },
        "existing_selector_values": {
            "action_gate": selector["action_gate"],
            "probability_tolerance": selector["probability_tolerance"],
            "minimum_event_support": selector["minimum_event_support"],
            "minimum_context_event_support": selector[
                "minimum_context_event_support"
            ],
            "execution_policy": selector["execution_policy"],
        },
        "runtime_calculation_contract": {
            "count": len(calculations),
            "base_interval_calculations": base_count,
            "context_relation_calculations": context_count,
            "lookup_policy": "CALCULATE_CURRENT_RELATIONS_ACROSS_FULL_MODEL",
            "selected_historical_output_filter_used": False,
            "context_relations_containing_own_family_signal": context_count,
            "environment_only_context_relations": 0,
        },
        "runtime_calculations": calculations,
        "historical_learning_evidence": {
            "count": len(evidence),
            "context_outputs": evidence_context_count,
            "base_outputs": evidence_base_count,
            "release_baton_count": len(release_batons),
            "role": "VALIDATION_ONLY_NOT_A_RUNTIME_WHITELIST",
            "used_by_runtime_router": False,
            "source_thought_audit": deepcopy(source.get("thought_audit", {})),
            "results": evidence,
            "stage10_to_stage11_release_batons": release_batons,
        },
    }


def export_executable_artifact(
    source_path: str | Path,
    output_path: str | Path,
    blueprint_path: str | Path | None = None,
) -> dict[str, Any]:
    source_file = Path(source_path).expanduser().resolve()
    output_file = Path(output_path).expanduser().resolve()
    blueprint_file = (
        Path(blueprint_path).expanduser().resolve()
        if blueprint_path is not None
        else source_file.with_name(BLUEPRINT_FILENAME)
    )
    source = _read_object(source_file)
    blueprint = _read_object(blueprint_file)
    artifact = build_executable_artifact(
        source,
        blueprint,
        source_sha256=_sha256(source_file),
        blueprint_sha256=_sha256(blueprint_file),
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        prefix=".twin-export-",
        dir=output_file.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(artifact, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
    try:
        os.replace(temporary, output_file)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "output": str(output_file),
        "sha256": _sha256(output_file),
        "runtime_calculation_contract": artifact["runtime_calculation_contract"],
        "historical_learning_evidence": {
            key: value
            for key, value in artifact["historical_learning_evidence"].items()
            if key != "results"
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source", help="existing JSON containing decision_program and learned relation model"
    )
    parser.add_argument("output", help="single executable JSON to create")
    parser.add_argument(
        "--blueprint",
        help=f"supplied {BLUEPRINT_FILENAME}; defaults beside the source JSON",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            export_executable_artifact(args.source, args.output, args.blueprint),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
