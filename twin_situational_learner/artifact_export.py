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


SCHEMA = "twin.executable-thought-program.v2"
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
    return {
        "calculation_id": calculation_id,
        "source_calculation_address": source_address,
        "background": record["background"],
        "output_source": "BASE",
        "calculation_group": "BASE_SINGLE_STRONGEST_CHANNEL",
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
    return {
        "calculation_id": calculation_id,
        "source_calculation_address": source_address,
        "background": record["background"],
        "output_source": "CONTEXT",
        "calculation_group": family,
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
            "conditions": [_condition(condition) for condition in item.get("conditions", [])],
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
                "role": "PAST_RELEASE_EVIDENCE_ONLY",
            }
        )
    return evidence


def build_executable_artifact(source: Mapping[str, Any], *, source_sha256: str) -> dict[str, Any]:
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

    selector = deepcopy(source["selector"])
    return {
        "schema": SCHEMA,
        "source": {
            "schema": source.get("schema"),
            "sha256": source_sha256,
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
            "stages": [
                {
                    "stage": number,
                    "module": STAGE_MODULES[number],
                    "executable_operator": EXECUTABLE_OPERATORS[number],
                    "source_stage_contract": _clean_stage(stages[number]),
                }
                for number in STAGE_ORDER
            ],
            "candidate_transition_axis": "REARM_ON_NEW_EXTREME_ELSE_HOLD",
            "trade_action_axis": "INDEPENDENT_WAIT_OR_NOW",
            "aggregation_across_methods": None,
            "single_total_score": None,
            "runtime_decision_dependency": "FULL_RELATION_MODEL_ONLY",
            "historical_evidence_dependency": False,
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
        },
        "runtime_calculations": calculations,
        "historical_learning_evidence": {
            "count": len(evidence),
            "context_outputs": evidence_context_count,
            "base_outputs": evidence_base_count,
            "role": "VALIDATION_ONLY_NOT_A_RUNTIME_WHITELIST",
            "used_by_runtime_router": False,
            "results": evidence,
        },
    }


def export_executable_artifact(source_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    source_file = Path(source_path).expanduser().resolve()
    output_file = Path(output_path).expanduser().resolve()
    source = _read_object(source_file)
    artifact = build_executable_artifact(source, source_sha256=_sha256(source_file))
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
    args = parser.parse_args()
    print(
        json.dumps(
            export_executable_artifact(args.source, args.output),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
