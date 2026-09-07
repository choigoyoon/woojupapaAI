"""Read-only audit for the private authentic TWIN transfer bundle."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from twin_core import TwinContractError
from twin_core.authentic_contract import (
    AUTHENTIC_SHA256,
    BEHAVIORS,
    EXPECTED,
    FORBIDDEN_RUNTIME_FIELDS,
    NOW_ALLOWED_STAGES,
    OBSERVATION_ONLY_STAGES,
    PARALLEL_METHODS,
    RUNTIME_INPUTS,
    STAGE_MODULES,
    STAGE_ORDER,
)


BUNDLE_PATHS = {
    "augmented_rules": Path("scratch/learning_audit_logs/test_entry_rules_with_stage_11.json"),
    "base_rules": Path("twin_fixed_tool/outputs/runtime_rules/entry_rules.json"),
    "official_event_ledger": Path(
        "canonical_inputs_v1/full_4136_distribution_base_v1/"
        "event_ledger_4136_study_only_v1.parquet"
    ),
    "stage_blueprint": Path("scratch/learning_audit_logs/stages_03_to_11_authentic_blueprint.json"),
    "source_manifest": Path("canonical_inputs_v1/CANONICAL_INPUTS_MANIFEST_v1.json"),
    "output_manifest": Path("twin_fixed_tool/outputs/manifest.json"),
    "sha256_sums": Path("SHA256SUMS.txt"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TwinContractError(f"expected a JSON object: {path}")
    return value


def resolve_bundle_paths(bundle_root: str | Path) -> dict[str, Path]:
    root = Path(bundle_root).expanduser().resolve()
    if not root.is_dir():
        raise TwinContractError(f"bundle root is not a directory: {root}")
    resolved = {name: (root / relative).resolve() for name, relative in BUNDLE_PATHS.items()}
    for name, path in resolved.items():
        if root not in path.parents or not path.is_file():
            raise TwinContractError(f"missing or unsafe bundle file {name}: {path}")
    return resolved


def _rule_indexes(rules: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    context: dict[str, Any] = {}
    base: dict[str, Any] = {}
    for background, behavior in rules["behaviors"].items():
        for channel in behavior.get("channels", []):
            feature = channel["feature"]
            for item in channel.get("bins", []):
                signature = f"{background}|{feature}|{item['bin']}"
                if signature in base:
                    raise TwinContractError(f"duplicate base signature: {signature}")
                base[signature] = (feature, item)
        for rule in behavior.get("family_context_rules", []):
            variant = rule.get("rule_variant", "SNAPSHOT_V1")
            family = rule.get("entry_family", "UNKNOWN")
            signature = f"{background}|CONTEXT::{family}::{variant}|{rule.get('context_bin', -1)}"
            if signature in context:
                raise TwinContractError(f"duplicate context signature: {signature}")
            context[signature] = rule
    return context, base


def _stage_audit(rules: dict[str, Any], blueprint: dict[str, Any]) -> dict[str, Any]:
    program = rules["decision_program"]
    order = tuple(program["stage_order"])
    stages = program["stages"]
    module_matches = {
        stage: stages[stage].get("module") == module for stage, module in STAGE_MODULES.items()
    }
    no_early_now = all(stages[stage].get("now_allowed") is False for stage in OBSERVATION_ONLY_STAGES)
    final_now = all(
        stages[stage].get("now_allowed") is True
        or any(item.get("action") == "NOW" for item in stages[stage].get("ordered_reasoning", []))
        for stage in NOW_ALLOWED_STAGES
    )
    stage11 = stages["11"]
    independent_axes = (
        stage11.get("candidate_transition_axis") == "REARM_ON_NEW_EXTREME_ELSE_HOLD"
        and stage11.get("trade_action_axis") == "INDEPENDENT_FROM_CANDIDATE_TRANSITION"
    )
    blueprint_stages = blueprint.get("stages", {})
    blueprint_modules = [value.get("module") for value in blueprint_stages.values()]
    return {
        "stage_order": list(order),
        "stage_order_exact": order == STAGE_ORDER,
        "module_matches": module_matches,
        "all_module_matches": all(module_matches.values()),
        "blueprint_module_count": len(blueprint_modules),
        "no_now_before_stage_11": no_early_now,
        "stage_11_now_allowed": final_now,
        "rearm_and_trade_action_are_independent_axes": independent_axes,
    }


def _rules_audit(rules: dict[str, Any], base_rules: dict[str, Any]) -> dict[str, Any]:
    selected = rules["decision_program"]["stages"]["11"]["learned_now_signatures"]
    context, base = _rule_indexes(rules)
    matched_context = [signature for signature in selected if signature in context]
    matched_base = [signature for signature in selected if signature in base]
    unmatched = [signature for signature in selected if signature not in context and signature not in base]
    duplicate_selected = len(selected) - len(set(selected))

    condition_features: set[str] = set()
    temporal_horizons: Counter[int] = Counter()
    invalid_operators: set[str] = set()
    family_counts: Counter[str] = Counter()
    persistence_counts: Counter[int] = Counter()
    for signature in matched_context:
        rule = context[signature]
        family_counts[str(rule["entry_family"])] += 1
        persistence_counts[int(rule.get("minimum_persistence_bars", 1))] += 1
        for condition in rule.get("conditions", []):
            operator = str(condition.get("operator"))
            if operator not in {">", "<="}:
                invalid_operators.add(operator)
            feature = str(condition["feature"])
            if "::" in feature:
                prefix, feature = feature.split("::", 1)
                if prefix.startswith("change_"):
                    temporal_horizons[int(prefix.removeprefix("change_"))] += 1
            condition_features.add(feature)

    runtime_inputs = tuple(rules["behaviors"]["ALIGNED"]["inputs"])
    per_behavior_inputs_equal = all(
        tuple(rules["behaviors"][name]["inputs"]) == runtime_inputs for name in BEHAVIORS
    )
    decision_architectures = {
        rules["behaviors"][name].get("decision_architecture") for name in BEHAVIORS
    }
    behavior_cross_intersection = {
        name: rules["behaviors"][name].get("cross_feature_intersection_used")
        for name in BEHAVIORS
    }
    source_hash = str(rules.get("source_official_entry_rules_sha256", "")).lower()
    base_bytes_equal = rules.get("selector") == base_rules.get("selector") and rules.get(
        "behaviors"
    ) == base_rules.get("behaviors")

    return {
        "selected_signature_count": len(selected),
        "duplicate_selected_signatures": duplicate_selected,
        "matched_context_rules": len(matched_context),
        "matched_base_bins": len(matched_base),
        "unmatched_signatures": unmatched,
        "family_counts": dict(sorted(family_counts.items())),
        "persistence_bars": {str(k): v for k, v in sorted(persistence_counts.items())},
        "temporal_change_horizons": {str(k): v for k, v in sorted(temporal_horizons.items())},
        "invalid_condition_operators": sorted(invalid_operators),
        "condition_base_features_outside_runtime_64": sorted(condition_features - set(RUNTIME_INPUTS)),
        "runtime_input_count": len(runtime_inputs),
        "runtime_inputs_exact": runtime_inputs == RUNTIME_INPUTS,
        "all_behaviors_share_runtime_inputs": per_behavior_inputs_equal,
        "decision_architectures": sorted(str(value) for value in decision_architectures),
        "behavior_cross_feature_intersection": behavior_cross_intersection,
        "augmented_source_hash_matches_base": source_hash == AUTHENTIC_SHA256["base_rules"],
        "augmented_base_payload_equal": base_bytes_equal,
    }


def _ledger_audit(path: Path) -> dict[str, Any]:
    ledger = pd.read_parquet(path)
    side = ledger["official_side_score_only"].astype(str)
    raw_index = pd.to_numeric(ledger["official_event_raw_index_score_only"], errors="raise")
    sequence = pd.to_numeric(ledger["official_event_sequence_study_only"], errors="raise")
    expected_sequence = pd.Series(range(1, len(ledger) + 1), index=ledger.index)

    forbidden_lists: list[list[str]] = []
    for raw in ledger["runtime_input_forbidden_fields_json"]:
        value = json.loads(raw)
        if not isinstance(value, list):
            raise TwinContractError("ledger forbidden field payload is not a list")
        forbidden_lists.append([str(item) for item in value])
    forbidden_union = set(item for values in forbidden_lists for item in values)

    return {
        "rows": len(ledger),
        "h": int(side.eq("H").sum()),
        "l": int(side.eq("L").sum()),
        "first_side": side.iloc[0],
        "alternating": bool(side.iloc[1:].ne(side.shift().iloc[1:]).all()),
        "event_sequence_exact": bool(sequence.eq(expected_sequence).all()),
        "raw_index_strictly_increasing": bool(raw_index.diff().dropna().gt(0).all()),
        "first_raw_index": int(raw_index.iloc[0]),
        "last_raw_index": int(raw_index.iloc[-1]),
        "forbidden_field_union": sorted(forbidden_union),
        "ledger_forbidden_fields_known": forbidden_union.issubset(FORBIDDEN_RUNTIME_FIELDS),
    }


def _find_by_hash(paths: Iterable[Path], expected_hash: str) -> Path | None:
    for path in paths:
        if path.is_file() and sha256(path).lower() == expected_hash.lower():
            return path
    return None


def _transfer_hash_audit(bundle_root: Path, sums_path: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for raw_line in sums_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        expected, raw_relative = line.split(maxsplit=1)
        relative = raw_relative.removeprefix("*").replace("/", "\\")
        path = (bundle_root / relative).resolve()
        safe = bundle_root in path.parents
        exists = safe and path.is_file()
        actual = sha256(path) if exists else None
        entries.append(
            {
                "path": raw_relative.removeprefix("*"),
                "safe": safe,
                "exists": exists,
                "sha256_matches": actual is not None and actual.lower() == expected.lower(),
            }
        )
    return {
        "listed_files": len(entries),
        "matched_files": sum(int(item["sha256_matches"]) for item in entries),
        "all_match": bool(entries) and all(item["sha256_matches"] for item in entries),
        "failures": [item for item in entries if not item["sha256_matches"]],
    }


def audit_observation_ohlcv(ohlcv_dir: str | Path, *, ledger_last_raw_index: int) -> dict[str, Any]:
    """Verify the public OHLCV as observation data, not as the private replay source."""
    directory = Path(ohlcv_dir).expanduser().resolve()
    manifest_path = directory / "manifest.json"
    manifest = _load_json(manifest_path)
    files = manifest.get("files", [])
    frames: list[pd.DataFrame] = []
    hash_matches: dict[str, bool] = {}
    for item in files:
        path = (directory / str(item["file"])).resolve()
        if directory not in path.parents or not path.is_file():
            raise TwinContractError(f"missing or unsafe OHLCV file: {path}")
        hash_matches[path.name] = sha256(path).lower() == str(item["sha256"]).lower()
        frames.append(pd.read_csv(path, usecols=["timestamp"]))
    timestamps = pd.concat(frames, ignore_index=True)["timestamp"]
    clock = pd.to_datetime(timestamps, utc=True, errors="raise")
    rows = len(clock)
    continuous = bool(clock.diff().dropna().eq(pd.Timedelta(minutes=5)).all())
    duplicate_count = int(clock.duplicated().sum())
    expected_rows = int(manifest.get("rows", manifest.get("total_rows", -1)))
    passed = bool(
        rows == expected_rows == 660_853
        and all(hash_matches.values())
        and continuous
        and duplicate_count == 0
        and ledger_last_raw_index < rows
    )
    return {
        "status": "PASS_OBSERVATION_ONLY" if passed else "FAIL",
        "rows": rows,
        "manifest_rows": expected_rows,
        "first_timestamp": clock.iloc[0].isoformat(),
        "last_timestamp": clock.iloc[-1].isoformat(),
        "continuous_5m": continuous,
        "duplicate_timestamps": duplicate_count,
        "file_hashes_match_manifest": all(hash_matches.values()),
        "ledger_indices_within_bounds": ledger_last_raw_index < rows,
        "official_exact_replay_source": False,
        "reason": (
            "This public Binance/Bitstamp table is valid observation input. The authentic manifest locks exact replay to causal_market_rows_v1.parquet with a different file contract."
        ),
    }


def audit_bundle(
    bundle_root: str | Path,
    *,
    causal_market_candidates: Iterable[str | Path] = (),
    ohlcv_dir: str | Path | None = None,
) -> dict[str, Any]:
    paths = resolve_bundle_paths(bundle_root)
    hash_checks = {
        name: sha256(paths[name]).lower() == AUTHENTIC_SHA256[name]
        for name in ("augmented_rules", "base_rules", "official_event_ledger", "stage_blueprint")
    }
    rules = _load_json(paths["augmented_rules"])
    base_rules = _load_json(paths["base_rules"])
    blueprint = _load_json(paths["stage_blueprint"])
    output_manifest = _load_json(paths["output_manifest"])
    transfer_hashes = _transfer_hash_audit(Path(bundle_root).expanduser().resolve(), paths["sha256_sums"])
    stage = _stage_audit(rules, blueprint)
    rule = _rules_audit(rules, base_rules)
    ledger = _ledger_audit(paths["official_event_ledger"])

    candidates = [Path(path).expanduser().resolve() for path in causal_market_candidates]
    causal_market = _find_by_hash(candidates, AUTHENTIC_SHA256["causal_market_rows"])
    source_present = causal_market is not None

    checks = {
        "all_provided_hashes_match": all(hash_checks.values()) and transfer_hashes["all_match"],
        "stages_exact": bool(
            stage["stage_order_exact"]
            and stage["all_module_matches"]
            and stage["no_now_before_stage_11"]
            and stage["stage_11_now_allowed"]
            and stage["rearm_and_trade_action_are_independent_axes"]
        ),
        "rules_exact": bool(
            rule["selected_signature_count"] == EXPECTED["selected_signatures"]
            and rule["matched_context_rules"] == EXPECTED["selected_context_rules"]
            and rule["matched_base_bins"] == EXPECTED["selected_base_bins"]
            and not rule["unmatched_signatures"]
            and rule["duplicate_selected_signatures"] == 0
            and rule["runtime_inputs_exact"]
            and not rule["condition_base_features_outside_runtime_64"]
            and not rule["invalid_condition_operators"]
            and rule["augmented_source_hash_matches_base"]
            and rule["augmented_base_payload_equal"]
        ),
        "ledger_exact": bool(
            ledger["rows"] == EXPECTED["official_events"]
            and ledger["h"] == EXPECTED["official_h"]
            and ledger["l"] == EXPECTED["official_l"]
            and ledger["first_side"] == "H"
            and ledger["alternating"]
            and ledger["event_sequence_exact"]
            and ledger["raw_index_strictly_increasing"]
            and ledger["ledger_forbidden_fields_known"]
        ),
        "exact_replay_source_present": source_present,
    }
    status = (
        "STRUCTURE_PASS_SOURCE_PRESENT_REPLAY_NOT_RUN"
        if source_present
        else "STRUCTURE_PASS_REPLAY_SOURCE_MISSING"
    )
    if not all(checks[key] for key in ("all_provided_hashes_match", "stages_exact", "rules_exact", "ledger_exact")):
        status = "FAIL"

    result = {
        "schema": "twin.authentic-bundle-audit.v1",
        "status": status,
        "hash_checks": hash_checks,
        "transfer_hashes": transfer_hashes,
        "checks": checks,
        "stage_audit": stage,
        "rule_audit": rule,
        "ledger_audit": ledger,
        "exact_replay": {
            "ready": False,
            "source_present": source_present,
            "replay_executed": False,
            "required_filename": "causal_market_rows_v1.parquet",
            "required_sha256": AUTHENTIC_SHA256["causal_market_rows"],
            "matched_path": str(causal_market) if causal_market else None,
            "reason_if_blocked": (
                "The transfer bundle contains the ledger and rules, but not the hash-locked causal market rows."
                if not source_present
                else "The source is present, but the 656,136-row causal feature build and sequential parity replay have not run."
            ),
        },
        "artifact_claims_not_recomputed": {
            "rule_now_events": EXPECTED["rule_now_events"],
            "fallback_now_events": EXPECTED["fallback_now_events"],
            "note": "These are source-artifact claims until exact sequential replay succeeds.",
        },
        "source_discrepancies": [
            {
                "field": "manifest.architecture.cross_feature_intersection_used",
                "manifest_value": output_manifest["architecture"]["cross_feature_intersection_used"],
                "behavior_rulebooks_value": all(
                    rules["behaviors"][name]["cross_feature_intersection_used"] for name in BEHAVIORS
                ),
                "resolution": "Do not silently choose; behavior rulebooks and decision_architecture remain authoritative for runtime.",
            },
            {
                "field": "decision_program.stage07.match_fields_order",
                "stage07_field_count": len(rules["decision_program"]["stages"]["07"]["match_fields_order"]),
                "fields_outside_runtime_64": sorted(
                    set(rules["decision_program"]["stages"]["07"]["match_fields_order"])
                    - set(RUNTIME_INPUTS)
                ),
                "resolution": "The evaluator uses the hash-locked 64 runtime inputs; omitted long-timeframe ZC counts cannot be supplied or invented.",
            },
        ],
    }
    if ohlcv_dir is not None:
        result["ohlcv_observation_audit"] = audit_observation_ohlcv(
            ohlcv_dir, ledger_last_raw_index=int(ledger["last_raw_index"])
        )
    return result
