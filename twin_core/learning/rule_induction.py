"""Stage 09: recognize the six learned methods from the existing rulebook.

This module evaluates learned relative boundaries. It never fits a new RSI-like
constant, averages methods, or turns Stage 09 into a trade action.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from twin_core import TwinContractError
from twin_core.authentic_contract import (
    AUTHENTIC_SHA256,
    BEHAVIORS,
    FORBIDDEN_RUNTIME_FIELDS,
    PARALLEL_METHODS,
    RUNTIME_INPUTS,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def base_feature_method(feature: str) -> str:
    """Map a single base channel to the method that owns its observation type."""
    if feature.startswith("macd_"):
        return "MACRO_TREND" if any(f"macd_{tf}_" in feature for tf in ("4h", "1d", "1w")) else "MID_MACD_TREND"
    if feature.startswith("volume_") or feature == "range_ratio20":
        return "VOLUME_RANGE"
    if feature.startswith("candle_") or feature in {
        "upper_wick_ratio",
        "lower_wick_ratio",
        "close_position_for_side",
    }:
        return "CANDLE_REVERSAL"
    if feature.startswith("speed_") or feature.startswith("candidate_rejection_"):
        return "MOMENTUM_SPEED"
    return "WAVE_REARM_AGE"


@dataclass(frozen=True)
class EvidenceCandidate:
    method: str
    source: str
    signature: str
    probability: float
    support: int
    required: int
    artifact_order: int
    feature: str | None = None
    bin: int | None = None
    rule_id: str | None = None
    variant: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _condition_value(
    feature: str,
    observation: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
) -> float | None:
    if "::" not in feature:
        return _number(observation.get(feature))
    prefix, base = feature.split("::", 1)
    if not prefix.startswith("change_"):
        return None
    horizon = int(prefix.removeprefix("change_"))
    if horizon <= 0 or len(history) < horizon:
        return None
    current = _number(observation.get(base))
    previous = _number(history[-horizon].get(base))
    return None if current is None or previous is None else current - previous


def _condition_matches(value: float | None, operator: str, threshold: Any) -> bool:
    if value is None:
        return False
    boundary = float(threshold)
    if operator == ">":
        return value > boundary
    if operator == "<=":
        return value <= boundary
    raise TwinContractError(f"unsupported learned operator: {operator}")


def _required(item: Mapping[str, Any], seek_side: str) -> int:
    per_side = item.get("minimum_persistence_bars_by_side", {})
    return int(per_side.get(seek_side, item.get("minimum_persistence_bars", 1)))


class RuleCatalog:
    """Hash-locked in-memory index of the supplied authentic rule package."""

    def __init__(self, payload: dict[str, Any]):
        self.payload = payload
        self.selector = payload["selector"]
        self.behaviors = payload["behaviors"]
        self.selected_signatures = frozenset(
            payload["decision_program"]["stages"]["11"]["learned_now_signatures"]
        )
        self._validate()

    @classmethod
    def load(cls, path: str | Path, *, verify_hash: bool = True) -> "RuleCatalog":
        source = Path(path).expanduser().resolve()
        if verify_hash and _sha256(source).lower() != AUTHENTIC_SHA256["augmented_rules"]:
            raise TwinContractError("authentic augmented rule SHA-256 mismatch")
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(payload)

    def _validate(self) -> None:
        if tuple(self.behaviors) != BEHAVIORS:
            raise TwinContractError("authentic behavior order or names changed")
        for name in BEHAVIORS:
            behavior = self.behaviors[name]
            if tuple(behavior["inputs"]) != RUNTIME_INPUTS:
                raise TwinContractError(f"{name} runtime inputs differ from the authentic 64")
            if behavior.get("decision_architecture") != "INDEPENDENT_EVIDENCE_NO_CROSS_FEATURE_INTERSECTION":
                raise TwinContractError(f"{name} has a different decision architecture")
        if len(self.selected_signatures) != 2_775:
            raise TwinContractError("authentic selected signature count is not 2,775")

    @staticmethod
    def validate_observation(observation: Mapping[str, Any]) -> None:
        forbidden = sorted(set(observation).intersection(FORBIDDEN_RUNTIME_FIELDS))
        if forbidden:
            raise TwinContractError(f"future/offline fields reached runtime: {forbidden}")
        missing = [feature for feature in RUNTIME_INPUTS if feature not in observation]
        if missing:
            raise TwinContractError(f"runtime observation is missing {len(missing)} authentic inputs")

    @staticmethod
    def _select_strongest(candidates: list[EvidenceCandidate]) -> EvidenceCandidate | None:
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda candidate: (
                -candidate.probability,
                -candidate.support,
                candidate.artifact_order,
                candidate.signature,
            ),
        )

    def evaluate(
        self,
        observation: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
        *,
        background: str,
        seek_side: str,
    ) -> dict[str, Any]:
        self.validate_observation(observation)
        if background not in BEHAVIORS:
            raise TwinContractError(f"unknown behavior background: {background}")
        side = seek_side.upper()
        if side not in {"H", "L"}:
            raise TwinContractError("seek_side must be H or L")
        behavior = self.behaviors[background]
        base_by_method: dict[str, list[EvidenceCandidate]] = {name: [] for name in PARALLEL_METHODS}
        context_by_method: dict[str, list[EvidenceCandidate]] = {name: [] for name in PARALLEL_METHODS}

        for channel_order, channel in enumerate(behavior.get("channels", [])):
            feature = str(channel["feature"])
            value = _number(observation.get(feature))
            if value is None:
                continue
            for item in channel.get("bins", []):
                lower = item.get("lower_exclusive")
                upper = item.get("upper_inclusive")
                if (lower is None or value > float(lower)) and (upper is None or value <= float(upper)):
                    signature = f"{background}|{feature}|{item['bin']}"
                    method = base_feature_method(feature)
                    base_by_method[method].append(
                        EvidenceCandidate(
                            method=method,
                            source="BASE",
                            signature=signature,
                            probability=float(item["state_probability"]),
                            support=int(item["event_support"]),
                            required=_required(item, side),
                            artifact_order=channel_order,
                            feature=feature,
                            bin=int(item["bin"]),
                        )
                    )
                    break

        for rule_order, rule in enumerate(behavior.get("family_context_rules", [])):
            conditions = rule.get("conditions", [])
            if not all(
                _condition_matches(
                    _condition_value(str(condition["feature"]), observation, history),
                    str(condition["operator"]),
                    condition["threshold"],
                )
                for condition in conditions
            ):
                continue
            method = str(rule["entry_family"])
            if method not in context_by_method:
                raise TwinContractError(f"unknown context rule family: {method}")
            variant = str(rule.get("rule_variant", "SNAPSHOT_V1"))
            bin_id = int(rule.get("context_bin", -1))
            signature = f"{background}|CONTEXT::{method}::{variant}|{bin_id}"
            context_by_method[method].append(
                EvidenceCandidate(
                    method=method,
                    source="CONTEXT",
                    signature=signature,
                    probability=float(rule["state_probability"]),
                    support=int(rule["event_support"]),
                    required=_required(rule, side),
                    artifact_order=rule_order,
                    rule_id=str(rule.get("candidate_id")),
                    variant=variant,
                    bin=bin_id,
                )
            )

        methods: dict[str, dict[str, Any]] = {}
        for method in PARALLEL_METHODS:
            base = self._select_strongest(base_by_method[method])
            context = self._select_strongest(context_by_method[method])
            methods[method] = {
                "base": base.to_dict() if base else None,
                "context": context.to_dict() if context else None,
            }
        return {
            "stage": "09",
            "methods": methods,
            "action_gate": float(self.selector["action_gate"]),
            "probability_tolerance": float(self.selector["probability_tolerance"]),
            "minimum_event_support": int(self.selector["minimum_event_support"]),
            "minimum_context_event_support": int(self.selector["minimum_context_event_support"]),
            "aggregation_across_methods": None,
            "single_total_score": None,
            "trade_action": "WAIT",
            "handoff": "10",
        }
