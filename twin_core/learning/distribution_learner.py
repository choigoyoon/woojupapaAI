"""Small orchestrator for the 16-module learning contract."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from twin_core.learning.artifact_export import export_learning_artifacts
from twin_core.learning.candidate_revision import (
    build_candidate_revision_table,
    probe_macd_lh_events,
)
from twin_core.learning.candle_volume_distribution import compute_candle_volume_distribution
from twin_core.learning.context_partition import partition_contexts
from twin_core.learning.flow_order_distribution import compute_flow_order_distribution
from twin_core.learning.learning_status import LearningStatus, get_learning_status
from twin_core.learning.mdd_problem import analyze_mdd_and_failures
from twin_core.learning.multitf_zc_distribution import compute_all_8tf_zc_distributions
from twin_core.learning.repair_queue import build_repair_queue
from twin_core.learning.replay_builder import build_prefix_replay
from twin_core.learning.rule_induction import fit_independent_evidence
from twin_core.learning.scorebook import score_candidate_outcomes
from twin_core.learning.threshold_frontier import compute_threshold_frontier
from twin_core.learning.time_distribution import compute_time_distribution


def run_learning_pipeline(
    ohlcv: pd.DataFrame,
    official_ledger: pd.DataFrame | None = None,
    *,
    output_dir: str | Path | None = None,
    audit_only: bool = False,
    strict_ledger: bool = True,
) -> dict[str, object]:
    status = LearningStatus()
    if official_ledger is None:
        with status.stage("candidate_revision.probe_macd_lh_events"):
            probe, audit = probe_macd_lh_events(ohlcv)
        audit["canonical_training_status"] = "BLOCKED_OFFICIAL_4136_LEDGER_REQUIRED"
        status.finish("AUDIT_ONLY")
        snapshot = get_learning_status(status)
        manifest = None
        if output_dir is not None:
            with status.stage("artifact_export.export_learning_artifacts"):
                manifest = export_learning_artifacts(
                    {"ohlcv_event_probe": probe, "ohlcv_audit": audit, "learning_status": snapshot},
                    output_dir,
                )
            snapshot = get_learning_status(status)
        return {
            "mode": "AUDIT_ONLY",
            "audit": audit,
            "status": snapshot,
            "manifest": manifest,
            "probe": probe,
        }
    if audit_only:
        raise ValueError("audit_only cannot be combined with an official ledger")

    with status.stage("candidate_revision.build_candidate_revision_table"):
        observations, events, contract = build_candidate_revision_table(
            ohlcv, official_ledger, strict=strict_ledger
        )
    with status.stage("time_distribution.compute_time_distribution"):
        observations = compute_time_distribution(observations)
    with status.stage("candle_volume_distribution.compute_candle_volume_distribution"):
        observations = compute_candle_volume_distribution(observations)
    with status.stage("multitf_zc_distribution.compute_all_8tf_zc_distributions"):
        observations = compute_all_8tf_zc_distributions(observations)
    with status.stage("flow_order_distribution.compute_flow_order_distribution"):
        observations = compute_flow_order_distribution(observations)
    with status.stage("context_partition.partition_contexts"):
        observations = partition_contexts(observations)
    with status.stage("rule_induction.fit_independent_evidence"):
        rules = fit_independent_evidence(observations)
    with status.stage("threshold_frontier.compute_threshold_frontier"):
        frontier = compute_threshold_frontier(observations, rules)
    with status.stage("scorebook.score_candidate_outcomes"):
        score, diagnostics = score_candidate_outcomes(frontier, rules)
    with status.stage("replay_builder.build_prefix_replay"):
        replay = build_prefix_replay(frontier, ohlcv)
    with status.stage("mdd_problem.analyze_mdd_and_failures"):
        mdd = analyze_mdd_and_failures(replay)
    with status.stage("repair_queue.build_repair_queue"):
        repair = build_repair_queue(frontier)

    manifest = None
    if output_dir is not None:
        with status.stage("artifact_export.export_learning_artifacts"):
            manifest = export_learning_artifacts(
                {
                    "rules": rules,
                    "score": score,
                    "mdd": mdd,
                    "ledger_contract": contract,
                    "diagnostics": diagnostics,
                    "repair_queue": repair,
                    "replay_actions": replay.loc[replay["action"].ne("WAIT")],
                    "learning_status": get_learning_status(status),
                },
                output_dir,
            )
    mode = "FULL_LEARNING" if contract["canonical"] else "NONCANONICAL_DIAGNOSTIC"
    status.finish("COMPLETE" if contract["canonical"] else "DIAGNOSTIC_COMPLETE")
    return {
        "mode": mode,
        "events": events,
        "observations": observations,
        "rules": rules,
        "frontier": frontier,
        "score": score,
        "replay": replay,
        "mdd": mdd,
        "repair_queue": repair,
        "ledger_contract": contract,
        "status": get_learning_status(status),
        "manifest": manifest,
    }

