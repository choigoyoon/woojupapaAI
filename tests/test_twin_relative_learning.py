from __future__ import annotations

import importlib
import unittest

import numpy as np
import pandas as pd

from twin_core import FORBIDDEN_RUNTIME_FIELDS, RUNTIME_ACTIONS, TwinContractError
from twin_core.learning import CANONICAL_MODULES
from twin_core.learning.distribution_learner import run_learning_pipeline
from twin_core.learning.rule_induction import fit_independent_evidence
from twin_core.learning.multitf_zc_distribution import compute_all_8tf_zc_distributions
from twin_core.learning.mdd_problem import analyze_mdd_and_failures
from twin_core.learning.threshold_frontier import THINKING_STAGES, compute_threshold_frontier


ENTRY_POINTS = {
    "learning_worker": "run_learning_worker",
    "distribution_learner": "run_learning_pipeline",
    "candidate_revision": "build_candidate_revision_table",
    "context_partition": "partition_contexts",
    "time_distribution": "compute_time_distribution",
    "candle_volume_distribution": "compute_candle_volume_distribution",
    "multitf_zc_distribution": "compute_all_8tf_zc_distributions",
    "flow_order_distribution": "compute_flow_order_distribution",
    "rule_induction": "fit_independent_evidence",
    "repair_queue": "build_repair_queue",
    "threshold_frontier": "compute_threshold_frontier",
    "scorebook": "score_candidate_outcomes",
    "replay_builder": "build_prefix_replay",
    "mdd_problem": "analyze_mdd_and_failures",
    "learning_status": "get_learning_status",
    "artifact_export": "export_learning_artifacts",
}


def synthetic_market() -> tuple[pd.DataFrame, pd.DataFrame]:
    count = 240
    timestamp = pd.date_range("2024-01-01", periods=count, freq="5min", tz="UTC")
    close = 100.0 + 8.0 * np.sin(np.arange(count) * np.pi / 20.0)
    open_ = np.r_[close[0], close[:-1]]
    bars = pd.DataFrame(
        {
            "timestamp": timestamp,
            "open": open_,
            "high": np.maximum(open_, close) + 0.5,
            "low": np.minimum(open_, close) - 0.5,
            "close": close,
            "volume": 120.0 + 20.0 * np.sin(np.arange(count) * 0.3),
        }
    )
    pivot_idx = np.arange(0, 221, 20)
    side = np.where(np.arange(len(pivot_idx)) % 2 == 0, "L", "H")
    price = [
        bars.at[index, "low" if event_side == "L" else "high"]
        for index, event_side in zip(pivot_idx, side)
    ]
    ledger = pd.DataFrame(
        {
            "side": side,
            "pivot_idx": pivot_idx,
            "pivot_time": timestamp[pivot_idx],
            "pivot_price": price,
        }
    )
    return bars, ledger


class RelativeLearningContractTest(unittest.TestCase):
    def test_all_16_modules_have_the_canonical_entry_point(self):
        self.assertEqual(len(CANONICAL_MODULES), 16)
        self.assertEqual(set(CANONICAL_MODULES), set(ENTRY_POINTS))
        for module_name, function_name in ENTRY_POINTS.items():
            module = importlib.import_module(f"twin_core.learning.{module_name}")
            self.assertTrue(callable(getattr(module, function_name)))

    def test_boundary_is_learned_from_values_instead_of_hard_coded(self):
        base = pd.DataFrame(
            {
                "target_side": ["L"] * 6,
                "market_context": ["MIXED"] * 6,
                "fixed_action": ["WAIT", "WAIT", "WAIT", "NOW", "NOW", "NOW"],
                "relative_move": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )
        first = fit_independent_evidence(base, feature_names=["relative_move"])
        shifted = base.copy()
        shifted["relative_move"] += 100.0
        second = fit_independent_evidence(shifted, feature_names=["relative_move"])
        self.assertEqual(len(first), 1)
        self.assertAlmostEqual(
            float(second.iloc[0]["threshold"]) - float(first.iloc[0]["threshold"]), 100.0
        )
        self.assertEqual(first.iloc[0]["boundary_source"], "first_counterexample_frontier")

    def test_offline_answer_cannot_be_requested_as_a_rule_feature(self):
        observations = pd.DataFrame(
            {
                "target_side": ["L", "L"],
                "market_context": ["MIXED", "MIXED"],
                "fixed_action": ["WAIT", "NOW"],
            }
        )
        for field in FORBIDDEN_RUNTIME_FIELDS:
            observations[field] = [0, 1]
            with self.assertRaises(TwinContractError):
                fit_independent_evidence(observations, feature_names=[field])

    def test_full_relative_pipeline_and_next_open_execution(self):
        bars, ledger = synthetic_market()
        result = run_learning_pipeline(bars, ledger, strict_ledger=False)
        frontier = result["frontier"]
        replay = result["replay"]
        self.assertEqual(len(THINKING_STAGES), 9)
        self.assertTrue(set(frontier["runtime_action"]).issubset(RUNTIME_ACTIONS))
        self.assertFalse(result["score"]["future_fields_in_rules"])
        executions = replay.loc[replay["execution_idx"].notna()]
        self.assertTrue(
            (executions["execution_idx"].astype(int) == executions["decision_idx"] + 1).all()
        )
        self.assertTrue((executions["decision_uses_future"] == False).all())  # noqa: E712

    def test_same_direction_is_hold_and_rearm_can_coexist_with_now(self):
        observations = pd.DataFrame(
            {
                "case_id": [1],
                "step_id": [5],
                "target_side": ["L"],
                "market_context": ["MIXED"],
                "rearm_event": [True],
                "relative_move": [2.0],
                "current_position": ["LONG"],
            }
        )
        rules = pd.DataFrame(
            {
                "rule_id": ["r1"],
                "target_side": ["L"],
                "market_context": ["MIXED"],
                "entry_family": ["RELATIVE_PATH"],
                "feature": ["relative_move"],
                "operator": [">"],
                "threshold": [1.0],
                "balanced_accuracy": [1.0],
                "now_support": [2],
                "wait_support": [2],
                "true_now_support": [2],
                "true_wait_support": [0],
                "source_rows": [4],
                "boundary_source": ["observed_fixed_action_frontier"],
            }
        )
        result = compute_threshold_frontier(observations, rules)
        self.assertEqual(result.at[0, "runtime_action"], "HOLD")
        self.assertTrue(bool(result.at[0, "rearm_and_now"]))

    def test_multitimeframe_features_are_prefix_causal(self):
        bars, _ = synthetic_market()
        prefix = bars.assign(
            global_idx=np.arange(len(bars)),
            target_side=np.where(np.arange(len(bars)) % 2, "L", "H"),
        )
        first = compute_all_8tf_zc_distributions(prefix)
        changed = prefix.copy()
        changed.loc[120:, "close"] *= 3.0
        second = compute_all_8tf_zc_distributions(changed)
        columns = [column for column in first if column.startswith("macd_")]
        pd.testing.assert_frame_equal(first.loc[:119, columns], second.loc[:119, columns])

    def test_drawdown_is_bounded_when_an_uncontrolled_short_is_insolvent(self):
        replay = pd.DataFrame({"action": ["NOW"], "realized_return": [-2.0]})
        result = analyze_mdd_and_failures(replay, fee_bps=0)
        self.assertEqual(result["total_return_after_fees"], -1.0)
        self.assertEqual(result["max_drawdown"], -1.0)
        self.assertTrue(result["insolvent_without_risk_controls"])


if __name__ == "__main__":
    unittest.main()

