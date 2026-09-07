from __future__ import annotations

import json
import os
from pathlib import Path
import unittest

from twin_core.authentic_audit import audit_bundle
from twin_core import TwinContractError
from twin_core.authentic_contract import EXPECTED, RUNTIME_INPUTS, STAGE_ORDER
from twin_core.learning import CANONICAL_MODULES, STAGE_OWNERS
from twin_core.learning.learning_worker import AuthenticLearningWorker
from twin_core.learning.rule_induction import RuleCatalog


BUNDLE = os.environ.get("TWIN_AUTHENTIC_BUNDLE")
OHLCV = os.environ.get("TWIN_OHLCV_DIR")


class AuthenticBundleTests(unittest.TestCase):
    def test_static_contract_counts(self) -> None:
        self.assertEqual(STAGE_ORDER, ("03", "04", "05", "06", "07", "08", "09", "10", "11"))
        self.assertEqual(len(RUNTIME_INPUTS), 64)
        self.assertEqual(
            EXPECTED["selected_context_rules"] + EXPECTED["selected_base_bins"], 2_775
        )
        self.assertEqual(len(CANONICAL_MODULES), 16)
        self.assertEqual(tuple(STAGE_OWNERS), STAGE_ORDER)

    @unittest.skipUnless(BUNDLE, "set TWIN_AUTHENTIC_BUNDLE to the private bundle root")
    def test_private_bundle_structure(self) -> None:
        result = audit_bundle(Path(BUNDLE))
        self.assertEqual(result["status"], "STRUCTURE_PASS_REPLAY_SOURCE_MISSING")
        self.assertTrue(result["checks"]["all_provided_hashes_match"])
        self.assertTrue(result["checks"]["stages_exact"])
        self.assertTrue(result["checks"]["rules_exact"])
        self.assertTrue(result["checks"]["ledger_exact"])
        self.assertFalse(result["checks"]["exact_replay_source_present"])
        self.assertEqual(result["rule_audit"]["matched_context_rules"], 2_499)
        self.assertEqual(result["rule_audit"]["matched_base_bins"], 276)
        self.assertEqual(result["rule_audit"]["unmatched_signatures"], [])

    @unittest.skipUnless(BUNDLE, "set TWIN_AUTHENTIC_BUNDLE to the private bundle root")
    def test_ordered_runtime_keeps_rearm_and_now_independent(self) -> None:
        path = Path(BUNDLE) / "scratch/learning_audit_logs/test_entry_rules_with_stage_11.json"
        catalog = RuleCatalog.load(path)
        observation = {feature: 0.0 for feature in RUNTIME_INPUTS}
        observation.update(
            {
                "side_code": 1.0,
                "new_extreme_now": 1.0,
                "macd_4h_sign_for_side": 1.0,
                "macd_1d_sign_for_side": 1.0,
                "macd_1w_sign_for_side": 1.0,
            }
        )
        result = AuthenticLearningWorker(catalog).step(
            observation,
            event_key="runtime-wave-1",
            seek_side="L",
            raw_position=100,
            observed_1h_zc_switch=True,
        )
        self.assertEqual(result["stage_order"], list(STAGE_ORDER))
        self.assertTrue(all(result["stages"][stage]["trade_action"] == "WAIT" for stage in STAGE_ORDER[:-1]))
        self.assertEqual(result["candidate_transition"], "REARM")
        self.assertEqual(result["trade_action"], "NOW")
        self.assertTrue(result["stages"]["11"]["rearm_and_now"])

    @unittest.skipUnless(BUNDLE, "set TWIN_AUTHENTIC_BUNDLE to the private bundle root")
    def test_runtime_rejects_offline_answer_fields(self) -> None:
        path = Path(BUNDLE) / "scratch/learning_audit_logs/test_entry_rules_with_stage_11.json"
        catalog = RuleCatalog.load(path)
        observation = {feature: 0.0 for feature in RUNTIME_INPUTS}
        observation["official_side_score_only"] = "H"
        with self.assertRaises(TwinContractError):
            catalog.validate_observation(observation)

    @unittest.skipUnless(BUNDLE and OHLCV, "set private bundle and public OHLCV paths")
    def test_public_ohlcv_is_observation_only(self) -> None:
        result = audit_bundle(Path(BUNDLE), ohlcv_dir=Path(OHLCV))
        audit = result["ohlcv_observation_audit"]
        self.assertEqual(audit["status"], "PASS_OBSERVATION_ONLY")
        self.assertEqual(audit["rows"], 660_853)
        self.assertFalse(audit["official_exact_replay_source"])

    def test_audit_result_is_json_serializable(self) -> None:
        self.assertEqual(json.dumps({"inputs": len(RUNTIME_INPUTS)}), '{"inputs": 64}')


if __name__ == "__main__":
    unittest.main()
