"""Load learned distributions; this corrected path does not relearn 16 thresholds."""

from __future__ import annotations

from pathlib import Path

from twin_core.learning.rule_induction import RuleCatalog


def load_authentic_distributions(augmented_rules_path: str | Path) -> RuleCatalog:
    return RuleCatalog.load(augmented_rules_path, verify_hash=True)
