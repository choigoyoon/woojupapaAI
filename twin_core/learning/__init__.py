"""The sixteen canonical TWIN learning modules."""

CANONICAL_MODULES = (
    "learning_worker",
    "distribution_learner",
    "candidate_revision",
    "context_partition",
    "time_distribution",
    "candle_volume_distribution",
    "multitf_zc_distribution",
    "flow_order_distribution",
    "rule_induction",
    "repair_queue",
    "threshold_frontier",
    "scorebook",
    "replay_builder",
    "mdd_problem",
    "learning_status",
    "artifact_export",
)

__all__ = ["CANONICAL_MODULES"]
