"""The 16 authentic TWIN modules and their Stage 03-11 ownership."""

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

STAGE_OWNERS = {
    "03": "candidate_revision",
    "04": "context_partition",
    "05": "time_distribution",
    "06": "candle_volume_distribution",
    "07": "multitf_zc_distribution",
    "08": "flow_order_distribution",
    "09": "rule_induction",
    "10": "repair_queue",
    "11": "threshold_frontier",
}

assert len(CANONICAL_MODULES) == 16

__all__ = ["CANONICAL_MODULES", "STAGE_OWNERS"]
