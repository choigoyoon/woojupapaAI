"""Immutable facts recovered from the authenticated 2,775-rule bundle.

The numeric values here are integrity locks or already-learned values from the
provided artifact.  This module never invents a market threshold.
"""

from __future__ import annotations

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

OBSERVATION_ONLY_STAGES = STAGE_ORDER[:-1]
NOW_ALLOWED_STAGES = ("11",)
BEHAVIORS = ("ALIGNED", "MIXED", "OPPOSED")
RAW_CONTEXT_SIGNATURES = ("+++", "++-", "+-+", "+--", "-++", "-+-", "--+", "---")
TIMEFRAMES = ("5m", "15m", "30m", "1h", "2h", "4h", "1d", "1w")
PARALLEL_METHODS = (
    "WAVE_REARM_AGE",
    "MOMENTUM_SPEED",
    "CANDLE_REVERSAL",
    "VOLUME_RANGE",
    "MACRO_TREND",
    "MID_MACD_TREND",
)
SAFETY_METHOD = "ZC_FALLBACK"

RUNTIME_INPUTS = (
    "side_code",
    "wave_age_log",
    "time_ratio_prev1",
    "time_ratio_prev2",
    "move_ratio_prev1",
    "move_ratio_prev2",
    "wave_move_pct_so_far",
    "zone_move_pct_so_far",
    "speed_recent3",
    "speed_ratio_3_to_12",
    "candidate_age_log",
    "rearm_count_log",
    "rearm_gap_bars_log",
    "rearm_extension_pct",
    "rearm_extension_ratio",
    "new_extreme_now",
    "candidate_rejection_pct",
    "candidate_rejection_fraction",
    "candidate_rejection_speed",
    "candidate_rejection_range_units",
    "candle_direction_for_side",
    "candle_body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "close_position_for_side",
    "volume_ratio20",
    "volume_ratio72",
    "range_ratio20",
    "macd_5m_hist_for_side",
    "macd_5m_delta_for_side",
    "macd_5m_sign_for_side",
    "macd_5m_zc_count_so_far",
    "macd_5m_zc_age_log",
    "macd_15m_hist_for_side",
    "macd_15m_delta_for_side",
    "macd_15m_sign_for_side",
    "macd_15m_zc_count_so_far",
    "macd_15m_zc_age_log",
    "macd_30m_hist_for_side",
    "macd_30m_delta_for_side",
    "macd_30m_sign_for_side",
    "macd_30m_zc_count_so_far",
    "macd_30m_zc_age_log",
    "macd_1h_hist_for_side",
    "macd_1h_delta_for_side",
    "macd_1h_sign_for_side",
    "macd_1h_zc_count_so_far",
    "macd_1h_zc_age_log",
    "macd_2h_hist_for_side",
    "macd_2h_delta_for_side",
    "macd_2h_sign_for_side",
    "macd_2h_zc_age_log",
    "macd_4h_hist_for_side",
    "macd_4h_delta_for_side",
    "macd_4h_sign_for_side",
    "macd_4h_zc_age_log",
    "macd_1d_hist_for_side",
    "macd_1d_delta_for_side",
    "macd_1d_sign_for_side",
    "macd_1d_zc_age_log",
    "macd_1w_hist_for_side",
    "macd_1w_delta_for_side",
    "macd_1w_sign_for_side",
    "macd_1w_zc_age_log",
)

EXPECTED = {
    "runtime_inputs": 64,
    "official_events": 4_136,
    "official_h": 2_068,
    "official_l": 2_068,
    "completed_transitions": 4_135,
    "prefix_observations": 656_136,
    "candidate_attempts": 40_133,
    "selected_signatures": 2_775,
    "selected_context_rules": 2_499,
    "selected_base_bins": 276,
    "rule_now_events": 4_000,
    "fallback_now_events": 136,
}

AUTHENTIC_SHA256 = {
    "augmented_rules": "88c7502e3899c7f704388135c4465b016af68089b99500e079309e1e7d0b47a6",
    "base_rules": "59ebed1a14cd39c0e02b6ade0866d7216750428ae46ad5d00f0630f20b62f9dd",
    "official_event_ledger": "7e4b012609535812d5f64654b744bf3d234f231565d8fd5be8b937c66133ac89",
    "stage_blueprint": "e253c0bc8e30b70c519d89a963e6484ddcccfc2d296b6f39d427ccdd216f28cb",
    "output_manifest": "1192f128e671e7a5efd982c3361e4f0c02482a7f797bf9df10e8d5c44950a4ae",
    # Required for exact 4,136-event replay. This 107 MB file is named in the
    # source manifest but deliberately absent from the small transfer bundle.
    "causal_market_rows": "17fdd7567236e22fa1374aeef93cf83a54e3766db2694d0f4650647b0493ea05",
}

FORBIDDEN_RUNTIME_FIELDS = frozenset(
    {
        "outcome_study_only",
        "survived_study_only",
        "future_answer_columns",
        "current_event_future",
        "official_action_position",
        "event_boundary_lookahead",
        "event_position_study_only",
        "official_rule_ready",
        "official_wait_state",
        "official_event_sequence_study_only",
        "official_side_score_only",
        "official_event_raw_index_score_only",
        "incoming_transition_sequence_score_only",
        "incoming_direction_score_only",
        "incoming_path_row_n_score_only",
        "coverage_status_study_only",
        "layer_keys_study_only_json",
    }
)

assert len(RUNTIME_INPUTS) == EXPECTED["runtime_inputs"]
