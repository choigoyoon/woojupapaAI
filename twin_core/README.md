# TWIN relative-action learning core

This rebuild makes one separation explicit:

- **Fixed values are actions:** `WAIT`, `NOW`, `HOLD`, `REARM`.
- **Relative values are evidence:** a selected anchor, comparisons with the prior
  wave/candidate, time and price progress, candle/volume change, divergence, and
  causal eight-timeframe MACD relationships.

There is no hand-written market threshold such as `RSI <= 30`.  Offline fixed
action outcomes teach each numeric boundary.  Moving the historical values moves
the learned boundary.

## Nine-stage reasoning

1. `POSITION_STATE` — read the current position first.
2. `TARGET_DIRECTION` — identify the intended LONG/SHORT direction.
3. `ANCHOR_SELECTION` — select the previous wave, current candidate, and previous
   candidate anchors.
4. `RELATIVE_MEASUREMENT` — measure relative time, move, speed, and rebound.
5. `SEQUENCE_DIVERGENCE` — compare candidate, candle, volume, and 8-TF sequences.
6. `HISTORICAL_RETRIEVAL` — retrieve applicable learned relations.
7. `COUNTEREXAMPLE_CHECK` — compare support with contradicting historical cases.
8. `FIXED_ACTION` — emit `WAIT`, `NOW`, `HOLD`, or `REARM`.
9. `POSITION_TRANSITION` — keep, open, or switch at the next actual open.

`REARM` is also retained as an event flag, so a new extreme and `NOW` may occur
on the same closed candle.  A same-direction `NOW` intent becomes `HOLD`.
Correlated features do not receive duplicate votes: the strongest historical
relationship from each of the six evidence families contributes a likelihood
ratio, starting from that context's observed `NOW`/`WAIT` base rate.
Each numeric rule begins at the selective edge and expands only to the first
historical `WAIT` counterexample; that observed entry point becomes its boundary.

## Sixteen modules

| Module | One responsibility |
|---|---|
| `learning_worker` | single locked entry point |
| `distribution_learner` | orchestration only |
| `candidate_revision` | official event journeys and fixed action labels |
| `context_partition` | aligned/mixed/opposed context, with no discarded cases |
| `time_distribution` | relative time, movement, and speed |
| `candle_volume_distribution` | causal candle and volume relationships |
| `multitf_zc_distribution` | forming-candle-safe 8-TF MACD/ZC relationships |
| `flow_order_distribution` | creation, replacement, survival, and divergence |
| `rule_induction` | data-derived relative boundaries |
| `repair_queue` | missed, premature, conflicting, and unseen cases |
| `threshold_frontier` | the nine-stage decision trace |
| `scorebook` | outcome and future-leakage audit |
| `replay_builder` | prefix-only replay and next-open execution |
| `mdd_problem` | posthoc return and drawdown diagnostics |
| `learning_status` | stage receipts and timing |
| `artifact_export` | validated atomic artifact export |

## Canonical-input rule

Full learning requires the official alternating 4,136-event ledger: 2,068 L and
2,068 H.  The ledger is an offline teaching answer and is never a runtime rule
input.  Without it the worker performs an audit only and refuses to describe the
result as canonical training.

```powershell
python -m twin_core.learning.learning_worker `
  --ohlcv twin_ohlcv/data `
  --audit-only `
  --output twin_core/audit/current
```

For full learning, add `--ledger path/to/official_4136.csv`.  Required columns are
`side,pivot_time,pivot_price`; an optional exact `pivot_idx` is accepted.

Run the contract tests with:

```powershell
python -m unittest discover -s tests -p "test_twin_*.py" -v
```

