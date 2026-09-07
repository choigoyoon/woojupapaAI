# Reconstructed Stage 03-11 intermediate ledger

This directory contains a formula-driven reconstruction made from the available
closed-bar OHLCV and the existing executable thought program.

- `professor_16_modules_intermediate_ledger_reconstructed_v1.parquet` connects
  all Stage 03-11 observations, relations, persistence states and handoffs for
  656,136 closed-bar rows.
- `stage11_event_actions_reconstructed_v1.parquet` contains one final action row
  for each of the 4,136 historical events.
- `manifest.json` records provenance, counts, hashes and the explicit
  reconstruction status.

The historical 2,775 selected outputs are not consulted while calculating this
ledger. They remain audit evidence, not a runtime whitelist. The files also do
not claim to match the unavailable hashes of the original intermediate ledgers.

The ledger now records the signal/environment distinction explicitly.  A BASE
interval match is a formula claim inside the current macro background, not an
entry by itself.  Every one of the 5,500 CONTEXT relations was verified to
contain at least one condition owned by its own trading method, so CONTEXT is a
contextualized version of that method's signal (signal formula plus relative
environment), not an independent environment-only trigger.  Stage 10 records
`EXCLUDE`, `WAIT`, `KEEP`, or `NOT_APPLICABLE` for every BASE and CONTEXT path;
only `KEEP` is eligible to reach Stage 11.

These verdict names do not add a formula, threshold, score, or method.  They
make the existing support/probability and persistence blockers observable:
support/probability failure is `EXCLUDE`, unfinished persistence is `WAIT`, and
all blockers clear is `KEEP`.  The six methods retain their individual support,
counterargument, and unresolved paths, while the existing official winner path
continues to control the final action.

The executable JSON also preserves 3,972 historical Stage 10-to-11 release
batons.  They contain the six methods' READY or WAIT reasons and are likewise
validation evidence only.  Reversing those batons corrected two observation
ownership errors: `candidate_rejection_*` belongs to `WAVE_REARM_AGE`, and
`macd_5m_*` belongs to `MOMENTUM_SPEED`.

`manifest.json` now compares every reconstructed aggregate with the source
thought audit.  The six-method parallel path is close (540,176 reconstructed
ready rows versus 543,628 source rows), but the official single-winner path is
not exact (50,702 versus 71,157).  Consequently the current no-whitelist final
result is 3,430 RULE_NOW and 706 fallback events, rather than the source
4,000/136.  The manifest deliberately records `MISMATCH`.

Exact row-level recovery requires the source files
`module_09_rule_recognition_patterns.parquet`,
`module_10_wait_reason_patterns.parquet`, and
`module_11_final_action_patterns.parquet`.  Their names and hashes are recorded
in the executable JSON, but their contents were not included in the supplied
ZIP.

The reconstruction can be repeated with
`export_reconstructed_intermediate_ledger()` in `rule_router.py`.
