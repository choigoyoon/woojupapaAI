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

The reconstruction can be repeated with
`export_reconstructed_intermediate_ledger()` in `rule_router.py`.
