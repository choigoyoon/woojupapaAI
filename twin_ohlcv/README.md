# TWIN OHLCV data

This branch stores reproducible BTCUSDT 5-minute OHLCV observations for TWIN
module inspection. The canonical TWIN window is fixed to
`2020-02-14T09:00:00Z` through `2026-05-28T00:00:00Z`, inclusive, with
`660,853` expected five-minute timestamps.

The generated files contain observations only. They do not contain official
L/H labels, future candles, post-trade profit, MAE, or any other answer field.

The build keeps Binance Spot observations and records every missing timestamp.
Missing timestamps must be repaired from the previously approved Bitstamp
BTCUSD five-minute lineage before the data can be marked canonical.

## Intended inspection flow

```text
OHLCV observations
-> 16 observation modules
-> fixed action ledger used only during training
-> nine-stage relative reasoning
-> WAIT / NOW / HOLD / REARM
-> post-hoc comparison and correction
```
