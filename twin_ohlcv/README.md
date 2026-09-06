# TWIN OHLCV data

This branch stores reproducible BTCUSDT 5-minute OHLCV observations for TWIN
module inspection. The data is downloaded from Binance Vision and split into
one gzip-compressed CSV per UTC year.

The generated files contain observations only. They do not contain official
L/H labels, future candles, post-trade profit, MAE, or any other answer field.

## Build

```bash
python twin_ohlcv/fetch_binance_5m.py \
  --symbol BTCUSDT \
  --interval 5m \
  --start 2020-01-01 \
  --output twin_ohlcv/data
```

`manifest.json` records the requested period, row counts, timestamp bounds,
missing intervals, duplicates, archive URLs, and SHA-256 hashes for every
generated file. Missing candles are reported rather than synthesized.

## Columns

```text
timestamp
open
high
low
close
volume
quote_volume
trades
taker_buy_base_volume
taker_buy_quote_volume
source
```

## Intended inspection flow

```text
OHLCV observations
-> 16 observation modules
-> fixed action ledger used only during training
-> nine-stage relative reasoning
-> WAIT / NOW / HOLD / REARM
-> post-hoc comparison and correction
```

