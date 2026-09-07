# TWIN canonical OHLCV data

This branch stores the reproducible BTCUSDT 5-minute observation set used for
TWIN module inspection. The exact inclusive UTC range is
`2020-02-14T09:00:00Z` through `2026-05-28T00:00:00Z` (660,853 rows).

Binance Spot BTCUSDT monthly klines are the primary source. Missing Binance
timestamps are repaired only from Bitstamp BTCUSD 5-minute candles, matching
the previously approved TWIN source lineage. Output is split into one
gzip-compressed CSV per UTC year.

The generated files contain observations only. They do not contain official
L/H labels, future candles, post-trade profit, MAE, or any other answer field.

## Build

```bash
pip install pandas
python twin_ohlcv/build_canonical_5m.py
```

`manifest.json` records the requested period, row counts, timestamp bounds,
repaired count, duplicates, interval checks, archive URLs, and SHA-256 hashes
for every source archive and generated file. The build fails if any timestamp
remains missing or if the total is not exactly 660,853 rows.

## Columns

```text
timestamp
open
high
low
close
volume
fill_source
```

`fill_source` is either `binance_btcusdt_5m` or `bitstamp_btcusd_5m`.

## Intended inspection flow

```text
public OHLCV observations
-> structural and observation-only checks
-> authentic Stage 03-11 modules
-> candidate axis: HOLD / REARM
-> trade-action axis: WAIT / NOW
-> post-hoc comparison and correction
```

This table is not automatically interchangeable with the private
`causal_market_rows_v1.parquet` locked by the authentic transfer manifest.
Exact 4,136-event parity must remain blocked until that source hash and the
656,136-row causal feature build are available.

