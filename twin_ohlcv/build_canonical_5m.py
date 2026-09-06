#!/usr/bin/env python3
"""Build the canonical TWIN BTCUSDT 5-minute OHLCV observation set.

Binance Spot is the primary source. Missing Binance timestamps are repaired
from Bitstamp BTCUSD 5-minute candles, matching the previously approved TWIN
source lineage. The resulting data contains observations only.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


START = pd.Timestamp("2020-02-14T09:00:00Z")
END = pd.Timestamp("2026-05-28T00:00:00Z")
EXPECTED_ROWS = 660_853
BINANCE = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/5m"
BITSTAMP = "https://www.bitstamp.net/api/v2/ohlc/btcusd/"
BINANCE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_base",
    "taker_quote",
    "ignore",
]
OUTPUT_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "fill_source"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def months():
    current = pd.Timestamp(START.year, START.month, 1, tz="UTC")
    stop = pd.Timestamp(END.year, END.month, 1, tz="UTC")
    while current <= stop:
        yield current.year, current.month
        current += pd.offsets.MonthBegin(1)


def normalize_epoch(value) -> int:
    number = int(value)
    return number // 1000 if number > 10**14 else number


def download(url: str, destination: Path) -> None:
    if destination.exists():
        return
    request = urllib.request.Request(url, headers={"User-Agent": "twin-ohlcv-builder/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as output:
        for block in iter(lambda: response.read(1024 * 1024), b""):
            output.write(block)


def verify_archive(url: str, path: Path) -> str:
    checksum_url = f"{url}.CHECKSUM"
    request = urllib.request.Request(checksum_url, headers={"User-Agent": "twin-ohlcv-builder/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        expected = response.read().decode("utf-8").strip().split()[0].lower()
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"ARCHIVE_CHECKSUM_FAIL {path.name} {actual} != {expected}")
    return actual


def read_binance(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise RuntimeError(f"ZIP_MEMBER_FAIL {path.name}: {members}")
        with archive.open(members[0]) as source:
            frame = pd.read_csv(source, header=None, names=BINANCE_COLUMNS)
    parsed_open_time = pd.to_numeric(frame["open_time"], errors="coerce")
    invalid = parsed_open_time.isna()
    if invalid.any():
        invalid_values = frame.loc[invalid, "open_time"].astype(str).str.strip().str.lower()
        if len(invalid_values) != 1 or invalid_values.iloc[0] not in {"open_time", "open time"}:
            raise RuntimeError(f"BINANCE_TIMESTAMP_FAIL {path.name}: {invalid_values.tolist()}")
        frame = frame.loc[~invalid].copy()
        parsed_open_time = parsed_open_time.loc[~invalid]
    frame["open_time"] = parsed_open_time.astype("int64").map(normalize_epoch)
    frame["timestamp"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["fill_source"] = "binance_btcusdt_5m"
    return frame[OUTPUT_COLUMNS]


def contiguous_runs(timestamps: list[pd.Timestamp]):
    if not timestamps:
        return []
    ordered = sorted(timestamps)
    runs = []
    start = previous = ordered[0]
    for current in ordered[1:]:
        if current - previous == pd.Timedelta(minutes=5):
            previous = current
            continue
        runs.append((start, previous))
        start = previous = current
    runs.append((start, previous))
    return runs


def read_bitstamp_run(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    query = urllib.parse.urlencode(
        {
            "step": 300,
            "limit": 1000,
            "start": int((start - pd.Timedelta(minutes=5)).timestamp()),
            "end": int((end + pd.Timedelta(minutes=5)).timestamp()),
        }
    )
    request = urllib.request.Request(
        f"{BITSTAMP}?{query}",
        headers={"User-Agent": "twin-ohlcv-builder/1.0", "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    frame = pd.DataFrame(payload["data"]["ohlc"])
    frame["timestamp"] = pd.to_datetime(pd.to_numeric(frame["timestamp"]), unit="s", utc=True)
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame[["timestamp", "open", "high", "low", "close", "volume"]]


def write_deterministic_gzip(frame: pd.DataFrame, path: Path) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.writer(text, lineterminator="\n")
                writer.writerow(OUTPUT_COLUMNS)
                for row in frame[OUTPUT_COLUMNS].itertuples(index=False, name=None):
                    values = list(row)
                    values[0] = pd.Timestamp(values[0]).isoformat().replace("+00:00", "Z")
                    writer.writerow(values)


def main() -> int:
    output = Path("twin_ohlcv/data")
    raw = Path("work/twin_ohlcv_monthly")
    output.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)

    frames = []
    archives = []
    for year, month in months():
        name = f"BTCUSDT-5m-{year:04d}-{month:02d}.zip"
        url = f"{BINANCE}/{name}"
        path = raw / name
        download(url, path)
        archive_hash = verify_archive(url, path)
        frame = read_binance(path)
        frames.append(frame)
        archives.append({"file": name, "rows": len(frame), "sha256": archive_hash, "url": url})

    binance = pd.concat(frames, ignore_index=True)
    binance = (
        binance[(binance.timestamp >= START) & (binance.timestamp <= END)]
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )
    expected = pd.date_range(START, END, freq="5min", tz="UTC")
    missing = expected.difference(pd.DatetimeIndex(binance.timestamp))

    repairs = []
    unresolved = []
    for run_start, run_end in contiguous_runs(list(missing)):
        candidate = read_bitstamp_run(run_start, run_end).set_index("timestamp")
        for timestamp in pd.date_range(run_start, run_end, freq="5min", tz="UTC"):
            if timestamp not in candidate.index:
                unresolved.append(timestamp)
                continue
            row = candidate.loc[timestamp]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            repairs.append(
                {
                    "timestamp": timestamp,
                    "open": float(row.open),
                    "high": float(row.high),
                    "low": float(row.low),
                    "close": float(row.close),
                    "volume": float(row.volume),
                    "fill_source": "bitstamp_btcusd_5m",
                }
            )

    if unresolved:
        raise RuntimeError(f"BITSTAMP_UNRESOLVED {len(unresolved)}")

    repair_frame = pd.DataFrame(repairs, columns=OUTPUT_COLUMNS)
    full = (
        pd.concat([binance, repair_frame], ignore_index=True)
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )
    full_index = pd.DatetimeIndex(full.timestamp)
    missing_after = expected.difference(full_index)
    extra_after = full_index.difference(expected)
    if len(full) != EXPECTED_ROWS or len(missing_after) or len(extra_after):
        raise RuntimeError(
            f"CONTINUITY_FAIL rows={len(full)} missing={len(missing_after)} extra={len(extra_after)}"
        )
    if full.timestamp.duplicated().any():
        raise RuntimeError("DUPLICATE_TIMESTAMP_FAIL")
    if not full.timestamp.diff().dropna().eq(pd.Timedelta(minutes=5)).all():
        raise RuntimeError("INTERVAL_FAIL")

    bad_ohlc = (
        (full.high < full[["open", "close", "low"]].max(axis=1))
        | (full.low > full[["open", "close", "high"]].min(axis=1))
        | (full.volume < 0)
    )
    if bad_ohlc.any():
        raise RuntimeError(f"OHLC_FAIL {int(bad_ohlc.sum())}")

    files = []
    for year, yearly in full.groupby(full.timestamp.dt.year, sort=True):
        path = output / f"BTCUSDT_5m_{year}.csv.gz"
        write_deterministic_gzip(yearly, path)
        files.append(
            {
                "file": path.name,
                "year": int(year),
                "rows": len(yearly),
                "first": yearly.timestamp.iloc[0].isoformat(),
                "last": yearly.timestamp.iloc[-1].isoformat(),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        )

    repaired = full[full.fill_source.eq("bitstamp_btcusd_5m")].copy()
    repaired["hour"] = repaired.timestamp.dt.floor("h")
    manifest = {
        "schema": "twin_canonical_ohlcv_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "symbol": "BTCUSDT",
        "interval": "5m",
        "start": START.isoformat(),
        "end": END.isoformat(),
        "expected_rows": EXPECTED_ROWS,
        "rows": len(full),
        "primary_source": "Binance Spot BTCUSDT monthly klines",
        "repair_source": "Bitstamp BTCUSD 5-minute OHLCV",
        "original_binance_rows": len(binance),
        "original_missing": len(missing),
        "bitstamp_repaired": len(repair_frame),
        "repaired_hours": int(repaired.hour.nunique()),
        "unresolved": 0,
        "duplicates": int(full.timestamp.duplicated().sum()),
        "non_5m_intervals": int((~full.timestamp.diff().dropna().eq(pd.Timedelta(minutes=5))).sum()),
        "columns": OUTPUT_COLUMNS,
        "files": files,
        "archives": archives,
        "runtime_forbidden_columns": [
            "official_lh",
            "future_candle",
            "posthoc_pnl",
            "posthoc_mae",
            "fixed_action_label",
        ],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ["rows", "original_missing", "bitstamp_repaired", "unresolved", "files"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

