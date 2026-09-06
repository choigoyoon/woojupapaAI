#!/usr/bin/env python3
"""Download normalized BTCUSDT 5-minute spot OHLCV from Binance Vision.

The script writes one gzip-compressed CSV per UTC year plus a manifest with
row counts, timestamp bounds, archive availability, gaps, duplicates, and
SHA-256 hashes. It intentionally preserves exchange observations and does not
invent candles for missing intervals.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import sys
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


BASE_URL = "https://data.binance.vision/data/spot"
INTERVAL_MS = 5 * 60 * 1000
HEADER = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "source",
]


@dataclass
class YearStats:
    year: int
    expected_rows: int = 0
    rows: int = 0
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    duplicate_timestamps: int = 0
    gap_intervals: int = 0
    non_5m_intervals: int = 0
    missing_expected_rows: int = 0
    sha256: str | None = None
    file: str | None = None


def parse_args() -> argparse.Namespace:
    yesterday = date.today() - timedelta(days=1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2020, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=yesterday)
    parser.add_argument("--output", type=Path, default=Path("twin_ohlcv/data"))
    return parser.parse_args()


def month_starts(start: date, end: date):
    current = start.replace(day=1)
    while current <= end:
        yield current
        current = (current.replace(day=28) + timedelta(days=4)).replace(day=1)


def days_in_month(month: date, start: date, end: date):
    next_month = (month.replace(day=28) + timedelta(days=4)).replace(day=1)
    current = max(start, month)
    stop = min(end, next_month - timedelta(days=1))
    while current <= stop:
        yield current
        current += timedelta(days=1)


def fetch(url: str) -> bytes | None:
    request = urllib.request.Request(url, headers={"User-Agent": "twin-ohlcv-builder/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def verified_archive(url: str) -> tuple[bytes | None, str]:
    payload = fetch(url)
    if payload is None:
        return None, "missing"

    checksum_payload = fetch(f"{url}.CHECKSUM")
    if checksum_payload:
        expected = checksum_payload.decode("utf-8").strip().split()[0].lower()
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch for {url}: {actual} != {expected}")
        return payload, "verified"

    return payload, "downloaded_without_checksum"


def archive_rows(payload: bytes):
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise RuntimeError(f"Expected one CSV member, found {members}")
        with archive.open(members[0]) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
            for row in csv.reader(text):
                if not row or not row[0].isdigit():
                    continue
                if len(row) < 11:
                    raise RuntimeError(f"Short Binance kline row: {row}")
                yield row


def normalize_epoch_ms(raw: str) -> int:
    value = int(raw)
    if value >= 10**15:
        return value // 1000
    return value


def iso_utc(epoch_ms: int) -> str:
    return datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_rows_for_year(year: int, start: date, end: date) -> int:
    year_start = max(start, date(year, 1, 1))
    year_end = min(end, date(year, 12, 31))
    if year_start > year_end:
        return 0
    return ((year_end - year_start).days + 1) * (24 * 60 // 5)


def archive_url(symbol: str, interval: str, cadence: str, stamp: str) -> str:
    return f"{BASE_URL}/{cadence}/klines/{symbol}/{interval}/{symbol}-{interval}-{stamp}.zip"


def main() -> int:
    args = parse_args()
    if args.start > args.end:
        raise SystemExit("--start must not be later than --end")

    args.output.mkdir(parents=True, exist_ok=True)
    writers: dict[int, tuple[gzip.GzipFile, io.TextIOWrapper, csv.writer]] = {}
    stats: dict[int, YearStats] = {}
    last_seen: dict[int, int] = {}
    archives: list[dict[str, str]] = []
    seen_global: set[int] = set()

    def writer_for(year: int):
        if year not in writers:
            path = args.output / f"{args.symbol}_{args.interval}_{year}.csv.gz"
            binary = gzip.GzipFile(filename=str(path), mode="wb", mtime=0)
            text = io.TextIOWrapper(binary, encoding="utf-8", newline="")
            writer = csv.writer(text, lineterminator="\n")
            writer.writerow(HEADER)
            writers[year] = (binary, text, writer)
            stats[year] = YearStats(year=year, file=path.name)
        return writers[year][2]

    try:
        for month in month_starts(args.start, args.end):
            month_stamp = month.strftime("%Y-%m")
            url = archive_url(args.symbol, args.interval, "monthly", month_stamp)
            payload, status = verified_archive(url)
            archive_payloads: list[tuple[str, bytes]] = []

            if payload is not None:
                archives.append({"archive": month_stamp, "cadence": "monthly", "status": status, "url": url})
                archive_payloads.append((url, payload))
            else:
                archives.append({"archive": month_stamp, "cadence": "monthly", "status": "missing", "url": url})
                for day in days_in_month(month, args.start, args.end):
                    day_stamp = day.isoformat()
                    day_url = archive_url(args.symbol, args.interval, "daily", day_stamp)
                    day_payload, day_status = verified_archive(day_url)
                    archives.append({"archive": day_stamp, "cadence": "daily", "status": day_status, "url": day_url})
                    if day_payload is not None:
                        archive_payloads.append((day_url, day_payload))

            for source_url, archive in archive_payloads:
                for row in archive_rows(archive):
                    epoch_ms = normalize_epoch_ms(row[0])
                    candle_day = datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc).date()
                    if candle_day < args.start or candle_day > args.end:
                        continue

                    year = candle_day.year
                    year_stats = stats.setdefault(year, YearStats(year=year))
                    if epoch_ms in seen_global:
                        year_stats.duplicate_timestamps += 1
                        continue
                    seen_global.add(epoch_ms)

                    prior = last_seen.get(year)
                    if prior is not None:
                        delta = epoch_ms - prior
                        if delta != INTERVAL_MS:
                            year_stats.non_5m_intervals += 1
                            if delta > INTERVAL_MS:
                                year_stats.gap_intervals += max(0, delta // INTERVAL_MS - 1)
                    last_seen[year] = epoch_ms

                    timestamp = iso_utc(epoch_ms)
                    if year_stats.first_timestamp is None:
                        year_stats.first_timestamp = timestamp
                    year_stats.last_timestamp = timestamp
                    year_stats.rows += 1

                    writer_for(year).writerow([
                        timestamp,
                        row[1],
                        row[2],
                        row[3],
                        row[4],
                        row[5],
                        row[7],
                        row[8],
                        row[9],
                        row[10],
                        "binance_spot",
                    ])
    finally:
        for binary, text, _ in writers.values():
            text.flush()
            text.detach()
            binary.close()

    ordered_stats = []
    for year in range(args.start.year, args.end.year + 1):
        item = stats.setdefault(year, YearStats(year=year))
        item.expected_rows = expected_rows_for_year(year, args.start, args.end)
        item.missing_expected_rows = max(0, item.expected_rows - item.rows)
        if item.file:
            item.sha256 = sha256_file(args.output / item.file)
        ordered_stats.append(asdict(item))

    manifest = {
        "schema": "twin_ohlcv_manifest_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": "Binance Vision spot klines",
        "symbol": args.symbol,
        "interval": args.interval,
        "requested_start": args.start.isoformat(),
        "requested_end": args.end.isoformat(),
        "columns": HEADER,
        "expected_rows": sum(item["expected_rows"] for item in ordered_stats),
        "total_rows": sum(item["rows"] for item in ordered_stats),
        "total_missing_expected_rows": sum(item["missing_expected_rows"] for item in ordered_stats),
        "total_duplicate_timestamps": sum(item["duplicate_timestamps"] for item in ordered_stats),
        "total_gap_intervals": sum(item["gap_intervals"] for item in ordered_stats),
        "total_non_5m_intervals": sum(item["non_5m_intervals"] for item in ordered_stats),
        "years": ordered_stats,
        "archives": archives,
        "notes": [
            "All timestamps are normalized to UTC ISO-8601.",
            "Missing intervals are reported and never fabricated.",
            "This dataset is observation input only; it contains no L/H labels or future-derived fields.",
        ],
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "manifest": str(manifest_path),
        "total_rows": manifest["total_rows"],
        "gaps": manifest["total_gap_intervals"],
        "duplicates": manifest["total_duplicate_timestamps"],
        "years": len(ordered_stats),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

