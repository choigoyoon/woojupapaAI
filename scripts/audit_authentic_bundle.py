"""CLI for read-only validation of an authentic TWIN transfer bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from twin_core.authentic_audit import audit_bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle_root", type=Path)
    parser.add_argument(
        "--causal-market",
        action="append",
        default=[],
        help="candidate causal_market_rows_v1.parquet path; may be repeated",
    )
    parser.add_argument(
        "--ohlcv-dir",
        type=Path,
        help="public observation-only OHLCV directory containing manifest.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_bundle(
        args.bundle_root,
        causal_market_candidates=args.causal_market,
        ohlcv_dir=args.ohlcv_dir,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 1 if result["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
