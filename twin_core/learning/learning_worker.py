"""Single locked entry point for OHLCV audit and full learning."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path

import pandas as pd

from twin_core import TwinContractError
from twin_core.learning.distribution_learner import run_learning_pipeline


def _load_ohlcv(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    files = sorted(source.glob("*.csv.gz")) if source.is_dir() else [source]
    if not files:
        raise TwinContractError(f"learning_worker: no OHLCV files at {source}")
    frames = [pd.read_csv(file) for file in files]
    result = pd.concat(frames, ignore_index=True)
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    return result.sort_values("timestamp").reset_index(drop=True)


def _load_ledger(path: str | Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    source = Path(path)
    if source.suffix.lower() == ".json":
        return pd.read_json(source)
    return pd.read_csv(source)


@contextmanager
def _exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise TwinContractError(f"learning_worker: another run owns {path}") from exc
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        yield
    finally:
        os.close(descriptor)
        path.unlink(missing_ok=True)


def run_learning_worker(
    ohlcv_path: str | Path,
    *,
    ledger_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    audit_only: bool = False,
) -> dict[str, object]:
    output = Path(output_dir or "twin_learning_output").resolve()
    with _exclusive_lock(output.parent / f".{output.name}.lock"):
        ohlcv = _load_ohlcv(ohlcv_path)
        ledger = _load_ledger(ledger_path)
        return run_learning_pipeline(
            ohlcv,
            ledger,
            output_dir=output,
            audit_only=audit_only,
        )


def _summary(result: dict[str, object]) -> dict[str, object]:
    return {
        "mode": result["mode"],
        "audit": result.get("audit"),
        "ledger_contract": result.get("ledger_contract"),
        "score": result.get("score"),
        "mdd": result.get("mdd"),
        "status": result.get("status"),
        "manifest": result.get("manifest"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ohlcv", required=True)
    parser.add_argument("--ledger")
    parser.add_argument("--output", default="twin_learning_output")
    parser.add_argument("--audit-only", action="store_true")
    arguments = parser.parse_args()
    result = run_learning_worker(
        arguments.ohlcv,
        ledger_path=arguments.ledger,
        output_dir=arguments.output,
        audit_only=arguments.audit_only,
    )
    print(json.dumps(_summary(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
