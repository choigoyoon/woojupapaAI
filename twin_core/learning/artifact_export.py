"""Validated, atomic export. No other learning module writes artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Mapping

import numpy as np
import pandas as pd

from twin_core import TwinContractError


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if value is pd.NaT:
        return None
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _safe_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise TwinContractError(f"artifact_export: unsafe artifact name {name!r}")
    return name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_learning_artifacts(
    artifacts: Mapping[str, object],
    output_dir: str | Path,
) -> dict[str, object]:
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest_entries: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix=".twin-stage-", dir=destination) as temporary:
        stage = Path(temporary)
        staged_files: list[tuple[Path, Path]] = []
        for raw_name, value in artifacts.items():
            name = _safe_name(raw_name)
            if isinstance(value, pd.DataFrame):
                filename = f"{name}.csv.gz"
                staged = stage / filename
                value.to_csv(
                    staged,
                    index=False,
                    compression={"method": "gzip", "mtime": 0},
                )
                rows = int(len(value))
            else:
                filename = f"{name}.json"
                staged = stage / filename
                staged.write_text(
                    json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n",
                    encoding="utf-8",
                )
                rows = None
            entry = {
                "file": filename,
                "bytes": staged.stat().st_size,
                "sha256": _sha256(staged),
            }
            if rows is not None:
                entry["rows"] = rows
            manifest_entries.append(entry)
            staged_files.append((staged, destination / filename))

        manifest = {"format": 1, "artifacts": manifest_entries}
        manifest_stage = stage / "manifest.json"
        manifest_stage.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        for source, target in staged_files:
            os.replace(source, target)
        os.replace(manifest_stage, destination / "manifest.json")
    return manifest
