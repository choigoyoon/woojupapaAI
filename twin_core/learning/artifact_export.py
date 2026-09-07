"""Export small audit JSON only; private teaching artifacts stay external."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from twin_core import TwinContractError


def export_audit(name: str, payload: Any, output_dir: str | Path) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise TwinContractError(f"unsafe audit artifact name: {name}")
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{name}.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target
