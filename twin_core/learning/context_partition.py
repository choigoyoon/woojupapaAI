"""Stage 04: preserve one of eight exact macro sign signatures."""

from __future__ import annotations

from typing import Any, Mapping

from twin_core import TwinContractError


def _sign(value: Any) -> str:
    number = float(value)
    if number > 0:
        return "+"
    if number < 0:
        return "-"
    raise TwinContractError("Stage 04 received a zero macro sign")


def evaluate_context(observation: Mapping[str, Any]) -> dict[str, Any]:
    signs = [_sign(observation[f"macd_{timeframe}_sign_for_side"]) for timeframe in ("4h", "1d", "1w")]
    raw = "".join(signs)
    background = "ALIGNED" if raw == "+++" else "OPPOSED" if raw == "---" else "MIXED"
    return {
        "stage": "04",
        "context_signature_raw": raw,
        "context_signature_for_side": f"{int(float(observation['side_code']))}|{raw}",
        "context_background_name": background,
        "trade_action": "WAIT",
        "handoff": "05",
    }
