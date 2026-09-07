"""Build the event-by-event candidate revision ledger.

Official L/H is an offline teaching answer.  It is used to cut completed
historical journeys and create fixed action labels, never as a runtime input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from twin_core import TwinContractError, require_columns


EXPECTED_EVENTS = 4_136
EXPECTED_EACH_SIDE = 2_068
OHLCV_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}
LEDGER_COLUMNS = {"side", "pivot_time", "pivot_price"}


def _normalise_ohlcv(ohlcv: pd.DataFrame) -> pd.DataFrame:
    require_columns(ohlcv.columns, OHLCV_COLUMNS, "candidate_revision.ohlcv")
    frame = ohlcv.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    if frame["timestamp"].duplicated().any():
        raise TwinContractError("candidate_revision: duplicate OHLCV timestamp")
    if not frame["timestamp"].diff().dropna().eq(pd.Timedelta(minutes=5)).all():
        raise TwinContractError("candidate_revision: OHLCV is not a continuous 5m clock")
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def _locate_pivots(ohlcv: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    require_columns(ledger.columns, LEDGER_COLUMNS, "candidate_revision.ledger")
    events = ledger.copy().reset_index(drop=True)
    events["side"] = events["side"].astype(str).str.upper()
    events["pivot_time"] = pd.to_datetime(events["pivot_time"], utc=True)
    events["pivot_price"] = pd.to_numeric(events["pivot_price"], errors="raise")
    if not events["side"].isin(["L", "H"]).all():
        raise TwinContractError("candidate_revision: event side must be L or H")
    if (events["side"] == events["side"].shift()).any():
        raise TwinContractError("candidate_revision: official L/H must alternate")

    if "pivot_idx" in events:
        located = pd.to_numeric(events["pivot_idx"], errors="raise").astype("int64")
    else:
        located_values: list[int] = []
        for event in events.itertuples(index=False):
            start = event.pivot_time
            hour = ohlcv[
                (ohlcv["timestamp"] >= start)
                & (ohlcv["timestamp"] < start + pd.Timedelta(hours=1))
            ]
            if hour.empty:
                raise TwinContractError(f"candidate_revision: no 5m bars for {start}")
            price_column = "low" if event.side == "L" else "high"
            distance = (hour[price_column] - float(event.pivot_price)).abs()
            located_values.append(int(distance.idxmin()))
        located = pd.Series(located_values, dtype="int64")
    events["pivot_idx"] = located.to_numpy()
    if not events["pivot_idx"].is_monotonic_increasing or events["pivot_idx"].duplicated().any():
        raise TwinContractError("candidate_revision: pivot indices must strictly increase")
    if events["pivot_idx"].min() < 0 or events["pivot_idx"].max() >= len(ohlcv):
        raise TwinContractError("candidate_revision: pivot index outside OHLCV")
    return events


def validate_official_ledger(events: pd.DataFrame, strict: bool = True) -> dict[str, object]:
    counts = events["side"].value_counts().to_dict()
    result = {
        "events": int(len(events)),
        "L": int(counts.get("L", 0)),
        "H": int(counts.get("H", 0)),
        "alternating": bool(not (events["side"] == events["side"].shift()).any()),
    }
    result["canonical"] = bool(
        result["events"] == EXPECTED_EVENTS
        and result["L"] == EXPECTED_EACH_SIDE
        and result["H"] == EXPECTED_EACH_SIDE
        and result["alternating"]
    )
    if strict and not result["canonical"]:
        raise TwinContractError(f"candidate_revision: official ledger mismatch {result}")
    return result


def probe_macd_lh_events(ohlcv: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Rebuild the old MACD event definition for diagnosis, never for training."""
    frame = _normalise_ohlcv(ohlcv)
    close = frame["close"].astype(float)
    line = close.ewm(span=144, adjust=False).mean() - close.ewm(span=312, adjust=False).mean()
    hist = line - line.ewm(span=108, adjust=False).mean()
    sign = np.sign(hist.to_numpy()).astype(np.int8)
    nonzero = np.flatnonzero(sign)
    if not len(nonzero):
        raise TwinContractError("candidate_revision: MACD event probe found no sign")
    sign[: nonzero[0]] = sign[nonzero[0]]
    for index in range(nonzero[0] + 1, len(sign)):
        if sign[index] == 0:
            sign[index] = sign[index - 1]
    starts = np.r_[0, np.flatnonzero(sign[1:] != sign[:-1]) + 1]
    records = []
    for number, (start, stop) in enumerate(zip(starts[:-1], starts[1:]), 1):
        end = int(stop - 1)
        if sign[start] > 0:
            pivot = int(start + np.argmax(frame.loc[start:end, "high"].to_numpy()))
            side, price = "H", float(frame.at[pivot, "high"])
        else:
            pivot = int(start + np.argmin(frame.loc[start:end, "low"].to_numpy()))
            side, price = "L", float(frame.at[pivot, "low"])
        records.append(
            {
                "event_no": number,
                "side": side,
                "pivot_idx": pivot,
                "pivot_time": frame.at[pivot, "timestamp"],
                "pivot_price": price,
                "probe_only_not_official": True,
            }
        )
    events = pd.DataFrame(records)
    audit = validate_official_ledger(events, strict=False)
    audit.update(
        {
            "status": "MATCH" if audit["canonical"] else "INPUT_MISMATCH",
            "shortfall": EXPECTED_EVENTS - len(events),
            "training_allowed": False,
        }
    )
    return events, audit


def build_candidate_revision_table(
    ohlcv: pd.DataFrame,
    official_ledger: pd.DataFrame,
    *,
    strict: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    frame = _normalise_ohlcv(ohlcv)
    events = _locate_pivots(frame, official_ledger)
    contract = validate_official_ledger(events, strict=strict)
    pieces: list[pd.DataFrame] = []

    for event_index in range(len(events) - 1):
        source = events.iloc[event_index]
        target = events.iloc[event_index + 1]
        start, stop = int(source.pivot_idx), int(target.pivot_idx)
        if stop <= start:
            raise TwinContractError(f"candidate_revision: reversed case {event_index + 1}")
        previous = events.iloc[event_index - 1] if event_index else None
        previous_bars = start - int(previous.pivot_idx) if previous is not None else stop - start
        previous_move = (
            abs(float(source.pivot_price) - float(previous.pivot_price)) / float(previous.pivot_price)
            if previous is not None
            else abs(float(target.pivot_price) - float(source.pivot_price)) / float(source.pivot_price)
        )
        previous_move = max(previous_move, np.finfo(float).eps)
        part = frame.iloc[start : stop + 1].copy()
        local_step = np.arange(len(part), dtype=np.int32)
        target_side = str(target.side)
        if target_side == "L":
            candidate_price = part["low"].cummin().to_numpy(float)
            new_extreme = np.r_[True, candidate_price[1:] < candidate_price[:-1]]
            extension = (float(source.pivot_price) - candidate_price) / float(source.pivot_price)
            rebound = (part["close"].to_numpy(float) - candidate_price) / candidate_price
        else:
            candidate_price = part["high"].cummax().to_numpy(float)
            new_extreme = np.r_[True, candidate_price[1:] > candidate_price[:-1]]
            extension = (candidate_price - float(source.pivot_price)) / float(source.pivot_price)
            rebound = (candidate_price - part["close"].to_numpy(float)) / candidate_price
        candidate_step = np.maximum.accumulate(np.where(new_extreme, local_step, 0))
        rearm_count = np.cumsum(new_extreme).astype(np.int32)
        fixed_action = np.full(len(part), "WAIT", dtype=object)
        fixed_action[new_extreme] = "REARM"
        fixed_action[-1] = "NOW"

        part["global_idx"] = np.arange(start, stop + 1, dtype=np.int64)
        part["case_id"] = event_index + 1
        part["step_id"] = local_step
        part["source_side"] = str(source.side)
        part["target_side"] = target_side
        part["previous_wave_bars"] = int(previous_bars)
        part["previous_wave_move"] = float(previous_move)
        part["elapsed_ratio"] = (local_step + 1) / max(previous_bars, 1)
        part["candidate_price"] = candidate_price
        part["candidate_step"] = candidate_step
        part["candidate_age"] = local_step - candidate_step
        part["rearm_count"] = rearm_count
        part["rearm_event"] = new_extreme
        part["relative_move"] = extension / previous_move
        part["relative_rebound"] = rebound / previous_move
        part["fixed_action"] = fixed_action
        part["is_final_survivor"] = local_step >= candidate_step[-1]
        pieces.append(part)

    prefix = pd.concat(pieces, ignore_index=True)
    contract.update(
        {
            "transition_cases": int(len(events) - 1),
            "prefix_rows": int(len(prefix)),
            "fixed_actions": {key: int(value) for key, value in prefix["fixed_action"].value_counts().items()},
        }
    )
    return prefix, events, contract
