from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_EVENTS = 4136
EXPECTED_H = 2068
EXPECTED_L = 2068


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def ema(values: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    out = np.empty(len(values), dtype=np.float64)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def macd_sign(values: np.ndarray) -> np.ndarray:
    line = ema(values, 12) - ema(values, 26)
    hist = line - ema(line, 9)
    sign = np.sign(hist).astype(np.int8)
    nz = np.flatnonzero(sign)
    if not len(nz):
        sign[:] = 1
        return sign
    sign[: nz[0]] = 1
    for i in range(nz[0], len(sign)):
        if sign[i] == 0:
            sign[i] = sign[i - 1] if i else 1
    return sign


def bin025(value: float | None) -> str:
    if value is None or pd.isna(value):
        return 'NA'
    lo = math.floor(float(value) / 0.25) * 0.25
    return f'{lo:.2f}~{lo + 0.25:.2f}'


def load_sources(path5: Path, ledger_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    five = pd.read_csv(path5)
    need5 = {'timestamp', 'open', 'high', 'low', 'close', 'volume', 'fill_source'}
    if not need5.issubset(five.columns):
        raise RuntimeError(f'5M_COLUMNS_FAIL missing={sorted(need5-set(five.columns))}')
    five['timestamp'] = pd.to_datetime(five['timestamp'], utc=True)
    if five['timestamp'].duplicated().any():
        raise RuntimeError('5M_DUPLICATE_TIME')
    if not five['timestamp'].diff().dropna().eq(pd.Timedelta(minutes=5)).all():
        raise RuntimeError('5M_GAP_FAIL')

    ledger = pd.read_csv(ledger_path)
    if len(ledger) != EXPECTED_EVENTS:
        raise RuntimeError(f'LEDGER_EVENT_COUNT_FAIL {len(ledger)}')
    counts = ledger['side'].value_counts().to_dict()
    if counts.get('H') != EXPECTED_H or counts.get('L') != EXPECTED_L:
        raise RuntimeError(f'LEDGER_SIDE_COUNT_FAIL {counts}')
    for col in ['pivot_time', 'zc_time', 'target_time']:
        ledger[col] = pd.to_datetime(ledger[col], utc=True, errors='coerce')
    return five, ledger


def add_visible_features(five: pd.DataFrame) -> pd.DataFrame:
    work = five.copy()
    rng = (work['high'] - work['low']).replace(0, np.nan)
    work['body_ratio'] = ((work['close'] - work['open']).abs() / rng).fillna(0.0)
    work['candle_dir'] = np.where(work['close'] > work['open'], 'U', np.where(work['close'] < work['open'], 'D', 'N'))
    dirs = work['candle_dir'].astype(str).tolist()
    work['seq4'] = [''.join(dirs[max(0, i - 3): i + 1]).rjust(4, '_') for i in range(len(work))]
    work['volume_ratio20'] = work['volume'] / work['volume'].rolling(20, min_periods=1).mean()
    work['macd5_sign'] = macd_sign(work['close'].to_numpy(dtype=np.float64))

    for minutes, name in [(15, 'macd15_sign'), (60, 'macd60_sign')]:
        tf = work.set_index('timestamp')['close'].resample(f'{minutes}min', label='left', closed='left').last().dropna()
        tf_sign = macd_sign(tf.to_numpy(dtype=np.float64))
        available_at = tf.index + pd.Timedelta(minutes=minutes - 5)
        visible = pd.Series(tf_sign, index=available_at).reindex(work['timestamp'], method='ffill').fillna(1).astype(np.int8)
        work[name] = visible.to_numpy()
    return work


def locate_events(five: pd.DataFrame, ledger: pd.DataFrame) -> tuple[dict[int, int], list[dict[str, object]]]:
    located: dict[int, int] = {}
    unresolved: list[dict[str, object]] = []
    for _, event in ledger.iterrows():
        start = event['pivot_time']
        hour = five[(five['timestamp'] >= start) & (five['timestamp'] < start + pd.Timedelta(hours=1))]
        field = 'low' if event['side'] == 'L' else 'high'
        delta = (hour[field] - float(event['pivot_price'])).abs()
        idx = int(delta.idxmin())
        if float(delta.loc[idx]) < 1e-8:
            located[int(event['event_no'])] = idx
        else:
            unresolved.append({
                'event_no': int(event['event_no']),
                'side': str(event['side']),
                'pivot_time_1h': start.isoformat(),
                'pivot_price_1h': float(event['pivot_price']),
                'nearest_5m_time': five.at[idx, 'timestamp'].isoformat(),
                'nearest_5m_price_delta': float(delta.loc[idx]),
            })
    return located, unresolved


def build(path5: Path, ledger_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    five, ledger = load_sources(path5, ledger_path)
    five = add_visible_features(five)
    located, unresolved_events = locate_events(five, ledger)

    memory_path = out_dir / 'CASE_STEP_MEMORY_v1.jsonl.gz'
    labels_path = out_dir / 'CASE_STEP_LEARNING_LABELS_POSTHOC_v1.jsonl.gz'
    index_paths = {name: out_dir / f'INDEX_{name}_ACTION_FREE_v1.jsonl.gz' for name in ['time', 'move', 'candle_sequence', 'mtf_macd', 'volume', 'previous_waves']}

    index_maps: dict[str, dict[str, list[str]]] = {name: {} for name in index_paths}
    stable_cases = 0
    steps = 0
    unresolved_case_ids: list[int] = []

    with gzip.open(memory_path, 'wt', encoding='utf-8') as mem, gzip.open(labels_path, 'wt', encoding='utf-8') as lab:
        for i in range(len(ledger) - 1):
            source = ledger.iloc[i]
            target = ledger.iloc[i + 1]
            case_id = int(source['event_no'])
            target_id = int(target['event_no'])
            if case_id not in located or target_id not in located:
                unresolved_case_ids.append(case_id)
                continue
            a, b = located[case_id], located[target_id]
            source_price = float(source['pivot_price'])
            side = str(source['side'])
            previous = ledger.iloc[i - 1] if i else None
            stable_cases += 1

            for step_id, idx in enumerate(range(a, b + 1)):
                row = five.iloc[idx]
                move = ((float(row['close']) / source_price) - 1.0) * 100.0 if side == 'L' else ((source_price / float(row['close'])) - 1.0) * 100.0
                prev_side = 'BOUNDARY' if previous is None else str(previous['side'])
                prev_bars = None if previous is None or pd.isna(previous['total_bars']) else int(previous['total_bars'])
                prev_move = None if previous is None or pd.isna(previous['total_move_pct']) else float(previous['total_move_pct'])

                visible = {
                    'case_id': case_id,
                    'step_id': step_id,
                    'timestamp': row['timestamp'].isoformat(),
                    'source_side': side,
                    'elapsed_5m': step_id,
                    'move_pct_from_source': round(move, 6),
                    'candle_dir': str(row['candle_dir']),
                    'seq4': str(row['seq4']),
                    'body_ratio': round(float(row['body_ratio']), 6),
                    'volume_ratio20': round(float(row['volume_ratio20']), 6),
                    'macd5_sign': int(row['macd5_sign']),
                    'macd15_sign': int(row['macd15_sign']),
                    'macd60_sign': int(row['macd60_sign']),
                    'fill_source': str(row['fill_source']),
                    'previous_wave': {'side': prev_side, 'total_1h_bars': prev_bars, 'total_move_pct': prev_move},
                }
                mem.write(json.dumps(visible, ensure_ascii=False, separators=(',', ':')) + '\n')

                posthoc = {
                    'case_id': case_id,
                    'step_id': step_id,
                    'actual_target_event_no': target_id,
                    'actual_target_side': str(target['side']),
                    'actual_target_time': five.at[b, 'timestamp'].isoformat(),
                    'actual_target_price': float(target['pivot_price']),
                    'remaining_5m_bars': b - idx,
                    'is_actual_target_step': idx == b,
                    'posthoc_only': True,
                }
                lab.write(json.dumps(posthoc, ensure_ascii=False, separators=(',', ':')) + '\n')

                ref = f'{case_id}:{step_id}'
                keys = {
                    'time': f'{side}|elapsed={step_id}',
                    'move': f'{side}|move={bin025(move)}',
                    'candle_sequence': f'{side}|seq4={row["seq4"]}',
                    'mtf_macd': f'{side}|m5={int(row["macd5_sign"])}|m15={int(row["macd15_sign"])}|m60={int(row["macd60_sign"])}',
                    'volume': f'{side}|vr20={bin025(min(float(row["volume_ratio20"]), 5.0))}',
                    'previous_waves': f'{side}|prevSide={prev_side}|prevBars={prev_bars if prev_bars is not None else "NA"}|prevMove={bin025(prev_move)}',
                }
                for name, key in keys.items():
                    index_maps[name].setdefault(key, []).append(ref)
                steps += 1

    for name, path in index_paths.items():
        with gzip.open(path, 'wt', encoding='utf-8') as f:
            for key in sorted(index_maps[name]):
                f.write(json.dumps({'key': key, 'refs': index_maps[name][key]}, ensure_ascii=False, separators=(',', ':')) + '\n')

    last = ledger.iloc[-1]
    (out_dir / 'BOUNDARY_EVENT_4136.json').write_text(json.dumps({
        'case_id': int(last['event_no']), 'status': 'BOUNDARY_OPEN_NO_TARGET', 'side': str(last['side']),
        'pivot_time_1h': last['pivot_time'].isoformat(), 'pivot_price_1h': float(last['pivot_price'])
    }, ensure_ascii=False, indent=2), encoding='utf-8')

    manifest = {
        'status': 'PASS_FULL' if not unresolved_events else 'PARTIAL_UNRESOLVED_SOURCE_EVENT',
        'events': EXPECTED_EVENTS,
        'H': EXPECTED_H,
        'L': EXPECTED_L,
        'exact_event_5m_matches': len(located),
        'unresolved_events': unresolved_events,
        'linked_transition_cases_expected': EXPECTED_EVENTS - 1,
        'stable_linked_transition_cases_built': stable_cases,
        'unresolved_transition_case_ids': unresolved_case_ids,
        'step_records_built': steps,
        'runtime_indexes_include_posthoc_labels': False,
        'learning_labels_separate': True,
        'averages_or_medians_created': False,
        'source_5m_sha256': sha256(path5),
        'source_ledger_sha256': sha256(ledger_path),
        'index_key_counts': {name: len(index_maps[name]) for name in index_maps},
    }
    (out_dir / 'MANIFEST_v1.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--five-minute', required=True, type=Path)
    p.add_argument('--ledger', required=True, type=Path)
    p.add_argument('--out-dir', type=Path, default=Path('lh_case_step_memory_v1'))
    args = p.parse_args()
    build(args.five_minute, args.ledger, args.out_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
