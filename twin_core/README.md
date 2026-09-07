# TWIN 정본 03~11단계 연결

이 디렉터리는 전달받은 정본 묶음의 구조를 코드로 잠근다. 새로운 RSI식 숫자를 만들거나 64개 입력을 하나의 점수로 평균내지 않는다.

## 고정값과 상대값

- 고정값: `WAIT`, `NOW`, 후보축의 `HOLD`, `REARM`처럼 시스템이 내리는 행동·상태다.
- 상대값: 현재 후보와 과거 후보, 이전 파동, 캔들, 거래량, 8개 시간봉 MACD를 비교해서 나온 관찰값이다.
- 학습된 경계: 정본 JSON에 이미 저장된 확률·지지 사건 수·구간·지속 봉 수만 사용한다. 실행 코드가 새 문턱을 임의로 만들지 않는다.

## 단계 순서

| 단계 | 담당 모듈 | 역할 | 행동 |
|---|---|---|---|
| 03 | `candidate_revision.py` | 현재 극값 유지/갱신 | 후보축 `HOLD/REARM`, 매매는 `WAIT` |
| 04 | `context_partition.py` | 4h·1d·1w의 8개 정확한 부호 배경 | `WAIT` |
| 05 | `time_distribution.py` | 현재/이전 파동의 상대 시간 | `WAIT` |
| 06 | `candle_volume_distribution.py` | 캔들·거래량 반응 | `WAIT` |
| 07 | `multitf_zc_distribution.py` | 8개 시간봉 MACD 증거 보존 | `WAIT` |
| 08 | `flow_order_distribution.py` | 8개 ZC 순서, 동시 발생 유지 | `WAIT` |
| 09 | `rule_induction.py` | 6개 학습법 병렬 인식 | `WAIT` |
| 10 | `repair_queue.py` | 확률·지지 수·지속 봉 블로커 | `WAIT` |
| 11 | `threshold_frontier.py` | 최초 규칙 해제 또는 1h ZC 안전 경로 | `NOW/WAIT` |

`REARM`은 후보 전환이고 `NOW`는 매매 행동이다. 두 축은 독립이므로 같은 닫힌 5분봉에서 `REARM + NOW`가 함께 발생할 수 있다.

## 정본 검증

정본 ZIP은 공개 저장소 밖에 풀고 다음처럼 검사한다.

```powershell
$env:TWIN_AUTHENTIC_BUNDLE='D:\private\twin_authentic_2775_stage03_11'
python scripts\audit_authentic_bundle.py $env:TWIN_AUTHENTIC_BUNDLE --ohlcv-dir twin_ohlcv\data
python -m unittest discover -s tests -v
```

검사기는 다음을 확인한다.

- 전달 ZIP에 기록된 12개 파일의 SHA-256
- 4,136개 사건, H/L 각 2,068개, 엄격한 교대와 증가 인덱스
- 2,775개 선택 서명 = 상대조건 규칙 2,499개 + 단일 채널 구간 276개, 누락 0
- 정확한 64개 실행 입력과 미래 정답 필드 차단
- 03~11단계 순서, 16개 모듈 소유권, 11단계 이전 NOW 금지

## 현재 재생 잠금

작은 전달 ZIP에는 정본 규칙과 공식 사건 장부는 있지만, 정본 매니페스트가 지정한 `causal_market_rows_v1.parquet`(SHA-256 `17FDD756...EA05`)은 없다. Git의 660,853행 Binance/Bitstamp OHLCV는 관찰용으로 정상이나 이 비공개 정본 파일의 대체물로 간주하지 않는다.

따라서 현재 상태는 `STRUCTURE_PASS_REPLAY_SOURCE_MISSING`이다. 4,136건 순차 재생, 4,000개 규칙 NOW, 136개 ZC 안전 경로, 승률 수치는 정본 원시 시장행과 656,136개 인과 특성을 다시 만든 뒤에만 재검증 완료로 바꿀 수 있다.

## 원본에서 발견한 불일치

최상위 실행 매니페스트는 `cross_feature_intersection_used=true`라고 기록하지만 세 개 행동 규칙집과 의사결정 구조는 모두 `false` 및 `INDEPENDENT_EVIDENCE_NO_CROSS_FEATURE_INTERSECTION`이다. 코드는 이 차이를 숨기지 않고 감사 결과에 남긴다.

또한 07단계 설명에는 8개 시간봉 × 5개 항목(40개)이 적혀 있지만 정본 64개 실행 입력에서는 2h·4h·1d·1w의 `zc_count_so_far` 네 항목이 빠져 있다. 선택 규칙의 조건은 모두 실제 64개 입력 안에 있으므로, 실행기는 없는 네 값을 만들어 넣지 않고 이 차이를 감사 기록으로 보존한다.
