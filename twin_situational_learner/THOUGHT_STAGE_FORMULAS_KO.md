# TWIN 사고 1~9단계·16개 모듈 계산식 정리

## 1. 정리 기준

이 문서는 다음 두 자료를 함께 기준으로 삼는다.

1. 제공된 정본 청사진의 기존 Stage 03~11 책임과 순서
2. 현재 `twin-authentic-export-router-v1` 실행기에서 실제 실행되는 계산식

사고 1~9단계는 기존 Stage 03~11에 다음처럼 대응한다.

| 사고 단계 | 기존 단계 | 직접 담당 모듈 |
|---:|---:|---|
| 1 | 03 | `candidate_revision.py` |
| 2 | 04 | `context_partition.py` |
| 3 | 05 | `time_distribution.py` |
| 4 | 06 | `candle_volume_distribution.py` |
| 5 | 07 | `multitf_zc_distribution.py` |
| 6 | 08 | `flow_order_distribution.py` |
| 7 | 09 | `rule_induction.py` |
| 8 | 10 | `repair_queue.py` |
| 9 | 11 | `threshold_frontier.py` |

Stage 03~08은 관찰값을 만들고 넘긴다. Stage 09는 여섯 매매사고가 수식 주장과 상대환경을 포함한 주장을 찾는다. Stage 10은 각 주장을 `EXCLUDE / WAIT / KEEP`으로 구분한다. Stage 11에는 `KEEP`만 전달된다.

학습 때만 최종 L/H, 후보 생존/교체, 과거 행동을 정답으로 사용한다. 실전 입력에는 이 정답값을 넣지 않는다.

## 2. 공통 기호

| 기호 | 의미 |
|---|---|
| `t` | 현재 닫힌 5분봉 위치 |
| `O_t, H_t, L_t, C_t, V_t` | 현재 봉의 시가·고가·저가·종가·거래량 |
| `side_code` | 찾는 방향을 수식에 맞추는 방향값 |
| `E*` | 현재 추적 중인 후보 극값 |
| `ε` | 0으로 나누는 것을 막는 아주 작은 값 |
| `S` | 그 관계에서 살아남은 과거 후보 수 |
| `R` | 그 관계에서 나중에 교체된 과거 후보 수 |
| `run` | 같은 관계가 연속해서 유지된 현재 봉 수 |
| `required` | 과거 가짜 후보의 최장 연속 유지보다 길어지기 위해 필요한 봉 수 |

## 3. 사고 1 — 후보 극값 생명주기

담당 모듈: `candidate_revision.py`  
기존 단계: Stage 03

### 볼 상황

- 현재 찾는 방향이 `H` 또는 `L`이고 현재 후보 극값이 존재할 때
- 최신 닫힌 5분봉의 고가 또는 저가가 들어왔을 때

### 보지 않을 상황

- 반대 방향 후보
- 학습 중 현재 사건 자신의 미래 정답
- 생존 여부나 최종 피벗 같은 사후값

### 핵심 계산식

방향 가격:

```text
p_t = H_t  (seek_side = H)
p_t = L_t  (seek_side = L)
```

극값 갱신:

```text
new_extreme_now = 1[p_t > E*]  (H 방향)
new_extreme_now = 1[p_t < E*]  (L 방향)
```

후보축 결론:

```text
candidate_transition = REARM  if new_extreme_now = 1
candidate_transition = HOLD   if new_extreme_now = 0
```

후보 상대값:

```text
candidate_age_log       = log(1 + 현재위치 - 후보탄생위치)
rearm_count_log         = log(1 + 현재 사건의 후보 갱신 횟수)
rearm_gap_bars_log      = log(1 + 현재 갱신위치 - 직전 갱신위치)
rearm_extension_pct     = |새 극값 - 직전 극값| / |직전 극값| × 100
rearm_extension_ratio   = 현재 갱신폭 / 직전 갱신폭
```

현재 후보에서 종가가 밀린 정도:

```text
candidate_rejection_pct =
    (E* - C_t) / |E*| × 100  (H 방향)
    (C_t - E*) / |E*| × 100  (L 방향)

candidate_rejection_fraction    = candidate_rejection_pct / max(zone_move_pct, ε)
candidate_rejection_speed       = candidate_rejection_pct / max(candidate_age_bars, 1)
candidate_rejection_range_units = candidate_rejection_pct / max(current_range_pct, ε)
```

### 다음 단계 전달값

`seek_side`, `candidate_transition`, `new_extreme_now`, 후보 나이, 갱신 횟수, 갱신 간격, 갱신폭, 갱신폭 비율을 Stage 04로 넘긴다. 이 단계는 매매 `NOW`를 만들지 않는다.

## 4. 사고 2 — 큰 상대환경 분할

담당 모듈: `context_partition.py`  
기존 단계: Stage 04

### 볼 상황

- Stage 03에서 넘어온 모든 현재 후보
- REARM 후보와 HOLD 후보를 모두 본다.

### 핵심 계산식

4시간·1일·1주 MACD 부호를 현재 찾는 방향에 맞춘다.

```text
s_tf = sign(macd_hist_tf) × side_code
tf ∈ {4h, 1d, 1w}
```

부호 문자열:

```text
context_signature = sign(s_4h) + sign(s_1d) + sign(s_1w)
```

8개 정확한 부호 배열을 보존한 뒤 세 실행 장부 중 하나를 고른다.

```text
+++  → ALIGNED
---  → OPPOSED
나머지 6개 부호 배열 → MIXED
```

### 제외 조건

- 0 또는 누락된 상위 시간대 부호
- 미래에 완성될 MACD 값
- 8개 부호 배열을 하나의 점수로 합치는 처리

### 다음 단계 전달값

정확한 부호 배열, 방향 포함 부호 배열, `ALIGNED / MIXED / OPPOSED` 배경을 Stage 05로 넘긴다. 이 배경은 뒤에서 같은 수식이 서로 다른 환경에 놓였는지를 구분한다.

## 5. 사고 3 — 현재 파동과 과거 파동의 상대시간·이동·속도

담당 모듈: `time_distribution.py`  
기존 단계: Stage 05

### 시작점

- 현재 파동 시작점
- 직전 파동의 두 피벗
- 전전 파동의 두 피벗

### 시간 복원

```text
wave_age_bars = round(expm1(wave_age_log))
zone_age_bars = round(expm1(zone_age_log))
```

### 상대시간

```text
time_ratio_prev1 = 현재 파동 경과봉 / 직전 파동 경과봉
time_ratio_prev2 = 현재 파동 경과봉 / 전전 파동 경과봉
```

### 상대이동

```text
wave_move_pct_so_far = |현재 후보가격 - 현재 파동 시작가격| / |시작가격| × 100
zone_move_pct_so_far = |현재 후보가격 - 현재 구간 시작시가| / |구간 시작시가| × 100

move_ratio_prev1 = 현재 파동 이동률 / 직전 파동 이동률
move_ratio_prev2 = 현재 파동 이동률 / 전전 파동 이동률
```

### 상대속도

```text
speed_N = max(wave_move_pct[t] - wave_move_pct[t-N], 0) / 실제 경과봉
speed_recent3 = speed_3
speed_ratio_3_to_12 = min(speed_3 / speed_12, 4)  if speed_12 > 0
speed_ratio_3_to_12 = 0                        if speed_12 = 0
```

### 제외 조건

- 직전 또는 전전 파동이 없으면 숫자 0을 만들어 넣지 않는다.
- 비교 이력이 없다는 상태로 유지한다.
- 사람이 정한 최소 대기봉은 넣지 않는다.

### 다음 단계 전달값

파동/구간 나이, 두 시간비율, 두 이동비율, 현재 이동률, 최근 속도와 속도비율을 Stage 06으로 넘긴다.

## 6. 사고 4 — 캔들 구조와 거래량 반응

담당 모듈: `candle_volume_distribution.py`  
기존 단계: Stage 06

### 시작점

최신 닫힌 5분봉과 그 이전 거래량·가격범위 기준 구간이다.

### 캔들 계산식

```text
range_t = H_t - L_t
candle_body_ratio = |C_t - O_t| / range_t
upper_wick_ratio   = (H_t - max(O_t, C_t)) / range_t
lower_wick_ratio   = (min(O_t, C_t) - L_t) / range_t
candle_direction_for_side = sign(C_t - O_t) × side_code
```

방향 기준 종가 위치:

```text
close_position_for_side = (C_t - L_t) / range_t  (H 방향)
close_position_for_side = (H_t - C_t) / range_t  (L 방향)
```

거래량·범위 상대값의 계약식:

```text
volume_ratio20 = V_t / volume_reference20
volume_ratio72 = V_t / volume_reference72
range_ratio20  = current_range / range_reference20
```

### 원본 계약과 현재 OHLCV 재구성기의 차이

제공된 청사진은 기준값 이름을 `volume_sma20`, `volume_sma72`, `range_sma20`으로 기록한다. 현재 OHLCV 재구성기는 다음처럼 현재 봉을 제외한 직전 구간의 중앙값을 사용한다.

```text
volume_reference20 = median(V[t-20 : t-1])
volume_reference72 = median(V[t-72 : t-1])
range_reference20  = median(range_pct[t-20 : t-1])
```

따라서 이 세 기준은 원본 Stage 06 행별 장부가 들어오면 반드시 원본과 다시 대조해야 한다.

### 제외 조건

- 가격범위가 0인 봉의 비율 계산
- 거래량이 없는데 임의의 숫자를 채우는 처리

### 다음 단계 전달값

방향 캔들, 몸통, 위·아래 꼬리, 방향 기준 종가 위치, 거래량 두 비율, 범위비율을 Stage 07로 넘긴다.

## 7. 사고 5 — 8개 시간대 모멘텀 관찰

담당 모듈: `multitf_zc_distribution.py`  
기존 단계: Stage 07

시간대 순서는 다음과 같다.

```text
5m → 15m → 30m → 1h → 2h → 4h → 1d → 1w
```

### 방향 상대값

각 시간대에서 다음 값을 따로 보존한다.

```text
macd_tf_hist_for_side  = macd_tf_hist × side_code
macd_tf_delta_for_side = macd_tf_delta × side_code
macd_tf_sign_for_side  = macd_tf_sign × side_code
zc_age_log             = log(1 + 현재위치 - 마지막 ZC 위치)
```

5m·15m·30m·1h에는 현재 1시간 파동 안에서 관찰된 ZC 누적 횟수도 보존한다.

### 현재 OHLCV 재구성기의 MACD 계산

5분봉 기준 시간대 배수 `M`은 다음과 같다.

```text
5m=1, 15m=3, 30m=6, 1h=12, 2h=24, 4h=48, 1d=288, 1w=2016
```

현재 재구성기는 다음 계산을 쓴다.

```text
macd_line = EMA(12M, close) - EMA(26M, close)
hist_raw  = macd_line - EMA(9M, macd_line)
hist      = clip(hist_raw / max(EMA(26M, |hist_raw|), ε), -20, 20)

delta_raw = hist_raw[t] - hist_raw[t-1]
delta     = clip(delta_raw / max(EMA(9M, |delta_raw|), ε), -20, 20)
sign      = +1 if hist_raw >= 0 else -1
```

원본 청사진은 Stage 07의 입력과 방향 변환은 보존하지만, OHLCV에서 이 정규화 MACD를 생성하는 내부식을 포함하지 않는다. 따라서 위 MACD 생성식은 현재 재구성기 식이며 원본 Stage 07 장부 해시와 아직 대조되지 않았다.

### 제외 조건

- 여덟 시간대를 합친 투표
- 하나의 모멘텀 점수
- 완성되지 않은 미래 시간대 값

### 다음 단계 전달값

각 시간대의 `hist / delta / sign / ZC count / ZC age`를 독립 증인으로 Stage 08에 넘긴다.

## 8. 사고 6 — 신호 발생 순서와 동시성

담당 모듈: `flow_order_distribution.py`  
기존 단계: Stage 08

### 핵심 계산식

```text
last_zc_position_tf = raw_position_t - round(expm1(zc_age_log_tf))
```

여덟 시간대의 마지막 ZC 위치를 오름차순으로 정렬한다.

```text
signal_order_sequence = sort(last_zc_position_5m ... last_zc_position_1w)
distinct_count = 서로 다른 last_zc_position의 개수
```

### 동시 발생 처리

같은 위치에서 발생한 여러 시간대는 한 그룹으로 유지한다.

```text
같은 raw_position → simultaneous_timeframes
```

동시 발생을 임의로 5m, 15m, 1h 순서처럼 쪼개지 않는다.

### 다음 단계 전달값

여덟 ZC 위치, 서로 다른 위치 수, 동시성을 보존한 발생 순서를 Stage 09로 넘긴다.

## 9. 사고 7 — 6개 매매사고의 수식 주장·상대환경 주장

담당 모듈: `rule_induction.py`  
기존 단계: Stage 09

### 여섯 매매사고와 값 소유권

| 매매사고 | 주로 소유하는 관찰값 |
|---|---|
| `WAVE_REARM_AGE` | 파동/구간 시간·이동, 후보 나이, 갱신 횟수·간격·폭, 후보 거부값 |
| `MOMENTUM_SPEED` | `speed_*`, 5분 MACD 힘·변화·부호·ZC |
| `CANDLE_REVERSAL` | 방향 캔들, 몸통, 꼬리, 종가 위치 |
| `VOLUME_RANGE` | 거래량 20/72 비율, 범위 20 비율 |
| `MACRO_TREND` | 4시간·1일·1주 MACD |
| `MID_MACD_TREND` | 15분·30분·1시간·2시간 MACD |

### BASE 수식 주장

현재 Stage 04 배경의 모든 기존 구간 수식을 계산한다.

```text
BASE_MATCH =
    (lower가 없거나 x_t > lower)
    AND
    (upper가 없거나 x_t <= upper)
```

`x_t`는 Stage 03~08에서 받은 상대값 중 하나다. BASE 일치는 행동이 아니라 수식 주장이다.

### CONTEXT 수식 + 상대환경 주장

각 조건의 피연산자는 두 종류다.

```text
CURRENT_VALUE:       x_t
CURRENT_MINUS_PRIOR: x_t - x_(t-k)
```

조건식:

```text
condition_i = operand_i > learned_boundary_i
           또는 operand_i <= learned_boundary_i

CONTEXT_MATCH = AND(condition_1 ... condition_n)
```

현재 프로그램의 CONTEXT 5,500개를 전수 검사한 결과, 5,500개 모두 자기 매매사고가 소유한 수식 조건을 최소 하나 포함한다. 따라서 CONTEXT는 환경만 따로 내는 독립 신호가 아니라 `같은 매매법 수식 + 상대환경`이다.

### 후보 선택 순서

같은 봉에서 여러 관계가 맞으면 다음 순서로 가장 강한 관계를 고른다.

```text
1. state_probability 높은 순서
2. event_support 높은 순서
3. 기존 매매사고 순서
4. 원본 장부의 채널/규칙/구간 순서
5. calculation_id 순서
```

### 제외 조건

- 2,775개 과거 주소와 현재 문자열을 직접 비교하는 화이트리스트
- 미래 정답값
- 여섯 매매사고를 한 점수로 합치는 처리

### 다음 단계 전달값

각 매매사고의 BASE 주장, CONTEXT 주장, 확률, 근거 수, 요구 지속봉, 계산식 식별자를 Stage 10으로 넘긴다.

## 10. 사고 8 — 반례·근거·지속성 블로커

담당 모듈: `repair_queue.py`  
기존 단계: Stage 10

이 단계 안의 세 판단렌즈는 새 모듈이 아니다.

1. `RELATION`: 현재 수식과 상대관계가 맞았는가
2. `ENVIRONMENT`: 같은 관계에서 살아남은 후보와 교체된 후보의 근거가 충분한가
3. `PERSISTENCE`: 가짜 후보보다 오래 유지됐는가

### 학습 확률

각 관계의 상태확률은 과거 살아남은 후보와 교체된 후보의 상대비율이다.

```text
state_probability = S / (S + R)
```

### 학습된 지속성

```text
required = 해당 관계가 가짜 후보에서 연속 유지된 최대 봉 수 + 1
```

방향별 값이 있으면 `H`와 `L`의 `required`를 따로 쓴다.

### 현재 연속 유지봉

```text
같은 relation signature가 계속되면 run_t = run_(t-1) + 1
관계가 바뀌면 run_t = 1
관계가 없으면 run_t = 0
```

현재 실행기에서는 새 극값, 관계 식별자 변경, 1시간 파동 변경 때 관련 연속상태를 초기화한다.

### 기존 블로커 식

프로그램에 저장된 기존 값은 다음과 같다.

```text
gate = 0.44204017519950867
tolerance = 0.0000001
minimum_event_support = 3
minimum_context_event_support = 3
```

이 값은 이번 정리에서 새로 만든 고정 시장 문턱이 아니라 기존 학습 결과물에 이미 있던 선택값이다.

경로 통과식:

```text
READY =
    relation_matched
    AND support >= minimum_support
    AND state_probability + tolerance >= gate
    AND run >= required
```

### 사고경로 결론

```text
관계 없음                         → NOT_APPLICABLE
support 부족 또는 probability 부족 → EXCLUDE
관계는 유효하지만 run < required    → WAIT
모든 블로커 해제                    → KEEP
```

`EXCLUDE`는 전체 매매를 WAIT시키는 것이 아니라 해당 사고경로만 제외한다. CONTEXT의 `KEEP`은 환경 단독 승인이 아니라 자기 수식과 상대환경을 함께 통과했다는 뜻이다.

매매사고별 현재 준비식:

```text
method_keep = base_keep OR contextualized_signal_keep
```

두 항은 단순 수식과 환경을 서로 독립 투표시키는 뜻이 아니다. BASE도 현재 배경·확률·반례 지속성 검사를 통과해야 하고, CONTEXT는 자기 수식과 상대환경을 이미 함께 포함한다.

### 다음 단계 전달값

여섯 매매사고의 지지 근거, 반례, 미완료 지속성, `EXCLUDE / WAIT / KEEP`, 최종 KEEP 후보를 Stage 11로 넘긴다.

## 11. 사고 9 — 최종 행동과 다음 봉 체결

담당 모듈: `threshold_frontier.py`  
기존 단계: Stage 11

### 후보축과 매매축

후보축은 Stage 03의 결과를 그대로 가진다.

```text
candidate_transition ∈ {REARM, HOLD}
```

매매축은 별도로 결정한다.

```text
if 현재 사건이 아직 미출시 AND Stage10 KEEP 후보 존재:
    trade_action = NOW
    action_source = RULE_NOW

elif 현재 사건이 아직 미출시 AND 실제 관찰된 1h ZC 전환 존재:
    trade_action = NOW
    action_source = OBSERVED_1H_ZC_STATE_NOW

else:
    trade_action = WAIT
```

REARM과 NOW는 서로 독립이므로 같은 닫힌 봉에서 둘 다 참일 수 있다.

### 최종 후보 순위

여러 KEEP 후보가 있으면 Stage 09와 같은 순위인 확률, 근거 수, 매매사고 순서, 원본 순서로 하나를 고른다.

### 체결식

```text
decision_bar = 닫힌 현재 5분봉 t
entry_fill   = 다음 닫힌 5분봉의 시가 O_(t+1)
```

사건당 첫 NOW만 허용한다. 현재 실행기는 2,775개 과거 출력 주소를 최종 선택 명단으로 조회하지 않는다.

## 12. 16개 모듈 전체 연결표

16개는 모두 매매 수식을 새로 만드는 모듈이 아니다. 9개가 사고 단계를 직접 담당하고, 7개는 실행·검산·재생·저장을 담당한다.

| 번호 | 모듈 | 연결 위치 | 계산 또는 책임 |
|---:|---|---|---|
| 1 | `learning_worker` | 전체 1→9 | 사건 상태·이력·지속성 상태를 들고 Stage 03→11을 순서대로 호출 |
| 2 | `distribution_learner` | 사고 전 | 기존 학습 장부와 규칙 묶음을 불러옴. 현재 정본 경로에서는 16개 임계값을 새로 학습하지 않음 |
| 3 | `candidate_revision` | 사고 1 | 현재 방향 가격과 후보 극값 비교, REARM/HOLD |
| 4 | `context_partition` | 사고 2 | 4h·1d·1w 방향 상대부호, 8개 부호 배열과 3개 배경 |
| 5 | `time_distribution` | 사고 3 | 파동/구간 나이 복원, 시간·이동·속도 상대비교 |
| 6 | `candle_volume_distribution` | 사고 4 | 캔들 몸통·꼬리·종가 위치와 거래량·범위 비율 |
| 7 | `multitf_zc_distribution` | 사고 5 | 8개 시간대 MACD 힘·변화·부호·ZC 관찰 |
| 8 | `flow_order_distribution` | 사고 6 | 시간대별 마지막 ZC 위치, 동시성, 발생 순서 |
| 9 | `rule_induction` | 사고 7 | 5,363개 BASE 구간과 5,500개 CONTEXT 상대관계 계산 |
| 10 | `repair_queue` | 사고 8 | 확률·근거·지속성 블로커, EXCLUDE/WAIT/KEEP |
| 11 | `threshold_frontier` | 사고 9 | 첫 KEEP 또는 실제 1h ZC 보조전환으로 NOW/WAIT 결정 |
| 12 | `scorebook` | 학습 후 검산 | 행동별 개수와 사후 성과를 계산하되 런타임 결정에는 참여하지 않음 |
| 13 | `replay_builder` | 사고 9 후 | 과거 각 시점의 접두 구간만 순서대로 재생하고 다음 봉 시가 체결을 연결 |
| 14 | `mdd_problem` | 매매 후 | `peak_t=max(equity_0..t)`, `drawdown_t=equity_t/peak_t-1`, `MDD=min(drawdown_t)` |
| 15 | `learning_status` | 전체 감시 | 완료 봉 수, 단계 상태, 검증 세부값 기록. 시장 판단식 없음 |
| 16 | `artifact_export` | 최종 저장 | 9단계 계약, 10,863개 계산, 2,775개 증거, 3,972개 과거 전달 바통을 검증 후 JSON으로 저장 |

## 13. 계산 수량과 저장 역할

```text
BASE 구간 계산       = 5,363개
CONTEXT 관계 계산    = 5,500개
현재 실행 계산 전체  = 10,863개

과거 선택 출력 증거  = 2,775개
과거 10→11 전달 바통 = 3,972개
```

10,863개는 현재 차트에서 계산하는 전체 관계 모델이다. 2,775개와 3,972개는 과거 학습과 검산 증거이며 현재 행동을 고르는 화이트리스트가 아니다.

## 14. 최종 한 줄 구조

```text
OHLCV
→ 후보 갱신
→ 큰 배경
→ 파동 시간·이동·속도
→ 캔들·거래량
→ 8개 시간대 모멘텀
→ 발생 순서
→ 6개 매매사고의 수식+상대환경 주장
→ 반례·지속성으로 EXCLUDE/WAIT/KEEP
→ KEEP만 NOW, 없으면 관찰 지속 또는 실제 1h ZC 보조전환
```
