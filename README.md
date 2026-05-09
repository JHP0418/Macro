# Macro Risk-Off Sentinel & ETF Screening System

이 프로젝트는 한국 상장 ETF 유니버스를 대상으로 매크로 환경, 시장 내부 breadth, 위험선호, 고점 취약성, 유사국면, 단기 조정 압력을 결합해 매주 또는 매일 현재 투자 매력도와 위험회피 필요성을 판단하는 스크리닝 시스템이다.

현재 활성 파이프라인에서는 LPPL/DTCAI를 사용하지 않는다. 이전 실험용 LPPL 스크립트와 캐시는 남아 있을 수 있지만, 현재 대시보드, Risk-Off Sentinel, Risk Vector, walk-forward 최적화 레이어의 점수 산출에는 LPPL이 들어가지 않는다.

2026-05-09 업데이트 이후 매크로/해외지수/환율/원자재 드라이버 패널은 `1995-01-01`부터 생성한다. 단, 한국 상장 ETF 자체 가격은 각 ETF 상장일 이전으로 존재하지 않으므로 ETF별 백테스트는 실제 상장 이후 구간에서만 평가된다.

## 핵심 목적

모델의 목표는 전체 시장 방향을 매일 80~90% 맞히는 것이 아니다. 1주 단위 상승/하락은 노이즈가 커서 전체 표본 기준으로 그런 정확도를 안정적으로 만들기 어렵다. 현재 구조의 목표는 다음처럼 분리되어 있다.

1. Risk-Off와 조정 위험을 가능한 한 빨리 감지한다.
2. 고점권에서 하락이 시작되기 전의 취약성을 잡는다.
3. Risk-Off 구간에서는 위험자산보다 안전자산 후보를 우선한다.
4. Risk-On 또는 중립 구간에서는 ETF 유니버스 안에서 상대적으로 더 나은 자산군을 랭킹한다.
5. 단순 OX 정확도보다 손실 회피, drawdown 감소, 고확신 구간 성능을 우선한다.

## 현재 유니버스

투자 가능 자산은 다음 파일에서 읽는다.

- `data/asset_universe.csv`

2026년 GAPS ETF 리스트 기준으로 한국 상장 ETF 186개를 사용한다. 보조 추출 파일은 다음과 같다.

- `data/gaps_etf_list_2026_05_09_extracted.csv`
- `data/asset_universe_expanded_2026_05_09.csv`

코드는 `A069500` 같은 형식으로 입력해도 내부적으로 `069500.KS`로 정규화된다.

ETF는 세부 그룹 외에 GAPS 운용 판단용 상위 바스켓으로도 재분류된다.

```text
해외지수
해외섹터
국내지수
국내섹터
FX및 원자재
국내채권_종합
국내채권_회사채
해외채권_종합
해외채권_회사채
금리연계형 및 초단기채권
```

분류 로직은 `scripts/basket_taxonomy.py`에 있다. 해외/국내 주식은 기존 세부 그룹으로 나누고, 채권은 GAPS ETF 추출 파일의 실제 ETF 이름을 이용해 국채/종합채권/회사채/금융채/초단기/금리연계형을 구분한다.

## 전체 파이프라인

현재 모델은 한 개의 거대한 모델이 아니라 여러 레이어로 나뉜다.

```text
가격/매크로 데이터
  -> Macro Regime Asset Screener
  -> Daily Risk-Off Sentinel
  -> Risk Vector Dashboard
  -> Peak Fragility Model
  -> Analog Macro Risk Model
  -> Correction Timing Model
  -> Walk-Forward Optimizer
  -> Screening HTML Dashboard
```

## 1. Macro Regime Asset Screener

실행:

```powershell
python scripts\macro_regime_asset_screener.py
```

출력:

- `outputs/macro_regime_asset_screener_latest/tables/current_asset_scores.csv`
- `outputs/macro_regime_asset_screener_latest/tables/driver_panel.csv`
- `outputs/macro_regime_asset_screener_latest/tables/driver_state.csv`
- `outputs/macro_regime_asset_screener_latest/tables/regime_history.csv`

이 레이어는 각 ETF의 기본 매력도를 만든다.

사용하는 정보:

- 자산 자체 5일, 20일, 60일, 120일 수익률
- 20일 변동성
- 252일 고점 대비 drawdown
- 자산군별 기대 매크로 드라이버 적합도
- 각 자산의 rolling beta와 driver alignment
- 과거 비슷한 regime에서의 조건부 승률

대표 드라이버:

- 미국 2년/10년 금리
- 미국 실질금리
- 달러 인덱스
- 원/달러
- 달러/엔
- VIX, VXN, MOVE
- HY OAS, IG OAS
- S&P500, Nasdaq100, SOX, Russell2000
- 구리, 유가, 금, 구리/금
- 항셍테크, CSI300
- KOSDAQ/KOSPI

출력 점수:

- `score_0_100`: 현재 자산 매력도
- `upside_prob_1w`: 1주 상승확률 추정
- `upside_prob_4w`: 4주 상승확률 추정
- `technical_score`: 자산 자체 추세 점수
- `driver_fit_score`: 현재 매크로 환경과 자산군의 궁합
- `rolling_beta_fit_score`: 최근 민감도와 기대 드라이버의 일치도

## 2. Daily Risk-Off Sentinel

실행:

```powershell
python scripts\daily_risk_off_sentinel.py --skip-download
```

출력:

- `outputs/daily_risk_off_sentinel_latest/tables/daily_sentinel_history.csv`
- `outputs/daily_risk_off_sentinel_latest/tables/sentinel_adjusted_current_scores.csv`
- `outputs/daily_risk_off_sentinel_latest/tables/sentinel_threshold_sweep.csv`
- `outputs/daily_risk_off_sentinel_latest/reports/daily_sentinel_report.md`

Sentinel은 뉴스 예측 모델이 아니다. 매일 관측되는 시장 가격 기반 위험 신호를 빠르게 점수화하는 레이어다.

각 지표는 원값을 그대로 쓰지 않고, 방향을 맞춘 뒤 1일, 5일, 20일 충격을 계산한다.

```text
1일 충격 = 오늘 변화 / 252일 표준편차
5일 충격 = 최근 5일 누적 변화 / 252일 표준편차 * sqrt(5)
20일 충격 = 최근 20일 누적 변화 / 252일 표준편차 * sqrt(20)

shock_score = max(1일, 5일, 20일 충격) * 가중치
```

방향 예시:

- VIX 상승: 위험 증가
- HY OAS 상승: 위험 증가
- HYG/IEF 하락: 위험 증가
- DXY 상승: 위험 증가
- USDKRW 상승: 한국 위험자산 부담
- USDJPY 하락: 엔화 안전자산 선호, 위험 증가
- Nasdaq 하락: 성장주 위험 증가
- SOX 하락: 반도체 위험 증가
- 구리 하락: 경기민감 위험 증가
- 금 상승: 헤지 수요 증가

Sentinel 상태:

- `Normal`: 위험 예산 정상
- `Watch`: 위험자산 비중 축소 검토
- `De-risk`: 위험자산 축소 우선
- `Cash`: 현금/단기채 중심 방어

## 3. RAI 프록시

현재 모델은 Credit Suisse 원본 RAI 데이터를 쓰지 않는다. 대신 ETF 유니버스와 주요 글로벌 지수로 RAI 프록시를 만든다.

2026-05-09 업데이트 이후 RAI는 자산군 중립 가중 회귀로 계산한다. 한국 상장 ETF에는 같은 노출의 ETF가 많이 중복되어 있으므로, 모든 ETF를 동일 가중으로 넣으면 미국 성장주나 특정 채권군이 과대표집될 수 있다. 지금은 같은 자산군 안의 ETF 수가 많을수록 개별 ETF 가중치를 낮춰 RAI가 유니버스 구성비 왜곡에 덜 흔들리게 했다.

매일 다음을 계산한다.

```text
각 자산의 6개월 수익률 = 126거래일 수익률
각 자산의 12개월 변동성 = 252거래일 일간수익률 표준편차

단면 회귀:
Y = 6개월 수익률
X = 12개월 변동성

RAI_raw = 자산군 중립 가중 회귀 beta
```

해석:

- `RAI_raw`와 `RAI_z`가 높다: 위험자산이 보상받는 위험선호/과열 구간
- `RAI_z`가 급락하거나 낮다: 위험선호 붕괴 또는 공포 구간

파생 점수:

- `RAI_z`: 756일 rolling z-score
- `RAI_level_0_100`: 0~100으로 변환한 RAI 수준
- `RAI_fear_score`: RAI가 낮을 때 상승
- `RAI_collapse_score`: RAI가 20일 동안 급락할 때 상승
- `RAI_overheat_score`: RAI가 과열권일 때 상승
- `RAI_shock_score`: fear와 collapse의 최대값

중요한 구분:

- RAI 저점/급락은 즉시 Risk-Off 위험이다.
- RAI 고점은 그 자체로 Risk-Off가 아니라 고점 취약성, 과열, 향후 조정 위험으로 쓴다.

2026-05-08 기준 RAI는 다음처럼 해석된다.

```text
RAI_z: +2.34
RAI_level_0_100: 85.17
RAI_overheat_score: 33.61
RAI_shock_score: 0.00
```

즉 공포가 아니라 위험선호 과열권이다.

## 4. ETF Breadth와 안전자산 로테이션

ETF 유니버스 내부 상태도 별도로 본다.

계산 항목:

- `ETF_risk_breadth_pct`: 위험자산 ETF 중 추세가 살아 있는 비율
- `ETF_below_60ma_pct`: 전체 ETF 중 60일선 아래 비율
- `ETF_below_20ma_pct`: 전체 ETF 중 20일선 아래 비율
- `ETF_20d_loss_pct`: 20일 수익률이 음수인 ETF 비율
- `ETF_20d_large_loss_pct`: 20일 수익률이 -5% 이하인 ETF 비율
- `ETF_breadth_shock_score`: breadth 악화 종합점수
- `SAFE_ROTATION_shock_score`: 안전자산이 위험자산보다 강해지는 정도

이 레이어는 지수는 버티는데 내부 ETF들이 무너지는 경우를 잡기 위한 것이다.

## 5. 3차원 Risk Vector

실행:

```powershell
python scripts\risk_vector_dashboard.py
```

출력:

- `outputs/risk_vector_dashboard_latest/tables/daily_risk_vector.csv`
- `outputs/risk_vector_dashboard_latest/tables/current_risk_vector.csv`
- `outputs/risk_vector_dashboard_latest/charts/`

Risk Vector는 위험을 하나의 점수로만 압축하지 않고 3개 축으로 나눈다.

### X축: Macro Liquidity Axis

거시 유동성, 신용, 환율, 변동성 스트레스다.

구성:

- 신용 스트레스
- 달러/원화/위안/엔화 스트레스
- 변동성 스트레스
- 유가/인플레 공급 충격
- RAI 위험선호 붕괴

### Y축: Market Breakdown Axis

실제 주식시장 붕괴와 내부 breadth 악화다.

구성:

- S&P500, Nasdaq, SOX, Russell 하락 충격
- VIX/VXN/MOVE
- 중국/경기민감 스트레스
- ETF breadth 악화
- Peak Fragility
- Analog Macro Risk
- Correction Pressure

### Z축: External Supply Axis

대외, 중국, 원자재, 안전자산 이동이다.

구성:

- DXY, USDKRW, USDJPY, USDCNH
- 중국/홍콩/구리
- WTI
- 금
- 안전자산 로테이션

최종 상태:

```text
0~25    Normal
25~40   Fragile
40~55   Warning
55~70   Risk-Off
70~100  Crisis
```

현재 2026-05-08 기준:

```text
Risk Phase: Fragile
Risk Archetype: Correction Pressure
Composite Vector Risk: 26.79
Risk-Off Sentinel: 6.92
Peak Fragility: 64.38
Analog Macro Risk: 57.67
Correction Pressure: 70.47
```

해석:

```text
시스템성 Risk-Off는 아직 아니다.
하지만 위험선호 과열, 고점 취약성, 단기 조정 압력이 높다.
신규 위험자산 매수는 보수적으로 접근하고, 분할/헤지/안전자산 후보를 우선 검토한다.
```

## 6. Peak Fragility Model

실행:

```powershell
python scripts\peak_fragility_model.py --min-train-days 504 --retrain-step-days 21
```

타깃:

```text
Nasdaq이 고점권에 있을 때,
향후 1개월 안에 의미 있는 조정이 발생하는가?
```

구체적 라벨:

- Nasdaq이 60일 고점 근처
- 향후 1개월 최소 drawdown이 -5% 이하
- 또는 1개월 종가수익률 -4% 이하
- 또는 1주 수익률 -2.5% 이하

사용 피처:

- Nasdaq, SOX, S&P500, Russell 모멘텀
- 고점 대비 거리
- 60일선 대비 위치
- VIX 저점권 complacency
- SOX 대비 Nasdaq 괴리
- Russell 대비 Nasdaq 괴리
- 신용/달러/금리 압력
- RAI 과열/붕괴
- ETF breadth
- Sentinel risk score

모델:

- Logistic Regression
- Random Forest
- 두 모델의 평균 확률
- expanding walk-forward 학습

최근 검증:

```text
samples: 1,682
positive_rate: 12.8%
accuracy: 73.0%
precision: 29.5%
recall: 80.0%
ROC-AUC: 0.801
```

해석:

- 고점 위험을 놓치지 않는 능력은 좋다.
- 대신 false alarm이 많다.
- 이 모델은 공격 매수 신호가 아니라 방어 경고 신호다.

## 7. Analog Macro Risk Model

실행:

```powershell
python scripts\analog_macro_risk_model.py --min-history-days 504 --exclude-recent-days 21 --neighbors 20,50,100 --retrain-step-days 21
```

로직:

1. 각 날짜의 매크로/위험 feature vector를 만든다.
2. 현재 feature를 과거 feature와 z-score 기준으로 비교한다.
3. 가장 유사한 과거 날짜 k개를 찾는다.
4. 그 날짜들의 이후 Nasdaq/SOX 1주, 1개월 수익률과 tail risk를 계산한다.
5. 유사국면 통계와 현재 위험축을 메타모델에 넣는다.

사용 k:

- 20
- 50
- 100

이 모델은 단독 방향 예측기보다는 현재 환경이 과거 위험 구간과 얼마나 비슷한지 보여주는 위험 가중치다.

## 8. Correction Timing Model

실행:

```powershell
python scripts\correction_timing_model.py --min-train-days 504 --retrain-step-days 63
```

타깃:

- Nasdaq 1주 -2% 이상
- Nasdaq 1개월 조정
- Nasdaq 지연형 1개월 조정
- SOX 1주 -3% 이상
- SOX 1개월 조정

사용 피처:

- Nasdaq/SOX/S&P/Russell 모멘텀
- 고점 대비 거리
- 60일선 대비 위치
- 변동성
- DXY, USDKRW, USDJPY
- US10Y, US2Y
- VIX, VXN, MOVE
- HY OAS
- Gold, WTI, Copper/Gold
- RAI
- ETF breadth
- Peak Fragility
- Analog Macro Risk
- Risk Vector 축들

최근 검증 중 가장 의미 있는 항목:

```text
Nasdaq 1개월 조정:
accuracy: 약 63.6%
precision: 약 37.5%
recall: 약 59.8%
ROC-AUC: 약 0.665
```

1주 예측은 아직 약하다. 1주 방향성은 노이즈가 커서 전체 정확도보다 고확신 신호만 따로 보는 방식이 더 현실적이다.

## 9. Walk-Forward Optimizer

실행:

```powershell
python scripts\risk_model_walkforward_optimizer.py
```

출력:

- `outputs/risk_model_walkforward_optimizer_latest/tables/current_optimized_risk_signal.csv`
- `outputs/risk_model_walkforward_optimizer_latest/tables/risk_probability_model_validation.csv`
- `outputs/risk_model_walkforward_optimizer_latest/tables/risk_probability_calibration_validation.csv`
- `outputs/risk_model_walkforward_optimizer_latest/tables/risk_threshold_optimization.csv`
- `outputs/risk_model_walkforward_optimizer_latest/tables/high_confidence_rule_validation.csv`
- `outputs/risk_model_walkforward_optimizer_latest/tables/false_alarm_taxonomy.csv`
- `outputs/risk_model_walkforward_optimizer_latest/tables/safe_asset_selector_validation.csv`
- `outputs/risk_model_walkforward_optimizer_latest/tables/current_safe_asset_recommendations.csv`
- `outputs/risk_model_walkforward_optimizer_latest/tables/fast_weekly_rank_summary.csv`
- `outputs/risk_model_walkforward_optimizer_latest/reports/risk_model_walkforward_optimizer_report.md`

이 레이어는 정확도 개선을 위해 모델 목적을 분리한다.

### 9.1 Risk-Off / Correction 타깃

검증 타깃:

- Nasdaq 1주 -2% 이상
- Nasdaq 1개월 조정
- Nasdaq 1개월 tail risk
- SOX 1주 -3% 이상
- SOX 1개월 조정
- KOSPI200 1주 -2% 이상
- KOSPI200 1개월 조정
- 위험자산 ETF 유니버스 1주 실전 손실
- 위험자산 ETF 유니버스 1개월 실전 손실
- 1주 안전자산 우위 필요
- 1개월 안전자산 우위 필요

실전 손실 라벨은 단순히 미래수익률이 0보다 작은지를 보지 않는다. 다음 조건을 같이 본다.

- 위험자산 유니버스의 forward drawdown
- 1주/1개월 미래수익률이 실전적으로 의미 있는 손실인지
- 안전자산 바스켓 대비 위험자산이 의미 있게 뒤처졌는지

이 라벨은 “맞혔다/틀렸다”보다 “그 시점에 위험자산을 줄였어야 했는가”에 더 가깝다.

### 9.1.1 Purged Walk-Forward / Embargo

현재 optimizer는 기본적으로 `purge-days=20`, `embargo-days=5`를 둔다.

이유는 1개월 forward return이나 forward drawdown 라벨이 서로 겹치기 때문이다. 어떤 시점의 미래 20거래일 성과를 라벨로 만들면, 바로 직전 날짜들의 라벨도 거의 같은 미래 구간을 공유한다. 이 상태에서 직전 데이터를 그대로 학습에 넣으면 실제보다 성능이 좋아 보일 수 있다.

따라서 각 test 구간을 예측할 때 학습 데이터 끝부분에서 purge/embargo 구간을 빼고 학습한다.

### 9.2 Threshold 최적화

threshold는 전체 데이터를 보고 고정하지 않는다. 각 시점에서 과거 데이터만 보고 expanding walk-forward 방식으로 선택한다.

목표함수는 손실회피 중심이다. recall과 precision뿐 아니라, 신호가 실제로 포착한 forward drawdown, 놓친 손실, 상승장에서 잘못 피한 opportunity cost까지 반영한다.

```text
objective =
2.2 * recall
+ 0.9 * precision
+ 0.25 * lift
+ 8.0 * caught_drawdown
- 6.0 * missed_drawdown
- 5.0 * false_alarm_upside
- 0.45 * signal_rate
```

이 방식은 전체 정확도보다 “위험을 놓치지 않되, 너무 자주 경고하지 않는 것”을 목표로 한다.

threshold도 단일 전역 threshold가 아니다. `Risk-On`, `Transition`, `Fragile`, `Risk-Off` regime별로 충분한 과거 표본이 있으면 해당 regime 안에서 별도 threshold를 고른다. 표본이 부족하면 전체 과거 데이터로 fallback한다.

### 9.2.1 자산군/Regime별 확률 Calibration

모델 원확률은 그대로 쓰지 않는다. walk-forward로 생성된 과거 예측확률과 실제 발생률을 이용해 calibration한다.

분리 기준:

- 타깃 자산군: `US growth`, `Semiconductor`, `Korea equity`, `Risk assets`, `Safe rotation`
- 현재 calibration group: `Risk-On`, `Transition`, `Fragile`, `Risk-Off`, `RAI fear`, `RAI overheat`, `Breadth break`

최신 검증에서는 보정 후 Brier score가 대부분 개선됐다. 예를 들어 2026-05-09 실행 기준:

```text
Nasdaq 1개월 조정:
raw Brier 약 0.2840 -> calibrated Brier 약 0.2063

위험자산 1개월 실전 손실:
raw Brier 약 0.2433 -> calibrated Brier 약 0.1498

1개월 안전자산 우위 필요:
raw Brier 약 0.2053 -> calibrated Brier 약 0.1209
```

### 9.3 False Alarm Taxonomy

false alarm을 모두 같은 오답으로 보지 않는다.

분류:

- `true_positive`: 실제 위험 발생
- `small_drawdown`: 큰 조정은 아니지만 작은 낙폭 발생
- `sideways`: 큰 움직임 없이 횡보
- `bad_false_alarm_up`: 경고 후 상승 지속

진짜 나쁜 오경보는 `bad_false_alarm_up`이다. `small_drawdown`과 `sideways`는 방어 신호로 어느 정도 허용할 수 있다.

### 9.4 Safe Asset Selector

Risk-Off 신호가 있을 때 안전자산 후보 중 무엇이 더 나은지 따로 검증한다.

후보 그룹:

- FX cash
- Cash/short bonds
- Gold
- Korea bonds
- US long bonds
- US IG bonds

현재 selector는 단순 모멘텀만 보지 않는다. 다음 두 축을 결합한다.

- 기술적 점수: 5일/20일/60일 모멘텀, 20일 변동성
- 매크로 점수: 환율 스트레스, 변동성, 신용/유동성, RAI, breadth, 공급충격, hedge demand

자산군별 macro sensitivity도 다르게 둔다.

예:

- FX cash: 환율/대외 스트레스, 변동성, 신용 스트레스
- Cash/short bonds: 위험회피 점수, 변동성, breadth 악화
- Gold: hedge demand, RAI 과열, 변동성, 공급충격
- Korea bonds: 조정 압력, 위험회피 점수, 변동성
- US long bonds: 조정 압력, hedge demand, 물가/공급충격 역방향
- US IG bonds: 조정 압력, 신용 스트레스 역방향

2026-05-09 실행 기준 현재 안전자산 추천 상위권은 한국채권과 단기채/CD금리형이다.

과거 검증 요약:

```text
1주:
Top1 hit-rate와 Top3 hit-rate를 별도 기록

3주:
Top1 hit-rate와 Top3 hit-rate를 별도 기록
```

안전자산 선택은 여전히 어려운 부분이다. 특히 금, 달러, 장기채는 같은 Risk-Off라도 트리거가 인플레인지, 성장충격인지, 환율충격인지에 따라 승자가 달라진다. 그래서 이 모델은 안전자산 전체를 하나로 묶지 않고 따로 랭킹한다.

### 9.5 Fast Weekly Risk-On Ranker

전체 186개 ETF의 주간 랭킹을 빠르게 검증한다.

로직:

- 5일/20일/60일/120일 모멘텀
- 20일 변동성 패널티
- cross-sectional z-score
- risk-off avoidance score로 위험구간 gate 적용
- `risk_off_avoidance_score >= 55`에서는 위험자산 랭킹 평가에서 제외

현재 성능:

```text
1주:
Risk-On 구간 TopK 평균수익률: 유니버스 평균수익률보다 높음

1개월:
Risk-On 구간 TopK 평균수익률: 유니버스 평균수익률보다 높음
```

즉 방향성 OX는 약하지만, 랭킹은 유니버스 평균 대비 초과수익을 만든다.

## 10. HTML Dashboard

실행:

```powershell
python scripts\generate_screening_dashboard_html.py --similar-days 50
```

출력:

- `outputs/screening_dashboard_latest/screening_dashboard.html`

HTML에는 다음이 포함된다.

- 현재 종합 위험점수
- Risk-Off 점수
- Peak Fragility
- Analog Macro Risk
- Correction Pressure
- RAI
- ETF breadth
- 미국/한국/일본/원자재 차트
- Risk Vector 축
- 유사 매크로 환경 30개 이상
- 자산군별 스크리닝 요약
- 상위 ETF 스크리닝
- walk-forward 최적화 현재 신호
- threshold 최적화 검증
- false alarm 분해
- 안전자산 선택 검증
- 빠른 Risk-On 랭킹 백테스트
- GAPS 바스켓 투자매력도
- 바스켓 내부 상위 ETF
- 바스켓 walk-forward 성능

## 11. GAPS 바스켓 스코어링

기존 모델은 ETF 전체를 일렬로 세워 점수를 매겼다. 2026-05-09 업데이트 이후에는 두 단계로 판단한다.

1. 먼저 10개 상위 바스켓 중 무엇이 좋은지 판단한다.
2. 그 다음 선택된 바스켓 안에서 어떤 ETF가 가장 좋은지 판단한다.

현재 바스켓 점수 산출물:

- `outputs/macro_regime_asset_screener_latest/tables/current_basket_scores.csv`
- `outputs/weekly_screening_rank_backtest_latest/tables/latest_basket_scores.csv`
- `outputs/weekly_screening_rank_backtest_latest/tables/latest_basket_constituent_scores.csv`
- `outputs/weekly_screening_rank_backtest_latest/tables/weekly_basket_backtest_summary.csv`

현재 바스켓 점수는 다음 요소를 결합한다.

- 바스켓 전체 ETF 평균 점수
- 바스켓 안 상위 5개 ETF 평균 점수
- 1주/1개월 보정 상승확률
- 최근 20일 수익률
- 위험 패널티

바스켓 백테스트는 매주 각 바스켓 점수를 만든 뒤, 다음 1주/1개월 실제 수익률이 가장 좋은 바스켓과 비교한다. 2026-05-09 실행 기준 바스켓 선택 성능은 다음과 같다.

```text
1주:
평가 주수: 885주
예측 1등 바스켓 평균수익률: 약 0.26%
전체 바스켓 평균수익률: 약 0.17%
실제 1등이 예측 Top3 안에 포함된 비율: 약 50.1%

1개월:
평가 주수: 885주
예측 1등 바스켓 평균수익률: 약 1.02%
전체 바스켓 평균수익률: 약 0.65%
실제 1등이 예측 Top3 안에 포함된 비율: 약 47.0%
```

즉 개별 ETF Top1을 정확히 맞히는 것보다, 먼저 유리한 바스켓을 좁힌 뒤 바스켓 내부에서 ETF를 고르는 방식이 더 안정적이다.

## 권장 실행 순서

데이터 다운로드까지 포함:

```powershell
python scripts\macro_regime_asset_screener.py
python scripts\daily_risk_off_sentinel.py --skip-download
python scripts\peak_fragility_model.py --min-train-days 504 --retrain-step-days 21
python scripts\risk_vector_dashboard.py
python scripts\analog_macro_risk_model.py --min-history-days 504 --exclude-recent-days 21 --neighbors 20,50,100 --retrain-step-days 21
python scripts\correction_timing_model.py --min-train-days 504 --retrain-step-days 63
python scripts\risk_vector_dashboard.py
python scripts\risk_model_walkforward_optimizer.py
python scripts\generate_screening_dashboard_html.py --similar-days 50
```

1995년부터 다시 받으려면 첫 줄은 다음처럼 명시한다.

```powershell
python scripts\macro_regime_asset_screener.py --start 1995-01-01
```

이미 캐시가 최신이면 이후 단계는 대부분 `--skip-download`를 사용한다.

바스켓 백테스트 산출물만 다시 만들 때는 다음을 실행한다.

```powershell
python scripts\basket_scoring_backtest.py
```

## 현재 모델의 한계

1주 OX 정확도는 구조적으로 높이기 어렵다. 시장의 1주 수익률은 뉴스, 포지셔닝, 옵션 만기, 단기 수급, 갭 상승/하락에 크게 흔들린다. 따라서 전체 표본 기준 80~90% 정확도는 현실적인 목표가 아니다.

현재 강점:

- Peak Fragility AUC가 약 0.80으로 고점 취약성 탐지는 쓸 만하다.
- 1개월 조정 탐지는 1주보다 유의미하다.
- Risk-On 랭킹은 유니버스 평균 대비 초과수익이 있다.
- RAI와 breadth를 분리해 과열과 공포를 다르게 해석한다.

현재 약점:

- 1주 급락/상승 OX는 아직 약하다.
- 안전자산 Top1 선택 정확도는 낮다.
- ETF 유니버스에 중복 노출이 많아 독립 표본 수가 실제보다 적다.
- 한국 상장 ETF 가격만으로 글로벌 RAI를 근사하므로 Credit Suisse 원본 RAI와는 다르다.
- false alarm은 방어형 모델 특성상 반드시 생긴다.

## 정확도를 더 높이려면 어떻게 수정해야 하는가

정확도 향상은 모델을 하나 더 복잡하게 만드는 것보다 라벨, 유니버스, threshold, 검증 구조를 더 정밀하게 바꾸는 쪽이 효과적이다.

### 1. 라벨을 실전형으로 더 바꾼다

현재보다 더 좋은 라벨:

```text
단순 상승:
미래수익률 > 0

개선 라벨:
미래수익률 > 거래비용 + 현금/단기채 대안수익률
또는 유니버스 평균 대비 초과수익 > 0
```

Risk-Off 라벨도 더 세분화한다.

```text
minor correction: 1주 -1%~-2%
tradeable correction: 1주 -2% 이상
monthly correction: 1개월 -3.5% 또는 intra-month -5.5%
tail event: 1개월 -6% 또는 intra-month -8%
crash: 1개월 -10% 이상
```

이렇게 나누면 작은 조정과 큰 급락을 같은 모델에 넣지 않아도 된다.

### 2. 자산군별 calibration을 분리한다

전체 ETF에 하나의 확률 보정을 쓰면 안 된다. 다음 단위로 따로 calibration해야 한다.

- 미국 성장/테크
- 미국 반도체
- 한국 반도체
- 한국 대표지수
- 중국/HK
- 일본
- 원자재
- 금
- 한국 채권
- 미국 장기채
- 달러/현금

이렇게 하면 확률의 과신이 줄어든다.

### 3. Risk-Off 모델과 Risk-On 랭킹 모델을 완전히 분리한다

현재도 분리되어 있지만 더 강하게 분리할 수 있다.

```text
Risk-Off 모델:
하락 위험만 예측
recall 우선
손실 회피 목적

Risk-On 랭킹 모델:
위험 신호가 낮을 때만 작동
상대수익률/TopK 성능 목적
```

Risk-Off 신호가 높은 날에는 공격자산 랭킹을 무시하거나 강하게 할인해야 한다.

### 4. 안전자산 선택 모델을 별도로 강화한다

현재 안전자산 Top1 적중률은 낮다. 개선하려면 안전자산 후보별로 다른 드라이버를 넣어야 한다.

금:

- 실질금리
- 달러
- VIX
- RAI collapse
- 지정학/유가 proxy

미국 장기채:

- 미국 10년물
- 실질금리
- MOVE
- 인플레 기대
- 경기민감 약화

달러/현금:

- DXY
- USDKRW
- HY OAS
- VIX
- 미국 2년물

한국 채권:

- 한국 금리 데이터
- 원/달러
- 미국 10년물
- 국내 인플레

현재는 안전자산을 단순 모멘텀/변동성 중심으로 고르기 때문에 정확도가 낮다. 안전자산별 macro driver model이 필요하다.

### 5. RAI 유니버스를 더 정제한다

현재 RAI는 186개 ETF와 글로벌 proxy를 함께 쓴다. 중복 ETF가 많다.

개선:

- 국가/자산군별 대표 ETF만 1개씩 선택
- 주식/채권/원자재/FX를 같은 비중으로 sector-neutral weighting
- 한국 상장 중복 ETF는 동일 노출끼리 평균 처리
- RAI를 global RAI, Korea RAI, US growth RAI, safe-asset RAI로 분리

이렇게 하면 RAI가 특정 ETF 군집에 끌려가는 문제가 줄어든다.

### 6. Feature selection을 타깃별로 다르게 한다

Nasdaq 1주 조정, KOSPI 1개월 조정, SOX 급락은 원인이 다르다.

예:

```text
Nasdaq:
VXN, US10Y_REAL, SOX, QQQ trend, RAI overheat

SOX:
SOX momentum, Nasdaq/SOX 괴리, AI/반도체 ETF breadth

KOSPI:
USDKRW, 외국인 수급 proxy, SOX, 중국, KOSDAQ/KOSPI

채권:
US10Y, 실질금리, MOVE, 인플레 기대
```

현재는 많은 피처를 공통으로 쓰기 때문에 일부 타깃에서 잡음이 늘어난다.

### 7. Threshold 목표함수를 손실 기반으로 바꾼다

현재 threshold objective는 recall, precision, signal_rate 중심이다. 실전적으로는 포트폴리오 손실함수를 써야 한다.

예:

```text
objective =
drawdown_saved
+ tail_loss_reduction
+ true_positive_profit
- false_alarm_opportunity_cost
- turnover_cost
```

즉 맞췄냐 틀렸냐보다, 틀렸을 때 돈이 얼마나 손해인지 기준으로 threshold를 골라야 한다.

### 8. Purged walk-forward와 embargo를 적용한다

현재도 과거 데이터만 사용하지만, 1개월 forward label은 인접 날짜끼리 미래 구간이 겹친다. 더 엄격하게 하려면 다음이 필요하다.

- train/test 사이 20거래일 embargo
- 같은 forward window가 겹치는 표본 제거
- 월별 또는 주별 리밸런싱 날짜만 검증

이렇게 하면 성능 수치는 낮아질 수 있지만 더 믿을 수 있다.

### 9. 앙상블을 regime별로 나눈다

같은 피처라도 regime마다 의미가 다르다.

분리 모델:

- Calm Risk-On
- Fragile Peak
- Correction Pressure
- Credit/Liquidity Shock
- FX/External Stress
- Full Risk-Off

각 regime마다 threshold와 feature importance가 달라야 한다.

### 10. 현재 우선순위

가장 효과가 큰 다음 수정 순서:

1. 안전자산 선택 모델을 macro-driver 기반으로 재작성한다.
2. RAI 유니버스를 대표자산 방식으로 정제한다.
3. 자산군별 calibration table을 만든다.
4. Risk-Off threshold objective를 손실 기반으로 바꾼다.
5. purged walk-forward / embargo 검증을 추가한다.
6. regime별 threshold를 따로 학습한다.
7. 빠른 주간 랭킹 백테스트에 거래비용과 turnover를 넣는다.
8. 고확신 구간만 따로 스크리닝 화면에 표시한다.

## 현재 판단 예시

2026-05-08 기준 최적화 레이어는 다음처럼 판단한다.

```text
위험회피 최적점수: 38.80
고점/조정 점수: 54.67
급락 Sentinel 점수: 4.10

Nasdaq 1주 -2% 신호: O
Nasdaq 1개월 조정 신호: O
SOX 1주 -3% 신호: O
KOSPI 1주 -2% 신호: X

최적화 액션:
De-risk: 신규 위험자산 축소, 안전자산 후보 우선
```

해석:

```text
급락장으로 확정할 신호는 약하다.
그러나 고점권 과열과 단기 조정 압력이 높다.
미국 성장/반도체는 단기 조정 리스크가 있고,
한국 시장은 미국보다 즉각적인 1주 급락 신호가 약하다.
```

## 주의

이 시스템은 투자 의사결정을 돕는 연구/스크리닝 도구다. 자동매매 신호가 아니며, 예측 확률은 실현 확률을 보장하지 않는다. 특히 1주 예측은 노이즈가 크므로, 포트폴리오에서는 포지션 크기, 손절 기준, 분산, 거래비용, 세금, 유동성을 별도로 관리해야 한다.
