# Macro Risk-Off Sentinel & ETF Screening System

이 프로젝트는 한국 상장 ETF 유니버스를 대상으로 매크로 환경, 시장 내부 breadth, 위험선호, 고점 취약성, 유사국면, 단기 조정 압력을 결합해 매주 또는 매일 현재 투자 매력도와 위험회피 필요성을 판단하는 스크리닝 시스템이다.

현재 활성 파이프라인에서는 LPPL/DTCAI를 사용하지 않는다. 이전 실험용 LPPL 스크립트와 캐시는 남아 있을 수 있지만, 현재 대시보드, Risk-Off Sentinel, Risk Vector, walk-forward 최적화 레이어의 점수 산출에는 LPPL이 들어가지 않는다.

2026-05-09 업데이트 이후 매크로/해외지수/환율/원자재 드라이버 패널은 `1995-01-01`부터 생성한다. 단, 한국 상장 ETF 자체 가격은 각 ETF 상장일 이전으로 존재하지 않으므로 ETF별 백테스트는 실제 상장 이후 구간에서만 평가된다.

위험자산 ETF 수익률 랭킹은 별도 모듈 `etf_leadership_model/`로 분리했다. 이 모듈은 시장 regime, Risk-Off Sentinel, 매크로 드라이버를 사용하지 않고 ETF 자체 상대강도와 구성종목 leadership/breadth만으로 기준지수 대비 초과수익 가능성을 랭킹한다.

## 핵심 목적

모델의 목표는 전체 시장 방향을 매일 80~90% 맞히는 것이 아니다. 1주 단위 상승/하락은 노이즈가 커서 전체 표본 기준으로 그런 정확도를 안정적으로 만들기 어렵다. 현재 구조의 목표는 다음처럼 분리되어 있다.

1. Risk-Off와 조정 위험을 가능한 한 빨리 감지한다.
2. 고점권에서 하락이 시작되기 전의 취약성을 잡는다.
3. Risk-Off 구간에서는 위험자산보다 안전자산 후보를 우선한다.
4. Risk-On 또는 중립 구간에서는 ETF 유니버스 안에서 상대적으로 더 나은 자산군을 랭킹한다.
5. 단순 OX 정확도보다 손실 회피, drawdown 감소, 고확신 구간 성능을 우선한다.

## 현재 모델 성능평가 요약

현재 모델은 "모든 ETF의 1주 상승/하락을 맞히는 단일 예측기"가 아니라, 위험회피 판단과 투자 후보 압축을 분리한 방어형 스크리닝 시스템이다. 따라서 성능도 목적별로 따로 봐야 한다.

### 1. Risk-Off 방어 성능

VIX 단일 지표와 비교한 최신 벤치마크 기준:

```text
전체 타깃 평균 ROC-AUC:
통합 모델 0.620
VIX 단일 0.604

포트폴리오 방어 타깃 평균:
통합 모델 ROC-AUC 0.718 / Recall 0.920
VIX 단일 ROC-AUC 0.564 / Recall 0.598
```

해석:

- 전체 방향 예측력은 VIX보다 약간 좋은 정도다.
- 하지만 KOSPI 하락, 위험자산 실전 손실, 안전자산 로테이션 필요 같은 방어 목적에서는 VIX보다 차이가 크다.
- Recall 0.920은 실제 위험 구간을 많이 잡는다는 뜻이다.
- 대신 false alarm이 생긴다. 이 모델은 "매수/매도 자동 신호"가 아니라 위험 노출을 줄일지 판단하는 조기경보 장치로 써야 한다.

### 2. 고점 취약성과 조정 압력

```text
Peak Fragility:
ROC-AUC 약 0.80
Recall 약 0.67~0.80
Precision 약 0.24~0.30

Nasdaq 1개월 조정:
ROC-AUC 약 0.66
Recall 약 0.60
Precision 약 0.38

Nasdaq 1주 하락:
ROC-AUC 약 0.53~0.57
```

해석:

- 고점권 취약성 탐지는 현재 모델의 강점이다.
- 1개월 단위 조정 위험은 어느 정도 유의미하게 본다.
- 1주 하락 OX는 아직 약하다. 1주 예측은 뉴스, 옵션 만기, 갭, 단기 수급 노이즈가 커서 고확신 구간만 따로 봐야 한다.

### 3. 바스켓/ETF 랭킹 성능

2026-05-09 이후 최신 GAPS ETF 유니버스와 바스켓 분류 기준의 주간 백테스트:

```text
바스켓 1주:
평가 주수 99주
예측 1등 바스켓 평균수익률 1.05%
전체 바스켓 평균수익률 0.44%
실제 1등이 예측 Top3 안에 포함된 비율 48.5%

바스켓 1개월:
평가 주수 99주
예측 1등 바스켓 평균수익률 3.72%
전체 바스켓 평균수익률 1.76%
실제 1등이 예측 Top3 안에 포함된 비율 58.6%

ETF Top10 1주:
예측 Top10 평균수익률 0.95%
전체 ETF 평균수익률 0.51%

ETF Top10 1개월:
예측 Top10 평균수익률 3.31%
전체 ETF 평균수익률 2.10%
```

해석:

- 개별 ETF Top1을 맞히는 모델이라기보다, 먼저 유리한 바스켓을 좁히고 그 안에서 ETF 후보를 고르는 방식이 더 안정적이다.
- 1개월 랭킹 성능이 1주보다 낫다.
- 1주 OX는 `O` 비율이 과도하게 높아 단독 의사결정 지표로 쓰면 안 된다.

### 4. 안전자산 선택 성능

Risk-Off 구간에서 안전자산 후보를 고르는 별도 모델의 최신 검증:

```text
1주:
Top1 적중률 16.7%
Top3 적중률 35.4%
예측 Top1 평균수익률 0.17%
안전자산 유니버스 평균수익률 0.05%

3주:
Top1 적중률 16.1%
Top3 적중률 34.3%
예측 Top1 평균수익률 0.37%
안전자산 유니버스 평균수익률 0.15%
```

해석:

- 안전자산 중 1등을 정확히 맞히는 능력은 아직 낮다.
- 그래도 평균 안전자산보다 높은 후보를 고르는 효과는 있다.
- 실제 운용에서는 Top1 몰빵보다 Top3 분산이 더 적합하다.

### 5. 현재 스크리닝 결과를 보는 순서

1. 먼저 `risk_off_avoidance_score`, `composite_vector_risk`, `peak_fragility`, `correction_pressure`를 본다.
2. 위험이 낮으면 바스켓 점수 Top3와 ETF Top10을 투자 후보군으로 본다.
3. 위험이 높거나 Transition이면 신규 위험자산 비중을 줄이고 안전자산/현금성 후보를 먼저 본다.
4. `upside_prob_1w`의 OX는 보조 지표다. 단독 매수 신호가 아니다.
5. RAI가 높고 Peak Fragility가 높으면 지수가 더 오르더라도 추격매수보다 리스크 관리가 우선이다.

현재 2026-05-08 기준 최적화 신호는 다음에 가깝다.

```text
model_regime: Transition
risk_off_avoidance_score: 38.77
composite_vector_risk: 28.52
peak_fragility: 74.07
correction_pressure: 68.82
RAI_z: 2.33
optimized_action: De-risk
```

즉 시스템성 폭락 경보는 아니지만, 고점 취약성과 단기 조정 압력이 높은 상태다. 신규 위험자산을 강하게 늘리기보다 바스켓/ETF 후보를 좁히고, 안전자산과 현금성 비중을 같이 검토하는 국면으로 해석한다.

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

바스켓 백테스트는 매주 각 바스켓 점수를 만든 뒤, 다음 1주/1개월 실제 수익률이 가장 좋은 바스켓과 비교한다. 2026-05-09 이후 최신 GAPS ETF 유니버스 기준 바스켓 선택 성능은 다음과 같다.

```text
1주:
평가 주수: 99주
예측 1등 바스켓 평균수익률: 약 1.05%
전체 바스켓 평균수익률: 약 0.44%
실제 1등이 예측 Top3 안에 포함된 비율: 약 48.5%

1개월:
평가 주수: 99주
예측 1등 바스켓 평균수익률: 약 3.72%
전체 바스켓 평균수익률: 약 1.76%
실제 1등이 예측 Top3 안에 포함된 비율: 약 58.6%
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
# 2026-05-10 ETF Leadership 실사용 업그레이드

## 구현 목적

위험자산 ETF의 수익률 예측 로직은 매크로/Regime을 제거하고, ETF 자체 가격 상대강도와 구성종목 리더십만으로 랭킹한다. 최종 포트폴리오 진입 여부는 별도 Risk-Off Sentinel 게이트로 제어한다.

```text
ETF 선택 = ETF Leadership Ranking Model
진입/축소 = Risk-Off Sentinel Gate
안전자산 선택 = 별도 안전자산 모델
```

## 새 실행 파일

```text
scripts/optimize_etf_leadership_selective_strategy.py
```

역할:

- 기존 ETF Leadership 룰 점수와 LightGBM Ranker 예측 결과를 읽는다.
- 2023-2024 구간에서만 진입 threshold를 고른다.
- 2025년 이후 구간은 out-of-sample test로 남겨 성능을 평가한다.
- 매주/매월 리밸런싱 기준으로 Top-K ETF를 선택한다.
- 확신도가 낮거나 Risk-Off Sentinel 위험신호가 많은 구간은 현금 대기 처리한다.
- 결과를 HTML 대시보드에 붙일 수 있도록 CSV로 저장한다.

## 출력 위치

```text
outputs/etf_leadership_selective_strategy/test_summary.csv
outputs/etf_leadership_selective_strategy/selected_trades.csv
outputs/etf_leadership_selective_strategy_risk_gated/test_summary.csv
outputs/etf_leadership_selective_strategy_risk_gated/selected_trades.csv
outputs/screening_dashboard_latest/screening_dashboard.html
```

## 2026-05-11 Institutional Tensor SSL V2 Foundation

상준 메모의 구조를 실제 코드 기반으로 옮기기 시작했다. 이번 단계는 head 모델을 다시 학습하기 전의 기반 공사다.

구현 파일:

```text
scripts/institutional_feature_tensor_ssl_v2.py
```

설치 추가:

```text
pyarrow
```

구현된 단계:

```text
1. Feature Store
   - macro / etf / safe 패널을 date, asset, role 키로 통일
   - parquet 저장
   - macro join은 backward asof 방식

2. Train-only scaling
   - tensor 생성 단계에서만 scaler를 fit
   - macro/ETF는 train_end=2022-12-31
   - safe는 원천 패널이 2024년 이후라 safe 전용 train_end=2025-12-31로 별도 생성

3. Multi-window tensor
   Risk-Off Sentinel:
   - macro 20/40/64/126일 window

   ETF Leadership:
   - ETF/internal 10/20/40/64일 window

   Safe Asset:
   - safe 20/60일 window
   - 120일은 현재 DB GAPS 안전자산 패널 기간이 짧아 생성 불가

4. SSL Encoder V2
   - PatchTST-style masked reconstruction
   - TS2Vec-style contrastive branch
   - reconstruction loss + InfoNCE loss

5. Regime / Confidence
   - KMeans VQ regime code
   - regime별 forward return / label prior statistics
   - RealNVP 기반 Normalizing Flow confidence
```

실행:

```powershell
python scripts\institutional_feature_tensor_ssl_v2.py --stage feature-store --roles macro,etf,safe --train-end 2022-12-31
python scripts\institutional_feature_tensor_ssl_v2.py --stage tensors --roles macro,etf,safe --train-end 2022-12-31
python scripts\institutional_feature_tensor_ssl_v2.py --stage ssl --roles macro,etf,safe --train-end 2022-12-31 --epochs 1 --flow-epochs 1 --max-train-samples 6000 --batch-size 256
python scripts\institutional_feature_tensor_ssl_v2.py --stage tensors --roles safe --train-end 2025-12-31
python scripts\institutional_feature_tensor_ssl_v2.py --stage ssl --roles safe --train-end 2025-12-31 --epochs 1 --flow-epochs 1 --max-train-samples 6000 --batch-size 256
```

생성된 feature store:

```text
outputs/institutional_tensor_ssl_v2_latest/parquet/macro_panel.parquet
outputs/institutional_tensor_ssl_v2_latest/parquet/etf_panel.parquet
outputs/institutional_tensor_ssl_v2_latest/parquet/safe_panel.parquet
```

생성된 tensor:

```text
macro_w20  X=(8169, 20, 76)
macro_w40  X=(8149, 40, 76)
macro_w64  X=(8125, 64, 76)
macro_w126 X=(8063, 126, 76)

etf_w10    X=(16197, 10, 32)
etf_w20    X=(15957, 20, 32)
etf_w40    X=(15477, 40, 32)
etf_w64    X=(14901, 64, 32)

safe_w20   X=(3786, 20, 24)
safe_w60   X=(1758, 60, 24)
```

SSL V2 embedding 생성 결과:

```text
etf   10/20/40/64 window: each 24-dim embedding, 24 VQ states
macro 20/40/64/126 window: each 24-dim embedding, 24 VQ states
safe  20/60 window: each 24-dim embedding, 24 VQ states
```

핵심 산출물:

```text
outputs/institutional_tensor_ssl_v2_latest/tables/ssl2_embedding_summary.csv
outputs/institutional_tensor_ssl_v2_latest/tables/macro_w64_ssl2_embeddings.csv
outputs/institutional_tensor_ssl_v2_latest/tables/etf_w20_ssl2_embeddings.csv
outputs/institutional_tensor_ssl_v2_latest/tables/safe_w20_ssl2_embeddings.csv
```

아직 남은 단계:

```text
1. SSL V2 embedding을 Risk-Off V4/V5 head에 연결
2. ETF Leadership Ranker에 10/20/40/64 window embedding concat
3. Safe Macro Ranker에 20/60 window embedding concat
4. walk-forward로 기존 V4/V5 대비 성능 비교
5. 성능이 좋아지는 window만 운영 모델에 채택
6. safe 120일 window는 장기 안전자산 프록시 데이터로 보강
7. TTA/UDA는 마지막에 rolling scaler/calibration/adapter만 제한 적용
```

## 2026-05-11 SSL2 Head Walk-Forward 검증

SSL2 embedding을 만든 뒤 바로 운영에 넣지 않고, 기존 head 대비 성능을 walk-forward로 비교했다.

구현 파일:

```text
scripts/backtest_ssl2_head_models.py
```

산출물:

```text
outputs/ssl2_head_backtest_latest/tables/risk_ssl2_metrics.csv
outputs/ssl2_head_backtest_latest/tables/etf_ssl2_backtest_summary.csv
outputs/ssl2_head_backtest_latest/tables/safe_ssl2_backtest_summary.csv
outputs/ssl2_head_backtest_latest/tables/operational_model_adoption.csv
outputs/ssl2_head_backtest_latest/tables/operational_model_adoption.json
```

검증 구조:

```text
Risk-Off:
- baseline macro/sentinel features
- baseline + macro SSL2 20/40/64/126/multi-window
- train <= 2022, valid 2023~2024, test >= 2025

ETF Leadership:
- baseline ETF/internal ranker
- baseline + ETF SSL2 10/20/40/64/multi-window
- train <= 2018, valid 2019~2021, test >= 2022

Safe Asset:
- baseline macro-conditioned safe ranker
- baseline + safe SSL2 20/60/multi-window
- train <= 2024, valid 2025, test 2026
```

채택 판단:

```text
Risk-Off label_large_loss_1m:
baseline AUC 0.974
best SSL2 w126 AUC 0.993
결론: 채택

Risk-Off label_large_loss_1w:
baseline AUC 0.904
best SSL2 w126 AUC 0.907
하지만 baseline recall 1.00, SSL2 recall 0.975
결론: baseline 유지

Risk-Off label_nasdaq_down_1m:
baseline AUC 0.749
best SSL2 multi AUC 0.804
recall 0.725 -> 0.908
결론: 채택

Risk-Off label_nasdaq_down_1w:
baseline AUC 0.848
best SSL2 w64 AUC 0.857
하지만 recall 하락
결론: baseline 유지

ETF Leadership 1M:
baseline Sharpe 1.445
best도 baseline
결론: SSL2 미채택

ETF Leadership 1W:
baseline Sharpe 1.318
best도 baseline
결론: SSL2 미채택

Safe Asset 1W:
baseline beat safe average 64.3%
SSL2 w20 71.4%
avg picked return 0.78% -> 1.33%
결론: 채택 후보

Safe Asset 1M:
baseline beat safe average 35.7%
SSL2 w20 42.9%
avg picked return -1.63% -> -0.09%
결론: 개선은 있으나 아직 약함
```

운영 판단:

- SSL2는 Risk-Off 1개월/나스닥 1개월 하락 감지에는 유효하다.
- SSL2는 ETF Leadership에는 현재 미채택이다. ETF는 구조화 피처와 룰/Ranker가 더 안정적이다.
- SSL2는 안전자산 1주 선택에는 개선이 있다. 다만 test 기간이 14개 주로 짧으므로 바로 A급으로 볼 수 없다.
- 다음 보강은 SSL epoch 증가가 아니라, 먼저 안전자산 장기 데이터 확장과 ETF embedding 과적합 제어다.

2026-05-11 재검증에서 운영 채택표를 스크립트가 자동 생성하도록 보강했다.

```text
python scripts/backtest_ssl2_head_models.py --tasks risk,etf,safe
```

최종 채택:

```text
Risk-Off 1개월 대형손실: SSL2 macro 126일 window 채택
Risk-Off 1주 대형손실: baseline 유지
Risk-Off Nasdaq 1개월 하락: SSL2 macro multi-window 채택
Risk-Off Nasdaq 1주 하락: baseline 유지
ETF Leadership 1주/1개월: baseline V5 유지
Safe Asset 1주/1개월: SSL2 safe 20일 window 채택 후보
```

## 현재 가장 좋은 테스트 결과

2025년 이후 out-of-sample 기준이다.

1주 선택형 모델:

```text
모델: static_rule
구조: 현재 구성종목 근사 룰베이스
Top-K: 1개
진입조건: confidence score 52.5% 이상 + Risk-Off 위험신호 2개 이하
진입비율: 47.6%
Sharpe: 1.99
상승 적중률: 56.7%
초과수익 적중률: 56.7%
MDD: -17.4%
```

1개월 선택형 모델:

```text
모델: static_blend
구조: 현재 구성종목 근사 리더십 점수 + Ranker 블렌드
Top-K: 5개
진입비율: 87.5%
Sharpe: 2.30
상승 적중률: 78.6%
초과수익 적중률: 64.3%
MDD: -5.0%
```

주의:

- `static_blend`는 현재 구성종목을 과거에도 적용한 근사 백테스트라 성능 상한 확인용이다.
- `strict_blend`는 구성종목 룩어헤드 위험을 줄인 보수적 성능 확인용이다.
- 재학습 후 Sharpe 3과 hit ratio 80%에는 아직 도달하지 못했다. 현재 최선은 1개월 `static_blend`의 Sharpe 2.30 / 상승 적중률 78.6%다.

## 현재 스크리닝 해석

2026-05-08 기준 ETF 리더십 상위는 AI 전력, 미국/한국 반도체, 나스닥/테크 계열이다. 하지만 Risk-Off 게이트는 `Transition` 상태이며 현재 액션은 다음과 같다.

```text
De-risk: 신규 위험자산 축소, 안전자산 후보 우선
```

따라서 현재 화면은 이렇게 읽는다.

```text
리더십 자체는 성장/반도체가 강하다.
하지만 고점 취약성과 조정 압력이 높아 신규 위험자산 진입은 축소한다.
진입하더라도 고확신 Top-K만 제한적으로 보고, 안전자산 바스켓을 함께 우선 검토한다.
```

## 다음 우선순위

1. ETF별 공식 운용사 holdings 수집기를 붙여 static approximation 의존도를 낮춘다.
2. Yahoo 검색으로 잘못 매핑된 구성종목 ticker를 정리한다.
3. LightGBM Ranker 모델을 5D와 20D로 분리 저장하고 현재 스크리닝에도 ML 점수를 직접 붙인다.
4. 거래비용, 회전율, 보유 중 risk gate 발동 시 청산 규칙을 백테스트에 반영한다.
5. strict 기준 Sharpe 3 / 상승 적중률 80%에 도달하는지 다시 검증한다.

---

## 2026-05-10 ETF Leadership V3 업그레이드

이번 버전은 과거 구성종목 변화를 추적하지 않고, **현재 구성종목을 2019년 이후에도 동일하게 적용하는 static holdings approximation**으로 학습한다. 대신 현재 구성종목들의 가격 히스토리를 최대한 보강해 `고점근접도`, `52주 고점 90% 이상 비중`, `60/200일선 위 구성종목 비중`이 실제로 계산되도록 수정했다.

### 변경된 핵심 구조

1. 가격 히스토리 보강
   - `scripts/repair_component_price_history.py`
   - 필요 티커 3,091개 점검
   - 가격 매트릭스 편입 2,366개
   - 실패 725개는 잘못된 야후 티커, 비상장/비가격 구성항목, 펀드 내부 코드 가능성이 높다.
   - 결과 파일:
     - `outputs/component_price_history_coverage.csv`
     - `outputs/component_price_history_repair_failed.csv`
     - `outputs/etf_leadership_from_cache/prices_adj_close.csv`

2. HP/Breadth 계산 수정
   - 파일: `etf_leadership_model/features.py`
   - 글로벌 혼합 캘린더 때문에 rolling window가 전부 NaN이 되는 문제를 수정했다.
   - 가격은 짧은 휴장 공백만 `ffill(limit=5)`로 보정한다.
   - `MA60`: 60일 창, 최소 45개 관측치
   - `MA200`: 200일 창, 최소 150개 관측치
   - `52주 고점`: 252일 창, 최소 180개 관측치

3. 바스켓/테마 내부 랭킹
   - 파일: `scripts/train_static_etf_leadership_v3.py`
   - 반도체는 반도체끼리, 배당/방어는 배당/방어끼리, 채권은 채권끼리 비교한다.
   - LightGBM Ranker group은 `date + ranking_group` 기준이다.
   - 전체 ETF를 한 날짜에 한 덩어리로 묶던 방식보다 테마/자산군 효과를 줄인다.

4. 1주 모델은 룰/필터 중심
   - 1주 수익률은 노이즈가 커서 Ranker를 최종 신호로 쓰지 않는다.
   - `rule_5d_score`가 기본 리더십 점수다.
   - 별도 `entry_prob_5d` 메타모델이 “이번 주 진입할 만한가”를 판단한다.

5. 1개월 모델은 Ranker 사용
   - `forward_20D_excess`를 바스켓 내부 랭킹 라벨로 변환한다.
   - LightGBM `LGBMRanker`는 같은 날짜, 같은 테마 ETF들을 하나의 query group으로 학습한다.
   - 최종적으로 `rule_20d_score`, `ranker_score`, `blend_20d_score`, `entry_prob_20d`를 모두 저장한다.

### Feature 사용 가능성 개선

기존에는 HP와 MA200 breadth가 거의 전부 비어 있었다. V3 feature 재생성 후 사용 가능 비율은 다음과 같다.

```text
weighted_HP         non-null 약 90.0%
HP90_share          non-null 약 90.0%
MA60_breadth        non-null 약 90.2%
MA200_breadth       non-null 약 90.0%
component RS 20D    non-null 약 90.2%
RS positive share   non-null 약 90.2%
```

### 최신 out-of-sample 성능 요약

검증 기준은 2025년 이후 테스트 구간이다. 아직 샘플 수가 짧기 때문에 “확정 성능”이 아니라 현재 구조의 검증 결과로 봐야 한다.

```text
1주 룰 기반 Top5:
Sharpe 3.53
상승 적중률 67.2%
초과수익 적중률 53.7%
MDD -8.1%

1주 진입필터 Top3/Top5:
진입비율 35.8%
상승 적중률 79.2%
초과수익 적중률 62.5%
Sharpe 약 2.12~2.14
```

해석:
- 1주에서는 항상 들어가는 룰 기반 Top5가 Sharpe는 가장 높았다.
- 다만 “진입/대기” 필터를 걸면 거래 횟수는 줄고 상승 적중률은 79% 근처까지 올라간다.
- 사용 목적이 공격적 수익률이면 룰 Top5, 실전 손실 회피/선별 진입이면 entry-adjusted Top3/Top5가 더 맞다.

```text
1개월 룰 기반 Top5:
Sharpe 2.83
상승 적중률 81.3%
초과수익 적중률 50.0%
MDD -4.0%

1개월 blend Top3:
Sharpe 2.56
상승 적중률 68.8%
초과수익 적중률 62.5%
MDD -6.1%
```

해석:
- 1개월은 Ranker 단독보다 룰 또는 룰+Ranker blend가 낫다.
- Ranker가 `MA200_breadth`, `median_component_return_60D`, `ETF_RS_120D`를 사용하기 시작한 것은 긍정적이다.
- 하지만 현재 test sample이 16개월뿐이라 threshold를 강하게 최적화하면 과최적화 위험이 크다.

### 현재 남은 약점

1. 실패 티커 725개
   - 구성종목 일부는 아직 가격 히스토리가 없다.
   - 특히 해외 ETF 내부의 비상장 클래스, 현금성 항목, 잘못된 코드가 섞여 있다.

2. 1개월 Ranker의 학습 강도
   - LightGBM best iteration이 1에서 멈췄다.
   - 이는 바스켓 내부 샘플이 작고, 2021~2022 학습 구간에서 feature/label 신호가 약하다는 뜻이다.
   - 지금은 Ranker를 단독 매수 신호로 쓰기보다 룰 점수 보조로 쓰는 게 맞다.

3. 1주 초과수익 적중률
   - 상승 적중률은 entry filter로 79%까지 올라가지만, 기준지수 대비 초과수익 적중률은 62.5% 수준이다.
   - 즉 “오를 ETF”는 꽤 걸러도, “벤치마크보다 더 오를 ETF”는 아직 더 어렵다.

### 다음 개선 우선순위

1. 실패 티커 정리
   - `outputs/component_price_history_repair_failed.csv`에서 실제 종목인데 매핑이 틀린 것부터 수동 보정한다.

2. 바스켓별 개별 threshold
   - 반도체/테크, 배당/방어, 채권, 원자재의 최적 진입 threshold가 다르다.
   - 현재는 전체 공통 threshold라 보수적이다.

3. 1개월 Ranker를 바스켓별로 분리
   - 현재는 group만 나누고 모델은 하나다.
   - 데이터가 충분한 바스켓은 별도 ranker, 부족한 바스켓은 룰 기반으로 유지하는 hybrid가 더 안정적이다.

4. 현재 스크리닝 HTML 반영
   - `v3_current_basket_scores_1w.csv`
   - `v3_current_basket_scores_1m.csv`
   - 위 두 파일을 대시보드에 붙이면 “큰 바스켓 점수”와 “바스켓 내부 ETF 점수”를 동시에 볼 수 있다.

---

## 2026-05-10 ETF Leadership V4 보강

V4는 V3의 두 약점, 즉 구성종목 가격 누락과 Ranker 학습 샘플 부족을 보강한 버전이다.

### 1. 구성종목 실패 티커 해결

추가 스크립트:

```text
scripts/resolve_failed_component_prices.py
```

처리 방식:

1. 기존 실패 티커 725개를 holdings의 `component_name`과 다시 연결한다.
2. 수동 보정 맵과 Yahoo Finance search API를 같이 사용해 실제 상장 심볼을 찾는다.
3. 찾은 심볼의 가격을 받아 원래 component ticker alias 파일로 저장한다.
4. 스왑, CP, 내부 코드처럼 독립 가격이 없는 항목은 ETF 가격 프록시로 대체한다.
5. 최종 repaired holdings를 별도 파일로 저장한다.

결과:

```text
입력 실패 티커: 725개
실제 가격 복구: 704개
프록시 처리 대상: 21개
최종 가격 매트릭스 편입: 3,089 / 3,100개
남은 실패: 11개
```

남은 11개는 repaired holdings에서 모두 weight 0인 항목이라 현재 ETF 내부 점수에는 실질 영향이 없다.

주요 출력:

```text
data/etf_holdings_static_2019_repaired.csv
outputs/component_price_resolved_aliases.csv
outputs/component_price_proxy_map.csv
outputs/component_price_history_coverage_after_resolve.csv
outputs/component_price_history_repair_failed_after_resolve.csv
outputs/etf_leadership_static_holdings_repaired_v4base/rule_scores.csv
```

가격 보강 후 feature coverage:

```text
weighted_HP          non-null 약 96.4%
HP90_share           non-null 약 96.4%
MA60_breadth         non-null 약 96.7%
MA200_breadth        non-null 약 96.5%
component RS 20D     non-null 약 96.8%
RS positive share    non-null 약 96.8%
```

### 2. Ranker 학습 샘플 부족 보강

기존 V3는 `date + 세부 테마` 단위로 Ranker group을 만들었다. 이 방식은 의미는 좋지만, 일부 세부 테마의 ETF 수가 2~4개에 불과해 LightGBM이 충분히 학습하기 어렵다.

V4에서는 다음처럼 바꿨다.

```text
학습 group: date + 큰 바스켓(asset_basket)
최종 선택: 세부 테마(ranking_group)별 대표 후보를 먼저 뽑고, 그 후보들끼리 Top-K 선택
```

즉 학습은 더 큰 바스켓에서 샘플 수를 확보하고, 실제 선택에서는 반도체/배당/채권/원자재 같은 세부 테마 쏠림을 제한한다.

또한 Ranker 학습 시작일을 2021년에서 2020년으로 앞당겨 학습 샘플을 늘렸다.

```text
V3 학습 행 수: 약 32,537
V4 학습 행 수: 약 47,435
```

### 3. V4 out-of-sample 성능

2025년 이후 테스트 기준:

```text
1개월 Ranker Top5:
Sharpe 3.15
상승 적중률 87.5%
초과수익 적중률 56.3%
MDD -4.8%

1개월 Ranker Top2:
Sharpe 3.11
상승 적중률 87.5%
초과수익 적중률 75.0%
MDD -2.1%

1개월 Ranker Top1:
Sharpe 2.19
상승 적중률 75.0%
초과수익 적중률 81.3%
MDD -7.4%
```

1주 룰 기반:

```text
1주 Rule Top5:
Sharpe 3.82
상승 적중률 69.1%
초과수익 적중률 51.5%
MDD -9.9%

1주 Rule Top2:
Sharpe 3.21
상승 적중률 67.6%
초과수익 적중률 57.4%
MDD -12.6%
```

해석:

- 1개월 Ranker는 가격 보강과 큰 바스켓 학습 전환 후 확실히 좋아졌다.
- 특히 1개월 Top2는 Sharpe 3 이상, 상승 적중률 87.5%, 초과수익 적중률 75%로 현재 목표에 가장 가깝다.
- 1주는 여전히 Ranker보다 룰 기반이 낫다. 단기 구간은 노이즈가 커서 “항상 Top-K”보다 risk-off sentinel 및 별도 진입 필터와 같이 쓰는 것이 맞다.
- 1개월 entry-adjusted 모델은 이번 V4에서는 과도하게 보수적으로 변해 성능이 낮다. 현재는 1개월 최종 선택에 `ranker_score`를 우선 쓰고, entry model은 참고 지표로만 둔다.

### 4. 현재 V4 스크리닝 결과 파일

```text
outputs/etf_leadership_static_v4_repaired/v3_current_basket_scores_1w.csv
outputs/etf_leadership_static_v4_repaired/v3_current_basket_scores_1m.csv
outputs/etf_leadership_static_v4_repaired/v3_backtest_summary.csv
```

현재 1주 기준 강한 바스켓/테마:

```text
국내섹터: Korea cyclical
해외섹터: US semiconductor
해외지수: US growth, Japan equity
국내지수: Korea growth
```

현재 1개월 기준 강한 바스켓/테마:

```text
국내섹터: Korea cyclical
해외섹터: US semiconductor
해외지수: US growth, Japan equity
국내지수: Korea growth
```
## 2026-05-11 SSL/VQ 기반 모델 보강

상준 메모의 아이디어를 바로 운영 가능한 형태로 1차 구현했다. 이번 버전은 세 모델에 공통으로 자기지도 시계열 임베딩을 붙인다.

### 구현한 구조

공통 임베딩 엔진:

```text
scripts/ssl_time_series_embeddings.py
```

- PatchTST 계열 아이디어를 단순화한 masked patch reconstruction encoder
- 입력 시계열을 window/patch로 나누고 일부 patch를 mask한 뒤 복원 학습
- 학습 후 `ssl_emb_00...` 임베딩 생성
- 임베딩을 MiniBatchKMeans로 묶어 `ssl_vq_state` 생성
- PCA + LedoitWolf covariance 기반 Gaussian likelihood로 `ssl_flow_nll`, `ssl_flow_confidence` 생성
- 각 VQ state별 과거 forward return/hit prior를 expanding 방식으로 생성해 룩어헤드 방지

주의: 현재 NF는 RealNVP/MAF 같은 full normalizing flow가 아니라, 임베딩 공간의 확률적 신뢰도를 계산하는 Gaussian-flow style likelihood다. Concept Whitening/SAE/TTA는 아직 운영 모델에 넣지 않았다.

### 1. Risk-Off Sentinel SSL 보강

```text
scripts/train_ssl_enhanced_risk_off.py
outputs/ssl_risk_off_sentinel_latest/tables/ssl_risk_off_predictions.csv
outputs/ssl_risk_off_sentinel_latest/tables/ssl_risk_off_metrics.csv
```

기존 risk-off sentinel 피처에 macro SSL 임베딩, VQ state, flow confidence를 추가했다. 목표 라벨은 위험자산 프록시의 실전 손실 기준이다.

```text
1주 실전손실 라벨: 위험자산 평균 5D forward return <= -2%
1개월 실전손실 라벨: 위험자산 평균 20D forward return <= -5%
```

현재 out-of-sample 성능:

```text
1주 loss AUC: 0.638
1개월 loss AUC: 0.808
```

해석:

- 1개월 위험 순위화 능력은 의미 있게 개선됐다.
- 다만 현재 threshold는 보수적으로 잡혀 test recall이 낮다.
- 다음 단계는 AUC가 아니라 recall/false alarm trade-off 기준으로 threshold를 재최적화해야 한다.

### 2. ETF Leadership V5 SSL

```text
scripts/train_static_etf_leadership_v5_ssl.py
outputs/etf_leadership_static_v5_ssl/v5_ssl_backtest_summary.csv
outputs/etf_leadership_static_v5_ssl/v5_ssl_current_basket_scores_1w.csv
outputs/etf_leadership_static_v5_ssl/v5_ssl_current_basket_scores_1m.csv
```

V4의 ETF 리더십 feature에 ETF별 SSL 임베딩 29개를 추가했다.

추가된 대표 피처:

```text
ssl_emb_00...ssl_emb_15
ssl_vq_state
ssl_vq_distance
ssl_flow_nll
ssl_flow_confidence
forward_20D_excess_state_mean_prior
forward_20D_excess_state_hit_prior
forward_20D_excess_state_count_prior
```

2025년 이후 out-of-sample 핵심 결과:

```text
1개월 Ranker Top2:
CAGR 158.9%
Sharpe 3.02
상승 적중률 87.5%
초과수익 적중률 62.5%
MDD -5.5%

1주 Rule Top5:
CAGR 111.3%
Sharpe 3.82
상승 적중률 69.1%
초과수익 적중률 51.5%
MDD -9.9%
```

V5에서 Ranker가 많이 사용한 SSL 계열 피처:

```text
forward_20D_excess_state_mean_prior
forward_20D_excess_state_hit_prior
forward_20D_return_state_mean_prior
ssl_emb_13
forward_20D_excess_state_count_prior
ssl_emb_06
ssl_vq_distance
```

해석:

- 1개월 ETF 리더십 모델은 SSL/VQ state prior를 실제로 강하게 사용한다.
- V4보다 수익 포텐셜은 커졌지만 MDD는 커졌다.
- 현재 실전형 선택은 `1개월 Ranker Top2 + Risk-Off Sentinel 게이트` 조합이 가장 타당하다.
- 1주는 여전히 ML Ranker보다 rule/필터 중심이 낫다.

### 3. 안전자산 선택 SSL 모델

```text
scripts/train_ssl_safe_asset_selector.py
outputs/ssl_safe_asset_selector_latest/tables/ssl_safe_asset_predictions.csv
outputs/ssl_safe_asset_selector_latest/tables/ssl_safe_asset_summary.csv
```

Risk-off 시점에 어떤 안전자산을 고를지 학습하는 별도 모델을 추가했다. 후보군은 초단기/현금성, 달러, 국내채권, 해외장기채, 해외IG, 금, 방어주로 나눈다.

현재 결과:

```text
Risk-off test 기간: 5개
1주 beat safe average rate: 20.0%
1개월 beat safe average rate: 80.0%
```

해석:

- 1개월은 방향성이 있으나 표본이 5개뿐이라 신뢰하기 어렵다.
- 1주는 현재 성능이 부족하다.
- 안전자산 모델은 더 긴 weekly panel과 위기구간 표본 확장이 필요하다.

### 다음 최적화 우선순위

1. Risk-Off Sentinel은 threshold를 AUC 기준이 아니라 손실회피 목적함수로 재학습한다.
2. 1개월 ETF Ranker는 SSL V5를 유지하되, risk-off probability가 일정 수준 이상이면 진입을 막는다.
3. 안전자산 모델은 2019년 이전까지 가능한 가격 히스토리를 붙이고 risk-off episode를 늘린다.
4. full NF, TS2Vec contrastive loss, TTA는 현재 구조가 안정화된 뒤 별도 실험 브랜치에서 추가한다.
5. 최종 포트폴리오는 `위험 탐지 모델`, `위험자산 리더십 모델`, `안전자산 선택 모델`을 분리해서 운영한다.

## 2026-05-11 Risk-Off V2 / 3D 축 / 안전자산 Macro Ranker

요청한 보강 중 다음을 별도 산출물로 구현했다.

```text
scripts/optimize_risk_off_3d_and_safe_assets.py
outputs/institutional_risk_off_v2_latest/tables/
```

### Risk-Off V2 구조

Risk-Off를 단일 점수가 아니라 3개 원인축으로 분해했다.

```text
축 1: 변동성/신용 스트레스
  VIX, VXN, MOVE, HY OAS, IG OAS, HYG/IEF, 기존 volatility/credit score

축 2: 달러/환율/유동성 스트레스
  DXY, USDKRW, USDCNH, USDJPY, NFCI/ANFCI, 기존 fx/liquidity score

축 3: 고점취약성/과열 피로도
  peak fragility, correction pressure, analog risk, RAI collapse/overheat, breadth damage
```

추가로 기존 peak fragility가 실제 고점 초입을 충분히 잡지 못해 아래 축을 새로 넣었다.

```text
complacent_peak_fragility_stress
= 60일 모멘텀 강함
+ 252일 고점 근접
+ 200일선 대비 과열
+ 낮은 실현변동성
+ 낮은 기존 스트레스
```

최종 Risk-Off V2 확률은 다음 ensemble이다.

```text
Risk-Off V2 probability
= 45% HistGradientBoostingClassifier probability
+ 55% 3년 rolling stress percentile
```

threshold는 AUC 최대화가 아니라 실전 손실회피 목적함수로 고른다.

```text
목적함수:
놓친 하락 비용 크게 부여
precision 보상
false alarm 감점
alert rate 과다 발생 감점
```

### Risk-Off V2 Walk-Forward 결과

1998년 이후 나스닥/S&P500/SOX 프록시를 사용해 2003년부터 연도별 expanding walk-forward로 검증했다.

```text
1주 대형손실:
평균 AUC 0.565
평균 recall 0.370
평균 precision 0.102
평균 false alarm rate 0.324
평균 alert rate 0.328

1개월 대형손실:
평균 AUC 0.589
평균 recall 0.368
평균 precision 0.080
평균 false alarm rate 0.405
평균 alert rate 0.408
```

구간별 특징:

```text
1개월 2003-2009 AUC: 0.658
1개월 2010-2019 AUC: 0.547
1개월 2020-2026 AUC: 0.575
```

해석:

- 3D 축과 stress percentile을 붙여 기존보다 구조는 강해졌다.
- 하지만 “큰 하락 recall 80% 이상 + false alarm 제한” 목표는 아직 달성하지 못했다.
- 현재 데이터/라벨 기준으로 recall 80%를 강제로 맞추면 alert rate가 지나치게 높아져 항상 위험하다고 말하는 모델이 된다.
- 따라서 이 버전은 A등급 운용모델이 아니라 B/B+급 risk overlay로 봐야 한다.

### 현재 Risk-Off V2 상태

```text
기준일: 2026-05-08
1주 Risk-Off V2 확률: 0.512 / threshold 0.495 / alert 1
1개월 Risk-Off V2 확률: 0.416 / threshold 0.385 / alert 1
기존 risk_off_score: 8.12
주된 원인축: 고점취약성/과열 피로도
축1 변동성/신용: 3.93
축2 달러/유동성: 27.04
축3 고점취약성: 37.68
```

해석:

- 기존 stress score는 낮지만, 고점권/과열 피로도 축이 더 높다.
- 즉 “이미 터진 risk-off”라기보다 “고점권 취약성 경보”에 가깝다.

### 안전자산 Macro-Conditioned Ranker

DB GAPS 안전자산 후보 전체를 포함했다.

```text
Gold
FX cash
Cash/short bonds
Korea bonds
US long bonds
US IG bonds
Korea defensive
```

사용한 macro feature:

```text
미국 10년물/2년물 변화
실질금리
DXY
USDKRW
VIX
HY OAS
금/달러 상대강도
장기채/단기채와 유사한 금리축
원화채/미국채 구분을 위한 그룹/바스켓 더미
Risk-Off 3D 축
Risk-Off V2 확률
```

검증 결과:

```text
1개월 전체:
기간 56개
선택 안전자산 평균 수익률 +3.09%
안전자산 평균 대비 target 승률 71.4%

1주 전체:
기간 56개
선택 안전자산 평균 수익률 +0.68%
안전자산 평균 대비 target 승률 57.1%
```

현재 1개월 안전자산 상위:

```text
TIGER 로우볼
ACE KRX금현물
KODEX 국고채30년액티브
KODEX 장기종합채권(AA-이상)액티브
TIGER 국고채30년스트립액티브
KODEX 은선물(H)
```

주의:

- 안전자산 ranker는 DB GAPS 주간 패널이 2024-05-16부터 2026-04-03까지라 표본이 아직 짧다.
- 1개월 안전자산 선택은 개선됐지만, 1주는 아직 약하다.
- 기관급으로 올리려면 2019년 이후 안전자산 가격 패널을 직접 재구성해 risk-off episode를 더 늘려야 한다.

## 2026-05-11 Risk-Off V3: 3D + SSL + Episode Objective

Risk-Off V2에서 한 단계 더 보강했다.

```text
Risk-Off V3 = 3D risk axes + stress percentile + macro SSL/VQ/NF features
Safe Ranker V3 = macro features + macro SSL/VQ/NF + safe-asset SSL/VQ/NF
```

### 반영한 요청 항목

```text
1. Risk-Off용 macro window tensor 생성: 완료
2. ETF Leadership용 internal window tensor 생성: 완료
3. PatchTST-style masked reconstruction encoder 구현: 완료
4. macro_emb, etf_emb 저장: 완료
5. VQ regime code 생성: 완료
6. regime별 forward return/drawdown 통계 생성: ETF/safe prior 완료
7. NF confidence 2단계 추가: Gaussian-flow style confidence 완료
8. 기존 Risk-Off / ETF Ranker / Safe Ranker에 embedding feature 추가: 완료
9. walk-forward로 V4 대비 성능 비교: 완료
```

산출물:

```text
outputs/institutional_risk_off_v2_latest/tables/risk_off_v2_walkforward_metrics.csv
outputs/institutional_risk_off_v2_latest/tables/risk_off_v2_walkforward_predictions.csv
outputs/institutional_risk_off_v2_latest/tables/macro_conditioned_safe_asset_summary.csv
outputs/institutional_risk_off_v2_latest/tables/current_safe_asset_recommendations_v2.csv
outputs/institutional_risk_off_v2_latest/tables/model_v4_v5_ssl_comparison_summary.csv
```

### Risk-Off V3 성능

날짜별 OX만 보면 risk-off는 희소 라벨이라 precision이 낮다. 그래서 “하락 episode 시작 전 20거래일 내 경보를 냈는가”를 별도 목적함수로 추가했다.

```text
1주 Risk-Off:
AUC 0.561
일간 recall 0.485
episode recall 20D 0.734
precision 0.105
false alarm rate 0.427
alert rate 0.432

1개월 Risk-Off:
AUC 0.568
일간 recall 0.417
episode recall 20D 0.745
precision 0.104
false alarm rate 0.427
alert rate 0.432
```

해석:

- 선제 episode 포착률은 약 73~75%까지 올라왔다.
- 목표였던 80%에는 아직 못 미친다.
- false alarm도 42% 수준이라 “A급 기관 운용”이라고 부르기엔 아직 높다.
- 다만 이전의 일간 OX 중심 모델보다 “하락 시작 전 경보” 관점은 더 실전에 맞다.

### ETF Leadership V4 vs V5 SSL

```text
ETF Leadership V4 1M Top2:
Sharpe 3.115
상승 적중률 87.5%
초과수익 적중률 75.0%
MDD -2.1%

ETF Leadership V5 SSL/VQ 1M Top2:
Sharpe 3.016
상승 적중률 87.5%
초과수익 적중률 62.5%
MDD -5.5%
```

해석:

- SSL/VQ는 수익률 포텐셜은 키웠지만, 리스크 조정 성능은 V4보다 낮았다.
- 현재 실전 선택은 V5 단독보다 V4/V5 ensemble + Risk-Off 게이트가 맞다.

### Safe Ranker V3 성능

```text
1개월 Safe Ranker:
안전자산 평균 대비 승률 75.9%
선택 안전자산 평균 수익률 +3.19%

1주 Safe Ranker:
안전자산 평균 대비 승률 66.7%
선택 안전자산 평균 수익률 +0.94%
```

이전보다 1주 안전자산 선택이 개선됐다. 단, 주간 패널이 짧기 때문에 아직 A급 검증이라고 보기는 어렵다.

현재 1개월 안전자산 상위:

```text
KODEX 국고채30년액티브
TIGER 국고채30년스트립액티브
KODEX 장기종합채권(AA-이상)액티브
TIGER 로우볼
KODEX 은선물(H)
TIGER 200 헬스케어
KODEX 미국달러선물
ACE KRX금현물
```

### 현재 등급

```text
ETF Leadership: B+ ~ A-
Risk-Off V3: B+
안전자산 선택 1개월: A-
안전자산 선택 1주: B
전체 통합: B+ ~ A-
기관급 프로토타입: 맞음
기관급 운용 시스템: 아직 아님
```

다음 병목:

1. Risk-Off episode recall 80%를 넘기려면 일간 라벨이 아니라 drawdown event dataset을 별도로 만들어야 한다.
2. false alarm을 줄이려면 alert를 “Watch / De-risk / Cash” 3단계로 분리해야 한다.
3. ETF V5 SSL은 V4보다 리스크가 커져서 ensemble과 MDD penalty가 필요하다.
4. Safe Ranker는 2019년 이후 안전자산 가격 패널을 직접 재구축해야 표본 부족이 줄어든다.

## 2026-05-11 포트폴리오 리밸런싱 검증 룰

사용자가 제시한 DB GAPS 편입비중 상한표를 `scripts/portfolio_rebalance_validator.py`에 하드 제약으로 반영했다.

### 적용한 하드 제약

```text
개별 ETF/상품 최대 비중: 20%

위험자산 전체 최대 비중: 70%
국내주식_지수  -> 국내지수              최대 30%
국내주식_섹터  -> 국내섹터              최대 15%
해외주식_지수  -> 해외지수              최대 30%
해외주식_섹터  -> 해외섹터              최대 10%
FX 및 원자재   -> FX및 원자재           최대 20%

안전자산 전체 최대 비중: 100%
국내채권_종합          최대 50%
국내채권_회사채        최대 30%
해외채권_종합          최대 50%
해외채권_회사채        최대 30%
금리연계형/초단기채권  -> 금리연계형 및 초단기채권 최대 50%
```

총 투자비중은 100%로 맞춘다. 다만 후보 ETF가 부족해 위 제약을 깨지 않고 100%를 채울 수 없는 날짜는 `미배분 KRW 현금`을 명시 포지션으로 넣는다. 이 현금은 ETF 상품이 아니므로 개별 ETF 20% 상한 검증 대상에서 제외한다.

### 리밸런싱 로직

1. `weekly_calibrated_rank_panel.csv`에서 위험자산/안전자산 후보를 읽는다.
2. 위험자산은 기존 ETF/바스켓 스코어의 cross-sectional percentile로 랭킹한다.
3. 안전자산은 가능하면 `macro_conditioned_safe_asset_predictions.csv`의 안전자산 ranker 점수를 사용한다.
4. Risk-Off V2 1개월 확률을 이용해 위험자산 목표비중을 동적으로 낮춘다.

```text
Risk-Off 확률 >= 0.65: 위험자산 15%, 안전자산/현금 85%
Risk-Off 확률 >= 0.50: 위험자산 35%, 안전자산/현금 65%
Risk-Off 확률 >= 0.38 또는 alert: 위험자산 50%, 안전자산/현금 50%
그 외: 위험자산 70%, 안전자산/현금 30%
```

이 값은 목표 비중이고, 이미지의 바스켓별 상한은 항상 우선 적용한다.

### 최신 검증 결과

실행:

```powershell
python scripts\portfolio_rebalance_validator.py
```

결과:

```text
검증 기간: 2024-05-16 ~ 2026-04-03
주간 리밸런싱 날짜 수: 102
제약 위반 날짜: 0
누적수익률: +82.21%
CAGR: +35.78%
MDD: -8.36%
Sharpe: 2.41
주간 양수 수익률 비율: 65.69%
평균 위험자산 비중: 49.95%
평균 안전자산 비중: 47.70%
평균 미배분 KRW 현금: 2.35%
```

최신 포트폴리오 예시는 2026-04-03 기준 Risk-Off 확률 0.438로 위험자산 50%, 안전자산 50% 목표가 적용됐다.

주요 산출물:

```text
outputs/portfolio_rebalance_validator_latest/tables/portfolio_constraint_rules.csv
outputs/portfolio_rebalance_validator_latest/tables/weekly_constrained_allocations.csv
outputs/portfolio_rebalance_validator_latest/tables/weekly_constraint_validation.csv
outputs/portfolio_rebalance_validator_latest/tables/weekly_constrained_portfolio_returns.csv
outputs/portfolio_rebalance_validator_latest/tables/weekly_constrained_portfolio_summary.csv
outputs/portfolio_rebalance_validator_latest/tables/latest_constrained_portfolio.csv
```

## 2026-05-11 Risk-Off V4: 실전 손실 이벤트 라벨 재설계

Risk-Off V3의 병목은 라벨이었다. 기존 라벨은 날짜별 `향후 5일 -3%`, `향후 20일 -7% 또는 최대낙폭 -9%` 중심이라 “하락이 이미 진행된 날짜”와 “하락 전에 피해야 하는 날짜”가 섞였다. V4에서는 Risk-Off를 단순 OX가 아니라 실전 손실 이벤트로 다시 정의했다.

실행:

```powershell
python scripts\risk_off_v4_event_label_retrainer.py
```

### 새 라벨

1주 라벨:

```text
label_event_loss_1w = 1 if
위험 프록시 5거래일 forward 최소낙폭 <= -2.5%
또는 Nasdaq/S&P500/SOX 중 최악 5거래일 forward 최소낙폭 <= -4.5%
또는 위험 프록시 5거래일 forward 수익률 <= -3.0%
```

1개월 라벨:

```text
label_event_loss_1m = 1 if
위험 프록시 20거래일 forward 최소낙폭 <= -6.0%
또는 Nasdaq/S&P500/SOX 중 최악 20거래일 forward 최소낙폭 <= -9.5%
또는 위험 프록시 20거래일 forward 수익률 <= -5.5%
또는 위험 프록시 40거래일 forward 최소낙폭 <= -8.5%
```

위험 프록시는 Nasdaq100, S&P500, SOX를 함께 쓴다. 목적은 “정확히 내일 하락하나”가 아니라 “지금부터 가까운 미래에 위험자산 손실 이벤트가 시작될 확률이 높은가”를 잡는 것이다.

### Threshold 재설계

V4는 단일 threshold가 아니다.

```text
Watch   : 선제 감시. 놓치지 않는 것이 목적.
De-risk : 위험자산 축소. 실제 포트폴리오 비중 조절 기준.
Cash    : 강한 방어. 위험자산 15% 이하로 낮추는 기준.
```

threshold 선택 목적함수도 바꿨다.

```text
목적함수 =
선제 event recall
+ 잡은 손실 비율
+ 일간 recall
+ precision
- false alarm
- 놓친 손실 비율
- 상승장 기회비용
- 과도한 alert rate penalty
```

또한 1개월 forward label의 중복을 줄이기 위해 purged walk-forward와 embargo를 적용했다.

### V3 대비 성능

```text
Risk-Off V3 previous 1개월:
event recall 20D 73.5%
precision 9.8%
false alarm 41.5%

Risk-Off V4 1개월:
Watch event recall 20D 98.1%
De-risk event recall 20D 81.0%
Cash event recall 20D 52.6%
De-risk precision 28.7%
De-risk false alarm 54.7%
De-risk caught loss ratio 52.4%

Risk-Off V3 previous 1주:
event recall 20D 72.7%
precision 10.4%
false alarm 42.3%

Risk-Off V4 1주:
Watch event recall 20D 94.0%
De-risk event recall 20D 80.2%
Cash event recall 20D 50.8%
De-risk precision 27.9%
De-risk false alarm 50.2%
De-risk caught loss ratio 60.2%
```

해석:

- 선제 감시 성능은 A급 기준에 가까워졌다. Watch 기준으로 1개월 손실 이벤트의 98.1%, 1주 손실 이벤트의 94.0%를 사전 구간에서 잡는다.
- 실제 비중 축소용 De-risk 기준도 1개월 81.0%, 1주 80.2%로 목표 80%를 넘겼다.
- 대신 false alarm은 아직 높다. 이것은 “놓친 하락을 가장 비싸게 둔” 설정의 비용이다.
- 따라서 V4는 자동 현금화 모델이 아니라 `Watch -> De-risk -> Cash` 단계형 운용 게이트로 써야 한다.

### 현재 상태

2026-05-08 기준:

```text
1주 Risk-Off V4: De-risk
확률 0.584 / De-risk threshold 0.378

1개월 Risk-Off V4: Cash
확률 0.420 / Cash threshold 0.383

주된 원인축: 고점취약성/과열 피로도
기존 risk_off_score 자체는 낮지만, 고점취약성 축이 높아 V4가 더 보수적으로 반응한다.
```

### 리밸런싱 연결

`scripts/portfolio_rebalance_validator.py`는 이제 V4가 있으면 V4 stage를 우선 사용한다.

```text
Normal  : 위험자산 목표 70%
Watch   : 위험자산 목표 50%
De-risk : 위험자산 목표 35%
Cash    : 위험자산 목표 15%
```

제약 검증은 여전히 DB GAPS 편입비중 상한을 우선한다.

최신 리밸런싱 검증:

```text
검증 기간: 2024-05-16 ~ 2026-04-03
제약 위반 날짜: 0
누적수익률: +56.37%
CAGR: +25.60%
MDD: -7.28%
Sharpe: 2.07
평균 위험자산 비중: 33.53%
평균 안전자산 비중: 64.12%
평균 미배분 KRW 현금: 2.35%
```

V2보다 수익률과 Sharpe는 낮아졌지만 MDD와 위험자산 노출은 줄었다. V4는 수익 극대화가 아니라 손실 이벤트 회피 목적의 방어 게이트다.

주요 산출물:

```text
outputs/risk_off_v4_event_label_latest/tables/risk_off_v4_event_label_panel.csv
outputs/risk_off_v4_event_label_latest/tables/risk_off_v4_walkforward_predictions.csv
outputs/risk_off_v4_event_label_latest/tables/risk_off_v4_walkforward_metrics.csv
outputs/risk_off_v4_event_label_latest/tables/current_risk_off_v4_state.csv
outputs/risk_off_v4_event_label_latest/tables/risk_off_v3_v4_comparison.csv
```

## 2026-05-11 장기 프록시 백테스트와 V4 Adaptive 보강

실제 DB GAPS ETF는 상장일과 구성종목 데이터 제약 때문에 장기 검증이 짧다. 그래서 Risk-Off 게이트 자체는 ETF 대신 장기 지수/매크로 프록시로 검증했다.

실행:

```powershell
python scripts\long_horizon_risk_off_proxy_backtest.py
```

검증 조건:

```text
기간: 2003-01-01 ~ 2026-05-08
위험자산 프록시: Nasdaq100 40%, S&P500 30%, SOX 20%, Russell2000 10%
안전자산 프록시: 현금, 미국 장기채 합성, 금 원화환산 프록시, 달러 현금 프록시
비교군: 위험자산 Buy&Hold, 60/40, VIX 단일 게이트, V3 게이트, V4 게이트, V4 adaptive
```

장기 성능:

```text
V4 adaptive:
CAGR 23.17%
Sharpe 2.17
MDD -17.56%
누적수익률 +15,440%

V4 combined defensive:
CAGR 16.84%
Sharpe 1.92
MDD -12.87%
누적수익률 +4,232%

V3 gate:
CAGR 21.09%
Sharpe 1.87
MDD -11.99%

VIX 단일 게이트:
CAGR 20.54%
Sharpe 1.87
MDD -11.21%

위험자산 Buy&Hold:
CAGR 13.31%
Sharpe 0.69
MDD -55.83%
```

해석:

- V4 defensive는 최대낙폭을 줄이는 데 강하다.
- V4 adaptive는 장기 Sharpe와 CAGR이 가장 좋다.
- VIX 단일 게이트는 GFC/Covid 같은 충격형 위기에서는 강하지만, 고점취약성/환율/RAI 기반 선제 구간은 설명하지 못한다.
- V4 adaptive의 약점은 GFC 같은 시스템 위기에서 V3/VIX보다 MDD가 커질 수 있다는 점이다.

V4 adaptive 보강:

```text
기존 V4:
Cash면 위험자산 15%

보강 V4 adaptive:
Cash라도 변동성/신용 스트레스 < 30,
달러/유동성 스트레스 < 35,
고점취약성 축만 우세,
VIX < 28이면
전면 Cash가 아니라 고점 경계로 보고 위험자산 70%까지 허용한다.
```

이유는 장기 검증에서 “고점취약성만 높은 Cash”가 수익을 과도하게 포기하는 구간으로 확인됐기 때문이다. 따라서 실전 운용은 두 모드로 나눈다.

```text
방어 우선: V4 combined defensive
수익/Sharpe 우선: V4 adaptive
```

산출물:

```text
outputs/long_horizon_risk_off_proxy_backtest_latest/tables/long_horizon_proxy_daily_panel.csv
outputs/long_horizon_risk_off_proxy_backtest_latest/tables/long_horizon_proxy_strategy_summary.csv
outputs/long_horizon_risk_off_proxy_backtest_latest/tables/long_horizon_proxy_annual_returns.csv
outputs/long_horizon_risk_off_proxy_backtest_latest/tables/long_horizon_proxy_crisis_windows.csv
outputs/long_horizon_risk_off_proxy_backtest_latest/tables/long_horizon_proxy_v4_stage_exposure.csv
```

## 2026-05-11 장기 ETF 리더십/안전자산 선택 프록시 백테스트

실제 DB GAPS ETF는 상장일과 과거 구성종목 데이터 제약 때문에 2010년대 초반부터 동일 조건으로 검증할 수 없다. 그래서 장기 검증은 `SPY/QQQ/SMH/SOXX/EWY/EEM/섹터 ETF/GLD/DBC` 등 오래 상장된 프록시 ETF로 수행하고, 실제 GAPS ETF 구성종목 리더십 모델은 별도 단기 실전 유니버스 검증으로 해석한다.

실행:

```powershell
python scripts\long_history_proxy_selection_backtest.py --start 2009-01-01 --backtest-start 2010-01-04 --end 2026-05-11 --initial-test-year 2014
```

검증 구조:

```text
ETF Leadership:
- 해외 ETF는 NASDAQ/QQQ 대비 상대강도와 forward excess return 기준
- 룰 점수: 20/60/120일 초과수익, RS slope, 고점근접도, MA60/MA200, 변동성 페널티
- ML: LightGBM LGBMRanker, 날짜별 ETF 그룹 랭킹, walk-forward

Safe Asset Selection:
- 후보: BIL, SHY, IEF, TLT, AGG, BND, LQD, TIP, MBB, GLD, UUP, FXY, FXF
- 룰 점수: 모멘텀, 변동성, 낙폭, 고점근접도
- macro-conditioned 점수: 금리 상승/하락, VIX, HY OAS, DXY, USDKRW, Risk-Off V4 3축에 따라 장기채/현금/달러/금/엔·스위스프랑 가중치 조정
- ML Ranker도 테스트했지만 안전자산은 현재 룰+매크로 조건부 점수가 더 안정적
```

핵심 결과:

```text
ETF Leadership 장기 프록시
- Ranker 주간 Top3: 2016-01-08 ~ 2026-05-08
  CAGR 23.41%, Sharpe 0.95, MDD -45.64%, QQQ 대비 주간 초과승률 50.65%
- Ranker 월간 Top5: 2016-01-29 ~ 2026-04-30
  CAGR 19.78%, Sharpe 1.02, MDD -23.20%, QQQ 대비 월간 초과승률 44.72%
- Rule 월간 Top5: 2010-01-29 ~ 2026-05-08
  CAGR 14.71%, Sharpe 0.99, MDD -21.78%

안전자산 선택 장기 프록시
- Macro-conditioned 주간 Top1: 2014-01-03 ~ 2026-05-08
  CAGR 7.26%, Sharpe 0.64, MDD -17.29%
- Macro-conditioned 주간 Risk-Off Top1:
  CAGR 5.57%, Sharpe 0.63, MDD -17.92%
- Rule 월간 Top3: 2010-01-29 ~ 2026-05-08
  CAGR 3.21%, Sharpe 0.55, MDD -11.32%
```

해석:

- 리더십 모델은 “절대수익이 좋은 ETF 묶음”을 고르는 데는 작동한다. 다만 2010년대 미국 성장주 강세장에서 QQQ 자체가 매우 강한 벤치마크라, QQQ 초과수익을 안정적으로 내는 수준은 아직 아니다.
- 리더십 Rank IC는 20일 Ranker 기준 평균 `0.047`, 양수 비율 `54.0%`로 약한 예측력이 있다. 실전에서는 Top1보다 Top3~Top5 분산이 낫다.
- 안전자산 선택은 ML Ranker보다 룰/매크로 조건부가 낫다. 안전자산은 가격 리더십보다 “금리하락형 방어인지, 금리상승형 방어인지, 달러 유동성 쇼크인지”가 더 중요하다.
- 안전자산 모델의 현재 등급은 아직 A가 아니다. 장기채/달러/금 선택의 방향성은 잡았지만, 국면별 손실 회피 기준과 금리 상승 충격 구간의 장기채 회피를 더 강화해야 한다.

산출물:

```text
outputs/long_history_proxy_selection_backtest_latest/tables/proxy_raw_prices.csv
outputs/long_history_proxy_selection_backtest_latest/tables/proxy_leadership_features_predictions.csv
outputs/long_history_proxy_selection_backtest_latest/tables/proxy_safe_features_predictions.csv
outputs/long_history_proxy_selection_backtest_latest/tables/proxy_selection_backtest_trades.csv
outputs/long_history_proxy_selection_backtest_latest/tables/proxy_selection_backtest_summary.csv
outputs/long_history_proxy_selection_backtest_latest/tables/proxy_selection_rank_ic.csv
outputs/long_history_proxy_selection_backtest_latest/README.md
```

## 2026-05-11 Selective ETF Leadership V3 / Safe Macro V3

요청한 방향대로 `항상 투자`가 아니라 `좋은 구간만 투자`하는 선택형 구조를 추가했다.

구현 파일:

```text
scripts/optimize_selective_leadership_and_safe_v3.py
```

산출물:

```text
outputs/selective_leadership_safe_v3_latest/tables/etf_selective_v3_summary.csv
outputs/selective_leadership_safe_v3_latest/tables/etf_selective_v3_trades.csv
outputs/selective_leadership_safe_v3_latest/tables/etf_selective_v3_threshold_grid.csv
outputs/selective_leadership_safe_v3_latest/tables/safe_v3_summary.csv
outputs/selective_leadership_safe_v3_latest/tables/safe_v3_predictions.csv
outputs/selective_leadership_safe_v3_latest/tables/safe_v3_importance_1m.csv
outputs/selective_leadership_safe_v3_latest/tables/safe_v3_importance_1w.csv
```

ETF Leadership V3 구조:

- 여러 후보를 동시에 만든다: `rule_5d`, `rule_20d`, `ranker`, `blend`, `entry_adjusted`, Top-K `1/2/3/5`.
- 각 리밸런싱 날짜마다 후보별 `score spread`, `top-k 평균 점수`, `breadth`, `HP`, `RS`, 내부 회귀 피처를 집계한다.
- GradientBoosting classifier가 `이번 주/이번 달 Top-K가 초과수익과 절대수익을 동시에 낼지`를 학습한다.
- GradientBoosting regressor가 실전 utility를 예측한다.
- 날짜별로 가장 높은 `dynamic_selection_score` 후보를 선택하고, validation에서 정한 threshold 미만이면 대기한다.
- 룩어헤드 방지를 위해 `train <= 2018`, `validation 2019~2021`, `test >= 2022` 구조를 썼다.

ETF Leadership V3 test 결과:

```text
1W dynamic selective:
기간 213주, 투자 113주, coverage 53.1%
CAGR 26.0%, Sharpe 1.22, MDD -19.8%
거래 초과수익 적중률 54.9%, 거래 상승 적중률 57.5%

1M dynamic selective:
기간 52개월, 투자 24개월, coverage 46.2%
CAGR 2.3%, Sharpe 0.22, MDD -20.7%
거래 초과수익 적중률 54.2%, 거래 상승 적중률 54.2%
```

판단:

- 선택형 gate를 구현했지만, 장기 test에서는 기존 `DB GAPS 장기상장 ETF 리더십 1개월 Rule Top5`보다 낮다.
- 기존 1개월 Rule Top5는 CAGR 38.3%, Sharpe 1.54, MDD -14.7%, 상승 적중률 71.2%, 초과수익 적중률 59.6%였다.
- 따라서 현재 운영 기본값은 선택형 V3가 아니라 기존 Rule Top5가 더 낫다.
- 선택형 모델의 validation 성능은 높았지만 test로 넘어오며 깨졌으므로, 아직 A급 entry gate가 아니다.

Safe Macro V3 구조:

- 기존 `macro_conditioned_safe_asset_panel`에 금리, 실질금리, DXY, USDKRW, VIX, HY, GOLD, HYG/IEF, Risk-Off 3축을 사용한다.
- 안전자산 그룹/바스켓 더미와 매크로 축의 상호작용 피처를 추가했다.
- LightGBM Ranker로 1주/1개월 안전자산 상대 순위를 재학습했다.
- 현재 확보된 GAPS 주간 패널이 2024년 이후 중심이라 test 기간은 짧다.

Safe Macro V3 test 결과:

```text
1W:
periods 14
avg picked return 1.13%
beat safe average rate 64.3%
top-k overlap 35.7%

1M:
periods 14
avg picked return 0.09%
beat safe average rate 64.3%
top-k overlap 16.7%
```

판단:

- 안전자산 V3는 평균 안전자산보다 나은 것을 고르는 비율은 64% 수준으로 개선 여지가 있다.
- 1개월 안전자산 모델은 아직 약하다. 금리 상승 충격 구간에서 장기채를 과대선택하는 문제가 남아 있다.
- 주요 중요 피처는 `score_0_100`, `calibrated_prob_1w`, `Gold x Risk-Off`, `Korea bonds x US10Y 20d change`, `USDKRW x overseas bonds`다.

최신 HTML 대시보드에도 위 결과를 연결했다:

```text
outputs/screening_dashboard_latest/screening_dashboard.html
```
