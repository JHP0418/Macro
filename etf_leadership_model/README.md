# ETF Leadership Ranking Model

이 모듈은 위험자산 ETF 수익률 예측에서 시장 regime과 매크로 변수를 완전히 제외하고, ETF 자체 주도력과 구성종목 내부 강도만으로 ETF를 랭킹한다.

## 입력 파일

`etf_universe.csv`

```text
etf_ticker,market,benchmark_ticker
069500.KS,KR,^KS11
QQQ,US,^NDX
```

필수 컬럼:

- `etf_ticker`
- `market`: `KR` 또는 `US`
- `benchmark_ticker`: 국내 ETF는 KOSPI 기준, 해외 ETF는 NASDAQ 기준

`etf_holdings.csv`

```text
date,etf_ticker,component_ticker,weight
2026-05-08,069500.KS,005930.KS,0.25
2026-05-08,QQQ,AAPL,0.08
```

필수 컬럼:

- `date`
- `etf_ticker`
- `component_ticker`
- `weight`: 0~1 사이 비중

holdings가 매일 없으면 각 feature date에서 해당 ETF의 가장 최근 holdings를 사용한다.

## 실행

```powershell
python -m etf_leadership_model.main `
  --universe data\etf_universe.csv `
  --holdings data\etf_holdings.csv `
  --start 2015-01-01 `
  --feature-frequency daily `
  --rebalance-frequency W-FRI `
  --train-end 2021-12-31 `
  --valid-end 2022-12-31 `
  --top-k 5
```

LightGBM 없이 룰베이스만 실행:

```powershell
python -m etf_leadership_model.main --skip-ml
```

## 출력 파일

기본 출력 위치:

```text
outputs/etf_leadership_model/
```

출력:

- `features.csv`
- `rule_scores.csv`
- `model_predictions.csv`
- `backtest_results.csv`
- `backtest_summary.csv`
- `feature_importance.csv`
- `prices_adj_close.csv`

## Feature

ETF 가격 기반:

- `ETF_RS_20D`
- `ETF_RS_60D`
- `ETF_RS_120D`
- `RS_slope_20D`

구성종목 고점근접도:

- `weighted_HP`
- `median_HP`
- `HP90_share`
- `HP_change_20D`

구성종목 상대모멘텀:

- `weighted_component_RS_20D`
- `weighted_component_RS_60D`
- `median_component_RS_20D`
- `RS_positive_share`

Breadth:

- `MA60_breadth`
- `MA200_breadth`
- `Breadth_change_20D`

수익률 분포:

- `median_component_return_20D`
- `median_component_return_60D`
- `mean_minus_median_return_20D`
- `top20_component_return_mean`
- `bottom20_component_return_mean`

집중도:

- `holding_count`
- `effective_N`
- `top5_weight_share`
- `top10_weight_share`
- `top5_return_contribution_share`

## Concentrated ETF 처리

구성종목 수가 적은 ETF는 breadth가 통계적으로 불안정하므로 로직을 다르게 쓴다.

```text
holding_count <= 5 또는 effective_N < 8:
concentrated logic

effective_N >= 20:
diversified logic

그 외:
mid logic
```

Concentrated ETF는 ETF 자체 상대강도, 가중 구성종목 상대모멘텀, 고점근접도를 중심으로 본다. Breadth와 내부 확산도는 거의 쓰지 않는다.

Diversified ETF는 breadth, HP90 share, median return, concentration penalty까지 적극 반영한다.

## Rule Score

```text
ETF_RS_Score =
0.4*z(ETF_RS_20D)
+0.4*z(ETF_RS_60D)
+0.2*z(ETF_RS_120D)

HP_Score =
0.5*z(weighted_HP)
+0.3*z(HP_change_20D)
+0.2*z(HP90_share)

Component_Momentum_Score =
0.4*z(weighted_component_RS_20D)
+0.4*z(weighted_component_RS_60D)
+0.2*z(median_component_RS_20D)

Breadth_Score =
0.35*z(MA60_breadth)
+0.25*z(MA200_breadth)
+0.25*z(RS_positive_share)
+0.15*z(Breadth_change_20D)
```

최종 룰 점수는 concentrated, mid, diversified ETF에 따라 다른 식을 적용한다.

## Target

```text
forward_5D_return = ETF_price[t+5] / ETF_price[t] - 1
forward_20D_return = ETF_price[t+20] / ETF_price[t] - 1

forward_5D_excess =
ETF forward_5D_return - benchmark forward_5D_return

forward_20D_excess =
ETF forward_20D_return - benchmark forward_20D_return
```

LightGBM Ranker용 label:

```text
date별 forward_20D_excess percentile rank
상위 20% = 4
20~40% = 3
40~60% = 2
60~80% = 1
하위 20% = 0
```

## LightGBM Ranker

모델은 `LGBMRanker(objective="lambdarank", metric="ndcg")`를 사용한다.

학습 데이터는 반드시 `date`, `etf_ticker` 순으로 정렬한다. 같은 날짜의 ETF들을 하나의 query group으로 묶는다.

```python
group_train = train_df.groupby("date").size().to_list()
```

`sum(group_train) == n_samples`가 되어야 한다.

## 룩어헤드 방지

- date t의 feature는 t까지의 가격만 사용한다.
- target만 t 이후 5거래일/20거래일을 사용한다.
- holdings는 date t 이전의 가장 최근 holdings만 사용한다.
- 시간순 split만 사용하고 random split은 사용하지 않는다.
- 시장 regime, macro feature, Risk-Off Sentinel 값은 위험자산 수익률 랭킹에 넣지 않는다.

## 백테스트 해석

비교 대상:

- benchmark 단독
- Rule Score Top K
- ML Ranker Top K

평가지표:

- CAGR
- 누적수익률
- MDD
- Sharpe
- 승률
- 평균 forward 20D excess return
- Hit Ratio
- Rank IC

이 모델의 목적은 시장 전체 위험을 판단하는 것이 아니라, 위험자산을 담을 수 있는 구간에서 어떤 ETF가 기준지수 대비 더 강할 가능성이 높은지 랭킹하는 것이다.
