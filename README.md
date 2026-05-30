# ETF Leadership Ranking Backtest

국내 상장 ETF 유니버스를 대상으로 단순 기술적 랭킹 모델과 Tree 기반 랭킹 모델을 같은 조건에서 비교하고, 월간 리밸런싱 포트폴리오로 검증하는 연구용 파이프라인입니다.

## 핵심 구성

- `v2`: 기존 리더십 피처 기반 LightGBM walk-forward ranker
- `v3.1 light overlay`: v2 점수에 지속성 상대강도와 단기 반등 패널티를 약하게 얹고, 벤치마크 강세장에서 라틴/일본 ETF를 제외하는 위험자산 100% overlay
- `current-only screener`: 백테스트용 long-lived 필터 없이 전체 대회 ETF를 최신 가격 기준으로 스크리닝
- `rule vs ranker compare`: 단순 rule ranking과 Tree ranker를 같은 output 형식으로 비교
- `beta RS / ATR compare`: Beta-adjusted RS와 ATR 이격도를 사용하는 단순 기술 분석 ranking vs Tree model 비교

## 저장소 구조

```text
data/
  etf_universe_leadership.csv
  processed/
    long_lived_scored_features.csv
  gaps_long_lived_cache/
    gaps_etf_benchmark_prices_2010-01-01_2026-05-18.csv

scripts/
  current_only_v31_screener.py
  walkforward_leadership_v2_backtest.py
  leadership_v2_constrained_70_30_backtest.py
  leadership_v2_sleeve_only_backtest.py
  leadership_v31_light_overlay_backtest.py
  walkforward_leadership_v31_excess_ranker.py
  leadership_rule_vs_ranker_compare.py
  walkforward_etf_beta_rs_atr_compare.py

results/
  v2_walkforward/
  v31_light_overlay/
  model_compare/
```

## 모델 비교 프레임워크

두 모델은 모두 아래 공통 output을 생성합니다.

```text
Date, Ticker, Predicted_Score, Actual_Return
```

평가 지표:

1. Spearman rank correlation
2. Z-score 정규화 후 RMSE fit
3. Top-K 평균 수익률과 decay 분석
4. Spearman과 RMSE 기반 composite score

## 주요 실행 명령

전체 대회 ETF current-only 스크리닝:

```bash
python scripts/current_only_v31_screener.py --end 2026-05-30
```

v2 walk-forward ranker:

```bash
python scripts/walkforward_leadership_v2_backtest.py --plot
```

위험자산 100% / 안전자산 100% sleeve 분리 백테스트:

```bash
python scripts/leadership_v2_sleeve_only_backtest.py --plot
```

v3.1 excess-ranker 실험:

```bash
python scripts/walkforward_leadership_v31_excess_ranker.py --plot
```

v3.1 light overlay:

```bash
python scripts/leadership_v31_light_overlay_backtest.py --plot
```

단순 rule ranking vs Tree ranker 비교:

```bash
python scripts/leadership_rule_vs_ranker_compare.py
```

Beta-adjusted RS / ATR 기반 모델 비교:

```bash
python scripts/walkforward_etf_beta_rs_atr_compare.py --plot
```

## 현재 핵심 결과

최신 스크리닝은 `current_only_v31_screener.py`를 기본으로 사용합니다. 이 스크리너는 전체 대회 ETF 중 가격 히스토리 120거래일 이상인 종목을 모두 평가하고, `KODEX 200 20D > 3%` 또는 `60D > 8%`인 벤치마크 강세장에서는 `TIGER 라틴35(105010.KS)`, `KODEX 일본TOPIX100(101280.KS)`를 후보에서 제외합니다.

위험자산 100% 기준:

| 모델 | 누적수익률 | CAGR | MDD | Sharpe | 누적 초과수익 |
|---|---:|---:|---:|---:|---:|
| v2 risk-only | 722.37% | 22.62% | -16.97% | 1.28 | 12.04% |
| v3.1 light overlay | 812.59% | 23.86% | -16.77% | 1.32 | 25.94% |

해석:

- v3.1 light는 장기 성과와 누적 초과수익을 개선했습니다.
- 적용된 추가 룰: `KODEX 200 20D > 3%` 또는 `60D > 8%`인 벤치마크 강세장에서는 `TIGER 라틴35(105010.KS)`, `KODEX 일본TOPIX100(101280.KS)`를 편입 후보에서 제외합니다.
- 2026 초과수익 열위는 줄었지만 2025 열위는 남아 있어, 다음 개선 방향은 regime별 내부비중 조절입니다.

## 데이터 주의사항

- 이 저장소의 데이터는 연구와 재현 목적의 캐시입니다.
- `outputs/`는 중간 산출물이므로 GitHub에 올리지 않습니다.
- 대용량 원천 캐시와 실험 결과는 필요할 때 로컬에서 다시 생성하는 구조를 권장합니다.
