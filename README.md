# Macro Regime Asset Screener

Daily macro-driver asset screening model inspired by the Eugene AI Transformation Cycle quant framework:

- build stationary macro/market driver features from level, diff, momentum, z-score, MA distance, slope, and volatility-adjusted changes
- classify current market state with a GMM-style unsupervised regime layer plus explicit economic post-labels
- estimate each asset's rolling sensitivity to core drivers
- blend technical confirmation, driver fit, historical conditional win rate, and risk penalties into a 0-100 current attractiveness score
- output 1-week and 4-week upside probabilities for each tradable asset

## Run

```powershell
python scripts\macro_regime_asset_screener.py
```

Outputs are written to:

- `outputs/macro_regime_asset_screener_latest/tables/current_asset_scores.csv`
- `outputs/macro_regime_asset_screener_latest/tables/driver_state.csv`
- `outputs/macro_regime_asset_screener_latest/reports/current_report.md`

The script caches downloaded data under `.cache/` and uses public FRED CSV and Yahoo Finance data first. It can run without paid API keys; unavailable series are skipped and listed in the report.

## Asset Universe

The investable universe is configured in:

- `data/asset_universe.csv`

Codes may be entered as Korean market codes with the `A` prefix, such as `A069500`; the model normalizes them to Yahoo/KRX style symbols such as `069500.KS`. Every script imports the same `ASSETS` object from `macro_regime_asset_screener.py`, so macro screening, RWKV/LPPL, walk-forward validation, and the daily risk-off sentinel use the same universe.

## RWKV + LPPL / DTCAI Version

```powershell
python scripts\rwkv_lppl_asset_screener.py
```

This heavier version adds:

- RWKV-style time-mix self-supervised macro sequence encoder
- GMM clustering on RWKV embeddings
- embedding drift as a regime-transition signal
- correlation + Granger-causality driver selection report against KODEX 200
- LPPL fitting for every asset class through genetic algorithm optimization
- rolling LPPL parameter-set generation for reliability training
- ANN / Random Forest / Logistic reliability model comparison, with recall-first selection
- built-in SMOTE-style minority oversampling for crash-label imbalance
- `DTC=(t2-t1)/(tc-t1)` and `DTCAI=DTC*AI_reliability`
- asset score adjustment by LPPL crash-proximity/bubble risk

Default LPPL settings intentionally follow a full, non-lightweight configuration:

- population: `200`
- generations: `700`
- fitting window: `504` trading days
- rolling step: `21` trading days
- LPPL parameter sets per window: `500`
- RWKV sequence: `48` monthly observations

Outputs:

- `outputs/rwkv_lppl_asset_screener_latest/tables/current_asset_scores_rwkv_lppl.csv`
- `outputs/rwkv_lppl_asset_screener_latest/tables/current_lppl_dtcai.csv`
- `outputs/rwkv_lppl_asset_screener_latest/tables/rwkv_embeddings.csv`
- `outputs/rwkv_lppl_asset_screener_latest/tables/rwkv_regime_history.csv`
- `outputs/rwkv_lppl_asset_screener_latest/tables/lppl_reliability_training_set.csv`
- `outputs/rwkv_lppl_asset_screener_latest/tables/lppl_reliability_training_scored.csv`
- `outputs/rwkv_lppl_asset_screener_latest/tables/lppl_signal_validation.csv`
- `outputs/rwkv_lppl_asset_screener_latest/tables/driver_selection_granger_corr.csv`
- `outputs/rwkv_lppl_asset_screener_latest/reports/current_report.md`

For a quick smoke test only:

```powershell
python scripts\rwkv_lppl_asset_screener.py --skip-download --rwkv-epochs 3 --sequence-length 12 --rwkv-frequency M --lppl-window 360 --lppl-population 12 --lppl-generations 5 --lppl-fits-per-window 4 --lppl-training-step 252 --max-training-windows 2 --refresh-lppl
```

The quick smoke test is only for execution validation. It intentionally does not create enough historical crash-positive labels to train a meaningful ANN/RF/Logistic reliability model.

## Walk-Forward Validation And Calibration

After running the RWKV + LPPL screener, validate and calibrate the model:

```powershell
python scripts\walkforward_calibrate_rwkv_lppl.py --skip-download --top-n 5 --cost-bps 10
```

Outputs:

- `outputs/rwkv_lppl_walkforward_validation_latest/tables/walkforward_raw_panel.csv`
- `outputs/rwkv_lppl_walkforward_validation_latest/tables/walkforward_calibrated_panel.csv`
- `outputs/rwkv_lppl_walkforward_validation_latest/tables/probability_calibration.csv`
- `outputs/rwkv_lppl_walkforward_validation_latest/tables/walkforward_strategy_monthly.csv`
- `outputs/rwkv_lppl_walkforward_validation_latest/tables/walkforward_summary.csv`
- `outputs/rwkv_lppl_walkforward_validation_latest/tables/lppl_false_alarm_validation.csv`
- `outputs/rwkv_lppl_walkforward_validation_latest/tables/calibrated_current_asset_scores.csv`
- `outputs/rwkv_lppl_walkforward_validation_latest/reports/validation_report.md`

## Daily Risk-Off Sentinel Overlay

Run this after the RWKV/LPPL screener and walk-forward calibration when you want the fastest risk-control layer:

```powershell
python scripts\daily_risk_off_sentinel.py --skip-download
```

The sentinel is a daily override layer. It does not predict news headlines; it reacts immediately to market fingerprints that often appear before or during crash windows:

- volatility shock: VIX, VXN, MOVE
- credit shock: HY OAS, IG OAS, HYG/IEF
- FX shock: DXY, USDKRW, USDCNH
- equity stress: S&P 500, Nasdaq 100, SOX, Russell 2000
- cyclical stress: copper, copper/gold, China/HK proxies
- supply/hedge shock: WTI, gold
- liquidity stress: NFCI, ANFCI

States:

- `Normal`: full risk budget
- `Watch`: early warning, size down risky assets
- `De-risk`: cut risky assets aggressively
- `Cash`: cash/short bonds become the priority overlay

Outputs:

- `outputs/daily_risk_off_sentinel_latest/tables/daily_sentinel_history.csv`
- `outputs/daily_risk_off_sentinel_latest/tables/sentinel_adjusted_current_scores.csv`
- `outputs/daily_risk_off_sentinel_latest/tables/sentinel_benchmark_validation.csv`
- `outputs/daily_risk_off_sentinel_latest/tables/sentinel_threshold_sweep.csv`
- `outputs/daily_risk_off_sentinel_latest/tables/sentinel_crash_episode_validation.csv`
- `outputs/daily_risk_off_sentinel_latest/tables/sentinel_event_case_studies.csv`
- `outputs/daily_risk_off_sentinel_latest/reports/daily_sentinel_report.md`
