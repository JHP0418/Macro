from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = Path(__file__).resolve().parent

DEFAULT_UNIVERSE_PATH = ROOT / "data" / "etf_universe.csv"
DEFAULT_HOLDINGS_PATH = ROOT / "data" / "etf_holdings.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "etf_leadership_model"

FORWARD_5D = 5
FORWARD_20D = 20
MIN_HISTORY_DAYS = 260

MARKET_BENCHMARKS = {
    "KR": "^KS11",
    "US": "^NDX",
}

FEATURE_COLUMNS = [
    "ETF_RS_20D",
    "ETF_RS_60D",
    "ETF_RS_120D",
    "RS_slope_20D",
    "weighted_HP",
    "median_HP",
    "HP90_share",
    "HP_change_20D",
    "weighted_component_RS_20D",
    "weighted_component_RS_60D",
    "median_component_RS_20D",
    "RS_positive_share",
    "MA60_breadth",
    "MA200_breadth",
    "Breadth_change_20D",
    "median_component_return_20D",
    "median_component_return_60D",
    "mean_minus_median_return_20D",
    "top20_component_return_mean",
    "bottom20_component_return_mean",
    "holding_count",
    "effective_N",
    "top5_weight_share",
    "top10_weight_share",
    "top5_return_contribution_share",
]

LGBM_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "n_estimators": 500,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "random_state": 42,
    "n_jobs": -1,
}

