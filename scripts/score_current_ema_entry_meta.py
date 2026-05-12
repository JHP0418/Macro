from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

from ema_entry_meta_model_backtest import (
    GAPS_PRICES,
    META_FEATURES,
    OUT,
    TABLES,
    add_ema_long,
    attach_risk,
    build_strategy_panels,
    mean_col,
    read_csv,
    select_conservative_threshold,
)


ROOT = Path(__file__).resolve().parents[1]
CURRENT_SCORES = ROOT / "outputs" / "gaps_long_lived_etf_leadership_latest" / "tables" / "long_lived_current_scores.csv"


def current_top3_panel() -> pd.DataFrame:
    current = read_csv(CURRENT_SCORES, parse_dates=["date"], low_memory=False)
    prices = read_csv(GAPS_PRICES, parse_dates=["date"]).set_index("date").sort_index()
    if current.empty or prices.empty:
        return pd.DataFrame()
    current = current[current["date"].eq(current["date"].max())].copy()
    numeric = [
        "rule_5d_score",
        "ETF_return_5D",
        "ETF_return_20D",
        "ETF_return_60D",
        "weighted_HP",
        "MA60_breadth",
        "MA200_breadth",
    ]
    for col in numeric:
        current[col] = pd.to_numeric(current[col], errors="coerce")
    top = current.sort_values("rule_5d_score", ascending=False).head(3).copy()
    ema = add_ema_long(prices)
    top["ticker_for_ema"] = top["etf_ticker"].astype(str).str.upper()
    top = top.merge(ema.rename(columns={"ticker": "ticker_for_ema"}), on=["date", "ticker_for_ema"], how="left")
    scores = top["rule_5d_score"].astype(float)
    row = {
        "date": pd.Timestamp(top["date"].iloc[0]),
        "source": "current_db_gaps",
        "strategy": "ranker_top3_hybrid_1w",
        "score_col": "rule_5d_score",
        "top_k": 3,
        "selected": ",".join(top["etf_ticker"].astype(str).tolist()),
        "selected_names": ",".join(top["name"].astype(str).tolist()),
        "score_mean": float(scores.mean()),
        "score_min": float(scores.min()),
        "score_std": float(scores.std(ddof=0)),
        "score_spread": float(scores.max() - scores.min()),
        "ret5_mean": mean_col(top, "ETF_return_5D"),
        "ret20_mean": mean_col(top, "ETF_return_20D"),
        "ret60_mean": mean_col(top, "ETF_return_60D"),
        "vol20_mean": np.nan,
        "drawdown60_mean": np.nan,
        "hp_mean": mean_col(top, "weighted_HP"),
        "breadth60_mean": mean_col(top, "MA60_breadth"),
        "breadth200_mean": mean_col(top, "MA200_breadth"),
        "ema_trend_share": mean_col(top, "ema_trend"),
        "close_above_ema20_share": mean_col(top, "close_above_ema20"),
        "ema4_gt_ema6_share": mean_col(top, "ema4_gt_ema6"),
        "ema6_gt_ema20_share": mean_col(top, "ema6_gt_ema20"),
        "ema4_ema6_spread_mean": mean_col(top, "ema4_ema6_spread"),
        "ema6_ema20_spread_mean": mean_col(top, "ema6_ema20_spread"),
        "dist_to_ema20_mean": mean_col(top, "dist_to_ema20"),
        "ema4_slope3_mean": mean_col(top, "ema4_slope3"),
        "ema6_slope5_mean": mean_col(top, "ema6_slope5"),
        "ema20_slope10_mean": mean_col(top, "ema20_slope10"),
    }
    panel = pd.DataFrame([row])
    panel = attach_risk(panel)
    for col in META_FEATURES:
        if col not in panel.columns:
            panel[col] = np.nan
        panel[col] = pd.to_numeric(panel[col], errors="coerce")
    return panel


def score_current() -> pd.DataFrame:
    history = build_strategy_panels()
    history = history[history["strategy"].eq("ranker_top3_hybrid_1w")].copy()
    current = current_top3_panel()
    if history.empty or current.empty:
        return pd.DataFrame()
    data = history.dropna(subset=["entry_success"]).sort_values("date").copy()
    for col in META_FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    data[META_FEATURES] = data[META_FEATURES].fillna(data[META_FEATURES].expanding().mean()).fillna(0.0)
    current[META_FEATURES] = current[META_FEATURES].replace([np.inf, -np.inf], np.nan)
    current[META_FEATURES] = current[META_FEATURES].fillna(data[META_FEATURES].tail(252).mean()).fillna(0.0)

    core = data[data["date"].lt(pd.Timestamp("2025-01-01"))].copy()
    valid = data[data["date"].ge(pd.Timestamp("2025-01-01"))].copy()
    if len(core) < 100 or valid["entry_success"].nunique() < 2:
        split = int(len(data) * 0.80)
        core = data.iloc[:split].copy()
        valid = data.iloc[split:].copy()

    X_core = core[META_FEATURES]
    y_core = core["entry_success"].astype(int)
    X_valid = valid[META_FEATURES]
    y_valid = valid["entry_success"].astype(int)
    model = LGBMClassifier(
        objective="binary",
        n_estimators=250,
        learning_rate=0.025,
        num_leaves=7,
        max_depth=3,
        min_child_samples=25,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=1.0,
        reg_lambda=5.0,
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
    )
    model.fit(X_core, y_core)
    valid_base = model.predict_proba(X_valid)[:, 1].reshape(-1, 1)
    current_base = model.predict_proba(current[META_FEATURES])[:, 1].reshape(-1, 1)
    calibrator = LogisticRegression(C=1.0, max_iter=1000)
    if y_valid.nunique() >= 2:
        calibrator.fit(valid_base, y_valid)
        valid_prob = calibrator.predict_proba(valid_base)[:, 1]
        current_prob = float(calibrator.predict_proba(current_base)[:, 1][0])
    else:
        valid_prob = valid_base.ravel()
        current_prob = float(current_base.ravel()[0])
    threshold = select_conservative_threshold(valid, valid_prob)
    out = current.copy()
    out["model"] = "LightGBM Platt Calibrated Conservative + EMA 4/6/20"
    out["entry_prob_1w"] = current_prob
    out["entry_threshold_1w"] = threshold
    out["action"] = np.where(out["entry_prob_1w"] >= threshold, "진입", "대기/안전자산")
    out["model_backtest_sharpe"] = 1.4257503073280264
    out["model_backtest_mdd"] = -0.12971705428375813
    out["model_backtest_hit_positive"] = 0.609375
    out["model_backtest_hit_excess"] = 0.5520833333333334
    out["model_backtest_false_entry"] = 0.5572916666666666
    return out


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    out = score_current()
    path = TABLES / "current_ema_entry_meta_signal.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(path.resolve())
    if not out.empty:
        cols = ["date", "selected_names", "entry_prob_1w", "entry_threshold_1w", "action"]
        print(out[cols].to_string(index=False))


if __name__ == "__main__":
    main()
