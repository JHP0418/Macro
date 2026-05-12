from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker, early_stopping, log_evaluation
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "selective_leadership_safe_v3_latest"
TABLES = OUT / "tables"


ETF_SCORE_COLS = {
    "1W": ["rule_5d_score", "entry_adjusted_5d_score"],
    "1M": ["rule_20d_score", "ranker_score", "blend_20d_score", "entry_adjusted_20d_score"],
}

SAFE_GROUPS = {
    "Cash/short bonds",
    "FX cash",
    "USD cash",
    "Korea bonds",
    "US long bonds",
    "US IG bonds",
    "Gold",
    "Korea defensive",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Selective ETF leadership entry gate and macro-conditioned safe asset V3.")
    p.add_argument("--long-lived-predictions", default=str(ROOT / "outputs/gaps_long_lived_etf_leadership_latest/tables/long_lived_predictions.csv"))
    p.add_argument("--safe-panel", default=str(ROOT / "outputs/institutional_risk_off_v2_latest/tables/macro_conditioned_safe_asset_panel.csv"))
    p.add_argument("--safe-current", default=str(ROOT / "outputs/institutional_risk_off_v2_latest/tables/current_safe_asset_recommendations_v2.csv"))
    p.add_argument("--output-dir", default=str(OUT))
    p.add_argument("--etf-train-end", default="2018-12-31")
    p.add_argument("--etf-valid-end", default="2021-12-31")
    p.add_argument("--safe-train-end", default="2024-12-31")
    p.add_argument("--safe-valid-end", default="2025-12-31")
    p.add_argument("--min-etf-valid-trades-1w", type=int, default=35)
    p.add_argument("--min-etf-valid-trades-1m", type=int, default=8)
    p.add_argument("--safe-top-k", type=int, default=3)
    return p.parse_args()


def read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    std = x.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - x.mean()) / std


def rebalance_dates(dates: pd.Series, horizon: str) -> list[pd.Timestamp]:
    idx = pd.DatetimeIndex(pd.to_datetime(dates).dropna().unique()).sort_values()
    if idx.empty:
        return []
    freq = "M" if horizon == "1M" else "W-FRI"
    return pd.Series(idx, index=idx).groupby(idx.to_period(freq)).max().tolist()


def target_cols(horizon: str) -> tuple[str, str, str, int]:
    if horizon == "1W":
        return "forward_5D_return", "benchmark_forward_5D_return", "forward_5D_excess", 52
    return "forward_20D_return", "benchmark_forward_20D_return", "forward_20D_excess", 12


def build_candidate_panel(features: pd.DataFrame, horizon: str, top_k_values: list[int]) -> pd.DataFrame:
    ret_col, bench_col, excess_col, _ = target_cols(horizon)
    score_cols = [c for c in ETF_SCORE_COLS[horizon] if c in features.columns]
    rows: list[dict] = []
    for score_col in score_cols:
        cols_needed = ["date", "etf_ticker", score_col, ret_col, bench_col, excess_col]
        data = features.dropna(subset=[c for c in cols_needed if c in features.columns]).copy()
        if "prediction_horizon" in data.columns:
            horizon_rows = data["prediction_horizon"].astype(str).eq(horizon)
            if horizon_rows.any():
                data = data[horizon_rows].copy()
        if data.empty:
            continue
        for dt in rebalance_dates(data["date"], horizon):
            sample = data[data["date"].eq(dt)].copy()
            if sample.empty:
                continue
            sample[score_col] = pd.to_numeric(sample[score_col], errors="coerce")
            sample = sample.dropna(subset=[score_col, ret_col, bench_col, excess_col])
            if sample.empty:
                continue
            sample = sample.sort_values(score_col, ascending=False)
            universe_scores = sample[score_col].astype(float)
            for top_k in top_k_values:
                if sample.shape[0] < top_k:
                    continue
                top = sample.head(top_k).copy()
                returns = pd.to_numeric(top[ret_col], errors="coerce")
                excess = pd.to_numeric(top[excess_col], errors="coerce")
                score_values = pd.to_numeric(top[score_col], errors="coerce")
                row = {
                    "date": dt,
                    "horizon": horizon,
                    "score_col": score_col,
                    "top_k": top_k,
                    "portfolio_return": float(returns.mean()),
                    "benchmark_return": float(pd.to_numeric(top[bench_col], errors="coerce").mean()),
                    "excess_return": float(excess.mean()),
                    "hit_excess": int(excess.mean() > 0),
                    "hit_positive": int(returns.mean() > 0),
                    "utility": float(excess.mean() + 0.35 * returns.mean() - 0.65 * max(0.0, -returns.mean())),
                    "top1_score": float(score_values.iloc[0]),
                    "topk_score_mean": float(score_values.mean()),
                    "topk_score_min": float(score_values.min()),
                    "score_spread": float(score_values.iloc[0] - score_values.iloc[-1]) if len(score_values) > 1 else 0.0,
                    "score_std": float(score_values.std(ddof=0)) if len(score_values) > 1 else 0.0,
                    "universe_score_std": float(universe_scores.std(ddof=0)),
                    "universe_score_iqr": float(universe_scores.quantile(0.75) - universe_scores.quantile(0.25)),
                    "selected": ",".join(top["etf_ticker"].astype(str)),
                    "selected_names": ",".join(top.get("name", top["etf_ticker"]).astype(str)),
                }
                for col in [
                    "ETF_RS_20D",
                    "ETF_RS_60D",
                    "ETF_RS_120D",
                    "RS_slope_20D",
                    "weighted_HP",
                    "HP90_share",
                    "MA60_breadth",
                    "MA200_breadth",
                    "Breadth_change_20D",
                    "RS_positive_share",
                    "top5_return_contribution_share",
                    "reg_r2",
                    "reg_coef_high_proximity",
                    "reg_residual_dispersion",
                    "entry_prob_5d",
                    "entry_prob_20d",
                ]:
                    if col in top.columns:
                        row[f"topk_{col}_mean"] = float(pd.to_numeric(top[col], errors="coerce").mean())
                rows.append(row)
    out = pd.DataFrame(rows).sort_values(["date", "horizon", "score_col", "top_k"]).reset_index(drop=True)
    if out.empty:
        return out
    out["entry_label"] = ((out["hit_excess"].eq(1)) & (out["hit_positive"].eq(1))).astype(int)
    return out


def encode_candidate_features(panel: pd.DataFrame, fit_columns: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    base = [
        "top_k",
        "top1_score",
        "topk_score_mean",
        "topk_score_min",
        "score_spread",
        "score_std",
        "universe_score_std",
        "universe_score_iqr",
    ]
    base += [c for c in panel.columns if c.startswith("topk_")]
    x = panel[base].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True)).fillna(0.0)
    dummies = pd.get_dummies(panel["score_col"].astype(str), prefix="score")
    x = pd.concat([x, dummies], axis=1)
    if fit_columns is not None:
        x = x.reindex(columns=fit_columns, fill_value=0.0)
        return x, fit_columns
    return x, list(x.columns)


def fit_entry_models(train: pd.DataFrame, valid: pd.DataFrame) -> tuple[GradientBoostingClassifier, GradientBoostingRegressor, list[str], dict]:
    x_train, cols = encode_candidate_features(train)
    x_valid, _ = encode_candidate_features(valid, cols)
    y_train = train["entry_label"].astype(int)
    clf = GradientBoostingClassifier(random_state=42, n_estimators=140, learning_rate=0.035, max_depth=2, subsample=0.8)
    clf.fit(x_train, y_train)
    reg = GradientBoostingRegressor(random_state=42, n_estimators=160, learning_rate=0.035, max_depth=2, subsample=0.8)
    reg.fit(x_train, train["utility"].astype(float))
    valid_prob = clf.predict_proba(x_valid)[:, 1]
    metrics = {"valid_entry_auc": safe_auc(valid["entry_label"], valid_prob)}
    return clf, reg, cols, metrics


def add_entry_predictions(panel: pd.DataFrame, clf: GradientBoostingClassifier, reg: GradientBoostingRegressor, cols: list[str]) -> pd.DataFrame:
    x, _ = encode_candidate_features(panel, cols)
    out = panel.copy()
    out["entry_gate_prob"] = clf.predict_proba(x)[:, 1]
    out["predicted_utility"] = reg.predict(x)
    out["dynamic_selection_score"] = 0.75 * out["entry_gate_prob"] + 0.25 * zscore(out["predicted_utility"]).fillna(0.0)
    return out


def choose_dynamic(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return panel
    idx = panel.sort_values(["date", "dynamic_selection_score"], ascending=[True, False]).groupby("date").head(1).index
    return panel.loc[idx].sort_values("date").reset_index(drop=True)


def apply_entry_threshold(panel: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = panel.copy()
    out["invested"] = pd.to_numeric(out["entry_gate_prob"], errors="coerce").ge(threshold).astype(int)
    out["strategy_return"] = out["portfolio_return"].where(out["invested"].eq(1), 0.0)
    out["strategy_excess_return"] = out["excess_return"].where(out["invested"].eq(1), 0.0)
    return out


def perf(panel: pd.DataFrame, label: str, periods_per_year: int) -> dict:
    if panel.empty:
        return {"label": label, "periods": 0}
    returns = pd.to_numeric(panel["strategy_return"], errors="coerce").fillna(0.0)
    invested = panel["invested"].astype(int)
    equity = (1.0 + returns).cumprod()
    active = panel[invested.eq(1)].copy()
    return {
        "label": label,
        "periods": int(panel.shape[0]),
        "invested_periods": int(invested.sum()),
        "coverage": float(invested.mean()),
        "cumulative_return": float(equity.iloc[-1] - 1.0),
        "CAGR": cagr(float(equity.iloc[-1]), panel.shape[0], periods_per_year),
        "MDD": max_drawdown(equity),
        "Sharpe": sharpe(returns, periods_per_year),
        "trade_hit_excess": float((active["excess_return"] > 0).mean()) if not active.empty else np.nan,
        "trade_hit_positive": float((active["portfolio_return"] > 0).mean()) if not active.empty else np.nan,
        "avg_trade_return": float(active["portfolio_return"].mean()) if not active.empty else np.nan,
        "avg_trade_excess": float(active["excess_return"].mean()) if not active.empty else np.nan,
        "avg_top_k": float(active["top_k"].mean()) if not active.empty else np.nan,
    }


def optimize_entry_threshold(valid: pd.DataFrame, horizon: str, min_trades: int) -> pd.DataFrame:
    _, _, _, periods_per_year = target_cols(horizon)
    rows = []
    for threshold in np.round(np.arange(0.40, 0.91, 0.025), 3):
        tested = apply_entry_threshold(valid, threshold)
        if int(tested["invested"].sum()) < min_trades:
            continue
        row = perf(tested, f"{horizon}_thr_{threshold}", periods_per_year)
        row["threshold"] = threshold
        row["objective"] = (
            row["Sharpe"] if not pd.isna(row["Sharpe"]) else -99
        ) + 1.2 * (row["trade_hit_excess"] if not pd.isna(row["trade_hit_excess"]) else 0) - 0.35 * abs(row["MDD"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["objective", "Sharpe"], ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


def run_etf_selective(args: argparse.Namespace, out_dir: Path) -> None:
    raw = read_csv(args.long_lived_predictions, parse_dates=["date"])
    if raw.empty:
        return
    top_k_values = [1, 2, 3, 5]
    summaries = []
    trades = []
    grids = []
    for horizon in ["1W", "1M"]:
        _, _, _, periods_per_year = target_cols(horizon)
        panel = build_candidate_panel(raw, horizon, top_k_values)
        if panel.empty:
            continue
        train = panel[panel["date"].le(pd.Timestamp(args.etf_train_end))].copy()
        valid = panel[panel["date"].gt(pd.Timestamp(args.etf_train_end)) & panel["date"].le(pd.Timestamp(args.etf_valid_end))].copy()
        test = panel[panel["date"].gt(pd.Timestamp(args.etf_valid_end))].copy()
        clf, reg, cols, model_metrics = fit_entry_models(train, valid)
        valid_pred = choose_dynamic(add_entry_predictions(valid, clf, reg, cols))
        test_pred = choose_dynamic(add_entry_predictions(test, clf, reg, cols))
        min_trades = args.min_etf_valid_trades_1w if horizon == "1W" else args.min_etf_valid_trades_1m
        grid = optimize_entry_threshold(valid_pred, horizon, min_trades=min_trades)
        if grid.empty:
            continue
        grid["horizon"] = horizon
        grids.append(grid)
        best = grid.iloc[0]
        threshold = float(best["threshold"])
        tested = apply_entry_threshold(test_pred, threshold)
        summary = perf(tested, f"dynamic_selective_{horizon}", periods_per_year)
        summary.update(
            {
                "horizon": horizon,
                "threshold": threshold,
                "valid_objective": float(best["objective"]),
                "valid_sharpe": float(best["Sharpe"]),
                "valid_trade_hit_excess": float(best["trade_hit_excess"]),
                **model_metrics,
            }
        )
        summaries.append(summary)
        tested["horizon"] = horizon
        tested["threshold"] = threshold
        trades.append(tested)
    if summaries:
        pd.DataFrame(summaries).to_csv(out_dir / "etf_selective_v3_summary.csv", index=False, encoding="utf-8-sig")
    if trades:
        pd.concat(trades, ignore_index=True).to_csv(out_dir / "etf_selective_v3_trades.csv", index=False, encoding="utf-8-sig")
    if grids:
        pd.concat(grids, ignore_index=True).to_csv(out_dir / "etf_selective_v3_threshold_grid.csv", index=False, encoding="utf-8-sig")


def make_rank_labels(frame: pd.DataFrame, target: str) -> pd.Series:
    pct = frame.groupby("date")[target].rank(pct=True, method="average")
    return np.ceil(pct * 5.0).sub(1).clip(0, 4).where(pct.notna())


def prepare_safe_panel(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str]]:
    panel = read_csv(args.safe_panel, parse_dates=["date"])
    if panel.empty:
        return panel, []
    panel = panel[panel["group"].isin(SAFE_GROUPS) | panel.get("is_safe_asset", False).astype(bool)].copy()
    panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)
    macro_cols = [c for c in panel.columns if c.startswith("macro_")]
    base_cols = [
        "score_0_100",
        "technical_score",
        "driver_fit_score",
        "beta_fit_score",
        "calibrated_prob_1w",
        "calibrated_prob_4w",
        "institutional_score_0_100",
    ]
    safe_ssl_cols = [c for c in panel.columns if c.startswith("safe_ssl_")]
    one_hot_cols = [c for c in panel.columns if c.startswith("group_") or c.startswith("basket_")]
    interaction_sources = [
        "macro_axis1_vol_credit_stress",
        "macro_axis2_fx_liquidity_stress",
        "macro_axis3_peak_fragility_stress",
        "macro_US10Y_driver_chg_20d",
        "macro_USDKRW_driver_ret_20d",
        "macro_GOLD_driver_ret_20d",
        "macro_HYG_IEF_ret_20d",
        "macro_risk_off_score",
    ]
    interaction_sources = [c for c in interaction_sources if c in panel.columns]
    for src in interaction_sources:
        src_num = pd.to_numeric(panel[src], errors="coerce")
        for dummy in one_hot_cols:
            panel[f"{dummy}_x_{src}"] = pd.to_numeric(panel[dummy], errors="coerce").fillna(0.0) * src_num
    interaction_cols = [c for c in panel.columns if "_x_macro_" in c]
    features = [c for c in base_cols + macro_cols + safe_ssl_cols + one_hot_cols + interaction_cols if c in panel.columns]
    for col in features + ["safe_target_1w", "safe_target_1m", "realized_return_1w", "realized_return_4w"]:
        if col in panel.columns:
            panel[col] = pd.to_numeric(panel[col], errors="coerce")
    panel["safe_v3_label_1w"] = make_rank_labels(panel, "safe_target_1w")
    panel["safe_v3_label_1m"] = make_rank_labels(panel, "safe_target_1m")
    return panel, features


def lgb_rank_data(frame: pd.DataFrame, features: list[str], label: str) -> tuple[pd.DataFrame, pd.Series, list[int], pd.DataFrame]:
    data = frame.dropna(subset=[label]).sort_values(["date", "symbol"]).copy()
    data = data[data.groupby("date")["symbol"].transform("size").ge(2)].reset_index(drop=True)
    x = data[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = x.groupby(data["date"]).transform(lambda s: s.fillna(s.median())).fillna(0.0)
    y = data[label].astype(int)
    group = data.groupby("date", sort=False).size().astype(int).tolist()
    return x, y, group, data


def train_safe_ranker(train: pd.DataFrame, valid: pd.DataFrame, features: list[str], label: str) -> LGBMRanker:
    x_train, y_train, g_train, _ = lgb_rank_data(train, features, label)
    x_valid, y_valid, g_valid, _ = lgb_rank_data(valid, features, label)
    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=600,
        learning_rate=0.025,
        num_leaves=15,
        max_depth=4,
        min_child_samples=12,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_alpha=1.0,
        reg_lambda=8.0,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        x_train,
        y_train,
        group=g_train,
        eval_set=[(x_valid, y_valid)],
        eval_group=[g_valid],
        eval_at=[1, 3],
        callbacks=[early_stopping(80), log_evaluation(100)],
    )
    return model


def predict_safe(model: LGBMRanker, frame: pd.DataFrame, features: list[str], label: str, split: str) -> pd.DataFrame:
    x, _, _, data = lgb_rank_data(frame, features, label)
    out = data.copy()
    out["split"] = split
    out["safe_v3_ranker_score"] = model.predict(x)
    return out


def safe_backtest(pred: pd.DataFrame, target: str, ret_col: str, top_k: int) -> tuple[pd.DataFrame, dict]:
    rows = []
    test = pred[pred["split"].eq("test")].copy()
    for date, part in test.groupby("date", sort=True):
        picks = part.nlargest(min(top_k, len(part)), "safe_v3_ranker_score")
        actual = part.nlargest(min(top_k, len(part)), target)
        rows.append(
            {
                "date": date,
                "picked_return": float(pd.to_numeric(picks[ret_col], errors="coerce").mean()),
                "picked_target": float(pd.to_numeric(picks[target], errors="coerce").mean()),
                "safe_avg_target": float(pd.to_numeric(part[target], errors="coerce").mean()),
                "actual_top_target": float(pd.to_numeric(actual[target], errors="coerce").mean()),
                "beat_safe_average": int(pd.to_numeric(picks[target], errors="coerce").mean() > pd.to_numeric(part[target], errors="coerce").mean()),
                "overlap": len(set(picks["symbol"]) & set(actual["symbol"])) / max(min(top_k, len(part)), 1),
                "selected": ",".join(picks["symbol"].astype(str)),
                "selected_names": ",".join(picks.get("name", picks["symbol"]).astype(str)),
            }
        )
    raw = pd.DataFrame(rows)
    summary = {
        "periods": int(raw.shape[0]),
        "avg_picked_return": float(raw["picked_return"].mean()) if not raw.empty else np.nan,
        "avg_picked_target": float(raw["picked_target"].mean()) if not raw.empty else np.nan,
        "avg_safe_target": float(raw["safe_avg_target"].mean()) if not raw.empty else np.nan,
        "beat_safe_average_rate": float(raw["beat_safe_average"].mean()) if not raw.empty else np.nan,
        "topk_overlap_rate": float(raw["overlap"].mean()) if not raw.empty else np.nan,
    }
    return raw, summary


def run_safe_v3(args: argparse.Namespace, out_dir: Path) -> None:
    panel, features = prepare_safe_panel(args)
    if panel.empty or not features:
        return
    train = panel[panel["date"].le(pd.Timestamp(args.safe_train_end))].copy()
    valid = panel[panel["date"].gt(pd.Timestamp(args.safe_train_end)) & panel["date"].le(pd.Timestamp(args.safe_valid_end))].copy()
    test = panel[panel["date"].gt(pd.Timestamp(args.safe_valid_end))].copy()
    preds = []
    raws = []
    summaries = []
    for horizon, label, target, ret_col in [
        ("1w", "safe_v3_label_1w", "safe_target_1w", "realized_return_1w"),
        ("1m", "safe_v3_label_1m", "safe_target_1m", "realized_return_4w"),
    ]:
        model = train_safe_ranker(train, valid, features, label)
        pred = pd.concat(
            [
                predict_safe(model, train, features, label, "train"),
                predict_safe(model, valid, features, label, "valid"),
                predict_safe(model, test, features, label, "test"),
            ],
            ignore_index=True,
        )
        pred["horizon"] = horizon
        raw, summary = safe_backtest(pred, target, ret_col, args.safe_top_k)
        raw["horizon"] = horizon
        summary["horizon"] = horizon
        summaries.append(summary)
        preds.append(pred)
        raws.append(raw)
        imp = pd.DataFrame({"feature": features, "importance_gain": model.booster_.feature_importance(importance_type="gain"), "horizon": horizon})
        imp.sort_values("importance_gain", ascending=False).to_csv(out_dir / f"safe_v3_importance_{horizon}.csv", index=False, encoding="utf-8-sig")
    pd.concat(preds, ignore_index=True).to_csv(out_dir / "safe_v3_predictions.csv", index=False, encoding="utf-8-sig")
    pd.concat(raws, ignore_index=True).to_csv(out_dir / "safe_v3_backtest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summaries).to_csv(out_dir / "safe_v3_summary.csv", index=False, encoding="utf-8-sig")
    current = read_csv(args.safe_current, parse_dates=["date"])
    if not current.empty:
        current.to_csv(out_dir / "safe_v3_current_reference_v2_latest.csv", index=False, encoding="utf-8-sig")


def cagr(final_value: float, periods: int, periods_per_year: int) -> float:
    if periods <= 0 or final_value <= 0:
        return np.nan
    return float(final_value ** (periods_per_year / periods) - 1.0)


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return np.nan
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def sharpe(returns: pd.Series, periods_per_year: int) -> float:
    std = returns.std(ddof=1)
    if pd.isna(std) or std == 0:
        return np.nan
    return float(returns.mean() / std * np.sqrt(periods_per_year))


def safe_auc(y_true: pd.Series, y_prob: np.ndarray) -> float:
    try:
        if pd.Series(y_true).nunique() < 2:
            return np.nan
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return np.nan


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_etf_selective(args, out_dir)
    run_safe_v3(args, out_dir)
    metadata = {
        "description": "Selective ETF leadership gate with dynamic Top-K and macro-conditioned safe asset V3.",
        "etf_train_end": args.etf_train_end,
        "etf_valid_end": args.etf_valid_end,
        "safe_train_end": args.safe_train_end,
        "safe_valid_end": args.safe_valid_end,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(out_dir.resolve())


if __name__ == "__main__":
    main()
