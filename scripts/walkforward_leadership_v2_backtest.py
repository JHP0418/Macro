from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("data/processed/long_lived_scored_features.csv")
DEFAULT_OUTPUT = Path("outputs/leadership_v2_walkforward")

FEATURES_1M = [
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
    "top5_return_contribution_share",
]

REGRESSION_FEATURES = [
    "reg_coef_high_proximity",
    "reg_coef_component_return_60d",
    "reg_coef_component_rs_60d",
    "reg_r2",
    "reg_residual_dispersion",
]


@dataclass(frozen=True)
class Window:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward ETF leadership v2 Tree Ranker with confidence gate.")
    p.add_argument("--input", default=str(DEFAULT_INPUT))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    p.add_argument("--min-date", default="2012-01-01")
    p.add_argument("--train-months", type=int, default=36)
    p.add_argument("--valid-months", type=int, default=12)
    p.add_argument("--test-months", type=int, default=1)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--group-score-std-threshold", type=float, default=0.03067)
    p.add_argument("--min-train-rows", type=int, default=1200)
    p.add_argument("--min-valid-rows", type=int, default=300)
    p.add_argument("--plot", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_frame(args.input, args.min_date)
    feature_cols = [c for c in FEATURES_1M + REGRESSION_FEATURES if c in data.columns]
    windows = make_windows(data["date"], args.train_months, args.valid_months, args.test_months)
    if not windows:
        raise RuntimeError("No walk-forward windows can be made from the input date range.")

    predictions = []
    importances = []
    for i, window in enumerate(windows, start=1):
        print(
            f"[window {i}/{len(windows)}] train {window.train_start.date()}~{window.train_end.date()} "
            f"valid {window.valid_start.date()}~{window.valid_end.date()} "
            f"test {window.test_start.date()}~{window.test_end.date()}",
            flush=True,
        )
        pred, imp = fit_predict_window(data, window, feature_cols, args.min_train_rows, args.min_valid_rows)
        if pred.empty:
            continue
        pred["window_id"] = i
        predictions.append(pred)
        if not imp.empty:
            imp["window_id"] = i
            importances.append(imp)

    if not predictions:
        raise RuntimeError("No walk-forward predictions generated.")

    pred = pd.concat(predictions, ignore_index=True).sort_values(["date", "etf_ticker"])
    pred.to_csv(output_dir / "walkforward_predictions.csv", index=False, encoding="utf-8-sig")

    if importances:
        importance = (
            pd.concat(importances, ignore_index=True)
            .groupby("feature", as_index=False)
            .agg(importance_gain=("importance_gain", "mean"), importance_split=("importance_split", "mean"))
            .sort_values("importance_gain", ascending=False)
        )
        importance.to_csv(output_dir / "feature_importance_mean.csv", index=False, encoding="utf-8-sig")

    trades_base, summary_base = run_monthly_backtest(
        pred,
        score_col="ranker_score",
        top_k=args.top_k,
        threshold=None,
        label="ranker_top3_no_gate",
    )
    trades_v2, summary_v2 = run_monthly_backtest(
        pred,
        score_col="ranker_score",
        top_k=args.top_k,
        threshold=args.group_score_std_threshold,
        label="ranker_top3_std_gate",
    )
    trades_rule, summary_rule = run_monthly_backtest(
        pred,
        score_col="rule_20d_score",
        top_k=args.top_k,
        threshold=None,
        label="rule_top3_no_gate",
    )

    trades = pd.concat([trades_base, trades_v2, trades_rule], ignore_index=True)
    summary = pd.DataFrame([summary_base, summary_v2, summary_rule])
    monthly = monthly_metrics(trades)
    group_stats = selected_group_stats(trades)
    ticker_stats = selected_ticker_stats(trades)

    trades.to_csv(output_dir / "backtest_trades.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "backtest_summary.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output_dir / "yearly_metrics.csv", index=False, encoding="utf-8-sig")
    group_stats.to_csv(output_dir / "selected_group_stats.csv", index=False, encoding="utf-8-sig")
    ticker_stats.to_csv(output_dir / "selected_ticker_stats.csv", index=False, encoding="utf-8-sig")

    if args.plot:
        plot_equity(trades, output_dir / "equity_curve.png")

    meta = {
        "input": str(args.input),
        "min_date": args.min_date,
        "train_months": args.train_months,
        "valid_months": args.valid_months,
        "test_months": args.test_months,
        "top_k": args.top_k,
        "group_score_std_threshold": args.group_score_std_threshold,
        "feature_cols": feature_cols,
        "windows": len(windows),
        "prediction_rows": int(len(pred)),
        "prediction_dates": int(pred["date"].nunique()),
    }
    (output_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


def load_frame(path: str | Path, min_date: str) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"])
    frame = frame[frame["date"].ge(pd.Timestamp(min_date))].copy()
    required = [
        "date",
        "etf_ticker",
        "model_group",
        "ranking_group",
        "rule_20d_score",
        "label_20D_group_rank_int",
        "forward_20D_return",
        "benchmark_forward_20D_return",
        "forward_20D_excess",
    ]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Input missing required columns: {missing}")
    numeric = sorted(set(FEATURES_1M + REGRESSION_FEATURES + required + ["holding_count"]))
    for col in numeric:
        if col in frame.columns and col not in {"date", "etf_ticker", "model_group", "ranking_group"}:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(
        subset=[
            "label_20D_group_rank_int",
            "rule_20d_score",
            "forward_20D_return",
            "benchmark_forward_20D_return",
            "forward_20D_excess",
        ]
    )
    if "holding_count" in frame.columns:
        frame = frame[frame["holding_count"].fillna(0).ge(2)]
    group_size = frame.groupby(["date", "model_group"])["etf_ticker"].transform("size")
    frame = frame[group_size.ge(2)]
    return frame.sort_values(["date", "model_group", "etf_ticker"]).reset_index(drop=True)


def make_windows(dates: pd.Series, train_months: int, valid_months: int, test_months: int) -> list[Window]:
    idx = pd.DatetimeIndex(pd.to_datetime(dates).dropna().unique()).sort_values()
    if idx.empty:
        return []
    first_test = pd.Timestamp(idx.min().year, idx.min().month, 1) + pd.DateOffset(months=train_months + valid_months)
    end = idx.max()
    windows = []
    test_start = first_test
    while test_start <= end:
        test_end = test_start + pd.DateOffset(months=test_months) - pd.DateOffset(days=1)
        valid_end = test_start - pd.DateOffset(days=1)
        valid_start = test_start - pd.DateOffset(months=valid_months)
        train_end = valid_start - pd.DateOffset(days=1)
        train_start = valid_start - pd.DateOffset(months=train_months)
        windows.append(Window(train_start, train_end, valid_start, valid_end, test_start, min(test_end, end)))
        test_start = test_start + pd.DateOffset(months=test_months)
    return windows


def fit_predict_window(
    frame: pd.DataFrame,
    window: Window,
    feature_cols: list[str],
    min_train_rows: int,
    min_valid_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = frame[(frame["date"] >= window.train_start) & (frame["date"] <= window.train_end)].copy()
    valid = frame[(frame["date"] >= window.valid_start) & (frame["date"] <= window.valid_end)].copy()
    test = frame[(frame["date"] >= window.test_start) & (frame["date"] <= window.test_end)].copy()
    if len(train) < min_train_rows or len(valid) < min_valid_rows or test.empty:
        return pd.DataFrame(), pd.DataFrame()

    try:
        from lightgbm import LGBMRanker, early_stopping, log_evaluation
    except ImportError as exc:
        raise ImportError("lightgbm is required for the leadership v2 walk-forward ranker.") from exc

    x_train, y_train, group_train, _ = rank_data(train, feature_cols)
    x_valid, y_valid, group_valid, _ = rank_data(valid, feature_cols)
    if x_train.empty or x_valid.empty:
        return pd.DataFrame(), pd.DataFrame()

    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=500,
        learning_rate=0.02,
        num_leaves=9,
        max_depth=3,
        min_child_samples=12,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=2.0,
        reg_lambda=12.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        x_train,
        y_train,
        group=group_train,
        eval_set=[(x_valid, y_valid)],
        eval_group=[group_valid],
        eval_at=[1, 3, 5],
        callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
    )

    x_test, _, _, test_data = rank_data(test, feature_cols)
    if x_test.empty:
        return pd.DataFrame(), pd.DataFrame()
    out = test_data.copy()
    out["ranker_score"] = model.predict(x_test)
    out = add_score_context(out, "ranker_score", "ranker")
    out = out.assign(
        train_start=window.train_start,
        train_end=window.train_end,
        valid_start=window.valid_start,
        valid_end=window.valid_end,
        test_start=window.test_start,
        test_end=window.test_end,
    )
    importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
            "importance_split": model.booster_.feature_importance(importance_type="split"),
        }
    )
    return out, importance


def rank_data(frame: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.Series, list[int], pd.DataFrame]:
    data = frame.dropna(subset=["label_20D_group_rank_int"]).sort_values(["date", "model_group", "etf_ticker"]).reset_index(drop=True)
    group_size = data.groupby(["date", "model_group"])["etf_ticker"].transform("size")
    data = data[group_size.ge(2)].reset_index(drop=True)
    x = data[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = x.groupby([data["date"], data["model_group"]]).transform(lambda s: s.fillna(s.median())).fillna(0.0)
    y = data["label_20D_group_rank_int"].astype(int)
    group = data.groupby(["date", "model_group"], sort=False).size().astype(int).tolist()
    return x, y, group, data


def add_score_context(frame: pd.DataFrame, score_col: str, prefix: str) -> pd.DataFrame:
    out = frame.copy()
    keys = ["date", "model_group"]
    score = pd.to_numeric(out[score_col], errors="coerce")
    group = out.groupby(keys)
    rank = group[score_col].rank(ascending=False, method="first")
    size = group[score_col].transform("size").astype(float)
    top = group[score_col].transform("max")
    mean = group[score_col].transform("mean")
    std = group[score_col].transform("std").replace(0, np.nan)
    out[f"{prefix}_group_rank"] = rank
    out[f"{prefix}_group_size"] = size
    out[f"{prefix}_group_rank_pct"] = (1.0 - (rank - 1.0) / (size - 1.0).replace(0, np.nan)).fillna(1.0)
    out[f"{prefix}_group_z"] = ((score - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out[f"{prefix}_gap_to_group_top"] = (top - score).fillna(0.0)
    out[f"{prefix}_group_score_std"] = std.fillna(0.0)
    return out


def monthly_dates(dates: pd.Series) -> list[pd.Timestamp]:
    idx = pd.DatetimeIndex(pd.to_datetime(dates).dropna().unique()).sort_values()
    return pd.Series(idx, index=idx).groupby(idx.to_period("M")).max().tolist()


def run_monthly_backtest(
    frame: pd.DataFrame,
    score_col: str,
    top_k: int,
    threshold: float | None,
    label: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    all_dates = monthly_dates(frame["date"])
    rows = []
    for dt in all_dates:
        sample = frame[frame["date"].eq(dt)].dropna(
            subset=[score_col, "forward_20D_return", "benchmark_forward_20D_return", "forward_20D_excess"]
        )
        if threshold is not None:
            if "ranker_group_score_std" not in sample.columns:
                raise ValueError("ranker_group_score_std is required for threshold-gated backtest.")
            sample = sample[sample["ranker_group_score_std"].ge(threshold)]
        if sample.empty:
            rows.append(flat_trade(dt, label, top_k, threshold))
            continue
        top = sample.nlargest(min(top_k, len(sample)), score_col)
        rows.append(
            {
                "date": dt,
                "label": label,
                "score_col": score_col,
                "top_k": top_k,
                "threshold": threshold,
                "invested": 1,
                "selected_count": int(len(top)),
                "portfolio_return": float(top["forward_20D_return"].mean()),
                "benchmark_return": float(top["benchmark_forward_20D_return"].mean()),
                "excess_return": float(top["forward_20D_excess"].mean()),
                "hit_excess": int(top["forward_20D_excess"].mean() > 0),
                "hit_positive": int(top["forward_20D_return"].mean() > 0),
                "selected": ",".join(top["etf_ticker"].astype(str).tolist()),
                "selected_names": ",".join(top.get("name", top["etf_ticker"]).astype(str).tolist()),
                "selected_groups": ",".join(top["ranking_group"].astype(str).tolist()),
            }
        )
    trades = pd.DataFrame(rows)
    return trades, summarize_trades(trades, label)


def flat_trade(date: pd.Timestamp, label: str, top_k: int, threshold: float | None) -> dict[str, object]:
    return {
        "date": date,
        "label": label,
        "score_col": "",
        "top_k": top_k,
        "threshold": threshold,
        "invested": 0,
        "selected_count": 0,
        "portfolio_return": 0.0,
        "benchmark_return": 0.0,
        "excess_return": 0.0,
        "hit_excess": 0,
        "hit_positive": 0,
        "selected": "",
        "selected_names": "",
        "selected_groups": "",
    }


def summarize_trades(trades: pd.DataFrame, label: str) -> dict[str, object]:
    returns = trades["portfolio_return"].astype(float)
    excess = trades["excess_return"].astype(float)
    invested = trades[trades["invested"].eq(1)]
    eq = (1.0 + returns.fillna(0.0)).cumprod()
    excess_eq = (1.0 + excess.fillna(0.0)).cumprod()
    return {
        "label": label,
        "periods": int(len(trades)),
        "invested_periods": int(trades["invested"].sum()),
        "coverage": float(trades["invested"].mean()) if len(trades) else np.nan,
        "avg_selected_count": float(invested["selected_count"].mean()) if not invested.empty else 0.0,
        "cumulative_return": float(eq.iloc[-1] - 1.0) if not eq.empty else np.nan,
        "cumulative_excess_return": float(excess_eq.iloc[-1] - 1.0) if not excess_eq.empty else np.nan,
        "CAGR": cagr(eq.iloc[-1], len(trades), 12) if not eq.empty else np.nan,
        "MDD": max_drawdown(eq),
        "Sharpe": sharpe(returns, 12),
        "avg_return": float(returns.mean()),
        "avg_excess": float(excess.mean()),
        "invested_avg_return": float(invested["portfolio_return"].mean()) if not invested.empty else np.nan,
        "invested_avg_excess": float(invested["excess_return"].mean()) if not invested.empty else np.nan,
        "hit_excess": float(invested["hit_excess"].mean()) if not invested.empty else np.nan,
        "hit_positive": float(invested["hit_positive"].mean()) if not invested.empty else np.nan,
    }


def cagr(final_value: float, periods: int, periods_per_year: int) -> float:
    if periods <= 0 or final_value <= 0:
        return np.nan
    return float(final_value ** (periods_per_year / periods) - 1.0)


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return np.nan
    return float((equity / equity.cummax() - 1.0).min())


def sharpe(returns: pd.Series, periods_per_year: int) -> float:
    std = returns.std()
    if pd.isna(std) or std == 0:
        return np.nan
    return float(returns.mean() / std * np.sqrt(periods_per_year))


def monthly_metrics(trades: pd.DataFrame) -> pd.DataFrame:
    out = trades.copy()
    out["year"] = pd.to_datetime(out["date"]).dt.year
    rows = []
    for (label, year), group in out.groupby(["label", "year"]):
        rows.append(summarize_trades(group.drop(columns=["year"]), f"{label}_{year}") | {"label_base": label, "year": int(year)})
    return pd.DataFrame(rows)


def selected_group_stats(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    invested = trades[trades["invested"].eq(1)].copy()
    for _, row in invested.iterrows():
        groups = str(row.get("selected_groups", "")).split(",")
        names = str(row.get("selected_names", "")).split(",")
        tickers = str(row.get("selected", "")).split(",")
        n = max(1, len([g for g in groups if g]))
        for ticker, name, group in zip(tickers, names, groups):
            if ticker:
                rows.append(
                    {
                        "label": row["label"],
                        "ticker": ticker,
                        "name": name,
                        "group": group,
                        "date": row["date"],
                        "allocated_excess": float(row["excess_return"]) / n,
                    }
                )
    if not rows:
        return pd.DataFrame()
    picks = pd.DataFrame(rows)
    return (
        picks.groupby(["label", "group"], as_index=False)
        .agg(picks=("date", "size"), dates=("date", "nunique"), avg_allocated_excess=("allocated_excess", "mean"))
        .sort_values(["label", "picks"], ascending=[True, False])
    )


def selected_ticker_stats(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    invested = trades[trades["invested"].eq(1)].copy()
    for _, row in invested.iterrows():
        groups = str(row.get("selected_groups", "")).split(",")
        names = str(row.get("selected_names", "")).split(",")
        tickers = str(row.get("selected", "")).split(",")
        n = max(1, len([t for t in tickers if t]))
        for ticker, name, group in zip(tickers, names, groups):
            if ticker:
                rows.append(
                    {
                        "label": row["label"],
                        "ticker": ticker,
                        "name": name,
                        "group": group,
                        "date": row["date"],
                        "allocated_excess": float(row["excess_return"]) / n,
                    }
                )
    if not rows:
        return pd.DataFrame()
    picks = pd.DataFrame(rows)
    return (
        picks.groupby(["label", "ticker", "name", "group"], as_index=False)
        .agg(picks=("date", "size"), dates=("date", "nunique"), avg_allocated_excess=("allocated_excess", "mean"))
        .sort_values(["label", "picks"], ascending=[True, False])
    )


def plot_equity(trades: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, group in trades.groupby("label"):
        group = group.sort_values("date")
        eq = (1.0 + group["portfolio_return"].fillna(0.0)).cumprod()
        ax.plot(group["date"], eq, label=label)
    ax.set_title("Leadership V2 Walk-forward Equity")
    ax.set_ylabel("Growth of 1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
