from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from leadership_v2_constrained_70_30_backtest import (
    ETF_CAP,
    RISK_CAPS,
    add_taxonomy,
    allocate_by_caps,
    cagr,
    max_drawdown,
    sharpe,
)


DEFAULT_INPUT = Path("data/processed/long_lived_scored_features.csv")
DEFAULT_OUTPUT = Path("outputs/leadership_v31_excess_ranker")

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

MEAN_REVERSION_GROUPS = {"China equity", "China/HK growth", "Korea cyclical", "Korea value"}
CYCLICAL_GROUPS = {
    "China equity",
    "China/HK growth",
    "Korea cyclical",
    "Korea value",
    "Korea defensive",
    "Commodity/Oil",
    "Oil",
}


@dataclass(frozen=True)
class Window:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Leadership v3.1 excess-return ranker and risk-only capped backtest.")
    p.add_argument("--input", default=str(DEFAULT_INPUT))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    p.add_argument("--min-date", default="2012-01-01")
    p.add_argument("--train-months", type=int, default=36)
    p.add_argument("--valid-months", type=int, default=12)
    p.add_argument("--test-months", type=int, default=1)
    p.add_argument("--min-train-rows", type=int, default=1200)
    p.add_argument("--min-valid-rows", type=int, default=300)
    p.add_argument("--tree-std-threshold", type=float, default=0.03067)
    p.add_argument("--plot", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_frame(args.input, args.min_date)
    feature_cols = [c for c in FEATURES_1M + REGRESSION_FEATURES if c in data.columns]
    windows = make_windows(data["date"], args.train_months, args.valid_months, args.test_months)

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
        raise RuntimeError("No predictions generated.")

    pred = pd.concat(predictions, ignore_index=True).sort_values(["date", "etf_ticker"])
    pred = add_taxonomy(pred)
    pred = add_v31_scores(pred)
    pred.to_csv(out_dir / "walkforward_predictions_v31.csv", index=False, encoding="utf-8-sig")

    if importances:
        importance = (
            pd.concat(importances, ignore_index=True)
            .groupby("feature", as_index=False)
            .agg(importance_gain=("importance_gain", "mean"), importance_split=("importance_split", "mean"))
            .sort_values("importance_gain", ascending=False)
        )
        importance.to_csv(out_dir / "feature_importance_mean.csv", index=False, encoding="utf-8-sig")

    portfolios = []
    holdings = []
    for label, score_col, adjusted in [
        ("v31_excess_ranker_raw", "ranker_score", False),
        ("v31_persistent_rs_overlay", "v31_score", True),
    ]:
        p, h = run_risk_only_backtest(pred, label, score_col, args.tree_std_threshold, adjusted)
        portfolios.append(p)
        holdings.append(h)

    portfolios = pd.concat(portfolios, ignore_index=True)
    holdings = pd.concat(holdings, ignore_index=True)
    summary = pd.DataFrame([summarize(g, label) for label, g in portfolios.groupby("strategy")])
    yearly = yearly_summary(portfolios)
    group_stats = selected_group_stats(holdings, portfolios)

    portfolios.to_csv(out_dir / "portfolio_returns.csv", index=False, encoding="utf-8-sig")
    holdings.to_csv(out_dir / "target_holdings.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(out_dir / "yearly.csv", index=False, encoding="utf-8-sig")
    group_stats.to_csv(out_dir / "selected_group_stats.csv", index=False, encoding="utf-8-sig")

    if args.plot:
        plot_equity(portfolios, out_dir / "equity_curve.png")

    meta = {
        "input": str(args.input),
        "target": "forward_20D_excess rank within date/model_group",
        "feature_cols": feature_cols,
        "windows": len(windows),
        "prediction_rows": int(len(pred)),
        "tree_std_threshold": args.tree_std_threshold,
        "risk_caps_100": risk_caps_100(),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(summary.sort_values("Sharpe", ascending=False).to_string(index=False))
    print()
    print(yearly.sort_values(["strategy", "year"]).to_string(index=False))


def load_frame(path: str | Path, min_date: str) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"])
    frame = frame[frame["date"].ge(pd.Timestamp(min_date))].copy()
    required = [
        "date",
        "etf_ticker",
        "model_group",
        "ranking_group",
        "rule_20d_score",
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
    frame = frame.dropna(subset=["rule_20d_score", "forward_20D_return", "benchmark_forward_20D_return", "forward_20D_excess"])
    if "holding_count" in frame.columns:
        frame = frame[frame["holding_count"].fillna(0).ge(2)]
    group_size = frame.groupby(["date", "model_group"])["etf_ticker"].transform("size")
    frame = frame[group_size.ge(2)].copy()
    frame["excess_rank_label_int"] = make_excess_rank_label(frame)
    return frame.sort_values(["date", "model_group", "etf_ticker"]).reset_index(drop=True)


def make_excess_rank_label(frame: pd.DataFrame) -> pd.Series:
    rank = frame.groupby(["date", "model_group"])["forward_20D_excess"].rank(ascending=True, method="first")
    return (rank - 1).astype(int)


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

    from lightgbm import LGBMRanker, early_stopping, log_evaluation

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
    data = frame.dropna(subset=["excess_rank_label_int"]).sort_values(["date", "model_group", "etf_ticker"]).reset_index(drop=True)
    group_size = data.groupby(["date", "model_group"])["etf_ticker"].transform("size")
    data = data[group_size.ge(2)].reset_index(drop=True)
    x = data[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = x.groupby([data["date"], data["model_group"]]).transform(lambda s: s.fillna(s.median())).fillna(0.0)
    y = data["excess_rank_label_int"].astype(int)
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


def add_v31_scores(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in ["ETF_RS_20D", "ETF_RS_60D", "ETF_RS_120D", "RS_slope_20D", "Breadth_change_20D", "MA60_breadth"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out["persistent_rs"] = (
        0.15 * z_by_date(out, "ETF_RS_20D")
        + 0.40 * z_by_date(out, "ETF_RS_60D")
        + 0.35 * z_by_date(out, "ETF_RS_120D")
        + 0.10 * z_by_date(out, "RS_slope_20D")
    )
    out["is_mean_reversion_group"] = out["ranking_group"].isin(MEAN_REVERSION_GROUPS)
    out["is_cyclical_group"] = out["ranking_group"].isin(CYCLICAL_GROUPS)

    short_rebound = out["ETF_RS_20D"].gt(0) & (out["ETF_RS_60D"].lt(0) | out["ETF_RS_120D"].lt(0))
    weak_persistence = out["ETF_RS_60D"].lt(0) & out["ETF_RS_120D"].lt(0)
    weak_trend = (out["MA60_breadth"].lt(0.45)) | (out["Breadth_change_20D"].lt(0))

    out["penalty_short_rebound"] = np.where(out["is_mean_reversion_group"] & short_rebound, -0.35, 0.0)
    out["penalty_weak_persistence"] = np.where(out["is_cyclical_group"] & weak_persistence, -0.30, 0.0)
    out["penalty_weak_trend"] = np.where(out["is_cyclical_group"] & weak_trend, -0.15, 0.0)
    out["v31_penalty"] = out["penalty_short_rebound"] + out["penalty_weak_persistence"] + out["penalty_weak_trend"]
    out["v31_score"] = out["ranker_score"] + 0.20 * out["persistent_rs"] + out["v31_penalty"]
    out = add_score_context(out, "v31_score", "v31")
    return out


def z_by_date(frame: pd.DataFrame, col: str) -> pd.Series:
    x = pd.to_numeric(frame[col], errors="coerce")
    mean = x.groupby(frame["date"]).transform("mean")
    std = x.groupby(frame["date"]).transform("std").replace(0, np.nan)
    return ((x - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def monthly_dates(frame: pd.DataFrame) -> list[pd.Timestamp]:
    idx = pd.DatetimeIndex(pd.to_datetime(frame["date"]).dropna().unique()).sort_values()
    return pd.Series(idx, index=idx).groupby(idx.to_period("M")).max().tolist()


def risk_caps_100() -> dict[str, float]:
    return {k: v / 0.70 for k, v in RISK_CAPS.items()}


def run_risk_only_backtest(
    pred: pd.DataFrame,
    label: str,
    score_col: str,
    threshold: float,
    adjusted: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    portfolio_rows = []
    holding_rows = []
    caps = risk_caps_100()
    for dt in monthly_dates(pred):
        sample = pred[pred["date"].eq(dt)].copy()
        leg = "tree"
        active_score = score_col
        if not sample["ranker_group_score_std"].ge(threshold).any():
            active_score = "rule_20d_score" if not adjusted else "v31_score"
            leg = "rule"
        selected = allocate_ranked(sample, active_score, caps)
        selected["date"] = dt
        selected["strategy"] = label
        selected["overlay_leg"] = leg
        returns = []
        benchmarks = []
        excesses = []
        source = pred[pred["date"].eq(dt)]
        for row in selected.itertuples(index=False):
            base = source[source["etf_ticker"].eq(row.ticker)]
            if base.empty:
                returns.append(0.0)
                benchmarks.append(0.0)
                excesses.append(0.0)
            else:
                returns.append(float(base["forward_20D_return"].iloc[0]))
                benchmarks.append(float(base["benchmark_forward_20D_return"].iloc[0]))
                excesses.append(float(base["forward_20D_excess"].iloc[0]))
        selected["forward_return"] = returns
        selected["benchmark_return"] = benchmarks
        selected["forward_excess"] = excesses
        selected["weighted_return"] = selected["target_weight"] * selected["forward_return"]
        selected["weighted_benchmark_return"] = selected["target_weight"] * selected["benchmark_return"]
        selected["weighted_excess"] = selected["target_weight"] * selected["forward_excess"]
        portfolio_return = float(selected["weighted_return"].sum())
        benchmark_return = float(selected["weighted_benchmark_return"].sum())
        excess_return = float(selected["weighted_excess"].sum())
        portfolio_rows.append(
            {
                "date": dt,
                "strategy": label,
                "overlay_leg": leg,
                "portfolio_return": portfolio_return,
                "benchmark_return": benchmark_return,
                "excess_return": excess_return,
                "risk_weight": float(selected["target_weight"].sum()),
                "holdings": ",".join(f"{r.ticker}:{r.target_weight:.4f}" for r in selected.itertuples(index=False)),
            }
        )
        holding_rows.extend(selected.to_dict("records"))
    return pd.DataFrame(portfolio_rows), pd.DataFrame(holding_rows)


def allocate_ranked(sample: pd.DataFrame, score_col: str, caps: dict[str, float]) -> pd.DataFrame:
    ordered = sample.sort_values(score_col, ascending=False).copy()
    rows = []
    used_cyclical_slots = 0
    for _, row in ordered.iterrows():
        if row.get("is_cyclical_group", False) and used_cyclical_slots >= 2:
            continue
        tmp = pd.DataFrame([row])
        tmp_alloc = allocate_by_caps(
            pd.concat([pd.DataFrame(rows), tmp], ignore_index=True) if rows else tmp,
            score_col=score_col,
            sleeve_weight=1.0,
            category_col="sub_asset",
            category_caps=caps,
            etf_cap=ETF_CAP,
        )
        if len(tmp_alloc[tmp_alloc["ticker"].eq(row["etf_ticker"])]) == 0:
            continue
        rows.append(row)
        if row.get("is_cyclical_group", False):
            used_cyclical_slots += 1
        full_alloc = allocate_by_caps(
            pd.DataFrame(rows),
            score_col=score_col,
            sleeve_weight=1.0,
            category_col="sub_asset",
            category_caps=caps,
            etf_cap=ETF_CAP,
        )
        if full_alloc["target_weight"].sum() >= 1.0 - 1e-12:
            return full_alloc
    return allocate_by_caps(pd.DataFrame(rows), score_col, 1.0, "sub_asset", caps, ETF_CAP)


def summarize(portfolio: pd.DataFrame, label: str) -> dict[str, object]:
    returns = portfolio["portfolio_return"].astype(float)
    excess = portfolio["excess_return"].astype(float)
    eq = (1 + returns.fillna(0)).cumprod()
    exeq = (1 + excess.fillna(0)).cumprod()
    return {
        "label": label,
        "periods": int(len(portfolio)),
        "cumulative_return": float(eq.iloc[-1] - 1),
        "cumulative_excess_return": float(exeq.iloc[-1] - 1),
        "CAGR": cagr(eq.iloc[-1], len(portfolio), 12),
        "MDD": max_drawdown(eq),
        "Sharpe": sharpe(returns, 12),
        "avg_monthly_return": float(returns.mean()),
        "avg_monthly_excess": float(excess.mean()),
        "hit_excess": float((excess > 0).mean()),
        "avg_risk_weight": float(portfolio["risk_weight"].mean()),
    }


def yearly_summary(portfolio: pd.DataFrame) -> pd.DataFrame:
    out = portfolio.copy()
    out["year"] = pd.to_datetime(out["date"]).dt.year
    rows = []
    for (strategy, year), group in out.groupby(["strategy", "year"]):
        row = summarize(group.drop(columns=["year"]), strategy)
        row["year"] = int(year)
        rows.append(row)
    return pd.DataFrame(rows)


def selected_group_stats(holdings: pd.DataFrame, portfolio: pd.DataFrame) -> pd.DataFrame:
    stats = (
        holdings.groupby(["strategy", "sub_asset"], as_index=False)
        .agg(months=("date", "nunique"), avg_weight=("target_weight", "mean"), total_weight=("target_weight", "sum"))
        .sort_values(["strategy", "total_weight"], ascending=[True, False])
    )
    stats["selection_share"] = stats["months"] / portfolio["date"].nunique()
    return stats


def plot_equity(portfolios: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for strategy, group in portfolios.groupby("strategy"):
        group = group.sort_values("date")
        eq = (1 + group["portfolio_return"].fillna(0)).cumprod()
        ax.plot(group["date"], eq, label=strategy)
    ax.set_title("Leadership v3.1 Excess Ranker Risk-Only")
    ax.set_ylabel("Growth of 1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
