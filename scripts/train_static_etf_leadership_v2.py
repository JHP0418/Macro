from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


FEATURES_1W = [
    "ETF_RS_20D",
    "RS_slope_20D",
    "weighted_component_RS_20D",
    "median_component_RS_20D",
    "RS_positive_share",
    "MA60_breadth",
    "Breadth_change_20D",
    "median_component_return_20D",
    "mean_minus_median_return_20D",
    "top20_component_return_mean",
    "bottom20_component_return_mean",
    "top5_return_contribution_share",
    "HP_change_20D",
]

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Static-holdings ETF leadership V2: separate 5D/20D rankers and selective entry.")
    p.add_argument("--input", default=str(ROOT / "outputs" / "etf_leadership_static_holdings_approx" / "rule_scores.csv"))
    p.add_argument("--output-dir", default=str(ROOT / "outputs" / "etf_leadership_static_v2"))
    p.add_argument("--min-date", default="2021-01-01")
    p.add_argument("--train-end", default="2022-12-31")
    p.add_argument("--valid-end", default="2024-12-31")
    p.add_argument("--test-start", default="2025-01-01")
    p.add_argument("--min-holdings", type=int, default=2)
    p.add_argument("--min-hp", type=float, default=-1.0)
    return p.parse_args()


def zscore_by_date(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in cols:
        if col not in out.columns:
            continue
        x = pd.to_numeric(out[col], errors="coerce")
        q = x.groupby(out["date"]).quantile([0.01, 0.99]).unstack()
        if q.shape[1] < 2:
            out[f"v2_z_{col}"] = 0.0
            continue
        lo = out["date"].map(q.iloc[:, 0])
        hi = out["date"].map(q.iloc[:, 1])
        x = x.clip(lo, hi)
        mean = x.groupby(out["date"]).transform("mean")
        std = x.groupby(out["date"]).transform("std").replace(0, np.nan)
        out[f"v2_z_{col}"] = ((x - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def z(frame: pd.DataFrame, col: str) -> pd.Series:
    return frame.get(f"v2_z_{col}", pd.Series(0.0, index=frame.index)).fillna(0.0)


def add_v2_rule_scores(frame: pd.DataFrame) -> pd.DataFrame:
    cols = sorted(set(FEATURES_1W + FEATURES_1M))
    out = zscore_by_date(frame, cols)

    # 1W leadership: fast relative momentum, fresh component participation, and breadth improvement.
    out["rule_5d_score"] = (
        0.24 * z(out, "ETF_RS_20D")
        + 0.14 * z(out, "RS_slope_20D")
        + 0.22 * z(out, "weighted_component_RS_20D")
        + 0.12 * z(out, "median_component_RS_20D")
        + 0.12 * z(out, "RS_positive_share")
        + 0.08 * z(out, "Breadth_change_20D")
        + 0.08 * z(out, "median_component_return_20D")
        - 0.06 * z(out, "top5_return_contribution_share")
    )

    # 1M leadership: persistent ETF RS plus broad component trend and high-proximity confirmation.
    out["rule_20d_score"] = (
        0.14 * z(out, "ETF_RS_20D")
        + 0.18 * z(out, "ETF_RS_60D")
        + 0.12 * z(out, "ETF_RS_120D")
        + 0.08 * z(out, "RS_slope_20D")
        + 0.12 * z(out, "weighted_component_RS_60D")
        + 0.08 * z(out, "weighted_component_RS_20D")
        + 0.08 * z(out, "RS_positive_share")
        + 0.08 * z(out, "MA60_breadth")
        + 0.06 * z(out, "MA200_breadth")
        + 0.06 * z(out, "HP90_share")
        + 0.04 * z(out, "HP_change_20D")
        - 0.04 * z(out, "top5_return_contribution_share")
    )
    out["rule_5d_rank"] = out.groupby("date")["rule_5d_score"].rank(ascending=False, method="first")
    out["rule_20d_rank"] = out.groupby("date")["rule_20d_score"].rank(ascending=False, method="first")
    return out


def filter_quality(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"])
    for col in [
        "holding_count",
        "weighted_HP",
        "MA60_breadth",
        "forward_5D_excess",
        "forward_20D_excess",
        "forward_5D_return",
        "forward_20D_return",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    mask = (
        out["date"].ge(pd.Timestamp(args.min_date))
        & out["holding_count"].ge(args.min_holdings)
        & (out["weighted_HP"].ge(args.min_hp) | out["weighted_HP"].isna())
        & out["weighted_component_RS_20D"].notna()
        & out["forward_5D_excess"].notna()
        & out["forward_20D_excess"].notna()
    )
    return out.loc[mask].sort_values(["date", "etf_ticker"]).reset_index(drop=True)


def prepare_rank_data(frame: pd.DataFrame, label_col: str, features: list[str]) -> tuple[pd.DataFrame, pd.Series, list[int], pd.DataFrame]:
    data = frame.dropna(subset=[label_col]).sort_values(["date", "etf_ticker"]).reset_index(drop=True)
    x = data[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = x.groupby(data["date"]).transform(lambda s: s.fillna(s.median())).fillna(0.0)
    y = data[label_col].astype(int)
    group = data.groupby("date", sort=False).size().astype(int).tolist()
    return x, y, group, data


def train_ranker(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, label_col: str, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    from lightgbm import LGBMRanker, early_stopping, log_evaluation

    x_train, y_train, group_train, _ = prepare_rank_data(train, label_col, features)
    x_valid, y_valid, group_valid, _ = prepare_rank_data(valid, label_col, features)
    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=700,
        learning_rate=0.025,
        num_leaves=11,
        max_depth=3,
        min_child_samples=40,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_alpha=2.0,
        reg_lambda=10.0,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        x_train,
        y_train,
        group=group_train,
        eval_set=[(x_valid, y_valid)],
        eval_group=[group_valid],
        eval_at=[3, 5, 10],
        callbacks=[early_stopping(80), log_evaluation(100)],
    )

    preds = []
    for split_name, part in [("train", train), ("valid", valid), ("test", test)]:
        x, _, _, data = prepare_rank_data(part, label_col, features)
        data = data.copy()
        data["split"] = split_name
        data["ranker_score"] = model.predict(x)
        data["ranker_rank"] = data.groupby("date")["ranker_score"].rank(ascending=False, method="first")
        preds.append(data)

    importance = pd.DataFrame(
        {
            "feature": features,
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
            "importance_split": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False)
    return pd.concat(preds, ignore_index=True), importance


def split_frame(frame: pd.DataFrame, train_end: str, valid_end: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = frame[frame["date"].le(pd.Timestamp(train_end))]
    valid = frame[frame["date"].gt(pd.Timestamp(train_end)) & frame["date"].le(pd.Timestamp(valid_end))]
    test = frame[frame["date"].gt(pd.Timestamp(valid_end))]
    return train, valid, test


def select_rebalance_dates(dates: pd.Series, horizon: str) -> list[pd.Timestamp]:
    idx = pd.DatetimeIndex(pd.to_datetime(dates).dropna().unique()).sort_values()
    if horizon == "1M":
        return pd.Series(idx, index=idx).groupby(idx.to_period("M")).max().tolist()
    return pd.Series(idx, index=idx).groupby(idx.to_period("W-FRI")).max().tolist()


def backtest(frame: pd.DataFrame, score_col: str, horizon: str, top_k: int, split: str = "test") -> tuple[pd.DataFrame, dict]:
    data = frame[frame["split"].eq(split)].copy() if "split" in frame.columns else frame.copy()
    ret_col = "forward_5D_return" if horizon == "1W" else "forward_20D_return"
    bench_col = "benchmark_forward_5D_return" if horizon == "1W" else "benchmark_forward_20D_return"
    excess_col = "forward_5D_excess" if horizon == "1W" else "forward_20D_excess"
    rows = []
    for dt in select_rebalance_dates(data["date"], horizon):
        sample = data[data["date"].eq(dt)].dropna(subset=[score_col, ret_col, bench_col, excess_col])
        if sample.shape[0] < top_k:
            continue
        top = sample.nlargest(top_k, score_col)
        rows.append(
            {
                "date": dt,
                "horizon": horizon,
                "score_col": score_col,
                "top_k": top_k,
                "portfolio_return": top[ret_col].mean(),
                "benchmark_return": top[bench_col].mean(),
                "excess_return": top[excess_col].mean(),
                "hit_excess": int(top[excess_col].mean() > 0),
                "hit_positive": int(top[ret_col].mean() > 0),
                "selected": ",".join(top["etf_ticker"].astype(str).tolist()),
            }
        )
    raw = pd.DataFrame(rows)
    return raw, summarize(raw, f"{score_col}_{horizon}_top{top_k}")


def summarize(raw: pd.DataFrame, label: str) -> dict:
    if raw.empty:
        return {"label": label, "periods": 0}
    periods_per_year = 52 if raw["horizon"].iloc[0] == "1W" else 12
    returns = raw["portfolio_return"].astype(float)
    excess = raw["excess_return"].astype(float)
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    bench = (1.0 + raw["benchmark_return"].astype(float).fillna(0.0)).cumprod()
    std = returns.std(ddof=1)
    sharpe = np.nan if pd.isna(std) or std == 0 else returns.mean() / std * np.sqrt(periods_per_year)
    mdd = (equity / equity.cummax() - 1.0).min()
    return {
        "label": label,
        "horizon": raw["horizon"].iloc[0],
        "top_k": int(raw["top_k"].iloc[0]),
        "periods": int(raw.shape[0]),
        "cumulative_return": float(equity.iloc[-1] - 1.0),
        "benchmark_cumulative_return": float(bench.iloc[-1] - 1.0),
        "CAGR": float(equity.iloc[-1] ** (periods_per_year / raw.shape[0]) - 1.0) if equity.iloc[-1] > 0 else np.nan,
        "MDD": float(mdd),
        "Sharpe": float(sharpe) if not pd.isna(sharpe) else np.nan,
        "hit_excess": float(raw["hit_excess"].mean()),
        "hit_positive": float(raw["hit_positive"].mean()),
        "avg_return": float(returns.mean()),
        "avg_excess": float(excess.mean()),
    }


def add_blends(pred_5d: pd.DataFrame, pred_20d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = pred_5d.copy()
    b = pred_20d.copy()
    a["z_ranker_5d"] = a.groupby("date")["ranker_score"].transform(cs_z)
    a["z_rule_5d"] = a.groupby("date")["rule_5d_score"].transform(cs_z)
    a["blend_5d_score"] = 0.55 * a["z_rule_5d"] + 0.45 * a["z_ranker_5d"]
    b["z_ranker_20d"] = b.groupby("date")["ranker_score"].transform(cs_z)
    b["z_rule_20d"] = b.groupby("date")["rule_20d_score"].transform(cs_z)
    b["blend_20d_score"] = 0.45 * b["z_rule_20d"] + 0.55 * b["z_ranker_20d"]
    return a, b


def cs_z(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    std = x.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=s.index)
    return (x - x.mean()) / std


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.input, parse_dates=["date"])
    filtered = filter_quality(raw, args)
    scored = add_v2_rule_scores(filtered)
    train, valid, test = split_frame(scored, args.train_end, args.valid_end)

    pred_5d, imp_5d = train_ranker(train, valid, test, "label_5D_rank_int", FEATURES_1W)
    pred_20d, imp_20d = train_ranker(train, valid, test, "label_20D_rank_int", FEATURES_1M)
    pred_5d, pred_20d = add_blends(pred_5d, pred_20d)

    scored.to_csv(out_dir / "v2_static_scored_features.csv", index=False, encoding="utf-8-sig")
    pred_5d.to_csv(out_dir / "v2_ranker_5d_predictions.csv", index=False, encoding="utf-8-sig")
    pred_20d.to_csv(out_dir / "v2_ranker_20d_predictions.csv", index=False, encoding="utf-8-sig")
    imp_5d.to_csv(out_dir / "v2_ranker_5d_feature_importance.csv", index=False, encoding="utf-8-sig")
    imp_20d.to_csv(out_dir / "v2_ranker_20d_feature_importance.csv", index=False, encoding="utf-8-sig")

    summaries = []
    raws = []
    for top_k in [1, 2, 3, 5]:
        for score_col in ["rule_5d_score", "ranker_score", "blend_5d_score"]:
            raw_bt, summary = backtest(pred_5d, score_col, "1W", top_k)
            summary["model"] = score_col
            summaries.append(summary)
            raw_bt["model"] = score_col
            raws.append(raw_bt)
        for score_col in ["rule_20d_score", "ranker_score", "blend_20d_score"]:
            raw_bt, summary = backtest(pred_20d, score_col, "1M", top_k)
            summary["model"] = score_col
            summaries.append(summary)
            raw_bt["model"] = score_col
            raws.append(raw_bt)

    summary_df = pd.DataFrame(summaries).sort_values(["horizon", "Sharpe"], ascending=[True, False])
    raw_df = pd.concat(raws, ignore_index=True) if raws else pd.DataFrame()
    summary_df.to_csv(out_dir / "v2_backtest_summary.csv", index=False, encoding="utf-8-sig")
    raw_df.to_csv(out_dir / "v2_backtest_trades.csv", index=False, encoding="utf-8-sig")
    print(f"input rows={raw.shape[0]:,} filtered rows={filtered.shape[0]:,} dates={filtered['date'].nunique():,} etfs={filtered['etf_ticker'].nunique():,}")
    print(summary_df.head(20).to_string(index=False))
    print("\n5D importance")
    print(imp_5d.head(12).to_string(index=False))
    print("\n20D importance")
    print(imp_20d.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
