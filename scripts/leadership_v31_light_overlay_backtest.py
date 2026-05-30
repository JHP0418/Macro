from __future__ import annotations

import argparse
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


DEFAULT_INPUT = Path("outputs/leadership_v2_walkforward/walkforward_predictions.csv")
DEFAULT_OUTPUT = Path("outputs/leadership_v31_light_overlay")

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
BENCHMARK_SENSITIVE_EXCLUDES = {"105010.KS", "101280.KS"}  # Latin America, Japan TOPIX


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Leadership v3.1 light overlay with benchmark-strong exclusions.")
    p.add_argument("--input", default=str(DEFAULT_INPUT))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    p.add_argument("--tree-std-threshold", type=float, default=0.03067)
    p.add_argument("--benchmark-20d-threshold", type=float, default=0.03)
    p.add_argument("--benchmark-60d-threshold", type=float, default=0.08)
    p.add_argument("--plot", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(args.input, parse_dates=["date"])
    pred = add_taxonomy(pred)
    pred = add_v31_light_score(pred, args.benchmark_20d_threshold, args.benchmark_60d_threshold)
    pred.to_csv(out_dir / "predictions_with_v31_light_score.csv", index=False, encoding="utf-8-sig")

    portfolio, holdings, orders = run_backtest(
        pred,
        tree_std_threshold=args.tree_std_threshold,
    )
    summary = pd.DataFrame([summarize(portfolio, "v31_light_overlay_benchmark_exclude")])
    yearly = yearly_summary(portfolio)

    portfolio.to_csv(out_dir / "portfolio_returns.csv", index=False, encoding="utf-8-sig")
    holdings.to_csv(out_dir / "target_holdings.csv", index=False, encoding="utf-8-sig")
    orders.to_csv(out_dir / "rebalance_orders.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(out_dir / "yearly.csv", index=False, encoding="utf-8-sig")

    if args.plot:
        plot_equity(portfolio, out_dir / "equity_curve.png")

    print(summary.to_string(index=False))
    print()
    print(yearly.to_string(index=False))


def add_v31_light_score(frame: pd.DataFrame, benchmark_20d_threshold: float, benchmark_60d_threshold: float) -> pd.DataFrame:
    out = frame.copy()
    numeric_cols = [
        "ETF_RS_20D",
        "ETF_RS_60D",
        "ETF_RS_120D",
        "RS_slope_20D",
        "Breadth_change_20D",
        "MA60_breadth",
        "ranker_score",
        "rule_20d_score",
        "benchmark_return_20D",
        "benchmark_return_60D",
    ]
    for col in numeric_cols:
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
    weak_trend = out["MA60_breadth"].lt(0.45) | out["Breadth_change_20D"].lt(0)
    out["benchmark_strong"] = out["benchmark_return_20D"].gt(benchmark_20d_threshold) | out["benchmark_return_60D"].gt(
        benchmark_60d_threshold
    )
    out["benchmark_strong_excluded"] = out["benchmark_strong"] & out["etf_ticker"].isin(BENCHMARK_SENSITIVE_EXCLUDES)

    out["v31_light_score"] = out["ranker_score"] + 0.03 * out["persistent_rs"]
    out["v31_light_score"] += np.where(out["is_mean_reversion_group"] & short_rebound, -0.15, 0.0)
    out["v31_light_score"] += np.where(out["is_cyclical_group"] & weak_persistence, -0.10, 0.0)
    out["v31_light_score"] += np.where(out["is_cyclical_group"] & weak_trend, -0.05, 0.0)
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


def run_backtest(pred: pd.DataFrame, tree_std_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    portfolio_rows = []
    holding_rows = []
    order_rows = []
    previous_weights: dict[str, float] = {}
    previous_meta: dict[str, object] = {}

    for dt in monthly_dates(pred):
        sample = pred[pred["date"].eq(dt)].copy()
        filtered = sample[~sample["benchmark_strong_excluded"]].copy()
        if filtered.empty:
            filtered = sample

        active_score = "v31_light_score" if sample["ranker_group_score_std"].ge(tree_std_threshold).any() else "rule_20d_score"
        overlay_leg = "tree" if active_score == "v31_light_score" else "rule"
        selected = allocate_ranked(filtered, active_score, risk_caps_100())
        if selected["target_weight"].sum() < 1.0 - 1e-9:
            selected = allocate_ranked(sample, active_score, risk_caps_100())

        selected["date"] = dt
        selected["strategy"] = "v31_light_overlay_benchmark_exclude"
        selected["overlay_leg"] = overlay_leg

        benchmark = benchmark_return(sample)
        returns = []
        excesses = []
        for row in selected.itertuples(index=False):
            base = sample[sample["etf_ticker"].eq(row.ticker)]
            if base.empty:
                returns.append(0.0)
                excesses.append(0.0)
                continue
            ret = float(base["forward_20D_return"].iloc[0])
            returns.append(ret)
            excesses.append(ret - benchmark)
        selected["forward_return"] = returns
        selected["benchmark_return"] = benchmark
        selected["forward_excess"] = excesses
        selected["weighted_return"] = selected["target_weight"] * selected["forward_return"]
        selected["weighted_excess"] = selected["target_weight"] * selected["forward_excess"]

        portfolio_rows.append(
            {
                "date": dt,
                "strategy": "v31_light_overlay_benchmark_exclude",
                "overlay_leg": overlay_leg,
                "portfolio_return": float(selected["weighted_return"].sum()),
                "benchmark_return": benchmark,
                "excess_return": float(selected["weighted_excess"].sum()),
                "risk_weight": float(selected["target_weight"].sum()),
                "excluded_by_benchmark_strong": ",".join(
                    sample.loc[sample["benchmark_strong_excluded"], "etf_ticker"].astype(str).tolist()
                ),
                "holdings": ",".join(f"{r.ticker}:{r.target_weight:.4f}" for r in selected.itertuples(index=False)),
            }
        )
        holding_rows.extend(selected.to_dict("records"))

        current_weights = dict(zip(selected["ticker"], selected["target_weight"]))
        current_meta = {row.ticker: row for row in selected.itertuples(index=False)}
        for ticker in sorted(set(previous_weights) | set(current_weights)):
            old = previous_weights.get(ticker, 0.0)
            new = current_weights.get(ticker, 0.0)
            delta = new - old
            if abs(delta) < 1e-9:
                continue
            meta = current_meta.get(ticker, previous_meta.get(ticker))
            order_rows.append(
                {
                    "date": dt,
                    "ticker": ticker,
                    "name": getattr(meta, "name", ticker),
                    "sub_asset": getattr(meta, "sub_asset", ""),
                    "previous_weight": old,
                    "target_weight": new,
                    "trade_weight": delta,
                    "action": "BUY" if delta > 0 else "SELL",
                    "overlay_leg": overlay_leg,
                }
            )
        previous_weights = current_weights
        previous_meta = current_meta

    return pd.DataFrame(portfolio_rows), pd.DataFrame(holding_rows), pd.DataFrame(order_rows)


def allocate_ranked(sample: pd.DataFrame, score_col: str, caps: dict[str, float]) -> pd.DataFrame:
    rows = []
    used_cyclical_slots = 0
    for _, row in sample.sort_values(score_col, ascending=False).iterrows():
        if bool(row.get("is_cyclical_group", False)) and used_cyclical_slots >= 2:
            continue
        rows.append(row)
        if bool(row.get("is_cyclical_group", False)):
            used_cyclical_slots += 1
        allocated = allocate_by_caps(pd.DataFrame(rows), score_col, 1.0, "sub_asset", caps, ETF_CAP)
        if allocated[allocated["ticker"].ne("CASH_KRW")]["target_weight"].sum() >= 1.0 - 1e-12:
            return allocated[allocated["ticker"].ne("CASH_KRW")]

    allocated = allocate_by_caps(pd.DataFrame(rows), score_col, 1.0, "sub_asset", caps, ETF_CAP)
    return allocated[allocated["ticker"].ne("CASH_KRW")]


def benchmark_return(sample: pd.DataFrame) -> float:
    kospi200 = sample[sample["etf_ticker"].eq("069500.KS")]
    if not kospi200.empty:
        return float(kospi200["forward_20D_return"].iloc[0])
    return float(sample["benchmark_forward_20D_return"].mean())


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
    for year, group in out.groupby("year"):
        row = summarize(group.drop(columns=["year"]), f"year_{year}")
        row["year"] = int(year)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_equity(portfolio: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    eq = (1 + portfolio["portfolio_return"].fillna(0)).cumprod()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(portfolio["date"], eq, label="v3.1 light benchmark exclude")
    ax.set_title("Leadership v3.1 Light Overlay")
    ax.set_ylabel("Growth of 1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
