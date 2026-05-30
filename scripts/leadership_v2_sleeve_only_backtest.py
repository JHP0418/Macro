from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from leadership_v2_constrained_70_30_backtest import (
    DEFAULT_PREDICTIONS,
    DEFAULT_PRICES,
    DEFAULT_UNIVERSE,
    ETF_CAP,
    RISK_CAPS,
    SAFE_CAPS,
    add_taxonomy,
    allocate_by_caps,
    allocate_safe,
    build_safe_universe,
    cagr,
    forward_price_return,
    max_drawdown,
    monthly_dates,
    select_risky,
    sharpe,
)


DEFAULT_OUTPUT = Path("outputs/leadership_v2_sleeve_only")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V2 leadership risk-only and safe-only sleeve backtests.")
    p.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    p.add_argument("--universe", default=str(DEFAULT_UNIVERSE))
    p.add_argument("--prices", default=str(DEFAULT_PRICES))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    p.add_argument("--tree-std-threshold", type=float, default=0.03067)
    p.add_argument("--horizon-days", type=int, default=20)
    p.add_argument("--plot", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(args.predictions, parse_dates=["date"])
    universe = pd.read_csv(args.universe)
    prices = pd.read_csv(args.prices, parse_dates=["date"]).set_index("date").sort_index()

    pred = add_taxonomy(pred)
    safe_universe = build_safe_universe(universe)
    dates = monthly_dates(pred)

    risk_caps_100 = {k: v / 0.70 for k, v in RISK_CAPS.items()}
    strategies = {
        "risk_only_100": {
            "asset_class": "risk",
            "risk_weight": 1.0,
            "safe_weight": 0.0,
            "risk_caps": risk_caps_100,
        },
        "safe_only_100": {
            "asset_class": "safe",
            "risk_weight": 0.0,
            "safe_weight": 1.0,
            "risk_caps": {},
        },
    }

    summaries = []
    yearly_frames = []
    all_portfolios = []
    all_orders = []
    all_holdings = []

    for label, config in strategies.items():
        portfolio, orders, holdings = run_strategy(
            label=label,
            dates=dates,
            pred=pred,
            prices=prices,
            safe_universe=safe_universe,
            tree_std_threshold=args.tree_std_threshold,
            horizon_days=args.horizon_days,
            risk_weight=float(config["risk_weight"]),
            safe_weight=float(config["safe_weight"]),
            risk_caps=dict(config["risk_caps"]),
        )
        summaries.append(summarize(portfolio, label))
        yf = yearly_summary(portfolio)
        yf.insert(0, "strategy", label)

        portfolio.insert(0, "strategy", label)
        orders.insert(0, "strategy", label)
        holdings.insert(0, "strategy", label)

        yearly_frames.append(yf)
        all_portfolios.append(portfolio)
        all_orders.append(orders)
        all_holdings.append(holdings)

    summary = pd.DataFrame(summaries)
    yearly = pd.concat(yearly_frames, ignore_index=True)
    portfolios = pd.concat(all_portfolios, ignore_index=True)
    orders = pd.concat(all_orders, ignore_index=True)
    holdings = pd.concat(all_holdings, ignore_index=True)

    summary.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(out_dir / "yearly.csv", index=False, encoding="utf-8-sig")
    portfolios.to_csv(out_dir / "portfolio_returns.csv", index=False, encoding="utf-8-sig")
    orders.to_csv(out_dir / "rebalance_orders.csv", index=False, encoding="utf-8-sig")
    holdings.to_csv(out_dir / "target_holdings.csv", index=False, encoding="utf-8-sig")

    if args.plot:
        plot_equity(portfolios, out_dir / "equity_curve.png")

    print(summary.to_string(index=False))
    print()
    print(yearly.to_string(index=False))
    print()
    print(orders.groupby("strategy").tail(25).to_string(index=False))


def run_strategy(
    label: str,
    dates: list[pd.Timestamp],
    pred: pd.DataFrame,
    prices: pd.DataFrame,
    safe_universe: pd.DataFrame,
    tree_std_threshold: float,
    horizon_days: int,
    risk_weight: float,
    safe_weight: float,
    risk_caps: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    portfolio_rows = []
    order_rows = []
    holding_rows = []
    previous_weights: dict[str, float] = {}
    previous_meta: dict[str, dict[str, str]] = {}

    for dt in dates:
        sample = pred[pred["date"].eq(dt)].copy()
        risky = pd.DataFrame()
        leg = "safe_only"
        if risk_weight > 0:
            risky, leg = select_risky(sample, tree_std_threshold)
            risky = allocate_by_caps(
                risky,
                score_col="active_score",
                sleeve_weight=risk_weight,
                category_col="sub_asset",
                category_caps=risk_caps,
                etf_cap=ETF_CAP,
            )
        safe = pd.DataFrame()
        if safe_weight > 0:
            safe = allocate_safe(dt, safe_universe, prices, safe_weight)
        target = pd.concat([risky, safe], ignore_index=True)
        target = target[target["target_weight"].gt(0)].copy()
        target["date"] = dt
        target["overlay_leg"] = leg

        returns = []
        for row in target.itertuples(index=False):
            if row.ticker == "CASH_KRW":
                returns.append(0.0)
            elif row.ticker in sample["etf_ticker"].values:
                r = sample.loc[sample["etf_ticker"].eq(row.ticker), "forward_20D_return"].iloc[0]
                returns.append(float(r))
            else:
                returns.append(forward_price_return(prices, row.ticker, dt, horizon_days))
        target["forward_return"] = returns
        target["weighted_return"] = target["target_weight"] * target["forward_return"].fillna(0.0)

        benchmark_return = benchmark_for_sample(sample, prices, dt, horizon_days, label)
        portfolio_return = float(target["weighted_return"].sum())
        portfolio_rows.append(
            {
                "date": dt,
                "overlay_leg": leg,
                "portfolio_return": portfolio_return,
                "benchmark_return": benchmark_return,
                "excess_return": portfolio_return - benchmark_return,
                "risk_weight": float(target[target["asset_class"].eq("risk")]["target_weight"].sum()),
                "safe_weight": float(target[target["asset_class"].eq("safe")]["target_weight"].sum()),
                "holdings": ",".join(f"{r.ticker}:{r.target_weight:.4f}" for r in target.itertuples(index=False)),
            }
        )

        for row in target.itertuples(index=False):
            holding_rows.append(
                {
                    "date": dt,
                    "ticker": row.ticker,
                    "name": row.name,
                    "asset_class": row.asset_class,
                    "sub_asset": row.sub_asset,
                    "target_weight": row.target_weight,
                    "score": row.score,
                    "overlay_leg": leg,
                }
            )

        current_weights = dict(zip(target["ticker"], target["target_weight"]))
        current_meta = {
            row.ticker: {"name": row.name, "asset_class": row.asset_class, "sub_asset": row.sub_asset}
            for row in target.itertuples(index=False)
        }
        all_tickers = sorted(set(previous_weights) | set(current_weights))
        for ticker in all_tickers:
            old = previous_weights.get(ticker, 0.0)
            new = current_weights.get(ticker, 0.0)
            delta = new - old
            if abs(delta) < 1e-9:
                continue
            meta = current_meta.get(ticker, previous_meta.get(ticker, {}))
            order_rows.append(
                {
                    "date": dt,
                    "ticker": ticker,
                    "name": meta.get("name", ticker),
                    "asset_class": meta.get("asset_class", ""),
                    "sub_asset": meta.get("sub_asset", ""),
                    "previous_weight": old,
                    "target_weight": new,
                    "trade_weight": delta,
                    "action": "BUY" if delta > 0 else "SELL",
                    "overlay_leg": leg,
                }
            )
        previous_weights = current_weights
        previous_meta = current_meta

    return pd.DataFrame(portfolio_rows), pd.DataFrame(order_rows), pd.DataFrame(holding_rows)


def benchmark_for_sample(
    sample: pd.DataFrame,
    prices: pd.DataFrame,
    date: pd.Timestamp,
    horizon_days: int,
    label: str,
) -> float:
    if label == "risk_only_100":
        if "069500.KS" in sample["etf_ticker"].values:
            return float(sample.loc[sample["etf_ticker"].eq("069500.KS"), "forward_20D_return"].iloc[0])
        return float(sample["benchmark_forward_20D_return"].mean())
    ticker = "423160.KS"
    if ticker in prices.columns:
        return forward_price_return(prices, ticker, date, horizon_days)
    return 0.0


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
        "avg_safe_weight": float(portfolio["safe_weight"].mean()),
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


def plot_equity(portfolios: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for strategy, group in portfolios.groupby("strategy"):
        group = group.sort_values("date")
        eq = (1 + group["portfolio_return"].fillna(0)).cumprod()
        ax.plot(group["date"], eq, label=strategy)
    ax.set_title("V2 Sleeve-Only Backtests")
    ax.set_ylabel("Growth of 1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
