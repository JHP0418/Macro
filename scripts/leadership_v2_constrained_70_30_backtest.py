from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_PREDICTIONS = Path("outputs/leadership_v2_walkforward/walkforward_predictions.csv")
DEFAULT_UNIVERSE = Path("data/etf_universe_leadership.csv")
DEFAULT_PRICES = Path("data/gaps_long_lived_cache/gaps_etf_benchmark_prices_2010-01-01_2026-05-18.csv")
DEFAULT_OUTPUT = Path("outputs/leadership_v2_constrained_70_30")

RISK_ASSET_WEIGHT = 0.70
SAFE_ASSET_WEIGHT = 0.30
ETF_CAP = 0.20

RISK_CAPS = {
    "domestic_equity_index": 0.30,
    "domestic_equity_sector": 0.15,
    "overseas_equity_index": 0.30,
    "overseas_equity_sector": 0.10,
    "fx_commodity": 0.20,
}

SAFE_CAPS = {
    "domestic_bond_total": 0.50,
    "domestic_bond_corp": 0.30,
    "overseas_bond_total": 0.50,
    "overseas_bond_corp": 0.30,
    "cash_short_bond": 0.50,
}

SAFE_PRIORITY = [
    "cash_short_bond",
    "domestic_bond_total",
    "overseas_bond_total",
    "overseas_bond_corp",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V2 leadership constrained 70/30 portfolio backtest.")
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

    portfolio_rows = []
    order_rows = []
    previous_weights: dict[str, float] = {}
    for dt in dates:
        sample = pred[pred["date"].eq(dt)].copy()
        risky, leg = select_risky(sample, args.tree_std_threshold)
        risky_alloc = allocate_by_caps(
            risky,
            score_col="active_score",
            sleeve_weight=RISK_ASSET_WEIGHT,
            category_col="sub_asset",
            category_caps=RISK_CAPS,
            etf_cap=ETF_CAP,
        )
        safe_alloc = allocate_safe(dt, safe_universe, prices, SAFE_ASSET_WEIGHT)
        target = pd.concat([risky_alloc, safe_alloc], ignore_index=True)
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
                returns.append(forward_price_return(prices, row.ticker, dt, args.horizon_days))
        target["forward_return"] = returns
        target["weighted_return"] = target["target_weight"] * target["forward_return"].fillna(0.0)
        portfolio_return = float(target["weighted_return"].sum())
        risky_return = float(target[target["asset_class"].eq("risk")]["weighted_return"].sum())
        safe_return = float(target[target["asset_class"].eq("safe")]["weighted_return"].sum())
        benchmark_return = float((sample.nlargest(1, "benchmark_forward_20D_return")["benchmark_forward_20D_return"].mean()))

        row = {
            "date": dt,
            "overlay_leg": leg,
            "portfolio_return": portfolio_return,
            "risky_sleeve_return_contribution": risky_return,
            "safe_sleeve_return_contribution": safe_return,
            "benchmark_return": benchmark_return,
            "excess_return": portfolio_return - benchmark_return,
            "risk_weight": float(target[target["asset_class"].eq("risk")]["target_weight"].sum()),
            "safe_weight": float(target[target["asset_class"].eq("safe")]["target_weight"].sum()),
            "holdings": ",".join(f"{r.ticker}:{r.target_weight:.4f}" for r in target.itertuples(index=False)),
        }
        portfolio_rows.append(row)

        current_weights = dict(zip(target["ticker"], target["target_weight"]))
        all_tickers = sorted(set(previous_weights) | set(current_weights))
        for ticker in all_tickers:
            old = previous_weights.get(ticker, 0.0)
            new = current_weights.get(ticker, 0.0)
            delta = new - old
            if abs(delta) < 1e-9:
                continue
            meta = target[target["ticker"].eq(ticker)]
            if meta.empty:
                asset_class = ""
                sub_asset = ""
                name = ticker
            else:
                asset_class = meta["asset_class"].iloc[0]
                sub_asset = meta["sub_asset"].iloc[0]
                name = meta["name"].iloc[0]
            order_rows.append(
                {
                    "date": dt,
                    "ticker": ticker,
                    "name": name,
                    "asset_class": asset_class,
                    "sub_asset": sub_asset,
                    "previous_weight": old,
                    "target_weight": new,
                    "trade_weight": delta,
                    "action": "BUY" if delta > 0 else "SELL",
                    "overlay_leg": leg,
                }
            )
        previous_weights = current_weights

    portfolio = pd.DataFrame(portfolio_rows)
    orders = pd.DataFrame(order_rows)
    holdings = rebuild_holdings_from_portfolio(portfolio)
    summary = summarize(portfolio, "v2_constrained_70_30")
    yearly = yearly_summary(portfolio)

    portfolio.to_csv(out_dir / "portfolio_returns.csv", index=False, encoding="utf-8-sig")
    orders.to_csv(out_dir / "rebalance_orders.csv", index=False, encoding="utf-8-sig")
    holdings.to_csv(out_dir / "target_holdings.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(out_dir / "yearly.csv", index=False, encoding="utf-8-sig")

    if args.plot:
        plot_equity(portfolio, out_dir / "equity_curve.png")

    print(pd.DataFrame([summary]).to_string(index=False))
    print()
    print(yearly.to_string(index=False))
    print()
    print(orders.tail(40).to_string(index=False))


def add_taxonomy(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["sub_asset"] = out["ranking_group"].map(risk_sub_asset).fillna("other")
    out["asset_class"] = "risk"
    return out


def risk_sub_asset(group: str) -> str:
    group = str(group)
    if group in {"Korea broad equity", "Korea growth"}:
        return "domestic_equity_index"
    if group in {"Korea semiconductor", "Korea IT", "Korea cyclical", "Korea value", "Korea defensive"}:
        return "domestic_equity_sector"
    if group in {"US broad equity", "US growth", "Global/Developed equity", "China/HK growth", "China equity", "India/EM", "Japan equity"}:
        return "overseas_equity_index"
    if group in {"US semiconductor", "US dividend/defensive", "US cyclical/sector", "US REIT"}:
        return "overseas_equity_sector"
    if group in {"Gold", "Commodity/Oil", "Oil", "FX cash", "USD cash"}:
        return "fx_commodity"
    return "other"


def safe_sub_asset(group: str) -> str:
    group = str(group)
    if group == "Cash/short bonds":
        return "cash_short_bond"
    if group == "Korea bonds":
        return "domestic_bond_total"
    if group == "US long bonds":
        return "overseas_bond_total"
    if group in {"US IG bonds", "US high yield"}:
        return "overseas_bond_corp"
    return "cash_short_bond"


def build_safe_universe(universe: pd.DataFrame) -> pd.DataFrame:
    safe_groups = {"Cash/short bonds", "Korea bonds", "US IG bonds", "US long bonds", "US high yield"}
    safe = universe[universe["group"].isin(safe_groups)][["etf_ticker", "name", "group"]].copy()
    safe["ticker"] = safe["etf_ticker"].astype(str)
    safe["sub_asset"] = safe["group"].map(safe_sub_asset)
    safe["asset_class"] = "safe"
    safe["priority"] = safe["sub_asset"].map({name: i for i, name in enumerate(SAFE_PRIORITY)}).fillna(99)
    return safe.sort_values(["priority", "ticker"]).reset_index(drop=True)


def monthly_dates(frame: pd.DataFrame) -> list[pd.Timestamp]:
    idx = pd.DatetimeIndex(pd.to_datetime(frame["date"]).dropna().unique()).sort_values()
    return pd.Series(idx, index=idx).groupby(idx.to_period("M")).max().tolist()


def select_risky(sample: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, str]:
    sample = sample.copy()
    if sample["ranker_group_score_std"].ge(threshold).any():
        sample["active_score"] = sample["ranker_score"]
        return sample.sort_values("active_score", ascending=False), "tree"
    sample["active_score"] = sample["rule_20d_score"]
    return sample.sort_values("active_score", ascending=False), "rule"


def allocate_by_caps(
    candidates: pd.DataFrame,
    score_col: str,
    sleeve_weight: float,
    category_col: str,
    category_caps: dict[str, float],
    etf_cap: float,
) -> pd.DataFrame:
    rows = []
    used_total = 0.0
    used_category = {k: 0.0 for k in category_caps}
    for _, row in candidates.sort_values(score_col, ascending=False).iterrows():
        if used_total >= sleeve_weight - 1e-12:
            break
        category = row[category_col]
        cap = category_caps.get(category, 0.0)
        remaining_category = cap - used_category.get(category, 0.0)
        remaining_total = sleeve_weight - used_total
        weight = min(etf_cap, remaining_category, remaining_total)
        if weight <= 1e-12:
            continue
        rows.append(
            {
                "ticker": row["etf_ticker"],
                "name": row.get("name", row["etf_ticker"]),
                "asset_class": "risk",
                "sub_asset": category,
                "target_weight": weight,
                "score": row[score_col],
            }
        )
        used_total += weight
        used_category[category] = used_category.get(category, 0.0) + weight
    if used_total < sleeve_weight - 1e-12:
        rows.append(
            {
                "ticker": "CASH_KRW",
                "name": "KRW cash filler for unfilled risk sleeve",
                "asset_class": "safe",
                "sub_asset": "cash_short_bond",
                "target_weight": sleeve_weight - used_total,
                "score": np.nan,
            }
        )
    return pd.DataFrame(rows)


def allocate_safe(date: pd.Timestamp, safe_universe: pd.DataFrame, prices: pd.DataFrame, sleeve_weight: float) -> pd.DataFrame:
    available = []
    for _, row in safe_universe.iterrows():
        ticker = row["ticker"]
        if ticker not in prices.columns:
            continue
        s = prices[ticker].dropna()
        if s.empty or s.index.min() > date or date not in prices.index or pd.isna(prices.at[date, ticker]):
            continue
        available.append(row)
    candidates = pd.DataFrame(available)
    rows = []
    used_total = 0.0
    used_category = {k: 0.0 for k in SAFE_CAPS}
    if not candidates.empty:
        for _, row in candidates.sort_values(["priority", "ticker"]).iterrows():
            if used_total >= sleeve_weight - 1e-12:
                break
            category = row["sub_asset"]
            remaining_category = SAFE_CAPS.get(category, 0.0) - used_category.get(category, 0.0)
            remaining_total = sleeve_weight - used_total
            weight = min(ETF_CAP, remaining_category, remaining_total)
            if weight <= 1e-12:
                continue
            rows.append(
                {
                    "ticker": row["ticker"],
                    "name": row["name"],
                    "asset_class": "safe",
                    "sub_asset": category,
                    "target_weight": weight,
                    "score": np.nan,
                }
            )
            used_total += weight
            used_category[category] = used_category.get(category, 0.0) + weight
    if used_total < sleeve_weight - 1e-12:
        rows.append(
            {
                "ticker": "CASH_KRW",
                "name": "KRW cash",
                "asset_class": "safe",
                "sub_asset": "cash_short_bond",
                "target_weight": sleeve_weight - used_total,
                "score": np.nan,
            }
        )
    return pd.DataFrame(rows)


def forward_price_return(prices: pd.DataFrame, ticker: str, date: pd.Timestamp, horizon_days: int) -> float:
    if ticker not in prices.columns or date not in prices.index:
        return 0.0
    s = prices[ticker]
    loc = prices.index.get_loc(date)
    if isinstance(loc, slice):
        loc = loc.start
    end_loc = loc + horizon_days
    if end_loc >= len(prices.index):
        return 0.0
    start = s.iloc[loc]
    end = s.iloc[end_loc]
    if pd.isna(start) or pd.isna(end) or start == 0:
        return 0.0
    return float(end / start - 1.0)


def rebuild_holdings_from_portfolio(portfolio: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in portfolio.iterrows():
        for item in str(row["holdings"]).split(","):
            if not item:
                continue
            ticker, weight = item.split(":")
            rows.append({"date": row["date"], "ticker": ticker, "target_weight": float(weight)})
    return pd.DataFrame(rows)


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


def cagr(final_value: float, periods: int, periods_per_year: int) -> float:
    if periods <= 0 or final_value <= 0:
        return np.nan
    return float(final_value ** (periods_per_year / periods) - 1)


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min()) if not equity.empty else np.nan


def sharpe(returns: pd.Series, periods_per_year: int) -> float:
    std = returns.std()
    if pd.isna(std) or std == 0:
        return np.nan
    return float(returns.mean() / std * np.sqrt(periods_per_year))


def plot_equity(portfolio: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    eq = (1 + portfolio["portfolio_return"].fillna(0)).cumprod()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(portfolio["date"], eq, label="v2 constrained 70/30")
    ax.set_title("V2 Constrained 70/30 Equity")
    ax.set_ylabel("Growth of 1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
