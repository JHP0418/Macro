from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from macro_regime_asset_screener import ASSETS, read_price_cache, safe_to_csv  # noqa: E402
from basket_taxonomy import classify_basket  # noqa: E402
from weekly_screening_rank_backtest import basket_backtest_outputs  # noqa: E402

WEEKLY_TABLES = ROOT / "outputs" / "weekly_screening_rank_backtest_latest" / "tables"


def main() -> None:
    source = WEEKLY_TABLES / "weekly_calibrated_rank_panel.csv"
    panel = pd.read_csv(source, parse_dates=["date"]) if source.exists() else pd.DataFrame()
    if panel.empty or "basket" not in panel.columns or panel["date"].min() > pd.Timestamp("2015-01-01"):
        panel = build_fast_weekly_asset_panel()
    else:
        if "basket" not in panel.columns:
            panel["basket"] = [
                classify_basket(group, name, symbol)
                for group, name, symbol in zip(panel.get("group", ""), panel.get("name", ""), panel.get("symbol", ""))
            ]
        if "basket_rank" not in panel.columns:
            panel["basket_rank"] = panel.groupby(["date", "basket"])["institutional_score_0_100"].rank(ascending=False, method="first")
    basket_panel, basket_summary, basket_current, basket_constituents = basket_backtest_outputs(panel)
    safe_to_csv(panel, WEEKLY_TABLES / "fast_weekly_asset_basket_panel.csv")
    safe_to_csv(basket_panel, WEEKLY_TABLES / "weekly_basket_panel.csv")
    safe_to_csv(basket_summary, WEEKLY_TABLES / "weekly_basket_backtest_summary.csv")
    safe_to_csv(basket_current, WEEKLY_TABLES / "latest_basket_scores.csv")
    safe_to_csv(basket_constituents, WEEKLY_TABLES / "latest_basket_constituent_scores.csv")
    print(basket_summary.to_string(index=False))
    print(basket_current.head(12).to_string(index=False))


def build_fast_weekly_asset_panel() -> pd.DataFrame:
    series = {}
    meta_rows = []
    for asset in ASSETS:
        hist = read_price_cache(asset.symbol)
        if hist.empty or "Close" not in hist:
            continue
        close = pd.to_numeric(hist["Close"], errors="coerce").dropna()
        if close.shape[0] < 260:
            continue
        series[asset.symbol] = close
        meta_rows.append({"symbol": asset.symbol, "name": asset.name, "group": asset.group, "basket": classify_basket(asset.group, asset.name, asset.symbol)})
    prices = pd.DataFrame(series).sort_index().ffill()
    meta = pd.DataFrame(meta_rows).set_index("symbol")
    if prices.empty:
        return pd.DataFrame()
    ret5 = prices.pct_change(5)
    ret20 = prices.pct_change(20)
    ret60 = prices.pct_change(60)
    ret120 = prices.pct_change(120)
    vol20 = prices.pct_change().rolling(20, min_periods=10).std()
    raw_score = 0.24 * cross_z(ret5) + 0.34 * cross_z(ret20) + 0.28 * cross_z(ret60) + 0.14 * cross_z(ret120) - 0.18 * cross_z(vol20)
    score = (50.0 + 12.0 * raw_score).clip(0, 100)
    prob1w = (0.50 + 0.10 * raw_score).clip(0.05, 0.95)
    prob1m = (0.50 + 0.12 * (0.45 * cross_z(ret20) + 0.35 * cross_z(ret60) + 0.20 * cross_z(ret120))).clip(0.05, 0.95)
    weekly = weekly_dates(prices.index)
    rows = []
    for date in weekly:
        if date not in score.index:
            continue
        loc = prices.index.get_loc(date)
        if isinstance(loc, slice) or loc + 20 >= len(prices.index):
            continue
        for symbol in prices.columns:
            if symbol not in meta.index or pd.isna(score.at[date, symbol]):
                continue
            px0 = prices.at[date, symbol]
            if not pd.notna(px0) or px0 <= 0:
                continue
            r1w = prices.iloc[loc + 5][symbol] / px0 - 1.0 if loc + 5 < len(prices.index) else np.nan
            r1m = prices.iloc[loc + 20][symbol] / px0 - 1.0
            if pd.isna(r1w) or pd.isna(r1m):
                continue
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "name": meta.at[symbol, "name"],
                    "group": meta.at[symbol, "group"],
                    "basket": meta.at[symbol, "basket"],
                    "institutional_score_0_100": float(score.at[date, symbol]),
                    "calibrated_prob_1w": float(prob1w.at[date, symbol]),
                    "calibrated_prob_4w": float(prob1m.at[date, symbol]),
                    "realized_return_1w": float(r1w),
                    "realized_return_4w": float(r1m),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["basket_rank"] = out.groupby(["date", "basket"])["institutional_score_0_100"].rank(ascending=False, method="first")
    return out


def cross_z(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=1)
    std = frame.std(axis=1).replace(0, pd.NA)
    return frame.sub(mean, axis=0).div(std, axis=0).replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)


def weekly_dates(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    idx = pd.DatetimeIndex(index).dropna().sort_values()
    return pd.Series(idx, index=idx).groupby(idx.to_period("W-FRI")).max().tolist()


if __name__ == "__main__":
    main()
