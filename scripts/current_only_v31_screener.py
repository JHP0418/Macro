from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from leadership_v2_constrained_70_30_backtest import ETF_CAP, RISK_CAPS, add_taxonomy, allocate_by_caps


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNIVERSE = ROOT / "data" / "etf_universe_leadership.csv"
DEFAULT_CACHE_DIR = ROOT / "data" / "gaps_long_lived_cache"
DEFAULT_OUTPUT = ROOT / "outputs" / "current_only_v31_screening"
BENCHMARK_SENSITIVE_EXCLUDES = {"105010.KS", "101280.KS"}  # Latin America, Japan TOPIX
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Current-only v3.1 screener for the full GAPS ETF universe.")
    p.add_argument("--universe", default=str(DEFAULT_UNIVERSE))
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--force-download", action="store_true")
    p.add_argument("--chunk-size", type=int, default=80)
    p.add_argument("--min-history-days", type=int, default=120)
    p.add_argument("--benchmark-20d-threshold", type=float, default=0.03)
    p.add_argument("--benchmark-60d-threshold", type=float, default=0.08)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    universe = pd.read_csv(args.universe)
    universe["etf_ticker"] = universe["etf_ticker"].astype(str).str.strip()
    universe["benchmark_ticker"] = universe["benchmark_ticker"].astype(str).str.strip()
    tickers = sorted(set(universe["etf_ticker"]).union(universe["benchmark_ticker"]))
    prices = load_or_download_prices(tickers, args)

    screen = build_current_screen(universe, prices, args.min_history_days)
    if screen.empty:
        raise RuntimeError("No current-only candidates could be scored.")
    screen = add_taxonomy(screen)
    screen = add_current_v31_score(screen, args.benchmark_20d_threshold, args.benchmark_60d_threshold)
    screen = screen.sort_values("current_v31_score", ascending=False).reset_index(drop=True)
    screen["screen_rank"] = np.arange(1, len(screen) + 1)

    filtered = screen[~screen["benchmark_strong_excluded"]].copy()
    if filtered.empty:
        filtered = screen.copy()
    target = allocate_by_caps(
        filtered,
        score_col="current_v31_score",
        sleeve_weight=1.0,
        category_col="sub_asset",
        category_caps={k: v / 0.70 for k, v in RISK_CAPS.items()},
        etf_cap=ETF_CAP,
    )
    target = target[target["ticker"].ne("CASH_KRW")].copy()
    if target["target_weight"].sum() < 1.0 - 1e-9:
        # If the exclusion leaves an unfilled sleeve, fill from the full ranked list.
        target = allocate_by_caps(
            screen,
            score_col="current_v31_score",
            sleeve_weight=1.0,
            category_col="sub_asset",
            category_caps={k: v / 0.70 for k, v in RISK_CAPS.items()},
            etf_cap=ETF_CAP,
        )
        target = target[target["ticker"].ne("CASH_KRW")].copy()

    target = target.merge(
        screen[
            [
                "date",
                "etf_ticker",
                "name",
                "group",
                "ranking_group",
                "sub_asset",
                "screen_rank",
                "benchmark_strong",
                "benchmark_strong_excluded",
                "ETF_return_20D",
                "ETF_return_60D",
                "ETF_return_120D",
                "ETF_RS_20D",
                "ETF_RS_60D",
                "ETF_RS_120D",
                "RS_slope_20D",
                "current_v31_score",
            ]
        ],
        left_on="ticker",
        right_on="etf_ticker",
        how="left",
        suffixes=("", "_screen"),
    )
    category = (
        target.groupby("sub_asset", as_index=False)
        .agg(target_weight=("target_weight", "sum"), count=("ticker", "count"))
        .sort_values("target_weight", ascending=False)
    )
    excluded = screen[screen["benchmark_strong_excluded"]].copy()

    screen.to_csv(out_dir / "current_only_screening_rank.csv", index=False, encoding="utf-8-sig")
    target.to_csv(out_dir / "current_only_target_portfolio.csv", index=False, encoding="utf-8-sig")
    category.to_csv(out_dir / "current_only_category_weights.csv", index=False, encoding="utf-8-sig")
    excluded.to_csv(out_dir / "current_only_benchmark_strong_excluded.csv", index=False, encoding="utf-8-sig")

    print(f"latest_date {screen['date'].max().date()}")
    print(f"universe_count {universe['etf_ticker'].nunique()}")
    print(f"scored_count {len(screen)}")
    print(f"benchmark_strong {bool(screen['benchmark_strong'].iloc[0])}")
    print()
    print("TARGET")
    print(
        target[
            [
                "screen_rank",
                "ticker",
                "name",
                "ranking_group",
                "sub_asset",
                "target_weight",
                "current_v31_score",
                "ETF_return_20D",
                "ETF_return_60D",
                "ETF_RS_20D",
                "ETF_RS_60D",
                "ETF_RS_120D",
            ]
        ].to_string(
            index=False,
            formatters={
                "target_weight": pct,
                "ETF_return_20D": pct,
                "ETF_return_60D": pct,
                "ETF_RS_20D": pct,
                "ETF_RS_60D": pct,
                "ETF_RS_120D": pct,
                "current_v31_score": lambda x: f"{x:.4f}",
            },
        )
    )
    print()
    print("TOP20")
    print(
        screen[
            [
                "screen_rank",
                "etf_ticker",
                "name",
                "ranking_group",
                "sub_asset",
                "current_v31_score",
                "benchmark_strong_excluded",
                "ETF_RS_20D",
                "ETF_RS_60D",
                "ETF_RS_120D",
            ]
        ]
        .head(20)
        .to_string(
            index=False,
            formatters={
                "current_v31_score": lambda x: f"{x:.4f}",
                "ETF_RS_20D": pct,
                "ETF_RS_60D": pct,
                "ETF_RS_120D": pct,
            },
        )
    )


def pct(x: float) -> str:
    return f"{float(x) * 100:.2f}%"


def load_or_download_prices(tickers: list[str], args: argparse.Namespace) -> pd.DataFrame:
    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    cache = Path(args.cache_dir) / f"gaps_etf_benchmark_prices_{args.start}_{end}.csv".replace(":", "-")
    if cache.exists() and not args.force_download:
        return pd.read_csv(cache, parse_dates=["date"]).set_index("date").sort_index()

    import yfinance as yf

    pieces = []
    failed = []
    for i in range(0, len(tickers), args.chunk_size):
        chunk = tickers[i : i + args.chunk_size]
        print(f"[download] {i + 1}-{min(i + args.chunk_size, len(tickers))}/{len(tickers)}", flush=True)
        try:
            raw = yf.download(chunk, start=args.start, end=args.end, auto_adjust=True, progress=False, threads=True, group_by="column")
            close = extract_close(raw, chunk)
            if not close.empty:
                pieces.append(close)
        except Exception:
            failed.extend(chunk)
    prices = pd.concat(pieces, axis=1).loc[:, lambda x: ~x.columns.duplicated()].sort_index() if pieces else pd.DataFrame()
    prices.index.name = "date"
    cache.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(cache, encoding="utf-8-sig")
    pd.DataFrame({"ticker": sorted(set(failed))}).to_csv(cache.with_name(cache.stem + "_failed.csv"), index=False, encoding="utf-8-sig")
    return prices


def extract_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw["Adj Close"]
    else:
        close = raw[["Close" if "Close" in raw else "Adj Close"]].copy()
        close.columns = tickers[:1]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.columns = [str(c).strip() for c in close.columns]
    return close.apply(pd.to_numeric, errors="coerce").dropna(how="all")


def build_current_screen(universe: pd.DataFrame, prices: pd.DataFrame, min_history_days: int) -> pd.DataFrame:
    px = prices.sort_index().apply(pd.to_numeric, errors="coerce").ffill(limit=5)
    latest = px.dropna(how="all").index.max()
    ret20 = px.pct_change(20, fill_method=None)
    ret60 = px.pct_change(60, fill_method=None)
    ret120 = px.pct_change(120, fill_method=None)
    rows = []
    for row in universe.itertuples(index=False):
        etf = str(row.etf_ticker)
        benchmark = str(row.benchmark_ticker)
        if etf not in px.columns or benchmark not in px.columns:
            continue
        s = px[etf].dropna()
        if len(s) < min_history_days or latest not in px.index or pd.isna(px.at[latest, etf]) or pd.isna(px.at[latest, benchmark]):
            continue
        ratio = (px[etf] / px[benchmark]).replace([np.inf, -np.inf], np.nan).dropna()
        slope = rolling_log_slope(ratio.tail(20))
        rows.append(
            {
                "date": latest,
                "etf_ticker": etf,
                "market": getattr(row, "market", ""),
                "benchmark_ticker": benchmark,
                "name": getattr(row, "name", etf),
                "group": getattr(row, "group", ""),
                "ranking_group": getattr(row, "group", ""),
                "ETF_return_20D": ret20.at[latest, etf],
                "ETF_return_60D": ret60.at[latest, etf],
                "ETF_return_120D": ret120.at[latest, etf],
                "benchmark_return_20D": ret20.at[latest, benchmark],
                "benchmark_return_60D": ret60.at[latest, benchmark],
                "benchmark_return_120D": ret120.at[latest, benchmark],
                "ETF_RS_20D": ret20.at[latest, etf] - ret20.at[latest, benchmark],
                "ETF_RS_60D": ret60.at[latest, etf] - ret60.at[latest, benchmark],
                "ETF_RS_120D": ret120.at[latest, etf] - ret120.at[latest, benchmark],
                "RS_slope_20D": slope,
                "price_obs": int(len(s)),
                "first_price_date": s.index.min().date().isoformat(),
            }
        )
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna(subset=["ETF_RS_20D", "ETF_RS_60D", "ETF_RS_120D"])


def rolling_log_slope(series: pd.Series) -> float:
    if len(series) < 20 or series.isna().any():
        return 0.0
    y = np.log(series.astype(float).to_numpy())
    x = np.arange(len(y), dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    denom = float((x * x).sum())
    return float((x * y).sum() / denom) if denom else 0.0


def add_current_v31_score(frame: pd.DataFrame, benchmark_20d_threshold: float, benchmark_60d_threshold: float) -> pd.DataFrame:
    out = frame.copy()
    out["persistent_rs"] = (
        0.15 * z(out, "ETF_RS_20D")
        + 0.40 * z(out, "ETF_RS_60D")
        + 0.35 * z(out, "ETF_RS_120D")
        + 0.10 * z(out, "RS_slope_20D")
    )
    out["is_mean_reversion_group"] = out["ranking_group"].isin(MEAN_REVERSION_GROUPS)
    out["is_cyclical_group"] = out["ranking_group"].isin(CYCLICAL_GROUPS)
    short_rebound = out["ETF_RS_20D"].gt(0) & (out["ETF_RS_60D"].lt(0) | out["ETF_RS_120D"].lt(0))
    weak_persistence = out["ETF_RS_60D"].lt(0) & out["ETF_RS_120D"].lt(0)
    weak_trend = out["ETF_RS_20D"].lt(0)
    out["benchmark_strong"] = out["benchmark_return_20D"].gt(benchmark_20d_threshold) | out["benchmark_return_60D"].gt(benchmark_60d_threshold)
    out["benchmark_strong_excluded"] = out["benchmark_strong"] & out["etf_ticker"].isin(BENCHMARK_SENSITIVE_EXCLUDES)
    out["current_v31_score"] = out["persistent_rs"]
    out["current_v31_score"] += np.where(out["is_mean_reversion_group"] & short_rebound, -0.50, 0.0)
    out["current_v31_score"] += np.where(out["is_cyclical_group"] & weak_persistence, -0.40, 0.0)
    out["current_v31_score"] += np.where(out["is_cyclical_group"] & weak_trend, -0.20, 0.0)
    return out


def z(frame: pd.DataFrame, col: str) -> pd.Series:
    x = pd.to_numeric(frame[col], errors="coerce")
    std = x.std()
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=frame.index)
    return ((x - x.mean()) / std).fillna(0.0)


if __name__ == "__main__":
    main()
