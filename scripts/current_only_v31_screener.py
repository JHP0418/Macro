from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from leadership_v2_constrained_70_30_backtest import ETF_CAP, RISK_CAPS, add_taxonomy, allocate_by_caps


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNIVERSE = ROOT / "data" / "etf_universe_leadership.csv"
DEFAULT_HOLDINGS = ROOT / "data" / "etf_holdings_static_2019_repaired.csv"
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
    p.add_argument("--holdings", default=str(DEFAULT_HOLDINGS))
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--force-download", action="store_true")
    p.add_argument("--chunk-size", type=int, default=80)
    p.add_argument("--min-history-days", type=int, default=120)
    p.add_argument("--benchmark-20d-threshold", type=float, default=0.03)
    p.add_argument("--benchmark-60d-threshold", type=float, default=0.08)
    p.add_argument("--skip-component-features", action="store_true")
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
    if not args.skip_component_features:
        holdings = pd.read_csv(args.holdings)
        component_prices = load_component_prices(universe, holdings, prices, args)
        component = build_latest_component_features(screen, holdings, component_prices)
        screen = screen.merge(component, on=["date", "etf_ticker", "benchmark_ticker"], how="left")
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
                "weighted_HP",
                "HP90_share",
                "weighted_component_RS_20D",
                "RS_positive_share",
                "MA60_breadth",
                "MA200_breadth",
                "Breadth_change_20D",
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
    print(f"component_feature_count {int(screen['MA60_breadth'].notna().sum()) if 'MA60_breadth' in screen else 0}")
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
                "MA60_breadth",
                "HP90_share",
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
                "MA60_breadth": pct,
                "HP90_share": pct,
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
                "MA60_breadth",
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
                "MA60_breadth": pct,
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


def load_component_prices(universe: pd.DataFrame, holdings: pd.DataFrame, etf_prices: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    holdings = holdings.copy()
    holdings["etf_ticker"] = holdings["etf_ticker"].astype(str).str.strip()
    holdings["component_ticker"] = holdings["component_ticker"].astype(str).str.strip()
    keep = set(universe["etf_ticker"].astype(str))
    holdings = holdings[holdings["etf_ticker"].isin(keep)].copy()
    required = sorted(
        set(universe["etf_ticker"].astype(str))
        .union(universe["benchmark_ticker"].astype(str))
        .union(holdings["component_ticker"].astype(str))
    )
    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    cache = Path(args.cache_dir) / f"current_only_component_prices_{args.start}_{end}.csv".replace(":", "-")
    if cache.exists() and not args.force_download:
        return pd.read_csv(cache, parse_dates=["date"]).set_index("date").sort_index()

    # Reuse ETF/benchmark prices already downloaded, then fetch only missing components.
    pieces = [etf_prices]
    existing = set(etf_prices.columns)
    missing = [ticker for ticker in required if ticker not in existing]
    if missing:
        component_args = argparse.Namespace(**vars(args))
        component_args.chunk_size = min(args.chunk_size, 80)
        downloaded = download_price_matrix(missing, component_args)
        if not downloaded.empty:
            pieces.append(downloaded)
    prices = pd.concat(pieces, axis=1).loc[:, lambda x: ~x.columns.duplicated()].sort_index()
    prices.index.name = "date"
    cache.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(cache, encoding="utf-8-sig")
    return prices


def download_price_matrix(tickers: list[str], args: argparse.Namespace) -> pd.DataFrame:
    import yfinance as yf

    pieces = []
    failed = []
    for i in range(0, len(tickers), args.chunk_size):
        chunk = tickers[i : i + args.chunk_size]
        print(f"[component download] {i + 1}-{min(i + args.chunk_size, len(tickers))}/{len(tickers)}", flush=True)
        try:
            raw = yf.download(chunk, start=args.start, end=args.end, auto_adjust=True, progress=False, threads=True, group_by="column")
            close = extract_close(raw, chunk)
            if not close.empty:
                pieces.append(close)
        except Exception:
            failed.extend(chunk)
    prices = pd.concat(pieces, axis=1).loc[:, lambda x: ~x.columns.duplicated()].sort_index() if pieces else pd.DataFrame()
    if failed:
        print(f"[component download] failed chunks/tickers: {len(set(failed))}", flush=True)
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


def build_latest_component_features(screen: pd.DataFrame, holdings: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    px = prices.sort_index().apply(pd.to_numeric, errors="coerce").ffill(limit=5)
    latest = pd.Timestamp(screen["date"].max())
    ret20 = px.pct_change(20, fill_method=None)
    ret60 = px.pct_change(60, fill_method=None)
    ma60 = px.rolling(60, min_periods=45).mean()
    ma200 = px.rolling(200, min_periods=150).mean()
    high252 = px.rolling(252, min_periods=180).max()

    holdings = holdings.copy()
    holdings["etf_ticker"] = holdings["etf_ticker"].astype(str).str.strip()
    holdings["component_ticker"] = holdings["component_ticker"].astype(str).str.strip()
    holdings["weight"] = pd.to_numeric(holdings["weight"], errors="coerce")
    rows = []
    for row in screen.itertuples(index=False):
        etf = str(row.etf_ticker)
        benchmark = str(row.benchmark_ticker)
        h = holdings[holdings["etf_ticker"].eq(etf)][["component_ticker", "weight"]].dropna().copy()
        h = h[(h["weight"] > 0) & h["component_ticker"].isin(px.columns)]
        if h.empty or latest not in px.index or benchmark not in px.columns:
            rows.append(empty_component_row(latest, etf, benchmark))
            continue
        h = h.groupby("component_ticker", as_index=False)["weight"].sum()
        h["weight"] = h["weight"] / h["weight"].sum()
        comps = h["component_ticker"].tolist()
        w = h.set_index("component_ticker")["weight"].astype(float)
        component_price = px.loc[latest, comps].astype(float)
        valid = component_price.notna()
        comps = [c for c in comps if bool(valid.get(c, False))]
        if not comps:
            rows.append(empty_component_row(latest, etf, benchmark))
            continue
        w = w.reindex(comps)
        w = w / w.sum()
        hp = (px.loc[latest, comps] / high252.loc[latest, comps]).astype(float)
        c_ret20 = ret20.loc[latest, comps].astype(float)
        c_ret60 = ret60.loc[latest, comps].astype(float)
        b_ret20 = float(ret20.at[latest, benchmark]) if pd.notna(ret20.at[latest, benchmark]) else np.nan
        b_ret60 = float(ret60.at[latest, benchmark]) if pd.notna(ret60.at[latest, benchmark]) else np.nan
        c_rs20 = c_ret20 - b_ret20
        c_rs60 = c_ret60 - b_ret60
        ma60_values = ma60.loc[latest, comps]
        ma200_values = ma200.loc[latest, comps]
        prior_idx = px.index.get_indexer([latest])[0] - 20
        if prior_idx >= 0:
            prior_date = px.index[prior_idx]
            prior_price = px.loc[prior_date, comps].astype(float)
            prior_hp = prior_price / high252.loc[prior_date, comps].astype(float)
            prior_ma60 = ma60.loc[prior_date, comps]
            prior_breadth = weighted_share(prior_price > prior_ma60, prior_price.notna() & prior_ma60.notna(), w)
            prior_hp90 = weighted_share(prior_hp >= 0.9, prior_hp.notna(), w)
        else:
            prior_breadth = np.nan
            prior_hp90 = np.nan
        ma60_breadth = weighted_share(px.loc[latest, comps] > ma60_values, px.loc[latest, comps].notna() & ma60_values.notna(), w)
        hp90 = weighted_share(hp >= 0.9, hp.notna(), w)
        rows.append(
            {
                "date": latest,
                "etf_ticker": etf,
                "benchmark_ticker": benchmark,
                "weighted_HP": weighted_mean(hp, w),
                "median_HP": float(hp.median(skipna=True)),
                "HP90_share": hp90,
                "HP_change_20D": hp90 - prior_hp90 if pd.notna(prior_hp90) else np.nan,
                "weighted_component_RS_20D": weighted_mean(c_rs20, w),
                "weighted_component_RS_60D": weighted_mean(c_rs60, w),
                "median_component_RS_20D": float(c_rs20.median(skipna=True)),
                "RS_positive_share": weighted_share(c_rs20 > 0, c_rs20.notna(), w),
                "MA60_breadth": ma60_breadth,
                "MA200_breadth": weighted_share(px.loc[latest, comps] > ma200_values, px.loc[latest, comps].notna() & ma200_values.notna(), w),
                "Breadth_change_20D": ma60_breadth - prior_breadth if pd.notna(prior_breadth) else np.nan,
                "median_component_return_20D": float(c_ret20.median(skipna=True)),
                "median_component_return_60D": float(c_ret60.median(skipna=True)),
                "holding_count": len(comps),
                "effective_N": float(1.0 / np.square(w).sum()) if np.square(w).sum() > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def empty_component_row(date: pd.Timestamp, etf: str, benchmark: str) -> dict[str, object]:
    return {
        "date": date,
        "etf_ticker": etf,
        "benchmark_ticker": benchmark,
        "weighted_HP": np.nan,
        "median_HP": np.nan,
        "HP90_share": np.nan,
        "HP_change_20D": np.nan,
        "weighted_component_RS_20D": np.nan,
        "weighted_component_RS_60D": np.nan,
        "median_component_RS_20D": np.nan,
        "RS_positive_share": np.nan,
        "MA60_breadth": np.nan,
        "MA200_breadth": np.nan,
        "Breadth_change_20D": np.nan,
        "median_component_return_20D": np.nan,
        "median_component_return_60D": np.nan,
        "holding_count": 0,
        "effective_N": np.nan,
    }


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    data = pd.concat([values.rename("value"), weights.rename("weight")], axis=1).dropna()
    if data.empty or data["weight"].sum() <= 0:
        return np.nan
    return float((data["value"] * data["weight"]).sum() / data["weight"].sum())


def weighted_share(condition: pd.Series, valid: pd.Series, weights: pd.Series) -> float:
    data = pd.concat([condition.rename("condition"), valid.rename("valid"), weights.rename("weight")], axis=1).dropna()
    data = data[data["valid"].astype(bool)]
    if data.empty or data["weight"].sum() <= 0:
        return np.nan
    return float(data.loc[data["condition"].astype(bool), "weight"].sum() / data["weight"].sum())


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
    component_score = pd.Series(0.0, index=out.index)
    for weight, col in [
        (0.20, "weighted_component_RS_20D"),
        (0.15, "weighted_component_RS_60D"),
        (0.15, "MA60_breadth"),
        (0.10, "HP90_share"),
        (0.05, "Breadth_change_20D"),
    ]:
        if col in out.columns:
            component_score += weight * z(out, col)
    out["current_v31_score"] = out["persistent_rs"] + component_score
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
