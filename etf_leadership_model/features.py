from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS, FORWARD_20D, FORWARD_5D, MIN_HISTORY_DAYS

COMPONENT_FEATURE_COLUMNS = [
    col
    for col in FEATURE_COLUMNS
    if col
    not in {
        "ETF_RS_20D",
        "ETF_RS_60D",
        "ETF_RS_120D",
        "RS_slope_20D",
        "HP_change_20D",
        "Breadth_change_20D",
    }
]


def pct_return(prices: pd.DataFrame | pd.Series, periods: int) -> pd.DataFrame | pd.Series:
    return prices / prices.shift(periods) - 1.0


def rolling_log_slope(series: pd.Series, window: int = 20) -> pd.Series:
    log_series = np.log(series.replace(0, np.nan))
    x = np.arange(window, dtype=float)
    x = x - x.mean()
    denom = float((x * x).sum())

    def slope(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        y = values - values.mean()
        return float((x * y).sum() / denom)

    return log_series.rolling(window).apply(slope, raw=True)


def feature_dates(index: pd.DatetimeIndex, frequency: str = "W-FRI") -> list[pd.Timestamp]:
    idx = pd.DatetimeIndex(index).dropna().sort_values()
    if frequency.lower() in {"d", "daily"}:
        return list(idx)
    return pd.Series(idx, index=idx).groupby(idx.to_period(frequency)).max().tolist()


def latest_holdings_for_date(holdings_by_etf: dict[str, pd.DataFrame], etf: str, date: pd.Timestamp) -> pd.DataFrame:
    holdings = holdings_by_etf.get(etf)
    if holdings is None or holdings.empty:
        return pd.DataFrame(columns=["component_ticker", "weight"])
    eligible_dates = holdings["date"].drop_duplicates()
    eligible_dates = eligible_dates[eligible_dates <= date]
    if eligible_dates.empty:
        return pd.DataFrame(columns=["component_ticker", "weight"])
    latest_date = eligible_dates.max()
    current = holdings[holdings["date"].eq(latest_date)][["component_ticker", "weight"]].copy()
    total = current["weight"].sum()
    if total > 0:
        current["weight"] = current["weight"] / total
    return current


def make_features(
    universe: pd.DataFrame,
    holdings: pd.DataFrame,
    prices: pd.DataFrame,
    frequency: str = "W-FRI",
    require_forward_targets: bool = True,
) -> pd.DataFrame:
    """Build point-in-time ETF leadership features and forward targets."""
    raw_prices = prices.sort_index().apply(pd.to_numeric, errors="coerce")
    # The universe mixes Korean, US, HK, JP, and China assets. A strict rolling
    # window over the merged calendar makes high-proximity and breadth unusable
    # because any local holiday creates NaNs. Limit the fill to short market
    # calendar gaps, then require enough observed history in each rolling window.
    prices = raw_prices.ffill(limit=5)
    dates = feature_dates(prices.index, frequency=frequency)
    holdings_by_etf = {k: v.sort_values("date").reset_index(drop=True) for k, v in holdings.groupby("etf_ticker")}

    ret5 = pct_return(prices, 5)
    ret20 = pct_return(prices, 20)
    ret60 = pct_return(prices, 60)
    ret120 = pct_return(prices, 120)
    ma60 = prices.rolling(60, min_periods=45).mean()
    ma200 = prices.rolling(200, min_periods=150).mean()
    high252 = prices.rolling(252, min_periods=180).max()

    frames: list[pd.DataFrame] = []
    for asset in universe.itertuples(index=False):
        etf = str(asset.etf_ticker)
        benchmark = str(asset.benchmark_ticker)
        if etf not in prices or benchmark not in prices:
            continue
        ratio = prices[etf] / prices[benchmark]
        rs_slope = rolling_log_slope(ratio, 20)
        valid_dates = valid_feature_dates(prices.index, dates, raw_prices[etf], prices[benchmark], require_forward_targets)
        if not valid_dates:
            continue
        frame = pd.DataFrame(
            {
                "date": valid_dates,
                "etf_ticker": etf,
                "market": asset.market,
                "benchmark_ticker": benchmark,
                "ETF_return_5D": ret5.loc[valid_dates, etf].to_numpy(),
                "ETF_return_20D": ret20.loc[valid_dates, etf].to_numpy(),
                "ETF_return_60D": ret60.loc[valid_dates, etf].to_numpy(),
                "ETF_return_120D": ret120.loc[valid_dates, etf].to_numpy(),
                "benchmark_return_5D": ret5.loc[valid_dates, benchmark].to_numpy(),
                "benchmark_return_20D": ret20.loc[valid_dates, benchmark].to_numpy(),
                "benchmark_return_60D": ret60.loc[valid_dates, benchmark].to_numpy(),
                "benchmark_return_120D": ret120.loc[valid_dates, benchmark].to_numpy(),
                "ETF_RS_20D": (ret20.loc[valid_dates, etf] - ret20.loc[valid_dates, benchmark]).to_numpy(),
                "ETF_RS_60D": (ret60.loc[valid_dates, etf] - ret60.loc[valid_dates, benchmark]).to_numpy(),
                "ETF_RS_120D": (ret120.loc[valid_dates, etf] - ret120.loc[valid_dates, benchmark]).to_numpy(),
                "RS_slope_20D": rs_slope.loc[valid_dates].to_numpy(),
                "forward_5D_return": prices[etf].shift(-FORWARD_5D).loc[valid_dates].to_numpy() / prices[etf].loc[valid_dates].to_numpy() - 1.0,
                "forward_20D_return": prices[etf].shift(-FORWARD_20D).loc[valid_dates].to_numpy() / prices[etf].loc[valid_dates].to_numpy() - 1.0,
                "benchmark_forward_5D_return": prices[benchmark].shift(-FORWARD_5D).loc[valid_dates].to_numpy() / prices[benchmark].loc[valid_dates].to_numpy() - 1.0,
                "benchmark_forward_20D_return": prices[benchmark].shift(-FORWARD_20D).loc[valid_dates].to_numpy() / prices[benchmark].loc[valid_dates].to_numpy() - 1.0,
            }
        ).set_index("date")
        component_frame = component_features_for_etf(
            etf=etf,
            dates=valid_dates,
            holdings_by_etf=holdings_by_etf,
            prices=prices,
            ret20=ret20,
            ret60=ret60,
            ma60=ma60,
            ma200=ma200,
            high252=high252,
            benchmark=benchmark,
        )
        frame = frame.join(component_frame, how="left").reset_index()
        frames.append(frame)

    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if frame.empty:
        return frame
    frame["forward_5D_excess"] = frame["forward_5D_return"] - frame["benchmark_forward_5D_return"]
    frame["forward_20D_excess"] = frame["forward_20D_return"] - frame["benchmark_forward_20D_return"]
    frame = add_holding_logic(frame)
    if require_forward_targets:
        frame = add_ranking_labels(frame)
    return frame.sort_values(["date", "etf_ticker"]).reset_index(drop=True)


def valid_feature_dates(
    index: pd.DatetimeIndex,
    dates: list[pd.Timestamp],
    etf_price: pd.Series,
    benchmark_price: pd.Series,
    require_forward_targets: bool = True,
) -> list[pd.Timestamp]:
    out = []
    n = len(index)
    for date in dates:
        if date not in index:
            continue
        loc = index.get_indexer([date])[0]
        if loc < MIN_HISTORY_DAYS:
            continue
        if require_forward_targets and loc + FORWARD_20D >= n:
            continue
        if pd.isna(etf_price.loc[date]) or pd.isna(benchmark_price.loc[date]):
            continue
        out.append(date)
    return out


def component_features_for_etf(
    etf: str,
    dates: list[pd.Timestamp],
    holdings_by_etf: dict[str, pd.DataFrame],
    prices: pd.DataFrame,
    ret20: pd.DataFrame,
    ret60: pd.DataFrame,
    ma60: pd.DataFrame,
    ma200: pd.DataFrame,
    high252: pd.DataFrame,
    benchmark: str,
) -> pd.DataFrame:
    idx = pd.DatetimeIndex(dates)
    holdings = holdings_by_etf.get(etf)
    if holdings is None or holdings.empty:
        return pd.DataFrame(empty_component_features(), index=idx)
    parts = []
    snapshot_dates = sorted(pd.to_datetime(holdings["date"].dropna().unique()))
    for pos, snapshot_date in enumerate(snapshot_dates):
        next_date = snapshot_dates[pos + 1] if pos + 1 < len(snapshot_dates) else pd.Timestamp.max
        segment_dates = idx[(idx >= snapshot_date) & (idx < next_date)]
        if len(segment_dates) == 0:
            continue
        current = holdings[holdings["date"].eq(snapshot_date)][["component_ticker", "weight"]].copy()
        current = current[current["component_ticker"].isin(prices.columns)]
        if current.empty:
            part = pd.DataFrame(empty_component_features(), index=segment_dates)
            parts.append(part)
            continue
        current = current.groupby("component_ticker", as_index=False)["weight"].sum()
        current["weight"] = current["weight"] / current["weight"].sum()
        comps = current["component_ticker"].tolist()
        weights = current.set_index("component_ticker")["weight"].astype(float)
        part = vector_component_features(segment_dates, comps, weights, prices, ret20, ret60, ma60, ma200, high252, benchmark)
        parts.append(part)
    if not parts:
        return pd.DataFrame(empty_component_features(), index=idx)
    return pd.concat(parts).reindex(idx)


def vector_component_features(
    dates: pd.DatetimeIndex,
    comps: list[str],
    weights: pd.Series,
    prices: pd.DataFrame,
    ret20: pd.DataFrame,
    ret60: pd.DataFrame,
    ma60: pd.DataFrame,
    ma200: pd.DataFrame,
    high252: pd.DataFrame,
    benchmark: str,
) -> pd.DataFrame:
    component_price = prices.loc[dates, comps].astype(float)
    hp = component_price / high252.loc[dates, comps].astype(float)
    c_ret20 = ret20.loc[dates, comps].astype(float)
    c_ret60 = ret60.loc[dates, comps].astype(float)
    c_rs20 = c_ret20.sub(ret20.loc[dates, benchmark].astype(float), axis=0)
    c_rs60 = c_ret60.sub(ret60.loc[dates, benchmark].astype(float), axis=0)
    w = weights.reindex(comps).astype(float)

    contrib = c_ret20.mul(w, axis=1)
    positive_contrib = contrib.clip(lower=0).sum(axis=1)
    top5_contrib = contrib.clip(lower=0).apply(lambda row: row.nlargest(5).sum(), axis=1)
    n_tail = max(1, int(np.ceil(len(comps) * 0.20)))

    out = pd.DataFrame(index=dates)
    out["weighted_HP"] = weighted_frame_mean(hp, w)
    out["median_HP"] = hp.median(axis=1)
    out["HP90_share"] = weighted_frame_share(hp >= 0.9, hp.notna(), w)
    out["weighted_component_RS_20D"] = weighted_frame_mean(c_rs20, w)
    out["weighted_component_RS_60D"] = weighted_frame_mean(c_rs60, w)
    out["median_component_RS_20D"] = c_rs20.median(axis=1)
    out["RS_positive_share"] = weighted_frame_share(c_rs20 > 0, c_rs20.notna(), w)
    ma60_values = ma60.loc[dates, comps]
    ma200_values = ma200.loc[dates, comps]
    out["MA60_breadth"] = weighted_frame_share(component_price > ma60_values, component_price.notna() & ma60_values.notna(), w)
    out["MA200_breadth"] = weighted_frame_share(component_price > ma200_values, component_price.notna() & ma200_values.notna(), w)
    out["median_component_return_20D"] = c_ret20.median(axis=1)
    out["median_component_return_60D"] = c_ret60.median(axis=1)
    out["mean_minus_median_return_20D"] = c_ret20.mean(axis=1) - c_ret20.median(axis=1)
    out["top20_component_return_mean"] = c_ret20.apply(lambda row: row.nlargest(n_tail).mean(), axis=1)
    out["bottom20_component_return_mean"] = c_ret20.apply(lambda row: row.nsmallest(n_tail).mean(), axis=1)
    out["holding_count"] = len(comps)
    out["effective_N"] = float(1.0 / np.square(w).sum()) if np.square(w).sum() > 0 else np.nan
    out["top5_weight_share"] = float(w.sort_values(ascending=False).head(5).sum())
    out["top10_weight_share"] = float(w.sort_values(ascending=False).head(10).sum())
    out["top5_return_contribution_share"] = (top5_contrib / positive_contrib.replace(0, np.nan)).fillna(0.0)
    return out


def weighted_frame_mean(values: pd.DataFrame, weights: pd.Series) -> pd.Series:
    aligned_weights = weights.reindex(values.columns).astype(float)
    valid = values.notna()
    denom = valid.astype(float).mul(aligned_weights, axis=1).sum(axis=1).replace(0, np.nan)
    numer = values.mul(aligned_weights, axis=1).sum(axis=1, min_count=1)
    return numer / denom


def weighted_frame_share(mask: pd.DataFrame, valid: pd.DataFrame, weights: pd.Series) -> pd.Series:
    aligned_weights = weights.reindex(mask.columns).astype(float)
    denom = valid.astype(float).mul(aligned_weights, axis=1).sum(axis=1).replace(0, np.nan)
    numer = mask.where(valid, False).astype(float).mul(aligned_weights, axis=1).sum(axis=1)
    return numer / denom


def component_feature_row(
    current_holdings: pd.DataFrame,
    prices: pd.DataFrame,
    ret20: pd.DataFrame,
    ret60: pd.DataFrame,
    ma60: pd.DataFrame,
    ma200: pd.DataFrame,
    high252: pd.DataFrame,
    benchmark: str,
    date: pd.Timestamp,
) -> dict[str, float]:
    if current_holdings.empty:
        return empty_component_features()
    comps = [c for c in current_holdings["component_ticker"].astype(str) if c in prices.columns]
    if not comps:
        return empty_component_features(holding_count=int(current_holdings.shape[0]))

    h = current_holdings[current_holdings["component_ticker"].isin(comps)].copy()
    h = h.groupby("component_ticker", as_index=False)["weight"].sum()
    h["weight"] = h["weight"] / h["weight"].sum()
    comps = h["component_ticker"].tolist()
    weights = h.set_index("component_ticker")["weight"]

    component_price = prices.loc[date, comps].astype(float)
    hp = component_price / high252.loc[date, comps].astype(float)
    c_ret20 = ret20.loc[date, comps].astype(float)
    c_ret60 = ret60.loc[date, comps].astype(float)
    b_ret20 = float(ret20.at[date, benchmark])
    b_ret60 = float(ret60.at[date, benchmark])
    c_rs20 = c_ret20 - b_ret20
    c_rs60 = c_ret60 - b_ret60

    ma60_breadth = weighted_indicator(component_price > ma60.loc[date, comps], weights)
    ma200_breadth = weighted_indicator(component_price > ma200.loc[date, comps], weights)
    contrib = weights * c_ret20.reindex(weights.index)
    positive_contrib = contrib[contrib > 0].sum()
    top5_contrib = contrib.sort_values(ascending=False).head(5).clip(lower=0).sum()

    holding_count = int(h.shape[0])
    effective_n = float(1.0 / np.square(weights).sum()) if np.square(weights).sum() > 0 else np.nan
    return {
        "weighted_HP": weighted_mean(hp, weights),
        "median_HP": safe_median(hp),
        "HP90_share": weighted_indicator(hp >= 0.9, weights),
        "weighted_component_RS_20D": weighted_mean(c_rs20, weights),
        "weighted_component_RS_60D": weighted_mean(c_rs60, weights),
        "median_component_RS_20D": safe_median(c_rs20),
        "RS_positive_share": weighted_indicator(c_rs20 > 0, weights),
        "MA60_breadth": ma60_breadth,
        "MA200_breadth": ma200_breadth,
        "median_component_return_20D": safe_median(c_ret20),
        "median_component_return_60D": safe_median(c_ret60),
        "mean_minus_median_return_20D": float(c_ret20.mean() - c_ret20.median()),
        "top20_component_return_mean": quantile_tail_mean(c_ret20, top=True),
        "bottom20_component_return_mean": quantile_tail_mean(c_ret20, top=False),
        "holding_count": holding_count,
        "effective_N": effective_n,
        "top5_weight_share": float(weights.sort_values(ascending=False).head(5).sum()),
        "top10_weight_share": float(weights.sort_values(ascending=False).head(10).sum()),
        "top5_return_contribution_share": float(top5_contrib / positive_contrib) if positive_contrib > 0 else 0.0,
    }


def empty_component_features(holding_count: int = 0) -> dict[str, float]:
    out = {col: np.nan for col in COMPONENT_FEATURE_COLUMNS if col not in {"holding_count", "effective_N"}}
    out["holding_count"] = holding_count
    out["effective_N"] = np.nan
    return out


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    aligned = pd.concat([values.rename("v"), weights.rename("w")], axis=1).dropna()
    if aligned.empty or aligned["w"].sum() <= 0:
        return np.nan
    return float((aligned["v"] * aligned["w"]).sum() / aligned["w"].sum())


def weighted_indicator(mask: pd.Series, weights: pd.Series) -> float:
    aligned = pd.concat([mask.astype(float).rename("v"), weights.rename("w")], axis=1).dropna()
    if aligned.empty or aligned["w"].sum() <= 0:
        return np.nan
    return float((aligned["v"] * aligned["w"]).sum() / aligned["w"].sum())


def safe_median(values: pd.Series) -> float:
    return float(pd.to_numeric(values, errors="coerce").median())


def quantile_tail_mean(values: pd.Series, top: bool) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    n = max(1, int(np.ceil(len(clean) * 0.20)))
    return float(clean.nlargest(n).mean() if top else clean.nsmallest(n).mean())


def forward_return(series: pd.Series, date: pd.Timestamp, horizon: int) -> float:
    loc = series.index.get_loc(date)
    if loc + horizon >= len(series.index):
        return np.nan
    start = series.iloc[loc]
    end = series.iloc[loc + horizon]
    return float(end / start - 1.0) if pd.notna(start) and pd.notna(end) and start else np.nan


def add_holding_logic(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["HP_change_20D"] = out.groupby("etf_ticker")["weighted_HP"].diff(20)
    out["Breadth_change_20D"] = out.groupby("etf_ticker")["MA60_breadth"].diff(20)
    effective = pd.to_numeric(out["effective_N"], errors="coerce")
    count = pd.to_numeric(out["holding_count"], errors="coerce").fillna(0)
    out["holding_logic"] = np.select(
        [count <= 5, effective < 8, effective >= 20],
        ["concentrated", "concentrated", "diversified"],
        default="mid",
    )
    return out


def add_ranking_labels(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for horizon, target in [("5D", "forward_5D_excess"), ("20D", "forward_20D_excess")]:
        pct_col = f"label_{horizon}_rank_pct"
        int_col = f"label_{horizon}_rank_int"
        out[pct_col] = out.groupby("date")[target].rank(pct=True, method="average")
        out[int_col] = np.floor(out[pct_col].fillna(0) * 5).clip(0, 4).astype(int)
    return out
