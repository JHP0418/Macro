from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from lightgbm import LGBMRanker, early_stopping, log_evaluation
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "long_history_proxy_selection_backtest_latest"
TABLES = OUT / "tables"
CACHE = ROOT / "data" / "proxy_long_history_cache"


LEADERSHIP_UNIVERSE = {
    "SPY": "US broad",
    "QQQ": "US growth",
    "DIA": "US broad",
    "IWM": "US small",
    "EFA": "Developed ex-US",
    "EEM": "EM",
    "EWY": "Korea",
    "EWJ": "Japan",
    "EWT": "Taiwan",
    "FXI": "China",
    "VGK": "Europe",
    "XLK": "US sector",
    "XLY": "US sector",
    "XLP": "US sector",
    "XLI": "US sector",
    "XLF": "US sector",
    "XLE": "US sector",
    "XLV": "US sector",
    "XLU": "US sector",
    "XLB": "US sector",
    "SMH": "Semiconductor",
    "SOXX": "Semiconductor",
    "IBB": "Biotech",
    "IYR": "REIT",
    "VNQ": "REIT",
    "GLD": "Gold",
    "DBC": "Commodity",
    "USO": "Oil",
}

SAFE_UNIVERSE = {
    "BIL": "Cash",
    "SHY": "Short Treasury",
    "IEF": "Intermediate Treasury",
    "TLT": "Long Treasury",
    "AGG": "Aggregate Bond",
    "BND": "Aggregate Bond",
    "LQD": "IG Credit",
    "TIP": "TIPS",
    "MBB": "MBS",
    "GLD": "Gold",
    "UUP": "Dollar",
    "FXY": "Yen",
    "FXF": "Swiss Franc",
}


LEADERSHIP_FEATURES = [
    "ret_20",
    "ret_60",
    "ret_120",
    "ret_252",
    "excess_20",
    "excess_60",
    "excess_120",
    "rs_slope_20",
    "rs_slope_60",
    "vol_20",
    "vol_60",
    "drawdown_60",
    "drawdown_252",
    "high_proximity_252",
    "above_ma60",
    "above_ma200",
    "rule_leadership_score",
]

SAFE_FEATURES = [
    "ret_5",
    "ret_20",
    "ret_60",
    "ret_120",
    "vol_20",
    "vol_60",
    "drawdown_60",
    "high_proximity_252",
    "above_ma60",
    "above_ma200",
    "macro_US2Y_level",
    "macro_US10Y_level",
    "macro_US10Y_5d_chg",
    "macro_US10Y_20d_chg",
    "macro_US2Y_20d_chg",
    "macro_DXY_20d_ret",
    "macro_USDKRW_20d_ret",
    "macro_VIX_level",
    "macro_VIX_20d_chg",
    "macro_HY_OAS_20d_chg",
    "macro_GOLD_20d_ret",
    "macro_NASDAQ100_20d_ret",
    "macro_HYG_IEF_20d_ret",
    "axis1_vol_credit_stress_1m",
    "axis2_fx_liquidity_stress_1m",
    "axis3_peak_fragility_stress_1m",
    "risk_off_v4_prob_1m",
    "stage_level_1m",
]


@dataclass
class BacktestSpec:
    name: str
    score_col: str
    universe: str
    top_k: int
    frequency: str
    benchmark: str | None = None
    risk_only: bool = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Long-history proxy ETF leadership and safe-asset selection backtest.")
    p.add_argument("--start", default="2009-01-01")
    p.add_argument("--backtest-start", default="2010-01-04")
    p.add_argument("--end", default="2026-05-11")
    p.add_argument("--initial-test-year", type=int, default=2014)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--safe-top-k", type=int, default=3)
    p.add_argument("--force-download", action="store_true")
    p.add_argument("--min-non-null-ratio", type=float, default=0.80)
    return p.parse_args()


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)


def flatten_yfinance_close(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"].copy()
        elif "Adj Close" in raw.columns.get_level_values(0):
            close = raw["Adj Close"].copy()
        else:
            close = raw.xs(raw.columns.get_level_values(0)[0], axis=1, level=0)
    else:
        close = raw[["Close"]].copy() if "Close" in raw.columns else raw.copy()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close.sort_index()
    close.columns = [str(c).upper() for c in close.columns]
    return close


def download_prices(tickers: list[str], start: str, end: str, force: bool = False) -> pd.DataFrame:
    cache_path = CACHE / f"proxy_prices_{start}_{end}.csv".replace(":", "-")
    if cache_path.exists() and not force:
        return pd.read_csv(cache_path, parse_dates=["Date"]).set_index("Date").sort_index()
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="column",
    )
    close = flatten_yfinance_close(raw)
    close = close.replace([np.inf, -np.inf], np.nan)
    close.to_csv(cache_path, index_label="Date", encoding="utf-8-sig")
    return close


def clean_price_universe(prices: pd.DataFrame, tickers: list[str], backtest_start: str, min_ratio: float) -> pd.DataFrame:
    available = [t for t in tickers if t in prices.columns]
    px = prices[available].copy().sort_index()
    bt = px.loc[pd.Timestamp(backtest_start) :]
    keep = []
    for col in px.columns:
        ratio = bt[col].notna().mean()
        if ratio >= min_ratio:
            keep.append(col)
    px = px[keep].ffill(limit=5)
    return px


def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    x = x - x.mean()
    denom = float(np.dot(x, x))

    def _slope(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        y = values - values.mean()
        return float(np.dot(x, y) / denom)

    return series.rolling(window, min_periods=window).apply(_slope, raw=True)


def max_drawdown_window(price: pd.Series, window: int) -> pd.Series:
    peak = price.rolling(window, min_periods=max(5, window // 4)).max()
    return price / peak - 1.0


def add_basic_asset_features(prices: pd.DataFrame, universe: dict[str, str], benchmark: str | None) -> pd.DataFrame:
    rows = []
    bench = prices[benchmark].copy() if benchmark and benchmark in prices.columns else None
    bench_rets = {w: bench.pct_change(w) if bench is not None else pd.Series(index=prices.index, dtype=float) for w in [5, 20, 60, 120, 252]}
    for ticker in [t for t in universe if t in prices.columns]:
        p = prices[ticker].astype(float)
        f = pd.DataFrame({"date": prices.index, "ticker": ticker, "category": universe[ticker]})
        for w in [5, 20, 60, 120, 252]:
            f[f"ret_{w}"] = p.pct_change(w).to_numpy()
        f["vol_20"] = p.pct_change().rolling(20).std().mul(np.sqrt(252)).to_numpy()
        f["vol_60"] = p.pct_change().rolling(60).std().mul(np.sqrt(252)).to_numpy()
        f["drawdown_60"] = max_drawdown_window(p, 60).to_numpy()
        f["drawdown_252"] = max_drawdown_window(p, 252).to_numpy()
        f["high_proximity_252"] = (p / p.rolling(252, min_periods=60).max()).to_numpy()
        f["above_ma60"] = (p > p.rolling(60, min_periods=30).mean()).astype(float).to_numpy()
        f["above_ma200"] = (p > p.rolling(200, min_periods=100).mean()).astype(float).to_numpy()
        if bench is not None:
            ratio = np.log((p / bench).replace([np.inf, -np.inf], np.nan))
            for w in [20, 60, 120]:
                f[f"excess_{w}"] = (p.pct_change(w) - bench_rets[w]).to_numpy()
            f["rs_slope_20"] = rolling_slope(ratio, 20).to_numpy()
            f["rs_slope_60"] = rolling_slope(ratio, 60).to_numpy()
        rows.append(f)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def add_forward_targets(features: pd.DataFrame, prices: pd.DataFrame, benchmark: str | None, horizons: tuple[int, ...] = (5, 20)) -> pd.DataFrame:
    out = features.copy()
    bench_fwd: dict[int, pd.Series] = {}
    if benchmark and benchmark in prices.columns:
        b = prices[benchmark].astype(float)
        for h in horizons:
            bench_fwd[h] = b.shift(-h) / b - 1.0
    for h in horizons:
        values = []
        for ticker, part in out.groupby("ticker", sort=False):
            p = prices[ticker].astype(float)
            fwd = p.shift(-h) / p - 1.0
            if h in bench_fwd:
                fwd_excess = fwd - bench_fwd[h]
            else:
                fwd_excess = fwd
            values.append(pd.DataFrame({"idx": part.index, f"fwd_{h}d_return": fwd.loc[part["date"]].to_numpy(), f"fwd_{h}d_excess": fwd_excess.loc[part["date"]].to_numpy()}))
        joined = pd.concat(values).set_index("idx").sort_index()
        out[f"fwd_{h}d_return"] = joined[f"fwd_{h}d_return"]
        out[f"fwd_{h}d_excess"] = joined[f"fwd_{h}d_excess"]
        pct = out.groupby("date")[f"fwd_{h}d_excess"].rank(pct=True, method="average")
        out[f"label_{h}d_rank_int"] = np.ceil(pct * 5).sub(1).clip(0, 4).where(pct.notna())
    return out


def zscore_by_date(frame: pd.DataFrame, cols: list[str], prefix: str = "z_") -> pd.DataFrame:
    out = frame.copy()
    for col in cols:
        x = pd.to_numeric(out[col], errors="coerce")
        lo = x.groupby(out["date"]).transform(lambda s: s.quantile(0.01))
        hi = x.groupby(out["date"]).transform(lambda s: s.quantile(0.99))
        x = x.clip(lo, hi)
        mean = x.groupby(out["date"]).transform("mean")
        std = x.groupby(out["date"]).transform("std").replace(0, np.nan)
        out[f"{prefix}{col}"] = ((x - mean) / std).fillna(0.0)
    return out


def build_leadership_features(prices: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    frame = add_basic_asset_features(prices, LEADERSHIP_UNIVERSE, "QQQ")
    frame = frame[frame["date"].ge(pd.Timestamp(args.backtest_start))].copy()
    frame = add_forward_targets(frame, prices, "QQQ", (5, 20))
    score_cols = [
        "excess_20",
        "excess_60",
        "excess_120",
        "rs_slope_20",
        "ret_20",
        "ret_60",
        "ret_120",
        "high_proximity_252",
        "above_ma60",
        "above_ma200",
        "vol_20",
        "drawdown_60",
    ]
    frame = zscore_by_date(frame, [c for c in score_cols if c in frame.columns])
    frame["rule_leadership_score"] = (
        0.24 * frame["z_excess_20"]
        + 0.24 * frame["z_excess_60"]
        + 0.14 * frame["z_excess_120"]
        + 0.12 * frame["z_rs_slope_20"]
        + 0.10 * frame["z_high_proximity_252"]
        + 0.06 * frame["z_above_ma60"]
        + 0.06 * frame["z_above_ma200"]
        - 0.04 * frame["z_vol_20"]
    )
    return frame.replace([np.inf, -np.inf], np.nan)


def load_macro_panel() -> pd.DataFrame:
    path = ROOT / "outputs" / "macro_regime_asset_screener_latest" / "tables" / "driver_panel.csv"
    macro = pd.read_csv(path, parse_dates=["Date"]).rename(columns={"Date": "date"}).sort_values("date")
    for col in macro.columns:
        if col != "date":
            macro[col] = pd.to_numeric(macro[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    out = macro[["date"]].copy()
    for src, dst in [("US2Y", "macro_US2Y_level"), ("US10Y", "macro_US10Y_level"), ("VIX", "macro_VIX_level")]:
        out[dst] = macro[src] if src in macro else np.nan
    for col in ["US10Y", "US2Y", "VIX", "HY_OAS"]:
        if col in macro:
            for w in [5, 20]:
                out[f"macro_{col}_{w}d_chg"] = macro[col].diff(w)
    for col in ["DXY", "USDKRW", "GOLD", "NASDAQ100", "HYG_IEF"]:
        if col in macro:
            out[f"macro_{col}_20d_ret"] = macro[col].pct_change(20)
    return out


def load_v4_wide() -> pd.DataFrame:
    path = ROOT / "outputs" / "risk_off_v4_event_label_latest" / "tables" / "risk_off_v4_walkforward_predictions.csv"
    pred = pd.read_csv(path, parse_dates=["date"])
    parts = []
    keep = [
        "date",
        "horizon",
        "risk_off_v4_prob",
        "risk_off_v4_stage",
        "axis1_vol_credit_stress",
        "axis2_fx_liquidity_stress",
        "axis3_peak_fragility_stress",
    ]
    for horizon, part in pred[[c for c in keep if c in pred.columns]].groupby("horizon"):
        x = part.drop(columns=["horizon"]).copy()
        x = x.rename(columns={c: f"{c}_{horizon}" for c in x.columns if c != "date"})
        parts.append(x)
    if not parts:
        return pd.DataFrame(columns=["date"])
    out = parts[0]
    for part in parts[1:]:
        out = out.merge(part, on="date", how="outer")
    stage = out.get("risk_off_v4_stage_1m", pd.Series(index=out.index, dtype=object)).map({"Normal": 0, "Watch": 1, "De-risk": 2, "Cash": 3})
    out["stage_level_1m"] = stage.fillna(0).astype(float)
    return out.sort_values("date")


def asof_merge(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    return pd.merge_asof(left.sort_values("date"), right.sort_values("date"), on="date", direction="backward")


def build_safe_features(prices: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    frame = add_basic_asset_features(prices, SAFE_UNIVERSE, None)
    frame = frame[frame["date"].ge(pd.Timestamp(args.backtest_start))].copy()
    frame = add_forward_targets(frame, prices, None, (5, 20))
    macro = load_macro_panel()
    v4 = load_v4_wide()
    frame = asof_merge(frame, macro)
    frame = asof_merge(frame, v4)
    score_cols = [
        "ret_5",
        "ret_20",
        "ret_60",
        "ret_120",
        "vol_20",
        "drawdown_60",
        "high_proximity_252",
        "above_ma60",
        "above_ma200",
    ]
    frame = zscore_by_date(frame, [c for c in score_cols if c in frame.columns])
    frame["rule_safe_score"] = (
        0.20 * frame["z_ret_20"]
        + 0.20 * frame["z_ret_60"]
        + 0.12 * frame["z_ret_120"]
        + 0.12 * frame["z_high_proximity_252"]
        + 0.08 * frame["z_above_ma60"]
        + 0.08 * frame["z_above_ma200"]
        - 0.10 * frame["z_vol_20"]
        - 0.10 * frame["z_drawdown_60"]
    )
    frame = add_macro_conditioned_safe_score(frame)
    return frame.replace([np.inf, -np.inf], np.nan)


def add_macro_conditioned_safe_score(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    rate_chg = pd.to_numeric(out.get("macro_US10Y_20d_chg", 0.0), errors="coerce").fillna(0.0)
    vix_chg = pd.to_numeric(out.get("macro_VIX_20d_chg", 0.0), errors="coerce").fillna(0.0)
    vix_level = pd.to_numeric(out.get("macro_VIX_level", 0.0), errors="coerce").fillna(0.0)
    dxy_ret = pd.to_numeric(out.get("macro_DXY_20d_ret", 0.0), errors="coerce").fillna(0.0)
    usdkrw_ret = pd.to_numeric(out.get("macro_USDKRW_20d_ret", 0.0), errors="coerce").fillna(0.0)
    hy_chg = pd.to_numeric(out.get("macro_HY_OAS_20d_chg", 0.0), errors="coerce").fillna(0.0)
    gold_ret = pd.to_numeric(out.get("macro_GOLD_20d_ret", 0.0), errors="coerce").fillna(0.0)
    nasdaq_ret = pd.to_numeric(out.get("macro_NASDAQ100_20d_ret", 0.0), errors="coerce").fillna(0.0)
    higheq_stress = (vix_level.sub(20).clip(lower=0) / 20.0 + vix_chg.clip(lower=0) / 15.0 + hy_chg.clip(lower=0) / 2.0).clip(0, 2.5)
    axis1 = pd.to_numeric(out.get("axis1_vol_credit_stress_1m", 0.0), errors="coerce").fillna(0.0) / 100.0
    axis2 = pd.to_numeric(out.get("axis2_fx_liquidity_stress_1m", 0.0), errors="coerce").fillna(0.0) / 100.0
    axis3 = pd.to_numeric(out.get("axis3_peak_fragility_stress_1m", 0.0), errors="coerce").fillna(0.0) / 100.0

    cat = out["category"].astype(str)
    overlay = pd.Series(0.0, index=out.index)
    long_bond = cat.eq("Long Treasury")
    intermediate = cat.isin(["Intermediate Treasury", "Aggregate Bond", "IG Credit", "MBS", "TIPS"])
    short_cash = cat.isin(["Cash", "Short Treasury"])
    dollar = cat.eq("Dollar")
    gold = cat.eq("Gold")
    yen_swiss = cat.isin(["Yen", "Swiss Franc"])

    # 금리 하락형 risk-off: 장기/중기 국채가 가장 잘 작동한다.
    overlay += np.where(long_bond, -2.4 * rate_chg + 0.45 * higheq_stress + 0.25 * axis1, 0.0)
    overlay += np.where(intermediate, -1.2 * rate_chg + 0.25 * higheq_stress + 0.12 * axis1, 0.0)

    # 금리 상승형 방어: 장기채보다 현금/초단기/달러가 낫다.
    overlay += np.where(short_cash, 1.5 * rate_chg.clip(lower=0) + 0.20 * higheq_stress, 0.0)
    overlay += np.where(dollar, 5.0 * dxy_ret + 2.5 * usdkrw_ret + 0.15 * higheq_stress + 0.45 * axis2, 0.0)

    # 금과 엔/스위스프랑은 공포·고점취약성에는 좋지만 달러 급등/금리급등에는 일부 감점한다.
    overlay += np.where(gold, 2.2 * gold_ret - 1.2 * dxy_ret - 0.6 * rate_chg.clip(lower=0) + 0.20 * higheq_stress + 0.35 * axis3, 0.0)
    overlay += np.where(yen_swiss, -2.0 * dxy_ret + 0.25 * higheq_stress + (-nasdaq_ret).clip(lower=0) + 0.25 * axis2, 0.0)

    out["macro_safe_overlay_raw"] = overlay
    out = zscore_by_date(out, ["macro_safe_overlay_raw"])
    out["macro_conditioned_safe_score"] = 0.55 * out["rule_safe_score"] + 0.45 * out["z_macro_safe_overlay_raw"]
    return out


def rank_data(frame: pd.DataFrame, features: list[str], label: str) -> tuple[pd.DataFrame, pd.Series, list[int], pd.DataFrame]:
    cols = ["date", "ticker", "category", *features, label]
    data = frame[[c for c in cols if c in frame.columns]].dropna(subset=[label]).sort_values(["date", "ticker"]).reset_index(drop=True)
    data = data[data.groupby("date")["ticker"].transform("size").ge(3)].reset_index(drop=True)
    x = data[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = x.groupby(data["date"]).transform(lambda s: s.fillna(s.median())).fillna(0.0)
    y = data[label].astype(int)
    group = data.groupby("date", sort=False).size().astype(int).tolist()
    return x, y, group, data


def fit_predict_walkforward(frame: pd.DataFrame, features: list[str], label: str, score_name: str, initial_year: int) -> pd.DataFrame:
    preds = []
    years = sorted(y for y in frame["date"].dt.year.dropna().unique() if y >= initial_year)
    for year in years:
        train_end = pd.Timestamp(f"{year - 1}-01-01")
        valid_start = pd.Timestamp(f"{year - 1}-01-01")
        valid_end = pd.Timestamp(f"{year}-01-01")
        test_start = pd.Timestamp(f"{year}-01-01")
        test_end = pd.Timestamp(f"{year + 1}-01-01")
        train = frame[frame["date"].lt(train_end)].copy()
        valid = frame[frame["date"].ge(valid_start) & frame["date"].lt(valid_end)].copy()
        test = frame[frame["date"].ge(test_start) & frame["date"].lt(test_end)].copy()
        if train["date"].nunique() < 250 or valid["date"].nunique() < 40 or test.empty:
            continue
        x_train, y_train, g_train, _ = rank_data(train, features, label)
        x_valid, y_valid, g_valid, _ = rank_data(valid, features, label)
        x_test, _, _, d_test = rank_data(test, features, label)
        if x_train.empty or x_valid.empty or x_test.empty:
            continue
        model = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=180,
            learning_rate=0.05,
            num_leaves=11,
            max_depth=4,
            min_child_samples=12,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=1.0,
            reg_lambda=6.0,
            random_state=42 + year,
            n_jobs=-1,
            verbose=-1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(
                x_train,
                y_train,
                group=g_train,
                eval_set=[(x_valid, y_valid)],
                eval_group=[g_valid],
                eval_at=[1, 3, 5],
                callbacks=[early_stopping(30, verbose=False), log_evaluation(0)],
            )
        d_test = d_test.copy()
        d_test[score_name] = model.predict(x_test)
        d_test["model_year"] = year
        preds.append(d_test[["date", "ticker", score_name, "model_year"]])
    if not preds:
        return frame[["date", "ticker"]].copy().assign(**{score_name: np.nan})
    return pd.concat(preds, ignore_index=True)


def add_model_predictions(frame: pd.DataFrame, features: list[str], labels: list[tuple[str, str]], initial_year: int) -> pd.DataFrame:
    out = frame.copy()
    for label, score_name in labels:
        pred = fit_predict_walkforward(out, features, label, score_name, initial_year)
        out = out.merge(pred, on=["date", "ticker"], how="left")
    return out


def rebalance_dates(prices: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, frequency: str) -> list[pd.Timestamp]:
    dates = prices.loc[start:end].index
    if frequency == "W":
        return list(pd.Series(dates, index=dates).resample("W-FRI").last().dropna().values)
    if frequency == "M":
        return list(pd.Series(dates, index=dates).resample("M").last().dropna().values)
    raise ValueError(frequency)


def model_sample_dates(prices: pd.DataFrame, start: str, end: str) -> set[pd.Timestamp]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    return set(rebalance_dates(prices, start_ts, end_ts, "W")) | set(rebalance_dates(prices, start_ts, end_ts, "M"))


def next_date(index: pd.DatetimeIndex, date: pd.Timestamp, frequency: str) -> pd.Timestamp | None:
    pos = index.searchsorted(date)
    if pos >= len(index) - 2:
        return None
    if frequency == "W":
        target_pos = min(pos + 5, len(index) - 1)
    else:
        target_pos = min(pos + 21, len(index) - 1)
    return index[target_pos]


def strategy_backtest(features: pd.DataFrame, prices: pd.DataFrame, spec: BacktestSpec, start: str, end: str) -> tuple[pd.DataFrame, dict]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    rows = []
    dates = rebalance_dates(prices, start_ts, end_ts, spec.frequency)
    f = features.dropna(subset=[spec.score_col]).copy()
    if spec.risk_only and "stage_level_1m" in f.columns:
        f = f[f["stage_level_1m"].fillna(0).ge(1)]
    for date in dates:
        next_d = next_date(prices.index, pd.Timestamp(date), spec.frequency)
        if next_d is None:
            continue
        part = f[f["date"].eq(pd.Timestamp(date))]
        part = part[part["ticker"].isin(prices.columns)]
        if part.empty:
            continue
        picks = part.nlargest(min(spec.top_k, len(part)), spec.score_col)
        selected = [t for t in picks["ticker"].tolist() if pd.notna(prices.at[pd.Timestamp(date), t]) and pd.notna(prices.at[next_d, t])]
        if not selected:
            continue
        returns = prices.loc[next_d, selected] / prices.loc[pd.Timestamp(date), selected] - 1.0
        port_return = float(returns.mean())
        bench_return = np.nan
        if spec.benchmark and spec.benchmark in prices.columns and pd.notna(prices.at[pd.Timestamp(date), spec.benchmark]) and pd.notna(prices.at[next_d, spec.benchmark]):
            bench_return = float(prices.at[next_d, spec.benchmark] / prices.at[pd.Timestamp(date), spec.benchmark] - 1.0)
        rows.append(
            {
                "date": pd.Timestamp(date),
                "exit_date": next_d,
                "strategy": spec.name,
                "frequency": spec.frequency,
                "top_k": spec.top_k,
                "selected": ",".join(selected),
                "period_return": port_return,
                "benchmark_return": bench_return,
                "excess_return": port_return - bench_return if pd.notna(bench_return) else np.nan,
            }
        )
    raw = pd.DataFrame(rows)
    return raw, summarize_backtest(raw, spec)


def summarize_backtest(raw: pd.DataFrame, spec: BacktestSpec) -> dict:
    if raw.empty:
        return {"strategy": spec.name, "frequency": spec.frequency, "top_k": spec.top_k, "periods": 0}
    r = raw["period_return"].fillna(0.0)
    equity = (1.0 + r).cumprod()
    years = max((raw["exit_date"].max() - raw["date"].min()).days / 365.25, 1e-9)
    per_year = 52 if spec.frequency == "W" else 12
    vol = float(r.std() * np.sqrt(per_year))
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0)
    mdd = float((equity / equity.cummax() - 1.0).min())
    sharpe = float(r.mean() * per_year / vol) if vol > 0 else np.nan
    out = {
        "strategy": spec.name,
        "frequency": spec.frequency,
        "top_k": spec.top_k,
        "start": raw["date"].min().date().isoformat(),
        "end": raw["exit_date"].max().date().isoformat(),
        "periods": int(len(raw)),
        "total_return": float(equity.iloc[-1] - 1.0),
        "CAGR": cagr,
        "ann_vol": vol,
        "Sharpe": sharpe,
        "MDD": mdd,
        "Calmar": float(cagr / abs(mdd)) if mdd < 0 else np.nan,
        "hit_rate_positive": float((r > 0).mean()),
    }
    if raw["excess_return"].notna().any():
        out["avg_excess_return"] = float(raw["excess_return"].mean())
        out["hit_rate_excess_positive"] = float((raw["excess_return"] > 0).mean())
    return out


def rank_ic(frame: pd.DataFrame, score_col: str, target_col: str) -> dict:
    vals = []
    for _, part in frame.dropna(subset=[score_col, target_col]).groupby("date"):
        if part["ticker"].nunique() < 4:
            continue
        corr = spearmanr(part[score_col], part[target_col], nan_policy="omit").correlation
        if pd.notna(corr):
            vals.append(float(corr))
    return {
        "score": score_col,
        "target": target_col,
        "dates": len(vals),
        "mean_rank_ic": float(np.mean(vals)) if vals else np.nan,
        "median_rank_ic": float(np.median(vals)) if vals else np.nan,
        "positive_ic_rate": float(np.mean(np.array(vals) > 0)) if vals else np.nan,
    }


def write_report(summary: pd.DataFrame, ic: pd.DataFrame, metadata: dict) -> None:
    lines = [
        "# Long History Proxy Selection Backtest",
        "",
        "이 검증은 2010년대부터 살아있는 장기 프록시 ETF로 수행한다. 실제 DB GAPS ETF의 과거 상장 전 구간과 과거 holdings 생존편향 문제를 피하기 위해, 장기 검증과 실전 유니버스 검증을 분리한다.",
        "",
        "## Metadata",
        "",
        "```json",
        json.dumps(metadata, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Backtest Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Rank IC",
        "",
        ic.to_markdown(index=False),
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dirs()
    all_tickers = sorted(set(LEADERSHIP_UNIVERSE) | set(SAFE_UNIVERSE) | {"QQQ", "SPY"})
    raw_prices = download_prices(all_tickers, args.start, args.end, args.force_download)
    leader_prices = clean_price_universe(raw_prices, list(LEADERSHIP_UNIVERSE), args.backtest_start, args.min_non_null_ratio)
    safe_prices = clean_price_universe(raw_prices, list(SAFE_UNIVERSE), args.backtest_start, args.min_non_null_ratio)
    combined_prices = raw_prices[sorted(set(leader_prices.columns) | set(safe_prices.columns) | {"QQQ", "SPY"} & set(raw_prices.columns))].ffill(limit=5)
    sample_dates = model_sample_dates(combined_prices, args.backtest_start, args.end)

    leadership = build_leadership_features(combined_prices, args)
    leadership = leadership[leadership["date"].isin(sample_dates)].copy()
    leadership_features = [c for c in LEADERSHIP_FEATURES if c in leadership.columns]
    leadership = add_model_predictions(
        leadership,
        leadership_features,
        [("label_5d_rank_int", "ranker_5d_score"), ("label_20d_rank_int", "ranker_20d_score")],
        args.initial_test_year,
    )

    safe = build_safe_features(combined_prices, args)
    safe = safe[safe["date"].isin(sample_dates)].copy()
    safe_features = [c for c in SAFE_FEATURES if c in safe.columns]
    safe = add_model_predictions(
        safe,
        safe_features,
        [("label_5d_rank_int", "safe_ranker_5d_score"), ("label_20d_rank_int", "safe_ranker_20d_score")],
        args.initial_test_year,
    )

    specs = []
    for k in sorted({1, 3, args.top_k}):
        specs.extend(
            [
                BacktestSpec(f"leader_rule_weekly_top{k}", "rule_leadership_score", "leader", k, "W", "QQQ"),
                BacktestSpec(f"leader_ranker_weekly_top{k}", "ranker_5d_score", "leader", k, "W", "QQQ"),
                BacktestSpec(f"leader_rule_monthly_top{k}", "rule_leadership_score", "leader", k, "M", "QQQ"),
                BacktestSpec(f"leader_ranker_monthly_top{k}", "ranker_20d_score", "leader", k, "M", "QQQ"),
            ]
        )
    for k in sorted({1, args.safe_top_k}):
        specs.extend(
            [
                BacktestSpec(f"safe_rule_weekly_top{k}", "rule_safe_score", "safe", k, "W", None),
                BacktestSpec(f"safe_macro_weekly_top{k}", "macro_conditioned_safe_score", "safe", k, "W", None),
                BacktestSpec(f"safe_ranker_weekly_top{k}", "safe_ranker_5d_score", "safe", k, "W", None),
                BacktestSpec(f"safe_rule_monthly_top{k}", "rule_safe_score", "safe", k, "M", None),
                BacktestSpec(f"safe_macro_monthly_top{k}", "macro_conditioned_safe_score", "safe", k, "M", None),
                BacktestSpec(f"safe_ranker_monthly_top{k}", "safe_ranker_20d_score", "safe", k, "M", None),
                BacktestSpec(f"safe_macro_weekly_riskoff_top{k}", "macro_conditioned_safe_score", "safe", k, "W", None, True),
                BacktestSpec(f"safe_ranker_weekly_riskoff_top{k}", "safe_ranker_5d_score", "safe", k, "W", None, True),
                BacktestSpec(f"safe_macro_monthly_riskoff_top{k}", "macro_conditioned_safe_score", "safe", k, "M", None, True),
                BacktestSpec(f"safe_ranker_monthly_riskoff_top{k}", "safe_ranker_20d_score", "safe", k, "M", None, True),
            ]
        )
    raw_bts = []
    summaries = []
    for spec in specs:
        frame = leadership if spec.universe == "leader" else safe
        raw, summary = strategy_backtest(frame, combined_prices, spec, args.backtest_start if "rule" in spec.name else f"{args.initial_test_year}-01-01", args.end)
        raw_bts.append(raw)
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries).sort_values(["strategy"])
    raw_bt_df = pd.concat(raw_bts, ignore_index=True) if raw_bts else pd.DataFrame()
    ic_df = pd.DataFrame(
        [
            rank_ic(leadership, "rule_leadership_score", "fwd_20d_excess"),
            rank_ic(leadership, "ranker_20d_score", "fwd_20d_excess"),
            rank_ic(leadership, "rule_leadership_score", "fwd_5d_excess"),
            rank_ic(leadership, "ranker_5d_score", "fwd_5d_excess"),
            rank_ic(safe, "rule_safe_score", "fwd_20d_return"),
            rank_ic(safe, "macro_conditioned_safe_score", "fwd_20d_return"),
            rank_ic(safe, "safe_ranker_20d_score", "fwd_20d_return"),
            rank_ic(safe, "rule_safe_score", "fwd_5d_return"),
            rank_ic(safe, "macro_conditioned_safe_score", "fwd_5d_return"),
            rank_ic(safe, "safe_ranker_5d_score", "fwd_5d_return"),
        ]
    )

    raw_prices.to_csv(TABLES / "proxy_raw_prices.csv", index_label="date", encoding="utf-8-sig")
    leadership.to_csv(TABLES / "proxy_leadership_features_predictions.csv", index=False, encoding="utf-8-sig")
    safe.to_csv(TABLES / "proxy_safe_features_predictions.csv", index=False, encoding="utf-8-sig")
    raw_bt_df.to_csv(TABLES / "proxy_selection_backtest_trades.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(TABLES / "proxy_selection_backtest_summary.csv", index=False, encoding="utf-8-sig")
    ic_df.to_csv(TABLES / "proxy_selection_rank_ic.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "start_download": args.start,
        "backtest_start": args.backtest_start,
        "end": args.end,
        "initial_ml_test_year": args.initial_test_year,
        "leadership_tickers": list(leader_prices.columns),
        "safe_tickers": list(safe_prices.columns),
        "benchmark_for_leadership": "QQQ",
        "note": "과거 holdings가 없는 ETF는 장기 프록시 가격 기반으로 검증했다. 실제 GAPS ETF 구성종목 리더십 검증은 별도 단기 검증으로 해석해야 한다.",
    }
    write_report(summary_df, ic_df, metadata)
    print("saved", OUT)
    print(summary_df.to_string(index=False))
    print("\nRank IC")
    print(ic_df.to_string(index=False))


if __name__ == "__main__":
    main()
