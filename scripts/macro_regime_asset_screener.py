from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".cache"
PRICE_CACHE = CACHE_DIR / "prices"
FRED_CACHE = CACHE_DIR / "fred"
ASSET_UNIVERSE_PATH = ROOT / "data" / "asset_universe.csv"
OUT_DIR = ROOT / "outputs" / "macro_regime_asset_screener_latest"
TABLE_DIR = OUT_DIR / "tables"
REPORT_DIR = OUT_DIR / "reports"

START_DATE = "1995-01-01"
FORWARD_1W = 5
FORWARD_4W = 20
ROLLING_BETA_WINDOW = 60

try:
    from basket_taxonomy import classify_basket, enrich_asset_name, load_gaps_name_map
except Exception:  # pragma: no cover
    classify_basket = None
    enrich_asset_name = None
    load_gaps_name_map = None


@dataclass(frozen=True)
class SeriesSpec:
    name: str
    symbol: str
    source: str
    kind: str
    higher_is_risk_on: bool
    transform: str = "auto"


@dataclass(frozen=True)
class AssetSpec:
    symbol: str
    name: str
    group: str
    expected_drivers: dict[str, float]


FRED_SERIES = [
    SeriesSpec("US2Y", "DGS2", "fred", "rate", False, "diff"),
    SeriesSpec("US10Y", "DGS10", "fred", "rate", False, "diff"),
    SeriesSpec("US10Y_2Y", "T10Y2Y", "fred", "curve", True, "diff"),
    SeriesSpec("US10Y_REAL", "DFII10", "fred", "real_rate", False, "diff"),
    SeriesSpec("US10Y_BEI", "T10YIE", "fred", "inflation_expectation", True, "diff"),
    SeriesSpec("HY_OAS", "BAMLH0A0HYM2", "fred", "credit", False, "diff"),
    SeriesSpec("IG_OAS", "BAMLC0A0CM", "fred", "credit", False, "diff"),
    SeriesSpec("NFCI", "NFCI", "fred", "financial_conditions", False, "diff"),
    SeriesSpec("ANFCI", "ANFCI", "fred", "financial_conditions", False, "diff"),
    SeriesSpec("USDKRW", "DEXKOUS", "fred", "fx", False, "diff"),
    SeriesSpec("USDJPY", "DEXJPUS", "fred", "fx", True, "diff"),
    SeriesSpec("WTI", "DCOILWTICO", "fred", "commodity", True, "return"),
]

YF_SERIES = [
    SeriesSpec("DXY", "DX-Y.NYB", "yahoo", "fx", False),
    SeriesSpec("USDKRW", "KRW=X", "yahoo", "fx", False),
    SeriesSpec("USDJPY", "JPY=X", "yahoo", "fx", True),
    SeriesSpec("USDCNH", "CNH=X", "yahoo", "fx", False),
    SeriesSpec("VIX", "^VIX", "yahoo", "volatility", False, "diff"),
    SeriesSpec("VXN", "^VXN", "yahoo", "volatility", False, "diff"),
    SeriesSpec("MOVE", "^MOVE", "yahoo", "volatility", False, "diff"),
    SeriesSpec("SP500", "^GSPC", "yahoo", "equity", True),
    SeriesSpec("NASDAQ100", "^NDX", "yahoo", "equity", True),
    SeriesSpec("SOX", "^SOX", "yahoo", "semiconductor", True),
    SeriesSpec("RUSSELL2000", "^RUT", "yahoo", "equity", True),
    SeriesSpec("VALUE_GROWTH", "IWD/IWF", "ratio", "style", True),
    SeriesSpec("CYCLICAL_DEFENSIVE", "XLY/XLP", "ratio", "style", True),
    SeriesSpec("HYG_IEF", "HYG/IEF", "ratio", "credit_risk_appetite", True),
    SeriesSpec("COPPER", "HG=F", "yahoo", "commodity", True),
    SeriesSpec("WTI", "CL=F", "yahoo", "commodity", True),
    SeriesSpec("GOLD", "GC=F", "yahoo", "commodity", True),
    SeriesSpec("COPPER_GOLD", "HG=F/GC=F", "ratio", "commodity_ratio", True),
    SeriesSpec("HANGSENG_TECH", "KWEB", "yahoo", "china", True),
    SeriesSpec("CSI300", "ASHR", "yahoo", "china", True),
    SeriesSpec("KOSDAQ_KOSPI", "229200.KS/069500.KS", "ratio", "korea_internal", True),
]

CORE_DRIVER_NAMES = [
    "US10Y",
    "US2Y",
    "DXY",
    "USDKRW",
    "VIX",
    "HY_OAS",
    "SOX",
    "NASDAQ100",
    "COPPER",
    "WTI",
    "GOLD",
    "HANGSENG_TECH",
    "CSI300",
]

ASSETS = [
    AssetSpec("069500.KS", "KODEX 200", "Korea broad equity", {"SP500": 0.6, "NASDAQ100": 0.3, "SOX": 0.7, "USDKRW": -0.8, "VIX": -0.7, "US10Y": -0.2, "HANGSENG_TECH": 0.3}),
    AssetSpec("229200.KS", "KODEX KOSDAQ150", "Korea growth", {"NASDAQ100": 0.5, "SOX": 0.8, "USDKRW": -0.7, "VIX": -0.7, "US10Y": -0.4, "KOSDAQ_KOSPI": 0.8}),
    AssetSpec("091160.KS", "KODEX Semiconductor", "Korea semiconductor", {"SOX": 1.0, "NASDAQ100": 0.6, "USDKRW": -0.5, "US10Y": -0.4, "VIX": -0.6}),
    AssetSpec("139260.KS", "TIGER 200 IT", "Korea IT", {"SOX": 0.8, "NASDAQ100": 0.7, "USDKRW": -0.5, "US10Y": -0.4, "VIX": -0.6}),
    AssetSpec("091180.KS", "KODEX Autos", "Korea cyclical", {"COPPER": 0.5, "WTI": 0.2, "USDKRW": -0.3, "HANGSENG_TECH": 0.4, "CSI300": 0.4, "VIX": -0.4}),
    AssetSpec("139240.KS", "TIGER 200 Materials", "Korea cyclical", {"COPPER": 0.8, "CSI300": 0.5, "DXY": -0.5, "VIX": -0.3}),
    AssetSpec("139270.KS", "TIGER 200 Financials", "Korea value", {"US10Y_2Y": 0.5, "US10Y": 0.2, "HY_OAS": -0.6, "VALUE_GROWTH": 0.5, "USDKRW": -0.2}),
    AssetSpec("227560.KS", "TIGER 200 Staples", "Korea defensive", {"VIX": 0.1, "CYCLICAL_DEFENSIVE": -0.5, "USDKRW": -0.1, "HY_OAS": -0.2}),
    AssetSpec("360750.KS", "TIGER S&P500", "US broad equity", {"SP500": 1.0, "VIX": -0.7, "HY_OAS": -0.6, "DXY": -0.2}),
    AssetSpec("133690.KS", "TIGER Nasdaq100", "US growth", {"NASDAQ100": 1.0, "SOX": 0.6, "US10Y_REAL": -0.5, "US10Y": -0.4, "VXN": -0.7}),
    AssetSpec("381180.KS", "TIGER US Philadelphia Semiconductor", "US semiconductor", {"SOX": 1.0, "NASDAQ100": 0.7, "US10Y_REAL": -0.4, "VIX": -0.6}),
    AssetSpec("280930.KS", "KODEX Russell2000", "US cyclical", {"RUSSELL2000": 1.0, "COPPER": 0.5, "HY_OAS": -0.7, "US10Y_2Y": 0.4}),
    AssetSpec("182480.KS", "TIGER US REITs", "US REIT", {"US10Y": -0.8, "US10Y_REAL": -0.7, "VIX": -0.4, "IG_OAS": -0.4}),
    AssetSpec("371160.KS", "TIGER Hang Seng Tech", "China/HK growth", {"HANGSENG_TECH": 1.0, "USDCNH": -0.8, "DXY": -0.5, "COPPER": 0.4}),
    AssetSpec("283580.KS", "KODEX China CSI300", "China equity", {"CSI300": 1.0, "USDCNH": -0.7, "DXY": -0.4, "COPPER": 0.5}),
    AssetSpec("453870.KS", "TIGER India Nifty50", "India/EM", {"DXY": -0.6, "US10Y": -0.4, "VIX": -0.5, "SP500": 0.4}),
    AssetSpec("238720.KS", "ACE Japan Nikkei225", "Japan equity", {"USDJPY": 0.5, "SP500": 0.5, "DXY": 0.1, "VIX": -0.4}),
    AssetSpec("411060.KS", "ACE KRX Gold", "Gold", {"GOLD": 1.0, "US10Y_REAL": -0.7, "DXY": -0.5, "VIX": 0.2}),
    AssetSpec("261220.KS", "KODEX WTI Oil", "Oil", {"WTI": 1.0, "DXY": -0.3, "COPPER": 0.2, "US10Y_BEI": 0.4}),
    AssetSpec("152380.KS", "KODEX 10Y KTB", "Korea bonds", {"US10Y": -0.5, "USDKRW": -0.2, "VIX": 0.2, "HY_OAS": -0.1}),
    AssetSpec("114260.KS", "KODEX KTB", "Korea bonds", {"US10Y": -0.3, "USDKRW": -0.1, "VIX": 0.1}),
    AssetSpec("484790.KS", "KODEX US 30Y Treasury Active(H)", "US long bonds", {"US10Y": -1.0, "US10Y_REAL": -0.8, "VIX": 0.2, "GOLD": 0.2}),
    AssetSpec("458260.KS", "TIGER US IG Corporate Active(H)", "US IG bonds", {"US10Y": -0.5, "IG_OAS": -0.7, "VIX": -0.2}),
    AssetSpec("468380.KS", "KODEX iShares US High Yield Active", "US high yield", {"HY_OAS": -1.0, "HYG_IEF": 0.8, "VIX": -0.6, "SP500": 0.4}),
    AssetSpec("261240.KS", "KODEX USD Futures", "USD cash", {"DXY": 0.7, "USDKRW": 1.0, "VIX": 0.4, "HY_OAS": 0.3}),
    AssetSpec("459580.KS", "KODEX CD Rate Active", "Cash/short bonds", {"VIX": 0.3, "HY_OAS": 0.3, "US2Y": 0.1}),
]

DRIVER_TEMPLATES: dict[str, dict[str, float]] = {
    "gold": {"GOLD": 1.0, "US10Y_REAL": -0.7, "DXY": -0.5, "VIX": 0.2},
    "oil": {"WTI": 1.0, "DXY": -0.3, "COPPER": 0.2, "US10Y_BEI": 0.4},
    "usd_cash": {"DXY": 0.7, "USDKRW": 1.0, "VIX": 0.4, "HY_OAS": 0.3},
    "cash": {"VIX": 0.3, "HY_OAS": 0.3, "US2Y": 0.1},
    "us_equity": {"SP500": 1.0, "NASDAQ100": 0.3, "VIX": -0.7, "HY_OAS": -0.6, "DXY": -0.2},
    "us_cyclical": {"RUSSELL2000": 1.0, "COPPER": 0.5, "HY_OAS": -0.7, "US10Y_2Y": 0.4},
    "us_growth": {"NASDAQ100": 1.0, "SOX": 0.6, "US10Y_REAL": -0.5, "US10Y": -0.4, "VXN": -0.7},
    "semiconductor": {"SOX": 1.0, "NASDAQ100": 0.7, "US10Y_REAL": -0.4, "USDKRW": -0.3, "VIX": -0.6},
    "china_growth": {"HANGSENG_TECH": 1.0, "USDCNH": -0.8, "DXY": -0.5, "COPPER": 0.4},
    "china_equity": {"CSI300": 1.0, "USDCNH": -0.7, "DXY": -0.4, "COPPER": 0.5},
    "em_equity": {"DXY": -0.6, "US10Y": -0.4, "VIX": -0.5, "SP500": 0.4, "COPPER": 0.2},
    "japan_equity": {"USDJPY": 0.5, "SP500": 0.5, "DXY": 0.1, "VIX": -0.4},
    "us_reit": {"US10Y": -0.8, "US10Y_REAL": -0.7, "VIX": -0.4, "IG_OAS": -0.4},
    "korea_broad": {"SP500": 0.6, "NASDAQ100": 0.3, "SOX": 0.7, "USDKRW": -0.8, "VIX": -0.7, "US10Y": -0.2, "HANGSENG_TECH": 0.3},
    "korea_growth": {"NASDAQ100": 0.5, "SOX": 0.8, "USDKRW": -0.7, "VIX": -0.7, "US10Y": -0.4, "KOSDAQ_KOSPI": 0.8},
    "korea_it": {"SOX": 0.8, "NASDAQ100": 0.7, "USDKRW": -0.5, "US10Y": -0.4, "VIX": -0.6},
    "korea_cyclical": {"COPPER": 0.7, "WTI": 0.2, "USDKRW": -0.3, "HANGSENG_TECH": 0.4, "CSI300": 0.4, "VIX": -0.4},
    "korea_value": {"US10Y_2Y": 0.5, "US10Y": 0.2, "HY_OAS": -0.6, "VALUE_GROWTH": 0.5, "USDKRW": -0.2},
    "korea_defensive": {"VIX": 0.1, "CYCLICAL_DEFENSIVE": -0.5, "USDKRW": -0.1, "HY_OAS": -0.2},
    "korea_bonds": {"US10Y": -0.4, "USDKRW": -0.15, "VIX": 0.15, "HY_OAS": -0.1},
    "long_bonds": {"US10Y": -1.0, "US10Y_REAL": -0.8, "VIX": 0.2, "GOLD": 0.2},
    "ig_bonds": {"US10Y": -0.5, "IG_OAS": -0.7, "VIX": -0.2},
    "high_yield": {"HY_OAS": -1.0, "HYG_IEF": 0.8, "VIX": -0.6, "SP500": 0.4},
}


def load_asset_universe(path: Path = ASSET_UNIVERSE_PATH) -> list[AssetSpec]:
    if not path.exists():
        return ASSETS
    frame = pd.read_csv(path)
    name_map = load_gaps_name_map() if load_gaps_name_map else {}
    assets: list[AssetSpec] = []
    seen: set[str] = set()
    for _, row in frame.iterrows():
        group = str(row.get("group", "Other")).strip() or "Other"
        template_name = str(row.get("driver_template", "")).strip()
        drivers = DRIVER_TEMPLATES.get(template_name, DRIVER_TEMPLATES.get("us_equity", {}))
        if "codes" in frame.columns and pd.notna(row.get("codes")):
            code_tokens = str(row.get("codes", "")).split()
            for token in code_tokens:
                symbol = normalize_asset_symbol(token)
                if symbol in seen:
                    continue
                seen.add(symbol)
                name = enrich_asset_name(symbol, token.strip(), name_map) if enrich_asset_name else token.strip()
                assets.append(AssetSpec(symbol, name, group, dict(drivers)))
        elif pd.notna(row.get("symbol")):
            symbol = normalize_asset_symbol(str(row["symbol"]))
            if symbol in seen:
                continue
            seen.add(symbol)
            fallback = str(row.get("name", symbol)).strip() or symbol
            name = enrich_asset_name(symbol, fallback, name_map) if enrich_asset_name else fallback
            assets.append(AssetSpec(symbol, name, group, dict(drivers)))
    return assets or ASSETS


def normalize_asset_symbol(token: str) -> str:
    code = token.strip().upper()
    if code.startswith("A"):
        code = code[1:]
    if "." in code:
        return code
    return f"{code}.KS"


ASSETS = load_asset_universe()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Macro regime asset screener with dynamic driver fit and upside probabilities.")
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--skip-download", action="store_true", help="Use only cached data.")
    parser.add_argument("--output", type=Path, default=OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (PRICE_CACHE, FRED_CACHE, args.output / "tables", args.output / "reports"):
        path.mkdir(parents=True, exist_ok=True)

    specs = FRED_SERIES + YF_SERIES
    raw, availability = load_driver_series(specs, args.start, args.skip_download)
    driver_panel = make_driver_panel(raw)
    if driver_panel.empty:
        raise SystemExit("No driver data available.")

    driver_features = make_driver_features(driver_panel, specs)
    regime_frame = classify_regimes(driver_panel, driver_features)
    driver_state = current_driver_state(driver_panel, driver_features, specs, regime_frame)

    asset_histories = load_asset_histories([asset.symbol for asset in ASSETS], args.start, args.skip_download)
    asset_scores = score_assets(ASSETS, asset_histories, driver_panel, driver_features, regime_frame, driver_state)
    if asset_scores.empty:
        raise SystemExit("No asset histories available.")

    tables = args.output / "tables"
    reports = args.output / "reports"
    safe_to_csv(driver_panel.reset_index().rename(columns={"index": "Date"}), tables / "driver_panel.csv")
    safe_to_csv(driver_state, tables / "driver_state.csv")
    safe_to_csv(regime_frame.reset_index().rename(columns={"index": "Date"}), tables / "regime_history.csv")
    safe_to_csv(asset_scores, tables / "current_asset_scores.csv")
    safe_to_csv(current_basket_scores(asset_scores), tables / "current_basket_scores.csv")
    pd.DataFrame(availability).to_csv(tables / "data_availability.csv", index=False, encoding="utf-8-sig")
    write_report(asset_scores, driver_state, regime_frame, availability, reports / "current_report.md")
    print(f"wrote {tables / 'current_asset_scores.csv'}")
    print(asset_scores.head(15).to_string(index=False))


def load_driver_series(specs: list[SeriesSpec], start: str, skip_download: bool) -> tuple[dict[str, pd.Series], list[dict[str, Any]]]:
    raw: dict[str, pd.Series] = {}
    availability: list[dict[str, Any]] = []
    base_yahoo_symbols = sorted({part for spec in specs if spec.source in {"yahoo", "ratio"} for part in spec.symbol.split("/") if spec.source != "fred"})
    yahoo_prices = load_yahoo_prices(base_yahoo_symbols, start, skip_download)

    for spec in specs:
        series = pd.Series(dtype=float)
        error = ""
        try:
            if spec.source == "fred":
                series = load_fred_series(spec.symbol, start, skip_download)
            elif spec.source == "yahoo":
                series = yahoo_prices.get(spec.symbol, pd.Series(dtype=float))
                if series.empty and spec.name == "USDCNH":
                    series = yahoo_prices.get("CNY=X", pd.Series(dtype=float))
            elif spec.source == "ratio":
                left, right = spec.symbol.split("/", 1)
                series = yahoo_prices.get(left, pd.Series(dtype=float)) / yahoo_prices.get(right, pd.Series(dtype=float))
        except Exception as exc:
            error = str(exc)
            series = pd.Series(dtype=float)
        series = clean_series(series)
        if not series.empty:
            if spec.name in raw and not raw[spec.name].empty:
                combined = raw[spec.name].combine_first(series).sort_index()
                if combined.index.min() > series.index.min():
                    combined = series.combine_first(raw[spec.name]).sort_index()
                raw[spec.name] = clean_series(combined)
            else:
                raw[spec.name] = series
        availability.append(
            {
                "name": spec.name,
                "symbol": spec.symbol,
                "source": spec.source,
                "available": not series.empty,
                "last_date": series.index.max().date().isoformat() if not series.empty else None,
                "points": int(series.shape[0]),
                "error": error,
            }
        )
    return raw, availability


def load_yahoo_prices(symbols: list[str], start: str, skip_download: bool) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    if "CNH=X" in symbols and "CNY=X" not in symbols:
        symbols = [*symbols, "CNY=X"]
    missing = []
    requested_start = pd.Timestamp(start)
    for symbol in symbols:
        cached = read_price_cache(symbol)
        cache_is_stale = cached.empty or cached.index.max() < pd.Timestamp.today().normalize() - pd.Timedelta(days=7)
        cache_starts_too_late = (not cached.empty) and cached.index.min() > requested_start + pd.Timedelta(days=30)
        if cache_is_stale or cache_starts_too_late:
            missing.append(symbol)
        else:
            out[symbol] = cached.loc[requested_start:, "Close"]
    if missing and not skip_download:
        import yfinance as yf

        for batch_start in range(0, len(missing), 24):
            batch = missing[batch_start : batch_start + 24]
            data = yf.download(batch, start=start, auto_adjust=True, group_by="ticker", threads=True, progress=False)
            if data.empty:
                continue
            if isinstance(data.columns, pd.MultiIndex):
                for symbol in batch:
                    if symbol in data.columns.get_level_values(0):
                        frame = data[symbol].dropna(how="all")
                        existing = read_price_cache(symbol)
                        frame = merge_price_frames(existing, frame)
                        write_price_cache(symbol, frame)
                        if "Close" in frame:
                            out[symbol] = clean_series(frame.loc[requested_start:, "Close"])
            else:
                symbol = batch[0]
                frame = data.dropna(how="all")
                existing = read_price_cache(symbol)
                frame = merge_price_frames(existing, frame)
                write_price_cache(symbol, frame)
                if "Close" in frame:
                    out[symbol] = clean_series(frame.loc[requested_start:, "Close"])
    for symbol in symbols:
        if symbol not in out:
            cached = read_price_cache(symbol)
            if not cached.empty and "Close" in cached:
                out[symbol] = clean_series(cached.loc[requested_start:, "Close"])
    return out


def merge_price_frames(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if existing is None or existing.empty:
        return new.sort_index()
    if new is None or new.empty:
        return existing.sort_index()
    cols = sorted(set(existing.columns).union(new.columns))
    return new.reindex(columns=cols).combine_first(existing.reindex(columns=cols)).sort_index()


def load_asset_histories(symbols: list[str], start: str, skip_download: bool) -> dict[str, pd.DataFrame]:
    load_yahoo_prices(symbols, start, skip_download)
    return {symbol: read_price_cache(symbol) for symbol in symbols if not read_price_cache(symbol).empty}


def load_fred_series(series_id: str, start: str, skip_download: bool) -> pd.Series:
    path = FRED_CACHE / f"{series_id}.csv"
    if path.exists():
        frame = read_fred_csv(path, series_id)
        series = pd.to_numeric(frame[series_id], errors="coerce")
        series.index = frame["Date"]
        if series.index.max() >= pd.Timestamp.today().normalize() - pd.Timedelta(days=7):
            return series.loc[pd.Timestamp(start) :]
    if skip_download:
        return pd.Series(dtype=float)
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    path.write_text(response.text, encoding="utf-8")
    frame = read_fred_csv(path, series_id)
    series = pd.to_numeric(frame[series_id], errors="coerce")
    series.index = frame["Date"]
    return series.loc[pd.Timestamp(start) :]


def read_fred_csv(path: Path, series_id: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    date_col = "DATE" if "DATE" in frame.columns else "observation_date"
    if date_col not in frame.columns:
        raise ValueError(f"FRED cache {path.name} has no date column")
    if series_id not in frame.columns:
        raise ValueError(f"FRED cache {path.name} has no {series_id} column")
    out = frame[[date_col, series_id]].copy()
    out = out.rename(columns={date_col: "Date"})
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out[series_id] = pd.to_numeric(out[series_id], errors="coerce")
    return out.dropna(subset=["Date"])


def read_price_cache(symbol: str) -> pd.DataFrame:
    path = PRICE_CACHE / f"{safe_name(symbol)}.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    except Exception:
        return pd.DataFrame()
    return frame.sort_index()


def write_price_cache(symbol: str, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    out = frame.copy()
    out.index.name = "Date"
    out.to_csv(PRICE_CACHE / f"{safe_name(symbol)}.csv", encoding="utf-8-sig")


def safe_name(symbol: str) -> str:
    return symbol.replace("^", "_idx_").replace("/", "_").replace("=", "_").replace(".", "_").replace("-", "_")


def clean_series(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out.sort_index()


def make_driver_panel(raw: dict[str, pd.Series]) -> pd.DataFrame:
    frame = pd.DataFrame(raw).sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame.ffill(limit=5).dropna(how="all")


def make_driver_features(panel: pd.DataFrame, specs: list[SeriesSpec]) -> pd.DataFrame:
    fields: dict[str, pd.Series] = {}
    spec_map = {spec.name: spec for spec in specs}
    for name in panel.columns:
        series = panel[name].dropna()
        spec = spec_map.get(name)
        base = series.diff() if spec and spec.transform == "diff" else series.pct_change()
        fields[f"{name}_chg_5d"] = change(series, 5, spec)
        fields[f"{name}_chg_20d"] = change(series, 20, spec)
        fields[f"{name}_chg_60d"] = change(series, 60, spec)
        fields[f"{name}_z_20d"] = zscore(series, 20)
        fields[f"{name}_z_60d"] = zscore(series, 60)
        fields[f"{name}_z_252d"] = zscore(series, 252)
        fields[f"{name}_ma20_pos"] = series / series.rolling(20).mean() - 1.0
        fields[f"{name}_ma60_pos"] = series / series.rolling(60).mean() - 1.0
        fields[f"{name}_slope_20d"] = rolling_slope(series, 20)
        vol = base.rolling(60).std().replace(0, np.nan)
        fields[f"{name}_vol_adj_20d"] = change(series, 20, spec) / vol
        direction = 1.0 if spec is None or spec.higher_is_risk_on else -1.0
        fields[f"{name}_riskon_score"] = direction * zscore(change(series, 20, spec), 252)
    return pd.DataFrame(fields).reindex(panel.index).replace([np.inf, -np.inf], np.nan)


def change(series: pd.Series, periods: int, spec: SeriesSpec | None) -> pd.Series:
    if spec and spec.transform == "diff":
        return series.diff(periods)
    return series.pct_change(periods)


def zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std().replace(0, np.nan)
    return (series - mean) / std


def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    x = x - x.mean()
    denom = float((x * x).sum())

    def slope(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        y = values - values.mean()
        return float((x * y).sum() / denom)

    return series.rolling(window).apply(slope, raw=True)


def classify_regimes(panel: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    inputs = []
    for name in CORE_DRIVER_NAMES:
        col = f"{name}_riskon_score"
        if col in features:
            inputs.append(col)
    x = features[inputs].copy().dropna()
    if x.shape[0] < 260 or x.shape[1] < 5:
        rule = rule_regime(features)
        return rule
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    z = scaler.fit_transform(x.clip(-5, 5))
    gmm = GaussianMixture(n_components=4, covariance_type="full", random_state=42, n_init=10)
    cluster = pd.Series(gmm.fit_predict(z), index=x.index, name="gmm_cluster")
    proba = pd.DataFrame(gmm.predict_proba(z), index=x.index, columns=[f"gmm_prob_{i}" for i in range(4)])
    post = post_label_gmm(cluster, features.loc[x.index])
    rule = rule_regime(features).reindex(x.index)
    out = pd.concat([cluster, proba, post, rule[["rule_regime", "rule_confidence"]]], axis=1)
    return out.reindex(panel.index).ffill()


def post_label_gmm(cluster: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
    labels: dict[int, str] = {}
    scores: dict[int, float] = {}
    for cid, idx in cluster.groupby(cluster).groups.items():
        f = features.loc[idx]
        growth = mean_cols(f, ["NASDAQ100_riskon_score", "SOX_riskon_score", "VIX_riskon_score", "HY_OAS_riskon_score", "DXY_riskon_score"])
        cyc = mean_cols(f, ["COPPER_riskon_score", "WTI_riskon_score", "HANGSENG_TECH_riskon_score", "CSI300_riskon_score", "US10Y_2Y_riskon_score"])
        defensive = mean_cols(f, ["US10Y_riskon_score", "US10Y_REAL_riskon_score", "GOLD_riskon_score"]) - mean_cols(f, ["COPPER_riskon_score", "SP500_riskon_score"])
        cash = -mean_cols(f, ["VIX_riskon_score", "HY_OAS_riskon_score", "DXY_riskon_score", "USDKRW_riskon_score", "SOX_riskon_score"])
        vals = {
            "Risk-On Growth": growth,
            "Risk-On Cyclical": cyc,
            "Defensive / Rate-Cut": defensive,
            "Risk-Off / Cash": cash,
        }
        label = max(vals, key=vals.get)
        labels[int(cid)] = label
        scores[int(cid)] = float(vals[label])
    return pd.DataFrame(
        {
            "gmm_regime": cluster.map(labels),
            "gmm_regime_score": cluster.map(scores),
        },
        index=cluster.index,
    )


def mean_cols(frame: pd.DataFrame, cols: list[str]) -> float:
    have = [col for col in cols if col in frame]
    if not have:
        return 0.0
    return float(frame[have].mean(axis=1).mean())


def rule_regime(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date, row in features.iterrows():
        scores = regime_scores(row)
        label = max(scores, key=scores.get)
        vals = sorted(scores.values(), reverse=True)
        confidence = sigmoid(vals[0] - vals[1]) if len(vals) > 1 else 0.5
        rows.append({"Date": date, "rule_regime": label, "rule_confidence": confidence, **{f"rule_score_{k}": v for k, v in scores.items()}})
    return pd.DataFrame(rows).set_index("Date")


def regime_scores(row: pd.Series) -> dict[str, float]:
    def v(name: str) -> float:
        val = row.get(f"{name}_riskon_score")
        return 0.0 if pd.isna(val) else float(np.clip(val, -3, 3))

    return {
        "Risk-On Growth": 0.22 * v("VIX") + 0.20 * v("HY_OAS") + 0.22 * v("NASDAQ100") + 0.22 * v("SOX") + 0.14 * v("DXY"),
        "Risk-On Cyclical": 0.22 * v("COPPER") + 0.12 * v("WTI") + 0.22 * v("HANGSENG_TECH") + 0.16 * v("CSI300") + 0.14 * v("DXY") + 0.14 * v("US10Y_2Y"),
        "Defensive / Rate-Cut": 0.28 * v("US10Y") + 0.18 * v("US10Y_REAL") + 0.22 * v("GOLD") - 0.16 * v("COPPER") - 0.16 * v("SP500"),
        "Risk-Off / Cash": -0.24 * v("VIX") - 0.24 * v("HY_OAS") - 0.18 * v("DXY") - 0.18 * v("USDKRW") - 0.16 * v("SOX"),
    }


def current_driver_state(panel: pd.DataFrame, features: pd.DataFrame, specs: list[SeriesSpec], regime_frame: pd.DataFrame) -> pd.DataFrame:
    latest_date = features.dropna(how="all").index.max()
    spec_map = {spec.name: spec for spec in specs}
    rows = []
    for name in panel.columns:
        spec = spec_map.get(name)
        rows.append(
            {
                "asof": latest_date.date().isoformat(),
                "driver": name,
                "kind": spec.kind if spec else None,
                "level": safe_float(panel[name].loc[:latest_date].dropna().iloc[-1] if name in panel else np.nan),
                "change_5d": safe_float(features.get(f"{name}_chg_5d", pd.Series(dtype=float)).loc[latest_date] if f"{name}_chg_5d" in features else np.nan),
                "change_20d": safe_float(features.get(f"{name}_chg_20d", pd.Series(dtype=float)).loc[latest_date] if f"{name}_chg_20d" in features else np.nan),
                "z_60d": safe_float(features.get(f"{name}_z_60d", pd.Series(dtype=float)).loc[latest_date] if f"{name}_z_60d" in features else np.nan),
                "riskon_score": safe_float(features.get(f"{name}_riskon_score", pd.Series(dtype=float)).loc[latest_date] if f"{name}_riskon_score" in features else np.nan),
                "higher_is_risk_on": spec.higher_is_risk_on if spec else None,
            }
        )
    out = pd.DataFrame(rows).sort_values("driver")
    latest_regime = regime_frame.loc[:latest_date].dropna(how="all").iloc[-1]
    out.attrs["latest_regime"] = latest_regime.to_dict()
    return out


def score_assets(
    assets: list[AssetSpec],
    histories: dict[str, pd.DataFrame],
    driver_panel: pd.DataFrame,
    driver_features: pd.DataFrame,
    regime_frame: pd.DataFrame,
    driver_state: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    latest_regime_row = regime_frame.dropna(how="all").iloc[-1]
    current_regime = str(latest_regime_row.get("gmm_regime") or latest_regime_row.get("rule_regime"))
    regime_confidence = safe_float(latest_regime_row.get("rule_confidence")) or 0.5
    for asset in assets:
        hist = histories.get(asset.symbol)
        if hist is None or hist.empty or "Close" not in hist:
            continue
        close = clean_series(hist["Close"])
        if close.shape[0] < 140:
            continue
        aligned = pd.concat([close.rename("asset"), driver_panel], axis=1).ffill().dropna(subset=["asset"])
        latest_date = aligned.index.max()
        ret = close.pct_change()
        technical = technical_score(close)
        driver_fit = asset_driver_fit(asset, driver_features.loc[:latest_date])
        betas = rolling_driver_betas(ret, driver_panel, asset.expected_drivers)
        beta_fit = beta_alignment_score(betas, asset.expected_drivers)
        win_1w, avg_1w = conditional_forward_stats(close, regime_frame, current_regime, FORWARD_1W)
        win_4w, avg_4w = conditional_forward_stats(close, regime_frame, current_regime, FORWARD_4W)
        prob_1w = blend_probability(win_1w, technical, driver_fit, beta_fit, horizon="1w")
        prob_4w = blend_probability(win_4w, technical, driver_fit, beta_fit, horizon="4w")
        risk_penalty = risk_score(close)
        final_score = clip100(0.34 * technical + 0.28 * driver_fit + 0.18 * beta_fit + 0.20 * (prob_4w * 100.0) - risk_penalty)
        rows.append(
            {
                "asof": latest_date.date().isoformat(),
                "symbol": asset.symbol,
                "name": asset.name,
                "group": asset.group,
                "basket": classify_basket(asset.group, asset.name, asset.symbol) if classify_basket else asset.group,
                "current_regime": current_regime,
                "regime_confidence": round(regime_confidence, 3),
                "score_0_100": round(final_score, 2),
                "upside_prob_1w": round(prob_1w, 4),
                "upside_prob_4w": round(prob_4w, 4),
                "conditional_win_rate_1w": round(win_1w, 4) if not math.isnan(win_1w) else np.nan,
                "conditional_win_rate_4w": round(win_4w, 4) if not math.isnan(win_4w) else np.nan,
                "conditional_avg_return_1w": round(avg_1w, 4) if not math.isnan(avg_1w) else np.nan,
                "conditional_avg_return_4w": round(avg_4w, 4) if not math.isnan(avg_4w) else np.nan,
                "technical_score": round(technical, 2),
                "driver_fit_score": round(driver_fit, 2),
                "rolling_beta_fit_score": round(beta_fit, 2),
                "risk_penalty": round(risk_penalty, 2),
                "return_5d": round(pct_return(close, 5), 4),
                "return_20d": round(pct_return(close, 20), 4),
                "return_60d": round(pct_return(close, 60), 4),
                "return_120d": round(pct_return(close, 120), 4),
                "volatility_20d_ann": round(ann_vol(close, 20), 4),
                "drawdown_252d": round(drawdown(close, 252), 4),
                "beta_snapshot_json": json.dumps({k: round(v, 4) for k, v in betas.items()}, ensure_ascii=False),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["rank"] = frame["score_0_100"].rank(ascending=False, method="first").astype(int)
    if "basket" in frame:
        frame["basket_rank"] = frame.groupby("basket")["score_0_100"].rank(ascending=False, method="first").astype(int)
    return frame.sort_values(["score_0_100", "upside_prob_4w"], ascending=[False, False]).reset_index(drop=True)


def current_basket_scores(asset_scores: pd.DataFrame) -> pd.DataFrame:
    if asset_scores.empty or "basket" not in asset_scores:
        return pd.DataFrame()
    rows = []
    for basket, group in asset_scores.groupby("basket"):
        ranked = group.sort_values("score_0_100", ascending=False)
        top = ranked.head(min(5, len(ranked)))
        rows.append(
            {
                "asof": ranked["asof"].iloc[0],
                "basket": basket,
                "asset_count": int(len(ranked)),
                "basket_score_0_100": float(0.55 * top["score_0_100"].mean() + 0.45 * ranked["score_0_100"].mean()),
                "basket_upside_prob_1w": float(top["upside_prob_1w"].mean()),
                "basket_upside_prob_4w": float(top["upside_prob_4w"].mean()),
                "basket_return_20d": float(ranked["return_20d"].mean()),
                "basket_risk_penalty": float(ranked["risk_penalty"].mean()),
                "top_symbols": ", ".join(top["symbol"].astype(str).tolist()),
                "top_names": " | ".join(top["name"].astype(str).tolist()),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["basket_rank"] = out["basket_score_0_100"].rank(ascending=False, method="first").astype(int)
    return out.sort_values("basket_rank").reset_index(drop=True)


def technical_score(close: pd.Series) -> float:
    r20 = pct_return(close, 20)
    r60 = pct_return(close, 60)
    r120 = pct_return(close, 120)
    vol20 = ann_vol(close, 20)
    ma200 = close.rolling(200).mean().iloc[-1]
    ma_dist = close.iloc[-1] / ma200 - 1 if pd.notna(ma200) and ma200 else 0.0
    trend_speed = rolling_slope(np.log(close.dropna()), 20).iloc[-1]
    edge = 3.0 * r20 + 2.0 * r60 + 1.2 * r120 + 1.0 * ma_dist + 18.0 * trend_speed - 0.8 * max(vol20 - 0.20, 0)
    raw = 50 + 42 * math.tanh(edge)
    return clip100(raw)


def asset_driver_fit(asset: AssetSpec, features: pd.DataFrame) -> float:
    latest = features.dropna(how="all").iloc[-1]
    weighted = []
    for driver, expected in asset.expected_drivers.items():
        val = latest.get(f"{driver}_riskon_score")
        if pd.isna(val):
            continue
        weighted.append(float(np.clip(val, -3, 3)) * expected)
    if not weighted:
        return 50.0
    raw = np.mean(weighted)
    return clip100(50 + 16 * raw)


def rolling_driver_betas(asset_returns: pd.Series, driver_panel: pd.DataFrame, drivers: dict[str, float]) -> dict[str, float]:
    betas = {}
    for driver in drivers:
        if driver not in driver_panel:
            continue
        dret = driver_panel[driver].pct_change()
        pair = pd.concat([asset_returns.rename("asset"), dret.rename("driver")], axis=1).dropna().tail(ROLLING_BETA_WINDOW)
        if pair.shape[0] < 30:
            continue
        var = pair["driver"].var()
        if not var or pd.isna(var):
            continue
        betas[driver] = float(pair["asset"].cov(pair["driver"]) / var)
    return betas


def beta_alignment_score(betas: dict[str, float], expected: dict[str, float]) -> float:
    vals = []
    for driver, beta in betas.items():
        exp = expected.get(driver)
        if exp is None:
            continue
        vals.append(np.sign(beta) * np.sign(exp) * min(abs(beta), 3.0) / 3.0)
    if not vals:
        return 50.0
    return clip100(50 + 35 * float(np.mean(vals)))


def conditional_forward_stats(close: pd.Series, regime_frame: pd.DataFrame, regime: str, horizon: int) -> tuple[float, float]:
    forward = close.shift(-horizon) / close - 1.0
    regimes = regime_frame["gmm_regime"] if "gmm_regime" in regime_frame else regime_frame["rule_regime"]
    data = pd.concat([forward.rename("forward"), regimes.rename("regime")], axis=1).dropna()
    sample = data[data["regime"].astype(str).eq(regime)]
    if sample.shape[0] < 20:
        sample = data.tail(252)
    if sample.empty:
        return np.nan, np.nan
    return float((sample["forward"] > 0).mean()), float(sample["forward"].mean())


def blend_probability(win_rate: float, technical: float, driver_fit: float, beta_fit: float, horizon: str) -> float:
    base = 0.52 if math.isnan(win_rate) else win_rate
    tech_edge = (technical - 50) / 100
    driver_edge = (driver_fit - 50) / 100
    beta_edge = (beta_fit - 50) / 100
    if horizon == "1w":
        logit = safe_logit(base) + 1.0 * tech_edge + 0.7 * driver_edge + 0.25 * beta_edge
    else:
        logit = safe_logit(base) + 0.7 * tech_edge + 1.0 * driver_edge + 0.45 * beta_edge
    return float(np.clip(sigmoid(logit), 0.05, 0.95))


def risk_score(close: pd.Series) -> float:
    vol = ann_vol(close, 20)
    dd = abs(drawdown(close, 252))
    return min(18.0, max(0.0, (vol - 0.18) * 25 + dd * 15))


def pct_return(close: pd.Series, periods: int) -> float:
    if close.shape[0] <= periods:
        return np.nan
    return float(close.iloc[-1] / close.iloc[-periods - 1] - 1.0)


def ann_vol(close: pd.Series, window: int) -> float:
    ret = close.pct_change().dropna()
    if ret.shape[0] < window:
        return np.nan
    return float(ret.tail(window).std() * np.sqrt(252))


def drawdown(close: pd.Series, window: int) -> float:
    recent = close.dropna().tail(window)
    if recent.empty:
        return np.nan
    return float(recent.iloc[-1] / recent.max() - 1.0)


def clip100(value: float) -> float:
    if pd.isna(value):
        return 50.0
    return float(np.clip(value, 0, 100))


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(np.clip(value, -20, 20))))


def safe_logit(prob: float) -> float:
    p = float(np.clip(prob, 0.01, 0.99))
    return math.log(p / (1 - p))


def safe_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def safe_to_csv(frame: pd.DataFrame, path: Path) -> Path:
    try:
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}{path.suffix}")
        frame.to_csv(fallback, index=False, encoding="utf-8-sig")
        return fallback


def write_report(asset_scores: pd.DataFrame, driver_state: pd.DataFrame, regime_frame: pd.DataFrame, availability: list[dict[str, Any]], path: Path) -> None:
    latest_regime = regime_frame.dropna(how="all").iloc[-1]
    regime = latest_regime.get("gmm_regime") or latest_regime.get("rule_regime")
    asof = asset_scores["asof"].iloc[0]
    top = asset_scores.head(20)[
        [
            "rank",
            "symbol",
            "name",
            "group",
            "score_0_100",
            "upside_prob_1w",
            "upside_prob_4w",
            "technical_score",
            "driver_fit_score",
            "rolling_beta_fit_score",
            "return_20d",
            "drawdown_252d",
        ]
    ]
    driver_top = driver_state.sort_values("riskon_score", ascending=False).head(12)
    missing = pd.DataFrame(availability)
    missing = missing[~missing["available"]][["name", "symbol", "source", "error"]]
    lines = [
        "# Macro Regime Asset Screener",
        "",
        f"- As of: {asof}",
        f"- Current regime: {regime}",
        f"- Rule confidence: {safe_float(latest_regime.get('rule_confidence'))}",
        "",
        "## Top Assets",
        top.to_markdown(index=False),
        "",
        "## Strongest Current Driver States",
        driver_top[["driver", "kind", "level", "change_20d", "z_60d", "riskon_score"]].to_markdown(index=False),
        "",
        "## Method Notes",
        "- Driver features use level/change/momentum/z-score/MA distance/slope/volatility-adjusted change.",
        "- Regime is classified with a 4-state GMM and post-labeled economically; rule scores are kept as an explicit fallback.",
        "- Asset score blends own technical trend, macro driver fit, rolling beta alignment, conditional historical win rate, and risk penalty.",
        "- Upside probability is not a calibrated guarantee; it is a ranking probability for 1-week and 4-week forward positive return.",
    ]
    if not missing.empty:
        lines.extend(["", "## Missing Series", missing.to_markdown(index=False)])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
