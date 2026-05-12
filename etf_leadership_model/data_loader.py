from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf


def load_universe(path: str | Path) -> pd.DataFrame:
    """Load ETF universe.

    Required columns:
    - etf_ticker
    - market: KR or US
    - benchmark_ticker
    """
    frame = pd.read_csv(path)
    required = {"etf_ticker", "market", "benchmark_ticker"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"universe file is missing columns: {sorted(missing)}")
    out = frame.copy()
    out["etf_ticker"] = out["etf_ticker"].astype(str).str.strip()
    out["market"] = out["market"].astype(str).str.strip().str.upper()
    out["benchmark_ticker"] = out["benchmark_ticker"].astype(str).str.strip()
    out = out[out["etf_ticker"].ne("")].drop_duplicates("etf_ticker")
    if not out["market"].isin(["KR", "US"]).all():
        bad = sorted(out.loc[~out["market"].isin(["KR", "US"]), "market"].dropna().unique())
        raise ValueError(f"market must be KR or US. Bad values: {bad}")
    return out.reset_index(drop=True)


def load_holdings(path: str | Path) -> pd.DataFrame:
    """Load ETF holdings.

    Required columns:
    - date
    - etf_ticker
    - component_ticker
    - weight
    """
    frame = pd.read_csv(path)
    required = {"date", "etf_ticker", "component_ticker", "weight"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"holdings file is missing columns: {sorted(missing)}")
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["etf_ticker"] = out["etf_ticker"].astype(str).str.strip()
    out["component_ticker"] = out["component_ticker"].astype(str).str.strip()
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce")
    out = out.dropna(subset=["date", "etf_ticker", "component_ticker", "weight"])
    out = out[(out["weight"] > 0) & (out["weight"] <= 1)]
    out = out.sort_values(["etf_ticker", "date", "weight"], ascending=[True, True, False])
    return out.reset_index(drop=True)


def download_prices(tickers: Iterable[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Download adjusted close prices with yfinance.

    Returns a DataFrame:
    - index: date
    - columns: ticker
    - values: adjusted close when available, otherwise close
    """
    tickers = sorted({str(t).strip() for t in tickers if str(t).strip()})
    if not tickers:
        return pd.DataFrame()
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        actions=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        field = "Adj Close" if "Adj Close" in raw.columns.get_level_values(0) else "Close"
        prices = raw[field].copy()
    else:
        col = "Adj Close" if "Adj Close" in raw.columns else "Close"
        prices = raw[[col]].copy()
        prices.columns = tickers[:1]
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.sort_index()
    prices = prices.loc[:, ~prices.columns.duplicated()]
    return prices.apply(pd.to_numeric, errors="coerce").dropna(how="all")


def load_prices_cache(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"])
    return frame.set_index("date").sort_index()


def save_prices_cache(prices: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = prices.copy()
    out.index.name = "date"
    out.to_csv(path, encoding="utf-8-sig")

