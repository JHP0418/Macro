from __future__ import annotations

import argparse
import json
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from baseline1_model import (
    BASELINE1_BENCHMARKS,
    BASELINE1_RS_TSL_FEATURES,
    baseline1_benchmark_symbol,
    baseline1_latest_features,
)


HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 screening-agent/1.0 (+https://example.local)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


KR_ETF_PREFIXES = (
    "ACE",
    "ARIRANG",
    "FOCUS",
    "HANARO",
    "HK",
    "KBSTAR",
    "KODEX",
    "KOSEF",
    "PLUS",
    "RISE",
    "SMART",
    "SOL",
    "TIGER",
    "TIMEFOLIO",
    "TREX",
    "UNICORN",
    "WOORI",
)


FACTOR_LABELS = {
    "return_3m": "Return (3M)",
    "return_6m": "Return (6M)",
    "return_12m": "Return (12M)",
    "rd_to_market_cap": "R&D / Market Cap",
    "rd_to_total_assets": "R&D / Total Assets",
    "und_gro_score": "Undgro Growth Score (Max 11)",
    "price_to_target": "Price / Target Price",
    "und_bal_score": "Undgro Balance Score (Max 16)",
    "minervini_qm_score": "Minervini QM Score (Max 17)",
    "volume_price_cv_1m": "Volume/Price CV (1M)",
    "innovative_roe": "Innovative ROE",
    "upside_potential": "Upside Potential",
    "gross_income_to_assets": "Gross Income / Total Assets",
    "trend_line_cps": "Trend Line of CPS",
    "high_proximity": "52W High Proximity",
    "overheat_penalty": "Overheat Penalty",
    "foreign_net_buy_strength_20d": "Foreign Net Buy Strength (20D)",
    "foreign_net_buy_strength_60d": "Foreign Net Buy Strength (60D)",
    "pension_net_buy_strength_20d": "Pension Net Buy Strength (20D)",
    "pension_net_buy_strength_60d": "Pension Net Buy Strength (60D)",
    "other_corp_net_buy_strength_20d": "Other Corp Net Buy Strength (20D)",
    "foreign_flow_persistence_20d": "Foreign Net Buy Persistence (20D)",
    "kr_smart_money_combo_20d": "KR Smart Money Combo (20D)",
    "kr_smart_money_accel": "KR Smart Money Acceleration",
    "roe": "ROE",
    "roa": "ROA",
    "debt_to_equity": "Debt / Equity",
    "operating_margin": "Operating Margin",
    "profit_margin": "Profit Margin",
    "revenue_growth": "Revenue Growth",
    "earnings_growth": "Earnings Growth",
    "fcf_to_assets": "FCF / Assets",
    "cash_to_assets": "Cash / Assets",
    "obv_slope_60d": "OBV Slope (60D)",
    "obv_high_proximity": "OBV High Proximity",
    "mfi_14": "Money Flow Index (14D)",
    "dollar_volume_ratio_20_120": "Dollar Volume Ratio (20D/120D)",
    "relative_volume_20_60": "Relative Volume (20D/60D)",
    "up_down_volume_ratio_20d": "Up/Down Volume Ratio (20D)",
    "accumulation_days_20d": "Accumulation Days (20D)",
    "distribution_days_20d": "Distribution Days (20D)",
    "weekly_bullish_divergence_score": "Weekly Bullish Divergence Score",
    "weekly_bearish_divergence_score": "Weekly Bearish Divergence Score",
    "monthly_bullish_divergence_score": "Monthly Bullish Divergence Score",
    "monthly_bearish_divergence_score": "Monthly Bearish Divergence Score",
    "divergence_net_score": "Divergence Net Score",
    "drawdown_from_52w_high": "Drawdown From 52W High",
    "ma50_reclaim_score": "MA50 Reclaim Score",
    "base_breakout_score": "Base Breakout Score",
    "dividend_yield": "Dividend Yield",
    "baseline1_rs_score": "Baseline1 RS Score",
    "baseline1_trend_line_score": "Baseline1 Trend Line Score",
    "baseline1_volume_score": "Baseline1 Volume Score",
    "baseline1_candidate_score": "Baseline1 Legacy Candidate Score",
    "baseline1_entry_signal": "Baseline1 Entry Signal",
    "baseline1_exit_signal": "Baseline1 Exit Signal",
    "baseline1_daily_rs": "Baseline1 Daily RS",
    "baseline1_rs_dist_pct": "Baseline1 RS Distance",
    "baseline1_rs_z_score": "Baseline1 RS Z-Score",
    "baseline1_rs_zero_cross_up": "Baseline1 RS Zero Cross Up",
    "baseline1_rs_zero_cross_down": "Baseline1 RS Zero Cross Down",
    "baseline1_rs_turn_up": "Baseline1 RS Turn Up",
    "baseline1_rs_dist_improve_5d": "Baseline1 RS Improve (5D)",
    "baseline1_rs_red_zone_bars": "Baseline1 RS Red-Zone Bars",
    "baseline1_daily_excess_rs": "Baseline1 Daily Excess RS",
    "baseline1_daily_excess_z": "Baseline1 Daily Excess Z",
    "baseline1_tsl_gap": "Baseline1 TSL Gap",
    "baseline1_tsl_green": "Baseline1 TSL Green",
    "baseline1_tsl_red": "Baseline1 TSL Red",
    "baseline1_tsl_green_start": "Baseline1 TSL Green Start",
    "baseline1_tsl_red_start": "Baseline1 TSL Red Start",
    "baseline1_tsl_wave_speed": "Baseline1 TSL Wave Speed",
    "baseline1_tsl_wave_speed_delta_5d": "Baseline1 TSL Speed Delta (5D)",
    "baseline1_tsl_wave_age": "Baseline1 TSL Wave Age",
    "baseline1_benchmark_available": "Baseline1 Benchmark Available",
}


@dataclass(frozen=True)
class UniverseRow:
    symbol: str
    name: str = ""
    source: str = ""
    exchange: str = ""


@dataclass
class ScreeningResult:
    symbol: str
    name: str
    source: str
    exchange: str
    comprehensive_score: float | None
    trend_score: float | None = None
    turnaround_score: float | None = None
    trend_bonus: float | None = None
    turnaround_bonus: float | None = None
    signal_bonus_score: float | None = None
    financial_survival_score: float | None = None
    fragility_score: float | None = None
    route_score: float | None = None
    candidate_type: str = "Unclassified"
    factor_values: dict[str, float | None] = field(default_factory=dict)
    factor_scores: dict[str, float | None] = field(default_factory=dict)
    checklist: list[dict[str, Any]] = field(default_factory=list)
    king8: dict[str, Any] = field(default_factory=dict)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_env_file(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and not os.environ.get(key):
            os.environ[key] = value


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


class UniverseProvider:
    def build(self, config: dict[str, Any], offline_universe: Path | None = None) -> pd.DataFrame:
        if offline_universe:
            return self._from_csv(offline_universe)

        frames: list[pd.DataFrame] = []
        universes = config["universes"]
        if universes.get("include_sp500", True):
            frames.append(self.sp500())
        if universes.get("include_arkk", True):
            frames.append(self.arkk())
        if universes.get("include_kospi_top", 0):
            frames.append(self.krx_top("KOSPI", int(universes["include_kospi_top"])))
        if universes.get("include_kosdaq_top", 0):
            frames.append(self.krx_top("KOSDAQ", int(universes["include_kosdaq_top"])))

        if not frames:
            return pd.DataFrame(columns=["symbol", "name", "source", "exchange"])

        universe = pd.concat(frames, ignore_index=True)
        universe["symbol"] = universe["symbol"].astype(str).str.strip()
        universe = universe[universe["symbol"].ne("")]
        return (
            universe.drop_duplicates("symbol", keep="first")
            .reset_index(drop=True)
        )

    def _from_csv(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        required = {"symbol"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Offline universe missing columns: {sorted(missing)}")
        for col in ["name", "source", "exchange"]:
            if col not in df.columns:
                df[col] = ""
        return df[["symbol", "name", "source", "exchange"]]

    def sp500(self) -> pd.DataFrame:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(url, headers=HTTP_HEADERS, timeout=30)
        response.raise_for_status()
        from io import StringIO

        table = pd.read_html(StringIO(response.text))[0]
        return pd.DataFrame(
            {
                "symbol": table["Symbol"].astype(str).str.replace(".", "-", regex=False),
                "name": table["Security"],
                "source": "S&P 500",
                "exchange": "US",
            }
        )

    def arkk(self) -> pd.DataFrame:
        urls = [
            "https://ark-funds.com/wp-content/fundsiteliterature/csv/ARK_INNOVATION_ETF_ARKK_HOLDINGS.csv",
            "https://ark-funds.com/wp-content/fundsiteliterature/csv/ARKK_HOLDINGS.csv",
        ]
        urls.extend(self._discover_arkk_csv_urls())
        last_error: Exception | None = None
        for url in urls:
            try:
                response = requests.get(url, headers=HTTP_HEADERS, timeout=20)
                response.raise_for_status()
                from io import StringIO

                raw = pd.read_csv(StringIO(response.text))
                symbol_col = next((c for c in raw.columns if c.lower() in {"ticker", "symbol"}), None)
                name_col = next((c for c in raw.columns if "company" in c.lower() or "name" in c.lower()), None)
                if not symbol_col:
                    continue
                out = pd.DataFrame(
                    {
                        "symbol": raw[symbol_col].astype(str).str.strip(),
                        "name": raw[name_col] if name_col else "",
                        "source": "ARKK",
                        "exchange": "US",
                    }
                )
                return out[out["symbol"].ne("") & out["symbol"].ne("nan")]
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        fallback = self._arkk_from_stockanalysis()
        if not fallback.empty:
            return fallback
        raise RuntimeError(f"Could not download ARKK holdings: {last_error}")

    def _discover_arkk_csv_urls(self) -> list[str]:
        try:
            response = requests.get("https://www.ark-funds.com/funds/arkk", headers=HTTP_HEADERS, timeout=20)
            response.raise_for_status()
        except Exception:  # noqa: BLE001
            return []
        urls = []
        pattern = r"https?://[^\"']+?\.csv[^\"']*|href=[\"'](?P<href>[^\"']+?\.csv[^\"']*)"
        for match in re.finditer(pattern, response.text, flags=re.I):
            url = match.group("href") or match.group(0)
            if url.lower().startswith("href="):
                url = url.split("=", 1)[1].strip("\"'")
            if not url:
                continue
            if url.startswith("/"):
                url = f"https://www.ark-funds.com{url}"
            if "ARKK" in url.upper() or "INNOVATION" in url.upper():
                urls.append(url.replace("&amp;", "&"))
        return urls

    def _arkk_from_stockanalysis(self) -> pd.DataFrame:
        try:
            response = requests.get("https://stockanalysis.com/etf/arkk/holdings/", headers=HTTP_HEADERS, timeout=20)
            response.raise_for_status()
            from io import StringIO

            tables = pd.read_html(StringIO(response.text))
        except Exception:  # noqa: BLE001
            return pd.DataFrame(columns=["symbol", "name", "source", "exchange"])

        for table in tables:
            columns = {str(c).lower(): c for c in table.columns}
            symbol_col = columns.get("symbol")
            name_col = columns.get("name")
            if symbol_col is None or name_col is None:
                continue
            out = pd.DataFrame(
                {
                    "symbol": table[symbol_col].astype(str).str.strip(),
                    "name": table[name_col],
                    "source": "ARKK",
                    "exchange": "US",
                }
            )
            return out[out["symbol"].ne("") & out["symbol"].ne("nan") & out["symbol"].ne("n/a")]
        return pd.DataFrame(columns=["symbol", "name", "source", "exchange"])

    def krx_top(self, market: str, limit: int) -> pd.DataFrame:
        try:
            return self._krx_top_naver(market, limit)
        except Exception as exc:  # noqa: BLE001
            print(f"Naver Finance failed for {market}; falling back to pykrx: {exc}")
            return self._krx_top_pykrx(market, limit)

    def _krx_top_pykrx(self, market: str, limit: int) -> pd.DataFrame:
        from pykrx import stock

        today = datetime.now().strftime("%Y%m%d")
        date = self._last_krx_business_day(today)
        caps = stock.get_market_cap_by_ticker(date, market=market)
        if caps.empty or "시가총액" not in caps.columns:
            raise RuntimeError("empty market-cap response")
        names = {ticker: stock.get_market_ticker_name(ticker) for ticker in caps.index}
        top = caps.sort_values("시가총액", ascending=False).head(limit)
        return self._krx_rows(top.index, names, market, limit)

    def _krx_top_naver(self, market: str, limit: int) -> pd.DataFrame:
        from bs4 import BeautifulSoup

        sosok = "0" if market == "KOSPI" else "1"
        rows: list[tuple[str, str]] = []
        for page in range(1, int(math.ceil(limit / 50)) + 3):
            url = f"https://finance.naver.com/sise/sise_market_sum.naver?sosok={sosok}&page={page}"
            response = requests.get(url, headers=HTTP_HEADERS, timeout=20)
            response.encoding = "euc-kr"
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for link in soup.select("table.type_2 a.tltle"):
                href = link.get("href", "")
                match = re.search(r"code=(\d{6})", href)
                if not match:
                    continue
                name = link.get_text(strip=True)
                if self._looks_like_kr_etf(name):
                    continue
                rows.append((match.group(1), name))
                if len(rows) >= limit:
                    break
            if len(rows) >= limit:
                break
        if not rows:
            raise RuntimeError(f"Could not collect Naver market-cap rows for {market}")
        tickers = [ticker for ticker, _ in rows[:limit]]
        names = {ticker: name for ticker, name in rows[:limit]}
        return self._krx_rows(tickers, names, market, limit)

    def _krx_rows(self, tickers: Any, names: dict[str, str], market: str, limit: int) -> pd.DataFrame:
        suffix = ".KS" if market == "KOSPI" else ".KQ"
        return pd.DataFrame(
            {
                "symbol": [f"{ticker}{suffix}" for ticker in tickers],
                "name": [names.get(ticker, "") for ticker in tickers],
                "source": f"{market} Top {limit}",
                "exchange": market,
            }
        )

    def _looks_like_kr_etf(self, name: str) -> bool:
        upper = name.upper()
        if upper.startswith(KR_ETF_PREFIXES):
            return True
        return any(token in upper for token in (" ETF", " ETN", "인버스", "레버리지"))

    def _last_krx_business_day(self, date_yyyymmdd: str) -> str:
        from pykrx import stock

        date = datetime.strptime(date_yyyymmdd, "%Y%m%d")
        for offset in range(0, 10):
            candidate = (date - timedelta(days=offset)).strftime("%Y%m%d")
            try:
                tickers = stock.get_market_ticker_list(candidate, market="KOSPI")
            except Exception:  # noqa: BLE001
                continue
            if tickers:
                return candidate
        return date_yyyymmdd


class MarketDataProvider:
    def __init__(
        self,
        include_krx_flows: bool = True,
        include_fundamentals: bool = True,
        include_us_v11_data: bool = True,
        use_yfinance_info: bool = True,
        us_live_estimates: bool = True,
        benchmark_histories: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        self.include_krx_flows = include_krx_flows
        self.include_fundamentals = include_fundamentals
        self.include_us_v11_data = include_us_v11_data
        self.use_yfinance_info = use_yfinance_info
        self.us_live_estimates = us_live_estimates
        self.benchmark_histories = benchmark_histories or {}
        self._benchmark_cache: dict[str, pd.DataFrame] = {}
        self._krx_flows = None
        self._us_v11 = None

    def fetch(self, symbol: str, hist: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
        import yfinance as yf

        if hist is None:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2y", auto_adjust=True)
        info = {}
        if self.use_yfinance_info:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info or {}
            except Exception:  # noqa: BLE001
                info = {}
        if re.match(r"^\d{6}\.(KS|KQ)$", symbol):
            try:
                if self.include_fundamentals:
                    from data_providers import DartProvider

                    dart_info = DartProvider().metrics_for_symbol(symbol, market_cap=safe_float(info.get("marketCap")))
                    info.update({k: v for k, v in dart_info.items() if v is not None})
            except Exception as exc:  # noqa: BLE001
                info["dartError"] = str(exc)
            if self.include_krx_flows:
                try:
                    from data_providers import KrxInvestorFlowProvider

                    if self._krx_flows is None:
                        self._krx_flows = KrxInvestorFlowProvider()
                    as_of = self._last_history_date(hist)
                    if as_of is not None:
                        flow_info = self._krx_flows.metrics_for_symbol(
                            symbol,
                            as_of,
                            self._trading_value_sum(hist, 20),
                            self._trading_value_sum(hist, 60),
                        )
                        info.update({k: v for k, v in flow_info.items() if v is not None})
                        if self._krx_flows.disabled_reason:
                            info["krxFlowError"] = self._krx_flows.disabled_reason
                except Exception as exc:  # noqa: BLE001
                    info["krxFlowError"] = str(exc)
        elif self.include_us_v11_data:
            try:
                from data_providers import SECTOR_ETF_MAP, UsV11DataProvider, _history_return

                if self._us_v11 is None:
                    self._us_v11 = UsV11DataProvider()
                as_of = self._last_history_date(hist)
                if as_of is not None:
                    market_history = self.benchmark_histories.get("SPY")
                    qqq_history = self.benchmark_histories.get("QQQ")
                    us_info = self._us_v11.metrics_for_symbol(
                        symbol,
                        as_of,
                        stock_history=hist,
                        market_history=market_history,
                        qqq_history=qqq_history,
                        live_estimates=self.us_live_estimates,
                    )
                    sector_symbol = self._us_v11.sector_etf_for(us_info.get("sector") or info.get("sector"))
                    sector_hist = self.benchmark_histories.get(sector_symbol) if sector_symbol else None
                    if sector_hist is not None:
                        us_info.update(
                            {
                                "sector_etf_return_3m": _history_return(sector_hist, as_of, 63),
                                "sector_etf_return_6m": _history_return(sector_hist, as_of, 126),
                                "sector_etf_return_12m": _history_return(sector_hist, as_of, 252),
                            }
                        )
                    info.update({k: v for k, v in us_info.items() if v is not None})
            except Exception as exc:  # noqa: BLE001
                info["usV11DataError"] = str(exc)
        benchmark_symbol = baseline1_benchmark_symbol(symbol, exchange=info.get("exchange"), source=info.get("source"))
        info["baseline1_benchmark_symbol"] = benchmark_symbol
        benchmark_history = self._benchmark_history(benchmark_symbol)
        if benchmark_history is not None and not benchmark_history.empty:
            info["_baseline1_benchmark_history"] = benchmark_history
        return hist, info

    def _benchmark_history(self, symbol: str) -> pd.DataFrame | None:
        if symbol in self.benchmark_histories:
            return self.benchmark_histories[symbol]
        if symbol in self._benchmark_cache:
            return self._benchmark_cache[symbol]
        try:
            import yfinance as yf

            history = yf.Ticker(symbol).history(period="2y", auto_adjust=True)
            if history is not None and not history.empty:
                self._benchmark_cache[symbol] = history
                return history
        except Exception:  # noqa: BLE001
            return None
        return None

    def _last_history_date(self, hist: pd.DataFrame) -> pd.Timestamp | None:
        if hist.empty:
            return None
        ts = pd.Timestamp(hist.index.max())
        return ts.tz_convert(None) if ts.tzinfo is not None else ts

    def _trading_value_sum(self, hist: pd.DataFrame, sessions: int) -> float | None:
        if "Close" not in hist or "Volume" not in hist:
            return None
        frame = hist[["Close", "Volume"]].dropna().tail(sessions)
        if frame.empty:
            return None
        value = (frame["Close"].astype(float) * frame["Volume"].astype(float)).sum()
        return safe_float(value)


class FactorEngine:
    def calculate(self, hist: pd.DataFrame, info: dict[str, Any]) -> dict[str, float | None]:
        benchmark_symbol = str(info.get("baseline1_benchmark_symbol") or "")
        if info.get("_baseline1_three_factor_only") or info.get("_baseline1_rs_tsl_only"):
            baseline1 = baseline1_latest_features(
                hist,
                info.get("_baseline1_benchmark_history"),
                benchmark_symbol=benchmark_symbol or None,
            )
            out = {key: baseline1.get(key) for key in BASELINE1_RS_TSL_FEATURES}
            if baseline1.get("baseline1_benchmark_symbol"):
                out["baseline1_benchmark_symbol"] = baseline1.get("baseline1_benchmark_symbol")
            return out

        close = hist["Close"].dropna() if "Close" in hist else pd.Series(dtype=float)
        volume = hist["Volume"].dropna() if "Volume" in hist else pd.Series(dtype=float)

        current = safe_float(close.iloc[-1]) if len(close) else None
        three_month_return = self._return_over_sessions(close, 63)
        six_month_return = self._return_over_sessions(close, 126)
        twelve_month_return = self._return_over_sessions(close, 252)
        target_mean = safe_float(info.get("targetMeanPrice"))
        market_cap = safe_float(info.get("marketCap"))
        total_assets = safe_float(info.get("totalAssets"))
        rd_expense = safe_float(info.get("researchDevelopment"))
        roe = safe_float(info.get("returnOnEquity"))
        roa = safe_float(info.get("returnOnAssets"))
        debt_to_equity = safe_float(info.get("debtToEquity"))
        operating_margin = safe_float(info.get("operatingMargins"))
        profit_margin = safe_float(info.get("profitMargins"))
        revenue_growth = safe_float(info.get("revenueGrowth"))
        earnings_growth = safe_float(info.get("earningsGrowth"))
        gross_profits = safe_float(info.get("grossProfits"))
        free_cashflow = safe_float(info.get("freeCashflow"))
        total_cash = safe_float(info.get("totalCash"))
        dividend_yield = safe_float(info.get("dividendYield"))

        price_to_target = None
        upside = None
        if current and target_mean and target_mean > 0:
            price_to_target = current / target_mean
            upside = (target_mean / current - 1) * 100

        rd_to_market_cap = self._ratio_pct(rd_expense, market_cap)
        rd_to_total_assets = self._ratio_pct(rd_expense, total_assets)
        gross_income_to_assets = self._ratio_pct(gross_profits, total_assets)
        volume_price_cv_1m = self._volume_price_cv(close, volume)
        trend_line_cps = self._linear_slope(close.tail(126))
        high_52w = safe_float(close.tail(252).max()) if len(close) else None
        low_52w = safe_float(close.tail(252).min()) if len(close) else None
        high_proximity = current / high_52w if current and high_52w else None
        drawdown_from_52w_high = (high_52w / current - 1) * 100 if current and high_52w else None
        overheat = self._overheat_penalty(three_month_return, six_month_return, twelve_month_return, high_proximity)

        minervini = self._minervini_score(close)
        growth = self._growth_score(close, volume, info)
        balance = self._balance_score(info)
        obv = self._obv(close, volume)
        obv_slope_60d = self._linear_slope(obv.tail(60)) if len(obv) else None
        obv_high = safe_float(obv.tail(120).max()) if len(obv) else None
        obv_current = safe_float(obv.iloc[-1]) if len(obv) else None
        obv_high_proximity = obv_current / obv_high if obv_current is not None and obv_high not in (None, 0) else None
        mfi_14 = self._money_flow_index(hist, 14)
        dollar_volume_ratio_20_120 = self._dollar_volume_ratio(close, volume, 20, 120)
        relative_volume_20_60 = self._volume_ratio(volume, 20, 60)
        up_down_volume_ratio_20d = self._up_down_volume_ratio(close, volume, 20)
        accumulation_days_20d = self._accumulation_days(close, volume, 20)
        distribution_days_20d = self._distribution_days(close, volume, 20)
        divergence = self._divergence_scores(hist)
        ma50_reclaim_score = self._ma_reclaim_score(close, 50, 20)
        base_breakout_score = self._base_breakout_score(close, volume)
        baseline1 = baseline1_latest_features(
            hist,
            info.get("_baseline1_benchmark_history"),
            benchmark_symbol=benchmark_symbol or None,
        )

        factors = {
            "return_3m": three_month_return,
            "return_6m": six_month_return,
            "return_12m": twelve_month_return,
            "rd_to_market_cap": rd_to_market_cap,
            "rd_to_total_assets": rd_to_total_assets,
            "und_gro_score": growth,
            "price_to_target": price_to_target,
            "und_bal_score": balance,
            "minervini_qm_score": minervini,
            "volume_price_cv_1m": volume_price_cv_1m,
            "innovative_roe": roe * 100 if roe is not None else None,
            "roe": roe,
            "roa": roa,
            "debt_to_equity": debt_to_equity,
            "operating_margin": operating_margin,
            "profit_margin": profit_margin,
            "revenue_growth": revenue_growth,
            "earnings_growth": earnings_growth,
            "fcf_to_assets": free_cashflow / total_assets if free_cashflow is not None and total_assets else None,
            "cash_to_assets": total_cash / total_assets if total_cash is not None and total_assets else None,
            "upside_potential": upside,
            "gross_income_to_assets": gross_income_to_assets,
            "trend_line_cps": trend_line_cps,
            "high_proximity": high_proximity,
            "drawdown_from_52w_high": drawdown_from_52w_high,
            "overheat_penalty": overheat,
            "foreign_net_buy_strength_20d": safe_float(info.get("foreign_net_buy_strength_20d")),
            "foreign_net_buy_strength_60d": safe_float(info.get("foreign_net_buy_strength_60d")),
            "pension_net_buy_strength_20d": safe_float(info.get("pension_net_buy_strength_20d")),
            "pension_net_buy_strength_60d": safe_float(info.get("pension_net_buy_strength_60d")),
            "other_corp_net_buy_strength_20d": safe_float(info.get("other_corp_net_buy_strength_20d")),
            "foreign_flow_persistence_20d": safe_float(info.get("foreign_flow_persistence_20d")),
            "kr_smart_money_combo_20d": safe_float(info.get("kr_smart_money_combo_20d")),
            "kr_smart_money_accel": safe_float(info.get("kr_smart_money_accel")),
            "obv_slope_60d": obv_slope_60d,
            "obv_high_proximity": obv_high_proximity,
            "mfi_14": mfi_14,
            "dollar_volume_ratio_20_120": dollar_volume_ratio_20_120,
            "relative_volume_20_60": relative_volume_20_60,
            "up_down_volume_ratio_20d": up_down_volume_ratio_20d,
            "accumulation_days_20d": accumulation_days_20d,
            "distribution_days_20d": distribution_days_20d,
            "weekly_bullish_divergence_score": divergence.get("weekly_bullish_divergence_score"),
            "weekly_bearish_divergence_score": divergence.get("weekly_bearish_divergence_score"),
            "monthly_bullish_divergence_score": divergence.get("monthly_bullish_divergence_score"),
            "monthly_bearish_divergence_score": divergence.get("monthly_bearish_divergence_score"),
            "divergence_net_score": divergence.get("divergence_net_score"),
            "ma50_reclaim_score": ma50_reclaim_score,
            "base_breakout_score": base_breakout_score,
            "dividend_yield": dividend_yield * 100 if dividend_yield is not None else None,
            "estimate_timestamp": info.get("estimate_timestamp"),
            "financial_filing_date": info.get("financial_filing_date"),
            "fy1_eps_estimate_current": safe_float(info.get("fy1_eps_estimate_current")),
            "fy1_eps_estimate_1m_ago": safe_float(info.get("fy1_eps_estimate_1m_ago")),
            "fy1_eps_estimate_3m_ago": safe_float(info.get("fy1_eps_estimate_3m_ago")),
            "fy2_eps_estimate_current": safe_float(info.get("fy2_eps_estimate_current")),
            "fy2_eps_estimate_1m_ago": safe_float(info.get("fy2_eps_estimate_1m_ago")),
            "fy2_eps_estimate_3m_ago": safe_float(info.get("fy2_eps_estimate_3m_ago")),
            "revenue_estimate_current": safe_float(info.get("revenue_estimate_current")),
            "revenue_estimate_3m_ago": safe_float(info.get("revenue_estimate_3m_ago")),
            "num_upward_revisions_3m": safe_float(info.get("num_upward_revisions_3m")),
            "num_downward_revisions_3m": safe_float(info.get("num_downward_revisions_3m")),
            "latest_eps_actual": safe_float(info.get("latest_eps_actual")),
            "latest_eps_consensus": safe_float(info.get("latest_eps_consensus")),
            "latest_revenue_actual": safe_float(info.get("latest_revenue_actual")),
            "latest_revenue_consensus": safe_float(info.get("latest_revenue_consensus")),
            "earnings_day_return": safe_float(info.get("earnings_day_return")),
            "post_earnings_5d_return": safe_float(info.get("post_earnings_5d_return")),
            "post_earnings_20d_return": safe_float(info.get("post_earnings_20d_return")),
            "post_earnings_volume_ratio": safe_float(info.get("post_earnings_volume_ratio")),
            "sector_etf_return_3m": safe_float(info.get("sector_etf_return_3m")),
            "sector_etf_return_6m": safe_float(info.get("sector_etf_return_6m")),
            "sector_etf_return_12m": safe_float(info.get("sector_etf_return_12m")),
            "industry_median_return_3m": safe_float(info.get("industry_median_return_3m")),
            "industry_median_return_6m": safe_float(info.get("industry_median_return_6m")),
            "stock_return_3m": safe_float(info.get("stock_return_3m")),
            "stock_return_6m": safe_float(info.get("stock_return_6m")),
            "stock_return_12m": safe_float(info.get("stock_return_12m")),
            "market_return_3m": safe_float(info.get("market_return_3m")),
            "market_return_6m": safe_float(info.get("market_return_6m")),
            "market_return_12m": safe_float(info.get("market_return_12m")),
            "revenue_growth_yoy": safe_float(info.get("revenue_growth_yoy")),
            "eps_growth_yoy": safe_float(info.get("eps_growth_yoy")),
            "fcf_growth_yoy": safe_float(info.get("fcf_growth_yoy")),
            "gross_margin": safe_float(info.get("gross_margin")),
            "fcf_margin": safe_float(info.get("fcf_margin")),
            "roic": safe_float(info.get("roic")),
            "rule_of_40": safe_float(info.get("rule_of_40")),
            "forward_pe": safe_float(info.get("forward_pe")),
            "ev_sales": safe_float(info.get("ev_sales")),
            "ev_ebitda": safe_float(info.get("ev_ebitda")),
            "price_sales": safe_float(info.get("price_sales")),
            "fcf_yield": safe_float(info.get("fcf_yield")),
            "sales_growth_yoy": safe_float(info.get("sales_growth_yoy")),
            "ten_year_yield": safe_float(info.get("ten_year_yield")),
            "ten_year_yield_3m_change": safe_float(info.get("ten_year_yield_3m_change")),
            "real_yield_3m_change": safe_float(info.get("real_yield_3m_change")),
            "qqq_return_3m": safe_float(info.get("qqq_return_3m")),
            "spy_return_3m": safe_float(info.get("spy_return_3m")),
            "stock_beta_to_qqq": safe_float(info.get("stock_beta_to_qqq")),
            "stock_beta_to_spy": safe_float(info.get("stock_beta_to_spy")),
            "stock_corr_to_qqq": safe_float(info.get("stock_corr_to_qqq")),
            "us_v11_data_sources": info.get("us_v11_data_sources"),
        }
        factors.update(baseline1)
        return factors

    def _return_over_sessions(self, close: pd.Series, sessions: int) -> float | None:
        if len(close) <= sessions:
            return None
        start = safe_float(close.iloc[-sessions])
        end = safe_float(close.iloc[-1])
        if not start or not end:
            return None
        return (end / start - 1) * 100

    def _ratio_pct(self, numerator: float | None, denominator: float | None) -> float | None:
        if numerator is None or denominator is None or denominator == 0:
            return None
        return numerator / denominator * 100

    def _volume_price_cv(self, close: pd.Series, volume: pd.Series) -> float | None:
        if len(close) < 20 or len(volume) < 20:
            return None
        dollar_volume = close.tail(21).to_numpy() * volume.tail(21).to_numpy()
        mean = np.nanmean(dollar_volume)
        if mean == 0 or np.isnan(mean):
            return None
        return float(np.nanstd(dollar_volume) / mean)

    def _obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        if len(close) < 2 or len(volume) < 2:
            return pd.Series(dtype=float)
        aligned = pd.concat([close, volume], axis=1).dropna()
        if aligned.empty:
            return pd.Series(dtype=float)
        aligned.columns = ["close", "volume"]
        direction = np.sign(aligned["close"].diff()).fillna(0)
        return (direction * aligned["volume"]).cumsum()

    def _money_flow_index(self, hist: pd.DataFrame, period: int = 14) -> float | None:
        required = {"High", "Low", "Close", "Volume"}
        if not required.issubset(hist.columns) or len(hist) <= period:
            return None
        frame = hist[list(required)].dropna().copy()
        if len(frame) <= period:
            return None
        typical = (frame["High"] + frame["Low"] + frame["Close"]) / 3
        raw_flow = typical * frame["Volume"]
        direction = typical.diff()
        positive = raw_flow.where(direction > 0, 0.0).rolling(period).sum()
        negative = raw_flow.where(direction < 0, 0.0).rolling(period).sum().abs()
        denom = positive + negative
        mfi = 100 * positive / denom.replace(0, np.nan)
        return safe_float(mfi.iloc[-1])

    def _dollar_volume_ratio(self, close: pd.Series, volume: pd.Series, short: int, long: int) -> float | None:
        if len(close) < long or len(volume) < long:
            return None
        dollar_volume = (close * volume).dropna()
        if len(dollar_volume) < long:
            return None
        long_avg = safe_float(dollar_volume.tail(long).mean())
        short_avg = safe_float(dollar_volume.tail(short).mean())
        if not long_avg:
            return None
        return short_avg / long_avg

    def _volume_ratio(self, volume: pd.Series, short: int, long: int) -> float | None:
        clean = volume.dropna()
        if len(clean) < long:
            return None
        long_avg = safe_float(clean.tail(long).mean())
        short_avg = safe_float(clean.tail(short).mean())
        if not long_avg:
            return None
        return short_avg / long_avg

    def _up_down_volume_ratio(self, close: pd.Series, volume: pd.Series, sessions: int) -> float | None:
        if len(close) < sessions + 1 or len(volume) < sessions + 1:
            return None
        frame = pd.concat([close, volume], axis=1).dropna().tail(sessions + 1)
        frame.columns = ["close", "volume"]
        changes = frame["close"].pct_change()
        up_volume = frame.loc[changes > 0, "volume"].sum()
        down_volume = frame.loc[changes < 0, "volume"].sum()
        if down_volume == 0:
            return None
        return float(up_volume / down_volume)

    def _accumulation_days(self, close: pd.Series, volume: pd.Series, sessions: int) -> float | None:
        if len(close) < sessions + 1 or len(volume) < sessions + 1:
            return None
        frame = pd.concat([close, volume], axis=1).dropna().tail(sessions + 1)
        frame.columns = ["close", "volume"]
        changes = frame["close"].pct_change()
        vol_avg = frame["volume"].rolling(10).mean()
        return float(((changes > 0.015) & (frame["volume"] > vol_avg)).sum())

    def _distribution_days(self, close: pd.Series, volume: pd.Series, sessions: int) -> float | None:
        if len(close) < sessions + 1 or len(volume) < sessions + 1:
            return None
        frame = pd.concat([close, volume], axis=1).dropna().tail(sessions + 1)
        frame.columns = ["close", "volume"]
        changes = frame["close"].pct_change()
        vol_avg = frame["volume"].rolling(10).mean()
        return float(((changes < -0.015) & (frame["volume"] > vol_avg)).sum())

    def _linear_slope(self, series: pd.Series) -> float | None:
        clean = series.dropna()
        if len(clean) < 20:
            return None
        y = clean.to_numpy(dtype=float)
        x = np.arange(len(y), dtype=float)
        slope = np.polyfit(x, y, 1)[0]
        base = np.nanmean(y)
        if base == 0 or np.isnan(base):
            return None
        return float(slope / base * 100)

    def _overheat_penalty(
        self,
        return_3m: float | None,
        return_6m: float | None,
        return_12m: float | None,
        high_proximity: float | None,
    ) -> float:
        penalty = 0.0
        if return_12m is not None and return_12m > 120:
            penalty += min(12.0, (return_12m - 120) / 15)
        if return_3m is not None and return_3m > 45:
            penalty += min(10.0, (return_3m - 45) / 7)
        if return_3m is not None and high_proximity is not None and high_proximity >= 0.98 and return_3m > 25:
            penalty += 5.0
        if return_3m is not None and return_6m is not None and return_6m > 0 and return_3m > 30 and (return_3m / return_6m) > 0.75:
            penalty += 4.0
        return round(min(25.0, penalty), 4)

    def _minervini_score(self, close: pd.Series) -> float | None:
        if len(close) < 200:
            return None
        score = 0
        ma50 = close.rolling(50).mean()
        ma150 = close.rolling(150).mean()
        ma200 = close.rolling(200).mean()
        price = close.iloc[-1]
        low_52w = close.tail(252).min()
        high_52w = close.tail(252).max()
        checks = [
            price > ma150.iloc[-1] and price > ma200.iloc[-1],
            ma150.iloc[-1] > ma200.iloc[-1],
            ma200.iloc[-1] > ma200.iloc[-22],
            ma50.iloc[-1] > ma150.iloc[-1] and ma50.iloc[-1] > ma200.iloc[-1],
            price > ma50.iloc[-1],
            price >= low_52w * 1.3,
            price >= high_52w * 0.75,
        ]
        score += sum(bool(x) for x in checks)
        return float(score / len(checks) * 17)

    def _divergence_scores(self, hist: pd.DataFrame) -> dict[str, float | None]:
        required = {"Open", "High", "Low", "Close", "Volume"}
        if not required.issubset(hist.columns) or len(hist) < 80:
            return {
                "weekly_bullish_divergence_score": None,
                "weekly_bearish_divergence_score": None,
                "monthly_bullish_divergence_score": None,
                "monthly_bearish_divergence_score": None,
                "divergence_net_score": None,
            }
        weekly = self._resample_ohlcv(hist, "W-FRI")
        monthly = self._resample_ohlcv(hist, "ME")
        weekly_bull, weekly_bear = self._divergence_score_for_frame(weekly, pivot_order=2, recent_bars=5)
        monthly_bull, monthly_bear = self._divergence_score_for_frame(monthly, pivot_order=1, recent_bars=4)
        return {
            "weekly_bullish_divergence_score": weekly_bull,
            "weekly_bearish_divergence_score": weekly_bear,
            "monthly_bullish_divergence_score": monthly_bull,
            "monthly_bearish_divergence_score": monthly_bear,
            "divergence_net_score": (weekly_bull or 0) + (monthly_bull or 0) - (weekly_bear or 0) - (monthly_bear or 0),
        }

    def _resample_ohlcv(self, hist: pd.DataFrame, rule: str) -> pd.DataFrame:
        frame = hist[["Open", "High", "Low", "Close", "Volume"]].dropna()
        out = frame.resample(rule).agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        return out.dropna()

    def _divergence_score_for_frame(self, frame: pd.DataFrame, pivot_order: int, recent_bars: int) -> tuple[float | None, float | None]:
        if len(frame) < max(12, pivot_order * 4 + recent_bars):
            return None, None
        close = frame["Close"]
        indicators = {
            "rsi": self._rsi_series(close, 14),
            "macd_hist": self._macd_hist_series(close),
            "obv": self._obv(close, frame["Volume"]),
            "mfi": self._mfi_series(frame, 14),
            "cmf": self._cmf_series(frame, 21),
        }
        bull = 0.0
        bear = 0.0
        for indicator in indicators.values():
            clean = indicator.reindex(frame.index).dropna()
            if len(clean) < 8:
                continue
            aligned = frame.loc[clean.index]
            bull_reg, bull_hidden, bear_reg, bear_hidden = self._detect_divergence(
                aligned["Low"],
                aligned["High"],
                clean,
                pivot_order,
                recent_bars,
            )
            bull += 25 if bull_reg else 0
            bull += 15 if bull_hidden else 0
            bear += 25 if bear_reg else 0
            bear += 15 if bear_hidden else 0
        return min(100.0, bull), min(100.0, bear)

    def _detect_divergence(
        self,
        lows: pd.Series,
        highs: pd.Series,
        indicator: pd.Series,
        pivot_order: int,
        recent_bars: int,
    ) -> tuple[bool, bool, bool, bool]:
        low_positions = self._pivot_positions(lows, pivot_order, want_high=False)
        high_positions = self._pivot_positions(highs, pivot_order, want_high=True)
        bull_regular = False
        bull_hidden = False
        bear_regular = False
        bear_hidden = False

        low_pair = self._recent_pivot_pair(low_positions, len(lows), recent_bars + pivot_order)
        if low_pair:
            prev, last = low_pair
            price_prev = safe_float(lows.iloc[prev])
            price_last = safe_float(lows.iloc[last])
            ind_prev = safe_float(indicator.iloc[prev])
            ind_last = safe_float(indicator.iloc[last])
            if None not in (price_prev, price_last, ind_prev, ind_last):
                bull_regular = bool(price_last < price_prev and ind_last > ind_prev)
                bull_hidden = bool(price_last > price_prev and ind_last < ind_prev)

        high_pair = self._recent_pivot_pair(high_positions, len(highs), recent_bars + pivot_order)
        if high_pair:
            prev, last = high_pair
            price_prev = safe_float(highs.iloc[prev])
            price_last = safe_float(highs.iloc[last])
            ind_prev = safe_float(indicator.iloc[prev])
            ind_last = safe_float(indicator.iloc[last])
            if None not in (price_prev, price_last, ind_prev, ind_last):
                bear_regular = bool(price_last > price_prev and ind_last < ind_prev)
                bear_hidden = bool(price_last < price_prev and ind_last > ind_prev)
        return bull_regular, bull_hidden, bear_regular, bear_hidden

    def _pivot_positions(self, series: pd.Series, order: int, want_high: bool) -> list[int]:
        values = series.to_numpy(dtype=float)
        positions: list[int] = []
        for idx in range(order, len(values) - order):
            window = values[idx - order : idx + order + 1]
            if np.isnan(window).any():
                continue
            if want_high and values[idx] == np.max(window):
                positions.append(idx)
            if not want_high and values[idx] == np.min(window):
                positions.append(idx)
        return positions

    def _recent_pivot_pair(self, positions: list[int], length: int, recent_bars: int) -> tuple[int, int] | None:
        recent = [pos for pos in positions if length - 1 - pos <= recent_bars]
        if not recent:
            return None
        last = recent[-1]
        prev_candidates = [pos for pos in positions if pos < last]
        if not prev_candidates:
            return None
        return prev_candidates[-1], last

    def _rsi_series(self, close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def _macd_hist_series(self, close: pd.Series) -> pd.Series:
        fast = close.ewm(span=12, adjust=False).mean()
        slow = close.ewm(span=26, adjust=False).mean()
        macd = fast - slow
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd - signal

    def _mfi_series(self, frame: pd.DataFrame, period: int) -> pd.Series:
        typical = (frame["High"] + frame["Low"] + frame["Close"]) / 3
        raw_flow = typical * frame["Volume"]
        direction = typical.diff()
        positive = raw_flow.where(direction > 0, 0.0).rolling(period).sum()
        negative = raw_flow.where(direction < 0, 0.0).rolling(period).sum().abs()
        return 100 * positive / (positive + negative).replace(0, np.nan)

    def _cmf_series(self, frame: pd.DataFrame, period: int) -> pd.Series:
        spread = (frame["High"] - frame["Low"]).replace(0, np.nan)
        multiplier = ((frame["Close"] - frame["Low"]) - (frame["High"] - frame["Close"])) / spread
        flow_volume = multiplier * frame["Volume"]
        return flow_volume.rolling(period).sum() / frame["Volume"].rolling(period).sum().replace(0, np.nan)

    def _ma_reclaim_score(self, close: pd.Series, window: int, lookback: int) -> float | None:
        if len(close) < window + lookback:
            return None
        ma = close.rolling(window).mean()
        current = safe_float(close.iloc[-1])
        current_ma = safe_float(ma.iloc[-1])
        prior_below = bool((close.tail(lookback) < ma.tail(lookback)).any())
        if current is None or current_ma is None:
            return None
        if current > current_ma and prior_below:
            return 100.0
        if current > current_ma:
            return 60.0
        return 0.0

    def _base_breakout_score(self, close: pd.Series, volume: pd.Series) -> float | None:
        if len(close) < 80 or len(volume) < 80:
            return None
        prior_high = safe_float(close.shift(1).tail(60).max())
        current = safe_float(close.iloc[-1])
        vol_ratio = self._volume_ratio(volume, 20, 60)
        if current is None or prior_high is None:
            return None
        if current > prior_high and vol_ratio is not None and vol_ratio > 1.2:
            return 100.0
        if current > prior_high:
            return 75.0
        if current > prior_high * 0.95:
            return 50.0
        return 0.0

    def _growth_score(self, close: pd.Series, volume: pd.Series, info: dict[str, Any]) -> float | None:
        score = 0
        total = 0
        for condition in [
            self._return_over_sessions(close, 63) is not None and self._return_over_sessions(close, 63) > 0,
            self._return_over_sessions(close, 126) is not None and self._return_over_sessions(close, 126) > 0,
            self._return_over_sessions(close, 252) is not None and self._return_over_sessions(close, 252) > 0,
            safe_float(info.get("revenueGrowth")) is not None and safe_float(info.get("revenueGrowth")) > 0,
            safe_float(info.get("earningsGrowth")) is not None and safe_float(info.get("earningsGrowth")) > 0,
        ]:
            total += 1
            score += int(bool(condition))
        return float(score / total * 11) if total else None

    def _balance_score(self, info: dict[str, Any]) -> float | None:
        score = 0
        total = 0
        checks = [
            ("debtToEquity", lambda x: x < 150),
            ("currentRatio", lambda x: x > 1),
            ("quickRatio", lambda x: x > 0.7),
            ("profitMargins", lambda x: x > 0),
            ("operatingMargins", lambda x: x > 0),
            ("returnOnAssets", lambda x: x > 0),
            ("returnOnEquity", lambda x: x > 0),
            ("freeCashflow", lambda x: x > 0),
        ]
        for key, predicate in checks:
            value = safe_float(info.get(key))
            if value is None:
                continue
            total += 1
            score += int(predicate(value))
        return float(score / total * 16) if total else None


class ScoreEngine:
    TREND_BONUS_WEIGHTS = {
        "baseline1_entry_signal": 4.0,
        "baseline1_tsl_green_start": 2.0,
    }
    TURNAROUND_BONUS_WEIGHTS = {
        "baseline1_entry_signal": 5.0,
        "baseline1_rs_zero_cross_up": 4.0,
        "baseline1_tsl_green_start": 3.0,
    }

    def normalize(self, rows: list[ScreeningResult], config: dict[str, Any]) -> None:
        scoring = config["scoring"]
        weights = scoring["weights"]
        higher_is_better = scoring["higher_is_better"]
        profile_models = config.get("profile_models", {})
        simple_baseline = bool(config.get("model_notes", {}).get("baseline1_three_factor_only"))
        profile_factor_names: set[str] = set()
        for profile_weights in profile_models.values():
            profile_factor_names.update(profile_weights.keys())
        factor_sources = [*weights.keys()] if simple_baseline else [
            *weights.keys(),
            *profile_factor_names,
            *self.TREND_BONUS_WEIGHTS.keys(),
            *self.TURNAROUND_BONUS_WEIGHTS.keys(),
        ]
        all_factor_names = list(dict.fromkeys(factor_sources))

        factor_frame = pd.DataFrame([r.factor_values for r in rows], index=[r.symbol for r in rows])
        score_frame = pd.DataFrame(index=factor_frame.index)

        for factor in all_factor_names:
            if factor in factor_frame.columns:
                values = pd.to_numeric(factor_frame[factor], errors="coerce")
            else:
                values = pd.Series(np.nan, index=factor_frame.index)
            if values.notna().sum() < 2:
                score_frame[factor] = np.nan
                continue
            ranks = values.rank(pct=True, method="average") * 100
            if not higher_is_better.get(factor, True):
                ranks = 101 - ranks
            score_frame[factor] = ranks.clip(0, 100)

        for result in rows:
            result.factor_scores = {
                factor: safe_float(score_frame.loc[result.symbol, factor])
                for factor in all_factor_names
                if factor in score_frame.columns
            }
            base_score = self._weighted_average(result.factor_scores, weights)
            if simple_baseline:
                result.comprehensive_score = base_score
                result.trend_score = None
                result.turnaround_score = None
                result.trend_bonus = None
                result.turnaround_bonus = None
                result.signal_bonus_score = None
                result.financial_survival_score = None
                result.fragility_score = None
                result.route_score = None
                result.candidate_type = ""
                continue
            trend_base = self._weighted_average(result.factor_scores, profile_models.get("trend_weights", {}))
            turnaround_base = self._weighted_average(result.factor_scores, profile_models.get("turnaround_weights", {}))
            result.trend_bonus = self._positive_bonus(result.factor_values, result.factor_scores, self.TREND_BONUS_WEIGHTS)
            result.turnaround_bonus = self._positive_bonus(result.factor_values, result.factor_scores, self.TURNAROUND_BONUS_WEIGHTS)
            result.signal_bonus_score = round((result.trend_bonus or 0) + (result.turnaround_bonus or 0), 2)
            result.trend_score = self._cap_score(trend_base, result.trend_bonus)
            result.turnaround_score = self._cap_score(turnaround_base, result.turnaround_bonus)
            result.financial_survival_score = self._weighted_average(result.factor_scores, profile_models.get("survival_weights", {}))
            risk_control_score = self._weighted_average(result.factor_scores, profile_models.get("risk_control_weights", {}))
            result.fragility_score = round(100 - risk_control_score, 2) if risk_control_score is not None else None
            result.route_score = self._route_score(result.trend_score, result.turnaround_score, result.financial_survival_score, result.fragility_score)
            result.candidate_type = self._candidate_type(result.trend_score, result.turnaround_score, result.financial_survival_score, result.fragility_score)
            result.comprehensive_score = result.route_score if result.route_score is not None else base_score

    def _weighted_average(self, scores: dict[str, float | None], weights: dict[str, float]) -> float | None:
        numerator = 0.0
        denominator = 0.0
        for factor, weight in weights.items():
            score = scores.get(factor)
            if score is None:
                continue
            numerator += score * weight
            denominator += weight
        if denominator == 0:
            return None
        return round(numerator / denominator, 2)

    def _positive_bonus(
        self,
        raw_values: dict[str, float | None],
        scores: dict[str, float | None],
        weights: dict[str, float],
    ) -> float:
        bonus = 0.0
        for factor, max_points in weights.items():
            raw = safe_float(raw_values.get(factor))
            score = safe_float(scores.get(factor))
            if raw is None or raw <= 0 or score is None:
                continue
            bonus += max_points * (score / 100)
        return round(bonus, 2)

    def _cap_score(self, base_score: float | None, bonus: float | None) -> float | None:
        if base_score is None:
            return None
        return round(max(0.0, min(100.0, base_score + (bonus or 0))), 2)

    def _route_score(
        self,
        trend_score: float | None,
        turnaround_score: float | None,
        survival_score: float | None,
        fragility_score: float | None,
    ) -> float | None:
        candidates = [score for score in (trend_score, turnaround_score) if score is not None]
        if not candidates:
            return None
        score = max(candidates)
        if survival_score is not None and survival_score < 35:
            score -= 20
        if fragility_score is not None and fragility_score > 75:
            score -= 15
        return round(max(0.0, min(100.0, score)), 2)

    def _candidate_type(
        self,
        trend_score: float | None,
        turnaround_score: float | None,
        survival_score: float | None,
        fragility_score: float | None,
    ) -> str:
        if survival_score is not None and survival_score < 30:
            return "Avoid - weak survival"
        if fragility_score is not None and fragility_score > 80:
            return "Avoid - fragile"
        trend = trend_score or 0
        turnaround = turnaround_score or 0
        if trend >= 80 and turnaround >= 70:
            return "Dual Signal"
        if trend >= 75:
            return "Trend Leader"
        if turnaround >= 70 and (survival_score is None or survival_score >= 50):
            return "Turnaround Candidate"
        return "Watch"


class ChecklistEngine:
    def build(self, hist: pd.DataFrame, info: dict[str, Any], factors: dict[str, float | None], config: dict[str, Any]) -> list[dict[str, Any]]:
        close = hist["Close"].dropna() if "Close" in hist else pd.Series(dtype=float)
        volume = hist["Volume"].dropna() if "Volume" in hist else pd.Series(dtype=float)
        thresholds = config["checklist_thresholds"]
        checklist: list[dict[str, Any]] = []

        def add(label: str, passed: bool | None, value: Any = None) -> None:
            checklist.append({"label": label, "passed": passed, "value": value})

        current = safe_float(close.iloc[-1]) if len(close) else None
        ma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else np.nan
        ma150 = close.rolling(150).mean().iloc[-1] if len(close) >= 150 else np.nan
        ma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan
        high_52w = close.tail(252).max() if len(close) >= 20 else np.nan
        low_52w = close.tail(252).min() if len(close) >= 20 else np.nan

        add("K-스코어 값(모멘텀 요소)은 0에서 100 사이입니다.", self._between(factors.get("return_6m"), -100, 500), factors.get("return_6m"))
        add("연구개발비/총자산 비율은 0%에서 50% 사이입니다.", self._between(factors.get("rd_to_total_assets"), 0, thresholds["max_rd_to_market_cap_pct"]), factors.get("rd_to_total_assets"))
        target = safe_float(info.get("targetMeanPrice"))
        add("현재 가격이 목표 가격보다 낮습니다.", current is not None and target is not None and current < target, target)
        add("역대 최고치에 근접해 있습니다.", current is not None and not np.isnan(high_52w) and current >= high_52w * (1 - thresholds["near_ath_drawdown_pct"] / 100), high_52w)
        add("주가는 150일 이동평균선과 200일 이동평균선 위에 있습니다.", self._above(current, ma150) and self._above(current, ma200), current)
        add("150일 이동평균선이 200일 이동평균선 위에 있을 것입니다.", self._above(ma150, ma200), ma150)
        add("200일 이동평균선이 상승 추세에 있습니다.", len(close) >= 222 and close.rolling(200).mean().iloc[-1] > close.rolling(200).mean().iloc[-22], None)
        add("50일 이동평균선이 150일 및 200일 이동평균선 위에 있습니다.", self._above(ma50, ma150) and self._above(ma50, ma200), ma50)
        add("주가가 50일 이동평균선 위에 있습니다.", self._above(current, ma50), current)
        add("가격은 52주 최저가보다 30% 이상 높을 것입니다.", current is not None and not np.isnan(low_52w) and current >= low_52w * (1 + thresholds["min_above_52w_low_pct"] / 100), low_52w)
        add("주가는 52주 최고가 대비 25% 이상 하락하지 않았습니다.", current is not None and not np.isnan(high_52w) and current >= high_52w * (1 - thresholds["max_52w_high_drawdown_pct"] / 100), high_52w)
        add("가격이 상승하는 날에는 거래량이 많습니다.", self._up_days_have_volume(close, volume), None)
        rs_dist = safe_float(factors.get("baseline1_rs_dist_pct"))
        add("Baseline1 RS is above its rolling average versus the configured benchmark.", rs_dist is not None and rs_dist >= 0, rs_dist)
        add("실적 발표가 예상을 뛰어넘으면서 주가가 상승했습니다.", safe_float(info.get("earningsQuarterlyGrowth")) is not None and safe_float(info.get("earningsQuarterlyGrowth")) > 0, info.get("earningsQuarterlyGrowth"))
        survival_pass, survival_reason = self._financial_survival(info)
        add("재무 생존 필터를 통과했습니다.", survival_pass, survival_reason)
        return checklist

    def _between(self, value: float | None, low: float, high: float) -> bool | None:
        if value is None:
            return None
        return low <= value <= high

    def _above(self, left: Any, right: Any) -> bool:
        left_value = safe_float(left)
        right_value = safe_float(right)
        return left_value is not None and right_value is not None and left_value > right_value

    def _up_days_have_volume(self, close: pd.Series, volume: pd.Series) -> bool | None:
        if len(close) < 25 or len(volume) < 25:
            return None
        changes = close.tail(25).pct_change()
        vol = volume.tail(25)
        up_volume = vol[changes > 0].mean()
        down_volume = vol[changes < 0].mean()
        if np.isnan(up_volume) or np.isnan(down_volume):
            return None
        return bool(up_volume > down_volume)

    def _financial_survival(self, info: dict[str, Any]) -> tuple[bool | None, str]:
        keys = ["debtToEquity", "operatingMargins", "profitMargins", "returnOnAssets", "freeCashflow", "totalAssets", "totalCash"]
        if all(safe_float(info.get(key)) is None for key in keys):
            return None, "no_fundamental_data"
        reasons: list[str] = []
        debt_to_equity = safe_float(info.get("debtToEquity"))
        operating_margin = safe_float(info.get("operatingMargins"))
        profit_margin = safe_float(info.get("profitMargins"))
        roa = safe_float(info.get("returnOnAssets"))
        fcf = safe_float(info.get("freeCashflow"))
        assets = safe_float(info.get("totalAssets"))
        cash = safe_float(info.get("totalCash"))
        fcf_to_assets = fcf / assets if fcf is not None and assets else None
        cash_to_assets = cash / assets if cash is not None and assets else None
        if debt_to_equity is not None and debt_to_equity > 300:
            reasons.append("debt_to_equity_gt_300")
        if operating_margin is not None and operating_margin < -0.15:
            reasons.append("operating_margin_lt_-15pct")
        if profit_margin is not None and profit_margin < -0.25:
            reasons.append("profit_margin_lt_-25pct")
        if roa is not None and roa < -0.15:
            reasons.append("roa_lt_-15pct")
        if cash_to_assets is not None and fcf_to_assets is not None and cash_to_assets < 0.01 and fcf_to_assets < -0.05:
            reasons.append("cash_low_and_fcf_negative")
        return not reasons, "pass" if not reasons else ";".join(reasons)


class King8Engine:
    def build(
        self,
        hist: pd.DataFrame,
        info: dict[str, Any],
        factors: dict[str, float | None],
        factor_scores: dict[str, float | None],
        checklist: list[dict[str, Any]],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        close = hist["Close"].dropna() if "Close" in hist else pd.Series(dtype=float)
        current = safe_float(close.iloc[-1]) if len(close) else None
        high_52w = safe_float(close.tail(252).max()) if len(close) else None
        market_cap = safe_float(info.get("marketCap"))
        total_cash = safe_float(info.get("totalCash"))
        total_debt = safe_float(info.get("totalDebt"))
        revenue_growth = safe_float(info.get("revenueGrowth"))
        earnings_growth = safe_float(info.get("earningsGrowth"))
        gross_margin = safe_float(info.get("grossMargins"))
        operating_margin = safe_float(info.get("operatingMargins"))
        profit_margin = safe_float(info.get("profitMargins"))
        roe = safe_float(info.get("returnOnEquity"))
        roa = safe_float(info.get("returnOnAssets"))
        debt_to_equity = safe_float(info.get("debtToEquity"))
        beta = safe_float(info.get("beta"))

        king_cfg = config.get("king8", {})
        thresholds = king_cfg.get("quant_thresholds", {})
        quant_items = self._quantitative_items(
            factors=factors,
            current=current,
            high_52w=high_52w,
            market_cap=market_cap,
            total_cash=total_cash,
            total_debt=total_debt,
            revenue_growth=revenue_growth,
            earnings_growth=earnings_growth,
            thresholds=thresholds,
        )

        stages = [
            self._stage(
                "0_data_sync",
                "데이터 동기화 및 2026 실시간 로딩",
                "실시간/최신성",
                [
                    self._item("가격 데이터 로딩", len(close) > 0, f"{len(close)} sessions"),
                    self._item("펀더멘털 데이터 로딩", bool(info), "yfinance/info 또는 대체 데이터"),
                    self._item("업계 리포트/뉴스 교차검증", None, "무료 API만으로는 별도 리서치 필요"),
                ],
            ),
            self._stage(
                "1_survival_mda",
                "생존 필터 & 밸류체인의 해부",
                "톨게이트 기업인지 단순 노동자인지 판별",
                [
                    self._item("경기침체 결제 지속성", None, "산업/제품 정성 판정 필요"),
                    self._item("기술 변화로 증발할 위험", None, "사업보고서와 산업 리포트 확인 필요"),
                    self._item("밸류체인 톨게이트성", None, "Upstream/Midstream/Downstream 리서치 필요"),
                ],
            ),
            self._stage(
                "2_moat_stack",
                "중첩 해자 & 독점의 기술",
                "전환비용, 규모, 규제, 브랜드, 가격결정력",
                [
                    self._item("가격결정력 프록시", gross_margin is not None and gross_margin > 0.35, gross_margin),
                    self._item("수익성 기반 해자 프록시", operating_margin is not None and operating_margin > 0.15, operating_margin),
                    self._item("독점/비대칭 우위", None, "점유율 데이터 소스 연결 필요"),
                ],
            ),
            self._stage(
                "3_fcf_unit_economics",
                "현금흐름 절대주의 & 단위당 경제성",
                "현금이 왕이고 단위당 경제성이 본질",
                [
                    self._item("FCF 양수", safe_float(info.get("freeCashflow")) is not None and safe_float(info.get("freeCashflow")) > 0, info.get("freeCashflow")),
                    self._item("Gross Margin", gross_margin is not None and gross_margin > 0, gross_margin),
                    self._item("R&D 효율성", factors.get("rd_to_market_cap") is not None, factors.get("rd_to_market_cap")),
                ],
            ),
            self._stage(
                "4_management_smart_money",
                "경영진의 노마드 정신 & 스마트 머니",
                "자본 배분과 13F 추적",
                [
                    self._item("자사주/소각 데이터", None, "SEC/DART 원문 또는 paid fundamentals 연결 필요"),
                    self._item("기관 13F 추적", None, "SEC 13F 원문 파서 추가 필요"),
                    self._item("장기 복리형 경영진", None, "IR/주주서한/콜 transcript 정성 분석 필요"),
                ],
            ),
            self._stage(
                "5_time_advantage",
                "시간 우위 & 거장의 철학적 오버레이",
                "10년 뒤에도 강해질 구조인지",
                [
                    self._item("6개월 가격 추세", factors.get("return_6m") is not None and factors["return_6m"] > 0, factors.get("return_6m")),
                    self._item("매출 성장", revenue_growth is not None and revenue_growth > 0, revenue_growth),
                    self._item("AI/인구/국가전략 수혜", None, "산업 분류와 뉴스/리포트 RAG 연결 필요"),
                ],
            ),
            self._stage(
                "6_dupont_quant",
                "재무적 맷집과 듀퐁/정량 필터",
                "이익률, 회전율, 레버리지로 ROE를 분해",
                [
                    self._item("ROE 양수", roe is not None and roe > 0, roe),
                    self._item("ROA 양수", roa is not None and roa > 0, roa),
                    self._item("레버리지 과열 아님", debt_to_equity is not None and debt_to_equity < 150, debt_to_equity),
                    self._item("순이익률 양수", profit_margin is not None and profit_margin > 0, profit_margin),
                ]
                + quant_items,
            ),
            self._stage(
                "7_technical_expectation",
                "모멘텀과 기대감",
                "신고가 근접, 목표가, 시장 기대",
                [
                    self._item("신고가 85% 이상", current is not None and high_52w is not None and current >= high_52w * thresholds.get("min_high_proximity", 0.85), {"price": current, "high_52w": high_52w}),
                    self._item("현재가 > 애널리스트 평균 목표가", self._target_expectation(current, info), {"price": current, "target": info.get("targetMeanPrice")}),
                    self._item("위험조정 수익 RAR", self._rar_ok(factors.get("return_6m"), beta, thresholds.get("max_rar", 50)), {"return_6m": factors.get("return_6m"), "beta": beta}),
                ],
            ),
        ]

        king_score = self._weighted_stage_score(stages, king_cfg.get("stage_weights", {}))
        verdict = self._verdict(king_score, stages, king_cfg.get("minimum_gate_score", 60))
        return {
            "king_score": king_score,
            "verdict": verdict,
            "stages": stages,
            "style_note": "헤지펀드 매니저처럼 차갑게 숫자를 보고, 동네 형처럼 이해되는 말로 판정한다.",
        }

    def _quantitative_items(
        self,
        factors: dict[str, float | None],
        current: float | None,
        high_52w: float | None,
        market_cap: float | None,
        total_cash: float | None,
        total_debt: float | None,
        revenue_growth: float | None,
        earnings_growth: float | None,
        thresholds: dict[str, Any],
    ) -> list[dict[str, Any]]:
        rd_to_market_cap = factors.get("rd_to_market_cap")
        prr = 100 / rd_to_market_cap if rd_to_market_cap and rd_to_market_cap > 0 else None
        net_cash_to_market_cap = None
        if market_cap and total_cash is not None:
            net_cash_to_market_cap = ((total_cash - (total_debt or 0)) / market_cap) * 100
        return [
            self._item("수익성: 최근 수익성 프록시 양호", factors.get("innovative_roe") is not None and factors["innovative_roe"] > 0, factors.get("innovative_roe")),
            self._item("R&D 가치: PRR <= 15", prr is not None and prr <= thresholds.get("max_prr", 15), prr),
            self._item("저평가: PSR < 산업 평균", None, "산업 평균 PSR 데이터 연결 필요"),
            self._item("성장 가속", revenue_growth is not None and revenue_growth > 0, revenue_growth),
            self._item("서프라이즈: 최근 2개 분기", None, "무료 earnings surprise 소스 연결 필요"),
            self._item("현금 부자: 순현금/시총 >= 10%", net_cash_to_market_cap is not None and net_cash_to_market_cap >= thresholds.get("min_net_cash_to_market_cap_pct", 10), net_cash_to_market_cap),
            self._item("모멘텀: 현재가 >= 신고가 * 0.85", current is not None and high_52w is not None and current >= high_52w * thresholds.get("min_high_proximity", 0.85), {"price": current, "high_52w": high_52w}),
            self._item("이익 성장", earnings_growth is not None and earnings_growth > 0, earnings_growth),
        ]

    def _stage(self, key: str, title: str, thesis: str, items: list[dict[str, Any]]) -> dict[str, Any]:
        known = [item for item in items if item["passed"] is not None]
        score = None
        if known:
            score = round(sum(1 for item in known if item["passed"]) / len(known) * 100, 2)
        gate = "research_required" if score is None else ("pass" if score >= 60 else "fail")
        return {"key": key, "title": title, "thesis": thesis, "score": score, "gate": gate, "items": items}

    def _item(self, label: str, passed: bool | None, value: Any) -> dict[str, Any]:
        return {"label": label, "passed": passed, "value": to_jsonable(value)}

    def _target_expectation(self, current: float | None, info: dict[str, Any]) -> bool | None:
        target = safe_float(info.get("targetMeanPrice"))
        if current is None or target is None:
            return None
        return current > target

    def _rar_ok(self, return_6m: float | None, beta: float | None, max_rar: float) -> bool | None:
        if return_6m is None:
            return None
        denominator = abs(beta) if beta else 1
        rar = return_6m / denominator
        return 0 <= rar <= max_rar

    def _verdict(self, king_score: float | None, stages: list[dict[str, Any]], minimum: float) -> str:
        if king_score is None:
            return "보류: 숫자 데이터가 부족하다. 지금은 판결보다 데이터 확보가 먼저다."
        failed = [stage["title"] for stage in stages if stage["gate"] == "fail"]
        research = [stage["title"] for stage in stages if stage["gate"] == "research_required"]
        if king_score >= minimum and not failed:
            return "통과: 숫자상으로는 심판의 문을 넘었다. 다만 정성 리서치 미확인 항목은 최종 투자 전 확인해야 한다."
        if failed:
            return f"탈락 후보: {', '.join(failed[:2])} 단계에서 약하다. 여기서 무리하면 돈이 먼저 지친다."
        if research:
            return "조건부 보류: 정량은 버티지만 해자/산업/경영진 리서치가 비어 있다."
        return "보류: 기준점에는 못 미친다."

    def _weighted_stage_score(self, stages: list[dict[str, Any]], weights: dict[str, float]) -> float | None:
        numerator = 0.0
        denominator = 0.0
        for stage in stages:
            score = stage.get("score")
            if score is None:
                continue
            weight = safe_float(weights.get(stage["key"])) or 1.0
            numerator += score * weight
            denominator += weight
        if denominator == 0:
            return None
        return round(numerator / denominator, 2)


def _safe_cache_name(symbol: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", symbol).replace(".", "_")


def _normalize_history_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    out.index = pd.to_datetime(out.index)
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    wanted = [col for col in ["Open", "High", "Low", "Close", "Volume"] if col in out.columns]
    out = out[wanted].copy() if wanted else out
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(how="all")


def _extract_download_history(raw: pd.DataFrame, symbol: str, chunk_size: int) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        if symbol in raw.columns.get_level_values(0):
            return _normalize_history_frame(raw[symbol])
        if symbol in raw.columns.get_level_values(1):
            return _normalize_history_frame(raw.xs(symbol, axis=1, level=1))
        return pd.DataFrame()
    if chunk_size == 1:
        return _normalize_history_frame(raw)
    return pd.DataFrame()


def download_live_histories(
    symbols: list[str],
    years: int,
    chunk_size: int,
    cache_dir: Path,
    refresh_hours: float,
) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    cache_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    histories: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for symbol in symbols:
        cache_path = cache_dir / f"{_safe_cache_name(symbol)}.csv"
        if cache_path.exists():
            age_hours = (now - datetime.fromtimestamp(cache_path.stat().st_mtime)).total_seconds() / 3600
            if age_hours <= refresh_hours:
                try:
                    hist = pd.read_csv(cache_path, parse_dates=["Date"]).set_index("Date")
                    hist = _normalize_history_frame(hist)
                    if not hist.empty:
                        histories[symbol] = hist
                        continue
                except Exception:  # noqa: BLE001
                    pass
        missing.append(symbol)

    if not missing:
        print(f"loaded {len(histories)} cached price histories", flush=True)
        return histories

    start = (pd.Timestamp(datetime.now().date()) - pd.DateOffset(years=years, months=1)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(datetime.now().date()) + pd.DateOffset(days=1)).strftime("%Y-%m-%d")
    for idx in range(0, len(missing), chunk_size):
        chunk = missing[idx : idx + chunk_size]
        print(f"downloading live price history {idx + 1}-{idx + len(chunk)} / {len(missing)}", flush=True)
        try:
            raw = yf.download(
                chunk,
                start=start,
                end=end,
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"price download failed for chunk {idx + 1}: {exc}", flush=True)
            raw = pd.DataFrame()
        for symbol in chunk:
            hist = _extract_download_history(raw, symbol, len(chunk))
            if hist.empty:
                continue
            histories[symbol] = hist
            out = hist.copy()
            out.index.name = "Date"
            out.to_csv(cache_dir / f"{_safe_cache_name(symbol)}.csv", encoding="utf-8-sig")
    print(f"price histories available: {len(histories)} / {len(symbols)}", flush=True)
    return histories


def _screen_one(
    row: Any,
    idx: int,
    total: int,
    config: dict[str, Any],
    provider: MarketDataProvider,
    hist: pd.DataFrame | None = None,
) -> ScreeningResult:
    factors = FactorEngine()
    checklist = ChecklistEngine()
    king8 = King8Engine()
    symbol = row.symbol
    started = time.perf_counter()
    print(f"[start {idx}/{total}] screening {symbol}", flush=True)
    try:
        hist, info = provider.fetch(symbol, hist=hist)
        if config.get("model_notes", {}).get("baseline1_three_factor_only") or config.get("model_notes", {}).get("rs_tsl_only"):
            info["_baseline1_three_factor_only"] = True
        factor_values = factors.calculate(hist, info)
        if config.get("model_notes", {}).get("baseline1_three_factor_only"):
            check_items = []
            king_items = {}
        else:
            check_items = checklist.build(hist, info, factor_values, config)
            king_items = king8.build(hist, info, factor_values, {}, check_items, config) if config.get("king8", {}).get("enabled", True) else {}
    except Exception as exc:  # noqa: BLE001
        factor_values = {}
        check_items = [{"label": "데이터 수집 또는 계산 실패", "passed": False, "value": str(exc)}]
        king_items = {"king_score": None, "verdict": f"데이터 실패: {exc}", "stages": []}
    elapsed = time.perf_counter() - started
    sources = factor_values.get("us_v11_data_sources") if factor_values else None
    status = "ok" if factor_values else "empty"
    print(f"[done {idx}/{total}] {symbol} {elapsed:.1f}s status={status} sources={sources or ''}", flush=True)
    return ScreeningResult(
        symbol=symbol,
        name=getattr(row, "name", ""),
        source=getattr(row, "source", ""),
        exchange=getattr(row, "exchange", ""),
        comprehensive_score=None,
        factor_values=factor_values,
        checklist=check_items,
        king8=king_items,
    )


def run_screen(
    config: dict[str, Any],
    universe: pd.DataFrame,
    limit: int | None = None,
    workers: int = 1,
    batch_history: bool = False,
    history_years: int = 2,
    chunk_size: int = 80,
    history_cache: Path = Path(".cache/live_prices"),
    history_refresh_hours: float = 12,
    use_yfinance_info: bool = True,
    us_live_estimates: bool = True,
) -> list[ScreeningResult]:
    krx_flow_cfg = config.get("krx_investor_flows", {})
    enrichment_cfg = config.get("data_enrichment", {})
    include_fundamentals = bool(enrichment_cfg.get("fundamentals", True))
    include_us_v11_data = bool(enrichment_cfg.get("us_v11", True))
    effective_use_yfinance_info = use_yfinance_info and bool(enrichment_cfg.get("yfinance_info", True))
    effective_us_live_estimates = us_live_estimates and bool(enrichment_cfg.get("us_live_estimates", True))
    effective_include_krx_flows = bool(krx_flow_cfg.get("enabled", True)) and bool(enrichment_cfg.get("krx_flows", True))
    market = MarketDataProvider(
        include_krx_flows=effective_include_krx_flows,
        include_fundamentals=include_fundamentals,
        include_us_v11_data=include_us_v11_data,
        use_yfinance_info=effective_use_yfinance_info,
        us_live_estimates=effective_us_live_estimates,
    )
    factors = FactorEngine()
    checklist = ChecklistEngine()
    king8 = King8Engine()
    rows: list[ScreeningResult] = []

    if batch_history or workers > 1 or not use_yfinance_info or not us_live_estimates:
        selected = (universe.head(limit) if limit else universe).reset_index(drop=True)
        total = len(selected)
        histories: dict[str, pd.DataFrame] = {}
        benchmark_histories: dict[str, pd.DataFrame] = {}
        if batch_history:
            try:
                symbols = selected["symbol"].dropna().astype(str).tolist()
                extra_symbols = [*BASELINE1_BENCHMARKS]
                if include_us_v11_data:
                    from data_providers import SECTOR_ETF_MAP

                    extra_symbols = ["SPY", "QQQ", *extra_symbols, *sorted(set(SECTOR_ETF_MAP.values()))]
                all_histories = download_live_histories(
                    list(dict.fromkeys([*symbols, *extra_symbols])),
                    history_years,
                    chunk_size,
                    history_cache,
                    history_refresh_hours,
                )
                histories = {symbol: all_histories[symbol] for symbol in symbols if symbol in all_histories}
                benchmark_histories = {symbol: all_histories[symbol] for symbol in extra_symbols if symbol in all_histories}
            except Exception as exc:  # noqa: BLE001
                print(f"batch price history failed; falling back to per-symbol fetch: {exc}")

        def make_provider() -> MarketDataProvider:
            return MarketDataProvider(
                include_krx_flows=effective_include_krx_flows,
                include_fundamentals=include_fundamentals,
                include_us_v11_data=include_us_v11_data,
                use_yfinance_info=effective_use_yfinance_info,
                us_live_estimates=effective_us_live_estimates,
                benchmark_histories=benchmark_histories,
            )

        if workers <= 1:
            fast_provider = make_provider()
            for idx, row in enumerate(selected.itertuples(index=False), start=1):
                rows.append(_screen_one(row, idx, total, config, fast_provider, histories.get(row.symbol)))
        else:
            thread_state = threading.local()
            progress_started = time.perf_counter()

            def provider_for_thread() -> MarketDataProvider:
                provider = getattr(thread_state, "provider", None)
                if provider is None:
                    provider = make_provider()
                    thread_state.provider = provider
                return provider

            def submit_row(idx: int, row: Any) -> ScreeningResult:
                return _screen_one(row, idx, total, config, provider_for_thread(), histories.get(row.symbol))

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(submit_row, idx, row): idx
                    for idx, row in enumerate(selected.itertuples(index=False), start=1)
                }
                for future in as_completed(futures):
                    rows.append(future.result())
                    completed = len(rows)
                    if completed <= 5 or completed % 10 == 0 or completed == total:
                        elapsed = time.perf_counter() - progress_started
                        rate = completed / elapsed if elapsed > 0 else 0
                        eta = (total - completed) / rate if rate > 0 else float("nan")
                        print(
                            f"[progress] completed={completed}/{total} elapsed={elapsed/60:.1f}m "
                            f"rate={rate*60:.1f}/min eta={eta/60:.1f}m",
                            flush=True,
                        )

        ScoreEngine().normalize(rows, config)
        for row in rows:
            if row.king8.get("stages"):
                row.king8["factor_normalized_context"] = row.factor_scores
        rows.sort(key=lambda r: (-1 if r.comprehensive_score is None else r.comprehensive_score), reverse=True)
        return rows

    selected = universe.head(limit) if limit else universe
    total = len(selected)
    for idx, row in enumerate(selected.itertuples(index=False), start=1):
        symbol = row.symbol
        print(f"[{idx}/{total}] screening {symbol}")
        try:
            hist, info = market.fetch(symbol)
            if config.get("model_notes", {}).get("baseline1_three_factor_only") or config.get("model_notes", {}).get("rs_tsl_only"):
                info["_baseline1_three_factor_only"] = True
            factor_values = factors.calculate(hist, info)
            if config.get("model_notes", {}).get("baseline1_three_factor_only"):
                check_items = []
                king_items = {}
            else:
                check_items = checklist.build(hist, info, factor_values, config)
                king_items = king8.build(hist, info, factor_values, {}, check_items, config) if config.get("king8", {}).get("enabled", True) else {}
        except Exception as exc:  # noqa: BLE001
            factor_values = {}
            check_items = [{"label": "데이터 수집 또는 계산 실패", "passed": False, "value": str(exc)}]
            king_items = {"king_score": None, "verdict": f"데이터 실패: {exc}", "stages": []}
        rows.append(
            ScreeningResult(
                symbol=symbol,
                name=getattr(row, "name", ""),
                source=getattr(row, "source", ""),
                exchange=getattr(row, "exchange", ""),
                comprehensive_score=None,
                factor_values=factor_values,
                checklist=check_items,
                king8=king_items,
            )
        )

    ScoreEngine().normalize(rows, config)
    for row in rows:
        if row.king8.get("stages"):
            row.king8["factor_normalized_context"] = row.factor_scores
    rows.sort(key=lambda r: (-1 if r.comprehensive_score is None else r.comprehensive_score), reverse=True)
    return rows


def write_outputs(rows: list[ScreeningResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    simple_baseline = bool(rows) and all(
        row.trend_score is None
        and row.turnaround_score is None
        and row.route_score is None
        and not row.candidate_type
        for row in rows
    )
    flat_rows = []
    for row in rows:
        base = {
            "symbol": row.symbol,
            "name": row.name,
            "source": row.source,
            "exchange": row.exchange,
            "comprehensive_score": row.comprehensive_score,
        }
        if not simple_baseline:
            base.update(
                {
                    "trend_score": row.trend_score,
                    "turnaround_score": row.turnaround_score,
                    "trend_bonus": row.trend_bonus,
                    "turnaround_bonus": row.turnaround_bonus,
                    "signal_bonus_score": row.signal_bonus_score,
                    "financial_survival_score": row.financial_survival_score,
                    "fragility_score": row.fragility_score,
                    "route_score": row.route_score,
                    "candidate_type": row.candidate_type,
                    "king8_score": row.king8.get("king_score"),
                    "king8_verdict": row.king8.get("verdict"),
                    "check_pass": sum(1 for item in row.checklist if item["passed"] is True),
                    "check_fail": sum(1 for item in row.checklist if item["passed"] is False),
                    "check_unknown": sum(1 for item in row.checklist if item["passed"] is None),
                }
            )
        for factor, value in row.factor_values.items():
            base[f"value_{factor}"] = value
        for factor, score in row.factor_scores.items():
            base[f"score_{factor}"] = score
        flat_rows.append(base)

    pd.DataFrame(flat_rows).to_csv(output_dir / "screening_results.csv", index=False, encoding="utf-8-sig")
    with (output_dir / "screening_results.json").open("w", encoding="utf-8") as f:
        json.dump([row_to_dict(r) for r in rows], f, ensure_ascii=False, indent=2)
    write_markdown(rows, output_dir / "top_ranked.md")
    write_macro_snapshot(output_dir)


def write_macro_snapshot(output_dir: Path) -> None:
    try:
        from data_providers import EcosProvider, FredProvider

        as_of = pd.Timestamp(datetime.now().date())
        snapshot = {"as_of_date": str(as_of.date())}
        try:
            snapshot.update(FredProvider().snapshot(as_of))
        except Exception as exc:  # noqa: BLE001
            snapshot["fred_error"] = str(exc)
        try:
            snapshot.update(EcosProvider().snapshot(as_of))
        except Exception as exc:  # noqa: BLE001
            snapshot["ecos_error"] = str(exc)
        (output_dir / "macro_snapshot.json").write_text(json.dumps(to_jsonable(snapshot), ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        (output_dir / "macro_snapshot.json").write_text(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2), encoding="utf-8")


def row_to_dict(row: ScreeningResult) -> dict[str, Any]:
    base = {
        "symbol": row.symbol,
        "name": row.name,
        "source": row.source,
        "exchange": row.exchange,
        "comprehensive_score": row.comprehensive_score,
        "factor_values": row.factor_values,
        "factor_scores": row.factor_scores,
    }
    if row.trend_score is not None or row.turnaround_score is not None or row.route_score is not None or row.candidate_type:
        base.update(
            {
                "trend_score": row.trend_score,
                "turnaround_score": row.turnaround_score,
                "trend_bonus": row.trend_bonus,
                "turnaround_bonus": row.turnaround_bonus,
                "signal_bonus_score": row.signal_bonus_score,
                "financial_survival_score": row.financial_survival_score,
                "fragility_score": row.fragility_score,
                "route_score": row.route_score,
                "candidate_type": row.candidate_type,
                "king8": row.king8,
                "checklist": row.checklist,
            }
        )
    return to_jsonable(base)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return safe_float(value)
    if isinstance(value, float):
        return safe_float(value)
    return value


def write_markdown(rows: list[ScreeningResult], path: Path, top_n: int = 30) -> None:
    lines = ["# Top Ranked Screening Results", ""]
    simple_baseline = bool(rows) and all(
        row.trend_score is None
        and row.turnaround_score is None
        and row.route_score is None
        and not row.candidate_type
        for row in rows
    )
    if simple_baseline:
        lines.append("| Rank | Symbol | Name | Source | Score | RS | Trend Line | Volume |")
        lines.append("|---:|---|---|---|---:|---:|---:|---:|")
    else:
        lines.append("| Rank | Symbol | Name | Source | Score | Type | Trend | Turnaround | Pass | Fail | Unknown |")
        lines.append("|---:|---|---|---|---:|---|---:|---:|---:|---:|---:|")
    for rank, row in enumerate(rows[:top_n], start=1):
        score = "" if row.comprehensive_score is None else f"{row.comprehensive_score:.2f}"
        if simple_baseline:
            rs = row.factor_values.get("baseline1_rs_score")
            trend_line = row.factor_values.get("baseline1_trend_line_score")
            volume = row.factor_values.get("baseline1_volume_score")
            rs_text = "" if rs is None else f"{rs:.2f}"
            trend_line_text = "" if trend_line is None else f"{trend_line:.2f}"
            volume_text = "" if volume is None else f"{volume:.2f}"
            lines.append(f"| {rank} | {row.symbol} | {row.name} | {row.source} | {score} | {rs_text} | {trend_line_text} | {volume_text} |")
        else:
            pass_count = sum(1 for item in row.checklist if item["passed"] is True)
            fail_count = sum(1 for item in row.checklist if item["passed"] is False)
            unknown_count = sum(1 for item in row.checklist if item["passed"] is None)
            trend = "" if row.trend_score is None else f"{row.trend_score:.2f}"
            turnaround = "" if row.turnaround_score is None else f"{row.turnaround_score:.2f}"
            king = "" if row.king8.get("king_score") is None else f" / K8 {row.king8['king_score']:.2f}"
            lines.append(f"| {rank} | {row.symbol} | {row.name} | {row.source} | {score}{king} | {row.candidate_type} | {trend} | {turnaround} | {pass_count} | {fail_count} | {unknown_count} |")

    if any(row.king8 for row in rows[:top_n]):
        lines.append("")
        lines.append("## KING-8 Verdicts")
        for row in rows[:top_n]:
            if not row.king8:
                continue
            verdict = row.king8.get("verdict", "")
            score = row.king8.get("king_score")
            score_text = "N/A" if score is None else f"{score:.2f}"
            lines.append(f"- **{row.symbol}** KING-8 {score_text}: {verdict}")

    lines.append("")
    lines.append("## Factor Definitions")
    present_factors: set[str] = set()
    for row in rows:
        present_factors.update(row.factor_values.keys())
        present_factors.update(row.factor_scores.keys())
    for key, label in FACTOR_LABELS.items():
        if key in present_factors:
            lines.append(f"- `{key}`: {label}")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen S&P 500, ARKK, KOSPI 200, and KOSDAQ 100 style universes.")
    parser.add_argument("--config", type=Path, default=Path("config/screener_config.json"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--limit", type=int, default=None, help="Limit universe size for test runs.")
    parser.add_argument("--offline-universe", type=Path, default=None, help="CSV with symbol,name,source,exchange columns.")
    parser.add_argument("--symbols", type=str, default="", help="Comma-separated symbols for a small manual run.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers for live full-universe screening.")
    parser.add_argument("--batch-history", action="store_true", help="Download price histories in yfinance batches before scoring.")
    parser.add_argument("--history-years", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=80)
    parser.add_argument("--history-cache", type=Path, default=Path(".cache/live_prices"))
    parser.add_argument("--history-refresh-hours", type=float, default=12)
    parser.add_argument("--skip-yfinance-info", action="store_true", help="Skip slow per-symbol yfinance info calls.")
    parser.add_argument("--no-us-live-estimates", action="store_true", help="Use cached/PIT US API fields without current-profile live estimate calls.")
    return parser.parse_args()


def main() -> None:
    load_env_file()
    args = parse_args()
    config = load_config(args.config)

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        universe = pd.DataFrame({"symbol": symbols, "name": "", "source": "manual", "exchange": ""})
    else:
        universe = UniverseProvider().build(config, args.offline_universe)

    if universe.empty:
        raise SystemExit("No symbols found.")

    rows = run_screen(
        config,
        universe,
        args.limit,
        workers=args.workers,
        batch_history=args.batch_history,
        history_years=args.history_years,
        chunk_size=args.chunk_size,
        history_cache=args.history_cache,
        history_refresh_hours=args.history_refresh_hours,
        use_yfinance_info=not args.skip_yfinance_info,
        us_live_estimates=not args.no_us_live_estimates,
    )
    write_outputs(rows, args.output)
    print(f"Wrote results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
