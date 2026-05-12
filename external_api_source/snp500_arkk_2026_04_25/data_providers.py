from __future__ import annotations

import json
import os
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any
import time
from xml.etree import ElementTree

import pandas as pd
import requests

from screener_agent import HTTP_HEADERS, safe_float

MARKET_API_TIMEOUT = float(os.getenv("MARKET_API_TIMEOUT", "12"))


DART_REPORT_CODES = {
    "annual": "11011",
    "q1": "11013",
    "half": "11012",
    "q3": "11014",
}


def parse_amount(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if text in {"", "-", "nan", "None"}:
        return None
    text = text.replace("(", "-").replace(")", "")
    return safe_float(text)


def first_present(metrics: dict[str, float | None], names: list[str]) -> float | None:
    for name in names:
        value = metrics.get(name)
        if value is not None:
            return value
    return None


def api_cache_only() -> bool:
    return os.getenv("MARKET_API_CACHE_ONLY", "0").strip().lower() in {"1", "true", "yes"}


def provider_disabled(provider_name: str) -> bool:
    normalized = provider_name.upper().replace("-", "_")
    keys = [
        f"DISABLE_{normalized}",
        f"{normalized}_DISABLED",
        f"{normalized}_ENABLED",
    ]
    for key in keys[:2]:
        if os.getenv(key, "").strip().lower() in {"1", "true", "yes"}:
            return True
    enabled_value = os.getenv(keys[2])
    if enabled_value is not None and enabled_value.strip().lower() in {"0", "false", "no"}:
        return True
    return False


def api_sleep_seconds(provider_name: str) -> float:
    key = f"{provider_name.upper()}_API_SLEEP_SEC"
    value = os.getenv(key) or os.getenv("MARKET_API_SLEEP_SEC")
    parsed = safe_float(value)
    return parsed or 0.0


def cacheable_json_payload(data: Any) -> bool:
    if data in ({}, [], None):
        return False
    if isinstance(data, dict):
        lowered = {str(k).lower(): v for k, v in data.items()}
        blocked_keys = {
            "note",
            "information",
            "error message",
            "error",
            "message",
        }
        if blocked_keys & set(lowered):
            text = " ".join(str(v).lower() for v in lowered.values())
            if any(token in text for token in ["rate", "limit", "premium", "error", "invalid", "thank you"]):
                return False
    return True


@dataclass(frozen=True)
class DartReportRef:
    bsns_year: int
    reprt_code: str
    label: str
    available_date: pd.Timestamp


class DartProvider:
    def __init__(self, api_key: str | None = None, cache_dir: Path = Path(".cache/dart")) -> None:
        self.api_key = api_key or os.getenv("OPENDART_API_KEY", "")
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._corp_map: dict[str, dict[str, str]] | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.api_key) and not provider_disabled("opendart")

    def metrics_for_symbol(
        self,
        symbol: str,
        as_of_date: pd.Timestamp | None = None,
        market_cap: float | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {}
        ticker = self._ticker_from_symbol(symbol)
        if not ticker:
            return {}
        corp = self.corp_map().get(ticker)
        if not corp:
            return {}
        as_of = as_of_date or pd.Timestamp(datetime.now().date())
        report = self.report_for_date(as_of)
        current = self.statement_metrics(corp["corp_code"], report.bsns_year, report.reprt_code)
        if not current:
            return {}
        previous = self.statement_metrics(corp["corp_code"], report.bsns_year - 1, report.reprt_code)
        if not previous and report.reprt_code != DART_REPORT_CODES["annual"]:
            previous = self.statement_metrics(corp["corp_code"], report.bsns_year - 1, DART_REPORT_CODES["annual"])
        return self._to_info(corp, report, current, previous, market_cap)

    def corp_map(self) -> dict[str, dict[str, str]]:
        if self._corp_map is not None:
            return self._corp_map
        cache_path = self.cache_dir / "corp_codes.json"
        if cache_path.exists():
            self._corp_map = json.loads(cache_path.read_text(encoding="utf-8"))
            return self._corp_map
        if api_cache_only():
            self._corp_map = {}
            return self._corp_map
        sleep_sec = api_sleep_seconds("opendart")
        if sleep_sec > 0:
            time.sleep(sleep_sec)

        url = "https://opendart.fss.or.kr/api/corpCode.xml"
        response = requests.get(url, params={"crtfc_key": self.api_key}, headers=HTTP_HEADERS, timeout=MARKET_API_TIMEOUT)
        response.raise_for_status()
        with zipfile.ZipFile(BytesIO(response.content)) as zf:
            xml_name = zf.namelist()[0]
            root = ElementTree.fromstring(zf.read(xml_name))
        mapping: dict[str, dict[str, str]] = {}
        for item in root.findall("list"):
            stock_code = (item.findtext("stock_code") or "").strip()
            if not stock_code:
                continue
            mapping[stock_code] = {
                "corp_code": (item.findtext("corp_code") or "").strip(),
                "corp_name": (item.findtext("corp_name") or "").strip(),
                "stock_code": stock_code,
            }
        cache_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
        self._corp_map = mapping
        return mapping

    def report_for_date(self, as_of_date: pd.Timestamp) -> DartReportRef:
        candidates: list[DartReportRef] = []
        for year in range(as_of_date.year - 3, as_of_date.year + 1):
            candidates.extend(
                [
                    DartReportRef(year - 1, DART_REPORT_CODES["annual"], "annual", pd.Timestamp(year=year, month=4, day=1)),
                    DartReportRef(year, DART_REPORT_CODES["q1"], "q1", pd.Timestamp(year=year, month=5, day=16)),
                    DartReportRef(year, DART_REPORT_CODES["half"], "half", pd.Timestamp(year=year, month=8, day=16)),
                    DartReportRef(year, DART_REPORT_CODES["q3"], "q3", pd.Timestamp(year=year, month=11, day=16)),
                ]
            )
        available = [candidate for candidate in candidates if candidate.available_date <= as_of_date]
        if not available:
            return DartReportRef(as_of_date.year - 2, DART_REPORT_CODES["annual"], "annual", pd.Timestamp(as_of_date.year - 1, 4, 1))
        return sorted(available, key=lambda item: item.available_date)[-1]

    def statement_metrics(self, corp_code: str, bsns_year: int, reprt_code: str) -> dict[str, float | None]:
        for fs_div in ("CFS", "OFS"):
            data = self._statement_json(corp_code, bsns_year, reprt_code, fs_div)
            if data.get("status") != "000" or not data.get("list"):
                continue
            metrics = self._parse_statement_rows(data["list"])
            if metrics:
                metrics["fs_div"] = fs_div  # type: ignore[assignment]
                return metrics
        return {}

    def _statement_json(self, corp_code: str, bsns_year: int, reprt_code: str, fs_div: str) -> dict[str, Any]:
        cache_path = self.cache_dir / f"fnltt_{corp_code}_{bsns_year}_{reprt_code}_{fs_div}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        if api_cache_only():
            return {}
        sleep_sec = api_sleep_seconds("opendart")
        if sleep_sec > 0:
            time.sleep(sleep_sec)
        url = "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"
        params = {
            "crtfc_key": self.api_key,
            "corp_code": corp_code,
            "bsns_year": str(bsns_year),
            "reprt_code": reprt_code,
            "fs_div": fs_div,
        }
        response = requests.get(url, params=params, headers=HTTP_HEADERS, timeout=MARKET_API_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return data

    def _parse_statement_rows(self, rows: list[dict[str, Any]]) -> dict[str, float | None]:
        parsed: dict[str, float | None] = {}
        account_map = {
            "totalAssets": ["자산총계"],
            "totalLiabilities": ["부채총계"],
            "totalStockholderEquity": ["자본총계", "자본총계(지배기업 소유주지분)"],
            "totalRevenue": ["매출액", "수익(매출액)", "영업수익"],
            "grossProfits": ["매출총이익", "매출총이익(손실)"],
            "operatingIncome": ["영업이익", "영업이익(손실)"],
            "netIncomeToCommon": ["당기순이익", "당기순이익(손실)", "분기순이익", "반기순이익"],
            "totalCash": ["현금및현금성자산", "현금 및 현금성자산"],
            "operatingCashflow": ["영업활동 현금흐름", "영업활동으로 인한 현금흐름"],
            "capitalExpenditures": ["유형자산의 취득", "유형자산 취득", "유형자산의 증가"],
            "researchDevelopment": ["연구개발비", "연구비", "경상연구개발비"],
        }
        for row in rows:
            account_nm = str(row.get("account_nm", "")).strip()
            amount = parse_amount(row.get("thstrm_amount"))
            if amount is None:
                continue
            normalized = re.sub(r"\s+", "", account_nm)
            for key, names in account_map.items():
                if key in parsed:
                    continue
                if any(re.sub(r"\s+", "", name) == normalized for name in names):
                    parsed[key] = amount
        return parsed

    def _to_info(
        self,
        corp: dict[str, str],
        report: DartReportRef,
        current: dict[str, float | None],
        previous: dict[str, float | None],
        market_cap: float | None,
    ) -> dict[str, Any]:
        assets = current.get("totalAssets")
        liabilities = current.get("totalLiabilities")
        equity = current.get("totalStockholderEquity")
        revenue = current.get("totalRevenue")
        operating_income = current.get("operatingIncome")
        net_income = current.get("netIncomeToCommon")
        gross_profit = current.get("grossProfits")
        cash = current.get("totalCash")
        operating_cf = current.get("operatingCashflow")
        capex = current.get("capitalExpenditures")
        free_cashflow = None
        if operating_cf is not None:
            free_cashflow = operating_cf - abs(capex or 0)

        prev_revenue = previous.get("totalRevenue")
        prev_net_income = previous.get("netIncomeToCommon")
        return {
            "dartCorpCode": corp["corp_code"],
            "shortName": corp["corp_name"],
            "dartReportYear": report.bsns_year,
            "dartReportCode": report.reprt_code,
            "dartReportLabel": report.label,
            "dartReportAvailableDate": str(report.available_date.date()),
            "totalAssets": assets,
            "totalDebt": liabilities,
            "totalStockholderEquity": equity,
            "totalRevenue": revenue,
            "grossProfits": gross_profit,
            "operatingIncome": operating_income,
            "netIncomeToCommon": net_income,
            "totalCash": cash,
            "operatingCashflow": operating_cf,
            "capitalExpenditures": capex,
            "freeCashflow": free_cashflow,
            "researchDevelopment": current.get("researchDevelopment"),
            "returnOnEquity": net_income / equity if net_income is not None and equity else None,
            "returnOnAssets": net_income / assets if net_income is not None and assets else None,
            "debtToEquity": liabilities / equity * 100 if liabilities is not None and equity else None,
            "profitMargins": net_income / revenue if net_income is not None and revenue else None,
            "operatingMargins": operating_income / revenue if operating_income is not None and revenue else None,
            "grossMargins": gross_profit / revenue if gross_profit is not None and revenue else None,
            "revenueGrowth": (revenue / prev_revenue - 1) if revenue is not None and prev_revenue else None,
            "earningsGrowth": (net_income / prev_net_income - 1) if net_income is not None and prev_net_income else None,
            "netCashToMarketCap": ((cash or 0) - (liabilities or 0)) / market_cap * 100 if market_cap else None,
        }

    def _ticker_from_symbol(self, symbol: str) -> str | None:
        match = re.match(r"^(\d{6})(?:\.(KS|KQ))?$", symbol)
        return match.group(1) if match else None


class FredProvider:
    def __init__(self, api_key: str | None = None, cache_dir: Path = Path(".cache/fred")) -> None:
        self.api_key = api_key or os.getenv("FRED_API_KEY", "")
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._obs_cache: dict[tuple[str, str], Any] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.api_key) and not provider_disabled("fred")

    def observation(self, series_id: str, as_of_date: pd.Timestamp) -> float | None:
        if not self.enabled:
            return None
        cache_key = (series_id, str(pd.Timestamp(as_of_date).date()))
        if cache_key in self._obs_cache:
            data = self._obs_cache[cache_key]
            rows = data.get("observations", []) if isinstance(data, dict) else []
            return safe_float(rows[0].get("value")) if rows else None
        cache_path = self.cache_dir / f"{series_id}_{as_of_date.date()}.json"
        if cache_path.exists():
            data = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            if api_cache_only():
                return None
            sleep_sec = api_sleep_seconds("fred")
            if sleep_sec > 0:
                time.sleep(sleep_sec)
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
                "observation_end": str(as_of_date.date()),
                "sort_order": "desc",
                "limit": 1,
            }
            response = requests.get(url, params=params, headers=HTTP_HEADERS, timeout=20)
            response.raise_for_status()
            data = response.json()
            cache_path.write_text(json.dumps(data), encoding="utf-8")
        self._obs_cache[cache_key] = data
        rows = data.get("observations", [])
        if not rows:
            return None
        return safe_float(rows[0].get("value"))

    def snapshot(self, as_of_date: pd.Timestamp) -> dict[str, float | None]:
        return {
            "fred_fedfunds": self.observation("FEDFUNDS", as_of_date),
            "fred_10y": self.observation("DGS10", as_of_date),
            "fred_2y": self.observation("DGS2", as_of_date),
            "fred_cpi_yoy": self.observation("CPIAUCSL", as_of_date),
            "fred_unemployment": self.observation("UNRATE", as_of_date),
        }


class EcosProvider:
    def __init__(self, api_key: str | None = None, cache_dir: Path = Path(".cache/ecos")) -> None:
        self.api_key = api_key or os.getenv("ECOS_API_KEY", "")
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return bool(self.api_key) and not provider_disabled("ecos")

    def statistic_search(
        self,
        stat_code: str,
        cycle: str,
        start: str,
        end: str,
        item_code: str | None = None,
    ) -> float | None:
        if not self.enabled:
            return None
        suffix = item_code or "none"
        cache_path = self.cache_dir / f"{stat_code}_{cycle}_{start}_{end}_{suffix}.json"
        if cache_path.exists():
            data = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            if api_cache_only():
                return None
            sleep_sec = api_sleep_seconds("ecos")
            if sleep_sec > 0:
                time.sleep(sleep_sec)
            parts = [
                "https://ecos.bok.or.kr/api/StatisticSearch",
                self.api_key,
                "json",
                "kr",
                "1",
                "1000",
                stat_code,
                cycle,
                start,
                end,
            ]
            if item_code:
                parts.append(item_code)
            url = "/".join(parts)
            response = requests.get(url, headers=HTTP_HEADERS, timeout=20)
            response.raise_for_status()
            data = response.json()
            cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        rows = data.get("StatisticSearch", {}).get("row", [])
        if not rows:
            return None
        return safe_float(rows[-1].get("DATA_VALUE"))

    def snapshot(self, as_of_date: pd.Timestamp) -> dict[str, float | None]:
        month = as_of_date.strftime("%Y%m")
        month_start = (as_of_date - timedelta(days=120)).strftime("%Y%m")
        day = as_of_date.strftime("%Y%m%d")
        day_start = (as_of_date - timedelta(days=14)).strftime("%Y%m%d")
        return {
            "ecos_base_rate": self.statistic_search("722Y001", "D", day_start, day, "0101000"),
            "ecos_usdkrw": self.statistic_search("731Y001", "D", day_start, day, "0000001"),
            "ecos_cpi": self.statistic_search("901Y009", "M", month_start, month, "0"),
        }


class KrxInvestorFlowProvider:
    """KRX investor-flow factors for Korean equities through pykrx.

    pykrx/KRX endpoints are not as stable as OpenDART/FRED, so this provider is
    deliberately fail-soft: after repeated empty/error responses it disables
    itself for the current process and returns no factors.
    """

    INVESTOR_ALIASES = {
        "foreign": ["외국인", "외국인합계"],
        "pension": ["연기금", "연기금등"],
        "other_corp": ["기타법인"],
        "institution": ["기관합계", "기관"],
    }

    DAILY_COLUMN_ALIASES = {
        "foreign": ["외국인합계", "외국인"],
        "other_corp": ["기타법인"],
        "institution": ["기관합계", "기관"],
    }

    def __init__(self, cache_dir: Path = Path(".cache/krx_flows"), max_failures: int = 5) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_failures = max_failures
        self._failures = 0
        self.disabled_reason: str | None = None

    @property
    def enabled(self) -> bool:
        return self.disabled_reason is None and not provider_disabled("krx")

    def metrics_for_symbol(
        self,
        symbol: str,
        as_of_date: pd.Timestamp,
        trading_value_20d: float | None = None,
        trading_value_60d: float | None = None,
    ) -> dict[str, float | None]:
        ticker = self._ticker_from_symbol(symbol)
        if not ticker or not self.enabled:
            return {}

        as_of = self._yyyymmdd(as_of_date)
        start_20 = self._yyyymmdd(as_of_date - pd.Timedelta(days=45))
        start_60 = self._yyyymmdd(as_of_date - pd.Timedelta(days=120))

        investor_20 = self._trading_value_by_investor(start_20, as_of, ticker)
        investor_60 = self._trading_value_by_investor(start_60, as_of, ticker)
        daily_60 = self._trading_value_by_date(start_60, as_of, ticker)

        if investor_20.empty and investor_60.empty and daily_60.empty:
            return {}

        metrics: dict[str, float | None] = {}
        for investor_key in ("foreign", "pension", "other_corp", "institution"):
            net_20 = self._net_buy_from_investor_frame(investor_20, investor_key)
            net_60 = self._net_buy_from_investor_frame(investor_60, investor_key)
            metrics[f"{investor_key}_net_buy_20d"] = net_20
            metrics[f"{investor_key}_net_buy_60d"] = net_60
            metrics[f"{investor_key}_net_buy_strength_20d"] = self._strength_pct(net_20, trading_value_20d)
            metrics[f"{investor_key}_net_buy_strength_60d"] = self._strength_pct(net_60, trading_value_60d)

        for investor_key in ("foreign", "other_corp", "institution"):
            metrics[f"{investor_key}_flow_persistence_20d"] = self._persistence(daily_60, investor_key, 20)
            metrics[f"{investor_key}_flow_persistence_60d"] = self._persistence(daily_60, investor_key, 60)

        combo_20 = self._weighted_combo(metrics, "20d")
        combo_60 = self._weighted_combo(metrics, "60d")
        metrics["kr_smart_money_combo_20d"] = combo_20
        metrics["kr_smart_money_combo_60d"] = combo_60
        metrics["kr_smart_money_accel"] = combo_20 - combo_60 if combo_20 is not None and combo_60 is not None else None
        return metrics

    def _trading_value_by_investor(self, start: str, end: str, ticker: str) -> pd.DataFrame:
        cache_path = self.cache_dir / f"investor_{ticker}_{start}_{end}.csv"
        if cache_path.exists():
            return pd.read_csv(cache_path, index_col=0)
        if api_cache_only():
            return pd.DataFrame()
        if not self.enabled:
            return pd.DataFrame()
        try:
            from pykrx import stock

            sleep_sec = api_sleep_seconds("krx")
            if sleep_sec > 0:
                time.sleep(sleep_sec)
            df = stock.get_market_trading_value_by_investor(start, end, ticker)
        except Exception as exc:  # noqa: BLE001
            self._record_failure(exc)
            return pd.DataFrame()
        if df is None or df.empty:
            self._record_failure(RuntimeError("empty investor-flow response"))
            return pd.DataFrame()
        df.to_csv(cache_path, encoding="utf-8-sig")
        return df

    def _trading_value_by_date(self, start: str, end: str, ticker: str) -> pd.DataFrame:
        cache_path = self.cache_dir / f"daily_{ticker}_{start}_{end}.csv"
        if cache_path.exists():
            return pd.read_csv(cache_path, index_col=0)
        if api_cache_only():
            return pd.DataFrame()
        if not self.enabled:
            return pd.DataFrame()
        try:
            from pykrx import stock

            sleep_sec = api_sleep_seconds("krx")
            if sleep_sec > 0:
                time.sleep(sleep_sec)
            df = stock.get_market_trading_value_by_date(start, end, ticker)
        except Exception as exc:  # noqa: BLE001
            self._record_failure(exc)
            return pd.DataFrame()
        if df is None or df.empty:
            self._record_failure(RuntimeError("empty daily investor-flow response"))
            return pd.DataFrame()
        df.to_csv(cache_path, encoding="utf-8-sig")
        return df

    def _record_failure(self, exc: Exception) -> None:
        self._failures += 1
        if self._failures >= self.max_failures:
            self.disabled_reason = f"KRX investor-flow disabled after repeated failures: {exc}"

    def _net_buy_from_investor_frame(self, frame: pd.DataFrame, investor_key: str) -> float | None:
        if frame.empty:
            return None
        row = self._row_by_alias(frame, self.INVESTOR_ALIASES[investor_key])
        if row is None:
            return None
        for col in ("순매수", "순매수거래대금", "순매수대금"):
            if col in row.index:
                return safe_float(row[col])
        sell = first_present({str(k): safe_float(v) for k, v in row.items()}, ["매도", "매도거래대금", "매도대금"])
        buy = first_present({str(k): safe_float(v) for k, v in row.items()}, ["매수", "매수거래대금", "매수대금"])
        if buy is None or sell is None:
            return None
        return buy - sell

    def _row_by_alias(self, frame: pd.DataFrame, aliases: list[str]) -> pd.Series | None:
        index_text = [str(idx).strip() for idx in frame.index]
        for alias in aliases:
            if alias in index_text:
                return frame.iloc[index_text.index(alias)]
        investor_col = next((col for col in frame.columns if "투자자" in str(col)), None)
        if investor_col is None:
            return None
        matches = frame[frame[investor_col].astype(str).str.strip().isin(aliases)]
        if matches.empty:
            return None
        return matches.iloc[0]

    def _persistence(self, frame: pd.DataFrame, investor_key: str, sessions: int) -> float | None:
        if frame.empty:
            return None
        col = self._column_by_alias(frame, self.DAILY_COLUMN_ALIASES[investor_key])
        if col is None:
            return None
        series = pd.to_numeric(frame[col], errors="coerce").dropna().tail(sessions)
        if series.empty:
            return None
        return float((series > 0).mean() * 100)

    def _column_by_alias(self, frame: pd.DataFrame, aliases: list[str]) -> str | None:
        columns = [str(col).strip() for col in frame.columns]
        for alias in aliases:
            if alias in columns:
                return frame.columns[columns.index(alias)]  # type: ignore[return-value]
        return None

    def _weighted_combo(self, metrics: dict[str, float | None], suffix: str) -> float | None:
        weights = {
            f"foreign_net_buy_strength_{suffix}": 1.0,
            f"pension_net_buy_strength_{suffix}": 1.0,
            f"other_corp_net_buy_strength_{suffix}": 0.5,
            f"institution_net_buy_strength_{suffix}": 0.35,
        }
        numerator = 0.0
        denominator = 0.0
        for key, weight in weights.items():
            value = safe_float(metrics.get(key))
            if value is None:
                continue
            numerator += value * weight
            denominator += weight
        return numerator / denominator if denominator else None

    def _strength_pct(self, net_buy: float | None, trading_value: float | None) -> float | None:
        if net_buy is None or not trading_value:
            return None
        return net_buy / trading_value * 100

    def _yyyymmdd(self, value: pd.Timestamp) -> str:
        ts = pd.Timestamp(value)
        if ts.tzinfo is not None:
            ts = ts.tz_convert(None)
        return ts.strftime("%Y%m%d")

    def _ticker_from_symbol(self, symbol: str) -> str | None:
        match = re.match(r"^(\d{6})(?:\.(KS|KQ))?$", symbol)
        return match.group(1) if match else None


US_V11_COLUMNS = [
    "estimate_timestamp",
    "financial_filing_date",
    "fy1_eps_estimate_current",
    "fy1_eps_estimate_1m_ago",
    "fy1_eps_estimate_3m_ago",
    "fy2_eps_estimate_current",
    "fy2_eps_estimate_1m_ago",
    "fy2_eps_estimate_3m_ago",
    "revenue_estimate_current",
    "revenue_estimate_3m_ago",
    "num_upward_revisions_3m",
    "num_downward_revisions_3m",
    "latest_eps_actual",
    "latest_eps_consensus",
    "latest_revenue_actual",
    "latest_revenue_consensus",
    "earnings_announcement_date",
    "earnings_day_return",
    "post_earnings_5d_return",
    "post_earnings_20d_return",
    "post_earnings_volume_ratio",
    "sector",
    "industry",
    "sector_etf_return_3m",
    "sector_etf_return_6m",
    "sector_etf_return_12m",
    "industry_median_return_3m",
    "industry_median_return_6m",
    "stock_return_3m",
    "stock_return_6m",
    "stock_return_12m",
    "market_return_3m",
    "market_return_6m",
    "market_return_12m",
    "revenue_growth_yoy",
    "eps_growth_yoy",
    "fcf_growth_yoy",
    "gross_margin",
    "operating_margin",
    "fcf_margin",
    "roic",
    "rule_of_40",
    "forward_pe",
    "ev_sales",
    "ev_ebitda",
    "price_sales",
    "fcf_yield",
    "sales_growth_yoy",
    "ten_year_yield",
    "ten_year_yield_3m_change",
    "real_yield_3m_change",
    "qqq_return_3m",
    "spy_return_3m",
    "stock_beta_to_qqq",
    "stock_beta_to_spy",
    "stock_corr_to_qqq",
    "us_v11_data_sources",
]


SECTOR_ETF_MAP = {
    "Technology": "XLK",
    "Information Technology": "XLK",
    "Consumer Cyclical": "XLY",
    "Consumer Discretionary": "XLY",
    "Communication Services": "XLC",
    "Healthcare": "XLV",
    "Health Care": "XLV",
    "Financial Services": "XLF",
    "Financial": "XLF",
    "Financials": "XLF",
    "Industrials": "XLI",
    "Consumer Defensive": "XLP",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Basic Materials": "XLB",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
}


def _as_float(value: Any) -> float | None:
    return safe_float(value)


def _first_number(row: dict[str, Any] | pd.Series | None, names: list[str]) -> float | None:
    if row is None:
        return None
    for name in names:
        if name in row:
            value = _as_float(row[name])
            if value is not None:
                return value
    lower = {str(k).lower(): v for k, v in dict(row).items()}
    for name in names:
        value = _as_float(lower.get(name.lower()))
        if value is not None:
            return value
    return None


def _first_text(row: dict[str, Any] | pd.Series | None, names: list[str]) -> str | None:
    if row is None:
        return None
    for name in names:
        if name in row and row[name] not in (None, ""):
            return str(row[name])
    lower = {str(k).lower(): v for k, v in dict(row).items()}
    for name in names:
        value = lower.get(name.lower())
        if value not in (None, ""):
            return str(value)
    return None


def _safe_pct_change(current: float | None, previous: float | None) -> float | None:
    if current is None or previous in (None, 0):
        return None
    return current / abs(previous) - 1


def _history_return(hist: pd.DataFrame | None, as_of_date: pd.Timestamp, sessions: int) -> float | None:
    if hist is None or hist.empty or "Close" not in hist:
        return None
    frame = hist.copy()
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    frame = frame[frame.index <= pd.Timestamp(as_of_date)]
    close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
    if len(close) <= sessions:
        return None
    start = _as_float(close.iloc[-sessions])
    end = _as_float(close.iloc[-1])
    if not start or end is None:
        return None
    return (end / start - 1) * 100


def _last_close(hist: pd.DataFrame | None, as_of_date: pd.Timestamp) -> float | None:
    if hist is None or hist.empty or "Close" not in hist:
        return None
    frame = hist.copy()
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    frame = frame[frame.index <= pd.Timestamp(as_of_date)]
    close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
    return _as_float(close.iloc[-1]) if len(close) else None


def _event_timestamp(value: Any) -> pd.Timestamp | None:
    event = pd.to_datetime(value, errors="coerce")
    if pd.isna(event):
        return None
    event = pd.Timestamp(event)
    if event.tzinfo is not None:
        event = event.tz_convert(None)
    return event


def _event_day_return(hist: pd.DataFrame | None, event_date: Any, as_of_date: pd.Timestamp) -> float | None:
    event = _event_timestamp(event_date)
    if hist is None or hist.empty or event is None or "Close" not in hist:
        return None
    frame = hist.copy()
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    frame = frame.sort_index()
    frame = frame[frame.index <= pd.Timestamp(as_of_date)]
    if len(frame) < 2:
        return None
    pos = frame.index.searchsorted(event, side="left")
    if pos <= 0 or pos >= len(frame):
        return None
    close = pd.to_numeric(frame["Close"], errors="coerce")
    start = _as_float(close.iloc[pos - 1])
    end = _as_float(close.iloc[pos])
    if not start or end is None:
        return None
    return (end / start - 1) * 100


def _post_event_return(hist: pd.DataFrame | None, event_date: Any, as_of_date: pd.Timestamp, sessions: int) -> float | None:
    event = _event_timestamp(event_date)
    if hist is None or hist.empty or event is None or "Close" not in hist:
        return None
    frame = hist.copy()
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    frame = frame.sort_index()
    frame = frame[frame.index <= pd.Timestamp(as_of_date)]
    pos = frame.index.searchsorted(event, side="left")
    end_pos = pos + sessions
    if pos < 0 or end_pos >= len(frame):
        return None
    close = pd.to_numeric(frame["Close"], errors="coerce")
    start = _as_float(close.iloc[pos])
    end = _as_float(close.iloc[end_pos])
    if not start or end is None:
        return None
    return (end / start - 1) * 100


def _post_event_volume_ratio(hist: pd.DataFrame | None, event_date: Any, as_of_date: pd.Timestamp) -> float | None:
    event = _event_timestamp(event_date)
    if hist is None or hist.empty or event is None or "Volume" not in hist:
        return None
    frame = hist.copy()
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    frame = frame.sort_index()
    frame = frame[frame.index <= pd.Timestamp(as_of_date)]
    pos = frame.index.searchsorted(event, side="left")
    if pos < 20 or pos >= len(frame):
        return None
    volume = pd.to_numeric(frame["Volume"], errors="coerce")
    post = _as_float(volume.iloc[pos : min(pos + 5, len(frame))].mean())
    pre = _as_float(volume.iloc[max(0, pos - 60) : pos].mean())
    if not pre or post is None:
        return None
    return post / pre


def _beta_corr(stock_hist: pd.DataFrame | None, bench_hist: pd.DataFrame | None, as_of_date: pd.Timestamp, sessions: int = 252) -> tuple[float | None, float | None]:
    if stock_hist is None or bench_hist is None or stock_hist.empty or bench_hist.empty:
        return None, None
    if "Close" not in stock_hist or "Close" not in bench_hist:
        return None, None
    left = stock_hist[["Close"]].copy()
    right = bench_hist[["Close"]].copy()
    left.index = pd.to_datetime(left.index).tz_localize(None)
    right.index = pd.to_datetime(right.index).tz_localize(None)
    frame = left.join(right, how="inner", lsuffix="_stock", rsuffix="_bench")
    frame = frame[frame.index <= pd.Timestamp(as_of_date)].tail(sessions + 1)
    returns = frame.pct_change().dropna()
    if len(returns) < 30:
        return None, None
    stock_ret = returns["Close_stock"]
    bench_ret = returns["Close_bench"]
    var = bench_ret.var()
    beta = stock_ret.cov(bench_ret) / var if var else None
    corr = stock_ret.corr(bench_ret)
    return safe_float(beta), safe_float(corr)


class _CachedHttpProvider:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._json_cache: dict[str, Any] = {}

    def _json_get(self, cache_name: str, url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> Any:
        if cache_name in self._json_cache:
            return self._json_cache[cache_name]
        cache_path = self.cache_dir / f"{cache_name}.json"
        provider_name = self.cache_dir.name
        if cache_path.exists():
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            self._json_cache[cache_name] = data
            return data
        if api_cache_only():
            data = {}
            self._json_cache[cache_name] = data
            return data
        try:
            sleep_sec = api_sleep_seconds(provider_name)
            if sleep_sec > 0:
                time.sleep(sleep_sec)
            response = requests.get(url, params=params, headers=headers or HTTP_HEADERS, timeout=MARKET_API_TIMEOUT)
            response.raise_for_status()
            data = response.json()
        except Exception:  # noqa: BLE001
            data = {}
        if cacheable_json_payload(data):
            cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        self._json_cache[cache_name] = data
        return data


class AlphaVantageProvider(_CachedHttpProvider):
    def __init__(self, api_key: str | None = None, cache_dir: Path = Path(".cache/alpha_vantage")) -> None:
        super().__init__(cache_dir)
        self.api_key = api_key or os.getenv("ALPHAVANTAGE_API_KEY", "")
        self.base_url = "https://www.alphavantage.co/query"

    @property
    def enabled(self) -> bool:
        return bool(self.api_key) and not provider_disabled("alpha_vantage") and not provider_disabled("alphavantage")

    def query(self, function: str, symbol: str) -> dict[str, Any]:
        if not self.enabled:
            return {}
        safe_symbol = symbol.replace("/", "_").replace(".", "_")
        data = self._json_get(
            f"{function}_{safe_symbol}",
            self.base_url,
            {"function": function, "symbol": symbol, "apikey": self.api_key},
        )
        if isinstance(data, dict) and any(k in data for k in ("Note", "Information", "Error Message")):
            return {}
        return data if isinstance(data, dict) else {}

    def overview(self, symbol: str) -> dict[str, Any]:
        return self.query("OVERVIEW", symbol)

    def earnings(self, symbol: str) -> dict[str, Any]:
        return self.query("EARNINGS", symbol)

    def earnings_estimates(self, symbol: str) -> dict[str, Any]:
        return self.query("EARNINGS_ESTIMATES", symbol)

    def metrics_for_symbol(
        self,
        symbol: str,
        as_of_date: pd.Timestamp,
        include_estimates: bool = True,
        include_current_overview: bool = True,
    ) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        overview = self.overview(symbol)
        if overview:
            metrics.update({"sector": overview.get("Sector"), "industry": overview.get("Industry")})
            if include_current_overview:
                metrics.update(
                    {
                        "forward_pe": _first_number(overview, ["ForwardPE", "PERatio"]),
                        "ev_sales": _first_number(overview, ["EVToRevenue"]),
                        "ev_ebitda": _first_number(overview, ["EVToEBITDA"]),
                        "price_sales": _first_number(overview, ["PriceToSalesRatioTTM"]),
                        "gross_margin": _first_number(overview, ["GrossProfitTTM"]),
                        "operating_margin": _first_number(overview, ["OperatingMarginTTM"]),
                        "revenue_growth_yoy": _first_number(overview, ["QuarterlyRevenueGrowthYOY"]),
                        "eps_growth_yoy": _first_number(overview, ["QuarterlyEarningsGrowthYOY"]),
                        "sales_growth_yoy": _first_number(overview, ["QuarterlyRevenueGrowthYOY"]),
                    }
                )
        earnings = self.earnings(symbol)
        quarterly = earnings.get("quarterlyEarnings", []) if isinstance(earnings, dict) else []
        if isinstance(quarterly, list):
            rows = []
            for row in quarterly:
                date = pd.to_datetime(row.get("reportedDate") or row.get("fiscalDateEnding"), errors="coerce")
                if pd.notna(date) and date <= as_of_date:
                    rows.append((date, row))
            if rows:
                _, latest = sorted(rows, key=lambda item: item[0])[-1]
                metrics.update(
                    {
                        "latest_eps_actual": _first_number(latest, ["reportedEPS"]),
                        "latest_eps_consensus": _first_number(latest, ["estimatedEPS"]),
                        "earnings_announcement_date": _first_text(latest, ["reportedDate", "fiscalDateEnding"]),
                    }
                )
        if include_estimates:
            estimates = self.earnings_estimates(symbol)
            rows = estimates.get("estimates") if isinstance(estimates, dict) else None
            if isinstance(rows, list):
                dated_rows = []
                for row in rows:
                    date = pd.to_datetime(row.get("date"), errors="coerce")
                    if pd.notna(date) and date <= as_of_date:
                        dated_rows.append((date, row))
                if dated_rows:
                    date, current = sorted(dated_rows, key=lambda item: item[0])[-1]
                    fy2 = {}
                    future_rows = []
                    for row in rows:
                        future_date = pd.to_datetime(row.get("date"), errors="coerce")
                        if pd.notna(future_date) and future_date > date:
                            future_rows.append((future_date, row))
                    if future_rows:
                        fy2 = sorted(future_rows, key=lambda item: item[0])[0][1]
                    metrics.update(
                        {
                            "estimate_timestamp": str(date.date()),
                            "fy1_eps_estimate_current": _first_number(current, ["eps_estimate_average"]),
                            "fy1_eps_estimate_1m_ago": _first_number(current, ["eps_estimate_average_30_days_ago"]),
                            "fy1_eps_estimate_3m_ago": _first_number(current, ["eps_estimate_average_90_days_ago"]),
                            "fy2_eps_estimate_current": _first_number(fy2, ["eps_estimate_average"]),
                            "fy2_eps_estimate_1m_ago": _first_number(fy2, ["eps_estimate_average_30_days_ago"]),
                            "fy2_eps_estimate_3m_ago": _first_number(fy2, ["eps_estimate_average_90_days_ago"]),
                            "revenue_estimate_current": _first_number(current, ["revenue_estimate_average"]),
                            "revenue_estimate_3m_ago": _first_number(current, ["revenue_estimate_average_90_days_ago"]),
                            "num_upward_revisions_3m": _first_number(current, ["eps_estimate_revision_up_trailing_30_days"]),
                            "num_downward_revisions_3m": _first_number(current, ["eps_estimate_revision_down_trailing_30_days"]),
                        }
                    )
            for key in ("quarterlyEstimates", "quarterlyEarningsEstimates", "quarterly"):
                rows = estimates.get(key) if isinstance(estimates, dict) else None
                if isinstance(rows, list) and rows:
                    current = rows[0]
                    previous = rows[1] if len(rows) > 1 else {}
                    fallback = {
                        "fy1_eps_estimate_current": _first_number(current, ["epsEstimateAverage", "estimatedEPS", "estimate"]),
                        "fy1_eps_estimate_1m_ago": _first_number(current, ["epsEstimateAverage1MonthAgo", "estimatedEPS1MonthAgo"]),
                        "fy1_eps_estimate_3m_ago": _first_number(current, ["epsEstimateAverage3MonthsAgo", "estimatedEPS3MonthsAgo"]),
                        "fy2_eps_estimate_current": _first_number(previous, ["epsEstimateAverage", "estimatedEPS", "estimate"]),
                        "revenue_estimate_current": _first_number(current, ["revenueEstimateAverage", "estimatedRevenue"]),
                        "revenue_estimate_3m_ago": _first_number(current, ["revenueEstimateAverage3MonthsAgo", "estimatedRevenue3MonthsAgo"]),
                        "num_upward_revisions_3m": _first_number(current, ["revisionsUpLast3Months", "upwardRevisions3M"]),
                        "num_downward_revisions_3m": _first_number(current, ["revisionsDownLast3Months", "downwardRevisions3M"]),
                    }
                    metrics.update({k: v for k, v in fallback.items() if metrics.get(k) in (None, "")})
                    break
        return {k: v for k, v in metrics.items() if v not in (None, "")}


class FinnhubProvider(_CachedHttpProvider):
    def __init__(self, api_key: str | None = None, cache_dir: Path = Path(".cache/finnhub")) -> None:
        super().__init__(cache_dir)
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY", "")
        self.base_url = "https://finnhub.io/api/v1"

    @property
    def enabled(self) -> bool:
        return bool(self.api_key) and not provider_disabled("finnhub")

    def endpoint(self, path: str, symbol: str, extra: dict[str, Any] | None = None) -> Any:
        if not self.enabled:
            return {}
        params = {"symbol": symbol, "token": self.api_key}
        if extra:
            params.update(extra)
        safe_path = path.strip("/").replace("/", "_")
        safe_symbol = symbol.replace("/", "_").replace(".", "_")
        return self._json_get(f"{safe_path}_{safe_symbol}_{hash(json.dumps(extra or {}, sort_keys=True))}", f"{self.base_url}/{path.strip('/')}", params)

    def metrics_for_symbol(self, symbol: str, as_of_date: pd.Timestamp) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        profile = self.endpoint("stock/profile2", symbol)
        if isinstance(profile, dict):
            metrics.update({"sector": profile.get("finnhubIndustry")})
        earnings = self.endpoint("stock/earnings", symbol)
        if isinstance(earnings, list):
            rows = []
            for row in earnings:
                date = pd.to_datetime(row.get("period"), errors="coerce")
                if pd.notna(date) and date <= as_of_date:
                    rows.append((date, row))
            if rows:
                _, latest = sorted(rows, key=lambda item: item[0])[-1]
                metrics.update(
                    {
                        "latest_eps_actual": _first_number(latest, ["actual"]),
                        "latest_eps_consensus": _first_number(latest, ["estimate"]),
                        "earnings_announcement_date": _first_text(latest, ["period"]),
                    }
                )
        return {k: v for k, v in metrics.items() if v not in (None, "")}


class FmpProvider(_CachedHttpProvider):
    def __init__(self, api_key: str | None = None, cache_dir: Path = Path(".cache/fmp")) -> None:
        super().__init__(cache_dir)
        self.api_key = api_key or os.getenv("FMP_API_KEY", "")
        self.stable_url = "https://financialmodelingprep.com/stable"
        self.legacy_url = "https://financialmodelingprep.com/api/v3"
        self.use_premium_endpoints = os.getenv("FMP_USE_PREMIUM_ENDPOINTS", "0").strip().lower() in {"1", "true", "yes"}

    @property
    def enabled(self) -> bool:
        return bool(self.api_key) and not provider_disabled("fmp")

    def _get_any(self, cache_name: str, urls: list[str], params: dict[str, Any]) -> Any:
        if not self.enabled:
            return {}
        if cache_name in self._json_cache:
            return self._json_cache[cache_name]
        cache_path = self.cache_dir / f"{cache_name}.json"
        if cache_path.exists():
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            self._json_cache[cache_name] = data
            return data
        if api_cache_only():
            data = {}
            self._json_cache[cache_name] = data
            return data
        last_error: Exception | None = None
        for url in urls:
            try:
                sleep_sec = api_sleep_seconds("fmp")
                if sleep_sec > 0:
                    time.sleep(sleep_sec)
                response = requests.get(url, params=params, headers=HTTP_HEADERS, timeout=MARKET_API_TIMEOUT)
                response.raise_for_status()
                data = response.json()
                if cacheable_json_payload(data):
                    cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                    self._json_cache[cache_name] = data
                    return data
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
        data = {}
        self._json_cache[cache_name] = data
        return data

    def endpoint(self, name: str, symbol: str, period: str | None = None, limit: int = 12) -> Any:
        safe_symbol = symbol.replace("/", "_").replace(".", "_")
        params = {"apikey": self.api_key, "limit": limit}
        if period:
            params["period"] = period
        stable = f"{self.stable_url}/{name}"
        v3 = f"{self.legacy_url}/{name}/{symbol}"
        if name in {"profile"}:
            stable = f"{self.stable_url}/profile"
            params["symbol"] = symbol
        else:
            params["symbol"] = symbol
        return self._get_any(f"fmpv5_{name}_{safe_symbol}_{period or 'none'}_{limit}", [stable, v3], params)

    def _rows_before(self, data: Any, as_of_date: pd.Timestamp) -> list[dict[str, Any]]:
        rows = data if isinstance(data, list) else data.get("data", []) if isinstance(data, dict) else []
        valid = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            availability_value = row.get("filingDate") or row.get("fillingDate") or row.get("acceptedDate") or row.get("reportedDate")
            period_value = row.get("date") or row.get("period") or row.get("fiscalDateEnding")
            if availability_value:
                date = pd.to_datetime(availability_value, errors="coerce")
            else:
                date = pd.to_datetime(period_value, errors="coerce")
                if pd.notna(date):
                    date = date + pd.DateOffset(days=45)
            if pd.isna(date) or date <= as_of_date:
                valid.append(row)
        return valid

    def metrics_for_symbol(
        self,
        symbol: str,
        as_of_date: pd.Timestamp,
        include_estimates: bool = True,
        include_current_profile: bool = True,
    ) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        profile = self.endpoint("profile", symbol, limit=1)
        profile_row = profile[0] if isinstance(profile, list) and profile else profile if isinstance(profile, dict) else {}
        metrics.update({"sector": _first_text(profile_row, ["sector"]), "industry": _first_text(profile_row, ["industry"])})
        if include_current_profile:
            metrics.update(
                {
                    "forward_pe": _first_number(profile_row, ["pe", "priceEarningsRatio"]),
                    "price_sales": _first_number(profile_row, ["priceToSalesRatio"]),
                }
            )
        ratios = self._rows_before(self.endpoint("ratios", symbol, "quarter", 12), as_of_date) if self.use_premium_endpoints else []
        ratio = ratios[0] if ratios else {}
        growths = self._rows_before(self.endpoint("financial-growth", symbol, "quarter", 5), as_of_date)
        growth = growths[0] if growths else {}
        metrics_row = (self._rows_before(self.endpoint("key-metrics", symbol, "quarter", 12), as_of_date) or [{}])[0] if self.use_premium_endpoints else {}
        estimates = self._rows_before(self.endpoint("analyst-estimates", symbol, "quarter", 12), as_of_date) if include_estimates and self.use_premium_endpoints else []
        estimate = estimates[0] if estimates else {}
        prev_estimate = estimates[1] if len(estimates) > 1 else {}
        income_rows = self._rows_before(self.endpoint("income-statement", symbol, "quarter", 5), as_of_date)
        cash_rows = self._rows_before(self.endpoint("cash-flow-statement", symbol, "quarter", 5), as_of_date)
        balance_rows = self._rows_before(self.endpoint("balance-sheet-statement", symbol, "quarter", 5), as_of_date)
        income = income_rows[0] if income_rows else {}
        prev_income = income_rows[4] if len(income_rows) > 4 else income_rows[1] if len(income_rows) > 1 else {}
        cash = cash_rows[0] if cash_rows else {}
        prev_cash = cash_rows[4] if len(cash_rows) > 4 else cash_rows[1] if len(cash_rows) > 1 else {}
        balance = balance_rows[0] if balance_rows else {}
        revenue = _first_number(income, ["revenue"])
        gross_profit = _first_number(income, ["grossProfit"])
        operating_income = _first_number(income, ["operatingIncome"])
        free_cashflow = _first_number(cash, ["freeCashFlow", "freeCashflow"])
        eps = _first_number(income, ["eps", "epsdiluted"])
        total_debt = _first_number(balance, ["totalDebt", "shortTermDebt", "longTermDebt"])
        cash_and_st = _first_number(balance, ["cashAndShortTermInvestments", "cashAndCashEquivalents"])
        equity = _first_number(balance, ["totalStockholdersEquity", "totalEquity"])
        invested_capital = (equity or 0) + (total_debt or 0) - (cash_and_st or 0) if any(v is not None for v in [equity, total_debt, cash_and_st]) else None
        metrics.update(
            {
                "financial_filing_date": _first_text(income, ["filingDate", "acceptedDate"]) or _first_text(balance, ["filingDate", "acceptedDate"]),
                "revenue_growth_yoy": _first_number(growth, ["revenueGrowth"]) or _safe_pct_change(revenue, _first_number(prev_income, ["revenue"])),
                "eps_growth_yoy": _first_number(growth, ["epsgrowth", "epsGrowth"]) or _safe_pct_change(eps, _first_number(prev_income, ["eps", "epsdiluted"])),
                "fcf_growth_yoy": _first_number(growth, ["freeCashFlowGrowth"]) or _safe_pct_change(free_cashflow, _first_number(prev_cash, ["freeCashFlow", "freeCashflow"])),
                "gross_margin": _first_number(ratio, ["grossProfitMargin"]) or (gross_profit / revenue if revenue else None),
                "operating_margin": _first_number(ratio, ["operatingProfitMargin"]) or (operating_income / revenue if revenue else None),
                "fcf_margin": free_cashflow / revenue if free_cashflow is not None and revenue else _first_number(ratio, ["freeCashFlowOperatingCashFlowRatio"]),
                "roic": _first_number(ratio, ["returnOnInvestedCapital"]) or _first_number(metrics_row, ["roic"]) or (operating_income / invested_capital if operating_income is not None and invested_capital and invested_capital > 0 else None),
                "forward_pe": metrics.get("forward_pe") or _first_number(ratio, ["priceEarningsRatio"]),
                "ev_sales": _first_number(metrics_row, ["evToSales", "enterpriseValueOverRevenue"]),
                "ev_ebitda": _first_number(metrics_row, ["enterpriseValueOverEBITDA", "evToEBITDA"]),
                "price_sales": metrics.get("price_sales") or _first_number(ratio, ["priceToSalesRatio"]),
                "fcf_yield": _first_number(metrics_row, ["freeCashFlowYield", "fcfYield"]),
                "sales_growth_yoy": _first_number(growth, ["revenueGrowth"]),
                "latest_revenue_actual": revenue,
                "latest_revenue_consensus": _first_number(estimate, ["estimatedRevenueAvg", "revenueAvg"]),
                "latest_free_cashflow": free_cashflow,
                "latest_shares_out": _first_number(income, ["weightedAverageShsOutDil", "weightedAverageShsOut"]),
                "latest_eps_actual": eps,
                "latest_operating_income": operating_income,
                "total_debt": total_debt,
                "cash_and_short_term_investments": cash_and_st,
                "fy1_eps_estimate_current": _first_number(estimate, ["estimatedEpsAvg", "epsAvg", "estimatedEPSAvg"]),
                "fy2_eps_estimate_current": _first_number(prev_estimate, ["estimatedEpsAvg", "epsAvg", "estimatedEPSAvg"]),
                "revenue_estimate_current": _first_number(estimate, ["estimatedRevenueAvg", "revenueAvg"]),
                "revenue_estimate_3m_ago": _first_number(prev_estimate, ["estimatedRevenueAvg", "revenueAvg"]),
            }
        )
        rev_growth = _as_float(metrics.get("revenue_growth_yoy"))
        fcf_margin = _as_float(metrics.get("fcf_margin"))
        metrics["rule_of_40"] = rev_growth + fcf_margin if rev_growth is not None and fcf_margin is not None else None
        return {k: v for k, v in metrics.items() if v not in (None, "")}


class TiingoProvider(_CachedHttpProvider):
    def __init__(self, api_key: str | None = None, cache_dir: Path = Path(".cache/tiingo")) -> None:
        super().__init__(cache_dir)
        self.api_key = api_key or os.getenv("TIINGO_API_KEY", "")

    @property
    def enabled(self) -> bool:
        return bool(self.api_key) and not provider_disabled("tiingo")

    def daily_prices(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        if not self.enabled:
            return pd.DataFrame()
        safe_symbol = symbol.replace("/", "_").replace(".", "_")
        cache_path = self.cache_dir / f"prices_{safe_symbol}_{start.date()}_{end.date()}.csv"
        if cache_path.exists():
            return pd.read_csv(cache_path, parse_dates=["date"]).set_index("date")
        if api_cache_only():
            return pd.DataFrame()
        sleep_sec = api_sleep_seconds("tiingo")
        if sleep_sec > 0:
            time.sleep(sleep_sec)
        url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
        params = {"startDate": str(start.date()), "endDate": str(end.date()), "token": self.api_key}
        response = requests.get(url, params=params, headers=HTTP_HEADERS, timeout=MARKET_API_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list) or not data:
            return pd.DataFrame()
        raw = pd.DataFrame(data)
        if "date" not in raw.columns:
            return pd.DataFrame()
        raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None)
        frame = pd.DataFrame(index=raw["date"])
        frame["Open"] = pd.to_numeric(raw.get("adjOpen", raw.get("open")), errors="coerce")
        frame["High"] = pd.to_numeric(raw.get("adjHigh", raw.get("high")), errors="coerce")
        frame["Low"] = pd.to_numeric(raw.get("adjLow", raw.get("low")), errors="coerce")
        frame["Close"] = pd.to_numeric(raw.get("adjClose", raw.get("close")), errors="coerce")
        frame["Volume"] = pd.to_numeric(raw.get("adjVolume", raw.get("volume")), errors="coerce")
        frame = frame.dropna(how="all")
        frame.to_csv(cache_path, encoding="utf-8-sig")
        return frame


class UsV11DataProvider:
    def __init__(self, cache_dir: Path = Path(".cache/us_v11")) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.alpha_vantage = AlphaVantageProvider(cache_dir=cache_dir / "alpha_vantage")
        self.finnhub = FinnhubProvider(cache_dir=cache_dir / "finnhub")
        self.fmp = FmpProvider(cache_dir=cache_dir / "fmp")
        self.fred = FredProvider(cache_dir=cache_dir / "fred")
        self.tiingo = TiingoProvider(cache_dir=cache_dir / "tiingo")
        self._yf_info_cache: dict[str, dict[str, Any]] = {}

    @property
    def enabled(self) -> bool:
        return any([self.alpha_vantage.enabled, self.finnhub.enabled, self.fmp.enabled, self.fred.enabled, self.tiingo.enabled])

    def yfinance_info(self, symbol: str) -> dict[str, Any]:
        if symbol in self._yf_info_cache:
            return self._yf_info_cache[symbol]
        try:
            import yfinance as yf

            info = yf.Ticker(symbol).info or {}
        except Exception:  # noqa: BLE001
            info = {}
        self._yf_info_cache[symbol] = info
        return info

    def metrics_for_symbol(
        self,
        symbol: str,
        as_of_date: pd.Timestamp | None = None,
        stock_history: pd.DataFrame | None = None,
        market_history: pd.DataFrame | None = None,
        qqq_history: pd.DataFrame | None = None,
        sector_etf_history: pd.DataFrame | None = None,
        live_estimates: bool = False,
    ) -> dict[str, Any]:
        if not symbol or symbol.startswith("^") or symbol.endswith((".KS", ".KQ")):
            return {}
        as_of = pd.Timestamp(as_of_date or datetime.now().date())
        metrics: dict[str, Any] = {col: None for col in US_V11_COLUMNS}
        sources: list[str] = []

        info = self.yfinance_info(symbol) if live_estimates else {}
        if info:
            sources.append("yfinance")
            metrics.update(
                {
                    "sector": info.get("sector"),
                    "industry": info.get("industry"),
                    "revenue_growth_yoy": safe_float(info.get("revenueGrowth")),
                    "eps_growth_yoy": safe_float(info.get("earningsGrowth")),
                    "gross_margin": safe_float(info.get("grossMargins")),
                    "operating_margin": safe_float(info.get("operatingMargins")),
                    "forward_pe": safe_float(info.get("forwardPE")),
                    "ev_sales": safe_float(info.get("enterpriseToRevenue")),
                    "ev_ebitda": safe_float(info.get("enterpriseToEbitda")),
                    "price_sales": safe_float(info.get("priceToSalesTrailing12Months")),
                    "fcf_yield": (safe_float(info.get("freeCashflow")) / safe_float(info.get("marketCap"))) if safe_float(info.get("freeCashflow")) is not None and safe_float(info.get("marketCap")) else None,
                }
            )
        for source_name, source_metrics in [
            ("fmp", self.fmp.metrics_for_symbol(symbol, as_of, include_estimates=live_estimates, include_current_profile=live_estimates)),
            ("alpha_vantage", self.alpha_vantage.metrics_for_symbol(symbol, as_of, include_estimates=True, include_current_overview=live_estimates)),
            ("finnhub", self.finnhub.metrics_for_symbol(symbol, as_of)),
        ]:
            if source_metrics:
                sources.append(source_name)
                for key, value in source_metrics.items():
                    if value not in (None, "") and metrics.get(key) in (None, ""):
                        metrics[key] = value
        if not live_estimates:
            estimate_ts = pd.to_datetime(metrics.get("estimate_timestamp"), errors="coerce")
            if pd.isna(estimate_ts) or estimate_ts > as_of:
                for key in [
                    "fy1_eps_estimate_current",
                    "fy1_eps_estimate_1m_ago",
                    "fy1_eps_estimate_3m_ago",
                    "fy2_eps_estimate_current",
                    "fy2_eps_estimate_1m_ago",
                    "fy2_eps_estimate_3m_ago",
                    "revenue_estimate_current",
                    "revenue_estimate_3m_ago",
                    "num_upward_revisions_3m",
                    "num_downward_revisions_3m",
                    "latest_revenue_consensus",
                ]:
                    metrics[key] = None

        event_date = metrics.get("earnings_announcement_date")
        if event_date and stock_history is not None:
            metrics["earnings_day_return"] = _event_day_return(stock_history, event_date, as_of)
            metrics["post_earnings_5d_return"] = _post_event_return(stock_history, event_date, as_of, 5)
            metrics["post_earnings_20d_return"] = _post_event_return(stock_history, event_date, as_of, 20)
            metrics["post_earnings_volume_ratio"] = _post_event_volume_ratio(stock_history, event_date, as_of)

        metrics["stock_return_3m"] = _history_return(stock_history, as_of, 63)
        metrics["stock_return_6m"] = _history_return(stock_history, as_of, 126)
        metrics["stock_return_12m"] = _history_return(stock_history, as_of, 252)
        metrics["market_return_3m"] = _history_return(market_history, as_of, 63)
        metrics["market_return_6m"] = _history_return(market_history, as_of, 126)
        metrics["market_return_12m"] = _history_return(market_history, as_of, 252)
        metrics["qqq_return_3m"] = _history_return(qqq_history, as_of, 63)
        metrics["spy_return_3m"] = metrics["market_return_3m"]
        metrics["sector_etf_return_3m"] = _history_return(sector_etf_history, as_of, 63)
        metrics["sector_etf_return_6m"] = _history_return(sector_etf_history, as_of, 126)
        metrics["sector_etf_return_12m"] = _history_return(sector_etf_history, as_of, 252)
        beta_spy, _ = _beta_corr(stock_history, market_history, as_of)
        beta_qqq, corr_qqq = _beta_corr(stock_history, qqq_history, as_of)
        metrics["stock_beta_to_spy"] = beta_spy
        metrics["stock_beta_to_qqq"] = beta_qqq
        metrics["stock_corr_to_qqq"] = corr_qqq

        close = _last_close(stock_history, as_of)
        shares = safe_float(metrics.get("latest_shares_out"))
        revenue = safe_float(metrics.get("latest_revenue_actual"))
        fcf = safe_float(metrics.get("latest_free_cashflow"))
        eps = safe_float(metrics.get("latest_eps_actual"))
        debt = safe_float(metrics.get("total_debt")) or 0.0
        cash = safe_float(metrics.get("cash_and_short_term_investments")) or 0.0
        market_cap = close * shares if close is not None and shares else None
        enterprise_value = market_cap + debt - cash if market_cap is not None else None
        annual_revenue = revenue * 4 if revenue is not None else None
        annual_fcf = fcf * 4 if fcf is not None else None
        annual_eps = eps * 4 if eps is not None else None
        if market_cap and annual_eps and annual_eps > 0 and metrics.get("forward_pe") is None:
            metrics["forward_pe"] = close / annual_eps if close is not None else None
        if enterprise_value and annual_revenue and annual_revenue > 0 and metrics.get("ev_sales") is None:
            metrics["ev_sales"] = enterprise_value / annual_revenue
        if market_cap and annual_revenue and annual_revenue > 0 and metrics.get("price_sales") is None:
            metrics["price_sales"] = market_cap / annual_revenue
        if market_cap and annual_fcf is not None and market_cap > 0 and metrics.get("fcf_yield") is None:
            metrics["fcf_yield"] = annual_fcf / market_cap

        if self.fred.enabled:
            ten_y = self.fred.observation("DGS10", as_of)
            ten_y_3m = self.fred.observation("DGS10", as_of - pd.DateOffset(months=3))
            real_y = self.fred.observation("DFII10", as_of)
            real_y_3m = self.fred.observation("DFII10", as_of - pd.DateOffset(months=3))
            metrics["ten_year_yield"] = ten_y
            metrics["ten_year_yield_3m_change"] = ten_y - ten_y_3m if ten_y is not None and ten_y_3m is not None else None
            metrics["real_yield_3m_change"] = real_y - real_y_3m if real_y is not None and real_y_3m is not None else None
            sources.append("fred")

        rev_growth = safe_float(metrics.get("revenue_growth_yoy"))
        fcf_margin = safe_float(metrics.get("fcf_margin"))
        if metrics.get("rule_of_40") is None and rev_growth is not None and fcf_margin is not None:
            metrics["rule_of_40"] = rev_growth + fcf_margin
        metrics["us_v11_data_sources"] = ",".join(sorted(set(sources)))
        return {k: v for k, v in metrics.items() if v is not None}

    def sector_etf_for(self, sector: str | None) -> str | None:
        if not sector:
            return None
        return SECTOR_ETF_MAP.get(str(sector).strip())
