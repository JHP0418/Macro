from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

from harvest_etf_holdings_and_component_ohlcv import safe_filename, yahoo_chart_ohlcv


ROOT = Path(__file__).resolve().parents[1]
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0"})

MANUAL_NAME_MAP = {
    "미래에셋증권": "006800.KS",
    "KT&G": "033780.KS",
    "SKC": "011790.KS",
    "KCC": "002380.KS",
    "WAL-MART STORES INC": "WMT",
    "Walmart Inc": "WMT",
    "BAKER HUGHES A GE": "BKR",
    "VISTRA ENERGY CORP": "VST",
    "Vistra Energy Corp": "VST",
    "MERCEDES-BENZ GROUP AG": "MBG.DE",
    "BAYERISCHE MOTOREN WERKE AG": "BMW.DE",
    "DHL GROUP": "DHL.DE",
    "SIEMENS AG-REG": "SIE.DE",
    "SIEMENS ENERGY AG": "ENR.DE",
    "BAYER AG-REG": "BAYN.DE",
    "RHEINMETALL AG": "RHM.DE",
    "ADIDAS AG": "ADS.DE",
    "BNP PARIBAS": "BNP.PA",
    "ING GROEP NV": "INGA.AS",
    "AXA SA": "CS.PA",
    "KONINKLIJKE AHOLD DELHAIZE N": "AD.AS",
    "BYD CO LTD-H": "1211.HK",
    "BYD CO LTD -A": "002594.SZ",
    "JD HEALTH INTERNATIONAL INC": "6618.HK",
    "PING AN INSURANCE GROUP CO-H": "2318.HK",
    "ANTA SPORTS PRODUCTS LTD": "2020.HK",
    "ZTO EXPRESS CAYMAN INC": "ZTO",
    "IND & COMM BK OF CHINA-H": "1398.HK",
    "AGRICULTURAL BANK OF CHINA-H": "1288.HK",
    "CSPC PHARMACEUTICAL GROUP LT": "1093.HK",
    "TONGCHENG TRAVEL HOLDINGS LT": "0780.HK",
    "UNIVERSAL MUSIC GROUP NV": "UMG.AS",
    "SYSMEX CORP": "6869.T",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Resolve missing component tickers via Yahoo search and proxy non-priced holdings.")
    p.add_argument("--failed", default=str(ROOT / "outputs" / "component_price_history_repair_failed.csv"))
    p.add_argument("--holdings", default=str(ROOT / "data" / "etf_holdings.csv"))
    p.add_argument("--universe", default=str(ROOT / "data" / "etf_universe_leadership.csv"))
    p.add_argument("--source-holdings-static", default=str(ROOT / "data" / "etf_holdings_static_2019_approx.csv"))
    p.add_argument("--out-holdings-static", default=str(ROOT / "data" / "etf_holdings_static_2019_repaired.csv"))
    p.add_argument("--component-cache", default=str(ROOT / ".cache" / "component_ohlcv"))
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--max-yahoo-search", type=int, default=0, help="0 means no limit.")
    p.add_argument("--sleep", type=float, default=0.05)
    return p.parse_args()


def cache_path(cache_dir: Path, ticker: str) -> Path:
    return cache_dir / f"{safe_filename(ticker)}.csv"


def read_ohlcv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if "Date" not in frame.columns or "Close" not in frame.columns:
        return pd.DataFrame()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
    return frame.dropna(subset=["Date", "Close"]).sort_values("Date").drop_duplicates("Date", keep="last")


def save_alias_price(cache_dir: Path, alias_ticker: str, source_ticker: str, start: str, end: str | None) -> bool:
    frame = yahoo_chart_ohlcv(source_ticker, start, end)
    if frame.empty or frame.shape[0] < 100:
        return False
    frame.to_csv(cache_path(cache_dir, alias_ticker), encoding="utf-8-sig")
    return True


def implied_candidates(ticker: str) -> list[str]:
    text = str(ticker).strip()
    out: list[str] = []
    suffixes = ["DE", "MU", "SG", "PA", "AS", "SW", "VI", "IL", "L", "HK", "BK", "F"]
    for suffix in suffixes:
        if text.endswith(suffix) and len(text) > len(suffix) + 1:
            out.append(f"{text[:-len(suffix)]}.{suffix}")
    if re.fullmatch(r"\d{6}\.KS", text):
        out.append(text[:-3] + ".KQ")
    if re.fullmatch(r"\d{6}\.KQ", text):
        out.append(text[:-3] + ".KS")
    return out


def yahoo_search(query: str) -> list[str]:
    if not query or query.lower() == "nan":
        return []
    try:
        resp = SESSION.get(
            "https://query2.finance.yahoo.com/v1/finance/search",
            params={"q": query, "quotesCount": 8, "newsCount": 0},
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []
    symbols = []
    for item in data.get("quotes", []):
        symbol = item.get("symbol")
        qtype = item.get("quoteType")
        if symbol and qtype in {"EQUITY", "ETF", "INDEX", "MUTUALFUND"}:
            symbols.append(str(symbol))
    return symbols


def candidate_symbols(ticker: str, names: list[str]) -> list[str]:
    out: list[str] = []
    for name in names:
        if name in MANUAL_NAME_MAP:
            out.append(MANUAL_NAME_MAP[name])
    out.extend(implied_candidates(ticker))
    for name in names[:3]:
        out.extend(yahoo_search(name))
    seen = set()
    uniq = []
    for symbol in out:
        if symbol and symbol not in seen:
            seen.add(symbol)
            uniq.append(symbol)
    return uniq


def resolve_aliases(failed: pd.DataFrame, holdings: pd.DataFrame, cache_dir: Path, start: str, end: str | None, max_search: int, sleep: float) -> pd.DataFrame:
    rows = []
    attempted = 0
    grouped = holdings[holdings["component_ticker"].isin(set(failed["ticker"].astype(str)))]
    for ticker, part in grouped.groupby("component_ticker"):
        if read_ohlcv(cache_path(cache_dir, ticker)).shape[0] >= 100:
            rows.append({"ticker": ticker, "resolved_ticker": ticker, "method": "already_available", "status": "resolved"})
            continue
        names = [str(x).strip() for x in part["component_name"].dropna().astype(str).unique() if str(x).strip()]
        candidates = candidate_symbols(str(ticker), names)
        resolved = None
        for cand in candidates:
            attempted += 1
            if max_search and attempted > max_search:
                break
            if save_alias_price(cache_dir, str(ticker), cand, start, end):
                resolved = cand
                break
            if sleep > 0:
                time.sleep(sleep)
        rows.append(
            {
                "ticker": ticker,
                "component_names": "|".join(names[:5]),
                "resolved_ticker": resolved or "",
                "candidate_count": len(candidates),
                "method": "yahoo_search_or_manual" if resolved else "",
                "status": "resolved" if resolved else "unresolved",
            }
        )
    return pd.DataFrame(rows)


def make_proxy_holdings(static_path: Path, universe_path: Path, cache_dir: Path, out_path: Path, unresolved: set[str]) -> pd.DataFrame:
    holdings = pd.read_csv(static_path)
    universe = pd.read_csv(universe_path)
    etf_set = set(universe["etf_ticker"].astype(str))
    rows = []
    proxy_rows = []
    for rec in holdings.to_dict("records"):
        ticker = str(rec.get("component_ticker"))
        etf = str(rec.get("etf_ticker"))
        if ticker in unresolved and float(rec.get("weight") or 0.0) > 0:
            proxy = f"PROXY_{safe_filename(etf)}_{safe_filename(ticker)}"
            source = etf if etf in etf_set else str(rec.get("benchmark_ticker", ""))
            source_frame = read_ohlcv(cache_path(cache_dir, source))
            if source_frame.empty:
                source_frame = pd.DataFrame()
            if not source_frame.empty:
                source_frame.to_csv(cache_path(cache_dir, proxy), index=False, encoding="utf-8-sig")
                rec["component_ticker"] = proxy
                rec["weight_source"] = "proxy_etf_price"
                proxy_rows.append({"proxy_ticker": proxy, "source_ticker": source, "original_ticker": ticker, "etf_ticker": etf, "component_name": rec.get("component_name", "")})
        rows.append(rec)
    out = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(proxy_rows).to_csv(ROOT / "outputs" / "component_price_proxy_map.csv", index=False, encoding="utf-8-sig")
    return out


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.component_cache)
    failed = pd.read_csv(args.failed)
    failed["ticker"] = failed["ticker"].astype(str)
    holdings = pd.read_csv(args.holdings)
    holdings["component_ticker"] = holdings["component_ticker"].astype(str)

    alias_map = resolve_aliases(failed, holdings, cache_dir, args.start, args.end, args.max_yahoo_search, args.sleep)
    alias_out = ROOT / "outputs" / "component_price_resolved_aliases.csv"
    alias_map.to_csv(alias_out, index=False, encoding="utf-8-sig")

    unresolved = set(alias_map.loc[alias_map["status"].ne("resolved"), "ticker"].astype(str))
    repaired_holdings = make_proxy_holdings(Path(args.source_holdings_static), Path(args.universe), cache_dir, Path(args.out_holdings_static), unresolved)
    unresolved_after_proxy = [
        t for t in sorted(unresolved)
        if t in set(repaired_holdings["component_ticker"].astype(str))
        and read_ohlcv(cache_path(cache_dir, t)).empty
    ]
    summary = {
        "failed_input": int(failed.shape[0]),
        "resolved_aliases": int(alias_map["status"].eq("resolved").sum()),
        "unresolved_before_proxy": int(len(unresolved)),
        "proxy_holdings_out": str(Path(args.out_holdings_static)),
        "alias_out": str(alias_out),
        "unresolved_after_proxy_remaining_original_rows": int(len(unresolved_after_proxy)),
    }
    out = ROOT / "outputs" / "component_price_resolve_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
