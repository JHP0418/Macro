from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yfinance as yf


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNIVERSE = ROOT / "data" / "asset_universe_expanded_2026_05_09.csv"
DEFAULT_HOLDINGS_OUT = ROOT / "data" / "etf_holdings.csv"
DEFAULT_NAME_MAP_OUT = ROOT / "data" / "component_name_ticker_map.csv"
DEFAULT_MARKET_CAP_OUT = ROOT / "data" / "component_market_caps.csv"
DEFAULT_COMPONENT_CACHE = ROOT / ".cache" / "component_ohlcv"
DEFAULT_DART_CORP_CODES = ROOT / ".cache" / "dart" / "corp_codes.json"
SOURCE_PRICE_DIRS = [
    ROOT / ".cache" / "prices_expanded",
    ROOT / ".cache" / "live_prices",
    ROOT / ".cache" / "etf_asset_allocation_prices",
]
WISE_ETF_URL = "https://navercomp.wisereport.co.kr/v2/ETF/index.aspx"
WISE_AUTOCOMPLETE_URL = "https://navercomp.wisereport.co.kr/v2/company/autocomplete.aspx"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://navercomp.wisereport.co.kr/v2/ETF/index.aspx",
}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def main() -> None:
    args = parse_args()
    universe = load_asset_universe(args.universe)
    args.component_cache.mkdir(parents=True, exist_ok=True)

    if args.download_only:
        holdings = pd.read_csv(args.holdings_out)
        component_tickers = sorted(set(holdings["component_ticker"].dropna().astype(str)))
        copied = copy_local_ohlcv(component_tickers, args.component_cache)
        missing = [ticker for ticker in component_tickers if not (args.component_cache / f"{safe_filename(ticker)}.csv").exists()]
        downloaded = download_missing_ohlcv(missing, args.component_cache, args.start, args.end, args.batch_size) if missing else []
        summary = {
            "mode": "download_only",
            "unique_component_tickers": int(len(component_tickers)),
            "local_ohlcv_copied_or_existing": int(copied),
            "missing_before_download": int(len(missing)),
            "downloaded_ohlcv": int(len(downloaded)),
            "component_cache": str(args.component_cache),
        }
        summary_out = ROOT / "outputs" / "component_ohlcv_download_summary.json"
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return

    code_suffix = discover_local_ticker_suffixes()
    name_map = load_existing_name_map(args.name_map_out)
    if args.retry_blank_map:
        name_map = {name: code for name, code in name_map.items() if code}
    name_map.update({k: v for k, v in load_dart_name_map(args.dart_corp_codes).items() if k not in name_map})
    if not args.no_seed_us_indices:
        name_map.update(load_us_index_seed_map(args.us_seed_map_out))
    save_name_map(name_map, args.name_map_out)
    raw_out = args.holdings_out.with_name("etf_holdings_raw_wisereport.csv")

    raw = harvest_raw_holdings(universe, code_suffix, raw_out, args.refresh_holdings, args.sleep)
    if args.raw_only:
        print(json.dumps({"raw_holding_rows": int(len(raw)), "raw_holdings_out": str(raw_out)}, ensure_ascii=False, indent=2))
        return
    holdings = build_mapped_holdings(raw, name_map, code_suffix, args.name_map_out, args.max_yahoo_lookups, args.lookup_workers)
    holdings.to_csv(args.holdings_out.with_name("etf_holdings_mapped_pre_weight.csv"), index=False, encoding="utf-8-sig")
    save_name_map(name_map, args.name_map_out)

    component_tickers = sorted(set(holdings["component_ticker"].dropna().astype(str))) if not holdings.empty else []
    copied = copy_local_ohlcv(component_tickers, args.component_cache)
    missing = [ticker for ticker in component_tickers if not (args.component_cache / f"{safe_filename(ticker)}.csv").exists()]
    downloaded = []
    if missing and not args.skip_download:
        downloaded = download_missing_ohlcv(missing, args.component_cache, args.start, args.end, args.batch_size)
    holdings = finalize_holding_weights(holdings, args.component_cache, args.market_cap_out)
    holdings.to_csv(args.holdings_out, index=False, encoding="utf-8-sig")

    summary = {
        "universe_count": int(len(universe)),
        "raw_holding_rows": int(len(raw)),
        "mapped_holding_rows": int(len(holdings)),
        "unique_component_tickers": int(len(component_tickers)),
        "local_ohlcv_copied": int(copied),
        "missing_before_download": int(len(missing)),
        "downloaded_ohlcv": int(len(downloaded)),
        "weight_sources": holdings["weight_source"].value_counts(dropna=False).to_dict() if "weight_source" in holdings else {},
        "holdings_out": str(args.holdings_out),
        "raw_holdings_out": str(raw_out),
        "name_map_out": str(args.name_map_out),
        "component_cache": str(args.component_cache),
    }
    summary_out = ROOT / "outputs" / "component_ohlcv_download_summary.json"
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Harvest WiseReport ETF holdings and component OHLCV cache.")
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--holdings-out", type=Path, default=DEFAULT_HOLDINGS_OUT)
    parser.add_argument("--name-map-out", type=Path, default=DEFAULT_NAME_MAP_OUT)
    parser.add_argument("--market-cap-out", type=Path, default=DEFAULT_MARKET_CAP_OUT)
    parser.add_argument("--us-seed-map-out", type=Path, default=ROOT / "data" / "us_index_name_ticker_seed_map.csv")
    parser.add_argument("--component-cache", type=Path, default=DEFAULT_COMPONENT_CACHE)
    parser.add_argument("--dart-corp-codes", type=Path, default=DEFAULT_DART_CORP_CODES)
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--refresh-holdings", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--raw-only", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--retry-blank-map", action="store_true")
    parser.add_argument("--no-seed-us-indices", action="store_true")
    parser.add_argument("--max-yahoo-lookups", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--lookup-workers", type=int, default=10)
    return parser.parse_args()


def load_asset_universe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"code", "name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    out = df.copy()
    out["code"] = out["code"].map(normalize_code)
    out["name"] = out["name"].astype(str).str.strip()
    return out.drop_duplicates("code").reset_index(drop=True)


def harvest_raw_holdings(
    universe: pd.DataFrame,
    code_suffix: dict[str, str],
    raw_out: Path,
    refresh: bool,
    sleep_seconds: float,
) -> pd.DataFrame:
    columns = ["date", "etf_ticker", "etf_code", "etf_name", "component_name", "component_shares", "weight"]
    if raw_out.exists() and not refresh:
        raw = pd.read_csv(raw_out, dtype={"etf_code": str})
        raw["etf_code"] = raw["etf_code"].map(normalize_code)
        raw = raw.drop_duplicates(columns)
        raw.to_csv(raw_out, index=False, encoding="utf-8-sig")
        done = set(raw["etf_code"].astype(str).map(normalize_code)) if "etf_code" in raw.columns else set()
        print(f"[raw] resume from {raw_out}: rows={len(raw)} done_etfs={len(done)}", flush=True)
    else:
        raw = pd.DataFrame(columns=columns)
        done = set()

    rows_buffer = raw.to_dict("records")
    for idx, asset in universe.iterrows():
        etf_code = normalize_code(asset["code"])
        if etf_code in done:
            continue
        etf_ticker = kr_ticker(etf_code, code_suffix)
        try:
            rows = fetch_wisereport_cu_rows(etf_code)
        except Exception as exc:
            print(f"[raw] {idx + 1:03d}/{len(universe)} {etf_ticker} failed: {exc}", flush=True)
            rows = []
        for row in rows:
            rows_buffer.append(
                {
                    "date": normalize_date(row.get("TRD_DT")),
                    "etf_ticker": etf_ticker,
                    "etf_code": etf_code,
                    "etf_name": asset["name"],
                    "component_name": str(row.get("STK_NM_KOR", "")).strip(),
                    "component_shares": row.get("AGMT_STK_CNT"),
                    "weight": parse_weight(row.get("ETF_WEIGHT")),
                }
            )
        raw = pd.DataFrame(rows_buffer, columns=columns)
        raw.to_csv(raw_out, index=False, encoding="utf-8-sig")
        print(f"[raw] {idx + 1:03d}/{len(universe)} {etf_ticker} rows={len(rows)} total_rows={len(raw)} saved", flush=True)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return raw


def build_mapped_holdings(
    raw: pd.DataFrame,
    name_map: dict[str, str],
    code_suffix: dict[str, str],
    name_map_out: Path,
    max_yahoo_lookups: int,
    lookup_workers: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    unique_names = sorted({str(x).strip() for x in raw["component_name"].dropna() if str(x).strip()})
    pending = [name for name in unique_names if name not in name_map]
    if max_yahoo_lookups > 0:
        pending = pending[:max_yahoo_lookups]
    completed = 0
    if pending:
        with ThreadPoolExecutor(max_workers=max(1, lookup_workers)) as executor:
            futures = {executor.submit(resolve_component_code, name, {}, True): name for name in pending}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    code, _ = future.result()
                except Exception:
                    code = None
                name_map[name] = normalize_component_code_value(code or "")
                completed += 1
                if completed % 100 == 0:
                    save_name_map(name_map, name_map_out)
                    print(f"[map] completed={completed}/{len(pending)} total_cached={len(name_map)}", flush=True)
    save_name_map(name_map, name_map_out)

    for rec in raw.to_dict("records"):
        name = str(rec.get("component_name") or "").strip()
        code = name_map.get(name, "")
        if not code:
            continue
        rows.append(
            {
                "date": rec.get("date"),
                "etf_ticker": rec.get("etf_ticker"),
                "component_ticker": component_ticker(code, code_suffix),
                "weight": rec.get("weight"),
                "raw_weight": rec.get("weight"),
                "weight_source": "actual" if float(rec.get("weight") or 0.0) > 0 else "missing",
                "component_name": name,
            "component_code": normalize_component_code_value(code),
                "etf_name": rec.get("etf_name"),
            }
        )

    holdings = pd.DataFrame(rows)
    if holdings.empty:
        return pd.DataFrame(columns=["date", "etf_ticker", "component_ticker", "weight", "raw_weight", "weight_source", "component_name", "component_code", "etf_name"])
    holdings["weight"] = pd.to_numeric(holdings["weight"], errors="coerce").fillna(0.0)
    holdings["raw_weight"] = pd.to_numeric(holdings["raw_weight"], errors="coerce").fillna(0.0)
    holdings = holdings.drop_duplicates(["date", "etf_ticker", "component_ticker"])
    holdings = holdings.sort_values(["etf_ticker", "weight"], ascending=[True, False])
    return holdings


def finalize_holding_weights(holdings: pd.DataFrame, component_cache: Path, market_cap_out: Path) -> pd.DataFrame:
    if holdings.empty:
        return holdings
    out = holdings.copy()
    market_caps = load_or_fetch_market_caps(sorted(out["component_ticker"].astype(str).unique()), market_cap_out)
    for _, idx in out.groupby(["date", "etf_ticker"], dropna=False).groups.items():
        raw = pd.to_numeric(out.loc[idx, "raw_weight"], errors="coerce").fillna(0.0)
        raw_sum = raw.sum()
        if raw_sum > 0:
            out.loc[idx, "weight"] = raw / raw_sum
            out.loc[idx, "weight_source"] = "actual"
            continue
        tickers = out.loc[idx, "component_ticker"].astype(str)
        caps = tickers.map(market_caps).astype(float).fillna(0.0)
        if caps.sum() > 0:
            out.loc[idx, "weight"] = caps / caps.sum()
            out.loc[idx, "weight_source"] = "market_cap_pseudo"
            continue
        adv = tickers.map(lambda ticker: recent_adv(ticker, component_cache)).astype(float).fillna(0.0)
        if adv.sum() > 0:
            out.loc[idx, "weight"] = adv / adv.sum()
            out.loc[idx, "weight_source"] = "adv_pseudo"
            continue
        if len(idx) > 0:
            out.loc[idx, "weight"] = 1.0 / len(idx)
            out.loc[idx, "weight_source"] = "equal_weight_fallback"
    return out


def load_or_fetch_market_caps(tickers: list[str], path: Path) -> dict[str, float]:
    if path.exists():
        df = pd.read_csv(path)
        if {"ticker", "market_cap"}.issubset(df.columns):
            cached = dict(zip(df["ticker"].astype(str), pd.to_numeric(df["market_cap"], errors="coerce").fillna(0.0)))
        else:
            cached = {}
    else:
        cached = {}
    missing = [ticker for ticker in tickers if ticker not in cached and not re.search(r"\.(KS|KQ)$", ticker)]
    for i in range(0, len(missing), 80):
        batch = missing[i : i + 80]
        caps = fetch_yahoo_market_caps(batch)
        cached.update(caps)
        pd.DataFrame([{"ticker": k, "market_cap": v} for k, v in sorted(cached.items())]).to_csv(path, index=False, encoding="utf-8-sig")
        if batch:
            print(f"[mcap] {min(i + 80, len(missing))}/{len(missing)} cached={len(cached)}", flush=True)
    return cached


def fetch_yahoo_market_caps(tickers: list[str]) -> dict[str, float]:
    if not tickers:
        return {}
    try:
        resp = SESSION.get(
            "https://query1.finance.yahoo.com/v7/finance/quote",
            params={"symbols": ",".join(tickers)},
            timeout=12,
        )
        resp.raise_for_status()
        results = resp.json().get("quoteResponse", {}).get("result", [])
    except Exception:
        return {ticker: 0.0 for ticker in tickers}
    out = {ticker: 0.0 for ticker in tickers}
    for item in results:
        ticker = str(item.get("symbol") or "")
        cap = item.get("marketCap")
        try:
            out[ticker] = float(cap or 0.0)
        except (TypeError, ValueError):
            out[ticker] = 0.0
    return out


def recent_adv(ticker: str, component_cache: Path, window: int = 60) -> float:
    path = component_cache / f"{safe_filename(ticker)}.csv"
    if not path.exists():
        return 0.0
    try:
        df = pd.read_csv(path)
    except Exception:
        return 0.0
    if not {"Close", "Volume"}.issubset(df.columns):
        return 0.0
    close = pd.to_numeric(df["Close"], errors="coerce")
    volume = pd.to_numeric(df["Volume"], errors="coerce")
    value = (close * volume).dropna().tail(window)
    return float(value.mean()) if not value.empty else 0.0


def normalize_code(value: Any) -> str:
    text = str(value).strip().upper()
    text = text[1:] if text.startswith("A") else text
    return re.sub(r"[^0-9A-Z]", "", text)


def normalize_date(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if re.fullmatch(r"\d{8}", text):
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return text[:10]


def parse_weight(value: Any) -> float:
    try:
        weight = float(value)
    except (TypeError, ValueError):
        return 0.0
    return weight / 100.0 if weight > 1.0 else weight


def fetch_wisereport_cu_rows(etf_code: str) -> list[dict[str, Any]]:
    resp = SESSION.get(WISE_ETF_URL, params={"cmp_cd": etf_code}, timeout=20)
    resp.raise_for_status()
    match = re.search(r"var\s+CU_data\s*=\s*(\{.*?\});", resp.text, re.S)
    if not match:
        return []
    data = json.loads(match.group(1))
    rows = data.get("grid_data") or []
    return rows if isinstance(rows, list) else []


def load_existing_name_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if not {"component_name", "component_code"}.issubset(df.columns):
        return {}
    out: dict[str, str] = {}
    for row in df.itertuples(index=False):
        name = str(row.component_name).strip()
        raw_code = "" if pd.isna(row.component_code) else str(row.component_code).strip()
        if name:
            out[name] = normalize_component_code_value(raw_code)
    return out


def load_dart_name_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, list):
        return {}
    out: dict[str, str] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        code = normalize_code(item.get("stock_code") or "")
        name = str(item.get("corp_name") or "").strip()
        if re.fullmatch(r"\d{6}", code) and name:
            out.setdefault(name, code)
    return out


def load_us_index_seed_map(path: Path) -> dict[str, str]:
    if path.exists():
        try:
            df = pd.read_csv(path)
            if {"component_name", "component_code"}.issubset(df.columns):
                return {
                    str(row.component_name).strip(): str(row.component_code).strip()
                    for row in df.itertuples(index=False)
                    if str(row.component_name).strip() and str(row.component_code).strip()
                }
        except Exception:
            pass
    rows: list[dict[str, str]] = []
    for url, symbol_col, name_col in [
        ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Symbol", "Security"),
        ("https://en.wikipedia.org/wiki/Nasdaq-100", "Ticker", "Company"),
    ]:
        try:
            resp = SESSION.get(url, timeout=20)
            resp.raise_for_status()
            tables = pd.read_html(StringIO(resp.text))
        except Exception:
            continue
        for table in tables:
            if symbol_col not in table.columns or name_col not in table.columns:
                continue
            for record in table[[symbol_col, name_col]].dropna().to_dict("records"):
                symbol = str(record[symbol_col]).strip().replace(".", "-")
                name = str(record[name_col]).strip()
                if not symbol or not name:
                    continue
                for alias in us_name_aliases(name):
                    rows.append({"component_name": alias, "component_code": symbol})
            break
    if rows:
        seed = pd.DataFrame(rows).drop_duplicates("component_name")
        seed.to_csv(path, index=False, encoding="utf-8-sig")
        return dict(zip(seed["component_name"], seed["component_code"]))
    return {}


def us_name_aliases(name: str) -> list[str]:
    base = re.sub(r"\s+", " ", name).strip()
    no_class = re.sub(r"\s+Class\s+[A-Z]$", "", base, flags=re.I).strip()
    no_punct = re.sub(r"[.,]", "", no_class)
    aliases = {base, base.upper(), no_class, no_class.upper(), no_punct.upper()}
    suffixes = [" CORP", " CORPORATION", " INC", " INC.", " CO", " COMPANY", " LTD", " PLC"]
    stem = re.sub(r"\b(CORPORATION|CORP|INC|COMPANY|CO|LTD|PLC)\.?$", "", no_punct, flags=re.I).strip()
    if stem:
        aliases.add(stem.upper())
        aliases.update((stem + suffix).upper() for suffix in suffixes)
    return sorted(a for a in aliases if a)


def save_name_map(name_map: dict[str, str], path: Path) -> None:
    rows = [{"component_name": name, "component_code": normalize_component_code_value(code)} for name, code in sorted(name_map.items())]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def resolve_component_code(name: str, name_map: dict[str, str], allow_yahoo: bool = True) -> tuple[str | None, bool]:
    if not name or name in {"원화현금", "설정현금액"} or looks_like_cash_or_future(name):
        return None, False
    if name in name_map:
        return name_map[name] or None, False
    code = None if looks_like_non_kr_name(name) else lookup_component_code(name)
    used_yahoo = False
    if not code:
        if not allow_yahoo:
            return None, False
        code = lookup_yahoo_symbol(name)
        used_yahoo = True
    if code:
        name_map[name] = code
    return code, used_yahoo


def looks_like_cash_or_future(name: str) -> bool:
    lowered = name.lower()
    blocked = [
        "선물",
        "현금",
        "cash",
        "deposit",
        "futures",
        "future",
        "treasury bill",
        "t-bill",
        "usd ",
        "krw ",
        "jpy ",
    ]
    return any(token in lowered for token in blocked)


def lookup_component_code(name: str) -> str | None:
    search_name = cleanup_component_search_name(name)
    try:
        resp = SESSION.get(
            WISE_AUTOCOMPLETE_URL,
            params={"searchTyp": "S", "searchVal": search_name},
            timeout=6,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None
    if not isinstance(data, list):
        return None
    normalized = normalize_name(name)
    candidates = []
    for item in data:
        item_name = str(item.get("item") or "").strip()
        item_code = normalize_code(item.get("item_cd") or "")
        if not re.fullmatch(r"\d{6}", item_code):
            continue
        score = 0
        if normalize_name(item_name) == normalized:
            score = 3
        elif normalized and normalized in normalize_name(str(item.get("item_nm") or "")):
            score = 2
        elif normalize_name(item_name) in normalized or normalized in normalize_name(item_name):
            score = 1
        candidates.append((score, item_code))
    candidates = [c for c in candidates if c[0] > 0]
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def looks_like_non_kr_name(name: str) -> bool:
    has_latin = bool(re.search(r"[A-Za-z]", name))
    has_hangul = bool(re.search(r"[\uac00-\ud7a3]", name))
    return has_latin and not has_hangul


def lookup_yahoo_symbol(name: str) -> str | None:
    if not re.search(r"[A-Za-z]", name):
        return None
    query = cleanup_yahoo_query(name)
    if not query:
        return None
    try:
        resp = SESSION.get(
            "https://query2.finance.yahoo.com/v1/finance/search",
            params={"q": query, "quotes_count": 8, "news_count": 0},
            timeout=6,
        )
        resp.raise_for_status()
        quotes = resp.json().get("quotes") or []
    except Exception:
        return None
    blocked_types = {"FUTURE", "CURRENCY", "CRYPTOCURRENCY"}
    for quote in quotes:
        symbol = str(quote.get("symbol") or "").strip()
        quote_type = str(quote.get("quoteType") or "").upper()
        exchange = str(quote.get("exchange") or "").upper()
        if not symbol or quote_type in blocked_types:
            continue
        if exchange in {"KSC", "KOE"}:
            continue
        if re.search(r"[-=]", symbol):
            continue
        if quote_type in {"EQUITY", "ETF"}:
            return symbol
    return None


def cleanup_yahoo_query(name: str) -> str:
    query = cleanup_component_search_name(name)
    query = re.sub(r"\b(CORP|INC|LTD|PLC|SA|NV|AG|CO)\.?$", "", query, flags=re.I).strip()
    return query


def cleanup_component_search_name(name: str) -> str:
    query = re.sub(r"^\((dup|선|합성|현물)\)\s*", "", name, flags=re.I).strip()
    query = re.sub(r"^\s*(dup)\s+", "", query, flags=re.I).strip()
    query = re.sub(r"\s+-[A-Z]$", "", query).strip()
    query = re.sub(r"\s+", " ", query).strip()
    return query


def normalize_name(value: str) -> str:
    return re.sub(r"\s+|\(.*?\)|\[.*?\]", "", value).upper()


def discover_local_ticker_suffixes() -> dict[str, str]:
    suffixes: dict[str, str] = {}
    for directory in SOURCE_PRICE_DIRS:
        if not directory.exists():
            continue
        for path in directory.glob("*.csv"):
            stem = path.stem.replace("_", ".")
            match = re.match(r"^(\d{6})\.(KS|KQ)$", stem, re.I)
            if match:
                suffixes.setdefault(match.group(1), match.group(2).upper())
    return suffixes


def kr_ticker(code: str, suffix_map: dict[str, str]) -> str:
    suffix = suffix_map.get(code, "KS")
    return f"{code}.{suffix}"


def component_ticker(code: str, suffix_map: dict[str, str]) -> str:
    code = normalize_component_code_value(code)
    if re.fullmatch(r"\d{6}", code):
        return kr_ticker(code, suffix_map)
    return code


def normalize_component_code_value(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text or text == "NAN":
        return ""
    if re.fullmatch(r"A\d{6}", text):
        return text[1:]
    if re.fullmatch(r"\d{6}(KS|KQ|SZ|SS)", text):
        return f"{text[:6]}.{text[6:]}"
    if re.fullmatch(r"\d{4,5}HK", text):
        return f"{text[:-2].zfill(4)}.HK"
    if re.fullmatch(r"\d{4}T", text):
        return f"{text[:4]}.T"
    if re.fullmatch(r"\d{4,6}\.(KS|KQ|SZ|SS|HK|T)", text):
        return text
    return text.replace("/", "-")


def safe_filename(ticker: str) -> str:
    return ticker.replace(".", "_").replace("/", "_")


def copy_local_ohlcv(tickers: list[str], out_dir: Path) -> int:
    copied = 0
    for ticker in tickers:
        out_path = out_dir / f"{safe_filename(ticker)}.csv"
        if out_path.exists():
            copied += 1
            continue
        candidates = [
            directory / f"{ticker}.csv"
            for directory in SOURCE_PRICE_DIRS
        ] + [
            directory / f"{safe_filename(ticker)}.csv"
            for directory in SOURCE_PRICE_DIRS
        ]
        source = next((p for p in candidates if p.exists()), None)
        if source is None:
            continue
        shutil.copy2(source, out_path)
        copied += 1
    return copied


def download_missing_ohlcv(tickers: list[str], out_dir: Path, start: str, end: str | None, batch_size: int) -> list[str]:
    downloaded: list[str] = []
    failed: list[str] = []
    workers = max(1, min(batch_size, 24))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_one_yahoo_chart, ticker, out_dir, start, end): ticker for ticker in tickers}
        for idx, future in enumerate(as_completed(futures), start=1):
            ticker = futures[future]
            try:
                saved_ticker = future.result()
            except Exception:
                saved_ticker = None
            if saved_ticker:
                downloaded.append(saved_ticker)
            else:
                failed.append(ticker)
            if idx % 100 == 0:
                print(f"[prices] completed={idx}/{len(tickers)} downloaded={len(downloaded)} failed={len(failed)}", flush=True)
    fail_out = ROOT / "outputs" / "component_ohlcv_failed_tickers.csv"
    fail_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticker": failed}).to_csv(fail_out, index=False, encoding="utf-8-sig")
    return downloaded


def download_one_yahoo_chart(ticker: str, out_dir: Path, start: str, end: str | None) -> str | None:
    candidates = [ticker]
    alt = flip_kr_suffix(ticker)
    if alt != ticker:
        candidates.append(alt)
    for candidate in candidates:
        frame = yahoo_chart_ohlcv(candidate, start, end)
        if frame.empty:
            continue
        frame.to_csv(out_dir / f"{safe_filename(candidate)}.csv", encoding="utf-8-sig")
        return candidate
    return None


def yahoo_chart_ohlcv(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    period1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    period2 = int((pd.Timestamp(end, tz="UTC") if end else pd.Timestamp(datetime.now(timezone.utc))).timestamp())
    try:
        resp = SESSION.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
            params={"period1": period1, "period2": period2, "interval": "1d", "events": "history", "includeAdjustedClose": "true"},
            timeout=12,
        )
        resp.raise_for_status()
        result = (resp.json().get("chart", {}).get("result") or [None])[0]
    except Exception:
        return pd.DataFrame()
    if not result or not result.get("timestamp"):
        return pd.DataFrame()
    quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    adj = ((result.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose")
    idx = pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert(None).normalize()
    out = pd.DataFrame(
        {
            "Open": quote.get("open"),
            "High": quote.get("high"),
            "Low": quote.get("low"),
            "Close": quote.get("close"),
            "Volume": quote.get("volume"),
        },
        index=idx,
    )
    if adj is not None:
        close = pd.to_numeric(out["Close"], errors="coerce")
        adj_close = pd.Series(adj, index=idx, dtype="float64")
        factor = (adj_close / close).replace([float("inf"), -float("inf")], pd.NA)
        for col in ["Open", "High", "Low", "Close"]:
            out[col] = pd.to_numeric(out[col], errors="coerce") * factor
    out = out.dropna(subset=["Close"])
    out.index.name = "Date"
    return out


def extract_ticker_frame(data: pd.DataFrame, ticker: str, batch_len: int) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        if ticker in data.columns.get_level_values(0):
            return data[ticker].copy()
        return pd.DataFrame()
    if batch_len == 1:
        return data.copy()
    return pd.DataFrame()


def normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(-1)
    columns = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in out.columns]
    if "Close" not in columns:
        return pd.DataFrame()
    out = out[columns].dropna(subset=["Close"])
    out.index = pd.to_datetime(out.index)
    out.index.name = "Date"
    return out


def flip_kr_suffix(ticker: str) -> str:
    if ticker.endswith(".KS"):
        return ticker[:-3] + ".KQ"
    if ticker.endswith(".KQ"):
        return ticker[:-3] + ".KS"
    return ticker


if __name__ == "__main__":
    main()
