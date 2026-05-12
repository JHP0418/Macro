from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_providers import AlphaVantageProvider, DartProvider, FinnhubProvider, FmpProvider, FredProvider, KrxInvestorFlowProvider  # noqa: E402
from screener_agent import load_env_file  # noqa: E402


SENSITIVE_QUERY_RE = re.compile(r"(?i)(crtfc_key|apikey|api_key|token|key)=([^&\s]+)")


def safe_error(exc: Exception) -> str:
    return SENSITIVE_QUERY_RE.sub(r"\1=<redacted>", str(exc))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Harvest free API caches with progress logs and resumable cache usage.")
    parser.add_argument("--panel", type=Path, default=ROOT / "output_monthly_walkforward_v3_analysis" / "mature_monthly_panel.csv")
    parser.add_argument("--providers", type=str, default="fmp,finnhub,alpha,fred,dart,krx")
    parser.add_argument("--market", choices=["all", "US", "KR"], default="all")
    parser.add_argument("--limit-symbols", type=int, default=0)
    parser.add_argument("--limit-dates", type=int, default=0)
    parser.add_argument("--progress-seconds", type=int, default=180)
    parser.add_argument("--status-file", type=Path, default=ROOT / "output_data_harvest" / "api_harvest_status.json")
    return parser.parse_args()


def load_panel(path: Path) -> pd.DataFrame:
    cols = ["market_case", "snapshot_date", "symbol"]
    panel = pd.read_csv(path, usecols=cols)
    panel["snapshot_date"] = panel["snapshot_date"].astype(str)
    return panel.dropna(subset=["market_case", "snapshot_date", "symbol"])


def ordered_symbols(panel: pd.DataFrame, market: str) -> list[str]:
    side = panel[panel["market_case"].eq(market)].copy()
    side = side.sort_values(["snapshot_date", "symbol"], ascending=[False, True])
    return list(dict.fromkeys(side["symbol"].astype(str).tolist()))


def ordered_symbol_dates(panel: pd.DataFrame, market: str) -> list[tuple[str, str]]:
    side = panel[panel["market_case"].eq(market)].copy()
    side = side[["symbol", "snapshot_date"]].drop_duplicates()
    side = side.sort_values(["snapshot_date", "symbol"], ascending=[False, True])
    return [(str(row.symbol), str(row.snapshot_date)) for row in side.itertuples(index=False)]


def report(
    task: str,
    idx: int,
    total: int,
    started: float,
    last_report: float,
    status_file: Path,
    item: str,
    force: bool = False,
    progress_seconds: int = 180,
) -> float:
    elapsed = time.perf_counter() - started
    if not force and idx != 1 and idx < total and elapsed - last_report < progress_seconds:
        return last_report
    rate = idx / elapsed if elapsed > 0 else 0.0
    eta = (total - idx) / rate if rate > 0 else None
    msg = (
        f"{task} {idx}/{total} item={item} elapsed={elapsed/60:.1f}m "
        f"rate={rate*60:.1f}/min eta={(eta/60 if eta else 0):.1f}m"
    )
    print(msg, flush=True)
    status_file.parent.mkdir(parents=True, exist_ok=True)
    status_file.write_text(
        json.dumps(
            {
                "task": task,
                "processed": idx,
                "total": total,
                "item": item,
                "elapsed_minutes": elapsed / 60,
                "items_per_minute": rate * 60,
                "eta_minutes": eta / 60 if eta else None,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return elapsed


def harvest_us(provider_name: str, symbols: list[str], status_file: Path, progress_seconds: int) -> None:
    as_of = pd.Timestamp.today().normalize()
    if provider_name == "fmp":
        provider: Any = FmpProvider(cache_dir=ROOT / ".cache" / "us_v11" / "fmp")
        fn = lambda symbol: provider.metrics_for_symbol(symbol, as_of, include_estimates=True, include_current_profile=True)
    elif provider_name == "finnhub":
        provider = FinnhubProvider(cache_dir=ROOT / ".cache" / "us_v11" / "finnhub")
        fn = lambda symbol: provider.metrics_for_symbol(symbol, as_of)
    elif provider_name == "alpha":
        provider = AlphaVantageProvider(cache_dir=ROOT / ".cache" / "us_v11" / "alpha_vantage")
        fn = lambda symbol: provider.metrics_for_symbol(symbol, as_of, include_estimates=True, include_current_overview=True)
    elif provider_name == "fred":
        provider = FredProvider(cache_dir=ROOT / ".cache" / "us_v11" / "fred")
        dates = sorted({pd.Timestamp.today().normalize() - pd.DateOffset(months=m) for m in range(0, 76)}, reverse=True)
        started = time.perf_counter()
        last = 0.0
        total = len(dates) * 4
        idx = 0
        for date in dates:
            for series in ["DGS10", "DFII10", "DGS2", "FEDFUNDS"]:
                idx += 1
                try:
                    provider.observation(series, date)
                except Exception as exc:  # noqa: BLE001
                    print(f"fred error {series} {date.date()}: {safe_error(exc)}", flush=True)
                last = report("harvest_fred", idx, total, started, last, status_file, f"{series}:{date.date()}", progress_seconds=progress_seconds)
        report("harvest_fred", total, total, started, last, status_file, "done", force=True, progress_seconds=progress_seconds)
        return
    else:
        return

    started = time.perf_counter()
    last = 0.0
    total = len(symbols)
    for idx, symbol in enumerate(symbols, start=1):
        try:
            fn(symbol)
        except Exception as exc:  # noqa: BLE001
            print(f"{provider_name} error {symbol}: {safe_error(exc)}", flush=True)
        last = report(f"harvest_{provider_name}", idx, total, started, last, status_file, symbol, progress_seconds=progress_seconds)
    report(f"harvest_{provider_name}", total, total, started, last, status_file, "done", force=True, progress_seconds=progress_seconds)


def harvest_kr(provider_name: str, symbol_dates: list[tuple[str, str]], status_file: Path, progress_seconds: int) -> None:
    if provider_name == "dart":
        provider: Any = DartProvider()
        fn = lambda symbol, date: provider.metrics_for_symbol(symbol, pd.Timestamp(date))
    elif provider_name == "krx":
        provider = KrxInvestorFlowProvider()
        fn = lambda symbol, date: provider.metrics_for_symbol(symbol, pd.Timestamp(date), None, None)
    else:
        return
    started = time.perf_counter()
    last = 0.0
    total = len(symbol_dates)
    for idx, (symbol, date) in enumerate(symbol_dates, start=1):
        try:
            fn(symbol, date)
        except Exception as exc:  # noqa: BLE001
            print(f"{provider_name} error {symbol} {date}: {safe_error(exc)}", flush=True)
        last = report(f"harvest_{provider_name}", idx, total, started, last, status_file, f"{symbol}:{date}", progress_seconds=progress_seconds)
    report(f"harvest_{provider_name}", total, total, started, last, status_file, "done", force=True, progress_seconds=progress_seconds)


def main() -> None:
    args = parse_args()
    load_env_file(ROOT / ".env")
    panel = load_panel(args.panel)
    providers = [p.strip().lower() for p in args.providers.split(",") if p.strip()]
    us_symbols = ordered_symbols(panel, "US") if args.market in {"all", "US"} else []
    kr_symbol_dates = ordered_symbol_dates(panel, "KR") if args.market in {"all", "KR"} else []
    if args.limit_symbols:
        us_symbols = us_symbols[: args.limit_symbols]
        allowed = {symbol for symbol, _ in kr_symbol_dates[: args.limit_symbols]}
        kr_symbol_dates = [row for row in kr_symbol_dates if row[0] in allowed]
    if args.limit_dates:
        dates = sorted({date for _, date in kr_symbol_dates}, reverse=True)[: args.limit_dates]
        kr_symbol_dates = [row for row in kr_symbol_dates if row[1] in dates]

    for provider in providers:
        if provider in {"fmp", "finnhub", "alpha", "fred"} and us_symbols:
            harvest_us(provider, us_symbols, args.status_file.with_name(f"{provider}_status.json"), args.progress_seconds)
        if provider in {"dart", "krx"} and kr_symbol_dates:
            harvest_kr(provider, kr_symbol_dates, args.status_file.with_name(f"{provider}_status.json"), args.progress_seconds)


if __name__ == "__main__":
    main()
