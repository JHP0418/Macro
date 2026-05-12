from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from harvest_etf_holdings_and_component_ohlcv import copy_local_ohlcv, download_one_yahoo_chart, safe_filename


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Repair ETF/component OHLCV cache and rebuild adjusted-close matrix.")
    p.add_argument("--universe", default=str(ROOT / "data" / "etf_universe_leadership.csv"))
    p.add_argument("--holdings", default=str(ROOT / "data" / "etf_holdings.csv"))
    p.add_argument("--component-cache", default=str(ROOT / ".cache" / "component_ohlcv"))
    p.add_argument("--prices-cache-out", default=str(ROOT / "outputs" / "etf_leadership_from_cache" / "prices_adj_close.csv"))
    p.add_argument("--summary-out", default=str(ROOT / "outputs" / "component_price_history_repair_summary.json"))
    p.add_argument("--failed-out", default=str(ROOT / "outputs" / "component_price_history_repair_failed.csv"))
    p.add_argument("--coverage-out", default=str(ROOT / "outputs" / "component_price_history_coverage.csv"))
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--min-rows", type=int, default=900)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--max-downloads", type=int, default=0, help="0 means no limit.")
    return p.parse_args()


def required_tickers(universe_path: Path, holdings_path: Path) -> list[str]:
    universe = pd.read_csv(universe_path)
    holdings = pd.read_csv(holdings_path)
    tickers = set(universe["etf_ticker"].dropna().astype(str))
    tickers.update(universe["benchmark_ticker"].dropna().astype(str))
    tickers.update(holdings["component_ticker"].dropna().astype(str))
    return sorted(t for t in tickers if t and t.lower() != "nan")


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
    frame = frame.dropna(subset=["Date", "Close"]).sort_values("Date")
    return frame.drop_duplicates("Date", keep="last")


def cache_path(cache_dir: Path, ticker: str) -> Path:
    return cache_dir / f"{safe_filename(ticker)}.csv"


def coverage_row(cache_dir: Path, ticker: str) -> dict:
    frame = read_ohlcv(cache_path(cache_dir, ticker))
    if frame.empty:
        return {"ticker": ticker, "exists": False, "rows": 0, "start": None, "end": None}
    return {
        "ticker": ticker,
        "exists": True,
        "rows": int(frame.shape[0]),
        "start": frame["Date"].min().date().isoformat(),
        "end": frame["Date"].max().date().isoformat(),
    }


def needs_repair(row: dict, start: str, min_rows: int) -> bool:
    if not row["exists"] or row["rows"] < min_rows:
        return True
    if row["start"] is None:
        return True
    return pd.Timestamp(row["start"]) > pd.Timestamp(start) + pd.Timedelta(days=120)


def download_many(tickers: list[str], cache_dir: Path, start: str, end: str | None, workers: int) -> tuple[list[str], list[str]]:
    downloaded: list[str] = []
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(download_one_yahoo_chart, ticker, cache_dir, start, end): ticker for ticker in tickers}
        for idx, future in enumerate(as_completed(futures), start=1):
            ticker = futures[future]
            try:
                saved = future.result()
            except Exception:
                saved = None
            if saved:
                downloaded.append(saved)
            else:
                failed.append(ticker)
            if idx % 100 == 0:
                print(f"[repair] downloaded={len(downloaded)} failed={len(failed)} completed={idx}/{len(tickers)}", flush=True)
    return downloaded, failed


def rebuild_price_matrix(tickers: list[str], cache_dir: Path, out_path: Path) -> tuple[pd.DataFrame, list[str]]:
    series: dict[str, pd.Series] = {}
    failed: list[str] = []
    for ticker in tickers:
        frame = read_ohlcv(cache_path(cache_dir, ticker))
        if frame.empty:
            failed.append(ticker)
            continue
        s = frame.set_index("Date")["Close"].sort_index()
        s.name = ticker
        series[ticker] = s
    matrix = pd.concat(series.values(), axis=1).sort_index() if series else pd.DataFrame()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.index.name = "date"
    matrix.to_csv(out_path, encoding="utf-8-sig")
    return matrix, failed


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.component_cache)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tickers = required_tickers(Path(args.universe), Path(args.holdings))

    copied = copy_local_ohlcv(tickers, cache_dir)
    before = pd.DataFrame([coverage_row(cache_dir, ticker) for ticker in tickers])
    to_download = [row["ticker"] for row in before.to_dict("records") if needs_repair(row, args.start, args.min_rows)]
    if args.max_downloads > 0:
        to_download = to_download[: args.max_downloads]

    downloaded, download_failed = download_many(to_download, cache_dir, args.start, args.end, args.workers) if to_download else ([], [])
    after = pd.DataFrame([coverage_row(cache_dir, ticker) for ticker in tickers])
    after.to_csv(args.coverage_out, index=False, encoding="utf-8-sig")

    matrix, matrix_failed = rebuild_price_matrix(tickers, cache_dir, Path(args.prices_cache_out))
    failed = sorted(set(download_failed).union(matrix_failed))
    pd.DataFrame({"ticker": failed}).to_csv(args.failed_out, index=False, encoding="utf-8-sig")

    summary = {
        "required_tickers": int(len(tickers)),
        "copied_or_existing": int(copied),
        "repair_candidates": int(len(to_download)),
        "downloaded": int(len(downloaded)),
        "download_failed": int(len(download_failed)),
        "price_matrix_rows": int(matrix.shape[0]),
        "price_matrix_columns": int(matrix.shape[1]),
        "matrix_failed_tickers": int(len(matrix_failed)),
        "prices_cache_out": str(Path(args.prices_cache_out)),
        "coverage_out": str(Path(args.coverage_out)),
        "failed_out": str(Path(args.failed_out)),
    }
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
