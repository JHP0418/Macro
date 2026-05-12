from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from etf_leadership_model.features import make_features  # noqa: E402
import train_static_etf_leadership_v3 as v3  # noqa: E402


OUT = ROOT / "outputs" / "gaps_long_lived_etf_leadership_latest"
TABLES = OUT / "tables"
CACHE = ROOT / "data" / "gaps_long_lived_cache"

REGRESSION_FEATURES = [
    "reg_coef_high_proximity",
    "reg_coef_component_return_60d",
    "reg_coef_component_rs_60d",
    "reg_r2",
    "reg_residual_dispersion",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Long-lived DB GAPS ETF leadership backtest using static current holdings and full component techniques.")
    p.add_argument("--universe", default=str(ROOT / "data" / "etf_universe_leadership.csv"))
    p.add_argument("--holdings", default=str(ROOT / "data" / "etf_holdings_static_2019_repaired.csv"))
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default="2026-05-11")
    p.add_argument("--min-first-date", default="2015-01-01")
    p.add_argument("--min-etf-obs", type=int, default=1200)
    p.add_argument("--feature-frequency", default="W-FRI")
    p.add_argument("--train-end", default="2018-12-31")
    p.add_argument("--valid-end", default="2021-12-31")
    p.add_argument("--min-date", default="2012-01-01")
    p.add_argument("--min-holdings", type=int, default=2)
    p.add_argument("--min-group-size", type=int, default=2)
    p.add_argument("--top-k-list", default="1,2,3,5")
    p.add_argument("--chunk-size", type=int, default=80)
    p.add_argument("--force-download", action="store_true")
    return p.parse_args()


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)


def safe_name(ticker: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(ticker))


def download_prices_chunked(tickers: list[str], start: str, end: str | None, cache_path: Path, chunk_size: int, force: bool = False) -> pd.DataFrame:
    if cache_path.exists() and not force:
        return pd.read_csv(cache_path, parse_dates=["date"]).set_index("date").sort_index()
    tickers = sorted({str(t).strip() for t in tickers if str(t).strip() and str(t).strip().lower() != "nan"})
    pieces: list[pd.DataFrame] = []
    failed: list[str] = []
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        print(f"[download] {i + 1}-{min(i + chunk_size, len(tickers))}/{len(tickers)}", flush=True)
        try:
            raw = yf.download(
                chunk,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
                group_by="column",
            )
            close = extract_close(raw, chunk)
            if not close.empty:
                pieces.append(close)
        except Exception as exc:
            print(f"[download] failed chunk size={len(chunk)} error={exc}", flush=True)
            failed.extend(chunk)
    if pieces:
        prices = pd.concat(pieces, axis=1)
        prices = prices.loc[:, ~prices.columns.duplicated()].sort_index()
    else:
        prices = pd.DataFrame()
    prices.index.name = "date"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(cache_path, encoding="utf-8-sig")
    pd.DataFrame({"ticker": sorted(set(failed))}).to_csv(cache_path.with_name(cache_path.stem + "_failed.csv"), index=False, encoding="utf-8-sig")
    return prices


def extract_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"].copy()
        elif "Adj Close" in raw.columns.get_level_values(0):
            close = raw["Adj Close"].copy()
        else:
            return pd.DataFrame()
    else:
        if "Close" in raw:
            close = raw[["Close"]].copy()
        elif "Adj Close" in raw:
            close = raw[["Adj Close"]].copy()
        else:
            return pd.DataFrame()
        close.columns = tickers[:1]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.columns = [str(c).strip() for c in close.columns]
    return close.apply(pd.to_numeric, errors="coerce").dropna(how="all")


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    universe = pd.read_csv(args.universe)
    holdings = pd.read_csv(args.holdings)
    universe["etf_ticker"] = universe["etf_ticker"].astype(str).str.strip()
    universe["benchmark_ticker"] = universe["benchmark_ticker"].astype(str).str.strip()
    holdings["etf_ticker"] = holdings["etf_ticker"].astype(str).str.strip()
    holdings["component_ticker"] = holdings["component_ticker"].astype(str).str.strip()
    holdings["weight"] = pd.to_numeric(holdings["weight"], errors="coerce")
    holdings = holdings.dropna(subset=["etf_ticker", "component_ticker", "weight"])
    holdings = holdings[(holdings["weight"] > 0) & (holdings["weight"] <= 1)].copy()
    return universe, holdings


def select_long_lived_universe(universe: pd.DataFrame, etf_prices: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    min_first = pd.Timestamp(args.min_first_date)
    for rec in universe.to_dict("records"):
        ticker = rec["etf_ticker"]
        if ticker not in etf_prices.columns:
            continue
        s = etf_prices[ticker].dropna()
        if s.empty:
            continue
        first = s.index.min()
        obs = int(s.loc[pd.Timestamp(args.start) :].shape[0])
        if first <= min_first and obs >= args.min_etf_obs:
            rows.append({**rec, "first_price_date": first.date().isoformat(), "price_obs": obs})
    out = pd.DataFrame(rows)
    return out.sort_values(["group", "etf_ticker"]).reset_index(drop=True) if not out.empty else out


def static_holdings_from_start(holdings: pd.DataFrame, selected: pd.DataFrame, start: str) -> pd.DataFrame:
    keep = set(selected["etf_ticker"].astype(str))
    out = holdings[holdings["etf_ticker"].isin(keep)].copy()
    out["date"] = pd.Timestamp(start)
    out = out.groupby(["date", "etf_ticker", "component_ticker"], as_index=False).agg(
        weight=("weight", "sum"),
        component_name=("component_name", "first") if "component_name" in out.columns else ("component_ticker", "first"),
    )
    out["weight"] = out.groupby(["date", "etf_ticker"])["weight"].transform(lambda s: s / s.sum() if s.sum() > 0 else s)
    return out


def add_internal_regression_features(features: pd.DataFrame, holdings: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return features
    print("[regression] building component internal regression features", flush=True)
    px = prices.sort_index().ffill(limit=5)
    ret20 = px.pct_change(20)
    ret60 = px.pct_change(60)
    high252 = px.rolling(252, min_periods=120).max()
    holdings_by = {
        etf: h.groupby("component_ticker", as_index=False)["weight"].sum().assign(
            weight=lambda x: x["weight"] / x["weight"].sum() if x["weight"].sum() > 0 else x["weight"]
        )
        for etf, h in holdings.groupby("etf_ticker")
    }
    rows = []
    for idx, row in features[["date", "etf_ticker", "benchmark_ticker"]].iterrows():
        if idx % 2500 == 0:
            print(f"[regression] row {idx:,}/{features.shape[0]:,}", flush=True)
        date = pd.Timestamp(row["date"])
        etf = str(row["etf_ticker"])
        bench = str(row["benchmark_ticker"])
        rows.append(regression_row(date, etf, bench, holdings_by, px, ret20, ret60, high252))
    reg = pd.DataFrame(rows, index=features.index)
    return pd.concat([features.reset_index(drop=True), reg.reset_index(drop=True)], axis=1)


def regression_row(
    date: pd.Timestamp,
    etf: str,
    bench: str,
    holdings_by: dict[str, pd.DataFrame],
    prices: pd.DataFrame,
    ret20: pd.DataFrame,
    ret60: pd.DataFrame,
    high252: pd.DataFrame,
) -> dict[str, float]:
    h = holdings_by.get(etf)
    if h is None or h.empty or date not in prices.index or bench not in prices.columns:
        return empty_regression()
    comps = [c for c in h["component_ticker"].astype(str) if c in prices.columns]
    if len(comps) < 5:
        return empty_regression()
    w = h.set_index("component_ticker")["weight"].reindex(comps).astype(float)
    y = ret20.loc[date, comps].astype(float)
    hp = (prices.loc[date, comps] / high252.loc[date, comps]).astype(float)
    c_ret60 = ret60.loc[date, comps].astype(float)
    b_ret60 = float(ret60.at[date, bench]) if bench in ret60.columns and pd.notna(ret60.at[date, bench]) else np.nan
    c_rs60 = c_ret60 - b_ret60
    x = pd.DataFrame(
        {
            "hp": hp,
            "ret60": c_ret60,
            "rs60": c_rs60,
        }
    )
    data = pd.concat([y.rename("y"), x, w.rename("w")], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if data.shape[0] < 5 or data["y"].std() == 0:
        return empty_regression()
    xmat = data[["hp", "ret60", "rs60"]].to_numpy(dtype=float)
    xmat = (xmat - np.nanmean(xmat, axis=0)) / np.where(np.nanstd(xmat, axis=0) == 0, 1, np.nanstd(xmat, axis=0))
    xmat = np.column_stack([np.ones(data.shape[0]), xmat])
    yvec = data["y"].to_numpy(dtype=float)
    weights = data["w"].to_numpy(dtype=float)
    weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
    sw = np.sqrt(weights)
    xw = xmat * sw[:, None]
    yw = yvec * sw
    try:
        beta = np.linalg.lstsq(xw, yw, rcond=None)[0]
    except Exception:
        return empty_regression()
    pred = xmat @ beta
    resid = yvec - pred
    sst = float(np.sum(weights * (yvec - np.average(yvec, weights=weights)) ** 2))
    sse = float(np.sum(weights * resid**2))
    r2 = 1.0 - sse / sst if sst > 0 else np.nan
    return {
        "reg_coef_high_proximity": float(beta[1]),
        "reg_coef_component_return_60d": float(beta[2]),
        "reg_coef_component_rs_60d": float(beta[3]),
        "reg_r2": float(r2) if pd.notna(r2) else np.nan,
        "reg_residual_dispersion": float(np.sqrt(max(sse, 0))),
    }


def empty_regression() -> dict[str, float]:
    return {k: np.nan for k in REGRESSION_FEATURES}


def run_v3_like_training(scored: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    class Obj:
        pass

    cfg = Obj()
    cfg.min_date = args.min_date
    cfg.min_holdings = args.min_holdings
    cfg.min_group_size = args.min_group_size
    filtered = v3.filter_quality(scored, cfg)
    scored2 = v3.add_group_labels(v3.add_rule_scores(filtered))
    train, valid, test = v3.split_frame(scored2, args.train_end, args.valid_end)

    rank_features_20d = v3.FEATURES_1M + REGRESSION_FEATURES
    meta_features_1w = v3.FEATURES_1W + v3.STRUCTURE_COLUMNS + v3.ENTRY_CONTEXT_1W + REGRESSION_FEATURES
    meta_features_1m = v3.FEATURES_1M + v3.STRUCTURE_COLUMNS + v3.ENTRY_CONTEXT_1M + REGRESSION_FEATURES

    pred_5d, imp_entry_5d = v3.train_entry_model(train, valid, test, "entry_5d_label", meta_features_1w, "entry_prob_5d")
    pred_20d_ranker, imp_ranker_20d = v3.train_ranker(train, valid, test, "label_20D_group_rank_int", rank_features_20d)
    pred_20d_ranker = v3.add_score_context(pred_20d_ranker, "rule_20d_score", "rule_20d")
    pred_20d, imp_entry_20d = v3.train_entry_model(
        pred_20d_ranker[pred_20d_ranker["split"].eq("train")],
        pred_20d_ranker[pred_20d_ranker["split"].eq("valid")],
        pred_20d_ranker[pred_20d_ranker["split"].eq("test")],
        "entry_20d_label",
        meta_features_1m,
        "entry_prob_20d",
    )
    pred_5d, pred_20d = v3.add_blends(pred_5d, pred_20d)
    threshold_5d = v3.optimize_threshold(pred_5d, "entry_adjusted_5d_score", "1W", 5, "entry_prob_5d")
    threshold_20d = v3.optimize_threshold(pred_20d, "entry_adjusted_20d_score", "1M", 5, "entry_prob_20d")

    summaries = []
    raws = []
    for top_k in [int(x) for x in args.top_k_list.split(",") if x.strip()]:
        specs = [
            (pred_5d, "rule_5d_score", "1W", None, None),
            (pred_5d, "entry_adjusted_5d_score", "1W", "entry_prob_5d", threshold_5d),
            (pred_20d, "rule_20d_score", "1M", None, None),
            (pred_20d, "ranker_score", "1M", None, None),
            (pred_20d, "blend_20d_score", "1M", None, None),
            (pred_20d, "entry_adjusted_20d_score", "1M", "entry_prob_20d", threshold_20d),
        ]
        for frame, score_col, horizon, prob_col, threshold in specs:
            raw_bt, summary = v3.backtest(frame, score_col, horizon, top_k, "test", prob_col, threshold)
            summary["model"] = score_col
            summary["entry_threshold"] = threshold
            summary["regression_features"] = len(REGRESSION_FEATURES)
            summaries.append(summary)
            raw_bt["model"] = score_col
            raw_bt["entry_threshold"] = threshold
            raws.append(raw_bt)
    summary_df = pd.DataFrame(summaries).sort_values(["horizon", "Sharpe"], ascending=[True, False])
    raw_df = pd.concat(raws, ignore_index=True) if raws else pd.DataFrame()
    importance = pd.concat(
        [
            imp_entry_5d.assign(model="entry_5d"),
            imp_ranker_20d.assign(model="ranker_20d"),
            imp_entry_20d.assign(model="entry_20d"),
        ],
        ignore_index=True,
    )
    predictions = pd.concat(
        [
            pred_5d.assign(prediction_horizon="1W"),
            pred_20d.assign(prediction_horizon="1M"),
        ],
        ignore_index=True,
        sort=False,
    )
    return scored2, predictions, summary_df, raw_df, importance


def build_current_scores(
    selected: pd.DataFrame,
    selected_holdings: pd.DataFrame,
    prices: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    current_features = make_features(
        selected[["etf_ticker", "market", "benchmark_ticker"]],
        selected_holdings,
        prices,
        frequency=args.feature_frequency,
        require_forward_targets=False,
    )
    if current_features.empty:
        return current_features
    latest_date = pd.to_datetime(current_features["date"]).max()
    current = current_features[pd.to_datetime(current_features["date"]).eq(latest_date)].copy()
    current = add_internal_regression_features(current, selected_holdings, prices)
    current = v3.attach_universe(current, Path(args.universe))
    current["date"] = pd.to_datetime(current["date"])
    numeric = sorted(set(v3.FEATURES_1W + v3.FEATURES_1M + v3.STRUCTURE_COLUMNS + REGRESSION_FEATURES))
    for col in numeric:
        if col in current.columns:
            current[col] = pd.to_numeric(current[col], errors="coerce")
    current = current[current["holding_count"].ge(args.min_holdings)].copy()
    if current.empty:
        return current
    group_size = current.groupby(["date", "model_group"])["etf_ticker"].transform("size")
    current = current[group_size.ge(args.min_group_size)].reset_index(drop=True)
    current = v3.add_rule_scores(current)
    current["entry_adjusted_5d_score"] = current["rule_5d_score"]
    current["entry_adjusted_20d_score"] = current["rule_20d_score"]
    current["ranker_score"] = np.nan
    current["entry_prob_5d"] = np.nan
    current["entry_prob_20d"] = np.nan
    current["prediction_source"] = "current_features_without_forward_targets"
    return current.sort_values(["date", "rule_20d_score"], ascending=[True, False]).reset_index(drop=True)


def write_report(selected: pd.DataFrame, summary: pd.DataFrame, metadata: dict) -> None:
    lines = [
        "# DB GAPS Long-Lived ETF Leadership Backtest",
        "",
        "현재 DB GAPS ETF 중 2010년대부터 가격 히스토리가 충분한 ETF만 골라, 현재 구성종목을 과거에도 동일하다고 가정한 장기 근사 백테스트입니다.",
        "",
        "## 중요한 한계",
        "",
        "- 과거 holdings가 없으므로 실제 과거 구성종목이 아니라 현재 구성종목을 과거로 고정했습니다.",
        "- 이 방식은 현재 스크리닝 로직의 장기 내구성을 보는 검증이지, 완전한 point-in-time holdings 검증은 아닙니다.",
        "- 구성종목 가격이 과거에 없던 경우 해당 시점 breadth/HP/회귀 피처에는 자연스럽게 결측이 반영됩니다.",
        "",
        "## Metadata",
        "",
        "```json",
        json.dumps(metadata, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Long-Lived ETF Universe",
        "",
        selected[["etf_ticker", "name", "group", "benchmark_ticker", "first_price_date", "price_obs"]].to_markdown(index=False),
        "",
        "## Backtest Summary",
        "",
        summary.to_markdown(index=False),
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dirs()
    universe, holdings = load_inputs(args)

    etf_bench = sorted(set(universe["etf_ticker"]).union(universe["benchmark_ticker"]))
    etf_price_cache = CACHE / f"gaps_etf_benchmark_prices_{args.start}_{args.end}.csv".replace(":", "-")
    etf_prices = download_prices_chunked(etf_bench, args.start, args.end, etf_price_cache, args.chunk_size, args.force_download)
    selected = select_long_lived_universe(universe, etf_prices, args)
    if selected.empty:
        raise RuntimeError("No long-lived DB GAPS ETFs selected. Relax --min-first-date or --min-etf-obs.")
    selected_holdings = static_holdings_from_start(holdings, selected, args.start)
    required = sorted(set(selected["etf_ticker"]).union(selected["benchmark_ticker"]).union(selected_holdings["component_ticker"]))
    price_cache = CACHE / f"gaps_long_lived_components_prices_{args.start}_{args.end}.csv".replace(":", "-")
    prices = download_prices_chunked(required, args.start, args.end, price_cache, args.chunk_size, args.force_download)
    prices = prices.loc[:, ~prices.columns.duplicated()].sort_index()

    print(f"[features] selected_etfs={selected.shape[0]} holdings={selected_holdings.shape[0]} price_cols={prices.shape[1]}", flush=True)
    features = make_features(selected[["etf_ticker", "market", "benchmark_ticker"]], selected_holdings, prices, frequency=args.feature_frequency)
    if features.empty:
        raise RuntimeError("No features generated for long-lived GAPS ETFs.")
    features = add_internal_regression_features(features, selected_holdings, prices)
    features = v3.attach_universe(features, Path(args.universe))
    scored, predictions, summary, raw_bt, importance = run_v3_like_training(features, args)
    current_scores = build_current_scores(selected, selected_holdings, prices, args)

    selected.to_csv(TABLES / "long_lived_gaps_etf_universe.csv", index=False, encoding="utf-8-sig")
    selected_holdings.to_csv(TABLES / "static_holdings_used_from_start.csv", index=False, encoding="utf-8-sig")
    prices.to_csv(TABLES / "long_lived_price_matrix.csv", index_label="date", encoding="utf-8-sig")
    features.to_csv(TABLES / "long_lived_features_with_regression.csv", index=False, encoding="utf-8-sig")
    scored.to_csv(TABLES / "long_lived_scored_features.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(TABLES / "long_lived_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(TABLES / "long_lived_backtest_summary.csv", index=False, encoding="utf-8-sig")
    raw_bt.to_csv(TABLES / "long_lived_backtest_trades.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(TABLES / "long_lived_feature_importance.csv", index=False, encoding="utf-8-sig")
    current_scores.to_csv(TABLES / "long_lived_current_scores.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "start": args.start,
        "end": args.end,
        "min_first_date": args.min_first_date,
        "min_etf_obs": args.min_etf_obs,
        "selected_etfs": int(selected.shape[0]),
        "static_holding_rows": int(selected_holdings.shape[0]),
        "price_columns": int(prices.shape[1]),
        "feature_rows": int(features.shape[0]),
        "train_end": args.train_end,
        "valid_end": args.valid_end,
        "test_start": str(pd.Timestamp(args.valid_end) + pd.Timedelta(days=1)),
        "techniques": [
            "ETF relative strength",
            "component high proximity",
            "component breadth MA60/MA200",
            "component relative momentum",
            "concentration penalty",
            "component internal weighted regression",
            "basket-aware rule scores",
            "LightGBM LGBMRanker",
            "entry meta-model",
        ],
    }
    write_report(selected, summary, metadata)
    print(summary.head(30).to_string(index=False))
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
