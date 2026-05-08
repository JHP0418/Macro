from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from macro_regime_asset_screener import (  # noqa: E402
    ASSETS,
    FRED_SERIES,
    YF_SERIES,
    asset_driver_fit,
    beta_alignment_score,
    blend_probability,
    clean_series,
    load_asset_histories,
    load_driver_series,
    make_driver_features,
    make_driver_panel,
    risk_score,
    rolling_driver_betas,
    safe_to_csv,
)
from rwkv_lppl_asset_screener import RWKV_OUT_DIR, dtcai_label  # noqa: E402
from walkforward_calibrate_rwkv_lppl import (  # noqa: E402
    add_institutional_score,
    first_valid_regime_date,
    forward_return,
    historical_technical_score,
    past_conditional_forward_stats,
    read_lppl_history,
)

OUT_DIR = ROOT / "outputs" / "weekly_screening_rank_backtest_latest"
FORWARD_1W = 5
FORWARD_1M = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weekly walk-forward screening OX and rank backtest.")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--input", type=Path, default=RWKV_OUT_DIR)
    parser.add_argument("--output", type=Path, default=OUT_DIR)
    parser.add_argument("--min-train-weeks", type=int, default=52)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables = args.output / "tables"
    reports = args.output / "reports"
    for path in (tables, reports):
        path.mkdir(parents=True, exist_ok=True)

    specs = FRED_SERIES + YF_SERIES
    raw, _ = load_driver_series(specs, args.start, args.skip_download)
    driver_panel = make_driver_panel(raw)
    driver_features = make_driver_features(driver_panel, specs)
    regime = read_table(args.input / "tables" / "rwkv_regime_history.csv", parse_dates=["Date"]).set_index("Date")
    lppl_hist = read_lppl_history(args.input / "tables" / "lppl_reliability_training_scored.csv")
    histories = load_asset_histories([asset.symbol for asset in ASSETS], args.start, args.skip_download)

    panel = build_weekly_panel(histories, driver_panel, driver_features, regime, lppl_hist)
    valid_start = first_valid_regime_date(regime)
    if valid_start is not None and not panel.empty:
        panel = panel[pd.to_datetime(panel["date"]).ge(valid_start)].reset_index(drop=True)
    calibrated = add_expanding_weekly_calibration(panel, args.min_train_weeks)
    calibrated = add_institutional_score(calibrated)
    calibrated = add_ranks(calibrated)
    ox = ox_metrics(calibrated)
    rank_metrics = ranking_metrics(calibrated, args.top_k)
    strategy = topk_strategy(calibrated, args.top_k)
    summary = strategy_summary(strategy)
    current = current_week_snapshot(calibrated)

    safe_to_csv(panel, tables / "weekly_raw_panel.csv")
    safe_to_csv(calibrated, tables / "weekly_calibrated_rank_panel.csv")
    safe_to_csv(ox, tables / "weekly_ox_metrics.csv")
    safe_to_csv(rank_metrics, tables / "weekly_rank_metrics.csv")
    safe_to_csv(strategy, tables / "weekly_topk_strategy.csv")
    safe_to_csv(summary, tables / "weekly_topk_strategy_summary.csv")
    safe_to_csv(current, tables / "latest_weekly_screening_verdict.csv")
    write_report(ox, rank_metrics, summary, current, reports / "weekly_screening_rank_backtest.md", args.top_k)

    print(f"wrote {reports / 'weekly_screening_rank_backtest.md'}")
    print(summary.to_string(index=False))
    print(rank_metrics.to_string(index=False))


def read_table(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing required input: {path}")
    return pd.read_csv(path, **kwargs)


def build_weekly_panel(
    histories: dict[str, pd.DataFrame],
    driver_panel: pd.DataFrame,
    driver_features: pd.DataFrame,
    regime: pd.DataFrame,
    lppl_hist: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    dates = weekly_dates(driver_features.index)
    asset_map = {asset.symbol: asset for asset in ASSETS}
    regime_series = regime["gmm_regime"] if "gmm_regime" in regime else regime.get("rwkv_regime")
    lppl_lookup = build_lppl_lookup(lppl_hist)

    for date in dates:
        if date not in driver_features.index:
            eligible_driver_dates = driver_features.index[driver_features.index <= date]
            if eligible_driver_dates.empty:
                continue
            date = eligible_driver_dates.max()
        current_regime = str(regime_series.loc[:date].dropna().iloc[-1]) if not regime_series.loc[:date].dropna().empty else "unknown"
        for symbol, asset in asset_map.items():
            hist = histories.get(symbol)
            if hist is None or hist.empty or "Close" not in hist:
                continue
            full_close = clean_series(hist["Close"])
            close = full_close.loc[:date]
            if close.shape[0] < 260:
                continue
            date_in_asset = close.index[-1]
            loc = full_close.index.get_loc(date_in_asset)
            if loc + FORWARD_1M >= full_close.shape[0]:
                continue
            technical = historical_technical_score(close)
            driver_fit = asset_driver_fit(asset, driver_features.loc[:date])
            betas = rolling_driver_betas(close.pct_change(), driver_panel.loc[:date], asset.expected_drivers)
            beta_fit = beta_alignment_score(betas, asset.expected_drivers)
            win_1w, avg_1w = past_conditional_forward_stats(full_close, regime_series, date_in_asset, current_regime, FORWARD_1W)
            win_1m, avg_1m = past_conditional_forward_stats(full_close, regime_series, date_in_asset, current_regime, FORWARD_1M)
            prob_1w = blend_probability(win_1w, technical, driver_fit, beta_fit, horizon="1w")
            prob_1m = blend_probability(win_1m, technical, driver_fit, beta_fit, horizon="4w")
            lppl = latest_lppl_lookup(lppl_lookup, symbol, date_in_asset)
            bubble = float(lppl.get("lppl_dtcai", 0.0))
            score_before_lppl = np.clip(0.34 * technical + 0.28 * driver_fit + 0.18 * beta_fit + 0.20 * (prob_1m * 100.0) - risk_score(close), 0, 100)
            score = np.clip(score_before_lppl - bubble * 28.0, 0, 100)
            ret_1w = forward_return(full_close, date_in_asset, FORWARD_1W)
            ret_1m = forward_return(full_close, date_in_asset, FORWARD_1M)
            rows.append(
                {
                    "date": date_in_asset,
                    "symbol": symbol,
                    "name": asset.name,
                    "group": asset.group,
                    "regime": current_regime,
                    "score_0_100": score,
                    "score_before_lppl": score_before_lppl,
                    "upside_prob_1w": prob_1w,
                    "upside_prob_4w": prob_1m,
                    "technical_score": technical,
                    "driver_fit_score": driver_fit,
                    "beta_fit_score": beta_fit,
                    "bubble_score_0_100": bubble * 100.0,
                    "lppl_risk_label": dtcai_label(bubble),
                    "conditional_avg_return_1w": avg_1w,
                    "conditional_avg_return_4w": avg_1m,
                    "realized_return_1w": ret_1w,
                    "realized_return_4w": ret_1m,
                    "realized_up_1w": int(ret_1w > 0),
                    "realized_up_4w": int(ret_1m > 0),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame.dropna(subset=["score_0_100", "upside_prob_1w", "upside_prob_4w", "realized_return_1w", "realized_return_4w"]).reset_index(drop=True)


def weekly_dates(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    idx = pd.DatetimeIndex(index).dropna().sort_values()
    return pd.Series(idx, index=idx).groupby(idx.to_period("W-FRI")).max().tolist()


def build_lppl_lookup(lppl: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if lppl.empty:
        return {}
    return {symbol: group.sort_values("asof").reset_index(drop=True) for symbol, group in lppl.groupby("symbol")}


def latest_lppl_lookup(lookup: dict[str, pd.DataFrame], symbol: str, date: pd.Timestamp) -> dict[str, Any]:
    frame = lookup.get(symbol)
    if frame is None or frame.empty:
        return {}
    idx = np.searchsorted(pd.to_datetime(frame["asof"]).to_numpy(dtype="datetime64[ns]"), np.datetime64(date), side="right") - 1
    if idx < 0:
        return {}
    return frame.iloc[int(idx)].to_dict()


def add_expanding_weekly_calibration(panel: pd.DataFrame, min_train_weeks: int) -> pd.DataFrame:
    if panel.empty:
        return panel
    from sklearn.isotonic import IsotonicRegression

    out = panel.copy().sort_values("date").reset_index(drop=True)
    unique_dates = sorted(pd.to_datetime(out["date"]).unique())
    min_train_obs = max(260, min_train_weeks * 8)
    for horizon, raw_col, target_col in [("1w", "upside_prob_1w", "realized_up_1w"), ("4w", "upside_prob_4w", "realized_up_4w")]:
        calibrated = pd.Series(np.nan, index=out.index, dtype=float)
        for date in unique_dates:
            train_idx = out.index[pd.to_datetime(out["date"]).lt(date)]
            test_idx = out.index[pd.to_datetime(out["date"]).eq(date)]
            if len(train_idx) < min_train_obs or out.loc[train_idx, target_col].nunique() < 2:
                calibrated.loc[test_idx] = out.loc[test_idx, raw_col].astype(float)
                continue
            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(out.loc[train_idx, raw_col].astype(float), out.loc[train_idx, target_col].astype(int))
            calibrated.loc[test_idx] = model.predict(out.loc[test_idx, raw_col].astype(float))
        out[f"calibrated_prob_{horizon}"] = calibrated.fillna(out[raw_col].astype(float))
        out[f"meta_prob_{horizon}"] = out[f"calibrated_prob_{horizon}"]
        out[f"predicted_up_{horizon}"] = out[f"calibrated_prob_{horizon}"].ge(0.5).astype(int)
    out["threshold_1w"] = expanding_threshold(out, "calibrated_prob_1w", "realized_up_1w", min_train_obs)
    out["threshold_4w"] = expanding_threshold(out, "calibrated_prob_4w", "realized_up_4w", min_train_obs)
    out["high_confidence_1w"] = out["calibrated_prob_1w"].ge(out["threshold_1w"])
    out["high_confidence_4w"] = out["calibrated_prob_4w"].ge(out["threshold_4w"])
    out["threshold_predicted_up_1w"] = out["high_confidence_1w"].astype(int)
    out["threshold_predicted_up_4w"] = out["high_confidence_4w"].astype(int)
    return out


def expanding_threshold(frame: pd.DataFrame, prob_col: str, target_col: str, min_train_obs: int) -> pd.Series:
    thresholds = pd.Series(0.80, index=frame.index, dtype=float)
    for date in sorted(pd.to_datetime(frame["date"]).unique()):
        train = frame[pd.to_datetime(frame["date"]).lt(date)]
        idx = frame.index[pd.to_datetime(frame["date"]).eq(date)]
        if train.shape[0] < min_train_obs:
            continue
        best = 0.80
        best_acc = -1.0
        for threshold in np.arange(0.50, 0.91, 0.01):
            signal = train[pd.to_numeric(train[prob_col], errors="coerce").ge(threshold)]
            if signal.shape[0] < max(20, int(train.shape[0] * 0.03)):
                continue
            acc = float(signal[target_col].astype(int).mean())
            if acc >= 0.80:
                best = float(threshold)
                break
            if acc > best_acc:
                best_acc = acc
                best = float(threshold)
        thresholds.loc[idx] = best
    return thresholds


def add_ranks(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["pred_rank"] = out.groupby("date")["institutional_score_0_100"].rank(ascending=False, method="first")
    out["actual_rank_1w"] = out.groupby("date")["realized_return_1w"].rank(ascending=False, method="first")
    out["actual_rank_4w"] = out.groupby("date")["realized_return_4w"].rank(ascending=False, method="first")
    return out.sort_values(["date", "pred_rank"]).reset_index(drop=True)


def ox_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    configs = [
        ("1w", "standard_50", "predicted_up_1w", "realized_up_1w", "calibrated_prob_1w"),
        ("1w", "walkforward_threshold", "threshold_predicted_up_1w", "realized_up_1w", "calibrated_prob_1w"),
        ("1m", "standard_50", "predicted_up_4w", "realized_up_4w", "calibrated_prob_4w"),
        ("1m", "walkforward_threshold", "threshold_predicted_up_4w", "realized_up_4w", "calibrated_prob_4w"),
    ]
    for horizon, rule, pred_col, target_col, prob_col in configs:
        data = panel.dropna(subset=[pred_col, target_col, prob_col])
        for segment_type, segments in [("__ALL__", [("__ALL__", data)]), ("group", list(data.groupby("group"))), ("regime", list(data.groupby("regime")))]:
            for segment, sample in segments:
                if sample.empty:
                    continue
                pred = sample[pred_col].astype(int)
                actual = sample[target_col].astype(int)
                tp = int((pred.eq(1) & actual.eq(1)).sum())
                fp = int((pred.eq(1) & actual.eq(0)).sum())
                fn = int((pred.eq(0) & actual.eq(1)).sum())
                rows.append(
                    {
                        "horizon": horizon,
                        "rule": rule,
                        "segment_type": segment_type,
                        "segment": str(segment),
                        "samples": int(sample.shape[0]),
                        "o_rate": float(pred.mean()),
                        "accuracy": float(pred.eq(actual).mean()),
                        "precision_up": tp / max(tp + fp, 1),
                        "recall_up": tp / max(tp + fn, 1),
                        "mean_prob": float(sample[prob_col].mean()),
                        "realized_up_rate": float(actual.mean()),
                    }
                )
    return pd.DataFrame(rows)


def ranking_metrics(panel: pd.DataFrame, top_k: int) -> pd.DataFrame:
    rows = []
    for horizon, actual_rank_col, ret_col in [("1w", "actual_rank_1w", "realized_return_1w"), ("1m", "actual_rank_4w", "realized_return_4w")]:
        per_date = []
        for date, group in panel.groupby("date"):
            if group.shape[0] < top_k * 2:
                continue
            predicted_top = set(group.nsmallest(top_k, "pred_rank")["symbol"])
            actual_top = set(group.nsmallest(top_k, actual_rank_col)["symbol"])
            top = group.nsmallest(top_k, "pred_rank")
            actual_best = group.nsmallest(top_k, actual_rank_col)
            per_date.append(
                {
                    "date": date,
                    "topk_hit_rate": len(predicted_top & actual_top) / top_k,
                    "topk_avg_return": float(top[ret_col].mean()),
                    "universe_avg_return": float(group[ret_col].mean()),
                    "actual_topk_avg_return": float(actual_best[ret_col].mean()),
                    "spearman": float(group["institutional_score_0_100"].corr(group[ret_col], method="spearman")),
                    "top1_is_actual_top10": int(top.iloc[0]["symbol"] in actual_top),
                    "top1_return": float(top.iloc[0][ret_col]),
                    "top1_actual_rank_pct": float(top.iloc[0][actual_rank_col] / group.shape[0]),
                }
            )
        frame = pd.DataFrame(per_date).dropna()
        rows.append(
            {
                "horizon": horizon,
                "weeks": int(frame.shape[0]),
                f"top{top_k}_hit_rate": float(frame["topk_hit_rate"].mean()),
                f"top{top_k}_avg_return": float(frame["topk_avg_return"].mean()),
                "universe_avg_return": float(frame["universe_avg_return"].mean()),
                "actual_topk_avg_return": float(frame["actual_topk_avg_return"].mean()),
                "mean_spearman_score_vs_return": float(frame["spearman"].mean()),
                "top1_in_actual_top10_rate": float(frame["top1_is_actual_top10"].mean()),
                "top1_avg_return": float(frame["top1_return"].mean()),
                "top1_avg_actual_rank_pct": float(frame["top1_actual_rank_pct"].mean()),
            }
        )
    return pd.DataFrame(rows)


def topk_strategy(panel: pd.DataFrame, top_k: int) -> pd.DataFrame:
    rows = []
    for date, group in panel.groupby("date"):
        picks = group.nsmallest(top_k, "pred_rank")
        if picks.empty:
            continue
        rows.append(
            {
                "date": date,
                "holdings": ",".join(picks["symbol"]),
                "gross_return_1w": float(picks["realized_return_1w"].mean()),
                "gross_return_1m": float(picks["realized_return_4w"].mean()),
                "universe_return_1w": float(group["realized_return_1w"].mean()),
                "universe_return_1m": float(group["realized_return_4w"].mean()),
            }
        )
    strategy = pd.DataFrame(rows).dropna()
    if not strategy.empty:
        strategy["equity_1w"] = (1 + strategy["gross_return_1w"]).cumprod()
        strategy["universe_equity_1w"] = (1 + strategy["universe_return_1w"]).cumprod()
    return strategy


def strategy_summary(strategy: pd.DataFrame) -> pd.DataFrame:
    if strategy.empty:
        return pd.DataFrame()
    rows = []
    for col, equity_col, periods in [("gross_return_1w", "equity_1w", 52), ("universe_return_1w", "universe_equity_1w", 52)]:
        r = strategy[col].dropna()
        equity = strategy[equity_col]
        total = float((1 + r).prod() - 1)
        ann = float((1 + total) ** (periods / max(len(r), 1)) - 1)
        vol = float(r.std() * math.sqrt(periods))
        dd = equity / equity.cummax() - 1
        rows.append({"series": col, "weeks": int(len(r)), "total_return": total, "ann_return": ann, "ann_vol": vol, "sharpe": ann / vol if vol else np.nan, "max_drawdown": float(dd.min()), "win_rate": float((r > 0).mean())})
    return pd.DataFrame(rows)


def current_week_snapshot(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return panel
    latest = panel[pd.to_datetime(panel["date"]).eq(pd.to_datetime(panel["date"]).max())].copy()
    cols = [
        "date",
        "pred_rank",
        "symbol",
        "name",
        "group",
        "regime",
        "institutional_score_0_100",
        "calibrated_prob_1w",
        "predicted_up_1w",
        "threshold_1w",
        "threshold_predicted_up_1w",
        "calibrated_prob_4w",
        "predicted_up_4w",
        "threshold_4w",
        "threshold_predicted_up_4w",
        "realized_return_1w",
        "actual_rank_1w",
        "realized_return_4w",
        "actual_rank_4w",
    ]
    return latest[[c for c in cols if c in latest]].sort_values("pred_rank")


def write_report(ox: pd.DataFrame, rank_metrics: pd.DataFrame, summary: pd.DataFrame, current: pd.DataFrame, path: Path, top_k: int) -> None:
    lines = ["# Weekly Screening Rank Backtest", ""]
    if not summary.empty:
        lines.extend(["## Weekly Top-K Strategy", summary.to_markdown(index=False), ""])
    if not rank_metrics.empty:
        lines.extend(["## Ranking Correctness", rank_metrics.to_markdown(index=False), ""])
    if not ox.empty:
        core = ox[ox["segment_type"].eq("__ALL__")]
        lines.extend(["## OX Direction Metrics", core.to_markdown(index=False), ""])
    if not current.empty:
        lines.extend([f"## Latest Weekly Screening Top {top_k}", current.head(top_k).to_markdown(index=False), ""])
    lines.extend(
        [
            "## Notes",
            "- Each screening date is the last available trading day of the week.",
            "- Features use only data available up to that weekly screening date.",
            "- OX is evaluated with expanding walk-forward probability calibration.",
            "- Ranking correctness compares predicted screening rank with future realized-return rank over 5 and 20 trading days.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
