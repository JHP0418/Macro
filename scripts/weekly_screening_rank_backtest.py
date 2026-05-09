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
from basket_taxonomy import classify_basket  # noqa: E402

OUT_DIR = ROOT / "outputs" / "weekly_screening_rank_backtest_latest"
RWKV_OUT_DIR = ROOT / "outputs" / "rwkv_macro_regime_latest"
LEGACY_RWKV_OUT_DIR = ROOT / "outputs" / ("rwkv_" + "lp" + "pl_asset_screener_latest")
SAFE_GROUPS = {"FX cash", "Cash/short bonds"}
DEFENSIVE_GROUPS = {"Gold", "Korea bonds", "US long bonds", "US IG bonds", "Korea defensive", "US dividend/defensive"}
FORWARD_1W = 5
FORWARD_1M = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weekly walk-forward screening OX and rank backtest.")
    parser.add_argument("--start", default="1995-01-01")
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
    regime_path = args.input / "tables" / "rwkv_regime_history.csv"
    if not regime_path.exists():
        regime_path = ROOT / "outputs" / "macro_regime_asset_screener_latest" / "tables" / "regime_history.csv"
    if not regime_path.exists():
        regime_path = LEGACY_RWKV_OUT_DIR / "tables" / "rwkv_regime_history.csv"
    regime = read_table(regime_path, parse_dates=["Date"]).set_index("Date")
    histories = load_asset_histories([asset.symbol for asset in ASSETS], args.start, args.skip_download)

    panel = build_weekly_panel(histories, driver_panel, driver_features, regime)
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
    basket_panel, basket_summary, basket_current, basket_constituents = basket_backtest_outputs(calibrated)
    safe_to_csv(basket_panel, tables / "weekly_basket_panel.csv")
    safe_to_csv(basket_summary, tables / "weekly_basket_backtest_summary.csv")
    safe_to_csv(basket_current, tables / "latest_basket_scores.csv")
    safe_to_csv(basket_constituents, tables / "latest_basket_constituent_scores.csv")
    write_report(ox, rank_metrics, summary, current, reports / "weekly_screening_rank_backtest.md", args.top_k)

    print(f"wrote {reports / 'weekly_screening_rank_backtest.md'}")
    print(summary.to_string(index=False))
    print(rank_metrics.to_string(index=False))


def read_table(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing required input: {path}")
    return pd.read_csv(path, **kwargs)


def first_valid_regime_date(regime: pd.DataFrame) -> pd.Timestamp | None:
    if regime.empty:
        return None
    col = "rwkv_regime" if "rwkv_regime" in regime else "rule_regime" if "rule_regime" in regime else "gmm_regime" if "gmm_regime" in regime else None
    if col is None:
        return None
    s = regime[col].dropna().astype(str)
    s = s[~s.str.lower().eq("unknown")]
    return None if s.empty else pd.Timestamp(s.index.min())


def historical_technical_score(close: pd.Series) -> float:
    if close.shape[0] < 130:
        return 50.0
    r20 = safe_pct(close, 20)
    r60 = safe_pct(close, 60)
    r120 = safe_pct(close, 120)
    vol20 = ann_vol(close, 20)
    ma200 = close.rolling(200).mean().iloc[-1]
    ma_dist = close.iloc[-1] / ma200 - 1 if pd.notna(ma200) and ma200 else 0.0
    edge = 3.0 * r20 + 2.0 * r60 + 1.2 * r120 + 1.0 * ma_dist - 0.8 * max(vol20 - 0.20, 0)
    return float(np.clip(50 + 42 * math.tanh(edge), 0, 100))


def safe_pct(close: pd.Series, periods: int) -> float:
    return 0.0 if close.shape[0] <= periods else float(close.iloc[-1] / close.iloc[-periods - 1] - 1.0)


def ann_vol(close: pd.Series, periods: int) -> float:
    ret = close.pct_change().tail(periods)
    return float(ret.std() * math.sqrt(252)) if ret.notna().sum() >= max(5, periods // 2) else 0.0


def past_conditional_forward_stats(close: pd.Series, regimes: pd.Series, asof: pd.Timestamp, regime: str, horizon: int) -> tuple[float, float]:
    forward = close.shift(-horizon) / close - 1.0
    data = pd.concat([forward.rename("forward"), regimes.rename("regime")], axis=1).dropna()
    data = data[data.index < asof - pd.Timedelta(days=horizon + 2)]
    sample = data[data["regime"].astype(str).eq(regime)]
    if sample.shape[0] < 20:
        sample = data.tail(252)
    if sample.empty:
        return 0.52, 0.0
    return float((sample["forward"] > 0).mean()), float(sample["forward"].mean())


def forward_return(close: pd.Series, date: pd.Timestamp, horizon: int) -> float:
    loc = close.index.get_loc(date)
    if loc + horizon >= close.shape[0]:
        return np.nan
    return float(close.iloc[loc + horizon] / close.iloc[loc] - 1.0)


def add_institutional_score(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    prob = pd.to_numeric(out.get("meta_prob_4w", out.get("calibrated_prob_4w", 0.5)), errors="coerce").fillna(0.5)
    expected_return = pd.to_numeric(out.get("conditional_avg_return_4w", 0.0), errors="coerce").fillna(0.0)
    technical = pd.to_numeric(out.get("technical_score", out.get("score_0_100", 50.0)), errors="coerce").fillna(50.0)
    base_score = pd.to_numeric(out.get("score_0_100", 50.0), errors="coerce").fillna(50.0)
    confidence_bonus = out.get("high_confidence_4w", False).astype(float) * 5.0 if "high_confidence_4w" in out else 0.0
    out["institutional_score_0_100"] = (
        0.52 * base_score
        + 24.0 * prob
        + 0.14 * technical
        + 260.0 * expected_return.clip(-0.10, 0.10)
        + confidence_bonus
        - cash_opportunity_drag(out)
    ).clip(0, 100)
    return out


def cash_opportunity_drag(frame: pd.DataFrame) -> pd.Series:
    group = frame.get("group", pd.Series("", index=frame.index)).astype(str)
    regime = frame.get("regime", pd.Series("", index=frame.index)).astype(str)
    is_cash = group.isin(SAFE_GROUPS)
    is_defensive = group.isin(DEFENSIVE_GROUPS)
    risk_off = regime.eq("Risk-Off / Cash")
    defensive = regime.eq("Defensive / Rate-Cut")
    drag = pd.Series(0.0, index=frame.index)
    drag.loc[is_cash & ~risk_off] = 18.0
    drag.loc[is_cash & defensive] = 7.0
    drag.loc[is_defensive & ~(risk_off | defensive)] = 5.0
    return drag


def build_weekly_panel(
    histories: dict[str, pd.DataFrame],
    driver_panel: pd.DataFrame,
    driver_features: pd.DataFrame,
    regime: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    dates = weekly_dates(driver_features.index)
    asset_map = {asset.symbol: asset for asset in ASSETS}
    regime_series = regime["rwkv_regime"] if "rwkv_regime" in regime else regime["rule_regime"] if "rule_regime" in regime else regime.get("gmm_regime")

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
            score = np.clip(0.34 * technical + 0.28 * driver_fit + 0.18 * beta_fit + 0.20 * (prob_1m * 100.0) - risk_score(close), 0, 100)
            ret_1w = forward_return(full_close, date_in_asset, FORWARD_1W)
            ret_1m = forward_return(full_close, date_in_asset, FORWARD_1M)
            rows.append(
                {
                    "date": date_in_asset,
                    "symbol": symbol,
                    "name": asset.name,
                    "group": asset.group,
                    "basket": classify_basket(asset.group, asset.name, symbol),
                    "regime": current_regime,
                    "score_0_100": score,
                    "upside_prob_1w": prob_1w,
                    "upside_prob_4w": prob_1m,
                    "technical_score": technical,
                    "driver_fit_score": driver_fit,
                    "beta_fit_score": beta_fit,
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
    if "basket" in out:
        out["basket_rank"] = out.groupby(["date", "basket"])["institutional_score_0_100"].rank(ascending=False, method="first")
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


def basket_backtest_outputs(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if panel.empty or "basket" not in panel:
        empty = pd.DataFrame()
        return empty, empty, empty, empty
    basket_rows = []
    for (date, basket), sample in panel.groupby(["date", "basket"]):
        ranked = sample.sort_values("institutional_score_0_100", ascending=False)
        top = ranked.head(min(5, len(ranked)))
        basket_rows.append(
            {
                "date": date,
                "basket": basket,
                "asset_count": int(len(sample)),
                "basket_score_0_100": float(0.55 * top["institutional_score_0_100"].mean() + 0.45 * sample["institutional_score_0_100"].mean()),
                "basket_prob_1w": float(top["calibrated_prob_1w"].mean()),
                "basket_prob_1m": float(top["calibrated_prob_4w"].mean()),
                "basket_realized_return_1w": float(top["realized_return_1w"].mean()),
                "basket_realized_return_1m": float(top["realized_return_4w"].mean()),
                "top_symbols": ", ".join(top["symbol"].astype(str).tolist()),
                "top_names": " | ".join(top["name"].astype(str).tolist()),
            }
        )
    basket_panel = pd.DataFrame(basket_rows)
    if basket_panel.empty:
        return basket_panel, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    basket_panel["basket_rank"] = basket_panel.groupby("date")["basket_score_0_100"].rank(ascending=False, method="first")
    basket_panel["actual_basket_rank_1w"] = basket_panel.groupby("date")["basket_realized_return_1w"].rank(ascending=False, method="first")
    basket_panel["actual_basket_rank_1m"] = basket_panel.groupby("date")["basket_realized_return_1m"].rank(ascending=False, method="first")

    summary_rows = []
    for horizon, ret_col, actual_rank_col in [
        ("1w", "basket_realized_return_1w", "actual_basket_rank_1w"),
        ("1m", "basket_realized_return_1m", "actual_basket_rank_1m"),
    ]:
        per_date = []
        for date, sample in basket_panel.groupby("date"):
            if sample.shape[0] < 3:
                continue
            pred_top = sample.nsmallest(1, "basket_rank").iloc[0]
            actual_top = sample.nsmallest(1, actual_rank_col).iloc[0]
            pred_top3 = set(sample.nsmallest(min(3, len(sample)), "basket_rank")["basket"])
            actual_top3 = set(sample.nsmallest(min(3, len(sample)), actual_rank_col)["basket"])
            per_date.append(
                {
                    "date": date,
                    "pred_top_return": pred_top[ret_col],
                    "basket_avg_return": sample[ret_col].mean(),
                    "top1_exact": int(pred_top["basket"] == actual_top["basket"]),
                    "actual_top1_in_pred_top3": int(actual_top["basket"] in pred_top3),
                    "top3_overlap_rate": len(pred_top3.intersection(actual_top3)) / max(len(actual_top3), 1),
                }
            )
        df = pd.DataFrame(per_date)
        if df.empty:
            continue
        summary_rows.append(
            {
                "horizon": horizon,
                "weeks": int(len(df)),
                "pred_top_avg_return": float(df["pred_top_return"].mean()),
                "basket_avg_return": float(df["basket_avg_return"].mean()),
                "top1_hit_rate": float(df["top1_exact"].mean()),
                "actual_top1_in_pred_top3_rate": float(df["actual_top1_in_pred_top3"].mean()),
                "top3_overlap_rate": float(df["top3_overlap_rate"].mean()),
            }
        )
    basket_summary = pd.DataFrame(summary_rows)
    latest_date = basket_panel["date"].max()
    basket_current = basket_panel[basket_panel["date"].eq(latest_date)].sort_values("basket_rank").reset_index(drop=True)
    constituent_current = panel[panel["date"].eq(latest_date)].sort_values(["basket", "basket_rank"]).reset_index(drop=True)
    return basket_panel, basket_summary, basket_current, constituent_current


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
