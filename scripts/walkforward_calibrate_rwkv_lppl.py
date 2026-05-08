from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from macro_regime_asset_screener import (  # noqa: E402
    ASSETS,
    FRED_SERIES,
    YF_SERIES,
    ann_vol,
    asset_driver_fit,
    beta_alignment_score,
    blend_probability,
    clean_series,
    load_asset_histories,
    load_driver_series,
    make_driver_features,
    make_driver_panel,
    pct_return,
    risk_score,
    rolling_driver_betas,
    safe_to_csv,
)
from rwkv_lppl_asset_screener import RWKV_OUT_DIR, dtcai_label  # noqa: E402

OUT_DIR = ROOT / "outputs" / "rwkv_lppl_walkforward_validation_latest"
FORWARD_1W = 5
FORWARD_4W = 20
SAFE_GROUPS = {"Cash/short bonds", "USD cash"}
DEFENSIVE_GROUPS = {"Korea bonds", "US long bonds", "US IG bonds", "Gold"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward validation, probability calibration, and LPPL false-alarm checks.")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--input", type=Path, default=RWKV_OUT_DIR)
    parser.add_argument("--output", type=Path, default=OUT_DIR)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--min-train-months", type=int, default=36)
    parser.add_argument("--target-high-confidence-accuracy", type=float, default=0.80)
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
    current_scores = read_table(args.input / "tables" / "current_asset_scores_rwkv_lppl.csv")
    histories = load_asset_histories([asset.symbol for asset in ASSETS], args.start, args.skip_download)

    valid_start = first_valid_regime_date(regime)
    panel = build_walkforward_panel(histories, driver_panel, driver_features, regime, lppl_hist)
    if valid_start is not None:
        panel = panel[pd.to_datetime(panel["date"]).ge(valid_start)].reset_index(drop=True)
    calibrated_panel, calibrators, calibration = calibrate_probabilities(panel)
    calibrated_panel, meta_models, meta_report = add_walkforward_meta_model(calibrated_panel, args.min_train_months)
    thresholds = learn_high_confidence_thresholds(calibrated_panel, args.target_high_confidence_accuracy)
    calibrated_panel = apply_high_confidence_flags(calibrated_panel, thresholds)
    dtcai_thresholds = learn_group_dtcai_thresholds(lppl_hist)
    calibrated_panel = apply_group_dtcai_thresholds(calibrated_panel, dtcai_thresholds)
    calibrated_panel = add_institutional_score(calibrated_panel)
    strategy, summary = run_strategy_backtest(calibrated_panel, histories, args.top_n, args.cost_bps)
    false_alarm = lppl_false_alarm_validation(lppl_hist)
    calibrated_current = calibrate_current_scores(current_scores, calibrators, meta_models, thresholds, dtcai_thresholds)

    safe_to_csv(panel, tables / "walkforward_raw_panel.csv")
    safe_to_csv(calibrated_panel, tables / "walkforward_calibrated_panel.csv")
    safe_to_csv(calibration, tables / "probability_calibration.csv")
    safe_to_csv(meta_report, tables / "meta_model_validation.csv")
    safe_to_csv(thresholds, tables / "high_confidence_thresholds.csv")
    safe_to_csv(dtcai_thresholds, tables / "group_dtcai_thresholds.csv")
    safe_to_csv(strategy, tables / "walkforward_strategy_monthly.csv")
    safe_to_csv(summary, tables / "walkforward_summary.csv")
    safe_to_csv(false_alarm, tables / "lppl_false_alarm_validation.csv")
    safe_to_csv(calibrated_current, tables / "calibrated_current_asset_scores.csv")
    write_report(summary, calibration, false_alarm, calibrated_current, meta_report, thresholds, dtcai_thresholds, reports / "validation_report.md")
    print(f"wrote {reports / 'validation_report.md'}")
    print(summary.to_string(index=False))


def read_table(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing required input: {path}")
    return pd.read_csv(path, **kwargs)


def read_lppl_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, parse_dates=["asof"])
    if frame.empty:
        return frame
    grouped = (
        frame.groupby(["symbol", "asof"], as_index=False)
        .agg(
            lppl_dtcai=("lppl_dtcai", lambda x: float(pd.to_numeric(x, errors="coerce").quantile(0.95))),
            lppl_dtcai_max=("lppl_dtcai", "max"),
            lppl_reliability=("lppl_reliability", "mean"),
            label=("label", "max"),
        )
    )
    return grouped


def first_valid_regime_date(regime: pd.DataFrame) -> pd.Timestamp | None:
    if regime.empty:
        return None
    col = "rwkv_regime" if "rwkv_regime" in regime else "gmm_regime" if "gmm_regime" in regime else None
    if col is None:
        return None
    s = regime[col].dropna().astype(str)
    s = s[~s.str.lower().eq("unknown")]
    return None if s.empty else pd.Timestamp(s.index.min())


def build_walkforward_panel(
    histories: dict[str, pd.DataFrame],
    driver_panel: pd.DataFrame,
    driver_features: pd.DataFrame,
    regime: pd.DataFrame,
    lppl_hist: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    rebalance_dates = monthly_dates(driver_features.index)
    asset_map = {asset.symbol: asset for asset in ASSETS}
    regime_series = regime["gmm_regime"] if "gmm_regime" in regime else regime.get("rwkv_regime")
    for date in rebalance_dates:
        if date not in driver_features.index:
            date = driver_features.index[driver_features.index <= date].max()
        if pd.isna(date):
            continue
        current_regime = str(regime_series.loc[:date].dropna().iloc[-1]) if not regime_series.loc[:date].dropna().empty else "unknown"
        for symbol, asset in asset_map.items():
            hist = histories.get(symbol)
            if hist is None or hist.empty or "Close" not in hist:
                continue
            close = clean_series(hist["Close"]).loc[:date]
            full_close = clean_series(hist["Close"])
            if close.shape[0] < 260:
                continue
            date_in_asset = close.index[-1]
            if full_close.index.get_loc(date_in_asset) + FORWARD_4W >= full_close.shape[0]:
                continue
            technical = historical_technical_score(close)
            driver_fit = asset_driver_fit(asset, driver_features.loc[:date])
            betas = rolling_driver_betas(close.pct_change(), driver_panel.loc[:date], asset.expected_drivers)
            beta_fit = beta_alignment_score(betas, asset.expected_drivers)
            win_1w, avg_1w = past_conditional_forward_stats(full_close, regime_series, date_in_asset, current_regime, FORWARD_1W)
            win_4w, avg_4w = past_conditional_forward_stats(full_close, regime_series, date_in_asset, current_regime, FORWARD_4W)
            prob_1w = blend_probability(win_1w, technical, driver_fit, beta_fit, horizon="1w")
            prob_4w = blend_probability(win_4w, technical, driver_fit, beta_fit, horizon="4w")
            lppl = latest_lppl_before(lppl_hist, symbol, date_in_asset)
            bubble = float(lppl.get("lppl_dtcai", 0.0))
            risk_penalty = risk_score(close)
            score_before_lppl = np.clip(0.34 * technical + 0.28 * driver_fit + 0.18 * beta_fit + 0.20 * (prob_4w * 100.0) - risk_penalty, 0, 100)
            score = np.clip(score_before_lppl - bubble * 28.0, 0, 100)
            fwd_1w = forward_return(full_close, date_in_asset, FORWARD_1W)
            fwd_4w = forward_return(full_close, date_in_asset, FORWARD_4W)
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
                    "upside_prob_4w": prob_4w,
                    "technical_score": technical,
                    "driver_fit_score": driver_fit,
                    "beta_fit_score": beta_fit,
                    "bubble_score_0_100": bubble * 100.0,
                    "lppl_risk_label": dtcai_label(bubble),
                    "realized_return_1w": fwd_1w,
                    "realized_return_4w": fwd_4w,
                    "realized_up_1w": int(fwd_1w > 0),
                    "realized_up_4w": int(fwd_4w > 0),
                    "conditional_avg_return_4w": avg_4w,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["score_0_100", "upside_prob_1w", "upside_prob_4w", "realized_return_1w", "realized_return_4w"])
    frame["rank_by_date"] = frame.groupby("date")["score_0_100"].rank(ascending=False, method="first")
    return frame.sort_values(["date", "rank_by_date"]).reset_index(drop=True)


def monthly_dates(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    idx = pd.DatetimeIndex(index).dropna().sort_values()
    return pd.Series(idx, index=idx).groupby(idx.to_period("M")).max().tolist()


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


def latest_lppl_before(lppl: pd.DataFrame, symbol: str, date: pd.Timestamp) -> dict[str, Any]:
    if lppl.empty:
        return {}
    sample = lppl[(lppl["symbol"].eq(symbol)) & (lppl["asof"].le(date))]
    if sample.empty:
        return {}
    return sample.sort_values("asof").iloc[-1].to_dict()


def forward_return(close: pd.Series, date: pd.Timestamp, horizon: int) -> float:
    loc = close.index.get_loc(date)
    if loc + horizon >= close.shape[0]:
        return np.nan
    return float(close.iloc[loc + horizon] / close.iloc[loc] - 1.0)


def calibrate_probabilities(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    if panel.empty:
        return panel, {}, pd.DataFrame()
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import brier_score_loss, log_loss

    out = panel.copy()
    calibrators: dict[str, Any] = {}
    rows = []
    for horizon, prob_col, target_col in [("1w", "upside_prob_1w", "realized_up_1w"), ("4w", "upside_prob_4w", "realized_up_4w")]:
        data = out.dropna(subset=[prob_col, target_col]).sort_values("date")
        if data.shape[0] < 80 or data[target_col].nunique() < 2:
            out[f"calibrated_prob_{horizon}"] = out[prob_col]
            continue
        split_date = data["date"].quantile(0.70)
        train = data[data["date"].le(split_date)]
        test = data[data["date"].gt(split_date)]
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(train[prob_col].astype(float), train[target_col].astype(int))
        out[f"calibrated_prob_{horizon}"] = cal.predict(out[prob_col].astype(float))
        calibrators[horizon] = cal
        for subset_name, sample in [("train", train), ("test", test), ("all", data)]:
            pred = cal.predict(sample[prob_col].astype(float))
            actual = sample[target_col].astype(int).to_numpy()
            rows.append(
                {
                    "horizon": horizon,
                    "subset": subset_name,
                    "samples": int(sample.shape[0]),
                    "brier": brier_score_loss(actual, pred),
                    "log_loss": log_loss(actual, np.clip(pred, 1e-4, 1 - 1e-4), labels=[0, 1]),
                    "ece_10bin": expected_calibration_error(pred, actual, bins=10),
                    "mean_pred": float(np.mean(pred)),
                    "realized_rate": float(np.mean(actual)),
                }
            )
        rows.extend(calibration_bins(data, prob_col, target_col, horizon, "raw"))
        rows.extend(calibration_bins(data.assign(_cal=out.loc[data.index, f"calibrated_prob_{horizon}"]), "_cal", target_col, horizon, "calibrated"))
    return out, calibrators, pd.DataFrame(rows)


def add_walkforward_meta_model(panel: pd.DataFrame, min_train_months: int) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    if panel.empty:
        return panel, {}, pd.DataFrame()
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, brier_score_loss, precision_score, recall_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    out = panel.copy().sort_values("date").reset_index(drop=True)
    feature_cols = meta_feature_columns(out)
    models: dict[str, Any] = {}
    rows = []
    unique_dates = sorted(pd.to_datetime(out["date"]).unique())
    min_train_obs = max(240, min_train_months * 8)
    for horizon, target_col in [("1w", "realized_up_1w"), ("4w", "realized_up_4w")]:
        prob_values = pd.Series(np.nan, index=out.index, dtype=float)
        for date in unique_dates:
            train_idx = out.index[pd.to_datetime(out["date"]).lt(date)]
            test_idx = out.index[pd.to_datetime(out["date"]).eq(date)]
            if len(train_idx) < min_train_obs or out.loc[train_idx, target_col].nunique() < 2:
                continue
            model = build_meta_classifier()
            x_train = feature_matrix(out.loc[train_idx], feature_cols)
            y_train = out.loc[train_idx, target_col].astype(int)
            train_columns = x_train.columns
            x_test = feature_matrix(out.loc[test_idx], feature_cols).reindex(columns=train_columns, fill_value=0.0)
            model.fit(x_train, y_train)
            prob_values.loc[test_idx] = model.predict_proba(x_test)[:, 1]
        fallback_col = f"calibrated_prob_{horizon}" if f"calibrated_prob_{horizon}" in out else f"upside_prob_{horizon}"
        out[f"meta_prob_{horizon}"] = prob_values.fillna(pd.to_numeric(out[fallback_col], errors="coerce").fillna(0.5))
        full_model = build_meta_classifier()
        x_full = feature_matrix(out, feature_cols)
        full_model.fit(x_full, out[target_col].astype(int))
        models[horizon] = {"model": full_model, "features": feature_cols, "columns": list(x_full.columns)}
        valid = out.dropna(subset=[f"meta_prob_{horizon}", target_col])
        pred = valid[f"meta_prob_{horizon}"].ge(0.5)
        actual = valid[target_col].astype(int).eq(1)
        rows.append(
            {
                "horizon": horizon,
                "samples": int(valid.shape[0]),
                "accuracy_at_50": accuracy_score(actual, pred),
                "precision_at_50": precision_score(actual, pred, zero_division=0),
                "recall_at_50": recall_score(actual, pred, zero_division=0),
                "brier": brier_score_loss(actual.astype(int), valid[f"meta_prob_{horizon}"]),
                "mean_prob": float(valid[f"meta_prob_{horizon}"].mean()),
                "realized_rate": float(actual.mean()),
            }
        )
    return out, models, pd.DataFrame(rows)


def build_meta_classifier():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1200, class_weight="balanced", C=0.7, random_state=42))


def meta_feature_columns(frame: pd.DataFrame) -> list[str]:
    base = [
        "score_0_100",
        "score_before_lppl",
        "upside_prob_1w",
        "upside_prob_4w",
        "calibrated_prob_1w",
        "calibrated_prob_4w",
        "technical_score",
        "driver_fit_score",
        "beta_fit_score",
        "bubble_score_0_100",
        "conditional_avg_return_4w",
    ]
    return [col for col in base if col in frame]


def feature_matrix(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    numeric = frame.reindex(columns=cols).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    cats = pd.get_dummies(frame[["group", "regime"]].astype(str), prefix=["group", "regime"], dtype=float)
    return pd.concat([numeric.reset_index(drop=True), cats.reset_index(drop=True)], axis=1)


def learn_high_confidence_thresholds(panel: pd.DataFrame, target_accuracy: float) -> pd.DataFrame:
    rows = []
    for horizon, prob_col, target_col in [("1w", "meta_prob_1w", "realized_up_1w"), ("4w", "meta_prob_4w", "realized_up_4w")]:
        data = panel.dropna(subset=[prob_col, target_col]).copy()
        if data.empty:
            continue
        for segment_col in ["__ALL__", "group", "regime"]:
            segments = [("__ALL__", data)] if segment_col == "__ALL__" else list(data.groupby(segment_col))
            for segment, sample in segments:
                threshold, stats = choose_threshold(sample, prob_col, target_col, target_accuracy)
                rows.append({"horizon": horizon, "segment_type": segment_col, "segment": str(segment), "threshold": threshold, **stats})
    return pd.DataFrame(rows)


def choose_threshold(sample: pd.DataFrame, prob_col: str, target_col: str, target_accuracy: float) -> tuple[float, dict[str, Any]]:
    best = (0.5, {"coverage": 0.0, "accuracy": 0.0, "signals": 0})
    for threshold in np.arange(0.50, 0.91, 0.01):
        signal = sample[pd.to_numeric(sample[prob_col], errors="coerce").ge(threshold)]
        if signal.shape[0] < max(10, int(sample.shape[0] * 0.03)):
            continue
        actual = signal[target_col].astype(int).eq(1)
        acc = float(actual.mean())
        coverage = float(signal.shape[0] / sample.shape[0])
        stats = {"coverage": coverage, "accuracy": acc, "signals": int(signal.shape[0])}
        if acc >= target_accuracy:
            return float(threshold), stats
        if acc > best[1]["accuracy"] or (acc == best[1]["accuracy"] and coverage > best[1]["coverage"]):
            best = (float(threshold), stats)
    return best


def apply_high_confidence_flags(panel: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    for horizon, prob_col in [("1w", "meta_prob_1w"), ("4w", "meta_prob_4w")]:
        default = threshold_lookup(thresholds, horizon, "__ALL__", "__ALL__")
        vals = []
        for _, row in out.iterrows():
            thr = threshold_lookup(thresholds, horizon, "group", str(row.get("group")), default)
            thr = threshold_lookup(thresholds, horizon, "regime", str(row.get("regime")), thr)
            vals.append(float(row.get(prob_col, 0.0)) >= thr)
        out[f"high_confidence_{horizon}"] = vals
    return out


def add_institutional_score(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    prob = pd.to_numeric(out.get("meta_prob_4w", out.get("calibrated_prob_4w", 0.5)), errors="coerce").fillna(0.5)
    expected_return = pd.to_numeric(out.get("conditional_avg_return_4w", 0.0), errors="coerce").fillna(0.0)
    technical = pd.to_numeric(out.get("technical_score", out.get("score_0_100", 50.0)), errors="coerce").fillna(50.0)
    base_score = pd.to_numeric(out.get("score_0_100", 50.0), errors="coerce").fillna(50.0)
    confidence_bonus = out.get("high_confidence_4w", False).astype(float) * 5.0 if "high_confidence_4w" in out else 0.0
    crash_penalty = out.get("adaptive_lppl_risk_label", "").eq("crash-alert").astype(float) * 10.0 if "adaptive_lppl_risk_label" in out else 0.0
    out["institutional_score_0_100"] = (
        0.52 * base_score
        + 24.0 * prob
        + 0.14 * technical
        + 260.0 * expected_return.clip(-0.10, 0.10)
        + confidence_bonus
        - crash_penalty
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


def threshold_lookup(thresholds: pd.DataFrame, horizon: str, segment_type: str, segment: str, fallback: float = 0.65) -> float:
    if thresholds.empty:
        return fallback
    hit = thresholds[(thresholds["horizon"].eq(horizon)) & (thresholds["segment_type"].eq(segment_type)) & (thresholds["segment"].eq(segment))]
    return fallback if hit.empty else float(hit.iloc[0]["threshold"])


def learn_group_dtcai_thresholds(lppl_hist: pd.DataFrame) -> pd.DataFrame:
    if lppl_hist.empty:
        return pd.DataFrame(columns=["segment", "caution_threshold", "crash_threshold"])
    rows = []
    for symbol, group in lppl_hist.groupby("symbol"):
        rows.append({"symbol": symbol, "caution_threshold": choose_dtcai_threshold(group, 0.40), "crash_threshold": choose_dtcai_threshold(group, 0.15)})
    rows.append({"symbol": "__ALL__", "caution_threshold": choose_dtcai_threshold(lppl_hist, 0.40), "crash_threshold": choose_dtcai_threshold(lppl_hist, 0.15)})
    return pd.DataFrame(rows)


def choose_dtcai_threshold(sample: pd.DataFrame, target_false_alarm: float) -> float:
    if sample.empty or "label" not in sample:
        return 0.6
    label = sample["label"].astype(int)
    score = pd.to_numeric(sample["lppl_dtcai"], errors="coerce").fillna(0)
    for thr in np.arange(0.25, 0.91, 0.01):
        signal = score.ge(thr)
        negatives = label.eq(0)
        far = float((signal & negatives).sum() / max(negatives.sum(), 1))
        if far <= target_false_alarm:
            return float(thr)
    return 0.9


def apply_group_dtcai_thresholds(panel: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    if thresholds.empty:
        return out
    default = thresholds[thresholds["symbol"].eq("__ALL__")]
    default_caution = float(default.iloc[0]["caution_threshold"]) if not default.empty else 0.3
    default_crash = float(default.iloc[0]["crash_threshold"]) if not default.empty else 0.6
    labels = []
    for _, row in out.iterrows():
        hit = thresholds[thresholds["symbol"].eq(row["symbol"])]
        caution = float(hit.iloc[0]["caution_threshold"]) if not hit.empty else default_caution
        crash = float(hit.iloc[0]["crash_threshold"]) if not hit.empty else default_crash
        score = float(row.get("bubble_score_0_100", 0.0)) / 100.0
        labels.append("crash-alert" if score >= crash else "caution" if score >= caution else "stable")
    out["adaptive_lppl_risk_label"] = labels
    return out


def expected_calibration_error(pred: np.ndarray, actual: np.ndarray, bins: int) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (pred >= lo) & (pred < hi if hi < 1 else pred <= hi)
        if mask.any():
            ece += mask.mean() * abs(float(pred[mask].mean()) - float(actual[mask].mean()))
    return float(ece)


def calibration_bins(data: pd.DataFrame, prob_col: str, target_col: str, horizon: str, kind: str) -> list[dict[str, Any]]:
    rows = []
    sample = data.copy()
    sample["bin"] = pd.qcut(sample[prob_col].rank(method="first"), q=min(10, sample.shape[0]), labels=False, duplicates="drop")
    for bin_id, group in sample.groupby("bin"):
        rows.append(
            {
                "horizon": horizon,
                "subset": f"{kind}_bin_{int(bin_id)}",
                "samples": int(group.shape[0]),
                "brier": np.nan,
                "log_loss": np.nan,
                "ece_10bin": np.nan,
                "mean_pred": float(group[prob_col].mean()),
                "realized_rate": float(group[target_col].mean()),
            }
        )
    return rows


def run_strategy_backtest(panel: pd.DataFrame, histories: dict[str, pd.DataFrame], top_n: int, cost_bps: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if panel.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows = []
    prev_symbols: set[str] = set()
    for date, group in panel.groupby("date"):
        eligible = group[group.get("high_confidence_4w", False).astype(bool)] if "high_confidence_4w" in group else group
        if eligible.empty:
            safe = group[group["group"].isin(["Cash/short bonds", "USD cash"])].copy()
            eligible = safe if not safe.empty else group
        regime = str(group["regime"].mode().iloc[0]) if "regime" in group and not group["regime"].mode().empty else ""
        if regime != "Risk-Off / Cash":
            non_cash = eligible[~eligible["group"].isin(SAFE_GROUPS)]
            if non_cash.shape[0] >= max(3, min(top_n, 3)):
                eligible = non_cash
        rank_col = "institutional_score_0_100" if "institutional_score_0_100" in eligible else "score_0_100"
        prob_col = "meta_prob_4w" if "meta_prob_4w" in eligible else "calibrated_prob_4w"
        picks = eligible.sort_values([rank_col, prob_col], ascending=False).head(top_n)
        symbols = set(picks["symbol"])
        gross = float(picks["realized_return_4w"].mean())
        turnover = 1.0 if not prev_symbols else len(symbols.symmetric_difference(prev_symbols)) / max(len(symbols.union(prev_symbols)), 1)
        cost = turnover * cost_bps / 10000.0
        net = gross - cost
        rows.append({"date": date, "holdings": ",".join(picks["symbol"]), "gross_return_4w": gross, "turnover": turnover, "cost": cost, "net_return_4w": net})
        prev_symbols = symbols
    strategy = pd.DataFrame(rows).dropna()
    if strategy.empty:
        return strategy, pd.DataFrame()
    strategy["equity"] = (1 + strategy["net_return_4w"]).cumprod()
    bench = benchmark_returns(histories, strategy["date"], "069500.KS", FORWARD_4W)
    strategy["benchmark_return_4w"] = bench
    strategy["benchmark_equity"] = (1 + strategy["benchmark_return_4w"].fillna(0)).cumprod()
    summary = pd.DataFrame(
        [
            summarize_returns(strategy["net_return_4w"], "strategy", strategy["equity"]),
            summarize_returns(strategy["benchmark_return_4w"], "benchmark_069500", strategy["benchmark_equity"]),
        ]
    )
    return strategy, summary


def benchmark_returns(histories: dict[str, pd.DataFrame], dates: pd.Series, symbol: str, horizon: int) -> list[float]:
    hist = histories.get(symbol)
    if hist is None or hist.empty:
        return [np.nan] * len(dates)
    close = clean_series(hist["Close"])
    out = []
    for date in dates:
        sample = close.loc[:date]
        if sample.empty:
            out.append(np.nan)
        else:
            out.append(forward_return(close, sample.index[-1], horizon))
    return out


def summarize_returns(returns: pd.Series, name: str, equity: pd.Series) -> dict[str, Any]:
    r = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return {"series": name}
    ann_factor = 12.0
    total = float((1 + r).prod() - 1)
    ann_ret = float((1 + total) ** (ann_factor / max(len(r), 1)) - 1)
    ann_vol_value = float(r.std() * math.sqrt(ann_factor))
    sharpe = ann_ret / ann_vol_value if ann_vol_value else np.nan
    dd = equity / equity.cummax() - 1
    return {"series": name, "periods": int(len(r)), "total_return": total, "ann_return": ann_ret, "ann_vol": ann_vol_value, "sharpe": sharpe, "max_drawdown": float(dd.min()), "win_rate": float((r > 0).mean())}


def lppl_false_alarm_validation(lppl_hist: pd.DataFrame) -> pd.DataFrame:
    if lppl_hist.empty or "label" not in lppl_hist:
        return pd.DataFrame()
    rows = []
    for threshold in [0.3, 0.6]:
        for symbol, group in lppl_hist.groupby("symbol"):
            rows.append(false_alarm_row(group, symbol, threshold))
        rows.append(false_alarm_row(lppl_hist, "__ALL__", threshold))
    return pd.DataFrame(rows)


def false_alarm_row(group: pd.DataFrame, symbol: str, threshold: float) -> dict[str, Any]:
    score = pd.to_numeric(group["lppl_dtcai"], errors="coerce").fillna(0)
    label = group["label"].astype(int)
    signal = score >= threshold
    tp = int((signal & label.eq(1)).sum())
    fp = int((signal & label.eq(0)).sum())
    fn = int((~signal & label.eq(1)).sum())
    tn = int((~signal & label.eq(0)).sum())
    return {"symbol": symbol, "threshold": threshold, "samples": int(group.shape[0]), "positives": int(label.sum()), "signals": int(signal.sum()), "recall": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1), "false_alarm_rate": fp / max(fp + tn, 1)}


def calibrate_current_scores(current: pd.DataFrame, calibrators: dict[str, Any], meta_models: dict[str, Any], thresholds: pd.DataFrame, dtcai_thresholds: pd.DataFrame) -> pd.DataFrame:
    out = current.copy()
    out = normalize_current_columns(out)
    for horizon, col in [("1w", "upside_prob_1w"), ("4w", "upside_prob_4w")]:
        if horizon in calibrators and col in out:
            out[f"calibrated_prob_{horizon}"] = calibrators[horizon].predict(pd.to_numeric(out[col], errors="coerce").fillna(0.5))
        elif col in out:
            out[f"calibrated_prob_{horizon}"] = out[col]
    for horizon in ["1w", "4w"]:
        bundle = meta_models.get(horizon)
        if bundle:
            x = feature_matrix(out, bundle["features"]).reindex(columns=bundle["columns"], fill_value=0.0)
            out[f"meta_prob_{horizon}"] = bundle["model"].predict_proba(x)[:, 1]
        elif f"calibrated_prob_{horizon}" in out:
            out[f"meta_prob_{horizon}"] = out[f"calibrated_prob_{horizon}"]
    if not thresholds.empty:
        for horizon in ["1w", "4w"]:
            default = threshold_lookup(thresholds, horizon, "__ALL__", "__ALL__")
            out[f"high_confidence_{horizon}"] = pd.to_numeric(out.get(f"meta_prob_{horizon}", 0), errors="coerce").ge(default)
    if not dtcai_thresholds.empty and "bubble_score_0_100" in out:
        default = dtcai_thresholds[dtcai_thresholds["symbol"].eq("__ALL__")]
        caution_default = float(default.iloc[0]["caution_threshold"]) if not default.empty else 0.3
        crash_default = float(default.iloc[0]["crash_threshold"]) if not default.empty else 0.6
        labels = []
        for _, row in out.iterrows():
            hit = dtcai_thresholds[dtcai_thresholds["symbol"].eq(row["symbol"])]
            caution = float(hit.iloc[0]["caution_threshold"]) if not hit.empty else caution_default
            crash = float(hit.iloc[0]["crash_threshold"]) if not hit.empty else crash_default
            score = float(row.get("bubble_score_0_100", 0.0)) / 100.0
            labels.append("crash-alert" if score >= crash else "caution" if score >= caution else "stable")
        out["adaptive_lppl_risk_label"] = labels
    if "calibrated_prob_4w" in out:
        prob = pd.to_numeric(out.get("meta_prob_4w", out["calibrated_prob_4w"]), errors="coerce").fillna(0.5)
        expected_return = pd.to_numeric(out.get("conditional_avg_return_4w", 0.0), errors="coerce").fillna(0.0)
        technical = pd.to_numeric(out.get("technical_score", out.get("score_0_100", 50.0)), errors="coerce").fillna(50.0)
        base_score = pd.to_numeric(out.get("score_0_100", 50.0), errors="coerce").fillna(50.0)
        confidence_bonus = out.get("high_confidence_4w", False).astype(float) * 5.0 if "high_confidence_4w" in out else 0.0
        crash_penalty = out.get("adaptive_lppl_risk_label", "").eq("crash-alert").astype(float) * 10.0 if "adaptive_lppl_risk_label" in out else 0.0
        out["institutional_score_0_100"] = (
            0.52 * base_score
            + 24.0 * prob
            + 0.14 * technical
            + 260.0 * expected_return.clip(-0.10, 0.10)
            + confidence_bonus
            - crash_penalty
            - cash_opportunity_drag(out)
        ).clip(0, 100).round(2)
        out = out.sort_values("institutional_score_0_100", ascending=False).reset_index(drop=True)
        out["institutional_rank"] = np.arange(1, len(out) + 1)
    return out


def normalize_current_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    aliases = {
        "score_before_lppl": "score_0_100_before_lppl",
        "beta_fit_score": "rolling_beta_fit_score",
        "bubble_score_0_100": "bubble_score_0_100",
    }
    for target, source in aliases.items():
        if target not in out and source in out:
            out[target] = out[source]
    if "conditional_avg_return_4w" not in out:
        out["conditional_avg_return_4w"] = 0.0
    if "regime" not in out and "current_regime" in out:
        out["regime"] = out["current_regime"]
    return out


def write_report(
    summary: pd.DataFrame,
    calibration: pd.DataFrame,
    false_alarm: pd.DataFrame,
    current: pd.DataFrame,
    meta_report: pd.DataFrame,
    thresholds: pd.DataFrame,
    dtcai_thresholds: pd.DataFrame,
    path: Path,
) -> None:
    lines = ["# Walk-Forward Validation And Calibration", ""]
    if not summary.empty:
        lines.extend(["## Strategy Summary", summary.to_markdown(index=False), ""])
    if not calibration.empty:
        lines.extend(["## Probability Metrics", calibration[calibration["subset"].isin(["train", "test", "all"])].to_markdown(index=False), ""])
    if not false_alarm.empty:
        lines.extend(["## LPPL False Alarm", false_alarm[false_alarm["symbol"].eq("__ALL__")].to_markdown(index=False), ""])
    if not meta_report.empty:
        lines.extend(["## Dynamic Meta Model", meta_report.to_markdown(index=False), ""])
    if not thresholds.empty:
        lines.extend(["## High Confidence Thresholds", thresholds[thresholds["segment_type"].eq("__ALL__")].to_markdown(index=False), ""])
    if not dtcai_thresholds.empty:
        lines.extend(["## Adaptive LPPL Thresholds", dtcai_thresholds.head(20).to_markdown(index=False), ""])
    if not current.empty:
        cols = [col for col in ["institutional_rank", "symbol", "name", "group", "institutional_score_0_100", "score_0_100", "meta_prob_1w", "meta_prob_4w", "high_confidence_1w", "high_confidence_4w", "bubble_score_0_100", "adaptive_lppl_risk_label"] if col in current]
        lines.extend(["## Calibrated Current Ranking", current[cols].head(20).to_markdown(index=False), ""])
    lines.extend(
        [
            "## Notes",
            "- Walk-forward scores only use information available before each rebalance date.",
            "- Probability calibration uses isotonic regression with a chronological train/test split.",
            "- Strategy backtest is monthly top-N equal-weight with transaction-cost deduction.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
