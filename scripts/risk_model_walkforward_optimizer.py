from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from macro_regime_asset_screener import ASSETS, read_price_cache, safe_to_csv  # noqa: E402

RISK_VECTOR = ROOT / "outputs" / "risk_vector_dashboard_latest" / "tables" / "daily_risk_vector.csv"
DRIVER_PANEL = ROOT / "outputs" / "macro_regime_asset_screener_latest" / "tables" / "driver_panel.csv"
OUT_DIR = ROOT / "outputs" / "risk_model_walkforward_optimizer_latest"

SAFE_GROUPS = {"FX cash", "Cash/short bonds", "Gold", "Korea bonds", "US long bonds", "US IG bonds"}
RISK_GROUPS = {
    "Korea broad equity",
    "Korea growth",
    "Korea semiconductor",
    "Korea IT",
    "Korea cyclical",
    "Korea value",
    "US broad equity",
    "US growth",
    "US semiconductor",
    "US cyclical/sector",
    "US REIT",
    "US high yield",
    "Global/Developed equity",
    "China/HK growth",
    "China equity",
    "India/EM",
    "Japan equity",
    "Commodity/Oil",
}


@dataclass(frozen=True)
class TargetSpec:
    target: str
    return_col: str
    drawdown_col: str
    description: str
    family: str


TARGETS = [
    TargetSpec("nasdaq_1w_drop_2pct", "NASDAQ100_fwd_1w", "NASDAQ100_fwd_min_1w", "Nasdaq 1주 -2% 이상", "US growth"),
    TargetSpec("nasdaq_1m_correction", "NASDAQ100_fwd_1m", "NASDAQ100_fwd_min_1m", "Nasdaq 1개월 조정", "US growth"),
    TargetSpec("nasdaq_tail_1m", "NASDAQ100_fwd_1m", "NASDAQ100_fwd_min_1m", "Nasdaq 1개월 급락/tail", "US growth"),
    TargetSpec("sox_1w_drop_3pct", "SOX_fwd_1w", "SOX_fwd_min_1w", "SOX 1주 -3% 이상", "Semiconductor"),
    TargetSpec("sox_1m_correction", "SOX_fwd_1m", "SOX_fwd_min_1m", "SOX 1개월 조정", "Semiconductor"),
    TargetSpec("kospi_1w_drop_2pct", "KOSPI200_fwd_1w", "KOSPI200_fwd_min_1w", "KOSPI200 1주 -2% 이상", "Korea equity"),
    TargetSpec("kospi_1m_correction", "KOSPI200_fwd_1m", "KOSPI200_fwd_min_1m", "KOSPI200 1개월 조정", "Korea equity"),
    TargetSpec("risk_assets_practical_loss_1w", "RISK_ASSET_fwd_1w", "RISK_ASSET_fwd_min_1w", "위험자산 유니버스 1주 실전 손실", "Risk assets"),
    TargetSpec("risk_assets_practical_loss_1m", "RISK_ASSET_fwd_1m", "RISK_ASSET_fwd_min_1m", "위험자산 유니버스 1개월 실전 손실", "Risk assets"),
    TargetSpec("safety_rotation_needed_1w", "RISK_MINUS_SAFE_fwd_1w", "RISK_ASSET_fwd_min_1w", "1주 안전자산 우위 필요", "Safe rotation"),
    TargetSpec("safety_rotation_needed_1m", "RISK_MINUS_SAFE_fwd_1m", "RISK_ASSET_fwd_min_1m", "1개월 안전자산 우위 필요", "Safe rotation"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward optimizer for separated risk-off, correction, safe-asset, and ranking models.")
    parser.add_argument("--risk-vector", type=Path, default=RISK_VECTOR)
    parser.add_argument("--driver-panel", type=Path, default=DRIVER_PANEL)
    parser.add_argument("--output", type=Path, default=OUT_DIR)
    parser.add_argument("--min-train-days", type=int, default=504)
    parser.add_argument("--retrain-step-days", type=int, default=126)
    parser.add_argument("--purge-days", type=int, default=20)
    parser.add_argument("--embargo-days", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables = args.output / "tables"
    reports = args.output / "reports"
    for path in (tables, reports):
        path.mkdir(parents=True, exist_ok=True)

    price_matrix, asset_meta = load_price_matrix()
    risk_panel = build_risk_panel(args.risk_vector, args.driver_panel)
    risk_panel = add_universe_forward_labels(risk_panel, price_matrix, asset_meta)
    optimized = add_risk_scores(risk_panel)
    optimized = add_regime_labels(optimized)
    optimized, model_validation, calibration_validation = add_walkforward_probabilities(
        optimized,
        args.min_train_days,
        args.retrain_step_days,
        args.purge_days,
        args.embargo_days,
    )
    threshold_validation, threshold_signals = optimize_thresholds(
        optimized,
        args.min_train_days,
        args.retrain_step_days,
        args.purge_days,
        args.embargo_days,
    )
    high_conf = high_confidence_rule_validation(optimized)
    false_alarm = false_alarm_taxonomy(optimized, threshold_signals)
    safe_eval, current_safe = safe_asset_selector(price_matrix, asset_meta, threshold_signals, optimized)
    rank_eval, rank_summary = fast_weekly_rank_backtest(price_matrix, asset_meta, optimized, args.top_k)
    current = current_signal(optimized, threshold_signals)

    safe_to_csv(risk_panel, tables / "risk_target_base_panel.csv")
    safe_to_csv(optimized, tables / "risk_target_scored_panel.csv")
    safe_to_csv(model_validation, tables / "risk_probability_model_validation.csv")
    safe_to_csv(calibration_validation, tables / "risk_probability_calibration_validation.csv")
    safe_to_csv(threshold_validation, tables / "risk_threshold_optimization.csv")
    safe_to_csv(threshold_signals, tables / "risk_threshold_signals.csv")
    safe_to_csv(high_conf, tables / "high_confidence_rule_validation.csv")
    safe_to_csv(false_alarm, tables / "false_alarm_taxonomy.csv")
    safe_to_csv(safe_eval, tables / "safe_asset_selector_validation.csv")
    safe_to_csv(current_safe, tables / "current_safe_asset_recommendations.csv")
    safe_to_csv(rank_eval, tables / "fast_weekly_rank_backtest.csv")
    safe_to_csv(rank_summary, tables / "fast_weekly_rank_summary.csv")
    safe_to_csv(current, tables / "current_optimized_risk_signal.csv")
    write_report(current, model_validation, calibration_validation, threshold_validation, high_conf, false_alarm, safe_eval, rank_summary, current_safe, reports / "risk_model_walkforward_optimizer_report.md")
    print(f"wrote {reports / 'risk_model_walkforward_optimizer_report.md'}")
    print(current.to_string(index=False))


def build_risk_panel(risk_vector_path: Path, driver_panel_path: Path) -> pd.DataFrame:
    rv = pd.read_csv(risk_vector_path, parse_dates=["Date"]).sort_values("Date")
    dp = pd.read_csv(driver_panel_path, parse_dates=["Date"]).sort_values("Date")
    df = rv.merge(dp, on="Date", how="left", suffixes=("", "_driver"))
    kospi = read_close("069500.KS").rename("KOSPI200")
    if not kospi.empty:
        df = df.merge(kospi.reset_index(), on="Date", how="left")
    for col in ["NASDAQ100", "SP500", "SOX", "RUSSELL2000", "KOSPI200"]:
        if col in df:
            s = pd.to_numeric(df[col], errors="coerce").replace(0, np.nan).ffill()
            df[f"{col}_fwd_1w"] = s.shift(-5) / s - 1.0
            df[f"{col}_fwd_1m"] = s.shift(-20) / s - 1.0
            df[f"{col}_fwd_min_1w"] = forward_min_return(s, 5)
            df[f"{col}_fwd_min_1m"] = forward_min_return(s, 20)
    df["target_nasdaq_1w_drop_2pct"] = df["NASDAQ100_fwd_min_1w"].le(-0.02).astype(int)
    df["target_nasdaq_1m_correction"] = (df["NASDAQ100_fwd_1m"].le(-0.035) | df["NASDAQ100_fwd_min_1m"].le(-0.055)).astype(int)
    df["target_nasdaq_tail_1m"] = (df["NASDAQ100_fwd_1m"].le(-0.06) | df["NASDAQ100_fwd_min_1m"].le(-0.08)).astype(int)
    df["target_sox_1w_drop_3pct"] = df["SOX_fwd_min_1w"].le(-0.03).astype(int)
    df["target_sox_1m_correction"] = (df["SOX_fwd_1m"].le(-0.055) | df["SOX_fwd_min_1m"].le(-0.09)).astype(int)
    df["target_kospi_1w_drop_2pct"] = df["KOSPI200_fwd_min_1w"].le(-0.02).astype(int)
    df["target_kospi_1m_correction"] = (df["KOSPI200_fwd_1m"].le(-0.035) | df["KOSPI200_fwd_min_1m"].le(-0.055)).astype(int)
    return df.replace([np.inf, -np.inf], np.nan)


def add_universe_forward_labels(df: pd.DataFrame, prices: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Add practical loss labels based on the actual ETF universe, not only index proxies."""
    out = df.copy().sort_values("Date")
    if prices.empty or meta.empty:
        return out
    risk_cols = [s for s in meta.loc[meta["group"].isin(RISK_GROUPS), "symbol"].tolist() if s in prices]
    safe_cols = [s for s in meta.loc[meta["group"].isin(SAFE_GROUPS), "symbol"].tolist() if s in prices]
    if len(risk_cols) < 5 or len(safe_cols) < 2:
        return out

    daily_ret = prices.pct_change().replace([np.inf, -np.inf], np.nan)
    risk_index = (1.0 + daily_ret[risk_cols].mean(axis=1).fillna(0.0)).cumprod().rename("RISK_ASSET")
    safe_index = (1.0 + daily_ret[safe_cols].mean(axis=1).fillna(0.0)).cumprod().rename("SAFE_ASSET")
    basket = pd.concat([risk_index, safe_index], axis=1).reindex(out["Date"]).ffill()
    out = out.merge(basket.reset_index().rename(columns={"index": "Date"}), on="Date", how="left")
    for col in ["RISK_ASSET", "SAFE_ASSET"]:
        s = pd.to_numeric(out[col], errors="coerce").replace(0, np.nan).ffill()
        out[f"{col}_fwd_1w"] = s.shift(-5) / s - 1.0
        out[f"{col}_fwd_1m"] = s.shift(-20) / s - 1.0
        out[f"{col}_fwd_min_1w"] = forward_min_return(s, 5)
        out[f"{col}_fwd_min_1m"] = forward_min_return(s, 20)
    out["RISK_MINUS_SAFE_fwd_1w"] = out["RISK_ASSET_fwd_1w"] - out["SAFE_ASSET_fwd_1w"]
    out["RISK_MINUS_SAFE_fwd_1m"] = out["RISK_ASSET_fwd_1m"] - out["SAFE_ASSET_fwd_1m"]

    # Practical loss labels are intentionally harsher than simple return<0.
    # They fire when there is tradable drawdown, negative forward return, or a
    # meaningful opportunity loss versus cash/bonds/gold/USD safe assets.
    out["target_risk_assets_practical_loss_1w"] = (
        out["RISK_ASSET_fwd_min_1w"].le(-0.020)
        | out["RISK_ASSET_fwd_1w"].le(-0.012)
        | out["RISK_MINUS_SAFE_fwd_1w"].le(-0.020)
    ).astype(int)
    out["target_risk_assets_practical_loss_1m"] = (
        out["RISK_ASSET_fwd_min_1m"].le(-0.050)
        | out["RISK_ASSET_fwd_1m"].le(-0.030)
        | out["RISK_MINUS_SAFE_fwd_1m"].le(-0.040)
    ).astype(int)
    out["target_safety_rotation_needed_1w"] = (
        out["RISK_MINUS_SAFE_fwd_1w"].le(-0.015)
        & (out["RISK_ASSET_fwd_min_1w"].le(-0.010) | out["SAFE_ASSET_fwd_1w"].gt(0.002))
    ).astype(int)
    out["target_safety_rotation_needed_1m"] = (
        out["RISK_MINUS_SAFE_fwd_1m"].le(-0.030)
        & (out["RISK_ASSET_fwd_min_1m"].le(-0.025) | out["SAFE_ASSET_fwd_1m"].gt(0.006))
    ).astype(int)
    return out.replace([np.inf, -np.inf], np.nan)


def add_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["crash_sentinel_score"] = weighted_sum(
        out,
        {
            "risk_off_score": 0.22,
            "macro_liquidity_axis_x": 0.20,
            "volatility_stress": 0.18,
            "liquidity_credit_stress": 0.16,
            "fx_external_stress": 0.14,
            "RAI_shock_score": 0.10,
        },
    )
    out["peak_correction_score"] = weighted_sum(
        out,
        {
            "peak_fragility": 0.26,
            "correction_pressure": 0.26,
            "analog_macro_risk": 0.18,
            "RAI_overheat_score": 0.12,
            "universe_breadth_stress": 0.10,
            "market_breakdown_axis_y": 0.08,
        },
    )
    out["risk_off_avoidance_score"] = weighted_sum(
        out,
        {
            "crash_sentinel_score": 0.22,
            "peak_correction_score": 0.28,
            "composite_vector_risk": 0.18,
            "correction_pressure": 0.17,
            "analog_macro_risk": 0.10,
            "safe_rotation_stress": 0.05,
        },
    )
    out["risk_on_permission_score"] = (100.0 - out["risk_off_avoidance_score"]).clip(0, 100)
    return out


def add_regime_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    risk = pd.to_numeric(out.get("risk_off_avoidance_score", 0.0), errors="coerce").fillna(0.0)
    crash = pd.to_numeric(out.get("crash_sentinel_score", 0.0), errors="coerce").fillna(0.0)
    peak = pd.to_numeric(out.get("peak_correction_score", 0.0), errors="coerce").fillna(0.0)
    out["model_regime"] = np.select(
        [
            (crash.ge(55) | risk.ge(62)),
            (risk.ge(45) | peak.ge(55)),
            risk.le(30) & peak.le(38),
        ],
        ["Risk-Off", "Fragile", "Risk-On"],
        default="Transition",
    )
    out["calibration_group"] = np.select(
        [
            out.get("RAI_shock_score", pd.Series(0, index=out.index)).ge(40),
            out.get("RAI_overheat_score", pd.Series(0, index=out.index)).ge(35),
            out.get("ETF_breadth_shock_score", pd.Series(0, index=out.index)).ge(45),
        ],
        ["RAI fear", "RAI overheat", "Breadth break"],
        default=out["model_regime"],
    )
    return out


def weighted_sum(frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    total = pd.Series(0.0, index=frame.index)
    wsum = 0.0
    for col, weight in weights.items():
        if col in frame:
            total += pd.to_numeric(frame[col], errors="coerce").fillna(0.0).clip(0, 100) * weight
            wsum += weight
    return (total / max(wsum, 1e-9)).clip(0, 100)


def add_walkforward_probabilities(
    frame: pd.DataFrame,
    min_train_days: int,
    retrain_step_days: int,
    purge_days: int,
    embargo_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, brier_score_loss, precision_score, recall_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    out = frame.copy().sort_values("Date").reset_index(drop=True)
    feature_cols = risk_feature_columns(out)
    rows = []
    for spec in TARGETS:
        print(f"training walk-forward probability: {spec.target}", flush=True)
        target = f"target_{spec.target}"
        prob_col = f"prob_{spec.target}"
        out[prob_col] = np.nan
        for start in range(min_train_days, len(out), retrain_step_days):
            train_end = max(0, start - purge_days)
            train = out.iloc[:train_end].dropna(subset=[target])
            if train[target].nunique() < 2:
                continue
            x_train = train[feature_cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            y_train = train[target].astype(int)
            models = [
                make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", C=0.45, random_state=42)),
                ExtraTreesClassifier(n_estimators=18, max_depth=5, min_samples_leaf=32, class_weight="balanced", random_state=42, n_jobs=-1),
            ]
            fitted = []
            for model in models:
                try:
                    model.fit(x_train, y_train)
                    fitted.append(model)
                except Exception:
                    continue
            if not fitted:
                continue
            end = min(start + retrain_step_days, len(out))
            x_test = out.loc[start : end - 1, feature_cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            out.loc[start : end - 1, prob_col] = np.mean([model.predict_proba(x_test)[:, 1] for model in fitted], axis=0)

        valid = out.dropna(subset=[prob_col, target]).copy()
        pred = valid[prob_col].ge(0.50).astype(int)
        actual = valid[target].astype(int)
        rows.append(validation_row(spec.target, spec.description, "model_prob_0.50", valid, pred, actual, prob_col, spec.return_col, spec.drawdown_col))
    out, calibration_rows = calibrate_probabilities(out, min_train_days, retrain_step_days, purge_days, embargo_days)
    for spec in TARGETS:
        target = f"target_{spec.target}"
        cal_col = f"prob_cal_{spec.target}"
        if cal_col not in out:
            continue
        valid = out.dropna(subset=[cal_col, target, spec.return_col, spec.drawdown_col]).copy()
        pred = valid[cal_col].ge(0.50).astype(int)
        actual = valid[target].astype(int)
        rows.append(validation_row(spec.target, spec.description, "asset_regime_calibrated_prob_0.50", valid, pred, actual, cal_col, spec.return_col, spec.drawdown_col))
    return out, pd.DataFrame(rows), pd.DataFrame(calibration_rows)


def calibrate_probabilities(
    frame: pd.DataFrame,
    min_train_days: int,
    retrain_step_days: int,
    purge_days: int,
    embargo_days: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    out = frame.copy()
    rows: list[dict[str, Any]] = []
    for spec in TARGETS:
        raw_col = f"prob_{spec.target}"
        target = f"target_{spec.target}"
        cal_col = f"prob_cal_{spec.target}"
        out[cal_col] = np.nan
        if raw_col not in out:
            continue
        for start in range(min_train_days, len(out), retrain_step_days):
            train_end = max(0, start - purge_days - embargo_days)
            train = out.iloc[:train_end].dropna(subset=[raw_col, target])
            if train[target].nunique() < 2 or train.shape[0] < 120:
                continue
            end = min(start + retrain_step_days, len(out))
            test = out.iloc[start:end]
            out.loc[test.index, cal_col] = vector_calibrated_probability(train, test, raw_col, target)
        valid = out.dropna(subset=[cal_col, raw_col, target]).copy()
        if not valid.empty:
            rows.append(
                {
                    "target": spec.target,
                    "family": spec.family,
                    "samples": int(len(valid)),
                    "raw_brier": brier(valid[target], valid[raw_col]),
                    "calibrated_brier": brier(valid[target], valid[cal_col]),
                    "raw_avg_prob": float(valid[raw_col].mean()),
                    "calibrated_avg_prob": float(valid[cal_col].mean()),
                    "actual_rate": float(valid[target].mean()),
                }
            )
    return out, rows


def vector_calibrated_probability(train: pd.DataFrame, test: pd.DataFrame, prob_col: str, target_col: str) -> pd.Series:
    base = float(train[target_col].mean())
    result = pd.Series(base, index=test.index, dtype=float)
    bins = np.unique(np.nanquantile(train[prob_col], np.linspace(0, 1, 9)))
    if len(bins) < 3:
        return result.clip(0.02, 0.98)
    bins[0] = -np.inf
    bins[-1] = np.inf
    train_bin = pd.cut(train[prob_col], bins=bins, include_lowest=True)
    test_bin = pd.cut(test[prob_col], bins=bins, include_lowest=True)
    group_col = "calibration_group" if "calibration_group" in train and "calibration_group" in test else None
    global_stats = train.groupby(train_bin, observed=False)[target_col].agg(["mean", "count"])
    if group_col:
        grouped_stats = train.groupby([group_col, train_bin], observed=False)[target_col].agg(["mean", "count"])
    else:
        grouped_stats = pd.DataFrame()
    for idx in test.index:
        bucket = test_bin.loc[idx]
        if pd.isna(bucket):
            continue
        local_mean = global_stats.at[bucket, "mean"] if bucket in global_stats.index else base
        local_count = global_stats.at[bucket, "count"] if bucket in global_stats.index else 0
        if group_col:
            key = (test.at[idx, group_col], bucket)
            if key in grouped_stats.index and grouped_stats.at[key, "count"] >= 40:
                local_mean = grouped_stats.at[key, "mean"]
                local_count = grouped_stats.at[key, "count"]
        shrink = min(float(local_count) / 180.0, 1.0)
        result.at[idx] = shrink * float(local_mean) + (1.0 - shrink) * base
    return result.clip(0.02, 0.98)


def brier(actual: pd.Series, prob: pd.Series) -> float:
    actual = pd.to_numeric(actual, errors="coerce")
    prob = pd.to_numeric(prob, errors="coerce").clip(0, 1)
    ok = actual.notna() & prob.notna()
    return float(((actual[ok] - prob[ok]) ** 2).mean()) if ok.any() else np.nan


def risk_feature_columns(frame: pd.DataFrame) -> list[str]:
    explicit = {
        "risk_off_score",
        "risk_off_momentum_5d",
        "macro_liquidity_axis_x",
        "market_breakdown_axis_y",
        "external_supply_axis_z",
        "composite_vector_risk",
        "liquidity_credit_stress",
        "equity_breakdown_stress",
        "volatility_stress",
        "fx_external_stress",
        "cyclical_china_stress",
        "inflation_supply_stress",
        "hedge_demand",
        "rai_appetite_stress",
        "universe_breadth_stress",
        "safe_rotation_stress",
        "peak_fragility",
        "analog_macro_risk",
        "correction_pressure",
        "RAI_z",
        "RAI_20d_change",
        "RAI_shock_score",
        "RAI_overheat_score",
        "ETF_risk_breadth_pct",
        "ETF_breadth_shock_score",
        "SAFE_ROTATION_shock_score",
        "crash_sentinel_score",
        "peak_correction_score",
        "risk_off_avoidance_score",
    }
    shock_cols = [c for c in frame.columns if c.endswith("_shock_score")]
    cols = [c for c in sorted(explicit | set(shock_cols)) if c in frame and pd.api.types.is_numeric_dtype(frame[c])]
    return cols


def optimize_thresholds(
    frame: pd.DataFrame,
    min_train_days: int,
    retrain_step_days: int,
    purge_days: int,
    embargo_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_map = {
        "prob": None,
        "prob_cal": "calibrated_probability",
        "risk_off_avoidance_score": "risk_off_avoidance_score",
        "peak_correction_score": "peak_correction_score",
        "crash_sentinel_score": "crash_sentinel_score",
        "composite_vector_risk": "composite_vector_risk",
    }
    validation_rows = []
    signal_frame = frame[["Date"]].copy()
    for spec in TARGETS:
        target = f"target_{spec.target}"
        for label, col in score_map.items():
            if col is None:
                score_col = f"prob_{spec.target}"
            elif col == "calibrated_probability":
                score_col = f"prob_cal_{spec.target}"
            else:
                score_col = col
            if score_col not in frame:
                continue
            signal, threshold = walkforward_threshold(
                frame,
                score_col,
                target,
                spec.return_col,
                spec.drawdown_col,
                min_train_days,
                retrain_step_days,
                purge_days,
                embargo_days,
                regime_aware=label in {"prob", "prob_cal", "risk_off_avoidance_score", "peak_correction_score"},
            )
            signal_col = f"signal_{spec.target}_{label}"
            threshold_col = f"threshold_{spec.target}_{label}"
            signal_frame[signal_col] = signal.astype(int)
            signal_frame[threshold_col] = threshold
            valid = frame.dropna(subset=[score_col, target, spec.return_col, spec.drawdown_col]).copy()
            pred = signal.loc[valid.index].astype(int)
            actual = valid[target].astype(int)
            validation_rows.append(validation_row(spec.target, spec.description, label, valid, pred, actual, score_col, spec.return_col, spec.drawdown_col))
    return pd.DataFrame(validation_rows), signal_frame


def walkforward_threshold(
    frame: pd.DataFrame,
    score_col: str,
    target_col: str,
    return_col: str,
    drawdown_col: str,
    min_train_days: int,
    retrain_step_days: int,
    purge_days: int,
    embargo_days: int,
    regime_aware: bool,
) -> tuple[pd.Series, pd.Series]:
    signal = pd.Series(False, index=frame.index)
    thresholds = pd.Series(np.nan, index=frame.index, dtype=float)
    current_threshold = np.nan
    last_fit = -10**9
    current_regime: object = None
    for i in range(len(frame)):
        if i < min_train_days:
            continue
        regime = frame.at[i, "model_regime"] if regime_aware and "model_regime" in frame else None
        if not np.isfinite(current_threshold) or i - last_fit >= retrain_step_days or regime != current_regime:
            train_end = max(0, i - purge_days - embargo_days)
            train = frame.iloc[:train_end].dropna(subset=[score_col, target_col, return_col, drawdown_col])
            if regime_aware and regime is not None and "model_regime" in train:
                same_regime = train[train["model_regime"].eq(regime)]
                if same_regime.shape[0] >= 160 and same_regime[target_col].nunique() > 1:
                    train = same_regime
            current_threshold = choose_threshold(
                train[score_col].astype(float),
                train[target_col].astype(int),
                pd.to_numeric(train[return_col], errors="coerce"),
                pd.to_numeric(train[drawdown_col], errors="coerce"),
            )
            last_fit = i
            current_regime = regime
        value = frame.at[i, score_col]
        thresholds.iat[i] = current_threshold
        signal.iat[i] = bool(pd.notna(value) and value >= current_threshold)
    return signal, thresholds


def choose_threshold(score: pd.Series, actual: pd.Series, fwd_return: pd.Series | None = None, fwd_drawdown: pd.Series | None = None) -> float:
    if score.empty or actual.nunique() < 2:
        return float(score.quantile(0.75)) if not score.empty else 50.0
    candidates = np.unique(np.nanpercentile(score, np.arange(50, 96, 2)))
    base_rate = float(actual.mean())
    best_threshold = float(np.nanpercentile(score, 75))
    best_objective = -np.inf
    for threshold in candidates:
        sig = score.ge(threshold)
        if sig.sum() < max(12, int(len(score) * 0.025)) or sig.mean() > 0.55:
            continue
        tp = int((sig & actual.eq(1)).sum())
        fp = int((sig & actual.eq(0)).sum())
        fn = int((~sig & actual.eq(1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        lift = precision / max(base_rate, 1e-9)
        signal_rate = float(sig.mean())
        caught_drawdown = 0.0
        missed_drawdown = 0.0
        false_alarm_upside = 0.0
        if fwd_drawdown is not None:
            caught_drawdown = float((-fwd_drawdown[sig]).clip(lower=0).mean()) if sig.any() else 0.0
            missed = ~sig & actual.eq(1)
            missed_drawdown = float((-fwd_drawdown[missed]).clip(lower=0).mean()) if missed.any() else 0.0
        if fwd_return is not None:
            false_alarm = sig & actual.eq(0)
            false_alarm_upside = float(fwd_return[false_alarm].clip(lower=0).mean()) if false_alarm.any() else 0.0
        objective = (
            2.2 * recall
            + 0.9 * precision
            + 0.25 * min(lift, 5.0)
            + 8.0 * caught_drawdown
            - 6.0 * missed_drawdown
            - 5.0 * false_alarm_upside
            - 0.45 * signal_rate
        )
        if objective > best_objective:
            best_objective = objective
            best_threshold = float(threshold)
    return best_threshold


def high_confidence_rule_validation(frame: pd.DataFrame) -> pd.DataFrame:
    rules = {
        "severe_crash_guard": frame["risk_off_score"].ge(55) | ((frame["RAI_shock_score"].ge(55)) & (frame["ETF_breadth_shock_score"].ge(25))),
        "peak_correction_guard": frame["peak_fragility"].ge(58) & frame["correction_pressure"].ge(62),
        "macro_de_risk_guard": frame["analog_macro_risk"].ge(55) & frame["correction_pressure"].ge(60),
        "combined_high_conf_de_risk": frame["risk_off_avoidance_score"].ge(55)
        | (frame["peak_fragility"].ge(58) & frame["correction_pressure"].ge(62))
        | (frame["risk_off_score"].ge(55)),
    }
    rows = []
    for spec in TARGETS:
        target = f"target_{spec.target}"
        valid = frame.dropna(subset=[target, spec.return_col, spec.drawdown_col]).copy()
        actual = valid[target].astype(int)
        for rule_name, sig in rules.items():
            pred = sig.loc[valid.index].astype(int)
            rows.append(validation_row(spec.target, spec.description, rule_name, valid, pred, actual, rule_name, spec.return_col, spec.drawdown_col))
    return pd.DataFrame(rows)


def false_alarm_taxonomy(frame: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    merged = frame.merge(signals, on="Date", how="left")
    preferred = [
        ("nasdaq_1w_drop_2pct", "signal_nasdaq_1w_drop_2pct_prob", "NASDAQ100_fwd_1w", "NASDAQ100_fwd_min_1w"),
        ("nasdaq_1m_correction", "signal_nasdaq_1m_correction_prob", "NASDAQ100_fwd_1m", "NASDAQ100_fwd_min_1m"),
        ("kospi_1w_drop_2pct", "signal_kospi_1w_drop_2pct_prob", "KOSPI200_fwd_1w", "KOSPI200_fwd_min_1w"),
        ("kospi_1m_correction", "signal_kospi_1m_correction_prob", "KOSPI200_fwd_1m", "KOSPI200_fwd_min_1m"),
    ]
    for target_name, signal_col, return_col, drawdown_col in preferred:
        target_col = f"target_{target_name}"
        if signal_col not in merged or target_col not in merged:
            continue
        sample = merged[merged[signal_col].fillna(0).astype(int).eq(1)].dropna(subset=[return_col, drawdown_col])
        if sample.empty:
            continue
        actual = sample[target_col].astype(int)
        taxonomy = pd.Series("bad_false_alarm_up", index=sample.index, dtype=object)
        taxonomy.loc[actual.eq(1)] = "true_positive"
        taxonomy.loc[(actual.eq(0)) & pd.to_numeric(sample[drawdown_col], errors="coerce").le(-0.01)] = "small_drawdown"
        taxonomy.loc[(actual.eq(0)) & pd.to_numeric(sample[return_col], errors="coerce").between(-0.01, 0.01)] = "sideways"
        counts = taxonomy.value_counts(normalize=False)
        rates = taxonomy.value_counts(normalize=True)
        for bucket in ["true_positive", "small_drawdown", "sideways", "bad_false_alarm_up"]:
            rows.append(
                {
                    "target": target_name,
                    "signal": signal_col,
                    "bucket": bucket,
                    "count": int(counts.get(bucket, 0)),
                    "rate": float(rates.get(bucket, 0.0)),
                    "avg_forward_return": float(pd.to_numeric(sample.loc[taxonomy.eq(bucket), return_col], errors="coerce").mean()) if counts.get(bucket, 0) else np.nan,
                    "avg_forward_drawdown": float(pd.to_numeric(sample.loc[taxonomy.eq(bucket), drawdown_col], errors="coerce").mean()) if counts.get(bucket, 0) else np.nan,
                }
            )
    return pd.DataFrame(rows)


SAFE_MACRO_WEIGHTS: dict[str, dict[str, float]] = {
    "FX cash": {"fx_external_stress": 0.45, "volatility_stress": 0.25, "liquidity_credit_stress": 0.20, "RAI_shock_score": 0.10},
    "Cash/short bonds": {"risk_off_avoidance_score": 0.40, "volatility_stress": 0.20, "liquidity_credit_stress": 0.20, "ETF_breadth_shock_score": 0.20},
    "Gold": {"hedge_demand": 0.35, "RAI_overheat_score": 0.15, "volatility_stress": 0.20, "fx_external_stress": 0.15, "external_supply_axis_z": 0.15},
    "Korea bonds": {"peak_correction_score": 0.30, "risk_off_avoidance_score": 0.25, "volatility_stress": 0.20, "fx_external_stress": -0.10, "RAI_shock_score": 0.15},
    "US long bonds": {"peak_correction_score": 0.35, "risk_off_avoidance_score": 0.25, "hedge_demand": 0.20, "inflation_supply_stress": -0.20},
    "US IG bonds": {"peak_correction_score": 0.20, "volatility_stress": 0.15, "liquidity_credit_stress": -0.25, "risk_off_avoidance_score": 0.20},
}


def safe_asset_selector(prices: pd.DataFrame, meta: pd.DataFrame, signals: pd.DataFrame, risk: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    safe_symbols = meta[meta["group"].isin(SAFE_GROUPS)]["symbol"].tolist()
    safe_symbols = [s for s in safe_symbols if s in prices]
    if len(safe_symbols) < 2:
        return pd.DataFrame(), pd.DataFrame()
    signal_cols = [c for c in signals.columns if c.startswith("signal_") and any(k in c for k in ["practical_loss", "safety_rotation", "nasdaq_1m_correction"])]
    signal = signals.set_index("Date")[signal_cols].max(axis=1).fillna(0).astype(bool) if signal_cols else pd.Series(False, index=signals["Date"])
    weekly = weekly_dates(prices.index)
    ret5 = prices[safe_symbols].pct_change(5)
    ret20 = prices[safe_symbols].pct_change(20)
    ret60 = prices[safe_symbols].pct_change(60)
    vol20 = prices[safe_symbols].pct_change().rolling(20, min_periods=10).std()
    tech_score = cross_z(ret20) * 0.40 + cross_z(ret5) * 0.22 + cross_z(ret60) * 0.20 - cross_z(vol20) * 0.18
    macro_score = build_safe_macro_score(tech_score.index, safe_symbols, meta, risk)
    score = tech_score * 0.58 + macro_score * 0.42
    rows = []
    for date in weekly:
        if date not in score.index or date not in signal.index or not bool(signal.loc[date]):
            continue
        scores = score.loc[date].dropna()
        if scores.shape[0] < 2:
            continue
        pred_rank = scores.sort_values(ascending=False)
        for horizon, days in [("1w", 5), ("3w", 15)]:
            future = forward_asset_return(prices[safe_symbols], date, days)
            future = future.dropna()
            common = pred_rank.index.intersection(future.index)
            if len(common) < 2:
                continue
            pred_top1 = pred_rank.loc[common].idxmax()
            actual_top1 = future.loc[common].idxmax()
            pred_top3 = set(pred_rank.loc[common].head(3).index)
            rows.append(
                {
                    "Date": date,
                    "horizon": horizon,
                    "safe_count": int(len(common)),
                    "pred_top1": pred_top1,
                    "actual_top1": actual_top1,
                    "top1_exact": int(pred_top1 == actual_top1),
                    "actual_top1_in_pred_top3": int(actual_top1 in pred_top3),
                    "pred_top1_return": float(future[pred_top1]),
                    "pred_top3_avg_return": float(future[list(pred_top3)].mean()),
                    "safe_universe_avg_return": float(future.loc[common].mean()),
                    "actual_top1_return": float(future[actual_top1]),
                    "model_version": "macro_technical_safe_selector_v2",
                }
            )
    if not rows:
        return pd.DataFrame(), current_safe_recommendations(score, prices, meta, risk)
    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby("horizon", as_index=False)
        .agg(
            signal_weeks=("Date", "count"),
            top1_hit_rate=("top1_exact", "mean"),
            top3_hit_rate=("actual_top1_in_pred_top3", "mean"),
            pred_top1_avg_return=("pred_top1_return", "mean"),
            pred_top3_avg_return=("pred_top3_avg_return", "mean"),
            safe_universe_avg_return=("safe_universe_avg_return", "mean"),
        )
        .assign(row_type="summary")
    )
    return pd.concat([summary, raw.assign(row_type="detail")], ignore_index=True, sort=False), current_safe_recommendations(score, prices, meta, risk)


def build_safe_macro_score(index: pd.DatetimeIndex, safe_symbols: list[str], meta: pd.DataFrame, risk: pd.DataFrame) -> pd.DataFrame:
    risk_idx = risk.set_index("Date").reindex(index).ffill()
    out = pd.DataFrame(0.0, index=index, columns=safe_symbols)
    group_map = meta.set_index("symbol")["group"].to_dict()
    for symbol in safe_symbols:
        group = group_map.get(symbol, "")
        weights = SAFE_MACRO_WEIGHTS.get(group, {})
        if not weights:
            continue
        total = pd.Series(0.0, index=index)
        wsum = 0.0
        for col, weight in weights.items():
            if col not in risk_idx:
                continue
            val = pd.to_numeric(risk_idx[col], errors="coerce").fillna(0.0).clip(0, 100) / 100.0
            total = total + val * weight
            wsum += abs(weight)
        out[symbol] = total / max(wsum, 1e-9)
    return cross_z(out)


def current_safe_recommendations(score: pd.DataFrame, prices: pd.DataFrame, meta: pd.DataFrame, risk: pd.DataFrame) -> pd.DataFrame:
    if score.empty:
        return pd.DataFrame()
    latest_date = score.dropna(how="all").index.max()
    if pd.isna(latest_date):
        return pd.DataFrame()
    latest = score.loc[latest_date].dropna().sort_values(ascending=False).head(12)
    meta_idx = meta.set_index("symbol")
    ret20 = prices[latest.index].pct_change(20).loc[latest_date]
    rows = []
    for rank, (symbol, value) in enumerate(latest.items(), start=1):
        rows.append(
            {
                "Date": latest_date,
                "rank": rank,
                "symbol": symbol,
                "name": meta_idx.at[symbol, "name"] if symbol in meta_idx.index else symbol,
                "group": meta_idx.at[symbol, "group"] if symbol in meta_idx.index else "",
                "safe_score": float(value),
                "return_20d": float(ret20.get(symbol, np.nan)),
            }
        )
    return pd.DataFrame(rows)


def fast_weekly_rank_backtest(prices: pd.DataFrame, meta: pd.DataFrame, risk: pd.DataFrame, top_k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = meta[meta["group"].isin(RISK_GROUPS)]["symbol"].tolist()
    eligible = [s for s in eligible if s in prices]
    if len(eligible) < top_k:
        return pd.DataFrame(), pd.DataFrame()
    px = prices[eligible]
    ret5 = px.pct_change(5)
    ret20 = px.pct_change(20)
    ret60 = px.pct_change(60)
    ret120 = px.pct_change(120)
    vol20 = px.pct_change().rolling(20, min_periods=10).std()
    rank_score = 0.20 * cross_z(ret5) + 0.35 * cross_z(ret20) + 0.30 * cross_z(ret60) + 0.15 * cross_z(ret120) - 0.20 * cross_z(vol20)
    risk_series = risk.set_index("Date")["risk_off_avoidance_score"].reindex(rank_score.index).ffill().fillna(0.0)
    risk_gate = (1.0 - (risk_series / 140.0).clip(0, 0.55)).rename("risk_gate")
    regime = risk.set_index("Date").get("model_regime", pd.Series(index=risk["Date"], dtype=object)).reindex(rank_score.index).ffill().fillna("Transition")
    weekly = weekly_dates(rank_score.index)
    rows = []
    for date in weekly:
        if date not in rank_score.index:
            continue
        if float(risk_series.loc[date]) >= 55:
            continue
        scores = (rank_score.loc[date] * float(risk_gate.loc[date])).dropna()
        if scores.shape[0] < top_k:
            continue
        for horizon, days in [("1w", 5), ("1m", 20)]:
            future = forward_asset_return(px, date, days).dropna()
            common = scores.index.intersection(future.index)
            if len(common) < top_k:
                continue
            pred_top = scores.loc[common].sort_values(ascending=False).head(top_k).index
            actual_top = future.loc[common].sort_values(ascending=False).head(top_k).index
            rows.append(
                {
                    "Date": date,
                    "horizon": horizon,
                    "model_regime": str(regime.loc[date]),
                    "risk_gate": float(risk_gate.loc[date]),
                    "topk_avg_return": float(future[pred_top].mean()),
                    "universe_avg_return": float(future.loc[common].mean()),
                    "actual_topk_avg_return": float(future[actual_top].mean()),
                    "topk_overlap_rate": len(set(pred_top).intersection(actual_top)) / top_k,
                    "top1_exact": int(pred_top[0] == actual_top[0]),
                    "actual_top1_in_pred_topk": int(actual_top[0] in set(pred_top)),
                }
            )
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw, pd.DataFrame()
    summary = (
        raw.groupby(["horizon", "model_regime"], as_index=False)
        .agg(
            weeks=("Date", "count"),
            avg_topk_return=("topk_avg_return", "mean"),
            avg_universe_return=("universe_avg_return", "mean"),
            avg_actual_topk_return=("actual_topk_avg_return", "mean"),
            topk_overlap_rate=("topk_overlap_rate", "mean"),
            top1_exact_hit_rate=("top1_exact", "mean"),
            actual_top1_in_pred_topk_rate=("actual_top1_in_pred_topk", "mean"),
        )
    )
    return raw, summary


def validation_row(
    target: str,
    description: str,
    model: str,
    valid: pd.DataFrame,
    pred: pd.Series,
    actual: pd.Series,
    score_col: str,
    return_col: str,
    drawdown_col: str,
) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, brier_score_loss, precision_score, recall_score, roc_auc_score

    pred = pred.astype(int)
    actual = actual.astype(int)
    score = pd.to_numeric(valid[score_col], errors="coerce") if score_col in valid else pred.astype(float)
    try:
        auc = roc_auc_score(actual, score) if actual.nunique() > 1 and score.notna().sum() > 10 else np.nan
    except Exception:
        auc = np.nan
    try:
        brier = brier_score_loss(actual, score.clip(0, 1)) if score.max(skipna=True) <= 1.0 and score.min(skipna=True) >= 0.0 else np.nan
    except Exception:
        brier = np.nan
    return {
        "target": target,
        "description": description,
        "model": model,
        "samples": int(len(valid)),
        "positive_rate": float(actual.mean()) if len(actual) else np.nan,
        "signal_rate": float(pred.mean()) if len(pred) else np.nan,
        "accuracy": accuracy_score(actual, pred) if len(actual) else np.nan,
        "precision": precision_score(actual, pred, zero_division=0) if len(actual) else np.nan,
        "recall": recall_score(actual, pred, zero_division=0) if len(actual) else np.nan,
        "false_alarm_rate": false_alarm_rate(pred, actual),
        "roc_auc": auc,
        "brier": brier,
        "avg_forward_return_when_signal": float(pd.to_numeric(valid.loc[pred.eq(1), return_col], errors="coerce").mean()) if pred.sum() else np.nan,
        "avg_forward_drawdown_when_signal": float(pd.to_numeric(valid.loc[pred.eq(1), drawdown_col], errors="coerce").mean()) if pred.sum() else np.nan,
    }


def current_signal(frame: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    latest = frame.tail(1).copy()
    sig_latest = signals.tail(1).copy()
    out = latest[[
        "Date",
        "risk_off_score",
        "composite_vector_risk",
        "risk_off_avoidance_score",
        "peak_correction_score",
        "crash_sentinel_score",
        "peak_fragility",
        "analog_macro_risk",
        "correction_pressure",
        "RAI_z",
        "RAI_overheat_score",
        "RAI_shock_score",
        "ETF_breadth_shock_score",
        "model_regime",
    ]].reset_index(drop=True)
    for col in sig_latest.columns:
        if col != "Date" and col.startswith("signal_"):
            out[col] = int(sig_latest[col].iloc[0]) if pd.notna(sig_latest[col].iloc[0]) else 0
    out["optimized_action"] = out.apply(action_from_current, axis=1)
    return out


def action_from_current(row: pd.Series) -> str:
    if (
        row.get("signal_risk_assets_practical_loss_1m_prob_cal", 0)
        or row.get("signal_safety_rotation_needed_1m_prob_cal", 0)
        or row.get("signal_nasdaq_1m_correction_prob_cal", 0)
        or row.get("risk_off_avoidance_score", 0) >= 60
    ):
        return "De-risk: 신규 위험자산 축소, 안전자산 후보 우선"
    if (
        row.get("signal_risk_assets_practical_loss_1w_prob_cal", 0)
        or row.get("signal_nasdaq_1w_drop_2pct_prob_cal", 0)
        or row.get("peak_correction_score", 0) >= 55
    ):
        return "Fragile: 비중 축소/분할, 단기 조정 대비"
    if row.get("crash_sentinel_score", 0) >= 55:
        return "Risk-Off Watch: 급락 방어 우선"
    return "Normal: 위험자산 허용, 단 조정확률 모니터링"


def false_alarm_rate(pred: pd.Series, actual: pd.Series) -> float:
    pred = pred.astype(int)
    actual = actual.astype(int)
    fp = int((pred.eq(1) & actual.eq(0)).sum())
    tp = int((pred.eq(1) & actual.eq(1)).sum())
    return fp / max(fp + tp, 1)


def load_price_matrix() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    series = {}
    for asset in ASSETS:
        close = read_close(asset.symbol)
        if close.shape[0] < 120:
            continue
        series[asset.symbol] = close
        rows.append({"symbol": asset.symbol, "name": asset.name, "group": asset.group})
    prices = pd.DataFrame(series).sort_index().ffill()
    return prices, pd.DataFrame(rows)


def read_close(symbol: str) -> pd.Series:
    hist = read_price_cache(symbol)
    if hist.empty or "Close" not in hist:
        return pd.Series(dtype=float, name=symbol)
    return pd.to_numeric(hist["Close"], errors="coerce").dropna().rename(symbol)


def forward_min_return(s: pd.Series, days: int) -> pd.Series:
    return s.shift(-1).rolling(days, min_periods=max(2, days // 2)).min().shift(-(days - 1)) / s - 1.0


def forward_asset_return(prices: pd.DataFrame, date: pd.Timestamp, days: int) -> pd.Series:
    if date not in prices.index:
        return pd.Series(dtype=float)
    loc = prices.index.get_loc(date)
    if isinstance(loc, slice) or loc + days >= len(prices.index):
        return pd.Series(dtype=float)
    return prices.iloc[loc + days] / prices.iloc[loc] - 1.0


def weekly_dates(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    idx = pd.DatetimeIndex(index).dropna().sort_values()
    return pd.Series(idx, index=idx).groupby(idx.to_period("W-FRI")).max().tolist()


def cross_z(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=1)
    std = frame.std(axis=1).replace(0, np.nan)
    return frame.sub(mean, axis=0).div(std, axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def write_report(
    current: pd.DataFrame,
    model_validation: pd.DataFrame,
    calibration_validation: pd.DataFrame,
    threshold_validation: pd.DataFrame,
    high_conf: pd.DataFrame,
    false_alarm: pd.DataFrame,
    safe_eval: pd.DataFrame,
    rank_summary: pd.DataFrame,
    current_safe: pd.DataFrame,
    path: Path,
) -> None:
    lines = [
        "# Walk-Forward Risk Model Optimizer",
        "",
        "## Current Optimized Signal",
        current.to_markdown(index=False),
        "",
        "## Probability Models",
        model_validation.to_markdown(index=False),
        "",
        "## Asset/Regime Probability Calibration",
        calibration_validation.to_markdown(index=False) if not calibration_validation.empty else "No calibration rows.",
        "",
        "## Best Threshold Signals",
        best_rows(threshold_validation).to_markdown(index=False),
        "",
        "## High Confidence Rules",
        high_conf.sort_values(["target", "model"]).to_markdown(index=False),
        "",
    ]
    if not false_alarm.empty:
        lines.extend(["## False Alarm Taxonomy", false_alarm.to_markdown(index=False), ""])
    if not safe_eval.empty:
        lines.extend(["## Safe Asset Selector", safe_eval[safe_eval["row_type"].eq("summary")].to_markdown(index=False), ""])
    if not current_safe.empty:
        lines.extend(["## Current Safe Asset Recommendations", current_safe.to_markdown(index=False), ""])
    if not rank_summary.empty:
        lines.extend(["## Fast Weekly Risk-On Ranker", rank_summary.to_markdown(index=False), ""])
    lines.extend(
        [
            "## Optimization Logic",
            "- Risk-off detection, correction timing, safe-asset selection, and risk-on ranking are evaluated separately.",
            "- Thresholds are selected only from past observations in an expanding walk-forward loop.",
            "- A purge/embargo gap is applied so labels generated from overlapping future windows do not leak into training.",
            "- The threshold objective favors loss avoidance: recall, precision, captured drawdown, missed-loss penalty, false-alarm opportunity cost, and excessive signal frequency.",
            "- Probabilities are recalibrated by current risk regime/calibration group using only past model predictions.",
            "- Safe assets are selected by a separate macro plus technical model; risky-asset ranking is evaluated only when the risk gate permits it.",
            "- False alarms are split into true false alarms, small drawdowns, and sideways outcomes instead of treating all misses as equal.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def best_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["objective"] = 1.6 * pd.to_numeric(out["recall"], errors="coerce").fillna(0) + pd.to_numeric(out["precision"], errors="coerce").fillna(0) - 0.35 * pd.to_numeric(out["signal_rate"], errors="coerce").fillna(0)
    idx = out.groupby("target")["objective"].idxmax()
    return out.loc[idx].drop(columns=["objective"]).sort_values("target")


if __name__ == "__main__":
    main()
