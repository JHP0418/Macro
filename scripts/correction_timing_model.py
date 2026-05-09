from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALOG_SCORES = ROOT / "outputs" / "analog_macro_risk_model_latest" / "tables" / "analog_macro_risk_scores.csv"
OUT_DIR = ROOT / "outputs" / "correction_timing_model_latest"


TARGETS = {
    "nasdaq_1w_drop_2pct": "나스닥 1주 -2% 이상",
    "nasdaq_1m_correction": "나스닥 1개월 조정",
    "nasdaq_delayed_1m_correction": "나스닥 지연형 1개월 조정",
    "sox_1w_drop_3pct": "SOX 1주 -3% 이상",
    "sox_1m_correction": "SOX 1개월 조정",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train point-in-time correction timing model.")
    parser.add_argument("--input", type=Path, default=ANALOG_SCORES)
    parser.add_argument("--output", type=Path, default=OUT_DIR)
    parser.add_argument("--min-train-days", type=int, default=504)
    parser.add_argument("--retrain-step-days", type=int, default=63)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables = args.output / "tables"
    reports = args.output / "reports"
    for path in (tables, reports):
        path.mkdir(parents=True, exist_ok=True)
    panel = build_panel(args.input)
    scored, validation, importances = walkforward_models(panel, args.min_train_days, args.retrain_step_days)
    scored = add_composite_policy(scored)
    current = scored.tail(1)
    scored.to_csv(tables / "correction_timing_daily_scores.csv", index=False, encoding="utf-8-sig")
    validation.to_csv(tables / "correction_timing_validation.csv", index=False, encoding="utf-8-sig")
    importances.to_csv(tables / "correction_timing_feature_importance.csv", index=False, encoding="utf-8-sig")
    current.to_csv(tables / "current_correction_timing_signal.csv", index=False, encoding="utf-8-sig")
    write_report(validation, current, reports / "correction_timing_report.md")
    print(f"wrote {reports / 'correction_timing_report.md'}")
    print(validation.to_string(index=False))
    print(current[["Date", "correction_pressure_score_0_100", "correction_pressure_state", "correction_1w_drop_prob", "correction_1m_prob", "delayed_correction_prob"]].to_string(index=False))


def build_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    for col in df.columns:
        if col != "Date" and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).ffill()

    nas_1w = pd.to_numeric(df.get("NASDAQ100_fwd_1w"), errors="coerce")
    nas_1m = pd.to_numeric(df.get("NASDAQ100_fwd_1m"), errors="coerce")
    nas_min = pd.to_numeric(df.get("NASDAQ100_fwd_min_1m"), errors="coerce")
    sox_1w = pd.to_numeric(df.get("SOX_fwd_1w"), errors="coerce")
    sox_1m = pd.to_numeric(df.get("SOX_fwd_1m"), errors="coerce")
    sox_min = pd.to_numeric(df.get("SOX_fwd_min_1m"), errors="coerce")

    df["target_nasdaq_1w_drop_2pct"] = nas_1w.le(-0.02).astype(int)
    df["target_nasdaq_1m_correction"] = (nas_1m.le(-0.035) | nas_min.le(-0.055)).astype(int)
    df["target_nasdaq_delayed_1m_correction"] = (nas_1w.gt(-0.01) & (nas_1m.le(-0.03) | nas_min.le(-0.055))).astype(int)
    df["target_sox_1w_drop_3pct"] = sox_1w.le(-0.03).astype(int)
    df["target_sox_1m_correction"] = (sox_1m.le(-0.055) | sox_min.le(-0.09)).astype(int)

    return add_features(df)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for asset in ["NASDAQ100", "SOX", "SP500", "RUSSELL2000", "HYG_IEF", "DXY", "USDKRW", "USDJPY", "US10Y", "US2Y", "VIX", "VXN", "MOVE", "HY_OAS", "COPPER_GOLD", "GOLD", "WTI", "RAI_z", "ETF_risk_breadth_pct"]:
        if asset not in out:
            continue
        s = pd.to_numeric(out[asset], errors="coerce").replace(0, np.nan).ffill()
        for win in (5, 10, 20, 60):
            out[f"{asset}_ret_{win}d_ct"] = s.pct_change(win)
        out[f"{asset}_z_60d_ct"] = rolling_z(s, 60)
        out[f"{asset}_z_252d_ct"] = rolling_z(s, 252)
        out[f"{asset}_dist_high_60d_ct"] = s / s.rolling(60, min_periods=20).max() - 1.0
        out[f"{asset}_dist_high_120d_ct"] = s / s.rolling(120, min_periods=40).max() - 1.0
        out[f"{asset}_dist_ma_60d_ct"] = s / s.rolling(60, min_periods=20).mean() - 1.0
        out[f"{asset}_vol_20d_ct"] = s.pct_change().rolling(20, min_periods=10).std()

    out["nasdaq_sox_rs_20d"] = out.get("NASDAQ100_ret_20d_ct", 0.0) - out.get("SOX_ret_20d_ct", 0.0)
    out["nasdaq_sp500_rs_20d"] = out.get("NASDAQ100_ret_20d_ct", 0.0) - out.get("SP500_ret_20d_ct", 0.0)
    out["nasdaq_russell_rs_20d"] = out.get("NASDAQ100_ret_20d_ct", 0.0) - out.get("RUSSELL2000_ret_20d_ct", 0.0)
    out["sox_sp500_rs_20d"] = out.get("SOX_ret_20d_ct", 0.0) - out.get("SP500_ret_20d_ct", 0.0)
    out["credit_equity_divergence"] = -out.get("HYG_IEF_ret_20d_ct", 0.0) + out.get("NASDAQ100_ret_20d_ct", 0.0)
    out["vol_complacency_reversal"] = -out.get("VIX_z_252d_ct", 0.0) + out.get("NASDAQ100_dist_high_60d_ct", 0.0).abs() * -1.0
    out["peak_without_riskoff"] = out.get("peak_fragility", 0.0) - out.get("risk_off_score", 0.0)
    out["analog_peak_combo"] = 0.5 * out.get("analog_macro_risk", 0.0) + 0.5 * out.get("peak_fragility", 0.0)
    out["breakdown_acceleration"] = -out.get("NASDAQ100_ret_5d_ct", 0.0) + out.get("VIX_ret_5d_ct", 0.0).clip(lower=0)
    out["late_cycle_pressure"] = (
        0.35 * out.get("peak_fragility", 0.0)
        + 0.25 * out.get("analog_macro_risk", 0.0)
        + 0.20 * out.get("fx_external_stress", 0.0)
        + 0.20 * out.get("liquidity_credit_stress", 0.0)
    )
    out["rai_breadth_collapse_pressure"] = (
        0.40 * out.get("rai_appetite_stress", 0.0)
        + 0.35 * out.get("universe_breadth_stress", 0.0)
        + 0.25 * out.get("safe_rotation_stress", 0.0)
    )
    return out.replace([np.inf, -np.inf], np.nan)


def rolling_z(s: pd.Series, window: int) -> pd.Series:
    mean = s.rolling(window, min_periods=max(10, window // 3)).mean()
    std = s.rolling(window, min_periods=max(10, window // 3)).std().replace(0, np.nan)
    return (s - mean) / std


def feature_cols(frame: pd.DataFrame) -> list[str]:
    prefixes = (
        "NASDAQ100_",
        "SOX_",
        "SP500_",
        "RUSSELL2000_",
        "HYG_IEF_",
        "DXY_",
        "USDKRW_",
        "USDJPY_",
        "US10Y_",
        "US2Y_",
        "VIX_",
        "VXN_",
        "MOVE_",
        "HY_OAS_",
        "COPPER_GOLD_",
        "GOLD_",
        "WTI_",
        "RAI_",
        "ETF_",
        "analog_k",
    )
    explicit = {
        "risk_off_score",
        "peak_fragility",
        "analog_macro_risk",
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
        "macro_liquidity_axis_x",
        "market_breakdown_axis_y",
        "external_supply_axis_z",
        "composite_vector_risk",
        "nasdaq_sox_rs_20d",
        "nasdaq_sp500_rs_20d",
        "nasdaq_russell_rs_20d",
        "sox_sp500_rs_20d",
        "credit_equity_divergence",
        "vol_complacency_reversal",
        "peak_without_riskoff",
        "analog_peak_combo",
        "breakdown_acceleration",
        "late_cycle_pressure",
        "rai_breadth_collapse_pressure",
    }
    blocked = ("_fwd_", "target_", "label_")
    cols = []
    for col in frame.columns:
        if any(token in col for token in blocked):
            continue
        if col in explicit or col.startswith(prefixes):
            if pd.api.types.is_numeric_dtype(frame[col]):
                cols.append(col)
    return sorted(set(cols))


def walkforward_models(panel: pd.DataFrame, min_train_days: int, retrain_step_days: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, brier_score_loss, precision_score, recall_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    out = panel.copy().sort_values("Date").reset_index(drop=True)
    cols = feature_cols(out)
    validation_rows = []
    importance_rows = []
    for target_key in TARGETS:
        target = f"target_{target_key}"
        probs = pd.Series(np.nan, index=out.index, dtype=float)
        for i in range(min_train_days, len(out), retrain_step_days):
            train = out.iloc[:i].dropna(subset=[target])
            if train[target].nunique() < 2:
                continue
            x_train = train[cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            y_train = train[target].astype(int)
            models = [
                make_pipeline(StandardScaler(), LogisticRegression(max_iter=900, class_weight="balanced", C=0.4, random_state=42)),
                RandomForestClassifier(n_estimators=60, max_depth=5, min_samples_leaf=16, class_weight="balanced_subsample", random_state=42, n_jobs=-1),
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
            end = min(i + retrain_step_days, len(out))
            x_test = out.loc[i : end - 1, cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            pred = np.mean([model.predict_proba(x_test)[:, 1] for model in fitted], axis=0)
            probs.iloc[i:end] = pred

        prob_col = f"prob_{target_key}"
        out[prob_col] = probs
        valid = out.dropna(subset=[prob_col, target]).copy()
        threshold, threshold_stats = choose_threshold(valid[prob_col], valid[target])
        signal = valid[prob_col].ge(threshold).astype(int)
        actual = valid[target].astype(int)
        validation_rows.append(
            {
                "target": target_key,
                "description": TARGETS[target_key],
                "samples": int(valid.shape[0]),
                "positive_rate": float(actual.mean()),
                "threshold": threshold,
                "signal_rate": float(signal.mean()),
                "accuracy": accuracy_score(actual, signal),
                "precision": precision_score(actual, signal, zero_division=0),
                "recall": recall_score(actual, signal, zero_division=0),
                "brier": brier_score_loss(actual, valid[prob_col].clip(0, 1)),
                "roc_auc": roc_auc_score(actual, valid[prob_col]) if actual.nunique() > 1 else np.nan,
                **threshold_stats,
            }
        )
        out[f"signal_{target_key}"] = out[prob_col].ge(threshold).astype(int)
        out[f"threshold_{target_key}"] = threshold

        try:
            last_rf = [m for m in (fitted or []) if isinstance(m, RandomForestClassifier)]
            if last_rf:
                imp = pd.Series(last_rf[-1].feature_importances_, index=cols).sort_values(ascending=False).head(25)
                for feature, value in imp.items():
                    importance_rows.append({"target": target_key, "feature": feature, "importance": float(value)})
        except Exception:
            pass

    return out, pd.DataFrame(validation_rows), pd.DataFrame(importance_rows)


def choose_threshold(prob: pd.Series, actual: pd.Series) -> tuple[float, dict[str, float]]:
    best_obj = -np.inf
    best = (0.5, {"threshold_precision": 0.0, "threshold_recall": 0.0})
    base_rate = float(actual.mean())
    for threshold in np.arange(0.25, 0.86, 0.01):
        signal = prob.ge(threshold)
        if signal.sum() < max(12, int(len(prob) * 0.025)) or signal.mean() > 0.42:
            continue
        tp = int((signal & actual.eq(1)).sum())
        fp = int((signal & actual.eq(0)).sum())
        fn = int((~signal & actual.eq(1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        lift = precision / max(base_rate, 1e-9)
        obj = 1.4 * recall + 1.0 * precision + 0.25 * min(lift, 4.0) - 0.2 * float(signal.mean())
        if obj > best_obj:
            best_obj = obj
            best = (float(threshold), {"threshold_precision": precision, "threshold_recall": recall})
    return best


def add_composite_policy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["correction_1w_drop_prob"] = out[["prob_nasdaq_1w_drop_2pct", "prob_sox_1w_drop_3pct"]].mean(axis=1)
    out["correction_1m_prob"] = out[["prob_nasdaq_1m_correction", "prob_sox_1m_correction"]].mean(axis=1)
    out["delayed_correction_prob"] = out["prob_nasdaq_delayed_1m_correction"]
    out["correction_pressure_score_0_100"] = (
        100.0
        * (
            0.28 * out["correction_1w_drop_prob"].fillna(0.0)
            + 0.42 * out["correction_1m_prob"].fillna(0.0)
            + 0.22 * out["delayed_correction_prob"].fillna(0.0)
            + 0.08 * (out.get("peak_fragility", 0.0).fillna(0.0) / 100.0)
        )
    ).clip(0, 100)
    out["correction_pressure_state"] = pd.cut(
        out["correction_pressure_score_0_100"],
        bins=[-np.inf, 35, 50, 65, np.inf],
        labels=["Normal", "Watch", "High", "Extreme"],
        right=False,
    ).astype(str)
    return out


def write_report(validation: pd.DataFrame, current: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Correction Timing Model",
        "",
        "This model separates immediate one-week downside, one-month correction, and delayed correction after peak conditions.",
        "",
        "## Current",
        "",
        current[["Date", "correction_pressure_score_0_100", "correction_pressure_state", "correction_1w_drop_prob", "correction_1m_prob", "delayed_correction_prob"]].to_markdown(index=False),
        "",
        "## Walk-forward validation",
        "",
        validation.to_markdown(index=False),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
