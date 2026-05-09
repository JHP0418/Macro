from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RISK_VECTOR = ROOT / "outputs" / "risk_vector_dashboard_latest" / "tables" / "daily_risk_vector.csv"
DRIVER_PANEL = ROOT / "outputs" / "macro_regime_asset_screener_latest" / "tables" / "driver_panel.csv"
OUT_DIR = ROOT / "outputs" / "analog_macro_risk_model_latest"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Point-in-time analog macro risk model.")
    parser.add_argument("--risk-vector", type=Path, default=RISK_VECTOR)
    parser.add_argument("--driver-panel", type=Path, default=DRIVER_PANEL)
    parser.add_argument("--output", type=Path, default=OUT_DIR)
    parser.add_argument("--min-history-days", type=int, default=504)
    parser.add_argument("--exclude-recent-days", type=int, default=21)
    parser.add_argument("--neighbors", type=str, default="20,50,100")
    parser.add_argument("--retrain-step-days", type=int, default=21)
    parser.add_argument("--analog-step-days", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables = args.output / "tables"
    reports = args.output / "reports"
    for path in (tables, reports):
        path.mkdir(parents=True, exist_ok=True)
    ks = [int(x.strip()) for x in args.neighbors.split(",") if x.strip()]
    panel = build_panel(args.risk_vector, args.driver_panel)
    analog = build_point_in_time_analog_features(panel, ks, args.min_history_days, args.exclude_recent_days, args.analog_step_days)
    scored, validation = walkforward_meta_model(analog, args.min_history_days, args.retrain_step_days)
    current = scored.tail(1).copy()

    panel.to_csv(tables / "analog_base_panel.csv", index=False, encoding="utf-8-sig")
    analog.to_csv(tables / "analog_feature_panel.csv", index=False, encoding="utf-8-sig")
    scored.to_csv(tables / "analog_macro_risk_scores.csv", index=False, encoding="utf-8-sig")
    validation.to_csv(tables / "analog_macro_validation.csv", index=False, encoding="utf-8-sig")
    current.to_csv(tables / "current_analog_macro_signal.csv", index=False, encoding="utf-8-sig")
    write_report(validation, current, reports / "analog_macro_risk_report.md")
    print(f"wrote {reports / 'analog_macro_risk_report.md'}")
    print(validation.to_string(index=False))
    print(current[["Date", "analog_risk_score_0_100", "analog_state", "analog_down_prob_1w_model", "analog_down_prob_1m_model", "analog_tail_prob_1m_model"]].to_string(index=False))


def build_panel(risk_vector_path: Path, driver_panel_path: Path) -> pd.DataFrame:
    rv = pd.read_csv(risk_vector_path, parse_dates=["Date"]).sort_values("Date")
    dp = pd.read_csv(driver_panel_path, parse_dates=["Date"]).sort_values("Date")
    df = rv.merge(dp, on="Date", how="left", suffixes=("", "_driver"))
    for col in df.columns:
        if col == "Date":
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).ffill()

    price_cols = ["NASDAQ100", "SP500", "SOX", "RUSSELL2000", "DXY", "USDKRW", "USDJPY", "GOLD", "WTI", "COPPER_GOLD"]
    for col in price_cols:
        if col in df:
            s = pd.to_numeric(df[col], errors="coerce")
            df[f"{col}_fwd_1w"] = s.shift(-5) / s - 1.0
            df[f"{col}_fwd_1m"] = s.shift(-20) / s - 1.0
    if "NASDAQ100" in df:
        nas = pd.to_numeric(df["NASDAQ100"], errors="coerce")
        df["NASDAQ100_fwd_min_1m"] = nas.shift(-1).rolling(20, min_periods=5).min().shift(-19) / nas - 1.0
    if "SOX" in df:
        sox = pd.to_numeric(df["SOX"], errors="coerce")
        df["SOX_fwd_min_1m"] = sox.shift(-1).rolling(20, min_periods=5).min().shift(-19) / sox - 1.0

    df["label_nasdaq_down_1w"] = df.get("NASDAQ100_fwd_1w", pd.Series(index=df.index)).lt(0).astype(int)
    df["label_nasdaq_down_1m"] = df.get("NASDAQ100_fwd_1m", pd.Series(index=df.index)).lt(0).astype(int)
    df["label_nasdaq_tail_1m"] = (
        df.get("NASDAQ100_fwd_1m", pd.Series(index=df.index)).lt(-0.05)
        | df.get("NASDAQ100_fwd_min_1m", pd.Series(index=df.index)).lt(-0.07)
    ).astype(int)
    df["label_sox_down_1w"] = df.get("SOX_fwd_1w", pd.Series(index=df.index)).lt(0).astype(int)
    df["label_sox_down_1m"] = df.get("SOX_fwd_1m", pd.Series(index=df.index)).lt(0).astype(int)
    return df.reset_index(drop=True)


def analog_feature_columns(frame: pd.DataFrame) -> list[str]:
    base = [
        "macro_liquidity_axis_x",
        "market_breakdown_axis_y",
        "external_supply_axis_z",
        "composite_vector_risk",
        "risk_off_score",
        "peak_fragility",
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
        "RAI_z",
        "RAI_level_0_100",
        "RAI_20d_change",
        "RAI_shock_score",
        "RAI_overheat_score",
        "ETF_risk_breadth_pct",
        "ETF_breadth_shock_score",
        "SAFE_ROTATION_shock_score",
        "NASDAQ100",
        "SOX",
        "SP500",
        "RUSSELL2000",
        "DXY",
        "USDKRW",
        "USDJPY",
        "US10Y",
        "US2Y",
        "VIX",
        "VXN",
        "MOVE",
        "HY_OAS",
        "IG_OAS",
        "HYG_IEF",
        "COPPER",
        "WTI",
        "GOLD",
        "COPPER_GOLD",
        "CSI300",
        "HANGSENG_TECH",
        "KOSDAQ_KOSPI",
    ]
    cols = [c for c in base if c in frame and pd.api.types.is_numeric_dtype(frame[c])]
    for col in ["NASDAQ100", "SOX", "SP500", "RUSSELL2000", "DXY", "USDKRW", "USDJPY", "US10Y", "VIX", "HY_OAS", "COPPER_GOLD", "RAI_z", "ETF_risk_breadth_pct"]:
        if col in frame:
            s = pd.to_numeric(frame[col], errors="coerce")
            frame[f"{col}_ret_5d_pt"] = s.pct_change(5)
            frame[f"{col}_ret_20d_pt"] = s.pct_change(20)
            frame[f"{col}_z_60d_pt"] = rolling_z(s, 60)
            cols.extend([f"{col}_ret_5d_pt", f"{col}_ret_20d_pt", f"{col}_z_60d_pt"])
    return [c for c in cols if c in frame]


def rolling_z(s: pd.Series, window: int) -> pd.Series:
    mean = s.rolling(window, min_periods=max(10, window // 3)).mean()
    std = s.rolling(window, min_periods=max(10, window // 3)).std().replace(0, np.nan)
    return (s - mean) / std


def build_point_in_time_analog_features(panel: pd.DataFrame, ks: list[int], min_history_days: int, exclude_recent_days: int, analog_step_days: int = 5) -> pd.DataFrame:
    out = panel.copy()
    cols = analog_feature_columns(out)
    targets = [
        "NASDAQ100_fwd_1w",
        "NASDAQ100_fwd_1m",
        "NASDAQ100_fwd_min_1m",
        "SOX_fwd_1w",
        "SOX_fwd_1m",
    ]
    labels = [
        "label_nasdaq_down_1w",
        "label_nasdaq_down_1m",
        "label_nasdaq_tail_1m",
        "label_sox_down_1w",
        "label_sox_down_1m",
    ]
    for k in ks:
        for name in ["nasdaq", "sox"]:
            for suffix in ["down_prob_1w", "down_prob_1m", "avg_return_1w", "avg_return_1m", "tail_prob_1m", "tail_return_p10_1m"]:
                out[f"analog_k{k}_{name}_{suffix}"] = np.nan
        out[f"analog_k{k}_distance_mean"] = np.nan
        out[f"analog_k{k}_distance_min"] = np.nan
        out[f"analog_k{k}_neighbor_count"] = 0

    x_all = out[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x_values = x_all.to_numpy(dtype=float)
    for i in range(len(out)):
        if i < min_history_days:
            continue
        if analog_step_days > 1 and i % analog_step_days != 0 and i != len(out) - 1:
            continue
        hist_end = i - exclude_recent_days
        if hist_end <= min_history_days // 2:
            continue
        hist_matrix = x_values[:hist_end]
        current = x_values[i]
        valid = np.isfinite(current) & (np.isfinite(hist_matrix).mean(axis=0) > 0.8)
        if int(valid.sum()) < 10:
            continue
        hist_v = hist_matrix[:, valid]
        cur_v = current[valid]
        means = np.nanmean(hist_v, axis=0)
        stds = np.nanstd(hist_v, axis=0)
        good = np.isfinite(means) & np.isfinite(stds) & (stds > 1e-12) & np.isfinite(cur_v)
        if int(good.sum()) < 10:
            continue
        hist_v = hist_v[:, good]
        cur_v = cur_v[good]
        means = means[good]
        stds = stds[good]
        z_hist = (hist_v - means) / stds
        z_cur = (cur_v - means) / stds
        z_hist = np.where(np.isfinite(z_hist), z_hist, 0.0)
        z_cur = np.where(np.isfinite(z_cur), z_cur, 0.0)
        distances = np.sqrt(np.mean((z_hist - z_cur) ** 2, axis=1))
        finite = np.isfinite(distances)
        if not finite.any():
            continue
        hist_positions = np.flatnonzero(finite)
        distances = distances[finite]
        order = np.argsort(distances)
        for k in ks:
            take = order[: min(k, len(order))]
            nidx = hist_positions[take]
            neighbor_dist = distances[take]
            hist = out.loc[nidx]
            out.loc[i, f"analog_k{k}_distance_mean"] = float(np.mean(neighbor_dist))
            out.loc[i, f"analog_k{k}_distance_min"] = float(np.min(neighbor_dist))
            out.loc[i, f"analog_k{k}_neighbor_count"] = int(len(neighbor_dist))
            fill_analog_stats(out, i, k, "nasdaq", hist, "NASDAQ100_fwd_1w", "NASDAQ100_fwd_1m", "NASDAQ100_fwd_min_1m", "label_nasdaq_down_1w", "label_nasdaq_down_1m", "label_nasdaq_tail_1m")
            fill_analog_stats(out, i, k, "sox", hist, "SOX_fwd_1w", "SOX_fwd_1m", "SOX_fwd_min_1m", "label_sox_down_1w", "label_sox_down_1m", "label_sox_down_1m")

    analog_cols = [c for c in out.columns if c.startswith("analog_k")]
    out[analog_cols] = out[analog_cols].ffill()
    return out


def fill_analog_stats(
    out: pd.DataFrame,
    row_idx: int,
    k: int,
    prefix: str,
    hist: pd.DataFrame,
    ret_1w: str,
    ret_1m: str,
    min_1m: str,
    label_1w: str,
    label_1m: str,
    label_tail: str,
) -> None:
    r1w = pd.to_numeric(hist.get(ret_1w), errors="coerce")
    r1m = pd.to_numeric(hist.get(ret_1m), errors="coerce")
    min1m = pd.to_numeric(hist.get(min_1m), errors="coerce") if min_1m in hist else r1m
    out.loc[row_idx, f"analog_k{k}_{prefix}_down_prob_1w"] = float(pd.to_numeric(hist[label_1w], errors="coerce").mean())
    out.loc[row_idx, f"analog_k{k}_{prefix}_down_prob_1m"] = float(pd.to_numeric(hist[label_1m], errors="coerce").mean())
    out.loc[row_idx, f"analog_k{k}_{prefix}_tail_prob_1m"] = float(pd.to_numeric(hist[label_tail], errors="coerce").mean())
    out.loc[row_idx, f"analog_k{k}_{prefix}_avg_return_1w"] = float(r1w.mean())
    out.loc[row_idx, f"analog_k{k}_{prefix}_avg_return_1m"] = float(r1m.mean())
    out.loc[row_idx, f"analog_k{k}_{prefix}_tail_return_p10_1m"] = float(pd.concat([r1m, min1m], axis=1).min(axis=1).quantile(0.10))


def walkforward_meta_model(panel: pd.DataFrame, min_train_days: int, retrain_step_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, brier_score_loss, precision_score, recall_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    out = panel.copy()
    feature_cols = [
        c
        for c in out.columns
        if c.startswith("analog_k")
        or c
        in {
            "risk_off_score",
            "peak_fragility",
            "composite_vector_risk",
            "macro_liquidity_axis_x",
            "market_breakdown_axis_y",
            "external_supply_axis_z",
            "rai_appetite_stress",
            "universe_breadth_stress",
            "safe_rotation_stress",
            "RAI_z",
            "RAI_shock_score",
            "RAI_overheat_score",
            "ETF_breadth_shock_score",
        }
    ]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(out[c])]
    targets = [
        ("nasdaq_down_1w", "label_nasdaq_down_1w"),
        ("nasdaq_down_1m", "label_nasdaq_down_1m"),
        ("nasdaq_tail_1m", "label_nasdaq_tail_1m"),
        ("sox_down_1w", "label_sox_down_1w"),
        ("sox_down_1m", "label_sox_down_1m"),
    ]
    validation_rows = []
    for name, target in targets:
        prob_col = f"analog_{name}_model_prob"
        out[prob_col] = np.nan
        fitted_models: list[Any] | None = None
        last_fit = -10**9
        for i in range(len(out)):
            if i < min_train_days:
                continue
            train = out.iloc[:i].dropna(subset=[target])
            train = train[train[feature_cols].notna().any(axis=1)]
            if train[target].nunique() < 2:
                continue
            if fitted_models is None or i - last_fit >= retrain_step_days:
                x_train = train[feature_cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                y_train = train[target].astype(int)
                models = [
                    make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", C=0.7, random_state=42)),
                    RandomForestClassifier(n_estimators=80, max_depth=4, min_samples_leaf=16, class_weight="balanced_subsample", random_state=42),
                ]
                fitted = []
                for model in models:
                    try:
                        model.fit(x_train, y_train)
                        fitted.append(model)
                    except Exception:
                        continue
                fitted_models = fitted
                last_fit = i
            if not fitted_models:
                continue
            x_test = out.loc[[i], feature_cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            out.loc[i, prob_col] = float(np.mean([m.predict_proba(x_test)[:, 1][0] for m in fitted_models]))
        valid = out.dropna(subset=[prob_col, target]).copy()
        threshold, stats = choose_threshold(valid[prob_col], valid[target])
        pred = valid[prob_col].ge(threshold).astype(int)
        actual = valid[target].astype(int)
        out[f"analog_{name}_signal"] = out[prob_col].ge(threshold).astype(int)
        validation_rows.append(
            {
                "target": name,
                "samples": int(valid.shape[0]),
                "positive_rate": float(actual.mean()) if not valid.empty else np.nan,
                "threshold": threshold,
                "signal_rate": float(pred.mean()) if not valid.empty else np.nan,
                "accuracy": accuracy_score(actual, pred) if not valid.empty else np.nan,
                "precision": precision_score(actual, pred, zero_division=0) if not valid.empty else np.nan,
                "recall": recall_score(actual, pred, zero_division=0) if not valid.empty else np.nan,
                "brier": brier_score_loss(actual, valid[prob_col].clip(0, 1)) if not valid.empty else np.nan,
                "roc_auc": roc_auc_score(actual, valid[prob_col]) if actual.nunique() > 1 else np.nan,
                **stats,
            }
        )

    out["analog_down_prob_1w_model"] = out["analog_nasdaq_down_1w_model_prob"]
    out["analog_down_prob_1m_model"] = out["analog_nasdaq_down_1m_model_prob"]
    out["analog_tail_prob_1m_model"] = out["analog_nasdaq_tail_1m_model_prob"]
    out["analog_risk_score_0_100"] = (
        100
        * (
            0.28 * out["analog_down_prob_1w_model"].fillna(0.0)
            + 0.32 * out["analog_down_prob_1m_model"].fillna(0.0)
            + 0.25 * out["analog_tail_prob_1m_model"].fillna(0.0)
            + 0.15 * pd.to_numeric(out.get("peak_fragility", 0.0), errors="coerce").fillna(0.0) / 100.0
        )
    ).clip(0, 100)
    out["analog_state"] = pd.cut(
        out["analog_risk_score_0_100"],
        bins=[-1, 35, 50, 65, 101],
        labels=["Normal", "Watch", "De-risk", "Cash"],
        right=False,
    ).astype(str)
    return out, pd.DataFrame(validation_rows)


def choose_threshold(prob: pd.Series, actual: pd.Series) -> tuple[float, dict[str, Any]]:
    best_score = -np.inf
    best = (0.5, {"threshold_precision": 0.0, "threshold_recall": 0.0, "threshold_signals": 0})
    base = float(actual.mean()) if len(actual) else 0.0
    for threshold in np.arange(0.25, 0.86, 0.01):
        sig = prob.ge(threshold)
        if sig.sum() < max(10, int(len(prob) * 0.03)) or sig.mean() > 0.45:
            continue
        tp = int((sig & actual.eq(1)).sum())
        fp = int((sig & actual.eq(0)).sum())
        fn = int((~sig & actual.eq(1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        lift = precision / max(base, 1e-9)
        score = 1.25 * recall + 0.9 * precision + 0.2 * min(lift, 5.0) - 0.2 * float(sig.mean())
        if score > best_score:
            best_score = score
            best = (float(threshold), {"threshold_precision": precision, "threshold_recall": recall, "threshold_signals": int(sig.sum())})
    return best


def write_report(validation: pd.DataFrame, current: pd.DataFrame, path: Path) -> None:
    lines = ["# Analog Macro Risk Model", ""]
    if not validation.empty:
        lines.extend(["## Walk-Forward Validation", validation.to_markdown(index=False), ""])
    if not current.empty:
        cols = [
            "Date",
            "analog_risk_score_0_100",
            "analog_state",
            "analog_down_prob_1w_model",
            "analog_down_prob_1m_model",
            "analog_tail_prob_1m_model",
            "analog_k50_nasdaq_down_prob_1w",
            "analog_k50_nasdaq_down_prob_1m",
            "analog_k50_nasdaq_avg_return_1w",
            "analog_k50_nasdaq_avg_return_1m",
            "analog_k50_nasdaq_tail_return_p10_1m",
        ]
        lines.extend(["## Current Signal", current[[c for c in cols if c in current]].to_markdown(index=False), ""])
    lines.extend(
        [
            "## Notes",
            "- Analog features are built point-in-time: each date only uses older historical neighbors.",
            "- Recent 21 trading days are excluded from neighbor search to reduce near-duplicate leakage.",
            "- The model is a supplemental risk layer, not a standalone trading rule.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
