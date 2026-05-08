from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WEEKLY_OUT_DIR = ROOT / "outputs" / "weekly_screening_rank_backtest_latest"
SENTINEL_OUT_DIR = ROOT / "outputs" / "daily_risk_off_sentinel_latest"
OUT_DIR = ROOT / "outputs" / "risk_off_short_horizon_model_latest"

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
    "US cyclical",
    "US REIT",
    "China/HK growth",
    "China equity",
    "India/EM",
    "Japan equity",
    "US high yield",
    "Oil",
}
SAFE_GROUPS = {"Cash/short bonds", "USD cash", "Korea bonds", "US long bonds", "US IG bonds", "Gold", "Korea defensive"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train walk-forward short-horizon risk-off classifier and safe-asset selector.")
    parser.add_argument("--weekly-output", type=Path, default=WEEKLY_OUT_DIR)
    parser.add_argument("--sentinel-output", type=Path, default=SENTINEL_OUT_DIR)
    parser.add_argument("--output", type=Path, default=OUT_DIR)
    parser.add_argument("--min-train-weeks", type=int, default=52)
    parser.add_argument("--top-k-safe", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables = args.output / "tables"
    reports = args.output / "reports"
    for path in (tables, reports):
        path.mkdir(parents=True, exist_ok=True)

    weekly = pd.read_csv(args.weekly_output / "tables" / "weekly_calibrated_rank_panel.csv", parse_dates=["date"])
    sentinel = pd.read_csv(args.sentinel_output / "tables" / "daily_sentinel_history.csv", parse_dates=["Date"])
    date_panel = build_date_panel(weekly, sentinel)
    predictions, model_report = walkforward_risk_models(date_panel, args.min_train_weeks)
    predictions = add_policy_state(predictions)
    safe_panel = build_safe_asset_panel(weekly, predictions)
    safe_ranked, safe_report = walkforward_safe_asset_selector(safe_panel, args.min_train_weeks, args.top_k_safe)
    current = current_signal(predictions, safe_ranked, args.top_k_safe)

    date_panel.to_csv(tables / "risk_off_training_date_panel.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(tables / "risk_off_walkforward_predictions.csv", index=False, encoding="utf-8-sig")
    model_report.to_csv(tables / "risk_off_model_validation.csv", index=False, encoding="utf-8-sig")
    safe_ranked.to_csv(tables / "safe_asset_walkforward_rankings.csv", index=False, encoding="utf-8-sig")
    safe_report.to_csv(tables / "safe_asset_selector_validation.csv", index=False, encoding="utf-8-sig")
    current.to_csv(tables / "current_risk_off_ml_signal.csv", index=False, encoding="utf-8-sig")
    write_report(model_report, safe_report, current, reports / "risk_off_short_horizon_model.md")

    print(f"wrote {reports / 'risk_off_short_horizon_model.md'}")
    print(model_report.to_string(index=False))
    print(safe_report.to_string(index=False))


def build_date_panel(weekly: pd.DataFrame, sentinel: pd.DataFrame) -> pd.DataFrame:
    weekly = weekly.copy()
    weekly["is_risk_asset"] = weekly["group"].isin(RISK_GROUPS)
    risk = (
        weekly[weekly["is_risk_asset"]]
        .groupby("date", as_index=False)
        .agg(
            risk_asset_return_1w=("realized_return_1w", "mean"),
            risk_asset_return_1m=("realized_return_4w", "mean"),
            risk_asset_left_tail_1w=("realized_return_1w", lambda x: float(np.nanquantile(x, 0.25))),
            risk_asset_left_tail_1m=("realized_return_4w", lambda x: float(np.nanquantile(x, 0.25))),
            risk_asset_negative_rate_1w=("realized_return_1w", lambda x: float(np.mean(np.asarray(x) < 0))),
            risk_asset_negative_rate_1m=("realized_return_4w", lambda x: float(np.mean(np.asarray(x) < 0))),
        )
    )
    sent = sentinel.sort_values("Date").copy()
    numeric_cols = [c for c in sent.columns if c != "Date" and pd.api.types.is_numeric_dtype(sent[c])]
    sent[numeric_cols] = sent[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    merged = pd.merge_asof(risk.sort_values("date"), sent.sort_values("Date"), left_on="date", right_on="Date", direction="backward")
    merged = merged.drop(columns=["Date"])
    merged["label_down_1w"] = merged["risk_asset_return_1w"].lt(0).astype(int)
    merged["label_down_1w_1pct"] = merged["risk_asset_return_1w"].lt(-0.01).astype(int)
    merged["label_down_1w_2pct"] = merged["risk_asset_return_1w"].lt(-0.02).astype(int)
    merged["label_down_1w_3pct"] = merged["risk_asset_return_1w"].lt(-0.03).astype(int)
    merged["label_down_1w_5pct"] = merged["risk_asset_return_1w"].lt(-0.05).astype(int)
    merged["label_down_1m_3pct"] = merged["risk_asset_return_1m"].lt(-0.03).astype(int)
    merged["label_down_1m_5pct"] = merged["risk_asset_return_1m"].lt(-0.05).astype(int)
    merged = add_point_in_time_risk_features(merged)
    return merged.sort_values("date").reset_index(drop=True)


def add_point_in_time_risk_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values("date").reset_index(drop=True).copy()
    engineered: dict[str, pd.Series] = {}

    known_after_close = [
        "risk_asset_return_1w",
        "risk_asset_return_1m",
        "risk_asset_left_tail_1w",
        "risk_asset_left_tail_1m",
        "risk_asset_negative_rate_1w",
        "risk_asset_negative_rate_1m",
    ]
    for col in known_after_close:
        if col not in out:
            continue
        prev = pd.to_numeric(out[col], errors="coerce").shift(1)
        engineered[f"prev_{col}"] = prev
        engineered[f"prev2_{col}"] = pd.to_numeric(out[col], errors="coerce").shift(2)
        engineered[f"{col}_prev4_mean"] = prev.rolling(4, min_periods=2).mean()
        engineered[f"{col}_prev8_mean"] = prev.rolling(8, min_periods=3).mean()
        engineered[f"{col}_prev4_min"] = prev.rolling(4, min_periods=2).min()
        engineered[f"{col}_prev4_max"] = prev.rolling(4, min_periods=2).max()

    score_cols = [
        c
        for c in out.columns
        if c.endswith("_shock_score")
        or c
        in {
            "volatility_score",
            "credit_score",
            "fx_score",
            "equity_score",
            "cyclical_score",
            "supply_shock_score",
            "hedge_bid_score",
            "risk_off_score_raw",
            "risk_off_score",
            "risk_off_momentum_5d",
            "risk_budget_pct",
            "equity_penalty",
            "safe_asset_boost",
        }
    ]
    for col in score_cols:
        s = pd.to_numeric(out[col], errors="coerce")
        engineered[f"{col}_lag1"] = s.shift(1)
        engineered[f"{col}_chg1w"] = s - s.shift(1)
        engineered[f"{col}_chg4w"] = s - s.shift(4)
        engineered[f"{col}_ma4"] = s.rolling(4, min_periods=2).mean()
        engineered[f"{col}_max4"] = s.rolling(4, min_periods=2).max()
        engineered[f"{col}_std4"] = s.rolling(4, min_periods=3).std()
        engineered[f"{col}_accel"] = s.diff() - s.diff().shift(1)

    state = out.get("sentinel_state", pd.Series("", index=out.index)).astype(str)
    engineered["sentinel_is_watch_or_worse"] = state.isin(["Watch", "De-risk", "Cash"]).astype(float)
    engineered["sentinel_is_derisk_or_cash"] = state.isin(["De-risk", "Cash"]).astype(float)
    engineered["sentinel_is_cash"] = state.eq("Cash").astype(float)

    dominant = out.get("dominant_component", pd.Series("", index=out.index)).astype(str)
    engineered["dominant_is_vol"] = dominant.str.contains("VIX|VXN|MOVE", regex=True).astype(float)
    engineered["dominant_is_credit"] = dominant.str.contains("HY|IG|HYG", regex=True).astype(float)
    engineered["dominant_is_fx"] = dominant.str.contains("DXY|USDKRW|USDCNH", regex=True).astype(float)
    engineered["dominant_is_equity"] = dominant.str.contains("SP500|NASDAQ|SOX|RUSSELL|CSI|HANGSENG", regex=True).astype(float)
    engineered["dominant_is_commodity"] = dominant.str.contains("WTI|COPPER|GOLD", regex=True).astype(float)
    return pd.concat([out, pd.DataFrame(engineered, index=out.index)], axis=1).copy()


def walkforward_risk_models(panel: pd.DataFrame, min_train_weeks: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, brier_score_loss, precision_score, recall_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    out = panel.copy().sort_values("date").reset_index(drop=True)
    feature_cols = risk_feature_columns(out)
    targets = [
        ("down_1w", "label_down_1w"),
        ("down_1w_1pct", "label_down_1w_1pct"),
        ("down_1w_2pct", "label_down_1w_2pct"),
        ("down_1w_3pct", "label_down_1w_3pct"),
        ("down_1w_5pct", "label_down_1w_5pct"),
        ("down_1m_3pct", "label_down_1m_3pct"),
        ("down_1m_5pct", "label_down_1m_5pct"),
    ]
    unique_dates = sorted(pd.to_datetime(out["date"]).unique())
    min_obs = max(min_train_weeks, 52)
    rows = []
    for name, target in targets:
        probs = pd.Series(np.nan, index=out.index, dtype=float)
        for date in unique_dates:
            train_idx = out.index[pd.to_datetime(out["date"]).lt(date)]
            test_idx = out.index[pd.to_datetime(out["date"]).eq(date)]
            if len(train_idx) < min_obs or out.loc[train_idx, target].nunique() < 2:
                continue
            x_train = out.loc[train_idx, feature_cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            y_train = out.loc[train_idx, target].astype(int)
            x_test = out.loc[test_idx, feature_cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            models = [
                make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", C=0.7, random_state=42)),
                RandomForestClassifier(n_estimators=60, max_depth=3, min_samples_leaf=8, class_weight="balanced_subsample", random_state=42),
            ]
            model_probs = []
            for model in models:
                try:
                    model.fit(x_train, y_train)
                    model_probs.append(model.predict_proba(x_test)[:, 1])
                except Exception:
                    continue
            if model_probs:
                probs.loc[test_idx] = np.vstack(model_probs).mean(axis=0)
        out[f"prob_{name}"] = probs
        valid = out.dropna(subset=[f"prob_{name}", target]).copy()
        if valid.empty:
            continue
        if name in {"down_1w_3pct", "down_1w_5pct"}:
            threshold, stats = choose_severe_loss_threshold(valid[f"prob_{name}"], valid[target])
        else:
            threshold, stats = choose_precision_threshold(valid[f"prob_{name}"], valid[target])
        valid_pred = valid[f"prob_{name}"].ge(threshold).astype(int)
        auc = roc_auc_score(valid[target], valid[f"prob_{name}"]) if valid[target].nunique() > 1 else np.nan
        rows.append(
            {
                "target": name,
                "samples": int(valid.shape[0]),
                "positive_rate": float(valid[target].mean()),
                "threshold": threshold,
                "signal_rate": float(valid_pred.mean()),
                "accuracy": accuracy_score(valid[target], valid_pred),
                "precision": precision_score(valid[target], valid_pred, zero_division=0),
                "recall": recall_score(valid[target], valid_pred, zero_division=0),
                "brier": brier_score_loss(valid[target], valid[f"prob_{name}"]),
                "roc_auc": auc,
                **stats,
            }
        )
        out[f"signal_{name}"] = out[f"prob_{name}"].ge(threshold).astype(int)
    return out, pd.DataFrame(rows)


def risk_feature_columns(frame: pd.DataFrame) -> list[str]:
    blocked_prefixes = (
        "risk_asset_return_",
        "risk_asset_left_tail_",
        "risk_asset_negative_rate_",
        "label_",
    )
    candidates = [
        c
        for c in frame.columns
        if not c.startswith(blocked_prefixes)
        and (
        c.endswith("_shock_score")
        or c.endswith("_score")
        or c.startswith("prev_")
        or c.startswith("prev2_")
        or c.endswith("_prev4_mean")
        or c.endswith("_prev8_mean")
        or c.endswith("_prev4_min")
        or c.endswith("_prev4_max")
        or c.endswith("_lag1")
        or c.endswith("_chg1w")
        or c.endswith("_chg4w")
        or c.endswith("_ma4")
        or c.endswith("_max4")
        or c.endswith("_std4")
        or c.endswith("_accel")
        or c.startswith("sentinel_is_")
        or c.startswith("dominant_is_")
        or c
        in {
            "risk_off_score_raw",
            "risk_off_score",
            "risk_off_momentum_5d",
            "risk_budget_pct",
            "equity_penalty",
            "safe_asset_boost",
        }
        )
    ]
    return [c for c in candidates if c in frame and pd.api.types.is_numeric_dtype(frame[c])]


def choose_precision_threshold(prob: pd.Series, actual: pd.Series) -> tuple[float, dict[str, Any]]:
    best = (0.5, {"threshold_precision": 0.0, "threshold_recall": 0.0, "threshold_signals": 0})
    for threshold in np.arange(0.35, 0.91, 0.01):
        signal = prob.ge(threshold)
        if signal.sum() < max(5, int(len(prob) * 0.05)):
            continue
        tp = int((signal & actual.eq(1)).sum())
        fp = int((signal & actual.eq(0)).sum())
        fn = int((~signal & actual.eq(1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision >= 0.70:
            return float(threshold), {"threshold_precision": precision, "threshold_recall": recall, "threshold_signals": int(signal.sum())}
        if precision > best[1]["threshold_precision"]:
            best = (float(threshold), {"threshold_precision": precision, "threshold_recall": recall, "threshold_signals": int(signal.sum())})
    return best


def choose_severe_loss_threshold(prob: pd.Series, actual: pd.Series) -> tuple[float, dict[str, Any]]:
    best_score = -math.inf
    best = (0.5, {"threshold_precision": 0.0, "threshold_recall": 0.0, "threshold_signals": 0})
    base_rate = float(actual.mean())
    for threshold in np.arange(0.10, 0.71, 0.01):
        signal = prob.ge(threshold)
        if signal.sum() < max(4, int(len(prob) * 0.03)):
            continue
        if signal.mean() > 0.45:
            continue
        tp = int((signal & actual.eq(1)).sum())
        fp = int((signal & actual.eq(0)).sum())
        fn = int((~signal & actual.eq(1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        lift = precision / max(base_rate, 1e-9)
        score = 2.5 * recall + 0.75 * precision + 0.25 * min(lift, 5.0) - 0.35 * float(signal.mean())
        if score > best_score:
            best_score = score
            best = (float(threshold), {"threshold_precision": precision, "threshold_recall": recall, "threshold_signals": int(signal.sum())})
    return best


def add_policy_state(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    p_down = pd.to_numeric(out.get("prob_down_1w", 0.0), errors="coerce").fillna(0.0)
    p_drop = pd.to_numeric(out.get("prob_down_1w_2pct", 0.0), errors="coerce").fillna(0.0)
    p_severe = pd.to_numeric(out.get("prob_down_1w_3pct", 0.0), errors="coerce").fillna(0.0)
    p_month = pd.to_numeric(out.get("prob_down_1m_5pct", 0.0), errors="coerce").fillna(0.0)
    rule_score = pd.to_numeric(out.get("risk_off_score", 0.0), errors="coerce").fillna(0.0) / 100.0
    out["ml_risk_off_score_0_100"] = (100.0 * (0.25 * p_down + 0.20 * p_drop + 0.25 * p_severe + 0.15 * p_month + 0.15 * rule_score)).clip(0, 100)
    if "signal_down_1w_3pct" in out:
        out.loc[out["signal_down_1w_3pct"].eq(1), "ml_risk_off_score_0_100"] = out.loc[
            out["signal_down_1w_3pct"].eq(1), "ml_risk_off_score_0_100"
        ].clip(lower=50)
    if "signal_down_1w_5pct" in out:
        out.loc[out["signal_down_1w_5pct"].eq(1), "ml_risk_off_score_0_100"] = out.loc[
            out["signal_down_1w_5pct"].eq(1), "ml_risk_off_score_0_100"
        ].clip(lower=65)
    out = apply_sentinel_hard_overlay(out)
    out["ml_policy_state"] = pd.cut(
        out["ml_risk_off_score_0_100"],
        bins=[-1, 35, 50, 65, 101],
        labels=["Normal", "Watch", "De-risk", "Cash"],
        right=False,
    ).astype(str)
    return out


def apply_sentinel_hard_overlay(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    state = out.get("sentinel_state", pd.Series("", index=out.index)).astype(str)
    dominant = out.get("dominant_component", pd.Series("", index=out.index)).astype(str)
    sentinel_watch = state.isin(["Watch", "De-risk", "Cash"])
    sentinel_derisk = state.isin(["De-risk", "Cash"])
    risk_off_score = pd.to_numeric(out.get("risk_off_score", 0.0), errors="coerce").fillna(0.0)
    momentum = pd.to_numeric(out.get("risk_off_momentum_5d", 0.0), errors="coerce").fillna(0.0)

    systemic_driver = dominant.str.contains("VIX|VXN|MOVE|HY|IG|HYG|DXY|USDKRW|USDCNH", regex=True)
    vol_credit_driver = dominant.str.contains("VIX|VXN|MOVE|HY|IG|HYG", regex=True)
    equity_driver = dominant.str.contains("SP500|NASDAQ|SOX|RUSSELL", regex=True)

    watch_overlay = sentinel_watch & (systemic_driver | risk_off_score.ge(28) | momentum.ge(8))
    derisk_overlay = sentinel_derisk | (sentinel_watch & vol_credit_driver & risk_off_score.ge(40))
    cash_overlay = state.eq("Cash") | (sentinel_derisk & vol_credit_driver & risk_off_score.ge(70))

    out["hard_overlay_watch"] = watch_overlay.astype(int)
    out["hard_overlay_derisk"] = derisk_overlay.astype(int)
    out["hard_overlay_cash"] = cash_overlay.astype(int)
    out["hard_overlay_equity_shock"] = (sentinel_watch & equity_driver & risk_off_score.ge(45)).astype(int)

    out.loc[watch_overlay, "ml_risk_off_score_0_100"] = out.loc[watch_overlay, "ml_risk_off_score_0_100"].clip(lower=35)
    out.loc[derisk_overlay, "ml_risk_off_score_0_100"] = out.loc[derisk_overlay, "ml_risk_off_score_0_100"].clip(lower=50)
    out.loc[cash_overlay, "ml_risk_off_score_0_100"] = out.loc[cash_overlay, "ml_risk_off_score_0_100"].clip(lower=65)
    return out


def build_safe_asset_panel(weekly: pd.DataFrame, risk_predictions: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "date",
        "ml_risk_off_score_0_100",
        "ml_policy_state",
        "prob_down_1w",
        "prob_down_1w_2pct",
        "prob_down_1w_3pct",
        "prob_down_1w_5pct",
        "prob_down_1m_5pct",
    ]
    pred = risk_predictions[[c for c in cols if c in risk_predictions]].copy()
    panel = weekly[weekly["group"].isin(SAFE_GROUPS)].merge(pred, on="date", how="left")
    panel["target_safe_return_1w"] = panel["realized_return_1w"]
    panel["target_safe_return_1m"] = panel["realized_return_4w"]
    return panel


def walkforward_safe_asset_selector(panel: pd.DataFrame, min_train_weeks: int, top_k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    out = panel.copy().sort_values("date").reset_index(drop=True)
    feature_cols = safe_feature_columns(out)
    unique_dates = sorted(pd.to_datetime(out["date"]).unique())
    min_obs = max(260, min_train_weeks * 4)
    rows = []
    for horizon, target in [("1w", "target_safe_return_1w"), ("1m", "target_safe_return_1m")]:
        pred = pd.Series(np.nan, index=out.index, dtype=float)
        for date in unique_dates:
            train_idx = out.index[pd.to_datetime(out["date"]).lt(date)]
            test_idx = out.index[pd.to_datetime(out["date"]).eq(date)]
            if len(train_idx) < min_obs:
                continue
            x_train = safe_matrix(out.loc[train_idx], feature_cols)
            y_train = out.loc[train_idx, target].astype(float)
            x_test = safe_matrix(out.loc[test_idx], feature_cols).reindex(columns=x_train.columns, fill_value=0.0)
            models = [
                make_pipeline(StandardScaler(), Ridge(alpha=5.0)),
                RandomForestRegressor(n_estimators=60, max_depth=4, min_samples_leaf=8, random_state=42),
            ]
            preds = []
            for model in models:
                try:
                    model.fit(x_train, y_train)
                    preds.append(model.predict(x_test))
                except Exception:
                    continue
            if preds:
                pred.loc[test_idx] = np.vstack(preds).mean(axis=0)
        out[f"safe_pred_return_{horizon}"] = pred
        rows.append(safe_selector_metrics(out, horizon, target, f"safe_pred_return_{horizon}", top_k))
    return out, pd.DataFrame(rows)


def safe_feature_columns(frame: pd.DataFrame) -> list[str]:
    base = [
        "institutional_score_0_100",
        "score_0_100",
        "technical_score",
        "driver_fit_score",
        "beta_fit_score",
        "calibrated_prob_1w",
        "calibrated_prob_4w",
        "ml_risk_off_score_0_100",
        "prob_down_1w",
        "prob_down_1w_2pct",
        "prob_down_1w_3pct",
        "prob_down_1w_5pct",
        "prob_down_1m_5pct",
    ]
    return [c for c in base if c in frame]


def safe_matrix(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    numeric = frame.reindex(columns=cols).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    cats = pd.get_dummies(frame[["group", "regime"]].astype(str), prefix=["group", "regime"], dtype=float)
    return pd.concat([numeric.reset_index(drop=True), cats.reset_index(drop=True)], axis=1)


def safe_selector_metrics(frame: pd.DataFrame, horizon: str, target: str, pred_col: str, top_k: int) -> dict[str, Any]:
    data = frame.dropna(subset=[target, pred_col, "ml_risk_off_score_0_100"]).copy()
    data = data[data["ml_risk_off_score_0_100"].ge(35)]
    rows = []
    for date, group in data.groupby("date"):
        if group.shape[0] < top_k * 2:
            continue
        picks = group.nlargest(top_k, pred_col)
        actual = group.nlargest(top_k, target)
        rows.append(
            {
                "picked_return": float(picks[target].mean()),
                "safe_avg_return": float(group[target].mean()),
                "actual_top_return": float(actual[target].mean()),
                "hit_rate": len(set(picks["symbol"]) & set(actual["symbol"])) / top_k,
                "beat_safe_avg": float(picks[target].mean() > group[target].mean()),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return {"horizon": horizon, "risk_off_weeks": 0}
    return {
        "horizon": horizon,
        "risk_off_weeks": int(result.shape[0]),
        "picked_return": float(result["picked_return"].mean()),
        "safe_avg_return": float(result["safe_avg_return"].mean()),
        "actual_top_return": float(result["actual_top_return"].mean()),
        "hit_rate": float(result["hit_rate"].mean()),
        "beat_safe_avg_rate": float(result["beat_safe_avg"].mean()),
    }


def current_signal(predictions: pd.DataFrame, safe_ranked: pd.DataFrame, top_k: int) -> pd.DataFrame:
    latest_date = pd.to_datetime(predictions["date"]).max()
    latest_pred = predictions[pd.to_datetime(predictions["date"]).eq(latest_date)].iloc[-1].to_dict()
    latest_safe = safe_ranked[pd.to_datetime(safe_ranked["date"]).eq(latest_date)].copy()
    if "safe_pred_return_1w" in latest_safe:
        latest_safe = latest_safe.sort_values(["safe_pred_return_1w", "safe_pred_return_1m"], ascending=False)
    rows = []
    for _, row in latest_safe.head(top_k).iterrows():
        rows.append(
            {
                "date": latest_date.date().isoformat(),
                "ml_policy_state": latest_pred.get("ml_policy_state"),
                "ml_risk_off_score_0_100": latest_pred.get("ml_risk_off_score_0_100"),
                "prob_down_1w": latest_pred.get("prob_down_1w"),
                "prob_down_1w_2pct": latest_pred.get("prob_down_1w_2pct"),
                "prob_down_1w_3pct": latest_pred.get("prob_down_1w_3pct"),
                "prob_down_1w_5pct": latest_pred.get("prob_down_1w_5pct"),
                "prob_down_1m_5pct": latest_pred.get("prob_down_1m_5pct"),
                "symbol": row.get("symbol"),
                "name": row.get("name"),
                "group": row.get("group"),
                "safe_pred_return_1w": row.get("safe_pred_return_1w"),
                "safe_pred_return_1m": row.get("safe_pred_return_1m"),
                "realized_return_1w": row.get("realized_return_1w"),
                "realized_return_4w": row.get("realized_return_4w"),
            }
        )
    return pd.DataFrame(rows)


def write_report(model_report: pd.DataFrame, safe_report: pd.DataFrame, current: pd.DataFrame, path: Path) -> None:
    lines = ["# Risk-Off Short-Horizon ML Model", ""]
    if not model_report.empty:
        lines.extend(["## Risk-Off Classifier Validation", model_report.to_markdown(index=False), ""])
    if not safe_report.empty:
        lines.extend(["## Safe-Asset Selector Validation", safe_report.to_markdown(index=False), ""])
    if not current.empty:
        lines.extend(["## Current ML Risk-Off Signal And Safe Assets", current.to_markdown(index=False), ""])
    lines.extend(
        [
            "## Notes",
            "- Models are trained in expanding weekly walk-forward order.",
            "- The classifier targets short-horizon risk-asset weakness, not only large crashes.",
            "- The safe-asset selector is evaluated only when the ML risk-off score is at least Watch level.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
