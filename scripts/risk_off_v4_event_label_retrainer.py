from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from optimize_risk_off_3d_and_safe_assets import (
    OUT,
    TABLES as V2_TABLES,
    ThresholdPolicy,
    build_risk_panel,
    confusion,
    finite_numeric,
    make_model,
    risk_features,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_V4 = ROOT / "outputs" / "risk_off_v4_event_label_latest"
TABLES = OUT_V4 / "tables"


@dataclass(frozen=True)
class LossPolicy:
    threshold: float
    recall: float
    precision: float
    false_alarm_rate: float
    alert_rate: float
    event_recall_20d: float
    caught_loss_ratio: float
    missed_loss_ratio: float
    opportunity_cost: float
    objective: float


@dataclass(frozen=True)
class StagePolicy:
    watch: LossPolicy
    derisk: LossPolicy
    cash: LossPolicy


def safe_auc(y: pd.Series, p: pd.Series) -> float:
    return float(roc_auc_score(y, p)) if y.nunique() > 1 else np.nan


def safe_ap(y: pd.Series, p: pd.Series) -> float:
    return float(average_precision_score(y, p)) if y.nunique() > 1 else np.nan


def forward_min_return(px: pd.Series, days: int) -> pd.Series:
    parts = [(px.shift(-i) / px - 1.0).rename(i) for i in range(1, days + 1)]
    return pd.concat(parts, axis=1).min(axis=1)


def add_event_labels(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy().sort_values("date").reset_index(drop=True)
    for asset in ["NASDAQ100", "SP500", "SOX"]:
        if asset not in out.columns:
            continue
        px = finite_numeric(out[asset])
        out[f"{asset}_min_fwd_5d"] = forward_min_return(px, 5)
        out[f"{asset}_min_fwd_10d"] = forward_min_return(px, 10)
        out[f"{asset}_min_fwd_20d_v4"] = forward_min_return(px, 20)
        out[f"{asset}_min_fwd_40d"] = forward_min_return(px, 40)

    min5 = [c for c in ["NASDAQ100_min_fwd_5d", "SP500_min_fwd_5d", "SOX_min_fwd_5d"] if c in out.columns]
    min20 = [c for c in ["NASDAQ100_min_fwd_20d_v4", "SP500_min_fwd_20d_v4", "SOX_min_fwd_20d_v4"] if c in out.columns]
    min40 = [c for c in ["NASDAQ100_min_fwd_40d", "SP500_min_fwd_40d", "SOX_min_fwd_40d"] if c in out.columns]
    out["risk_proxy_min_fwd_5d_v4"] = out[min5].mean(axis=1, skipna=True)
    out["risk_proxy_worst_fwd_5d_v4"] = out[min5].min(axis=1, skipna=True)
    out["risk_proxy_min_fwd_20d_v4"] = out[min20].mean(axis=1, skipna=True)
    out["risk_proxy_worst_fwd_20d_v4"] = out[min20].min(axis=1, skipna=True)
    out["risk_proxy_min_fwd_40d_v4"] = out[min40].mean(axis=1, skipna=True)
    out["risk_proxy_worst_fwd_40d_v4"] = out[min40].min(axis=1, skipna=True)

    # 실전 손실 라벨: 단순 종가 수익률이 아니라 앞으로의 최소 낙폭과
    # SOX/Nasdaq 같은 취약 위험자산의 tail loss를 함께 본다.
    out["label_event_loss_1w"] = (
        out["risk_proxy_min_fwd_5d_v4"].le(-0.025)
        | out["risk_proxy_worst_fwd_5d_v4"].le(-0.045)
        | out["risk_proxy_fwd_5d"].le(-0.030)
    ).astype(int)
    out["label_event_loss_1m"] = (
        out["risk_proxy_min_fwd_20d_v4"].le(-0.060)
        | out["risk_proxy_worst_fwd_20d_v4"].le(-0.095)
        | out["risk_proxy_fwd_20d"].le(-0.055)
        | out["risk_proxy_min_fwd_40d_v4"].le(-0.085)
    ).astype(int)

    # Severity는 threshold 선택 때 놓친 손실의 비용으로 쓴다.
    out["severity_1w"] = (
        -out[["risk_proxy_min_fwd_5d_v4", "risk_proxy_worst_fwd_5d_v4", "risk_proxy_fwd_5d"]].min(axis=1)
    ).clip(lower=0)
    out["severity_1m"] = (
        -out[["risk_proxy_min_fwd_20d_v4", "risk_proxy_worst_fwd_20d_v4", "risk_proxy_fwd_20d", "risk_proxy_min_fwd_40d_v4"]].min(axis=1)
    ).clip(lower=0)
    return out


def event_start_indices(y: np.ndarray, min_gap: int = 15) -> list[int]:
    starts: list[int] = []
    last = -10_000
    in_event = False
    for i, value in enumerate(y.astype(bool)):
        if value and not in_event:
            if i - last >= min_gap:
                starts.append(i)
                last = i
            in_event = True
        elif not value:
            in_event = False
    return starts


def event_recall(y: pd.Series, pred: pd.Series, lookback_days: int = 20) -> float:
    yb = y.astype(bool).to_numpy()
    pb = pred.astype(bool).to_numpy()
    starts = event_start_indices(yb)
    if not starts:
        return np.nan
    hits = 0
    for start in starts:
        lo = max(0, start - lookback_days)
        if pb[lo : start + 1].any():
            hits += 1
    return hits / len(starts)


def choose_loss_policy(
    y: pd.Series,
    p: pd.Series,
    severity: pd.Series,
    forward_return: pd.Series,
    max_alert_rate: float,
    min_event_recall: float = 0.80,
) -> LossPolicy:
    yb = y.astype(bool).to_numpy()
    pp = p.astype(float).to_numpy()
    sev = severity.fillna(0.0).astype(float).to_numpy()
    fwd = forward_return.fillna(0.0).astype(float).to_numpy()
    total_loss = float(sev[yb].sum()) or 1.0
    best: LossPolicy | None = None
    fallback: LossPolicy | None = None
    for th in np.linspace(0.03, 0.92, 180):
        pred = pp >= th
        tp = int(np.sum(pred & yb))
        fp = int(np.sum(pred & ~yb))
        fn = int(np.sum(~pred & yb))
        tn = int(np.sum(~pred & ~yb))
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        false_alarm_rate = fp / max(fp + tn, 1)
        alert_rate = float(np.mean(pred))
        ep_recall = event_recall(pd.Series(yb), pd.Series(pred), 20)
        if np.isnan(ep_recall):
            ep_recall = recall
        caught_loss = float(sev[pred & yb].sum()) / total_loss
        missed_loss = float(sev[(~pred) & yb].sum()) / total_loss
        opportunity_cost = float(np.maximum(fwd[pred & ~yb], 0).mean()) if np.any(pred & ~yb) else 0.0
        objective = (
            5.8 * ep_recall
            + 2.8 * caught_loss
            + 1.1 * recall
            + 0.7 * precision
            - 1.0 * false_alarm_rate
            - 2.2 * opportunity_cost
            - 4.0 * max(alert_rate - max_alert_rate, 0.0)
            - 3.0 * max(min_event_recall - ep_recall, 0.0)
            - 2.5 * missed_loss
        )
        cand = LossPolicy(
            threshold=float(th),
            recall=float(recall),
            precision=float(precision),
            false_alarm_rate=float(false_alarm_rate),
            alert_rate=float(alert_rate),
            event_recall_20d=float(ep_recall),
            caught_loss_ratio=float(caught_loss),
            missed_loss_ratio=float(missed_loss),
            opportunity_cost=float(opportunity_cost),
            objective=float(objective),
        )
        if fallback is None or cand.objective > fallback.objective:
            fallback = cand
        if alert_rate <= max_alert_rate and (best is None or cand.objective > best.objective):
            best = cand
    return best or fallback or LossPolicy(0.5, 0, 0, 0, 0, 0, 0, 1, 0, -999)


def choose_cash_policy(y: pd.Series, p: pd.Series, severity: pd.Series, forward_return: pd.Series, max_alert_rate: float) -> LossPolicy:
    yb = y.astype(bool).to_numpy()
    pp = p.astype(float).to_numpy()
    sev = severity.fillna(0.0).astype(float).to_numpy()
    fwd = forward_return.fillna(0.0).astype(float).to_numpy()
    total_loss = float(sev[yb].sum()) or 1.0
    best: LossPolicy | None = None
    for th in np.linspace(0.10, 0.95, 172):
        pred = pp >= th
        tp = int(np.sum(pred & yb))
        fp = int(np.sum(pred & ~yb))
        fn = int(np.sum(~pred & yb))
        tn = int(np.sum(~pred & ~yb))
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        false_alarm_rate = fp / max(fp + tn, 1)
        alert_rate = float(np.mean(pred))
        ep_recall = event_recall(pd.Series(yb), pd.Series(pred), 20)
        if np.isnan(ep_recall):
            ep_recall = recall
        caught_loss = float(sev[pred & yb].sum()) / total_loss
        missed_loss = float(sev[(~pred) & yb].sum()) / total_loss
        opportunity_cost = float(np.maximum(fwd[pred & ~yb], 0).mean()) if np.any(pred & ~yb) else 0.0
        objective = (
            2.2 * precision
            + 2.0 * caught_loss
            + 1.0 * ep_recall
            - 1.5 * false_alarm_rate
            - 2.0 * opportunity_cost
            - 5.0 * max(alert_rate - max_alert_rate, 0.0)
            - 1.2 * missed_loss
        )
        cand = LossPolicy(float(th), float(recall), float(precision), float(false_alarm_rate), float(alert_rate), float(ep_recall), float(caught_loss), float(missed_loss), float(opportunity_cost), float(objective))
        if best is None or cand.objective > best.objective:
            best = cand
    return best or choose_loss_policy(y, p, severity, forward_return, max_alert_rate=max_alert_rate)


def choose_stage_policy(y: pd.Series, p: pd.Series, severity: pd.Series, forward_return: pd.Series, horizon: str) -> StagePolicy:
    # Watch is intentionally sensitive. De-risk is the portfolio cut signal.
    # Cash is the high-conviction stress signal.
    watch = choose_loss_policy(
        y,
        p,
        severity,
        forward_return,
        max_alert_rate=0.78 if horizon == "1m" else 0.72,
        min_event_recall=0.90,
    )
    derisk = choose_loss_policy(
        y,
        p,
        severity,
        forward_return,
        max_alert_rate=0.50 if horizon == "1m" else 0.43,
        min_event_recall=0.82 if horizon == "1m" else 0.78,
    )
    cash = choose_cash_policy(y, p, severity, forward_return, max_alert_rate=0.28 if horizon == "1m" else 0.24)
    derisk_threshold = max(derisk.threshold, watch.threshold + 0.02)
    cash_threshold = min(max(cash.threshold, derisk_threshold + 0.08), 0.95)
    derisk = LossPolicy(derisk_threshold, derisk.recall, derisk.precision, derisk.false_alarm_rate, derisk.alert_rate, derisk.event_recall_20d, derisk.caught_loss_ratio, derisk.missed_loss_ratio, derisk.opportunity_cost, derisk.objective)
    cash = LossPolicy(cash_threshold, cash.recall, cash.precision, cash.false_alarm_rate, cash.alert_rate, cash.event_recall_20d, cash.caught_loss_ratio, cash.missed_loss_ratio, cash.opportunity_cost, cash.objective)
    return StagePolicy(watch=watch, derisk=derisk, cash=cash)


def add_probability_blend(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, features: list[str], label: str) -> tuple[pd.Series, pd.Series]:
    model = make_model()
    model.fit(train[features], train[label].astype(int))
    valid_model_prob = pd.Series(model.predict_proba(valid[features])[:, 1], index=valid.index)
    test_model_prob = pd.Series(model.predict_proba(test[features])[:, 1], index=test.index)
    valid_rule = finite_numeric(valid.get("stress_rule_percentile_3y", pd.Series(index=valid.index))).fillna(0.0).clip(0, 1)
    test_rule = finite_numeric(test.get("stress_rule_percentile_3y", pd.Series(index=test.index))).fillna(0.0).clip(0, 1)
    valid_peak = finite_numeric(valid.get("complacent_peak_fragility_stress", pd.Series(index=valid.index))).fillna(0.0).clip(0, 100) / 100.0
    test_peak = finite_numeric(test.get("complacent_peak_fragility_stress", pd.Series(index=test.index))).fillna(0.0).clip(0, 100) / 100.0
    valid_axis = finite_numeric(valid.get("risk_3d_vector_score", pd.Series(index=valid.index))).fillna(0.0).clip(0, 100) / 100.0
    test_axis = finite_numeric(test.get("risk_3d_vector_score", pd.Series(index=test.index))).fillna(0.0).clip(0, 100) / 100.0
    valid_prob = (0.46 * valid_model_prob + 0.32 * valid_rule + 0.12 * valid_peak + 0.10 * valid_axis).clip(0, 1)
    test_prob = (0.46 * test_model_prob + 0.32 * test_rule + 0.12 * test_peak + 0.10 * test_axis).clip(0, 1)
    return valid_prob, test_prob


def stage_from_prob(prob: pd.Series, watch: float, derisk: float, cash: float) -> pd.Series:
    return pd.cut(
        prob,
        bins=[-np.inf, watch, derisk, cash, np.inf],
        labels=["Normal", "Watch", "De-risk", "Cash"],
        right=False,
    ).astype(str)


def walkforward_v4(panel: pd.DataFrame, label: str, horizon: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = risk_features(panel)
    severity_col = "severity_1w" if horizon == "1w" else "severity_1m"
    fwd_col = "risk_proxy_fwd_5d" if horizon == "1w" else "risk_proxy_fwd_20d"
    pred_rows = []
    metric_rows = []
    for test_year in range(2003, 2027):
        test_start = pd.Timestamp(f"{test_year}-01-01")
        test_end = pd.Timestamp(f"{test_year}-12-31")
        valid_start = test_start - pd.DateOffset(years=2)
        # Purge train/valid around overlapping forward labels.
        train_end = valid_start - pd.Timedelta(days=45 if horizon == "1m" else 15)
        embargo_start = test_start - pd.Timedelta(days=45 if horizon == "1m" else 15)
        train = panel[panel["date"].lt(train_end)]
        valid = panel[(panel["date"].ge(valid_start)) & (panel["date"].lt(embargo_start))]
        test = panel[(panel["date"].ge(test_start)) & (panel["date"].le(test_end))]
        if train[label].sum() < 25 or valid[label].sum() < 4 or test.empty:
            continue
        valid_prob, test_prob = add_probability_blend(train, valid, test, features, label)
        policy = choose_stage_policy(
            valid[label],
            valid_prob,
            valid[severity_col],
            valid[fwd_col],
            horizon,
        )
        cm = confusion(test[label], test_prob, policy.derisk.threshold, test["date"])
        watch_pred = test_prob.ge(policy.watch.threshold)
        pred = test_prob.ge(policy.derisk.threshold)
        cash_pred = test_prob.ge(policy.cash.threshold)
        test_event_recall = event_recall(test[label], pred, 20)
        test_watch_event_recall = event_recall(test[label], watch_pred, 20)
        test_cash_event_recall = event_recall(test[label], cash_pred, 20)
        severity = test[severity_col].fillna(0.0)
        total_loss = float(severity[test[label].astype(bool)].sum()) or 1.0
        metric_rows.append(
            {
                "model": "Risk-Off V4 event-label",
                "horizon": horizon,
                "test_year": test_year,
                "watch_threshold": policy.watch.threshold,
                "derisk_threshold": policy.derisk.threshold,
                "cash_threshold": policy.cash.threshold,
                "valid_watch_event_recall_20d": policy.watch.event_recall_20d,
                "valid_derisk_event_recall_20d": policy.derisk.event_recall_20d,
                "valid_cash_event_recall_20d": policy.cash.event_recall_20d,
                "valid_watch_alert_rate": policy.watch.alert_rate,
                "valid_derisk_alert_rate": policy.derisk.alert_rate,
                "valid_cash_alert_rate": policy.cash.alert_rate,
                "valid_derisk_caught_loss_ratio": policy.derisk.caught_loss_ratio,
                "valid_derisk_missed_loss_ratio": policy.derisk.missed_loss_ratio,
                "test_auc": safe_auc(test[label], test_prob),
                "test_ap": safe_ap(test[label], test_prob),
                "test_brier": float(brier_score_loss(test[label], test_prob)) if test[label].nunique() > 1 else np.nan,
                "test_watch_event_recall_20d": test_watch_event_recall,
                "test_derisk_event_recall_20d": test_event_recall,
                "test_cash_event_recall_20d": test_cash_event_recall,
                "test_watch_alert_rate": float(watch_pred.mean()),
                "test_derisk_alert_rate": float(pred.mean()),
                "test_cash_alert_rate": float(cash_pred.mean()),
                "test_derisk_caught_loss_ratio": float(severity[pred & test[label].astype(bool)].sum()) / total_loss,
                "test_derisk_missed_loss_ratio": float(severity[(~pred) & test[label].astype(bool)].sum()) / total_loss,
                **{f"test_{k}": v for k, v in cm.items()},
                "test_positive_rate": float(test[label].mean()),
                "n_train": int(train.shape[0]),
                "n_valid": int(valid.shape[0]),
                "n_test": int(test.shape[0]),
            }
        )
        out = test[
            [
                "date",
                label,
                severity_col,
                "risk_off_score",
                "risk_3d_dominant_axis",
                "axis1_vol_credit_stress",
                "axis2_fx_liquidity_stress",
                "axis3_peak_fragility_stress",
                "complacent_peak_fragility_stress",
                "stress_rule_percentile_3y",
                "risk_proxy_fwd_5d",
                "risk_proxy_fwd_20d",
                "risk_proxy_min_fwd_20d_v4",
                "risk_proxy_worst_fwd_20d_v4",
            ]
        ].copy()
        out["horizon"] = horizon
        out["risk_off_v4_prob"] = test_prob.to_numpy()
        out["risk_off_v4_watch_threshold"] = policy.watch.threshold
        out["risk_off_v4_derisk_threshold"] = policy.derisk.threshold
        out["risk_off_v4_cash_threshold"] = policy.cash.threshold
        out["risk_off_v4_threshold"] = policy.derisk.threshold
        out["risk_off_v4_watch"] = out["risk_off_v4_prob"].ge(policy.watch.threshold).astype(int)
        out["risk_off_v4_alert"] = pred.astype(int).to_numpy()
        out["risk_off_v4_cash"] = out["risk_off_v4_prob"].ge(policy.cash.threshold).astype(int)
        out["risk_off_v4_stage"] = stage_from_prob(out["risk_off_v4_prob"], policy.watch.threshold, policy.derisk.threshold, policy.cash.threshold)
        pred_rows.append(out)
    return pd.concat(pred_rows, ignore_index=True), pd.DataFrame(metric_rows)


def current_v4_state(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    features = risk_features(panel)
    for horizon, label in [("1w", "label_event_loss_1w"), ("1m", "label_event_loss_1m")]:
        severity_col = "severity_1w" if horizon == "1w" else "severity_1m"
        fwd_col = "risk_proxy_fwd_5d" if horizon == "1w" else "risk_proxy_fwd_20d"
        train = panel[panel[label].notna()].copy()
        valid = train[train["date"].ge(train["date"].max() - pd.DateOffset(years=2))]
        fit = train[train["date"].lt(valid["date"].min() - pd.Timedelta(days=45 if horizon == "1m" else 15))]
        if fit.empty:
            fit = train.iloc[:-252]
        model = make_model()
        model.fit(fit[features], fit[label].astype(int))
        model_prob = pd.Series(model.predict_proba(train[features])[:, 1], index=train.index)
        rule = finite_numeric(train.get("stress_rule_percentile_3y", pd.Series(index=train.index))).fillna(0.0).clip(0, 1)
        peak = finite_numeric(train.get("complacent_peak_fragility_stress", pd.Series(index=train.index))).fillna(0.0).clip(0, 100) / 100.0
        axis = finite_numeric(train.get("risk_3d_vector_score", pd.Series(index=train.index))).fillna(0.0).clip(0, 100) / 100.0
        prob = (0.46 * model_prob + 0.32 * rule + 0.12 * peak + 0.10 * axis).clip(0, 1)
        policy = choose_stage_policy(
            valid[label],
            prob.loc[valid.index],
            valid[severity_col],
            valid[fwd_col],
            horizon,
        )
        latest = train.iloc[-1]
        latest_prob = float(prob.loc[latest.name])
        rows.append(
            {
                "horizon": horizon,
                "date": latest["date"],
                "risk_off_v4_prob": latest_prob,
                "risk_off_v4_watch_threshold": policy.watch.threshold,
                "risk_off_v4_derisk_threshold": policy.derisk.threshold,
                "risk_off_v4_cash_threshold": policy.cash.threshold,
                "risk_off_v4_threshold": policy.derisk.threshold,
                "risk_off_v4_watch": int(latest_prob >= policy.watch.threshold),
                "risk_off_v4_alert": int(latest_prob >= policy.derisk.threshold),
                "risk_off_v4_cash": int(latest_prob >= policy.cash.threshold),
                "risk_off_v4_stage": stage_from_prob(pd.Series([latest_prob]), policy.watch.threshold, policy.derisk.threshold, policy.cash.threshold).iloc[0],
                "risk_off_score": float(latest["risk_off_score"]),
                "axis1_vol_credit_stress": float(latest["axis1_vol_credit_stress"]),
                "axis2_fx_liquidity_stress": float(latest["axis2_fx_liquidity_stress"]),
                "axis3_peak_fragility_stress": float(latest["axis3_peak_fragility_stress"]),
                "dominant_axis": str(latest["risk_3d_dominant_axis"]),
            }
        )
    return pd.DataFrame(rows)


def compare_with_v3(v4_metrics: pd.DataFrame) -> pd.DataFrame:
    v3_path = V2_TABLES / "risk_off_v2_walkforward_metrics.csv"
    rows = []
    if v3_path.exists():
        v3 = pd.read_csv(v3_path)
        for horizon, part in v3.groupby("horizon"):
            rows.append(
                {
                    "model": "Risk-Off V3 previous",
                    "horizon": horizon,
                    "auc": part["test_auc"].mean(),
                    "daily_recall": part["test_recall"].mean(),
                    "precision": part["test_precision"].mean(),
                    "false_alarm_rate": part["test_false_alarm_rate"].mean(),
                    "alert_rate": part["test_alert_rate"].mean(),
                    "watch_event_recall_20d": part.get("test_episode_recall_20d", pd.Series(dtype=float)).mean(),
                    "derisk_event_recall_20d": part.get("test_episode_recall_20d", pd.Series(dtype=float)).mean(),
                    "cash_event_recall_20d": np.nan,
                    "caught_loss_ratio": np.nan,
                    "watch_alert_rate": np.nan,
                    "derisk_alert_rate": part["test_alert_rate"].mean(),
                    "cash_alert_rate": np.nan,
                }
            )
    for horizon, part in v4_metrics.groupby("horizon"):
        rows.append(
            {
                "model": "Risk-Off V4 event-label",
                "horizon": horizon,
                "auc": part["test_auc"].mean(),
                "daily_recall": part["test_recall"].mean(),
                "precision": part["test_precision"].mean(),
                "false_alarm_rate": part["test_false_alarm_rate"].mean(),
                "alert_rate": part["test_alert_rate"].mean(),
                "watch_event_recall_20d": part["test_watch_event_recall_20d"].mean(),
                "derisk_event_recall_20d": part["test_derisk_event_recall_20d"].mean(),
                "cash_event_recall_20d": part["test_cash_event_recall_20d"].mean(),
                "caught_loss_ratio": part["test_derisk_caught_loss_ratio"].mean(),
                "watch_alert_rate": part["test_watch_alert_rate"].mean(),
                "derisk_alert_rate": part["test_derisk_alert_rate"].mean(),
                "cash_alert_rate": part["test_cash_alert_rate"].mean(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    panel = add_event_labels(build_risk_panel())
    pred_frames = []
    metric_frames = []
    for label, horizon in [("label_event_loss_1w", "1w"), ("label_event_loss_1m", "1m")]:
        preds, metrics = walkforward_v4(panel, label, horizon)
        pred_frames.append(preds)
        metric_frames.append(metrics)
    predictions = pd.concat(pred_frames, ignore_index=True)
    metrics = pd.concat(metric_frames, ignore_index=True)
    current = current_v4_state(panel)
    comparison = compare_with_v3(metrics)

    panel.to_csv(TABLES / "risk_off_v4_event_label_panel.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(TABLES / "risk_off_v4_walkforward_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(TABLES / "risk_off_v4_walkforward_metrics.csv", index=False, encoding="utf-8-sig")
    current.to_csv(TABLES / "current_risk_off_v4_state.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(TABLES / "risk_off_v3_v4_comparison.csv", index=False, encoding="utf-8-sig")

    print("Risk-Off V4 comparison")
    print(comparison.to_string(index=False))
    print("\nCurrent V4 state")
    print(current.to_string(index=False))


if __name__ == "__main__":
    main()
