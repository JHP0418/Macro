from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "institutional_risk_off_v2_latest"
TABLES = OUT / "tables"

SAFE_GROUPS = {
    "Cash/short bonds",
    "FX cash",
    "Korea bonds",
    "US long bonds",
    "US IG bonds",
    "Gold",
    "Korea defensive",
}

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
    "US cyclical/sector",
    "US REIT",
    "China/HK growth",
    "China equity",
    "India/EM",
    "Japan equity",
    "US high yield",
    "Commodity/Oil",
    "Oil",
}


@dataclass(frozen=True)
class ThresholdPolicy:
    threshold: float
    recall: float
    precision: float
    false_alarm_rate: float
    alert_rate: float
    objective: float
    episode_recall_20d: float = np.nan


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, **kwargs)


def finite_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def mean_existing(df: pd.DataFrame, cols: list[str], weights: list[float] | None = None) -> pd.Series:
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return pd.Series(np.nan, index=df.index)
    x = df[existing].apply(finite_numeric)
    if weights is None:
        return x.mean(axis=1, skipna=True)
    w = pd.Series({c: weights[cols.index(c)] for c in existing}, dtype=float)
    return x.mul(w, axis=1).sum(axis=1, skipna=True) / x.notna().mul(w, axis=1).sum(axis=1).replace(0, np.nan)


def add_risk_axes(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["axis1_vol_credit_stress"] = mean_existing(
        out,
        [
            "volatility_score",
            "credit_score",
            "VIX_shock_score",
            "VXN_shock_score",
            "MOVE_shock_score",
            "HY_OAS_shock_score",
            "IG_OAS_shock_score",
            "HYG_IEF_shock_score",
        ],
        [1.25, 1.25, 1.10, 0.75, 0.85, 1.20, 0.70, 0.85],
    ).clip(0, 100)
    out["axis2_fx_liquidity_stress"] = mean_existing(
        out,
        [
            "fx_score",
            "liquidity_score",
            "DXY_shock_score",
            "USDKRW_shock_score",
            "USDCNH_shock_score",
            "USDJPY_shock_score",
            "NFCI_shock_score",
            "ANFCI_shock_score",
        ],
        [1.05, 1.10, 0.80, 1.25, 0.95, 0.65, 0.90, 0.90],
    ).clip(0, 100)
    out["axis3_peak_fragility_stress"] = mean_existing(
        out,
        [
            "peak_fragility_score_0_100",
            "correction_pressure_score_0_100",
            "analog_risk_score_0_100",
            "RAI_overheat_score",
            "RAI_collapse_score",
            "ETF_breadth_shock_score",
            "breadth_score",
            "safe_rotation_score",
        ],
        [1.25, 1.20, 1.05, 0.80, 0.75, 1.05, 0.85, 0.65],
    ).clip(0, 100)
    out["risk_3d_max_axis"] = out[["axis1_vol_credit_stress", "axis2_fx_liquidity_stress", "axis3_peak_fragility_stress"]].max(axis=1)
    out["risk_3d_mean_axis"] = out[["axis1_vol_credit_stress", "axis2_fx_liquidity_stress", "axis3_peak_fragility_stress"]].mean(axis=1)
    out["risk_3d_breadth"] = out[["axis1_vol_credit_stress", "axis2_fx_liquidity_stress", "axis3_peak_fragility_stress"]].ge(35).sum(axis=1)
    out["risk_3d_vector_score"] = (
        np.sqrt(
            out["axis1_vol_credit_stress"].fillna(0).pow(2)
            + out["axis2_fx_liquidity_stress"].fillna(0).pow(2)
            + out["axis3_peak_fragility_stress"].fillna(0).pow(2)
        )
        / math.sqrt(3)
    ).clip(0, 100)
    labels = np.select(
        [
            (out["axis1_vol_credit_stress"] >= out[["axis2_fx_liquidity_stress", "axis3_peak_fragility_stress"]].max(axis=1)),
            (out["axis2_fx_liquidity_stress"] >= out[["axis1_vol_credit_stress", "axis3_peak_fragility_stress"]].max(axis=1)),
        ],
        ["변동성/신용 스트레스", "달러/유동성 스트레스"],
        default="고점취약성/과열 피로도",
    )
    out["risk_3d_dominant_axis"] = labels
    return out


def build_risk_panel() -> pd.DataFrame:
    base = read_csv(OUT.parent / "analog_macro_risk_model_latest" / "tables" / "analog_base_panel.csv", parse_dates=["Date"])
    base = add_risk_axes(base)
    base = base.rename(columns={"Date": "date"}).sort_values("date")
    macro_ssl_path = OUT.parent / "ssl_market_embeddings_latest" / "macro_ssl_embeddings.csv"
    if macro_ssl_path.exists():
        macro_ssl = read_csv(macro_ssl_path, parse_dates=["date"]).drop(columns=["entity"], errors="ignore")
        keep = ["date", *[c for c in macro_ssl.columns if c.startswith("ssl_emb_")][:16], "ssl_vq_state", "ssl_vq_distance", "ssl_flow_nll", "ssl_flow_confidence"]
        macro_ssl = macro_ssl[[c for c in keep if c in macro_ssl.columns]].rename(columns={c: f"macro_ssl_{c}" for c in keep if c != "date"})
        base = pd.merge_asof(base.sort_values("date"), macro_ssl.sort_values("date"), on="date", direction="backward")
    for c in ["NASDAQ100", "SP500", "SOX"]:
        if c in base.columns:
            px = finite_numeric(base[c])
            ret1 = px.pct_change()
            vol20 = ret1.rolling(20, min_periods=10).std() * math.sqrt(252)
            vol60 = ret1.rolling(60, min_periods=20).std() * math.sqrt(252)
            high252 = px.rolling(252, min_periods=60).max()
            ma200 = px.rolling(200, min_periods=60).mean()
            base[f"{c}_ret_20d"] = px / px.shift(20) - 1.0
            base[f"{c}_ret_60d"] = px / px.shift(60) - 1.0
            base[f"{c}_ret_120d"] = px / px.shift(120) - 1.0
            base[f"{c}_dist_252h"] = px / high252 - 1.0
            base[f"{c}_dist_ma200"] = px / ma200 - 1.0
            base[f"{c}_vol20"] = vol20
            base[f"{c}_vol60"] = vol60
            base[f"{c}_ret_fwd_5d"] = px.shift(-5) / px - 1.0
            base[f"{c}_ret_fwd_20d"] = px.shift(-20) / px - 1.0
            base[f"{c}_min_fwd_20d"] = pd.concat([(px.shift(-i) / px - 1.0).rename(i) for i in range(1, 21)], axis=1).min(axis=1)
    proxy_cols_5d = [c for c in ["NASDAQ100_ret_fwd_5d", "SP500_ret_fwd_5d", "SOX_ret_fwd_5d"] if c in base.columns]
    proxy_cols_20d = [c for c in ["NASDAQ100_ret_fwd_20d", "SP500_ret_fwd_20d", "SOX_ret_fwd_20d"] if c in base.columns]
    min_cols_20d = [c for c in ["NASDAQ100_min_fwd_20d", "SP500_min_fwd_20d", "SOX_min_fwd_20d"] if c in base.columns]
    base["risk_proxy_fwd_5d"] = base[proxy_cols_5d].mean(axis=1, skipna=True)
    base["risk_proxy_fwd_20d"] = base[proxy_cols_20d].mean(axis=1, skipna=True)
    base["risk_proxy_min_fwd_20d"] = base[min_cols_20d].mean(axis=1, skipna=True)
    base["label_large_loss_1w"] = base["risk_proxy_fwd_5d"].le(-0.03).astype(int)
    base["label_large_loss_1m"] = (base["risk_proxy_fwd_20d"].le(-0.07) | base["risk_proxy_min_fwd_20d"].le(-0.09)).astype(int)
    # This is the missing "high-point fragility" channel: crashes often begin
    # when realized risk is still suppressed, prices are near highs, and
    # momentum is extended. It intentionally complements the stress axes.
    mom = mean_existing(base, ["NASDAQ100_ret_60d", "SP500_ret_60d", "SOX_ret_60d"])
    near_high = (100.0 + 250.0 * mean_existing(base, ["NASDAQ100_dist_252h", "SP500_dist_252h", "SOX_dist_252h"])).clip(0, 100)
    above_ma = (50.0 + 180.0 * mean_existing(base, ["NASDAQ100_dist_ma200", "SP500_dist_ma200", "SOX_dist_ma200"])).clip(0, 100)
    low_vol = (35.0 - 100.0 * mean_existing(base, ["NASDAQ100_vol20", "SP500_vol20", "SOX_vol20"])).clip(0, 100)
    low_stress = (35.0 - mean_existing(base, ["axis1_vol_credit_stress", "axis2_fx_liquidity_stress"])).clip(0, 100)
    base["complacent_peak_fragility_stress"] = (
        0.30 * (mom.clip(lower=0) * 350.0).clip(0, 100)
        + 0.25 * near_high
        + 0.20 * above_ma
        + 0.15 * low_vol
        + 0.10 * low_stress
    ).clip(0, 100)
    base["risk_3d_vector_score"] = (
        0.72 * base["risk_3d_vector_score"].fillna(0)
        + 0.28 * base["complacent_peak_fragility_stress"].fillna(0)
    ).clip(0, 100)
    stress_rule_raw = pd.concat(
        [
            base.get("NASDAQ100_vol20", pd.Series(index=base.index, dtype=float)) * 100.0,
            base.get("SOX_vol20", pd.Series(index=base.index, dtype=float)) * 80.0,
            base.get("SP500_vol20", pd.Series(index=base.index, dtype=float)) * 110.0,
            base.get("VIX", pd.Series(index=base.index, dtype=float)) / 2.0,
            base.get("risk_off_score", pd.Series(index=base.index, dtype=float)),
            base.get("axis1_vol_credit_stress", pd.Series(index=base.index, dtype=float)),
        ],
        axis=1,
    ).max(axis=1)
    base["stress_rule_raw"] = stress_rule_raw
    base["stress_rule_percentile_3y"] = rolling_percentile(stress_rule_raw, 756)
    base = base.replace([np.inf, -np.inf], np.nan)
    usable = base["NASDAQ100"].pipe(finite_numeric).gt(0) & base["date"].ge(pd.Timestamp("1998-01-01"))
    return base.loc[usable].reset_index(drop=True)


def rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    out = np.full(len(values), np.nan, dtype=float)
    arr = values.to_numpy(dtype=float)
    for i, value in enumerate(arr):
        start = max(0, i - window)
        hist = arr[start:i]
        hist = hist[np.isfinite(hist)]
        if hist.size >= max(100, window // 4) and np.isfinite(value):
            out[i] = float(np.mean(hist <= value))
    return pd.Series(out, index=series.index)


def risk_features(panel: pd.DataFrame) -> list[str]:
    preferred = [
        "risk_off_score",
        "risk_off_momentum_5d",
        "axis1_vol_credit_stress",
        "axis2_fx_liquidity_stress",
        "axis3_peak_fragility_stress",
        "risk_3d_max_axis",
        "risk_3d_mean_axis",
        "risk_3d_breadth",
        "risk_3d_vector_score",
        "complacent_peak_fragility_stress",
        "stress_rule_raw",
        "stress_rule_percentile_3y",
        "volatility_score",
        "credit_score",
        "fx_score",
        "liquidity_score",
        "equity_score",
        "cyclical_score",
        "rai_score",
        "breadth_score",
        "safe_rotation_score",
        "RAI_z",
        "RAI_20d_change",
        "RAI_level_0_100",
        "ETF_below_60ma_pct",
        "ETF_20d_large_loss_pct",
        "peak_fragility_score_0_100",
        "analog_risk_score_0_100",
        "correction_pressure_score_0_100",
        "analog_down_prob_1w_model",
        "analog_down_prob_1m_model",
        "analog_tail_prob_1m_model",
        "US2Y",
        "US10Y_driver",
        "US10Y_REAL",
        "US10Y_2Y",
        "HY_OAS",
        "IG_OAS",
        "DXY_driver",
        "USDKRW_driver",
        "USDJPY_driver",
        "VIX",
        "MOVE",
        "HYG_IEF",
        "GOLD_driver",
        "WTI_driver",
        "COPPER_GOLD",
        "NASDAQ100_ret_20d",
        "NASDAQ100_ret_60d",
        "NASDAQ100_ret_120d",
        "NASDAQ100_dist_252h",
        "NASDAQ100_dist_ma200",
        "NASDAQ100_vol20",
        "SP500_ret_20d",
        "SP500_ret_60d",
        "SP500_dist_252h",
        "SP500_dist_ma200",
        "SP500_vol20",
        "SOX_ret_20d",
        "SOX_ret_60d",
        "SOX_dist_252h",
        "SOX_dist_ma200",
        "SOX_vol20",
    ]
    ssl_cols = [c for c in panel.columns if c.startswith("macro_ssl_")]
    return [c for c in preferred + ssl_cols if c in panel.columns and pd.api.types.is_numeric_dtype(panel[c])]


def make_model() -> object:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        HistGradientBoostingClassifier(
            max_iter=260,
            learning_rate=0.035,
            max_leaf_nodes=15,
            min_samples_leaf=55,
            l2_regularization=1.2,
            random_state=42,
        ),
    )


def choose_policy(y: pd.Series, p: pd.Series, min_recall: float = 0.80, max_alert_rate: float = 0.35) -> ThresholdPolicy:
    yb = y.astype(bool).to_numpy()
    pp = pd.Series(p).astype(float).to_numpy()
    best: ThresholdPolicy | None = None
    fallback: ThresholdPolicy | None = None
    constrained_fallback: ThresholdPolicy | None = None
    for th in np.linspace(0.03, 0.95, 185):
        pred = pp >= th
        tp = int(np.sum(pred & yb))
        fp = int(np.sum(pred & ~yb))
        fn = int(np.sum(~pred & yb))
        tn = int(np.sum(~pred & ~yb))
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        false_alarm_rate = fp / max(fp + tn, 1)
        alert_rate = float(np.mean(pred))
        # Missed drawdowns are intentionally expensive. False alarms are allowed,
        # but the quadratic alert-rate term prevents an always-on sentinel.
        objective = 4.2 * recall + 1.5 * precision - 1.4 * false_alarm_rate - 4.0 * max(alert_rate - max_alert_rate, 0) - 1.8 * max(min_recall - recall, 0)
        cand = ThresholdPolicy(float(th), float(recall), float(precision), float(false_alarm_rate), float(alert_rate), float(objective))
        if fallback is None or cand.objective > fallback.objective:
            fallback = cand
        if alert_rate <= max_alert_rate and (constrained_fallback is None or cand.objective > constrained_fallback.objective):
            constrained_fallback = cand
        if recall >= min_recall and alert_rate <= max_alert_rate:
            if best is None or cand.objective > best.objective:
                best = cand
    return best or constrained_fallback or fallback or ThresholdPolicy(0.5, 0.0, 0.0, 0.0, 0.0, -999.0)


def episode_starts(dates: pd.Series, y: pd.Series, min_gap_days: int = 30) -> list[pd.Timestamp]:
    frame = pd.DataFrame({"date": pd.to_datetime(dates), "y": y.astype(int)}).dropna().sort_values("date")
    starts: list[pd.Timestamp] = []
    previous: pd.Timestamp | None = None
    in_episode = False
    for _, row in frame.iterrows():
        date = pd.Timestamp(row["date"])
        is_pos = bool(row["y"])
        if is_pos and not in_episode:
            if previous is None or (date - previous).days >= min_gap_days:
                starts.append(date)
                previous = date
            in_episode = True
        elif not is_pos:
            in_episode = False
    return starts


def episode_recall(dates: pd.Series, y: pd.Series, pred: pd.Series, lookback_calendar_days: int = 35) -> float:
    frame = pd.DataFrame({"date": pd.to_datetime(dates), "y": y.astype(int), "pred": pred.astype(bool)}).dropna().sort_values("date")
    starts = episode_starts(frame["date"], frame["y"])
    if not starts:
        return np.nan
    hits = 0
    for start in starts:
        window = frame[(frame["date"] >= start - pd.Timedelta(days=lookback_calendar_days)) & (frame["date"] <= start)]
        if bool(window["pred"].any()):
            hits += 1
    return hits / len(starts)


def fast_episode_recall(date_values: np.ndarray, y_bool: np.ndarray, pred_bool: np.ndarray, lookback_days: int = 35) -> float:
    starts: list[int] = []
    in_episode = False
    previous_day: float | None = None
    day_values = date_values.astype("datetime64[D]").astype(float)
    for i, is_pos in enumerate(y_bool):
        if is_pos and not in_episode:
            if previous_day is None or day_values[i] - previous_day >= 30:
                starts.append(i)
                previous_day = day_values[i]
            in_episode = True
        elif not is_pos:
            in_episode = False
    if not starts:
        return np.nan
    hits = 0
    for i in starts:
        start_day = day_values[i] - lookback_days
        j = int(np.searchsorted(day_values, start_day, side="left"))
        if pred_bool[j : i + 1].any():
            hits += 1
    return hits / len(starts)


def choose_episode_policy(dates: pd.Series, y: pd.Series, p: pd.Series, max_alert_rate: float = 0.35) -> ThresholdPolicy:
    yb = y.astype(bool).to_numpy()
    pp = pd.Series(p).astype(float).to_numpy()
    date_values = pd.to_datetime(dates).to_numpy()
    best: ThresholdPolicy | None = None
    for th in np.linspace(0.08, 0.90, 67):
        pred = pp >= th
        tp = int(np.sum(pred & yb))
        fp = int(np.sum(pred & ~yb))
        fn = int(np.sum(~pred & yb))
        tn = int(np.sum(~pred & ~yb))
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        false_alarm_rate = fp / max(fp + tn, 1)
        alert_rate = float(np.mean(pred))
        ep_recall = fast_episode_recall(date_values, yb, pred)
        if np.isnan(ep_recall):
            ep_recall = recall
        objective = 5.0 * ep_recall + 1.0 * recall + 1.0 * precision - 1.2 * false_alarm_rate - 4.0 * max(alert_rate - max_alert_rate, 0)
        cand = ThresholdPolicy(float(th), float(recall), float(precision), float(false_alarm_rate), float(alert_rate), float(objective), float(ep_recall))
        if best is None or cand.objective > best.objective:
            best = cand
    return best or choose_policy(y, p, max_alert_rate=max_alert_rate)


def confusion(y: pd.Series, p: pd.Series, th: float, dates: pd.Series | None = None) -> dict[str, float]:
    actual = y.astype(bool)
    pred = p.astype(float).ge(th)
    tp = int((pred & actual).sum())
    fp = int((pred & ~actual).sum())
    fn = int((~pred & actual).sum())
    tn = int((~pred & ~actual).sum())
    out = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "recall": tp / max(tp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "false_alarm_rate": fp / max(fp + tn, 1),
        "alert_rate": (tp + fp) / max(tp + fp + fn + tn, 1),
        "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
    }
    if dates is not None:
        out["episode_recall_20d"] = episode_recall(dates, y, pred)
    return out


def safe_auc(y: pd.Series, p: pd.Series) -> float:
    return float(roc_auc_score(y, p)) if y.nunique() > 1 else np.nan


def safe_ap(y: pd.Series, p: pd.Series) -> float:
    return float(average_precision_score(y, p)) if y.nunique() > 1 else np.nan


def walkforward_risk_model(panel: pd.DataFrame, label: str, horizon: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = risk_features(panel)
    rows = []
    pred_rows = []
    for test_year in range(2003, 2027):
        test_start = pd.Timestamp(f"{test_year}-01-01")
        test_end = pd.Timestamp(f"{test_year}-12-31")
        valid_start = test_start - pd.DateOffset(years=2)
        embargo_start = test_start - pd.Timedelta(days=30)
        train = panel[panel["date"].lt(valid_start)]
        valid = panel[(panel["date"].ge(valid_start)) & (panel["date"].lt(embargo_start))]
        test = panel[(panel["date"].ge(test_start)) & (panel["date"].le(test_end))]
        if train[label].sum() < 20 or valid[label].sum() < 3 or test.empty:
            continue
        model = make_model()
        model.fit(train[features], train[label].astype(int))
        # Threshold optimization is done on raw probabilities. Isotonic
        # calibration is useful for reporting, but on rare crash labels it can
        # collapse many observations to the same tiny value and force an
        # always-on threshold.
        valid_model_prob = pd.Series(model.predict_proba(valid[features])[:, 1], index=valid.index)
        test_model_prob = pd.Series(model.predict_proba(test[features])[:, 1], index=test.index)
        valid_rule = finite_numeric(valid.get("stress_rule_percentile_3y", pd.Series(index=valid.index))).fillna(0.0).clip(0, 1)
        test_rule = finite_numeric(test.get("stress_rule_percentile_3y", pd.Series(index=test.index))).fillna(0.0).clip(0, 1)
        valid_prob = (0.45 * valid_model_prob + 0.55 * valid_rule).clip(0, 1)
        test_prob = (0.45 * test_model_prob + 0.55 * test_rule).clip(0, 1)
        policy = choose_episode_policy(valid["date"], valid[label], valid_prob, max_alert_rate=0.42 if horizon == "1m" else 0.34)
        cm = confusion(test[label], test_prob, policy.threshold, test["date"])
        rows.append(
            {
                "horizon": horizon,
                "test_year": test_year,
                "threshold": policy.threshold,
                "valid_recall": policy.recall,
                "valid_precision": policy.precision,
                "valid_false_alarm_rate": policy.false_alarm_rate,
                "valid_alert_rate": policy.alert_rate,
                "valid_episode_recall_20d": policy.episode_recall_20d,
                "test_auc": safe_auc(test[label], test_prob),
                "test_ap": safe_ap(test[label], test_prob),
                "test_brier": float(brier_score_loss(test[label], test_prob)) if test[label].nunique() > 1 else np.nan,
                **{f"test_{k}": v for k, v in cm.items()},
                "test_positive_rate": float(test[label].mean()),
                "n_train": int(train.shape[0]),
                "n_valid": int(valid.shape[0]),
                "n_test": int(test.shape[0]),
            }
        )
        out = test[["date", label, "risk_off_score", "risk_3d_dominant_axis", "axis1_vol_credit_stress", "axis2_fx_liquidity_stress", "axis3_peak_fragility_stress", "complacent_peak_fragility_stress", "stress_rule_percentile_3y", "risk_proxy_fwd_5d", "risk_proxy_fwd_20d", "risk_proxy_min_fwd_20d"]].copy()
        out["horizon"] = horizon
        out["risk_off_v2_prob"] = test_prob.to_numpy()
        out["risk_off_v2_threshold"] = policy.threshold
        out["risk_off_v2_alert"] = out["risk_off_v2_prob"].ge(policy.threshold).astype(int)
        pred_rows.append(out)
    return pd.concat(pred_rows, ignore_index=True), pd.DataFrame(rows)


def train_current_risk(panel: pd.DataFrame, label: str, horizon: str) -> pd.DataFrame:
    features = risk_features(panel)
    train = panel[panel[label].notna()].copy()
    valid = train[train["date"].ge(train["date"].max() - pd.DateOffset(years=2))]
    fit = train[train["date"].lt(valid["date"].min())]
    if fit.empty:
        fit = train
    model = make_model()
    model.fit(fit[features], fit[label].astype(int))
    model_prob = pd.Series(model.predict_proba(train[features])[:, 1], index=train.index)
    rule_prob = finite_numeric(train.get("stress_rule_percentile_3y", pd.Series(index=train.index))).fillna(0.0).clip(0, 1)
    prob = (0.45 * model_prob + 0.55 * rule_prob).clip(0, 1)
    policy = choose_policy(valid[label], pd.Series(prob.loc[valid.index], index=valid.index), min_recall=0.80, max_alert_rate=0.38 if horizon == "1m" else 0.32)
    latest = train.iloc[-1:].copy()
    latest_prob = float(prob.loc[latest.index[0]])
    return pd.DataFrame(
        [
            {
                "horizon": horizon,
                "date": latest["date"].iloc[0],
                "prob": latest_prob,
                "threshold": policy.threshold,
                "alert": int(latest_prob >= policy.threshold),
                "risk_off_score": float(latest["risk_off_score"].iloc[0]),
                "axis1_vol_credit_stress": float(latest["axis1_vol_credit_stress"].iloc[0]),
                "axis2_fx_liquidity_stress": float(latest["axis2_fx_liquidity_stress"].iloc[0]),
                "axis3_peak_fragility_stress": float(latest["axis3_peak_fragility_stress"].iloc[0]),
                "dominant_axis": str(latest["risk_3d_dominant_axis"].iloc[0]),
            }
        ]
    )


def macro_safe_features(macro: pd.DataFrame) -> pd.DataFrame:
    out = macro[["date"]].copy()
    candidates = [
        "US10Y_driver",
        "US2Y",
        "US10Y_REAL",
        "DXY_driver",
        "USDKRW_driver",
        "VIX",
        "HY_OAS",
        "GOLD_driver",
        "HYG_IEF",
        "axis1_vol_credit_stress",
        "axis2_fx_liquidity_stress",
        "axis3_peak_fragility_stress",
        "risk_3d_vector_score",
        "risk_off_score",
    ]
    for c in candidates:
        if c in macro.columns:
            s = finite_numeric(macro[c])
            out[f"macro_{c}"] = s
            out[f"macro_{c}_chg_5d"] = s.diff(5)
            out[f"macro_{c}_chg_20d"] = s.diff(20)
            if c in {"DXY_driver", "USDKRW_driver", "GOLD_driver", "HYG_IEF"}:
                out[f"macro_{c}_ret_20d"] = s / s.shift(20) - 1.0
    if {"US10Y_driver", "US2Y"}.issubset(macro.columns):
        out["macro_us10y_minus_2y"] = finite_numeric(macro["US10Y_driver"]) - finite_numeric(macro["US2Y"])
    if {"GOLD_driver", "DXY_driver"}.issubset(macro.columns):
        out["macro_gold_vs_dollar_20d"] = finite_numeric(macro["GOLD_driver"]) / finite_numeric(macro["GOLD_driver"]).shift(20) - finite_numeric(macro["DXY_driver"]) / finite_numeric(macro["DXY_driver"]).shift(20)
    for c in [c for c in macro.columns if c.startswith("macro_ssl_")]:
        out[f"macro_{c}"] = finite_numeric(macro[c])
    return out.replace([np.inf, -np.inf], np.nan)


def build_safe_panel(risk_panel: pd.DataFrame) -> pd.DataFrame:
    weekly = read_csv(OUT.parent / "weekly_screening_rank_backtest_latest" / "tables" / "weekly_calibrated_rank_panel.csv", parse_dates=["date"])
    macro = macro_safe_features(risk_panel)
    panel = pd.merge_asof(weekly.sort_values("date"), macro.sort_values("date"), on="date", direction="backward")
    safe_ssl_path = OUT.parent / "ssl_safe_asset_embeddings_latest" / "safe_ssl_embeddings.csv"
    if safe_ssl_path.exists():
        safe_ssl = read_csv(safe_ssl_path, parse_dates=["date"]).rename(columns={"entity": "symbol"})
        keep = ["date", "symbol", *[c for c in safe_ssl.columns if c.startswith("ssl_emb_")][:12], "ssl_vq_state", "ssl_vq_distance", "ssl_flow_nll", "ssl_flow_confidence"]
        safe_ssl = safe_ssl[[c for c in keep if c in safe_ssl.columns]].rename(columns={c: f"safe_ssl_{c}" for c in keep if c not in {"date", "symbol"}})
        panel = panel.merge(safe_ssl, on=["date", "symbol"], how="left")
    panel["is_safe_asset"] = panel["group"].isin(SAFE_GROUPS)
    panel["is_risk_asset"] = panel["group"].isin(RISK_GROUPS)
    risk = panel[panel["is_risk_asset"]].groupby("date", as_index=False).agg(
        risk_avg_1w=("realized_return_1w", "mean"),
        risk_avg_1m=("realized_return_4w", "mean"),
    )
    panel = panel.merge(risk, on="date", how="left")
    panel = panel[panel["is_safe_asset"]].copy()
    panel["safe_target_1w"] = panel["realized_return_1w"] - 0.75 * panel["realized_return_1w"].clip(upper=0).abs() - panel["risk_avg_1w"].fillna(0)
    panel["safe_target_1m"] = panel["realized_return_4w"] - 0.75 * panel["realized_return_4w"].clip(upper=0).abs() - panel["risk_avg_1m"].fillna(0)
    group_dummies = pd.get_dummies(panel["group"], prefix="group", dtype=float)
    basket_dummies = pd.get_dummies(panel["basket"], prefix="basket", dtype=float)
    panel = pd.concat([panel, group_dummies, basket_dummies], axis=1)
    return panel.replace([np.inf, -np.inf], np.nan)


def rank_label(frame: pd.DataFrame, target: str) -> pd.Series:
    pct = frame.groupby("date")[target].rank(pct=True, method="average")
    return np.ceil(pct * 5.0).sub(1).clip(0, 4).where(pct.notna())


def safe_rank_features(panel: pd.DataFrame) -> list[str]:
    base = [
        "score_0_100",
        "upside_prob_1w",
        "upside_prob_4w",
        "technical_score",
        "driver_fit_score",
        "beta_fit_score",
        "calibrated_prob_1w",
        "calibrated_prob_4w",
        "institutional_score_0_100",
    ]
    macro = [c for c in panel.columns if c.startswith("macro_")]
    ssl = [c for c in panel.columns if c.startswith("safe_ssl_")]
    dummies = [c for c in panel.columns if c.startswith("group_") or c.startswith("basket_")]
    return [c for c in base + macro + ssl + dummies if c in panel.columns]


def make_rank_group(data: pd.DataFrame, features: list[str], label: str) -> tuple[pd.DataFrame, pd.Series, list[int], pd.DataFrame]:
    data = data.dropna(subset=[label]).sort_values(["date", "symbol"]).copy()
    data = data[data.groupby("date")["symbol"].transform("size").ge(2)].copy()
    x = data[features].apply(pd.to_numeric, errors="coerce")
    x = x.groupby(data["date"]).transform(lambda s: s.fillna(s.median())).fillna(0.0)
    y = data[label].astype(int)
    group = data.groupby("date", sort=False).size().astype(int).tolist()
    return x, y, group, data


def train_safe_ranker_once(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, features: list[str], label: str) -> pd.DataFrame:
    x_train, y_train, g_train, _ = make_rank_group(train, features, label)
    x_valid, y_valid, g_valid, _ = make_rank_group(valid, features, label)
    if x_train.empty or x_valid.empty:
        return pd.DataFrame()
    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=260,
        learning_rate=0.035,
        num_leaves=11,
        max_depth=3,
        min_child_samples=8,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=1.0,
        reg_lambda=6.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(x_train, y_train, group=g_train, eval_set=[(x_valid, y_valid)], eval_group=[g_valid], eval_at=[1, 3])
    x_test, _, _, test_data = make_rank_group(test, features, label)
    if x_test.empty:
        return pd.DataFrame()
    out = test_data.copy()
    out["safe_v2_ranker_score"] = model.predict(x_test)
    return out


def train_current_safe_ranker(panel: pd.DataFrame, features: list[str], label: str, horizon: str) -> pd.DataFrame:
    labeled = panel.dropna(subset=[label]).copy()
    latest_date = panel["date"].max()
    current = panel[panel["date"].eq(latest_date)].sort_values(["date", "symbol"]).copy()
    if labeled.empty or current.empty:
        return pd.DataFrame()
    train = labeled[labeled["date"].lt(latest_date)].copy()
    x_train, y_train, g_train, _ = make_rank_group(train, features, label)
    if x_train.empty:
        return pd.DataFrame()
    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=220,
        learning_rate=0.035,
        num_leaves=11,
        max_depth=3,
        min_child_samples=8,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=1.0,
        reg_lambda=6.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(x_train, y_train, group=g_train)
    x_current = current[features].apply(pd.to_numeric, errors="coerce")
    x_current = x_current.fillna(x_train.median()).fillna(0.0)
    out = current.copy()
    out["horizon"] = horizon
    out["safe_v2_ranker_score"] = model.predict(x_current)
    return out


def walkforward_safe_ranker(panel: pd.DataFrame, risk_preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = panel.copy()
    panel["safe_label_1w"] = rank_label(panel, "safe_target_1w")
    panel["safe_label_1m"] = rank_label(panel, "safe_target_1m")
    features = safe_rank_features(panel)
    risk_alerts = risk_preds[risk_preds["horizon"].eq("1m")][["date", "risk_off_v2_prob", "risk_off_v2_alert"]]
    rows = []
    preds = []
    for horizon, label, target, ret_col in [
        ("1w", "safe_label_1w", "safe_target_1w", "realized_return_1w"),
        ("1m", "safe_label_1m", "safe_target_1m", "realized_return_4w"),
    ]:
        for year in range(2025, 2027):
            test_start = pd.Timestamp(f"{year}-01-01")
            test_end = pd.Timestamp(f"{year}-12-31")
            valid_start = test_start - pd.DateOffset(months=6)
            train = panel[panel["date"].lt(valid_start)]
            valid = panel[(panel["date"].ge(valid_start)) & (panel["date"].lt(test_start))]
            test = panel[(panel["date"].ge(test_start)) & (panel["date"].le(test_end))]
            if train.empty or valid.empty or test.empty:
                continue
            pred = train_safe_ranker_once(train, valid, test, features, label)
            if pred.empty:
                continue
            pred["horizon"] = horizon
            pred["test_year"] = year
            preds.append(pred)
    if not preds:
        return pd.DataFrame(), pd.DataFrame()
    all_preds = pd.concat(preds, ignore_index=True)
    risk_alerts = risk_alerts.sort_values("date")
    all_preds = pd.merge_asof(all_preds.sort_values("date"), risk_alerts, on="date", direction="backward")
    for (horizon, year, date), part in all_preds.groupby(["horizon", "test_year", "date"]):
        risk_prob = float(part["risk_off_v2_prob"].iloc[0]) if "risk_off_v2_prob" in part else np.nan
        risk_alert = int(part["risk_off_v2_alert"].iloc[0]) if "risk_off_v2_alert" in part and pd.notna(part["risk_off_v2_alert"].iloc[0]) else 0
        if risk_prob < 0.20 and not risk_alert:
            continue
        target = "safe_target_1w" if horizon == "1w" else "safe_target_1m"
        ret_col = "realized_return_1w" if horizon == "1w" else "realized_return_4w"
        top_k = min(3, len(part))
        picks = part.nlargest(top_k, "safe_v2_ranker_score")
        actual = part.nlargest(top_k, target)
        rows.append(
            {
                "horizon": horizon,
                "test_year": year,
                "date": date,
                "risk_off_v2_prob": risk_prob,
                "risk_off_v2_alert": risk_alert,
                "picked_return": picks[ret_col].mean(),
                "picked_target": picks[target].mean(),
                "safe_avg_target": part[target].mean(),
                "actual_top_target": actual[target].mean(),
                "beat_safe_average": picks[target].mean() > part[target].mean(),
                "topk_overlap": len(set(picks["symbol"]) & set(actual["symbol"])) / max(top_k, 1),
                "selected": ",".join(picks["symbol"].astype(str)),
                "selected_names": " | ".join(picks["name"].astype(str)),
            }
        )
    eval_df = pd.DataFrame(rows)
    if eval_df.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            eval_df.groupby(["horizon", "test_year"], as_index=False)
            .agg(
                periods=("date", "count"),
                avg_picked_return=("picked_return", "mean"),
                avg_picked_target=("picked_target", "mean"),
                avg_safe_target=("safe_avg_target", "mean"),
                beat_safe_average_rate=("beat_safe_average", "mean"),
                topk_overlap_rate=("topk_overlap", "mean"),
            )
            .sort_values(["horizon", "test_year"])
        )
        overall = (
            eval_df.groupby("horizon", as_index=False)
            .agg(
                periods=("date", "count"),
                avg_picked_return=("picked_return", "mean"),
                avg_picked_target=("picked_target", "mean"),
                avg_safe_target=("safe_avg_target", "mean"),
                beat_safe_average_rate=("beat_safe_average", "mean"),
                topk_overlap_rate=("topk_overlap", "mean"),
            )
            .assign(test_year="ALL")
        )
        summary = pd.concat([summary, overall], ignore_index=True)
    return all_preds, summary


def current_safe_recommendations(panel: pd.DataFrame, risk_current: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel["safe_label_1w"] = rank_label(panel, "safe_target_1w")
    panel["safe_label_1m"] = rank_label(panel, "safe_target_1m")
    features = safe_rank_features(panel)
    preds = pd.concat(
        [
            train_current_safe_ranker(panel, features, "safe_label_1w", "1w"),
            train_current_safe_ranker(panel, features, "safe_label_1m", "1m"),
        ],
        ignore_index=True,
    )
    if preds.empty:
        return pd.DataFrame()
    risk_map = risk_current.set_index("horizon")["prob"].to_dict() if not risk_current.empty else {}
    alert_map = risk_current.set_index("horizon")["alert"].to_dict() if not risk_current.empty else {}
    preds["risk_off_v2_prob"] = preds["horizon"].map(risk_map)
    preds["risk_off_v2_alert"] = preds["horizon"].map(alert_map)
    latest = preds[preds["date"].eq(preds["date"].max())].copy()
    cols = [
        "date",
        "horizon",
        "symbol",
        "name",
        "group",
        "basket",
        "safe_v2_ranker_score",
        "risk_off_v2_prob",
        "risk_off_v2_alert",
        "technical_score",
        "driver_fit_score",
        "beta_fit_score",
        "realized_return_1w",
        "realized_return_4w",
    ]
    out = latest.sort_values(["horizon", "safe_v2_ranker_score"], ascending=[True, False])
    out["rank"] = out.groupby("horizon")["safe_v2_ranker_score"].rank(ascending=False, method="first")
    return out[[c for c in ["rank", *cols] if c in out.columns]]


def write_comparison_summary(risk_metrics: pd.DataFrame, safe_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    v4_path = OUT.parent / "etf_leadership_static_v4_repaired" / "v3_backtest_summary.csv"
    v5_path = OUT.parent / "etf_leadership_static_v5_ssl" / "v5_ssl_backtest_summary.csv"
    if v4_path.exists():
        v4 = read_csv(v4_path)
        best = v4[v4["label"].eq("ranker_score_1M_top2")].head(1)
        if not best.empty:
            r = best.iloc[0]
            rows.append(
                {
                    "model": "ETF Leadership V4",
                    "scope": "1M Top2",
                    "primary_metric": "Sharpe",
                    "value": float(r.get("Sharpe", np.nan)),
                    "hit_positive": float(r.get("hit_positive", np.nan)),
                    "hit_excess": float(r.get("hit_excess", np.nan)),
                    "MDD": float(r.get("MDD", np.nan)),
                }
            )
    if v5_path.exists():
        v5 = read_csv(v5_path)
        best = v5[v5["label"].eq("ranker_score_1M_top2")].head(1)
        if not best.empty:
            r = best.iloc[0]
            rows.append(
                {
                    "model": "ETF Leadership V5 SSL/VQ",
                    "scope": "1M Top2",
                    "primary_metric": "Sharpe",
                    "value": float(r.get("Sharpe", np.nan)),
                    "hit_positive": float(r.get("hit_positive", np.nan)),
                    "hit_excess": float(r.get("hit_excess", np.nan)),
                    "MDD": float(r.get("MDD", np.nan)),
                }
            )
    for horizon, part in risk_metrics.groupby("horizon"):
        rows.append(
            {
                "model": "Risk-Off V3 3D+SSL",
                "scope": horizon,
                "primary_metric": "AUC / Episode Recall",
                "value": float(part["test_auc"].mean()),
                "recall": float(part["test_recall"].mean()),
                "episode_recall_20d": float(part.get("test_episode_recall_20d", pd.Series(dtype=float)).mean()),
                "precision": float(part["test_precision"].mean()),
                "false_alarm_rate": float(part["test_false_alarm_rate"].mean()),
                "alert_rate": float(part["test_alert_rate"].mean()),
            }
        )
    if not safe_summary.empty:
        for _, r in safe_summary[safe_summary["test_year"].astype(str).eq("ALL")].iterrows():
            rows.append(
                {
                    "model": "Safe Asset Macro SSL Ranker",
                    "scope": r["horizon"],
                    "primary_metric": "Beat safe average",
                    "value": float(r.get("beat_safe_average_rate", np.nan)),
                    "avg_picked_return": float(r.get("avg_picked_return", np.nan)),
                    "avg_picked_target": float(r.get("avg_picked_target", np.nan)),
                    "avg_safe_target": float(r.get("avg_safe_target", np.nan)),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "model_v4_v5_ssl_comparison_summary.csv", index=False, encoding="utf-8-sig")
    return out


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    risk_panel = build_risk_panel()
    pred_frames = []
    metric_frames = []
    for label, horizon in [("label_large_loss_1w", "1w"), ("label_large_loss_1m", "1m")]:
        preds, metrics = walkforward_risk_model(risk_panel, label, horizon)
        pred_frames.append(preds)
        metric_frames.append(metrics)
    risk_preds = pd.concat(pred_frames, ignore_index=True)
    risk_metrics = pd.concat(metric_frames, ignore_index=True)
    current = pd.concat(
        [
            train_current_risk(risk_panel, "label_large_loss_1w", "1w"),
            train_current_risk(risk_panel, "label_large_loss_1m", "1m"),
        ],
        ignore_index=True,
    )

    safe_panel = build_safe_panel(risk_panel)
    safe_preds, safe_summary = walkforward_safe_ranker(safe_panel, risk_preds)
    latest_safe = current_safe_recommendations(safe_panel, current)
    comparison = write_comparison_summary(risk_metrics, safe_summary)

    risk_panel.to_csv(TABLES / "risk_3d_training_panel.csv", index=False, encoding="utf-8-sig")
    risk_preds.to_csv(TABLES / "risk_off_v2_walkforward_predictions.csv", index=False, encoding="utf-8-sig")
    risk_metrics.to_csv(TABLES / "risk_off_v2_walkforward_metrics.csv", index=False, encoding="utf-8-sig")
    current.to_csv(TABLES / "current_risk_off_v2_state.csv", index=False, encoding="utf-8-sig")
    safe_panel.to_csv(TABLES / "macro_conditioned_safe_asset_panel.csv", index=False, encoding="utf-8-sig")
    safe_preds.to_csv(TABLES / "macro_conditioned_safe_asset_predictions.csv", index=False, encoding="utf-8-sig")
    safe_summary.to_csv(TABLES / "macro_conditioned_safe_asset_summary.csv", index=False, encoding="utf-8-sig")
    latest_safe.to_csv(TABLES / "current_safe_asset_recommendations_v2.csv", index=False, encoding="utf-8-sig")

    print("Risk-Off V2 summary")
    print(risk_metrics.groupby("horizon").agg(test_auc=("test_auc", "mean"), test_recall=("test_recall", "mean"), test_precision=("test_precision", "mean"), test_false_alarm_rate=("test_false_alarm_rate", "mean"), test_alert_rate=("test_alert_rate", "mean")).to_string())
    print("\nCurrent risk state")
    print(current.to_string(index=False))
    print("\nSafe asset summary")
    print(safe_summary.to_string(index=False))
    print("\nLatest safe asset recommendations")
    print(latest_safe.head(20).to_string(index=False))
    print("\nComparison summary")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
