from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker, early_stopping, log_evaluation
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_static_etf_leadership_v3 as v3  # noqa: E402


OUT = ROOT / "outputs" / "ssl2_head_backtest_latest"
TABLES = OUT / "tables"
SSL2 = ROOT / "outputs" / "institutional_tensor_ssl_v2_latest" / "tables"
PARQUET = ROOT / "outputs" / "institutional_tensor_ssl_v2_latest" / "parquet"


RISK_LABELS = ["label_large_loss_1w", "label_large_loss_1m", "label_nasdaq_down_1w", "label_nasdaq_down_1m"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward head model comparison: baseline vs SSL2 embeddings.")
    p.add_argument("--output-dir", default=str(OUT))
    p.add_argument("--risk-train-end", default="2022-12-31")
    p.add_argument("--risk-valid-end", default="2024-12-31")
    p.add_argument("--etf-train-end", default="2018-12-31")
    p.add_argument("--etf-valid-end", default="2021-12-31")
    p.add_argument("--safe-train-end", default="2024-12-31")
    p.add_argument("--safe-valid-end", default="2025-12-31")
    p.add_argument("--tasks", default="risk,etf,safe")
    return p.parse_args()


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def emb_cols(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith("ssl2_emb_")] + [
        c for c in ["ssl2_vq_state", "ssl2_vq_distance", "ssl2_nf_nll", "ssl2_nf_confidence"] if c in frame.columns
    ]


def safe_auc(y: pd.Series, p: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y, p)) if pd.Series(y).nunique() > 1 else np.nan
    except Exception:
        return np.nan


def safe_ap(y: pd.Series, p: np.ndarray) -> float:
    try:
        return float(average_precision_score(y, p)) if pd.Series(y).nunique() > 1 else np.nan
    except Exception:
        return np.nan


def classification_metrics(preds: pd.DataFrame, label: str, prob_col: str, threshold: float) -> dict:
    out: dict[str, float | int | str] = {"threshold": threshold}
    for split, part in preds.groupby("split"):
        y = part[label].astype(int)
        p = part[prob_col].astype(float)
        pred = p.ge(threshold)
        actual = y.astype(bool)
        tp = int((pred & actual).sum())
        fp = int((pred & ~actual).sum())
        fn = int((~pred & actual).sum())
        tn = int((~pred & ~actual).sum())
        out[f"{split}_samples"] = int(part.shape[0])
        out[f"{split}_positive_rate"] = float(y.mean())
        out[f"{split}_auc"] = safe_auc(y, p.to_numpy())
        out[f"{split}_ap"] = safe_ap(y, p.to_numpy())
        out[f"{split}_brier"] = float(brier_score_loss(y, p)) if y.nunique() > 1 else np.nan
        out[f"{split}_recall"] = tp / max(tp + fn, 1)
        out[f"{split}_precision"] = tp / max(tp + fp, 1)
        out[f"{split}_false_alarm"] = fp / max(tp + fp, 1)
        out[f"{split}_accuracy"] = (tp + tn) / max(tp + fp + fn + tn, 1)
    return out


def choose_risk_threshold(valid: pd.DataFrame, label: str, prob_col: str) -> float:
    best = (0.5, -1e9)
    for th in np.linspace(0.10, 0.90, 33):
        y = valid[label].astype(bool)
        pred = valid[prob_col].ge(th)
        tp = int((pred & y).sum())
        fp = int((pred & ~y).sum())
        fn = int((~pred & y).sum())
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        alarm_rate = pred.mean()
        objective = 2.0 * recall + 0.55 * precision - 0.35 * alarm_rate
        if objective > best[1]:
            best = (float(th), float(objective))
    return best[0]


def numeric_feature_cols(frame: pd.DataFrame, exclude: set[str], max_cols: int = 140) -> list[str]:
    cols = []
    for col in frame.columns:
        if col in exclude:
            continue
        s = pd.to_numeric(frame[col], errors="coerce")
        if s.notna().mean() >= 0.45:
            cols.append(col)
    return cols[:max_cols]


def run_risk(args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    macro = pd.read_parquet(PARQUET / "macro_panel.parquet")
    macro["date"] = pd.to_datetime(macro["date"])
    labels = [c for c in RISK_LABELS if c in macro.columns]
    exclude = {"date", "asset", "role", *labels}
    base_features = numeric_feature_cols(macro, exclude, max_cols=120)
    configs: list[tuple[str, list[str], pd.DataFrame]] = [("baseline", base_features, macro[["date", *labels, *base_features]].copy())]
    for window in [20, 40, 64, 126]:
        e = read_csv(SSL2 / f"macro_w{window}_ssl2_embeddings.csv", parse_dates=["date"])
        if e.empty:
            continue
        keep = ["date", *emb_cols(e)]
        merged = macro[["date", *labels, *base_features]].merge(e[keep], on="date", how="inner")
        configs.append((f"ssl2_w{window}", base_features + emb_cols(e), merged))
    if all((SSL2 / f"macro_w{w}_ssl2_embeddings.csv").exists() for w in [20, 40, 64, 126]):
        merged = macro[["date", *labels, *base_features]].copy()
        ssl_features = []
        for window in [20, 40, 64, 126]:
            e = read_csv(SSL2 / f"macro_w{window}_ssl2_embeddings.csv", parse_dates=["date"])
            rename = {c: f"w{window}_{c}" for c in emb_cols(e)}
            e = e[["date", *emb_cols(e)]].rename(columns=rename)
            merged = merged.merge(e, on="date", how="inner")
            ssl_features += list(rename.values())
        configs.append(("ssl2_multi", base_features + ssl_features, merged))

    rows = []
    pred_frames = []
    for label in labels:
        for model_name, features, frame in configs:
            data = frame.dropna(subset=[label]).sort_values("date")
            train = data[data["date"].le(pd.Timestamp(args.risk_train_end))]
            valid = data[data["date"].gt(pd.Timestamp(args.risk_train_end)) & data["date"].le(pd.Timestamp(args.risk_valid_end))]
            test = data[data["date"].gt(pd.Timestamp(args.risk_valid_end))]
            if train.empty or valid.empty or test.empty or train[label].nunique() < 2:
                continue
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                HistGradientBoostingClassifier(max_iter=220, learning_rate=0.035, max_leaf_nodes=13, min_samples_leaf=45, l2_regularization=0.8, random_state=42),
            )
            model.fit(train[features], train[label].astype(int))
            parts = []
            for split, part in [("train", train), ("valid", valid), ("test", test)]:
                p = model.predict_proba(part[features])[:, 1]
                sub = part[["date", label]].copy()
                sub["split"] = split
                sub["risk_model"] = model_name
                sub["label"] = label
                sub["prob"] = p
                parts.append(sub)
            pred = pd.concat(parts, ignore_index=True)
            th = choose_risk_threshold(pred[pred["split"].eq("valid")], label, "prob")
            m = classification_metrics(pred, label, "prob", th)
            m.update({"task": "risk_off", "model": model_name, "label": label, "features": len(features)})
            rows.append(m)
            pred_frames.append(pred)
    metrics = pd.DataFrame(rows)
    if pred_frames:
        pd.concat(pred_frames, ignore_index=True).to_csv(out_dir / "risk_ssl2_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(out_dir / "risk_ssl2_metrics.csv", index=False, encoding="utf-8-sig")
    return metrics


def split_frame(frame: pd.DataFrame, train_end: str, valid_end: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = frame[frame["date"].le(pd.Timestamp(train_end))].copy()
    valid = frame[frame["date"].gt(pd.Timestamp(train_end)) & frame["date"].le(pd.Timestamp(valid_end))].copy()
    test = frame[frame["date"].gt(pd.Timestamp(valid_end))].copy()
    return train, valid, test


def rank_data(frame: pd.DataFrame, features: list[str], label: str) -> tuple[pd.DataFrame, pd.Series, list[int], pd.DataFrame]:
    data = frame.dropna(subset=[label]).sort_values(["date", "etf_ticker"]).copy()
    data = data[data.groupby("date")["etf_ticker"].transform("size").ge(2)].reset_index(drop=True)
    x = data[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = x.groupby(data["date"]).transform(lambda s: s.fillna(s.median())).fillna(0.0)
    y = data[label].astype(int)
    group = data.groupby("date", sort=False).size().astype(int).tolist()
    return x, y, group, data


def train_ranker(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, features: list[str], label: str) -> pd.DataFrame:
    x_train, y_train, g_train, _ = rank_data(train, features, label)
    x_valid, y_valid, g_valid, _ = rank_data(valid, features, label)
    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=420,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=4,
        min_child_samples=18,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_alpha=1.0,
        reg_lambda=7.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(x_train, y_train, group=g_train, eval_set=[(x_valid, y_valid)], eval_group=[g_valid], eval_at=[1, 3, 5], callbacks=[early_stopping(80), log_evaluation(0)])
    frames = []
    for split, part in [("train", train), ("valid", valid), ("test", test)]:
        x, _, _, data = rank_data(part, features, label)
        data = data.copy()
        data["split"] = split
        data["ranker_score"] = model.predict(x)
        frames.append(data)
    return pd.concat(frames, ignore_index=True)


def etf_backtest(pred: pd.DataFrame, score_col: str, horizon: str, top_k: int) -> tuple[pd.DataFrame, dict]:
    ret_col, bench_col, excess_col = ("forward_5D_return", "benchmark_forward_5D_return", "forward_5D_excess") if horizon == "1W" else ("forward_20D_return", "benchmark_forward_20D_return", "forward_20D_excess")
    dates = pd.DatetimeIndex(pred[pred["split"].eq("test")]["date"].dropna().unique()).sort_values()
    freq = "W-FRI" if horizon == "1W" else "M"
    dates = pd.Series(dates, index=dates).groupby(dates.to_period(freq)).max().tolist()
    rows = []
    for dt in dates:
        part = pred[pred["date"].eq(dt)].dropna(subset=[score_col, ret_col, bench_col, excess_col]).copy()
        if part.shape[0] < top_k:
            continue
        picks = part.nlargest(top_k, score_col)
        rows.append(
            {
                "date": dt,
                "return": float(picks[ret_col].mean()),
                "benchmark_return": float(picks[bench_col].mean()),
                "excess": float(picks[excess_col].mean()),
                "selected": ",".join(picks["etf_ticker"].astype(str)),
            }
        )
    raw = pd.DataFrame(rows)
    periods_per_year = 52 if horizon == "1W" else 12
    if raw.empty:
        return raw, {"periods": 0}
    eq = (1 + raw["return"].astype(float)).cumprod()
    summary = {
        "periods": int(raw.shape[0]),
        "cumulative_return": float(eq.iloc[-1] - 1),
        "CAGR": cagr(float(eq.iloc[-1]), raw.shape[0], periods_per_year),
        "MDD": max_drawdown(eq),
        "Sharpe": sharpe(raw["return"], periods_per_year),
        "hit_excess": float((raw["excess"] > 0).mean()),
        "hit_positive": float((raw["return"] > 0).mean()),
        "avg_return": float(raw["return"].mean()),
        "avg_excess": float(raw["excess"].mean()),
    }
    return raw, summary


def rank_ic(pred: pd.DataFrame, score_col: str, target_col: str) -> dict:
    vals = []
    for _, part in pred[pred["split"].eq("test")].groupby("date"):
        if part[score_col].notna().sum() < 3 or part[target_col].notna().sum() < 3:
            continue
        corr = spearmanr(part[score_col], part[target_col], nan_policy="omit").correlation
        if not pd.isna(corr):
            vals.append(float(corr))
    return {"mean_rank_ic": float(np.mean(vals)) if vals else np.nan, "positive_ic_rate": float((np.array(vals) > 0).mean()) if vals else np.nan, "ic_dates": len(vals)}


def run_etf(args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    raw = read_csv(ROOT / "outputs/gaps_long_lived_etf_leadership_latest/tables/long_lived_scored_features.csv", parse_dates=["date"])
    if raw.empty:
        return pd.DataFrame()
    raw = raw.rename(columns={"asset": "asset_old"})
    configs = [("baseline", raw.copy(), v3.FEATURES_1W, v3.FEATURES_1M)]
    for windows in [[10], [20], [40], [64], [10, 20, 40, 64]]:
        frame = raw.copy()
        ssl_cols = []
        for w in windows:
            e = read_csv(SSL2 / f"etf_w{w}_ssl2_embeddings.csv", parse_dates=["date"])
            if e.empty:
                continue
            rename = {c: f"w{w}_{c}" for c in emb_cols(e)}
            e = e.rename(columns={"asset": "etf_ticker"})
            frame = frame.merge(e[["date", "etf_ticker", *emb_cols(e)]].rename(columns=rename), on=["date", "etf_ticker"], how="left")
            ssl_cols += list(rename.values())
        if ssl_cols:
            configs.append((f"ssl2_w{'_'.join(map(str, windows))}", frame, v3.FEATURES_1W + ssl_cols, v3.FEATURES_1M + ssl_cols))
    rows = []
    all_trades = []
    for model_name, frame, f1w, f1m in configs:
        train, valid, test = split_frame(frame, args.etf_train_end, args.etf_valid_end)
        if train.empty or valid.empty or test.empty:
            continue
        for horizon, label, features, top_k in [
            ("1W", "label_5D_group_rank_int", f1w, 3),
            ("1M", "label_20D_group_rank_int", f1m, 5),
        ]:
            features = [c for c in features if c in frame.columns]
            pred = train_ranker(train, valid, test, features, label)
            raw_bt, summary = etf_backtest(pred, "ranker_score", horizon, top_k)
            summary.update({"task": "etf_leadership", "model": model_name, "horizon": horizon, "top_k": top_k, "features": len(features)})
            target_col = "forward_5D_excess" if horizon == "1W" else "forward_20D_excess"
            summary.update(rank_ic(pred, "ranker_score", target_col))
            rows.append(summary)
            if not raw_bt.empty:
                raw_bt["model"] = model_name
                raw_bt["horizon"] = horizon
                all_trades.append(raw_bt)
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "etf_ssl2_backtest_summary.csv", index=False, encoding="utf-8-sig")
    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv(out_dir / "etf_ssl2_backtest_trades.csv", index=False, encoding="utf-8-sig")
    return out


SAFE_BASE = [
    "score_0_100",
    "technical_score",
    "driver_fit_score",
    "beta_fit_score",
    "calibrated_prob_1w",
    "calibrated_prob_4w",
    "institutional_score_0_100",
    "macro_US10Y_driver_chg_5d",
    "macro_US10Y_driver_chg_20d",
    "macro_US2Y_chg_5d",
    "macro_US2Y_chg_20d",
    "macro_US10Y_REAL_chg_20d",
    "macro_DXY_driver_ret_20d",
    "macro_USDKRW_driver_ret_20d",
    "macro_VIX_chg_20d",
    "macro_HY_OAS_chg_20d",
    "macro_GOLD_driver_ret_20d",
    "macro_HYG_IEF_ret_20d",
    "macro_axis1_vol_credit_stress",
    "macro_axis2_fx_liquidity_stress",
    "macro_axis3_peak_fragility_stress",
    "macro_risk_off_score",
]


def safe_label(frame: pd.DataFrame, target: str) -> pd.Series:
    pct = frame.groupby("date")[target].rank(pct=True, method="average")
    return np.ceil(pct * 5).sub(1).clip(0, 4).where(pct.notna())


def safe_rank_data(frame: pd.DataFrame, features: list[str], label: str) -> tuple[pd.DataFrame, pd.Series, list[int], pd.DataFrame]:
    data = frame.dropna(subset=[label]).sort_values(["date", "symbol"]).copy()
    data = data[data.groupby("date")["symbol"].transform("size").ge(2)].reset_index(drop=True)
    x = data[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = x.groupby(data["date"]).transform(lambda s: s.fillna(s.median())).fillna(0.0)
    y = data[label].astype(int)
    group = data.groupby("date", sort=False).size().astype(int).tolist()
    return x, y, group, data


def train_safe_ranker(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, features: list[str], label: str) -> pd.DataFrame:
    x_train, y_train, g_train, _ = safe_rank_data(train, features, label)
    x_valid, y_valid, g_valid, _ = safe_rank_data(valid, features, label)
    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=480,
        learning_rate=0.025,
        num_leaves=15,
        max_depth=4,
        min_child_samples=12,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_alpha=1.0,
        reg_lambda=8.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(x_train, y_train, group=g_train, eval_set=[(x_valid, y_valid)], eval_group=[g_valid], eval_at=[1, 3], callbacks=[early_stopping(80), log_evaluation(0)])
    frames = []
    for split, part in [("train", train), ("valid", valid), ("test", test)]:
        x, _, _, data = safe_rank_data(part, features, label)
        data = data.copy()
        data["split"] = split
        data["safe_ranker_score"] = model.predict(x)
        frames.append(data)
    return pd.concat(frames, ignore_index=True)


def safe_bt(pred: pd.DataFrame, target: str, ret_col: str, top_k: int = 3) -> dict:
    rows = []
    for date, part in pred[pred["split"].eq("test")].groupby("date"):
        picks = part.nlargest(min(top_k, len(part)), "safe_ranker_score")
        actual = part.nlargest(min(top_k, len(part)), target)
        rows.append(
            {
                "date": date,
                "picked_return": float(picks[ret_col].mean()),
                "picked_target": float(picks[target].mean()),
                "safe_avg_target": float(part[target].mean()),
                "beat_safe_average": int(picks[target].mean() > part[target].mean()),
                "overlap": len(set(picks["symbol"]) & set(actual["symbol"])) / max(min(top_k, len(part)), 1),
            }
        )
    raw = pd.DataFrame(rows)
    return {
        "periods": int(raw.shape[0]),
        "avg_picked_return": float(raw["picked_return"].mean()) if not raw.empty else np.nan,
        "avg_picked_target": float(raw["picked_target"].mean()) if not raw.empty else np.nan,
        "avg_safe_target": float(raw["safe_avg_target"].mean()) if not raw.empty else np.nan,
        "beat_safe_average_rate": float(raw["beat_safe_average"].mean()) if not raw.empty else np.nan,
        "topk_overlap_rate": float(raw["overlap"].mean()) if not raw.empty else np.nan,
    }


def run_safe(args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    raw = pd.read_parquet(PARQUET / "safe_panel.parquet")
    raw["date"] = pd.to_datetime(raw["date"])
    raw["safe_label_1w"] = safe_label(raw, "safe_target_1w")
    raw["safe_label_1m"] = safe_label(raw, "safe_target_1m")
    configs = [("baseline", raw.copy(), [c for c in SAFE_BASE if c in raw.columns])]
    for windows in [[20], [60], [20, 60]]:
        frame = raw.copy()
        ssl_cols = []
        for w in windows:
            e = read_csv(SSL2 / f"safe_w{w}_ssl2_embeddings.csv", parse_dates=["date"])
            if e.empty:
                continue
            rename = {c: f"w{w}_{c}" for c in emb_cols(e)}
            e = e.rename(columns={"asset": "symbol"})
            frame = frame.merge(e[["date", "symbol", *emb_cols(e)]].rename(columns=rename), on=["date", "symbol"], how="left")
            ssl_cols += list(rename.values())
        if ssl_cols:
            configs.append((f"ssl2_w{'_'.join(map(str, windows))}", frame, [c for c in SAFE_BASE if c in frame.columns] + ssl_cols))
    rows = []
    for model_name, frame, features in configs:
        train = frame[frame["date"].le(pd.Timestamp(args.safe_train_end))]
        valid = frame[frame["date"].gt(pd.Timestamp(args.safe_train_end)) & frame["date"].le(pd.Timestamp(args.safe_valid_end))]
        test = frame[frame["date"].gt(pd.Timestamp(args.safe_valid_end))]
        if train.empty or valid.empty or test.empty:
            continue
        for horizon, label, target, ret_col in [
            ("1W", "safe_label_1w", "safe_target_1w", "realized_return_1w"),
            ("1M", "safe_label_1m", "safe_target_1m", "realized_return_4w"),
        ]:
            pred = train_safe_ranker(train, valid, test, features, label)
            summary = safe_bt(pred, target, ret_col, top_k=3)
            summary.update({"task": "safe_asset", "model": model_name, "horizon": horizon, "features": len(features)})
            summary.update(rank_ic(pred.rename(columns={"symbol": "etf_ticker"}), "safe_ranker_score", target))
            rows.append(summary)
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "safe_ssl2_backtest_summary.csv", index=False, encoding="utf-8-sig")
    return out


def cagr(final_value: float, periods: int, periods_per_year: int) -> float:
    return float(final_value ** (periods_per_year / periods) - 1.0) if periods > 0 and final_value > 0 else np.nan


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min()) if not equity.empty else np.nan


def sharpe(returns: pd.Series, periods_per_year: int) -> float:
    std = returns.std(ddof=1)
    return float(returns.mean() / std * np.sqrt(periods_per_year)) if std and not pd.isna(std) else np.nan


def build_operational_decision_table(risk: pd.DataFrame, etf: pd.DataFrame, safe: pd.DataFrame) -> pd.DataFrame:
    """Choose only SSL windows that beat the existing operating baseline.

    The acceptance rule is intentionally different by head:
    - Risk-Off is recall-sensitive, so a tiny AUC gain is not enough when recall falls.
    - ETF Leadership is portfolio-performance-sensitive, so Sharpe and hit rate must improve.
    - Safe selection is defense-sensitive, so beating the safe basket average is primary.
    """
    rows: list[dict] = []

    if not risk.empty:
        for label, part in risk.groupby("label"):
            base = part[part["model"].eq("baseline")]
            if base.empty:
                continue
            base = base.iloc[0]
            candidates = part.dropna(subset=["test_auc"]).sort_values("test_auc", ascending=False)
            if candidates.empty:
                continue
            best = candidates.iloc[0]
            auc_delta = float(best["test_auc"] - base["test_auc"])
            recall_delta = float(best["test_recall"] - base["test_recall"])
            adopt = bool(best["model"] != "baseline" and auc_delta >= 0.01 and recall_delta >= -0.03)
            rows.append(
                {
                    "task": "risk_off",
                    "target": label,
                    "baseline_model": "baseline",
                    "best_model": best["model"],
                    "baseline_primary": float(base["test_auc"]),
                    "best_primary": float(best["test_auc"]),
                    "primary_metric": "test_auc",
                    "baseline_secondary": float(base["test_recall"]),
                    "best_secondary": float(best["test_recall"]),
                    "secondary_metric": "test_recall",
                    "delta_primary": auc_delta,
                    "delta_secondary": recall_delta,
                    "decision": "adopt" if adopt else "keep_baseline",
                }
            )

    if not etf.empty:
        for horizon, part in etf.groupby("horizon"):
            base = part[part["model"].eq("baseline")]
            if base.empty:
                continue
            base = base.iloc[0]
            candidates = part.dropna(subset=["Sharpe"]).sort_values("Sharpe", ascending=False)
            if candidates.empty:
                continue
            best = candidates.iloc[0]
            sharpe_delta = float(best["Sharpe"] - base["Sharpe"])
            hit_delta = float(best["hit_excess"] - base["hit_excess"])
            adopt = bool(best["model"] != "baseline" and sharpe_delta >= 0.05 and hit_delta >= 0.0)
            rows.append(
                {
                    "task": "etf_leadership",
                    "target": horizon,
                    "baseline_model": "baseline",
                    "best_model": best["model"],
                    "baseline_primary": float(base["Sharpe"]),
                    "best_primary": float(best["Sharpe"]),
                    "primary_metric": "Sharpe",
                    "baseline_secondary": float(base["hit_excess"]),
                    "best_secondary": float(best["hit_excess"]),
                    "secondary_metric": "hit_excess",
                    "delta_primary": sharpe_delta,
                    "delta_secondary": hit_delta,
                    "decision": "adopt" if adopt else "keep_baseline",
                }
            )

    if not safe.empty:
        for horizon, part in safe.groupby("horizon"):
            base = part[part["model"].eq("baseline")]
            if base.empty:
                continue
            base = base.iloc[0]
            candidates = part.dropna(subset=["beat_safe_average_rate"]).sort_values(
                ["beat_safe_average_rate", "avg_picked_return"], ascending=[False, False]
            )
            if candidates.empty:
                continue
            best = candidates.iloc[0]
            beat_delta = float(best["beat_safe_average_rate"] - base["beat_safe_average_rate"])
            ret_delta = float(best["avg_picked_return"] - base["avg_picked_return"])
            adopt = bool(best["model"] != "baseline" and beat_delta >= 0.03 and ret_delta >= -0.002)
            rows.append(
                {
                    "task": "safe_asset",
                    "target": horizon,
                    "baseline_model": "baseline",
                    "best_model": best["model"],
                    "baseline_primary": float(base["beat_safe_average_rate"]),
                    "best_primary": float(best["beat_safe_average_rate"]),
                    "primary_metric": "beat_safe_average_rate",
                    "baseline_secondary": float(base["avg_picked_return"]),
                    "best_secondary": float(best["avg_picked_return"]),
                    "secondary_metric": "avg_picked_return",
                    "delta_primary": beat_delta,
                    "delta_secondary": ret_delta,
                    "decision": "adopt" if adopt else "keep_baseline",
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = {x.strip() for x in args.tasks.split(",") if x.strip()}
    risk = run_risk(args, out_dir) if "risk" in tasks else pd.DataFrame()
    etf = run_etf(args, out_dir) if "etf" in tasks else pd.DataFrame()
    safe = run_safe(args, out_dir) if "safe" in tasks else pd.DataFrame()
    summary = {
        "risk_rows": int(risk.shape[0]),
        "etf_rows": int(etf.shape[0]),
        "safe_rows": int(safe.shape[0]),
    }
    decisions = build_operational_decision_table(risk, etf, safe)
    if not decisions.empty:
        decisions.to_csv(out_dir / "operational_model_adoption.csv", index=False, encoding="utf-8-sig")
        (out_dir / "operational_model_adoption.json").write_text(
            decisions.to_json(orient="records", force_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary["decision_rows"] = int(decisions.shape[0])
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    for name, frame in [("risk", risk), ("etf", etf), ("safe", safe)]:
        if not frame.empty:
            print(f"\n[{name}]\n{frame.head(20).to_string(index=False)}", flush=True)
    if not decisions.empty:
        print(f"\n[operational_decisions]\n{decisions.to_string(index=False)}", flush=True)


if __name__ == "__main__":
    main()
