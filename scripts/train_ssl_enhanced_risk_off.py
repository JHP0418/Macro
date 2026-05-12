from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Risk-Off Sentinel upgraded with macro SSL/VQ/NF embeddings.")
    p.add_argument("--sentinel", default=str(ROOT / "outputs" / "daily_risk_off_sentinel_latest" / "tables" / "daily_sentinel_history.csv"))
    p.add_argument("--macro-ssl", default=str(ROOT / "outputs" / "ssl_market_embeddings_latest" / "macro_ssl_embeddings.csv"))
    p.add_argument("--prices", default=str(ROOT / "outputs" / "etf_leadership_from_cache" / "prices_adj_close.csv"))
    p.add_argument("--output-dir", default=str(ROOT / "outputs" / "ssl_risk_off_sentinel_latest"))
    p.add_argument("--train-end", default="2022-12-31")
    p.add_argument("--valid-end", default="2024-12-31")
    return p.parse_args()


def load_prices(path: Path) -> pd.DataFrame:
    prices = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    return prices.apply(pd.to_numeric, errors="coerce")


def build_targets(panel: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    candidates = {
        "qqq": "QQQ",
        "kospi": "069500.KS",
        "sox": "381180.KS",
        "us_growth": "133690.KS",
    }
    for name, ticker in candidates.items():
        if ticker not in prices.columns:
            continue
        px = prices[ticker].reindex(out["date"]).ffill()
        out[f"{name}_fwd_5d"] = px.shift(-5).to_numpy() / px.to_numpy() - 1.0
        out[f"{name}_fwd_20d"] = px.shift(-20).to_numpy() / px.to_numpy() - 1.0
    fwd5_cols = [c for c in out.columns if c.endswith("_fwd_5d")]
    fwd20_cols = [c for c in out.columns if c.endswith("_fwd_20d")]
    out["risk_assets_fwd_5d_avg"] = out[fwd5_cols].mean(axis=1, skipna=True)
    out["risk_assets_fwd_20d_avg"] = out[fwd20_cols].mean(axis=1, skipna=True)
    out["label_1w_practical_loss"] = out["risk_assets_fwd_5d_avg"].le(-0.02).astype(int)
    out["label_1m_practical_loss"] = out["risk_assets_fwd_20d_avg"].le(-0.05).astype(int)
    return out


def prepare_panel(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str]]:
    sentinel = pd.read_csv(args.sentinel, parse_dates=["Date"]).rename(columns={"Date": "date"})
    ssl = pd.read_csv(args.macro_ssl, parse_dates=["date"]).drop(columns=["entity"], errors="ignore")
    panel = sentinel.merge(ssl, on="date", how="left")
    prices = load_prices(Path(args.prices))
    panel = build_targets(panel, prices)
    exclude = {
        "date",
        "dominant_component",
        "sentinel_state",
        "label_1w_practical_loss",
        "label_1m_practical_loss",
        "risk_assets_fwd_5d_avg",
        "risk_assets_fwd_20d_avg",
    }
    features = []
    for col in panel.columns:
        if col in exclude or col.endswith("_fwd_5d") or col.endswith("_fwd_20d"):
            continue
        if pd.api.types.is_numeric_dtype(panel[col]):
            features.append(col)
    return panel.dropna(subset=["label_1w_practical_loss", "label_1m_practical_loss"]), features


def train_one(panel: pd.DataFrame, features: list[str], label: str, train_end: str, valid_end: str) -> tuple[pd.DataFrame, dict]:
    train = panel[panel["date"].le(pd.Timestamp(train_end))]
    valid = panel[panel["date"].gt(pd.Timestamp(train_end)) & panel["date"].le(pd.Timestamp(valid_end))]
    test = panel[panel["date"].gt(pd.Timestamp(valid_end))]
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        HistGradientBoostingClassifier(
            max_iter=220,
            learning_rate=0.035,
            max_leaf_nodes=13,
            min_samples_leaf=45,
            l2_regularization=0.8,
            random_state=42,
        ),
    )
    model.fit(train[features], train[label].astype(int))
    rows = []
    for split_name, part in [("train", train), ("valid", valid), ("test", test)]:
        pred = model.predict_proba(part[features])[:, 1]
        sub = part[["date", "risk_off_score", "sentinel_state", "dominant_component", label]].copy()
        sub["split"] = split_name
        sub[f"{label}_ssl_prob"] = pred
        rows.append(sub)
    preds = pd.concat(rows, ignore_index=True)
    threshold = choose_threshold(preds[preds["split"].eq("valid")], label, f"{label}_ssl_prob")
    metrics = score_metrics(preds, label, f"{label}_ssl_prob", threshold)
    metrics["threshold"] = threshold
    metrics["label"] = label
    return preds, metrics


def choose_threshold(valid: pd.DataFrame, label: str, prob_col: str) -> float:
    best = (0.5, -1.0)
    for th in np.linspace(0.25, 0.85, 25):
        pred = valid[prob_col].ge(th)
        actual = valid[label].astype(bool)
        tp = int((pred & actual).sum())
        fp = int((pred & ~actual).sum())
        fn = int((~pred & actual).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        false_alarm = fp / max(tp + fp, 1)
        objective = 1.3 * recall + 0.7 * precision - 0.35 * false_alarm
        if objective > best[1]:
            best = (float(th), float(objective))
    return best[0]


def score_metrics(preds: pd.DataFrame, label: str, prob_col: str, threshold: float) -> dict:
    rows = {}
    for split_name, part in preds.groupby("split"):
        y = part[label].astype(int)
        p = part[prob_col].astype(float)
        pred = p.ge(threshold)
        actual = y.astype(bool)
        tp = int((pred & actual).sum())
        fp = int((pred & ~actual).sum())
        fn = int((~pred & actual).sum())
        tn = int((~pred & ~actual).sum())
        rows[f"{split_name}_samples"] = int(part.shape[0])
        rows[f"{split_name}_positive_rate"] = float(y.mean())
        rows[f"{split_name}_auc"] = safe_auc(y, p)
        rows[f"{split_name}_ap"] = safe_ap(y, p)
        rows[f"{split_name}_brier"] = float(brier_score_loss(y, p)) if y.nunique() > 1 else np.nan
        rows[f"{split_name}_precision"] = tp / max(tp + fp, 1)
        rows[f"{split_name}_recall"] = tp / max(tp + fn, 1)
        rows[f"{split_name}_false_alarm"] = fp / max(tp + fp, 1)
        rows[f"{split_name}_accuracy"] = (tp + tn) / max(tp + fp + fn + tn, 1)
    return rows


def safe_auc(y: pd.Series, p: pd.Series) -> float:
    return float(roc_auc_score(y, p)) if y.nunique() > 1 else np.nan


def safe_ap(y: pd.Series, p: pd.Series) -> float:
    return float(average_precision_score(y, p)) if y.nunique() > 1 else np.nan


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    panel, features = prepare_panel(args)
    all_preds = []
    metrics = []
    for label in ["label_1w_practical_loss", "label_1m_practical_loss"]:
        preds, m = train_one(panel, features, label, args.train_end, args.valid_end)
        all_preds.append(preds)
        metrics.append(m)
    pred = all_preds[0].merge(
        all_preds[1].drop(columns=["risk_off_score", "sentinel_state", "dominant_component"]),
        on=["date", "split"],
        how="outer",
    )
    pred.to_csv(tables / "ssl_risk_off_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(metrics).to_csv(tables / "ssl_risk_off_metrics.csv", index=False, encoding="utf-8-sig")
    panel[["date", *features, "label_1w_practical_loss", "label_1m_practical_loss"]].to_csv(tables / "ssl_risk_off_training_panel.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(metrics).to_string(index=False))


if __name__ == "__main__":
    main()
