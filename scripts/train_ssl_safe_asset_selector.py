from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker, early_stopping, log_evaluation


ROOT = Path(__file__).resolve().parents[1]
SAFE_GROUPS = {"Cash/short bonds", "USD cash", "Korea bonds", "US long bonds", "US IG bonds", "Gold", "Korea defensive"}
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
    "Oil",
    "Commodity/Oil",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SSL-enhanced safe asset selector for risk-off weeks.")
    p.add_argument("--weekly", default=str(ROOT / "outputs" / "weekly_screening_rank_backtest_latest" / "tables" / "weekly_calibrated_rank_panel.csv"))
    p.add_argument("--sentinel", default=str(ROOT / "outputs" / "daily_risk_off_sentinel_latest" / "tables" / "daily_sentinel_history.csv"))
    p.add_argument("--macro-ssl", default=str(ROOT / "outputs" / "ssl_market_embeddings_latest" / "macro_ssl_embeddings.csv"))
    p.add_argument("--safe-ssl", default=str(ROOT / "outputs" / "ssl_safe_asset_embeddings_latest" / "safe_ssl_embeddings.csv"))
    p.add_argument("--output-dir", default=str(ROOT / "outputs" / "ssl_safe_asset_selector_latest"))
    p.add_argument("--train-end", default="2024-12-31")
    p.add_argument("--valid-end", default="2025-12-31")
    p.add_argument("--risk-threshold", type=float, default=25.0)
    p.add_argument("--top-k", type=int, default=3)
    return p.parse_args()


def attach_asof(left: pd.DataFrame, right: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    return pd.merge_asof(left.sort_values(date_col), right.sort_values(date_col), on=date_col, direction="backward")


def prepare(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str]]:
    weekly = pd.read_csv(args.weekly, parse_dates=["date"])
    sentinel = pd.read_csv(args.sentinel, parse_dates=["Date"]).rename(columns={"Date": "date"})
    sentinel_cols = ["date", "risk_off_score", "risk_off_momentum_5d", "risk_budget_pct", "equity_penalty", "safe_asset_boost"]
    sentinel = sentinel[[c for c in sentinel_cols if c in sentinel.columns]]
    macro_ssl = pd.read_csv(args.macro_ssl, parse_dates=["date"]).drop(columns=["entity"], errors="ignore")
    macro_cols = ["date", *[c for c in macro_ssl.columns if c.startswith("ssl_emb_")][:12], "ssl_vq_state", "ssl_vq_distance", "ssl_flow_nll", "ssl_flow_confidence"]
    macro_ssl = macro_ssl[[c for c in macro_cols if c in macro_ssl.columns]].rename(columns={c: f"macro_{c}" for c in macro_cols if c != "date"})
    safe_ssl = pd.read_csv(args.safe_ssl, parse_dates=["date"]).rename(columns={"entity": "symbol"})
    safe_cols = ["date", "symbol", *[c for c in safe_ssl.columns if c.startswith("ssl_emb_")][:12], "ssl_vq_state", "ssl_vq_distance", "ssl_flow_nll", "ssl_flow_confidence"]
    safe_ssl = safe_ssl[[c for c in safe_cols if c in safe_ssl.columns]].rename(columns={c: f"safe_{c}" for c in safe_cols if c not in {"date", "symbol"}})

    panel = attach_asof(weekly.sort_values("date"), sentinel)
    panel = attach_asof(panel, macro_ssl)
    panel = panel.merge(safe_ssl, on=["date", "symbol"], how="left")
    panel["is_safe_asset"] = panel["group"].isin(SAFE_GROUPS)
    panel["is_risk_asset"] = panel["group"].isin(RISK_GROUPS)
    risk = panel[panel["is_risk_asset"]].groupby("date", as_index=False).agg(
        risk_avg_1w=("realized_return_1w", "mean"),
        risk_avg_1m=("realized_return_4w", "mean"),
    )
    panel = panel.merge(risk, on="date", how="left")
    panel["safe_target_1w"] = panel["realized_return_1w"] - 0.60 * panel["realized_return_1w"].clip(upper=0).abs() - panel["risk_avg_1w"].fillna(0)
    panel["safe_target_1m"] = panel["realized_return_4w"] - 0.60 * panel["realized_return_4w"].clip(upper=0).abs() - panel["risk_avg_1m"].fillna(0)
    panel = panel[panel["is_safe_asset"]].copy()
    panel["risk_week"] = pd.to_numeric(panel["risk_off_score"], errors="coerce").fillna(0).ge(args.risk_threshold)
    feature_base = [
        "score_0_100",
        "upside_prob_1w",
        "upside_prob_4w",
        "technical_score",
        "driver_fit_score",
        "beta_fit_score",
        "calibrated_prob_1w",
        "calibrated_prob_4w",
        "institutional_score_0_100",
        "risk_off_score",
        "risk_off_momentum_5d",
        "risk_budget_pct",
        "equity_penalty",
        "safe_asset_boost",
    ]
    ssl_features = [c for c in panel.columns if c.startswith("macro_ssl_") or c.startswith("safe_ssl_")]
    features = [c for c in feature_base + ssl_features if c in panel.columns]
    for col in features + ["safe_target_1w", "safe_target_1m"]:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")
    return panel.sort_values(["date", "symbol"]), features


def add_rank_labels(panel: pd.DataFrame, target: str, label: str) -> pd.DataFrame:
    out = panel.copy()
    pct = out.groupby("date")[target].rank(pct=True, method="average")
    out[label] = np.ceil(pct * 5.0).sub(1).clip(0, 4).where(pct.notna())
    return out


def rank_data(frame: pd.DataFrame, features: list[str], label: str) -> tuple[pd.DataFrame, pd.Series, list[int], pd.DataFrame]:
    data = frame.dropna(subset=[label]).sort_values(["date", "symbol"]).reset_index(drop=True)
    data = data[data.groupby("date")["symbol"].transform("size").ge(2)].reset_index(drop=True)
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
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=11,
        max_depth=3,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=6.0,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train, group=g_train, eval_set=[(x_valid, y_valid)], eval_group=[g_valid], eval_at=[1, 3], callbacks=[early_stopping(80), log_evaluation(100)])
    frames = []
    for split, part in [("train", train), ("valid", valid), ("test", test)]:
        x, _, _, data = rank_data(part, features, label)
        data = data.copy()
        data["split"] = split
        data["safe_ssl_ranker_score"] = model.predict(x)
        frames.append(data)
    return pd.concat(frames, ignore_index=True)


def backtest(pred: pd.DataFrame, target: str, top_k: int) -> tuple[pd.DataFrame, dict]:
    rows = []
    test = pred[pred["split"].eq("test")].copy()
    risk_dates = set(test.groupby("date")["risk_week"].max().loc[lambda s: s].index)
    for date, part in test[test["date"].isin(risk_dates)].groupby("date"):
        if part.empty:
            continue
        picks = part.nlargest(min(top_k, len(part)), "safe_ssl_ranker_score")
        actual = part.nlargest(min(top_k, len(part)), target)
        rows.append(
            {
                "date": date,
                "picked_return": picks["realized_return_4w" if target.endswith("1m") else "realized_return_1w"].mean(),
                "picked_target": picks[target].mean(),
                "safe_avg_target": part[target].mean(),
                "actual_top_target": actual[target].mean(),
                "overlap": len(set(picks["symbol"]) & set(actual["symbol"])) / max(min(top_k, len(part)), 1),
                "beat_safe_average": picks[target].mean() > part[target].mean(),
                "selected": ",".join(picks["symbol"].astype(str)),
            }
        )
    raw = pd.DataFrame(rows)
    summary = {
        "risk_off_periods": int(raw.shape[0]),
        "avg_picked_target": float(raw["picked_target"].mean()) if not raw.empty else np.nan,
        "avg_safe_target": float(raw["safe_avg_target"].mean()) if not raw.empty else np.nan,
        "beat_safe_average_rate": float(raw["beat_safe_average"].mean()) if not raw.empty else np.nan,
        "topk_overlap_rate": float(raw["overlap"].mean()) if not raw.empty else np.nan,
    }
    return raw, summary


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    panel, features = prepare(args)
    panel = add_rank_labels(add_rank_labels(panel, "safe_target_1w", "safe_label_1w"), "safe_target_1m", "safe_label_1m")
    train = panel[panel["date"].le(pd.Timestamp(args.train_end))]
    valid = panel[panel["date"].gt(pd.Timestamp(args.train_end)) & panel["date"].le(pd.Timestamp(args.valid_end))]
    test = panel[panel["date"].gt(pd.Timestamp(args.valid_end))]
    preds = []
    summaries = []
    raws = []
    for horizon, label, target in [("1w", "safe_label_1w", "safe_target_1w"), ("1m", "safe_label_1m", "safe_target_1m")]:
        pred = train_ranker(train, valid, test, features, label)
        pred["horizon"] = horizon
        raw, summary = backtest(pred, target, args.top_k)
        summary["horizon"] = horizon
        summaries.append(summary)
        raw["horizon"] = horizon
        preds.append(pred)
        raws.append(raw)
    pd.concat(preds, ignore_index=True).to_csv(tables / "ssl_safe_asset_predictions.csv", index=False, encoding="utf-8-sig")
    pd.concat(raws, ignore_index=True).to_csv(tables / "ssl_safe_asset_backtest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summaries).to_csv(tables / "ssl_safe_asset_summary.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
