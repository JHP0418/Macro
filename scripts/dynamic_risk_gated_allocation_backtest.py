from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "dynamic_risk_gated_allocation_latest"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"

EMA_PRED = ROOT / "outputs" / "ema_entry_meta_model_latest" / "tables" / "ema_entry_meta_predictions.csv"
CURRENT_ENTRY = ROOT / "outputs" / "ema_entry_meta_model_latest" / "tables" / "current_ema_entry_meta_signal.csv"
PROXY_SAFE_TRADES = ROOT / "outputs" / "long_history_proxy_selection_backtest_latest" / "tables" / "proxy_selection_backtest_trades.csv"
SAFE_V2_PRED = ROOT / "outputs" / "institutional_risk_off_v2_latest" / "tables" / "macro_conditioned_safe_asset_predictions.csv"
CURRENT_SAFE = ROOT / "outputs" / "institutional_risk_off_v2_latest" / "tables" / "current_safe_asset_recommendations_v2.csv"
RISK_V4 = ROOT / "outputs" / "risk_off_v4_event_label_latest" / "tables" / "risk_off_v4_walkforward_predictions.csv"
RISK_V4_CURRENT = ROOT / "outputs" / "risk_off_v4_event_label_latest" / "tables" / "current_risk_off_v4_state.csv"
SSL2_PRED = ROOT / "outputs" / "ssl2_head_backtest_latest" / "tables" / "risk_ssl2_predictions.csv"
SSL2_METRICS = ROOT / "outputs" / "ssl2_head_backtest_latest" / "tables" / "risk_ssl2_metrics.csv"
SSL2_ADOPTION = ROOT / "outputs" / "ssl2_head_backtest_latest" / "tables" / "operational_model_adoption.csv"


STAGE_RANK = {"Normal": 0, "Watch": 1, "De-risk": 2, "Cash": 3}


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def safe_float(x: object, fallback: float = np.nan) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else fallback
    except Exception:
        return fallback


def load_risk_sleeve() -> pd.DataFrame:
    pred = read_csv(EMA_PRED, parse_dates=["date"], low_memory=False)
    if pred.empty:
        return pd.DataFrame()
    x = pred[
        pred["strategy"].eq("ranker_top3_hybrid_1w")
        & pred["model"].eq("lightgbm_platt_calibrated_conservative")
    ].copy()
    x["risk_return"] = pd.to_numeric(x["portfolio_return"], errors="coerce")
    x["risk_excess"] = pd.to_numeric(x["excess_return"], errors="coerce")
    x["benchmark_return"] = x["risk_return"] - x["risk_excess"]
    x["entry_prob"] = pd.to_numeric(x["entry_prob"], errors="coerce")
    x["entry_threshold"] = pd.to_numeric(x["threshold"], errors="coerce")
    x["entry_allowed"] = x["invested"].astype(str).str.lower().eq("true")
    return x[["date", "source", "selected", "risk_return", "risk_excess", "benchmark_return", "entry_prob", "entry_threshold", "entry_allowed"]]


def weekly_dates(dates: pd.Series) -> set[pd.Timestamp]:
    idx = pd.DatetimeIndex(pd.to_datetime(dates).dropna().unique()).sort_values()
    if idx.empty:
        return set()
    return set(pd.Series(idx, index=idx).groupby(idx.to_period("W-FRI")).max())


def load_safe_sleeve() -> pd.DataFrame:
    proxy = read_csv(PROXY_SAFE_TRADES, parse_dates=["date", "exit_date"], low_memory=False)
    rows = []
    if not proxy.empty:
        p = proxy[
            proxy["strategy"].eq("safe_macro_weekly_top3")
            & proxy["date"].lt(pd.Timestamp("2025-01-01"))
        ].copy()
        p["safe_return"] = pd.to_numeric(p["period_return"], errors="coerce")
        rows.append(
            p[["date", "selected", "safe_return"]].assign(
                safe_source="long_proxy_safe_macro_top3",
                selected_names=p["selected"],
            )
        )
    safe = read_csv(SAFE_V2_PRED, parse_dates=["date"], low_memory=False)
    if not safe.empty:
        s = safe[
            safe["horizon"].astype(str).str.lower().eq("1w")
            & safe["date"].ge(pd.Timestamp("2025-01-01"))
            & safe["date"].isin(weekly_dates(safe["date"]))
        ].copy()
        s["safe_v2_ranker_score"] = pd.to_numeric(s["safe_v2_ranker_score"], errors="coerce")
        s["realized_return_1w"] = pd.to_numeric(s["realized_return_1w"], errors="coerce")
        s = s.dropna(subset=["safe_v2_ranker_score", "realized_return_1w"])
        s["rank"] = s.groupby("date")["safe_v2_ranker_score"].rank(ascending=False, method="first")
        top = s[s["rank"].le(3)].copy()
        if not top.empty:
            g = top.groupby("date").agg(
                safe_return=("realized_return_1w", "mean"),
                selected=("symbol", lambda x: ",".join(map(str, x))),
                selected_names=("name", lambda x: ",".join(map(str, x))),
            ).reset_index()
            g["safe_source"] = "db_gaps_macro_safe_v2_top3"
            rows.append(g[["date", "selected", "selected_names", "safe_return", "safe_source"]])
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True).sort_values("date")
    return out


def load_v4() -> pd.DataFrame:
    risk = read_csv(RISK_V4, parse_dates=["date"], low_memory=False)
    if risk.empty:
        return pd.DataFrame()
    frames = []
    for horizon in ["1w", "1m"]:
        x = risk[risk["horizon"].eq(horizon)].copy()
        cols = ["date", "risk_off_v4_prob", "risk_off_v4_stage", "risk_off_score", "axis1_vol_credit_stress", "axis2_fx_liquidity_stress", "axis3_peak_fragility_stress"]
        x = x[[c for c in cols if c in x.columns]].copy()
        x = x.rename(columns={c: f"{c}_{horizon}" for c in x.columns if c != "date"})
        frames.append(x.sort_values("date"))
    out = frames[0]
    for f in frames[1:]:
        out = pd.merge_asof(out.sort_values("date"), f.sort_values("date"), on="date", direction="backward")
    return out.sort_values("date")


def selected_ssl2_models() -> list[tuple[str, str, float]]:
    adoption = read_csv(SSL2_ADOPTION)
    metrics = read_csv(SSL2_METRICS)
    out = []
    if adoption.empty or metrics.empty:
        return out
    for _, row in adoption[adoption["task"].eq("risk_off")].iterrows():
        label = str(row["target"])
        model = str(row["best_model"] if row["decision"] == "adopt" else row["baseline_model"])
        m = metrics[metrics["label"].eq(label) & metrics["model"].eq(model)]
        if m.empty:
            continue
        out.append((label, model, safe_float(m.iloc[0]["threshold"], 1.0)))
    return out


def load_ssl2_alerts() -> pd.DataFrame:
    pred = read_csv(SSL2_PRED, parse_dates=["date"], low_memory=False)
    selected = selected_ssl2_models()
    if pred.empty or not selected:
        return pd.DataFrame()
    parts = []
    for label, model, threshold in selected:
        x = pred[pred["label"].eq(label) & pred["risk_model"].eq(model)].copy()
        if x.empty:
            continue
        x[f"ssl2_prob_{label}"] = pd.to_numeric(x["prob"], errors="coerce")
        x[f"ssl2_alert_{label}"] = x[f"ssl2_prob_{label}"] >= threshold
        parts.append(x[["date", f"ssl2_prob_{label}", f"ssl2_alert_{label}"]].sort_values("date"))
    if not parts:
        return pd.DataFrame()
    out = parts[0]
    for p in parts[1:]:
        out = pd.merge_asof(out.sort_values("date"), p.sort_values("date"), on="date", direction="backward")
    alert_cols = [c for c in out.columns if c.startswith("ssl2_alert_")]
    prob_cols = [c for c in out.columns if c.startswith("ssl2_prob_")]
    out["ssl2_alert_count"] = out[alert_cols].fillna(False).sum(axis=1)
    out["ssl2_max_prob"] = out[prob_cols].max(axis=1)
    out["ssl2_large_loss_alert"] = False
    for col in alert_cols:
        if "large_loss" in col:
            out["ssl2_large_loss_alert"] |= out[col].fillna(False).astype(bool)
    return out.sort_values("date")


def combine_panel() -> pd.DataFrame:
    risk_sleeve = load_risk_sleeve()
    safe_sleeve = load_safe_sleeve()
    v4 = load_v4()
    ssl2 = load_ssl2_alerts()
    if risk_sleeve.empty:
        return pd.DataFrame()
    panel = risk_sleeve.sort_values("date")
    if not safe_sleeve.empty:
        panel = pd.merge_asof(panel, safe_sleeve.sort_values("date"), on="date", direction="backward", tolerance=pd.Timedelta(days=10))
    if not v4.empty:
        panel = pd.merge_asof(panel.sort_values("date"), v4.sort_values("date"), on="date", direction="backward")
    if not ssl2.empty:
        panel = pd.merge_asof(panel.sort_values("date"), ssl2.sort_values("date"), on="date", direction="backward")
    panel["safe_return"] = pd.to_numeric(panel.get("safe_return"), errors="coerce").fillna(0.0)
    panel["ssl2_alert_count"] = pd.to_numeric(panel.get("ssl2_alert_count"), errors="coerce").fillna(0)
    panel["ssl2_large_loss_alert"] = panel.get("ssl2_large_loss_alert", False)
    return panel.sort_values("date")


def worst_stage(row: pd.Series) -> str:
    s1 = str(row.get("risk_off_v4_stage_1w", "Normal"))
    s2 = str(row.get("risk_off_v4_stage_1m", "Normal"))
    return max([s1, s2], key=lambda s: STAGE_RANK.get(s, 0))


def decide_weight(row: pd.Series) -> tuple[float, str]:
    stage = worst_stage(row)
    entry_allowed = bool(row.get("entry_allowed", False))
    ssl_large = bool(row.get("ssl2_large_loss_alert", False))
    ssl_count = int(safe_float(row.get("ssl2_alert_count"), 0))
    if stage == "Normal":
        risk = 0.60 if entry_allowed else 0.25
    elif stage == "Watch":
        risk = 0.45 if entry_allowed else 0.15
    elif stage == "De-risk":
        risk = 0.20 if entry_allowed else 0.05
    else:
        risk = 0.05
    if ssl_large:
        risk = min(risk, 0.20)
    if ssl_count >= 2:
        risk = min(risk, 0.25)
    if not entry_allowed:
        risk = min(risk, 0.25)
    risk = max(0.0, min(0.60, risk))
    return risk, stage


def backtest(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for _, row in panel.iterrows():
        risk_weight, stage = decide_weight(row)
        # Top3 sleeves respect the 20% per-product cap: a top3 sleeve can use at most 60%.
        safe_weight_raw = 1.0 - risk_weight
        safe_weight = min(0.60, safe_weight_raw)
        cash_weight = max(0.0, 1.0 - risk_weight - safe_weight)
        risk_return = safe_float(row.get("risk_return"), 0.0)
        safe_return = safe_float(row.get("safe_return"), 0.0)
        bench_return = safe_float(row.get("benchmark_return"), 0.0)
        port = risk_weight * risk_return + safe_weight * safe_return
        rows.append(
            {
                "date": row["date"],
                "risk_stage": stage,
                "risk_weight": risk_weight,
                "safe_weight": safe_weight,
                "cash_weight": cash_weight,
                "portfolio_return": port,
                "benchmark_return": bench_return,
                "excess_return": port - bench_return,
                "risk_return": risk_return,
                "safe_return": safe_return,
                "entry_allowed": bool(row.get("entry_allowed", False)),
                "entry_prob": row.get("entry_prob"),
                "entry_threshold": row.get("entry_threshold"),
                "ssl2_alert_count": row.get("ssl2_alert_count"),
                "ssl2_large_loss_alert": bool(row.get("ssl2_large_loss_alert", False)),
                "risk_selected": row.get("selected"),
                "safe_selected": row.get("selected_names"),
                "safe_source": row.get("safe_source"),
                "risk_off_v4_prob_1w": row.get("risk_off_v4_prob_1w"),
                "risk_off_v4_prob_1m": row.get("risk_off_v4_prob_1m"),
            }
        )
    trades = pd.DataFrame(rows).sort_values("date")
    summaries = []
    summaries.append(summarize(trades, "dynamic_risk_gated_allocation", "portfolio_return"))
    summaries.append(summarize(trades, "qqq_benchmark", "benchmark_return"))
    tmp = trades.copy()
    tmp["always_risk_return"] = tmp["risk_return"]
    tmp["always_entry_meta_return"] = np.where(tmp["entry_allowed"], tmp["risk_return"], tmp["safe_return"])
    tmp["always_60_40_return"] = 0.60 * tmp["risk_return"] + 0.40 * tmp["safe_return"]
    summaries.append(summarize(tmp, "always_risk_top3", "always_risk_return"))
    summaries.append(summarize(tmp, "entry_meta_switch_risk_or_safe", "always_entry_meta_return"))
    summaries.append(summarize(tmp, "static_60_risk_40_safe", "always_60_40_return"))
    return trades, pd.DataFrame(summaries).sort_values("Sharpe", ascending=False)


def summarize(frame: pd.DataFrame, strategy: str, ret_col: str) -> dict:
    x = frame.dropna(subset=[ret_col]).sort_values("date").copy()
    if x.empty:
        return {"strategy": strategy, "periods": 0}
    ret = pd.to_numeric(x[ret_col], errors="coerce").fillna(0.0)
    equity = (1.0 + ret).cumprod()
    years = max((x["date"].max() - x["date"].min()).days / 365.25, 1e-9)
    ann_vol = float(ret.std(ddof=1) * np.sqrt(52))
    mdd = float((equity / equity.cummax() - 1.0).min())
    bench = pd.to_numeric(x.get("benchmark_return", pd.Series(0.0, index=x.index)), errors="coerce").fillna(0.0)
    excess = ret - bench
    return {
        "strategy": strategy,
        "start": x["date"].min().date().isoformat(),
        "end": x["date"].max().date().isoformat(),
        "periods": int(len(x)),
        "cumulative_return": float(equity.iloc[-1] - 1.0),
        "CAGR": float(equity.iloc[-1] ** (1.0 / years) - 1.0) if equity.iloc[-1] > 0 else np.nan,
        "ann_vol": ann_vol,
        "Sharpe": float(ret.mean() * 52 / ann_vol) if ann_vol > 0 else np.nan,
        "MDD": mdd,
        "Calmar": float((equity.iloc[-1] ** (1.0 / years) - 1.0) / abs(mdd)) if mdd < 0 and equity.iloc[-1] > 0 else np.nan,
        "hit_positive": float((ret > 0).mean()),
        "hit_excess": float((excess > 0).mean()),
        "avg_return": float(ret.mean()),
        "avg_excess": float(excess.mean()),
    }


def current_allocation() -> pd.DataFrame:
    entry = read_csv(CURRENT_ENTRY, parse_dates=["date"], low_memory=False)
    risk = read_csv(RISK_V4_CURRENT, parse_dates=["date"], low_memory=False)
    safe = read_csv(CURRENT_SAFE, parse_dates=["date"], low_memory=False)
    if entry.empty:
        return pd.DataFrame()
    row = entry.iloc[-1].copy()
    risk_row = risk.copy()
    if not risk_row.empty:
        r1w = risk_row[risk_row["horizon"].eq("1w")].iloc[-1] if not risk_row[risk_row["horizon"].eq("1w")].empty else pd.Series()
        r1m = risk_row[risk_row["horizon"].eq("1m")].iloc[-1] if not risk_row[risk_row["horizon"].eq("1m")].empty else pd.Series()
        row["risk_off_v4_stage_1w"] = r1w.get("risk_off_v4_stage", "Normal")
        row["risk_off_v4_stage_1m"] = r1m.get("risk_off_v4_stage", "Normal")
        row["risk_off_v4_prob_1w"] = r1w.get("risk_off_v4_prob", np.nan)
        row["risk_off_v4_prob_1m"] = r1m.get("risk_off_v4_prob", np.nan)
    row["entry_allowed"] = safe_float(row.get("entry_prob_1w"), 0.0) >= safe_float(row.get("entry_threshold_1w"), 1.0)
    row["ssl2_alert_count"] = 0
    row["ssl2_large_loss_alert"] = False
    risk_weight, stage = decide_weight(row)
    safe_weight_raw = 1.0 - risk_weight
    safe_weight = min(0.60, safe_weight_raw)
    cash_weight = max(0.0, 1.0 - risk_weight - safe_weight)
    safe_top = safe.sort_values("rank").head(3) if "rank" in safe.columns else safe.head(3)
    return pd.DataFrame(
        [
            {
                "date": row.get("date"),
                "risk_stage": stage,
                "entry_prob_1w": row.get("entry_prob_1w"),
                "entry_threshold_1w": row.get("entry_threshold_1w"),
                "risk_weight": risk_weight,
                "safe_weight": safe_weight,
                "cash_weight": cash_weight,
                "risk_assets": row.get("selected_names", row.get("selected")),
                "safe_assets": ",".join(safe_top.get("name", pd.Series(dtype=str)).astype(str).tolist()),
                "risk_off_v4_prob_1w": row.get("risk_off_v4_prob_1w"),
                "risk_off_v4_prob_1m": row.get("risk_off_v4_prob_1m"),
            }
        ]
    )


def write_report(trades: pd.DataFrame, summary: pd.DataFrame, current: pd.DataFrame) -> None:
    lines = [
        "# Dynamic Risk-Gated ETF Allocation Backtest",
        "",
        "Risk-Off Sentinel V4, SSL2 risk heads, EMA entry meta model, ETF leadership risk sleeve, macro-conditioned safe sleeve를 결합한 주간 동적 자산배분 엔진이다.",
        "",
        "## Current Allocation",
        "",
        current.to_markdown(index=False) if not current.empty else "current allocation not available",
        "",
        "## Backtest Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Allocation Rules",
        "",
        "- Normal + entry allowed: risk sleeve up to 60%, safe sleeve 40%.",
        "- Watch: risk sleeve 45% if entry allowed, otherwise 15%.",
        "- De-risk: risk sleeve 20% if entry allowed, otherwise 5%.",
        "- Cash: risk sleeve 5%.",
        "- SSL2 large-loss alert caps risk at 20%; multiple SSL2 alerts cap risk at 25%.",
        "- Top3 sleeve respects 20% product cap, so each sleeve can deploy at most 60%; residual is cash.",
        "",
        "## Recent Trades",
        "",
        trades.tail(20).to_markdown(index=False),
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    panel = combine_panel()
    if panel.empty:
        raise RuntimeError("allocation panel is empty")
    trades, summary = backtest(panel)
    current = current_allocation()
    panel.to_csv(TABLES / "dynamic_allocation_panel.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(TABLES / "dynamic_allocation_trades.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(TABLES / "dynamic_allocation_summary.csv", index=False, encoding="utf-8-sig")
    current.to_csv(TABLES / "current_dynamic_allocation.csv", index=False, encoding="utf-8-sig")
    write_report(trades, summary, current)
    print(f"saved {OUT}")
    print(summary.to_string(index=False))
    print("\nCurrent")
    print(current.to_string(index=False))


if __name__ == "__main__":
    main()
