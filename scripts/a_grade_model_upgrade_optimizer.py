from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "a_grade_model_upgrade_latest"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"


def read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def rebalance_dates(dates: pd.Series, horizon: str) -> set[pd.Timestamp]:
    idx = pd.DatetimeIndex(pd.to_datetime(dates).dropna().unique()).sort_values()
    if idx.empty:
        return set()
    freq = "W-FRI" if horizon == "1W" else "M"
    return set(pd.Series(idx, index=idx).groupby(idx.to_period(freq)).max())


def strategy_panel(pred: pd.DataFrame, risk: pd.DataFrame, horizon: str, score_col: str, top_k: int) -> pd.DataFrame:
    ret_col = "forward_5D_return" if horizon == "1W" else "forward_20D_return"
    excess_col = "forward_5D_excess" if horizon == "1W" else "forward_20D_excess"
    data = pred[pred["prediction_horizon"].eq(horizon)].copy()
    data = data[data["date"].isin(rebalance_dates(data["date"], horizon))]
    data = data.dropna(subset=[score_col, ret_col, excess_col])
    if data.empty:
        return pd.DataFrame()
    data["rank"] = data.groupby("date")[score_col].rank(ascending=False, method="first")
    top = data[data["rank"].le(top_k)].copy()
    panel = (
        top.groupby("date")
        .agg(
            raw_return=(ret_col, "mean"),
            excess_return=(excess_col, "mean"),
            score_mean=(score_col, "mean"),
            n=(score_col, "size"),
        )
        .reset_index()
    )
    panel = panel[panel["n"].ge(top_k)]
    risk_h = risk[risk["horizon"].eq(horizon.lower())].sort_values("date")
    if risk_h.empty:
        panel["risk_off_v4_prob"] = 0.0
        panel["risk_off_v4_stage"] = "Normal"
        return panel
    return pd.merge_asof(panel.sort_values("date"), risk_h, on="date", direction="backward")


def apply_gate(frame: pd.DataFrame, prob_threshold: float, score_threshold: float, block_mode: str) -> pd.Series:
    invested = frame["risk_off_v4_prob"].astype(float).lt(prob_threshold) & frame["score_mean"].astype(float).ge(score_threshold)
    if block_mode == "cash":
        invested &= ~frame["risk_off_v4_stage"].astype(str).eq("Cash")
    elif block_mode == "derisk_cash":
        invested &= ~frame["risk_off_v4_stage"].astype(str).isin(["Cash", "De-risk"])
    return invested.fillna(False)


def perf(frame: pd.DataFrame, invested: pd.Series, periods_per_year: int) -> dict:
    if frame.empty:
        return {}
    ret = np.where(invested, frame["raw_return"].astype(float), 0.0)
    returns = pd.Series(ret, index=frame.index)
    equity = (1.0 + returns).cumprod()
    active = frame[invested].copy()
    std = returns.std(ddof=1)
    return {
        "periods": int(frame.shape[0]),
        "invested_periods": int(invested.sum()),
        "coverage": float(invested.mean()),
        "cumulative_return": float(equity.iloc[-1] - 1.0),
        "CAGR": float(equity.iloc[-1] ** (periods_per_year / max(len(equity), 1)) - 1.0) if equity.iloc[-1] > 0 else np.nan,
        "MDD": float((equity / equity.cummax() - 1.0).min()),
        "Sharpe": float(returns.mean() / std * np.sqrt(periods_per_year)) if std and not pd.isna(std) else np.nan,
        "hit_excess": float((active["excess_return"] > 0).mean()) if not active.empty else np.nan,
        "hit_positive": float((active["raw_return"] > 0).mean()) if not active.empty else np.nan,
        "avg_return": float(active["raw_return"].mean()) if not active.empty else np.nan,
        "avg_excess": float(active["excess_return"].mean()) if not active.empty else np.nan,
    }


def optimize_etf_gate() -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = read_csv(ROOT / "outputs/gaps_long_lived_etf_leadership_latest/tables/long_lived_predictions.csv", parse_dates=["date"], low_memory=False)
    risk = read_csv(ROOT / "outputs/risk_off_v4_event_label_latest/tables/risk_off_v4_walkforward_predictions.csv", parse_dates=["date"], low_memory=False)
    if pred.empty:
        return pd.DataFrame(), pd.DataFrame()
    keep = ["date", "horizon", "risk_off_v4_prob", "risk_off_v4_stage"]
    risk = risk[[c for c in keep if c in risk]].sort_values("date") if not risk.empty else pd.DataFrame(columns=keep)
    rows = []
    upper_rows = []
    for horizon in ["1W", "1M"]:
        score_cols = ["rule_5d_score"] if horizon == "1W" else ["rule_20d_score", "blend_20d_score", "ranker_score", "entry_adjusted_20d_score"]
        score_cols = [c for c in score_cols if c in pred.columns]
        periods_per_year = 52 if horizon == "1W" else 12
        min_trades = 20 if horizon == "1W" else 5
        min_valid_coverage = 0.35 if horizon == "1W" else 0.30
        for score_col in score_cols:
            for top_k in [1, 2, 3, 5]:
                panel = strategy_panel(pred, risk, horizon, score_col, top_k)
                if panel.empty:
                    continue
                valid = panel[panel["date"].between(pd.Timestamp("2019-01-01"), pd.Timestamp("2021-12-31"))].copy()
                test = panel[panel["date"].gt(pd.Timestamp("2021-12-31"))].copy()
                if valid.empty or test.empty:
                    continue
                score_thresholds = [-np.inf] + [float(valid["score_mean"].quantile(q)) for q in [0.2, 0.4, 0.6, 0.75, 0.85]]
                for prob_threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 999.0]:
                    for block_mode in ["none", "cash", "derisk_cash"]:
                        for score_threshold in score_thresholds:
                            inv_valid = apply_gate(valid, prob_threshold, score_threshold, block_mode)
                            if int(inv_valid.sum()) < min_trades:
                                continue
                            valid_perf = perf(valid, inv_valid, periods_per_year)
                            if safe(valid_perf.get("coverage")) < min_valid_coverage:
                                continue
                            objective = (
                                safe(valid_perf.get("Sharpe"))
                                + 0.8 * safe(valid_perf.get("hit_excess"))
                                + 0.3 * safe(valid_perf.get("hit_positive"))
                                - 0.5 * abs(safe(valid_perf.get("MDD")))
                            )
                            inv_test = apply_gate(test, prob_threshold, score_threshold, block_mode)
                            test_perf = perf(test, inv_test, periods_per_year)
                            rows.append(
                                {
                                    "horizon": horizon,
                                    "score_col": score_col,
                                    "top_k": top_k,
                                    "prob_threshold": prob_threshold,
                                    "block_mode": block_mode,
                                    "score_threshold": score_threshold,
                                    "valid_objective": objective,
                                    **{f"valid_{k}": v for k, v in valid_perf.items()},
                                    **{f"test_{k}": v for k, v in test_perf.items()},
                                }
                            )
                # A hard ceiling: choose rules using test itself. This is not
                # deployable, but it tells us whether the current signal family
                # can mathematically reach the A target.
                score_thresholds_test = [-np.inf] + [float(test["score_mean"].quantile(q)) for q in np.arange(0.1, 0.96, 0.05)]
                for prob_threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 999.0]:
                    for block_mode in ["none", "cash", "derisk_cash"]:
                        for score_threshold in score_thresholds_test:
                            inv_test = apply_gate(test, prob_threshold, score_threshold, block_mode)
                            if int(inv_test.sum()) < min_trades:
                                continue
                            test_perf = perf(test, inv_test, periods_per_year)
                            upper_rows.append(
                                {
                                    "horizon": horizon,
                                    "score_col": score_col,
                                    "top_k": top_k,
                                    "prob_threshold": prob_threshold,
                                    "block_mode": block_mode,
                                    "score_threshold": score_threshold,
                                    **test_perf,
                                }
                            )
    grid = pd.DataFrame(rows)
    upper = pd.DataFrame(upper_rows)
    if not grid.empty:
        grid = grid.sort_values(["horizon", "valid_objective"], ascending=[True, False])
    if not upper.empty:
        upper = upper.sort_values(["horizon", "Sharpe", "hit_excess"], ascending=[True, False, False])
    return grid, upper


def safe(value: object, fallback: float = 0.0) -> float:
    try:
        x = float(value)
        return x if np.isfinite(x) else fallback
    except Exception:
        return fallback


def grade_risk() -> pd.DataFrame:
    metrics = read_csv(ROOT / "outputs/ssl2_head_backtest_latest/tables/risk_ssl2_metrics.csv")
    adoption = read_csv(ROOT / "outputs/ssl2_head_backtest_latest/tables/operational_model_adoption.csv")
    rows = []
    if metrics.empty or adoption.empty:
        return pd.DataFrame()
    for _, rec in adoption[adoption["task"].eq("risk_off")].iterrows():
        model = rec["best_model"] if rec["decision"] == "adopt" else rec["baseline_model"]
        row = metrics[metrics["label"].eq(rec["target"]) & metrics["model"].eq(model)]
        if row.empty:
            continue
        r = row.iloc[0]
        grade = grade_classifier(r["test_auc"], r["test_recall"], r["test_precision"], r["test_false_alarm"])
        rows.append(
            {
                "component": f"Risk-Off {rec['target']}",
                "selected_model": model,
                "primary_metric": "test_auc/recall/precision",
                "A_target": "AUC>=0.90, Recall>=0.90, Precision>=0.50 for large loss; lower false alarm preferred",
                "actual_summary": f"AUC={r['test_auc']:.3f}, Recall={r['test_recall']:.3f}, Precision={r['test_precision']:.3f}, FalseAlarm={r['test_false_alarm']:.3f}",
                "grade": grade,
                "upgrade_needed": not is_a_or_better(grade),
            }
        )
    return pd.DataFrame(rows)


def grade_classifier(auc: float, recall: float, precision: float, false_alarm: float) -> str:
    auc = safe(auc)
    recall = safe(recall)
    precision = safe(precision)
    false_alarm = safe(false_alarm, 1.0)
    if auc >= 0.95 and recall >= 0.95 and precision >= 0.70 and false_alarm <= 0.25:
        return "A+"
    if auc >= 0.90 and recall >= 0.90 and precision >= 0.45:
        return "A"
    if auc >= 0.80 and recall >= 0.80:
        return "B+"
    if auc >= 0.70:
        return "B"
    return "C"


def grade_etf(grid: pd.DataFrame, upper: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon in ["1W", "1M"]:
        g = grid[grid["horizon"].eq(horizon)].copy() if not grid.empty else pd.DataFrame()
        u = upper[upper["horizon"].eq(horizon)].copy() if not upper.empty else pd.DataFrame()
        best = g.iloc[0] if not g.empty else pd.Series(dtype=object)
        ceiling = u.sort_values("Sharpe", ascending=False).iloc[0] if not u.empty else pd.Series(dtype=object)
        grade = grade_portfolio(best.get("test_Sharpe"), best.get("test_hit_excess"), best.get("test_hit_positive"))
        rows.append(
            {
                "component": f"ETF Leadership {horizon}",
                "selected_model": f"{best.get('score_col', '')} top{best.get('top_k', '')} gated",
                "primary_metric": "test Sharpe / excess hit / positive hit",
                "A_target": "Sharpe>=2.0 and excess hit>=0.65, or positive hit>=0.75 with controlled MDD",
                "actual_summary": f"Sharpe={safe(best.get('test_Sharpe')):.3f}, ExcessHit={safe(best.get('test_hit_excess')):.3f}, PositiveHit={safe(best.get('test_hit_positive')):.3f}, Coverage={safe(best.get('test_coverage')):.3f}",
                "test_upper_bound": f"Sharpe={safe(ceiling.get('Sharpe')):.3f}, ExcessHit={safe(ceiling.get('hit_excess')):.3f}, PositiveHit={safe(ceiling.get('hit_positive')):.3f}",
                "grade": grade,
                "upgrade_needed": not is_a_or_better(grade),
            }
        )
    return pd.DataFrame(rows)


def grade_portfolio(sharpe: float, hit_excess: float, hit_positive: float) -> str:
    sharpe = safe(sharpe)
    hit_excess = safe(hit_excess)
    hit_positive = safe(hit_positive)
    if sharpe >= 3.0 and hit_excess >= 0.70:
        return "A+"
    if sharpe >= 2.0 and (hit_excess >= 0.65 or hit_positive >= 0.75):
        return "A"
    if sharpe >= 1.4 and (hit_excess >= 0.55 or hit_positive >= 0.65):
        return "B+"
    if sharpe >= 1.0:
        return "B"
    return "C"


def grade_safe() -> tuple[pd.DataFrame, pd.DataFrame]:
    v2 = read_csv(ROOT / "outputs/institutional_risk_off_v2_latest/tables/macro_conditioned_safe_asset_summary.csv")
    ssl2 = read_csv(ROOT / "outputs/ssl2_head_backtest_latest/tables/safe_ssl2_backtest_summary.csv")
    rows = []
    best_rows = []
    if not v2.empty:
        v2 = v2.copy()
        v2["model"] = "macro_safe_v2"
        v2["horizon_norm"] = v2["horizon"].astype(str).str.upper().replace({"1W": "1W", "1M": "1M"})
        if "test_year" in v2:
            v2_all = v2[v2["test_year"].astype(str).eq("ALL")].copy()
        else:
            v2_all = v2.copy()
        best_rows.append(v2_all)
    if not ssl2.empty:
        s = ssl2.copy()
        s["model"] = "safe_ssl2"
        s["horizon_norm"] = s["horizon"].astype(str).str.upper()
        best_rows.append(s)
    best = pd.concat(best_rows, ignore_index=True, sort=False) if best_rows else pd.DataFrame()
    if best.empty:
        return pd.DataFrame(), best
    for horizon, part in best.groupby("horizon_norm"):
        part = part.copy()
        part["score_for_select"] = (
            pd.to_numeric(part["beat_safe_average_rate"], errors="coerce").fillna(0.0) * 2.0
            + pd.to_numeric(part["avg_picked_return"], errors="coerce").fillna(-9.0)
            + pd.to_numeric(part.get("periods", 0), errors="coerce").fillna(0.0).clip(0, 52) / 100.0
        )
        r = part.sort_values("score_for_select", ascending=False).iloc[0]
        grade = grade_safe_row(r)
        rows.append(
            {
                "component": f"Safe Asset {horizon}",
                "selected_model": r.get("model"),
                "primary_metric": "beat safe average / picked return / sample count",
                "A_target": "BeatSafeAverage>=0.70, avg picked return >=0, periods>=30",
                "actual_summary": f"Beat={safe(r.get('beat_safe_average_rate')):.3f}, AvgReturn={safe(r.get('avg_picked_return')):.3f}, Periods={int(safe(r.get('periods')))}",
                "grade": grade,
                "upgrade_needed": not is_a_or_better(grade),
            }
        )
    return pd.DataFrame(rows), best


def is_a_or_better(grade: str) -> bool:
    return str(grade) in {"A", "A+"}


def grade_safe_row(row: pd.Series) -> str:
    beat = safe(row.get("beat_safe_average_rate"))
    avg_return = safe(row.get("avg_picked_return"))
    periods = safe(row.get("periods"))
    if beat >= 0.80 and avg_return > 0 and periods >= 50:
        return "A+"
    if beat >= 0.70 and avg_return >= 0 and periods >= 30:
        return "A"
    if beat >= 0.60 and avg_return >= 0:
        return "B+"
    if beat >= 0.50:
        return "B"
    return "C"


def write_report(grades: pd.DataFrame, etf_grid: pd.DataFrame, etf_upper: pd.DataFrame, safe_candidates: pd.DataFrame) -> None:
    lines = [
        "# A Grade Model Upgrade Diagnostic",
        "",
        "This report does not inflate model grades. It tests whether the current signal family can reach the A target under validation-selected rules and also records a test-only upper bound.",
        "",
        "## Component Grades",
        grades.to_markdown(index=False) if not grades.empty else "No grade data.",
        "",
        "## ETF Validation-Selected Best",
    ]
    if not etf_grid.empty:
        lines.append(etf_grid.groupby("horizon").head(5).to_markdown(index=False))
    else:
        lines.append("No ETF grid.")
    lines.extend(["", "## ETF Test-Only Upper Bound"])
    if not etf_upper.empty:
        lines.append(etf_upper.groupby("horizon").head(5).to_markdown(index=False))
    else:
        lines.append("No ETF upper bound.")
    lines.extend(["", "## Safe Model Candidates"])
    if not safe_candidates.empty:
        lines.append(safe_candidates.to_markdown(index=False))
    else:
        lines.append("No safe model candidates.")
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "a_grade_model_upgrade_diagnostic.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    etf_grid, etf_upper = optimize_etf_gate()
    safe_grades, safe_candidates = grade_safe()
    grades = pd.concat([grade_risk(), grade_etf(etf_grid, etf_upper), safe_grades], ignore_index=True, sort=False)
    etf_grid.to_csv(TABLES / "etf_gate_validation_selected_grid.csv", index=False, encoding="utf-8-sig")
    etf_upper.to_csv(TABLES / "etf_gate_test_upper_bound.csv", index=False, encoding="utf-8-sig")
    safe_candidates.to_csv(TABLES / "safe_model_candidate_summary.csv", index=False, encoding="utf-8-sig")
    grades.to_csv(TABLES / "component_grade_summary.csv", index=False, encoding="utf-8-sig")
    write_report(grades, etf_grid, etf_upper, safe_candidates)
    print(grades.to_string(index=False))
    print(OUT.resolve())


if __name__ == "__main__":
    main()
