from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "hybrid_proxy_gaps_etf_leadership_latest"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"

PROXY_TRADES = ROOT / "outputs" / "long_history_proxy_selection_backtest_latest" / "tables" / "proxy_selection_backtest_trades.csv"
PROXY_SUMMARY = ROOT / "outputs" / "long_history_proxy_selection_backtest_latest" / "tables" / "proxy_selection_backtest_summary.csv"
PROXY_IC = ROOT / "outputs" / "long_history_proxy_selection_backtest_latest" / "tables" / "proxy_selection_rank_ic.csv"
GAPS_TRADES = ROOT / "outputs" / "etf_leadership_static_v4_repaired" / "v3_backtest_trades.csv"
GAPS_SUMMARY = ROOT / "outputs" / "etf_leadership_static_v4_repaired" / "v3_backtest_summary.csv"
RISK_V4 = ROOT / "outputs" / "risk_off_v4_event_label_latest" / "tables" / "risk_off_v4_walkforward_predictions.csv"


STRATEGY_MAP = [
    {
        "hybrid_strategy": "hybrid_rule_1w_top1",
        "horizon": "1W",
        "proxy_strategy": "leader_rule_weekly_top1",
        "gaps_score_col": "rule_5d_score",
        "top_k": 1,
        "description": "2010~2024: long-listed ETF price leadership rule, 2025+: DB GAPS constituent leadership rule",
    },
    {
        "hybrid_strategy": "hybrid_rule_1w_top3",
        "horizon": "1W",
        "proxy_strategy": "leader_rule_weekly_top3",
        "gaps_score_col": "rule_5d_score",
        "top_k": 3,
        "description": "Weekly Top3 rule bridge",
    },
    {
        "hybrid_strategy": "hybrid_ranker_1w_top3",
        "horizon": "1W",
        "proxy_strategy": "leader_ranker_weekly_top3",
        "gaps_score_col": "rule_5d_score",
        "top_k": 3,
        "description": "Proxy weekly ranker, DB GAPS weekly rule because short-horizon ranker is intentionally not deployed",
    },
    {
        "hybrid_strategy": "hybrid_rule_1m_top3",
        "horizon": "1M",
        "proxy_strategy": "leader_rule_monthly_top3",
        "gaps_score_col": "rule_20d_score",
        "top_k": 3,
        "description": "Monthly Top3 rule bridge",
    },
    {
        "hybrid_strategy": "hybrid_rule_1m_top5",
        "horizon": "1M",
        "proxy_strategy": "leader_rule_monthly_top5",
        "gaps_score_col": "rule_20d_score",
        "top_k": 5,
        "description": "Monthly Top5 rule bridge",
    },
    {
        "hybrid_strategy": "hybrid_ranker_1m_top2",
        "horizon": "1M",
        "proxy_strategy": "leader_ranker_monthly_top3",
        "gaps_score_col": "ranker_score",
        "top_k": 2,
        "description": "2016~2024 proxy ranker, 2025+ DB GAPS constituent ranker",
    },
    {
        "hybrid_strategy": "hybrid_ranker_1m_top5",
        "horizon": "1M",
        "proxy_strategy": "leader_ranker_monthly_top5",
        "gaps_score_col": "ranker_score",
        "top_k": 5,
        "description": "Monthly Top5 ranker bridge",
    },
]


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def normalize_proxy_trades(raw: pd.DataFrame, spec: dict) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    x = raw[raw["strategy"].eq(spec["proxy_strategy"])].copy()
    x = x[pd.to_datetime(x["date"]).lt(pd.Timestamp("2025-01-01"))]
    if x.empty:
        return pd.DataFrame()
    x["date"] = pd.to_datetime(x["date"])
    x["exit_date"] = pd.to_datetime(x["exit_date"])
    out = pd.DataFrame(
        {
            "date": x["date"],
            "exit_date": x["exit_date"],
            "hybrid_strategy": spec["hybrid_strategy"],
            "phase": "2010-2024 long-listed ETF proxy",
            "horizon": spec["horizon"],
            "top_k": spec["top_k"],
            "portfolio_return": pd.to_numeric(x["period_return"], errors="coerce"),
            "benchmark_return": pd.to_numeric(x["benchmark_return"], errors="coerce"),
            "excess_return": pd.to_numeric(x["excess_return"], errors="coerce"),
            "selected": x["selected"].astype(str),
            "selected_names": x["selected"].astype(str),
            "score_col": spec["proxy_strategy"],
        }
    )
    return out


def normalize_gaps_trades(raw: pd.DataFrame, spec: dict) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    x = raw[
        raw["horizon"].eq(spec["horizon"])
        & raw["score_col"].eq(spec["gaps_score_col"])
        & pd.to_numeric(raw["top_k"], errors="coerce").eq(spec["top_k"])
    ].copy()
    x = x[pd.to_datetime(x["date"]).ge(pd.Timestamp("2025-01-01"))]
    if x.empty:
        return pd.DataFrame()
    x["date"] = pd.to_datetime(x["date"])
    offset = pd.Timedelta(days=7 if spec["horizon"] == "1W" else 31)
    out = pd.DataFrame(
        {
            "date": x["date"],
            "exit_date": x["date"] + offset,
            "hybrid_strategy": spec["hybrid_strategy"],
            "phase": "2025-current DB GAPS full universe",
            "horizon": spec["horizon"],
            "top_k": spec["top_k"],
            "portfolio_return": pd.to_numeric(x["portfolio_return"], errors="coerce"),
            "benchmark_return": pd.to_numeric(x["benchmark_return"], errors="coerce"),
            "excess_return": pd.to_numeric(x["excess_return"], errors="coerce"),
            "selected": x["selected"].astype(str),
            "selected_names": x.get("selected_names", x["selected"]).astype(str),
            "score_col": spec["gaps_score_col"],
        }
    )
    return out


def summarize(trades: pd.DataFrame, strategy: str, phase: str | None = None) -> dict:
    x = trades[trades["hybrid_strategy"].eq(strategy)].copy()
    if phase is not None:
        x = x[x["phase"].eq(phase)].copy()
    if x.empty:
        return {"hybrid_strategy": strategy, "phase": phase or "all", "periods": 0}
    x = x.sort_values("date")
    ret = x["portfolio_return"].fillna(0.0).astype(float)
    equity = (1.0 + ret).cumprod()
    years = max((x["exit_date"].max() - x["date"].min()).days / 365.25, 1e-9)
    per_year = 52 if str(x["horizon"].iloc[0]) == "1W" else 12
    ann_vol = float(ret.std(ddof=1) * np.sqrt(per_year))
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if equity.iloc[-1] > 0 else np.nan
    mdd = float((equity / equity.cummax() - 1.0).min())
    sharpe = float(ret.mean() * per_year / ann_vol) if ann_vol > 0 else np.nan
    excess = x["excess_return"].astype(float)
    return {
        "hybrid_strategy": strategy,
        "phase": phase or "all",
        "horizon": x["horizon"].iloc[0],
        "top_k": int(x["top_k"].iloc[0]),
        "start": x["date"].min().date().isoformat(),
        "end": x["date"].max().date().isoformat(),
        "periods": int(len(x)),
        "cumulative_return": float(equity.iloc[-1] - 1.0),
        "CAGR": cagr,
        "ann_vol": ann_vol,
        "Sharpe": sharpe,
        "MDD": mdd,
        "Calmar": float(cagr / abs(mdd)) if mdd < 0 else np.nan,
        "hit_positive": float((ret > 0).mean()),
        "hit_excess": float((excess > 0).mean()),
        "avg_return": float(ret.mean()),
        "avg_excess": float(excess.mean()),
    }


def grade(row: pd.Series) -> str:
    sharpe = float(row.get("Sharpe", np.nan))
    hit_excess = float(row.get("hit_excess", np.nan))
    hit_positive = float(row.get("hit_positive", np.nan))
    mdd = float(row.get("MDD", np.nan))
    if sharpe >= 3.0 and hit_excess >= 0.70 and mdd >= -0.12:
        return "A+"
    if sharpe >= 2.0 and (hit_excess >= 0.65 or hit_positive >= 0.75) and mdd >= -0.18:
        return "A"
    if sharpe >= 1.4 and (hit_excess >= 0.55 or hit_positive >= 0.65):
        return "B+"
    if sharpe >= 0.8 and (hit_excess >= 0.50 or hit_positive >= 0.58):
        return "B"
    return "C"


def build_diagnostics(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary[summary["phase"].eq("all")].iterrows():
        phase_rows = summary[
            summary["hybrid_strategy"].eq(row["hybrid_strategy"])
            & summary["phase"].ne("all")
        ].copy()
        proxy = phase_rows[phase_rows["phase"].str.contains("proxy", na=False)]
        gaps = phase_rows[phase_rows["phase"].str.contains("GAPS", na=False)]
        rows.append(
            {
                "hybrid_strategy": row["hybrid_strategy"],
                "horizon": row["horizon"],
                "grade": row["grade"],
                "all_Sharpe": row["Sharpe"],
                "all_hit_excess": row["hit_excess"],
                "all_hit_positive": row["hit_positive"],
                "all_MDD": row["MDD"],
                "proxy_Sharpe": proxy["Sharpe"].iloc[0] if not proxy.empty else np.nan,
                "proxy_hit_excess": proxy["hit_excess"].iloc[0] if not proxy.empty else np.nan,
                "gaps_Sharpe": gaps["Sharpe"].iloc[0] if not gaps.empty else np.nan,
                "gaps_hit_excess": gaps["hit_excess"].iloc[0] if not gaps.empty else np.nan,
                "diagnosis": diagnose(row, proxy.iloc[0] if not proxy.empty else None, gaps.iloc[0] if not gaps.empty else None),
            }
        )
    return pd.DataFrame(rows)


def merge_risk(trades: pd.DataFrame, risk: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or risk.empty:
        out = trades.copy()
        out["risk_off_v4_prob"] = 0.0
        out["risk_off_v4_stage"] = "Normal"
        return out
    pieces = []
    r = risk.copy()
    r["date"] = pd.to_datetime(r["date"])
    r["horizon"] = r["horizon"].str.upper()
    keep = ["date", "horizon", "risk_off_v4_prob", "risk_off_v4_stage", "risk_3d_dominant_axis"]
    r = r[[c for c in keep if c in r.columns]].sort_values("date")
    for horizon, part in trades.copy().groupby("horizon"):
        left = part.sort_values("date")
        right = r[r["horizon"].eq(str(horizon).upper())].sort_values("date")
        if right.empty:
            left["risk_off_v4_prob"] = 0.0
            left["risk_off_v4_stage"] = "Normal"
            pieces.append(left)
            continue
        right = right.drop(columns=["horizon"], errors="ignore")
        merged = pd.merge_asof(left, right, on="date", direction="backward")
        pieces.append(merged)
    return pd.concat(pieces, ignore_index=True).sort_values(["hybrid_strategy", "date"])


def apply_risk_gate(frame: pd.DataFrame, prob_threshold: float, block_mode: str) -> pd.Series:
    prob = pd.to_numeric(frame["risk_off_v4_prob"], errors="coerce").fillna(0.0)
    stage = frame["risk_off_v4_stage"].astype(str)
    invested = prob.lt(prob_threshold)
    if block_mode == "cash":
        invested &= ~stage.eq("Cash")
    elif block_mode == "derisk_cash":
        invested &= ~stage.isin(["De-risk", "Cash"])
    return invested.fillna(False)


def summarize_gated(frame: pd.DataFrame, invested: pd.Series, strategy: str, label: str) -> dict:
    if frame.empty:
        return {"hybrid_strategy": strategy, "gate_label": label, "periods": 0}
    x = frame.sort_values("date").copy()
    inv = invested.reindex(x.index).fillna(False)
    ret = pd.Series(np.where(inv, pd.to_numeric(x["portfolio_return"], errors="coerce").fillna(0.0), 0.0), index=x.index)
    excess_active = pd.to_numeric(x.loc[inv, "excess_return"], errors="coerce")
    raw_active = pd.to_numeric(x.loc[inv, "portfolio_return"], errors="coerce")
    equity = (1.0 + ret).cumprod()
    years = max((x["exit_date"].max() - x["date"].min()).days / 365.25, 1e-9)
    per_year = 52 if str(x["horizon"].iloc[0]) == "1W" else 12
    ann_vol = float(ret.std(ddof=1) * np.sqrt(per_year))
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if equity.iloc[-1] > 0 else np.nan
    mdd = float((equity / equity.cummax() - 1.0).min())
    return {
        "hybrid_strategy": strategy,
        "gate_label": label,
        "horizon": x["horizon"].iloc[0],
        "top_k": int(x["top_k"].iloc[0]),
        "start": x["date"].min().date().isoformat(),
        "end": x["date"].max().date().isoformat(),
        "periods": int(len(x)),
        "invested_periods": int(inv.sum()),
        "coverage": float(inv.mean()),
        "cumulative_return": float(equity.iloc[-1] - 1.0),
        "CAGR": cagr,
        "ann_vol": ann_vol,
        "Sharpe": float(ret.mean() * per_year / ann_vol) if ann_vol > 0 else np.nan,
        "MDD": mdd,
        "Calmar": float(cagr / abs(mdd)) if mdd < 0 else np.nan,
        "hit_positive": float((raw_active > 0).mean()) if not raw_active.empty else np.nan,
        "hit_excess": float((excess_active > 0).mean()) if not excess_active.empty else np.nan,
        "avg_return": float(raw_active.mean()) if not raw_active.empty else np.nan,
        "avg_excess": float(excess_active.mean()) if not excess_active.empty else np.nan,
    }


def optimize_risk_gates(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    full_rows = []
    for strategy, part in trades.groupby("hybrid_strategy"):
        part = part.sort_values("date").copy()
        horizon = str(part["horizon"].iloc[0])
        min_trades = 40 if horizon == "1W" else 10
        valid = part[part["date"].lt(pd.Timestamp("2020-01-01"))].copy()
        test = part[part["date"].ge(pd.Timestamp("2020-01-01"))].copy()
        if valid.empty or test.empty:
            continue
        candidates = []
        for prob_threshold in [0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 999.0]:
            for block_mode in ["none", "cash", "derisk_cash"]:
                inv_valid = apply_risk_gate(valid, prob_threshold, block_mode)
                if int(inv_valid.sum()) < min_trades or float(inv_valid.mean()) < 0.35:
                    continue
                valid_perf = summarize_gated(valid, inv_valid, strategy, "valid")
                objective = (
                    _safe(valid_perf.get("Sharpe"))
                    + 0.6 * _safe(valid_perf.get("hit_positive"))
                    + 0.4 * _safe(valid_perf.get("hit_excess"))
                    - 0.6 * abs(_safe(valid_perf.get("MDD")))
                )
                candidates.append((objective, prob_threshold, block_mode, valid_perf))
        if not candidates:
            continue
        candidates.sort(key=lambda x: x[0], reverse=True)
        objective, prob_threshold, block_mode, valid_perf = candidates[0]
        inv_test = apply_risk_gate(test, prob_threshold, block_mode)
        inv_all = apply_risk_gate(part, prob_threshold, block_mode)
        test_perf = summarize_gated(test, inv_test, strategy, "test")
        all_perf = summarize_gated(part, inv_all, strategy, "all")
        rows.append(
            {
                "hybrid_strategy": strategy,
                "horizon": horizon,
                "selected_prob_threshold": prob_threshold,
                "selected_block_mode": block_mode,
                "valid_objective": objective,
                **{f"valid_{k}": v for k, v in valid_perf.items() if k not in {"hybrid_strategy", "gate_label", "horizon", "top_k"}},
                **{f"test_{k}": v for k, v in test_perf.items() if k not in {"hybrid_strategy", "gate_label", "horizon", "top_k"}},
                **{f"all_{k}": v for k, v in all_perf.items() if k not in {"hybrid_strategy", "gate_label", "horizon", "top_k"}},
            }
        )
        all_perf["grade"] = grade(pd.Series(all_perf))
        all_perf["selected_prob_threshold"] = prob_threshold
        all_perf["selected_block_mode"] = block_mode
        full_rows.append(all_perf)
    grid = pd.DataFrame(rows)
    selected = pd.DataFrame(full_rows)
    if not selected.empty:
        selected = selected.sort_values(["horizon", "Sharpe"], ascending=[True, False])
    return grid, selected


def _safe(value: object, fallback: float = 0.0) -> float:
    try:
        x = float(value)
        return x if np.isfinite(x) else fallback
    except Exception:
        return fallback


def diagnose(row: pd.Series, proxy: pd.Series | None, gaps: pd.Series | None) -> str:
    reasons = []
    if float(row.get("hit_excess", 0.0)) < 0.55:
        reasons.append("초과수익 적중률이 낮아 ETF 간 순위 신호가 약함")
    if float(row.get("MDD", 0.0)) < -0.25:
        reasons.append("리스크오프/고점권 필터 없이 항상 투자해 MDD가 큼")
    if proxy is not None and float(proxy.get("hit_excess", 0.0)) < 0.50:
        reasons.append("2010~2024 장기 프록시 구간은 가격 모멘텀만으로는 벤치마크 초과가 어려움")
    if gaps is not None and float(gaps.get("hit_excess", 0.0)) >= 0.55:
        reasons.append("2025년 이후 DB GAPS 구성종목 모델은 단기 표본이 작지만 작동 신호가 있음")
    if not reasons:
        reasons.append("A등급 전에는 검증 표본 확대와 진입 게이트 고정이 필요")
    return "; ".join(reasons)


def write_report(
    summary: pd.DataFrame,
    diagnostic: pd.DataFrame,
    gated: pd.DataFrame,
    proxy_ic: pd.DataFrame,
    proxy_summary: pd.DataFrame,
    gaps_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Hybrid Proxy + DB GAPS ETF Leadership Backtest",
        "",
        "이 검증은 2010~2024년에는 DB GAPS에 없더라도 오래 상장된 글로벌 ETF 프록시를 사용하고, 2025년 이후에는 DB GAPS ETF 구성종목 기반 리더십 모델을 사용한다.",
        "목적은 과거 ETF 생존 기간 문제를 피하면서도 2025년 이후 실제 대회 유니버스 검증을 분리해서 보는 것이다.",
        "",
        "## Hybrid Summary",
        "",
        summary[summary["phase"].eq("all")].to_markdown(index=False),
        "",
        "## Phase Summary",
        "",
        summary[summary["phase"].ne("all")].to_markdown(index=False),
        "",
        "## Diagnostics",
        "",
        diagnostic.to_markdown(index=False),
        "",
        "## Validation-Selected Risk Gate",
        "",
        gated.to_markdown(index=False) if not gated.empty else "risk gate result not available",
        "",
        "## Proxy Rank IC",
        "",
        proxy_ic.to_markdown(index=False) if not proxy_ic.empty else "proxy rank IC not found",
        "",
        "## Existing Proxy Summary",
        "",
        proxy_summary.head(20).to_markdown(index=False) if not proxy_summary.empty else "proxy summary not found",
        "",
        "## Existing DB GAPS 2025+ Summary",
        "",
        gaps_summary.head(20).to_markdown(index=False) if not gaps_summary.empty else "GAPS summary not found",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    proxy_trades = read_csv(PROXY_TRADES)
    gaps_trades = read_csv(GAPS_TRADES)
    proxy_summary = read_csv(PROXY_SUMMARY)
    gaps_summary = read_csv(GAPS_SUMMARY)
    proxy_ic = read_csv(PROXY_IC)
    risk = read_csv(RISK_V4)
    if proxy_trades.empty:
        raise FileNotFoundError(PROXY_TRADES)
    if gaps_trades.empty:
        raise FileNotFoundError(GAPS_TRADES)

    pieces = []
    for spec in STRATEGY_MAP:
        pieces.append(normalize_proxy_trades(proxy_trades, spec))
        pieces.append(normalize_gaps_trades(gaps_trades, spec))
    trades = pd.concat([p for p in pieces if not p.empty], ignore_index=True).sort_values(["hybrid_strategy", "date"])
    trades = merge_risk(trades, risk)

    rows = []
    for strategy in trades["hybrid_strategy"].dropna().unique():
        rows.append(summarize(trades, strategy))
        for phase in trades.loc[trades["hybrid_strategy"].eq(strategy), "phase"].dropna().unique():
            rows.append(summarize(trades, strategy, phase))
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["grade"] = summary.apply(lambda r: grade(r) if r["phase"] == "all" else "", axis=1)
        summary = summary.sort_values(["horizon", "phase", "Sharpe"], ascending=[True, True, False])
    diagnostic = build_diagnostics(summary)
    gate_grid, gated_selected = optimize_risk_gates(trades)

    trades.to_csv(TABLES / "hybrid_proxy_gaps_trades.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(TABLES / "hybrid_proxy_gaps_summary.csv", index=False, encoding="utf-8-sig")
    diagnostic.to_csv(TABLES / "hybrid_proxy_gaps_diagnostics.csv", index=False, encoding="utf-8-sig")
    gate_grid.to_csv(TABLES / "hybrid_risk_gate_validation_grid.csv", index=False, encoding="utf-8-sig")
    gated_selected.to_csv(TABLES / "hybrid_risk_gate_selected_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(STRATEGY_MAP).to_csv(TABLES / "hybrid_strategy_map.csv", index=False, encoding="utf-8-sig")
    write_report(summary, diagnostic, gated_selected, proxy_ic, proxy_summary, gaps_summary)
    print(f"saved {OUT}")
    print(summary[summary["phase"].eq("all")].to_string(index=False))
    print("\nValidation-selected risk gates")
    print(gated_selected.to_string(index=False))
    print("\nDiagnostics")
    print(diagnostic.to_string(index=False))


if __name__ == "__main__":
    main()
