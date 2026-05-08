from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from daily_risk_off_sentinel import OUT_DIR as SENTINEL_OUT_DIR  # noqa: E402

WEEKLY_OUT_DIR = ROOT / "outputs" / "weekly_screening_rank_backtest_latest"
OUT_DIR = ROOT / "outputs" / "risk_off_safe_asset_backtest_latest"
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
    "US REIT",
    "China/HK growth",
    "China equity",
    "India/EM",
    "Japan equity",
    "US high yield",
    "Oil",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Risk-off detection and safe-asset selection backtest.")
    parser.add_argument("--weekly-output", type=Path, default=WEEKLY_OUT_DIR)
    parser.add_argument("--sentinel-output", type=Path, default=SENTINEL_OUT_DIR)
    parser.add_argument("--output", type=Path, default=OUT_DIR)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--risk-off-threshold", type=float, default=25.0)
    parser.add_argument("--crash-threshold-1m", type=float, default=-0.07)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables = args.output / "tables"
    reports = args.output / "reports"
    for path in (tables, reports):
        path.mkdir(parents=True, exist_ok=True)

    weekly = pd.read_csv(args.weekly_output / "tables" / "weekly_calibrated_rank_panel.csv", parse_dates=["date"])
    sentinel = pd.read_csv(args.sentinel_output / "tables" / "daily_sentinel_history.csv", parse_dates=["Date"]).set_index("Date")
    panel = attach_weekly_sentinel(weekly, sentinel, args.risk_off_threshold, args.crash_threshold_1m)
    risk_detection = risk_off_detection_metrics(panel)
    safe_selection = safe_asset_selection_metrics(panel, args.top_k)
    safe_rank_by_group = safe_group_metrics(panel, args.top_k)
    episodes = risk_off_episodes(panel)
    current = current_safe_asset_verdict(panel, args.top_k)

    panel.to_csv(tables / "risk_off_weekly_panel.csv", index=False, encoding="utf-8-sig")
    risk_detection.to_csv(tables / "risk_off_detection_metrics.csv", index=False, encoding="utf-8-sig")
    safe_selection.to_csv(tables / "risk_off_safe_asset_selection.csv", index=False, encoding="utf-8-sig")
    safe_rank_by_group.to_csv(tables / "risk_off_safe_group_metrics.csv", index=False, encoding="utf-8-sig")
    episodes.to_csv(tables / "risk_off_episodes.csv", index=False, encoding="utf-8-sig")
    current.to_csv(tables / "current_risk_off_safe_asset_verdict.csv", index=False, encoding="utf-8-sig")
    write_report(risk_detection, safe_selection, safe_rank_by_group, episodes, current, reports / "risk_off_safe_asset_backtest.md", args.top_k)

    print(f"wrote {reports / 'risk_off_safe_asset_backtest.md'}")
    print(risk_detection.to_string(index=False))
    print(safe_selection.to_string(index=False))


def attach_weekly_sentinel(weekly: pd.DataFrame, sentinel: pd.DataFrame, threshold: float, crash_threshold: float) -> pd.DataFrame:
    out = weekly.copy()
    sent = sentinel[["risk_off_score", "sentinel_state", "dominant_component", "risk_budget_pct"]].copy()
    sent = sent.sort_index()
    aligned = []
    for date in pd.to_datetime(out["date"]):
        sample = sent.loc[:date]
        aligned.append(sample.iloc[-1].to_dict() if not sample.empty else {})
    sent_cols = pd.DataFrame(aligned)
    out = pd.concat([out.reset_index(drop=True), sent_cols.reset_index(drop=True)], axis=1)
    out["sentinel_warning"] = pd.to_numeric(out["risk_off_score"], errors="coerce").fillna(0).ge(threshold)
    out["is_safe_asset"] = out["group"].isin(SAFE_GROUPS)
    out["is_risk_asset"] = out["group"].isin(RISK_GROUPS)
    date_forward = (
        out[out["is_risk_asset"]]
        .groupby("date", as_index=False)
        .agg(
            risk_asset_avg_1w=("realized_return_1w", "mean"),
            risk_asset_avg_1m=("realized_return_4w", "mean"),
            risk_asset_min_1m=("realized_return_4w", "min"),
        )
    )
    out = out.merge(date_forward, on="date", how="left")
    out["future_risk_off_label"] = out["risk_asset_avg_1m"].le(crash_threshold)
    out["risk_off_safe_score"] = out.apply(score_safe_asset_for_stress, axis=1)
    return out


def score_safe_asset_for_stress(row: pd.Series) -> float:
    base = float(row.get("institutional_score_0_100", row.get("score_0_100", 50.0)) or 50.0)
    group = str(row.get("group", ""))
    component = str(row.get("dominant_component", ""))
    state = str(row.get("sentinel_state", "Normal"))
    risk_score = float(row.get("risk_off_score", 0.0) or 0.0)
    if group not in SAFE_GROUPS:
        return base

    boost = 0.0
    if group == "Cash/short bonds":
        boost += 15.0
        if component in {"credit", "volatility", "equity"}:
            boost += 8.0
        if component in {"supply_shock", "hedge_bid"}:
            boost -= 4.0
    elif group == "USD cash":
        boost += 8.0
        if component in {"fx", "credit", "volatility", "equity"}:
            boost += 16.0
        if component == "supply_shock":
            boost += 4.0
    elif group == "Gold":
        boost += 4.0
        if component in {"supply_shock", "hedge_bid", "volatility", "fx"}:
            boost += 18.0
        if component == "credit":
            boost += 5.0
    elif group in {"Korea bonds", "US long bonds"}:
        boost += 5.0
        if component in {"volatility", "equity", "credit"}:
            boost += 14.0
        if component in {"supply_shock", "fx"}:
            boost -= 12.0
    elif group == "US IG bonds":
        boost += 4.0
        if component in {"volatility", "equity"}:
            boost += 10.0
        if component == "credit":
            boost -= 12.0
    elif group == "Korea defensive":
        boost += 2.0
        if component in {"equity", "volatility"}:
            boost += 8.0

    if state == "Cash":
        boost += 8.0
    elif state == "De-risk":
        boost += 5.0
    elif state == "Watch":
        boost += 2.0
    boost += min(max(risk_score - 25.0, 0.0), 50.0) * 0.12
    return float(np.clip(base + boost, 0, 100))


def risk_off_detection_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    by_date = panel.groupby("date", as_index=False).agg(
        sentinel_warning=("sentinel_warning", "max"),
        risk_off_score=("risk_off_score", "max"),
        future_risk_off_label=("future_risk_off_label", "max"),
        risk_asset_avg_1m=("risk_asset_avg_1m", "first"),
    )
    pred = by_date["sentinel_warning"].astype(bool)
    actual = by_date["future_risk_off_label"].astype(bool)
    tp = int((pred & actual).sum())
    fp = int((pred & ~actual).sum())
    fn = int((~pred & actual).sum())
    tn = int((~pred & ~actual).sum())
    return pd.DataFrame(
        [
            {
                "weeks": int(by_date.shape[0]),
                "risk_off_weeks": int(actual.sum()),
                "warning_weeks": int(pred.sum()),
                "precision": tp / max(tp + fp, 1),
                "recall": tp / max(tp + fn, 1),
                "false_alarm_rate": fp / max(tp + fp, 1),
                "accuracy": float(pred.eq(actual).mean()),
                "avg_future_risk_asset_1m_when_warning": float(by_date.loc[pred, "risk_asset_avg_1m"].mean()),
                "avg_future_risk_asset_1m_without_warning": float(by_date.loc[~pred, "risk_asset_avg_1m"].mean()),
            }
        ]
    )


def safe_asset_selection_metrics(panel: pd.DataFrame, top_k: int) -> pd.DataFrame:
    rows = []
    risk_dates = panel.groupby("date")["sentinel_warning"].max()
    risk_dates = set(risk_dates[risk_dates].index)
    for horizon, ret_col, rank_col in [("1w", "realized_return_1w", "actual_rank_1w"), ("1m", "realized_return_4w", "actual_rank_4w")]:
        per = []
        for date, group in panel[panel["date"].isin(risk_dates)].groupby("date"):
            safe = group[group["is_safe_asset"]].copy()
            risk = group[group["is_risk_asset"]].copy()
            if safe.empty or risk.empty:
                continue
            safe["safe_rank"] = safe["risk_off_safe_score"].rank(ascending=False, method="first") if "risk_off_safe_score" in safe else safe["institutional_score_0_100"].rank(ascending=False, method="first")
            picks = safe.nsmallest(top_k, "safe_rank")
            actual_safe_best = safe.nlargest(top_k, ret_col)
            per.append(
                {
                    "date": date,
                    "picked_return": float(picks[ret_col].mean()),
                    "safe_universe_return": float(safe[ret_col].mean()),
                    "risk_universe_return": float(risk[ret_col].mean()),
                    "actual_safe_top_return": float(actual_safe_best[ret_col].mean()),
                    "hit_rate_safe_top": len(set(picks["symbol"]) & set(actual_safe_best["symbol"])) / top_k,
                    "beat_risk_assets": float(picks[ret_col].mean() > risk[ret_col].mean()),
                    "beat_safe_average": float(picks[ret_col].mean() > safe[ret_col].mean()),
                }
            )
        frame = pd.DataFrame(per)
        rows.append(
            {
                "horizon": horizon,
                "risk_warning_weeks": int(frame.shape[0]),
                f"safe_top{top_k}_return": float(frame["picked_return"].mean()) if not frame.empty else np.nan,
                "safe_universe_return": float(frame["safe_universe_return"].mean()) if not frame.empty else np.nan,
                "risk_universe_return": float(frame["risk_universe_return"].mean()) if not frame.empty else np.nan,
                "actual_safe_top_return": float(frame["actual_safe_top_return"].mean()) if not frame.empty else np.nan,
                "safe_top_hit_rate": float(frame["hit_rate_safe_top"].mean()) if not frame.empty else np.nan,
                "beat_risk_assets_rate": float(frame["beat_risk_assets"].mean()) if not frame.empty else np.nan,
                "beat_safe_average_rate": float(frame["beat_safe_average"].mean()) if not frame.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def safe_group_metrics(panel: pd.DataFrame, top_k: int) -> pd.DataFrame:
    risk_panel = panel[panel["sentinel_warning"] & panel["is_safe_asset"]].copy()
    if risk_panel.empty:
        return pd.DataFrame()
    rows = []
    for horizon, ret_col in [("1w", "realized_return_1w"), ("1m", "realized_return_4w")]:
        for group, sample in risk_panel.groupby("group"):
            rows.append(
                {
                    "horizon": horizon,
                    "group": group,
                    "samples": int(sample.shape[0]),
                    "avg_return": float(sample[ret_col].mean()),
                    "positive_rate": float(sample[ret_col].gt(0).mean()),
                    "median_return": float(sample[ret_col].median()),
                }
            )
    return pd.DataFrame(rows).sort_values(["horizon", "avg_return"], ascending=[True, False])


def risk_off_episodes(panel: pd.DataFrame) -> pd.DataFrame:
    by_date = panel.groupby("date", as_index=False).agg(
        sentinel_warning=("sentinel_warning", "max"),
        future_risk_off_label=("future_risk_off_label", "max"),
        risk_off_score=("risk_off_score", "max"),
        risk_asset_avg_1m=("risk_asset_avg_1m", "first"),
    )
    rows = []
    previous = False
    for _, row in by_date.iterrows():
        current = bool(row["future_risk_off_label"])
        if current and not previous:
            start = pd.Timestamp(row["date"])
            prior = by_date[(pd.to_datetime(by_date["date"]).ge(start - pd.Timedelta(days=28))) & (pd.to_datetime(by_date["date"]).le(start))]
            warnings = prior[prior["sentinel_warning"]]
            rows.append(
                {
                    "episode_start": start.date().isoformat(),
                    "detected_before_or_at_start": not warnings.empty,
                    "first_warning": pd.Timestamp(warnings.iloc[0]["date"]).date().isoformat() if not warnings.empty else None,
                    "lead_days": int((start - pd.Timestamp(warnings.iloc[0]["date"])).days) if not warnings.empty else None,
                    "max_pre_start_score": float(prior["risk_off_score"].max()),
                    "risk_asset_1m_at_start": float(row["risk_asset_avg_1m"]),
                }
            )
        previous = current
    return pd.DataFrame(rows)


def current_safe_asset_verdict(panel: pd.DataFrame, top_k: int) -> pd.DataFrame:
    latest_date = pd.to_datetime(panel["date"]).max()
    latest = panel[pd.to_datetime(panel["date"]).eq(latest_date)].copy()
    latest_safe = latest[latest["is_safe_asset"]].copy()
    sort_col = "risk_off_safe_score" if "risk_off_safe_score" in latest_safe else "institutional_score_0_100"
    latest_safe = latest_safe.sort_values(sort_col, ascending=False)
    return latest_safe.head(top_k)[
        [
            "date",
            "symbol",
            "name",
            "group",
            "sentinel_state",
            "risk_off_score",
            "dominant_component",
            "risk_off_safe_score",
            "institutional_score_0_100",
            "calibrated_prob_1w",
            "calibrated_prob_4w",
            "realized_return_1w",
            "realized_return_4w",
        ]
    ]


def write_report(
    detection: pd.DataFrame,
    selection: pd.DataFrame,
    group_metrics: pd.DataFrame,
    episodes: pd.DataFrame,
    current: pd.DataFrame,
    path: Path,
    top_k: int,
) -> None:
    lines = ["# Risk-Off And Safe-Asset Backtest", ""]
    if not detection.empty:
        lines.extend(["## Risk-Off Detection", detection.to_markdown(index=False), ""])
    if not selection.empty:
        lines.extend([f"## Safe Asset Top {top_k} Selection", selection.to_markdown(index=False), ""])
    if not group_metrics.empty:
        lines.extend(["## Safe Asset Group Results", group_metrics.to_markdown(index=False), ""])
    if not episodes.empty:
        lines.extend(["## Risk-Off Episodes", episodes.to_markdown(index=False), ""])
    if not current.empty:
        lines.extend([f"## Current Safe-Asset Candidates", current.to_markdown(index=False), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
