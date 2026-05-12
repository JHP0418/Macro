from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class StrategySpec:
    name: str
    source_dir: Path
    file_name: str
    score_col: str
    score_label: str


SPECS = [
    StrategySpec("strict_rule", ROOT / "outputs" / "etf_leadership_from_cache", "rule_scores.csv", "Final_Rule_Score", "룰베이스"),
    StrategySpec("strict_ml", ROOT / "outputs" / "etf_leadership_from_cache", "model_predictions.csv", "pred_score", "LightGBM Ranker"),
    StrategySpec("static_rule", ROOT / "outputs" / "etf_leadership_static_holdings_approx", "rule_scores.csv", "Final_Rule_Score", "구성종목근사 룰베이스"),
    StrategySpec("static_ml", ROOT / "outputs" / "etf_leadership_static_holdings_approx", "model_predictions.csv", "pred_score", "구성종목근사 Ranker"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize selective ETF leadership entry filters without using final test data.")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "etf_leadership_selective_strategy"))
    parser.add_argument("--valid-start", default="2023-01-01")
    parser.add_argument("--valid-end", default="2024-12-31")
    parser.add_argument("--test-start", default="2025-01-01")
    parser.add_argument("--top-k-list", default="1,2,3,5")
    parser.add_argument("--min-trades-weekly", type=int, default=16)
    parser.add_argument("--min-trades-monthly", type=int, default=5)
    parser.add_argument("--current-scores", default=str(ROOT / "outputs" / "etf_leadership_current" / "current_etf_leadership_scores.csv"))
    parser.add_argument("--risk-signals", default=str(ROOT / "outputs" / "risk_model_walkforward_optimizer_latest" / "tables" / "risk_threshold_signals.csv"))
    return parser.parse_args()


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def add_blended_score(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if {"Final_Rule_Score", "pred_score"}.issubset(out.columns):
        out["z_rule_for_blend"] = out.groupby("date")["Final_Rule_Score"].transform(zscore)
        out["z_pred_for_blend"] = out.groupby("date")["pred_score"].transform(zscore)
        out["blend_score"] = 0.5 * out["z_rule_for_blend"] + 0.5 * out["z_pred_for_blend"]
    return out


def zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    std = x.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (x - x.mean()) / std


def select_rebalance_dates(dates: pd.Series, frequency: str) -> list[pd.Timestamp]:
    idx = pd.DatetimeIndex(pd.to_datetime(dates).dropna().unique()).sort_values()
    if idx.empty:
        return []
    if frequency == "1M":
        return pd.Series(idx, index=idx).groupby(idx.to_period("M")).max().tolist()
    return pd.Series(idx, index=idx).groupby(idx.to_period("W-FRI")).max().tolist()


def target_columns(horizon: str) -> tuple[str, str, str]:
    if horizon == "1W":
        return "forward_5D_return", "benchmark_forward_5D_return", "forward_5D_excess"
    return "forward_20D_return", "benchmark_forward_20D_return", "forward_20D_excess"


def make_rebalance_panel(frame: pd.DataFrame, score_col: str, top_k: int, horizon: str) -> pd.DataFrame:
    ret_col, bench_col, excess_col = target_columns(horizon)
    needed = ["date", "etf_ticker", score_col, ret_col, bench_col, excess_col]
    if not set(needed).issubset(frame.columns):
        return pd.DataFrame()
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"])
    dates = select_rebalance_dates(data["date"], horizon)
    rows = []
    for dt in dates:
        sample = data[data["date"].eq(dt)].copy()
        sample = sample.dropna(subset=[score_col, ret_col, bench_col, excess_col])
        if sample.shape[0] < top_k:
            continue
        sample["score_rank"] = sample[score_col].rank(ascending=False, method="first")
        top = sample.nsmallest(top_k, "score_rank")
        ordered = top.sort_values(score_col, ascending=False)
        score_values = ordered[score_col].astype(float)
        row = {
            "date": dt,
            "horizon": horizon,
            "top_k": top_k,
            "score_col": score_col,
            "portfolio_return": float(ordered[ret_col].mean()),
            "benchmark_return": float(ordered[bench_col].mean()),
            "excess_return": float(ordered[excess_col].mean()),
            "universe_return": float(sample[ret_col].mean()),
            "universe_excess_return": float(sample[excess_col].mean()),
            "hit": int(ordered[excess_col].mean() > 0),
            "positive_return_hit": int(ordered[ret_col].mean() > 0),
            "top1_score": float(score_values.iloc[0]),
            "topk_score_mean": float(score_values.mean()),
            "topk_score_min": float(score_values.min()),
            "score_spread": float(score_values.iloc[0] - score_values.iloc[-1]) if len(score_values) > 1 else 0.0,
            "selected": ",".join(ordered["etf_ticker"].astype(str).tolist()),
            "selected_names": ",".join(ordered.get("name", ordered["etf_ticker"]).astype(str).tolist()),
        }
        for col in ["ETF_RS_20D", "ETF_RS_60D", "ETF_RS_120D", "RS_positive_share", "MA60_breadth", "Final_Rule_Score_0_100"]:
            if col in ordered.columns:
                row[f"topk_{col}_mean"] = float(pd.to_numeric(ordered[col], errors="coerce").mean())
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if out.empty:
        return out
    for col in ["top1_score", "topk_score_mean", "topk_score_min", "score_spread", "topk_ETF_RS_20D_mean", "topk_ETF_RS_60D_mean"]:
        if col in out.columns:
            out[f"{col}_pctile_expanding"] = expanding_percentile(out[col])
    out["confidence_score"] = build_confidence_score(out)
    return out


def expanding_percentile(s: pd.Series, min_periods: int = 26) -> pd.Series:
    vals = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    for i, val in enumerate(vals):
        if i < min_periods or np.isnan(val):
            continue
        hist = vals[:i]
        hist = hist[~np.isnan(hist)]
        if hist.size < min_periods:
            continue
        out[i] = float((hist <= val).mean())
    return pd.Series(out, index=s.index)


def build_confidence_score(panel: pd.DataFrame) -> pd.Series:
    parts = []
    weights = []
    for col, weight in [
        ("top1_score_pctile_expanding", 0.30),
        ("topk_score_mean_pctile_expanding", 0.30),
        ("score_spread_pctile_expanding", 0.10),
        ("topk_ETF_RS_20D_mean_pctile_expanding", 0.15),
        ("topk_ETF_RS_60D_mean_pctile_expanding", 0.15),
    ]:
        if col in panel.columns:
            parts.append(panel[col].astype(float).fillna(0.5) * weight)
            weights.append(weight)
    if not parts:
        return pd.Series(np.full(panel.shape[0], 0.5), index=panel.index)
    return sum(parts) / sum(weights)


def load_risk_signals(path: Path) -> pd.DataFrame:
    risk = load_csv(path)
    if risk.empty:
        return risk
    if "Date" in risk.columns:
        risk = risk.rename(columns={"Date": "date"})
    risk["date"] = pd.to_datetime(risk["date"])
    return risk


def attach_risk_overlay(panel: pd.DataFrame, risk: pd.DataFrame, horizon: str) -> pd.DataFrame:
    if panel.empty or risk.empty:
        out = panel.copy()
        out["risk_block_count"] = 0
        return out
    if horizon == "1W":
        cols = [
            "signal_risk_assets_practical_loss_1w_prob_cal",
            "signal_nasdaq_1w_drop_2pct_prob_cal",
            "signal_sox_1w_drop_3pct_prob_cal",
            "signal_kospi_1w_drop_2pct_prob_cal",
            "signal_safety_rotation_needed_1w_prob_cal",
        ]
    else:
        cols = [
            "signal_risk_assets_practical_loss_1m_prob_cal",
            "signal_nasdaq_1m_correction_prob_cal",
            "signal_sox_1m_correction_prob_cal",
            "signal_kospi_1m_correction_prob_cal",
            "signal_safety_rotation_needed_1m_prob_cal",
        ]
    cols = [c for c in cols if c in risk.columns]
    if not cols:
        out = panel.copy()
        out["risk_block_count"] = 0
        return out
    risk_small = risk[["date", *cols]].copy()
    for c in cols:
        risk_small[c] = pd.to_numeric(risk_small[c], errors="coerce").fillna(0).astype(int)
    risk_small["risk_block_count"] = risk_small[cols].sum(axis=1)
    risk_small = risk_small[["date", "risk_block_count"]].sort_values("date")
    out = pd.merge_asof(panel.sort_values("date"), risk_small, on="date", direction="backward")
    out["risk_block_count"] = out["risk_block_count"].fillna(0).astype(int)
    return out


def apply_threshold(panel: pd.DataFrame, threshold_col: str, threshold: float, risk_block_limit: int = 99) -> pd.DataFrame:
    out = panel.copy()
    risk_ok = out.get("risk_block_count", pd.Series(np.zeros(len(out)), index=out.index)).astype(int).le(risk_block_limit)
    out["risk_allowed"] = risk_ok.astype(int)
    out["invested"] = (pd.to_numeric(out[threshold_col], errors="coerce").ge(threshold) & risk_ok).astype(int)
    out["strategy_return"] = out["portfolio_return"].where(out["invested"].eq(1), 0.0)
    out["strategy_excess_return"] = out["excess_return"].where(out["invested"].eq(1), 0.0)
    out["strategy_hit"] = np.where(out["invested"].eq(1), out["hit"], np.nan)
    out["strategy_positive_return_hit"] = np.where(out["invested"].eq(1), out["positive_return_hit"], np.nan)
    return out


def summarize(panel: pd.DataFrame, label: str, periods_per_year: int) -> dict:
    if panel.empty:
        return {"label": label, "periods": 0}
    returns = panel["strategy_return"].astype(float).fillna(0.0)
    invested = panel["invested"].astype(int) if "invested" in panel else pd.Series(np.ones(len(panel)), index=panel.index)
    trade_excess = panel.loc[invested.eq(1), "excess_return"].astype(float)
    trade_returns = panel.loc[invested.eq(1), "portfolio_return"].astype(float)
    equity = (1.0 + returns).cumprod()
    bench_equity = (1.0 + panel["benchmark_return"].astype(float).fillna(0.0)).cumprod()
    return {
        "label": label,
        "periods": int(panel.shape[0]),
        "invested_periods": int(invested.sum()),
        "coverage": float(invested.mean()) if len(invested) else np.nan,
        "cumulative_return": float(equity.iloc[-1] - 1.0),
        "benchmark_cumulative_return": float(bench_equity.iloc[-1] - 1.0),
        "CAGR": cagr(float(equity.iloc[-1]), int(panel.shape[0]), periods_per_year),
        "MDD": max_drawdown(equity),
        "Sharpe": sharpe(returns, periods_per_year),
        "trade_hit_ratio_excess_gt0": float((trade_excess > 0).mean()) if not trade_excess.empty else np.nan,
        "trade_positive_return_ratio": float((trade_returns > 0).mean()) if not trade_returns.empty else np.nan,
        "avg_trade_return": float(trade_returns.mean()) if not trade_returns.empty else np.nan,
        "avg_trade_excess_return": float(trade_excess.mean()) if not trade_excess.empty else np.nan,
    }


def cagr(final_value: float, periods: int, periods_per_year: int) -> float:
    if periods <= 0 or final_value <= 0:
        return np.nan
    return float(final_value ** (periods_per_year / periods) - 1.0)


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return np.nan
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def sharpe(returns: pd.Series, periods_per_year: int) -> float:
    std = returns.std(ddof=1)
    if pd.isna(std) or std == 0:
        return np.nan
    return float(returns.mean() / std * np.sqrt(periods_per_year))


def optimize_thresholds(panel: pd.DataFrame, min_trades: int, periods_per_year: int) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()
    threshold_cols = ["confidence_score"]
    threshold_cols += [c for c in ["topk_score_mean_pctile_expanding", "top1_score_pctile_expanding"] if c in panel.columns]
    rows = []
    for col in threshold_cols:
        for threshold in np.round(np.arange(0.50, 0.96, 0.025), 3):
            for risk_block_limit in [99, 4, 3, 2, 1, 0]:
                tested = apply_threshold(panel, col, threshold, risk_block_limit=risk_block_limit)
                invested_periods = int(tested["invested"].sum())
                if invested_periods < min_trades:
                    continue
                summary = summarize(tested, f"{col}>={threshold}", periods_per_year)
                summary.update({"threshold_col": col, "threshold": threshold, "risk_block_limit": risk_block_limit})
                rows.append(summary)
    grid = pd.DataFrame(rows)
    if grid.empty:
        return grid
    grid["objective"] = (
        grid["Sharpe"].fillna(-99)
        + 0.75 * grid["trade_hit_ratio_excess_gt0"].fillna(0)
        - 0.25 * grid["MDD"].abs().fillna(1)
        - 0.10 * (1.0 - grid["coverage"].fillna(0))
    )
    grid["objective_high_precision"] = (
        grid["Sharpe"].fillna(-99)
        + 1.25 * grid["trade_positive_return_ratio"].fillna(0)
        + 0.50 * grid["trade_hit_ratio_excess_gt0"].fillna(0)
        - 0.35 * grid["MDD"].abs().fillna(1)
        - 0.15 * (1.0 - grid["coverage"].fillna(0))
    )
    return grid.sort_values(["objective", "Sharpe", "trade_hit_ratio_excess_gt0"], ascending=False).reset_index(drop=True)


def load_strategy_frame(spec: StrategySpec) -> pd.DataFrame:
    frame = load_csv(spec.source_dir / spec.file_name)
    if frame.empty:
        return frame
    if spec.score_col == "pred_score" and "Final_Rule_Score" not in frame.columns:
        rule = load_csv(spec.source_dir / "rule_scores.csv")
        keep = [c for c in ["date", "etf_ticker", "Final_Rule_Score", "Final_Rule_Score_0_100", "rule_rank"] if c in rule.columns]
        if keep:
            frame = frame.merge(rule[keep], on=["date", "etf_ticker"], how="left")
    if spec.score_col == "blend_score":
        frame = add_blended_score(frame)
    return frame


def current_signal(current_scores_path: Path, best_rows: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    current = load_csv(current_scores_path)
    if current.empty:
        return pd.DataFrame()
    if "leadership_score_0_100" not in current.columns and "Final_Rule_Score_0_100" in current.columns:
        current["leadership_score_0_100"] = current["Final_Rule_Score_0_100"]
    # The current screen has no forward target, so use the same score ranking and report the active threshold as guidance.
    top = current.sort_values("Final_Rule_Score", ascending=False).head(20).copy()
    signal_rows = []
    for _, best in best_rows.iterrows():
        if best.get("score_col") != "Final_Rule_Score":
            continue
        top_k = int(best["top_k"])
        selected = current.sort_values("Final_Rule_Score", ascending=False).head(top_k)
        signal_rows.append(
            {
                "model": best["model"],
                "horizon": best["horizon"],
                "score_type": best["score_label"],
                "top_k": top_k,
                "threshold_col": best["threshold_col"],
                "threshold": best["threshold"],
                "current_topk_mean_score": float(selected["Final_Rule_Score"].mean()),
                "selected_etfs": ",".join(selected["etf_ticker"].astype(str).tolist()),
                "selected_names": ",".join(selected.get("name", selected["etf_ticker"]).astype(str).tolist()),
                "note": "현재 화면은 최신 단면 점수 기준입니다. threshold 백분위는 과거 패널 기반이므로 실전에서는 백테스트 결과와 함께 해석해야 합니다.",
            }
        )
    signal = pd.DataFrame(signal_rows)
    if not signal.empty:
        signal.to_csv(output_dir / "current_selective_signal.csv", index=False, encoding="utf-8-sig")
    top.to_csv(output_dir / "current_top20_for_selective_strategy.csv", index=False, encoding="utf-8-sig")
    return signal


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_start = pd.Timestamp(args.valid_start)
    valid_end = pd.Timestamp(args.valid_end)
    test_start = pd.Timestamp(args.test_start)
    top_k_list = [int(x.strip()) for x in args.top_k_list.split(",") if x.strip()]
    risk_signals = load_risk_signals(Path(args.risk_signals))

    validation_rows = []
    test_rows = []
    trade_frames = []
    best_configs = []

    all_specs = SPECS + [
        StrategySpec("strict_blend", ROOT / "outputs" / "etf_leadership_from_cache", "model_predictions.csv", "blend_score", "룰+Ranker 블렌드"),
        StrategySpec("static_blend", ROOT / "outputs" / "etf_leadership_static_holdings_approx", "model_predictions.csv", "blend_score", "구성종목근사 블렌드"),
    ]

    for spec in all_specs:
        frame = load_strategy_frame(spec)
        if frame.empty or spec.score_col not in frame.columns:
            continue
        for horizon in ["1W", "1M"]:
            periods_per_year = 52 if horizon == "1W" else 12
            min_trades = args.min_trades_weekly if horizon == "1W" else args.min_trades_monthly
            for top_k in top_k_list:
                panel = make_rebalance_panel(frame, spec.score_col, top_k, horizon)
                if panel.empty:
                    continue
                panel = attach_risk_overlay(panel, risk_signals, horizon)
                valid_panel = panel[panel["date"].between(valid_start, valid_end)].copy()
                test_panel = panel[panel["date"].ge(test_start)].copy()
                if valid_panel.empty or test_panel.empty:
                    continue
                grid = optimize_thresholds(valid_panel, min_trades=min_trades, periods_per_year=periods_per_year)
                if grid.empty:
                    continue
                selection_modes = [
                    ("balanced", "objective", ["objective", "Sharpe", "trade_hit_ratio_excess_gt0"]),
                    ("high_precision", "objective_high_precision", ["objective_high_precision", "trade_positive_return_ratio", "Sharpe"]),
                ]
                for mode_name, objective_col, sort_cols in selection_modes:
                    ranked_grid = grid.sort_values(sort_cols, ascending=False).reset_index(drop=True)
                    best = ranked_grid.iloc[0].to_dict()
                    best.update(
                        {
                            "strategy_mode": mode_name,
                            "model": spec.name,
                            "score_label": spec.score_label,
                            "horizon": horizon,
                            "top_k": top_k,
                            "score_col": spec.score_col,
                        }
                    )
                    validation_rows.append(best)
                    threshold_col = best["threshold_col"]
                    threshold = float(best["threshold"])
                    risk_block_limit = int(best.get("risk_block_limit", 99))
                    tested = apply_threshold(test_panel, threshold_col, threshold, risk_block_limit=risk_block_limit)
                    summary = summarize(tested, f"{spec.name}_{mode_name}_{horizon}_top{top_k}", periods_per_year)
                    summary.update(
                        {
                            "strategy_mode": mode_name,
                            "model": spec.name,
                            "score_label": spec.score_label,
                            "horizon": horizon,
                            "top_k": top_k,
                            "score_col": spec.score_col,
                            "threshold_col": threshold_col,
                            "threshold": threshold,
                            "risk_block_limit": risk_block_limit,
                            "valid_objective": best["objective"],
                            "valid_objective_high_precision": best["objective_high_precision"],
                            "valid_sharpe": best["Sharpe"],
                            "valid_hit_ratio": best["trade_hit_ratio_excess_gt0"],
                            "valid_positive_return_ratio": best["trade_positive_return_ratio"],
                            "valid_coverage": best["coverage"],
                        }
                    )
                    test_rows.append(summary)
                    tested["strategy_mode"] = mode_name
                    tested["model"] = spec.name
                    tested["score_label"] = spec.score_label
                    tested["threshold_col"] = threshold_col
                    tested["threshold"] = threshold
                    tested["risk_block_limit"] = risk_block_limit
                    trade_frames.append(tested)
                    best_configs.append(summary)

    validation = pd.DataFrame(validation_rows)
    test_summary = pd.DataFrame(test_rows)
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()

    if not validation.empty:
        validation.sort_values(["horizon", "objective"], ascending=[True, False]).to_csv(output_dir / "validation_best_by_config.csv", index=False, encoding="utf-8-sig")
    if not test_summary.empty:
        test_summary["practical_rank"] = test_summary.groupby("horizon")["Sharpe"].rank(ascending=False, method="first")
        test_summary.sort_values(["horizon", "Sharpe"], ascending=[True, False]).to_csv(output_dir / "test_summary.csv", index=False, encoding="utf-8-sig")
        top_best = test_summary.sort_values(["horizon", "Sharpe"], ascending=[True, False]).groupby("horizon").head(3)
        top_best.to_csv(output_dir / "best_selective_models.csv", index=False, encoding="utf-8-sig")
        current_signal(Path(args.current_scores), top_best, output_dir)
    if not trades.empty:
        trades.sort_values(["model", "horizon", "top_k", "date"]).to_csv(output_dir / "selected_trades.csv", index=False, encoding="utf-8-sig")

    print(f"saved: {output_dir}")
    if not test_summary.empty:
        cols = ["strategy_mode", "model", "horizon", "top_k", "threshold_col", "threshold", "risk_block_limit", "invested_periods", "coverage", "Sharpe", "trade_hit_ratio_excess_gt0", "trade_positive_return_ratio", "CAGR", "MDD"]
        print(test_summary.sort_values(["horizon", "Sharpe"], ascending=[True, False])[cols].head(16).to_string(index=False))


if __name__ == "__main__":
    main()
