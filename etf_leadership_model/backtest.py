from __future__ import annotations

import numpy as np
import pandas as pd


def rank_ic(frame: pd.DataFrame, score_col: str, target_col: str = "forward_20D_excess") -> float:
    vals = []
    for _, group in frame.dropna(subset=[score_col, target_col]).groupby("date"):
        if group.shape[0] < 3:
            continue
        vals.append(group[score_col].corr(group[target_col], method="spearman"))
    return float(np.nanmean(vals)) if vals else np.nan


def select_rebalance_dates(dates: pd.Series, frequency: str) -> list[pd.Timestamp]:
    idx = pd.DatetimeIndex(pd.to_datetime(dates).dropna().unique()).sort_values()
    if frequency.lower() in {"m", "month", "monthly", "monthend"}:
        return pd.Series(idx, index=idx).groupby(idx.to_period("M")).max().tolist()
    return pd.Series(idx, index=idx).groupby(idx.to_period("W-FRI")).max().tolist()


def run_topk_backtest(
    frame: pd.DataFrame,
    score_col: str,
    top_k: int = 5,
    frequency: str = "W-FRI",
    target_col: str = "forward_20D_excess",
    return_col: str = "forward_20D_return",
    benchmark_col: str = "benchmark_forward_20D_return",
    split: str | None = "test",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"])
    if split and "split" in data.columns:
        data = data[data["split"].eq(split)]
    dates = select_rebalance_dates(data["date"], frequency)
    rows = []
    for date in dates:
        sample = data[data["date"].eq(date)].dropna(subset=[score_col, return_col, benchmark_col, target_col])
        if sample.shape[0] < top_k:
            continue
        top = sample.nlargest(top_k, score_col)
        rows.append(
            {
                "date": date,
                "score_col": score_col,
                "top_k": top_k,
                "portfolio_return": float(top[return_col].mean()),
                "benchmark_return": float(top[benchmark_col].mean()),
                "excess_return": float(top[target_col].mean()),
                "universe_return": float(sample[return_col].mean()),
                "universe_excess_return": float(sample[target_col].mean()),
                "hit": int(top[target_col].mean() > 0),
                "selected": ",".join(top["etf_ticker"].astype(str).tolist()),
            }
        )
    raw = pd.DataFrame(rows)
    summary = performance_summary(raw, score_col, target_col=target_col)
    return raw, summary


def performance_summary(raw: pd.DataFrame, score_col: str, target_col: str) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    returns = raw["portfolio_return"].astype(float)
    bench = raw["benchmark_return"].astype(float)
    excess = raw["excess_return"].astype(float)
    periods_per_year = 52 if raw["date"].diff().dt.days.median() <= 10 else 12
    equity = (1 + returns.fillna(0)).cumprod()
    bench_equity = (1 + bench.fillna(0)).cumprod()
    rows = [
        {
            "score_col": score_col,
            "target_col": target_col,
            "periods": int(raw.shape[0]),
            "cumulative_return": float(equity.iloc[-1] - 1),
            "benchmark_cumulative_return": float(bench_equity.iloc[-1] - 1),
            "CAGR": cagr(equity.iloc[-1], raw.shape[0], periods_per_year),
            "MDD": max_drawdown(equity),
            "Sharpe": sharpe(returns, periods_per_year),
            "monthly_or_weekly_win_rate": float((returns > bench).mean()),
            "avg_forward_20D_excess_return": float(excess.mean()),
            "Hit_Ratio": float(raw["hit"].mean()),
        }
    ]
    return pd.DataFrame(rows)


def cagr(final_value: float, periods: int, periods_per_year: int) -> float:
    if periods <= 0 or final_value <= 0:
        return np.nan
    return float(final_value ** (periods_per_year / periods) - 1)


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def sharpe(returns: pd.Series, periods_per_year: int) -> float:
    std = returns.std()
    if std == 0 or pd.isna(std):
        return np.nan
    return float(returns.mean() / std * np.sqrt(periods_per_year))


def evaluate_scores(
    frame: pd.DataFrame,
    score_cols: list[str],
    top_k: int = 5,
    frequency: str = "W-FRI",
    split: str | None = "test",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raws = []
    summaries = []
    for score_col in score_cols:
        raw, summary = run_topk_backtest(frame, score_col, top_k=top_k, frequency=frequency, split=split)
        if not raw.empty:
            raw["Rank_IC"] = rank_ic(frame[frame.get("split", split).eq(split)] if split and "split" in frame else frame, score_col)
        if not summary.empty:
            summary["Rank_IC"] = rank_ic(frame[frame.get("split", split).eq(split)] if split and "split" in frame else frame, score_col)
        raws.append(raw)
        summaries.append(summary)
    return pd.concat(raws, ignore_index=True) if raws else pd.DataFrame(), pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()

