from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("outputs/gaps_long_lived_etf_leadership_latest/tables/long_lived_predictions.csv")
DEFAULT_OUTPUT = Path("outputs/leadership_rule_vs_ranker_compare")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standardized comparison of the original ETF leadership rule ranking "
            "and plugged-in LightGBM tree ranker."
        )
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--split", default="test", help="Use test, valid, train, all, or none.")
    parser.add_argument("--prediction-horizon", default="1M")
    parser.add_argument(
        "--target-col",
        default="forward_20D_excess",
        choices=["forward_20D_excess", "forward_20D_return", "forward_5D_excess", "forward_5D_return"],
    )
    parser.add_argument("--rule-score-col", default="rule_20d_score")
    parser.add_argument("--tree-score-col", default="ranker_score")
    parser.add_argument("--top-k", default="1,2,3,5,10,20")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.input, parse_dates=["date"])
    data = filter_data(raw, split=args.split, prediction_horizon=args.prediction_horizon)
    required = ["date", "etf_ticker", args.target_col, args.rule_score_col, args.tree_score_col]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    data = data.dropna(subset=[args.target_col, args.rule_score_col, args.tree_score_col]).copy()
    if data.empty:
        raise RuntimeError("No comparable rows after split/horizon/NaN filtering.")

    outputs = {
        "Model_A_Rule_Ranking": common_output(data, args.rule_score_col, args.target_col),
        "Model_B_Tree_Ranker": common_output(data, args.tree_score_col, args.target_col),
    }
    outputs["Model_A_Rule_Ranking"].to_csv(output_dir / "model_a_rule_output.csv", index=False, encoding="utf-8-sig")
    outputs["Model_B_Tree_Ranker"].to_csv(output_dir / "model_b_tree_ranker_output.csv", index=False, encoding="utf-8-sig")

    ks = tuple(int(k.strip()) for k in str(args.top_k).split(",") if k.strip())
    summary, topk, daily_topk, topk_diff = evaluate_models(outputs, ks=ks)
    summary.to_csv(output_dir / "evaluation_summary.csv", index=False, encoding="utf-8-sig")
    topk.to_csv(output_dir / "topk_decay.csv", index=False, encoding="utf-8-sig")
    daily_topk.to_csv(output_dir / "daily_topk_returns.csv", index=False, encoding="utf-8-sig")
    topk_diff.to_csv(output_dir / "topk_model_difference.csv", index=False, encoding="utf-8-sig")

    if args.plot:
        plot_topk_decay(topk, output_dir / "topk_decay.png")
        plot_metric_bars(summary, output_dir / "model_metric_bars.png")

    meta = {
        "input": str(args.input),
        "split": args.split,
        "prediction_horizon": args.prediction_horizon,
        "target_col": args.target_col,
        "rule_score_col": args.rule_score_col,
        "tree_score_col": args.tree_score_col,
        "rows": int(len(data)),
        "dates": int(data["date"].nunique()),
        "tickers": int(data["etf_ticker"].nunique()),
        "outputs": {
            "model_a": "model_a_rule_output.csv",
            "model_b": "model_b_tree_ranker_output.csv",
            "summary": "evaluation_summary.csv",
            "topk_decay": "topk_decay.csv",
            "daily_topk_returns": "daily_topk_returns.csv",
            "topk_model_difference": "topk_model_difference.csv",
        },
    }
    (output_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(summary.to_string(index=False))
    print()
    print(topk.to_string(index=False))
    print()
    print(topk_diff.to_string(index=False))


def filter_data(frame: pd.DataFrame, split: str, prediction_horizon: str) -> pd.DataFrame:
    out = frame.copy()
    if split.lower() not in {"all", "none", ""} and "split" in out.columns:
        out = out[out["split"].astype(str).eq(split)]
    if prediction_horizon.lower() not in {"all", "none", ""} and "prediction_horizon" in out.columns:
        out = out[out["prediction_horizon"].astype(str).eq(prediction_horizon)]
    return out


def common_output(frame: pd.DataFrame, score_col: str, target_col: str) -> pd.DataFrame:
    out = frame[["date", "etf_ticker", score_col, target_col]].copy()
    out = out.rename(
        columns={
            "date": "Date",
            "etf_ticker": "Ticker",
            score_col: "Predicted_Score",
            target_col: "Actual_Return",
        }
    )
    return out.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def evaluate_models(outputs: dict[str, pd.DataFrame], ks: tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    topk_parts = []
    daily_parts = []
    for model_name, output in outputs.items():
        spearman = spearman_by_date(output)
        rmse = fit_rmse(output)
        mean_spearman = float(spearman["Spearman"].mean()) if not spearman.empty else np.nan
        summary_rows.append(
            {
                "Model": model_name,
                "Mean_Spearman": mean_spearman,
                "Fit_RMSE": rmse,
                "Composite_Score": composite_score(mean_spearman, rmse),
                "Evaluated_Dates": int(output["Date"].nunique()),
                "Evaluated_Rows": int(len(output)),
            }
        )
        topk, daily = top_k_returns(output, model_name=model_name, ks=ks)
        topk_parts.append(topk)
        daily_parts.append(daily)

    summary = pd.DataFrame(summary_rows)
    topk = pd.concat(topk_parts, ignore_index=True)
    daily_topk = pd.concat(daily_parts, ignore_index=True)
    diff = topk_difference(daily_topk)
    return summary, topk, daily_topk, diff


def spearman_by_date(output: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import spearmanr

    rows = []
    clean = output.dropna(subset=["Predicted_Score", "Actual_Return"])
    for date, group in clean.groupby("Date"):
        if group.shape[0] < 3 or group["Predicted_Score"].nunique() < 2 or group["Actual_Return"].nunique() < 2:
            continue
        corr, _ = spearmanr(group["Predicted_Score"], group["Actual_Return"])
        rows.append({"Date": date, "Spearman": corr})
    return pd.DataFrame(rows)


def fit_rmse(output: pd.DataFrame) -> float:
    from sklearn.metrics import mean_squared_error

    data = output.copy()
    data["z_pred"] = data.groupby("Date")["Predicted_Score"].transform(zscore)
    data["z_actual"] = data.groupby("Date")["Actual_Return"].transform(zscore)
    data = data.dropna(subset=["z_pred", "z_actual"])
    if data.empty:
        return np.nan
    return float(np.sqrt(mean_squared_error(data["z_actual"], data["z_pred"])))


def top_k_returns(output: pd.DataFrame, model_name: str, ks: tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = output.dropna(subset=["Predicted_Score", "Actual_Return"]).copy()
    summary_rows = []
    daily_rows = []
    for k in ks:
        daily = (
            clean.sort_values(["Date", "Predicted_Score"], ascending=[True, False])
            .groupby("Date")
            .head(k)
            .groupby("Date")["Actual_Return"]
            .mean()
        )
        if daily.empty:
            summary_rows.append(empty_topk_row(model_name, k))
            continue
        std = daily.std()
        sharpe = float(daily.mean() / std * np.sqrt(252)) if pd.notna(std) and std != 0 else np.nan
        summary_rows.append(
            {
                "Model": model_name,
                "K": k,
                "Mean_Return": float(daily.mean()),
                "Median_Return": float(daily.median()),
                "Sharpe": sharpe,
                "Hit_Ratio": float((daily > 0).mean()),
                "Periods": int(daily.shape[0]),
            }
        )
        daily_rows.extend({"Date": date, "Model": model_name, "K": k, "TopK_Return": value} for date, value in daily.items())
    return pd.DataFrame(summary_rows), pd.DataFrame(daily_rows)


def empty_topk_row(model_name: str, k: int) -> dict[str, object]:
    return {
        "Model": model_name,
        "K": k,
        "Mean_Return": np.nan,
        "Median_Return": np.nan,
        "Sharpe": np.nan,
        "Hit_Ratio": np.nan,
        "Periods": 0,
    }


def topk_difference(daily_topk: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import ttest_rel, wilcoxon

    rows = []
    models = sorted(daily_topk["Model"].unique())
    if len(models) != 2:
        return pd.DataFrame()
    left, right = models
    for k, group in daily_topk.groupby("K"):
        pivot = group.pivot(index="Date", columns="Model", values="TopK_Return").dropna()
        if pivot.empty:
            continue
        diff = pivot[right] - pivot[left]
        t_stat, t_pvalue = ttest_rel(pivot[right], pivot[left])
        try:
            w_stat, w_pvalue = wilcoxon(diff)
        except ValueError:
            w_stat, w_pvalue = np.nan, np.nan
        rows.append(
            {
                "K": int(k),
                "Left_Model": left,
                "Right_Model": right,
                "Mean_Diff_Right_Minus_Left": float(diff.mean()),
                "Median_Diff_Right_Minus_Left": float(diff.median()),
                "Right_Better_Ratio": float((diff > 0).mean()),
                "Paired_T_Stat": float(t_stat),
                "Paired_T_PValue": float(t_pvalue),
                "Wilcoxon_Stat": float(w_stat) if pd.notna(w_stat) else np.nan,
                "Wilcoxon_PValue": float(w_pvalue) if pd.notna(w_pvalue) else np.nan,
                "Periods": int(pivot.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def zscore(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    std = values.std()
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=values.index)
    return ((values - values.mean()) / std).replace([np.inf, -np.inf], np.nan)


def composite_score(mean_spearman: float, rmse: float, w_corr: float = 0.6, w_rmse: float = 0.4) -> float:
    if pd.isna(mean_spearman) or pd.isna(rmse):
        return np.nan
    return float(w_corr * mean_spearman + w_rmse * (-rmse))


def plot_topk_decay(topk: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for model, group in topk.groupby("Model"):
        group = group.sort_values("K")
        ax.plot(group["K"], group["Mean_Return"], marker="o", label=model)
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_title("Original Leadership Model: Top-K Return Decay")
    ax.set_xlabel("K")
    ax.set_ylabel("Average Actual Return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_metric_bars(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    metrics = ["Mean_Spearman", "Fit_RMSE", "Composite_Score"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric in zip(axes, metrics):
        ax.bar(summary["Model"], summary[metric])
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
