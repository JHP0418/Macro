from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "outputs" / "risk_model_walkforward_optimizer_latest" / "tables" / "risk_target_scored_panel.csv"
OUT_DIR = ROOT / "outputs" / "risk_model_vs_single_indicator_benchmark_latest"

TARGET_SPECS = {
    "nasdaq_1w_drop_2pct": {
        "label": "나스닥 1주 -2% 이상/주중 낙폭",
        "target_col": "target_nasdaq_1w_drop_2pct",
        "return_col": "NASDAQ100_fwd_1w",
        "drawdown_col": "NASDAQ100_fwd_min_1w",
        "prob_col": "prob_cal_nasdaq_1w_drop_2pct",
    },
    "nasdaq_1m_correction": {
        "label": "나스닥 1개월 조정",
        "target_col": "target_nasdaq_1m_correction",
        "return_col": "NASDAQ100_fwd_1m",
        "drawdown_col": "NASDAQ100_fwd_min_1m",
        "prob_col": "prob_cal_nasdaq_1m_correction",
    },
    "nasdaq_tail_1m": {
        "label": "나스닥 1개월 급락/tail",
        "target_col": "target_nasdaq_tail_1m",
        "return_col": "NASDAQ100_fwd_1m",
        "drawdown_col": "NASDAQ100_fwd_min_1m",
        "prob_col": "prob_cal_nasdaq_tail_1m",
    },
    "sox_1w_drop_3pct": {
        "label": "SOX 1주 -3% 이상",
        "target_col": "target_sox_1w_drop_3pct",
        "return_col": "SOX_fwd_1w",
        "drawdown_col": "SOX_fwd_min_1w",
        "prob_col": "prob_cal_sox_1w_drop_3pct",
    },
    "kospi_1w_drop_2pct": {
        "label": "KOSPI200 1주 -2% 이상",
        "target_col": "target_kospi_1w_drop_2pct",
        "return_col": "KOSPI200_fwd_1w",
        "drawdown_col": "KOSPI200_fwd_min_1w",
        "prob_col": "prob_cal_kospi_1w_drop_2pct",
    },
    "risk_assets_practical_loss_1w": {
        "label": "위험자산 유니버스 1주 실전 손실",
        "target_col": "target_risk_assets_practical_loss_1w",
        "return_col": "RISK_ASSET_fwd_1w",
        "drawdown_col": "RISK_ASSET_fwd_min_1w",
        "prob_col": "prob_cal_risk_assets_practical_loss_1w",
    },
    "risk_assets_practical_loss_1m": {
        "label": "위험자산 유니버스 1개월 실전 손실",
        "target_col": "target_risk_assets_practical_loss_1m",
        "return_col": "RISK_ASSET_fwd_1m",
        "drawdown_col": "RISK_ASSET_fwd_min_1m",
        "prob_col": "prob_cal_risk_assets_practical_loss_1m",
    },
    "safety_rotation_needed_1w": {
        "label": "1주 안전자산 우위 필요",
        "target_col": "target_safety_rotation_needed_1w",
        "return_col": "RISK_MINUS_SAFE_fwd_1w",
        "drawdown_col": "RISK_ASSET_fwd_min_1w",
        "prob_col": "prob_cal_safety_rotation_needed_1w",
    },
    "safety_rotation_needed_1m": {
        "label": "1개월 안전자산 우위 필요",
        "target_col": "target_safety_rotation_needed_1m",
        "return_col": "RISK_MINUS_SAFE_fwd_1m",
        "drawdown_col": "RISK_ASSET_fwd_min_1m",
        "prob_col": "prob_cal_safety_rotation_needed_1m",
    },
}

SCORE_SPECS = {
    "integrated_ml_prob": {"label": "통합 ML 확률", "kind": "target_prob"},
    "risk_off_avoidance_score": {"label": "통합 Risk-Off Sentinel", "col": "risk_off_avoidance_score"},
    "crash_sentinel_score": {"label": "급락 Sentinel 축", "col": "crash_sentinel_score"},
    "peak_correction_score": {"label": "고점/조정 축", "col": "peak_correction_score"},
    "composite_vector_risk": {"label": "3축 종합 위험점수", "col": "composite_vector_risk"},
    "analog_macro_risk": {"label": "유사 매크로 위험", "col": "analog_macro_risk"},
    "correction_pressure": {"label": "조정 압력", "col": "correction_pressure"},
    "rai_appetite_stress": {"label": "RAI 위험선호 붕괴", "col": "rai_appetite_stress"},
    "vix_level": {"label": "VIX 단일: 레벨", "col": "VIX"},
    "vix_shock_score": {"label": "VIX 단일: shock score", "col": "VIX_shock_score"},
    "vix_20d_change": {"label": "VIX 단일: 20일 변화율", "derived": "pct_change", "base": "VIX", "window": 20},
    "vix_60d_z": {"label": "VIX 단일: 60일 z-score", "derived": "zscore", "base": "VIX", "window": 60},
    "hy_oas_shock_score": {"label": "HY OAS 단일", "col": "HY_OAS_shock_score"},
    "dxy_shock_score": {"label": "달러 단일", "col": "DXY_shock_score"},
    "nasdaq_shock_score": {"label": "나스닥 가격충격 단일", "col": "NASDAQ100_shock_score"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare integrated risk model against VIX and other single-indicator baselines.")
    parser.add_argument("--panel", type=Path, default=PANEL)
    parser.add_argument("--output", type=Path, default=OUT_DIR)
    parser.add_argument("--min-train-days", type=int, default=504)
    parser.add_argument("--retrain-step-days", type=int, default=126)
    parser.add_argument("--purge-days", type=int, default=20)
    parser.add_argument("--embargo-days", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables = args.output / "tables"
    charts = args.output / "charts"
    reports = args.output / "reports"
    for path in (tables, charts, reports):
        path.mkdir(parents=True, exist_ok=True)

    panel = load_panel(args.panel)
    results, predictions = run_benchmark(panel, args.min_train_days, args.retrain_step_days, args.purge_days, args.embargo_days)
    comparison = build_comparison(results)

    results.to_csv(tables / "risk_model_vs_single_indicator_metrics.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(tables / "risk_model_vs_vix_summary.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(tables / "risk_model_vs_single_indicator_predictions.csv", index=False, encoding="utf-8-sig")
    plot_auc_comparison(results, charts / "auc_vs_vix_by_target.png")
    write_report(results, comparison, reports / "risk_model_vs_single_indicator_report.md")
    print((reports / "risk_model_vs_single_indicator_report.md").resolve())
    print(comparison.to_string(index=False))


def load_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    for key, spec in SCORE_SPECS.items():
        if spec.get("derived") == "pct_change":
            base = spec["base"]
            df[key] = pd.to_numeric(df[base], errors="coerce").pct_change(int(spec["window"]))
        elif spec.get("derived") == "zscore":
            base = spec["base"]
            window = int(spec["window"])
            s = pd.to_numeric(df[base], errors="coerce")
            df[key] = (s - s.rolling(window).mean()) / s.rolling(window).std()
    return df.replace([np.inf, -np.inf], np.nan)


def run_benchmark(df: pd.DataFrame, min_train: int, step: int, purge: int, embargo: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    pred_frames: list[pd.DataFrame] = []
    for target_name, target_spec in TARGET_SPECS.items():
        target_col = target_spec["target_col"]
        if target_col not in df:
            continue
        for score_name, score_spec in SCORE_SPECS.items():
            score_col = resolve_score_col(target_spec, score_spec, score_name)
            if score_col not in df:
                continue
            use = df[["Date", target_col, score_col, target_spec["return_col"], target_spec["drawdown_col"]]].copy()
            use.columns = ["Date", "target", "score", "forward_return", "forward_drawdown"]
            use["score"] = pd.to_numeric(use["score"], errors="coerce")
            use["target"] = pd.to_numeric(use["target"], errors="coerce")
            use = use.dropna(subset=["Date", "target", "score"]).reset_index(drop=True)
            if len(use) < min_train + step or use["target"].nunique() < 2:
                continue
            pred = walkforward_threshold_predictions(use, min_train, step, purge, embargo)
            metrics = score_metrics(pred)
            metrics.update(
                {
                    "target": target_name,
                    "target_label": target_spec["label"],
                    "score": score_name,
                    "score_label": score_spec["label"],
                    "score_col": score_col,
                }
            )
            rows.append(metrics)
            keep = pred.copy()
            keep["target_name"] = target_name
            keep["score"] = score_name
            pred_frames.append(keep)
    results = pd.DataFrame(rows)
    predictions = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    if not results.empty:
        cols = ["target", "target_label", "score", "score_label", "score_col"] + [c for c in results.columns if c not in {"target", "target_label", "score", "score_label", "score_col"}]
        results = results[cols].sort_values(["target", "roc_auc"], ascending=[True, False])
    return results, predictions


def resolve_score_col(target_spec: dict[str, str], score_spec: dict[str, object], score_name: str) -> str:
    if score_spec.get("kind") == "target_prob":
        return target_spec["prob_col"]
    return str(score_spec.get("col", score_name))


def walkforward_threshold_predictions(use: pd.DataFrame, min_train: int, step: int, purge: int, embargo: int) -> pd.DataFrame:
    out: list[pd.DataFrame] = []
    scores = pd.to_numeric(use["score"], errors="coerce").to_numpy(dtype=float)
    targets = pd.to_numeric(use["target"], errors="coerce").fillna(0).to_numpy(dtype=int)
    start = min_train + purge + embargo
    while start < len(use):
        train_end = max(0, start - purge - embargo)
        test = use.iloc[start : min(start + step, len(use))].copy()
        if train_end < min_train or test.empty:
            start += step
            continue
        threshold = choose_threshold_np(scores[:train_end], targets[:train_end])
        test["threshold"] = threshold
        test["signal"] = test["score"].ge(threshold).astype(int)
        out.append(test)
        start += step
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=list(use.columns) + ["threshold", "signal"])


def choose_threshold_np(score: np.ndarray, target: np.ndarray) -> float:
    mask = np.isfinite(score) & np.isfinite(target)
    valid_score = score[mask]
    valid_target = target[mask].astype(int)
    if valid_score.size == 0:
        return float("inf")
    quantiles = np.unique(np.nanquantile(valid_score, np.linspace(0.45, 0.95, 51)))
    best_threshold = float(quantiles[0])
    best_score = -1e18
    positives = int(valid_target.sum())
    negatives = int(valid_target.size - positives)
    base = max(positives / max(valid_target.size, 1), 1e-9)
    for threshold in quantiles:
        pred = valid_score >= threshold
        tp = int(np.sum(pred & (valid_target == 1)))
        fp = int(np.sum(pred & (valid_target == 0)))
        fn = positives - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        signal_rate = float(np.mean(pred))
        lift = precision / base
        objective = 1.9 * recall + 0.9 * precision + 0.25 * min(lift, 4.0) - 0.35 * signal_rate
        if objective > best_score:
            best_score = objective
            best_threshold = float(threshold)
    return best_threshold


def score_metrics(pred: pd.DataFrame) -> dict[str, object]:
    from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, precision_score, recall_score, roc_auc_score

    if pred.empty:
        return {}
    y = pred["target"].astype(int)
    signal = pred["signal"].astype(int)
    score = pd.to_numeric(pred["score"], errors="coerce")
    prob_like = score_to_percent_rank(score)
    tp = int(((signal == 1) & (y == 1)).sum())
    fp = int(((signal == 1) & (y == 0)).sum())
    tn = int(((signal == 0) & (y == 0)).sum())
    fn = int(((signal == 0) & (y == 1)).sum())
    signaled = pred[signal.eq(1)]
    return {
        "samples": int(len(pred)),
        "positive_rate": float(y.mean()),
        "signal_rate": float(signal.mean()),
        "accuracy": float(accuracy_score(y, signal)),
        "precision": float(precision_score(y, signal, zero_division=0)),
        "recall": float(recall_score(y, signal, zero_division=0)),
        "miss_rate": float(fn / max(tp + fn, 1)),
        "false_alarm_among_signals": float(fp / max(tp + fp, 1)),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "roc_auc": float(roc_auc_score(y, score)) if y.nunique() > 1 else np.nan,
        "pr_auc": float(average_precision_score(y, score)) if y.nunique() > 1 else np.nan,
        "rank_brier": float(brier_score_loss(y, prob_like.clip(0, 1))) if y.nunique() > 1 else np.nan,
        "avg_forward_return_when_signal": float(pd.to_numeric(signaled["forward_return"], errors="coerce").mean()) if not signaled.empty else np.nan,
        "avg_forward_drawdown_when_signal": float(pd.to_numeric(signaled["forward_drawdown"], errors="coerce").mean()) if not signaled.empty else np.nan,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "avg_threshold": float(pd.to_numeric(pred["threshold"], errors="coerce").mean()),
    }


def score_to_percent_rank(score: pd.Series) -> pd.Series:
    return score.rank(pct=True).fillna(0.5)


def build_comparison(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    rows = []
    for target, group in results.groupby("target"):
        integrated = pick_score(group, ["integrated_ml_prob", "risk_off_avoidance_score"])
        vix = pick_score(group, ["vix_level", "vix_shock_score", "vix_20d_change", "vix_60d_z"])
        best_single = group[group["score"].isin(["vix_level", "vix_shock_score", "vix_20d_change", "vix_60d_z", "hy_oas_shock_score", "dxy_shock_score", "nasdaq_shock_score"])].sort_values("roc_auc", ascending=False).head(1)
        if integrated.empty or vix.empty:
            continue
        row = {
            "target": target,
            "target_label": group["target_label"].iloc[0],
            "model_score": integrated["score_label"].iloc[0],
            "model_roc_auc": integrated["roc_auc"].iloc[0],
            "model_pr_auc": integrated["pr_auc"].iloc[0],
            "model_precision": integrated["precision"].iloc[0],
            "model_recall": integrated["recall"].iloc[0],
            "model_false_alarm": integrated["false_alarm_among_signals"].iloc[0],
            "vix_score": vix["score_label"].iloc[0],
            "vix_roc_auc": vix["roc_auc"].iloc[0],
            "vix_pr_auc": vix["pr_auc"].iloc[0],
            "vix_precision": vix["precision"].iloc[0],
            "vix_recall": vix["recall"].iloc[0],
            "vix_false_alarm": vix["false_alarm_among_signals"].iloc[0],
            "auc_advantage_vs_vix": integrated["roc_auc"].iloc[0] - vix["roc_auc"].iloc[0],
            "recall_advantage_vs_vix": integrated["recall"].iloc[0] - vix["recall"].iloc[0],
        }
        if not best_single.empty:
            row.update(
                {
                    "best_single_score": best_single["score_label"].iloc[0],
                    "best_single_roc_auc": best_single["roc_auc"].iloc[0],
                    "auc_advantage_vs_best_single": integrated["roc_auc"].iloc[0] - best_single["roc_auc"].iloc[0],
                }
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("auc_advantage_vs_vix", ascending=False)


def pick_score(group: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    cand = group[group["score"].isin(names)].copy()
    if cand.empty:
        return cand
    return cand.sort_values(["roc_auc", "pr_auc"], ascending=False).head(1)


def plot_auc_comparison(results: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    if results.empty:
        return
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    rows = []
    for target, group in results.groupby("target"):
        for name in ["integrated_ml_prob", "risk_off_avoidance_score", "vix_level", "vix_shock_score"]:
            one = group[group["score"].eq(name)]
            if one.empty:
                continue
            rows.append({"target": target, "score": one["score_label"].iloc[0], "roc_auc": one["roc_auc"].iloc[0]})
    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        return
    pivot = plot_df.pivot_table(index="target", columns="score", values="roc_auc", aggfunc="first")
    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
    pivot.plot(kind="bar", ax=ax, width=0.82)
    ax.axhline(0.5, color="#999999", lw=1.0, ls="--")
    ax.set_ylim(0.35, max(0.75, float(np.nanmax(pivot.values)) + 0.04))
    ax.set_ylabel("Walk-forward ROC-AUC")
    ax.set_title("통합 위험모델 vs VIX 단일지표 ROC-AUC 비교")
    ax.grid(True, axis="y", color="#e5e7eb")
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_report(results: pd.DataFrame, comparison: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Risk-Off 모델 성능평가: 통합 모델 vs VIX 단일지표",
        "",
        "동일한 라벨, 동일한 walk-forward threshold 최적화, 동일한 out-of-sample 구간으로 비교했다.",
        "급락 회피 모델은 단순 accuracy보다 recall, precision, false alarm, miss rate, ROC-AUC, PR-AUC를 함께 봐야 한다.",
        "",
        "## 핵심 비교",
        comparison_to_markdown(comparison),
        "",
        "## 전체 지표 상위권",
        results.sort_values(["target", "roc_auc"], ascending=[True, False]).groupby("target").head(5).to_markdown(index=False),
        "",
        "## 해석",
        "- VIX는 변동성 충격이 이미 가격에 반영된 뒤 강해지는 경향이 있어, 급락 진행형 탐지에는 강하지만 고점 취약성/유사 매크로/RAI/신용·환율 스트레스를 함께 보는 통합 모델보다 선행성이 약할 수 있다.",
        "- 다만 단기 1주 하락 라벨은 노이즈가 커서 통합 모델도 모든 타깃에서 압도적으로 우월하지는 않다. 1개월 조정, tail, 안전자산 로테이션처럼 손실 회피형 라벨에서 비교 우위가 더 중요하다.",
        "- false alarm은 의도적으로 낮추면 recall이 떨어진다. 현재 모델은 손실회피 목적이라 recall 가중 threshold를 썼고, 따라서 일부 오경보를 감수하는 구조다.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def comparison_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "비교 결과가 없습니다."
    use = df[
        [
            "target_label",
            "model_score",
            "model_roc_auc",
            "model_pr_auc",
            "model_precision",
            "model_recall",
            "model_false_alarm",
            "vix_score",
            "vix_roc_auc",
            "vix_pr_auc",
            "vix_precision",
            "vix_recall",
            "vix_false_alarm",
            "auc_advantage_vs_vix",
            "best_single_score",
            "best_single_roc_auc",
            "auc_advantage_vs_best_single",
        ]
    ].copy()
    for col in use.columns:
        if use[col].dtype.kind in "fc":
            use[col] = use[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    return use.to_markdown(index=False)


if __name__ == "__main__":
    main()
