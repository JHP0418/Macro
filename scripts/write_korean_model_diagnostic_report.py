from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "outputs" / "rwkv_lppl_walkforward_validation_latest" / "tables"
RWKV = ROOT / "outputs" / "rwkv_lppl_asset_screener_latest" / "tables"
REPORT_DIR = ROOT / "outputs" / "rwkv_lppl_walkforward_validation_latest" / "reports"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    panel = read_csv(VALIDATION / "walkforward_calibrated_panel.csv", parse_dates=["date"])
    summary = read_csv(VALIDATION / "walkforward_summary.csv")
    calibration = read_csv(VALIDATION / "probability_calibration.csv")
    false_alarm = read_csv(VALIDATION / "lppl_false_alarm_validation.csv")
    current = read_csv(VALIDATION / "calibrated_current_asset_scores.csv")
    meta = read_csv(VALIDATION / "meta_model_validation.csv")
    thresholds = read_csv(VALIDATION / "high_confidence_thresholds.csv")
    dtcai_thresholds = read_csv(VALIDATION / "group_dtcai_thresholds.csv")
    lppl_train = read_csv(RWKV / "lppl_reliability_training_set.csv")
    driver_selection = read_csv(RWKV / "driver_selection_granger_corr.csv")

    diagnostics = build_diagnostics(panel)
    group_diag = group_diagnostics(panel)
    regime_diag = regime_diagnostics(panel)
    top_miss = top_mistakes(panel)
    lines = report_lines(summary, calibration, false_alarm, current, lppl_train, driver_selection, diagnostics, group_diag, regime_diag, top_miss, meta, thresholds, dtcai_thresholds)
    out = REPORT_DIR / "korean_model_diagnostic_report.md"
    out.write_text("\n".join(lines), encoding="utf-8-sig")
    print(out)


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def build_diagnostics(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()
    out = []
    for horizon, prob, ret, up in [
        ("1주", "upside_prob_1w", "realized_return_1w", "realized_up_1w"),
        ("4주", "upside_prob_4w", "realized_return_4w", "realized_up_4w"),
    ]:
        pred_up = pd.to_numeric(panel[prob], errors="coerce").ge(0.5)
        actual_up = pd.to_numeric(panel[up], errors="coerce").eq(1)
        out.append(
            {
                "구간": horizon,
                "표본수": len(panel),
                "방향정확도": float((pred_up == actual_up).mean()),
                "상승예측비율": float(pred_up.mean()),
                "실제상승비율": float(actual_up.mean()),
                "예측상승_실제하락": int((pred_up & ~actual_up).sum()),
                "예측하락_실제상승": int((~pred_up & actual_up).sum()),
                "평균실현수익률": float(pd.to_numeric(panel[ret], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(out)


def group_diagnostics(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()
    g = panel.copy()
    g["pred_up_4w"] = pd.to_numeric(g["upside_prob_4w"], errors="coerce").ge(0.5)
    g["actual_up_4w"] = pd.to_numeric(g["realized_up_4w"], errors="coerce").eq(1)
    out = (
        g.groupby("group")
        .agg(
            표본수=("symbol", "size"),
            방향정확도=("pred_up_4w", lambda x: np.nan),
            평균점수=("score_0_100", "mean"),
            평균버블점수=("bubble_score_0_100", "mean"),
            평균4주수익률=("realized_return_4w", "mean"),
            실제상승비율=("actual_up_4w", "mean"),
        )
        .reset_index()
    )
    acc = g.groupby("group").apply(lambda x: (x["pred_up_4w"].to_numpy() == x["actual_up_4w"].to_numpy()).mean(), include_groups=False)
    out["방향정확도"] = out["group"].map(acc)
    return out.sort_values("방향정확도")


def regime_diagnostics(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()
    g = panel.copy()
    g["pred_up_4w"] = pd.to_numeric(g["upside_prob_4w"], errors="coerce").ge(0.5)
    g["actual_up_4w"] = pd.to_numeric(g["realized_up_4w"], errors="coerce").eq(1)
    out = (
        g.groupby("regime")
        .agg(
            표본수=("symbol", "size"),
            평균점수=("score_0_100", "mean"),
            실제상승비율=("actual_up_4w", "mean"),
            평균4주수익률=("realized_return_4w", "mean"),
        )
        .reset_index()
    )
    acc = g.groupby("regime").apply(lambda x: (x["pred_up_4w"].to_numpy() == x["actual_up_4w"].to_numpy()).mean(), include_groups=False)
    out["방향정확도"] = out["regime"].map(acc)
    return out.sort_values("방향정확도")


def top_mistakes(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()
    miss = panel[(pd.to_numeric(panel["upside_prob_4w"], errors="coerce") >= 0.6) & (pd.to_numeric(panel["realized_return_4w"], errors="coerce") < 0)].copy()
    miss["오차크기"] = pd.to_numeric(miss["upside_prob_4w"], errors="coerce") + abs(pd.to_numeric(miss["realized_return_4w"], errors="coerce"))
    cols = ["date", "symbol", "name", "group", "regime", "score_0_100", "upside_prob_4w", "calibrated_prob_4w", "bubble_score_0_100", "realized_return_4w", "오차크기"]
    return miss.sort_values("오차크기", ascending=False)[cols].head(20)


def report_lines(
    summary: pd.DataFrame,
    calibration: pd.DataFrame,
    false_alarm: pd.DataFrame,
    current: pd.DataFrame,
    lppl_train: pd.DataFrame,
    driver_selection: pd.DataFrame,
    diagnostics: pd.DataFrame,
    group_diag: pd.DataFrame,
    regime_diag: pd.DataFrame,
    top_miss: pd.DataFrame,
    meta: pd.DataFrame,
    thresholds: pd.DataFrame,
    dtcai_thresholds: pd.DataFrame,
) -> list[str]:
    strategy = summary[summary.get("series", pd.Series()).eq("strategy")] if not summary.empty else pd.DataFrame()
    bench = summary[summary.get("series", pd.Series()).eq("benchmark_069500")] if not summary.empty else pd.DataFrame()
    lppl_positive = int(pd.to_numeric(lppl_train.get("label", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not lppl_train.empty else 0
    lppl_samples = int(len(lppl_train)) if not lppl_train.empty else 0
    selected_drivers = int(driver_selection.get("selected_corr_and_granger", pd.Series(dtype=str)).astype(str).str.lower().eq("true").sum()) if not driver_selection.empty else 0

    lines = [
        "# 한국어 모델 진단 리포트",
        "",
        "## 1. 한 줄 결론",
        "",
        "현재 모델은 자산군 스크리닝 엔진으로는 유용한 신호를 만들고 있지만, 아직 기관급 운용모델이라고 보기에는 부족합니다. 가장 큰 문제는 `확률이 과신되어 있고`, `LPPL 버블 신호의 학습 라벨이 부족하며`, `초기 구간 레짐이 unknown으로 남아 walk-forward 검증에 섞인 것`입니다.",
        "",
        "## 2. 성과 요약",
    ]
    if not strategy.empty and not bench.empty:
        lines.extend(
            [
                f"- 전략 총수익률: {pct(strategy.iloc[0]['total_return'])}",
                f"- 벤치마크 총수익률: {pct(bench.iloc[0]['total_return'])}",
                f"- 전략 연환산 수익률: {pct(strategy.iloc[0]['ann_return'])}",
                f"- 벤치마크 연환산 수익률: {pct(bench.iloc[0]['ann_return'])}",
                f"- 전략 Sharpe: {num(strategy.iloc[0]['sharpe'])}",
                f"- 벤치마크 Sharpe: {num(bench.iloc[0]['sharpe'])}",
                f"- 전략 최대낙폭: {pct(strategy.iloc[0]['max_drawdown'])}",
                f"- 벤치마크 최대낙폭: {pct(bench.iloc[0]['max_drawdown'])}",
                "",
                "해석: 수익률은 벤치마크보다 낮았지만, 변동성과 낙폭이 줄면서 Sharpe는 더 높았습니다. 즉 공격적인 초과수익 모델이라기보다 방어적 위험조절 모델에 가까운 결과입니다.",
            ]
        )
    lines.extend(
        [
            "",
            "## 3. 왜 많이 틀렸나",
            "",
            "### 원인 A. 확률이 실제 확률처럼 보정되지 않았다",
            "",
            "초기 모델의 `upside_prob`는 진짜 확률이라기보다 점수형 확률입니다. calibration 결과를 보면 4주 test 구간에서 평균 예측확률은 실제 상승비율보다 낮거나 높게 어긋났고, 일부 고확률 구간은 실제 성과가 기대보다 낮았습니다.",
        ]
    )
    if not calibration.empty:
        metrics = calibration[calibration["subset"].isin(["train", "test", "all"])]
        lines.extend(["", metrics.to_markdown(index=False)])
    if not meta.empty:
        lines.extend(
            [
                "",
                "### 업그레이드 후 메타모델 성능",
                "",
                "동적 매크로/기술/LPPL 피처를 다시 학습하는 메타모델을 추가했습니다. 전체 표본 기준 0.5 단순 기준 정확도는 아직 80~90%가 아니며, 고확신 구간만 따로 골라야 합니다.",
                "",
                meta.to_markdown(index=False),
            ]
        )
    if not thresholds.empty:
        lines.extend(
            [
                "",
                "### 고확신 신호 정확도",
                "",
                "목표 정확도 80%를 맞추기 위해 확률 threshold를 올렸습니다. 전체 기준으로는 1주/4주 모두 약 76~77% 수준이었고, 레짐/자산군별 일부 구간에서만 80% 이상이 나왔습니다.",
                "",
                thresholds.sort_values("accuracy", ascending=False).head(20).to_markdown(index=False),
            ]
        )
    lines.extend(
        [
            "",
            "### 원인 B. LPPL reliability 모델이 학습되지 않았다",
            "",
            f"- LPPL 학습 표본수: {lppl_samples}",
            f"- crash positive label 수: {lppl_positive}",
            "",
            "현재 reduced 실행에서는 crash positive가 0개였습니다. 이 경우 ANN/RF/Logistic reliability 모델은 의미 있게 학습될 수 없고, fallback 신뢰도만 쓰게 됩니다. 그래서 LPPL은 과열 탐지라기보다 가격 곡선 모양에 대한 휴리스틱 패널티로 작동했습니다.",
        ]
    )
    if not false_alarm.empty:
        fa = false_alarm[false_alarm["symbol"].eq("__ALL__")]
        lines.extend(["", fa.to_markdown(index=False)])
    if not dtcai_thresholds.empty:
        lines.extend(
            [
                "",
                "자산별 adaptive DTCAI threshold도 생성했습니다. false alarm이 높은 자산은 더 높은 threshold가 필요합니다.",
                "",
                dtcai_thresholds.head(20).to_markdown(index=False),
            ]
        )
    lines.extend(
        [
            "",
            "### 원인 C. 레짐 학습 구간이 짧고 초기 walk-forward가 unknown regime으로 들어갔다",
            "",
            "RWKV 레짐은 월간 시퀀스가 충분히 쌓인 뒤부터 의미가 있습니다. 하지만 walk-forward 검증은 2019년부터 시작되어 초기 표본에는 `unknown` 레짐이 포함됩니다. 이 구간은 PDF식 RWKV 레짐 모델이 실제로 작동한 구간이라고 보기 어렵습니다.",
        ]
    )
    if not regime_diag.empty:
        lines.extend(["", regime_diag.to_markdown(index=False)])
    lines.extend(
        [
            "",
            "### 원인 D. 드라이버 선별은 아직 약하다",
            "",
            f"상관관계와 Granger 검정을 동시에 통과한 드라이버 수는 {selected_drivers}개입니다. 사용 가능한 Yahoo/FRED 기반 시장가격 프록시는 편리하지만, PDF에서 사용한 월간 실물 매크로 16개 변수와 다릅니다. 따라서 레짐 설명력은 아직 제한적입니다.",
        ]
    )
    if not driver_selection.empty:
        lines.extend(["", driver_selection.head(15).to_markdown(index=False)])
    lines.extend(
        [
            "",
            "### 원인 E. 자산군별 오답 편차가 크다",
            "",
            "모든 자산군이 같은 품질의 데이터와 같은 반응 구조를 갖지 않습니다. 특히 원자재, 채권, 환율형 ETF는 주식형 모멘텀/레짐 신호와 다르게 움직일 때가 많습니다.",
        ]
    )
    if not group_diag.empty:
        lines.extend(["", group_diag.to_markdown(index=False)])
    lines.extend(
        [
            "",
            "## 4. 대표 오답 사례",
            "",
            "아래는 4주 상승확률을 60% 이상으로 봤지만 실제 4주 수익률이 음수였던 사례입니다.",
        ]
    )
    if not top_miss.empty:
        lines.extend(["", top_miss.to_markdown(index=False)])
    lines.extend(
        [
            "",
            "## 5. 지금 당장 고쳐야 할 우선순위",
            "",
            "1. RWKV 레짐이 실제로 생성된 날짜 이후만 walk-forward 검증에 사용해야 합니다.",
            "2. LPPL crash positive label이 충분히 생기도록 전체 rolling window와 충분한 parameter archive를 생성해야 합니다.",
            "3. 확률 calibration을 자산군별 또는 레짐별로 분리해야 합니다.",
            "4. LPPL false alarm이 높은 자산군은 DTCAI threshold를 자산군별로 다르게 둬야 합니다.",
            "5. KODEX 200 하나가 아니라 KOSPI, Nasdaq, SOX, 채권, 금 등 각 자산군 기준으로 Granger/상관 driver 선별을 따로 해야 합니다.",
            "6. 월간 실물 매크로 변수, 한국 수출/반도체 수출, PMI/ISM, 신용스프레드 계열을 더 넣어야 합니다.",
            "",
            "## 6. 결론",
            "",
            "현재 모델은 `방향성 스크리닝 + 위험조절`에는 쓸 수 있지만, 아직 확률을 그대로 믿고 매매하기에는 이릅니다. 많이 틀린 핵심 이유는 모델 구조보다 데이터/검증 단계의 미완성입니다. 특히 LPPL reliability 학습 라벨 부족과 초기 unknown regime 혼입이 가장 큽니다.",
        ]
    )
    return lines


def pct(value) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "n/a"


def num(value) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return "n/a"


if __name__ == "__main__":
    main()
