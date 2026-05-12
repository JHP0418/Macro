from __future__ import annotations

import html
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "screening_dashboard_latest"
TABLE_DIR = OUT_DIR / "best_model_tables"
CHART_DIR = OUT_DIR / "charts_best"

PATHS = {
    "driver_panel": ROOT / "outputs" / "macro_regime_asset_screener_latest" / "tables" / "driver_panel.csv",
    "asset_scores": ROOT / "outputs" / "macro_regime_asset_screener_latest" / "tables" / "current_asset_scores.csv",
    "risk_v4_current": ROOT / "outputs" / "risk_off_v4_event_label_latest" / "tables" / "current_risk_off_v4_state.csv",
    "risk_v4_predictions": ROOT / "outputs" / "risk_off_v4_event_label_latest" / "tables" / "risk_off_v4_walkforward_predictions.csv",
    "risk_ssl2_predictions": ROOT / "outputs" / "ssl2_head_backtest_latest" / "tables" / "risk_ssl2_predictions.csv",
    "risk_ssl2_metrics": ROOT / "outputs" / "ssl2_head_backtest_latest" / "tables" / "risk_ssl2_metrics.csv",
    "risk_adoption": ROOT / "outputs" / "ssl2_head_backtest_latest" / "tables" / "operational_model_adoption.csv",
    "etf_current": ROOT / "outputs" / "gaps_long_lived_etf_leadership_latest" / "tables" / "long_lived_current_scores.csv",
    "ema_entry": ROOT / "outputs" / "ema_entry_meta_model_latest" / "tables" / "current_ema_entry_meta_signal.csv",
    "ema_perf": ROOT / "outputs" / "ema_entry_meta_model_latest" / "tables" / "ema_entry_meta_backtest_summary.csv",
    "safe_perf": ROOT / "outputs" / "institutional_risk_off_v2_latest" / "tables" / "macro_conditioned_safe_asset_summary.csv",
    "hybrid_perf": ROOT / "outputs" / "hybrid_proxy_gaps_etf_leadership_latest" / "tables" / "hybrid_proxy_gaps_summary.csv",
    "grade_summary": ROOT / "outputs" / "a_grade_model_upgrade_latest" / "tables" / "component_grade_summary.csv",
    "dynamic_current": ROOT / "outputs" / "dynamic_risk_gated_allocation_latest" / "tables" / "current_dynamic_allocation.csv",
    "dynamic_summary": ROOT / "outputs" / "dynamic_risk_gated_allocation_latest" / "tables" / "dynamic_allocation_summary.csv",
}


SAFE_BASKETS = {
    "국내채권_종합",
    "국내채권_회사채",
    "해외채권_종합",
    "해외채권_회사채",
    "금리연계형 및 초단기채권",
    "FX및 원자재",
}


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def pct(x: object, digits: int = 1) -> str:
    try:
        v = float(x)
        if not np.isfinite(v):
            return "-"
        return f"{v * 100:.{digits}f}%"
    except Exception:
        return "-"


def num(x: object, digits: int = 2) -> str:
    try:
        v = float(x)
        if not np.isfinite(v):
            return "-"
        return f"{v:.{digits}f}"
    except Exception:
        return "-"


def esc(x: object) -> str:
    if pd.isna(x):
        return "-"
    return html.escape(str(x))


def latest(frame: pd.DataFrame, date_col: str) -> pd.DataFrame:
    if frame.empty or date_col not in frame.columns:
        return pd.DataFrame()
    x = frame.copy()
    x[date_col] = pd.to_datetime(x[date_col])
    return x[x[date_col].eq(x[date_col].max())].copy()


def load_data() -> dict[str, pd.DataFrame]:
    data = {}
    for key, path in PATHS.items():
        parse = ["Date"] if key == "driver_panel" else ["asof"] if key == "asset_scores" else ["date"]
        try:
            data[key] = read_csv(path, parse_dates=parse, low_memory=False)
        except Exception:
            data[key] = read_csv(path, low_memory=False)
    return data


def current_risk_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    v4 = data["risk_v4_current"].copy()
    if v4.empty:
        return pd.DataFrame()
    v4["date"] = pd.to_datetime(v4["date"])
    cols = [
        "date",
        "horizon",
        "risk_off_v4_prob",
        "risk_off_v4_watch_threshold",
        "risk_off_v4_derisk_threshold",
        "risk_off_v4_cash_threshold",
        "risk_off_v4_stage",
        "risk_off_score",
        "axis1_vol_credit_stress",
        "axis2_fx_liquidity_stress",
        "axis3_peak_fragility_stress",
        "risk_3d_dominant_axis",
    ]
    out = v4[[c for c in cols if c in v4.columns]].copy()
    out["기간"] = out["horizon"].map({"1w": "1주", "1m": "1개월"}).fillna(out["horizon"])
    out["Risk-Off 확률"] = out["risk_off_v4_prob"].map(lambda x: pct(x))
    out["단계"] = out["risk_off_v4_stage"]
    out["종합 위험점수"] = out["risk_off_score"].map(lambda x: num(x, 1))
    out["1축 변동성/신용"] = out["axis1_vol_credit_stress"].map(lambda x: num(x, 1))
    out["2축 달러/유동성"] = out["axis2_fx_liquidity_stress"].map(lambda x: num(x, 1))
    out["3축 고점취약성"] = out["axis3_peak_fragility_stress"].map(lambda x: num(x, 1))
    out["주요 위험축"] = out.get("risk_3d_dominant_axis", "")
    return out[["date", "기간", "Risk-Off 확률", "단계", "종합 위험점수", "1축 변동성/신용", "2축 달러/유동성", "3축 고점취약성", "주요 위험축"]]


def ssl2_best_risk_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pred = data["risk_ssl2_predictions"].copy()
    metrics = data["risk_ssl2_metrics"].copy()
    adoption = data["risk_adoption"].copy()
    if pred.empty or metrics.empty or adoption.empty:
        return pd.DataFrame()
    names = {
        "label_large_loss_1w": "1주 큰 손실",
        "label_large_loss_1m": "1개월 큰 손실",
        "label_nasdaq_down_1w": "나스닥 1주 하락",
        "label_nasdaq_down_1m": "나스닥 1개월 하락",
    }
    rows = []
    pred["date"] = pd.to_datetime(pred["date"])
    for _, a in adoption[adoption["task"].eq("risk_off")].iterrows():
        label = str(a["target"])
        model = str(a["best_model"] if a["decision"] == "adopt" else a["baseline_model"])
        p = pred[pred["label"].eq(label) & pred["risk_model"].eq(model)].sort_values("date")
        m = metrics[metrics["label"].eq(label) & metrics["model"].eq(model)]
        if p.empty or m.empty:
            continue
        last = p.iloc[-1]
        met = m.iloc[0]
        rows.append(
            {
                "date": last["date"],
                "위험 라벨": names.get(label, label),
                "채택 모델": model,
                "현재 확률": pct(last["prob"]),
                "운영 기준": pct(met["threshold"]),
                "판정": "위험" if float(last["prob"]) >= float(met["threshold"]) else "정상",
                "검증 AUC": num(met["test_auc"], 3),
                "Recall": num(met["test_recall"], 3),
                "Precision": num(met["test_precision"], 3),
            }
        )
    return pd.DataFrame(rows).sort_values("현재 확률", ascending=False)


def etf_tables(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    etf = latest(data["etf_current"], "date")
    if etf.empty:
        return pd.DataFrame(), pd.DataFrame()
    for c in ["rule_5d_score", "rule_20d_score", "ETF_RS_20D", "ETF_RS_60D", "weighted_HP", "MA60_breadth", "MA200_breadth", "reg_r2"]:
        if c in etf.columns:
            etf[c] = pd.to_numeric(etf[c], errors="coerce")
    base_cols = ["date", "etf_ticker", "name", "asset_basket", "group", "rule_5d_score", "rule_20d_score", "ETF_RS_20D", "ETF_RS_60D", "weighted_HP", "MA60_breadth", "MA200_breadth", "reg_r2"]
    one_w = etf.sort_values("rule_5d_score", ascending=False)[base_cols].head(15).copy()
    one_m = etf.sort_values("rule_20d_score", ascending=False)[base_cols].head(15).copy()
    return one_w, one_m


def safe_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    scores = latest(data["asset_scores"], "asof")
    if scores.empty:
        return pd.DataFrame()
    safe = scores[scores["basket"].isin(SAFE_BASKETS)].copy()
    if safe.empty:
        return pd.DataFrame()
    cols = ["asof", "symbol", "name", "basket", "score_0_100", "upside_prob_1w", "upside_prob_4w", "return_20d", "volatility_20d_ann", "risk_penalty"]
    return safe.sort_values("score_0_100", ascending=False)[[c for c in cols if c in safe.columns]].head(15)


def all_asset_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    scores = latest(data["asset_scores"], "asof")
    if scores.empty:
        return pd.DataFrame()
    cols = ["asof", "symbol", "name", "basket", "score_0_100", "upside_prob_1w", "upside_prob_4w", "current_regime", "return_20d", "risk_penalty"]
    return scores.sort_values("score_0_100", ascending=False)[[c for c in cols if c in scores.columns]].head(20)


def entry_signal_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    entry = data["ema_entry"].copy()
    if entry.empty:
        return pd.DataFrame()
    entry["date"] = pd.to_datetime(entry["date"])
    cols = [
        "date",
        "selected_names",
        "entry_prob_1w",
        "entry_threshold_1w",
        "action",
        "ema_trend_share",
        "close_above_ema20_share",
        "dist_to_ema20_mean",
        "risk_off_prob",
        "model_backtest_sharpe",
        "model_backtest_mdd",
        "model_backtest_hit_positive",
        "model_backtest_hit_excess",
    ]
    return entry[[c for c in cols if c in entry.columns]].copy()


def performance_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    risk = data["grade_summary"].copy()
    if not risk.empty:
        for comp in ["Risk-Off label_large_loss_1m", "Risk-Off label_large_loss_1w"]:
            part = risk[risk["component"].eq(comp)]
            if not part.empty:
                r = part.iloc[0]
                rows.append({"모듈": r["component"], "채택 모델": r["selected_model"], "성능": r["actual_summary"], "등급": r["grade"]})
    ema = data["ema_perf"].copy()
    if not ema.empty:
        best = ema[(ema["strategy"] == "ranker_top3_hybrid_1w") & (ema["model"] == "lightgbm_platt_calibrated_conservative")]
        if not best.empty:
            r = best.iloc[0]
            rows.append(
                {
                    "모듈": "ETF 1주 진입/대기 Meta",
                    "채택 모델": "LightGBM Platt Calibrated Conservative + EMA 4/6/20",
                    "성능": f"Sharpe={num(r['Sharpe'], 2)}, MDD={pct(r['MDD'])}, 양수적중={pct(r['hit_positive'])}, 초과적중={pct(r['hit_excess'])}",
                    "등급": "B+",
                }
            )
    safe = data["safe_perf"].copy()
    if not safe.empty:
        row = safe.sort_values("beat_safe_average_rate", ascending=False).head(1)
        if not row.empty:
            r = row.iloc[0]
            rows.append(
                {
                    "모듈": "안전자산 선택",
                    "채택 모델": "Macro-conditioned Safe Ranker V2",
                    "성능": f"안전자산 평균 초과={pct(r['beat_safe_average_rate'])}, 선택 평균수익={pct(r['avg_picked_return'])}",
                    "등급": "A",
                }
            )
    return pd.DataFrame(rows)


def dynamic_current_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    x = data["dynamic_current"].copy()
    if x.empty:
        return pd.DataFrame()
    x["date"] = pd.to_datetime(x["date"])
    return x


def dynamic_summary_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    x = data["dynamic_summary"].copy()
    if x.empty:
        return pd.DataFrame()
    order = [
        "dynamic_risk_gated_allocation",
        "entry_meta_switch_risk_or_safe",
        "qqq_benchmark",
        "static_60_risk_40_safe",
        "always_risk_top3",
    ]
    x["strategy_order"] = x["strategy"].map({s: i for i, s in enumerate(order)}).fillna(99)
    return x.sort_values("strategy_order").drop(columns=["strategy_order"])


def create_charts(data: dict[str, pd.DataFrame]) -> list[Path]:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    charts = []
    driver = data["driver_panel"].copy()
    risk = data["risk_v4_predictions"].copy()
    if driver.empty or risk.empty:
        return charts
    driver["Date"] = pd.to_datetime(driver["Date"])
    risk["date"] = pd.to_datetime(risk["date"])
    r1m = risk[risk["horizon"].eq("1m")].copy()
    merged = pd.merge_asof(
        driver.sort_values("Date"),
        r1m[["date", "risk_off_v4_prob", "axis1_vol_credit_stress", "axis2_fx_liquidity_stress", "axis3_peak_fragility_stress"]].sort_values("date"),
        left_on="Date",
        right_on="date",
        direction="backward",
    )
    for months, name in [(12, "최근 12개월"), (6, "최근 6개월")]:
        start = merged["Date"].max() - pd.DateOffset(months=months)
        x = merged[merged["Date"].ge(start)].copy()
        if x.empty:
            continue
        fig, ax1 = plt.subplots(figsize=(13, 5))
        for col, label in [("NASDAQ100", "나스닥100"), ("KOSPI200", "KOSPI200"), ("GOLD", "금"), ("WTI", "WTI")]:
            if col in x.columns:
                y = pd.to_numeric(x[col], errors="coerce")
                y = y / y.dropna().iloc[0] * 100 if y.notna().any() else y
                ax1.plot(x["Date"], y, label=label, linewidth=1.8)
        ax1.set_ylabel("지수화 = 100")
        ax1.grid(alpha=0.25)
        ax2 = ax1.twinx()
        ax2.plot(x["Date"], pd.to_numeric(x["risk_off_v4_prob"], errors="coerce") * 100, color="#d62728", label="Risk-Off 확률", linewidth=2.2, alpha=0.85)
        ax2.set_ylabel("Risk-Off 확률(%)")
        ax2.set_ylim(0, 100)
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper left", ncol=5)
        ax1.set_title(f"{name} 시장지수와 최신 Risk-Off Sentinel")
        fig.tight_layout()
        path = CHART_DIR / f"latest_risk_market_{months}m.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        charts.append(path)
    return charts


def table_html(df: pd.DataFrame, columns: list[tuple[str, str, str | None]], limit: int | None = None) -> str:
    if df.empty:
        return "<p class='muted'>데이터 없음</p>"
    x = df.head(limit).copy() if limit else df.copy()
    head = "".join(f"<th>{esc(label)}</th>" for _, label, _ in columns)
    rows = []
    for _, row in x.iterrows():
        cells = []
        for col, _, kind in columns:
            val = row.get(col, "")
            if kind == "pct":
                text = pct(val)
            elif kind == "num":
                text = num(val)
            elif kind == "date":
                text = pd.to_datetime(val).strftime("%Y-%m-%d") if not pd.isna(val) else "-"
            else:
                text = esc(val)
            cells.append(f"<td>{text}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def render(data: dict[str, pd.DataFrame], charts: list[Path]) -> str:
    risk = current_risk_table(data)
    ssl2 = ssl2_best_risk_table(data)
    entry = entry_signal_table(data)
    etf_1w, etf_1m = etf_tables(data)
    safe = safe_table(data)
    assets = all_asset_table(data)
    perf = performance_table(data)
    dyn_current = dynamic_current_table(data)
    dyn_summary = dynamic_summary_table(data)

    asof_values = []
    for frame, col in [(risk, "date"), (entry, "date"), (assets, "asof")]:
        if not frame.empty and col in frame.columns:
            asof_values.append(pd.to_datetime(frame[col]).max())
    asof = max(asof_values).strftime("%Y-%m-%d") if asof_values else "-"

    risk_cards = ""
    if not risk.empty:
        for _, r in risk.iterrows():
            cls = "danger" if str(r["단계"]).lower() in {"cash", "de-risk"} else "ok"
            risk_cards += f"""
            <div class="card {cls}">
              <div class="label">{esc(r['기간'])} Risk-Off</div>
              <div class="big">{esc(r['Risk-Off 확률'])}</div>
              <div>{esc(r['단계'])} · 주요축: {esc(r['주요 위험축'])}</div>
            </div>"""

    entry_card = ""
    if not entry.empty:
        e = entry.iloc[0]
        entry_card = f"""
        <div class="card primary">
          <div class="label">1주 ETF 진입/대기 Meta</div>
          <div class="big">{pct(e['entry_prob_1w'])}</div>
          <div>기준 {pct(e['entry_threshold_1w'])} · 판정 <b>{esc(e['action'])}</b></div>
          <div class="small">Top3: {esc(e['selected_names'])}</div>
        </div>"""

    allocation_card = ""
    if not dyn_current.empty:
        d = dyn_current.iloc[0]
        allocation_card = f"""
        <div class="card primary">
          <div class="label">동적 리밸런싱 권고</div>
          <div class="big">위험 {pct(d['risk_weight'])}</div>
          <div>안전자산 {pct(d['safe_weight'])} · 현금 {pct(d['cash_weight'])}</div>
          <div class="small">상태: {esc(d['risk_stage'])}</div>
        </div>"""

    chart_html = "".join(f"<img class='chart' src='charts_best/{p.name}' alt='{p.name}'>" for p in charts)

    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>최신 운영모델 스크리닝</title>
  <style>
    body {{ margin:0; font-family: Arial, 'Malgun Gothic', sans-serif; background:#f6f7f9; color:#172033; }}
    .wrap {{ max-width: 1480px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 6px; font-size: 30px; }}
    h2 {{ margin: 28px 0 12px; font-size: 21px; }}
    .muted {{ color:#667085; }}
    .grid {{ display:grid; grid-template-columns: repeat(3, 1fr); gap:14px; margin-top:18px; }}
    .card {{ background:white; border:1px solid #d8dee8; border-radius:8px; padding:16px; box-shadow:0 1px 2px rgba(16,24,40,.04); }}
    .card.primary {{ border-color:#1f77b4; }}
    .card.danger {{ border-color:#d62728; }}
    .card.ok {{ border-color:#2ca02c; }}
    .label {{ color:#667085; font-size:13px; margin-bottom:8px; }}
    .big {{ font-size:34px; font-weight:800; margin-bottom:6px; }}
    .small {{ color:#667085; font-size:12px; margin-top:8px; line-height:1.4; }}
    table {{ width:100%; border-collapse:collapse; background:white; border:1px solid #d8dee8; border-radius:8px; overflow:hidden; }}
    th, td {{ padding:10px 11px; border-bottom:1px solid #edf0f5; text-align:left; font-size:13px; white-space:nowrap; }}
    th {{ background:#eef2f7; color:#344054; font-weight:700; }}
    tr:last-child td {{ border-bottom:none; }}
    .section {{ overflow-x:auto; }}
    .chart {{ width:100%; background:white; border:1px solid #d8dee8; border-radius:8px; margin:8px 0 16px; }}
    .note {{ background:#fff; border-left:4px solid #1f77b4; padding:12px 14px; margin:16px 0; color:#344054; }}
    .explain {{ background:#ffffff; border:1px solid #d8dee8; border-radius:8px; padding:14px 16px; margin:10px 0 14px; line-height:1.55; color:#344054; }}
    .explain b {{ color:#172033; }}
    .glossary {{ display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:10px; margin-top:10px; }}
    .term {{ background:#f8fafc; border:1px solid #edf0f5; border-radius:8px; padding:10px 12px; font-size:13px; line-height:1.45; }}
    .term b {{ display:block; margin-bottom:4px; }}
  </style>
</head>
<body>
<div class="wrap">
  <h1>최신 운영모델 스크리닝</h1>
  <div class="muted">기준일 {esc(asof)} · 과거 후보/비채택 모델 정보 제거 · 최신 채택 모델 결과만 표시</div>
  <div class="grid">{risk_cards}{entry_card}{allocation_card}</div>

  <div class="note">
    현재 1주 진입/대기 모델은 <b>LightGBM Platt Calibrated Conservative + EMA 4/6/20</b>를 사용합니다.
    ETF Ranker가 고른 Top3에 대해 EMA 정렬, 20EMA 이격, 후보군 점수 품질, Risk-Off 축을 함께 보고 진입 여부를 판단합니다.
  </div>

  <h2>한눈에 읽는 법</h2>
  <div class="explain">
    이 화면은 <b>시장 위험을 먼저 확인</b>하고, 그다음 <b>이번 주 진입해도 되는지</b>, 마지막으로 <b>어떤 ETF와 안전자산이 상대적으로 좋은지</b>를 보는 순서로 설계했습니다.
    Risk-Off 확률이 높으면 공격적인 ETF 점수가 좋아도 비중을 줄이거나 안전자산 후보를 먼저 봐야 합니다.
    반대로 Risk-Off가 낮고 1주 진입확률이 기준보다 높으면 ETF 리더십 상위 후보를 검토합니다.
    모든 점수는 매수 확정 신호가 아니라, 같은 날 후보들끼리 비교하기 위한 스크리닝 신호입니다.
  </div>
  <div class="glossary">
    <div class="term"><b>Risk-Off 확률</b>향후 1주 또는 1개월 동안 위험자산이 흔들릴 가능성입니다. 높을수록 현금, 단기채, 달러, 금 같은 방어 후보를 우선합니다.</div>
    <div class="term"><b>진입확률</b>ETF Top3에 지금 들어갔을 때 1주 동안 수익, 초과수익, 단기 낙폭 조건을 동시에 만족할 가능성입니다.</div>
    <div class="term"><b>ETF 리더십</b>ETF 자체 상대강도, 구성종목 고점근접도, breadth, 내부 회귀 구조를 합친 상대 순위 점수입니다.</div>
    <div class="term"><b>안전자산 후보</b>Risk-Off 또는 대기 구간에서 들고 갈 만한 채권, 금리형, 원자재/FX 계열 후보입니다.</div>
  </div>

  <h2>시장 차트</h2>
  <div class="explain">
    최근 12개월과 6개월 동안 나스닥, KOSPI200, 금, 유가를 100으로 맞춰 비교한 차트입니다.
    빨간 선은 Risk-Off 확률입니다. 지수는 오르는데 Risk-Off 확률도 같이 올라가면 <b>상승 중 고점취약성</b>이 커지는 구간으로 해석합니다.
  </div>
  {chart_html}

  <h2>Risk-Off Sentinel V4 현재상태</h2>
  <div class="explain">
    시장 전체의 방어 필요성을 보는 핵심 표입니다.
    <b>Normal</b>은 정상, <b>De-risk</b>는 위험자산 비중 축소, <b>Cash</b>는 현금/안전자산 우선 상태입니다.
    1축은 변동성·신용, 2축은 달러·환율·유동성, 3축은 고점권 피로와 과열 취약성을 뜻합니다.
    주요 위험축이 무엇인지에 따라 안전자산 선택도 달라집니다.
  </div>
  <div class="section">{table_html(risk, [('date','기준일','date'),('기간','기간',None),('Risk-Off 확률','Risk-Off 확률',None),('단계','단계',None),('종합 위험점수','종합 위험점수',None),('1축 변동성/신용','1축 변동성/신용',None),('2축 달러/유동성','2축 달러/유동성',None),('3축 고점취약성','3축 고점취약성',None),('주요 위험축','주요 위험축',None)])}</div>

  <h2>SSL2 Risk-Off 보조 헤드</h2>
  <div class="explain">
    Risk-Off Sentinel을 보조하는 딥러닝/머신러닝 헤드입니다.
    큰 손실, 나스닥 하락 같은 라벨별로 현재 확률이 운영 기준을 넘는지 봅니다.
    이 표는 주 판단이 아니라 <b>위험 경고가 여러 모델에서 동시에 나오는지 확인하는 보조 확인표</b>입니다.
  </div>
  <div class="section">{table_html(ssl2, [('date','기준일','date'),('위험 라벨','위험 라벨',None),('채택 모델','채택 모델',None),('현재 확률','현재 확률',None),('운영 기준','운영 기준',None),('판정','판정',None),('검증 AUC','검증 AUC',None),('Recall','Recall',None),('Precision','Precision',None)])}</div>

  <h2>1주 진입/대기 Meta Signal</h2>
  <div class="explain">
    ETF 리더십 모델이 고른 Top3를 이번 주에 바로 살지 판단합니다.
    <b>진입확률</b>이 <b>진입기준</b>보다 높으면 진입, 낮으면 대기 또는 안전자산을 우선합니다.
    EMA 정렬비율은 Top3 중 4EMA &gt; 6EMA &gt; 20EMA 조건을 만족하는 비율이고, 20EMA 위 비율은 단기 추세가 훼손되지 않았는지 보는 값입니다.
    20EMA 평균이격이 너무 크면 추세는 강하지만 단기 과열 가능성도 함께 봐야 합니다.
  </div>
  <div class="section">{table_html(entry, [('date','기준일','date'),('selected_names','Top3 ETF',None),('entry_prob_1w','진입확률','pct'),('entry_threshold_1w','진입기준','pct'),('action','판정',None),('ema_trend_share','EMA 정렬비율','pct'),('close_above_ema20_share','20EMA 위 비율','pct'),('dist_to_ema20_mean','20EMA 평균이격','pct'),('risk_off_prob','Risk-Off 확률','pct')])}</div>

  <h2>동적 자산배분 리밸런싱</h2>
  <div class="explain">
    Risk-Off Sentinel V4가 위험예산을 정하고, SSL2 보조 헤드가 큰 손실 경고를 확인하며, 1주 진입/대기 Meta Signal이 위험자산 진입 허가를 냅니다.
    위험자산은 ETF 리더십 Top3, 안전자산은 Macro-conditioned Safe Ranker Top3를 사용합니다.
    개별 ETF 20% 상한을 반영해 Top3 슬리브는 최대 60%까지만 배정하고 남는 비중은 현금으로 둡니다.
  </div>
  <div class="section">{table_html(dyn_current, [('date','기준일','date'),('risk_stage','위험단계',None),('entry_prob_1w','진입확률','pct'),('risk_weight','위험자산 비중','pct'),('safe_weight','안전자산 비중','pct'),('cash_weight','현금 비중','pct'),('risk_assets','위험자산 후보',None),('safe_assets','안전자산 후보',None)])}</div>

  <h2>동적 배분 백테스트</h2>
  <div class="explain">
    `dynamic_risk_gated_allocation`은 현재 운영 규칙 그대로의 주간 리밸런싱 결과입니다.
    QQQ 벤치마크와 비교하면 원수익률은 낮지만, 최대낙폭을 크게 줄이고 Sharpe를 높이는 방어형 성과가 나왔습니다.
    초과수익을 더 노리려면 Normal 구간의 위험자산 후보 수를 Top5로 늘려 20% 개별상한 안에서 위험자산 비중을 더 쓸 수 있게 보강해야 합니다.
  </div>
  <div class="section">{table_html(dyn_summary, [('strategy','전략',None),('start','시작',None),('end','종료',None),('periods','주간수',None),('cumulative_return','누적수익률','pct'),('CAGR','CAGR','pct'),('Sharpe','Sharpe','num'),('MDD','MDD','pct'),('hit_positive','양수적중률','pct'),('hit_excess','초과수익 적중률','pct'),('avg_excess','평균 초과수익','pct')])}</div>

  <h2>ETF 리더십 Top 15 - 1주</h2>
  <div class="explain">
    이번 주 기준으로 상대적으로 강한 ETF 후보입니다.
    1주 리더십은 단기 상대강도와 구성종목 내부 강도를 합친 점수입니다.
    20일 상대강도는 기준지수 대비 얼마나 앞섰는지, 고점근접도는 구성종목이 52주 고점에 얼마나 가까운지, 60일선 위 비중은 상승 참여도가 넓은지 보여줍니다.
  </div>
  <div class="section">{table_html(etf_1w, [('date','기준일','date'),('etf_ticker','ETF',None),('name','이름',None),('asset_basket','바스켓',None),('group','그룹',None),('rule_5d_score','1주 리더십','num'),('ETF_RS_20D','20일 상대강도','pct'),('weighted_HP','고점근접도','pct'),('MA60_breadth','60일선 위 비중','pct'),('reg_r2','내부 회귀 R2','num')])}</div>

  <h2>ETF 리더십 Top 15 - 1개월</h2>
  <div class="explain">
    1개월 보유 관점의 ETF 순위입니다.
    1주 표보다 노이즈가 적고, 중기 추세와 구성종목 breadth가 더 중요합니다.
    200일선 위 비중이 높으면 장기 추세가 살아 있는 ETF이고, 내부 회귀 R2가 높으면 ETF 상승이 구성종목 특성으로 잘 설명되는 상태입니다.
  </div>
  <div class="section">{table_html(etf_1m, [('date','기준일','date'),('etf_ticker','ETF',None),('name','이름',None),('asset_basket','바스켓',None),('group','그룹',None),('rule_20d_score','1개월 리더십','num'),('ETF_RS_60D','60일 상대강도','pct'),('weighted_HP','고점근접도','pct'),('MA200_breadth','200일선 위 비중','pct'),('reg_r2','내부 회귀 R2','num')])}</div>

  <h2>안전자산 후보</h2>
  <div class="explain">
    Risk-Off, 대기, 변동성 확대 구간에서 우선 검토할 후보입니다.
    투자매력도는 가격 추세, 변동성, 매크로 환경, 위험감점을 반영합니다.
    단기채와 금리형은 변동성 방어, 금·달러·원자재는 위험 원인에 따라 방어력이 달라질 수 있습니다.
  </div>
  <div class="section">{table_html(safe, [('asof','기준일','date'),('symbol','티커',None),('name','이름',None),('basket','바스켓',None),('score_0_100','투자매력도','num'),('upside_prob_1w','1주 상승확률','pct'),('upside_prob_4w','1개월 상승확률','pct'),('return_20d','20일 수익률','pct'),('volatility_20d_ann','20일 연율변동성','pct'),('risk_penalty','위험감점','num')])}</div>

  <h2>전체 자산 투자매력도 Top 20</h2>
  <div class="explain">
    위험자산과 안전자산을 모두 포함한 종합 순위입니다.
    투자매력도는 자체 모멘텀, 매크로 적합도, 유사국면 승률, 변동성 감점, Risk-Off 감점을 합친 0~100 점수입니다.
    이 표는 최종 포트폴리오 후보를 넓게 보는 용도이고, 실제 진입은 위의 Risk-Off와 진입/대기 신호를 함께 봐야 합니다.
  </div>
  <div class="section">{table_html(assets, [('asof','기준일','date'),('symbol','티커',None),('name','이름',None),('basket','바스켓',None),('score_0_100','투자매력도','num'),('upside_prob_1w','1주 상승확률','pct'),('upside_prob_4w','1개월 상승확률','pct'),('current_regime','현재 Regime',None),('return_20d','20일 수익률','pct'),('risk_penalty','위험감점','num')])}</div>

  <h2>채택 모델 성능</h2>
  <div class="explain">
    현재 HTML에 표시되는 모델만 요약한 성능표입니다.
    Sharpe는 위험 대비 수익, MDD는 최대낙폭, Recall은 실제 위험을 놓치지 않는 능력, Precision은 위험 경고가 맞을 확률에 가깝게 보면 됩니다.
    Risk-Off 모델은 수익률 예측보다 손실 회피가 목적이므로 Recall을 특히 중요하게 봅니다.
  </div>
  <div class="section">{table_html(perf, [('모듈','모듈',None),('채택 모델','채택 모델',None),('성능','성능',None),('등급','등급',None)])}</div>
</div>
</body>
</html>"""


def generate(output: Path = OUT_DIR) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    charts = create_charts(data)
    current_risk_table(data).to_csv(TABLE_DIR / "current_risk_off_v4.csv", index=False, encoding="utf-8-sig")
    ssl2_best_risk_table(data).to_csv(TABLE_DIR / "current_ssl2_risk_heads.csv", index=False, encoding="utf-8-sig")
    entry_signal_table(data).to_csv(TABLE_DIR / "current_ema_entry_signal.csv", index=False, encoding="utf-8-sig")
    etf_1w, etf_1m = etf_tables(data)
    etf_1w.to_csv(TABLE_DIR / "current_etf_leadership_1w.csv", index=False, encoding="utf-8-sig")
    etf_1m.to_csv(TABLE_DIR / "current_etf_leadership_1m.csv", index=False, encoding="utf-8-sig")
    safe_table(data).to_csv(TABLE_DIR / "current_safe_candidates.csv", index=False, encoding="utf-8-sig")
    all_asset_table(data).to_csv(TABLE_DIR / "current_all_asset_scores.csv", index=False, encoding="utf-8-sig")
    performance_table(data).to_csv(TABLE_DIR / "selected_model_performance.csv", index=False, encoding="utf-8-sig")
    dynamic_current_table(data).to_csv(TABLE_DIR / "current_dynamic_allocation.csv", index=False, encoding="utf-8-sig")
    dynamic_summary_table(data).to_csv(TABLE_DIR / "dynamic_allocation_summary.csv", index=False, encoding="utf-8-sig")
    out = output / "screening_dashboard.html"
    out.write_text(render(data, charts), encoding="utf-8")
    return out


def main() -> None:
    print(generate().resolve())


if __name__ == "__main__":
    main()
