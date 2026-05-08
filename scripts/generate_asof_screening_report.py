from __future__ import annotations

import html
from pathlib import Path
import argparse

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


KR_GROUP = {
    "Korea semiconductor": "한국 반도체",
    "Korea broad equity": "한국 대표지수",
    "Korea cyclical": "한국 경기민감",
    "Korea growth": "한국 성장주",
    "Korea IT": "한국 IT",
    "Korea value": "한국 가치",
    "Korea defensive": "한국 방어",
    "Korea bonds": "한국 채권",
    "US growth": "미국 성장주",
    "US semiconductor": "미국 반도체",
    "US broad equity": "미국 대표지수",
    "US cyclical": "미국 경기민감",
    "US REIT": "미국 리츠",
    "US IG bonds": "미국 투자등급채",
    "US long bonds": "미국 장기채",
    "US high yield": "미국 하이일드",
    "Japan equity": "일본 주식",
    "China equity": "중국 본토",
    "China/HK growth": "중국/홍콩 성장",
    "India/EM": "인도/신흥국",
    "Gold": "금",
    "USD cash": "달러/달러예금",
    "Cash/short bonds": "현금/단기채",
}

KR_ARCH = {
    "Calm Risk-On": "평온한 Risk-On",
    "Fragile Peak": "고점 취약",
    "Peak Warning": "고점 경고",
    "Mixed/Transition": "혼합/전환",
    "Full Risk-Off": "전면 Risk-Off",
    "Technical Equity Breakdown": "주식 기술적 붕괴",
    "Credit/Liquidity Shock": "신용/유동성 충격",
    "Normal": "정상",
    "Fragile": "취약",
    "Warning": "경고",
    "Risk-Off": "Risk-Off",
}

LABELS = {
    "similarity_rank": "순위",
    "Date": "날짜",
    "similarity_score": "유사도점수",
    "similarity_distance": "거리",
    "risk_archetype": "위험유형",
    "risk_phase": "위험단계",
    "dominant_risk_vector": "주도위험",
    "next_1w_nasdaq_return": "나스닥 1주후",
    "next_1m_nasdaq_return": "나스닥 1개월후",
    "next_1w_sox_return": "SOX 1주후",
    "next_1m_sox_return": "SOX 1개월후",
    "next_1w_sp500_return": "S&P500 1주후",
    "next_1m_sp500_return": "S&P500 1개월후",
    "next_1w_russell2000_return": "러셀2000 1주후",
    "next_1m_russell2000_return": "러셀2000 1개월후",
    "next_1w_gold_return": "금 1주후",
    "next_1m_gold_return": "금 1개월후",
    "next_1w_dxy_return": "달러지수 1주후",
    "next_1m_dxy_return": "달러지수 1개월후",
    "next_1w_usdkrw_return": "원/달러 1주후",
    "next_1m_usdkrw_return": "원/달러 1개월후",
    "macro_liquidity_axis_x": "유동성/신용/환율",
    "market_breakdown_axis_y": "주가/변동성",
    "external_supply_axis_z": "대외/원자재",
    "peak_fragility": "고점취약성",
    "analog_macro_risk": "유사환경위험",
    "group": "자산군",
    "count": "개수",
    "avg_score": "평균점수",
    "avg_prob_1w": "1주 상승확률",
    "avg_prob_4w": "1개월 상승확률",
    "avg_realized_1w": "실제 1주",
    "avg_realized_4w": "실제 1개월",
    "max_bubble": "최대 LPPL",
    "pred_rank": "순위",
    "symbol": "종목코드",
    "name": "이름",
    "institutional_score_0_100": "종합점수",
    "meta_prob_1w": "1주 상승확률",
    "meta_prob_4w": "1개월 상승확률",
    "score_0_100": "원점수",
    "technical_score": "기술점수",
    "driver_fit_score": "매크로 적합도",
    "bubble_score_0_100": "LPPL 버블점수",
    "lppl_risk_label": "LPPL 위험",
    "realized_return_1w": "실제 1주",
    "realized_return_4w": "실제 1개월",
    "asset": "자산",
    "horizon": "기간",
    "n": "표본수",
    "avg_return": "평균수익률",
    "up_prob": "상승확률",
    "down_prob": "하락확률",
    "p10": "10% 분위",
    "worst": "최악",
    "best": "최고",
}


def pct(value: object) -> str:
    try:
        v = float(value)
    except Exception:
        return esc(value)
    return "" if pd.isna(v) else f"{v:.2%}"


def num(value: object) -> str:
    try:
        v = float(value)
    except Exception:
        return esc(value)
    return "" if pd.isna(v) else f"{v:,.2f}"


def esc(value: object) -> str:
    if pd.isna(value):
        return ""
    return html.escape(str(value))


def state(value: object) -> str:
    return KR_ARCH.get(str(value), str(value))


def table(df: pd.DataFrame, cols: list[str]) -> str:
    if df.empty:
        return "<p class='note'>데이터가 없습니다.</p>"
    pct_cols = {
        "avg_prob_1w",
        "avg_prob_4w",
        "avg_realized_1w",
        "avg_realized_4w",
        "realized_return_1w",
        "realized_return_4w",
        "avg_return",
        "up_prob",
        "down_prob",
        "p10",
        "worst",
        "best",
        "1주후",
        "1개월후",
    }
    rows = ["<table><thead><tr>"]
    for col in cols:
        rows.append(f"<th>{esc(LABELS.get(col, col))}</th>")
    rows.append("</tr></thead><tbody>")
    for _, row in df.iterrows():
        rows.append("<tr>")
        for col in cols:
            value = row.get(col, "")
            if col in {"risk_archetype", "risk_phase"}:
                text = esc(state(value))
            elif col == "group":
                text = esc(KR_GROUP.get(str(value), str(value)))
            elif col in pct_cols or "return" in col:
                text = pct(value)
            elif isinstance(value, pd.Timestamp):
                text = value.date().isoformat()
            elif isinstance(value, (float, np.floating, int, np.integer)):
                text = num(value)
            else:
                text = esc(value)
            rows.append(f"<td>{text}</td>")
        rows.append("</tr>")
    rows.append("</tbody></table>")
    return "".join(rows)


def generate(asof_text: str = "2025-10-30") -> Path:
    asof = pd.Timestamp(asof_text)
    out_dir = ROOT / "outputs" / f"asof_screening_{asof.date().isoformat()}"
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "charts").mkdir(parents=True, exist_ok=True)

    risk = pd.read_csv(ROOT / "outputs/risk_vector_dashboard_latest/tables/daily_risk_vector.csv", parse_dates=["Date"]).sort_values("Date")
    analog = pd.read_csv(ROOT / "outputs/analog_macro_risk_model_latest/tables/analog_macro_risk_scores.csv", parse_dates=["Date"]).sort_values("Date")
    weekly = pd.read_csv(ROOT / "outputs/weekly_screening_rank_backtest_latest/tables/weekly_calibrated_rank_panel.csv", parse_dates=["date"]).sort_values("date")
    for frame in (risk, analog, weekly):
        for col in frame.select_dtypes(include=[np.number]).columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    asof_risk = risk[risk["Date"].le(asof)].tail(1).iloc[0]
    asof_analog = analog[analog["Date"].eq(asof_risk["Date"])].tail(1).iloc[0]
    weekly_date = weekly[weekly["date"].le(asof)]["date"].max()
    weekly_panel = weekly[weekly["date"].eq(weekly_date)].copy().sort_values("institutional_score_0_100", ascending=False)

    cutoff = asof - pd.Timedelta(days=35)
    hist = analog[analog["Date"].le(cutoff)].copy()
    feature_cols = [
        "macro_liquidity_axis_x",
        "market_breakdown_axis_y",
        "external_supply_axis_z",
        "composite_vector_risk",
        "risk_off_score",
        "peak_fragility",
        "analog_macro_risk",
        "liquidity_credit_stress",
        "equity_breakdown_stress",
        "volatility_stress",
        "fx_external_stress",
        "cyclical_china_stress",
        "inflation_supply_stress",
        "NASDAQ100_ret_5d_pt",
        "NASDAQ100_ret_20d_pt",
        "SOX_ret_5d_pt",
        "SOX_ret_20d_pt",
        "DXY_ret_20d_pt",
        "USDKRW_ret_20d_pt",
        "VIX_ret_20d_pt",
        "HY_OAS_ret_20d_pt",
        "COPPER_GOLD_ret_20d_pt",
    ]
    feature_cols = [col for col in feature_cols if col in hist.columns and col in asof_analog.index]
    hist = hist.dropna(subset=feature_cols).copy()
    means = hist[feature_cols].mean()
    stds = hist[feature_cols].std().replace(0, np.nan)
    z_hist = ((hist[feature_cols] - means) / stds).astype(float)
    z_asof = ((asof_analog[feature_cols] - means) / stds).astype(float)
    hist["similarity_distance"] = np.sqrt(((z_hist - z_asof) ** 2).mean(axis=1))
    sim = hist.sort_values("similarity_distance").head(50).copy()
    sim["similarity_rank"] = np.arange(1, len(sim) + 1)
    sim["similarity_score"] = 100 / (1 + sim["similarity_distance"])
    sim = sim.rename(
        columns={
            "NASDAQ100_fwd_1w": "next_1w_nasdaq_return",
            "NASDAQ100_fwd_1m": "next_1m_nasdaq_return",
            "SOX_fwd_1w": "next_1w_sox_return",
            "SOX_fwd_1m": "next_1m_sox_return",
            "SP500_fwd_1w": "next_1w_sp500_return",
            "SP500_fwd_1m": "next_1m_sp500_return",
            "RUSSELL2000_fwd_1w": "next_1w_russell2000_return",
            "RUSSELL2000_fwd_1m": "next_1m_russell2000_return",
            "GOLD_fwd_1w": "next_1w_gold_return",
            "GOLD_fwd_1m": "next_1m_gold_return",
            "DXY_fwd_1w": "next_1w_dxy_return",
            "DXY_fwd_1m": "next_1m_dxy_return",
            "USDKRW_fwd_1w": "next_1w_usdkrw_return",
            "USDKRW_fwd_1m": "next_1m_usdkrw_return",
        }
    )
    sim_cols = [
        "similarity_rank",
        "Date",
        "similarity_score",
        "similarity_distance",
        "risk_archetype",
        "risk_phase",
        "dominant_risk_vector",
        "next_1w_nasdaq_return",
        "next_1m_nasdaq_return",
        "next_1w_sox_return",
        "next_1m_sox_return",
        "next_1w_sp500_return",
        "next_1m_sp500_return",
        "next_1w_russell2000_return",
        "next_1m_russell2000_return",
        "next_1w_gold_return",
        "next_1m_gold_return",
        "next_1w_dxy_return",
        "next_1m_dxy_return",
        "next_1w_usdkrw_return",
        "next_1m_usdkrw_return",
        "macro_liquidity_axis_x",
        "market_breakdown_axis_y",
        "external_supply_axis_z",
        "peak_fragility",
        "analog_macro_risk",
    ]
    sim = sim[[col for col in sim_cols if col in sim.columns]]

    stat_rows = []
    for asset, name in [("nasdaq", "나스닥100"), ("sox", "SOX"), ("sp500", "S&P500"), ("russell2000", "러셀2000"), ("gold", "금"), ("dxy", "달러지수"), ("usdkrw", "원/달러")]:
        for horizon, horizon_name in [("1w", "1주"), ("1m", "1개월")]:
            col = f"next_{horizon}_{asset}_return"
            if col not in sim:
                continue
            s = sim[col].dropna()
            stat_rows.append(
                {
                    "asset": name,
                    "horizon": horizon_name,
                    "n": len(s),
                    "avg_return": s.mean(),
                    "up_prob": (s > 0).mean(),
                    "down_prob": (s < 0).mean(),
                    "p10": s.quantile(0.1),
                    "worst": s.min(),
                    "best": s.max(),
                }
            )
    stats = pd.DataFrame(stat_rows)

    group_summary = (
        weekly_panel.groupby("group", as_index=False)
        .agg(
            count=("symbol", "count"),
            avg_score=("institutional_score_0_100", "mean"),
            avg_prob_1w=("meta_prob_1w", "mean"),
            avg_prob_4w=("meta_prob_4w", "mean"),
            avg_realized_1w=("realized_return_1w", "mean"),
            avg_realized_4w=("realized_return_4w", "mean"),
            max_bubble=("bubble_score_0_100", "max"),
        )
        .sort_values("avg_score", ascending=False)
    )
    top_assets = weekly_panel.head(50).copy()

    date_slug = asof.date().isoformat()
    sim.to_csv(out_dir / f"tables/similar_macro_cases_asof_{date_slug}.csv", index=False, encoding="utf-8-sig")
    stats.to_csv(out_dir / "tables/similar_case_return_stats.csv", index=False, encoding="utf-8-sig")
    group_summary.to_csv(out_dir / "tables/asset_group_screening_asof.csv", index=False, encoding="utf-8-sig")
    top_assets.to_csv(out_dir / "tables/top_assets_screening_asof.csv", index=False, encoding="utf-8-sig")

    chart_path = out_dir / f"charts/asof_{date_slug}_risk_market.png"
    make_chart(risk, asof, chart_path)

    actual = pd.DataFrame(
        [
            {"asset": "나스닥100", "1주후": asof_analog.get("NASDAQ100_fwd_1w"), "1개월후": asof_analog.get("NASDAQ100_fwd_1m")},
            {"asset": "SOX", "1주후": asof_analog.get("SOX_fwd_1w"), "1개월후": asof_analog.get("SOX_fwd_1m")},
            {"asset": "S&P500", "1주후": asof_analog.get("SP500_fwd_1w"), "1개월후": asof_analog.get("SP500_fwd_1m")},
            {"asset": "금", "1주후": asof_analog.get("GOLD_fwd_1w"), "1개월후": asof_analog.get("GOLD_fwd_1m")},
            {"asset": "달러지수", "1주후": asof_analog.get("DXY_fwd_1w"), "1개월후": asof_analog.get("DXY_fwd_1m")},
            {"asset": "원/달러", "1주후": asof_analog.get("USDKRW_fwd_1w"), "1개월후": asof_analog.get("USDKRW_fwd_1m")},
        ]
    )

    report = out_dir / f"asof_screening_{date_slug}.html"
    report.write_text(
        render_html(
            asof=asof,
            asof_risk=asof_risk,
            weekly_date=weekly_date,
            chart_path=chart_path,
            actual=actual,
            sim=sim,
            sim_cols=sim_cols,
            stats=stats,
            group_summary=group_summary,
            top_assets=top_assets,
        ),
        encoding="utf-8",
    )
    return report


def make_chart(risk: pd.DataFrame, asof: pd.Timestamp, path: Path) -> None:
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    panel = risk[(risk["Date"] >= asof - pd.DateOffset(months=12)) & (risk["Date"] <= asof)].copy()
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]})
    ax = axes[0]
    for col, label, color in [
        ("composite_vector_risk", "종합 위험", "#111827"),
        ("risk_off_score", "Risk-Off", "#c00000"),
        ("peak_fragility", "고점 취약성", "#7030a0"),
        ("analog_macro_risk", "유사환경 위험", "#0f766e"),
    ]:
        if col in panel:
            ax.plot(panel["Date"], panel[col], label=label, lw=1.8, color=color)
    ax.axvline(asof, color="#000000", ls="--", lw=1.2)
    for level in [35, 50, 65]:
        ax.axhline(level, color="#bfbfbf", ls="--", lw=0.8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("위험 점수")
    ax.legend(ncol=4, loc="upper left")
    ax.grid(True, axis="y", color="#dddddd")
    ax = axes[1]
    for col, label, color in [
        ("NASDAQ100", "나스닥100", "#1f77b4"),
        ("SOX", "SOX", "#7030a0"),
        ("SP500", "S&P500", "#2f5597"),
        ("GOLD", "금", "#b7791f"),
        ("DXY", "달러지수", "#111827"),
    ]:
        if col in panel:
            s = panel[col].replace(0, np.nan).dropna()
            if not s.empty:
                ax.plot(panel.loc[s.index, "Date"], s / s.iloc[0] * 100, label=label, lw=1.7, color=color)
    ax.axvline(asof, color="#000000", ls="--", lw=1.2)
    ax.set_ylabel("12개월 누적지수=100")
    ax.legend(ncol=5, loc="upper left")
    ax.grid(True, axis="y", color="#dddddd")
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.suptitle("2025-10-30 기준 Risk Vector와 주요 시장", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_html(
    asof: pd.Timestamp,
    asof_risk: pd.Series,
    weekly_date: pd.Timestamp,
    chart_path: Path,
    actual: pd.DataFrame,
    sim: pd.DataFrame,
    sim_cols: list[str],
    stats: pd.DataFrame,
    group_summary: pd.DataFrame,
    top_assets: pd.DataFrame,
) -> str:
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>{asof.date().isoformat()} 가상 스크리닝</title>
  <style>
    body{{font-family:"Malgun Gothic",Arial,sans-serif;background:#f4f6f8;color:#1f2933;margin:0}}
    header{{background:#111827;color:#fff;padding:24px 30px}}
    main{{max-width:1700px;margin:0 auto;padding:24px}}
    section{{margin-bottom:24px}}
    .cards{{display:grid;grid-template-columns:repeat(6,1fr);gap:12px}}
    .card{{background:#fff;border:1px solid #d7dde8;border-radius:8px;padding:14px}}
    .label{{color:#667085;font-size:12px}}
    .value{{font-size:24px;font-weight:800;margin-top:6px}}
    table{{width:100%;border-collapse:collapse;background:#fff;font-size:13px}}
    th,td{{border-bottom:1px solid #d7dde8;padding:7px 8px;text-align:right;white-space:nowrap}}
    th{{background:#eef2f7;position:sticky;top:0}}
    td:first-child,th:first-child{{text-align:left}}
    .wrap{{overflow:auto;border:1px solid #d7dde8;border-radius:8px}}
    img{{width:100%;border:1px solid #d7dde8;border-radius:8px;background:white}}
    .note{{color:#667085}}
  </style>
</head>
<body>
<header>
  <h1>{asof.date().isoformat()} 기준 가상 스크리닝</h1>
  <p>당시까지 알 수 있었던 데이터 기준. 유사환경 사례는 2025-09-25 이전 과거 구간만 사용했습니다.</p>
</header>
<main>
  <section class="cards">
    <div class="card"><div class="label">위험 유형</div><div class="value">{esc(state(asof_risk["risk_archetype"]))}</div></div>
    <div class="card"><div class="label">위험 단계</div><div class="value">{esc(state(asof_risk["risk_phase"]))}</div></div>
    <div class="card"><div class="label">종합 위험</div><div class="value">{num(asof_risk["composite_vector_risk"])}</div></div>
    <div class="card"><div class="label">Risk-Off</div><div class="value">{num(asof_risk["risk_off_score"])}</div></div>
    <div class="card"><div class="label">고점 취약성</div><div class="value">{num(asof_risk["peak_fragility"])}</div></div>
    <div class="card"><div class="label">유사환경 위험</div><div class="value">{num(asof_risk["analog_macro_risk"])}</div></div>
  </section>
  <section><h2>12개월 위험점수와 주요 시장</h2><img src="{chart_path.resolve().as_posix()}"></section>
  <section><h2>판단 요약</h2><div class="card">
    <p>{asof.date().isoformat()}은 전면 Risk-Off가 아니라 <b>고점 취약/Fragile Peak</b> 상태였습니다. Risk-Off 점수는 낮은 편이지만 고점 취약성은 {num(asof_risk["peak_fragility"])}, 유사환경 위험은 {num(asof_risk["analog_macro_risk"])}로 높아 1개월 조정 위험을 더 크게 봐야 하는 구간입니다.</p>
    <p class="note">자산 랭킹은 미래누수 방지를 위해 직전 주간 스크리닝일 {weekly_date.date().isoformat()} 값을 사용했습니다.</p>
  </div></section>
  <section><h2>{asof.date().isoformat()} 이후 실제 결과 검증</h2><div class="wrap">{table(actual, ["asset", "1주후", "1개월후"])}</div></section>
  <section><h2>현재와 유사했던 과거 매크로/Risk 환경 50개</h2><p class="note">유사도 높은 순서입니다. 각 행의 1주/1개월 후 수익률은 해당 과거 날짜 이후 실제 결과입니다.</p><div class="wrap">{table(sim, sim_cols)}</div></section>
  <section><h2>유사환경 50개 기준 사후 수익률 통계</h2><div class="wrap">{table(stats, ["asset", "horizon", "n", "avg_return", "up_prob", "down_prob", "p10", "worst", "best"])}</div></section>
  <section><h2>자산군별 스크리닝 요약 - {weekly_date.date().isoformat()}</h2><div class="wrap">{table(group_summary, ["group", "count", "avg_score", "avg_prob_1w", "avg_prob_4w", "avg_realized_1w", "avg_realized_4w", "max_bubble"])}</div></section>
  <section><h2>상위 스크리닝 자산 50개 - {weekly_date.date().isoformat()}</h2><div class="wrap">{table(top_assets, ["pred_rank", "symbol", "name", "group", "institutional_score_0_100", "meta_prob_1w", "meta_prob_4w", "score_0_100", "technical_score", "driver_fit_score", "bubble_score_0_100", "lppl_risk_label", "realized_return_1w", "realized_return_4w"])}</div></section>
</main>
</body>
</html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate point-in-time screening report.")
    parser.add_argument("--asof", default="2025-10-30")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(generate(args.asof).resolve())
