from __future__ import annotations

import html
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "annual_regime_market_charts_latest"

REGIME_KR = {
    "Calm Risk-On": "평온한 Risk-On",
    "Fragile Peak": "고점 취약",
    "Peak Warning": "고점 경고",
    "Correction Pressure": "조정 압력",
    "Mixed/Transition": "혼합/전환",
    "Technical Equity Breakdown": "주식 붕괴",
    "Credit/Liquidity Shock": "신용/유동성 충격",
    "Full Risk-Off": "전면 Risk-Off",
    "Inflation/Supply Shock": "물가/공급 충격",
    "Cyclical/China Stress": "경기/중국 스트레스",
    "FX/External Stress": "환율/대외 스트레스",
}

REGIME_COLORS = {
    "Calm Risk-On": "#d9ead3",
    "Fragile Peak": "#fff2cc",
    "Peak Warning": "#fce4d6",
    "Correction Pressure": "#f4b183",
    "Mixed/Transition": "#ddebf7",
    "Technical Equity Breakdown": "#eadcf8",
    "Credit/Liquidity Shock": "#f4cccc",
    "Full Risk-Off": "#c00000",
    "Inflation/Supply Shock": "#f8cbad",
    "Cyclical/China Stress": "#e2f0d9",
    "FX/External Stress": "#d9e2f3",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "charts").mkdir(exist_ok=True)
    risk = load_risk_panel()
    yearly = []
    chart_paths = []
    for year, frame in risk.groupby(risk["Date"].dt.year):
        if year < 2019 or frame["NASDAQ100"].replace(0, np.nan).dropna().empty:
            continue
        path = OUT / "charts" / f"regime_market_{year}.png"
        yearly.append(year_summary(frame, year))
        plot_year(frame, year, path)
        chart_paths.append(path)
    summary = pd.DataFrame(yearly)
    summary.to_csv(OUT / "annual_regime_summary.csv", index=False, encoding="utf-8-sig")
    html_path = OUT / "annual_regime_market_dashboard.html"
    html_path.write_text(render_html(summary, chart_paths), encoding="utf-8")
    print(html_path.resolve())


def load_risk_panel() -> pd.DataFrame:
    risk = pd.read_csv(ROOT / "outputs/risk_vector_dashboard_latest/tables/daily_risk_vector.csv", parse_dates=["Date"]).sort_values("Date")
    kospi_path = ROOT / ".cache/prices/069500_KS.csv"
    if kospi_path.exists():
        kospi = pd.read_csv(kospi_path, parse_dates=["Date"]).sort_values("Date")
        close_col = "Close" if "Close" in kospi.columns else "Adj Close"
        kospi = kospi[["Date", close_col]].rename(columns={close_col: "KOSPI200"})
        risk = risk.merge(kospi, on="Date", how="left")
    else:
        risk["KOSPI200"] = np.nan
    for col in risk.select_dtypes(include=[np.number]).columns:
        risk[col] = pd.to_numeric(risk[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    risk["KOSPI200"] = risk["KOSPI200"].ffill()
    return risk


def year_summary(frame: pd.DataFrame, year: int) -> dict[str, object]:
    y = frame.dropna(subset=["NASDAQ100"]).copy()
    kospi = y.dropna(subset=["KOSPI200"]).copy()
    dominant = y["risk_archetype"].mode().iloc[0] if not y["risk_archetype"].dropna().empty else ""
    high_risk = y[y["composite_vector_risk"].ge(35)]
    risk_off = y[y["risk_phase"].isin(["Warning", "Risk-Off"])]
    peak = y[y["peak_fragility"].ge(48)]
    return {
        "year": year,
        "dominant_regime": REGIME_KR.get(str(dominant), str(dominant)),
        "nasdaq_return": total_return(y["NASDAQ100"]),
        "kospi200_return": total_return(kospi["KOSPI200"]) if not kospi.empty else np.nan,
        "max_composite_risk": y["composite_vector_risk"].max(),
        "avg_composite_risk": y["composite_vector_risk"].mean(),
        "max_risk_off": y["risk_off_score"].max(),
        "max_peak_fragility": y["peak_fragility"].max(),
        "max_analog_risk": y["analog_macro_risk"].max(),
        "high_risk_days": int(len(high_risk)),
        "warning_or_riskoff_days": int(len(risk_off)),
        "peak_warning_days": int(len(peak)),
    }


def total_return(series: pd.Series) -> float:
    s = series.replace(0, np.nan).dropna()
    if len(s) < 2:
        return np.nan
    return float(s.iloc[-1] / s.iloc[0] - 1.0)


def plot_year(frame: pd.DataFrame, year: int, path: Path) -> None:
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    y = frame.copy()
    fig, axes = plt.subplots(3, 1, figsize=(18, 10.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 0.55, 1.35]})
    ax0, ax1, ax2 = axes

    shade_regimes(ax0, y)
    for col, label, color, lw in [
        ("NASDAQ100", "나스닥100", "#1f77b4", 2.3),
        ("KOSPI200", "KOSPI200 ETF", "#c00000", 2.1),
    ]:
        s = y[col].replace(0, np.nan).dropna()
        if s.empty:
            continue
        ax0.plot(y.loc[s.index, "Date"], s / s.iloc[0] * 100, label=label, color=color, lw=lw)
    ax0.set_ylabel("연초=100")
    ax0.set_title(f"{year} 나스닥/한국지수와 동적 Regime", fontsize=16, fontweight="bold")
    ax0.legend(loc="upper left", ncol=2)
    ax0.grid(True, axis="y", color="#dddddd", lw=0.8)

    regime_blocks(ax1, y)
    ax1.set_yticks([])
    ax1.set_ylabel("Regime")

    ax2.plot(y["Date"], y["composite_vector_risk"], label="종합 위험", color="#111827", lw=2.0)
    ax2.plot(y["Date"], y["risk_off_score"], label="Risk-Off", color="#c00000", lw=1.5)
    ax2.plot(y["Date"], y["peak_fragility"], label="고점 취약성", color="#7030a0", lw=1.5)
    ax2.plot(y["Date"], y["analog_macro_risk"], label="유사환경 위험", color="#0f766e", lw=1.5)
    if "correction_pressure" in y:
        ax2.plot(y["Date"], y["correction_pressure"], label="조정 압력", color="#d97706", lw=1.6)
    for level, label in [(35, "주의"), (50, "위험"), (65, "현금")]:
        ax2.axhline(level, color="#bfbfbf", ls="--", lw=0.8)
        ax2.text(y["Date"].min(), level + 1, label, fontsize=9, color="#666666")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("위험 점수")
    ax2.legend(loc="upper left", ncol=4)
    ax2.grid(True, axis="y", color="#dddddd", lw=0.8)

    legend = [mpatches.Patch(color=color, label=REGIME_KR.get(regime, regime)) for regime, color in REGIME_COLORS.items() if regime in set(y["risk_archetype"].dropna())]
    if legend:
        ax1.legend(handles=legend, loc="center left", bbox_to_anchor=(0, -0.2), ncol=5, fontsize=8, frameon=False)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def shade_regimes(ax: plt.Axes, y: pd.DataFrame) -> None:
    for start, end, regime in contiguous_regimes(y):
        ax.axvspan(start, end, color=REGIME_COLORS.get(regime, "#eeeeee"), alpha=0.22, lw=0)


def regime_blocks(ax: plt.Axes, y: pd.DataFrame) -> None:
    for start, end, regime in contiguous_regimes(y):
        ax.axvspan(start, end, color=REGIME_COLORS.get(regime, "#eeeeee"), alpha=0.95, lw=0)
        mid = start + (end - start) / 2
        if (end - start).days >= 12:
            ax.text(mid, 0.5, REGIME_KR.get(regime, regime), ha="center", va="center", fontsize=8, color="#111827")
    ax.set_ylim(0, 1)


def contiguous_regimes(y: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    data = y[["Date", "risk_archetype"]].dropna().reset_index(drop=True)
    if data.empty:
        return []
    blocks = []
    start = data.loc[0, "Date"]
    prev_date = data.loc[0, "Date"]
    current = str(data.loc[0, "risk_archetype"])
    for _, row in data.iloc[1:].iterrows():
        regime = str(row["risk_archetype"])
        date = row["Date"]
        if regime != current:
            blocks.append((start, prev_date, current))
            start = date
            current = regime
        prev_date = date
    blocks.append((start, prev_date, current))
    return blocks


def render_html(summary: pd.DataFrame, chart_paths: list[Path]) -> str:
    rows = []
    for _, row in summary.iterrows():
        rows.append("<tr>")
        for col in summary.columns:
            value = row[col]
            if col.endswith("return"):
                text = "" if pd.isna(value) else f"{value:.2%}"
            elif isinstance(value, (float, np.floating)):
                text = "" if pd.isna(value) else f"{value:,.2f}"
            else:
                text = html.escape(str(value))
            rows.append(f"<td>{text}</td>")
        rows.append("</tr>")
    cards = []
    for path in chart_paths:
        year = path.stem.split("_")[-1]
        cards.append(f"<section><h2>{year}</h2><img src='{path.resolve().as_posix()}'></section>")
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>연간 Regime 차트</title>
  <style>
    body{{font-family:"Malgun Gothic",Arial,sans-serif;background:#f4f6f8;color:#1f2933;margin:0}}
    header{{background:#111827;color:white;padding:24px 30px}}
    main{{max-width:1800px;margin:0 auto;padding:24px}}
    section{{margin-bottom:28px}}
    img{{width:100%;border:1px solid #d7dde8;border-radius:8px;background:white}}
    table{{width:100%;border-collapse:collapse;background:white;font-size:13px;margin-bottom:24px}}
    th,td{{padding:8px 9px;border-bottom:1px solid #d7dde8;text-align:right;white-space:nowrap}}
    th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}
    th{{background:#eef2f7}}
  </style>
</head>
<body>
<header><h1>연간 나스닥·한국지수 Regime 차트</h1><p>나스닥100과 KOSPI200 ETF를 연초=100으로 비교하고, 배경색과 하단 막대로 해당 구간의 동적 regime을 표시합니다.</p></header>
<main>
<section><h2>연간 요약</h2><table><thead><tr>{''.join(f'<th>{html.escape(c)}</th>' for c in summary.columns)}</tr></thead><tbody>{''.join(rows)}</tbody></table></section>
{''.join(cards)}
</main>
</body>
</html>"""


if __name__ == "__main__":
    main()
