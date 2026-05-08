from __future__ import annotations

import html
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from annual_regime_market_charts import REGIME_COLORS, REGIME_KR, load_risk_panel


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "semiannual_regime_market_charts_latest"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "charts").mkdir(exist_ok=True)
    frame = load_risk_panel()
    frame = frame[frame["Date"].dt.year >= 2019].copy()
    summaries = []
    chart_paths = []
    for year in sorted(frame["Date"].dt.year.unique()):
        for half, start_month, end_month in [("H1", 1, 6), ("H2", 7, 12)]:
            start = pd.Timestamp(year=int(year), month=start_month, day=1)
            end = pd.Timestamp(year=int(year), month=end_month, day=1) + pd.offsets.MonthEnd(1)
            sub = frame[(frame["Date"] >= start) & (frame["Date"] <= end)].copy()
            sub = sub.dropna(subset=["NASDAQ100"], how="all")
            if sub.empty:
                continue
            label = f"{int(year)}_{half}"
            path = OUT / "charts" / f"regime_market_{label}.png"
            plot_period(sub, f"{int(year)} {half}", path)
            chart_paths.append(path)
            summaries.append(period_summary(sub, int(year), half))
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "semiannual_regime_summary.csv", index=False, encoding="utf-8-sig")
    html_path = OUT / "semiannual_regime_market_dashboard.html"
    html_path.write_text(render_html(summary, chart_paths), encoding="utf-8")
    print(html_path.resolve())


def period_summary(y: pd.DataFrame, year: int, half: str) -> dict[str, object]:
    dominant = y["risk_archetype"].mode().iloc[0] if not y["risk_archetype"].dropna().empty else ""
    return {
        "period": f"{year} {half}",
        "start": y["Date"].min().date().isoformat(),
        "end": y["Date"].max().date().isoformat(),
        "dominant_regime": REGIME_KR.get(str(dominant), str(dominant)),
        "nasdaq_return": total_return(y["NASDAQ100"]),
        "kospi200_return": total_return(y["KOSPI200"]),
        "max_total_risk": y["composite_vector_risk"].max(),
        "avg_total_risk": y["composite_vector_risk"].mean(),
        "max_risk_off": y["risk_off_score"].max(),
        "max_peak_fragility": y["peak_fragility"].max(),
        "max_analog_risk": y["analog_macro_risk"].max(),
        "max_correction_pressure": y["correction_pressure"].max() if "correction_pressure" in y else np.nan,
        "warning_or_worse_days": int(y["risk_phase"].isin(["Warning", "Risk-Off", "Crisis"]).sum()),
        "fragile_or_worse_days": int(y["risk_phase"].isin(["Fragile", "Warning", "Risk-Off", "Crisis"]).sum()),
        "regime_switches": max(len(contiguous_regimes(y)) - 1, 0),
    }


def total_return(series: pd.Series) -> float:
    s = series.replace(0, np.nan).dropna()
    if len(s) < 2:
        return np.nan
    return float(s.iloc[-1] / s.iloc[0] - 1.0)


def plot_period(y: pd.DataFrame, label: str, path: Path) -> None:
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(3, 1, figsize=(18, 10.5), sharex=True, gridspec_kw={"height_ratios": [2.1, 0.78, 1.35]})
    ax0, ax1, ax2 = axes
    shade_regimes(ax0, y)
    for col, name, color, lw in [
        ("NASDAQ100", "나스닥100", "#1f77b4", 2.3),
        ("KOSPI200", "KOSPI200 ETF", "#c00000", 2.1),
    ]:
        s = y[col].replace(0, np.nan).dropna()
        if not s.empty:
            ax0.plot(y.loc[s.index, "Date"], s / s.iloc[0] * 100, label=name, color=color, lw=lw)
    ax0.set_title(f"{label} 나스닥/한국지수와 세부 Regime", fontsize=16, fontweight="bold")
    ax0.set_ylabel("기간초=100")
    ax0.legend(loc="upper left", ncol=2)
    ax0.grid(True, axis="y", color="#dddddd", lw=0.8)

    regime_blocks(ax1, y)
    ax1.set_yticks([])
    ax1.set_ylabel("Regime")

    ax2.plot(y["Date"], y["composite_vector_risk"], label="종합 위험", color="#111827", lw=2.0)
    ax2.plot(y["Date"], y["risk_off_score"], label="Risk-Off", color="#c00000", lw=1.45)
    ax2.plot(y["Date"], y["peak_fragility"], label="고점 취약성", color="#7030a0", lw=1.45)
    ax2.plot(y["Date"], y["analog_macro_risk"], label="유사환경 위험", color="#0f766e", lw=1.45)
    if "correction_pressure" in y:
        ax2.plot(y["Date"], y["correction_pressure"], label="조정 압력", color="#d97706", lw=1.55)
    for level, text in [(35, "주의"), (50, "위험"), (65, "현금")]:
        ax2.axhline(level, color="#bfbfbf", ls="--", lw=0.8)
        ax2.text(y["Date"].min(), level + 1, text, fontsize=9, color="#666666")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("위험 점수")
    ax2.legend(loc="upper left", ncol=5, fontsize=9)
    ax2.grid(True, axis="y", color="#dddddd", lw=0.8)

    legend = [mpatches.Patch(color=color, label=REGIME_KR.get(regime, regime)) for regime, color in REGIME_COLORS.items() if regime in set(y["risk_archetype"].dropna())]
    if legend:
        ax1.legend(handles=legend, loc="upper left", bbox_to_anchor=(0, -0.28), ncol=6, fontsize=8, frameon=False)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    fig.savefig(path, dpi=155, bbox_inches="tight")
    plt.close(fig)


def shade_regimes(ax: plt.Axes, y: pd.DataFrame) -> None:
    for start, end, regime, _ in contiguous_regimes(y):
        ax.axvspan(start, end, color=REGIME_COLORS.get(regime, "#eeeeee"), alpha=0.22, lw=0)


def regime_blocks(ax: plt.Axes, y: pd.DataFrame) -> None:
    blocks = contiguous_regimes(y)
    for idx, (start, end, regime, days) in enumerate(blocks, start=1):
        ax.axvspan(start, end, color=REGIME_COLORS.get(regime, "#eeeeee"), alpha=0.96, lw=0)
        mid = start + (end - start) / 2
        label = REGIME_KR.get(regime, regime)
        text = label if days >= 5 else str(idx)
        ax.text(mid, 0.55, text, ha="center", va="center", fontsize=8, color="#111827")
        if days < 5:
            ax.text(mid, 0.18, f"{idx}", ha="center", va="center", fontsize=7, color="#111827")
    ax.set_ylim(0, 1)
    short = [f"{idx}. {start.date()}~{end.date()} {REGIME_KR.get(regime, regime)}" for idx, (start, end, regime, days) in enumerate(blocks, start=1) if days < 5]
    if short:
        ax.text(0.005, -0.54, "짧은 구간: " + " / ".join(short[:8]), transform=ax.transAxes, fontsize=8, color="#333333", va="top")


def contiguous_regimes(y: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp, str, int]]:
    data = y[["Date", "risk_archetype"]].dropna().reset_index(drop=True)
    if data.empty:
        return []
    blocks = []
    start = data.loc[0, "Date"]
    prev_date = data.loc[0, "Date"]
    current = str(data.loc[0, "risk_archetype"])
    days = 1
    for _, row in data.iloc[1:].iterrows():
        regime = str(row["risk_archetype"])
        date = row["Date"]
        if regime != current:
            blocks.append((start, prev_date, current, days))
            start = date
            current = regime
            days = 1
        else:
            days += 1
        prev_date = date
    blocks.append((start, prev_date, current, days))
    return blocks


def render_html(summary: pd.DataFrame, chart_paths: list[Path]) -> str:
    headers = "".join(f"<th>{html.escape(c)}</th>" for c in summary.columns)
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
    sections = []
    for path in chart_paths:
        title = path.stem.replace("regime_market_", "").replace("_", " ")
        sections.append(f"<section><h2>{html.escape(title)}</h2><img src='{path.resolve().as_posix()}'></section>")
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>반기별 Regime 차트</title>
  <style>
    body{{font-family:"Malgun Gothic",Arial,sans-serif;background:#f4f6f8;color:#1f2933;margin:0}}
    header{{background:#111827;color:white;padding:24px 30px}}
    main{{max-width:1840px;margin:0 auto;padding:24px}}
    section{{margin-bottom:30px}}
    img{{width:100%;border:1px solid #d7dde8;border-radius:8px;background:white}}
    table{{width:100%;border-collapse:collapse;background:white;font-size:13px;margin-bottom:24px}}
    th,td{{padding:8px 9px;border-bottom:1px solid #d7dde8;text-align:right;white-space:nowrap}}
    th:first-child,td:first-child,th:nth-child(4),td:nth-child(4){{text-align:left}}
    th{{background:#eef2f7;position:sticky;top:0}}
    .wrap{{overflow:auto;border:1px solid #d7dde8;border-radius:8px}}
    .note{{color:#667085}}
  </style>
</head>
<body>
<header>
  <h1>2019년 이후 반기별 나스닥·한국지수 Regime 차트</h1>
  <p>짧은 전환 regime도 숫자 마커로 표시했습니다. 아래 막대의 숫자는 5거래일 미만의 짧은 regime 구간입니다.</p>
</header>
<main>
<section><h2>반기 요약</h2><div class="wrap"><table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
{''.join(sections)}
</main>
</body>
</html>"""


if __name__ == "__main__":
    main()
