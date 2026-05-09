from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SENTINEL_HISTORY = ROOT / "outputs" / "daily_risk_off_sentinel_latest" / "tables" / "daily_sentinel_history.csv"
DRIVER_PANEL = ROOT / "outputs" / "macro_regime_asset_screener_latest" / "tables" / "driver_panel.csv"
PEAK_SCORES = ROOT / "outputs" / "peak_fragility_model_latest" / "tables" / "peak_fragility_daily_scores.csv"
ANALOG_SCORES = ROOT / "outputs" / "analog_macro_risk_model_latest" / "tables" / "analog_macro_risk_scores.csv"
CORRECTION_SCORES = ROOT / "outputs" / "correction_timing_model_latest" / "tables" / "correction_timing_daily_scores.csv"
OUT_DIR = ROOT / "outputs" / "risk_vector_dashboard_latest"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create multi-dimensional risk vector dashboard from Sentinel internals.")
    parser.add_argument("--sentinel-history", type=Path, default=SENTINEL_HISTORY)
    parser.add_argument("--driver-panel", type=Path, default=DRIVER_PANEL)
    parser.add_argument("--peak-scores", type=Path, default=PEAK_SCORES)
    parser.add_argument("--analog-scores", type=Path, default=ANALOG_SCORES)
    parser.add_argument("--correction-scores", type=Path, default=CORRECTION_SCORES)
    parser.add_argument("--output", type=Path, default=OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables = args.output / "tables"
    charts = args.output / "charts"
    reports = args.output / "reports"
    for path in (tables, charts, reports):
        path.mkdir(parents=True, exist_ok=True)

    vector = build_risk_vector(args.sentinel_history, args.driver_panel, args.peak_scores, args.analog_scores, args.correction_scores)
    vector.to_csv(tables / "daily_risk_vector.csv", index=False, encoding="utf-8-sig")
    vector.tail(1).to_csv(tables / "current_risk_vector.csv", index=False, encoding="utf-8-sig")
    yearly = yearly_summary(vector)
    yearly.to_csv(tables / "yearly_risk_vector_summary.csv", index=False, encoding="utf-8-sig")
    chart_paths = create_charts(vector, charts)
    write_report(vector, yearly, chart_paths, reports / "risk_vector_dashboard_report.md")
    print(f"wrote {reports / 'risk_vector_dashboard_report.md'}")
    print(vector.tail(1).T.to_string())


def build_risk_vector(sentinel_path: Path, driver_path: Path, peak_path: Path, analog_path: Path | None = None, correction_path: Path | None = None) -> pd.DataFrame:
    sent = pd.read_csv(sentinel_path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    driver = pd.read_csv(driver_path, parse_dates=["Date"]).sort_values("Date")
    keep = ["Date", "NASDAQ100", "SOX", "SP500", "RUSSELL2000", "DXY", "USDKRW", "USDJPY", "USDCNH", "US10Y", "GOLD", "WTI"]
    df = sent.merge(driver[[c for c in keep if c in driver]], on="Date", how="left")
    if peak_path.exists():
        peak = pd.read_csv(peak_path, parse_dates=["Date"])
        cols = ["Date", "peak_fragility_score_0_100", "peak_fragility_state"]
        df = df.merge(peak[[c for c in cols if c in peak]], on="Date", how="left")
    else:
        df["peak_fragility_score_0_100"] = 0.0
        df["peak_fragility_state"] = "Unknown"
    if analog_path and analog_path.exists():
        analog = pd.read_csv(analog_path, parse_dates=["Date"])
        cols = [
            "Date",
            "analog_risk_score_0_100",
            "analog_state",
            "analog_down_prob_1w_model",
            "analog_down_prob_1m_model",
            "analog_tail_prob_1m_model",
            "analog_k50_nasdaq_down_prob_1w",
            "analog_k50_nasdaq_down_prob_1m",
            "analog_k50_nasdaq_avg_return_1w",
            "analog_k50_nasdaq_avg_return_1m",
            "analog_k50_nasdaq_tail_return_p10_1m",
        ]
        df = df.merge(analog[[c for c in cols if c in analog]], on="Date", how="left")
    else:
        df["analog_risk_score_0_100"] = 0.0
        df["analog_state"] = "Unknown"
    if correction_path and correction_path.exists():
        correction = pd.read_csv(correction_path, parse_dates=["Date"])
        cols = [
            "Date",
            "correction_pressure_score_0_100",
            "correction_pressure_state",
            "correction_1w_drop_prob",
            "correction_1m_prob",
            "delayed_correction_prob",
        ]
        df = df.merge(correction[[c for c in cols if c in correction]], on="Date", how="left")
    else:
        df["correction_pressure_score_0_100"] = 0.0
        df["correction_pressure_state"] = "Unknown"

    for col in df.columns:
        if col != "Date" and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    out = df.copy()
    out["liquidity_credit_stress"] = weighted_mean(
        out,
        {
            "HY_OAS_shock_score": 0.26,
            "IG_OAS_shock_score": 0.16,
            "HYG_IEF_shock_score": 0.20,
            "MOVE_shock_score": 0.18,
            "credit_score": 0.20,
        },
    )
    out["equity_breakdown_stress"] = weighted_mean(
        out,
        {
            "SP500_shock_score": 0.22,
            "NASDAQ100_shock_score": 0.25,
            "SOX_shock_score": 0.24,
            "RUSSELL2000_shock_score": 0.14,
            "equity_score": 0.15,
        },
    )
    out["volatility_stress"] = weighted_mean(
        out,
        {
            "VIX_shock_score": 0.34,
            "VXN_shock_score": 0.30,
            "MOVE_shock_score": 0.20,
            "volatility_score": 0.16,
        },
    )
    out["fx_external_stress"] = weighted_mean(
        out,
        {
            "DXY_shock_score": 0.25,
            "USDKRW_shock_score": 0.30,
            "USDJPY_shock_score": 0.14,
            "USDCNH_shock_score": 0.20,
            "fx_score": 0.15,
            "COPPER_GOLD_shock_score": 0.10,
        },
    )
    out["cyclical_china_stress"] = weighted_mean(
        out,
        {
            "COPPER_shock_score": 0.22,
            "COPPER_GOLD_shock_score": 0.20,
            "CSI300_shock_score": 0.20,
            "HANGSENG_TECH_shock_score": 0.18,
            "cyclical_score": 0.20,
        },
    )
    out["inflation_supply_stress"] = weighted_mean(out, {"WTI_shock_score": 0.65, "supply_shock_score": 0.35})
    out["hedge_demand"] = weighted_mean(out, {"GOLD_shock_score": 0.55, "hedge_bid_score": 0.45})
    out["rai_appetite_stress"] = weighted_mean(
        out,
        {
            "RAI_shock_score": 0.38,
            "RAI_fear_score": 0.22,
            "RAI_collapse_score": 0.25,
            "RAI_overheat_score": 0.15,
        },
    )
    out["universe_breadth_stress"] = weighted_mean(
        out,
        {
            "ETF_breadth_shock_score": 0.48,
            "ETF_below_60ma_pct": 0.20,
            "ETF_below_20ma_pct": 0.14,
            "ETF_20d_loss_pct": 0.14,
            "ETF_20d_large_loss_pct": 0.04,
        },
    )
    out["safe_rotation_stress"] = weighted_mean(out, {"SAFE_ROTATION_shock_score": 0.70, "hedge_demand": 0.30})
    out["peak_fragility"] = pd.to_numeric(out.get("peak_fragility_score_0_100", 0.0), errors="coerce").fillna(0.0).clip(0, 100)
    out["analog_macro_risk"] = pd.to_numeric(out.get("analog_risk_score_0_100", 0.0), errors="coerce").fillna(0.0).clip(0, 100)
    out["correction_pressure"] = pd.to_numeric(out.get("correction_pressure_score_0_100", 0.0), errors="coerce").fillna(0.0).clip(0, 100)

    out["macro_liquidity_axis_x"] = (
        0.42 * out["liquidity_credit_stress"]
        + 0.28 * out["fx_external_stress"]
        + 0.20 * out["volatility_stress"]
        + 0.10 * out["inflation_supply_stress"]
        + 0.10 * out["rai_appetite_stress"]
    ).clip(0, 100)
    out["market_breakdown_axis_y"] = (
        0.36 * out["equity_breakdown_stress"]
        + 0.20 * out["volatility_stress"]
        + 0.15 * out["cyclical_china_stress"]
        + 0.14 * out["universe_breadth_stress"]
        + 0.08 * out["rai_appetite_stress"]
        + 0.10 * out["peak_fragility"]
        + 0.07 * out["analog_macro_risk"]
        + 0.08 * out["correction_pressure"]
    ).clip(0, 100)
    out["external_supply_axis_z"] = (
        0.36 * out["fx_external_stress"]
        + 0.26 * out["cyclical_china_stress"]
        + 0.22 * out["inflation_supply_stress"]
        + 0.10 * out["hedge_demand"]
        + 0.10 * out["safe_rotation_stress"]
    ).clip(0, 100)
    out["composite_vector_risk"] = (
        0.25 * out["macro_liquidity_axis_x"]
        + 0.25 * out["market_breakdown_axis_y"]
        + 0.16 * out["external_supply_axis_z"]
        + 0.20 * pd.to_numeric(out.get("risk_off_score", 0.0), errors="coerce").fillna(0.0)
        + 0.10 * out["peak_fragility"]
        + 0.08 * out["analog_macro_risk"]
        + 0.06 * out["correction_pressure"]
        + 0.07 * out["rai_appetite_stress"]
        + 0.08 * out["universe_breadth_stress"]
    ).clip(0, 100)

    axis_cols = [
        "liquidity_credit_stress",
        "equity_breakdown_stress",
        "volatility_stress",
        "fx_external_stress",
        "cyclical_china_stress",
        "inflation_supply_stress",
        "hedge_demand",
        "rai_appetite_stress",
        "universe_breadth_stress",
        "safe_rotation_stress",
        "peak_fragility",
        "analog_macro_risk",
        "correction_pressure",
    ]
    out["dominant_risk_vector"] = out[axis_cols].idxmax(axis=1).str.replace("_", " ").str.title()
    out["risk_archetype"] = out.apply(classify_archetype, axis=1)
    out["risk_phase"] = out.apply(classify_phase, axis=1)
    return out


def weighted_mean(frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    total = pd.Series(0.0, index=frame.index)
    wsum = 0.0
    for col, weight in weights.items():
        if col not in frame:
            continue
        total = total + pd.to_numeric(frame[col], errors="coerce").fillna(0.0).clip(0, 100) * weight
        wsum += weight
    return (total / max(wsum, 1e-9)).clip(0, 100)


def classify_archetype(row: pd.Series) -> str:
    x = row["macro_liquidity_axis_x"]
    y = row["market_breakdown_axis_y"]
    z = row["external_supply_axis_z"]
    peak = row["peak_fragility"]
    credit = row["liquidity_credit_stress"]
    fx = row["fx_external_stress"]
    vol = row["volatility_stress"]
    equity = row["equity_breakdown_stress"]
    supply = row["inflation_supply_stress"]
    cyc = row["cyclical_china_stress"]
    correction = row.get("correction_pressure", 0.0)
    rai = row.get("rai_appetite_stress", 0.0)
    breadth = row.get("universe_breadth_stress", 0.0)

    if x >= 55 and y >= 55:
        return "Full Risk-Off"
    if rai >= 58 and breadth >= 45:
        return "RAI/Breadth Risk-Off"
    if breadth >= 65 and y >= 35:
        return "Universe Breadth Breakdown"
    if rai >= 60 and y < 45:
        return "Risk Appetite Collapse"
    if correction >= 60 and peak >= 45 and x < 45:
        return "Correction Pressure"
    if peak >= 60 and y < 45 and x < 45:
        return "Fragile Peak"
    if credit >= 55 and vol >= 45:
        return "Credit/Liquidity Shock"
    if fx >= 55 and z >= 45:
        return "FX/External Stress"
    if supply >= 55 and equity < 45:
        return "Inflation/Supply Shock"
    if equity >= 50 and x < 45:
        return "Technical Equity Breakdown"
    if cyc >= 50 and fx >= 35:
        return "Cyclical/China Stress"
    if peak >= 48:
        return "Peak Warning"
    if max(x, y, z) < 30:
        return "Calm Risk-On"
    return "Mixed/Transition"


def classify_phase(row: pd.Series) -> str:
    total = row["composite_vector_risk"]
    if total >= 70:
        return "Crisis"
    if total >= 55:
        return "Risk-Off"
    if total >= 40:
        return "Warning"
    if total >= 25:
        return "Fragile"
    return "Normal"


def yearly_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, y in frame.groupby(frame["Date"].dt.year):
        rows.append(
            {
                "year": int(year),
                "start": y["Date"].min().date().isoformat(),
                "end": y["Date"].max().date().isoformat(),
                "nasdaq_return": float(y["NASDAQ100"].iloc[-1] / y["NASDAQ100"].iloc[0] - 1.0) if "NASDAQ100" in y and y["NASDAQ100"].iloc[0] else np.nan,
                "max_composite_vector_risk": float(y["composite_vector_risk"].max()),
                "max_liquidity_credit": float(y["liquidity_credit_stress"].max()),
                "max_equity_breakdown": float(y["equity_breakdown_stress"].max()),
                "max_fx_external": float(y["fx_external_stress"].max()),
                "max_peak_fragility": float(y["peak_fragility"].max()),
                "max_rai_appetite_stress": float(y.get("rai_appetite_stress", pd.Series(0.0, index=y.index)).max()),
                "max_universe_breadth_stress": float(y.get("universe_breadth_stress", pd.Series(0.0, index=y.index)).max()),
                "warning_or_worse_days": int(y["risk_phase"].isin(["Warning", "Risk-Off", "Crisis"]).sum()),
                "top_archetype": y["risk_archetype"].value_counts().idxmax(),
            }
        )
    return pd.DataFrame(rows)


def create_charts(frame: pd.DataFrame, charts_dir: Path) -> list[Path]:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from PIL import Image

    yearly_dir = charts_dir / "yearly_vector_vs_nasdaq"
    yearly_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for year, y in frame.groupby(frame["Date"].dt.year):
        y = y.dropna(subset=["NASDAQ100"]).copy()
        if y.empty:
            continue
        fig, axes = plt.subplots(3, 1, figsize=(15, 10), dpi=150, sharex=True, gridspec_kw={"height_ratios": [2.0, 1.4, 1.4]})
        fig.patch.set_facecolor("white")

        ax = axes[0]
        ax.plot(y["Date"], y["NASDAQ100"], color="#1f4e79", lw=2.0, label="Nasdaq 100")
        warn = y[y["risk_phase"].isin(["Warning", "Risk-Off", "Crisis"])]
        ax.scatter(warn["Date"], warn["NASDAQ100"], s=12, color="#c00000", alpha=0.65, label="Vector Warning+")
        ax.set_ylabel("Nasdaq 100")
        ax.grid(True, axis="y", color="#d9d9d9", lw=0.8)
        ax.legend(loc="upper left", fontsize=9)

        ax = axes[1]
        ax.plot(y["Date"], y["macro_liquidity_axis_x"], color="#c00000", lw=1.4, label="X Liquidity/Credit/FX")
        ax.plot(y["Date"], y["market_breakdown_axis_y"], color="#7030a0", lw=1.4, label="Y Equity/Vol Breakdown")
        ax.plot(y["Date"], y["external_supply_axis_z"], color="#548235", lw=1.2, label="Z External/Supply")
        ax.axhline(40, color="#f4b183", ls="--", lw=0.9)
        ax.axhline(55, color="#c00000", ls="--", lw=0.9)
        ax.set_ylim(0, 100)
        ax.set_ylabel("2D/3D Axes")
        ax.grid(True, axis="y", color="#e6e6e6", lw=0.8)
        ax.legend(loc="upper left", ncol=3, fontsize=8)

        ax = axes[2]
        ax.stackplot(
            y["Date"],
            y["liquidity_credit_stress"],
            y["equity_breakdown_stress"],
            y["fx_external_stress"],
            y.get("rai_appetite_stress", pd.Series(0.0, index=y.index)),
            y.get("universe_breadth_stress", pd.Series(0.0, index=y.index)),
            y["peak_fragility"],
            labels=["Liquidity/Credit", "Equity Breakdown", "FX/External", "RAI", "ETF Breadth", "Peak Fragility"],
            colors=["#c00000", "#7030a0", "#ed7d31", "#00a6a6", "#7f7f7f", "#4472c4"],
            alpha=0.72,
        )
        ax.set_ylabel("Risk Components")
        ax.grid(True, axis="y", color="#e6e6e6", lw=0.8)
        ax.legend(loc="upper left", ncol=3, fontsize=8)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

        title = f"{int(year)} Risk Vector vs Nasdaq"
        subtitle = f"Max vector risk {y['composite_vector_risk'].max():.1f} | dominant archetype {y['risk_archetype'].value_counts().idxmax()}"
        fig.suptitle(title + "\n" + subtitle, fontsize=15)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        path = yearly_dir / f"risk_vector_vs_nasdaq_{int(year)}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    scatter_path = create_2d_scatter(frame, charts_dir)
    paths.append(scatter_path)
    sheet_path = create_contact_sheet(paths=[p for p in paths if "risk_vector_vs_nasdaq_" in p.name], out_dir=yearly_dir)
    if sheet_path:
        paths.append(sheet_path)
    return paths


def create_2d_scatter(frame: pd.DataFrame, charts_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    y = frame.dropna(subset=["macro_liquidity_axis_x", "market_breakdown_axis_y"]).copy()
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    colors = {
        "Normal": "#70ad47",
        "Fragile": "#ffd966",
        "Warning": "#f4b183",
        "Risk-Off": "#c00000",
        "Crisis": "#7f0000",
    }
    for phase, group in y.groupby("risk_phase"):
        ax.scatter(group["macro_liquidity_axis_x"], group["market_breakdown_axis_y"], s=10, alpha=0.55, label=phase, color=colors.get(phase, "#999999"))
    latest = y.iloc[-1]
    ax.scatter([latest["macro_liquidity_axis_x"]], [latest["market_breakdown_axis_y"]], s=130, color="#000000", marker="*", label="Latest")
    ax.axvline(40, color="#bfbfbf", ls="--")
    ax.axvline(55, color="#c00000", ls="--")
    ax.axhline(40, color="#bfbfbf", ls="--")
    ax.axhline(55, color="#c00000", ls="--")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("X: Liquidity / Credit / FX Stress")
    ax.set_ylabel("Y: Equity / Volatility Breakdown")
    ax.set_title("2D Risk Map: Liquidity Stress vs Market Breakdown")
    ax.grid(True, color="#e6e6e6")
    ax.legend(loc="upper left", fontsize=8)
    path = charts_dir / "risk_vector_2d_map.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def create_contact_sheet(paths: list[Path], out_dir: Path) -> Path | None:
    from PIL import Image

    if not paths:
        return None
    thumbs = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        img.thumbnail((720, 480), Image.LANCZOS)
        canvas = Image.new("RGB", (740, 520), "white")
        canvas.paste(img, ((740 - img.width) // 2, 20))
        thumbs.append(canvas)
    cols = 2
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 740, rows * 520), "white")
    for i, img in enumerate(thumbs):
        sheet.paste(img, ((i % cols) * 740, (i // cols) * 520))
    path = out_dir / "risk_vector_vs_nasdaq_yearly_contact_sheet.png"
    sheet.save(path, quality=95)
    return path


def write_report(frame: pd.DataFrame, yearly: pd.DataFrame, chart_paths: list[Path], path: Path) -> None:
    current = frame.tail(1).T
    lines = ["# Risk Vector Dashboard", ""]
    lines.extend(["## Current", current.to_markdown(), ""])
    lines.extend(["## Yearly Summary", yearly.to_markdown(index=False), ""])
    lines.extend(["## Charts"])
    for chart in chart_paths:
        lines.append(f"- {chart}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "- X axis is macro liquidity, credit, volatility, and FX stress.",
            "- Y axis is equity and volatility breakdown plus peak fragility.",
            "- Z axis is external, China/cyclical, supply, and hedge demand stress.",
            "- Archetype explains the trigger mix instead of compressing everything into one risk score.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
