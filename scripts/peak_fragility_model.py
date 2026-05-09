from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DRIVER_PANEL = ROOT / "outputs" / "macro_regime_asset_screener_latest" / "tables" / "driver_panel.csv"
SENTINEL_HISTORY = ROOT / "outputs" / "daily_risk_off_sentinel_latest" / "tables" / "daily_sentinel_history.csv"
OUT_DIR = ROOT / "outputs" / "peak_fragility_model_latest"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a point-in-time peak fragility model for Nasdaq drawdown risk.")
    parser.add_argument("--driver-panel", type=Path, default=DRIVER_PANEL)
    parser.add_argument("--sentinel-history", type=Path, default=SENTINEL_HISTORY)
    parser.add_argument("--output", type=Path, default=OUT_DIR)
    parser.add_argument("--min-train-days", type=int, default=504)
    parser.add_argument("--retrain-step-days", type=int, default=21)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables = args.output / "tables"
    charts = args.output / "charts" / "yearly_peak_fragility_vs_nasdaq"
    reports = args.output / "reports"
    for path in (tables, charts, reports):
        path.mkdir(parents=True, exist_ok=True)

    panel = build_panel(args.driver_panel, args.sentinel_history)
    scored, validation = walkforward_peak_model(panel, args.min_train_days, args.retrain_step_days)
    scored = add_policy(scored)
    yearly_stats = yearly_summary(scored)

    scored.to_csv(tables / "peak_fragility_daily_scores.csv", index=False, encoding="utf-8-sig")
    validation.to_csv(tables / "peak_fragility_validation.csv", index=False, encoding="utf-8-sig")
    yearly_stats.to_csv(tables / "peak_fragility_yearly_summary.csv", index=False, encoding="utf-8-sig")
    chart_paths = create_yearly_charts(scored, charts)
    write_report(validation, yearly_stats, chart_paths, reports / "peak_fragility_report.md")

    print(f"wrote {reports / 'peak_fragility_report.md'}")
    print(validation.to_string(index=False))
    print(f"created {len(chart_paths)} yearly charts in {charts}")


def build_panel(driver_path: Path, sentinel_path: Path) -> pd.DataFrame:
    df = pd.read_csv(driver_path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    sent = pd.read_csv(sentinel_path, parse_dates=["Date"]).sort_values("Date")
    keep = [
        "Date",
        "risk_off_score",
        "sentinel_state",
        "dominant_component",
        "RAI_z",
        "RAI_level_0_100",
        "RAI_20d_change",
        "RAI_shock_score",
        "RAI_overheat_score",
        "ETF_risk_breadth_pct",
        "ETF_breadth_shock_score",
        "SAFE_ROTATION_shock_score",
    ]
    df = df.merge(sent[[c for c in keep if c in sent]], on="Date", how="left")

    for col in df.columns:
        if col != "Date" and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).ffill()

    px = pd.to_numeric(df["NASDAQ100"], errors="coerce")
    df["nasdaq_ret_5d"] = px.pct_change(5)
    df["nasdaq_ret_20d"] = px.pct_change(20)
    df["nasdaq_ret_60d"] = px.pct_change(60)
    df["nasdaq_fwd_5d"] = px.shift(-5) / px - 1.0
    df["nasdaq_fwd_20d"] = px.shift(-20) / px - 1.0
    df["nasdaq_fwd_min_20d"] = px.shift(-1).rolling(20, min_periods=5).min().shift(-19) / px - 1.0

    # Peak-fragility target: downside starts from a high/extended area.
    high_60 = px.rolling(60, min_periods=20).max()
    high_120 = px.rolling(120, min_periods=40).max()
    df["near_60d_high"] = (px / high_60 - 1.0).ge(-0.03).astype(float)
    df["near_120d_high"] = (px / high_120 - 1.0).ge(-0.05).astype(float)
    df["target_peak_fragility_1m"] = (
        (df["near_60d_high"].eq(1.0))
        & ((df["nasdaq_fwd_min_20d"].lt(-0.05)) | (df["nasdaq_fwd_20d"].lt(-0.04)) | (df["nasdaq_fwd_5d"].lt(-0.025)))
    ).astype(int)

    feature_map: dict[str, pd.Series] = {}
    for asset in ["NASDAQ100", "SOX", "SP500", "RUSSELL2000", "HYG_IEF", "DXY", "USDKRW", "USDJPY", "US10Y", "US2Y", "VIX", "VXN", "MOVE", "HY_OAS", "COPPER_GOLD", "GOLD"]:
        if asset not in df:
            continue
        s = pd.to_numeric(df[asset], errors="coerce").ffill()
        for win in (5, 20, 60):
            feature_map[f"{asset}_ret_{win}d"] = s.pct_change(win)
        feature_map[f"{asset}_z_60d"] = rolling_z(s, 60)
        feature_map[f"{asset}_dist_high_60d"] = s / s.rolling(60, min_periods=20).max() - 1.0
        feature_map[f"{asset}_dist_ma_60d"] = s / s.rolling(60, min_periods=20).mean() - 1.0

    nas_ret20 = feature_map.get("NASDAQ100_ret_20d", pd.Series(index=df.index, dtype=float))
    for peer in ["SOX", "SP500", "RUSSELL2000", "HYG_IEF", "COPPER_GOLD"]:
        peer_ret = feature_map.get(f"{peer}_ret_20d")
        if peer_ret is not None:
            feature_map[f"NASDAQ_vs_{peer}_ret20_gap"] = nas_ret20 - peer_ret

    feature_map["nasdaq_momentum_deceleration"] = feature_map["NASDAQ100_ret_20d"] - feature_map["NASDAQ100_ret_60d"]
    feature_map["low_vol_complacency"] = -rolling_z(pd.to_numeric(df["VIX"], errors="coerce").ffill(), 252) if "VIX" in df else 0.0
    feature_map["risk_off_score"] = pd.to_numeric(df.get("risk_off_score", 0.0), errors="coerce").fillna(0.0)
    for col in ["RAI_z", "RAI_level_0_100", "RAI_20d_change", "RAI_shock_score", "RAI_overheat_score", "ETF_risk_breadth_pct", "ETF_breadth_shock_score", "SAFE_ROTATION_shock_score"]:
        if col in df:
            feature_map[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    if "RAI_overheat_score" in feature_map and "RAI_shock_score" in feature_map:
        feature_map["RAI_overheat_to_collapse"] = feature_map["RAI_overheat_score"].shift(20).fillna(0.0) + feature_map["RAI_shock_score"].fillna(0.0)

    engineered = pd.DataFrame(feature_map, index=df.index).replace([np.inf, -np.inf], np.nan)
    out = pd.concat([df, engineered], axis=1)
    out["rule_peak_fragility_score"] = rule_peak_score(out)
    return out.copy()


def rolling_z(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(10, window // 3)).mean()
    std = series.rolling(window, min_periods=max(10, window // 3)).std()
    return (series - mean) / std.replace(0, np.nan)


def rule_peak_score(frame: pd.DataFrame) -> pd.Series:
    def pct_rank(s: pd.Series, inverse: bool = False) -> pd.Series:
        r = s.rolling(252, min_periods=60).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        return 1.0 - r if inverse else r

    near_high = pd.to_numeric(frame["near_60d_high"], errors="coerce").fillna(0.0)
    extended = pct_rank(pd.to_numeric(frame.get("NASDAQ100_dist_ma_60d", 0.0), errors="coerce"))
    decel = pct_rank(-pd.to_numeric(frame.get("nasdaq_momentum_deceleration", 0.0), errors="coerce"))
    vix_complacency = pct_rank(pd.to_numeric(frame.get("low_vol_complacency", 0.0), errors="coerce"))
    sox_div = pct_rank(pd.to_numeric(frame.get("NASDAQ_vs_SOX_ret20_gap", 0.0), errors="coerce"))
    small_div = pct_rank(pd.to_numeric(frame.get("NASDAQ_vs_RUSSELL2000_ret20_gap", 0.0), errors="coerce"))
    credit_div = pct_rank(-pd.to_numeric(frame.get("HYG_IEF_ret_20d", 0.0), errors="coerce"))
    dollar_tight = pct_rank(pd.to_numeric(frame.get("DXY_ret_20d", 0.0), errors="coerce"))
    rate_tight = pct_rank(pd.to_numeric(frame.get("US10Y_ret_20d", 0.0), errors="coerce"))
    score = (
        100.0
        * (
            0.18 * near_high
            + 0.14 * extended
            + 0.16 * decel
            + 0.12 * vix_complacency
            + 0.12 * sox_div
            + 0.10 * small_div
            + 0.08 * credit_div
            + 0.05 * dollar_tight
            + 0.05 * rate_tight
        )
    )
    return score.clip(0, 100).fillna(0.0)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    suffixes = ("_ret_5d", "_ret_20d", "_ret_60d", "_z_60d", "_dist_high_60d", "_dist_ma_60d")
    cols = [
        c
        for c in frame.columns
        if c.endswith(suffixes)
        or c.startswith("NASDAQ_vs_")
        or c
        in {
            "near_60d_high",
            "near_120d_high",
            "nasdaq_momentum_deceleration",
            "low_vol_complacency",
            "rule_peak_fragility_score",
            "risk_off_score",
            "RAI_z",
            "RAI_level_0_100",
            "RAI_20d_change",
            "RAI_shock_score",
            "RAI_overheat_score",
            "RAI_overheat_to_collapse",
            "ETF_risk_breadth_pct",
            "ETF_breadth_shock_score",
            "SAFE_ROTATION_shock_score",
        }
    ]
    blocked = {"nasdaq_fwd_5d", "nasdaq_fwd_20d", "nasdaq_fwd_min_20d", "target_peak_fragility_1m"}
    return [c for c in cols if c not in blocked and pd.api.types.is_numeric_dtype(frame[c])]


def walkforward_peak_model(panel: pd.DataFrame, min_train_days: int, retrain_step_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, brier_score_loss, precision_score, recall_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    out = panel.copy().sort_values("Date").reset_index(drop=True)
    target = "target_peak_fragility_1m"
    cols = feature_columns(out)
    probs = pd.Series(np.nan, index=out.index, dtype=float)
    last_models: list[Any] | None = None
    last_train_end = -10**9

    for i in range(len(out)):
        if i < min_train_days:
            continue
        train = out.iloc[:i].dropna(subset=[target])
        if train[target].nunique() < 2:
            continue
        if last_models is None or i - last_train_end >= retrain_step_days:
            x_train = train[cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            y_train = train[target].astype(int)
            models = [
                make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5, random_state=42)),
                RandomForestClassifier(n_estimators=80, max_depth=4, min_samples_leaf=12, class_weight="balanced_subsample", random_state=42),
            ]
            fitted = []
            for model in models:
                try:
                    model.fit(x_train, y_train)
                    fitted.append(model)
                except Exception:
                    continue
            last_models = fitted
            last_train_end = i
        if not last_models:
            continue
        x_test = out.loc[[i], cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        model_probs = [float(model.predict_proba(x_test)[:, 1][0]) for model in last_models]
        probs.iloc[i] = float(np.mean(model_probs))

    out["peak_fragility_prob"] = probs
    rule = pd.to_numeric(out["rule_peak_fragility_score"], errors="coerce").fillna(0.0) / 100.0
    out["peak_fragility_score_0_100"] = (100.0 * (0.65 * out["peak_fragility_prob"].fillna(0.0) + 0.35 * rule)).clip(0, 100)

    valid = out.dropna(subset=["peak_fragility_prob", target]).copy()
    threshold, stats = choose_threshold(valid["peak_fragility_score_0_100"], valid[target])
    out["peak_fragility_signal"] = out["peak_fragility_score_0_100"].ge(threshold).astype(int)
    pred = valid["peak_fragility_score_0_100"].ge(threshold).astype(int)
    actual = valid[target].astype(int)
    validation = pd.DataFrame(
        [
            {
                "target": target,
                "samples": int(valid.shape[0]),
                "positive_rate": float(actual.mean()),
                "threshold": threshold,
                "signal_rate": float(pred.mean()),
                "accuracy": accuracy_score(actual, pred),
                "precision": precision_score(actual, pred, zero_division=0),
                "recall": recall_score(actual, pred, zero_division=0),
                "brier": brier_score_loss(actual, valid["peak_fragility_prob"].clip(0, 1)),
                "roc_auc": roc_auc_score(actual, valid["peak_fragility_prob"]) if actual.nunique() > 1 else np.nan,
                **stats,
            }
        ]
    )
    out["peak_fragility_threshold"] = threshold
    return out, validation


def choose_threshold(score: pd.Series, actual: pd.Series) -> tuple[float, dict[str, Any]]:
    best_score = -np.inf
    best = (60.0, {"threshold_precision": 0.0, "threshold_recall": 0.0, "threshold_signals": 0})
    base_rate = float(actual.mean())
    for threshold in np.arange(35, 86, 1):
        signal = score.ge(threshold)
        if signal.sum() < max(10, int(len(score) * 0.03)) or signal.mean() > 0.35:
            continue
        tp = int((signal & actual.eq(1)).sum())
        fp = int((signal & actual.eq(0)).sum())
        fn = int((~signal & actual.eq(1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        lift = precision / max(base_rate, 1e-9)
        objective = 1.7 * recall + 0.8 * precision + 0.2 * min(lift, 5.0) - 0.25 * float(signal.mean())
        if objective > best_score:
            best_score = objective
            best = (float(threshold), {"threshold_precision": precision, "threshold_recall": recall, "threshold_signals": int(signal.sum())})
    return best


def add_policy(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    threshold = float(out["peak_fragility_threshold"].dropna().iloc[-1]) if "peak_fragility_threshold" in out and out["peak_fragility_threshold"].notna().any() else 60.0
    out["peak_fragility_state"] = pd.cut(
        out["peak_fragility_score_0_100"],
        bins=[-1, max(35, threshold - 15), threshold, min(100, threshold + 15), 101],
        labels=["Normal", "Fragile", "Peak Warning", "Peak Alert"],
        right=False,
    ).astype(str)
    return out


def yearly_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, y in frame.groupby(frame["Date"].dt.year):
        yy = y.dropna(subset=["NASDAQ100"])
        if yy.empty:
            continue
        rows.append(
            {
                "year": int(year),
                "start": yy["Date"].min().date().isoformat(),
                "end": yy["Date"].max().date().isoformat(),
                "nasdaq_return": float(yy["NASDAQ100"].iloc[-1] / yy["NASDAQ100"].iloc[0] - 1.0),
                "max_peak_fragility_score": float(yy["peak_fragility_score_0_100"].max()),
                "peak_warning_days": int(yy["peak_fragility_state"].isin(["Peak Warning", "Peak Alert"]).sum()),
                "peak_alert_days": int(yy["peak_fragility_state"].eq("Peak Alert").sum()),
                "max_risk_off_score": float(numeric_column(yy, "risk_off_score").max()),
            }
        )
    return pd.DataFrame(rows)


def numeric_column(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    value = frame[column]
    if isinstance(value, pd.DataFrame):
        value = value.iloc[:, 0]
    return pd.to_numeric(value, errors="coerce").fillna(default)


def create_yearly_charts(frame: pd.DataFrame, out_dir: Path) -> list[Path]:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from PIL import Image

    paths: list[Path] = []
    for year, y in frame.groupby(frame["Date"].dt.year):
        y = y.dropna(subset=["NASDAQ100"]).copy()
        if y.empty:
            continue
        fig, ax1 = plt.subplots(figsize=(15, 7.5), dpi=150)
        fig.patch.set_facecolor("white")
        ax1.set_facecolor("#fbfbfb")
        ax1.plot(y["Date"], y["NASDAQ100"], color="#1f4e79", lw=2.0, label="Nasdaq 100")
        ax1.set_ylabel("Nasdaq 100", color="#1f4e79")
        ax1.tick_params(axis="y", labelcolor="#1f4e79")
        ax1.grid(True, axis="y", color="#d9d9d9", lw=0.8)

        ax2 = ax1.twinx()
        ax2.plot(y["Date"], y["peak_fragility_score_0_100"], color="#7030a0", lw=1.8, label="Peak Fragility Score")
        ax2.plot(y["Date"], y["risk_off_score"], color="#c00000", lw=1.2, alpha=0.6, label="Risk-Off Score")
        threshold = float(y["peak_fragility_threshold"].dropna().iloc[-1]) if y["peak_fragility_threshold"].notna().any() else 60.0
        for lvl, color, label in [(max(35, threshold - 15), "#b4c7e7", "Fragile"), (threshold, "#7030a0", "Peak Warning"), (min(100, threshold + 15), "#c00000", "Peak Alert")]:
            ax2.axhline(lvl, color=color, lw=1.0, ls="--", alpha=0.85)
            ax2.text(y["Date"].min(), lvl + 1, label, color=color, fontsize=9, va="bottom")
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("Score", color="#7030a0")
        ax2.tick_params(axis="y", labelcolor="#7030a0")

        warn = y[y["peak_fragility_state"].isin(["Peak Warning", "Peak Alert"])]
        ax1.scatter(warn["Date"], warn["NASDAQ100"], s=18, color="#7030a0", alpha=0.75, label="Peak Warning Days", zorder=4)

        title = f"{int(year)} Peak Fragility / Risk-Off vs Nasdaq 100"
        subtitle = f"Nasdaq return {y['NASDAQ100'].iloc[-1] / y['NASDAQ100'].iloc[0] - 1.0:+.1%} | max peak fragility {y['peak_fragility_score_0_100'].max():.1f}"
        ax1.set_title(title + "\n" + subtitle, fontsize=15, pad=15)
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True, fontsize=9)
        fig.tight_layout()
        path = out_dir / f"peak_fragility_vs_nasdaq_{int(year)}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    thumbs = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        img.thumbnail((720, 360), Image.LANCZOS)
        canvas = Image.new("RGB", (740, 400), "white")
        canvas.paste(img, ((740 - img.width) // 2, 20))
        thumbs.append(canvas)
    if thumbs:
        cols = 2
        rows = (len(thumbs) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * 740, rows * 400), "white")
        for i, img in enumerate(thumbs):
            sheet.paste(img, ((i % cols) * 740, (i // cols) * 400))
        sheet_path = out_dir / "peak_fragility_vs_nasdaq_yearly_contact_sheet.png"
        sheet.save(sheet_path, quality=95)
        paths.append(sheet_path)
    return paths


def write_report(validation: pd.DataFrame, yearly_stats: pd.DataFrame, chart_paths: list[Path], path: Path) -> None:
    lines = ["# Peak Fragility Model", ""]
    lines.extend(["## Validation", validation.to_markdown(index=False), ""])
    lines.extend(["## Yearly Summary", yearly_stats.to_markdown(index=False), ""])
    lines.extend(["## Charts"])
    for chart in chart_paths:
        lines.append(f"- {chart}")
    lines.extend(
        [
            "",
            "## Notes",
            "- Peak Fragility is separate from Risk-Off Sentinel.",
            "- It targets high/extended Nasdaq conditions that later suffer a 1-month drawdown or fast 1-week downside.",
            "- It is trained point-in-time with expanding walk-forward models and combined with a rule score.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
