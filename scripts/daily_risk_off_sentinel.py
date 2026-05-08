from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from macro_regime_asset_screener import (  # noqa: E402
    ASSETS,
    FRED_SERIES,
    YF_SERIES,
    load_asset_histories,
    load_driver_series,
    make_driver_panel,
    safe_to_csv,
)

OUT_DIR = ROOT / "outputs" / "daily_risk_off_sentinel_latest"
VALIDATION_OUT_DIR = ROOT / "outputs" / "rwkv_lppl_walkforward_validation_latest"
RWKV_OUT_DIR = ROOT / "outputs" / "rwkv_lppl_asset_screener_latest"
MACRO_OUT_DIR = ROOT / "outputs" / "macro_regime_asset_screener_latest"


@dataclass(frozen=True)
class ShockSpec:
    name: str
    component: str
    direction: int
    weight: float
    mode: str = "auto"


SHOCK_SPECS = [
    ShockSpec("VIX", "volatility", 1, 1.25, "diff"),
    ShockSpec("VXN", "volatility", 1, 0.70, "diff"),
    ShockSpec("MOVE", "volatility", 1, 0.85, "diff"),
    ShockSpec("HY_OAS", "credit", 1, 1.30, "diff"),
    ShockSpec("IG_OAS", "credit", 1, 0.75, "diff"),
    ShockSpec("HYG_IEF", "credit", -1, 0.80, "return"),
    ShockSpec("DXY", "fx", 1, 0.70, "return"),
    ShockSpec("USDKRW", "fx", 1, 1.10, "return"),
    ShockSpec("USDCNH", "fx", 1, 0.80, "return"),
    ShockSpec("SP500", "equity", -1, 0.75, "return"),
    ShockSpec("NASDAQ100", "equity", -1, 0.90, "return"),
    ShockSpec("SOX", "equity", -1, 1.10, "return"),
    ShockSpec("RUSSELL2000", "equity", -1, 0.70, "return"),
    ShockSpec("COPPER", "cyclical", -1, 0.75, "return"),
    ShockSpec("COPPER_GOLD", "cyclical", -1, 0.95, "return"),
    ShockSpec("CSI300", "cyclical", -1, 0.50, "return"),
    ShockSpec("HANGSENG_TECH", "cyclical", -1, 0.55, "return"),
    ShockSpec("WTI", "supply_shock", 1, 0.60, "return"),
    ShockSpec("GOLD", "hedge_bid", 1, 0.45, "return"),
    ShockSpec("NFCI", "liquidity", 1, 0.65, "diff"),
    ShockSpec("ANFCI", "liquidity", 1, 0.65, "diff"),
]

RISK_GROUPS = {
    "Korea broad equity",
    "Korea growth",
    "Korea semiconductor",
    "Korea IT",
    "Korea cyclical",
    "Korea value",
    "US broad equity",
    "US growth",
    "US semiconductor",
    "US cyclical",
    "US REIT",
    "China/HK growth",
    "China equity",
    "India/EM",
    "Japan equity",
    "US high yield",
    "Oil",
}
DEFENSIVE_GROUPS = {"Korea defensive", "Korea bonds", "US long bonds", "US IG bonds", "Gold", "USD cash", "Cash/short bonds"}
SAFE_GROUPS = {"USD cash", "Cash/short bonds"}
HEDGE_GROUPS = {"Gold", "Korea bonds", "US long bonds", "US IG bonds"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Daily risk-off sentinel overlay for RWKV/LPPL macro asset scores.")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--input", type=Path, default=VALIDATION_OUT_DIR)
    parser.add_argument("--rwkv-input", type=Path, default=RWKV_OUT_DIR)
    parser.add_argument("--output", type=Path, default=OUT_DIR)
    parser.add_argument("--benchmark", default="069500.KS", help="Primary crash-validation benchmark.")
    parser.add_argument("--forward-days", type=int, default=20)
    parser.add_argument("--crash-threshold", type=float, default=-0.07)
    parser.add_argument("--episode-lookback-days", type=int, default=20)
    parser.add_argument("--watch-threshold", type=float, default=25.0)
    parser.add_argument("--derisk-threshold", type=float, default=55.0)
    parser.add_argument("--cash-threshold", type=float, default=70.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables = args.output / "tables"
    reports = args.output / "reports"
    for path in (tables, reports):
        path.mkdir(parents=True, exist_ok=True)

    specs = FRED_SERIES + YF_SERIES
    raw, availability = load_driver_series(specs, args.start, args.skip_download)
    driver_panel = make_driver_panel(raw)
    if driver_panel.empty:
        raise SystemExit("No driver data available.")

    sentinel = build_sentinel(driver_panel, args.watch_threshold, args.derisk_threshold, args.cash_threshold)
    current_scores = read_current_scores(args.input, args.rwkv_input)
    overlay = overlay_current_scores(current_scores, sentinel)
    histories = load_asset_histories([asset.symbol for asset in ASSETS], args.start, args.skip_download)
    benchmark_validation = validate_crash_warning(
        sentinel,
        histories,
        args.benchmark,
        args.forward_days,
        args.crash_threshold,
        args.watch_threshold,
    )
    threshold_sweep = validate_threshold_sweep(
        sentinel,
        histories,
        args.benchmark,
        args.forward_days,
        args.crash_threshold,
    )
    episode_validation = validate_crash_episodes(
        sentinel,
        histories,
        args.benchmark,
        args.forward_days,
        args.crash_threshold,
        args.watch_threshold,
        args.episode_lookback_days,
    )
    asset_validation = validate_assets_by_group(
        sentinel,
        histories,
        args.forward_days,
        args.crash_threshold,
        args.watch_threshold,
    )
    events = event_case_studies(sentinel, histories)

    safe_to_csv(sentinel.reset_index().rename(columns={"index": "Date"}), tables / "daily_sentinel_history.csv")
    safe_to_csv(overlay, tables / "sentinel_adjusted_current_scores.csv")
    safe_to_csv(benchmark_validation, tables / "sentinel_benchmark_validation.csv")
    safe_to_csv(threshold_sweep, tables / "sentinel_threshold_sweep.csv")
    safe_to_csv(episode_validation, tables / "sentinel_crash_episode_validation.csv")
    safe_to_csv(asset_validation, tables / "sentinel_asset_group_validation.csv")
    safe_to_csv(events, tables / "sentinel_event_case_studies.csv")
    pd.DataFrame(availability).to_csv(tables / "data_availability.csv", index=False, encoding="utf-8-sig")
    write_report(
        sentinel,
        overlay,
        benchmark_validation,
        threshold_sweep,
        episode_validation,
        asset_validation,
        events,
        reports / "daily_sentinel_report.md",
    )

    print(f"wrote {tables / 'sentinel_adjusted_current_scores.csv'}")
    print(overlay.head(15).to_string(index=False))


def build_sentinel(panel: pd.DataFrame, watch: float, derisk: float, cash: float) -> pd.DataFrame:
    panel = panel.sort_index().ffill()
    rows: dict[str, pd.Series] = {}
    component_scores: dict[str, list[pd.Series]] = {}
    contribution_cols: dict[str, pd.Series] = {}

    for spec in SHOCK_SPECS:
        if spec.name not in panel:
            continue
        base = panel[spec.name].astype(float)
        move = signed_move(base, spec.mode) * spec.direction
        score = shock_score(move, spec.weight)
        rows[f"{spec.name}_shock_score"] = score
        contribution_cols[spec.name] = score
        component_scores.setdefault(spec.component, []).append(score)

    out = pd.DataFrame(rows, index=panel.index)
    for component, scores in component_scores.items():
        out[f"{component}_score"] = pd.concat(scores, axis=1).mean(axis=1)

    component_weights = {
        "volatility_score": 1.25,
        "credit_score": 1.25,
        "fx_score": 1.05,
        "equity_score": 1.20,
        "cyclical_score": 0.95,
        "supply_shock_score": 0.55,
        "hedge_bid_score": 0.35,
        "liquidity_score": 0.75,
    }
    weighted = []
    weights = []
    for col, weight in component_weights.items():
        if col in out:
            weighted.append(out[col] * weight)
            weights.append(weight)
    if weighted:
        raw = pd.concat(weighted, axis=1).sum(axis=1) / sum(weights)
    else:
        raw = pd.Series(0.0, index=panel.index)

    broad_confirmation = pd.concat(
        [out.get(col, pd.Series(index=out.index, dtype=float)) for col in ["volatility_score", "credit_score", "fx_score", "equity_score"]],
        axis=1,
    ).gt(35).sum(axis=1)
    confirmation_boost = np.select(
        [broad_confirmation.ge(4), broad_confirmation.ge(3), broad_confirmation.ge(2)],
        [16.0, 10.0, 5.0],
        default=0.0,
    )
    persistence = raw.ewm(span=3, adjust=False).mean()
    out["risk_off_score_raw"] = raw.clip(0, 100)
    out["risk_off_score"] = (0.65 * raw + 0.35 * persistence + confirmation_boost).clip(0, 100)
    out["risk_off_momentum_5d"] = out["risk_off_score"].diff(5)
    out["dominant_component"] = dominant_component(out)
    out["sentinel_state"] = state_machine(out["risk_off_score"], watch, derisk, cash)
    out["risk_budget_pct"] = out["sentinel_state"].map({"Normal": 100, "Watch": 70, "De-risk": 35, "Cash": 10}).astype(float)
    out["equity_penalty"] = out["sentinel_state"].map({"Normal": 0, "Watch": 8, "De-risk": 20, "Cash": 38}).astype(float)
    out["safe_asset_boost"] = out["sentinel_state"].map({"Normal": 0, "Watch": 3, "De-risk": 8, "Cash": 16}).astype(float)
    return out.dropna(how="all")


def signed_move(series: pd.Series, mode: str) -> pd.Series:
    if mode == "diff":
        return series.diff()
    if mode == "return":
        return series.pct_change()
    if series.abs().median(skipna=True) > 20:
        return series.diff()
    return series.pct_change()


def shock_score(move: pd.Series, weight: float) -> pd.Series:
    std = move.rolling(252, min_periods=60).std()
    z1 = move / std.replace(0, np.nan)
    z5 = move.rolling(5, min_periods=3).sum() / (std * math.sqrt(5)).replace(0, np.nan)
    z20 = move.rolling(20, min_periods=10).sum() / (std * math.sqrt(20)).replace(0, np.nan)
    shock = pd.concat([z1, z5, z20], axis=1).max(axis=1).clip(lower=0)
    score = 32.0 * shock * weight
    return score.clip(0, 100)


def dominant_component(frame: pd.DataFrame) -> pd.Series:
    cols = [c for c in frame.columns if c.endswith("_score") and c not in {"risk_off_score", "risk_off_score_raw"}]
    if not cols:
        return pd.Series("none", index=frame.index)
    labels = frame[cols].idxmax(axis=1).str.replace("_score", "", regex=False)
    return labels.fillna("none")


def state_machine(score: pd.Series, watch: float, derisk: float, cash: float) -> pd.Series:
    states = []
    previous = "Normal"
    for value in score.fillna(0):
        if value >= cash:
            state = "Cash"
        elif value >= derisk:
            state = "De-risk"
        elif value >= watch:
            state = "Watch"
        else:
            state = "Normal"
        if previous == "Cash" and value >= derisk - 5:
            state = "Cash"
        elif previous == "De-risk" and value >= watch - 5:
            state = "De-risk" if state in {"Normal", "Watch"} else state
        previous = state
        states.append(state)
    return pd.Series(states, index=score.index)


def read_current_scores(validation_dir: Path, rwkv_dir: Path) -> pd.DataFrame:
    candidates = [
        validation_dir / "tables" / "calibrated_current_asset_scores.csv",
        rwkv_dir / "tables" / "current_asset_scores_rwkv_lppl.csv",
        MACRO_OUT_DIR / "tables" / "current_asset_scores.csv",
    ]
    frames = []
    for path in candidates:
        if path.exists():
            frame = pd.read_csv(path)
            frame.attrs["source_path"] = str(path)
            frames.append(frame)
    if frames:
        return max(frames, key=lambda frame: frame["symbol"].nunique() if "symbol" in frame else frame.shape[0])
    raise SystemExit("Missing current asset score table. Run RWKV/LPPL and walk-forward calibration first.")


def overlay_current_scores(scores: pd.DataFrame, sentinel: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return scores
    latest = sentinel.dropna(subset=["risk_off_score"]).iloc[-1]
    out = scores.copy()
    out["sentinel_asof"] = latest.name.date().isoformat()
    out["sentinel_state"] = latest["sentinel_state"]
    out["sentinel_risk_off_score"] = round(float(latest["risk_off_score"]), 2)
    out["sentinel_dominant_component"] = latest["dominant_component"]
    out["risk_budget_pct"] = float(latest["risk_budget_pct"])

    base_score_col = "institutional_score_0_100" if "institutional_score_0_100" in out else "score_0_100"
    p1_col = "meta_prob_1w" if "meta_prob_1w" in out else "upside_prob_1w"
    p4_col = "meta_prob_4w" if "meta_prob_4w" in out else "upside_prob_4w"
    out["score_before_sentinel"] = pd.to_numeric(out[base_score_col], errors="coerce")
    out["prob_1w_before_sentinel"] = pd.to_numeric(out[p1_col], errors="coerce")
    out["prob_4w_before_sentinel"] = pd.to_numeric(out[p4_col], errors="coerce")
    out["sentinel_adjustment"] = out.apply(lambda row: group_adjustment(str(row.get("group", "")), latest), axis=1)
    out["sentinel_adjusted_score_0_100"] = (out["score_before_sentinel"] + out["sentinel_adjustment"]).clip(0, 100).round(2)

    prob_shift = out["sentinel_adjustment"] / 100.0
    out["sentinel_adjusted_prob_1w"] = (out["prob_1w_before_sentinel"] + 0.75 * prob_shift).clip(0.01, 0.99).round(4)
    out["sentinel_adjusted_prob_4w"] = (out["prob_4w_before_sentinel"] + 0.90 * prob_shift).clip(0.01, 0.99).round(4)
    out["sentinel_action"] = out.apply(lambda row: asset_action(str(row.get("group", "")), str(latest["sentinel_state"])), axis=1)
    out = out.sort_values(["sentinel_adjusted_score_0_100", "sentinel_adjusted_prob_4w"], ascending=False).reset_index(drop=True)
    out["sentinel_rank"] = np.arange(1, len(out) + 1)
    return out


def group_adjustment(group: str, latest: pd.Series) -> float:
    state = str(latest["sentinel_state"])
    risk_score = float(latest["risk_off_score"])
    if group in SAFE_GROUPS:
        return float(latest["safe_asset_boost"])
    if group in HEDGE_GROUPS:
        return {"Normal": 0, "Watch": 2, "De-risk": 6, "Cash": 10}.get(state, 0)
    if group in RISK_GROUPS:
        penalty = float(latest["equity_penalty"])
        if group in {"US high yield", "Korea growth", "Korea semiconductor", "Korea IT", "US growth", "US semiconductor", "China/HK growth", "India/EM"}:
            penalty *= 1.15
        if group == "Oil" and str(latest["dominant_component"]) == "supply_shock":
            penalty *= 0.45
        return -min(45.0, penalty + max(0.0, risk_score - 70.0) * 0.20)
    if group in DEFENSIVE_GROUPS:
        return {"Normal": 0, "Watch": 1, "De-risk": 3, "Cash": 5}.get(state, 0)
    return 0.0


def asset_action(group: str, state: str) -> str:
    if state == "Cash":
        if group in SAFE_GROUPS:
            return "core cash"
        if group in HEDGE_GROUPS:
            return "hedge only"
        return "avoid risk"
    if state == "De-risk":
        if group in SAFE_GROUPS:
            return "raise cash"
        if group in HEDGE_GROUPS:
            return "defensive sleeve"
        if group in RISK_GROUPS:
            return "cut weight"
    if state == "Watch" and group in RISK_GROUPS:
        return "size down"
    return "normal"


def validate_crash_warning(
    sentinel: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    symbol: str,
    forward_days: int,
    crash_threshold: float,
    watch_threshold: float,
) -> pd.DataFrame:
    hist = histories.get(symbol)
    if hist is None or hist.empty or "Close" not in hist:
        return pd.DataFrame()
    close = hist["Close"].dropna()
    joined = pd.concat([sentinel["risk_off_score"], sentinel["sentinel_state"], close.rename("close")], axis=1).ffill().dropna()
    joined["forward_return"] = joined["close"].shift(-forward_days) / joined["close"] - 1.0
    joined["forward_drawdown"] = forward_drawdown(joined["close"], forward_days)
    joined["crash_label"] = joined["forward_drawdown"].le(crash_threshold)
    joined["warning"] = joined["risk_off_score"].ge(watch_threshold)
    joined = joined.dropna(subset=["forward_drawdown"])
    if joined.empty:
        return pd.DataFrame()
    return validation_metrics(joined, "benchmark", symbol)


def validate_threshold_sweep(
    sentinel: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    symbol: str,
    forward_days: int,
    crash_threshold: float,
) -> pd.DataFrame:
    hist = histories.get(symbol)
    if hist is None or hist.empty or "Close" not in hist:
        return pd.DataFrame()
    close = hist["Close"].dropna()
    joined = pd.concat([sentinel["risk_off_score"], sentinel["sentinel_state"], close.rename("close")], axis=1).ffill().dropna()
    joined["forward_return"] = joined["close"].shift(-forward_days) / joined["close"] - 1.0
    joined["forward_drawdown"] = forward_drawdown(joined["close"], forward_days)
    joined["crash_label"] = joined["forward_drawdown"].le(crash_threshold)
    joined = joined.dropna(subset=["forward_drawdown"])
    rows = []
    for threshold in range(15, 86, 5):
        test = joined.copy()
        test["warning"] = test["risk_off_score"].ge(threshold)
        metric = validation_metrics(test, "benchmark", symbol)
        metric.insert(2, "threshold", threshold)
        rows.append(metric)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def validate_crash_episodes(
    sentinel: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    symbol: str,
    forward_days: int,
    crash_threshold: float,
    watch_threshold: float,
    lookback_days: int,
) -> pd.DataFrame:
    hist = histories.get(symbol)
    if hist is None or hist.empty or "Close" not in hist:
        return pd.DataFrame()
    close = hist["Close"].dropna()
    joined = pd.concat([sentinel["risk_off_score"], close.rename("close")], axis=1).ffill().dropna()
    joined["forward_drawdown"] = forward_drawdown(joined["close"], forward_days)
    joined["crash_label"] = joined["forward_drawdown"].le(crash_threshold)
    joined["warning"] = joined["risk_off_score"].ge(watch_threshold)
    joined = joined.dropna(subset=["forward_drawdown"])
    if joined.empty:
        return pd.DataFrame()

    starts = []
    last_start: pd.Timestamp | None = None
    previous = False
    for date, is_crash in joined["crash_label"].items():
        if bool(is_crash) and not previous:
            if last_start is None or (pd.Timestamp(date) - last_start).days > lookback_days:
                starts.append(pd.Timestamp(date))
                last_start = pd.Timestamp(date)
        previous = bool(is_crash)

    rows = []
    for start in starts:
        prior = joined.loc[start - pd.Timedelta(days=lookback_days) : start]
        warnings = prior[prior["warning"]]
        first_warning = None if warnings.empty else pd.Timestamp(warnings.index[0])
        max_score = prior["risk_off_score"].max()
        rows.append(
            {
                "symbol": symbol,
                "crash_start": start.date().isoformat(),
                "detected_before_or_at_start": not warnings.empty,
                "first_warning_date": date_str(first_warning),
                "lead_days": (start - first_warning).days if first_warning is not None else None,
                "max_pre_start_score": round_float(max_score),
                "drawdown_next_window": round_float(joined.loc[start, "forward_drawdown"]),
            }
        )
    episodes = pd.DataFrame(rows)
    if episodes.empty:
        return episodes
    summary = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "crash_episodes": int(episodes.shape[0]),
                "detected_episodes": int(episodes["detected_before_or_at_start"].sum()),
                "episode_detection_rate": round_float(episodes["detected_before_or_at_start"].mean()),
                "median_lead_days": round_float(pd.to_numeric(episodes["lead_days"], errors="coerce").median()),
                "avg_max_pre_start_score": round_float(episodes["max_pre_start_score"].mean()),
            }
        ]
    )
    summary.attrs["episodes"] = episodes
    return pd.concat([summary.assign(row_type="summary"), episodes.assign(row_type="episode")], ignore_index=True, sort=False)


def validate_assets_by_group(
    sentinel: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    forward_days: int,
    crash_threshold: float,
    watch_threshold: float,
) -> pd.DataFrame:
    rows = []
    asset_map = {asset.symbol: asset for asset in ASSETS}
    for symbol, hist in histories.items():
        asset = asset_map.get(symbol)
        if asset is None or hist.empty or "Close" not in hist:
            continue
        close = hist["Close"].dropna()
        joined = pd.concat([sentinel["risk_off_score"], sentinel["sentinel_state"], close.rename("close")], axis=1).ffill().dropna()
        joined["forward_return"] = joined["close"].shift(-forward_days) / joined["close"] - 1.0
        joined["forward_drawdown"] = forward_drawdown(joined["close"], forward_days)
        joined["crash_label"] = joined["forward_drawdown"].le(crash_threshold)
        joined["warning"] = joined["risk_off_score"].ge(watch_threshold)
        joined = joined.dropna(subset=["forward_drawdown"])
        if joined.empty:
            continue
        metric = validation_metrics(joined, asset.group, symbol)
        metric["name"] = asset.name
        rows.append(metric)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def forward_drawdown(close: pd.Series, forward_days: int) -> pd.Series:
    values = []
    arr = close.to_numpy(dtype=float)
    for idx in range(len(arr)):
        future = arr[idx + 1 : idx + forward_days + 1]
        if len(future) == 0 or not np.isfinite(arr[idx]) or arr[idx] == 0:
            values.append(np.nan)
        else:
            values.append(float(np.nanmin(future / arr[idx] - 1.0)))
    return pd.Series(values, index=close.index)


def validation_metrics(frame: pd.DataFrame, group: str, symbol: str) -> pd.DataFrame:
    tp = int((frame["warning"] & frame["crash_label"]).sum())
    fp = int((frame["warning"] & ~frame["crash_label"]).sum())
    fn = int((~frame["warning"] & frame["crash_label"]).sum())
    tn = int((~frame["warning"] & ~frame["crash_label"]).sum())
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    false_alarm = fp / (tp + fp) if tp + fp else np.nan
    lead_days = warning_lead_days(frame)
    return pd.DataFrame(
        [
            {
                "group": group,
                "symbol": symbol,
                "samples": int(frame.shape[0]),
                "crash_days": int(frame["crash_label"].sum()),
                "warning_days": int(frame["warning"].sum()),
                "true_positive_days": tp,
                "false_positive_days": fp,
                "false_negative_days": fn,
                "true_negative_days": tn,
                "precision": round_float(precision),
                "recall": round_float(recall),
                "false_alarm_rate": round_float(false_alarm),
                "median_warning_lead_days": round_float(np.nanmedian(lead_days) if lead_days else np.nan),
                "avg_forward_return_when_warning": round_float(frame.loc[frame["warning"], "forward_return"].mean()),
                "avg_forward_drawdown_when_warning": round_float(frame.loc[frame["warning"], "forward_drawdown"].mean()),
            }
        ]
    )


def warning_lead_days(frame: pd.DataFrame) -> list[int]:
    crash_dates = frame.index[frame["crash_label"]].tolist()
    warning_dates = frame.index[frame["warning"]].tolist()
    leads = []
    for crash_date in crash_dates:
        prior = [d for d in warning_dates if d <= crash_date and (crash_date - d).days <= 20]
        if prior:
            leads.append((crash_date - prior[-1]).days)
    return leads


def event_case_studies(sentinel: pd.DataFrame, histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    cases = [
        ("COVID crash", "2020-01-01", "2020-04-30"),
        ("Liberation Day tariff shock", "2025-03-01", "2025-05-31"),
        ("Iran war oil shock", "2026-02-01", "2026-05-06"),
    ]
    rows = []
    benchmark_close = histories.get("069500.KS", pd.DataFrame()).get("Close", pd.Series(dtype=float))
    for name, start, end in cases:
        window = sentinel.loc[pd.Timestamp(start) : pd.Timestamp(end)]
        if window.empty:
            continue
        first_watch = first_state_date(window, {"Watch", "De-risk", "Cash"})
        first_derisk = first_state_date(window, {"De-risk", "Cash"})
        first_cash = first_state_date(window, {"Cash"})
        bench = benchmark_close.loc[pd.Timestamp(start) : pd.Timestamp(end)].dropna()
        rows.append(
            {
                "event": name,
                "window_start": start,
                "window_end": end,
                "max_risk_off_score": round_float(window["risk_off_score"].max()),
                "max_state": max_state(window["sentinel_state"]),
                "first_watch_or_worse": date_str(first_watch),
                "first_derisk_or_worse": date_str(first_derisk),
                "first_cash": date_str(first_cash),
                "dominant_component_at_max": str(window.loc[window["risk_off_score"].idxmax(), "dominant_component"]),
                "benchmark_return_window": round_float(bench.iloc[-1] / bench.iloc[0] - 1.0) if bench.shape[0] > 1 else np.nan,
                "benchmark_max_drawdown_window": round_float((bench / bench.cummax() - 1.0).min()) if bench.shape[0] > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def first_state_date(window: pd.DataFrame, states: set[str]) -> pd.Timestamp | None:
    hit = window[window["sentinel_state"].isin(states)]
    return None if hit.empty else pd.Timestamp(hit.index[0])


def max_state(states: pd.Series) -> str:
    order = {"Normal": 0, "Watch": 1, "De-risk": 2, "Cash": 3}
    rev = {v: k for k, v in order.items()}
    return rev.get(max(order.get(str(s), 0) for s in states.dropna()), "Normal")


def write_report(
    sentinel: pd.DataFrame,
    overlay: pd.DataFrame,
    benchmark_validation: pd.DataFrame,
    threshold_sweep: pd.DataFrame,
    episode_validation: pd.DataFrame,
    asset_validation: pd.DataFrame,
    events: pd.DataFrame,
    path: Path,
) -> None:
    latest = sentinel.dropna(subset=["risk_off_score"]).iloc[-1]
    top_cols = [
        "sentinel_rank",
        "symbol",
        "name",
        "group",
        "sentinel_adjusted_score_0_100",
        "sentinel_adjusted_prob_4w",
        "sentinel_action",
    ]
    lines = [
        "# Daily Risk-Off Sentinel Report",
        "",
        f"- 기준일: {latest.name.date().isoformat()}",
        f"- Sentinel 상태: {latest['sentinel_state']}",
        f"- Risk-off 점수: {float(latest['risk_off_score']):.2f} / 100",
        f"- 지배 경고요인: {latest['dominant_component']}",
        f"- 위험자산 예산: {float(latest['risk_budget_pct']):.0f}%",
        "",
        "## 현재 Sentinel 반영 상위 자산",
        overlay[[c for c in top_cols if c in overlay]].head(15).to_markdown(index=False),
        "",
    ]
    if not benchmark_validation.empty:
        lines.extend(["## KODEX 200 기준 일간 경보 검증", benchmark_validation.to_markdown(index=False), ""])
    if not threshold_sweep.empty:
        useful = threshold_sweep[
            ["threshold", "warning_days", "precision", "recall", "false_alarm_rate", "avg_forward_drawdown_when_warning"]
        ].sort_values("threshold")
        lines.extend(["## 경보 임계값 민감도", useful.to_markdown(index=False), ""])
    if not episode_validation.empty:
        lines.extend(["## 폭락 에피소드 선제 감지", episode_validation.to_markdown(index=False), ""])
    if not asset_validation.empty:
        group_summary = (
            asset_validation.groupby("group", as_index=False)
            .agg(
                symbols=("symbol", "count"),
                avg_precision=("precision", "mean"),
                avg_recall=("recall", "mean"),
                avg_false_alarm_rate=("false_alarm_rate", "mean"),
            )
            .sort_values("avg_recall", ascending=False)
        )
        lines.extend(["## 자산군별 경보 검증 요약", group_summary.to_markdown(index=False), ""])
    if not events.empty:
        lines.extend(["## 사건별 Sentinel 반응", events.to_markdown(index=False), ""])
    lines.extend(
        [
            "## 해석",
            "- 이 레이어는 뉴스 자체를 예측하지 않고, 매일 관측되는 변동성·신용·환율·주식 breadth·원자재 충격을 조합해 월간 RWKV/LPPL 점수를 즉시 보정한다.",
            "- Watch는 위험자산 사이즈 축소, De-risk는 위험자산 대폭 축소, Cash는 현금·단기채 중심 전환 신호다.",
            "- 완전한 선제형에 가깝게 쓰려면 이 Sentinel을 매일 먼저 실행하고, 월간 모델 신호보다 우선순위가 높게 적용해야 한다.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def round_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return round(float(value), 6)
    except Exception:
        return None


def date_str(value: pd.Timestamp | None) -> str | None:
    return None if value is None else value.date().isoformat()


if __name__ == "__main__":
    main()
