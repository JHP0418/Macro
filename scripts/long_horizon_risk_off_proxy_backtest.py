from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "long_horizon_risk_off_proxy_backtest_latest"
TABLES = OUT / "tables"


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, **kwargs)


def pct_change(frame: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).pct_change().fillna(0.0)


def build_return_panel() -> pd.DataFrame:
    prices = read_csv(ROOT / "outputs" / "macro_regime_asset_screener_latest" / "tables" / "driver_panel.csv", parse_dates=["Date"])
    prices = prices.rename(columns={"Date": "date"}).sort_values("date")
    for col in prices.columns:
        if col != "date":
            prices[col] = pd.to_numeric(prices[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    out = prices[["date"]].copy()
    risk_cols = [c for c in ["NASDAQ100", "SP500", "SOX", "RUSSELL2000"] if c in prices.columns]
    weights = {"NASDAQ100": 0.40, "SP500": 0.30, "SOX": 0.20, "RUSSELL2000": 0.10}
    risk_return = pd.Series(0.0, index=prices.index)
    weight_sum = 0.0
    for col in risk_cols:
        w = weights[col]
        risk_return += w * pct_change(prices, col)
        weight_sum += w
    out["risk_asset_return"] = risk_return / max(weight_sum, 1e-9)
    out["nasdaq_return"] = pct_change(prices, "NASDAQ100") if "NASDAQ100" in prices else np.nan
    out["sp500_return"] = pct_change(prices, "SP500") if "SP500" in prices else np.nan
    out["sox_return"] = pct_change(prices, "SOX") if "SOX" in prices else np.nan

    us2y = pd.to_numeric(prices.get("US2Y", pd.Series(index=prices.index)), errors="coerce").ffill()
    us10y = pd.to_numeric(prices.get("US10Y", pd.Series(index=prices.index)), errors="coerce").ffill()
    usdkrw_ret = pct_change(prices, "USDKRW") if "USDKRW" in prices else pd.Series(0.0, index=prices.index)
    dxy_ret = pct_change(prices, "DXY") if "DXY" in prices else pd.Series(0.0, index=prices.index)
    gold_ret = pct_change(prices, "GOLD") if "GOLD" in prices else pd.Series(0.0, index=prices.index)

    # Proxy returns for long-horizon testing. They are not ETF total-return
    # series, but they allow regime-defense validation before ETF listing dates.
    out["cash_return"] = (us2y.fillna(0.0) / 100.0 / 252.0).clip(-0.001, 0.001).fillna(0.0)
    out["us_long_bond_proxy_return"] = (-12.0 * us10y.diff().fillna(0.0) / 100.0 + us10y.fillna(0.0) / 100.0 / 252.0).clip(-0.08, 0.08)
    out["usd_cash_proxy_return"] = (0.70 * usdkrw_ret + 0.30 * dxy_ret + out["cash_return"]).clip(-0.08, 0.08)
    out["gold_krw_proxy_return"] = (gold_ret + 0.50 * usdkrw_ret).clip(-0.10, 0.10)
    out["safe_equal_return"] = (
        0.35 * out["cash_return"]
        + 0.30 * out["us_long_bond_proxy_return"]
        + 0.20 * out["gold_krw_proxy_return"]
        + 0.15 * out["usd_cash_proxy_return"]
    )
    keep = ["date", "VIX", "US10Y", "US2Y", "DXY", "USDKRW", "GOLD"]
    for col in keep[1:]:
        if col in prices:
            out[col] = prices[col]
    return out.replace([np.inf, -np.inf], np.nan)


def load_v4_predictions() -> pd.DataFrame:
    pred = read_csv(ROOT / "outputs" / "risk_off_v4_event_label_latest" / "tables" / "risk_off_v4_walkforward_predictions.csv", parse_dates=["date"])
    wide_parts = []
    for horizon, part in pred.groupby("horizon"):
        keep = [
            "date",
            "risk_off_v4_prob",
            "risk_off_v4_stage",
            "risk_off_v4_watch",
            "risk_off_v4_alert",
            "risk_off_v4_cash",
            "axis1_vol_credit_stress",
            "axis2_fx_liquidity_stress",
            "axis3_peak_fragility_stress",
            "risk_3d_dominant_axis",
        ]
        x = part[[c for c in keep if c in part.columns]].copy()
        x = x.rename(columns={c: f"{c}_{horizon}" for c in x.columns if c != "date"})
        wide_parts.append(x)
    out = wide_parts[0]
    for part in wide_parts[1:]:
        out = out.merge(part, on="date", how="outer")
    return out.sort_values("date")


def load_v3_predictions() -> pd.DataFrame:
    pred = read_csv(ROOT / "outputs" / "institutional_risk_off_v2_latest" / "tables" / "risk_off_v2_walkforward_predictions.csv", parse_dates=["date"])
    pred = pred[pred["horizon"].astype(str).eq("1m")].copy()
    return pred[["date", "risk_off_v2_prob", "risk_off_v2_alert"]].sort_values("date")


def stage_level(stage: object) -> int:
    return {"Normal": 0, "Watch": 1, "De-risk": 2, "Cash": 3}.get(str(stage), 0)


def risk_budget_from_stage(stage: object) -> float:
    return {"Cash": 0.15, "De-risk": 0.35, "Watch": 0.50, "Normal": 0.70}.get(str(stage), 0.70)


def risk_budget_from_v3(prob: float, alert: int) -> float:
    if prob >= 0.65:
        return 0.15
    if prob >= 0.50:
        return 0.35
    if prob >= 0.38 or alert:
        return 0.50
    return 0.70


def risk_budget_from_vix(vix: float) -> float:
    if pd.isna(vix):
        return 0.70
    if vix >= 35:
        return 0.15
    if vix >= 25:
        return 0.35
    if vix >= 20:
        return 0.50
    return 0.70


def choose_combined_stage(row: pd.Series) -> str:
    s1w = str(row.get("risk_off_v4_stage_1w", "Normal"))
    s1m = str(row.get("risk_off_v4_stage_1m", "Normal"))
    return s1w if stage_level(s1w) >= stage_level(s1m) else s1m


def dynamic_safe_return(row: pd.Series) -> float:
    axis1 = float(row.get("axis1_vol_credit_stress_1m", row.get("axis1_vol_credit_stress_1w", 0)) or 0)
    axis2 = float(row.get("axis2_fx_liquidity_stress_1m", row.get("axis2_fx_liquidity_stress_1w", 0)) or 0)
    axis3 = float(row.get("axis3_peak_fragility_stress_1m", row.get("axis3_peak_fragility_stress_1w", 0)) or 0)
    cash = float(row.get("cash_return", 0) or 0)
    bond = float(row.get("us_long_bond_proxy_return", 0) or 0)
    gold = float(row.get("gold_krw_proxy_return", 0) or 0)
    usd = float(row.get("usd_cash_proxy_return", 0) or 0)
    if axis2 >= max(axis1, axis3):
        return 0.20 * cash + 0.15 * bond + 0.25 * gold + 0.40 * usd
    if axis1 >= max(axis2, axis3):
        return 0.20 * cash + 0.45 * bond + 0.20 * gold + 0.15 * usd
    return 0.25 * cash + 0.35 * bond + 0.30 * gold + 0.10 * usd


def add_strategy_returns(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["v4_stage_1m"] = out["risk_off_v4_stage_1m"].fillna("Normal")
    out["v4_stage_combined"] = out.apply(choose_combined_stage, axis=1)
    out["v4_1m_risk_weight"] = out["v4_stage_1m"].map(risk_budget_from_stage).astype(float)
    out["v4_combined_risk_weight"] = out["v4_stage_combined"].map(risk_budget_from_stage).astype(float)
    # Adaptive improvement: V4 intentionally catches peak-fragility early, but
    # long tests show that "peak fragility only" Cash signals are often costly
    # false alarms when volatility/credit and FX/liquidity stress are still low.
    # Keep hard Cash only for systemic stress; relax isolated peak-fragility to
    # the normal 70% risk cap and let ETF leadership handle selection.
    axis1 = out.get("axis1_vol_credit_stress_1m", out.get("axis1_vol_credit_stress_1w", pd.Series(0.0, index=out.index))).fillna(0.0)
    axis2 = out.get("axis2_fx_liquidity_stress_1m", out.get("axis2_fx_liquidity_stress_1w", pd.Series(0.0, index=out.index))).fillna(0.0)
    axis3 = out.get("axis3_peak_fragility_stress_1m", out.get("axis3_peak_fragility_stress_1w", pd.Series(0.0, index=out.index))).fillna(0.0)
    vix = out.get("VIX", pd.Series(np.nan, index=out.index))
    out["v4_adaptive_risk_weight"] = out["v4_stage_combined"].map({"Normal": 0.70, "Watch": 0.60, "De-risk": 0.45, "Cash": 0.15}).astype(float)
    peak_only_cash = (
        out["v4_stage_combined"].eq("Cash")
        & axis1.lt(30)
        & axis2.lt(35)
        & axis3.ge(axis1)
        & axis3.ge(axis2)
        & vix.lt(28)
    )
    out.loc[peak_only_cash, "v4_adaptive_risk_weight"] = 0.70
    out["v4_adaptive_peak_only_relaxed"] = peak_only_cash.astype(int)
    out["v3_risk_weight"] = [
        risk_budget_from_v3(float(p) if pd.notna(p) else 0.0, int(a) if pd.notna(a) else 0)
        for p, a in zip(out.get("risk_off_v2_prob", 0), out.get("risk_off_v2_alert", 0))
    ]
    out["vix_risk_weight"] = out["VIX"].map(risk_budget_from_vix).astype(float)
    out["safe_dynamic_return"] = out.apply(dynamic_safe_return, axis=1)
    out["buyhold_risk_return"] = out["risk_asset_return"]
    out["balanced_60_40_return"] = 0.60 * out["risk_asset_return"] + 0.40 * out["safe_equal_return"]
    out["vix_gate_return"] = out["vix_risk_weight"] * out["risk_asset_return"] + (1 - out["vix_risk_weight"]) * out["safe_dynamic_return"]
    out["v3_gate_return"] = out["v3_risk_weight"] * out["risk_asset_return"] + (1 - out["v3_risk_weight"]) * out["safe_dynamic_return"]
    out["v4_1m_gate_return"] = out["v4_1m_risk_weight"] * out["risk_asset_return"] + (1 - out["v4_1m_risk_weight"]) * out["safe_dynamic_return"]
    out["v4_combined_gate_return"] = out["v4_combined_risk_weight"] * out["risk_asset_return"] + (1 - out["v4_combined_risk_weight"]) * out["safe_dynamic_return"]
    out["v4_adaptive_gate_return"] = out["v4_adaptive_risk_weight"] * out["risk_asset_return"] + (1 - out["v4_adaptive_risk_weight"]) * out["safe_dynamic_return"]
    return out


def max_drawdown(equity: pd.Series) -> float:
    dd = equity / equity.cummax() - 1.0
    return float(dd.min())


def metrics(frame: pd.DataFrame, return_col: str) -> dict[str, float | str | int]:
    r = pd.to_numeric(frame[return_col], errors="coerce").fillna(0.0)
    equity = (1.0 + r).cumprod()
    years = len(r) / 252.0
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 and not equity.empty else np.nan
    vol = float(r.std() * np.sqrt(252))
    sharpe = float(r.mean() * 252 / vol) if vol > 0 else np.nan
    downside = r[r < 0].std() * np.sqrt(252)
    sortino = float(r.mean() * 252 / downside) if downside and downside > 0 else np.nan
    mdd = max_drawdown(equity)
    return {
        "strategy": return_col.replace("_return", ""),
        "start": frame["date"].min().date().isoformat(),
        "end": frame["date"].max().date().isoformat(),
        "days": int(len(frame)),
        "total_return": float(equity.iloc[-1] - 1),
        "CAGR": cagr,
        "ann_vol": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MDD": mdd,
        "Calmar": float(cagr / abs(mdd)) if mdd < 0 else np.nan,
        "positive_day_rate": float((r > 0).mean()),
        "avg_daily_return": float(r.mean()),
    }


def annual_returns(frame: pd.DataFrame, return_cols: list[str]) -> pd.DataFrame:
    rows = []
    x = frame.copy()
    x["year"] = x["date"].dt.year
    for year, part in x.groupby("year"):
        row = {"year": year}
        for col in return_cols:
            row[col.replace("_return", "")] = float((1.0 + pd.to_numeric(part[col], errors="coerce").fillna(0.0)).prod() - 1.0)
        rows.append(row)
    return pd.DataFrame(rows)


def crisis_windows(frame: pd.DataFrame, return_cols: list[str]) -> pd.DataFrame:
    windows = {
        "GFC_2008": ("2007-10-01", "2009-03-31"),
        "Euro_2011": ("2011-07-01", "2011-12-31"),
        "China_FX_2015": ("2015-06-01", "2016-02-29"),
        "Volmageddon_Q4_2018": ("2018-01-26", "2018-12-31"),
        "Covid_2020": ("2020-02-15", "2020-04-30"),
        "Inflation_2022": ("2022-01-01", "2022-12-31"),
        "AI_high_2024_2026": ("2024-01-01", "2026-05-08"),
    }
    rows = []
    for name, (start, end) in windows.items():
        part = frame[(frame["date"].ge(start)) & (frame["date"].le(end))]
        if part.empty:
            continue
        for col in return_cols:
            r = pd.to_numeric(part[col], errors="coerce").fillna(0.0)
            eq = (1.0 + r).cumprod()
            rows.append(
                {
                    "window": name,
                    "strategy": col.replace("_return", ""),
                    "start": start,
                    "end": end,
                    "total_return": float(eq.iloc[-1] - 1.0),
                    "MDD": max_drawdown(eq),
                    "vol": float(r.std() * np.sqrt(252)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    returns = build_return_panel()
    v4 = load_v4_predictions()
    v3 = load_v3_predictions()
    panel = returns.merge(v4, on="date", how="inner").merge(v3, on="date", how="left")
    panel = panel[panel["date"].ge("2003-01-01")].copy()
    panel = add_strategy_returns(panel)
    return_cols = [
        "buyhold_risk_return",
        "balanced_60_40_return",
        "vix_gate_return",
        "v3_gate_return",
        "v4_1m_gate_return",
        "v4_combined_gate_return",
        "v4_adaptive_gate_return",
    ]
    summary = pd.DataFrame([metrics(panel, col) for col in return_cols]).sort_values("Sharpe", ascending=False)
    annual = annual_returns(panel, return_cols)
    crisis = crisis_windows(panel, return_cols)
    exposure = panel.groupby("v4_stage_combined", as_index=False).agg(
        days=("date", "count"),
        avg_risk_return_next_day=("risk_asset_return", "mean"),
        avg_safe_return_next_day=("safe_dynamic_return", "mean"),
        avg_v4_combined_return=("v4_combined_gate_return", "mean"),
        avg_risk_weight=("v4_combined_risk_weight", "mean"),
        avg_adaptive_risk_weight=("v4_adaptive_risk_weight", "mean"),
        peak_only_relaxed_days=("v4_adaptive_peak_only_relaxed", "sum"),
    )
    panel.to_csv(TABLES / "long_horizon_proxy_daily_panel.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(TABLES / "long_horizon_proxy_strategy_summary.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(TABLES / "long_horizon_proxy_annual_returns.csv", index=False, encoding="utf-8-sig")
    crisis.to_csv(TABLES / "long_horizon_proxy_crisis_windows.csv", index=False, encoding="utf-8-sig")
    exposure.to_csv(TABLES / "long_horizon_proxy_v4_stage_exposure.csv", index=False, encoding="utf-8-sig")
    print("Long-horizon proxy summary")
    print(summary.to_string(index=False))
    print("\nCrisis windows")
    print(crisis.pivot(index="window", columns="strategy", values="MDD").to_string())
    print("\nV4 stage exposure")
    print(exposure.to_string(index=False))


if __name__ == "__main__":
    main()
