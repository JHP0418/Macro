from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "portfolio_rebalance_validator_latest"
TABLES = OUT / "tables"

INDIVIDUAL_CAP = 0.20
RISK_TOTAL_CAP = 0.70
SAFE_TOTAL_CAP = 1.00

RISK_BASKET_CAPS = {
    "국내지수": 0.30,
    "국내섹터": 0.15,
    "해외지수": 0.30,
    "해외섹터": 0.10,
    "FX및 원자재": 0.20,
}

SAFE_BASKET_CAPS = {
    "국내채권_종합": 0.50,
    "국내채권_회사채": 0.30,
    "해외채권_종합": 0.50,
    "해외채권_회사채": 0.30,
    "금리연계형 및 초단기채권": 0.50,
}

CASH_SYMBOL = "CASH_KRW"
CASH_BASKET = "KRW 현금"

RISK_BASKETS = set(RISK_BASKET_CAPS)
SAFE_BASKETS = set(SAFE_BASKET_CAPS)


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, **kwargs)


def load_panel() -> pd.DataFrame:
    panel = read_csv(ROOT / "outputs" / "weekly_screening_rank_backtest_latest" / "tables" / "weekly_calibrated_rank_panel.csv", parse_dates=["date"])
    panel = panel[panel["basket"].isin(RISK_BASKETS | SAFE_BASKETS)].copy()
    safe_path = ROOT / "outputs" / "institutional_risk_off_v2_latest" / "tables" / "macro_conditioned_safe_asset_predictions.csv"
    if safe_path.exists():
        safe = read_csv(safe_path, parse_dates=["date"])
        safe = safe[safe.get("horizon", "").astype(str).eq("1m")][["date", "symbol", "safe_v2_ranker_score"]].copy()
        panel = panel.merge(safe, on=["date", "symbol"], how="left")
    else:
        panel["safe_v2_ranker_score"] = np.nan

    risk_v4_path = ROOT / "outputs" / "risk_off_v4_event_label_latest" / "tables" / "risk_off_v4_walkforward_predictions.csv"
    risk_v2_path = ROOT / "outputs" / "institutional_risk_off_v2_latest" / "tables" / "risk_off_v2_walkforward_predictions.csv"
    if risk_v4_path.exists():
        risk = read_csv(risk_v4_path, parse_dates=["date"])
        risk = risk[risk["horizon"].astype(str).eq("1m")][
            [
                "date",
                "risk_off_v4_prob",
                "risk_off_v4_watch",
                "risk_off_v4_alert",
                "risk_off_v4_cash",
                "risk_off_v4_stage",
            ]
        ].sort_values("date")
        panel = pd.merge_asof(panel.sort_values("date"), risk, on="date", direction="backward")
        panel["risk_model_version"] = "V4_event_label"
        panel["risk_off_prob"] = pd.to_numeric(panel["risk_off_v4_prob"], errors="coerce")
        panel["risk_off_alert"] = pd.to_numeric(panel["risk_off_v4_alert"], errors="coerce").fillna(0).astype(int)
        panel["risk_off_stage"] = panel["risk_off_v4_stage"].fillna("Normal")
    elif risk_v2_path.exists():
        risk = read_csv(risk_v2_path, parse_dates=["date"])
        risk = risk[risk["horizon"].astype(str).eq("1m")][["date", "risk_off_v2_prob", "risk_off_v2_alert"]].sort_values("date")
        panel = pd.merge_asof(panel.sort_values("date"), risk, on="date", direction="backward")
        panel["risk_model_version"] = "V2_legacy"
        panel["risk_off_prob"] = pd.to_numeric(panel["risk_off_v2_prob"], errors="coerce")
        panel["risk_off_alert"] = pd.to_numeric(panel["risk_off_v2_alert"], errors="coerce").fillna(0).astype(int)
        panel["risk_off_stage"] = np.where(panel["risk_off_alert"].eq(1), "De-risk", "Normal")
    else:
        panel["risk_model_version"] = "none"
        panel["risk_off_prob"] = np.nan
        panel["risk_off_alert"] = 0
        panel["risk_off_stage"] = "Normal"

    panel["risk_off_prob"] = pd.to_numeric(panel["risk_off_prob"], errors="coerce").fillna(0.0)
    panel["risk_off_alert"] = pd.to_numeric(panel["risk_off_alert"], errors="coerce").fillna(0).astype(int)
    panel["segment"] = np.where(panel["basket"].isin(RISK_BASKETS), "risk", "safe")
    panel["base_score"] = pd.to_numeric(panel["institutional_score_0_100"], errors="coerce").fillna(50.0)
    safe_score = pd.to_numeric(panel["safe_v2_ranker_score"], errors="coerce")
    panel["portfolio_score"] = np.where(panel["segment"].eq("safe") & safe_score.notna(), safe_score, panel["base_score"])
    # Cross-sectional percentile stabilizes heterogeneous score scales.
    panel["portfolio_score_pct"] = panel.groupby(["date", "segment"])["portfolio_score"].rank(pct=True, method="average").fillna(0.5)
    return panel.sort_values(["date", "segment", "portfolio_score_pct"], ascending=[True, True, False])


def risk_budget_from_state(prob: float, alert: int, stage: str) -> float:
    stage = str(stage)
    if stage == "Cash":
        return 0.15
    if stage == "De-risk":
        return 0.35
    if stage == "Watch":
        return 0.50
    if prob >= 0.65:
        return 0.15
    if prob >= 0.50:
        return 0.35
    if prob >= 0.38 or alert:
        return 0.50
    return RISK_TOTAL_CAP


def allocate_segment(frame: pd.DataFrame, target_weight: float, basket_caps: dict[str, float]) -> pd.DataFrame:
    if frame.empty or target_weight <= 0:
        return frame.assign(weight=0.0).iloc[0:0]
    rows = []
    remaining = float(target_weight)
    used_by_basket = {basket: 0.0 for basket in basket_caps}
    ranked = frame.sort_values(["portfolio_score_pct", "portfolio_score", "symbol"], ascending=[False, False, True])
    for _, row in ranked.iterrows():
        basket = str(row["basket"])
        basket_remaining = basket_caps.get(basket, 0.0) - used_by_basket.get(basket, 0.0)
        weight = min(INDIVIDUAL_CAP, basket_remaining, remaining)
        if weight <= 1e-9:
            continue
        item = row.to_dict()
        item["weight"] = weight
        rows.append(item)
        used_by_basket[basket] = used_by_basket.get(basket, 0.0) + weight
        remaining -= weight
        if remaining <= 1e-9:
            break
    return pd.DataFrame(rows)


def allocate_portfolio(date_frame: pd.DataFrame) -> pd.DataFrame:
    date = date_frame["date"].iloc[0]
    prob = float(date_frame["risk_off_prob"].max())
    alert = int(date_frame["risk_off_alert"].max())
    stage_order = {"Normal": 0, "Watch": 1, "De-risk": 2, "Cash": 3}
    stage = max(date_frame["risk_off_stage"].astype(str), key=lambda x: stage_order.get(x, 0))
    risk_target = min(RISK_TOTAL_CAP, risk_budget_from_state(prob, alert, stage))
    safe_target = min(SAFE_TOTAL_CAP, 1.0 - risk_target)
    risk_alloc = allocate_segment(date_frame[date_frame["segment"].eq("risk")], risk_target, RISK_BASKET_CAPS)
    safe_alloc = allocate_segment(date_frame[date_frame["segment"].eq("safe")], safe_target, SAFE_BASKET_CAPS)
    out = pd.concat([risk_alloc, safe_alloc], ignore_index=True)
    if out.empty:
        return out
    out["target_risk_weight"] = risk_target
    out["target_safe_weight"] = safe_target
    out["applied_risk_off_prob"] = prob
    out["applied_risk_off_stage"] = stage
    out["date"] = date
    total = out["weight"].sum()
    if total < 0.999 and not out[out["basket"].eq("금리연계형 및 초단기채권")].empty:
        # Try to park residual cash in short-rate products without breaking caps.
        residual = 1.0 - total
        cash_idx = out[out["basket"].eq("금리연계형 및 초단기채권")].sort_values("portfolio_score_pct", ascending=False).index
        for idx in cash_idx:
            add = min(residual, INDIVIDUAL_CAP - out.at[idx, "weight"], SAFE_BASKET_CAPS["금리연계형 및 초단기채권"] - out.loc[out["basket"].eq("금리연계형 및 초단기채권"), "weight"].sum())
            if add > 1e-9:
                out.at[idx, "weight"] += add
                residual -= add
            if residual <= 1e-9:
                break
    total = out["weight"].sum()
    if total < 0.999999:
        # If the investable universe cannot fill 100% without breaking ETF caps,
        # keep the remainder as explicit uninvested KRW cash. Cash is not an ETF
        # product, so the individual ETF cap is not applied to this synthetic row.
        residual = 1.0 - total
        template = out.iloc[0].to_dict()
        template.update(
            {
                "symbol": CASH_SYMBOL,
                "name": "미배분 KRW 현금",
                "basket": CASH_BASKET,
                "segment": "cash",
                "weight": residual,
                "portfolio_score": 0.0,
                "portfolio_score_pct": 0.0,
                "realized_return_1w": 0.0,
                "realized_return_4w": 0.0,
            }
        )
        out = pd.concat([out, pd.DataFrame([template])], ignore_index=True)
    return out


def build_allocations(panel: pd.DataFrame) -> pd.DataFrame:
    allocations = []
    for _, part in panel.groupby("date", sort=True):
        alloc = allocate_portfolio(part)
        if not alloc.empty:
            allocations.append(alloc)
    return pd.concat(allocations, ignore_index=True) if allocations else pd.DataFrame()


def validate_constraints(alloc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date, part in alloc.groupby("date"):
        total = part["weight"].sum()
        risk_total = part.loc[part["basket"].isin(RISK_BASKETS), "weight"].sum()
        safe_total = part.loc[part["basket"].isin(SAFE_BASKETS), "weight"].sum()
        product_part = part[~part["symbol"].eq(CASH_SYMBOL)]
        cash_weight = part.loc[part["symbol"].eq(CASH_SYMBOL), "weight"].sum()
        max_single = product_part["weight"].max() if not product_part.empty else 0.0
        rows.append(
            {
                "date": date,
                "total_weight": total,
                "risk_total": risk_total,
                "safe_total": safe_total,
                "cash_weight": cash_weight,
                "max_single": max_single,
                "total_weight_violation": abs(total - 1.0) > 1e-8,
                "single_cap_violation": max_single > INDIVIDUAL_CAP + 1e-8,
                "risk_total_violation": risk_total > RISK_TOTAL_CAP + 1e-8,
                "safe_total_violation": safe_total > SAFE_TOTAL_CAP + 1e-8,
            }
        )
        by_basket = part.groupby("basket")["weight"].sum()
        for basket, cap in {**RISK_BASKET_CAPS, **SAFE_BASKET_CAPS}.items():
            rows[-1][f"{basket}_weight"] = float(by_basket.get(basket, 0.0))
            rows[-1][f"{basket}_violation"] = by_basket.get(basket, 0.0) > cap + 1e-8
    return pd.DataFrame(rows)


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def performance(alloc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ret_col = "realized_return_1w"
    alloc["weighted_return_1w"] = alloc["weight"] * pd.to_numeric(alloc[ret_col], errors="coerce").fillna(0.0)
    port = (
        alloc.groupby("date", as_index=False)
        .agg(
            portfolio_return_1w=("weighted_return_1w", "sum"),
            names=("name", lambda x: " | ".join(x.astype(str).head(8))),
            symbols=("symbol", lambda x: ",".join(x.astype(str).head(8))),
            n_positions=("symbol", "count"),
            risk_weight=("weight", lambda s: alloc.loc[s.index][alloc.loc[s.index, "basket"].isin(RISK_BASKETS)]["weight"].sum()),
            safe_weight=("weight", lambda s: alloc.loc[s.index][alloc.loc[s.index, "basket"].isin(SAFE_BASKETS)]["weight"].sum()),
            cash_weight=("weight", lambda s: alloc.loc[s.index][alloc.loc[s.index, "symbol"].eq(CASH_SYMBOL)]["weight"].sum()),
            risk_off_prob=("risk_off_prob", "max"),
        )
        .sort_values("date")
    )
    port["equity"] = (1.0 + port["portfolio_return_1w"]).cumprod()
    weeks = max(len(port), 1)
    cagr = float(port["equity"].iloc[-1] ** (52.0 / weeks) - 1.0) if not port.empty else np.nan
    vol = float(port["portfolio_return_1w"].std() * np.sqrt(52)) if weeks > 2 else np.nan
    sharpe = float((port["portfolio_return_1w"].mean() * 52) / vol) if vol and vol > 0 else np.nan
    summary = pd.DataFrame(
        [
            {
                "periods": weeks,
                "start": port["date"].min(),
                "end": port["date"].max(),
                "cumulative_return": float(port["equity"].iloc[-1] - 1.0) if not port.empty else np.nan,
                "CAGR": cagr,
                "MDD": max_drawdown(port["equity"]) if not port.empty else np.nan,
                "Sharpe": sharpe,
                "hit_positive": float((port["portfolio_return_1w"] > 0).mean()) if not port.empty else np.nan,
                "avg_weekly_return": float(port["portfolio_return_1w"].mean()) if not port.empty else np.nan,
                "avg_risk_weight": float(port["risk_weight"].mean()) if not port.empty else np.nan,
                "avg_safe_weight": float(port["safe_weight"].mean()) if not port.empty else np.nan,
                "avg_cash_weight": float(port["cash_weight"].mean()) if not port.empty else np.nan,
            }
        ]
    )
    return port, summary


def write_rule_table() -> pd.DataFrame:
    rows = [{"scope": "개별 상품", "cap": INDIVIDUAL_CAP}]
    rows.append({"scope": "위험자산 전체", "cap": RISK_TOTAL_CAP})
    for k, v in RISK_BASKET_CAPS.items():
        rows.append({"scope": k, "cap": v})
    rows.append({"scope": "안전자산 전체", "cap": SAFE_TOTAL_CAP})
    for k, v in SAFE_BASKET_CAPS.items():
        rows.append({"scope": k, "cap": v})
    return pd.DataFrame(rows)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    alloc = build_allocations(panel)
    constraint_check = validate_constraints(alloc)
    port, summary = performance(alloc)
    latest_alloc = alloc[alloc["date"].eq(alloc["date"].max())].sort_values("weight", ascending=False)
    rule_table = write_rule_table()

    panel.to_csv(TABLES / "portfolio_candidate_panel.csv", index=False, encoding="utf-8-sig")
    alloc.to_csv(TABLES / "weekly_constrained_allocations.csv", index=False, encoding="utf-8-sig")
    constraint_check.to_csv(TABLES / "weekly_constraint_validation.csv", index=False, encoding="utf-8-sig")
    port.to_csv(TABLES / "weekly_constrained_portfolio_returns.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(TABLES / "weekly_constrained_portfolio_summary.csv", index=False, encoding="utf-8-sig")
    latest_alloc.to_csv(TABLES / "latest_constrained_portfolio.csv", index=False, encoding="utf-8-sig")
    rule_table.to_csv(TABLES / "portfolio_constraint_rules.csv", index=False, encoding="utf-8-sig")

    violations = constraint_check[[c for c in constraint_check.columns if c.endswith("_violation")]].any(axis=1).sum()
    print("Constraint rules")
    print(rule_table.to_string(index=False))
    print("\nPortfolio summary")
    print(summary.to_string(index=False))
    print(f"\nViolation dates: {violations}")
    print("\nLatest constrained portfolio")
    cols = ["date", "symbol", "name", "basket", "weight", "portfolio_score_pct", "applied_risk_off_prob", "applied_risk_off_stage", "target_risk_weight", "target_safe_weight"]
    print(latest_alloc[[c for c in cols if c in latest_alloc.columns]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
