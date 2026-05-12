from __future__ import annotations

import numpy as np
import pandas as pd


def cross_sectional_zscore(df: pd.DataFrame, cols: list[str], winsorize: bool = True) -> pd.DataFrame:
    out = df.copy()
    grouped_dates = out["date"]
    for col in cols:
        if col not in out:
            continue
        values = pd.to_numeric(out[col], errors="coerce")
        if winsorize:
            group_size = values.groupby(grouped_dates).transform("count")
            if group_size.max() >= 20:
                quantiles = values.groupby(grouped_dates).quantile([0.01, 0.99]).unstack()
                lo = grouped_dates.map(quantiles[0.01])
                hi = grouped_dates.map(quantiles[0.99])
                values = values.mask(group_size >= 20, values.clip(lo, hi))
        mean = values.groupby(grouped_dates).transform("mean")
        std = values.groupby(grouped_dates).transform("std").replace(0, np.nan)
        out[f"z_{col}"] = ((values - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def add_rule_scores(features: pd.DataFrame) -> pd.DataFrame:
    needed = [
        "ETF_RS_20D",
        "ETF_RS_60D",
        "ETF_RS_120D",
        "weighted_HP",
        "HP_change_20D",
        "HP90_share",
        "weighted_component_RS_20D",
        "weighted_component_RS_60D",
        "median_component_RS_20D",
        "MA60_breadth",
        "MA200_breadth",
        "RS_positive_share",
        "Breadth_change_20D",
        "top5_weight_share",
        "top5_return_contribution_share",
    ]
    out = cross_sectional_zscore(features, needed)

    out["ETF_RS_Score"] = 0.4 * z(out, "ETF_RS_20D") + 0.4 * z(out, "ETF_RS_60D") + 0.2 * z(out, "ETF_RS_120D")
    out["HP_Score"] = 0.5 * z(out, "weighted_HP") + 0.3 * z(out, "HP_change_20D") + 0.2 * z(out, "HP90_share")
    out["Component_Momentum_Score"] = (
        0.4 * z(out, "weighted_component_RS_20D")
        + 0.4 * z(out, "weighted_component_RS_60D")
        + 0.2 * z(out, "median_component_RS_20D")
    )
    out["Breadth_Score"] = (
        0.35 * z(out, "MA60_breadth")
        + 0.25 * z(out, "MA200_breadth")
        + 0.25 * z(out, "RS_positive_share")
        + 0.15 * z(out, "Breadth_change_20D")
    )
    out["Concentration_Penalty"] = z(out, "top5_weight_share") + z(out, "top5_return_contribution_share")

    out["Concentrated_ETF_Score"] = (
        0.45 * out["ETF_RS_Score"]
        + 0.30 * out["Component_Momentum_Score"]
        + 0.20 * out["HP_Score"]
        + 0.05 * z(out, "RS_positive_share")
    )
    out["Diversified_ETF_Score"] = (
        0.30 * out["ETF_RS_Score"]
        + 0.25 * out["Component_Momentum_Score"]
        + 0.20 * out["HP_Score"]
        + 0.20 * out["Breadth_Score"]
        - 0.05 * out["Concentration_Penalty"]
    )

    reliability = pd.to_numeric(out["effective_N"], errors="coerce") / (pd.to_numeric(out["effective_N"], errors="coerce") + 20.0)
    reliability = reliability.clip(0, 1).fillna(0)
    out["Adjusted_Breadth_Score"] = reliability * out["Breadth_Score"] + (1 - reliability) * out["ETF_RS_Score"]
    out["Mid_ETF_Score"] = (
        0.35 * out["ETF_RS_Score"]
        + 0.25 * out["Component_Momentum_Score"]
        + 0.20 * out["HP_Score"]
        + 0.15 * out["Adjusted_Breadth_Score"]
        - 0.05 * out["Concentration_Penalty"]
    )

    out["Final_Rule_Score"] = np.select(
        [
            out["holding_logic"].eq("concentrated"),
            out["holding_logic"].eq("diversified"),
        ],
        [
            out["Concentrated_ETF_Score"],
            out["Diversified_ETF_Score"],
        ],
        default=out["Mid_ETF_Score"],
    )
    out["Final_Rule_Score_0_100"] = rule_score_to_0_100(out["Final_Rule_Score"])
    out["rule_rank"] = out.groupby("date")["Final_Rule_Score"].rank(ascending=False, method="first")
    return out


def z(frame: pd.DataFrame, name: str) -> pd.Series:
    return frame.get(f"z_{name}", pd.Series(0.0, index=frame.index)).fillna(0.0)


def rule_score_to_0_100(score: pd.Series) -> pd.Series:
    return (50 + 18 * np.tanh(score / 2.0)).clip(0, 100)
