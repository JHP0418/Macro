from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

FEATURES_1W = [
    "ETF_RS_20D",
    "RS_slope_20D",
    "weighted_component_RS_20D",
    "median_component_RS_20D",
    "RS_positive_share",
    "MA60_breadth",
    "Breadth_change_20D",
    "median_component_return_20D",
    "mean_minus_median_return_20D",
    "top20_component_return_mean",
    "bottom20_component_return_mean",
    "top5_return_contribution_share",
    "HP_change_20D",
]

FEATURES_1M = [
    "ETF_RS_20D",
    "ETF_RS_60D",
    "ETF_RS_120D",
    "RS_slope_20D",
    "weighted_HP",
    "median_HP",
    "HP90_share",
    "HP_change_20D",
    "weighted_component_RS_20D",
    "weighted_component_RS_60D",
    "median_component_RS_20D",
    "RS_positive_share",
    "MA60_breadth",
    "MA200_breadth",
    "Breadth_change_20D",
    "median_component_return_20D",
    "median_component_return_60D",
    "mean_minus_median_return_20D",
    "top20_component_return_mean",
    "bottom20_component_return_mean",
    "top5_return_contribution_share",
]

STRUCTURE_COLUMNS = [
    "holding_count",
    "effective_N",
    "top5_weight_share",
    "top10_weight_share",
]

ENTRY_CONTEXT_1W = [
    "rule_5d_score",
    "rule_5d_group_z",
    "rule_5d_group_rank_pct",
    "rule_5d_gap_to_group_top",
    "rule_5d_group_score_std",
    "rule_5d_group_size",
]

ENTRY_CONTEXT_1M = [
    "ranker_score",
    "blend_20d_score",
    "rule_20d_score",
    "rule_20d_group_z",
    "rule_20d_group_rank_pct",
    "rule_20d_gap_to_group_top",
    "rule_20d_group_score_std",
    "rule_20d_group_size",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ETF leadership V3 with static holdings, basket ranking, and entry meta-model.")
    p.add_argument("--input", default=str(ROOT / "outputs" / "etf_leadership_static_holdings_approx_v3base" / "rule_scores.csv"))
    p.add_argument("--universe", default=str(ROOT / "data" / "etf_universe_leadership.csv"))
    p.add_argument("--output-dir", default=str(ROOT / "outputs" / "etf_leadership_static_v3"))
    p.add_argument("--min-date", default="2021-01-01")
    p.add_argument("--train-end", default="2022-12-31")
    p.add_argument("--valid-end", default="2024-12-31")
    p.add_argument("--min-holdings", type=int, default=2)
    p.add_argument("--min-group-size", type=int, default=2)
    p.add_argument("--top-k-list", default="1,2,3,5")
    return p.parse_args()


def broad_basket(group: str) -> str:
    group = str(group or "")
    if group in {"US broad equity", "US growth", "Global/Developed equity", "China/HK growth", "China equity", "India/EM", "Japan equity"}:
        return "해외지수"
    if group in {"US semiconductor", "US dividend/defensive", "US cyclical/sector", "US REIT"}:
        return "해외섹터"
    if group in {"Korea broad equity", "Korea growth"}:
        return "국내지수"
    if group in {"Korea semiconductor", "Korea IT", "Korea cyclical", "Korea value", "Korea defensive"}:
        return "국내섹터"
    if group in {"Gold", "Commodity/Oil", "Oil", "FX cash", "USD cash"}:
        return "FX및 원자재"
    if group == "Korea bonds":
        return "국내채권_종합"
    if group == "US IG bonds":
        return "해외채권_회사채"
    if group in {"US long bonds"}:
        return "해외채권_종합"
    if group == "US high yield":
        return "해외채권_회사채"
    if group == "Cash/short bonds":
        return "금리연계형 및 초단기채권"
    return "기타"


def attach_universe(frame: pd.DataFrame, universe_path: Path) -> pd.DataFrame:
    universe = pd.read_csv(universe_path)
    cols = [c for c in ["etf_ticker", "name", "group"] if c in universe.columns]
    out = frame.merge(universe[cols].drop_duplicates("etf_ticker"), on="etf_ticker", how="left")
    out["group"] = out["group"].fillna(out.get("market", "기타")).astype(str)
    out["ranking_group"] = out["group"]
    out["asset_basket"] = out["group"].map(broad_basket)
    out["model_group"] = out["asset_basket"]
    return out


def filter_quality(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"])
    numeric = sorted(set(FEATURES_1W + FEATURES_1M + STRUCTURE_COLUMNS + [
        "forward_5D_excess",
        "forward_20D_excess",
        "forward_5D_return",
        "forward_20D_return",
        "benchmark_forward_5D_return",
        "benchmark_forward_20D_return",
    ]))
    for col in numeric:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    mask = (
        out["date"].ge(pd.Timestamp(args.min_date))
        & out["holding_count"].ge(args.min_holdings)
        & out["forward_5D_excess"].notna()
        & out["forward_20D_excess"].notna()
    )
    out = out.loc[mask].sort_values(["date", "ranking_group", "etf_ticker"]).reset_index(drop=True)
    group_size = out.groupby(["date", "model_group"])["etf_ticker"].transform("size")
    return out[group_size.ge(args.min_group_size)].reset_index(drop=True)


def grouped_zscore(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = frame.copy()
    keys = [out["date"], out["model_group"]]
    fallback_key = out["date"]
    for col in cols:
        if col not in out.columns:
            continue
        x = pd.to_numeric(out[col], errors="coerce")
        size = x.groupby(keys).transform("count")
        mean = x.groupby(keys).transform("mean")
        std = x.groupby(keys).transform("std").replace(0, np.nan)
        z = (x - mean) / std
        fb_mean = x.groupby(fallback_key).transform("mean")
        fb_std = x.groupby(fallback_key).transform("std").replace(0, np.nan)
        z_fb = (x - fb_mean) / fb_std
        out[f"v3_z_{col}"] = z.where(size.ge(3), z_fb).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def z(frame: pd.DataFrame, col: str) -> pd.Series:
    return frame.get(f"v3_z_{col}", pd.Series(0.0, index=frame.index)).fillna(0.0)


def add_rule_scores(frame: pd.DataFrame) -> pd.DataFrame:
    out = grouped_zscore(frame, sorted(set(FEATURES_1W + FEATURES_1M + ["top5_return_contribution_share"])))
    out["rule_5d_score"] = (
        0.28 * z(out, "ETF_RS_20D")
        + 0.16 * z(out, "RS_slope_20D")
        + 0.22 * z(out, "weighted_component_RS_20D")
        + 0.12 * z(out, "median_component_RS_20D")
        + 0.12 * z(out, "RS_positive_share")
        + 0.06 * z(out, "Breadth_change_20D")
        + 0.06 * z(out, "median_component_return_20D")
        - 0.04 * z(out, "top5_return_contribution_share")
    )
    out["rule_20d_score"] = (
        0.14 * z(out, "ETF_RS_20D")
        + 0.18 * z(out, "ETF_RS_60D")
        + 0.12 * z(out, "ETF_RS_120D")
        + 0.08 * z(out, "RS_slope_20D")
        + 0.10 * z(out, "weighted_component_RS_60D")
        + 0.08 * z(out, "weighted_component_RS_20D")
        + 0.08 * z(out, "RS_positive_share")
        + 0.08 * z(out, "MA60_breadth")
        + 0.06 * z(out, "MA200_breadth")
        + 0.06 * z(out, "HP90_share")
        + 0.04 * z(out, "HP_change_20D")
        - 0.04 * z(out, "top5_return_contribution_share")
    )
    return add_score_context(add_score_context(out, "rule_5d_score", "rule_5d"), "rule_20d_score", "rule_20d")


def add_score_context(frame: pd.DataFrame, score_col: str, prefix: str) -> pd.DataFrame:
    out = frame.copy()
    keys = ["date", "model_group"]
    score = pd.to_numeric(out[score_col], errors="coerce")
    group = out.groupby(keys)
    rank = group[score_col].rank(ascending=False, method="first")
    size = group[score_col].transform("size").astype(float)
    top = group[score_col].transform("max")
    mean = group[score_col].transform("mean")
    std = group[score_col].transform("std").replace(0, np.nan)
    out[f"{prefix}_group_rank"] = rank
    out[f"{prefix}_group_size"] = size
    out[f"{prefix}_group_rank_pct"] = (1.0 - (rank - 1.0) / (size - 1.0).replace(0, np.nan)).fillna(1.0)
    out[f"{prefix}_group_z"] = ((score - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out[f"{prefix}_gap_to_group_top"] = (top - score).fillna(0.0)
    out[f"{prefix}_group_score_std"] = std.fillna(0.0)
    return out


def add_group_labels(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for horizon, target in [("5D", "forward_5D_excess"), ("20D", "forward_20D_excess")]:
        pct = out.groupby(["date", "model_group"])[target].rank(pct=True, method="average")
        out[f"label_{horizon}_group_rank_pct"] = pct
        out[f"label_{horizon}_group_rank_int"] = np.ceil(pct * 5.0).sub(1).clip(0, 4)
        out[f"label_{horizon}_group_rank_int"] = out[f"label_{horizon}_group_rank_int"].where(pct.notna())
    out["entry_5d_label"] = ((out["forward_5D_excess"] > 0) & (out["forward_5D_return"] > 0)).astype(int)
    out["entry_20d_label"] = ((out["forward_20D_excess"] > 0) & (out["forward_20D_return"] > 0)).astype(int)
    return out


def split_frame(frame: pd.DataFrame, train_end: str, valid_end: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = frame[frame["date"].le(pd.Timestamp(train_end))]
    valid = frame[frame["date"].gt(pd.Timestamp(train_end)) & frame["date"].le(pd.Timestamp(valid_end))]
    test = frame[frame["date"].gt(pd.Timestamp(valid_end))]
    return train, valid, test


def rank_features(frame: pd.DataFrame, label_col: str, features: list[str]) -> tuple[pd.DataFrame, pd.Series, list[int], pd.DataFrame]:
    data = frame.dropna(subset=[label_col]).sort_values(["date", "model_group", "etf_ticker"]).reset_index(drop=True)
    group_size = data.groupby(["date", "model_group"])["etf_ticker"].transform("size")
    data = data[group_size.ge(2)].reset_index(drop=True)
    x = data[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = x.groupby([data["date"], data["model_group"]]).transform(lambda s: s.fillna(s.median())).fillna(0.0)
    y = data[label_col].astype(int)
    group = data.groupby(["date", "model_group"], sort=False).size().astype(int).tolist()
    return x, y, group, data


def train_ranker(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, label_col: str, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    from lightgbm import LGBMRanker, early_stopping, log_evaluation

    x_train, y_train, group_train, _ = rank_features(train, label_col, features)
    x_valid, y_valid, group_valid, _ = rank_features(valid, label_col, features)
    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=800,
        learning_rate=0.02,
        num_leaves=9,
        max_depth=3,
        min_child_samples=12,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=2.0,
        reg_lambda=12.0,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        x_train,
        y_train,
        group=group_train,
        eval_set=[(x_valid, y_valid)],
        eval_group=[group_valid],
        eval_at=[1, 3, 5],
        callbacks=[early_stopping(100), log_evaluation(100)],
    )
    preds = []
    for split_name, part in [("train", train), ("valid", valid), ("test", test)]:
        x, _, _, data = rank_features(part, label_col, features)
        data = data.copy()
        data["split"] = split_name
        data["ranker_score"] = model.predict(x)
        data = add_score_context(data, "ranker_score", "ranker")
        preds.append(data)
    importance = pd.DataFrame(
        {
            "feature": features,
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
            "importance_split": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False)
    return pd.concat(preds, ignore_index=True), importance


def train_entry_model(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    features: list[str],
    out_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    cols = [c for c in features if c in train.columns]
    frames = []
    if train[target_col].nunique(dropna=True) < 2:
        base = float(train[target_col].mean()) if len(train) else 0.5
        for split_name, part in [("train", train), ("valid", valid), ("test", test)]:
            data = part.copy()
            data["split"] = split_name
            data[out_col] = base
            frames.append(data)
        return pd.concat(frames, ignore_index=True), pd.DataFrame({"feature": cols, "importance": 0.0})

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.035,
            max_leaf_nodes=9,
            min_samples_leaf=35,
            l2_regularization=0.6,
            random_state=42,
        ),
    )
    x_train = train[cols].apply(pd.to_numeric, errors="coerce")
    y_train = train[target_col].astype(int)
    model.fit(x_train, y_train)

    for split_name, part in [("train", train), ("valid", valid), ("test", test)]:
        data = part.copy()
        x = data[cols].apply(pd.to_numeric, errors="coerce")
        data["split"] = split_name
        data[out_col] = model.predict_proba(x)[:, 1]
        frames.append(data)
    importance = permutation_importance(valid, target_col, cols, model)
    return pd.concat(frames, ignore_index=True), importance


def permutation_importance(valid: pd.DataFrame, target_col: str, cols: list[str], model) -> pd.DataFrame:
    if valid.empty or not cols:
        return pd.DataFrame({"feature": cols, "importance": 0.0})
    rng = np.random.default_rng(42)
    x = valid[cols].apply(pd.to_numeric, errors="coerce")
    y = valid[target_col].astype(int).to_numpy()
    base = neg_logloss(y, model.predict_proba(x)[:, 1])
    rows = []
    for col in cols:
        x_perm = x.copy()
        x_perm[col] = rng.permutation(x_perm[col].to_numpy())
        score = neg_logloss(y, model.predict_proba(x_perm)[:, 1])
        rows.append({"feature": col, "importance": float(base - score)})
    return pd.DataFrame(rows).sort_values("importance", ascending=False)


def neg_logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def add_blends(pred_5d: pd.DataFrame, pred_20d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = pred_5d.copy()
    b = pred_20d.copy()
    a["entry_adjusted_5d_score"] = a["rule_5d_score"] + 0.75 * a.get("entry_prob_5d", 0.5)
    b["z_ranker_20d"] = b.groupby(["date", "model_group"])["ranker_score"].transform(cs_z)
    b["z_rule_20d"] = b.groupby(["date", "model_group"])["rule_20d_score"].transform(cs_z)
    b["blend_20d_score"] = 0.45 * b["z_rule_20d"] + 0.55 * b["z_ranker_20d"]
    b = add_score_context(b, "blend_20d_score", "blend_20d")
    b["entry_adjusted_20d_score"] = b["blend_20d_score"] + 0.75 * b.get("entry_prob_20d", 0.5)
    return a, b


def cs_z(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    std = x.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=s.index)
    return (x - x.mean()) / std


def select_rebalance_dates(dates: pd.Series, horizon: str) -> list[pd.Timestamp]:
    idx = pd.DatetimeIndex(pd.to_datetime(dates).dropna().unique()).sort_values()
    if horizon == "1M":
        return pd.Series(idx, index=idx).groupby(idx.to_period("M")).max().tolist()
    return pd.Series(idx, index=idx).groupby(idx.to_period("W-FRI")).max().tolist()


def backtest(
    frame: pd.DataFrame,
    score_col: str,
    horizon: str,
    top_k: int,
    split: str = "test",
    entry_prob_col: str | None = None,
    threshold: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    data = frame[frame["split"].eq(split)].copy() if "split" in frame.columns else frame.copy()
    ret_col = "forward_5D_return" if horizon == "1W" else "forward_20D_return"
    bench_col = "benchmark_forward_5D_return" if horizon == "1W" else "benchmark_forward_20D_return"
    excess_col = "forward_5D_excess" if horizon == "1W" else "forward_20D_excess"
    rows = []
    for dt in select_rebalance_dates(data["date"], horizon):
        sample = data[data["date"].eq(dt)].dropna(subset=[score_col, ret_col, bench_col, excess_col]).copy()
        if entry_prob_col and threshold is not None and entry_prob_col in sample.columns:
            sample = sample[pd.to_numeric(sample[entry_prob_col], errors="coerce").ge(threshold)]
        if sample.empty:
            rows.append(flat_row(dt, horizon, score_col, top_k))
            continue
        # Basket-aware: first rank ETFs inside each theme, then compete the best
        # representative candidates across baskets.
        leaders = sample.sort_values(score_col, ascending=False).groupby("ranking_group", as_index=False).head(1)
        pool = leaders if leaders.shape[0] >= top_k else sample
        top = pool.nlargest(min(top_k, pool.shape[0]), score_col)
        rows.append(
            {
                "date": dt,
                "horizon": horizon,
                "score_col": score_col,
                "top_k": top_k,
                "portfolio_return": top[ret_col].mean(),
                "benchmark_return": top[bench_col].mean(),
                "excess_return": top[excess_col].mean(),
                "hit_excess": int(top[excess_col].mean() > 0),
                "hit_positive": int(top[ret_col].mean() > 0),
                "invested": 1,
                "selected": ",".join(top["etf_ticker"].astype(str).tolist()),
                "selected_names": ",".join(top.get("name", top["etf_ticker"]).astype(str).tolist()),
                "selected_groups": ",".join(top["ranking_group"].astype(str).tolist()),
            }
        )
    raw = pd.DataFrame(rows)
    return raw, summarize(raw, f"{score_col}_{horizon}_top{top_k}")


def flat_row(dt: pd.Timestamp, horizon: str, score_col: str, top_k: int) -> dict:
    return {
        "date": dt,
        "horizon": horizon,
        "score_col": score_col,
        "top_k": top_k,
        "portfolio_return": 0.0,
        "benchmark_return": 0.0,
        "excess_return": 0.0,
        "hit_excess": np.nan,
        "hit_positive": np.nan,
        "invested": 0,
        "selected": "",
        "selected_names": "",
        "selected_groups": "",
    }


def summarize(raw: pd.DataFrame, label: str) -> dict:
    if raw.empty:
        return {"label": label, "periods": 0}
    periods_per_year = 52 if raw["horizon"].iloc[0] == "1W" else 12
    returns = raw["portfolio_return"].astype(float).fillna(0.0)
    excess = raw["excess_return"].astype(float)
    equity = (1.0 + returns).cumprod()
    invested = raw.get("invested", pd.Series(1, index=raw.index)).astype(int)
    trade = raw[invested.eq(1)]
    std = returns.std(ddof=1)
    sharpe = np.nan if pd.isna(std) or std == 0 else returns.mean() / std * np.sqrt(periods_per_year)
    mdd = (equity / equity.cummax() - 1.0).min()
    return {
        "label": label,
        "horizon": raw["horizon"].iloc[0],
        "top_k": int(raw["top_k"].iloc[0]),
        "periods": int(raw.shape[0]),
        "invested_periods": int(invested.sum()),
        "coverage": float(invested.mean()),
        "cumulative_return": float(equity.iloc[-1] - 1.0),
        "CAGR": float(equity.iloc[-1] ** (periods_per_year / raw.shape[0]) - 1.0) if equity.iloc[-1] > 0 else np.nan,
        "MDD": float(mdd),
        "Sharpe": float(sharpe) if not pd.isna(sharpe) else np.nan,
        "hit_excess": float(trade["hit_excess"].mean()) if not trade.empty else np.nan,
        "hit_positive": float(trade["hit_positive"].mean()) if not trade.empty else np.nan,
        "avg_return": float(trade["portfolio_return"].mean()) if not trade.empty else np.nan,
        "avg_excess": float(trade["excess_return"].mean()) if not trade.empty else np.nan,
    }


def optimize_threshold(frame: pd.DataFrame, score_col: str, horizon: str, top_k: int, prob_col: str) -> float:
    best = (0.0, 0.0, -999.0)
    for threshold in np.arange(0.45, 0.81, 0.03):
        raw, summary = backtest(frame, score_col, horizon, top_k, split="valid", entry_prob_col=prob_col, threshold=float(threshold))
        coverage = summary.get("coverage", 0.0)
        sharpe = summary.get("Sharpe", -999.0)
        hit = summary.get("hit_excess", 0.0)
        if coverage < 0.20:
            continue
        objective = (0 if pd.isna(hit) else hit) + 0.08 * (0 if pd.isna(sharpe) else sharpe)
        if objective > best[2]:
            best = (float(threshold), float(coverage), float(objective))
    return best[0] or 0.50


def basket_scores(current: pd.DataFrame, score_col: str, prob_col: str | None = None) -> pd.DataFrame:
    rows = []
    for (asset_basket, group), part in current.groupby(["asset_basket", "ranking_group"]):
        top = part.nlargest(min(3, len(part)), score_col)
        rows.append(
            {
                "asset_basket": asset_basket,
                "ranking_group": group,
                "etf_count": int(part.shape[0]),
                "top_score": float(top[score_col].mean()),
                "top_entry_prob": float(top[prob_col].mean()) if prob_col and prob_col in top else np.nan,
                "top_etfs": ",".join(top["etf_ticker"].astype(str).tolist()),
                "top_names": ",".join(top.get("name", top["etf_ticker"]).astype(str).tolist()),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["basket_score_0_100"] = out.groupby("asset_basket")["top_score"].transform(lambda s: 50 + 18 * np.tanh(cs_z(s) / 2.0))
    return out.sort_values(["asset_basket", "basket_score_0_100"], ascending=[True, False])


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.input, parse_dates=["date"])
    raw = attach_universe(raw, Path(args.universe))
    filtered = filter_quality(raw, args)
    scored = add_group_labels(add_rule_scores(filtered))
    train, valid, test = split_frame(scored, args.train_end, args.valid_end)

    # 1W: rule/filter driven. The meta-model decides whether the high-RS setup is
    # worth taking; the 5D ranker is intentionally not used in final scoring.
    meta_features_1w = FEATURES_1W + STRUCTURE_COLUMNS + ENTRY_CONTEXT_1W
    pred_5d, imp_entry_5d = train_entry_model(train, valid, test, "entry_5d_label", meta_features_1w, "entry_prob_5d")

    # 1M: basket-grouped learning-to-rank, then a separate entry filter.
    rank_features_20d = FEATURES_1M
    pred_20d_ranker, imp_ranker_20d = train_ranker(train, valid, test, "label_20D_group_rank_int", rank_features_20d)
    pred_20d_ranker = add_score_context(pred_20d_ranker, "rule_20d_score", "rule_20d")
    meta_features_1m = FEATURES_1M + STRUCTURE_COLUMNS + ENTRY_CONTEXT_1M
    pred_20d, imp_entry_20d = train_entry_model(
        pred_20d_ranker[pred_20d_ranker["split"].eq("train")],
        pred_20d_ranker[pred_20d_ranker["split"].eq("valid")],
        pred_20d_ranker[pred_20d_ranker["split"].eq("test")],
        "entry_20d_label",
        meta_features_1m,
        "entry_prob_20d",
    )

    pred_5d, pred_20d = add_blends(pred_5d, pred_20d)
    threshold_5d = optimize_threshold(pred_5d, "entry_adjusted_5d_score", "1W", 5, "entry_prob_5d")
    threshold_20d = optimize_threshold(pred_20d, "entry_adjusted_20d_score", "1M", 5, "entry_prob_20d")

    summaries = []
    raws = []
    for top_k in [int(x) for x in args.top_k_list.split(",") if x.strip()]:
        specs = [
            (pred_5d, "rule_5d_score", "1W", None, None),
            (pred_5d, "entry_adjusted_5d_score", "1W", "entry_prob_5d", threshold_5d),
            (pred_20d, "rule_20d_score", "1M", None, None),
            (pred_20d, "ranker_score", "1M", None, None),
            (pred_20d, "blend_20d_score", "1M", None, None),
            (pred_20d, "entry_adjusted_20d_score", "1M", "entry_prob_20d", threshold_20d),
        ]
        for frame, score_col, horizon, prob_col, threshold in specs:
            raw_bt, summary = backtest(frame, score_col, horizon, top_k, "test", prob_col, threshold)
            summary["model"] = score_col
            summary["entry_threshold"] = threshold
            summaries.append(summary)
            raw_bt["model"] = score_col
            raw_bt["entry_threshold"] = threshold
            raws.append(raw_bt)

    summary_df = pd.DataFrame(summaries).sort_values(["horizon", "Sharpe"], ascending=[True, False])
    raw_df = pd.concat(raws, ignore_index=True) if raws else pd.DataFrame()

    latest_5d = pred_5d[pred_5d["date"].eq(pred_5d["date"].max())].copy()
    latest_20d = pred_20d[pred_20d["date"].eq(pred_20d["date"].max())].copy()
    basket_5d = basket_scores(latest_5d, "entry_adjusted_5d_score", "entry_prob_5d")
    basket_20d = basket_scores(latest_20d, "entry_adjusted_20d_score", "entry_prob_20d")

    scored.to_csv(out_dir / "v3_scored_features.csv", index=False, encoding="utf-8-sig")
    pred_5d.to_csv(out_dir / "v3_1w_rule_entry_predictions.csv", index=False, encoding="utf-8-sig")
    pred_20d.to_csv(out_dir / "v3_1m_ranker_entry_predictions.csv", index=False, encoding="utf-8-sig")
    imp_entry_5d.to_csv(out_dir / "v3_entry_5d_importance.csv", index=False, encoding="utf-8-sig")
    imp_ranker_20d.to_csv(out_dir / "v3_ranker_20d_importance.csv", index=False, encoding="utf-8-sig")
    imp_entry_20d.to_csv(out_dir / "v3_entry_20d_importance.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "v3_backtest_summary.csv", index=False, encoding="utf-8-sig")
    raw_df.to_csv(out_dir / "v3_backtest_trades.csv", index=False, encoding="utf-8-sig")
    basket_5d.to_csv(out_dir / "v3_current_basket_scores_1w.csv", index=False, encoding="utf-8-sig")
    basket_20d.to_csv(out_dir / "v3_current_basket_scores_1m.csv", index=False, encoding="utf-8-sig")

    print(f"input rows={raw.shape[0]:,} filtered rows={filtered.shape[0]:,} dates={filtered['date'].nunique():,} etfs={filtered['etf_ticker'].nunique():,}")
    print(f"entry thresholds: 1W={threshold_5d:.2f}, 1M={threshold_20d:.2f}")
    print(summary_df.head(24).to_string(index=False))
    print("\n20D ranker importance")
    print(imp_ranker_20d.head(14).to_string(index=False))
    print("\n1W entry importance")
    print(imp_entry_5d.head(14).to_string(index=False))


if __name__ == "__main__":
    main()
