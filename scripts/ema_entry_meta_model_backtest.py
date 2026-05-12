from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "ema_entry_meta_model_latest"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"

PROXY_FEATURES = ROOT / "outputs" / "long_history_proxy_selection_backtest_latest" / "tables" / "proxy_leadership_features_predictions.csv"
PROXY_PRICES = ROOT / "outputs" / "long_history_proxy_selection_backtest_latest" / "tables" / "proxy_raw_prices.csv"
GAPS_FEATURES = ROOT / "outputs" / "etf_leadership_static_v4_repaired" / "v3_scored_features.csv"
GAPS_PRICES = ROOT / "data" / "gaps_long_lived_cache" / "gaps_etf_benchmark_prices_2010-01-01_2026-05-12.csv"
RISK_V4 = ROOT / "outputs" / "risk_off_v4_event_label_latest" / "tables" / "risk_off_v4_walkforward_predictions.csv"


META_FEATURES = [
    "score_mean",
    "score_min",
    "score_std",
    "score_spread",
    "ret5_mean",
    "ret20_mean",
    "ret60_mean",
    "vol20_mean",
    "drawdown60_mean",
    "hp_mean",
    "breadth60_mean",
    "breadth200_mean",
    "ema_trend_share",
    "close_above_ema20_share",
    "ema4_gt_ema6_share",
    "ema6_gt_ema20_share",
    "ema4_ema6_spread_mean",
    "ema6_ema20_spread_mean",
    "dist_to_ema20_mean",
    "ema4_slope3_mean",
    "ema6_slope5_mean",
    "ema20_slope10_mean",
    "risk_off_prob",
    "risk_axis1",
    "risk_axis2",
    "risk_axis3",
]


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def to_num(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def weekly_dates(dates: pd.Series) -> set[pd.Timestamp]:
    idx = pd.DatetimeIndex(pd.to_datetime(dates).dropna().unique()).sort_values()
    if idx.empty:
        return set()
    return set(pd.Series(idx, index=idx).groupby(idx.to_period("W-FRI")).max())


def add_ema_long(prices: pd.DataFrame) -> pd.DataFrame:
    px = prices.copy().sort_index()
    rows = []
    for ticker in px.columns:
        s = pd.to_numeric(px[ticker], errors="coerce").ffill(limit=5)
        ema4 = s.ewm(span=4, adjust=False, min_periods=4).mean()
        ema6 = s.ewm(span=6, adjust=False, min_periods=6).mean()
        ema20 = s.ewm(span=20, adjust=False, min_periods=20).mean()
        part = pd.DataFrame(
            {
                "date": px.index,
                "ticker": ticker,
                "close": s.values,
                "ema_trend": ((ema4 > ema6) & (ema6 > ema20)).astype(float).values,
                "close_above_ema20": (s > ema20).astype(float).values,
                "ema4_gt_ema6": (ema4 > ema6).astype(float).values,
                "ema6_gt_ema20": (ema6 > ema20).astype(float).values,
                "ema4_ema6_spread": ((ema4 / ema6) - 1.0).values,
                "ema6_ema20_spread": ((ema6 / ema20) - 1.0).values,
                "dist_to_ema20": ((s / ema20) - 1.0).values,
                "ema4_slope3": ema4.pct_change(3).values,
                "ema6_slope5": ema6.pct_change(5).values,
                "ema20_slope10": ema20.pct_change(10).values,
            }
        )
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def forward_path_min_return(prices: pd.DataFrame, date: pd.Timestamp, tickers: list[str], horizon: int = 5) -> float:
    tickers = [t for t in tickers if t in prices.columns]
    if not tickers or date not in prices.index:
        return np.nan
    loc = prices.index.get_loc(date)
    if isinstance(loc, slice):
        loc = loc.start
    end = min(int(loc) + horizon, len(prices.index) - 1)
    if end <= loc:
        return np.nan
    base = prices.iloc[int(loc)][tickers].astype(float)
    path = prices.iloc[int(loc) + 1 : end + 1][tickers].astype(float)
    valid = base.replace(0.0, np.nan).dropna().index.tolist()
    if not valid:
        return np.nan
    rel = path[valid].divide(base[valid], axis=1) - 1.0
    portfolio_path = rel.mean(axis=1)
    return float(portfolio_path.min()) if not portfolio_path.empty else np.nan


def prepare_proxy_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    features = read_csv(PROXY_FEATURES, parse_dates=["date"], low_memory=False)
    prices = read_csv(PROXY_PRICES, parse_dates=["date"]).set_index("date").sort_index()
    numeric = [
        "rule_leadership_score",
        "ranker_5d_score",
        "ret_5",
        "ret_20",
        "ret_60",
        "vol_20",
        "drawdown_60",
        "high_proximity_252",
        "above_ma60",
        "above_ma200",
        "fwd_5d_return",
        "fwd_5d_excess",
    ]
    features = to_num(features, numeric)
    return features, prices


def prepare_gaps_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    features = read_csv(GAPS_FEATURES, parse_dates=["date"], low_memory=False)
    prices = read_csv(GAPS_PRICES, parse_dates=["date"]).set_index("date").sort_index()
    numeric = [
        "rule_5d_score",
        "ETF_return_5D",
        "ETF_return_20D",
        "ETF_return_60D",
        "weighted_HP",
        "MA60_breadth",
        "MA200_breadth",
        "forward_5D_return",
        "forward_5D_excess",
    ]
    features = to_num(features, numeric)
    return features, prices


def aggregate_panel(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    ema_long: pd.DataFrame,
    source: str,
    strategy: str,
    score_col: str,
    top_k: int,
    start: str,
    end: str,
    ticker_col: str,
    colmap: dict[str, str],
) -> pd.DataFrame:
    x = features.copy()
    x = x[pd.to_datetime(x["date"]).between(pd.Timestamp(start), pd.Timestamp(end))]
    x = x[x["date"].isin(weekly_dates(x["date"]))]
    x = x.dropna(subset=[score_col, colmap["fwd_return"], colmap["fwd_excess"]])
    if x.empty:
        return pd.DataFrame()
    x["ticker_for_ema"] = x[ticker_col].astype(str).str.upper()
    e = ema_long.rename(columns={"ticker": "ticker_for_ema"})
    x = x.merge(e, on=["date", "ticker_for_ema"], how="left")
    x["rank"] = x.groupby("date")[score_col].rank(ascending=False, method="first")
    x = x[x["rank"].le(top_k)].copy()
    rows = []
    for date, part in x.groupby("date"):
        if part[ticker_col].nunique() < top_k:
            continue
        scores = part[score_col].astype(float)
        selected = part[ticker_col].astype(str).tolist()
        fwd_return = float(part[colmap["fwd_return"]].astype(float).mean())
        fwd_excess = float(part[colmap["fwd_excess"]].astype(float).mean())
        min_fwd = forward_path_min_return(prices, pd.Timestamp(date), [s.upper() for s in selected], 5)
        row = {
            "date": pd.Timestamp(date),
            "source": source,
            "strategy": strategy,
            "score_col": score_col,
            "top_k": top_k,
            "selected": ",".join(selected),
            "portfolio_return": fwd_return,
            "excess_return": fwd_excess,
            "forward_min_return": min_fwd,
            "entry_success": int(fwd_return > 0 and fwd_excess > 0 and (pd.isna(min_fwd) or min_fwd > -0.02)),
            "score_mean": float(scores.mean()),
            "score_min": float(scores.min()),
            "score_std": float(scores.std(ddof=0)),
            "score_spread": float(scores.max() - scores.min()),
            "ret5_mean": mean_col(part, colmap.get("ret5")),
            "ret20_mean": mean_col(part, colmap.get("ret20")),
            "ret60_mean": mean_col(part, colmap.get("ret60")),
            "vol20_mean": mean_col(part, colmap.get("vol20")),
            "drawdown60_mean": mean_col(part, colmap.get("drawdown60")),
            "hp_mean": mean_col(part, colmap.get("hp")),
            "breadth60_mean": mean_col(part, colmap.get("breadth60")),
            "breadth200_mean": mean_col(part, colmap.get("breadth200")),
            "ema_trend_share": mean_col(part, "ema_trend"),
            "close_above_ema20_share": mean_col(part, "close_above_ema20"),
            "ema4_gt_ema6_share": mean_col(part, "ema4_gt_ema6"),
            "ema6_gt_ema20_share": mean_col(part, "ema6_gt_ema20"),
            "ema4_ema6_spread_mean": mean_col(part, "ema4_ema6_spread"),
            "ema6_ema20_spread_mean": mean_col(part, "ema6_ema20_spread"),
            "dist_to_ema20_mean": mean_col(part, "dist_to_ema20"),
            "ema4_slope3_mean": mean_col(part, "ema4_slope3"),
            "ema6_slope5_mean": mean_col(part, "ema6_slope5"),
            "ema20_slope10_mean": mean_col(part, "ema20_slope10"),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def mean_col(frame: pd.DataFrame, col: str | None) -> float:
    if not col or col not in frame.columns:
        return np.nan
    return float(pd.to_numeric(frame[col], errors="coerce").mean())


def attach_risk(panel: pd.DataFrame) -> pd.DataFrame:
    risk = read_csv(RISK_V4, parse_dates=["date"], low_memory=False)
    if panel.empty or risk.empty:
        panel["risk_off_prob"] = 0.0
        panel["risk_axis1"] = 0.0
        panel["risk_axis2"] = 0.0
        panel["risk_axis3"] = 0.0
        return panel
    r = risk[risk["horizon"].astype(str).str.lower().eq("1w")].copy()
    cols = [
        "date",
        "risk_off_v4_prob",
        "axis1_vol_credit_stress",
        "axis2_fx_liquidity_stress",
        "axis3_peak_fragility_stress",
    ]
    r = r[[c for c in cols if c in r.columns]].sort_values("date")
    out = pd.merge_asof(panel.sort_values("date"), r, on="date", direction="backward")
    out = out.rename(
        columns={
            "risk_off_v4_prob": "risk_off_prob",
            "axis1_vol_credit_stress": "risk_axis1",
            "axis2_fx_liquidity_stress": "risk_axis2",
            "axis3_peak_fragility_stress": "risk_axis3",
        }
    )
    return out


def build_strategy_panels() -> pd.DataFrame:
    proxy, proxy_prices = prepare_proxy_features()
    gaps, gaps_prices = prepare_gaps_features()
    proxy_ema = add_ema_long(proxy_prices)
    gaps_ema = add_ema_long(gaps_prices)
    proxy_map = {
        "ret5": "ret_5",
        "ret20": "ret_20",
        "ret60": "ret_60",
        "vol20": "vol_20",
        "drawdown60": "drawdown_60",
        "hp": "high_proximity_252",
        "breadth60": "above_ma60",
        "breadth200": "above_ma200",
        "fwd_return": "fwd_5d_return",
        "fwd_excess": "fwd_5d_excess",
    }
    gaps_map = {
        "ret5": "ETF_return_5D",
        "ret20": "ETF_return_20D",
        "ret60": "ETF_return_60D",
        "hp": "weighted_HP",
        "breadth60": "MA60_breadth",
        "breadth200": "MA200_breadth",
        "fwd_return": "forward_5D_return",
        "fwd_excess": "forward_5D_excess",
    }
    panels = []
    panels.append(
        aggregate_panel(
            proxy,
            proxy_prices,
            proxy_ema,
            "long_proxy",
            "rule_top3_hybrid_1w",
            "rule_leadership_score",
            3,
            "2010-01-01",
            "2024-12-31",
            "ticker",
            proxy_map,
        )
    )
    panels.append(
        aggregate_panel(
            proxy,
            proxy_prices,
            proxy_ema,
            "long_proxy",
            "ranker_top3_hybrid_1w",
            "ranker_5d_score",
            3,
            "2016-01-01",
            "2024-12-31",
            "ticker",
            proxy_map,
        )
    )
    for strategy in ["rule_top3_hybrid_1w", "ranker_top3_hybrid_1w"]:
        panels.append(
            aggregate_panel(
                gaps,
                gaps_prices,
                gaps_ema,
                "db_gaps",
                strategy,
                "rule_5d_score",
                3,
                "2025-01-01",
                "2026-04-30",
                "etf_ticker",
                gaps_map,
            )
        )
    panel = pd.concat([p for p in panels if not p.empty], ignore_index=True)
    panel = attach_risk(panel)
    for col in META_FEATURES:
        if col not in panel.columns:
            panel[col] = np.nan
        panel[col] = pd.to_numeric(panel[col], errors="coerce")
    return panel.sort_values(["strategy", "date"])


def perf(frame: pd.DataFrame, invested: pd.Series, label: str, model: str, threshold: float | None = None) -> dict:
    x = frame.sort_values("date").copy()
    inv = invested.reindex(x.index).fillna(False).astype(bool)
    ret = pd.Series(np.where(inv, x["portfolio_return"].astype(float), 0.0), index=x.index)
    active = x[inv]
    equity = (1.0 + ret).cumprod()
    ann_vol = float(ret.std(ddof=1) * np.sqrt(52))
    mdd = float((equity / equity.cummax() - 1.0).min()) if not equity.empty else np.nan
    years = max((x["date"].max() - x["date"].min()).days / 365.25, 1e-9)
    return {
        "strategy": label,
        "model": model,
        "threshold": threshold,
        "start": x["date"].min().date().isoformat(),
        "end": x["date"].max().date().isoformat(),
        "periods": int(len(x)),
        "invested_periods": int(inv.sum()),
        "coverage": float(inv.mean()),
        "cumulative_return": float(equity.iloc[-1] - 1.0) if not equity.empty else np.nan,
        "CAGR": float(equity.iloc[-1] ** (1.0 / years) - 1.0) if not equity.empty and equity.iloc[-1] > 0 else np.nan,
        "ann_vol": ann_vol,
        "Sharpe": float(ret.mean() * 52 / ann_vol) if ann_vol > 0 else np.nan,
        "MDD": mdd,
        "hit_positive": float((active["portfolio_return"] > 0).mean()) if not active.empty else np.nan,
        "hit_excess": float((active["excess_return"] > 0).mean()) if not active.empty else np.nan,
        "entry_success_rate": float(active["entry_success"].mean()) if not active.empty else np.nan,
        "avg_return": float(active["portfolio_return"].mean()) if not active.empty else np.nan,
        "avg_excess": float(active["excess_return"].mean()) if not active.empty else np.nan,
        "false_entry_rate": float((active["entry_success"] == 0).mean()) if not active.empty else np.nan,
    }


def select_threshold(valid: pd.DataFrame, prob: np.ndarray) -> float:
    candidates = sorted(set(np.linspace(0.25, 0.75, 21).tolist() + np.quantile(prob, [0.4, 0.5, 0.6, 0.7, 0.8]).tolist()))
    best = (float("-inf"), 0.5)
    p = pd.Series(prob, index=valid.index)
    for th in candidates:
        inv = p >= th
        if inv.sum() < max(8, int(len(valid) * 0.20)):
            continue
        res = perf(valid, inv, "valid", "threshold")
        objective = (
            safe(res["Sharpe"])
            + 0.8 * safe(res["entry_success_rate"])
            + 0.5 * safe(res["hit_positive"])
            + 0.3 * safe(res["hit_excess"])
            - 0.8 * abs(safe(res["MDD"]))
        )
        if objective > best[0]:
            best = (objective, float(th))
    return best[1]


def select_conservative_threshold(valid: pd.DataFrame, prob: np.ndarray) -> float:
    candidates = sorted(set(np.linspace(0.35, 0.90, 23).tolist() + np.quantile(prob, [0.55, 0.65, 0.75, 0.85, 0.90]).tolist()))
    best = (float("-inf"), 0.65)
    p = pd.Series(prob, index=valid.index)
    for th in candidates:
        inv = p >= th
        if inv.sum() < max(6, int(len(valid) * 0.15)):
            continue
        res = perf(valid, inv, "valid", "threshold_conservative")
        objective = (
            0.6 * safe(res["Sharpe"])
            + 1.4 * safe(res["entry_success_rate"])
            + 0.8 * safe(res["hit_positive"])
            + 0.6 * safe(res["hit_excess"])
            - 1.2 * safe(res["false_entry_rate"])
            - 0.8 * abs(safe(res["MDD"]))
        )
        if objective > best[0]:
            best = (objective, float(th))
    return best[1]


def safe(value: object, fallback: float = 0.0) -> float:
    try:
        x = float(value)
        return x if np.isfinite(x) else fallback
    except Exception:
        return fallback


def fit_predict_walk_forward(panel: pd.DataFrame, strategy: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = panel[panel["strategy"].eq(strategy)].dropna(subset=["entry_success"]).copy()
    data = data.sort_values("date")
    for col in META_FEATURES:
        data[col] = data[col].replace([np.inf, -np.inf], np.nan)
    data[META_FEATURES] = data[META_FEATURES].fillna(data[META_FEATURES].expanding().mean()).fillna(0.0)
    pred_rows = []
    metric_rows = []
    years = sorted(data["date"].dt.year.unique())
    for year in years:
        if year < 2014:
            continue
        test = data[data["date"].dt.year.eq(year)].copy()
        train_all = data[data["date"].dt.year.lt(year)].copy()
        if len(train_all) < 120 or test.empty or train_all["entry_success"].nunique() < 2:
            continue
        valid_year = year - 1
        core = train_all[train_all["date"].dt.year.lt(valid_year)].copy()
        valid = train_all[train_all["date"].dt.year.eq(valid_year)].copy()
        if len(core) < 80 or len(valid) < 20 or core["entry_success"].nunique() < 2:
            split_idx = int(len(train_all) * 0.75)
            core = train_all.iloc[:split_idx].copy()
            valid = train_all.iloc[split_idx:].copy()
        if core["entry_success"].nunique() < 2 or valid.empty:
            continue
        X_core, y_core = core[META_FEATURES], core["entry_success"].astype(int)
        X_valid, y_valid = valid[META_FEATURES], valid["entry_success"].astype(int)
        X_test = test[META_FEATURES]

        models = {}
        models["logistic_elasticnet"] = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        penalty="elasticnet",
                        solver="saga",
                        l1_ratio=0.25,
                        C=0.6,
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )
        models["lightgbm"] = LGBMClassifier(
            objective="binary",
            n_estimators=250,
            learning_rate=0.025,
            num_leaves=7,
            max_depth=3,
            min_child_samples=25,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=1.0,
            reg_lambda=5.0,
            class_weight="balanced",
            random_state=42,
            verbosity=-1,
        )
        for name, model in models.items():
            model.fit(X_core, y_core)
            valid_prob = model.predict_proba(X_valid)[:, 1]
            test_prob = model.predict_proba(X_test)[:, 1]
            threshold = select_threshold(valid, valid_prob)
            conservative_threshold = select_conservative_threshold(valid, valid_prob)
            pred = test[["date", "source", "strategy", "selected", "portfolio_return", "excess_return", "forward_min_return", "entry_success"]].copy()
            pred["model"] = name
            pred["entry_prob"] = test_prob
            pred["threshold"] = threshold
            pred["invested"] = pred["entry_prob"] >= threshold
            pred["ema_gate"] = ema_hard_filter(test).values
            pred_rows.append(pred)
            pred_con = pred.copy()
            pred_con["model"] = f"{name}_conservative"
            pred_con["threshold"] = conservative_threshold
            pred_con["invested"] = pred_con["entry_prob"] >= conservative_threshold
            pred_rows.append(pred_con)
            pred_ema = pred.copy()
            pred_ema["model"] = f"{name}_plus_ema"
            pred_ema["invested"] = pred_ema["invested"] & pred_ema["ema_gate"].astype(bool)
            pred_rows.append(pred_ema)
            metric_rows.append(metric_classification(strategy, name, year, y_valid, valid_prob, threshold, "valid"))
            metric_rows.append(metric_classification(strategy, name, year, test["entry_success"].astype(int), test_prob, threshold, "test"))

        # Platt-calibrated LightGBM: fit on core, learn a probability calibration layer on the previous year.
        base = models["lightgbm"]
        valid_base = base.predict_proba(X_valid)[:, 1].reshape(-1, 1)
        test_base = base.predict_proba(X_test)[:, 1].reshape(-1, 1)
        calibrator = LogisticRegression(C=1.0, max_iter=1000)
        if y_valid.nunique() >= 2:
            calibrator.fit(valid_base, y_valid)
            valid_prob = calibrator.predict_proba(valid_base)[:, 1]
            test_prob = calibrator.predict_proba(test_base)[:, 1]
        else:
            valid_prob = valid_base.ravel()
            test_prob = test_base.ravel()
        threshold = select_threshold(valid, valid_prob)
        conservative_threshold = select_conservative_threshold(valid, valid_prob)
        pred = test[["date", "source", "strategy", "selected", "portfolio_return", "excess_return", "forward_min_return", "entry_success"]].copy()
        pred["model"] = "lightgbm_platt_calibrated"
        pred["entry_prob"] = test_prob
        pred["threshold"] = threshold
        pred["invested"] = pred["entry_prob"] >= threshold
        pred["ema_gate"] = ema_hard_filter(test).values
        pred_rows.append(pred)
        pred_con = pred.copy()
        pred_con["model"] = "lightgbm_platt_calibrated_conservative"
        pred_con["threshold"] = conservative_threshold
        pred_con["invested"] = pred_con["entry_prob"] >= conservative_threshold
        pred_rows.append(pred_con)
        pred_ema = pred.copy()
        pred_ema["model"] = "lightgbm_platt_calibrated_plus_ema"
        pred_ema["invested"] = pred_ema["invested"] & pred_ema["ema_gate"].astype(bool)
        pred_rows.append(pred_ema)
        metric_rows.append(metric_classification(strategy, "lightgbm_platt_calibrated", year, y_valid, valid_prob, threshold, "valid"))
        metric_rows.append(metric_classification(strategy, "lightgbm_platt_calibrated", year, test["entry_success"].astype(int), test_prob, threshold, "test"))

    predictions = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    metrics = pd.DataFrame(metric_rows)
    return predictions, metrics


def metric_classification(strategy: str, model: str, year: int, y: pd.Series, prob: np.ndarray, threshold: float, split: str) -> dict:
    y_true = y.astype(int).values
    y_pred = (prob >= threshold).astype(int)
    try:
        auc = roc_auc_score(y_true, prob) if len(np.unique(y_true)) > 1 else np.nan
    except Exception:
        auc = np.nan
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {
        "strategy": strategy,
        "model": model,
        "year": year,
        "split": split,
        "threshold": threshold,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positive_rate": float(np.mean(y_true)),
        "predicted_entry_rate": float(np.mean(y_pred)),
    }


def ema_hard_filter(panel: pd.DataFrame) -> pd.Series:
    return (
        panel["ema_trend_share"].fillna(0.0).ge(2 / 3)
        & panel["close_above_ema20_share"].fillna(0.0).ge(2 / 3)
        & panel["ema6_slope5_mean"].fillna(0.0).gt(0.0)
        & panel["ema20_slope10_mean"].fillna(0.0).gt(-0.005)
    )


def summarize_models(panel: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy, part in panel.groupby("strategy"):
        part = part[part["date"].ge(pd.Timestamp("2014-01-01"))].copy()
        rows.append(perf(part, pd.Series(True, index=part.index), strategy, "baseline_always_enter"))
        rows.append(perf(part, ema_hard_filter(part), strategy, "ema_4_6_20_hard_filter"))
    if not predictions.empty:
        for (strategy, model), pred in predictions.groupby(["strategy", "model"]):
            rows.append(perf(pred, pred["invested"].astype(bool), strategy, model, float(pred["threshold"].median())))
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["strategy", "Sharpe"], ascending=[True, False])
    return out


def feature_importance_logit(panel: pd.DataFrame, strategy: str) -> pd.DataFrame:
    data = panel[panel["strategy"].eq(strategy)].dropna(subset=["entry_success"]).copy()
    train = data[data["date"].dt.year.lt(2025)].copy()
    if train.empty or train["entry_success"].nunique() < 2:
        return pd.DataFrame()
    X = train[META_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = train["entry_success"].astype(int)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratio=0.25,
                    C=0.6,
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X, y)
    coef = model.named_steps["model"].coef_[0]
    return pd.DataFrame({"strategy": strategy, "feature": META_FEATURES, "coef": coef, "abs_coef": np.abs(coef)}).sort_values("abs_coef", ascending=False)


def write_report(summary: pd.DataFrame, metrics: pd.DataFrame, importance: pd.DataFrame) -> None:
    lines = [
        "# EMA 4/6/20 Entry Meta Model Backtest",
        "",
        "ETF Leadership Top-K를 먼저 고른 뒤, 4EMA/6EMA/20EMA 정렬과 후보군 품질, Risk-Off V4 상태로 진입/대기를 결정하는 1주 Meta Model 검증이다.",
        "2010~2024년은 장기 상장 ETF 프록시, 2025년 이후는 DB GAPS ETF 구성종목 기반 모델을 이어 붙였다.",
        "",
        "## Performance Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Classification Metrics",
        "",
        metrics.groupby(["strategy", "model", "split"]).agg(
            auc=("auc", "mean"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean"),
            predicted_entry_rate=("predicted_entry_rate", "mean"),
        ).reset_index().to_markdown(index=False)
        if not metrics.empty
        else "metrics not available",
        "",
        "## Logistic / ElasticNet Feature Coefficients",
        "",
        importance.head(40).to_markdown(index=False) if not importance.empty else "importance not available",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    panel = build_strategy_panels()
    panel.to_csv(TABLES / "ema_entry_meta_panel.csv", index=False, encoding="utf-8-sig")
    all_predictions = []
    all_metrics = []
    all_importance = []
    for strategy in sorted(panel["strategy"].dropna().unique()):
        preds, metrics = fit_predict_walk_forward(panel, strategy)
        all_predictions.append(preds)
        all_metrics.append(metrics)
        all_importance.append(feature_importance_logit(panel, strategy))
    predictions = pd.concat([p for p in all_predictions if not p.empty], ignore_index=True) if all_predictions else pd.DataFrame()
    metrics = pd.concat([m for m in all_metrics if not m.empty], ignore_index=True) if all_metrics else pd.DataFrame()
    importance = pd.concat([i for i in all_importance if not i.empty], ignore_index=True) if all_importance else pd.DataFrame()
    summary = summarize_models(panel, predictions)
    predictions.to_csv(TABLES / "ema_entry_meta_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(TABLES / "ema_entry_meta_classification_metrics.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(TABLES / "ema_entry_meta_feature_importance.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(TABLES / "ema_entry_meta_backtest_summary.csv", index=False, encoding="utf-8-sig")
    write_report(summary, metrics, importance)
    print(f"saved {OUT}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
