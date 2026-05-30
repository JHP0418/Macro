from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_UNIVERSE = Path("data/etf_universe_leadership.csv")
DEFAULT_OUTPUT = Path("outputs/etf_beta_rs_atr_compare")


@dataclass(frozen=True)
class WalkWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Walk-forward comparison of a simple Beta-adjusted RS / ATR-MA "
            "ranking model and a tree regression ranking model."
        )
    )
    parser.add_argument("--universe", default=str(DEFAULT_UNIVERSE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--max-etfs", type=int, default=0, help="Use the first N ETFs. 0 means all ETFs in the universe.")
    parser.add_argument("--download-batch-size", type=int, default=20, help="Number of tickers per yfinance download batch.")
    parser.add_argument("--benchmark", default=None, help="Override all row-level benchmarks with one benchmark ticker.")
    parser.add_argument("--horizon", type=int, default=20, help="Forward return horizon used as Actual_Return.")
    parser.add_argument("--beta-window", type=int, default=60)
    parser.add_argument("--rs-window", type=int, default=20)
    parser.add_argument("--ma-window", type=int, default=50)
    parser.add_argument("--atr-window", type=int, default=14)
    parser.add_argument("--train-months", type=int, default=36)
    parser.add_argument("--test-months", type=int, default=1)
    parser.add_argument("--min-train-rows", type=int, default=500)
    parser.add_argument("--top-k", default="3,5,10,20")
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--skip-download", action="store_true", help="Require an existing ohlcv cache.")
    parser.add_argument("--plot", action="store_true", help="Write a Top-K decay chart PNG.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    universe = load_universe(args.universe, max_etfs=args.max_etfs, benchmark_override=args.benchmark)
    tickers = sorted(set(universe["Ticker"]).union(universe["Benchmark"]))
    ohlcv_cache = output_dir / "ohlcv.parquet"

    if args.skip_download or (args.use_cache and ohlcv_cache.exists()):
        ohlcv = pd.read_parquet(ohlcv_cache)
    else:
        ohlcv = download_ohlcv(tickers, args.start, args.end, batch_size=args.download_batch_size)
        ohlcv.to_parquet(ohlcv_cache)

    dataset = build_dataset(
        universe=universe,
        ohlcv=ohlcv,
        horizon=args.horizon,
        beta_window=args.beta_window,
        rs_window=args.rs_window,
        ma_window=args.ma_window,
        atr_window=args.atr_window,
    )
    if dataset.empty:
        raise RuntimeError("No model dataset was generated. Check universe tickers, benchmark tickers, and price history.")

    dataset.to_csv(output_dir / "features_and_labels.csv", index=False, encoding="utf-8-sig")

    outputs = run_walk_forward(
        dataset,
        train_months=args.train_months,
        test_months=args.test_months,
        min_train_rows=args.min_train_rows,
    )
    if not outputs:
        raise RuntimeError("No walk-forward predictions were generated. Try a shorter train window or longer history.")

    model_a = outputs["Model_A_Technical"]
    model_b = outputs["Model_B_Tree"]
    model_a.to_csv(output_dir / "model_a_output.csv", index=False, encoding="utf-8-sig")
    model_b.to_csv(output_dir / "model_b_output.csv", index=False, encoding="utf-8-sig")

    ks = tuple(int(k.strip()) for k in str(args.top_k).split(",") if k.strip())
    summary, topk = evaluate_models({"Model_A_Technical": model_a, "Model_B_Tree": model_b}, ks=ks)
    summary.to_csv(output_dir / "evaluation_summary.csv", index=False, encoding="utf-8-sig")
    topk.to_csv(output_dir / "topk_decay.csv", index=False, encoding="utf-8-sig")

    if args.plot:
        plot_topk_decay(topk, output_dir / "topk_decay.png")

    meta = {
        "universe": str(args.universe),
        "start": args.start,
        "end": args.end,
        "max_etfs": args.max_etfs,
        "download_batch_size": args.download_batch_size,
        "horizon": args.horizon,
        "beta_window": args.beta_window,
        "rs_window": args.rs_window,
        "ma_window": args.ma_window,
        "atr_window": args.atr_window,
        "outputs": {
            "features": "features_and_labels.csv",
            "model_a": "model_a_output.csv",
            "model_b": "model_b_output.csv",
            "summary": "evaluation_summary.csv",
            "topk_decay": "topk_decay.csv",
        },
    }
    (output_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


def load_universe(path: str | Path, max_etfs: int | None, benchmark_override: str | None) -> pd.DataFrame:
    raw = pd.read_csv(path)
    ticker_col = first_existing(raw, ["etf_ticker", "ticker", "symbol", "code"])
    benchmark_col = first_existing(raw, ["benchmark_ticker", "benchmark", "bench"])
    name_col = first_existing(raw, ["name", "etf_name"], required=False)

    frame = pd.DataFrame()
    frame["Ticker"] = normalize_tickers(raw[ticker_col])
    frame["Benchmark"] = str(benchmark_override).strip() if benchmark_override else normalize_tickers(raw[benchmark_col])
    frame["Name"] = raw[name_col].astype(str) if name_col else frame["Ticker"]
    frame = frame[frame["Ticker"].ne("") & frame["Benchmark"].ne("")]
    frame = frame.drop_duplicates("Ticker").reset_index(drop=True)
    if max_etfs:
        frame = frame.head(max_etfs)
    return frame


def first_existing(frame: pd.DataFrame, cols: list[str], required: bool = True) -> str | None:
    lower = {c.lower(): c for c in frame.columns}
    for col in cols:
        if col.lower() in lower:
            return lower[col.lower()]
    if required:
        raise ValueError(f"Missing required column. Tried: {cols}")
    return None


def normalize_tickers(values: Iterable[object]) -> pd.Series:
    out = pd.Series(values, dtype="string").str.strip()
    out = out.str.replace(r"^A(?=\d{6}$)", "", regex=True)
    out = out.mask(out.str.fullmatch(r"\d{6}", na=False), out + ".KS")
    return out.fillna("").astype(str)


def download_ohlcv(tickers: list[str], start: str, end: str | None, batch_size: int = 20) -> pd.DataFrame:
    import yfinance as yf

    parts = []
    batch_size = max(1, int(batch_size))
    for start_pos in range(0, len(tickers), batch_size):
        batch = tickers[start_pos : start_pos + batch_size]
        print(f"downloading OHLCV batch {start_pos // batch_size + 1}: {len(batch)} tickers", flush=True)
        raw = yf.download(
            tickers=batch,
            start=start,
            end=end,
            auto_adjust=True,
            actions=False,
            group_by="column",
            progress=False,
            threads=True,
        )
        if raw.empty:
            continue
        if not isinstance(raw.columns, pd.MultiIndex):
            raw.columns = pd.MultiIndex.from_product([raw.columns, batch[:1]])
        fields = [field for field in ["Open", "High", "Low", "Close", "Volume"] if field in raw.columns.get_level_values(0)]
        part = raw.loc[:, raw.columns.get_level_values(0).isin(fields)].copy()
        part.index = pd.to_datetime(part.index).tz_localize(None)
        part.index.name = "Date"
        part.columns = pd.MultiIndex.from_tuples([(str(field), str(ticker)) for field, ticker in part.columns])
        parts.append(part)
    if not parts:
        return pd.DataFrame()
    frame = pd.concat(parts, axis=1)
    frame = frame.loc[:, ~frame.columns.duplicated()]
    return frame.sort_index()


def build_dataset(
    universe: pd.DataFrame,
    ohlcv: pd.DataFrame,
    horizon: int,
    beta_window: int,
    rs_window: int,
    ma_window: int,
    atr_window: int,
) -> pd.DataFrame:
    frames = []
    close = field_frame(ohlcv, "Close").ffill(limit=5)
    high = field_frame(ohlcv, "High").ffill(limit=5)
    low = field_frame(ohlcv, "Low").ffill(limit=5)
    volume = field_frame(ohlcv, "Volume").ffill(limit=5)

    for row in universe.itertuples(index=False):
        ticker = str(row.Ticker)
        benchmark = str(row.Benchmark)
        if ticker not in close.columns or benchmark not in close.columns:
            continue

        px = close[ticker].astype(float)
        bench_px = close[benchmark].astype(float)
        ret = np.log(px / px.shift(1))
        bench_ret = np.log(bench_px / bench_px.shift(1))

        beta = ret.rolling(beta_window).cov(bench_ret) / bench_ret.rolling(beta_window).var()
        daily_alpha = ret - beta * bench_ret
        beta_adjusted_rs = daily_alpha.rolling(rs_window).sum()

        ma = px.rolling(ma_window).mean()
        prev_close = px.shift(1)
        tr = pd.concat(
            [
                high[ticker] - low[ticker],
                (high[ticker] - prev_close).abs(),
                (low[ticker] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(atr_window).mean()
        atr_pct = atr / px
        ma_gap_pct = px / ma - 1.0
        atr_multiple_from_ma = ma_gap_pct / atr_pct.replace(0, np.nan)

        out = pd.DataFrame(
            {
                "Date": px.index,
                "Ticker": ticker,
                "Benchmark": benchmark,
                "Name": getattr(row, "Name", ticker),
                "Close": px.to_numpy(),
                "Return_5D": px / px.shift(5) - 1.0,
                "Return_20D": px / px.shift(20) - 1.0,
                "Volatility_20D": ret.rolling(20).std() * np.sqrt(252),
                "Beta_60D": beta,
                "Beta_Adjusted_RS": beta_adjusted_rs,
                "Daily_Alpha": daily_alpha,
                "ATR_Pct": atr_pct,
                "ATR_Multiple_From_MA": atr_multiple_from_ma,
                "MA_Gap_20D": px / px.rolling(20).mean() - 1.0,
                "MA_Gap_50D": ma_gap_pct,
                "Volume_Z_50D": rolling_zscore(volume[ticker].astype(float), 50),
                "Actual_Return": px.shift(-horizon) / px - 1.0,
            }
        )
        frames.append(out)

    if not frames:
        return pd.DataFrame()
    dataset = pd.concat(frames, ignore_index=True)
    dataset = add_cross_sectional_features(dataset)
    needed = ["Beta_Adjusted_RS", "ATR_Multiple_From_MA", "Actual_Return"]
    dataset = dataset.dropna(subset=needed)
    return dataset.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def field_frame(ohlcv: pd.DataFrame, field: str) -> pd.DataFrame:
    if ohlcv.empty or field not in ohlcv.columns.get_level_values(0):
        return pd.DataFrame(index=ohlcv.index)
    return ohlcv.xs(field, axis=1, level=0).apply(pd.to_numeric, errors="coerce")


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std().replace(0, np.nan)
    return (series - mean) / std


def add_cross_sectional_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in [
        "Beta_Adjusted_RS",
        "ATR_Multiple_From_MA",
        "Return_5D",
        "Return_20D",
        "Volatility_20D",
        "MA_Gap_20D",
        "MA_Gap_50D",
        "ATR_Pct",
        "Volume_Z_50D",
    ]:
        out[f"z_{col}"] = out.groupby("Date")[col].transform(zscore)

    # Model A: reward residual momentum, penalize large positive MA distance in ATR units.
    out["Technical_Rank_Score"] = 0.70 * out["z_Beta_Adjusted_RS"] - 0.30 * out["z_ATR_Multiple_From_MA"]
    return out


def zscore(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    std = values.std()
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=values.index)
    return ((values - values.mean()) / std).replace([np.inf, -np.inf], np.nan)


def run_walk_forward(
    dataset: pd.DataFrame,
    train_months: int,
    test_months: int,
    min_train_rows: int,
) -> dict[str, pd.DataFrame]:
    data = dataset.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    windows = make_walk_windows(data["Date"], train_months=train_months, test_months=test_months)
    feature_cols = [
        "Beta_Adjusted_RS",
        "ATR_Multiple_From_MA",
        "z_Beta_Adjusted_RS",
        "z_ATR_Multiple_From_MA",
        "Return_5D",
        "Return_20D",
        "Volatility_20D",
        "MA_Gap_20D",
        "MA_Gap_50D",
        "ATR_Pct",
        "Volume_Z_50D",
        "z_Return_5D",
        "z_Return_20D",
        "z_Volatility_20D",
        "z_MA_Gap_20D",
        "z_MA_Gap_50D",
        "z_ATR_Pct",
        "z_Volume_Z_50D",
    ]
    feature_cols = [c for c in feature_cols if c in data.columns]

    model_a_parts = []
    model_b_parts = []
    for window in windows:
        train = data[(data["Date"] >= window.train_start) & (data["Date"] <= window.train_end)].copy()
        test = data[(data["Date"] >= window.test_start) & (data["Date"] <= window.test_end)].copy()
        train = train.dropna(subset=feature_cols + ["Actual_Return"])
        test = test.dropna(subset=feature_cols + ["Actual_Return"])
        if len(train) < min_train_rows or test.empty:
            continue

        model_a_parts.append(common_output(test, "Technical_Rank_Score"))

        model = make_tree_model()
        x_train = fill_features(train, feature_cols)
        y_train = train["Actual_Return"].astype(float)
        x_test = fill_features(test, feature_cols)
        model.fit(x_train, y_train)
        ml_test = test.copy()
        ml_test["Tree_Predicted_Return"] = model.predict(x_test)
        model_b_parts.append(common_output(ml_test, "Tree_Predicted_Return"))

    if not model_a_parts or not model_b_parts:
        return {}
    return {
        "Model_A_Technical": pd.concat(model_a_parts, ignore_index=True),
        "Model_B_Tree": pd.concat(model_b_parts, ignore_index=True),
    }


def make_walk_windows(dates: pd.Series, train_months: int, test_months: int) -> list[WalkWindow]:
    idx = pd.DatetimeIndex(pd.to_datetime(dates).dropna().unique()).sort_values()
    if idx.empty:
        return []
    start = idx.min() + pd.DateOffset(months=train_months)
    end = idx.max()
    windows = []
    test_start = pd.Timestamp(start.year, start.month, 1)
    while test_start <= end:
        test_end = test_start + pd.DateOffset(months=test_months) - pd.DateOffset(days=1)
        train_end = test_start - pd.DateOffset(days=1)
        train_start = test_start - pd.DateOffset(months=train_months)
        windows.append(WalkWindow(train_start, train_end, test_start, min(test_end, end)))
        test_start = test_start + pd.DateOffset(months=test_months)
    return windows


def make_tree_model():
    try:
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            num_leaves=15,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=5.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    except ImportError:
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1,
        )


def fill_features(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    x = frame[feature_cols].astype(float).replace([np.inf, -np.inf], np.nan)
    x = x.groupby(frame["Date"]).transform(lambda s: s.fillna(s.median()))
    return x.fillna(0.0)


def common_output(frame: pd.DataFrame, score_col: str) -> pd.DataFrame:
    out = frame[["Date", "Ticker", score_col, "Actual_Return"]].copy()
    out = out.rename(columns={score_col: "Predicted_Score"})
    return out.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def evaluate_models(outputs: dict[str, pd.DataFrame], ks: tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    topk_rows = []
    for model_name, output in outputs.items():
        spearman = spearman_by_date(output)
        rmse = fit_rmse(output)
        mean_spearman = float(spearman["Spearman"].mean()) if not spearman.empty else np.nan
        composite = composite_score(mean_spearman, rmse)
        summary_rows.append(
            {
                "Model": model_name,
                "Mean_Spearman": mean_spearman,
                "Fit_RMSE": rmse,
                "Composite_Score": composite,
                "Evaluated_Dates": int(output["Date"].nunique()),
                "Evaluated_Rows": int(len(output)),
            }
        )
        topk = top_k_returns(output, ks=ks)
        topk["Model"] = model_name
        topk_rows.append(topk)
    return pd.DataFrame(summary_rows), pd.concat(topk_rows, ignore_index=True)


def spearman_by_date(output: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import spearmanr

    rows = []
    for date, group in output.dropna(subset=["Predicted_Score", "Actual_Return"]).groupby("Date"):
        if group.shape[0] < 3 or group["Predicted_Score"].nunique() < 2 or group["Actual_Return"].nunique() < 2:
            continue
        corr, _ = spearmanr(group["Predicted_Score"], group["Actual_Return"])
        rows.append({"Date": date, "Spearman": corr})
    return pd.DataFrame(rows)


def fit_rmse(output: pd.DataFrame) -> float:
    from sklearn.metrics import mean_squared_error

    data = output.copy()
    data["z_pred"] = data.groupby("Date")["Predicted_Score"].transform(zscore)
    data["z_actual"] = data.groupby("Date")["Actual_Return"].transform(zscore)
    data = data.dropna(subset=["z_pred", "z_actual"])
    if data.empty:
        return np.nan
    return float(np.sqrt(mean_squared_error(data["z_actual"], data["z_pred"])))


def top_k_returns(output: pd.DataFrame, ks: tuple[int, ...]) -> pd.DataFrame:
    rows = []
    clean = output.dropna(subset=["Predicted_Score", "Actual_Return"]).copy()
    for k in ks:
        daily = (
            clean.sort_values(["Date", "Predicted_Score"], ascending=[True, False])
            .groupby("Date")
            .head(k)
            .groupby("Date")["Actual_Return"]
            .mean()
        )
        if daily.empty:
            rows.append({"K": k, "Mean_Return": np.nan, "Sharpe": np.nan, "Hit_Ratio": np.nan})
            continue
        std = daily.std()
        sharpe = float(daily.mean() / std * np.sqrt(252)) if pd.notna(std) and std != 0 else np.nan
        rows.append(
            {
                "K": k,
                "Mean_Return": float(daily.mean()),
                "Median_Return": float(daily.median()),
                "Sharpe": sharpe,
                "Hit_Ratio": float((daily > 0).mean()),
                "Periods": int(daily.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def composite_score(mean_spearman: float, rmse: float, w_corr: float = 0.6, w_rmse: float = 0.4) -> float:
    if pd.isna(mean_spearman) or pd.isna(rmse):
        return np.nan
    return float(w_corr * mean_spearman + w_rmse * (-rmse))


def plot_topk_decay(topk: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for model, group in topk.groupby("Model"):
        ax.plot(group["K"], group["Mean_Return"], marker="o", label=model)
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_title("Top-K Portfolio Return Decay")
    ax.set_xlabel("K")
    ax.set_ylabel("Average Forward Return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
