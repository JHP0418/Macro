from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .backtest import evaluate_scores
from .config import DEFAULT_HOLDINGS_PATH, DEFAULT_OUTPUT_DIR, DEFAULT_UNIVERSE_PATH
from .data_loader import download_prices, load_holdings, load_prices_cache, load_universe, save_prices_cache
from .features import make_features
from .scoring import add_rule_scores
from .train_ranker import train_lgbm_ranker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ETF Leadership Ranking Model without macro/regime features.")
    parser.add_argument("--universe", default=str(DEFAULT_UNIVERSE_PATH))
    parser.add_argument("--holdings", default=str(DEFAULT_HOLDINGS_PATH))
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--feature-frequency", default="daily", help="Feature calculation frequency: daily, W-FRI, or M.")
    parser.add_argument("--rebalance-frequency", default="W-FRI", help="Backtest rebalance frequency: W-FRI or M.")
    parser.add_argument("--train-end", default="2021-12-31")
    parser.add_argument("--valid-end", default="2022-12-31")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prices-cache", default=None, help="Optional CSV cache for adjusted close prices.")
    parser.add_argument("--use-price-cache", action="store_true")
    parser.add_argument("--skip-ml", action="store_true", help="Build rule scores and rule backtest only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    universe = load_universe(args.universe)
    holdings = load_holdings(args.holdings)
    tickers = required_tickers(universe, holdings)

    cache = Path(args.prices_cache) if args.prices_cache else output_dir / "prices_adj_close.csv"
    if args.use_price_cache and cache.exists():
        prices = load_prices_cache(cache)
    else:
        prices = download_prices(tickers, args.start, args.end)
        save_prices_cache(prices, cache)

    features = make_features(universe, holdings, prices, frequency=args.feature_frequency)
    if features.empty:
        raise RuntimeError("No features were generated. Check universe, holdings, prices, and date range.")
    features.to_csv(output_dir / "features.csv", index=False, encoding="utf-8-sig")

    scored = add_rule_scores(features)
    scored.to_csv(output_dir / "rule_scores.csv", index=False, encoding="utf-8-sig")

    combined = scored.copy()
    score_cols = ["Final_Rule_Score"]
    if not args.skip_ml:
        predictions, importance, _ = train_lgbm_ranker(
            scored,
            label_col="label_20D_rank_int",
            train_end=args.train_end,
            valid_end=args.valid_end,
            output_dir=output_dir,
        )
        combined = predictions
        score_cols = ["Final_Rule_Score", "pred_score"]
        importance.to_csv(output_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
    else:
        combined["split"] = "all"

    raw_bt, summary = evaluate_scores(
        combined,
        score_cols=score_cols,
        top_k=args.top_k,
        frequency=args.rebalance_frequency,
        split=None if args.skip_ml else "test",
    )
    raw_bt.to_csv(output_dir / "backtest_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "backtest_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


def required_tickers(universe: pd.DataFrame, holdings: pd.DataFrame) -> list[str]:
    tickers = set(universe["etf_ticker"].astype(str))
    tickers.update(universe["benchmark_ticker"].astype(str))
    tickers.update(holdings["component_ticker"].astype(str))
    return sorted(tickers)


if __name__ == "__main__":
    main()
