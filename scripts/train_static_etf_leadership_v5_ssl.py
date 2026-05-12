from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_static_etf_leadership_v3 as v3  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ETF Leadership V5: V4 repaired universe plus SSL/VQ/NF features.")
    p.add_argument("--input", default=str(ROOT / "outputs" / "etf_leadership_static_holdings_repaired_v4base" / "rule_scores.csv"))
    p.add_argument("--universe", default=str(ROOT / "data" / "etf_universe_leadership.csv"))
    p.add_argument("--ssl-etf", default=str(ROOT / "outputs" / "ssl_etf_embeddings_latest" / "etf_ssl_embeddings.csv"))
    p.add_argument("--output-dir", default=str(ROOT / "outputs" / "etf_leadership_static_v5_ssl"))
    p.add_argument("--min-date", default="2020-01-01")
    p.add_argument("--train-end", default="2022-12-31")
    p.add_argument("--valid-end", default="2024-12-31")
    p.add_argument("--min-holdings", type=int, default=2)
    p.add_argument("--min-group-size", type=int, default=2)
    p.add_argument("--top-k-list", default="1,2,3,5")
    p.add_argument("--ssl-emb-dim", type=int, default=16)
    return p.parse_args()


def merge_ssl_features(frame: pd.DataFrame, ssl_path: Path, emb_dim: int) -> tuple[pd.DataFrame, list[str]]:
    ssl = pd.read_csv(ssl_path, parse_dates=["date"])
    ssl = ssl.rename(columns={"entity": "etf_ticker"})
    emb_cols = [f"ssl_emb_{i:02d}" for i in range(emb_dim) if f"ssl_emb_{i:02d}" in ssl.columns]
    stat_candidates = [
        "ssl_vq_state",
        "ssl_vq_distance",
        "ssl_flow_nll",
        "ssl_flow_confidence",
        "forward_5D_return_state_mean_prior",
        "forward_5D_return_state_hit_prior",
        "forward_5D_excess_state_mean_prior",
        "forward_5D_excess_state_hit_prior",
        "forward_20D_return_state_mean_prior",
        "forward_20D_return_state_hit_prior",
        "forward_20D_excess_state_mean_prior",
        "forward_20D_excess_state_hit_prior",
        "forward_20D_excess_state_count_prior",
    ]
    stat_cols = [c for c in stat_candidates if c in ssl.columns]
    keep = ["date", "etf_ticker", *emb_cols, *stat_cols]
    out = frame.merge(ssl[keep], on=["date", "etf_ticker"], how="left")
    ssl_cols = emb_cols + stat_cols
    for col in ssl_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if col.endswith("_state_count_prior"):
            out[col] = out[col].fillna(0)
    return out, ssl_cols


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.input, parse_dates=["date"])
    raw = v3.attach_universe(raw, Path(args.universe))
    filtered = v3.filter_quality(raw, args)
    scored = v3.add_group_labels(v3.add_rule_scores(filtered))
    scored, ssl_cols = merge_ssl_features(scored, Path(args.ssl_etf), args.ssl_emb_dim)
    train, valid, test = v3.split_frame(scored, args.train_end, args.valid_end)

    rank_features_20d = v3.FEATURES_1M + ssl_cols
    meta_features_1w = v3.FEATURES_1W + v3.STRUCTURE_COLUMNS + v3.ENTRY_CONTEXT_1W + ssl_cols
    pred_5d, imp_entry_5d = v3.train_entry_model(train, valid, test, "entry_5d_label", meta_features_1w, "entry_prob_5d")

    pred_20d_ranker, imp_ranker_20d = v3.train_ranker(train, valid, test, "label_20D_group_rank_int", rank_features_20d)
    pred_20d_ranker = v3.add_score_context(pred_20d_ranker, "rule_20d_score", "rule_20d")
    meta_features_1m = v3.FEATURES_1M + v3.STRUCTURE_COLUMNS + v3.ENTRY_CONTEXT_1M + ssl_cols
    pred_20d, imp_entry_20d = v3.train_entry_model(
        pred_20d_ranker[pred_20d_ranker["split"].eq("train")],
        pred_20d_ranker[pred_20d_ranker["split"].eq("valid")],
        pred_20d_ranker[pred_20d_ranker["split"].eq("test")],
        "entry_20d_label",
        meta_features_1m,
        "entry_prob_20d",
    )

    pred_5d, pred_20d = v3.add_blends(pred_5d, pred_20d)
    threshold_5d = v3.optimize_threshold(pred_5d, "entry_adjusted_5d_score", "1W", 5, "entry_prob_5d")
    threshold_20d = v3.optimize_threshold(pred_20d, "entry_adjusted_20d_score", "1M", 5, "entry_prob_20d")

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
            raw_bt, summary = v3.backtest(frame, score_col, horizon, top_k, "test", prob_col, threshold)
            summary["model"] = score_col
            summary["entry_threshold"] = threshold
            summary["ssl_features"] = len(ssl_cols)
            summaries.append(summary)
            raw_bt["model"] = score_col
            raw_bt["entry_threshold"] = threshold
            raws.append(raw_bt)

    summary_df = pd.DataFrame(summaries).sort_values(["horizon", "Sharpe"], ascending=[True, False])
    raw_df = pd.concat(raws, ignore_index=True) if raws else pd.DataFrame()
    latest_5d = pred_5d[pred_5d["date"].eq(pred_5d["date"].max())].copy()
    latest_20d = pred_20d[pred_20d["date"].eq(pred_20d["date"].max())].copy()
    basket_5d = v3.basket_scores(latest_5d, "entry_adjusted_5d_score", "entry_prob_5d")
    basket_20d = v3.basket_scores(latest_20d, "entry_adjusted_20d_score", "entry_prob_20d")

    scored.to_csv(out_dir / "v5_ssl_scored_features.csv", index=False, encoding="utf-8-sig")
    pred_5d.to_csv(out_dir / "v5_ssl_1w_rule_entry_predictions.csv", index=False, encoding="utf-8-sig")
    pred_20d.to_csv(out_dir / "v5_ssl_1m_ranker_entry_predictions.csv", index=False, encoding="utf-8-sig")
    imp_entry_5d.to_csv(out_dir / "v5_ssl_entry_5d_importance.csv", index=False, encoding="utf-8-sig")
    imp_ranker_20d.to_csv(out_dir / "v5_ssl_ranker_20d_importance.csv", index=False, encoding="utf-8-sig")
    imp_entry_20d.to_csv(out_dir / "v5_ssl_entry_20d_importance.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "v5_ssl_backtest_summary.csv", index=False, encoding="utf-8-sig")
    raw_df.to_csv(out_dir / "v5_ssl_backtest_trades.csv", index=False, encoding="utf-8-sig")
    basket_5d.to_csv(out_dir / "v5_ssl_current_basket_scores_1w.csv", index=False, encoding="utf-8-sig")
    basket_20d.to_csv(out_dir / "v5_ssl_current_basket_scores_1m.csv", index=False, encoding="utf-8-sig")

    print(f"filtered rows={filtered.shape[0]:,} ssl_features={len(ssl_cols)}")
    print(f"entry thresholds: 1W={threshold_5d:.2f}, 1M={threshold_20d:.2f}")
    print(summary_df.head(24).to_string(index=False))
    print("\n20D ranker importance")
    print(imp_ranker_20d.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
