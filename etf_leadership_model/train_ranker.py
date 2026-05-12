from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS, LGBM_PARAMS


def time_split(
    frame: pd.DataFrame,
    train_end: str = "2021-12-31",
    valid_end: str = "2022-12-31",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"])
    train = out[out["date"].le(pd.Timestamp(train_end))]
    valid = out[out["date"].gt(pd.Timestamp(train_end)) & out["date"].le(pd.Timestamp(valid_end))]
    test = out[out["date"].gt(pd.Timestamp(valid_end))]
    return train, valid, test


def prepare_rank_data(frame: pd.DataFrame, label_col: str, feature_cols: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series, list[int], list[str]]:
    cols = feature_cols or FEATURE_COLUMNS
    cols = [c for c in cols if c in frame.columns]
    data = frame.dropna(subset=[label_col]).copy().sort_values(["date", "etf_ticker"]).reset_index(drop=True)
    x = data[cols].astype(float).replace([np.inf, -np.inf], np.nan)
    x = x.groupby(data["date"]).transform(lambda s: s.fillna(s.median())).fillna(0.0)
    y = data[label_col].astype(int)
    group = data.groupby("date", sort=False).size().astype(int).tolist()
    if sum(group) != len(data):
        raise ValueError("LightGBM group sizes do not sum to n_samples")
    return x, y, group, cols


def train_lgbm_ranker(
    frame: pd.DataFrame,
    label_col: str = "label_20D_rank_int",
    train_end: str = "2021-12-31",
    valid_end: str = "2022-12-31",
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    try:
        from lightgbm import LGBMRanker, early_stopping, log_evaluation
    except ImportError as exc:
        raise ImportError("lightgbm is required. Install with: pip install lightgbm") from exc

    train, valid, test = time_split(frame, train_end=train_end, valid_end=valid_end)
    x_train, y_train, group_train, cols = prepare_rank_data(train, label_col)
    x_valid, y_valid, group_valid, _ = prepare_rank_data(valid, label_col, cols)
    if x_train.empty or x_valid.empty:
        raise ValueError("train or validation set is empty. Check date split and feature data.")

    model = LGBMRanker(**LGBM_PARAMS)
    model.fit(
        x_train,
        y_train,
        group=group_train,
        eval_set=[(x_valid, y_valid)],
        eval_group=[group_valid],
        eval_at=[3, 5, 10],
        callbacks=[early_stopping(50), log_evaluation(50)],
    )

    predictions = []
    for split_name, part in [("train", train), ("valid", valid), ("test", test)]:
        if part.empty:
            continue
        x_part, _, _, _ = prepare_rank_data(part, label_col, cols)
        pred_part = part.dropna(subset=[label_col]).copy().sort_values(["date", "etf_ticker"]).reset_index(drop=True)
        pred_part["split"] = split_name
        pred_part["pred_score"] = model.predict(x_part)
        pred_part["pred_rank"] = pred_part.groupby("date")["pred_score"].rank(ascending=False, method="first")
        predictions.append(pred_part)
    pred = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()

    importance = pd.DataFrame(
        {
            "feature": cols,
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
            "importance_split": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False)

    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        pred.to_csv(output / "model_predictions.csv", index=False, encoding="utf-8-sig")
        importance.to_csv(output / "feature_importance.csv", index=False, encoding="utf-8-sig")
    return pred, importance, model

