from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from macro_regime_asset_screener import (  # noqa: E402
    ASSETS,
    CACHE_DIR,
    CORE_DRIVER_NAMES,
    FRED_SERIES,
    OUT_DIR,
    YF_SERIES,
    asset_driver_fit,
    beta_alignment_score,
    blend_probability,
    clean_series,
    current_driver_state,
    drawdown,
    load_asset_histories,
    load_driver_series,
    make_driver_features,
    make_driver_panel,
    pct_return,
    risk_score,
    rolling_driver_betas,
    safe_float,
    safe_to_csv,
    score_assets,
    technical_score,
)

RWKV_OUT_DIR = ROOT / "outputs" / "rwkv_lppl_asset_screener_latest"
MODEL_CACHE = CACHE_DIR / "models"
LPPL_CACHE = CACHE_DIR / "lppl"
SEQUENCE_LENGTH = 48
EMBED_DIM = 96
HIDDEN_DIM = 128
RWKV_EPOCHS = 220
LPPL_WINDOW = 504
LPPL_STEP = 21
LPPL_POPULATION = 200
LPPL_GENERATIONS = 700
LPPL_FITS_PER_WINDOW = 500
LPPL_RELIABILITY_MIN_SAMPLES = 40


@dataclass
class LPPLFit:
    A: float
    B: float
    C: float
    tc: float
    phi: float
    omega: float
    beta: float
    rmse: float
    dtc: float
    reliability: float
    dtcai: float
    label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RWKV macro regime + LPPL/DTCAI asset screener.")
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--output", type=Path, default=RWKV_OUT_DIR)
    parser.add_argument("--rwkv-epochs", type=int, default=RWKV_EPOCHS)
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--rwkv-frequency", choices=["D", "M"], default="M")
    parser.add_argument("--lppl-window", type=int, default=LPPL_WINDOW)
    parser.add_argument("--lppl-step", type=int, default=LPPL_STEP)
    parser.add_argument("--lppl-population", type=int, default=LPPL_POPULATION)
    parser.add_argument("--lppl-generations", type=int, default=LPPL_GENERATIONS)
    parser.add_argument("--lppl-fits-per-window", type=int, default=LPPL_FITS_PER_WINDOW)
    parser.add_argument("--lppl-training-step", type=int, default=LPPL_STEP)
    parser.add_argument("--max-training-windows", type=int, default=None)
    parser.add_argument("--refresh-lppl", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (MODEL_CACHE, LPPL_CACHE, args.output / "tables", args.output / "reports"):
        path.mkdir(parents=True, exist_ok=True)

    specs = FRED_SERIES + YF_SERIES
    raw, availability = load_driver_series(specs, args.start, args.skip_download)
    driver_panel = make_driver_panel(raw)
    driver_features = make_driver_features(driver_panel, specs)
    asset_histories = load_asset_histories([asset.symbol for asset in ASSETS], args.start, args.skip_download)
    driver_selection = macro_driver_selection_report(driver_features, asset_histories)
    regime_frame, embeddings, rwkv_meta = rwkv_regime_frame(driver_features, args.sequence_length, args.rwkv_epochs, args.rwkv_frequency)
    driver_state = current_driver_state(driver_panel, driver_features, specs, regime_frame)
    lppl_training = build_lppl_training_table(
        asset_histories,
        args.lppl_window,
        args.lppl_training_step,
        args.lppl_population,
        args.lppl_generations,
        args.lppl_fits_per_window,
        args.max_training_windows,
        args.refresh_lppl,
    )
    reliability_model, reliability_meta = fit_lppl_reliability_model(lppl_training)
    lppl_training_scored = score_lppl_training_with_model(lppl_training, reliability_model)
    lppl_validation = validate_lppl_training_signals(lppl_training_scored)
    base_scores = score_assets(ASSETS, asset_histories, driver_panel, driver_features, regime_frame, driver_state)
    lppl_current = current_lppl_table(
        asset_histories,
        args.lppl_window,
        args.lppl_population,
        args.lppl_generations,
        args.lppl_fits_per_window,
        reliability_model,
        args.refresh_lppl,
    )
    final_scores = merge_lppl_scores(base_scores, lppl_current)

    tables = args.output / "tables"
    reports = args.output / "reports"
    safe_to_csv(regime_frame.reset_index().rename(columns={"index": "Date"}), tables / "rwkv_regime_history.csv")
    safe_to_csv(embeddings.reset_index().rename(columns={"index": "Date"}), tables / "rwkv_embeddings.csv")
    safe_to_csv(driver_state, tables / "driver_state.csv")
    safe_to_csv(driver_selection, tables / "driver_selection_granger_corr.csv")
    safe_to_csv(lppl_training, tables / "lppl_reliability_training_set.csv")
    safe_to_csv(lppl_training_scored, tables / "lppl_reliability_training_scored.csv")
    safe_to_csv(lppl_validation, tables / "lppl_signal_validation.csv")
    safe_to_csv(lppl_current, tables / "current_lppl_dtcai.csv")
    safe_to_csv(final_scores, tables / "current_asset_scores_rwkv_lppl.csv")
    pd.DataFrame(availability).to_csv(tables / "data_availability.csv", index=False, encoding="utf-8-sig")
    write_rwkv_lppl_report(final_scores, driver_state, regime_frame, rwkv_meta, reliability_meta, lppl_validation, driver_selection, reports / "current_report.md")
    print(f"wrote {tables / 'current_asset_scores_rwkv_lppl.csv'}")
    print(final_scores.head(15).to_string(index=False))


def rwkv_regime_frame(features: pd.DataFrame, sequence_length: int, epochs: int, frequency: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    inputs = [f"{name}_riskon_score" for name in CORE_DRIVER_NAMES if f"{name}_riskon_score" in features]
    raw_x = features[inputs].copy().replace([np.inf, -np.inf], np.nan)
    if frequency == "M":
        raw_x = raw_x.groupby(raw_x.index.to_period("M")).tail(1)
    min_extra = 24 if frequency == "M" else 80
    min_required = sequence_length + min_extra
    inputs = [col for col in raw_x.columns if raw_x[col].dropna().shape[0] >= min_required]
    x = raw_x[inputs].ffill().dropna()
    if x.shape[1] < 5:
        inputs = raw_x.count().sort_values(ascending=False).head(8).index.tolist()
        x = raw_x[inputs].ffill().dropna()
    if x.shape[0] < sequence_length + min_extra:
        raise SystemExit("Not enough macro history for RWKV sequence training.")
    clipped = x.clip(-5, 5)
    scaler = StandardScaler()
    z = pd.DataFrame(scaler.fit_transform(clipped), index=clipped.index, columns=clipped.columns).astype("float32")
    sequences, dates = make_sequences(z, sequence_length)
    tensor = torch.tensor(sequences, dtype=torch.float32)
    model = RWKVMacroAutoencoder(input_dim=tensor.shape[-1], hidden_dim=HIDDEN_DIM, embed_dim=EMBED_DIM)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_loss = float("inf")
    patience = 35
    stale = 0
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(tensor.shape[0])
        losses = []
        for batch_idx in order.split(64):
            batch = tensor[batch_idx]
            optimizer.zero_grad()
            recon, pred, _ = model(batch)
            recon_loss = F.mse_loss(recon, batch)
            pred_loss = F.mse_loss(pred[:, :-1], batch[:, 1:])
            smooth_loss = torch.mean(torch.square(recon[:, 1:] - recon[:, :-1] - (batch[:, 1:] - batch[:, :-1])))
            loss = recon_loss + 0.7 * pred_loss + 0.1 * smooth_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        avg = float(np.mean(losses))
        if avg < best_loss - 1e-5:
            best_loss = avg
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.eval()
    with torch.no_grad():
        _, _, emb = model(tensor)
    emb_np = emb.numpy()
    emb_cols = [f"rwkv_emb_{i:03d}" for i in range(emb_np.shape[1])]
    embeddings = pd.DataFrame(emb_np, index=pd.Index(dates, name="Date"), columns=emb_cols)
    gmm = GaussianMixture(n_components=4, covariance_type="full", random_state=42, n_init=20)
    cluster = pd.Series(gmm.fit_predict(emb_np), index=embeddings.index, name="rwkv_cluster")
    proba = pd.DataFrame(gmm.predict_proba(emb_np), index=embeddings.index, columns=[f"rwkv_prob_{i}" for i in range(4)])
    post = post_label_rwkv_clusters(cluster, features.reindex(embeddings.index))
    drift = embedding_drift(embeddings)
    out = pd.concat([cluster, proba, post, drift], axis=1)
    out["gmm_regime"] = out["rwkv_regime"]
    out["rule_regime"] = out["rwkv_regime"]
    out["rule_confidence"] = out[[col for col in out.columns if col.startswith("rwkv_prob_")]].max(axis=1)
    return out.reindex(features.index).ffill(), embeddings, {
        "model": "RWKV-style time-mix self-supervised macro encoder",
        "input_features": inputs,
        "frequency": "monthly_last_observation" if frequency == "M" else "daily",
        "sequence_length": sequence_length,
        "embedding_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "epochs_requested": epochs,
        "epochs_run": epoch + 1,
        "best_loss": best_loss,
        "training_sequences": int(tensor.shape[0]),
    }


def torch_nn_module():
    import torch.nn as nn

    return nn.Module


class RWKVTimeMix(torch_nn_module()):
    def __init__(self, dim: int):
        super().__init__()
        import torch
        import torch.nn as nn

        self.time_mix_k = nn.Parameter(torch.rand(1, 1, dim))
        self.time_mix_v = nn.Parameter(torch.rand(1, 1, dim))
        self.time_mix_r = nn.Parameter(torch.rand(1, 1, dim))
        self.time_decay = nn.Parameter(torch.zeros(dim))
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        self.layer_norm = nn.LayerNorm(dim)
        self._torch = torch

    def forward(self, x):
        torch = self._torch
        shifted = torch.cat([x[:, :1] * 0.0, x[:, :-1]], dim=1)
        xk = x * self.time_mix_k + shifted * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + shifted * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + shifted * (1 - self.time_mix_r)
        k = self.key(xk)
        v = self.value(xv)
        r = torch.sigmoid(self.receptance(xr))
        decay = torch.sigmoid(self.time_decay).view(1, 1, -1)
        state = torch.zeros_like(v[:, 0])
        outs = []
        for t in range(x.shape[1]):
            state = decay[:, 0] * state + (1 - decay[:, 0]) * torch.tanh(k[:, t]) * v[:, t]
            outs.append(state)
        wkv = torch.stack(outs, dim=1)
        return self.layer_norm(x + self.output(r * wkv))


class RWKVChannelMix(torch_nn_module()):
    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        import torch
        import torch.nn as nn

        self.time_mix_k = nn.Parameter(torch.rand(1, 1, dim))
        self.time_mix_r = nn.Parameter(torch.rand(1, 1, dim))
        self.key = nn.Linear(dim, dim * expansion, bias=False)
        self.value = nn.Linear(dim * expansion, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.layer_norm = nn.LayerNorm(dim)
        self._torch = torch

    def forward(self, x):
        torch = self._torch
        shifted = torch.cat([x[:, :1] * 0.0, x[:, :-1]], dim=1)
        xk = x * self.time_mix_k + shifted * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + shifted * (1 - self.time_mix_r)
        k = torch.relu(self.key(xk)) ** 2
        r = torch.sigmoid(self.receptance(xr))
        return self.layer_norm(x + r * self.value(k))


class RWKVMacroAutoencoder(torch_nn_module()):
    def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int):
        super().__init__()
        import torch.nn as nn

        self.input = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([nn.Sequential(RWKVTimeMix(hidden_dim), RWKVChannelMix(hidden_dim)) for _ in range(3)])
        self.embed = nn.Linear(hidden_dim, embed_dim)
        self.decoder = nn.Linear(embed_dim, input_dim)
        self.predictor = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, input_dim))

    def forward(self, x):
        import torch

        h = self.input(x)
        for block in self.blocks:
            h = block(h)
        pooled = 0.65 * h[:, -1] + 0.35 * h.mean(dim=1)
        emb = torch.tanh(self.embed(pooled))
        recon_step = self.decoder(emb)
        recon = recon_step[:, None, :].repeat(1, x.shape[1], 1)
        pred = self.predictor(torch.tanh(self.embed(h)))
        return recon, pred, emb


def make_sequences(z: pd.DataFrame, sequence_length: int) -> tuple[np.ndarray, list[pd.Timestamp]]:
    arr = z.to_numpy(dtype=np.float32)
    seqs = []
    dates = []
    for end in range(sequence_length, len(z) + 1):
        seqs.append(arr[end - sequence_length : end])
        dates.append(z.index[end - 1])
    return np.stack(seqs), dates


def post_label_rwkv_clusters(cluster: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
    labels = {}
    scores = {}
    for cid, idx in cluster.groupby(cluster).groups.items():
        f = features.loc[idx]
        vals = {
            "Risk-On Growth": feature_mean(f, ["NASDAQ100_riskon_score", "SOX_riskon_score", "VIX_riskon_score", "HY_OAS_riskon_score", "DXY_riskon_score"]),
            "Risk-On Cyclical": feature_mean(f, ["COPPER_riskon_score", "WTI_riskon_score", "HANGSENG_TECH_riskon_score", "CSI300_riskon_score", "US10Y_2Y_riskon_score"]),
            "Defensive / Rate-Cut": feature_mean(f, ["US10Y_riskon_score", "US10Y_REAL_riskon_score", "GOLD_riskon_score"]) - feature_mean(f, ["SP500_riskon_score", "COPPER_riskon_score"]),
            "Risk-Off / Cash": -feature_mean(f, ["VIX_riskon_score", "HY_OAS_riskon_score", "DXY_riskon_score", "USDKRW_riskon_score", "SOX_riskon_score"]),
        }
        label = max(vals, key=vals.get)
        labels[int(cid)] = label
        scores[int(cid)] = float(vals[label])
    return pd.DataFrame({"rwkv_regime": cluster.map(labels), "rwkv_regime_score": cluster.map(scores)}, index=cluster.index)


def feature_mean(frame: pd.DataFrame, cols: list[str]) -> float:
    have = [col for col in cols if col in frame]
    return 0.0 if not have else float(frame[have].mean(axis=1).mean())


def embedding_drift(embeddings: pd.DataFrame) -> pd.DataFrame:
    diff = embeddings.diff()
    drift_20 = np.sqrt((diff**2).sum(axis=1)).rolling(20).mean()
    drift_z = (drift_20 - drift_20.rolling(252).mean()) / drift_20.rolling(252).std()
    return pd.DataFrame({"embedding_drift_20d": drift_20, "embedding_drift_z252": drift_z}, index=embeddings.index)


def macro_driver_selection_report(features: pd.DataFrame, histories: dict[str, pd.DataFrame], target_symbol: str = "069500.KS") -> pd.DataFrame:
    target_hist = histories.get(target_symbol)
    if target_hist is None or target_hist.empty or "Close" not in target_hist:
        return pd.DataFrame()
    target = clean_series(target_hist["Close"]).resample("ME").last().pct_change(12).rename("target_yoy_return")
    monthly = features[[col for col in features.columns if col.endswith("_riskon_score")]].copy()
    monthly = monthly.groupby(monthly.index.to_period("M")).tail(1)
    monthly.index = monthly.index.to_period("M").to_timestamp("M")
    rows = []
    for col in monthly.columns:
        pair = pd.concat([target, monthly[col].rename("driver")], axis=1).dropna()
        if pair.shape[0] < 48:
            continue
        corr_full = pair["target_yoy_return"].corr(pair["driver"])
        corr_recent_5y = pair.tail(60)["target_yoy_return"].corr(pair.tail(60)["driver"]) if pair.shape[0] >= 60 else np.nan
        pvalue = granger_min_pvalue(pair[["target_yoy_return", "driver"]], maxlag=6)
        rows.append(
            {
                "driver_feature": col,
                "driver": col.replace("_riskon_score", ""),
                "target": target_symbol,
                "observations": int(pair.shape[0]),
                "corr_full": corr_full,
                "corr_recent_5y": corr_recent_5y,
                "granger_min_pvalue_lag1_6": pvalue,
                "selected_corr_and_granger": bool(abs(corr_recent_5y if pd.notna(corr_recent_5y) else corr_full) >= 0.20 and pvalue <= 0.10),
            }
        )
    return pd.DataFrame(rows).sort_values(["selected_corr_and_granger", "granger_min_pvalue_lag1_6", "corr_recent_5y"], ascending=[False, True, False])


def granger_min_pvalue(pair: pd.DataFrame, maxlag: int) -> float:
    try:
        from statsmodels.tsa.stattools import grangercausalitytests

        data = pair[["target_yoy_return", "driver"]].astype(float)
        result = grangercausalitytests(data, maxlag=maxlag, verbose=False)
        return float(min(result[lag][0]["ssr_ftest"][1] for lag in result))
    except Exception:
        return np.nan


def build_lppl_training_table(
    histories: dict[str, pd.DataFrame],
    window: int,
    step: int,
    population: int,
    generations: int,
    fits_per_window: int,
    max_training_windows: int | None,
    refresh: bool,
) -> pd.DataFrame:
    rows = []
    for symbol, hist in histories.items():
        close = clean_series(hist["Close"]) if "Close" in hist else pd.Series(dtype=float)
        if close.shape[0] < window + 80:
            continue
        end_points = list(range(window, close.shape[0] - 60, step))
        if max_training_windows is not None:
            end_points = end_points[-max_training_windows:]
        print(f"LPPL training {symbol}: {len(end_points)} windows x {fits_per_window} fits", flush=True)
        for end in end_points:
            date = close.index[end - 1]
            cache_path = LPPL_CACHE / f"train_{safe_lppl_name(symbol)}_{date.date()}_{window}_{population}_{generations}_{fits_per_window}.json"
            if cache_path.exists() and not refresh:
                window_rows = json.loads(cache_path.read_text(encoding="utf-8"))
            else:
                fits = fit_lppl_ensemble(close.iloc[end - window : end], population, generations, fits_per_window, seed=stable_seed(symbol, str(date.date())))
                window_rows = [lppl_fit_to_row(symbol, date, fit, actual_crash_label(close.iloc[: end + 61], end - 1, fit.tc, window)) for fit in fits]
                cache_path.write_text(json.dumps(window_rows, ensure_ascii=False), encoding="utf-8")
            rows.extend(window_rows)
    return pd.DataFrame(rows)


def current_lppl_table(
    histories: dict[str, pd.DataFrame],
    window: int,
    population: int,
    generations: int,
    fits_per_window: int,
    reliability_model: Any,
    refresh: bool,
) -> pd.DataFrame:
    rows = []
    for symbol, hist in histories.items():
        close = clean_series(hist["Close"]) if "Close" in hist else pd.Series(dtype=float)
        if close.shape[0] < window:
            continue
        date = close.index[-1]
        print(f"Current LPPL {symbol}: {fits_per_window} fits", flush=True)
        cache_path = LPPL_CACHE / f"current_{safe_lppl_name(symbol)}_{date.date()}_{window}_{population}_{generations}_{fits_per_window}.json"
        if cache_path.exists() and not refresh:
            fit_rows = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            fits = fit_lppl_ensemble(close.tail(window), population, generations, fits_per_window, seed=stable_seed(symbol, "current"))
            fit_rows = [lppl_fit_to_row(symbol, date, fit, label=None) for fit in fits]
            cache_path.write_text(json.dumps(fit_rows, ensure_ascii=False), encoding="utf-8")
        scored = []
        for row in fit_rows:
            reliability = predict_lppl_reliability(reliability_model, row)
            dtc = float(row["lppl_dtc"])
            row = dict(row)
            row["lppl_reliability"] = reliability
            row["lppl_dtcai"] = float(np.clip(dtc * reliability, 0, 1))
            scored.append(row)
        scored_frame = pd.DataFrame(scored)
        risk_value = float(scored_frame["lppl_dtcai"].quantile(0.95))
        best = scored_frame.sort_values(["lppl_dtcai", "lppl_reliability"], ascending=False).iloc[0].to_dict()
        best.update(
            {
                "lppl_dtcai": risk_value,
                "lppl_dtcai_max": float(scored_frame["lppl_dtcai"].max()),
                "lppl_dtcai_p95": risk_value,
                "lppl_dtcai_median": float(scored_frame["lppl_dtcai"].median()),
                "lppl_reliability_mean": float(scored_frame["lppl_reliability"].mean()),
                "lppl_reliability_p95": float(scored_frame["lppl_reliability"].quantile(0.95)),
                "lppl_ensemble_size": int(scored_frame.shape[0]),
                "lppl_risk_label": dtcai_label(risk_value),
            }
        )
        rows.append(best)
    return pd.DataFrame(rows)


def fit_lppl_ensemble(close: pd.Series, population: int, generations: int, fits_per_window: int, seed: int) -> list[LPPLFit]:
    return fit_lppl_ga_archive(close, population, generations, fits_per_window, seed)


def fit_lppl_ga(close: pd.Series, population: int, generations: int, seed: int) -> LPPLFit:
    return fit_lppl_ga_archive(close, population, generations, 1, seed)[0]


def fit_lppl_ga_archive(close: pd.Series, population: int, generations: int, archive_size: int, seed: int) -> list[LPPLFit]:
    rng = np.random.default_rng(seed)
    y = np.log(close.dropna().to_numpy(dtype=float))
    t = np.arange(len(y), dtype=float)
    n = len(y)
    lower = np.array([float(n + 1), 0.10, 6.0], dtype=float)
    upper = np.array([float(n * 1.8), 1.00, 13.0], dtype=float)
    pop = rng.uniform(lower, upper, size=(population, 3))
    pop[: min(20, population)] = gyration_initial_population(y, population=min(20, population), rng=rng, lower=lower, upper=upper)
    fitness, linear = lppl_population_rmse(t, y, pop)
    archive: list[tuple[float, np.ndarray, np.ndarray]] = []
    archive_take = max(5, min(population, archive_size // 10 if archive_size >= 50 else archive_size))
    for _ in range(generations):
        order = np.argsort(fitness)
        for idx in order[:archive_take]:
            archive.append((float(fitness[idx]), pop[idx].copy(), linear[idx].copy()))
        elite_count = max(10, int(population * 0.05))
        elites = pop[order[:elite_count]]
        weights = 1.0 / (fitness + 1e-8)
        weights = weights / weights.sum()
        parent_idx = rng.choice(np.arange(population), size=population * 2, replace=True, p=weights)
        parents = pop[parent_idx].reshape(population, 2, 3)
        children = parents.mean(axis=1)
        mutation_count = max(1, int(population * 0.20))
        mutation_idx = rng.choice(population, size=mutation_count, replace=False)
        scale = (upper - lower) * np.array([0.035, 0.04, 0.04])
        children[mutation_idx] += rng.normal(0.0, scale, size=(mutation_count, 3))
        children = np.clip(children, lower, upper)
        pop = np.vstack([elites, children])[:population]
        fitness, linear = lppl_population_rmse(t, y, pop)
    order = np.argsort(fitness)
    for idx in order[: max(archive_take, archive_size)]:
        archive.append((float(fitness[idx]), pop[idx].copy(), linear[idx].copy()))
    return archive_to_lppl_fits(archive, archive_size, n)


def archive_to_lppl_fits(archive: list[tuple[float, np.ndarray, np.ndarray]], archive_size: int, n: int) -> list[LPPLFit]:
    fits = []
    seen = set()
    for rmse, params, coefs in sorted(archive, key=lambda item: item[0]):
        tc, beta, omega = params
        a, b, bc, bs = coefs
        if not np.isfinite([rmse, tc, beta, omega, a, b, bc, bs]).all():
            continue
        key = (round(float(tc), 2), round(float(beta), 4), round(float(omega), 4), round(float(rmse), 6))
        if key in seen:
            continue
        seen.add(key)
        fits.append(lppl_fit_from_params(float(rmse), float(tc), float(beta), float(omega), float(a), float(b), float(bc), float(bs), n))
        if len(fits) >= archive_size:
            break
    return fits


def lppl_fit_from_params(rmse: float, tc: float, beta: float, omega: float, a: float, b: float, bc: float, bs: float, n: int) -> LPPLFit:
    phi = math.atan2(-bs, bc)
    amp = math.sqrt(bc * bc + bs * bs)
    c_signed = float(np.clip(amp / b if abs(b) > 1e-8 else 0.0, -1, 1))
    dtc = float(np.clip((n - 1) / (tc - 0), 0, 1))
    return LPPLFit(float(a), float(b), c_signed, float(tc), float(phi % (2 * math.pi)), float(omega), float(beta), float(rmse), dtc, np.nan, np.nan, "")


def lppl_population_rmse(t: np.ndarray, y: np.ndarray, pop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rmses = np.empty(pop.shape[0], dtype=float)
    linear = np.empty((pop.shape[0], 4), dtype=float)
    for i, (tc, beta, omega) in enumerate(pop):
        tau = np.maximum(tc - t, 1e-6)
        f = tau**beta
        cos_term = f * np.cos(omega * np.log(tau))
        sin_term = f * np.sin(omega * np.log(tau))
        x = np.column_stack([np.ones_like(t), f, cos_term, sin_term])
        try:
            coef, *_ = np.linalg.lstsq(x, y, rcond=None)
            pred = x @ coef
            penalty = 0.0
            if coef[1] >= 0:
                penalty += abs(coef[1]) + 0.05
            rmses[i] = float(np.sqrt(np.mean((y - pred) ** 2)) + penalty)
            linear[i] = coef
        except np.linalg.LinAlgError:
            rmses[i] = 1e9
            linear[i] = np.nan
    return rmses, linear


def gyration_initial_population(y: np.ndarray, population: int, rng: np.random.Generator, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    peaks = local_peaks(y)
    out = rng.uniform(lower, upper, size=(population, 3))
    if len(peaks) < 3:
        return out
    triples = []
    for i in range(len(peaks) - 2):
        a, b, c = peaks[i], peaks[i + 1], peaks[i + 2]
        if b - a > 0 and c - b > 0:
            rho = (b - a) / (c - b)
            if rho > 1.02:
                tc = (rho * c - b) / (rho - 1)
                omega = 2 * math.pi / max(math.log(rho), 1e-6)
                if tc > len(y) and 6 <= omega <= 13:
                    triples.append((tc, omega))
    if not triples:
        return out
    weights = np.linspace(0.2, 1.0, len(triples))
    weights = weights / weights.sum()
    for row in range(population):
        tc, omega = triples[rng.choice(len(triples), p=weights)]
        out[row, 0] = np.clip(tc + rng.normal(0, len(y) * 0.05), lower[0], upper[0])
        out[row, 1] = np.clip(1.0 + rng.normal(0, 0.06), lower[1], upper[1])
        out[row, 2] = np.clip(omega + rng.normal(0, 0.35), lower[2], upper[2])
    return out


def local_peaks(y: np.ndarray) -> list[int]:
    return [i for i in range(2, len(y) - 2) if y[i] >= y[i - 1] and y[i] >= y[i + 1] and y[i] >= y[i - 2] and y[i] >= y[i + 2]]


def actual_crash_label(close: pd.Series, end_idx: int, tc: float, window: int) -> int:
    future = close.iloc[end_idx : min(end_idx + 61, len(close))]
    if future.shape[0] < 30:
        return 0
    peak_offset = int(np.argmax(future.to_numpy()))
    peak_price = float(future.iloc[peak_offset])
    after_peak = future.iloc[peak_offset : min(peak_offset + 61, future.shape[0])]
    crash = after_peak.min() / peak_price - 1.0 <= -0.25
    predicted_close = abs((tc - (window - 1)) - peak_offset) <= 10
    return int(crash and predicted_close)


def fit_lppl_reliability_model(training: pd.DataFrame) -> tuple[Any, dict[str, Any]]:
    if training.empty or training["label"].sum() < 3 or training.shape[0] < LPPL_RELIABILITY_MIN_SAMPLES:
        return None, {"model": "fallback_logistic_prior", "samples": int(training.shape[0]), "positives": int(training["label"].sum()) if "label" in training else 0}
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    cols = lppl_feature_columns()
    data = training.dropna(subset=cols + ["label"]).copy()
    if data["label"].sum() < 3 or data.shape[0] < LPPL_RELIABILITY_MIN_SAMPLES:
        return None, {"model": "fallback_logistic_prior", "samples": int(data.shape[0]), "positives": int(data["label"].sum())}
    scaler = StandardScaler()
    x = scaler.fit_transform(data[cols].astype(float))
    y = data["label"].astype(int).to_numpy()
    stratify = y if len(np.unique(y)) == 2 and min(np.bincount(y.astype(int))) >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, stratify=stratify)
    x_bal, y_bal = simple_smote(x_train, y_train, random_state=42)
    candidates = {
        "ANN": MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-4, learning_rate_init=1e-3, max_iter=700, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=400, max_depth=8, min_samples_leaf=3, class_weight="balanced_subsample", random_state=42, n_jobs=-1),
        "Logistic": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    }
    comparison = []
    fitted: dict[str, Any] = {}
    for name, model in candidates.items():
        model.fit(x_bal, y_bal)
        prob = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(x_test)
        pred = prob >= 0.5
        recall = recall_score(y_test, pred, zero_division=0)
        precision = precision_score(y_test, pred, zero_division=0)
        auc = roc_auc_score(y_test, prob) if len(np.unique(y_test)) == 2 else np.nan
        comparison.append({"model": name, "recall": float(recall), "precision": float(precision), "auc": float(auc) if pd.notna(auc) else np.nan})
        fitted[name] = model
    comparison_frame = pd.DataFrame(comparison).sort_values(["recall", "precision", "auc"], ascending=False)
    selected_name = str(comparison_frame.iloc[0]["model"])
    return {"model": fitted[selected_name], "scaler": scaler, "columns": cols, "type": selected_name}, {
        "model": selected_name,
        "samples": int(data.shape[0]),
        "positives": int(data["label"].sum()),
        "comparison": comparison,
        "smote_train_samples": int(x_bal.shape[0]),
    }


def predict_lppl_reliability(model_bundle: Any, row: dict[str, Any]) -> float:
    if model_bundle is None:
        rmse = float(row.get("lppl_rmse", 1.0))
        c = abs(float(row.get("lppl_C", 0.0)))
        beta = float(row.get("lppl_beta", 0.5))
        omega = float(row.get("lppl_omega", 8.0))
        shape = 0.35 + 0.25 * min(c, 1.0) + 0.20 * (1 - abs(beta - 0.5)) + 0.20 * (1 - min(abs(omega - 9.5) / 5.0, 1.0))
        return float(np.clip(shape * math.exp(-8 * max(rmse - 0.015, 0)), 0, 1))
    cols = model_bundle["columns"]
    x = np.array([[float(row[col]) for col in cols]], dtype=float)
    xz = model_bundle["scaler"].transform(x)
    model = model_bundle["model"]
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(xz)[:, 1][0])
    raw = float(model.decision_function(xz)[0])
    return float(1.0 / (1.0 + math.exp(-raw)))


def lppl_feature_columns() -> list[str]:
    return ["lppl_A", "lppl_B", "lppl_C", "lppl_tc", "lppl_phi", "lppl_omega", "lppl_beta"]


def simple_smote(x: np.ndarray, y: np.ndarray, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    counts = np.bincount(y.astype(int), minlength=2)
    if counts.min() == 0 or counts[1] >= counts[0]:
        return x, y
    minority = x[y == 1]
    needed = counts[0] - counts[1]
    if minority.shape[0] < 2:
        return x, y
    synth = []
    for _ in range(needed):
        i, j = rng.choice(minority.shape[0], size=2, replace=False)
        lam = rng.random()
        synth.append(minority[i] + lam * (minority[j] - minority[i]))
    xs = np.vstack([x, np.asarray(synth)])
    ys = np.concatenate([y, np.ones(len(synth), dtype=int)])
    return xs, ys


def score_lppl_training_with_model(training: pd.DataFrame, model_bundle: Any) -> pd.DataFrame:
    if training.empty:
        return training
    out = training.copy()
    reliabilities = []
    for row in out.to_dict("records"):
        reliabilities.append(predict_lppl_reliability(model_bundle, row))
    out["lppl_reliability"] = reliabilities
    out["lppl_dtcai"] = (pd.to_numeric(out["lppl_dtc"], errors="coerce") * out["lppl_reliability"]).clip(0, 1)
    out["lppl_risk_label"] = out["lppl_dtcai"].map(dtcaiai_label_alias)
    return out


def dtcaiai_label_alias(value: float) -> str:
    return dtcai_label(float(value))


def validate_lppl_training_signals(training_scored: pd.DataFrame) -> pd.DataFrame:
    if training_scored.empty or "label" not in training_scored:
        return pd.DataFrame()
    rows = []
    thresholds = [0.3, 0.6]
    for symbol, frame in training_scored.groupby("symbol"):
        labels = frame["label"].astype(int)
        for threshold in thresholds:
            signal = pd.to_numeric(frame["lppl_dtcai"], errors="coerce").fillna(0).ge(threshold)
            tp = int((signal & labels.eq(1)).sum())
            fp = int((signal & labels.eq(0)).sum())
            fn = int((~signal & labels.eq(1)).sum())
            tn = int((~signal & labels.eq(0)).sum())
            rows.append(
                {
                    "symbol": symbol,
                    "threshold": threshold,
                    "samples": int(frame.shape[0]),
                    "positives": int(labels.sum()),
                    "signals": int(signal.sum()),
                    "recall": tp / max(tp + fn, 1),
                    "precision": tp / max(tp + fp, 1),
                    "false_alarm_rate": fp / max(fp + tn, 1),
                }
            )
    all_frame = training_scored
    labels = all_frame["label"].astype(int)
    for threshold in thresholds:
        signal = pd.to_numeric(all_frame["lppl_dtcai"], errors="coerce").fillna(0).ge(threshold)
        tp = int((signal & labels.eq(1)).sum())
        fp = int((signal & labels.eq(0)).sum())
        fn = int((~signal & labels.eq(1)).sum())
        tn = int((~signal & labels.eq(0)).sum())
        rows.append(
            {
                "symbol": "__ALL__",
                "threshold": threshold,
                "samples": int(all_frame.shape[0]),
                "positives": int(labels.sum()),
                "signals": int(signal.sum()),
                "recall": tp / max(tp + fn, 1),
                "precision": tp / max(tp + fp, 1),
                "false_alarm_rate": fp / max(fp + tn, 1),
            }
        )
    return pd.DataFrame(rows).sort_values(["symbol", "threshold"])


def lppl_fit_to_row(symbol: str, date: pd.Timestamp, fit: LPPLFit, label: int | None) -> dict[str, Any]:
    row = {
        "symbol": symbol,
        "asof": date.date().isoformat(),
        "lppl_A": fit.A,
        "lppl_B": fit.B,
        "lppl_C": fit.C,
        "lppl_tc": fit.tc,
        "lppl_phi": fit.phi,
        "lppl_omega": fit.omega,
        "lppl_beta": fit.beta,
        "lppl_rmse": fit.rmse,
        "lppl_dtc": fit.dtc,
    }
    if label is not None:
        row["label"] = int(label)
    return row


def merge_lppl_scores(base: pd.DataFrame, lppl: pd.DataFrame) -> pd.DataFrame:
    if lppl.empty:
        return base
    lppl_cols = [
        "symbol",
        "lppl_dtc",
        "lppl_reliability",
        "lppl_dtcai",
        "lppl_dtcai_max",
        "lppl_dtcai_p95",
        "lppl_dtcai_median",
        "lppl_reliability_mean",
        "lppl_reliability_p95",
        "lppl_ensemble_size",
        "lppl_risk_label",
        "lppl_tc",
        "lppl_beta",
        "lppl_omega",
        "lppl_rmse",
    ]
    out = base.merge(lppl[[col for col in lppl_cols if col in lppl]], on="symbol", how="left")
    risk_penalty = pd.to_numeric(out["lppl_dtcai"], errors="coerce").fillna(0) * 28.0
    out["score_0_100_before_lppl"] = out["score_0_100"]
    out["score_0_100"] = (out["score_0_100"] - risk_penalty).clip(0, 100).round(2)
    out["bubble_score_0_100"] = (pd.to_numeric(out["lppl_dtcai"], errors="coerce").fillna(0) * 100).round(2)
    out["rank"] = out["score_0_100"].rank(ascending=False, method="first").astype(int)
    return out.sort_values(["score_0_100", "upside_prob_4w"], ascending=[False, False]).reset_index(drop=True)


def dtcai_label(value: float) -> str:
    if value > 0.6:
        return "crash-alert"
    if value >= 0.3:
        return "caution"
    return "stable"


def safe_lppl_name(symbol: str) -> str:
    return symbol.replace("^", "_idx_").replace("/", "_").replace("=", "_").replace(".", "_").replace("-", "_")


def stable_seed(*parts: str) -> int:
    text = "|".join(parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def write_rwkv_lppl_report(
    scores: pd.DataFrame,
    driver_state: pd.DataFrame,
    regime_frame: pd.DataFrame,
    rwkv_meta: dict[str, Any],
    reliability_meta: dict[str, Any],
    lppl_validation: pd.DataFrame,
    driver_selection: pd.DataFrame,
    path: Path,
) -> None:
    latest = regime_frame.dropna(how="all").iloc[-1]
    top = scores.head(20)
    bubble = scores.sort_values("bubble_score_0_100", ascending=False).head(15)
    lines = [
        "# RWKV + LPPL/DTCAI Asset Screener",
        "",
        f"- Current RWKV regime: {latest.get('rwkv_regime')}",
        f"- RWKV confidence: {safe_float(latest.get('rule_confidence'))}",
        f"- RWKV training: {json.dumps(rwkv_meta, ensure_ascii=False)}",
        f"- LPPL reliability model: {json.dumps(reliability_meta, ensure_ascii=False)}",
        "",
        "## Top Assets After LPPL Risk Adjustment",
        top[["rank", "symbol", "name", "group", "score_0_100", "score_0_100_before_lppl", "upside_prob_4w", "bubble_score_0_100", "lppl_risk_label", "lppl_reliability"]].to_markdown(index=False),
        "",
        "## Highest Bubble / Crash-Proximity Scores",
        bubble[["symbol", "name", "group", "bubble_score_0_100", "lppl_dtc", "lppl_reliability", "lppl_risk_label", "lppl_tc", "lppl_beta", "lppl_omega", "lppl_rmse"]].to_markdown(index=False),
        "",
        "## Method",
        "- RWKV-style time-mix encoder learns macro-sequence embeddings through reconstruction and next-step self-supervised losses.",
        "- GMM clusters the RWKV embeddings, then clusters are post-labeled into economic regimes.",
        "- LPPL uses log price, GA optimization, and the seven parameters A/B/C/tc/phi/omega/beta.",
        "- DTC=(t2-t1)/(tc-t1). Reliability P is selected from ANN/RF/Logistic by test recall after SMOTE balancing. DTCAI=DTC*P.",
        "- DTCAI labels follow the article convention: >0.6 crash-alert, 0.3-0.6 caution, <0.3 stable.",
    ]
    if not lppl_validation.empty:
        lines.extend(
            [
                "",
                "## LPPL Signal Validation",
                lppl_validation[lppl_validation["symbol"].eq("__ALL__")].to_markdown(index=False),
            ]
        )
    if not driver_selection.empty:
        lines.extend(
            [
                "",
                "## Selected Macro Drivers",
                driver_selection.head(20).to_markdown(index=False),
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
