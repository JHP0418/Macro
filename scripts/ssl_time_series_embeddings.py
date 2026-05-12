from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[1]

ETF_FEATURES = [
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

SAFE_FEATURES = [
    "score_0_100",
    "upside_prob_1w",
    "upside_prob_4w",
    "technical_score",
    "driver_fit_score",
    "beta_fit_score",
    "calibrated_prob_1w",
    "calibrated_prob_4w",
    "institutional_score_0_100",
]

EXCLUDE_MACRO = {"dominant_component", "sentinel_state"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Self-supervised masked patch embeddings for macro, ETF, or safe-asset panels.")
    p.add_argument("--mode", choices=["macro", "etf", "safe"], required=True)
    p.add_argument("--input", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--date-col", default=None)
    p.add_argument("--entity-col", default=None)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--patch-len", type=int, default=8)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--embedding-dim", type=int, default=32)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=192)
    p.add_argument("--max-train-samples", type=int, default=30000)
    p.add_argument("--train-end", default="2024-12-31")
    p.add_argument("--clusters", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


class RobustPanelScaler:
    def __init__(self) -> None:
        self.median_: pd.Series | None = None
        self.iqr_: pd.Series | None = None

    def fit(self, frame: pd.DataFrame) -> "RobustPanelScaler":
        self.median_ = frame.median(axis=0)
        q75 = frame.quantile(0.75)
        q25 = frame.quantile(0.25)
        iqr = (q75 - q25).replace(0, np.nan)
        self.iqr_ = iqr.fillna(frame.std(axis=0).replace(0, np.nan)).fillna(1.0)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.median_ is None or self.iqr_ is None:
            raise RuntimeError("Scaler is not fitted.")
        out = (frame - self.median_) / self.iqr_
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-8, 8)


class MaskedPatchEncoder(torch.nn.Module):
    def __init__(
        self,
        n_features: int,
        window: int,
        patch_len: int,
        d_model: int,
        embedding_dim: int,
        layers: int,
        heads: int,
    ) -> None:
        super().__init__()
        if window % patch_len != 0:
            raise ValueError("window must be divisible by patch_len")
        self.n_features = n_features
        self.window = window
        self.patch_len = patch_len
        self.n_patches = window // patch_len
        self.patch_dim = n_features * patch_len
        self.patch_proj = torch.nn.Linear(self.patch_dim, d_model)
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = torch.nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 3,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.to_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model),
            torch.nn.Linear(d_model, embedding_dim),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model),
            torch.nn.Linear(d_model, self.patch_dim),
        )

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        return x.reshape(b, self.n_patches, self.patch_len, c).reshape(b, self.n_patches, self.patch_dim)

    def forward(self, x: torch.Tensor, mask_ratio: float = 0.35) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        patches = self.patchify(x)
        h = self.patch_proj(patches)
        if self.training and mask_ratio > 0:
            mask = torch.rand(h.shape[:2], device=h.device) < mask_ratio
            if not mask.any(dim=1).all():
                missing = ~mask.any(dim=1)
                mask[missing, torch.randint(0, h.shape[1], (int(missing.sum()),), device=h.device)] = True
            h = torch.where(mask.unsqueeze(-1), self.mask_token.expand_as(h), h)
        else:
            mask = torch.zeros(h.shape[:2], dtype=torch.bool, device=h.device)
        h = self.encoder(h + self.pos)
        recon = self.decoder(h)
        emb = self.to_embedding(h.mean(dim=1))
        return recon, patches, mask, emb


def default_paths(args: argparse.Namespace) -> argparse.Namespace:
    if args.mode == "macro":
        args.input = args.input or str(ROOT / "outputs" / "daily_risk_off_sentinel_latest" / "tables" / "daily_sentinel_history.csv")
        args.output_dir = args.output_dir or str(ROOT / "outputs" / "ssl_market_embeddings_latest")
        args.date_col = args.date_col or "Date"
        args.entity_col = args.entity_col or ""
    elif args.mode == "etf":
        args.input = args.input or str(ROOT / "outputs" / "etf_leadership_static_holdings_repaired_v4base" / "rule_scores.csv")
        args.output_dir = args.output_dir or str(ROOT / "outputs" / "ssl_etf_embeddings_latest")
        args.date_col = args.date_col or "date"
        args.entity_col = args.entity_col or "etf_ticker"
    else:
        args.input = args.input or str(ROOT / "outputs" / "weekly_screening_rank_backtest_latest" / "tables" / "weekly_calibrated_rank_panel.csv")
        args.output_dir = args.output_dir or str(ROOT / "outputs" / "ssl_safe_asset_embeddings_latest")
        args.date_col = args.date_col or "date"
        args.entity_col = args.entity_col or "symbol"
    return args


def select_features(frame: pd.DataFrame, mode: str, date_col: str, entity_col: str | None) -> list[str]:
    if mode == "etf":
        return [c for c in ETF_FEATURES if c in frame.columns]
    if mode == "safe":
        return [c for c in SAFE_FEATURES if c in frame.columns]
    cols = []
    for col in frame.columns:
        if col == date_col or col == entity_col or col in EXCLUDE_MACRO:
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            valid = pd.to_numeric(frame[col], errors="coerce").notna().mean()
            if valid >= 0.35:
                cols.append(col)
    return cols


def prepare_frame(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str], list[str]]:
    frame = pd.read_csv(args.input, parse_dates=[args.date_col])
    if args.mode == "macro":
        frame["__entity__"] = "macro"
        entity_col = "__entity__"
    else:
        entity_col = args.entity_col
    features = select_features(frame, args.mode, args.date_col, entity_col)
    keep = [args.date_col, entity_col, *features]
    targets = [c for c in ["forward_5D_return", "forward_20D_return", "forward_5D_excess", "forward_20D_excess", "realized_return_1w", "realized_return_4w"] if c in frame.columns]
    keep += targets
    if args.mode == "safe":
        keep += [c for c in ["group", "basket", "name"] if c in frame.columns]
    out = frame[keep].copy()
    out = out.rename(columns={args.date_col: "date", entity_col: "entity"})
    out["date"] = pd.to_datetime(out["date"])
    for col in features + targets:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.sort_values(["entity", "date"]).reset_index(drop=True), features, targets


def make_training_windows(frame: pd.DataFrame, features: list[str], scaler: RobustPanelScaler, window: int, max_samples: int, train_end: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples = []
    for _, part in frame[frame["date"].le(pd.Timestamp(train_end))].groupby("entity", sort=False):
        values = scaler.transform(part[features].ffill().bfill()).to_numpy(dtype=np.float32)
        if values.shape[0] < window:
            continue
        idx = np.arange(window - 1, values.shape[0])
        if idx.size > 0:
            take = idx
            samples.extend(values[i - window + 1 : i + 1] for i in take)
    if not samples:
        raise RuntimeError("No SSL training windows were generated.")
    if len(samples) > max_samples:
        choice = rng.choice(len(samples), size=max_samples, replace=False)
        samples = [samples[i] for i in choice]
    return np.stack(samples).astype(np.float32)


def train_model(x_train: np.ndarray, args: argparse.Namespace, n_features: int) -> MaskedPatchEncoder:
    torch.manual_seed(args.seed)
    model = MaskedPatchEncoder(n_features, args.window, args.patch_len, args.d_model, args.embedding_dim, args.layers, args.heads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    tensor = torch.from_numpy(x_train)
    n = tensor.shape[0]
    model.train()
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(n)
        losses = []
        for start in range(0, n, args.batch_size):
            batch = tensor[perm[start : start + args.batch_size]].to(device)
            recon, patches, mask, _ = model(batch, mask_ratio=0.35)
            if mask.any():
                loss = ((recon - patches) ** 2)[mask].mean()
            else:
                loss = ((recon - patches) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"[ssl] epoch={epoch}/{args.epochs} loss={np.mean(losses):.5f}", flush=True)
    return model.cpu().eval()


@torch.no_grad()
def embed_panel(frame: pd.DataFrame, features: list[str], scaler: RobustPanelScaler, model: MaskedPatchEncoder, window: int, batch_size: int) -> pd.DataFrame:
    rows = []
    buffers = []
    metas = []
    for entity, part in frame.groupby("entity", sort=False):
        part = part.sort_values("date").reset_index(drop=True)
        values = scaler.transform(part[features].ffill().bfill()).to_numpy(dtype=np.float32)
        if values.shape[0] < window:
            continue
        for i in range(window - 1, values.shape[0]):
            buffers.append(values[i - window + 1 : i + 1])
            metas.append((part.loc[i, "date"], entity, i))
            if len(buffers) >= batch_size:
                rows.extend(emit_embeddings(model, buffers, metas))
                buffers, metas = [], []
    if buffers:
        rows.extend(emit_embeddings(model, buffers, metas))
    return pd.DataFrame(rows)


def emit_embeddings(model: MaskedPatchEncoder, buffers: list[np.ndarray], metas: list[tuple[pd.Timestamp, str, int]]) -> list[dict]:
    x = torch.from_numpy(np.stack(buffers).astype(np.float32))
    _, _, _, emb = model(x, mask_ratio=0.0)
    arr = emb.numpy()
    rows = []
    for j, (date, entity, idx) in enumerate(metas):
        rec = {"date": date, "entity": entity, "row_position": idx}
        for k in range(arr.shape[1]):
            rec[f"ssl_emb_{k:02d}"] = float(arr[j, k])
        rows.append(rec)
    return rows


def add_vq_and_confidence(emb: pd.DataFrame, train_end: str, clusters: int, seed: int) -> tuple[pd.DataFrame, dict]:
    out = emb.copy()
    emb_cols = [c for c in out.columns if c.startswith("ssl_emb_")]
    x = out[emb_cols].to_numpy(dtype=float)
    train_mask = pd.to_datetime(out["date"]).le(pd.Timestamp(train_end))
    x_train = x[train_mask]
    n_clusters = max(4, min(clusters, max(4, x_train.shape[0] // 20)))
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=2048, random_state=seed, n_init="auto")
    kmeans.fit(x_train)
    out["ssl_vq_state"] = kmeans.predict(x).astype(int)
    dist = np.linalg.norm(x - kmeans.cluster_centers_[out["ssl_vq_state"].to_numpy()], axis=1)
    out["ssl_vq_distance"] = dist

    pca_dim = max(2, min(16, x_train.shape[1], x_train.shape[0] - 1))
    pca = PCA(n_components=pca_dim, random_state=seed)
    z_train = pca.fit_transform(x_train)
    z = pca.transform(x)
    cov = LedoitWolf().fit(z_train)
    loglik_train = gaussian_loglik(z_train, cov.location_, cov.covariance_)
    loglik = gaussian_loglik(z, cov.location_, cov.covariance_)
    out["ssl_flow_loglik"] = loglik
    out["ssl_flow_nll"] = -loglik
    out["ssl_flow_confidence"] = percentile_against_train(loglik, loglik_train) * 100.0
    meta = {
        "n_clusters": int(n_clusters),
        "pca_dim": int(pca_dim),
        "train_rows": int(x_train.shape[0]),
        "embedding_dim": int(x.shape[1]),
    }
    return out, meta


def gaussian_loglik(values: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    dim = values.shape[1]
    cov = np.asarray(cov, dtype=float)
    jitter = 1e-6 * np.eye(dim)
    inv = np.linalg.pinv(cov + jitter)
    sign, logdet = np.linalg.slogdet(cov + jitter)
    if sign <= 0:
        logdet = np.log(np.linalg.det(cov + 1e-4 * np.eye(dim)))
    diff = values - mean
    quad = np.einsum("ij,jk,ik->i", diff, inv, diff)
    return -0.5 * (dim * np.log(2 * np.pi) + logdet + quad)


def percentile_against_train(values: np.ndarray, train_values: np.ndarray) -> np.ndarray:
    sorted_train = np.sort(train_values)
    return np.searchsorted(sorted_train, values, side="right") / max(len(sorted_train), 1)


def add_expanding_state_stats(emb: pd.DataFrame, original: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    if not targets:
        return emb
    keyed = original.reset_index(drop=True).copy()
    keyed["row_position"] = keyed.groupby("entity").cumcount()
    cols = ["date", "entity", "row_position", *targets]
    out = emb.merge(keyed[cols], on=["date", "entity", "row_position"], how="left")
    out = out.sort_values(["date", "entity"]).reset_index(drop=True)
    for target in targets:
        val = pd.to_numeric(out[target], errors="coerce")
        hit = (val > 0).astype(float).where(val.notna())
        for stat_name, series in [(f"{target}_state_mean_prior", val), (f"{target}_state_hit_prior", hit)]:
            pieces = []
            for _, idx in out.groupby("ssl_vq_state").groups.items():
                s = series.loc[idx].sort_index()
                pieces.append(s.expanding().mean().shift(1))
            out[stat_name] = pd.concat(pieces).sort_index() if pieces else np.nan
        out[f"{target}_state_count_prior"] = out.groupby("ssl_vq_state").cumcount()
    return out


def main() -> None:
    args = default_paths(parse_args())
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame, features, targets = prepare_frame(args)
    train_rows = frame[frame["date"].le(pd.Timestamp(args.train_end))]
    scaler = RobustPanelScaler().fit(train_rows[features].ffill().bfill())
    x_train = make_training_windows(frame, features, scaler, args.window, args.max_train_samples, args.train_end, args.seed)
    model = train_model(x_train, args, len(features))
    emb = embed_panel(frame, features, scaler, model, args.window, args.batch_size)
    emb, meta = add_vq_and_confidence(emb, args.train_end, args.clusters, args.seed)
    emb = add_expanding_state_stats(emb, frame, targets)

    out_file = out_dir / f"{args.mode}_ssl_embeddings.csv"
    emb.to_csv(out_file, index=False, encoding="utf-8-sig")
    torch.save(model.state_dict(), out_dir / f"{args.mode}_ssl_encoder.pt")
    meta.update(
        {
            "mode": args.mode,
            "input": str(args.input),
            "output": str(out_file),
            "features": features,
            "targets": targets,
            "window": int(args.window),
            "patch_len": int(args.patch_len),
            "rows": int(emb.shape[0]),
        }
    )
    (out_dir / f"{args.mode}_ssl_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
