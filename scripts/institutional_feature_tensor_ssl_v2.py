from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "institutional_tensor_ssl_v2_latest"
TABLES = OUT / "tables"
PARQUET = OUT / "parquet"
TENSORS = OUT / "tensors"
MODELS = OUT / "models"

ROLE_WINDOWS = {
    "macro": [20, 40, 64, 126],
    "etf": [10, 20, 40, 64],
    "safe": [20, 60, 120],
}

TARGET_PREFIXES = (
    "forward_",
    "realized_",
    "label_",
    "safe_target",
    "entry_",
    "actual_",
    "benchmark_forward_",
)

META_COLUMNS = {
    "date",
    "asset",
    "role",
    "name",
    "group",
    "basket",
    "market",
    "benchmark_ticker",
    "ranking_group",
    "asset_basket",
    "model_group",
    "regime",
    "holding_logic",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Institutional feature store, multi-window tensors, and SSL v2 embeddings.")
    p.add_argument("--stage", choices=["all", "feature-store", "tensors", "ssl"], default="all")
    p.add_argument("--roles", default="macro,etf,safe")
    p.add_argument("--train-end", default="2022-12-31")
    p.add_argument("--valid-end", default="2024-12-31")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-train-samples", type=int, default=12000)
    p.add_argument("--embedding-dim", type=int, default=24)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--clusters", type=int, default=24)
    p.add_argument("--flow-epochs", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def numeric_columns(frame: pd.DataFrame, min_valid: float = 0.35) -> list[str]:
    cols = []
    for col in frame.columns:
        if col in META_COLUMNS:
            continue
        s = pd.to_numeric(frame[col], errors="coerce")
        if s.notna().mean() >= min_valid:
            cols.append(col)
    return cols


def normalize_date_asset(frame: pd.DataFrame, asset_col: str, role: str) -> pd.DataFrame:
    out = frame.copy()
    if "Date" in out.columns and "date" not in out.columns:
        out = out.rename(columns={"Date": "date"})
    out["date"] = pd.to_datetime(out["date"])
    out["asset"] = out[asset_col].astype(str)
    out["role"] = role
    return out


def build_feature_store(args: argparse.Namespace) -> dict:
    PARQUET.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    sentinel = read_csv(ROOT / "outputs/daily_risk_off_sentinel_latest/tables/daily_sentinel_history.csv", parse_dates=["Date"])
    driver = read_csv(ROOT / "outputs/macro_regime_asset_screener_latest/tables/driver_panel.csv", parse_dates=["Date"])
    risk3d = read_csv(ROOT / "outputs/institutional_risk_off_v2_latest/tables/risk_3d_training_panel.csv", parse_dates=["date"])
    macro = pd.DataFrame()
    if not driver.empty:
        macro = driver.rename(columns={"Date": "date"}).sort_values("date")
    if not sentinel.empty:
        sent = sentinel.rename(columns={"Date": "date"}).sort_values("date")
        macro = pd.merge_asof(macro, sent, on="date", direction="backward") if not macro.empty else sent
    if not risk3d.empty:
        risk_cols = [c for c in risk3d.columns if c == "date" or c.startswith("label_") or c.endswith("_target") or c.endswith("_fwd_return")]
        if len(risk_cols) > 1:
            macro = pd.merge_asof(macro.sort_values("date"), risk3d[risk_cols].sort_values("date"), on="date", direction="backward")
    if not macro.empty:
        macro["asset"] = "MACRO"
        macro["role"] = "macro"
        macro = macro.sort_values(["asset", "date"]).reset_index(drop=True)
        macro.to_parquet(PARQUET / "macro_panel.parquet", index=False)

    etf = read_csv(ROOT / "outputs/gaps_long_lived_etf_leadership_latest/tables/long_lived_scored_features.csv", parse_dates=["date"])
    if etf.empty:
        etf = read_csv(ROOT / "outputs/etf_leadership_static_v5_ssl/v5_ssl_scored_features.csv", parse_dates=["date"])
    if not etf.empty:
        etf = normalize_date_asset(etf, "etf_ticker", "etf").sort_values(["asset", "date"]).reset_index(drop=True)
        etf.to_parquet(PARQUET / "etf_panel.parquet", index=False)

    safe = read_csv(ROOT / "outputs/institutional_risk_off_v2_latest/tables/macro_conditioned_safe_asset_panel.csv", parse_dates=["date"])
    if not safe.empty:
        safe = normalize_date_asset(safe, "symbol", "safe").sort_values(["asset", "date"]).reset_index(drop=True)
        safe.to_parquet(PARQUET / "safe_panel.parquet", index=False)

    meta = {
        "train_end": args.train_end,
        "valid_end": args.valid_end,
        "macro_rows": int(macro.shape[0]) if not macro.empty else 0,
        "etf_rows": int(etf.shape[0]) if not etf.empty else 0,
        "safe_rows": int(safe.shape[0]) if not safe.empty else 0,
        "parquet_files": [str(p) for p in PARQUET.glob("*.parquet")],
        "notes": [
            "All panels use date/asset/role keys.",
            "Macro joins use merge_asof(direction='backward') where external macro labels are merged.",
            "Scaling is not stored in parquet; tensor scaler is fitted only on train dates.",
        ],
    }
    (TABLES / "feature_store_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


@dataclass
class RobustScalerState:
    median: dict[str, float]
    iqr: dict[str, float]


def fit_scaler(frame: pd.DataFrame, features: list[str], train_end: str) -> RobustScalerState:
    train = frame[frame["date"].le(pd.Timestamp(train_end))]
    vals = train[features].apply(pd.to_numeric, errors="coerce")
    med = vals.median(axis=0)
    iqr = vals.quantile(0.75) - vals.quantile(0.25)
    fallback = vals.std(axis=0)
    scale = iqr.replace(0, np.nan).fillna(fallback.replace(0, np.nan)).fillna(1.0)
    return RobustScalerState(median=med.to_dict(), iqr=scale.to_dict())


def transform(frame: pd.DataFrame, features: list[str], scaler: RobustScalerState) -> np.ndarray:
    vals = frame[features].apply(pd.to_numeric, errors="coerce").ffill().bfill()
    med = pd.Series(scaler.median)
    iqr = pd.Series(scaler.iqr)
    out = (vals - med) / iqr
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-8, 8).to_numpy(dtype=np.float32)


def choose_features(role: str, frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    target_cols = [c for c in frame.columns if c.startswith(TARGET_PREFIXES) or c in {"risk_assets_fwd_5d_avg", "risk_assets_fwd_20d_avg"}]
    nums = numeric_columns(frame)
    features = [c for c in nums if c not in target_cols and not c.startswith("ssl_")]
    if role == "macro":
        features = [c for c in features if not c.startswith("macro_macro_ssl")][:96]
    elif role == "etf":
        preferred = [
            "ETF_return_5D",
            "ETF_return_20D",
            "ETF_return_60D",
            "ETF_return_120D",
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
            "reg_coef_high_proximity",
            "reg_coef_component_return_60d",
            "reg_coef_component_rs_60d",
            "reg_r2",
            "reg_residual_dispersion",
            "rule_5d_score",
            "rule_20d_score",
        ]
        features = [c for c in preferred if c in frame.columns]
    else:
        safe_pref = [
            "score_0_100",
            "upside_prob_1w",
            "upside_prob_4w",
            "technical_score",
            "driver_fit_score",
            "beta_fit_score",
            "calibrated_prob_1w",
            "calibrated_prob_4w",
            "institutional_score_0_100",
            "macro_US10Y_driver_chg_5d",
            "macro_US10Y_driver_chg_20d",
            "macro_US2Y_chg_5d",
            "macro_US2Y_chg_20d",
            "macro_US10Y_REAL_chg_20d",
            "macro_DXY_driver_ret_20d",
            "macro_USDKRW_driver_ret_20d",
            "macro_VIX_chg_20d",
            "macro_HY_OAS_chg_20d",
            "macro_GOLD_driver_ret_20d",
            "macro_HYG_IEF_ret_20d",
            "macro_axis1_vol_credit_stress",
            "macro_axis2_fx_liquidity_stress",
            "macro_axis3_peak_fragility_stress",
            "macro_risk_off_score",
        ]
        features = [c for c in safe_pref if c in frame.columns]
    return features, target_cols


def build_tensors(args: argparse.Namespace) -> dict:
    TENSORS.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, dict] = {}
    for role in requested_roles(args):
        path = PARQUET / f"{role}_panel.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        frame["date"] = pd.to_datetime(frame["date"])
        features, targets = choose_features(role, frame)
        if not features:
            continue
        scaler = fit_scaler(frame, features, args.train_end)
        (TENSORS / f"{role}_scaler.json").write_text(json.dumps({"features": features, "scaler": scaler.__dict__}, ensure_ascii=False, indent=2), encoding="utf-8")
        role_meta = {"features": features, "targets": targets, "windows": {}}
        for window in ROLE_WINDOWS[role]:
            x_list = []
            rows = []
            y_rows = []
            for asset, part in frame.sort_values(["asset", "date"]).groupby("asset", sort=False):
                part = part.sort_values("date").reset_index(drop=True)
                if part.shape[0] < window:
                    continue
                values = transform(part, features, scaler)
                target_part = part[["date", "asset", *[c for c in targets if c in part.columns]]].copy()
                for i in range(window - 1, len(part)):
                    x_list.append(values[i - window + 1 : i + 1])
                    rows.append({"date": part.loc[i, "date"], "asset": asset, "row_position": int(i)})
                    if targets:
                        y_rows.append(target_part.iloc[i].to_dict())
            if not x_list:
                continue
            x = np.stack(x_list).astype(np.float32)
            stem = f"{role}_w{window}"
            np.savez_compressed(TENSORS / f"{stem}.npz", X=x)
            meta_df = pd.DataFrame(rows)
            meta_df.to_csv(TENSORS / f"{stem}_meta.csv", index=False, encoding="utf-8-sig")
            if y_rows:
                pd.DataFrame(y_rows).to_csv(TENSORS / f"{stem}_targets.csv", index=False, encoding="utf-8-sig")
            role_meta["windows"][str(window)] = {"samples": int(x.shape[0]), "features": int(x.shape[2])}
            print(f"[tensor] {stem} X={x.shape}", flush=True)
        metadata[role] = role_meta
    (TABLES / "tensor_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


class MaskedContrastiveEncoder(torch.nn.Module):
    def __init__(self, n_features: int, window: int, patch_len: int, d_model: int, embedding_dim: int, layers: int, heads: int):
        super().__init__()
        if window % patch_len != 0:
            raise ValueError("window must be divisible by patch_len")
        self.window = window
        self.patch_len = patch_len
        self.n_patches = window // patch_len
        self.patch_dim = patch_len * n_features
        self.patch_proj = torch.nn.Linear(self.patch_dim, d_model)
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = torch.nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        enc = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 3,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(enc, num_layers=layers)
        self.decoder = torch.nn.Sequential(torch.nn.LayerNorm(d_model), torch.nn.Linear(d_model, self.patch_dim))
        self.projector = torch.nn.Sequential(torch.nn.LayerNorm(d_model), torch.nn.Linear(d_model, embedding_dim))

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        return x.reshape(b, self.n_patches, self.patch_len, c).reshape(b, self.n_patches, self.patch_dim)

    def forward(self, x: torch.Tensor, mask_ratio: float = 0.35) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        patches = self.patchify(x)
        h = self.patch_proj(patches)
        if self.training and mask_ratio > 0:
            mask = torch.rand(h.shape[:2], device=h.device) < mask_ratio
            h = torch.where(mask.unsqueeze(-1), self.mask_token.expand_as(h), h)
        else:
            mask = torch.zeros(h.shape[:2], device=h.device, dtype=torch.bool)
        h = self.encoder(h + self.pos)
        recon = self.decoder(h)
        emb = self.projector(h.mean(dim=1))
        return recon, patches, mask, emb


def augment(x: torch.Tensor) -> torch.Tensor:
    noise = torch.randn_like(x) * 0.025
    out = x + noise
    if x.shape[1] > 8:
        mask = torch.rand(x.shape[0], x.shape[1], 1, device=x.device) < 0.08
        out = torch.where(mask, torch.zeros_like(out), out)
    return out


def info_nce(z1: torch.Tensor, z2: torch.Tensor, temp: float = 0.15) -> torch.Tensor:
    z1 = torch.nn.functional.normalize(z1, dim=1)
    z2 = torch.nn.functional.normalize(z2, dim=1)
    logits = z1 @ z2.T / temp
    labels = torch.arange(z1.shape[0], device=z1.device)
    return 0.5 * (torch.nn.functional.cross_entropy(logits, labels) + torch.nn.functional.cross_entropy(logits.T, labels))


class CouplingLayer(torch.nn.Module):
    def __init__(self, dim: int, hidden: int, flip: bool):
        super().__init__()
        mask = torch.arange(dim) % 2
        if flip:
            mask = 1 - mask
        self.register_buffer("mask", mask.float())
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, dim * 2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xm = x * self.mask
        s, t = self.net(xm).chunk(2, dim=1)
        s = torch.tanh(s) * (1 - self.mask) * 1.5
        t = t * (1 - self.mask)
        y = xm + (1 - self.mask) * (x * torch.exp(s) + t)
        logdet = s.sum(dim=1)
        return y, logdet


class RealNVP(torch.nn.Module):
    def __init__(self, dim: int, layers: int = 6, hidden: int = 96):
        super().__init__()
        self.layers = torch.nn.ModuleList([CouplingLayer(dim, hidden, bool(i % 2)) for i in range(layers)])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logdet = torch.zeros(x.shape[0], device=x.device)
        z = x
        for layer in self.layers:
            z, ld = layer(z)
            logdet = logdet + ld
        return z, logdet

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z, logdet = self.forward(x)
        base = -0.5 * (z.pow(2).sum(dim=1) + z.shape[1] * math.log(2 * math.pi))
        return base + logdet


def train_ssl_for_tensor(args: argparse.Namespace, role: str, window: int) -> pd.DataFrame | None:
    tensor_path = TENSORS / f"{role}_w{window}.npz"
    meta_path = TENSORS / f"{role}_w{window}_meta.csv"
    if not tensor_path.exists() or not meta_path.exists():
        return None
    x = np.load(tensor_path)["X"].astype(np.float32)
    meta = pd.read_csv(meta_path, parse_dates=["date"])
    train_idx = np.flatnonzero(meta["date"].le(pd.Timestamp(args.train_end)).to_numpy())
    if train_idx.size == 0:
        return None
    rng = np.random.default_rng(args.seed)
    if train_idx.size > args.max_train_samples:
        train_idx = rng.choice(train_idx, size=args.max_train_samples, replace=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_len = choose_patch_len(window)
    model = MaskedContrastiveEncoder(x.shape[2], window, patch_len, args.d_model, args.embedding_dim, args.layers, args.heads).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_tensor = torch.from_numpy(x[train_idx])
    model.train()
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(train_tensor.shape[0])
        losses = []
        for start in range(0, train_tensor.shape[0], args.batch_size):
            batch = train_tensor[perm[start : start + args.batch_size]].to(device)
            recon, patches, mask, _ = model(batch, mask_ratio=0.35)
            recon_loss = ((recon - patches) ** 2)[mask].mean() if mask.any() else ((recon - patches) ** 2).mean()
            _, _, _, z1 = model(augment(batch), mask_ratio=0.0)
            _, _, _, z2 = model(augment(batch), mask_ratio=0.0)
            c_loss = info_nce(z1, z2)
            loss = recon_loss + 0.15 * c_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        print(f"[ssl-v2] role={role} window={window} epoch={epoch}/{args.epochs} loss={np.mean(losses):.5f}", flush=True)
    model.cpu().eval()
    emb = embed_all(model, x, args.batch_size)
    out = meta.copy()
    for i in range(emb.shape[1]):
        out[f"ssl2_emb_{i:02d}"] = emb[:, i]
    out = add_vq_stats(out, args, role, window)
    out = add_realnvp_confidence(out, args)
    targets_path = TENSORS / f"{role}_w{window}_targets.csv"
    if targets_path.exists():
        targets = pd.read_csv(targets_path, parse_dates=["date"])
        out = out.merge(targets, on=["date", "asset"], how="left")
        out = add_regime_target_stats(out)
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS / f"{role}_w{window}_ssl2_encoder.pt")
    out_file = TABLES / f"{role}_w{window}_ssl2_embeddings.csv"
    out.to_csv(out_file, index=False, encoding="utf-8-sig")
    return out


def choose_patch_len(window: int) -> int:
    for p in [10, 8, 5, 4, 2]:
        if window % p == 0:
            return p
    return 1


@torch.no_grad()
def embed_all(model: MaskedContrastiveEncoder, x: np.ndarray, batch_size: int) -> np.ndarray:
    arrs = []
    for start in range(0, x.shape[0], batch_size):
        batch = torch.from_numpy(x[start : start + batch_size])
        _, _, _, emb = model(batch, mask_ratio=0.0)
        arrs.append(emb.numpy())
    return np.vstack(arrs)


def add_vq_stats(out: pd.DataFrame, args: argparse.Namespace, role: str, window: int) -> pd.DataFrame:
    emb_cols = [c for c in out.columns if c.startswith("ssl2_emb_")]
    x = out[emb_cols].to_numpy(dtype=float)
    train_mask = out["date"].le(pd.Timestamp(args.train_end)).to_numpy()
    x_train = x[train_mask]
    n_clusters = max(4, min(args.clusters, max(4, x_train.shape[0] // 20)))
    km = MiniBatchKMeans(n_clusters=n_clusters, batch_size=2048, random_state=args.seed, n_init="auto")
    km.fit(x_train)
    out["ssl2_vq_state"] = km.predict(x).astype(int)
    out["ssl2_vq_distance"] = np.linalg.norm(x - km.cluster_centers_[out["ssl2_vq_state"].to_numpy()], axis=1)
    (MODELS / f"{role}_w{window}_vq_metadata.json").write_text(json.dumps({"clusters": int(n_clusters)}, indent=2), encoding="utf-8")
    return out


def add_realnvp_confidence(out: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    emb_cols = [c for c in out.columns if c.startswith("ssl2_emb_")]
    x = out[emb_cols].to_numpy(dtype=np.float32)
    train_mask = out["date"].le(pd.Timestamp(args.train_end)).to_numpy()
    train = torch.from_numpy(x[train_mask])
    if train.shape[0] < 50:
        out["ssl2_nf_loglik"] = np.nan
        out["ssl2_nf_nll"] = np.nan
        out["ssl2_nf_confidence"] = np.nan
        return out
    flow = RealNVP(dim=x.shape[1])
    opt = torch.optim.AdamW(flow.parameters(), lr=8e-4, weight_decay=1e-5)
    flow.train()
    for _ in range(args.flow_epochs):
        perm = torch.randperm(train.shape[0])
        for start in range(0, train.shape[0], args.batch_size):
            batch = train[perm[start : start + args.batch_size]]
            loss = -flow.log_prob(batch).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 2.0)
            opt.step()
    flow.eval()
    with torch.no_grad():
        loglik = flow.log_prob(torch.from_numpy(x)).numpy()
        train_loglik = loglik[train_mask]
    out["ssl2_nf_loglik"] = loglik
    out["ssl2_nf_nll"] = -loglik
    out["ssl2_nf_confidence"] = percentile(loglik, train_loglik) * 100.0
    return out


def percentile(values: np.ndarray, train_values: np.ndarray) -> np.ndarray:
    sorted_train = np.sort(train_values)
    return np.searchsorted(sorted_train, values, side="right") / max(len(sorted_train), 1)


def add_regime_target_stats(out: pd.DataFrame) -> pd.DataFrame:
    target_cols = [c for c in out.columns if c.startswith(TARGET_PREFIXES) or c in {"risk_assets_fwd_5d_avg", "risk_assets_fwd_20d_avg"}]
    out = out.sort_values(["date", "asset"]).reset_index(drop=True)
    for target in target_cols:
        val = pd.to_numeric(out[target], errors="coerce")
        hit = (val > 0).astype(float).where(val.notna())
        state = out["ssl2_vq_state"]
        out[f"{target}_vq_mean_prior"] = val.groupby(state).expanding().mean().shift(1).reset_index(level=0, drop=True).sort_index()
        out[f"{target}_vq_hit_prior"] = hit.groupby(state).expanding().mean().shift(1).reset_index(level=0, drop=True).sort_index()
        out[f"{target}_vq_count_prior"] = out.groupby("ssl2_vq_state").cumcount()
    return out


def run_ssl(args: argparse.Namespace) -> dict:
    TABLES.mkdir(parents=True, exist_ok=True)
    summary = []
    for role in requested_roles(args):
        for window in ROLE_WINDOWS[role]:
            emb = train_ssl_for_tensor(args, role, window)
            if emb is not None:
                summary.append(
                    {
                        "role": role,
                        "window": window,
                        "rows": int(emb.shape[0]),
                        "embedding_cols": len([c for c in emb.columns if c.startswith("ssl2_emb_")]),
                        "vq_states": int(emb["ssl2_vq_state"].nunique()) if "ssl2_vq_state" in emb else 0,
                        "mean_nf_confidence": float(pd.to_numeric(emb.get("ssl2_nf_confidence"), errors="coerce").mean()),
                    }
                )
    df = pd.DataFrame(summary)
    df.to_csv(TABLES / "ssl2_embedding_summary.csv", index=False, encoding="utf-8-sig")
    return {"runs": summary}


def requested_roles(args: argparse.Namespace) -> list[str]:
    requested = [r.strip() for r in args.roles.split(",") if r.strip()]
    return [r for r in requested if r in ROLE_WINDOWS]


def main() -> None:
    args = parse_args()
    for d in [OUT, TABLES, PARQUET, TENSORS, MODELS]:
        d.mkdir(parents=True, exist_ok=True)
    meta: dict[str, object] = {"stage": args.stage}
    if args.stage in {"all", "feature-store"}:
        meta["feature_store"] = build_feature_store(args)
    if args.stage in {"all", "tensors"}:
        meta["tensors"] = build_tensors(args)
    if args.stage in {"all", "ssl"}:
        meta["ssl"] = run_ssl(args)
    (TABLES / "run_summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
