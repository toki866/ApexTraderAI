# tools/run_stepd_prime_features.py
# -*- coding: utf-8 -*-
"""
StepD' Prime Features (Transformer embedding) generator.

Purpose
-------
Train a small Transformer encoder on StepA features (train split) and export
fixed-length embeddings ("StepD' embeddings") for each requested source+scale.

Key concept
-----------
- "fit-split": which split is used to train the model (default: train)
- "export-split": which dates we export embeddings for:
    test  -> legacy behavior (test dates only)
    train -> train dates only
    all   -> train+test dates (recommended for StepE RL training)

Why "all" export is recommended
-------------------------------
If you only export for test dates (e.g., 62 rows in 2022Q1), StepE can't train
a policy over the full train period. Exporting "all" gives you ~2000 rows.

Inputs (StepA outputs)
----------------------
output/stepA/<mode>/
  stepA_prices_train_<SYMBOL>.csv
  stepA_prices_test_<SYMBOL>.csv
  stepA_periodic_train_<SYMBOL>.csv
  stepA_periodic_test_<SYMBOL>.csv
  stepA_tech_train_<SYMBOL>.csv
  stepA_tech_test_<SYMBOL>.csv

Outputs
-------
output/stepD_prime/<mode>/embeddings/
  stepDprime_<source>_hXX_<SYMBOL>_embeddings_<export-split>.csv   (if export-split != test)
  stepDprime_<source>_hXX_<SYMBOL>_embeddings.csv                 (legacy name when export-split=test)

output/stepD_prime/<mode>/models/
  stepDprime_<source>_hXX_<SYMBOL>.pt

Notes
-----
- Embedding columns are always "emb_000".."emb_031" (32 dims).
- The script is CPU-friendly by default.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path

# Resolve repo root and fix CWD so relative paths like 'output/...' are stable.
_REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    os.chdir(_REPO_ROOT)
except Exception:
    pass

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


# ---------------------------
# Utility
# ---------------------------

def _safe_to_datetime(x) -> pd.Series:
    return pd.to_datetime(x, errors="coerce").dt.tz_localize(None)

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _zscore(s: pd.Series, win: int) -> pd.Series:
    m = s.rolling(win, min_periods=win).mean()
    v = s.rolling(win, min_periods=win).std(ddof=0)
    return (s - m) / (v.replace(0, np.nan))

def _atr_norm(df: pd.DataFrame, win: int = 14) -> pd.Series:
    # True range based on prior close; avoid leakage by using shift(1) close.
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close_prev = df["Close"].astype(float).shift(1)
    tr = pd.concat([
        (high - low),
        (high - close_prev).abs(),
        (low - close_prev).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(win, min_periods=win).mean()
    close_prev2 = close_prev.replace(0, np.nan)
    return (atr / close_prev2).replace([np.inf, -np.inf], np.nan)

def _gap(df: pd.DataFrame) -> pd.Series:
    # Gap = Open / prev Close - 1
    open_ = df["Open"].astype(float)
    close_prev = df["Close"].astype(float).shift(1)
    return (open_ / close_prev.replace(0, np.nan) - 1.0).replace([np.inf, -np.inf], np.nan)

def _oc_ret(df: pd.DataFrame) -> pd.Series:
    # Open->Close return for same day
    open_ = df["Open"].astype(float)
    close_ = df["Close"].astype(float)
    return (close_ / open_.replace(0, np.nan) - 1.0).replace([np.inf, -np.inf], np.nan)

def _make_bnf_features(df_prices: pd.DataFrame) -> pd.DataFrame:
    df = df_prices.copy()
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c not in df.columns:
            raise KeyError(f"prices missing column: {c}")

    # Use previous day's info to avoid "same-day close" leakage.
    close_prev = df["Close"].astype(float).shift(1)
    vol_prev = df["Volume"].astype(float).shift(1)

    ret_1 = close_prev.pct_change(1)
    ret_5 = close_prev.pct_change(5)
    ret_20 = close_prev.pct_change(20)

    dev_z_25 = _zscore(close_prev, 25)
    vol_log = np.log(vol_prev.replace(0, np.nan))
    vol_log_ratio_20 = vol_log - np.log(vol_prev.rolling(20, min_periods=20).mean().replace(0, np.nan))
    bnf_score = dev_z_25 * vol_log_ratio_20

    atr_norm_14 = _atr_norm(df, 14)
    gap_1 = _gap(df)

    # Candle ratios (based on same day O/H/L, but those are known only end of day;
    # if you run at close-10min, it's acceptable; otherwise, keep for now.)
    o = df["Open"].astype(float)
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)

    rng = (h - l).replace(0, np.nan)
    body = (c - o).abs()
    upper = (h - np.maximum(o, c)).clip(lower=0)
    lower = (np.minimum(o, c) - l).clip(lower=0)

    body_ratio = (body / rng).replace([np.inf, -np.inf], np.nan)
    upper_wick_ratio = (upper / rng).replace([np.inf, -np.inf], np.nan)
    lower_wick_ratio = (lower / rng).replace([np.inf, -np.inf], np.nan)

    vol_chg_1 = vol_prev.pct_change(1)
    vol_z_20 = _zscore(vol_prev, 20)

    out = pd.DataFrame({
        "dev_z_25": dev_z_25,
        "vol_log_ratio_20": vol_log_ratio_20,
        "bnf_score": bnf_score,
        "ret_1": ret_1,
        "ret_5": ret_5,
        "ret_20": ret_20,
        "atr_norm_14": atr_norm_14,
        "gap_1": gap_1,
        "body_ratio": body_ratio,
        "upper_wick_ratio": upper_wick_ratio,
        "lower_wick_ratio": lower_wick_ratio,
        "vol_chg_1": vol_chg_1,
        "vol_z_20": vol_z_20,
        "oc_ret_0": _oc_ret(df),  # sometimes useful
    })
    return out


def _load_stepA_split(output_root: Path, mode: str, symbol: str) -> Dict[str, str]:
    p = output_root / "stepA" / mode / f"stepA_split_summary_{symbol}.csv"
    if not p.exists():
        cand = list((output_root / "stepA" / mode).glob(f"*split*summary*{symbol}*.csv"))
        if cand:
            p = cand[0]
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    if "key" in df.columns and "value" in df.columns:
        return dict(zip(df["key"].astype(str), df["value"].astype(str)))
    out: Dict[str, str] = {}
    if len(df) == 1:
        for c in df.columns:
            out[c] = str(df.iloc[0][c])
    return out


def _load_stepA_pair(output_root: Path, mode: str, symbol: str, stem: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = output_root / "stepA" / mode
    p_tr = base / f"{stem}_train_{symbol}.csv"
    p_te = base / f"{stem}_test_{symbol}.csv"
    if not p_tr.exists() or not p_te.exists():
        raise FileNotFoundError(f"Missing StepA files: {p_tr} or {p_te}")
    df_tr = pd.read_csv(p_tr)
    df_te = pd.read_csv(p_te)
    for df in (df_tr, df_te):
        if "Date" not in df.columns:
            raise KeyError(f"{stem} missing Date column")
        df["Date"] = _safe_to_datetime(df["Date"])
    return df_tr, df_te


def _merge_stepA(output_root: Path, mode: str, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      df_all: merged all dates (train+test)
      df_train: merged train dates
      df_test: merged test dates
    """
    prices_tr, prices_te = _load_stepA_pair(output_root, mode, symbol, "stepA_prices")
    per_tr, per_te = _load_stepA_pair(output_root, mode, symbol, "stepA_periodic")
    tech_tr, tech_te = _load_stepA_pair(output_root, mode, symbol, "stepA_tech")

    def _merge_one(pr, pe, te):
        df = pr.merge(pe, on="Date", how="left").merge(te, on="Date", how="left")
        df = df.sort_values("Date").reset_index(drop=True)
        return df

    df_train = _merge_one(prices_tr, per_tr, tech_tr)
    df_test = _merge_one(prices_te, per_te, tech_te)
    df_all = pd.concat([df_train, df_test], axis=0, ignore_index=True).sort_values("Date").reset_index(drop=True)

    # add engineered features
    bnf = _make_bnf_features(df_all[["Open", "High", "Low", "Close", "Volume"]].assign(Date=df_all["Date"]))
    bnf_cols = bnf.columns.tolist()
    df_all = pd.concat([df_all, bnf], axis=1)

    # Fill engineered NaNs with 0 (keeps tensors clean). Keep raw O/H/L/C/V intact.
    for c in bnf_cols:
        df_all[c] = df_all[c].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

    # train/test slices by original boundaries
    n_tr = len(df_train)
    df_train2 = df_all.iloc[:n_tr].copy().reset_index(drop=True)
    df_test2 = df_all.iloc[n_tr:].copy().reset_index(drop=True)
    return df_all, df_train2, df_test2


# ---------------------------
# Dataset
# ---------------------------

@dataclass
class ScaleSpec:
    scale_id: int
    seq_len: int
    horizon: int  # label horizon days


SCALES: Dict[int, ScaleSpec] = {
    1: ScaleSpec(scale_id=1, seq_len=8, horizon=1),
    2: ScaleSpec(scale_id=2, seq_len=12, horizon=2),
    3: ScaleSpec(scale_id=3, seq_len=63, horizon=3),
}


class SeqDataset(Dataset):
    def __init__(self, X_seq: np.ndarray, y: np.ndarray):
        assert X_seq.ndim == 3
        self.X = X_seq.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)


# ---------------------------
# Model
# ---------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        T = x.shape[1]
        return x + self.pe[:, :T, :]


class PrimeEncoder(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 64, emb_dim: int = 32, nhead: int = 4, nlayers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        self.posenc = PositionalEncoding(d_model, max_len=512)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.pool = nn.LayerNorm(d_model)
        self.to_emb = nn.Linear(d_model, emb_dim)
        self.cls = nn.Linear(emb_dim, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, in_dim)
        h = self.in_proj(x)
        h = self.posenc(h)
        h = self.encoder(h)
        # mean pool
        h = self.pool(h).mean(dim=1)
        emb = torch.tanh(self.to_emb(h))
        logits = self.cls(emb)
        return emb, logits


# ---------------------------
# Build sequences
# ---------------------------

def _select_feature_cols(df: pd.DataFrame, source: str) -> List[str]:
    ignore = {"Date"}
    price_cols = {"Open", "High", "Low", "Close", "Volume"}

    if source == "bnf":
        cols = [
            "dev_z_25", "vol_log_ratio_20", "bnf_score",
            "ret_1", "ret_5", "ret_20",
            "atr_norm_14", "gap_1",
            "body_ratio", "upper_wick_ratio", "lower_wick_ratio",
            "vol_chg_1", "vol_z_20",
            "oc_ret_0",
        ]
        # keep only existing
        cols = [c for c in cols if c in df.columns]
        return cols

    if source == "all_features":
        cols: List[str] = []
        # periodic features
        cols += [c for c in df.columns if c not in ignore and c.startswith("per_")]
        # If periodic columns are not prefixed, include all non-price columns from periodic part
        if not cols:
            # heuristic: StepA periodic often has 44 sin/cos columns; include any column that looks like sin/cos or "day_"
            cols += [c for c in df.columns if c not in ignore and c not in price_cols and ("sin" in c or "cos" in c or "day" in c or "retro" in c)]

        # tech features: include common names if present
        tech_keep = [
            "RSI", "RSI_14", "MACD", "MACD_hist", "MACD_hist_12_26_9",
            "ATR", "ATR_norm", "atr_norm_14", "gap", "Gap", "gap_1",
        ]
        for name in tech_keep:
            for c in df.columns:
                if c == name:
                    cols.append(c)

        # add BNF engineered
        cols += [c for c in df.columns if c in {
            "dev_z_25", "vol_log_ratio_20", "bnf_score",
            "ret_1", "ret_5", "ret_20",
            "atr_norm_14", "gap_1",
            "body_ratio", "upper_wick_ratio", "lower_wick_ratio",
            "vol_chg_1", "vol_z_20",
        }]

        # add scale-free OHLCV derived (avoid raw Close scale)
        extra = ["ret_1", "ret_5", "ret_20", "atr_norm_14", "gap_1", "body_ratio", "upper_wick_ratio", "lower_wick_ratio"]
        cols += [c for c in extra if c in df.columns]

        # Dedup preserving order
        seen = set()
        out = []
        for c in cols:
            if c in ignore:
                continue
            if c in price_cols:
                continue
            if c not in seen and pd.api.types.is_numeric_dtype(df[c]):
                out.append(c)
                seen.add(c)
        return out

    raise ValueError(f"unknown source={source}")


def _build_train_sequences(df_train: pd.DataFrame, feat_cols: List[str], spec: ScaleSpec) -> Tuple[np.ndarray, np.ndarray]:
    L = spec.seq_len
    H = spec.horizon

    feats = df_train[feat_cols].astype(np.float32).to_numpy()
    close = df_train["Close"].astype(float).to_numpy()

    Xs = []
    ys = []
    n = len(df_train)

    for i in range(L - 1, n - H):
        x = feats[i - (L - 1): i + 1, :]
        # label: future close direction relative to today's close
        y = 1 if (close[i + H] - close[i]) > 0 else 0
        Xs.append(x)
        ys.append(y)

    if not Xs:
        return np.zeros((0, L, len(feat_cols)), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    X = np.stack(Xs, axis=0)
    y = np.array(ys, dtype=np.int64)
    return X, y


def _export_embeddings(model: PrimeEncoder, df_all: pd.DataFrame, feat_cols: List[str], spec: ScaleSpec, anchors: pd.Series, device: torch.device) -> pd.DataFrame:
    L = spec.seq_len
    feats = df_all[feat_cols].astype(np.float32).to_numpy()
    date_i64 = pd.to_datetime(df_all['Date'], errors='coerce').values.astype('datetime64[ns]').astype('int64')
    date_to_idx = {int(k): i for i, k in enumerate(date_i64.tolist())}

    emb_dim = 32
    emb_cols = [f"emb_{i:03d}" for i in range(emb_dim)]

    rows = []
    model.eval()
    with torch.no_grad():
        for d in anchors:
            dts = pd.to_datetime(d, errors='coerce')
            key = None if pd.isna(dts) else int(dts.value)
            idx = None if key is None else date_to_idx.get(key, None)
            date_str = '' if pd.isna(dts) else dts.strftime('%Y-%m-%d')
            if idx is None:
                row = {'Date': date_str, 'label_available': 0, 'label': 0}
                for c in emb_cols:
                    row[c] = 0.0
                rows.append(row)
                continue

            if idx - (L - 1) < 0:
                row = {'Date': date_str, 'label_available': 0, 'label': 0}
                for c in emb_cols:
                    row[c] = 0.0
                rows.append(row)
                continue

            x = feats[idx - (L - 1): idx + 1, :]
            x_t = torch.from_numpy(x).unsqueeze(0).to(device)
            emb, _ = model(x_t)
            emb = emb.squeeze(0).cpu().numpy().astype(np.float32)

            row = {'Date': date_str, 'label_available': 1, 'label': 0}
            for c, v in zip(emb_cols, emb):
                row[c] = float(v)
            rows.append(row)

    out = pd.DataFrame(rows)
    out["Date"] = _safe_to_datetime(out["Date"])
    out = out.sort_values("Date").reset_index(drop=True)
    return out


# ---------------------------
# Train loop
# ---------------------------

def _train_model(X: np.ndarray, y: np.ndarray, in_dim: int, device: torch.device, seed: int = 7,
                 epochs: int = 200, batch: int = 64, lr: float = 1e-3, patience: int = 10) -> Tuple[PrimeEncoder, Dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    if len(X) < 200:
        # keep batch small for tiny datasets
        batch = min(batch, max(16, len(X)))

    # time-split validation: last 20%
    n = len(X)
    n_val = max(1, int(0.2 * n))
    n_tr = max(1, n - n_val)
    X_tr, y_tr = X[:n_tr], y[:n_tr]
    X_val, y_val = X[n_tr:], y[n_tr:]

    ds_tr = SeqDataset(X_tr, y_tr)
    ds_val = SeqDataset(X_val, y_val)

    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True, drop_last=False)
    dl_val = DataLoader(ds_val, batch_size=batch, shuffle=False, drop_last=False)

    model = PrimeEncoder(in_dim=in_dim, d_model=64, emb_dim=32, nhead=4, nlayers=2, dropout=0.1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best = float("inf")
    best_state = None
    wait = 0

    def _eval(dl: DataLoader) -> Tuple[float, float]:
        model.eval()
        tot_loss = 0.0
        tot = 0
        correct = 0
        with torch.no_grad():
            for xb, yb in dl:
                xb = xb.to(device)
                yb = yb.to(device)
                _, logits = model(xb)
                loss = loss_fn(logits, yb)
                tot_loss += float(loss.item()) * len(xb)
                tot += len(xb)
                pred = logits.argmax(dim=1)
                correct += int((pred == yb).sum().item())
        return (tot_loss / max(1, tot), correct / max(1, tot))

    for ep in range(1, epochs + 1):
        model.train()
        tot_loss = 0.0
        tot = 0
        correct = 0

        for xb, yb in dl_tr:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            _, logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tot_loss += float(loss.item()) * len(xb)
            tot += len(xb)
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())

        tr_loss = tot_loss / max(1, tot)
        tr_acc = correct / max(1, tot)
        val_loss, val_acc = _eval(dl_val)

        print(f"[StepD'feat] ep={ep:03d} train_loss={tr_loss:.4f} acc={tr_acc:.3f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        if val_loss + 1e-6 < best:
            best = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"[StepD'feat] early stop at ep={ep}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = {
        "val_loss": float(best),
    }
    return model, metrics


# ---------------------------
# Main
# ---------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--mode", default="sim", choices=["sim", "live", "ops", "display"])
    ap.add_argument("--output-root", default="output")
    ap.add_argument("--sources", default="bnf,all_features")
    ap.add_argument("--scales", default="1,2,3")
    ap.add_argument("--fit-split", default="train", choices=["train"])
    ap.add_argument("--export-split", default="test", choices=["test", "train", "all"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    symbol = args.symbol
    mode = args.mode
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = _REPO_ROOT / output_root
    device = torch.device(args.device)

    df_all, df_train, df_test = _merge_stepA(output_root, mode, symbol)

    # anchors
    if args.export_split == "test":
        anchors = df_test["Date"].copy()
    elif args.export_split == "train":
        anchors = df_train["Date"].copy()
    else:
        anchors = df_all["Date"].copy()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    scale_ids = [int(x) for x in args.scales.split(",") if x.strip()]

    out_dir = output_root / "stepD_prime" / mode
    emb_dir = out_dir / "embeddings"
    mdl_dir = out_dir / "models"
    _ensure_dir(emb_dir)
    _ensure_dir(mdl_dir)

    print(f"[StepD'feat] anchors_fit={len(df_train)} anchors_export={len(anchors)} mode={mode} symbol={symbol} device={device}")

    for src in sources:
        feat_cols = _select_feature_cols(df_all, src)
        if not feat_cols:
            raise RuntimeError(f"No feature columns for source={src}")

        # standardize based on train split only
        mu = df_train[feat_cols].astype(float).mean(axis=0)
        sd = df_train[feat_cols].astype(float).std(axis=0, ddof=0).replace(0, 1.0)
        df_all_std = df_all.copy()
        for c in feat_cols:
            df_all_std[c] = ((df_all_std[c].astype(float) - float(mu[c])) / float(sd[c])).astype(float)

        df_train_std = df_all_std.iloc[:len(df_train)].copy()

        for scale_id in scale_ids:
            if scale_id not in SCALES:
                raise ValueError(f"unknown scale={scale_id}. allowed={sorted(SCALES)}")
            spec = SCALES[scale_id]
            hh = f"h{spec.horizon:02d}"

            # build training sequences
            X, y = _build_train_sequences(df_train_std, feat_cols, spec)
            if len(X) == 0:
                raise RuntimeError(f"No training samples for source={src} scale={scale_id}")

            print(f"[StepD'feat] train src={src} scale={scale_id} ({hh}) samples={len(X)} seq_len={spec.seq_len} dim={len(feat_cols)} device={device}")

            model, metrics = _train_model(
                X, y, in_dim=len(feat_cols), device=device, seed=args.seed,
                epochs=int(args.epochs), batch=int(args.batch), lr=float(args.lr), patience=int(args.patience)
            )

            # export embeddings
            df_emb = _export_embeddings(model, df_all_std, feat_cols, spec, anchors=anchors, device=device)

            # append metadata
            df_emb.insert(1, "source", src)
            df_emb.insert(2, "horizon", spec.horizon)
            df_emb.insert(3, "scale", spec.scale_id)
            df_emb.insert(4, "seq_len", spec.seq_len)

            # output names
            if args.export_split == "test":
                out_csv = emb_dir / f"stepDprime_{src}_{hh}_{symbol}_embeddings.csv"
            else:
                out_csv = emb_dir / f"stepDprime_{src}_{hh}_{symbol}_embeddings_{args.export_split}.csv"

            df_emb.to_csv(out_csv, index=False)

            out_mdl = mdl_dir / f"stepDprime_{src}_{hh}_{symbol}.pt"
            torch.save({
                "state_dict": model.state_dict(),
                "feat_cols": feat_cols,
                "scale": spec.scale_id,
                "seq_len": spec.seq_len,
                "horizon": spec.horizon,
                "metrics": metrics,
            }, out_mdl)

            print(f"[StepD'feat] wrote embeddings -> {out_csv} shape={df_emb.shape}")
            print(f"[StepD'feat] wrote model -> {out_mdl}")

    print("[StepD'feat] ALL DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
