# -*- coding: utf-8 -*-
'''
step_dprime_service.py

StepD' (StepD-prime): Train transformer summarizers on StepB predicted paths and output embeddings.

Current scope (v1):
- Supports sources:
    - mamba         : output/stepB/<mode>/daily/stepB_daily_pred_mamba_hXX_<SYMBOL>_YYYY_MM_DD.csv
    - mamba_periodic: output/stepB/<mode>/daily_periodic/stepB_daily_pred_mamba_periodic_hXX_<SYMBOL>_YYYY_MM_DD.csv

- For each (source, horizon), trains a small transformer encoder on fixed-length padded sequences.
- Supervised objective: binary classification of realized return sign from Date_anchor -> Date_target(horizon).
- Produces embeddings CSV for downstream StepE.

Notes:
- This trainer needs realized Close prices to create labels. It reads them from StepA prices_train/test CSVs.
- If StepB daily paths only exist for the test period, training on "train only" will yield zero samples.
  In that case, set fit_split="available" to train on available samples (typically test period) for a quick experiment.
'''

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ai_core.models.transformer_summarizer import (
    TransformerSummarizer,
    TransformerSummarizerConfig,
)


@dataclass
class StepDPrimeConfig:
    symbol: str
    mode: str = "sim"  # sim/live/display/ops etc.
    output_root: str = "output"
    stepA_root: Optional[str] = None  # default: <output_root>/stepA/<mode>
    stepB_root: Optional[str] = None  # default: <output_root>/stepB/<mode>
    stepDprime_root: Optional[str] = None  # default: <output_root>/stepD_prime/<mode>

    sources: Tuple[str, ...] = ("mamba_periodic", "mamba")
    horizons: Tuple[int, ...] = (1, 5, 10, 20)

    # sequence building
    max_len: int = 20  # pad/truncate to this

    # model/training
    seed: int = 42
    device: str = "auto"  # auto/cpu/cuda
    embedding_dim: int = 32
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    batch_size: int = 64
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-3
    val_ratio: float = 0.15
    early_stop_patience: int = 5

    # training data selection
    fit_split: str = "train"  # train / test / train+test / available
    # Leak guard / debug export controls
    # By default, StepD' embeddings CSV will NOT contain any label/target columns.
    # Use export_labels_in_embeddings=True only for debugging; it is never consumed by StepE.
    export_labels_in_embeddings: bool = False
    # Optional: export labels into a separate meta CSV under <stepDprime_root>/meta/
    # choices: 'none' or 'labels'
    export_meta_mode: str = "none"
    verbose: bool = True


class _SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, M: np.ndarray, y: np.ndarray, dates: List[str]):
        self.X = torch.from_numpy(X).float()
        self.M = torch.from_numpy(M).long()
        self.y = torch.from_numpy(y).long()
        self.dates = dates

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.M[idx], self.y[idx], self.dates[idx]


def _set_seed(seed: Optional[int]) -> None:
    import random
    if seed is None:
        seed = 42
    print(f"[StepDPrime] seed={seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _pick_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _read_prices(stepA_dir: Path, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_path = stepA_dir / f"stepA_prices_train_{symbol}.csv"
    test_path = stepA_dir / f"stepA_prices_test_{symbol}.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"missing StepA train prices: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"missing StepA test prices: {test_path}")
    tr = pd.read_csv(train_path)
    te = pd.read_csv(test_path)
    for df in (tr, te):
        if "Date" not in df.columns or "Close" not in df.columns:
            raise ValueError("StepA prices CSV must include Date, Close")
        df["Date"] = pd.to_datetime(df["Date"])
    return tr, te


def _date_to_key(dt: pd.Timestamp) -> str:
    return dt.strftime("%Y-%m-%d")


def _build_close_map(pr_train: pd.DataFrame, pr_test: pd.DataFrame) -> Dict[str, float]:
    df = pd.concat([pr_train[["Date", "Close"]], pr_test[["Date", "Close"]]], axis=0, ignore_index=True)
    df = df.drop_duplicates(subset=["Date"]).sort_values("Date")
    return {_date_to_key(d): float(c) for d, c in zip(df["Date"], df["Close"])}


def _infer_train_end(pr_train: pd.DataFrame) -> str:
    return _date_to_key(pr_train["Date"].max())


def _collect_daily_pred_files(stepB_dir: Path, symbol: str, source: str, horizon: int) -> List[Path]:
    hh = f"{horizon:02d}"
    if source == "mamba_periodic":
        pat = f"stepB_daily_pred_mamba_periodic_h{hh}_{symbol}_*.csv"
        root = stepB_dir / "daily_periodic"
    elif source == "mamba":
        pat = f"stepB_daily_pred_mamba_h{hh}_{symbol}_*.csv"
        root = stepB_dir / "daily"
    else:
        raise ValueError(f"unsupported source: {source}")
    if not root.exists():
        return []
    return sorted(root.glob(pat))


def _pred_close_column(source: str, horizon: int) -> str:
    if source == "mamba_periodic":
        return f"Pred_Close_MAMBA_PERIODIC_h{horizon:02d}"
    if source == "mamba":
        return f"Pred_Close_MAMBA_h{horizon:02d}"
    raise ValueError(f"unsupported source: {source}")


def _load_pred_close_aggregate(stepB_dir: Path, symbol: str, source: str) -> Optional[pd.DataFrame]:
    if source == "mamba_periodic":
        name = f"stepB_pred_close_mamba_periodic_{symbol}.csv"
    elif source == "mamba":
        name = f"stepB_pred_close_mamba_{symbol}.csv"
    else:
        raise ValueError(f"unsupported source: {source}")

    cands = [stepB_dir / name] + sorted(stepB_dir.glob(f"**/{name}"))
    for p in cands:
        if p.exists():
            df = pd.read_csv(p)
            if "Date_anchor" not in df.columns:
                raise ValueError(f"{p} missing Date_anchor")
            return df
    return None


def _load_path_csv(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    req = ["Date_anchor", "step_ahead_bdays", "Date_target", "Pred_ret_from_anchor"]
    for c in req:
        if c not in df.columns:
            raise ValueError(f"{p} missing column: {c}")
    return df


def _path_to_sequence(df: pd.DataFrame, max_len: int) -> Tuple[np.ndarray, np.ndarray, str, str]:
    df = df.sort_values("step_ahead_bdays").reset_index(drop=True)
    anchor = str(df.loc[0, "Date_anchor"])
    target_last = str(df.loc[df.index.max(), "Date_target"])

    cum = df["Pred_ret_from_anchor"].astype(float).to_numpy()
    inc = np.empty_like(cum)
    inc[0] = cum[0]
    inc[1:] = cum[1:] - cum[:-1]

    T = len(cum)
    T_eff = min(T, max_len)

    x = np.zeros((max_len, 2), dtype=np.float32)
    m = np.zeros((max_len,), dtype=np.int64)
    x[:T_eff, 0] = inc[:T_eff]
    x[:T_eff, 1] = cum[:T_eff]
    m[:T_eff] = 1
    return x, m, anchor, target_last


def _make_label(close_map: Dict[str, float], anchor: str, target: str) -> Optional[int]:
    ca = close_map.get(anchor)
    ct = close_map.get(target)
    if ca is None or ct is None:
        return None
    ret = math.log(ct / ca) if ca > 0 and ct > 0 else (ct - ca)
    return 1 if ret > 0 else 0


def _build_calendar_index(pr_train: pd.DataFrame, pr_test: pd.DataFrame) -> Tuple[List[str], Dict[str, int]]:
    cal = pd.concat([pr_train[["Date"]], pr_test[["Date"]]], ignore_index=True)
    cal = cal.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    keys = [_date_to_key(dt) for dt in cal["Date"]]
    return keys, {k: int(i) for i, k in enumerate(keys)}


def _shift_calendar_date(calendar_keys: List[str], calendar_idx: Dict[str, int], offset: int, anchor: str) -> Optional[str]:
    i = calendar_idx.get(anchor)
    if i is None:
        return None
    j = int(i + offset)
    if j < 0 or j >= len(calendar_keys):
        return None
    return calendar_keys[j]


def _build_sequences_from_pred_close(
    df: pd.DataFrame,
    source: str,
    horizons: Tuple[int, ...],
    max_len: int,
    close_map: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if df is None or len(df) == 0:
        return np.zeros((0, max_len, 2), dtype=np.float32), np.zeros((0, max_len), dtype=np.int64), []

    work = df.copy()
    work["Date_anchor"] = pd.to_datetime(work["Date_anchor"], errors="coerce")
    work = work.dropna(subset=["Date_anchor"]).sort_values("Date_anchor").reset_index(drop=True)

    X_list, M_list, dates = [], [], []
    for _, row in work.iterrows():
        anchor = _date_to_key(pd.to_datetime(row["Date_anchor"]))
        close_anchor = close_map.get(anchor)
        if close_anchor is None or close_anchor == 0:
            continue

        cum = []
        for h in horizons:
            col = _pred_close_column(source, int(h))
            if col not in work.columns:
                continue
            pred_close = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
            if pd.isna(pred_close):
                continue
            cum.append(float(pred_close) / float(close_anchor) - 1.0)

        if not cum:
            continue

        cum = np.asarray(cum, dtype=np.float32)
        inc = np.empty_like(cum)
        inc[0] = cum[0]
        if len(cum) > 1:
            inc[1:] = cum[1:] - cum[:-1]

        t_eff = min(len(cum), int(max_len))
        x = np.zeros((int(max_len), 2), dtype=np.float32)
        m = np.zeros((int(max_len),), dtype=np.int64)
        x[:t_eff, 0] = inc[:t_eff]
        x[:t_eff, 1] = cum[:t_eff]
        m[:t_eff] = 1

        X_list.append(x)
        M_list.append(m)
        dates.append(anchor)

    if not X_list:
        return np.zeros((0, max_len, 2), dtype=np.float32), np.zeros((0, max_len), dtype=np.int64), []
    return np.stack(X_list, axis=0), np.stack(M_list, axis=0), dates


def _train_one(
    cfg: StepDPrimeConfig,
    X: np.ndarray,
    M: np.ndarray,
    y: np.ndarray,
    out_dir: Path,
    model_key: str,
    device: torch.device,
) -> Tuple[TransformerSummarizer, Dict[str, float]]:
    n = X.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_val = max(1, int(n * cfg.val_ratio)) if n >= 10 else max(1, min(3, n // 3))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    if len(tr_idx) < 1:
        tr_idx = val_idx
        val_idx = idx[:0]

    Xtr, Mtr, ytr = X[tr_idx], M[tr_idx], y[tr_idx]
    Xva, Mva, yva = (X[val_idx], M[val_idx], y[val_idx]) if len(val_idx) else (None, None, None)

    ds_tr = _SeqDataset(Xtr, Mtr, ytr, dates=[""] * len(ytr))
    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    dl_va = None
    if Xva is not None:
        ds_va = _SeqDataset(Xva, Mva, yva, dates=[""] * len(yva))
        dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    mcfg = TransformerSummarizerConfig(
        feature_dim=int(X.shape[2]),
        max_len=int(cfg.max_len),
        d_model=int(cfg.d_model),
        nhead=int(cfg.nhead),
        num_layers=int(cfg.num_layers),
        dropout=float(cfg.dropout),
        embedding_dim=int(cfg.embedding_dim),
        num_classes=2,
        use_cls_token=False,
    )
    model = TransformerSummarizer(mcfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_val = float("inf")
    best_state = None
    bad = 0

    def _eval(dl):
        model.eval()
        tot = 0.0
        cnt = 0
        correct = 0
        with torch.no_grad():
            for xb, mb, yb, _ in dl:
                xb = xb.to(device)
                mb = mb.to(device)
                yb = yb.to(device)
                logits, _ = model(xb, mb, return_embedding=True)
                loss = loss_fn(logits, yb)
                tot += float(loss.item()) * len(yb)
                cnt += len(yb)
                pred = torch.argmax(logits, dim=1)
                correct += int((pred == yb).sum().item())
        return (tot / max(1, cnt)), (correct / max(1, cnt))

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for ep in range(1, cfg.epochs + 1):
        model.train()
        tot = 0.0
        cnt = 0
        correct = 0
        for xb, mb, yb, _ in dl_tr:
            xb = xb.to(device)
            mb = mb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits, _ = model(xb, mb, return_embedding=True)
            loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tot += float(loss.item()) * len(yb)
            cnt += len(yb)
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == yb).sum().item())

        tr_loss = tot / max(1, cnt)
        tr_acc = correct / max(1, cnt)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)

        if dl_va is not None:
            va_loss, va_acc = _eval(dl_va)
            history["val_loss"].append(va_loss)
            history["val_acc"].append(va_acc)

            if cfg.verbose:
                print(f"[StepD'] {model_key} ep={ep:03d} train_loss={tr_loss:.4f} acc={tr_acc:.3f} val_loss={va_loss:.4f} val_acc={va_acc:.3f}")

            if va_loss + 1e-6 < best_val:
                best_val = va_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= cfg.early_stop_patience:
                    if cfg.verbose:
                        print(f"[StepD'] {model_key} early stop at ep={ep}")
                    break
        else:
            if cfg.verbose:
                print(f"[StepD'] {model_key} ep={ep:03d} train_loss={tr_loss:.4f} acc={tr_acc:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{model_key}.pt"
    meta_path = out_dir / f"{model_key}.json"

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(mcfg),
            "model_key": model_key,
            "history": history,
            "feature_dim": int(X.shape[2]),
            "max_len": int(cfg.max_len),
        },
        ckpt_path,
    )
    meta = {
        "model_key": model_key,
        "ckpt_path": str(ckpt_path.as_posix()),
        "config": asdict(mcfg),
        "history_last": {k: (v[-1] if v else None) for k, v in history.items()},
        "n_samples": int(X.shape[0]),
        "best_val_loss": None if best_state is None else float(best_val),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    metrics = {
        "train_loss_last": float(history["train_loss"][-1]) if history["train_loss"] else float("nan"),
        "train_acc_last": float(history["train_acc"][-1]) if history["train_acc"] else float("nan"),
        "val_loss_best": float(best_val) if best_state is not None else float("nan"),
    }
    return model, metrics

class StepDPrimeService:
    def __init__(self):
        pass

    def run(self, cfg: StepDPrimeConfig) -> Dict[str, object]:
        _set_seed(cfg.seed)
        device = _pick_device(cfg.device)

        out_root = Path(cfg.output_root)
        stepA_dir = Path(cfg.stepA_root) if cfg.stepA_root else (out_root / "stepA" / cfg.mode)
        stepB_dir = Path(cfg.stepB_root) if cfg.stepB_root else (out_root / "stepB" / cfg.mode)
        stepD_dir = Path(cfg.stepDprime_root) if cfg.stepDprime_root else (out_root / "stepD_prime" / cfg.mode)

        pr_train, pr_test = _read_prices(stepA_dir, cfg.symbol)
        close_map = _build_close_map(pr_train, pr_test)
        train_end = _infer_train_end(pr_train)
        calendar_keys, calendar_idx = _build_calendar_index(pr_train, pr_test)

        results = {
            "mode": cfg.mode,
            "symbol": cfg.symbol,
            "output_dir": str(stepD_dir.as_posix()),
            "train_end": train_end,
            "models": {},
            "embeddings": {},
            "warnings": [],
        }

        models_dir = stepD_dir / "models"
        embeds_dir = stepD_dir / "embeddings"
        models_dir.mkdir(parents=True, exist_ok=True)
        embeds_dir.mkdir(parents=True, exist_ok=True)

        for source in cfg.sources:
            daily_files_by_h = {int(h): _collect_daily_pred_files(stepB_dir, cfg.symbol, source, int(h)) for h in cfg.horizons}
            use_daily = any(len(v) > 0 for v in daily_files_by_h.values())

            if use_daily:
                for h in cfg.horizons:
                    files = daily_files_by_h[int(h)]
                    if cfg.verbose:
                        print(f"[StepD'] collect: source={source} h={h} files={len(files)} root={stepB_dir.as_posix()}")
                    if not files:
                        results["warnings"].append(f"no files for source={source} h={h} under {stepB_dir}")
                        continue

                    X_list, M_list, y_list, avail_list, dates = [], [], [], [], []
                    for fp in files:
                        try:
                            df = _load_path_csv(fp)
                            x, m, anchor, target = _path_to_sequence(df, cfg.max_len)

                            label = _make_label(close_map, anchor, target)
                            if label is None:
                                y_list.append(np.nan)
                                avail_list.append(False)
                            else:
                                y_list.append(int(label))
                                avail_list.append(True)

                            X_list.append(x)
                            M_list.append(m)
                            dates.append(anchor)
                        except Exception as e:
                            results["warnings"].append(f"failed read {fp}: {e}")
                            continue
                    if not X_list:
                        results["warnings"].append(f"no valid samples for source={source} h={h}")
                        continue

                    X = np.stack(X_list, axis=0)
                    M = np.stack(M_list, axis=0)
                    y = np.asarray(y_list, dtype=np.float32)
                    label_avail = np.asarray(avail_list, dtype=bool)

                    anchors = np.array(dates)
                    is_train = anchors <= train_end
                    is_test = anchors > train_end

                    if cfg.fit_split == "train":
                        sel = is_train & label_avail
                    elif cfg.fit_split == "test":
                        sel = is_test & label_avail
                    elif cfg.fit_split in {"train+test", "available"}:
                        sel = label_avail
                    else:
                        raise ValueError(f"invalid fit_split: {cfg.fit_split}")

                    if not np.any(sel):
                        results["warnings"].append(
                            f"fit_split={cfg.fit_split} yielded 0 labeled samples for source={source} h={h}. "
                            f"Fallback to fit_split=available for this model."
                        )
                        sel = label_avail

                    Xfit, Mfit = X[sel], M[sel]
                    yfit = y[sel].astype(np.int64)

                    model_key = f"stepDprime_{source}_h{h:02d}_{cfg.symbol}"
                    model, metrics = _train_one(cfg, Xfit, Mfit, yfit, models_dir, model_key, device)

                    model.eval()
                    embs = []
                    with torch.no_grad():
                        bs = 256
                        for i in range(0, X.shape[0], bs):
                            xb = torch.from_numpy(X[i:i+bs]).float().to(device)
                            mb = torch.from_numpy(M[i:i+bs]).long().to(device)
                            e = model.encode(xb, mb).detach().cpu().numpy()
                            embs.append(e)
                    E = np.concatenate(embs, axis=0)

                    out_csv = embeds_dir / f"{model_key}_embeddings.csv"
                    out_df = pd.DataFrame(E, columns=[f"emb_{k:03d}" for k in range(E.shape[1])])
                    out_df.insert(0, "Date", dates)
                    out_df.insert(1, "label_up", y)
                    out_df.insert(2, "label_available", label_avail.astype(int))
                    out_df.insert(3, "source", source)
                    out_df.insert(4, "horizon_model", int(h))
                    out_df.to_csv(out_csv, index=False)

                    results["models"][model_key] = {
                        "source": source,
                        "horizon": int(h),
                        "n_samples_all": int(X.shape[0]),
                        "n_labels_available": int(label_avail.sum()),
                        "n_samples_fit": int(Xfit.shape[0]),
                        "metrics": metrics,
                        "ckpt": str((models_dir / f"{model_key}.pt").as_posix()),
                        "meta": str((models_dir / f"{model_key}.json").as_posix()),
                    }
                    results["embeddings"][model_key] = str(out_csv.as_posix())
                continue

            agg = _load_pred_close_aggregate(stepB_dir, cfg.symbol, source)
            if agg is None:
                results["warnings"].append(f"no StepB inputs for source={source} under {stepB_dir}")
                continue
            if cfg.verbose:
                print(f"[StepD'] fallback aggregate: source={source} rows={len(agg)} root={stepB_dir.as_posix()}")

            X, M, dates = _build_sequences_from_pred_close(
                df=agg,
                source=source,
                horizons=cfg.horizons,
                max_len=cfg.max_len,
                close_map=close_map,
            )
            if X.shape[0] == 0:
                results["warnings"].append(f"aggregate pred_close produced 0 usable samples for source={source}")
                continue

            anchors = np.array(dates)
            is_train = anchors <= train_end
            is_test = anchors > train_end

            for h in cfg.horizons:
                y_list, avail_list = [], []
                for anchor in dates:
                    target = _shift_calendar_date(calendar_keys, calendar_idx, int(h), anchor)
                    label = _make_label(close_map, anchor, target) if target is not None else None
                    if label is None:
                        y_list.append(np.nan)
                        avail_list.append(False)
                    else:
                        y_list.append(int(label))
                        avail_list.append(True)

                y = np.asarray(y_list, dtype=np.float32)
                label_avail = np.asarray(avail_list, dtype=bool)

                if cfg.fit_split == "train":
                    sel = is_train & label_avail
                elif cfg.fit_split == "test":
                    sel = is_test & label_avail
                elif cfg.fit_split in {"train+test", "available"}:
                    sel = label_avail
                else:
                    raise ValueError(f"invalid fit_split: {cfg.fit_split}")

                if not np.any(sel):
                    results["warnings"].append(
                        f"fit_split={cfg.fit_split} yielded 0 labeled samples for source={source} h={h}. "
                        f"Fallback to fit_split=available for this model."
                    )
                    sel = label_avail
                if not np.any(sel):
                    results["warnings"].append(f"no labeled samples for source={source} h={h} from aggregate pred_close")
                    continue

                Xfit, Mfit = X[sel], M[sel]
                yfit = y[sel].astype(np.int64)

                model_key = f"stepDprime_{source}_h{h:02d}_{cfg.symbol}"
                model, metrics = _train_one(cfg, Xfit, Mfit, yfit, models_dir, model_key, device)

                model.eval()
                embs = []
                with torch.no_grad():
                    bs = 256
                    for i in range(0, X.shape[0], bs):
                        xb = torch.from_numpy(X[i:i+bs]).float().to(device)
                        mb = torch.from_numpy(M[i:i+bs]).long().to(device)
                        e = model.encode(xb, mb).detach().cpu().numpy()
                        embs.append(e)
                E = np.concatenate(embs, axis=0)

                out_csv = embeds_dir / f"{model_key}_embeddings.csv"
                out_df = pd.DataFrame(E, columns=[f"emb_{k:03d}" for k in range(E.shape[1])])
                out_df.insert(0, "Date", dates)
                out_df.insert(1, "label_up", y)
                out_df.insert(2, "label_available", label_avail.astype(int))
                out_df.insert(3, "source", source)
                out_df.insert(4, "horizon_model", int(h))
                out_df.to_csv(out_csv, index=False)

                results["models"][model_key] = {
                    "source": source,
                    "horizon": int(h),
                    "n_samples_all": int(X.shape[0]),
                    "n_labels_available": int(label_avail.sum()),
                    "n_samples_fit": int(Xfit.shape[0]),
                    "metrics": metrics,
                    "ckpt": str((models_dir / f"{model_key}.pt").as_posix()),
                    "meta": str((models_dir / f"{model_key}.json").as_posix()),
                }
                results["embeddings"][model_key] = str(out_csv.as_posix())

        summary_path = stepD_dir / f"stepDprime_summary_{cfg.symbol}.json"
        summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        if cfg.verbose:
            print(f"[StepD'] wrote summary: {summary_path.as_posix()}")
        return results
