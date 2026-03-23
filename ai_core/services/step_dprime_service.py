from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils.extmath import randomized_svd

from ai_core.utils.manifest_path_utils import normalize_output_artifact_path
from ai_core.utils.timing_logger import TimingLogger
from ai_core.services.stepdprime_path_utils import resolve_stepdprime_dir
from ai_core.services.dprime_cluster_components import (
    ClusterRuntimeConfig,
    DPrimeClusterService,
)

_LAST_DPRIME_PROFILE = ""
_DPRIME_LOGGER = logging.getLogger(__name__)


def _log_dprime(msg: str) -> None:
    _DPRIME_LOGGER.info(msg)
    print(msg)


def _log_timing(label: str, started_at: float) -> None:
    elapsed = time.perf_counter() - started_at
    _log_dprime(f"[DPRIME][TIMING] {label}={elapsed:.4f}s")


def _cuda_available() -> bool:
    if str(os.environ.get("DPRIME_FORCE_CPU", "0")).strip().lower() in {"1", "true", "yes", "on"}:
        return False
    return bool(torch.cuda.is_available())

_PROFILES: Tuple[str, ...] = (
    "dprime_bnf_h01",
    "dprime_bnf_h02",
    "dprime_bnf_3scale",
    "dprime_mix_h01",
    "dprime_mix_h02",
    "dprime_mix_3scale",
    "dprime_all_features_h01",
    "dprime_all_features_h02",
    "dprime_all_features_h03",
    "dprime_all_features_3scale",
)


@dataclass
class StepDPrimeConfig:
    symbol: str
    mode: str = "sim"
    output_root: str = "output"
    stepA_root: Optional[str] = None
    stepB_root: Optional[str] = None
    stepC_root: Optional[str] = None
    stepDprime_root: Optional[str] = None  # explicit canonical StepDPrime dir; write path prefers this when set
    legacy_stepDprime_root: Optional[str] = None  # legacy read fallback only; new writes never target stepD_prime
    profiles: Tuple[str, ...] = _PROFILES
    l_past: int = 63
    pred_k: int = 20
    z_past_dim: int = 32
    z_pred_dim: int = 32
    z_state_dim: int = 64
    verbose: bool = True
    encoder_type: str = "legacy"
    transformer_d_model: int = 64
    transformer_nhead: int = 4
    transformer_num_layers: int = 2
    transformer_ff_dim: int = 128
    transformer_dropout: float = 0.1
    transformer_epochs: int = 4
    transformer_batch_size: int = 64
    transformer_lr: float = 1e-3
    transformer_mask_ratio: float = 0.15
    transformer_seed: int = 42
    feature_version: str = "stepdprime_rl_v2"
    cluster_source_version: str = "stepA_only_cluster_v1"
    # cluster regime scaffold (some options are placeholders / not yet wired)
    enable_cluster_regime: bool = True
    enable_cluster_monthly_refit: bool = True
    enable_cluster_daily_assign: bool = True
    enable_cluster_in_rl_state: bool = True
    cluster_backend: str = "ticc"
    cluster_raw_k: int = 20
    cluster_k_eff_min: int = 12
    cluster_small_share_threshold: float = 0.01
    cluster_small_mean_run_threshold: float = 3.0
    cluster_short_window_days: int = 20
    cluster_mid_window_weeks: int = 8
    cluster_long_window_months: int = 6
    cluster_enable_8y_context: bool = True
    cluster_rare_flag_enabled: bool = True
    timing_logger: Optional[TimingLogger] = None


def _normalize_mode(mode: str) -> str:
    m = str(mode or "sim").strip().lower()
    if m in {"ops", "op", "prod", "production", "real"}:
        return "live"
    if m not in {"sim", "live", "display"}:
        raise ValueError(f"StepDPrime invalid mode={mode}. expected sim/live/display")
    return m


def _read_split_summary(stepa_dir: Path, symbol: str) -> Dict[str, str]:
    p = stepa_dir / f"stepA_split_summary_{symbol}.csv"
    if not p.exists():
        raise FileNotFoundError(f"missing StepA split summary: {p}")
    df = pd.read_csv(p)
    if {"key", "value"}.issubset(df.columns):
        kv = {str(k): str(v) for k, v in zip(df["key"], df["value"])}
    else:
        kv = {c: str(df.iloc[0][c]) for c in df.columns}
    req = ["train_start", "train_end", "test_start", "test_end"]
    out = {}
    for k in req:
        if k not in kv or kv[k] in {"None", "nan", "NaT"}:
            raise ValueError(f"split summary missing {k}")
        out[k] = str(pd.to_datetime(kv[k]).date())
    return out


def _read_pair(base: Path, stem: str, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    p_tr = base / f"{stem}_train_{symbol}.csv"
    p_te = base / f"{stem}_test_{symbol}.csv"
    if not (p_tr.exists() and p_te.exists()):
        raise FileNotFoundError(f"missing {stem} pair: {p_tr} / {p_te}")
    tr, te = pd.read_csv(p_tr), pd.read_csv(p_te)
    for df in (tr, te):
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    return tr, te


def _utcnow_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_split_summary_csv(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if {"key", "value"}.issubset(df.columns):
        return {str(k): str(v) for k, v in zip(df["key"], df["value"])}
    if len(df.index) > 0:
        return {str(c): str(df.iloc[0][c]) for c in df.columns}
    return {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _normalize_output_path(raw_path: str | Path | None, *, output_root: Path) -> str:
    return normalize_output_artifact_path(raw_path, output_root=output_root)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _parse_profile(profile: str) -> Tuple[str, str]:
    return _infer_family(profile), _infer_pred_type(profile)


def _parse_available_horizons(raw: Any) -> List[int]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        out: List[int] = []
        for item in raw:
            try:
                out.append(int(item))
            except Exception:
                continue
        return sorted(set(out))

    s = str(raw or "").strip()
    if not s:
        return []
    out = []
    for part in s.replace("|", ",").split(","):
        token = part.strip()
        if not token:
            continue
        try:
            out.append(int(token))
        except Exception:
            continue
    return sorted(set(out))


def _cleanup_running_marker(marker_dir: Path, marker_name: str) -> None:
    running = marker_dir / f"{marker_name}.RUNNING.json"
    if running.exists():
        running.unlink()


def _cleanup_running_markers(marker_dir: Path, *, prefixes: Optional[Sequence[str]] = None) -> None:
    wanted = tuple(str(p) for p in (prefixes or ()))
    for path in marker_dir.glob("*.RUNNING.json"):
        if wanted and not any(path.name.startswith(prefix) for prefix in wanted):
            continue
        path.unlink()


def _safe(df: pd.DataFrame, c: str, default: float = 0.0) -> pd.Series:
    if c in df.columns:
        return pd.to_numeric(df[c], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def _compute_base_features(prices: pd.DataFrame, tech: pd.DataFrame) -> pd.DataFrame:
    df = prices.sort_values("Date").reset_index(drop=True).copy()
    close = _safe(df, "Close")
    open_ = _safe(df, "Open")
    high = _safe(df, "High")
    low = _safe(df, "Low")
    vol = _safe(df, "Volume")
    prev_close = close.shift(1)
    prev_vol = vol.shift(1)

    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=14).mean()
    rng = (high - low).replace(0, np.nan)
    macd = _safe(tech, "MACD")
    macd_sig = _safe(tech, "MACD_signal")
    macd_hist = _safe(tech, "MACD_hist") if "MACD_hist" in tech.columns else (macd - macd_sig)

    out = pd.DataFrame({"Date": df["Date"]})
    out["ret_1"] = close.pct_change(1)
    out["ret_5"] = close.pct_change(5)
    out["ret_20"] = close.pct_change(20)
    out["range_atr"] = (high - low) / atr14.replace(0, np.nan)
    out["body_ratio"] = (close - open_) / rng
    out["body_atr"] = (close - open_).abs() / atr14.replace(0, np.nan)
    out["upper_wick_ratio"] = (high - np.maximum(open_, close)).clip(lower=0) / rng
    out["lower_wick_ratio"] = (np.minimum(open_, close) - low).clip(lower=0) / rng
    out["Gap"] = _safe(tech, "Gap") if "Gap" in tech.columns else (open_ / prev_close.replace(0, np.nan) - 1.0)
    out["ATR_norm"] = _safe(tech, "ATR_norm") if "ATR_norm" in tech.columns else (atr14 / prev_close.replace(0, np.nan))
    out["gap_atr"] = out["Gap"] / out["ATR_norm"].replace(0, np.nan)
    out["vol_log_ratio_20"] = np.log(vol.replace(0, np.nan)) - np.log(vol.rolling(20, min_periods=20).mean().replace(0, np.nan))
    out["vol_chg"] = vol.pct_change(1)
    z_mu = close.rolling(25, min_periods=25).mean()
    z_sd = close.rolling(25, min_periods=25).std(ddof=0).replace(0, np.nan)
    out["dev_z_25"] = (close - z_mu) / z_sd
    out["bnf_score"] = out["dev_z_25"] * out["vol_log_ratio_20"]
    out["RSI"] = _safe(tech, "RSI") if "RSI" in tech.columns else _safe(tech, "RSI_14")
    out["MACD_hist"] = macd_hist
    out["macd_hist_delta"] = out["MACD_hist"].diff()
    mh = out["MACD_hist"].fillna(0.0)
    out["macd_hist_cross_up"] = ((mh > 0.0) & (mh.shift(1) <= 0.0)).astype(float)
    out["clv"] = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    out["distribution_day"] = ((close < prev_close) & (vol > prev_vol)).astype(float)
    out["dist_count_25"] = out["distribution_day"].rolling(25, min_periods=1).sum()
    out["absorption_day"] = ((close > prev_close) & (vol < prev_vol)).astype(float)
    mf_mult = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mfv = mf_mult * vol
    out["cmf_20"] = mfv.rolling(20, min_periods=20).sum() / vol.rolling(20, min_periods=20).sum().replace(0, np.nan)

    for c in out.columns:
        if c != "Date":
            out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _sanitize_matrix(
    x: np.ndarray,
    *,
    name: str,
    fill_value: float = 0.0,
    clip_abs: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    arr = np.asarray(x, dtype=float)
    nonfinite_before = int(arr.size - int(np.isfinite(arr).sum()))
    arr = np.nan_to_num(arr, nan=fill_value, posinf=fill_value, neginf=fill_value)
    if clip_abs is not None:
        arr = np.clip(arr, -float(clip_abs), float(clip_abs))
    nonfinite_after = int(arr.size - int(np.isfinite(arr).sum()))
    zero_var_cols = 0
    if arr.ndim == 2 and arr.shape[1] > 0:
        zero_var_cols = int(np.count_nonzero(np.std(arr, axis=0) < 1e-8))
    diag = {
        "name": name,
        "shape": [int(v) for v in arr.shape],
        "nonfinite_count_before": nonfinite_before,
        "nonfinite_count_after": nonfinite_after,
        "zero_var_cols": zero_var_cols,
    }
    return arr, diag


def _fit_scaler(train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    train_arr, _ = _sanitize_matrix(train, name="fit_scaler.train", clip_abs=1e6)
    mu = train_arr.mean(axis=0)
    sd = train_arr.std(axis=0)
    mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    sd = np.nan_to_num(sd, nan=1.0, posinf=1.0, neginf=1.0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu, sd


def _fit_pca(train: np.ndarray, dim: int) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    train_san, train_diag = _sanitize_matrix(train, name="fit_pca.train", clip_abs=1e6)
    if train_san.ndim != 2 or train_san.shape[0] < 2:
        raise ValueError("too few train rows for PCA fit")

    x = train_san - train_san.mean(axis=0, keepdims=True)
    var = np.var(x, axis=0)
    keep = var >= 1e-12
    keep_count = int(np.count_nonzero(keep))
    if keep_count <= 0:
        d = int(min(dim, max(1, min(train_san.shape[0] - 1, train_san.shape[1]))))
        comp = np.zeros((train_san.shape[1], d), dtype=float)
        if train_san.shape[1] > 0:
            for j in range(d):
                comp[j % train_san.shape[1], j] = 1.0
        diag = {
            "train": train_diag,
            "pca_method_used": "degenerate_identity",
            "fallback_used": True,
            "rankable_dim": 0,
            "dim_requested": int(dim),
            "dim_used": int(d),
            "zero_var_cols_dropped": int(train_san.shape[1]),
        }
        return comp, d, diag

    x_reduced = x[:, keep]
    rankable_dim = min(x_reduced.shape[0] - 1, x_reduced.shape[1])
    d = int(min(dim, rankable_dim))
    if d <= 0:
        d = 1

    method = "numpy_svd"
    fallback_used = False
    try:
        use_cuda = _cuda_available() and int(x_reduced.size) >= 200_000
        if use_cuda:
            _log_dprime("[DPRIME][DEVICE] pca_device=cuda")
            xt = torch.as_tensor(x_reduced, dtype=torch.float64, device="cuda")
            _, _, vt = torch.linalg.svd(xt, full_matrices=False)
            comp_reduced = vt[:d, :].transpose(0, 1).detach().cpu().numpy().astype(float)
            method = "torch_cuda_svd"
        else:
            _, _, vt = np.linalg.svd(x_reduced, full_matrices=False)
            comp_reduced = vt[:d].T.astype(float)
    except Exception as exc:
        fallback_used = True
        method = "randomized_svd"
        _log_dprime(f"[DPRIME][DEVICE] PCA fallback to CPU randomized_svd reason={type(exc).__name__}: {exc}")
        _, _, vt = randomized_svd(x_reduced, n_components=d, random_state=42)
        comp_reduced = vt.T.astype(float)

    comp = np.zeros((train_san.shape[1], d), dtype=float)
    comp[keep, :] = comp_reduced
    comp, _ = _sanitize_matrix(comp, name="fit_pca.comp", clip_abs=1e6)
    diag = {
        "train": train_diag,
        "pca_method_used": method,
        "fallback_used": bool(fallback_used),
        "rankable_dim": int(rankable_dim),
        "dim_requested": int(dim),
        "dim_used": int(d),
        "zero_var_cols_dropped": int(train_san.shape[1] - keep_count),
    }
    return comp, d, diag


def _project(x: np.ndarray, mu: np.ndarray, sd: np.ndarray, comp: np.ndarray) -> np.ndarray:
    x_san, _ = _sanitize_matrix(x, name="project.x", clip_abs=1e6)
    mu_san = np.nan_to_num(np.asarray(mu, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    sd_san = np.nan_to_num(np.asarray(sd, dtype=float), nan=1.0, posinf=1.0, neginf=1.0)
    sd_san = np.where(sd_san < 1e-8, 1.0, sd_san)
    comp_san, _ = _sanitize_matrix(comp, name="project.comp", clip_abs=1e6)
    xn = (x_san - mu_san) / sd_san
    xn, _ = _sanitize_matrix(xn, name="project.xn", clip_abs=1e6)
    use_cuda = _cuda_available() and int(xn.size + comp_san.size) >= 200_000
    if use_cuda:
        try:
            _log_dprime("[DPRIME][DEVICE] projection_device=cuda")
            xn_t = torch.as_tensor(xn, dtype=torch.float64, device="cuda")
            comp_t = torch.as_tensor(comp_san, dtype=torch.float64, device="cuda")
            out = (xn_t @ comp_t).detach().cpu().numpy()
        except Exception as exc:
            _log_dprime(f"[DPRIME][DEVICE] projection fallback to CPU reason={type(exc).__name__}: {exc}")
            out = xn @ comp_san
    else:
        out = xn @ comp_san
    out, _ = _sanitize_matrix(out, name="project.out", clip_abs=1e6)
    return out


def _collect_horizon_cols(df: pd.DataFrame) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for c in df.columns:
        lc = str(c).lower()
        if "pred_close_t_plus_" in lc:
            try:
                h = int(lc.split("pred_close_t_plus_")[-1])
                out[h] = c
            except Exception:
                continue
            continue
        for pfx in ("pred_close_mamba_h", "close_pred_h", "pred_close_mamba_periodic_h"):
            if pfx in lc:
                try:
                    h = int(lc.split(pfx)[-1])
                    out[h] = c
                except Exception:
                    pass
    if 1 not in out:
        for c in df.columns:
            if str(c).lower() in {"pred_close_mamba", "pred_close", "pred_close_mamba_periodic"}:
                out[1] = c
                break
    return out


def _normalize_pred_frame(df: pd.DataFrame, pred_k: int) -> Tuple[pd.DataFrame, List[int]]:
    dcol = "Date" if "Date" in df.columns else ("Date_anchor" if "Date_anchor" in df.columns else None)
    if dcol is None:
        raise ValueError("prediction source missing Date/Date_anchor")
    out = pd.DataFrame()
    out["Date"] = pd.to_datetime(df[dcol], errors="coerce").dt.normalize()
    hmap = _collect_horizon_cols(df)
    for h, src in hmap.items():
        if 1 <= h <= pred_k:
            out[f"Pred_Close_t_plus_{h:02d}"] = pd.to_numeric(df[src], errors="coerce")
    out = out.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    return out, sorted([h for h in hmap.keys() if 1 <= h <= pred_k])


def _build_pred_from_stepb(stepb_dir: Path, symbol: str, pred_k: int) -> Tuple[pd.DataFrame, Dict[str, object]]:
    cands_pathseq = sorted(stepb_dir.glob(f"stepB_pred_pathseq_*_h{pred_k:02d}_{symbol}.csv"))
    cands_path = sorted(stepb_dir.glob(f"stepB_pred_path_mamba_{symbol}.csv"))
    cands_time = sorted(stepb_dir.glob(f"stepB_pred_time_all_{symbol}.csv"))
    cands_periodic_close = sorted(stepb_dir.glob(f"stepB_pred_close_mamba_periodic_{symbol}.csv"))
    cands_close = sorted(stepb_dir.glob(f"stepB_pred_close_mamba_{symbol}.csv"))
    candidate_files = [str(p.as_posix()) for p in (cands_pathseq + cands_path + cands_time + cands_periodic_close + cands_close)]

    selected: Optional[Path] = None
    mode = ""
    if cands_pathseq:
        selected, mode = cands_pathseq[0], "pathseq"
        src_df = pd.read_csv(selected)
    elif cands_path:
        selected, mode = cands_path[0], "path_mamba"
        src_df = pd.read_csv(selected)
    elif cands_time:
        selected, mode = cands_time[0], "time_all"
        src_df = pd.read_csv(selected)
    elif cands_periodic_close or cands_close:
        mode = "periodic_fallback"
        selected = cands_periodic_close[0] if cands_periodic_close else cands_close[0]
        base_df = pd.read_csv(selected)
        if cands_periodic_close and cands_close:
            close_df = pd.read_csv(cands_close[0])
            src_df = base_df.merge(close_df, on="Date", how="outer", suffixes=("", "_close"))
        else:
            src_df = base_df
    else:
        raise FileNotFoundError(f"missing StepB prediction source under {stepb_dir}; candidates={candidate_files}")

    norm_df, available = _normalize_pred_frame(src_df, pred_k)
    if not available:
        raise FileNotFoundError(f"selected StepB source has no usable prediction horizons: {selected}")
    meta = {
        "pred_source_candidate_files": candidate_files,
        "pred_source_selected": str(selected.as_posix()) if selected else "",
        "pred_source_mode": mode,
        "pred_available_horizons": available,
    }
    return norm_df, meta




def _load_pred_time_priority(stepc_dir: Path, stepb_dir: Path, symbol: str) -> Tuple[Optional[pd.DataFrame], str]:
    for root, label in ((stepc_dir, "stepC"), (stepb_dir, "stepB")):
        if root is None or not root.exists():
            continue
        cands = sorted(root.glob(f"stepC_pred_time*_{symbol}.csv")) if label == "stepC" else sorted(root.glob(f"stepB_pred_time*_{symbol}.csv"))
        if not cands:
            continue
        p = cands[0]
        df = pd.read_csv(p)
        dcol = "Date" if "Date" in df.columns else ("Date_anchor" if "Date_anchor" in df.columns else None)
        if dcol is None:
            continue
        df["Date"] = pd.to_datetime(df[dcol], errors="coerce").dt.normalize()
        return df, str(p.as_posix())
    return None, ""

def _infer_pred_type(profile: str) -> str:
    if profile.endswith("_h01"):
        return "h01"
    if profile.endswith("_h02"):
        return "h02"
    if profile.endswith("_h03"):
        return "h03"
    return "3scale"


def _infer_family(profile: str) -> str:
    if "all_features" in profile:
        return "all_features"
    if "mix" in profile:
        return "mix"
    return "bnf"


def _normalize_encoder_type(value: str) -> str:
    token = str(value or "legacy").strip().lower()
    if token not in {"legacy", "transformer"}:
        raise ValueError(f"unsupported StepDPrime encoder_type={value!r}. expected legacy|transformer")
    return token


def _fit_sequence_scaler(train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(train, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"expected 3D array for sequence scaler, got shape={arr.shape}")
    flat = arr.reshape(-1, arr.shape[-1])
    mu = np.nan_to_num(flat.mean(axis=0), nan=0.0, posinf=0.0, neginf=0.0)
    sd = np.nan_to_num(flat.std(axis=0), nan=1.0, posinf=1.0, neginf=1.0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu.astype(float), sd.astype(float)


def _apply_sequence_scaler(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    out = (arr - np.asarray(mu, dtype=float).reshape(1, 1, -1)) / np.asarray(sd, dtype=float).reshape(1, 1, -1)
    out, _ = _sanitize_matrix(out.reshape(arr.shape[0], -1), name="seq_scale.out", clip_abs=1e6)
    return out.reshape(arr.shape[0], arr.shape[1], arr.shape[2])


_SCALAR_CONTEXT_CLOSE_CANDIDATES: Tuple[str, ...] = ("Close_anchor", "Close")
_SCALAR_CONTEXT_VOLUME_CANDIDATES: Tuple[str, ...] = ("Volume", "vol_log_ratio_20", "vol_chg", "BNF_RVOL20", "BNF_VolZ20")
_SCALAR_CONTEXT_REQUIRED_COLUMNS: Tuple[str, ...] = (
    "gap_atr",
    "ATR_norm",
    "ret_20",
)


def _series_from_frame_value(
    frame: pd.DataFrame,
    column: str,
    *,
    replace_zero_with_nan: bool = False,
) -> Optional[pd.Series]:
    if column not in frame.columns:
        return None
    raw = frame[column]
    if isinstance(raw, pd.DataFrame):
        if raw.shape[1] == 0:
            return None
        raw = raw.iloc[:, 0]
    if isinstance(raw, pd.Series):
        out = pd.to_numeric(raw, errors="coerce")
    else:
        scalar = pd.to_numeric(pd.Series([raw]), errors="coerce").iloc[0]
        out = pd.Series(scalar, index=frame.index, dtype=float)
    if replace_zero_with_nan:
        out = out.replace(0, np.nan)
    return out.astype(float)


def _pick_scalar_context_series(
    frame: pd.DataFrame,
    candidates: Sequence[str],
    *,
    replace_zero_with_nan: bool = False,
) -> Tuple[pd.Series, str]:
    for column in candidates:
        series = _series_from_frame_value(frame, column, replace_zero_with_nan=replace_zero_with_nan)
        if series is not None:
            return series, str(column)
    return pd.Series(np.nan, index=frame.index, dtype=float), ""


def _add_rl_scalar_context(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = data.copy()
    available_columns = [str(c) for c in out.columns]
    close, close_source = _pick_scalar_context_series(out, _SCALAR_CONTEXT_CLOSE_CANDIDATES, replace_zero_with_nan=True)
    volume, volume_source = _pick_scalar_context_series(out, _SCALAR_CONTEXT_VOLUME_CANDIDATES)
    if volume_source == "Volume":
        volume = volume.replace(0, np.nan)
    ret_1_raw = _series_from_frame_value(out, "ret_1")
    ret_1 = ret_1_raw if ret_1_raw is not None else pd.Series(np.nan, index=out.index, dtype=float)
    if ret_1.isna().all():
        ret_1 = close.pct_change(1)

    rolling_vol20 = ret_1.rolling(20, min_periods=5).std(ddof=0)
    rolling_vol63 = ret_1.rolling(63, min_periods=10).std(ddof=0)
    rolling_vol252 = ret_1.rolling(252, min_periods=20).std(ddof=0)
    ma20 = close.rolling(20, min_periods=5).mean()
    ma63 = close.rolling(63, min_periods=10).mean()
    ma252 = close.rolling(252, min_periods=20).mean()
    rolling_max252 = close.rolling(252, min_periods=20).max()
    volume_based_context_enabled = bool(volume_source)
    vol_ma20 = volume.rolling(20, min_periods=5).mean() if volume_based_context_enabled else pd.Series(np.nan, index=out.index, dtype=float)
    vol_ma63 = volume.rolling(63, min_periods=10).mean() if volume_based_context_enabled else pd.Series(np.nan, index=out.index, dtype=float)
    yearly_ret = close.pct_change(252)
    background_ret_8y = close.pct_change(252 * 8)

    out["ctx_volatility_20"] = rolling_vol20
    out["ctx_volatility_63"] = rolling_vol63
    out["ctx_volatility_252"] = rolling_vol252
    out["ctx_trend_20"] = close / ma20 - 1.0
    out["ctx_trend_63"] = close / ma63 - 1.0
    out["ctx_trend_252"] = close / ma252 - 1.0
    out["ctx_turnover_20"] = volume / vol_ma20 - 1.0 if volume_based_context_enabled else 0.0
    out["ctx_turnover_63"] = volume / vol_ma63 - 1.0 if volume_based_context_enabled else 0.0
    out["ctx_drawdown_252"] = close / rolling_max252 - 1.0
    out["ctx_long_ret_252"] = yearly_ret
    out["ctx_long_ret_8y"] = background_ret_8y
    out["ctx_long_vol_252"] = rolling_vol252
    gap_atr, _ = _pick_scalar_context_series(out, ("gap_atr",))
    atr_norm, _ = _pick_scalar_context_series(out, ("ATR_norm",))
    ret_20, _ = _pick_scalar_context_series(out, ("ret_20",))
    out["ctx_gap_atr"] = gap_atr
    out["ctx_atr_norm"] = atr_norm
    out["ctx_ret_20"] = ret_20

    for col in [c for c in out.columns if str(c).startswith("ctx_")]:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    missing_required_columns = [
        col for col in _SCALAR_CONTEXT_REQUIRED_COLUMNS
        if col not in out.columns
    ]
    warnings: List[str] = []
    if not close_source:
        warnings.append("missing close source; scalar trend/drawdown context will degrade to zero-filled defaults")
    if not volume_source:
        warnings.append("missing volume source and proxy columns; turnover context will be zero-filled")
    elif volume_source != "Volume":
        warnings.append(f"raw Volume unavailable; using proxy column {volume_source} for turnover context")
    if missing_required_columns:
        warnings.append(f"missing scalar context inputs: {', '.join(missing_required_columns)}")
    diagnostics = {
        "available_columns": available_columns,
        "close_source": close_source or None,
        "volume_source": volume_source or None,
        "volume_proxy_candidates": list(_SCALAR_CONTEXT_VOLUME_CANDIDATES),
        "volume_proxy_used": bool(volume_source and volume_source != "Volume"),
        "missing_required_columns": missing_required_columns,
        "warnings": warnings,
        "encoder_paths_checked": ["legacy", "transformer"],
    }
    summary = (
        f"[STEPDPRIME_SCALAR_CONTEXT] close_source={close_source or 'missing'} "
        f"volume_source={volume_source or 'missing'} "
        f"missing_required={missing_required_columns} "
        f"columns={len(available_columns)}"
    )
    _log_dprime(summary)
    for warning in warnings:
        _log_dprime(f"[STEPDPRIME_SCALAR_CONTEXT][WARN] {warning}")
    return out, diagnostics


def _load_json_dict(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _update_json_profiles(path: Path, profile: str, payload: Dict[str, Any], *, root_defaults: Optional[Dict[str, Any]] = None) -> None:
    base = _load_json_dict(path)
    if root_defaults:
        for key, value in root_defaults.items():
            base.setdefault(key, _json_safe(value))
    profiles = base.get("profiles")
    if not isinstance(profiles, dict):
        profiles = {}
    profiles[str(profile)] = _json_safe(payload)
    base["profiles"] = profiles
    path.write_text(json.dumps(_json_safe(base), indent=2, ensure_ascii=False), encoding="utf-8")


class _DPrimeTwoTowerTransformer(nn.Module):
    def __init__(
        self,
        *,
        past_dim: int,
        pred_dim: int,
        past_len: int,
        pred_len: int,
        cluster_dim: int,
        scalar_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
        z_past_dim: int,
        z_pred_dim: int,
        z_state_dim: int,
        next_feature_dim: int,
    ) -> None:
        super().__init__()
        self.past_len = int(past_len)
        self.pred_len = int(pred_len)
        self.past_in = nn.Linear(past_dim, d_model)
        self.pred_in = nn.Linear(pred_dim, d_model)
        self.past_pos = nn.Parameter(torch.zeros(1, past_len, d_model))
        self.pred_pos = nn.Parameter(torch.zeros(1, pred_len, d_model))
        enc_cfg = dict(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.past_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(**enc_cfg), num_layers=num_layers)
        self.pred_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(**enc_cfg), num_layers=num_layers)
        self.past_proj = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, z_past_dim), nn.GELU())
        self.pred_proj = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, z_pred_dim), nn.GELU())
        self.past_recon = nn.Linear(d_model, past_dim)
        self.pred_recon = nn.Linear(d_model, pred_dim)
        fusion_in = z_past_dim + z_pred_dim + cluster_dim + scalar_dim
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_in),
            nn.Linear(fusion_in, max(z_state_dim, fusion_in // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(z_state_dim, fusion_in // 2), z_state_dim),
        )
        self.next_feature_head = nn.Linear(z_state_dim, next_feature_dim)
        self.future_return_head = nn.Linear(z_state_dim, 1)
        self.future_sign_head = nn.Linear(z_state_dim, 1)
        self.future_vol_head = nn.Linear(z_state_dim, 1)

    def _encode_tower(self, x: torch.Tensor, proj: nn.Linear, pos: torch.Tensor, encoder: nn.Module, out_proj: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        h = proj(x) + pos[:, : x.shape[1], :]
        h = encoder(h)
        pooled = h.mean(dim=1)
        z = out_proj(pooled)
        return h, z

    def forward(self, past_x: torch.Tensor, pred_x: torch.Tensor, cluster_x: torch.Tensor, scalar_x: torch.Tensor) -> Dict[str, torch.Tensor]:
        past_tokens, z_past = self._encode_tower(past_x, self.past_in, self.past_pos, self.past_encoder, self.past_proj)
        pred_tokens, z_pred = self._encode_tower(pred_x, self.pred_in, self.pred_pos, self.pred_encoder, self.pred_proj)
        fused = torch.cat([z_past, z_pred, cluster_x, scalar_x], dim=1)
        z_state = self.fusion(fused)
        return {
            "past_tokens": past_tokens,
            "pred_tokens": pred_tokens,
            "z_past": z_past,
            "z_pred": z_pred,
            "z_state": z_state,
            "past_recon": self.past_recon(past_tokens),
            "pred_recon": self.pred_recon(pred_tokens),
            "next_features": self.next_feature_head(z_state),
            "future_return": self.future_return_head(z_state).squeeze(1),
            "future_sign_logits": self.future_sign_head(z_state).squeeze(1),
            "future_vol": self.future_vol_head(z_state).squeeze(1),
        }



class DPrimeRLService:
    """Build RL state by integrating past features, cluster IDs, and prediction summaries."""

    def run(
        self,
        cfg: StepDPrimeConfig,
        *,
        timing: TimingLogger,
        data: pd.DataFrame,
        split: Dict[str, str],
        stepb_dir: Path,
        stepc_dir: Path,
        stepd_dir: Path,
        cluster_daily: pd.DataFrame,
        cluster_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, object]:
        t_pred_load = time.perf_counter()
        tr_s, tr_e = pd.to_datetime(split["train_start"]), pd.to_datetime(split["train_end"])
        te_s, te_e = pd.to_datetime(split["test_start"]), pd.to_datetime(split["test_end"])

        cluster_summary = dict(cluster_summary or {})
        cluster_status = str(cluster_summary.get("status", "unknown"))

        pred_df, pred_meta = _build_pred_from_stepb(stepb_dir, cfg.symbol, cfg.pred_k)
        pred_time_df, pred_time_src = _load_pred_time_priority(stepc_dir, stepb_dir, cfg.symbol)
        _log_timing("rl_pred_source_load", t_pred_load)
        print(f"[STEPDPRIME_INPUT] stepb_dir={stepb_dir}")
        print(f"[STEPDPRIME_INPUT] pred_source_candidate_files={pred_meta['pred_source_candidate_files']}")
        print(f"[STEPDPRIME_INPUT] pred_source_selected={pred_meta['pred_source_selected']}")
        print(f"[STEPDPRIME_INPUT] pred_source_mode={pred_meta['pred_source_mode']}")
        print(f"[STEPDPRIME_INPUT] pred_available_horizons={pred_meta['pred_available_horizons']}")

        pred_cols = [f"Pred_Close_t_plus_{i:02d}" for i in range(1, cfg.pred_k + 1)]
        available = sorted([int(c.split("_")[-1]) for c in pred_df.columns if c.startswith("Pred_Close_t_plus_")])
        missing_filled: Dict[str, object] = {}
        for h in range(1, cfg.pred_k + 1):
            c = f"Pred_Close_t_plus_{h:02d}"
            if c in pred_df.columns:
                continue
            if available:
                lower = [x for x in available if x <= h]
                upper = [x for x in available if x > h]
                src_h = (max(lower) if lower else min(upper))
                pred_df[c] = pd.to_numeric(pred_df.get(f"Pred_Close_t_plus_{src_h:02d}"), errors="coerce")
                missing_filled[f"h{h:02d}"] = f"reuse_h{src_h:02d}"
            else:
                pred_df[c] = 0.0
                missing_filled[f"h{h:02d}"] = "zero_fill"
        print(f"[STEPDPRIME_INPUT] pred_missing_horizons_filled={missing_filled}")
        t_merge = time.perf_counter()
        data = data.merge(pred_df[["Date"] + pred_cols], on="Date", how="left")

        if pred_time_df is not None:
            for h in (1, 5, 10, 20):
                cands = [f"Pred_Close_MAMBA_h{h:02d}", f"Close_pred_h{h}", f"Pred_Close_t_plus_{h:02d}"]
                col = next((c for c in cands if c in pred_time_df.columns), None)
                if col is not None:
                    sub = pred_time_df[["Date", col]].rename(columns={col: f"Pred_Close_t_plus_{h:02d}"})
                    data = data.drop(columns=[f"Pred_Close_t_plus_{h:02d}"], errors="ignore").merge(sub, on="Date", how="left")
            if pred_time_src:
                pred_meta["pred_source_selected"] = pred_time_src + "|" + str(pred_meta.get("pred_source_selected", ""))

        data = data.merge(cluster_daily[["Date", "cluster_id_raw20", "cluster_id_stable", "rare_flag_raw20"]], on="Date", how="left")
        _log_timing("rl_dataframe_merge", t_merge)
        data["cluster_id_raw20"] = pd.to_numeric(data.get("cluster_id_raw20"), errors="coerce").fillna(0).astype(int)
        data["cluster_id_stable"] = pd.to_numeric(data.get("cluster_id_stable"), errors="coerce").fillna(0).astype(int)
        data["rare_flag_raw20"] = pd.to_numeric(data.get("rare_flag_raw20"), errors="coerce").fillna(0).astype(int)

        for i in range(1, cfg.pred_k + 1):
            c = f"Pred_Close_t_plus_{i:02d}"
            data[f"pred_ret_{i:02d}"] = pd.to_numeric(data[c], errors="coerce") / data["Close_anchor"].replace(0, np.nan) - 1.0
            data[f"pred_ret_{i:02d}"] = data[f"pred_ret_{i:02d}"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        data, scalar_context_diag = _add_rl_scalar_context(data)
        scalar_context_diag_path = stepd_dir / f"stepDprime_scalar_context_diagnostics_{cfg.symbol}.json"
        scalar_context_diag_path.write_text(
            json.dumps(_json_safe({
                **scalar_context_diag,
                "symbol": cfg.symbol,
                "mode": str(cfg.mode),
            }), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        encoder_type = _normalize_encoder_type(cfg.encoder_type)
        num_cols = [c for c in data.columns if c != "Date" and pd.api.types.is_numeric_dtype(data[c])]
        bnf_cols = [
            "ret_1", "ret_5", "ret_20", "range_atr", "body_ratio", "body_atr", "upper_wick_ratio", "lower_wick_ratio",
            "Gap", "ATR_norm", "gap_atr", "vol_log_ratio_20", "vol_chg", "dev_z_25", "bnf_score",
        ]
        mix_cols = bnf_cols + ["RSI", "MACD_hist", "macd_hist_delta", "macd_hist_cross_up", "clv", "distribution_day", "dist_count_25", "absorption_day", "cmf_20"]
        all_cols = [c for c in num_cols if c not in {"Open", "High", "Low", "Close", "Volume", "Close_anchor"}]
        scalar_cols = [
            "ctx_volatility_20",
            "ctx_volatility_63",
            "ctx_volatility_252",
            "ctx_trend_20",
            "ctx_trend_63",
            "ctx_trend_252",
            "ctx_turnover_20",
            "ctx_turnover_63",
            "ctx_drawdown_252",
            "ctx_long_ret_252",
            "ctx_long_ret_8y",
            "ctx_long_vol_252",
            "ctx_gap_atr",
            "ctx_atr_norm",
            "ctx_ret_20",
        ]
        scalar_cols = [c for c in scalar_cols if c in data.columns]

        results: Dict[str, object] = {
            "mode": cfg.mode,
            "symbol": cfg.symbol,
            "encoder_type": encoder_type,
            "profiles": {},
            "output_dir": str(stepd_dir),
            "pred_source_selected": pred_meta.get("pred_source_selected", ""),
            "pred_source_mode": pred_meta.get("pred_source_mode", ""),
            "pred_available_horizons": available,
            "pred_missing_horizons_filled": missing_filled,
            "pca_diagnostics_files": [],
            "scalar_context_diagnostics_path": str(scalar_context_diag_path),
            "scalar_context_diagnostics": scalar_context_diag,
        }
        date_list = data["Date"].tolist()
        idx_train = [i for i, d in enumerate(date_list) if tr_s <= d <= tr_e]
        idx_test = [i for i, d in enumerate(date_list) if te_s <= d <= te_e]
        if min(idx_test) < cfg.l_past - 1:
            raise RuntimeError("insufficient history before test_start for L_past window")

        meta_root_defaults = {
            "symbol": cfg.symbol,
            "mode": str(cfg.mode),
            "encoder_type": encoder_type,
            "hidden_dim": int(cfg.transformer_d_model),
            "num_layers": int(cfg.transformer_num_layers),
            "input_blocks": ["past_context", "prediction_context", "cluster_context", "scalar_context"],
            "train_range": {"start": str(tr_s.date()), "end": str(tr_e.date())},
            "feature_version": str(cfg.feature_version),
            "cluster_source_version": str(cfg.cluster_source_version),
            "cluster_contract": "stepA_only_cluster",
        }
        meta_path = stepd_dir / "stepDprime_embedding_meta.json"
        schema_path = stepd_dir / "stepDprime_state_schema.json"

        last_profile = ""
        with timing.stage("stepDPrimeRL.total"):
            for profile in cfg.profiles:
                t_profile_total = time.perf_counter()
                last_profile = profile
                global _LAST_DPRIME_PROFILE
                _LAST_DPRIME_PROFILE = profile
                with timing.stage("stepDPrimeRL.profile.loop", agent_id=str(profile), meta={"profile": str(profile), "stage_group": "rl_profile", "agent_kind": "profile", "profile_name": str(profile)}):
                    fam = _infer_family(profile)
                    pred_type = _infer_pred_type(profile)
                    past_cols = all_cols if fam == "all_features" else (mix_cols if fam == "mix" else bnf_cols)
                    past_cols = [c for c in past_cols if c in data.columns]

                    pred_steps = [1] if pred_type == "h01" else ([1, 5, 10, 20] if pred_type == "h02" else list(range(1, cfg.pred_k + 1)))

                    with timing.stage("stepDPrimeRL.profile.state_build", agent_id=str(profile), meta={"profile": str(profile), "agent_kind": "profile", "profile_name": str(profile)}):
                        pass
                    pred_use_cols = [f"pred_ret_{s:02d}" for s in pred_steps]
                    pred_all_cols = [f"pred_ret_{s:02d}" for s in range(1, cfg.pred_k + 1)]
                    next_feature_cols = past_cols[: min(8, len(past_cols))]

                    def _build_rows(indices: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[pd.Timestamp]]:
                        xp_seq, pred_seq, x_cluster, x_scalar, next_targets, aux_targets, ds = [], [], [], [], [], [], []
                        for i in indices:
                            if i < cfg.l_past - 1:
                                continue
                            hist = data.iloc[i - cfg.l_past + 1 : i + 1][past_cols].to_numpy(dtype=float)
                            xp_seq.append(hist)
                            row_pred = pd.to_numeric(data.iloc[i][pred_all_cols], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                            prev_pred = np.concatenate([[0.0], row_pred[:-1]])
                            delta_pred = row_pred - prev_pred
                            horiz_frac = np.arange(1, cfg.pred_k + 1, dtype=float) / float(max(cfg.pred_k, 1))
                            active_mask = np.array([1.0 if (k in pred_steps or pred_type == "3scale") else 0.0 for k in range(1, cfg.pred_k + 1)], dtype=float)
                            pred_block = np.stack([
                                row_pred * active_mask,
                                delta_pred * active_mask,
                                horiz_frac * active_mask,
                            ], axis=1)
                            pred_seq.append(pred_block)
                            ds.append(data.iloc[i]["Date"])
                            x_cluster.append([
                                float(pd.to_numeric(pd.Series([data.iloc[i]["cluster_id_stable"]]), errors="coerce").iloc[0]) / 20.0,
                                float(pd.to_numeric(pd.Series([data.iloc[i]["cluster_id_raw20"]]), errors="coerce").iloc[0]) / 20.0,
                                float(pd.to_numeric(pd.Series([data.iloc[i]["rare_flag_raw20"]]), errors="coerce").iloc[0]),
                            ])
                            x_scalar.append(pd.to_numeric(data.iloc[i][scalar_cols], errors="coerce").fillna(0.0).to_numpy(dtype=float))
                            next_idx = min(i + 1, len(data) - 1)
                            next_targets.append(pd.to_numeric(data.iloc[next_idx][next_feature_cols], errors="coerce").fillna(0.0).to_numpy(dtype=float))
                            aux_targets.append([
                                float(pd.to_numeric(pd.Series([data.iloc[next_idx].get("ret_1", 0.0)]), errors="coerce").fillna(0.0).iloc[0]),
                                float(pd.to_numeric(pd.Series([data.iloc[next_idx].get("ret_1", 0.0)]), errors="coerce").fillna(0.0).iloc[0] > 0.0),
                                float(pd.to_numeric(pd.Series([data.iloc[next_idx].get("ctx_volatility_20", 0.0)]), errors="coerce").fillna(0.0).iloc[0]),
                            ])
                        return (
                            np.asarray(xp_seq, dtype=float),
                            np.asarray(pred_seq, dtype=float),
                            np.asarray(x_cluster, dtype=float),
                            np.asarray(x_scalar, dtype=float),
                            np.asarray(next_targets, dtype=float),
                            np.asarray(aux_targets, dtype=float),
                            ds,
                        )

                    t_state_rows = time.perf_counter()
                    Xp_tr_seq, Xf_tr_seq, Xc_tr, Xs_tr, Ynext_tr, Yaux_tr, Dtr = _build_rows(idx_train)
                    Xp_te_seq, Xf_te_seq, Xc_te, Xs_te, Ynext_te, Yaux_te, Dte = _build_rows(idx_test)
                    _log_timing(f"profile_{profile}.state_generation", t_state_rows)
                    if len(Dtr) == 0 or len(Dte) == 0:
                        raise RuntimeError(f"profile={profile}: no rows for train/test")
                    Xp_tr_before = int(np.asarray(Xp_tr_seq, dtype=float).size - int(np.isfinite(np.asarray(Xp_tr_seq, dtype=float)).sum()))
                    Xp_tr_seq = np.nan_to_num(Xp_tr_seq, nan=0.0, posinf=0.0, neginf=0.0)
                    Xp_te_seq = np.nan_to_num(Xp_te_seq, nan=0.0, posinf=0.0, neginf=0.0)
                    Xf_tr_seq = np.nan_to_num(Xf_tr_seq, nan=0.0, posinf=0.0, neginf=0.0)
                    Xf_te_seq = np.nan_to_num(Xf_te_seq, nan=0.0, posinf=0.0, neginf=0.0)
                    Xc_tr = np.nan_to_num(Xc_tr, nan=0.0, posinf=0.0, neginf=0.0)
                    Xc_te = np.nan_to_num(Xc_te, nan=0.0, posinf=0.0, neginf=0.0)
                    Xs_tr = np.nan_to_num(Xs_tr, nan=0.0, posinf=0.0, neginf=0.0)
                    Xs_te = np.nan_to_num(Xs_te, nan=0.0, posinf=0.0, neginf=0.0)
                    Ynext_tr = np.nan_to_num(Ynext_tr, nan=0.0, posinf=0.0, neginf=0.0)
                    Ynext_te = np.nan_to_num(Ynext_te, nan=0.0, posinf=0.0, neginf=0.0)
                    Yaux_tr = np.nan_to_num(Yaux_tr, nan=0.0, posinf=0.0, neginf=0.0)
                    Yaux_te = np.nan_to_num(Yaux_te, nan=0.0, posinf=0.0, neginf=0.0)

                    if encoder_type == "legacy":
                        Xp_tr = Xp_tr_seq.reshape(Xp_tr_seq.shape[0], -1)
                        Xp_te = Xp_te_seq.reshape(Xp_te_seq.shape[0], -1)
                        Xf_tr = Xf_tr_seq.reshape(Xf_tr_seq.shape[0], -1)
                        Xf_te = Xf_te_seq.reshape(Xf_te_seq.shape[0], -1)
                        Xp_tr, xp_tr_diag = _sanitize_matrix(Xp_tr, name=f"{profile}.Xp_tr", clip_abs=1e6)
                        Xf_tr, xf_tr_diag = _sanitize_matrix(Xf_tr, name=f"{profile}.Xf_tr", clip_abs=1e6)
                        Xp_te, _ = _sanitize_matrix(Xp_te, name=f"{profile}.Xp_te", clip_abs=1e6)
                        Xf_te, _ = _sanitize_matrix(Xf_te, name=f"{profile}.Xf_te", clip_abs=1e6)

                        mu_p, sd_p = _fit_scaler(Xp_tr)
                        mu_f, sd_f = _fit_scaler(Xf_tr)

                        Xp_tr_n, _ = _sanitize_matrix((Xp_tr - mu_p) / sd_p, name=f"{profile}.Xp_tr_norm", clip_abs=1e6)
                        Xf_tr_n, _ = _sanitize_matrix((Xf_tr - mu_f) / sd_f, name=f"{profile}.Xf_tr_norm", clip_abs=1e6)

                        t_pca_fit = time.perf_counter()
                        comp_p, d_p, pca_p_diag = _fit_pca(Xp_tr_n, cfg.z_past_dim)
                        pred_dim = cfg.z_pred_dim * 3 if pred_type == "3scale" else cfg.z_pred_dim
                        comp_f, d_f, pca_f_diag = _fit_pca(Xf_tr_n, pred_dim)
                        _log_timing(f"profile_{profile}.pca_fit", t_pca_fit)

                        t_proj = time.perf_counter()
                        Zp_tr, Zp_te = _project(Xp_tr, mu_p, sd_p, comp_p), _project(Xp_te, mu_p, sd_p, comp_p)
                        Zf_tr, Zf_te = _project(Xf_tr, mu_f, sd_f, comp_f), _project(Xf_te, mu_f, sd_f, comp_f)
                        Zs_tr = np.concatenate([Zp_tr, Zf_tr], axis=1)
                        Zs_te = np.concatenate([Zp_te, Zf_te], axis=1)
                        _log_timing(f"profile_{profile}.projection", t_proj)
                        model_diag: Dict[str, Any] = {
                            "profile": profile,
                            "Xp_tr_shape": [int(v) for v in Xp_tr.shape],
                            "Xf_tr_shape": [int(v) for v in Xf_tr.shape],
                            "Xp_tr_nonfinite_count_before": int(Xp_tr_before),
                            "Xp_tr_nonfinite_count_after": int(xp_tr_diag["nonfinite_count_after"]),
                            "Xp_tr_zero_var_cols": int(xp_tr_diag["zero_var_cols"]),
                            "pca_method_used": str(pca_p_diag["pca_method_used"]),
                            "fallback_used": bool(pca_p_diag["fallback_used"]),
                            "z_past_dim_requested": int(cfg.z_past_dim),
                            "z_past_dim_used": int(d_p),
                            "z_pred_dim_requested": int(pred_dim),
                            "z_pred_dim_used": int(d_f),
                            "pred_pca_method_used": str(pca_f_diag["pca_method_used"]),
                            "pred_fallback_used": bool(pca_f_diag["fallback_used"]),
                            "Xf_tr_nonfinite_count_after": int(xf_tr_diag["nonfinite_count_after"]),
                            "encoder_type": "legacy",
                        }
                    else:
                        t_transformer = time.perf_counter()
                        mu_p, sd_p = _fit_sequence_scaler(Xp_tr_seq)
                        mu_f, sd_f = _fit_sequence_scaler(Xf_tr_seq)
                        mu_s, sd_s = _fit_scaler(Xs_tr)
                        Xp_tr_n = _apply_sequence_scaler(Xp_tr_seq, mu_p, sd_p)
                        Xp_te_n = _apply_sequence_scaler(Xp_te_seq, mu_p, sd_p)
                        Xf_tr_n = _apply_sequence_scaler(Xf_tr_seq, mu_f, sd_f)
                        Xf_te_n = _apply_sequence_scaler(Xf_te_seq, mu_f, sd_f)
                        Xs_tr_n, _ = _sanitize_matrix((Xs_tr - mu_s) / sd_s, name=f"{profile}.Xs_tr_norm", clip_abs=1e6)
                        Xs_te_n, _ = _sanitize_matrix((Xs_te - mu_s) / sd_s, name=f"{profile}.Xs_te_norm", clip_abs=1e6)
                        device = torch.device("cuda" if _cuda_available() else "cpu")
                        torch.manual_seed(int(cfg.transformer_seed))
                        if device.type == "cuda":
                            torch.cuda.manual_seed_all(int(cfg.transformer_seed))
                        model = _DPrimeTwoTowerTransformer(
                            past_dim=int(Xp_tr_n.shape[2]),
                            pred_dim=int(Xf_tr_n.shape[2]),
                            past_len=int(Xp_tr_n.shape[1]),
                            pred_len=int(Xf_tr_n.shape[1]),
                            cluster_dim=int(Xc_tr.shape[1]),
                            scalar_dim=int(Xs_tr_n.shape[1]),
                            d_model=int(cfg.transformer_d_model),
                            nhead=int(cfg.transformer_nhead),
                            num_layers=int(cfg.transformer_num_layers),
                            ff_dim=int(cfg.transformer_ff_dim),
                            dropout=float(cfg.transformer_dropout),
                            z_past_dim=int(cfg.z_past_dim),
                            z_pred_dim=int(cfg.z_pred_dim),
                            z_state_dim=int(cfg.z_state_dim),
                            next_feature_dim=int(Ynext_tr.shape[1]),
                        ).to(device)
                        optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.transformer_lr))
                        batch_size = max(8, int(cfg.transformer_batch_size))
                        mask_ratio = float(cfg.transformer_mask_ratio)
                        n_train = int(Xp_tr_n.shape[0])
                        epoch_losses: List[float] = []

                        past_tr_t = torch.tensor(Xp_tr_n, dtype=torch.float32, device=device)
                        pred_tr_t = torch.tensor(Xf_tr_n, dtype=torch.float32, device=device)
                        cluster_tr_t = torch.tensor(Xc_tr, dtype=torch.float32, device=device)
                        scalar_tr_t = torch.tensor(Xs_tr_n, dtype=torch.float32, device=device)
                        next_tr_t = torch.tensor(Ynext_tr, dtype=torch.float32, device=device)
                        ret_tr_t = torch.tensor(Yaux_tr[:, 0], dtype=torch.float32, device=device)
                        sign_tr_t = torch.tensor(Yaux_tr[:, 1], dtype=torch.float32, device=device)
                        vol_tr_t = torch.tensor(Yaux_tr[:, 2], dtype=torch.float32, device=device)

                        for _epoch in range(max(1, int(cfg.transformer_epochs))):
                            perm = torch.randperm(n_train, device=device)
                            batch_losses: List[float] = []
                            model.train()
                            for start in range(0, n_train, batch_size):
                                idx = perm[start : start + batch_size]
                                past_batch = past_tr_t[idx]
                                pred_batch = pred_tr_t[idx]
                                cluster_batch = cluster_tr_t[idx]
                                scalar_batch = scalar_tr_t[idx]
                                next_batch = next_tr_t[idx]
                                ret_batch = ret_tr_t[idx]
                                sign_batch = sign_tr_t[idx]
                                vol_batch = vol_tr_t[idx]
                                past_mask = (torch.rand_like(past_batch[..., :1]) < mask_ratio).expand_as(past_batch)
                                pred_mask = (torch.rand_like(pred_batch[..., :1]) < mask_ratio).expand_as(pred_batch)
                                past_in = past_batch.masked_fill(past_mask, 0.0)
                                pred_in = pred_batch.masked_fill(pred_mask, 0.0)
                                out = model(past_in, pred_in, cluster_batch, scalar_batch)
                                recon_loss = F.mse_loss(out["past_recon"][past_mask], past_batch[past_mask]) if bool(past_mask.any().item()) else torch.tensor(0.0, device=device)
                                pred_recon_loss = F.mse_loss(out["pred_recon"][pred_mask], pred_batch[pred_mask]) if bool(pred_mask.any().item()) else torch.tensor(0.0, device=device)
                                next_loss = F.mse_loss(out["next_features"], next_batch)
                                ret_loss = F.mse_loss(out["future_return"], ret_batch)
                                sign_loss = F.binary_cross_entropy_with_logits(out["future_sign_logits"], sign_batch)
                                vol_loss = F.mse_loss(out["future_vol"], vol_batch)
                                loss = recon_loss + pred_recon_loss + 0.5 * next_loss + 0.25 * ret_loss + 0.1 * sign_loss + 0.25 * vol_loss
                                optimizer.zero_grad(set_to_none=True)
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                                optimizer.step()
                                batch_losses.append(float(loss.detach().cpu().item()))
                            epoch_losses.append(float(np.mean(batch_losses)) if batch_losses else 0.0)

                        @torch.no_grad()
                        def _encode_arrays(
                            past_arr: np.ndarray,
                            pred_arr: np.ndarray,
                            cluster_arr: np.ndarray,
                            scalar_arr: np.ndarray,
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
                            model.eval()
                            out = model(
                                torch.tensor(past_arr, dtype=torch.float32, device=device),
                                torch.tensor(pred_arr, dtype=torch.float32, device=device),
                                torch.tensor(cluster_arr, dtype=torch.float32, device=device),
                                torch.tensor(scalar_arr, dtype=torch.float32, device=device),
                            )
                            return (
                                out["z_past"].detach().cpu().numpy().astype(float),
                                out["z_pred"].detach().cpu().numpy().astype(float),
                                out["z_state"].detach().cpu().numpy().astype(float),
                            )

                        Zp_tr, Zf_tr, Zs_tr = _encode_arrays(Xp_tr_n, Xf_tr_n, Xc_tr, Xs_tr_n)
                        Zp_te, Zf_te, Zs_te = _encode_arrays(Xp_te_n, Xf_te_n, Xc_te, Xs_te_n)
                        d_p = int(Zp_tr.shape[1])
                        d_f = int(Zf_tr.shape[1])
                        model_diag = {
                            "profile": profile,
                            "encoder_type": "transformer",
                            "device": device.type,
                            "past_seq_shape": [int(v) for v in Xp_tr_seq.shape],
                            "pred_seq_shape": [int(v) for v in Xf_tr_seq.shape],
                            "cluster_shape": [int(v) for v in Xc_tr.shape],
                            "scalar_shape": [int(v) for v in Xs_tr.shape],
                            "masked_reconstruction": "enabled",
                            "next_step_feature_prediction": "enabled",
                            "aux_targets": ["future_return", "future_sign", "future_volatility"],
                            "transformer_epochs": int(cfg.transformer_epochs),
                            "epoch_losses": epoch_losses,
                            "z_past_dim_used": d_p,
                            "z_pred_dim_used": d_f,
                            "z_state_dim_used": int(Zs_tr.shape[1]),
                        }
                        _log_timing(f"profile_{profile}.transformer_fit", t_transformer)
                        print(f"[STEPDPRIME_XFMR] profile={profile} device={device.type} epochs={cfg.transformer_epochs} z_state_dim={Zs_tr.shape[1]}")

                    pca_diag_path = stepd_dir / f"stepDprime_pca_diagnostics_{profile}_{cfg.symbol}.json"
                    pca_diag_path.write_text(json.dumps(_json_safe(model_diag), indent=2, ensure_ascii=False), encoding="utf-8")
                    results["pca_diagnostics_files"].append(str(pca_diag_path))
                    print(f"[STEPDPRIME_MODEL] profile={profile} encoder_type={encoder_type} z_past_dim_used={d_p} z_pred_dim_used={d_f}")

                    t_groupby = time.perf_counter()
                    ref = (
                        data.sort_values("Date")
                        .dropna(subset=["Date"])
                        .groupby("Date", as_index=True)
                        .last()
                    )
                    _log_timing(f"profile_{profile}.groupby_last", t_groupby)

                    def _scalar(date_value: pd.Timestamp, column: str, default: float = 0.0) -> float:
                        ts = pd.to_datetime(date_value)
                        if ts not in ref.index or column not in ref.columns:
                            return float(default)
                        val = pd.to_numeric(pd.Series([ref.at[ts, column]]), errors="coerce").iloc[0]
                        return float(0.0 if pd.isna(val) else val)

                    def _to_df(ds: List[pd.Timestamp], zp: np.ndarray, zf: np.ndarray, zs: np.ndarray) -> pd.DataFrame:
                        payload: Dict[str, Any] = {"Date": pd.to_datetime(ds).strftime("%Y-%m-%d")}
                        for j in range(zp.shape[1]):
                            payload[f"zp_{j:03d}"] = zp[:, j]
                        for j in range(zf.shape[1]):
                            payload[f"zf_{j:03d}"] = zf[:, j]
                        for j in range(zs.shape[1]):
                            payload[f"zs_{j:03d}"] = zs[:, j]
                        payload["gap_atr"] = [_scalar(d, "gap_atr", 0.0) for d in pd.to_datetime(ds)]
                        payload["ATR_norm"] = [_scalar(d, "ATR_norm", 0.0) for d in pd.to_datetime(ds)]
                        payload["cluster_id_raw20"] = [int(round(_scalar(d, "cluster_id_raw20", 0.0))) for d in pd.to_datetime(ds)]
                        payload["cluster_id_stable"] = [int(round(_scalar(d, "cluster_id_stable", 0.0))) for d in pd.to_datetime(ds)]
                        payload["rare_flag_raw20"] = [int(round(_scalar(d, "rare_flag_raw20", 0.0))) for d in pd.to_datetime(ds)]
                        for scalar_col in scalar_cols:
                            payload[scalar_col] = [_scalar(d, scalar_col, 0.0) for d in pd.to_datetime(ds)]
                        payload["pos_prev"] = 0.0
                        payload["action_prev"] = 0.0
                        payload["time_in_trade"] = 0.0
                        return pd.DataFrame(payload)

                    df_tr, df_te = _to_df(Dtr, Zp_tr, Zf_tr, Zs_tr), _to_df(Dte, Zp_te, Zf_te, Zs_te)
                    p_tr = stepd_dir / f"stepDprime_state_train_{profile}_{cfg.symbol}.csv"
                    p_te = stepd_dir / f"stepDprime_state_test_{profile}_{cfg.symbol}.csv"
                    s_path = stepd_dir / f"stepDprime_split_summary_{profile}_{cfg.symbol}.csv"
                    t_save = time.perf_counter()
                    df_tr.to_csv(p_tr, index=False)
                    df_te.to_csv(p_te, index=False)
                    base_range_dates = (
                        data.loc[(data["Date"] >= tr_s) & (data["Date"] <= te_e), ["Date"]]
                        .dropna()
                        .drop_duplicates(subset=["Date"])
                        .sort_values("Date")
                        .reset_index(drop=True)
                    )
                    first_valid_ts = pd.to_datetime(pd.Series([*Dtr, *Dte]), errors="coerce").dropna().min()
                    warmup_rows = int((pd.to_datetime(base_range_dates["Date"], errors="coerce") < first_valid_ts).sum()) if pd.notna(first_valid_ts) else 0
                    expected_base_rows = int(len(base_range_dates))
                    effective_embedding_rows = int(len(df_tr) + len(df_te))
                    eligible_rows = max(0, expected_base_rows - warmup_rows)
                    expected_join_ratio_after_warmup = float(effective_embedding_rows / eligible_rows) if eligible_rows > 0 else 1.0
                    unexpected_missing_rows = max(0, eligible_rows - effective_embedding_rows)
                    pd.DataFrame([
                        {"key": "mode", "value": cfg.mode}, {"key": "symbol", "value": cfg.symbol}, {"key": "profile", "value": profile},
                        {"key": "train_start", "value": str(tr_s.date())}, {"key": "train_end", "value": str(tr_e.date())},
                        {"key": "test_start", "value": str(te_s.date())}, {"key": "test_end", "value": str(te_e.date())},
                        {"key": "L_past", "value": cfg.l_past}, {"key": "pred_type", "value": pred_type}, {"key": "pred_k", "value": cfg.pred_k},
                        {"key": "encoder_type", "value": encoder_type},
                        {"key": "z_past_dim", "value": int(d_p)}, {"key": "z_pred_dim", "value": int(d_f)},
                        {"key": "z_state_dim", "value": int(Zs_tr.shape[1])},
                        {"key": "rows_train_written", "value": int(len(df_tr))}, {"key": "rows_test_written", "value": int(len(df_te))},
                        {"key": "warmup_rows", "value": int(warmup_rows)},
                        {"key": "first_valid_date", "value": "" if pd.isna(first_valid_ts) else str(first_valid_ts.date())},
                        {"key": "expected_base_rows", "value": int(expected_base_rows)},
                        {"key": "effective_embedding_rows", "value": int(effective_embedding_rows)},
                        {"key": "expected_join_ratio_after_warmup", "value": float(expected_join_ratio_after_warmup)},
                        {"key": "unexpected_missing_rows", "value": int(unexpected_missing_rows)},
                        {"key": "past_feature_channels", "value": "|".join(past_cols)},
                        {"key": "pred_source_file", "value": pred_meta.get("pred_source_selected", "")},
                        {"key": "pred_source_mode", "value": pred_meta.get("pred_source_mode", "")},
                        {"key": "pred_available_horizons", "value": "|".join(str(x) for x in available)},
                        {"key": "pred_missing_horizons_filled", "value": json.dumps(missing_filled, ensure_ascii=False)},
                        {"key": "scalar_context_diagnostics_path", "value": str(scalar_context_diag_path)},
                        {"key": "scalar_context_close_source", "value": scalar_context_diag.get("close_source", "") or ""},
                        {"key": "scalar_context_volume_source", "value": scalar_context_diag.get("volume_source", "") or ""},
                        {"key": "scalar_context_missing_required", "value": "|".join(str(x) for x in scalar_context_diag.get("missing_required_columns", []))},
                        {"key": "scalar_context_warnings", "value": json.dumps(scalar_context_diag.get("warnings", []), ensure_ascii=False)},
                        {"key": "dprime_cluster_source", "value": "stepDprime_cluster_daily_assign"},
                        {"key": "dprime_cluster_status", "value": cluster_status},
                        {"key": "fit_stats", "value": f"train_only:{tr_s.date()}..{tr_e.date()}"},
                        {"key": "model_components_shape", "value": json.dumps({"z_past": list(Zp_tr.shape), "z_pred": list(Zf_tr.shape), "z_state": list(Zs_tr.shape)})},
                    ]).to_csv(s_path, index=False)

                    emb_cols = [c for c in df_tr.columns if c.startswith("zs_")]
                    if not emb_cols:
                        emb_cols = [c for c in df_tr.columns if c.startswith("zp_") or c.startswith("zf_")]

                    def _to_emb_df(df_state: pd.DataFrame) -> pd.DataFrame:
                        out = pd.DataFrame({"Date": pd.to_datetime(df_state["Date"], errors="coerce").dt.strftime("%Y-%m-%d")})
                        for i, c in enumerate(emb_cols):
                            out[f"emb_{i:03d}"] = pd.to_numeric(df_state[c], errors="coerce").astype(float)
                        return out

                    df_emb_tr = _to_emb_df(df_tr)
                    df_emb_te = _to_emb_df(df_te)
                    df_emb_all = pd.concat([df_emb_tr, df_emb_te], axis=0, ignore_index=True)
                    df_emb_all["Date"] = pd.to_datetime(df_emb_all["Date"], errors="coerce")
                    df_emb_all = (
                        df_emb_all.dropna(subset=["Date"])
                        .sort_values("Date")
                        .drop_duplicates(subset=["Date"], keep="last")
                        .reset_index(drop=True)
                    )
                    df_emb_all["Date"] = df_emb_all["Date"].dt.strftime("%Y-%m-%d")
                    emb_dir = stepd_dir / "embeddings"
                    emb_dir.mkdir(parents=True, exist_ok=True)
                    ep_tr = emb_dir / f"stepDprime_{profile}_{cfg.symbol}_embeddings_train.csv"
                    ep_te = emb_dir / f"stepDprime_{profile}_{cfg.symbol}_embeddings_test.csv"
                    ep_all = emb_dir / f"stepDprime_{profile}_{cfg.symbol}_embeddings_all.csv"
                    df_emb_tr.to_csv(ep_tr, index=False)
                    df_emb_te.to_csv(ep_te, index=False)
                    df_emb_all.to_csv(ep_all, index=False)

                    ep_all_named = emb_dir / f"stepDprime_{fam}_{pred_type}_{cfg.symbol}_embeddings_all.csv"
                    df_emb_all.to_csv(ep_all_named, index=False)
                    _log_timing(f"profile_{profile}.csv_save", t_save)

                    emb_values = df_emb_all[[c for c in df_emb_all.columns if c.startswith("emb_")]].to_numpy(dtype=float) if len(df_emb_all) else np.zeros((0, 0), dtype=float)
                    nonzero_ratio = float(np.count_nonzero(np.abs(emb_values) > 1e-12) / emb_values.size) if emb_values.size > 0 else 0.0
                    merge_ratio = float(len(df_emb_all) / expected_base_rows) if expected_base_rows > 0 else 0.0
                    _update_json_profiles(
                        meta_path,
                        profile,
                        {
                            "encoder_type": encoder_type,
                            "family": fam,
                            "pred_type": pred_type,
                            "hidden_dim": int(cfg.transformer_d_model) if encoder_type == "transformer" else None,
                            "num_layers": int(cfg.transformer_num_layers) if encoder_type == "transformer" else None,
                            "input_blocks": {
                                "past_context": list(past_cols),
                                "prediction_context": list(pred_use_cols if pred_type != "3scale" else pred_all_cols),
                                "cluster_context": ["cluster_id_stable", "cluster_id_raw20", "rare_flag_raw20"],
                                "scalar_context": list(scalar_cols),
                            },
                            "train_range": {"start": str(tr_s.date()), "end": str(tr_e.date())},
                            "feature_version": str(cfg.feature_version),
                            "cluster_source_version": str(cfg.cluster_source_version),
                            "merge_ratio": merge_ratio,
                            "nonzero_ratio": nonzero_ratio,
                            "effective_embedding_rows": int(effective_embedding_rows),
                            "expected_base_rows": int(expected_base_rows),
                            "diagnostics_path": str(pca_diag_path),
                            "scalar_context_diagnostics_path": str(scalar_context_diag_path),
                            "scalar_context": scalar_context_diag,
                        },
                        root_defaults=meta_root_defaults,
                    )
                    _update_json_profiles(
                        schema_path,
                        profile,
                        {
                            "state_columns": {
                                "date": ["Date"],
                                "past_embedding": [c for c in df_tr.columns if c.startswith("zp_")],
                                "pred_embedding": [c for c in df_tr.columns if c.startswith("zf_")],
                                "state_embedding": [c for c in df_tr.columns if c.startswith("zs_")],
                                "cluster_context": ["cluster_id_raw20", "cluster_id_stable", "rare_flag_raw20"],
                                "scalar_context": list(scalar_cols),
                                "compatibility": ["gap_atr", "ATR_norm", "pos_prev", "action_prev", "time_in_trade"],
                            }
                        },
                        root_defaults={"symbol": cfg.symbol, "mode": str(cfg.mode), "encoder_type": encoder_type},
                    )
                    _log_timing(f"profile_{profile}.total", t_profile_total)

                    results["profiles"][profile] = {
                        "train": str(p_tr),
                        "test": str(p_te),
                        "summary": str(s_path),
                        "encoder_type": encoder_type,
                        "warmup_rows": int(warmup_rows),
                        "first_valid_date": "" if pd.isna(first_valid_ts) else str(first_valid_ts.date()),
                        "expected_base_rows": int(expected_base_rows),
                        "effective_embedding_rows": int(effective_embedding_rows),
                        "expected_join_ratio_after_warmup": float(expected_join_ratio_after_warmup),
                        "unexpected_missing_rows": int(unexpected_missing_rows),
                    }

        pca_all_path = stepd_dir / f"stepDprime_pca_diagnostics_{cfg.symbol}.json"
        pca_all_path.write_text(json.dumps(_json_safe({"profiles": results.get("pca_diagnostics_files", [])}), indent=2, ensure_ascii=False), encoding="utf-8")
        results["pca_diagnostics_index"] = str(pca_all_path)
        results["last_profile"] = last_profile
        return results




class _DPrimeForceCPUContext:
    def __init__(self, enabled: bool):
        self.enabled = bool(enabled)
        self.prev = None

    def __enter__(self):
        if not self.enabled:
            return self
        self.prev = os.environ.get("DPRIME_FORCE_CPU")
        os.environ["DPRIME_FORCE_CPU"] = "1"
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.enabled:
            return False
        if self.prev is None:
            os.environ.pop("DPRIME_FORCE_CPU", None)
        else:
            os.environ["DPRIME_FORCE_CPU"] = self.prev
        return False


@dataclass
class StepDPrimeBaseClusterResult:
    status: str
    base_features_path: str
    cluster_daily_path: str
    cluster_summary_path: str
    ready_path: str


@dataclass
class StepDPrimeProfileResult:
    status: str
    profile: str
    state_train_path: str
    state_test_path: str
    summary_path: str
    embedding_all_path: str
    ready_path: str


class StepDPrimeService:
    def _resolve_paths(self, cfg: StepDPrimeConfig) -> Dict[str, Path]:
        mode = _normalize_mode(cfg.mode)
        out_root = Path(cfg.output_root)
        return {
            "mode": Path(mode),
            "stepa_dir": Path(cfg.stepA_root) if cfg.stepA_root else out_root / "stepA" / mode,
            "stepb_dir": Path(cfg.stepB_root) if cfg.stepB_root else out_root / "stepB" / mode,
            "stepc_dir": Path(cfg.stepC_root) if cfg.stepC_root else out_root / "stepC" / mode,
            "stepd_dir": resolve_stepdprime_dir(out_root, mode, explicit_root=cfg.stepDprime_root, for_write=True),
        }

    def _build_all_df(self, cfg: StepDPrimeConfig, stepa_dir: Path) -> Tuple[Dict[str, str], pd.DataFrame]:
        split = _read_split_summary(stepa_dir, cfg.symbol)
        pr_tr, pr_te = _read_pair(stepa_dir, "stepA_prices", cfg.symbol)
        tc_tr, tc_te = _read_pair(stepa_dir, "stepA_tech", cfg.symbol)
        pe_tr, pe_te = _read_pair(stepa_dir, "stepA_periodic", cfg.symbol)
        prices = pd.concat([pr_tr, pr_te], ignore_index=True).sort_values("Date").reset_index(drop=True)
        tech = pd.concat([tc_tr, tc_te], ignore_index=True).sort_values("Date").reset_index(drop=True)
        periodic = pd.concat([pe_tr, pe_te], ignore_index=True).sort_values("Date").reset_index(drop=True)
        base = _compute_base_features(prices, tech)
        all_df = base.merge(tech, on="Date", how="left", suffixes=("", "_tech")).merge(periodic, on="Date", how="left")
        all_df["Close_anchor"] = _safe(prices, "Close")
        for c in all_df.columns:
            if c == "Date":
                continue
            all_df[c] = pd.to_numeric(all_df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return split, all_df

    def run_base_cluster(self, cfg: StepDPrimeConfig) -> StepDPrimeBaseClusterResult:
        from ai_core.utils.file_ready_utils import write_status_marker

        timing = cfg.timing_logger if isinstance(cfg.timing_logger, TimingLogger) else TimingLogger.disabled()
        paths = self._resolve_paths(cfg)
        stepd_dir = paths["stepd_dir"]
        output_root = Path(cfg.output_root)
        stepa_dir = paths["stepa_dir"]
        stepd_dir.mkdir(parents=True, exist_ok=True)
        marker_dir = stepd_dir / "pipeline_markers"
        write_status_marker(marker_dir, "DPrimeBaseCluster", "RUNNING", {"symbol": cfg.symbol})
        try:
            with timing.stage("stepDPrime.base_cluster"):
                _, all_df = self._build_all_df(cfg, stepa_dir)
                base_path = stepd_dir / f"stepDprime_base_features_{cfg.symbol}.csv"
                all_df.to_csv(base_path, index=False)
                cluster_service = DPrimeClusterService()
                cluster_cfg = ClusterRuntimeConfig(
                    symbol=cfg.symbol,
                    mode=cfg.mode,
                    cluster_backend=cfg.cluster_backend,
                    cluster_raw_k=cfg.cluster_raw_k,
                    cluster_k_eff_min=cfg.cluster_k_eff_min,
                    cluster_small_share_threshold=cfg.cluster_small_share_threshold,
                    cluster_small_mean_run_threshold=cfg.cluster_small_mean_run_threshold,
                    cluster_short_window_days=cfg.cluster_short_window_days,
                    cluster_mid_window_weeks=cfg.cluster_mid_window_weeks,
                    cluster_long_window_months=cfg.cluster_long_window_months,
                    cluster_enable_8y_context=cfg.cluster_enable_8y_context,
                    cluster_rare_flag_enabled=cfg.cluster_rare_flag_enabled,
                )
                periodic = all_df[["Date"]].copy()
                cluster_out = cluster_service.run(cluster_cfg, data=all_df, periodic=periodic, stepd_dir=stepd_dir)
                cluster_daily_path = stepd_dir / f"stepDprime_cluster_daily_assign_{cfg.symbol}.csv"
                cluster_summary_path = stepd_dir / f"stepDprime_cluster_summary_{cfg.symbol}.json"
                cluster_raw_stats_path = stepd_dir / f"stepDprime_cluster_raw_stats_{cfg.symbol}.csv"
                if not cluster_daily_path.exists() or not cluster_summary_path.exists():
                    raise FileNotFoundError("DPrimeCluster artifacts missing")
                base_meta_path = stepd_dir / f"stepDprime_base_meta_{cfg.symbol}.json"
                _write_json(base_meta_path, {
                    "symbol": cfg.symbol,
                    "mode": str(cfg.mode),
                    "base_features_path": _normalize_output_path(base_path, output_root=output_root),
                    "cluster_daily_path": _normalize_output_path(cluster_daily_path, output_root=output_root),
                    "cluster_summary_path": _normalize_output_path(cluster_summary_path, output_root=output_root),
                    "cluster_raw_stats_path": _normalize_output_path(cluster_raw_stats_path, output_root=output_root),
                    "status": "READY",
                    "created_at": _utcnow_iso(),
                })
                ready = write_status_marker(marker_dir, "DPrimeBaseCluster", "READY", {
                    "symbol": cfg.symbol,
                    "base_features_path": _normalize_output_path(base_path, output_root=output_root),
                    "cluster_daily_path": _normalize_output_path(cluster_daily_path, output_root=output_root),
                    "cluster_summary_path": _normalize_output_path(cluster_summary_path, output_root=output_root),
                    "base_meta_path": _normalize_output_path(base_meta_path, output_root=output_root),
                })
                return StepDPrimeBaseClusterResult(
                    status="READY",
                    base_features_path=str(base_path),
                    cluster_daily_path=str(cluster_daily_path),
                    cluster_summary_path=str(cluster_summary_path),
                    ready_path=str(ready),
                )
        except Exception as e:
            diag_payload = {
                "symbol": cfg.symbol,
                "mode": str(cfg.mode),
                "error": repr(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc(),
                "config": {
                    "cluster_backend": str(cfg.cluster_backend),
                    "cluster_raw_k": int(cfg.cluster_raw_k),
                    "cluster_k_eff_min": int(cfg.cluster_k_eff_min),
                    "cluster_small_share_threshold": float(cfg.cluster_small_share_threshold),
                    "cluster_small_mean_run_threshold": float(cfg.cluster_small_mean_run_threshold),
                },
                "artifacts_present": {
                    "base_features": str((stepd_dir / f"stepDprime_base_features_{cfg.symbol}.csv").exists()).lower(),
                    "cluster_daily": str((stepd_dir / f"stepDprime_cluster_daily_assign_{cfg.symbol}.csv").exists()).lower(),
                    "cluster_summary": str((stepd_dir / f"stepDprime_cluster_summary_{cfg.symbol}.json").exists()).lower(),
                },
            }
            diag_path = stepd_dir / f"stepDprime_base_cluster_diagnostics_{cfg.symbol}.json"
            diag_path.write_text(json.dumps(_json_safe(diag_payload), indent=2, ensure_ascii=False), encoding="utf-8")
            base_meta_path = stepd_dir / f"stepDprime_base_meta_{cfg.symbol}.json"
            _write_json(base_meta_path, {
                "symbol": cfg.symbol,
                "mode": str(cfg.mode),
                "base_features_path": _normalize_output_path(stepd_dir / f"stepDprime_base_features_{cfg.symbol}.csv", output_root=output_root),
                "cluster_daily_path": _normalize_output_path(stepd_dir / f"stepDprime_cluster_daily_assign_{cfg.symbol}.csv", output_root=output_root),
                "cluster_summary_path": _normalize_output_path(stepd_dir / f"stepDprime_cluster_summary_{cfg.symbol}.json", output_root=output_root),
                "status": "FAILED",
                "created_at": _utcnow_iso(),
                "diagnostics_path": _normalize_output_path(diag_path, output_root=output_root),
                "error": repr(e),
                "error_type": type(e).__name__,
            })
            write_status_marker(
                marker_dir,
                "DPrimeBaseCluster",
                "FAILED",
                {
                    "symbol": cfg.symbol,
                    "error": repr(e),
                    "diagnostics_path": _normalize_output_path(diag_path, output_root=output_root),
                    "base_meta_path": _normalize_output_path(base_meta_path, output_root=output_root),
                },
            )
            raise
        finally:
            _cleanup_running_marker(marker_dir, "DPrimeBaseCluster")

    def run_final_profile(self, cfg: StepDPrimeConfig, profile: str, *, force_cpu: bool = False) -> StepDPrimeProfileResult:
        from ai_core.utils.file_ready_utils import write_status_marker

        timing = cfg.timing_logger if isinstance(cfg.timing_logger, TimingLogger) else TimingLogger.disabled()
        paths = self._resolve_paths(cfg)
        stepd_dir = paths["stepd_dir"]
        output_root = Path(cfg.output_root)
        marker_dir = stepd_dir / "pipeline_markers"
        marker_name = f"DPrimeFinal_{profile}"
        family, pred_type = _parse_profile(profile)
        write_status_marker(marker_dir, marker_name, "RUNNING", {"symbol": cfg.symbol, "profile": profile, "force_cpu": bool(force_cpu)})
        try:
            with _DPrimeForceCPUContext(force_cpu):
                base_path = stepd_dir / f"stepDprime_base_features_{cfg.symbol}.csv"
                if base_path.exists():
                    all_df = pd.read_csv(base_path)
                    all_df["Date"] = pd.to_datetime(all_df["Date"], errors="coerce").dt.normalize()
                    split = _read_split_summary(paths["stepa_dir"], cfg.symbol)
                else:
                    split, all_df = self._build_all_df(cfg, paths["stepa_dir"])

                cluster_daily_path = stepd_dir / f"stepDprime_cluster_daily_assign_{cfg.symbol}.csv"
                if not cluster_daily_path.exists():
                    raise FileNotFoundError(f"missing cluster daily: {cluster_daily_path}")
                cluster_daily = pd.read_csv(cluster_daily_path)
                cluster_daily["Date"] = pd.to_datetime(cluster_daily["Date"], errors="coerce").dt.normalize()

                cluster_summary_path = stepd_dir / f"stepDprime_cluster_summary_{cfg.symbol}.json"
                cluster_summary = {}
                if cluster_summary_path.exists():
                    cluster_summary = json.loads(cluster_summary_path.read_text(encoding="utf-8"))

                rl_service = DPrimeRLService()
                profile_cfg = StepDPrimeConfig(**{**cfg.__dict__, "profiles": (profile,)})
                rl_service.run(
                    profile_cfg,
                    timing=timing,
                    data=all_df.copy(),
                    split=split,
                    stepb_dir=paths["stepb_dir"],
                    stepc_dir=paths["stepc_dir"],
                    stepd_dir=stepd_dir,
                    cluster_daily=cluster_daily,
                    cluster_summary=cluster_summary,
                )

            summary_path = stepd_dir / f"stepDprime_split_summary_{profile}_{cfg.symbol}.csv"
            train_path = stepd_dir / f"stepDprime_state_train_{profile}_{cfg.symbol}.csv"
            test_path = stepd_dir / f"stepDprime_state_test_{profile}_{cfg.symbol}.csv"
            emb_path = stepd_dir / "embeddings" / f"stepDprime_{profile}_{cfg.symbol}_embeddings_all.csv"
            if not train_path.exists() or not test_path.exists() or not emb_path.exists():
                raise FileNotFoundError(f"StepDPrime final profile artifacts missing: {profile}")
            split_meta = _read_split_summary_csv(summary_path)
            warmup_rows = _safe_int(split_meta.get("warmup_rows", 0), 0)
            first_valid_date = str(split_meta.get("first_valid_date", "") or "")
            expected_base_rows = _safe_int(split_meta.get("expected_base_rows", 0), 0)
            effective_embedding_rows = _safe_int(split_meta.get("effective_embedding_rows", 0), 0)
            expected_join_ratio_after_warmup = _safe_float(split_meta.get("expected_join_ratio_after_warmup", 0.0), 0.0)
            unexpected_missing_rows = _safe_int(split_meta.get("unexpected_missing_rows", 0), 0)
            if not first_valid_date or expected_base_rows <= 0 or effective_embedding_rows <= 0:
                train_df = pd.read_csv(train_path)
                test_df = pd.read_csv(test_path)
                emb_df = pd.read_csv(emb_path)
                first_valid_ts = pd.to_datetime(
                    pd.concat(
                        [
                            pd.to_datetime(train_df.get("Date"), errors="coerce"),
                            pd.to_datetime(test_df.get("Date"), errors="coerce"),
                            pd.to_datetime(emb_df.get("Date"), errors="coerce"),
                        ],
                        axis=0,
                        ignore_index=True,
                    ),
                    errors="coerce",
                ).dropna().min()
                base_dates = (
                    all_df.loc[
                        (all_df["Date"] >= pd.to_datetime(split["train_start"])) & (all_df["Date"] <= pd.to_datetime(split["test_end"])),
                        "Date",
                    ]
                    .dropna()
                    .drop_duplicates()
                    .sort_values()
                    .reset_index(drop=True)
                )
                if not first_valid_date and pd.notna(first_valid_ts):
                    first_valid_date = str(first_valid_ts.date())
                if expected_base_rows <= 0:
                    expected_base_rows = int(len(base_dates))
                if effective_embedding_rows <= 0:
                    effective_embedding_rows = int(pd.to_datetime(emb_df.get("Date"), errors="coerce").dropna().nunique())
                if warmup_rows <= 0 and pd.notna(first_valid_ts):
                    warmup_rows = int((pd.to_datetime(base_dates, errors="coerce") < first_valid_ts).sum())
                eligible_rows = max(0, expected_base_rows - warmup_rows)
                if expected_join_ratio_after_warmup <= 0.0:
                    expected_join_ratio_after_warmup = float(effective_embedding_rows / eligible_rows) if eligible_rows > 0 else 1.0
                if unexpected_missing_rows <= 0 and eligible_rows >= effective_embedding_rows:
                    unexpected_missing_rows = int(max(0, eligible_rows - effective_embedding_rows))
            contract_ready = bool(first_valid_date) and expected_base_rows > 0 and effective_embedding_rows > 0
            if not contract_ready:
                raise RuntimeError(
                    f"StepDPrime profile contract incomplete: profile={profile} "
                    f"warmup_rows={warmup_rows} first_valid_date={first_valid_date!r} "
                    f"expected_base_rows={expected_base_rows} effective_embedding_rows={effective_embedding_rows}"
                )
            profile_summary_path = stepd_dir / f"stepDprime_profile_summary_{profile}_{cfg.symbol}.json"
            _write_json(profile_summary_path, {
                "symbol": cfg.symbol,
                "mode": str(cfg.mode),
                "profile": profile,
                "encoder_type": _normalize_encoder_type(cfg.encoder_type),
                "family": family,
                "pred_type": pred_type,
                "pred_source_selected": _normalize_output_path(split_meta.get("pred_source_file", ""), output_root=output_root),
                "pred_source_mode": str(split_meta.get("pred_source_mode", "")),
                "pred_available_horizons": _parse_available_horizons(split_meta.get("pred_available_horizons", "")),
                "state_train_path": _normalize_output_path(train_path, output_root=output_root),
                "state_test_path": _normalize_output_path(test_path, output_root=output_root),
                "embeddings_all_path": _normalize_output_path(emb_path, output_root=output_root),
                "embedding_meta_path": _normalize_output_path(stepd_dir / "stepDprime_embedding_meta.json", output_root=output_root),
                "state_schema_path": _normalize_output_path(stepd_dir / "stepDprime_state_schema.json", output_root=output_root),
                "warmup_rows": warmup_rows,
                "first_valid_date": first_valid_date,
                "expected_base_rows": expected_base_rows,
                "effective_embedding_rows": effective_embedding_rows,
                "expected_join_ratio_after_warmup": expected_join_ratio_after_warmup,
                "unexpected_missing_rows": unexpected_missing_rows,
                "force_cpu": bool(force_cpu),
                "gpu_execution_guard": "dprime_final_completes_before_stepe_agent_start",
                "status": "READY",
                "created_at": _utcnow_iso(),
            })
            ready = write_status_marker(marker_dir, marker_name, "READY", {
                "profile": profile,
                "symbol": cfg.symbol,
                "force_cpu": bool(force_cpu),
                "profile_summary_path": _normalize_output_path(profile_summary_path, output_root=output_root),
                "warmup_rows": warmup_rows,
                "first_valid_date": first_valid_date,
                "expected_base_rows": expected_base_rows,
                "effective_embedding_rows": effective_embedding_rows,
                "expected_join_ratio_after_warmup": expected_join_ratio_after_warmup,
                "gpu_execution_guard": "dprime_final_completes_before_stepe_agent_start",
            })
            return StepDPrimeProfileResult(
                status="READY",
                profile=profile,
                state_train_path=str(train_path),
                state_test_path=str(test_path),
                summary_path=str(summary_path),
                embedding_all_path=str(emb_path),
                ready_path=str(ready),
            )
        except Exception as e:
            profile_summary_path = stepd_dir / f"stepDprime_profile_summary_{profile}_{cfg.symbol}.json"
            split_meta = _read_split_summary_csv(stepd_dir / f"stepDprime_split_summary_{profile}_{cfg.symbol}.csv")
            _write_json(profile_summary_path, {
                "symbol": cfg.symbol,
                "mode": str(cfg.mode),
                "profile": profile,
                "encoder_type": _normalize_encoder_type(cfg.encoder_type),
                "family": family,
                "pred_type": pred_type,
                "pred_source_selected": _normalize_output_path(split_meta.get("pred_source_file", ""), output_root=output_root),
                "pred_source_mode": str(split_meta.get("pred_source_mode", "")),
                "pred_available_horizons": _parse_available_horizons(split_meta.get("pred_available_horizons", "")),
                "state_train_path": _normalize_output_path(stepd_dir / f"stepDprime_state_train_{profile}_{cfg.symbol}.csv", output_root=output_root),
                "state_test_path": _normalize_output_path(stepd_dir / f"stepDprime_state_test_{profile}_{cfg.symbol}.csv", output_root=output_root),
                "embeddings_all_path": _normalize_output_path(stepd_dir / "embeddings" / f"stepDprime_{profile}_{cfg.symbol}_embeddings_all.csv", output_root=output_root),
                "embedding_meta_path": _normalize_output_path(stepd_dir / "stepDprime_embedding_meta.json", output_root=output_root),
                "state_schema_path": _normalize_output_path(stepd_dir / "stepDprime_state_schema.json", output_root=output_root),
                "warmup_rows": _safe_int(split_meta.get("warmup_rows", 0), 0),
                "first_valid_date": str(split_meta.get("first_valid_date", "") or ""),
                "expected_base_rows": _safe_int(split_meta.get("expected_base_rows", 0), 0),
                "effective_embedding_rows": _safe_int(split_meta.get("effective_embedding_rows", 0), 0),
                "expected_join_ratio_after_warmup": _safe_float(split_meta.get("expected_join_ratio_after_warmup", 0.0), 0.0),
                "unexpected_missing_rows": _safe_int(split_meta.get("unexpected_missing_rows", 0), 0),
                "force_cpu": bool(force_cpu),
                "gpu_execution_guard": "dprime_final_completes_before_stepe_agent_start",
                "status": "FAILED",
                "created_at": _utcnow_iso(),
                "error": repr(e),
                "error_type": type(e).__name__,
            })
            write_status_marker(marker_dir, marker_name, "FAILED", {
                "symbol": cfg.symbol,
                "profile": profile,
                "error": repr(e),
                "force_cpu": bool(force_cpu),
                "profile_summary_path": _normalize_output_path(profile_summary_path, output_root=output_root),
                "gpu_execution_guard": "dprime_final_completes_before_stepe_agent_start",
            })
            raise
        finally:
            _cleanup_running_marker(marker_dir, marker_name)

    def run(self, cfg: StepDPrimeConfig) -> Dict[str, object]:
        stepd_dir = resolve_stepdprime_dir(Path(cfg.output_root), _normalize_mode(cfg.mode), explicit_root=cfg.stepDprime_root, for_write=True)
        marker_dir = stepd_dir / "pipeline_markers"
        try:
            base = self.run_base_cluster(cfg)
            profile_results: Dict[str, Dict[str, str]] = {}
            for profile in cfg.profiles:
                p = self.run_final_profile(cfg, profile)
                profile_results[profile] = {
                    "train": _normalize_output_path(p.state_train_path, output_root=Path(cfg.output_root)),
                    "test": _normalize_output_path(p.state_test_path, output_root=Path(cfg.output_root)),
                    "summary": _normalize_output_path(p.summary_path, output_root=Path(cfg.output_root)),
                    "embedding_all": _normalize_output_path(p.embedding_all_path, output_root=Path(cfg.output_root)),
                }
            out = {
                "mode": cfg.mode,
                "symbol": cfg.symbol,
                "encoder_type": _normalize_encoder_type(cfg.encoder_type),
                "profiles": profile_results,
                "output_dir": _normalize_output_path(stepd_dir, output_root=Path(cfg.output_root)),
                "embedding_meta": _normalize_output_path(stepd_dir / "stepDprime_embedding_meta.json", output_root=Path(cfg.output_root)),
                "state_schema": _normalize_output_path(stepd_dir / "stepDprime_state_schema.json", output_root=Path(cfg.output_root)),
                "dprime_cluster": {
                    "status": base.status,
                    "cluster_daily_path": _normalize_output_path(base.cluster_daily_path, output_root=Path(cfg.output_root)),
                    "cluster_summary_path": _normalize_output_path(base.cluster_summary_path, output_root=Path(cfg.output_root)),
                },
            }
            (stepd_dir / f"stepDprime_summary_{cfg.symbol}.json").write_text(json.dumps(_json_safe(out), indent=2, ensure_ascii=False), encoding="utf-8")
            return out
        except Exception as exc:
            tb_text = traceback.format_exc()
            traceback_path = stepd_dir / f"stepDprime_traceback_{cfg.symbol}.log"
            failure_summary_path = stepd_dir / f"stepDprime_failure_summary_{cfg.symbol}.json"
            stepd_dir.mkdir(parents=True, exist_ok=True)
            traceback_path.write_text(tb_text, encoding="utf-8")
            failure_payload = {
                "symbol": cfg.symbol,
                "mode": str(cfg.mode),
                "status": "FAILED",
                "error": repr(exc),
                "error_type": type(exc).__name__,
                "traceback_path": _normalize_output_path(traceback_path, output_root=Path(cfg.output_root)),
                "created_at": _utcnow_iso(),
                "base_meta_path": _normalize_output_path(stepd_dir / f"stepDprime_base_meta_{cfg.symbol}.json", output_root=Path(cfg.output_root)),
                "profile_summaries": [
                    _normalize_output_path(stepd_dir / f"stepDprime_profile_summary_{profile}_{cfg.symbol}.json", output_root=Path(cfg.output_root))
                    for profile in cfg.profiles
                ],
            }
            _write_json(failure_summary_path, failure_payload)
            print("[STEPDPRIME_FAIL_TRACEBACK_BEGIN]")
            print(tb_text.rstrip())
            print("[STEPDPRIME_FAIL_TRACEBACK_END]")
            raise
        finally:
            marker_dir.mkdir(parents=True, exist_ok=True)
            _cleanup_running_markers(marker_dir, prefixes=("DPrimeBaseCluster", "DPrimeFinal_", "StepB", "StepC", "StepE_"))
