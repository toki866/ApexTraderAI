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
from sklearn.utils.extmath import randomized_svd

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
    verbose: bool = True
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

        num_cols = [c for c in data.columns if c != "Date" and pd.api.types.is_numeric_dtype(data[c])]
        bnf_cols = [
            "ret_1", "ret_5", "ret_20", "range_atr", "body_ratio", "body_atr", "upper_wick_ratio", "lower_wick_ratio",
            "Gap", "ATR_norm", "gap_atr", "vol_log_ratio_20", "vol_chg", "dev_z_25", "bnf_score",
        ]
        mix_cols = bnf_cols + ["RSI", "MACD_hist", "macd_hist_delta", "macd_hist_cross_up", "clv", "distribution_day", "dist_count_25", "absorption_day", "cmf_20"]
        all_cols = [c for c in num_cols if c not in {"Open", "High", "Low", "Close", "Volume", "Close_anchor"}]

        results: Dict[str, object] = {
            "mode": cfg.mode,
            "symbol": cfg.symbol,
            "profiles": {},
            "output_dir": str(stepd_dir),
            "pred_source_selected": pred_meta.get("pred_source_selected", ""),
            "pred_source_mode": pred_meta.get("pred_source_mode", ""),
            "pred_available_horizons": available,
            "pred_missing_horizons_filled": missing_filled,
            "pca_diagnostics_files": [],
        }
        date_list = data["Date"].tolist()
        idx_train = [i for i, d in enumerate(date_list) if tr_s <= d <= tr_e]
        idx_test = [i for i, d in enumerate(date_list) if te_s <= d <= te_e]
        if min(idx_test) < cfg.l_past - 1:
            raise RuntimeError("insufficient history before test_start for L_past window")

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

                    def _build_rows(indices: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
                        xp, xf, ds = [], [], []
                        for i in indices:
                            if i < cfg.l_past - 1:
                                continue
                            hist = data.iloc[i - cfg.l_past + 1 : i + 1][past_cols].to_numpy(dtype=float)
                            xp.append(hist.reshape(-1))
                            ds.append(data.iloc[i]["Date"])
                            if pred_type == "3scale":
                                p = data.iloc[i][[f"pred_ret_{k:02d}" for k in range(1, cfg.pred_k + 1)]].to_numpy(dtype=float)
                                xf.append(np.concatenate([p[:5], p[:10], p[:20]], axis=0))
                            else:
                                xf.append(data.iloc[i][pred_use_cols].to_numpy(dtype=float))
                        return np.asarray(xp, float), np.asarray(xf, float), ds

                    t_state_rows = time.perf_counter()
                    Xp_tr, Xf_tr, Dtr = _build_rows(idx_train)
                    Xp_te, Xf_te, Dte = _build_rows(idx_test)
                    _log_timing(f"profile_{profile}.state_generation", t_state_rows)
                    if len(Dtr) == 0 or len(Dte) == 0:
                        raise RuntimeError(f"profile={profile}: no rows for train/test")

                    Xp_tr_before = int(np.asarray(Xp_tr, dtype=float).size - int(np.isfinite(np.asarray(Xp_tr, dtype=float)).sum()))
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
                    _log_timing(f"profile_{profile}.projection", t_proj)

                    pca_diag = {
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
                    }
                    pca_diag_path = stepd_dir / f"stepDprime_pca_diagnostics_{profile}_{cfg.symbol}.json"
                    pca_diag_path.write_text(json.dumps(_json_safe(pca_diag), indent=2, ensure_ascii=False), encoding="utf-8")
                    results["pca_diagnostics_files"].append(str(pca_diag_path))
                    print(f"[STEPDPRIME_PCA] profile={profile} Xp_tr_shape={Xp_tr.shape} Xf_tr_shape={Xf_tr.shape}")
                    print(f"[STEPDPRIME_PCA] nonfinite_before={Xp_tr_before} zero_var_cols={xp_tr_diag['zero_var_cols']}")
                    print(f"[STEPDPRIME_PCA] method={pca_p_diag['pca_method_used']} fallback_used={pca_p_diag['fallback_used']}")
                    print(f"[STEPDPRIME_PCA] z_past_dim_used={d_p} z_pred_dim_used={d_f}")

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

                    def _to_df(ds: List[pd.Timestamp], zp: np.ndarray, zf: np.ndarray) -> pd.DataFrame:
                        out = pd.DataFrame({"Date": pd.to_datetime(ds).strftime("%Y-%m-%d")})
                        for j in range(zp.shape[1]):
                            out[f"zp_{j:03d}"] = zp[:, j]
                        for j in range(zf.shape[1]):
                            out[f"zf_{j:03d}"] = zf[:, j]
                        out["gap_atr"] = [_scalar(d, "gap_atr", 0.0) for d in pd.to_datetime(ds)]
                        out["ATR_norm"] = [_scalar(d, "ATR_norm", 0.0) for d in pd.to_datetime(ds)]
                        out["cluster_id_raw20"] = [int(round(_scalar(d, "cluster_id_raw20", 0.0))) for d in pd.to_datetime(ds)]
                        out["cluster_id_stable"] = [int(round(_scalar(d, "cluster_id_stable", 0.0))) for d in pd.to_datetime(ds)]
                        out["rare_flag_raw20"] = [int(round(_scalar(d, "rare_flag_raw20", 0.0))) for d in pd.to_datetime(ds)]
                        out["pos_prev"] = 0.0
                        out["action_prev"] = 0.0
                        out["time_in_trade"] = 0.0
                        return out

                    df_tr, df_te = _to_df(Dtr, Zp_tr, Zf_tr), _to_df(Dte, Zp_te, Zf_te)
                    p_tr = stepd_dir / f"stepDprime_state_train_{profile}_{cfg.symbol}.csv"
                    p_te = stepd_dir / f"stepDprime_state_test_{profile}_{cfg.symbol}.csv"
                    s_path = stepd_dir / f"stepDprime_split_summary_{profile}_{cfg.symbol}.csv"
                    t_save = time.perf_counter()
                    df_tr.to_csv(p_tr, index=False)
                    df_te.to_csv(p_te, index=False)
                    pd.DataFrame([
                        {"key": "mode", "value": cfg.mode}, {"key": "symbol", "value": cfg.symbol}, {"key": "profile", "value": profile},
                        {"key": "train_start", "value": str(tr_s.date())}, {"key": "train_end", "value": str(tr_e.date())},
                        {"key": "test_start", "value": str(te_s.date())}, {"key": "test_end", "value": str(te_e.date())},
                        {"key": "L_past", "value": cfg.l_past}, {"key": "pred_type", "value": pred_type}, {"key": "pred_k", "value": cfg.pred_k},
                        {"key": "z_past_dim", "value": int(d_p)}, {"key": "z_pred_dim", "value": int(d_f)},
                        {"key": "rows_train_written", "value": int(len(df_tr))}, {"key": "rows_test_written", "value": int(len(df_te))},
                        {"key": "past_feature_channels", "value": "|".join(past_cols)},
                        {"key": "pred_source_file", "value": pred_meta.get("pred_source_selected", "")},
                        {"key": "pred_source_mode", "value": pred_meta.get("pred_source_mode", "")},
                        {"key": "pred_available_horizons", "value": "|".join(str(x) for x in available)},
                        {"key": "pred_missing_horizons_filled", "value": json.dumps(missing_filled, ensure_ascii=False)},
                        {"key": "dprime_cluster_source", "value": "stepDprime_cluster_daily_assign"},
                        {"key": "dprime_cluster_status", "value": cluster_status},
                        {"key": "fit_stats", "value": f"train_only:{tr_s.date()}..{tr_e.date()}"},
                        {"key": "pca_components_shape", "value": f"past={comp_p.shape},pred={comp_f.shape}"},
                    ]).to_csv(s_path, index=False)

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

                    _log_timing(f"profile_{profile}.total", t_profile_total)

                    results["profiles"][profile] = {"train": str(p_tr), "test": str(p_te), "summary": str(s_path)}

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
                    "base_features_path": str(base_path),
                    "cluster_daily_path": str(cluster_daily_path),
                    "cluster_summary_path": str(cluster_summary_path),
                    "cluster_raw_stats_path": str(cluster_raw_stats_path),
                    "status": "READY",
                    "created_at": _utcnow_iso(),
                })
                ready = write_status_marker(marker_dir, "DPrimeBaseCluster", "READY", {
                    "symbol": cfg.symbol,
                    "base_features_path": str(base_path),
                    "cluster_daily_path": str(cluster_daily_path),
                    "cluster_summary_path": str(cluster_summary_path),
                    "base_meta_path": str(base_meta_path),
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
                "base_features_path": str(stepd_dir / f"stepDprime_base_features_{cfg.symbol}.csv"),
                "cluster_daily_path": str(stepd_dir / f"stepDprime_cluster_daily_assign_{cfg.symbol}.csv"),
                "cluster_summary_path": str(stepd_dir / f"stepDprime_cluster_summary_{cfg.symbol}.json"),
                "status": "FAILED",
                "created_at": _utcnow_iso(),
                "diagnostics_path": str(diag_path),
            })
            write_status_marker(
                marker_dir,
                "DPrimeBaseCluster",
                "FAILED",
                {"symbol": cfg.symbol, "error": repr(e), "diagnostics_path": str(diag_path), "base_meta_path": str(base_meta_path)},
            )
            raise

    def run_final_profile(self, cfg: StepDPrimeConfig, profile: str, *, force_cpu: bool = False) -> StepDPrimeProfileResult:
        from ai_core.utils.file_ready_utils import write_status_marker

        timing = cfg.timing_logger if isinstance(cfg.timing_logger, TimingLogger) else TimingLogger.disabled()
        paths = self._resolve_paths(cfg)
        stepd_dir = paths["stepd_dir"]
        marker_dir = stepd_dir / "pipeline_markers"
        marker_name = f"DPrimeFinal_{profile}"
        write_status_marker(marker_dir, marker_name, "RUNNING", {"symbol": cfg.symbol, "profile": profile})
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
            family, pred_type = _parse_profile(profile)
            profile_summary_path = stepd_dir / f"stepDprime_profile_summary_{profile}_{cfg.symbol}.json"
            _write_json(profile_summary_path, {
                "profile": profile,
                "family": family,
                "pred_type": pred_type,
                "state_train_path": str(train_path),
                "state_test_path": str(test_path),
                "embeddings_all_path": str(emb_path),
                "pred_source_selected": str(split_meta.get("pred_source_file", "")),
                "pred_source_mode": str(split_meta.get("pred_source_mode", "")),
                "pred_available_horizons": str(split_meta.get("pred_available_horizons", "")),
                "force_cpu": bool(force_cpu),
                "status": "READY",
                "created_at": _utcnow_iso(),
            })
            ready = write_status_marker(marker_dir, marker_name, "READY", {"profile": profile, "symbol": cfg.symbol, "force_cpu": bool(force_cpu), "profile_summary_path": str(profile_summary_path)})
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
            family, pred_type = _parse_profile(profile)
            profile_summary_path = stepd_dir / f"stepDprime_profile_summary_{profile}_{cfg.symbol}.json"
            _write_json(profile_summary_path, {
                "profile": profile,
                "family": family,
                "pred_type": pred_type,
                "state_train_path": str(stepd_dir / f"stepDprime_state_train_{profile}_{cfg.symbol}.csv"),
                "state_test_path": str(stepd_dir / f"stepDprime_state_test_{profile}_{cfg.symbol}.csv"),
                "embeddings_all_path": str(stepd_dir / "embeddings" / f"stepDprime_{profile}_{cfg.symbol}_embeddings_all.csv"),
                "pred_source_selected": "",
                "pred_source_mode": "",
                "pred_available_horizons": "",
                "force_cpu": bool(force_cpu),
                "status": "FAILED",
                "created_at": _utcnow_iso(),
                "error": repr(e),
            })
            write_status_marker(marker_dir, marker_name, "FAILED", {"symbol": cfg.symbol, "profile": profile, "error": repr(e), "force_cpu": bool(force_cpu), "profile_summary_path": str(profile_summary_path)})
            raise

    def run(self, cfg: StepDPrimeConfig) -> Dict[str, object]:
        base = self.run_base_cluster(cfg)
        profile_results: Dict[str, Dict[str, str]] = {}
        for profile in cfg.profiles:
            p = self.run_final_profile(cfg, profile)
            profile_results[profile] = {
                "train": p.state_train_path,
                "test": p.state_test_path,
                "summary": p.summary_path,
                "embedding_all": p.embedding_all_path,
            }
        out = {
            "mode": cfg.mode,
            "symbol": cfg.symbol,
            "profiles": profile_results,
            "output_dir": str(resolve_stepdprime_dir(Path(cfg.output_root), _normalize_mode(cfg.mode), explicit_root=cfg.stepDprime_root, for_write=True)),
            "dprime_cluster": {
                "status": base.status,
                "cluster_daily_path": base.cluster_daily_path,
                "cluster_summary_path": base.cluster_summary_path,
            },
        }
        stepd_dir = Path(out["output_dir"])
        (stepd_dir / f"stepDprime_summary_{cfg.symbol}.json").write_text(json.dumps(_json_safe(out), indent=2, ensure_ascii=False), encoding="utf-8")
        return out
