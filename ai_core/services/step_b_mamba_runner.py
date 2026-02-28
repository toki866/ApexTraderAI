"""ai_core.services.step_b_mamba_runner

StepB(Mamba full: periodic + tech + OHLCV) — horizon-specific multi-model forecasting.

Why this exists
- You correctly pointed out: "take first 5 rows from a 20-step model" is not the same as
  a model trained specifically for horizon=5 (loss / capacity differs).
- Therefore this runner trains **separate models per horizon** (e.g. H=1,5,10,20).
  Each model predicts the path 1..H. Daily outputs are written per horizon.

Inputs
- prices_df: must include Date, Close (and optionally OHLCV)
- features_df: must include Date and numeric features
- StepA daily window snapshots:
    <output_root>/stepA/<mode>/daily/stepA_daily_features_<SYMBOL>_YYYY_MM_DD.csv

Core outputs (mode-separated under <output_root>/stepB/<mode>/)
- stepB_pred_path_mamba_<SYMBOL>.csv
    1 row = anchor date, columns:
      Pred_Close_t_plus_01 (from H=1 model),
      Pred_Close_t_plus_05 (from H=5 model),
      Pred_Close_t_plus_10 (from H=10 model),
      Pred_Close_t_plus_20 (from H=20 model) ...
- stepB_pred_close_mamba_<SYMBOL>.csv
    target-date aligned, columns:
      Pred_Close_MAMBA_h01/h05/h10/h20 and Delta_Close_pred_* for each horizon
    plus legacy aliases: Pred_Close_MAMBA, Delta_Close_pred_MAMBA (h=1)
- stepB_daily_manifest_<SYMBOL>.csv
    1 row = anchor date, columns:
      pred_path_h01/h05/h10/h20 ... (each points to a daily path csv)
      stepA_features_path
- daily/stepB_daily_pred_mamba_hXX_<SYMBOL>_YYYY_MM_DD.csv
    per anchor date, **H rows** with:
      step_ahead_bdays=1..H, Date_target, Pred_Close, Pred_ret_from_anchor

Leakage control (SIM)
- Standardization is fit on TRAIN slice only.
- Training samples for horizon H require all targets t+1..t+H <= train_end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ai_core.utils.paths import resolve_repo_path
from typing import Any, Dict, List, Tuple, Optional

import json
import random
import os
import re

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Optional Wavelet front-end (leakage-safe)
#   - Enabled by env: STEPB_MAMBA_USE_WAVELET=1
#   - Applies to FULL variant only (not periodic).
#   - Wavelet features are computed **per lookback window** from Close (past-only),
#     and appended as additional channels to the sequence input.
#   - This avoids future-leakage that would happen if you wavelet-transform the full
#     time-series globally and then slice windows.
#
# Env:
#   STEPB_MAMBA_USE_WAVELET=0/1
#   STEPB_MAMBA_WAVELET_TYPE=db4
#   STEPB_MAMBA_WAVELET_LEVELS=3
#   STEPB_MAMBA_WAVELET_MODE=periodization
#   STEPB_MAMBA_WAVELET_SOURCE_COL=Close
# ----------------------------------------------------------------------

def _env_flag(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "")
    if v == "":
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _wavelet_levels_eff(window_len: int, wavelet_type: str, levels_req: int) -> int:
    try:
        import pywt  # type: ignore
    except Exception as e:
        raise RuntimeError("PyWavelets is required for wavelet features. pip install PyWavelets") from e

    w = pywt.Wavelet(str(wavelet_type))
    max_level = pywt.dwt_max_level(int(window_len), w.dec_len)
    return max(0, min(int(levels_req), int(max_level)))


def _wavelet_detail_matrix(
    x: np.ndarray,
    wavelet_type: str,
    levels_eff: int,
    mode: str = "periodization",
) -> np.ndarray:
    """Return reconstructed detail-series matrix for a 1D window.

    Returns
    -------
    details : np.ndarray shape (T, levels_eff)
    """
    try:
        import pywt  # type: ignore
    except Exception as e:
        raise RuntimeError("PyWavelets is required for wavelet features. pip install PyWavelets") from e

    x = np.asarray(x, dtype=np.float32)
    T = int(len(x))
    if levels_eff <= 0 or T <= 0:
        return np.zeros((T, 0), dtype=np.float32)

    w = pywt.Wavelet(str(wavelet_type))
    coeffs = pywt.wavedec(x, w, level=int(levels_eff), mode=str(mode))
    details_list: List[np.ndarray] = []
    for j in range(1, int(levels_eff) + 1):
        coeffs_j = [np.zeros_like(c) for c in coeffs]
        coeffs_j[-j] = coeffs[-j]
        rec = pywt.waverec(coeffs_j, w, mode=str(mode))
        if len(rec) > T:
            rec = rec[:T]
        elif len(rec) < T:
            rec = np.pad(rec, (0, T - len(rec)), mode="edge")
        details_list.append(rec.astype(np.float32))

    if not details_list:
        return np.zeros((T, 0), dtype=np.float32)
    return np.stack(details_list, axis=1).astype(np.float32)


def _zscore_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mu = float(np.nanmean(x)) if len(x) else 0.0
    sd = float(np.nanstd(x)) if len(x) else 1.0
    if not np.isfinite(sd) or sd < 1e-6:
        sd = 1.0
    return ((x - mu) / sd).astype(np.float32)


def _require_torch() -> None:
    try:
        import torch  # noqa: F401
        import torch.nn as nn  # noqa: F401
        from torch.utils.data import DataLoader, TensorDataset  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "PyTorch import failed. StepB(Mamba) requires torch. "
            f"Original error: {e}"
        ) from e


@dataclass
class StepBAgentResult:
    success: bool
    message: str
    agent: str
    out_dir: str
    csv_paths: Dict[str, str]
    metrics: Dict[str, Any]
    csv_paths_list: List[str] = field(default_factory=list)

    def iter_csv_paths(self) -> List[str]:
        if self.csv_paths_list:
            return list(self.csv_paths_list)
        return list(self.csv_paths.values())


def _get(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)

def _is_periodic_variant(cfg: Any) -> bool:
    v = str(_get(cfg, 'variant', 'full')).strip().lower()
    return v in ('periodic', 'periodic_only', 'p', 'cal', 'calendar')

def _select_periodic_feature_cols(df: pd.DataFrame) -> List[str]:
    """Select periodic-only feature columns (44 columns by default).

    StepA periodic feature set (vFinal) is **44 columns**:
      - per_cal_* (10)
      - per_astro_* (4)
      - per_planet_* (16)  # retro_flag + speed per planet
      - per_h2_* / per_h3_* (14)  # harmonics (usually single-value per harmonic)

    Therefore periodic-only selection must NOT rely on only sin/cos names.

    Selection rule (preferred):
      - columns whose name starts with 'per_' (excluding 'Date')
      - keep the original order in df

    Strict mode can be disabled with:
      STEPB_PERIODIC_ONLY_STRICT=0
    """
    strict = os.environ.get("STEPB_PERIODIC_ONLY_STRICT", "1").strip().lower() not in ("0", "false", "no", "off")

    # Preferred: StepA vFinal periodic feature names start with 'per_'
    cols_per: List[str] = []
    for c in df.columns:
        if str(c) == "Date":
            continue
        if str(c).startswith("per_") and pd.api.types.is_numeric_dtype(df[c]):
            cols_per.append(str(c))

    if len(cols_per) > 0:
        if strict and len(cols_per) != 44:
            preview = ", ".join(cols_per[:14]) + (" ..." if len(cols_per) > 14 else "")
            raise RuntimeError(
                f"Periodic-only feature selection expected 44 'per_*' columns, got {len(cols_per)}. "
                f"Check StepA periodic feature generation. Selected: [{preview}]"
            )
        return cols_per

    # ---- Legacy fallback (older projects): sin/cos only naming ----
    cols_primary: List[str] = []
    pat1 = re.compile(r"^(sin|cos)[_]?k?\d{1,2}$", re.IGNORECASE)
    pat2 = re.compile(r"^(sin|cos)[_]\w*\d{1,2}$", re.IGNORECASE)  # e.g., sin_k01, cos_k22
    for c in df.columns:
        if c in ("Date", "Close"):
            continue
        lc = str(c).lower()
        if lc.startswith("sin_") or lc.startswith("cos_"):
            cols_primary.append(str(c))
            continue
        if pat1.match(lc) or pat2.match(lc):
            cols_primary.append(str(c))
            continue

    if len(cols_primary) == 0:
        pat_f = re.compile(r"(sin|cos).*\d{1,2}", re.IGNORECASE)
        for c in df.columns:
            if c in ("Date", "Close"):
                continue
            lc = str(c).lower()
            if "periodic" in lc and pat_f.search(lc):
                cols_primary.append(str(c))

    cols_primary = [c for c in cols_primary if pd.api.types.is_numeric_dtype(df[c])]

    if strict and len(cols_primary) != 44:
        preview = ", ".join(cols_primary[:14]) + (" ..." if len(cols_primary) > 14 else "")
        raise RuntimeError(
            f"Periodic-only feature selection expected 44 columns, got {len(cols_primary)}. "
            f"Check StepA periodic feature names. Selected: [{preview}]"
        )

    return cols_primary


def _get_target_mode(cfg: Any, is_periodic: bool) -> str:
    """Return target mode for Mamba forecasting.

    Supported:
      - close  : predict absolute Close_{t+s}
      - delta  : predict Close_{t+s} - Close_t
      - ret    : predict Close_{t+s}/Close_t - 1
      - logret : predict log(Close_{t+s}/Close_t)

    Defaults:
      - periodic-only: logret (because periodic features alone do not carry absolute price level)
      - full: close

    Overrides:
      - cfg.target_mode (full)
      - cfg.periodic_target_mode (periodic-only)
      - env STEPB_MAMBA_TARGET_MODE (global)
      - env STEPB_PERIODIC_TARGET_MODE (periodic-only)
    """
    env_global = os.environ.get("STEPB_MAMBA_TARGET_MODE", "").strip().lower()
    env_periodic = os.environ.get("STEPB_PERIODIC_TARGET_MODE", "").strip().lower()

    if is_periodic:
        v = str(_get(cfg, "periodic_target_mode", "")).strip().lower()
        mode = v or env_periodic or env_global or "logret"
    else:
        v = str(_get(cfg, "target_mode", "")).strip().lower()
        mode = v or env_global or "close"

    if mode in ("price", "abs", "absolute", "close_price"):
        mode = "close"
    if mode in ("return", "r"):
        mode = "ret"
    if mode in ("log_return", "log", "lr"):
        mode = "logret"
    if mode not in ("close", "delta", "ret", "logret"):
        mode = "logret" if is_periodic else "close"
    return mode


def _target_transform(close_t_plus_s: float, close_t: float, mode: str) -> float:
    """Transform absolute Close into the target value (y)."""
    if mode == "close":
        return float(close_t_plus_s)
    if not np.isfinite(close_t_plus_s) or not np.isfinite(close_t) or close_t <= 0.0 or close_t_plus_s <= 0.0:
        return float("nan")
    if mode == "delta":
        return float(close_t_plus_s - close_t)
    if mode == "ret":
        return float((close_t_plus_s / close_t) - 1.0)
    # logret
    return float(np.log(close_t_plus_s / close_t))


def _target_inverse(pred_y: float, close_t: float, mode: str) -> float:
    """Inverse transform: target value (y) -> predicted Close."""
    if mode == "close":
        return float(pred_y)
    if not np.isfinite(pred_y) or not np.isfinite(close_t) or close_t <= 0.0:
        return float("nan")
    if mode == "delta":
        return float(close_t + pred_y)
    if mode == "ret":
        return float(close_t * (1.0 + pred_y))
    # logret (clip to avoid exp overflow from pathological outputs)
    return float(close_t * float(np.exp(np.clip(pred_y, -20.0, 20.0))))


def _target_inverse_vec(pred_y: np.ndarray, close_t: np.ndarray, mode: str) -> np.ndarray:
    """Vectorized inverse transform: y -> Close."""
    pred_y = pred_y.astype(float)
    close_t = close_t.astype(float)

    if mode == "close":
        return pred_y

    out = np.full_like(pred_y, np.nan, dtype=float)
    ok = np.isfinite(pred_y) & np.isfinite(close_t) & (close_t > 0.0)
    if not np.any(ok):
        return out

    if mode == "delta":
        out[ok] = close_t[ok] + pred_y[ok]
        return out
    if mode == "ret":
        out[ok] = close_t[ok] * (1.0 + pred_y[ok])
        return out

    # logret
    out[ok] = close_t[ok] * np.exp(np.clip(pred_y[ok], -20.0, 20.0))
    return out

def _as_date(x: Any) -> pd.Timestamp:
    if x is None:
        return pd.NaT
    if isinstance(x, pd.Timestamp):
        return x
    return pd.to_datetime(x, errors="coerce")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _jsonify(x: Any) -> Any:
    if isinstance(x, (pd.Timestamp,)):
        return x.strftime("%Y-%m-%d")
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, dict):
        return {k: _jsonify(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_jsonify(v) for v in x]
    return x


def _parse_horizons(cfg: Any) -> List[int]:
    if _is_periodic_variant(cfg):
        hz = _get(cfg, 'periodic_snapshot_horizons', (1, 5, 10, 20))
    else:
        hz = _get(cfg, 'horizons', (1, 5, 10, 20))
    if isinstance(hz, (list, tuple)):
        hs = [int(x) for x in hz]
    else:
        s = str(hz).strip()
        if not s:
            hs = [1, 5, 10, 20]
        else:
            hs = [int(x) for x in re.split(r"[,\s]+", s) if x.strip()]
    hs = sorted(set([h for h in hs if h >= 1]))
    return hs if hs else [1]


def _infer_output_root(app_config: Any) -> Path:
    # AppConfig may store output_root nested
    try:
        data = getattr(app_config, "data", None)
        if data is not None and getattr(data, "output_root", None):
            return Path(getattr(data, "output_root"))
    except Exception:
        pass
    try:
        v = getattr(app_config, "output_root", None)
        if v:
            return Path(v)
    except Exception:
        pass
    return resolve_repo_path("output")


def _infer_split(app_config: Any, cfg: Any) -> Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp, str]:
    mode = str(_get(cfg, "mode", "sim")).strip().lower()
    if mode in ("ops", "prod", "production"):
        mode = "live"

    train_start = _as_date(_get(cfg, "train_start", None))
    train_end = _as_date(_get(cfg, "train_end", None))
    test_start = _as_date(_get(cfg, "test_start", None))
    test_end = _as_date(_get(cfg, "test_end", None))

    dr = _get(app_config, "date_range", None)
    if dr is not None:
        try:
            if pd.isna(train_start):
                train_start = _as_date(getattr(dr, "train_start", None))
            if pd.isna(train_end):
                train_end = _as_date(getattr(dr, "train_end", None))
            if pd.isna(test_start):
                test_start = _as_date(getattr(dr, "test_start", None))
            if pd.isna(test_end):
                test_end = _as_date(getattr(dr, "test_end", None))
        except Exception:
            pass

    if pd.isna(train_start):
        train_start = pd.Timestamp("1900-01-01")
    if pd.isna(test_end):
        test_end = pd.Timestamp("2100-01-01")

    if mode == "sim":
        if (not pd.isna(test_start)) and (pd.isna(train_end) or train_end >= test_start):
            train_end = (pd.Timestamp(test_start) - pd.tseries.offsets.BDay(1)).normalize()
        if pd.isna(test_start):
            test_start = (pd.Timestamp(test_end) - pd.DateOffset(months=3)).normalize()
    else:
        if pd.isna(train_end):
            train_end = pd.Timestamp(test_end)
        if pd.isna(test_start):
            test_start = pd.Timestamp(train_end)

    return train_start, train_end, test_start, test_end, mode


def _build_model(input_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2):
    import torch
    import torch.nn as nn

    class FallbackMambaBlock(nn.Module):
        """Pure PyTorch SSM-like fallback block (non-GRU/Transformer)."""

        def __init__(self, d_model: int) -> None:
            super().__init__()
            self.in_proj = nn.Linear(d_model, d_model * 2)
            self.a = nn.Parameter(torch.randn(d_model) * 0.02)
            self.b = nn.Parameter(torch.randn(d_model) * 0.02)
            self.c = nn.Parameter(torch.randn(d_model) * 0.02)
            self.out_proj = nn.Linear(d_model, d_model)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            u, g = self.in_proj(x).chunk(2, dim=-1)
            g = torch.sigmoid(g)
            bsz, tlen, dim = u.shape
            state = torch.zeros((bsz, dim), device=x.device, dtype=x.dtype)
            outs: List[torch.Tensor] = []
            a = -torch.nn.functional.softplus(self.a)
            b = torch.tanh(self.b)
            c = torch.tanh(self.c)
            for t in range(tlen):
                ut = u[:, t, :]
                state = torch.exp(a) * state + b * ut
                yt = c * state
                outs.append((yt * g[:, t, :]).unsqueeze(1))
            y = torch.cat(outs, dim=1)
            return self.out_proj(y)

    class MambaSeqModel(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, out_dim: int, num_layers: int) -> None:
            super().__init__()
            self.in_proj = nn.Linear(input_dim, hidden_dim)
            self.norm_in = nn.LayerNorm(hidden_dim)
            self.layers = nn.ModuleList()
            self.has_mamba_ssm = False
            try:
                from mamba_ssm import Mamba  # type: ignore

                for _ in range(int(max(1, num_layers))):
                    self.layers.append(
                        nn.Sequential(
                            nn.LayerNorm(hidden_dim),
                            Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2),
                        )
                    )
                self.has_mamba_ssm = True
            except Exception:
                for _ in range(int(max(1, num_layers))):
                    self.layers.append(nn.Sequential(nn.LayerNorm(hidden_dim), FallbackMambaBlock(hidden_dim)))
            self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, out_dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.norm_in(self.in_proj(x))
            for layer in self.layers:
                h = h + layer(h)
            return self.head(h[:, -1, :])

    return MambaSeqModel(input_dim, hidden_dim, out_dim, num_layers=num_layers)


def _scan_stepa_daily_dir(out_root: Path, mode: str, symbol: str, override_dir: Optional[str]) -> List[Tuple[pd.Timestamp, Path]]:
    if override_dir:
        ddir = Path(override_dir)
    else:
        ddir = out_root / "stepA" / mode / "daily"
    if not ddir.exists():
        return []
    pat = re.compile(rf"^stepA_daily_features_{re.escape(symbol)}_(\d{{4}})_(\d{{2}})_(\d{{2}})\.csv$")
    items: List[Tuple[pd.Timestamp, Path]] = []
    for p in sorted(ddir.glob(f"stepA_daily_features_{symbol}_*.csv")):
        m = pat.match(p.name)
        if not m:
            continue
        ds = pd.Timestamp(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
        items.append((ds, p))
    return items


def run_mamba_multi_model_by_horizon(
    app_config: Any,
    symbol: str,
    prices_df: pd.DataFrame,
    features_df: pd.DataFrame,
    cfg: Any,
) -> StepBAgentResult:
    _require_torch()
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    sym = str(symbol)
    train_start, train_end, test_start, test_end, mode = _infer_split(app_config, cfg)

    out_root = _infer_output_root(app_config)
    stepb_dir = out_root / "stepB" / mode
    stepb_dir.mkdir(parents=True, exist_ok=True)
    # Variant (full vs periodic snapshot)
    is_periodic = _is_periodic_variant(cfg)
    target_mode = _get_target_mode(cfg, is_periodic)
    out_tag = str(_get(cfg, 'periodic_output_tag', 'mamba_periodic')) if is_periodic else 'mamba'
    col_agent = "MAMBA_PERIODIC" if is_periodic else "MAMBA"
    manifest_suffix = str(_get(cfg, 'periodic_manifest_suffix', 'periodic')) if is_periodic else ''
    daily_dirname = str(_get(cfg, 'periodic_daily_dirname', 'daily_periodic')) if is_periodic else 'daily'
    daily_dir = stepb_dir / daily_dirname
    daily_dir.mkdir(parents=True, exist_ok=True)
    periodic_endpoints = list(_get(cfg, 'periodic_endpoints', (1, 5, 10, 20)))
    (stepb_dir / "models").mkdir(parents=True, exist_ok=True)

    lookback_days = int(_get(cfg, "lookback_days", 30))
    horizons = _parse_horizons(cfg)

    seed = int(_get(cfg, "seed", 42))
    epochs = int(_get(cfg, "epochs", 20))
    batch_size = int(_get(cfg, "batch_size", 128))
    lr = float(_get(cfg, "lr", 1e-3))
    hidden_dim = int(_get(cfg, "hidden_dim", 64))
    standardize = bool(_get(cfg, "standardize", True))

    use_stepa_daily = bool(_get(cfg, "use_stepa_daily_windows", True))
    stepa_daily_dir_override = _get(cfg, "stepa_daily_dir", None)

    _set_seed(seed)

    # ---- Validate / prepare base DF ----
    if "Date" not in prices_df.columns or "Close" not in prices_df.columns:
        raise ValueError("prices_df must contain Date and Close.")
    if "Date" not in features_df.columns:
        raise ValueError("features_df must contain Date.")

    df_p = prices_df.copy()
    df_f = features_df.copy()
    df_p["Date"] = pd.to_datetime(df_p["Date"], errors="coerce")
    df_f["Date"] = pd.to_datetime(df_f["Date"], errors="coerce")
    df_p = df_p.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df_f = df_f.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    df = df_p.merge(df_f, on="Date", how="left", suffixes=("", "_feat")).sort_values("Date").reset_index(drop=True)
    df = df[(df["Date"] >= train_start) & (df["Date"] <= test_end)].copy().reset_index(drop=True)

    max_h = int(max(horizons))
    if len(df) < (lookback_days + max_h + 10):
        raise RuntimeError(f"Not enough rows for mamba. rows={len(df)} lookback={lookback_days} max_h={max_h}")


    # Feature columns
    #   - full: numeric excluding Date/Close, with OHLCV basics ensured
    #   - periodic: sin/cos 44 columns only (strict by default)
    if is_periodic:
        feature_cols = _select_periodic_feature_cols(df)
    else:
        feature_cols: List[str] = []
        for c in df.columns:
            if c in ("Date", "Close"):
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                feature_cols.append(c)

        # Ensure OHLCV basics included if present (except Close)
        for c in ("Open", "High", "Low", "Volume"):
            if c in df.columns and c not in feature_cols:
                feature_cols.append(c)

    if not feature_cols:
        raise RuntimeError("No numeric feature columns found.")

    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce").ffill().fillna(0.0)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce").ffill().fillna(0.0)

    # Standardization fit on TRAIN only (SIM leakage rule)
    train_mask = (df["Date"] >= train_start) & (df["Date"] <= train_end)
    if int(train_mask.sum()) < (lookback_days + max_h + 10):
        raise RuntimeError(f"Too few train rows: {int(train_mask.sum())}")

    X_all = df[feature_cols].to_numpy(dtype=np.float32)
    mu = None
    sd = None
    if standardize:
        mu = X_all[train_mask.to_numpy()].mean(axis=0)
        sd = X_all[train_mask.to_numpy()].std(axis=0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        X_all = (X_all - mu) / sd

    close = df["Close"].to_numpy(dtype=np.float32)
    dates: List[pd.Timestamp] = [pd.Timestamp(x) for x in df["Date"].to_list()]

    # Map Date->(idx, Close) for anchor lookup
    idx_of_date: Dict[pd.Timestamp, int] = {pd.Timestamp(d).normalize(): i for i, d in enumerate(dates)}
    close_of_date: Dict[pd.Timestamp, float] = {pd.Timestamp(d).normalize(): float(close[i]) for i, d in enumerate(dates)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Train a model per horizon, then infer for all anchors ----
    all_anchor_idx = list(range(lookback_days - 1, len(df) - 1))

    # ---- Optional wavelet augmentation (FULL only, leakage-safe per window) ----
    wavelet_enabled = (not is_periodic) and _env_flag("STEPB_MAMBA_USE_WAVELET", False)
    wavelet_type = os.getenv("STEPB_MAMBA_WAVELET_TYPE", "db4")
    wavelet_levels_req = int(os.getenv("STEPB_MAMBA_WAVELET_LEVELS", "3") or "3")
    wavelet_mode = os.getenv("STEPB_MAMBA_WAVELET_MODE", "periodization")
    wavelet_source_col = os.getenv("STEPB_MAMBA_WAVELET_SOURCE_COL", "Close")

    wavelet_levels_eff = 0
    if wavelet_enabled:
        wavelet_levels_eff = _wavelet_levels_eff(lookback_days, wavelet_type, wavelet_levels_req)
        if wavelet_levels_eff <= 0:
            wavelet_enabled = False

    base_dim = int(X_all.shape[1])
    final_dim = int(base_dim + (wavelet_levels_eff if wavelet_enabled else 0))

    if _env_flag("STEPB_MAMBA_USE_WAVELET", False) or wavelet_enabled:
        print(f"[StepB:mamba:{out_tag}] wavelet_enabled={wavelet_enabled} type={wavelet_type} levels_req={wavelet_levels_req} levels_eff={wavelet_levels_eff} source={wavelet_source_col} feature_dim_base={base_dim} feature_dim_final={final_dim}")

    def _append_wavelet_to_seq(base_seq: np.ndarray, close_seq: np.ndarray) -> np.ndarray:
        if not wavelet_enabled:
            return base_seq.astype(np.float32, copy=False)
        cw = _zscore_1d(close_seq)
        details = _wavelet_detail_matrix(
            cw, wavelet_type=wavelet_type, levels_eff=wavelet_levels_eff, mode=wavelet_mode
        )  # (T, L)
        if details.shape[0] != base_seq.shape[0]:
            T = int(base_seq.shape[0])
            if details.shape[0] > T:
                details = details[:T, :]
            else:
                details = np.pad(details, ((0, T - details.shape[0]), (0, 0)), mode="edge")
        return np.concatenate([base_seq.astype(np.float32, copy=False), details.astype(np.float32, copy=False)], axis=1)

    if not wavelet_enabled:
        X_inf = np.stack([X_all[t - lookback_days + 1 : t + 1, :] for t in all_anchor_idx], axis=0).astype(np.float32)
    else:
        X_inf = np.empty((len(all_anchor_idx), lookback_days, final_dim), dtype=np.float32)
        for i, t in enumerate(all_anchor_idx):
            base_seq = X_all[t - lookback_days + 1 : t + 1, :]
            close_seq = close[t - lookback_days + 1 : t + 1]
            X_inf[i, :, :] = _append_wavelet_to_seq(base_seq, close_seq)


    # Store endpoint predictions per horizon for anchor-date alignment
    #   - pred_endpoint_y_by_h: endpoint in target space (y)
    #   - pred_endpoint_close_by_h: endpoint converted back to Close
    pred_endpoint_y_by_h: Dict[int, np.ndarray] = {}
    pred_endpoint_close_by_h: Dict[int, np.ndarray] = {}
    loss_by_h: Dict[int, List[float]] = {}
    train_samples_by_h: Dict[int, int] = {}

    for H in horizons:
        steps = list(range(1, H + 1))

        # Build training set for this horizon
        anchors: List[int] = []
        y_list: List[List[float]] = []
        for t in range(lookback_days - 1, len(df) - H):
            # require all targets inside TRAIN period
            if dates[t + H] > train_end:
                continue
            c0 = float(close[t])
            y = [_target_transform(float(close[t + s]), c0, target_mode) for s in steps]
            if not np.all(np.isfinite(y)):
                continue
            anchors.append(t)
            y_list.append(y)

        if len(anchors) < 200:
            raise RuntimeError(f"Too few training samples for H={H}: {len(anchors)}")


        if not wavelet_enabled:
            X_train = np.stack([X_all[t - lookback_days + 1 : t + 1, :] for t in anchors], axis=0).astype(np.float32)
        else:
            X_train = np.empty((len(anchors), lookback_days, final_dim), dtype=np.float32)
            for i, t in enumerate(anchors):
                base_seq = X_all[t - lookback_days + 1 : t + 1, :]
                close_seq = close[t - lookback_days + 1 : t + 1]
                X_train[i, :, :] = _append_wavelet_to_seq(base_seq, close_seq)

        y_train = np.array(y_list, dtype=np.float32)

        model = _build_model(input_dim=X_train.shape[2], hidden_dim=hidden_dim, out_dim=H, num_layers=int(_get(cfg, "num_layers", 2))).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

        losses: List[float] = []
        model.train()
        for _ep in range(epochs):
            total = 0.0
            n = 0
            for xb, yb in dl:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                total += float(loss.detach().cpu().item()) * int(xb.size(0))
                n += int(xb.size(0))
            losses.append(total / max(1, n))

        # Save model
        model_path = stepb_dir / "models" / f"{out_tag}_h{H:02d}.pt"
        torch.save({"state_dict": model.state_dict(), "H": H, "hidden_dim": hidden_dim, "feature_dim": int(X_train.shape[2])}, model_path)

        # Infer endpoints for all anchors
        model.eval()
        preds: List[np.ndarray] = []
        with torch.no_grad():
            for i0 in range(0, len(X_inf), 1024):
                xb = torch.from_numpy(X_inf[i0:i0+1024]).to(device)
                pb = model(xb).detach().cpu().numpy()  # (B, H)
                preds.append(pb)
        P = np.concatenate(preds, axis=0)  # (N_anchor, H)
        pred_endpoint_y_by_h[H] = P[:, -1].astype(float)  # endpoint in target space (y) at t+H

        loss_by_h[H] = losses
        train_samples_by_h[H] = int(len(anchors))

    
    # Convert endpoint predictions (y) -> Close using anchor Close_t
    anchor_close_arr = np.array([float(close[t]) for t in all_anchor_idx], dtype=float)
    for H in horizons:
        y_end = pred_endpoint_y_by_h.get(H)
        if y_end is None:
            continue
        pred_endpoint_close_by_h[H] = _target_inverse_vec(y_end.astype(float), anchor_close_arr, target_mode)

# ---- Build aggregated outputs ----
    # Anchor-date aligned endpoint CSV
    df_path = pd.DataFrame({"Date_anchor": [dates[t].strftime("%Y-%m-%d") for t in all_anchor_idx]})
    for H in horizons:
        df_path[f"Pred_Close_t_plus_{H:02d}"] = pred_endpoint_close_by_h[H]

    # Target-date aligned CSV
    out = pd.DataFrame({"Date": [d.strftime("%Y-%m-%d") for d in dates]})
    for H in horizons:
        out[f"Pred_Close_{col_agent}_h{H:02d}"] = np.nan
        out[f"Delta_Close_pred_{col_agent}_h{H:02d}"] = np.nan

    for k, t in enumerate(all_anchor_idx):
        for H in horizons:
            target = t + H
            if 0 <= target < len(out):
                out.at[target, f"Pred_Close_{col_agent}_h{H:02d}"] = float(pred_endpoint_close_by_h[H][k])

    close_prev = pd.Series(close).shift(1).to_numpy(dtype=np.float32)
    for H in horizons:
        pc = out[f"Pred_Close_{col_agent}_h{H:02d}"].to_numpy(dtype=float)
        out[f"Delta_Close_pred_{col_agent}_h{H:02d}"] = pc - close_prev

    # Legacy aliases (h=1)
    h1 = 1 if 1 in horizons else horizons[0]
    out[f"Pred_Close_{col_agent}"] = out[f"Pred_Close_{col_agent}_h{h1:02d}"]
    out[f"Delta_Close_pred_{col_agent}"] = out[f"Delta_Close_pred_{col_agent}_h{h1:02d}"]

    out_dates = pd.to_datetime(out["Date"])
    mask = (out_dates >= train_start) & (out_dates <= test_end)
    out_win = out.loc[mask].copy()

    # ---- Daily path outputs per horizon (from StepA daily windows) ----
    manifest_rows: List[Dict[str, Any]] = []

    if use_stepa_daily:
        stepa_items = _scan_stepa_daily_dir(out_root, mode, sym, stepa_daily_dir_override)

        # selection
        if mode == "live":
            stepa_items = stepa_items[-1:] if stepa_items else []
        else:
            stepa_items = [(ds, fp) for (ds, fp) in stepa_items if (ds >= test_start) and (ds <= test_end)]

        # For Date_target mapping beyond known dates (business days)
        dates_ext: List[pd.Timestamp] = list(dates)
        last_known = dates_ext[-1].normalize()

        # Cache: load each horizon model once and reuse it for all anchor days
        import torch
        model_cache: Dict[int, torch.nn.Module] = {}

        def _load_horizon_model(H: int, input_dim: int):
            mdl = model_cache.get(H)
            if mdl is not None:
                return mdl
            # IMPORTANT: the cached model filename depends on the variant tag.
            #   full:     models/mamba_hXX.pt
            #   periodic: models/mamba_periodic_hXX.pt

            pack_path = stepb_dir / "models" / f"{out_tag}_h{H:02d}.pt"
            try:
                pack = torch.load(pack_path, map_location=device, weights_only=True)
            except TypeError:
                # older torch without weights_only
                pack = torch.load(pack_path, map_location=device)
            state = pack["state_dict"] if isinstance(pack, dict) and ("state_dict" in pack) else pack
            mdl = _build_model(input_dim=input_dim, hidden_dim=hidden_dim, out_dim=H, num_layers=int(_get(cfg, "num_layers", 2))).to(device)
            mdl.load_state_dict(state)
            mdl.eval()
            model_cache[H] = mdl
            return mdl

        for ds, fp in stepa_items:
            ds_n = pd.Timestamp(ds).normalize()
            wdf = pd.read_csv(fp)
            if "Date" in wdf.columns:
                wdf["Date"] = pd.to_datetime(wdf["Date"], errors="coerce")
                wdf = wdf.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

            for c in feature_cols:
                if c not in wdf.columns:
                    wdf[c] = 0.0
            wdf[feature_cols] = wdf[feature_cols].apply(pd.to_numeric, errors="coerce").ffill().fillna(0.0)

            Xw = wdf[feature_cols].to_numpy(dtype=np.float32)
            if len(Xw) >= lookback_days:
                Xw = Xw[-lookback_days:, :]
            else:
                pad = np.zeros((lookback_days - len(Xw), Xw.shape[1]), dtype=np.float32)
                Xw = np.vstack([pad, Xw])

            if standardize and (mu is not None) and (sd is not None):
                Xw = (Xw - mu.astype(np.float32)) / sd.astype(np.float32)


            if wavelet_enabled:
                # Close window aligned to Xw (for wavelet augmentation)
                if wavelet_source_col in wdf.columns:
                    cw = pd.to_numeric(wdf[wavelet_source_col], errors="coerce").ffill().fillna(0.0).to_numpy(dtype=np.float32)
                elif "Date" in wdf.columns:
                    tmp = []
                    for _d in wdf["Date"].tolist():
                        dn = pd.Timestamp(_d).normalize()
                        tmp.append(float(close_of_date.get(dn, np.nan)))
                    cw = pd.Series(tmp).ffill().fillna(0.0).to_numpy(dtype=np.float32)
                else:
                    cw = np.zeros((len(Xw),), dtype=np.float32)

                if len(cw) >= lookback_days:
                    cw = cw[-lookback_days:]
                else:
                    if len(cw) == 0:
                        cw = np.zeros((lookback_days,), dtype=np.float32)
                    else:
                        pad = np.full((lookback_days - len(cw),), float(cw[0]), dtype=np.float32)
                        cw = np.concatenate([pad, cw], axis=0)

                details = _wavelet_detail_matrix(_zscore_1d(cw), wavelet_type=wavelet_type, levels_eff=wavelet_levels_eff, mode=wavelet_mode)
                Xw = np.concatenate([Xw, details.astype(np.float32, copy=False)], axis=1)

            anchor_close = float(close_of_date.get(ds_n, np.nan))
            t_idx = idx_of_date.get(ds_n, None)

            need_last = (t_idx + max_h) if t_idx is not None else (len(dates_ext) - 1 + max_h)
            if need_last >= len(dates_ext):
                extra = need_last - (len(dates_ext) - 1)
                start = last_known + pd.tseries.offsets.BDay(1)
                ext = pd.bdate_range(start=start, periods=extra).to_list()
                dates_ext.extend([pd.Timestamp(x) for x in ext])
                last_known = dates_ext[-1].normalize()

            row: Dict[str, Any] = {"Date": ds_n.strftime("%Y-%m-%d"), "agent": out_tag, "stepA_features_path": str(fp).replace("\\", "/")}
            # per horizon daily file
            for H in horizons:
                model = _load_horizon_model(H, input_dim=Xw.shape[1])

                xb = torch.from_numpy(Xw[None, :, :]).to(device)
                with torch.no_grad():
                    yhat = model(xb).detach().cpu().numpy().reshape(-1)  # (H,)

                rows = []
                for step in range(1, H + 1):
                    if t_idx is not None and (t_idx + step) < len(dates_ext):
                        tgt = dates_ext[t_idx + step]
                    else:
                        tgt = (ds_n + pd.tseries.offsets.BDay(step))
                    pred_y = float(yhat[step - 1])
                    pred = _target_inverse(pred_y, anchor_close, target_mode)
                    pret = (pred / anchor_close) - 1.0 if (anchor_close == anchor_close and anchor_close != 0.0) else float("nan")
                    rows.append({
                        "Date_anchor": ds_n.strftime("%Y-%m-%d"),
                        "Close_anchor": anchor_close,
                        "mode": mode,
                        "symbol": sym,
                        "stepA_features_path": str(fp).replace("\\", "/"),
                        "step_ahead_bdays": int(step),
                        "Date_target": pd.Timestamp(tgt).strftime("%Y-%m-%d"),
                        "Pred_y_from_anchor": pred_y,
                        "target_mode": target_mode,
                        "Pred_Close": pred,
                        "Pred_ret_from_anchor": pret,
                        "horizon_model": int(H),
                    })

                snap_name = f"stepB_daily_pred_{out_tag}_h{H:02d}_{sym}_{ds_n.strftime('%Y_%m_%d')}.csv"
                snap_path = daily_dir / snap_name
                pd.DataFrame(rows).to_csv(snap_path, index=False, encoding="utf-8-sig")

                pred_rel = str(resolve_repo_path("output") / "stepB" / mode / daily_dirname / snap_name).replace("\\", "/")
                row[f"pred_path_h{H:02d}"] = pred_rel

            manifest_rows.append(row)

        if manifest_rows:
            df_mani = pd.DataFrame(manifest_rows).sort_values("Date").reset_index(drop=True)
            mani_path = stepb_dir / (f"stepB_daily_manifest_{manifest_suffix + '_' if manifest_suffix else ''}{sym}.csv")
            df_mani.to_csv(mani_path, index=False, encoding="utf-8")
            print(f"[StepB:mamba] wrote daily manifest -> {mani_path} rows={len(df_mani)}")

    pred_future_periodic_path: Optional[Path] = None
    if is_periodic and mode == "live" and int(_get(cfg, "live_future_bdays", 0)) > 0:
        future_bdays = int(_get(cfg, "live_future_bdays", 63))
        model_h1_path = stepb_dir / "models" / f"{out_tag}_h01.pt"
        try:
            pack_h1 = torch.load(model_h1_path, map_location=device, weights_only=True)
        except TypeError:
            pack_h1 = torch.load(model_h1_path, map_location=device)
        state_h1 = pack_h1["state_dict"] if isinstance(pack_h1, dict) and ("state_dict" in pack_h1) else pack_h1
        h1_model = _build_model(input_dim=final_dim, hidden_dim=hidden_dim, out_dim=1, num_layers=int(_get(cfg, "num_layers", 2))).to(device)
        h1_model.load_state_dict(state_h1)
        h1_model.eval()

        fdf = df_f.copy()
        fdf["Date"] = pd.to_datetime(fdf["Date"], errors="coerce")
        fdf = fdf.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
        missing_cols = [c for c in feature_cols if c not in fdf.columns]
        if missing_cols:
            raise RuntimeError(f"periodic future requires missing feature columns: {missing_cols[:5]}")
        fdf[feature_cols] = fdf[feature_cols].apply(pd.to_numeric, errors="coerce").ffill().bfill()

        last_known_price_date = max(close_of_date.keys())
        future_dates = [pd.Timestamp(d).normalize() for d in fdf["Date"].tolist() if pd.Timestamp(d).normalize() > last_known_price_date]
        future_dates = future_dates[:future_bdays]
        if len(future_dates) < future_bdays:
            raise RuntimeError(f"Insufficient periodic future dates: {len(future_dates)} < {future_bdays}")

        anchor_date = pd.Timestamp(last_known_price_date).normalize()
        anchor_close = float(close_of_date[last_known_price_date])
        rows_future: List[Dict[str, Any]] = []

        for tgt_date in future_dates:
            hist = fdf[fdf["Date"] <= anchor_date].tail(lookback_days)
            if len(hist) < lookback_days:
                raise RuntimeError(f"Insufficient periodic window at {anchor_date.date()}: {len(hist)} < {lookback_days}")
            Xw = hist[feature_cols].to_numpy(dtype=np.float32)
            if standardize and mu is not None and sd is not None:
                Xw = (Xw - mu.astype(np.float32)) / sd.astype(np.float32)
            if wavelet_enabled:
                close_seq = np.full((lookback_days,), float(anchor_close), dtype=np.float32)
                Xw = _append_wavelet_to_seq(Xw, close_seq)

            xb = torch.from_numpy(Xw[None, :, :]).to(device)
            with torch.no_grad():
                yhat = h1_model(xb).detach().cpu().numpy().reshape(-1)

            pred_y = float(yhat[0])
            pred_close = float(_target_inverse(pred_y, anchor_close, target_mode))
            rows_future.append({
                "Date_anchor": anchor_date.strftime("%Y-%m-%d"),
                "Date_target": pd.Timestamp(tgt_date).strftime("%Y-%m-%d"),
                "Pred_y_from_anchor": pred_y,
                "target_mode": target_mode,
                "Close_anchor": anchor_close,
                "Pred_Close": pred_close,
            })
            anchor_date = pd.Timestamp(tgt_date).normalize()
            anchor_close = pred_close

        pred_future_periodic_path = stepb_dir / f"stepB_pred_future_periodic_{sym}.csv"
        pd.DataFrame(rows_future).to_csv(pred_future_periodic_path, index=False, encoding="utf-8-sig")

    # ---- Write outputs ----
    pred_close_path = stepb_dir / f"stepB_pred_close_{out_tag}_{sym}.csv"
    pred_path_path = stepb_dir / f"stepB_pred_path_{out_tag}_{sym}.csv"
    delta_path = stepb_dir / f"stepB_delta_{out_tag}_{sym}.csv"
    meta_path = stepb_dir / f"stepB_{out_tag}_meta_{sym}.json"

    out_win.to_csv(pred_close_path, index=False, encoding="utf-8-sig")
    df_path.to_csv(pred_path_path, index=False, encoding="utf-8-sig")
    out_win[["Date", f"Delta_Close_pred_{col_agent}"]].to_csv(delta_path, index=False, encoding="utf-8-sig")

    meta = {
        "symbol": sym,
        "variant": ("periodic" if is_periodic else "full"),
        "out_tag": out_tag,
        "col_agent": col_agent,
        "feature_cols_used": feature_cols,
        "mode": mode,
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "lookback_days": lookback_days,
        "horizons": horizons,
        "max_h": max_h,
        "feature_dim": int(final_dim),
        "feature_dim_base": int(len(feature_cols)),
        "feature_dim_final": int(final_dim),
        "wavelet": {
            "enabled": bool(wavelet_enabled),
            "type": str(wavelet_type),
            "levels_req": int(wavelet_levels_req),
            "levels_eff": int(wavelet_levels_eff),
            "mode": str(wavelet_mode),
            "source_col": str(wavelet_source_col),
        },
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "hidden_dim": hidden_dim,
        "model_arch": "Mamba",
        "model_name": "Mamba",
        "backend": str(_get(cfg, "backend", "mamba")),
        "standardize": standardize,
        "train_samples_by_h": train_samples_by_h,
        "loss_by_h": {f"h{h:02d}": loss_by_h[h] for h in horizons},
        "daily_manifest_rows": int(len(manifest_rows)),
        "csv": {
            "pred_close": str(pred_close_path),
            "pred_path": str(pred_path_path),
            "delta": str(delta_path),
            "meta": str(meta_path),
            "pred_future_periodic": (str(pred_future_periodic_path) if pred_future_periodic_path else ""),
        },
    }
    meta_path.write_text(json.dumps(_jsonify(meta), indent=2, ensure_ascii=False), encoding="utf-8")

    return StepBAgentResult(
        success=True,
        message=f"mamba ok (variant={'periodic' if is_periodic else 'full'}; horizons={horizons})",
        agent="mamba",
        out_dir=str(stepb_dir),
        csv_paths={
            "pred_close": str(pred_close_path),
            "pred_path": str(pred_path_path),
            "delta": str(delta_path),
            "meta": str(meta_path),
            "pred_future_periodic": (str(pred_future_periodic_path) if pred_future_periodic_path else ""),
        },
        metrics={
            "daily_manifest_rows": int(len(manifest_rows)),
            "train_samples_by_h": train_samples_by_h,
        },
        csv_paths_list=[str(pred_close_path), str(pred_path_path), str(delta_path), str(meta_path)],
    )


# ---------------------------------------------------------------------
# Backward-compatible entrypoint expected by StepBService
# ---------------------------------------------------------------------
def run_stepB_mamba(app_config, symbol, prices_df, features_df, cfg):
    """Entry point StepBService expects.

    StepBService imports:
        from ai_core.services.step_b_mamba_runner import run_stepB_mamba
    """
    return run_mamba_multi_model_by_horizon(
        app_config=app_config,
        symbol=symbol,
        prices_df=prices_df,
        features_df=features_df,
        cfg=cfg,
    )
