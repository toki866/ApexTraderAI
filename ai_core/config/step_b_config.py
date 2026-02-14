# ai_core/config/step_b_config.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

from ai_core.types.common import DateRange


@dataclass
class XSRTrainConfig:
    """XSR training configuration."""

    # Ridge / lag estimation
    l2_reg: float = 1e-3
    max_lag_days: int = 5
    max_freq: Optional[float] = None

    # FFT / reconstruction settings (required by step_b_xsr_runner)
    fft_bins: int = 256
    use_price_scale: bool = True
    random_state: int = 42

    # Whether to use frequency-domain phase lag estimation
    use_phase_lag: bool = True

    # StepBService compatibility
    enabled: bool = True
    date_range: Optional[DateRange] = None


@dataclass
class WaveletMambaTrainConfig:
    """Wavelet-Mamba training configuration."""

    # Model structure
    seq_len: int = 256
    horizon: int = 1
    hidden_dim: int = 96
    num_layers: int = 4
    dropout: float = 0.1
    activation: str = "gelu"
    use_layer_norm: bool = True

    # Training
    batch_size: int = 64
    num_epochs: int = 60
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    loss_type: str = "mse"

    # Wavelet frontend
    wavelet_type: str = "db4"
    wavelet_levels: int = 3

    # Runtime
    device: str = "auto"  # "cpu" | "cuda" | "auto"
    preset: Optional[str] = None

    # --- Runner-compat fields (direct multi-horizon runner) ---
    # run mode:
    #   - "sim": strict leakage control (train_end = prev business day before test_start)
    #   - "live"/"ops"/"prod": train_end = test_end
    mode: str = "sim"
    # Optional alias used by some CLI scripts
    mamba_mode: Optional[str] = None

    # Data split (string dates "YYYY-MM-DD"; runner converts to pandas.Timestamp)
    train_start: Optional[str] = None
    train_end: Optional[str] = None
    test_start: Optional[str] = None
    test_end: Optional[str] = None

    # Direct runner hyperparams
    lookback_days: int = 60
    horizons: str = "1,5,10,20"
    seed: int = 42
    # Aliases for legacy fields (filled in __post_init__ if None)
    epochs: Optional[int] = None
    lr: Optional[float] = None
    standardize: bool = True

    # --- StepA daily window integration ---
    # If True, StepB(Mamba) will read StepA daily window CSVs from:
    #   <output_root>/stepA/<mode>/daily/stepA_daily_features_<SYMBOL>_YYYY_MM_DD.csv
    # and generate per-day forecast path CSVs.
    use_stepa_daily_windows: bool = True
    # Optional override: directory path that contains StepA daily window CSVs.
    stepa_daily_dir: Optional[str] = None
    # Daily snapshot file format: 'path' writes 1..max_h rows (recommended).
    daily_snapshot_format: str = 'path'
    # StepBService compatibility
    enabled: bool = True
    # Variant switch
    #   'full'     : periodic + tech + OHLCV (endpoint forecasting; typically use horizons=(1,))
    #   'periodic' : periodic-only (sin/cos) and snapshot outputs (future-chart) for RL compression
    variant: str = 'full'


    # If True, also run periodic-only snapshot generation (disabled by default; full-only is default).
    enable_periodic_snapshots: bool = False
    # Periodic snapshot settings (used when variant == 'periodic')
    # snapshot horizons mean: for each H, predict a path of length H business-days.
    periodic_snapshot_horizons: Tuple[int, ...] = (20,)
    periodic_daily_dirname: str = 'daily_periodic'
    periodic_manifest_suffix: str = 'periodic'
    periodic_output_tag: str = 'mamba_periodic'
    periodic_endpoints: Tuple[int, ...] = (1, 5, 10, 20)
    date_range: Optional[DateRange] = None


@dataclass
class FEDformerTrainConfig:
    """FEDformer training configuration."""

    # Model structure
    seq_len: int = 256
    d_model: int = 128
    n_heads: int = 4
    e_layers: int = 3
    d_ff: int = 256
    dropout: float = 0.1

    # Frequency block
    freq_top_k: int = 16
    decompose_kernel: int = 25

    # Training
    batch_size: int = 64
    num_epochs: int = 60
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    lr_scheduler: str = "step"  # "none" | "step" | "cosine"

    # Runtime
    device: str = "auto"  # "cpu" | "cuda" | "auto"

    # StepBService compatibility
    enabled: bool = True
    date_range: Optional[DateRange] = None





def _dr_get(dr: Any, key: str, default: Any = None) -> Any:
    """Best-effort getter for DateRange-like objects or mappings."""
    if dr is None:
        return default
    if isinstance(dr, Mapping):
        return dr.get(key, default)
    return getattr(dr, key, default)


@dataclass
class StepBConfig:
    """StepB configuration.

    This dataclass keeps legacy boolean flags (train_xsr, train_wavelet_mamba, train_fedformer)
    while also providing StepBService-friendly attributes (xsr, mamba, fedformer) and helpers.
    """

    symbol: str
    date_range: Any

    # Whether StepBService should try to ensure contract artifacts such as stepB_delta_* files.
    ensure_contract_artifacts: bool = True

    # Run mode ("sim" or "ops"). StepBService resolves mode from:
    #   cfg.mamba.mode -> cfg.run_mode -> defaults to "sim"
    run_mode: str = "sim"

    # Legacy flags
    train_xsr: bool = True
    train_wavelet_mamba: bool = True
    train_fedformer: bool = True

    # Legacy config holders
    xsr_config: XSRTrainConfig = field(default_factory=XSRTrainConfig)
    wavelet_mamba_config: WaveletMambaTrainConfig = field(default_factory=WaveletMambaTrainConfig)
    fedformer_config: FEDformerTrainConfig = field(default_factory=FEDformerTrainConfig)

    # StepBService fields (populated in __post_init__)
    xsr: XSRTrainConfig = field(init=False)
    mamba: WaveletMambaTrainConfig = field(init=False)
    fedformer: FEDformerTrainConfig = field(init=False)

    def __post_init__(self) -> None:
        # Reuse the existing config objects
        self.xsr = self.xsr_config
        self.mamba = self.wavelet_mamba_config
        self.fedformer = self.fedformer_config

        # Reflect legacy enable toggles
        self.xsr.enabled = bool(self.train_xsr)
        self.mamba.enabled = bool(self.train_wavelet_mamba)
        self.fedformer.enabled = bool(self.train_fedformer)

        # Propagate date_range (best-effort)
        # NOTE: StepB(Mamba) leakage control is enforced via mamba.test_start/test_end.
        #       date_range is kept flexible because DateRange signature may evolve.
        self.xsr.date_range = self.date_range if isinstance(self.date_range, DateRange) else None
        self.mamba.date_range = self.date_range if isinstance(self.date_range, DateRange) else None
        self.fedformer.date_range = self.date_range if isinstance(self.date_range, DateRange) else None

        # --- Mode normalization / aliases ---
        # If CLI sets mamba_mode, prefer it.
        if isinstance(self.mamba.mamba_mode, str) and self.mamba.mamba_mode.strip():
            self.mamba.mode = self.mamba.mamba_mode.strip()

        # If mamba.mode is default and run_mode is provided, propagate.
        if isinstance(self.run_mode, str) and self.run_mode.strip():
            if str(self.mamba.mode).strip().lower() in ("", "sim") and self.run_mode.strip().lower() in ("sim", "ops"):
                self.mamba.mode = self.run_mode.strip().lower()

        # --- Runner hyperparam aliases ---
        # epochs <-> num_epochs
        if getattr(self.mamba, "epochs", None) is None:
            self.mamba.epochs = int(self.mamba.num_epochs)
        else:
            try:
                self.mamba.num_epochs = int(self.mamba.epochs)
            except Exception:
                pass

        # lr <-> learning_rate
        if getattr(self.mamba, "lr", None) is None:
            self.mamba.lr = float(self.mamba.learning_rate)
        else:
            try:
                self.mamba.learning_rate = float(self.mamba.lr)
            except Exception:
                pass

        # lookback_days fallback from seq_len if user hasn't set it explicitly
        try:
            if int(getattr(self.mamba, "lookback_days", 0) or 0) <= 0:
                self.mamba.lookback_days = int(self.mamba.seq_len)
        except Exception:
            pass

        # --- Leakage-control dates ---
        # Prefer explicit cfg.mamba.test_start/test_end. If absent, try to fill from date_range.
        if not getattr(self.mamba, "test_start", None):
            ts = _dr_get(self.date_range, "test_start", _dr_get(self.date_range, "start", None))
            if ts is not None:
                self.mamba.test_start = str(ts)
        if not getattr(self.mamba, "test_end", None):
            te = _dr_get(self.date_range, "test_end", _dr_get(self.date_range, "end", None))
            if te is not None:
                self.mamba.test_end = str(te)

        if not getattr(self.mamba, "train_start", None):
            tr_s = _dr_get(self.date_range, "train_start", None)
            if tr_s is not None:
                self.mamba.train_start = str(tr_s)
        if not getattr(self.mamba, "train_end", None):
            tr_e = _dr_get(self.date_range, "train_end", None)
            if tr_e is not None:
                self.mamba.train_end = str(tr_e)

    def enabled_agents(self) -> list[str]:
        """Return enabled agent keys in stable order."""
        out: list[str] = []
        if getattr(self.xsr, "enabled", False):
            out.append("xsr")
        if getattr(self.mamba, "enabled", False):
            out.append("mamba")
        if getattr(self.fedformer, "enabled", False):
            out.append("fedformer")
        return out

    @classmethod
    def from_any(cls, value: Any) -> "StepBConfig":
        """Best-effort converter used by StepBService.

        Supported:
          - StepBConfig (returned as-is)
          - Mapping (dict-like) with at least symbol and date_range
          - Any object whose vars() contains symbol and date_range
        """
        if isinstance(value, cls):
            return value
        if value is None:
            raise TypeError("StepBConfig.from_any: value is None")

        # Mapping (dict-like)
        if isinstance(value, Mapping):
            d = dict(value)
            # Normalize date_range if provided as a mapping
            if "date_range" in d and isinstance(d["date_range"], Mapping):
                try:
                    d["date_range"] = DateRange(**dict(d["date_range"]))
                except TypeError:
                    # Keep flexible: allow dict date_range if DateRange signature differs
                    d["date_range"] = dict(d["date_range"])

            # Filter only init fields (ignore unknown keys)
            init_keys = {k for k, f in cls.__dataclass_fields__.items() if f.init}  # type: ignore[attr-defined]
            filtered = {k: v for k, v in d.items() if k in init_keys}
            try:
                return cls(**filtered)
            except TypeError as e:
                raise TypeError(f"StepBConfig.from_any: cannot build StepBConfig from mapping: {e}")

        # Generic object: use vars() if possible
        try:
            obj_dict = vars(value)
        except Exception as e:
            raise TypeError(f"StepBConfig.from_any: unsupported type {type(value)}: {e}")

        return cls.from_any(obj_dict)