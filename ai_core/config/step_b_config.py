from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

from ai_core.types.common import DateRange


@dataclass
class WaveletMambaTrainConfig:
    """Wavelet-Mamba training configuration."""

    seq_len: int = 256
    horizon: int = 1
    hidden_dim: int = 96
    num_layers: int = 4
    dropout: float = 0.1
    activation: str = "gelu"
    use_layer_norm: bool = True
    batch_size: int = 64
    num_epochs: int = 60
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    loss_type: str = "mse"
    wavelet_type: str = "db4"
    wavelet_levels: int = 3
    device: str = "auto"
    preset: Optional[str] = None
    mode: str = "sim"
    mamba_mode: Optional[str] = None
    train_start: Optional[str] = None
    train_end: Optional[str] = None
    test_start: Optional[str] = None
    test_end: Optional[str] = None
    lookback_days: int = 128
    horizons: str = "1,5,10,20"
    seed: int = 42
    epochs: Optional[int] = None
    lr: Optional[float] = None
    standardize: bool = True
    use_stepa_daily_windows: bool = True
    stepa_daily_dir: Optional[str] = None
    daily_snapshot_format: str = "path"
    enabled: bool = True
    variant: str = "full"
    enable_periodic_snapshots: bool = True
    periodic_snapshot_horizons: Tuple[int, ...] = (1, 5, 10, 20)
    periodic_daily_dirname: str = "daily_periodic"
    periodic_manifest_suffix: str = "periodic"
    periodic_output_tag: str = "mamba_periodic"
    periodic_endpoints: Tuple[int, ...] = (1, 5, 10, 20)
    date_range: Optional[DateRange] = None


def _dr_get(dr: Any, key: str, default: Any = None) -> Any:
    if dr is None:
        return default
    if isinstance(dr, Mapping):
        return dr.get(key, default)
    return getattr(dr, key, default)


@dataclass
class StepBConfig:
    symbol: str
    date_range: Any
    ensure_contract_artifacts: bool = True
    run_mode: str = "sim"
    train_wavelet_mamba: bool = True
    wavelet_mamba_config: WaveletMambaTrainConfig = field(default_factory=WaveletMambaTrainConfig)
    mamba: WaveletMambaTrainConfig = field(init=False)

    def __post_init__(self) -> None:
        self.mamba = self.wavelet_mamba_config
        self.mamba.enabled = bool(self.train_wavelet_mamba)
        self.mamba.date_range = self.date_range if isinstance(self.date_range, DateRange) else None

        if isinstance(self.mamba.mamba_mode, str) and self.mamba.mamba_mode.strip():
            self.mamba.mode = self.mamba.mamba_mode.strip()
        if isinstance(self.run_mode, str) and self.run_mode.strip():
            if str(self.mamba.mode).strip().lower() in ("", "sim") and self.run_mode.strip().lower() in ("sim", "ops"):
                self.mamba.mode = self.run_mode.strip().lower()

        if getattr(self.mamba, "epochs", None) is None:
            self.mamba.epochs = int(self.mamba.num_epochs)
        if getattr(self.mamba, "lr", None) is None:
            self.mamba.lr = float(self.mamba.learning_rate)

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
        return ["mamba"] if getattr(self.mamba, "enabled", False) else []

    @classmethod
    def from_any(cls, value: Any) -> "StepBConfig":
        if isinstance(value, cls):
            return value
        if value is None:
            raise TypeError("StepBConfig.from_any: value is None")
        if isinstance(value, Mapping):
            d = dict(value)
            if "date_range" in d and isinstance(d["date_range"], Mapping):
                try:
                    d["date_range"] = DateRange(**dict(d["date_range"]))
                except TypeError:
                    d["date_range"] = dict(d["date_range"])
            init_keys = {k for k, f in cls.__dataclass_fields__.items() if f.init}  # type: ignore[attr-defined]
            filtered = {k: v for k, v in d.items() if k in init_keys}
            return cls(**filtered)
        return cls.from_any(vars(value))
