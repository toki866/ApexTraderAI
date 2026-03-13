from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
from ai_core.types.common import DateRange
from ai_core.config.rl_config import EnvConfig, RLSingleConfig, RLMARLConfig
from ai_core.utils.paths import resolve_repo_path

@dataclass
class DataConfig:
    """Data-related configuration including directories and symbols."""
    data_root: Path
    output_root: Path
    symbols: List[str]

    @property
    def data_dir(self) -> Path:
        """Alias for data_root used by CLI options such as --data-dir."""
        return self.data_root

    @data_dir.setter
    def data_dir(self, value: str | Path) -> None:
        self.data_root = resolve_repo_path(Path(value).expanduser())


@dataclass
class ClusterRegimeConfig:
    """Cluster regime settings scaffold (may be partially not yet wired)."""
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

@dataclass
class AppConfig:
    """Root class for application-wide settings."""
    data: DataConfig
    env: EnvConfig = field(default_factory=EnvConfig)
    rl_single: RLSingleConfig = field(default_factory=RLSingleConfig)
    rl_marl: RLMARLConfig = field(default_factory=RLMARLConfig)
    cluster_regime: ClusterRegimeConfig = field(default_factory=ClusterRegimeConfig)
    default_symbol: Optional[str] = None
    default_date_range: Optional[DateRange] = None
    stepF: Optional[Any] = None

    @classmethod
    def load_from_yaml(cls, path: str | Path) -> "AppConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        data_raw = raw.get("data", {}) or {}
        data_cfg = DataConfig(
            data_root=resolve_repo_path(Path(data_raw.get("data_root", "data")).expanduser()),
            output_root=resolve_repo_path(Path(data_raw.get("output_root", "output")).expanduser()),
            symbols=list(data_raw.get("symbols", [])),
        )
        env_raw = raw.get("env", raw.get("env_config", {})) or {}
        env_cfg = EnvConfig.from_dict(env_raw) if isinstance(env_raw, dict) else EnvConfig()
        rl_single_raw = raw.get("rl_single", {}) or {}
        rl_marl_raw = raw.get("rl_marl", {}) or {}
        rl_single_cfg = RLSingleConfig.from_dict(rl_single_raw) if isinstance(rl_single_raw, dict) else RLSingleConfig()
        rl_marl_cfg = RLMARLConfig.from_dict(rl_marl_raw) if isinstance(rl_marl_raw, dict) else RLMARLConfig()
        cluster_raw = raw.get("cluster_regime", {}) or {}
        cluster_cfg = ClusterRegimeConfig(
            enable_cluster_regime=bool(cluster_raw.get("enable_cluster_regime", True)),
            enable_cluster_monthly_refit=bool(cluster_raw.get("enable_cluster_monthly_refit", True)),
            enable_cluster_daily_assign=bool(cluster_raw.get("enable_cluster_daily_assign", True)),
            enable_cluster_in_rl_state=bool(cluster_raw.get("enable_cluster_in_rl_state", True)),
            cluster_backend=str(cluster_raw.get("cluster_backend", "ticc")),
            cluster_raw_k=int(cluster_raw.get("cluster_raw_k", 20)),
            cluster_k_eff_min=int(cluster_raw.get("cluster_k_eff_min", 12)),
            cluster_small_share_threshold=float(cluster_raw.get("cluster_small_share_threshold", 0.01)),
            cluster_small_mean_run_threshold=float(cluster_raw.get("cluster_small_mean_run_threshold", 3.0)),
            cluster_short_window_days=int(cluster_raw.get("cluster_short_window_days", 20)),
            cluster_mid_window_weeks=int(cluster_raw.get("cluster_mid_window_weeks", 8)),
            cluster_long_window_months=int(cluster_raw.get("cluster_long_window_months", 6)),
            cluster_enable_8y_context=bool(cluster_raw.get("cluster_enable_8y_context", True)),
            cluster_rare_flag_enabled=bool(cluster_raw.get("cluster_rare_flag_enabled", True)),
        )
        default_symbol = raw.get("default_symbol")
        default_date_range = None
        stepf_cfg = None
        stepf_raw = raw.get("stepF", {}) or {}
        if isinstance(stepf_raw, dict):
            try:
                from ai_core.services.step_f_service import StepFRouterConfig

                stepf_cfg = StepFRouterConfig(**stepf_raw)
            except Exception:
                stepf_cfg = stepf_raw
        if isinstance(raw.get("default_date_range", None), dict):
            try:
                default_date_range = DateRange.from_dict(raw["default_date_range"])
            except Exception:
                default_date_range = None
        return cls(
            data=data_cfg,
            env=env_cfg,
            rl_single=rl_single_cfg,
            rl_marl=rl_marl_cfg,
            cluster_regime=cluster_cfg,
            default_symbol=default_symbol,
            default_date_range=default_date_range,
            stepF=stepf_cfg,
        )
