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

@dataclass
class AppConfig:
    """Root class for application-wide settings."""
    data: DataConfig
    env: EnvConfig = field(default_factory=EnvConfig)
    rl_single: RLSingleConfig = field(default_factory=RLSingleConfig)
    rl_marl: RLMARLConfig = field(default_factory=RLMARLConfig)
    default_symbol: Optional[str] = None
    default_date_range: Optional[DateRange] = None

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
        default_symbol = raw.get("default_symbol")
        default_date_range = None
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
            default_symbol=default_symbol,
            default_date_range=default_date_range,
        )
