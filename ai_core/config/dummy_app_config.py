# ai_core/config/dummy_app_config.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Sequence

from ai_core.config.rl_config import EnvConfig, RLSingleConfig, RLMARLConfig


# StepE / StepF で使いたいエージェント名のデフォルト
DEFAULT_AGENT_NAMES: tuple[str, ...] = ("xsr", "lstm", "fed")


@dataclass
class DummyPaths:
    """
    AppConfig.paths の簡易版。
    StepE / StepF からは少なくとも:
      - paths.output_root
      - paths.data_root
    が参照される想定。
    """
    output_root: Path = Path("output")
    data_root: Path = Path("data")


@dataclass
class DummyRL:
    """
    AppConfig.rl の簡易版。

    StepE / StepF から参照されるフィールドを最小限だけ用意する。

    想定されるフィールド:
      - env_config / env
      - single_configs: Dict[str, RLSingleConfig]
      - single / single_config（互換用）
      - marl_config / marl / marl_configs
    """
    env_config: EnvConfig = field(default_factory=EnvConfig)
    marl_config: RLMARLConfig = field(default_factory=RLMARLConfig)
    single_configs: Dict[str, RLSingleConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # 互換用エイリアス
        self.env = self.env_config

        # MARL 用
        self.marl = self.marl_config
        self.marl_configs = {"default": self.marl_config}

        # 単体RLの設定が無ければデフォルトで作成
        if not self.single_configs:
            self.single_configs = {
                name: RLSingleConfig()
                for name in DEFAULT_AGENT_NAMES
            }

        # 代表の single / single_config
        first_cfg = next(iter(self.single_configs.values()))
        self.single = first_cfg
        self.single_config = first_cfg


@dataclass
class DummyAppConfig:
    """
    元の AppConfig のごく一部だけを再現したダミー版。
    """
    paths: DummyPaths = field(default_factory=DummyPaths)
    rl: DummyRL = field(default_factory=DummyRL)


def create_dummy_app_config(
    agent_names: Sequence[str] = DEFAULT_AGENT_NAMES,
    output_root: str | Path = "output",
    data_root: str | Path = "data",
) -> DummyAppConfig:
    """
    REPL やテストから一発で DummyAppConfig を作るためのヘルパ。
    """
    paths = DummyPaths(output_root=Path(output_root), data_root=Path(data_root))

    single_configs: Dict[str, RLSingleConfig] = {
        name: RLSingleConfig() for name in agent_names
    }

    rl = DummyRL(
        env_config=EnvConfig(),
        marl_config=RLMARLConfig(),
        single_configs=single_configs,
    )

    return DummyAppConfig(paths=paths, rl=rl)

# ---------------------------------------------------------------------------
# Compatibility for headless runner / autodebug
# ---------------------------------------------------------------------------
# The headless runner shipped with auto_dev_suite expects:
#   from ai_core.config.dummy_app_config import make_dummy_app_config
# If it does not exist, the runner treats dummy_app_config as "unavailable".
#
# This function attempts to create the real AppConfig (if available) using
# best-effort heuristics, and falls back to DummyAppConfig for minimal runs.
#
# NOTE: Returning DummyAppConfig is acceptable as a fallback as long as the
# downstream Step services only access the fields we provide (paths/rl).
# If a service later needs more fields, the autodebug loop will surface it
# and we can extend DummyAppConfig accordingly.
from typing import Any
import inspect


def make_dummy_app_config(repo_root: str | Path) -> Any:
    """
    Create an AppConfig-like object for headless / autodebug runs.

    Priority:
      1) Try to instantiate the real AppConfig (ai_core.config.app_config.AppConfig)
      2) Fallback to this module's DummyAppConfig

    Parameters
    ----------
    repo_root : str | Path
        Path to soxl_rl_gui repository root.

    Returns
    -------
    Any
        AppConfig instance if possible, otherwise DummyAppConfig.
    """
    rr = Path(repo_root).resolve()

    # best-effort: create common directories used by services
    for p in (rr / "output", rr / "data", rr / "config", rr / "artifacts", rr / "logs"):
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    # 1) Try real AppConfig
    try:
        from ai_core.config.app_config import AppConfig  # type: ignore

        # a) parameterless
        try:
            return AppConfig()
        except Exception:
            pass

        # b) signature-based kwargs for common fields
        try:
            sig = inspect.signature(AppConfig)
            kwargs: dict[str, Any] = {}
            for name, param in sig.parameters.items():
                if name == "self":
                    continue
                lname = name.lower()
                if "repo" in lname and "root" in lname:
                    kwargs[name] = rr
                elif "output" in lname and "root" in lname:
                    kwargs[name] = rr / "output"
                elif "data" in lname and ("root" in lname or "dir" in lname):
                    kwargs[name] = rr / "data"
                elif "config" in lname and ("root" in lname or "dir" in lname or "path" in lname):
                    kwargs[name] = rr / "config"
                elif "artifact" in lname and ("root" in lname or "dir" in lname or "path" in lname):
                    kwargs[name] = rr / "artifacts"
                elif "log" in lname and ("root" in lname or "dir" in lname or "path" in lname):
                    kwargs[name] = rr / "logs"

            if kwargs:
                return AppConfig(**kwargs)
        except Exception:
            pass

    except Exception:
        # no real AppConfig or import failed
        pass

    # 2) Fallback to DummyAppConfig
    return create_dummy_app_config(output_root=rr / "output", data_root=rr / "data")
