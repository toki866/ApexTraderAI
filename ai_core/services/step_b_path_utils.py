from __future__ import annotations

"""ai_core.services.step_b_path_utils

StepB 系で使う「output_root（出力先）」取得ユーティリティ。

背景
- step_b_fedformer_runner / step_b_mamba_runner に同名の
  `_get_output_root_from_app()` が重複定義されており、
  build_global_symbols の Duplicate Symbols に出ていました。
- ここに “正本” を 1つだけ定義し、各 runner は import して使います。

対応する AppConfig の揺れ（互換）
- app_config.output_root
- app_config.output_dir
- app_config.paths.output_root / app_config.paths.output_dir
- どれも無ければ "output" を返す
"""

from pathlib import Path

from ai_core.config.app_config import AppConfig


def _get_output_root_from_app(app_config: AppConfig) -> Path:
    """AppConfig から output_root を安全に取得する。"""
    if hasattr(app_config, "output_root"):
        return Path(getattr(app_config, "output_root"))

    if hasattr(app_config, "output_dir"):
        return Path(getattr(app_config, "output_dir"))

    if hasattr(app_config, "paths"):
        paths = getattr(app_config, "paths")
        if hasattr(paths, "output_root"):
            return Path(getattr(paths, "output_root"))
        if hasattr(paths, "output_dir"):
            return Path(getattr(paths, "output_dir"))

    return Path("output")


__all__ = ["_get_output_root_from_app"]
