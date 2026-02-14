from __future__ import annotations

"""ai_core.services._path_helpers (COMPAT SHIM)

目的
- StepB 系の output_root 解決ヘルパが複数ファイルに重複定義され、
  build_global_symbols の Duplicate Symbols に出ていました。
- 正本は ai_core.services.step_b_path_utils に統一し、このファイルは互換 import のみを残します。

注意
- このファイルに関数定義を置かないことで「同名定義の重複」にならないようにしています。
"""

from ai_core.services.step_b_path_utils import _get_output_root_from_app

__all__ = [
    "_get_output_root_from_app",
]
