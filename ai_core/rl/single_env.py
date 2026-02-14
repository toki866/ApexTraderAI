# ai_core/rl/single_env.py
"""
互換 shim（正本への参照を1本化するための薄いラッパ）

過去の実装で
- ai_core/rl/single_env.py
- ai_core/rl/env_single.py
の両方に RLSingleEnv が存在し、同名重複（かつ中身が違う）で事故を起こしていました。

今後は ai_core/rl/env_single.py の RLSingleEnv を「正本」とし、
本ファイルは **import の受け口**だけを提供します。

例:
    # 旧コード（そのまま動く）
    from ai_core.rl.single_env import RLSingleEnv

    # 新コード（推奨）
    from ai_core.rl.env_single import RLSingleEnv
"""

from ai_core.rl.env_single import RLSingleEnv

__all__ = ["RLSingleEnv"]
