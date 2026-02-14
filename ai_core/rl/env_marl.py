from __future__ import annotations

"""
ai_core.rl.env_marl
===================

⚠ Legacy compatibility module.

以前はここに「実験用の MultiAgentTradingEnv（ダミー環境）」が定義されていましたが、
現在は本番用の実装が ai_core.rl.envs.MultiAgentTradingEnv に集約されています。

新しいコードは必ずこちらを import してください：

    from ai_core.rl.envs import MultiAgentTradingEnv

このモジュールは、既存コードで

    from ai_core.rl.env_marl import MultiAgentTradingEnv

という import を使っている場合のための**互換ラッパー**です。
中身の実体は ai_core.rl.envs.MultiAgentTradingEnv です。

将来的には、このファイルごと削除しても良いですが、
当面は「互換用ブリッジ」として残しておきます。
"""

from dataclasses import dataclass
from typing import Any, Dict

# 本番用 MultiAgentTradingEnv を再エクスポート
from ai_core.rl.envs import MultiAgentTradingEnv  # noqa: F401


@dataclass
class MARLEpisodeStats:
    """
    （オプション）MARL の 1 エピソード分の統計情報。

    旧 env_marl.py で使っていた構造体との互換用に残しています。
    現行実装では必須ではありませんが、参照しているコードがあっても壊れないように、
    シンプルな dataclass として定義だけ残してあります。
    """

    total_reward: float
    length: int
    info: Dict[str, Any]


__all__ = [
    "MultiAgentTradingEnv",
    "MARLEpisodeStats",
]
