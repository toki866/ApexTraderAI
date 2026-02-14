# ai_core/rl/algo_marl_legacy.py

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

from ai_core.rl.env_marl import MultiAgentTradingEnv


@dataclass
class MARLTrainResult:
    """
    Legacy MARL 学習結果サマリ（レガシー API 保持用）。

    現行実装では使わず、参考用としてのみ残しています。
    """
    ok: bool
    message: str = ""
    metrics: Dict[str, Any] | None = None


class MARLAlgoBase(ABC):
    """
    Legacy MARL アルゴリズム抽象クラス。

    新しい実装では ai_core.rl.algos.MARLAlgo（PPO 実装）を使用してください。
    """

    def __init__(self, env: MultiAgentTradingEnv) -> None:
        self.env = env

    # ------------------------------------------------------------------
    # 抽象インターフェース
    # ------------------------------------------------------------------

    @abstractmethod
    def train(self, num_episodes: int) -> MARLTrainResult:
        """
        MARL 学習を実行する抽象メソッド。

        Parameters
        ----------
        num_episodes : int
            学習エピソード数

        Returns
        -------
        MARLTrainResult
        """
        raise NotImplementedError

    @abstractmethod
    def load_policies(self, policy_dir: Path) -> None:
        """
        保存済みポリシー群をロードする抽象メソッド。

        Parameters
        ----------
        policy_dir : Path
            ポリシーが保存されているディレクトリ
        """
        raise NotImplementedError

    @abstractmethod
    def act(self, states: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        各エージェントの状態から行動を返す抽象メソッド。

        Parameters
        ----------
        states : Dict[str, np.ndarray]
            {agent_id: state_vector}

        Returns
        -------
        Dict[str, float]
            {agent_id: action_ratio}
        """
        raise NotImplementedError
