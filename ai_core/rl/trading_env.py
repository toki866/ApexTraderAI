# ai_core/rl/trading_env.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


@dataclass
class TradingEnvConfig:
    """
    SB3 PPO 用のトレーディング環境設定。

    Attributes
    ----------
    initial_equity : float
        初期資産（資金曲線のスケール）。
    max_position : float
        1日あたりの最大ポジション比率（-1.0〜+1.0）。
        実際のアクションは [-1, 1] をこの値でスケーリングして解釈する。
    trading_cost_pct : float
        売買コスト率（片道）。|Δポジション| × equity × trading_cost_pct をコストとして控除。
    reward_scale : float
        報酬スケール。PnL / initial_equity に reward_scale を掛けて reward とする。
    """
    initial_equity: float = 1.0
    max_position: float = 1.0
    trading_cost_pct: float = 0.001
    reward_scale: float = 1.0


class TradingEnv(gym.Env):
    """
    1日1ステップのシンプルなトレーディング環境（SOXL/SOXSペア対応）。

    - 観測: 任意次元ベクトル（通常は24次元）
    - アクション: ratio ∈ [-1, +1]
        ratio > 0 → SOXL ロング（max_position でスケーリング）
        ratio < 0 → SOXS ロング
        ratio = 0 → ノーポジ
    - 報酬: 当日PnL - コスト を initial_equity で割って reward_scale を掛けた値
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df_obs: pd.DataFrame,
        price_long: pd.Series,
        price_short: pd.Series,
        config: Optional[TradingEnvConfig] = None,
    ) -> None:
        super().__init__()

        if not (len(df_obs) == len(price_long) == len(price_short)):
            raise ValueError(
                f"df_obs({len(df_obs)}), price_long({len(price_long)}), "
                f"price_short({len(price_short)}) の長さが一致しません。"
            )

        self.df_obs = df_obs.reset_index(drop=True)
        self.price_long = pd.Series(price_long, copy=True).reset_index(drop=True).astype(float)
        self.price_short = pd.Series(price_short, copy=True).reset_index(drop=True).astype(float)
        self.cfg = config or TradingEnvConfig()

        self.n_steps = len(self.df_obs)
        self.n_features = self.df_obs.shape[1]

        # Gym の定義
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_features,),
            dtype=np.float32,
        )

        # 内部状態
        self._t: int = 0
        self._position: float = 0.0  # [-max_position, +max_position]
        self._equity: float = self.cfg.initial_equity

    # ----------------------------
    # Gymnasium API
    # ----------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        """
        エピソードをリセットし、初期観測と info を返す。
        """
        super().reset(seed=seed)

        self._t = 0
        self._position = 0.0
        self._equity = self.cfg.initial_equity

        obs = self._get_obs()
        info = {
            "equity": self._equity,
            "position": self._position,
            "pnl": 0.0,
        }
        return obs, info

    def step(self, action):
        """
        1ステップ進める。

        Parameters
        ----------
        action : np.ndarray | float
            [-1, +1] の連続値。max_position でスケーリングして解釈。
        """
        # action をスカラーに
        if isinstance(action, (list, tuple, np.ndarray)):
            ratio_raw = float(np.asarray(action).reshape(-1)[0])
        else:
            ratio_raw = float(action)

        # -1〜+1 にクリップし、max_position を掛ける
        ratio = float(
            np.clip(ratio_raw, -1.0, 1.0) * float(self.cfg.max_position)
        )

        # すでに最終時点なら即終了
        if self._t >= self.n_steps - 1:
            obs = self._get_obs()
            info = {
                "equity": self._equity,
                "position": self._position,
                "pnl": 0.0,
            }
            return obs, 0.0, True, False, info

        t = self._t
        t_next = t + 1

        price_long_t = float(self.price_long.iloc[t])
        price_long_next = float(self.price_long.iloc[t_next])
        price_short_t = float(self.price_short.iloc[t])
        price_short_next = float(self.price_short.iloc[t_next])

        # 日次リターン
        ret_long = (price_long_next / price_long_t) - 1.0 if price_long_t > 0 else 0.0
        ret_short = (price_short_next / price_short_t) - 1.0 if price_short_t > 0 else 0.0

        # 前日までのポジションでPnL計算
        prev_pos = self._position
        weight_long = max(prev_pos, 0.0)
        weight_short = max(-prev_pos, 0.0)

        pnl = (weight_long * ret_long + weight_short * ret_short) * self._equity

        # 新しいターゲットポジションへの変更コスト
        pos_change = ratio - prev_pos
        trade_cost = abs(pos_change) * self._equity * float(self.cfg.trading_cost_pct)

        # 資産更新
        self._equity = self._equity + pnl - trade_cost
        self._position = ratio
        self._t = t_next

        # エピソード終了判定
        done = False
        if self._t >= self.n_steps - 1:
            done = True

        # 報酬（initial_equity でスケール）
        base = float(self.cfg.initial_equity) if self.cfg.initial_equity != 0 else 1.0
        reward = (pnl - trade_cost) / base * float(self.cfg.reward_scale)

        obs = self._get_obs()
        info = {
            "equity": self._equity,
            "position": self._position,
            "pnl": pnl,
            "ret_long": ret_long,
            "ret_short": ret_short,
            "trade_cost": trade_cost,
        }
        return obs, float(reward), done, False, info

    # ----------------------------
    # 補助
    # ----------------------------

    def _get_obs(self) -> np.ndarray:
        row = self.df_obs.iloc[self._t]
        return row.astype(np.float32).to_numpy()
