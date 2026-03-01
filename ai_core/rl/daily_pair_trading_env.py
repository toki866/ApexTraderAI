from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DailyPairTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        X: np.ndarray,
        r_soxl_next: np.ndarray,
        r_soxs_next: np.ndarray,
        trade_cost_bps: float = 5.0,
        pos_limit: float = 1.0,
        pos_l2: float = 1e-3,
    ) -> None:
        super().__init__()
        self.X = np.asarray(X, dtype=np.float32)
        self.r_soxl_next = np.asarray(r_soxl_next, dtype=np.float32)
        self.r_soxs_next = np.asarray(r_soxs_next, dtype=np.float32)
        if not (len(self.X) == len(self.r_soxl_next) == len(self.r_soxs_next)):
            raise ValueError("X/r_soxl_next/r_soxs_next must have identical length")

        self.T = int(len(self.X))
        self.obs_dim = int(self.X.shape[1])
        self.trade_cost_bps = float(trade_cost_bps)
        self.pos_limit = float(pos_limit)
        self.pos_l2 = float(pos_l2)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.float32(-np.inf),
            high=np.float32(np.inf),
            shape=(self.obs_dim + 1,),
            dtype=np.float32,
        )

        self.t = 0
        self.pos_prev = np.float32(0.0)

    def _obs(self) -> np.ndarray:
        if self.t >= self.T:
            core = np.zeros((self.obs_dim,), dtype=np.float32)
        else:
            core = self.X[self.t]
        return np.concatenate([core, np.array([self.pos_prev], dtype=np.float32)], axis=0).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.t = 0
        self.pos_prev = np.float32(0.0)
        return self._obs(), {}

    def step(self, action):
        if self.t >= self.T:
            return self._obs(), 0.0, True, False, {}

        pos = float(np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[0], -self.pos_limit, self.pos_limit))
        gross = max(pos, 0.0) * float(self.r_soxl_next[self.t]) + max(-pos, 0.0) * float(self.r_soxs_next[self.t])
        cost = float(self.trade_cost_bps) * 1e-4 * abs(pos - float(self.pos_prev))
        penalty = float(self.pos_l2) * (pos * pos)
        reward_next = float(gross - cost - penalty)

        self.pos_prev = np.float32(pos)
        self.t += 1
        done = self.t == self.T
        info = {"pos": pos, "gross": gross, "cost": cost, "penalty": penalty, "reward_next": reward_next}
        return self._obs(), reward_next, done, False, info
