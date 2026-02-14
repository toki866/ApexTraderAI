# ai_core/rl/env_single.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ai_core.config.rl_config import EnvConfig


@dataclass
class RLEpisodeStats:
    """1エピソード分の結果（資金曲線やDDなど）をまとめる。"""

    final_equity: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    num_steps: int


class RLSingleEnv:
    """単体RL（StepE）用の環境。

    本プロジェクトでは過去に以下の2系統が併存し、同名重複の原因になっていました。

    1) Array Mode:
       - obs_array / price_array / date_array を受け取り、
         事前に構築した観測ベクトル列で学習する方式。
       - 速度が速く、PPO などのループと相性が良い。

    2) DataFrame Mode:
       - prices_df / features_df / preds_df / envelope_daily_df を受け取り、
         その場で 24次元 obs を生成して学習する方式。
       - デバッグしやすく、GUI と相性が良い。

    v2 では RLSingleEnv をこの1ファイルに統一し、
    どちらのモードでも動く「正本」として運用します。
    """

    def __init__(self, *args, **kwargs) -> None:
        # ---- モード判定（互換性重視）----
        # Array Mode: 先頭が np.ndarray なら配列モード
        if len(args) >= 2 and isinstance(args[0], np.ndarray) and isinstance(args[1], np.ndarray):
            obs_array = args[0]
            price_array = args[1]
            date_array = args[2] if len(args) >= 3 else kwargs.get("date_array", None)
            env_cfg = args[3] if len(args) >= 4 else kwargs.get("env_cfg", kwargs.get("env_config", EnvConfig()))
            agent_name = args[4] if len(args) >= 5 else kwargs.get("agent_name", "xsr")
            self._init_array_mode(
                obs_array=obs_array,
                price_array=price_array,
                date_array=date_array,
                env_cfg=env_cfg,
                agent_name=agent_name,
            )
            return

        # DataFrame Mode: legacy signature or keyword
        if (len(args) >= 7 and isinstance(args[0], str)) or ("prices_df" in kwargs and "features_df" in kwargs):
            if len(args) >= 7:
                symbol = args[0]
                agent_name = args[1]
                date_range = args[2]
                prices_df = args[3]
                features_df = args[4]
                preds_df = args[5]
                envelope_daily_df = args[6]
                env_cfg = args[7] if len(args) >= 8 else kwargs.get("env_config", kwargs.get("env_cfg", EnvConfig()))
            else:
                symbol = kwargs["symbol"]
                agent_name = kwargs.get("agent_name", "xsr")
                date_range = kwargs["date_range"]
                prices_df = kwargs["prices_df"]
                features_df = kwargs["features_df"]
                preds_df = kwargs["preds_df"]
                envelope_daily_df = kwargs["envelope_daily_df"]
                env_cfg = kwargs.get("env_config", kwargs.get("env_cfg", EnvConfig()))
            self._init_df_mode(
                symbol=symbol,
                agent_name=agent_name,
                date_range=date_range,
                prices_df=prices_df,
                features_df=features_df,
                preds_df=preds_df,
                envelope_daily_df=envelope_daily_df,
                env_cfg=env_cfg,
            )
            return

        raise TypeError(
            "RLSingleEnv: invalid constructor arguments. "
            "Use (obs_array, price_array, ...) for Array Mode or "
            "(symbol, agent_name, date_range, prices_df, features_df, preds_df, envelope_daily_df, ...) for DataFrame Mode."
        )

    # =========================================================
    # Common helpers
    # =========================================================

    @staticmethod
    def _as_env_cfg(env_cfg: Any) -> EnvConfig:
        # 既存コードが dict を渡してくる場合に備える
        if isinstance(env_cfg, EnvConfig):
            return env_cfg
        if isinstance(env_cfg, dict):
            return EnvConfig.from_dict(env_cfg)
        return EnvConfig()

    # =========================================================
    # Array Mode
    # =========================================================

    def _init_array_mode(
        self,
        obs_array: np.ndarray,
        price_array: np.ndarray,
        date_array: Optional[np.ndarray],
        env_cfg: Any,
        agent_name: str,
    ) -> None:
        self._mode = "array"
        self.agent_name = str(agent_name)
        self.env_cfg = self._as_env_cfg(env_cfg)

        self.obs_array = np.asarray(obs_array, dtype=np.float32)
        self.price_array = np.asarray(price_array, dtype=np.float64)

        if self.obs_array.ndim != 2:
            raise ValueError(f"obs_array must be 2D. got shape={self.obs_array.shape}")
        if self.price_array.ndim != 1:
            raise ValueError(f"price_array must be 1D. got shape={self.price_array.shape}")
        if len(self.obs_array) != len(self.price_array):
            raise ValueError("obs_array and price_array length mismatch")

        if date_array is None:
            self.date_array = np.arange(len(self.price_array))
        else:
            self.date_array = np.asarray(date_array)

        # state
        self.t = 0
        self.position = 0.0
        self.equity = float(self.env_cfg.initial_cash)
        self.bh_equity = float(self.env_cfg.initial_cash)
        self.episode_log: List[Dict[str, Any]] = []

    def reset(self, *args, **kwargs) -> np.ndarray:
        """環境をリセットし初期観測を返す。

        Array Mode:
            reset(evaluation: bool=False)

        DataFrame Mode:
            reset(mode: str="train"|"test")
        """
        if getattr(self, "_mode", None) == "df":
            return self._reset_df(*args, **kwargs)
        return self._reset_array(*args, **kwargs)

    def _reset_array(self, evaluation: bool = False, **kwargs) -> np.ndarray:
        self.t = 0
        self.position = 0.0
        self.equity = float(self.env_cfg.initial_cash)
        self.bh_equity = float(self.env_cfg.initial_cash)
        self.episode_log = []
        return self.obs_array[self.t].copy()

    def step(self, action: Union[float, np.ndarray]) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if getattr(self, "_mode", None) == "df":
            return self._step_df(action)
        return self._step_array(action)

    def _step_array(self, action: Union[float, np.ndarray]) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        # action: ratio in [-1, 1]
        if isinstance(action, np.ndarray):
            ratio = float(action.reshape(-1)[0])
        else:
            ratio = float(action)

        # clip
        ratio = float(np.clip(ratio, -abs(self.env_cfg.max_leverage), abs(self.env_cfg.max_leverage)))

        # return
        if self.t >= len(self.price_array) - 1:
            # already terminal
            obs = np.zeros(self.obs_array.shape[1], dtype=np.float32)
            return obs, 0.0, True, {"reason": "terminal"}

        p0 = float(self.price_array[self.t])
        p1 = float(self.price_array[self.t + 1])
        daily_ret = 0.0 if p0 <= 0 else (p1 / p0 - 1.0)

        # trading cost: bp -> pct
        cost_pct = float(self.env_cfg.trading_cost_bp) / 10_000.0
        trade_cost = cost_pct * abs(ratio - self.position)

        pnl_rate = ratio * daily_ret - trade_cost
        reward = (pnl_rate - float(self.env_cfg.risk_penalty) * (ratio ** 2)) * float(self.env_cfg.reward_scale)

        # update equity (penalty は実損益ではないので equity に入れない)
        self.equity *= (1.0 + pnl_rate)
        self.bh_equity *= (1.0 + daily_ret)

        info = {
            "t": self.t,
            "date": self.date_array[self.t],
            "price": p0,
            "next_price": p1,
            "daily_ret": float(daily_ret),
            "position": float(ratio),
            "prev_position": float(self.position),
            "trade_cost": float(trade_cost),
            "pnl_rate": float(pnl_rate),
            "reward": float(reward),
            "equity": float(self.equity),
            "bh_equity": float(self.bh_equity),
        }
        self.episode_log.append(info)

        self.position = ratio
        self.t += 1

        done = self.t >= len(self.price_array) - 1
        if done:
            next_obs = np.zeros(self.obs_array.shape[1], dtype=np.float32)
        else:
            next_obs = self.obs_array[self.t].copy()

        return next_obs, float(reward), bool(done), info

    # =========================================================
    # DataFrame Mode (legacy / GUI-friendly)
    # =========================================================

    def _init_df_mode(
        self,
        symbol: str,
        agent_name: str,
        date_range: Any,
        prices_df: pd.DataFrame,
        features_df: pd.DataFrame,
        preds_df: pd.DataFrame,
        envelope_daily_df: pd.DataFrame,
        env_cfg: Any,
    ) -> None:
        self._mode = "df"
        self.symbol = str(symbol)
        self.agent_name = str(agent_name)
        self.date_range = date_range
        self.env_cfg = self._as_env_cfg(env_cfg)

        # common_observation を正本として参照（重複禁止方針）
        from ai_core.rl.common_observation import (
            AGENT_NAMES,
            build_observation_24d,
            prepare_common_base_df,
        )

        if self.agent_name not in AGENT_NAMES:
            raise ValueError(f"Invalid agent_name={self.agent_name}. expected one of {AGENT_NAMES}")

        base_df = prepare_common_base_df(
            prices_df=prices_df,
            features_df=features_df,
            preds_df=preds_df,
            envelope_daily_df=envelope_daily_df,
            agent_names=AGENT_NAMES,
        )

        # base_df は Date 列を持つ前提（prepare_common_base_df の仕様）
        if "Date" not in base_df.columns:
            raise KeyError("base_df must have 'Date' column. check prepare_common_base_df")

        base_df = base_df.copy()
        base_df["Date"] = pd.to_datetime(base_df["Date"])
        base_df = base_df.sort_values("Date").reset_index(drop=True)

        # train/test indices
        dr = date_range
        train_start = pd.to_datetime(getattr(dr, "train_start"))
        train_end = pd.to_datetime(getattr(dr, "train_end"))
        test_start = pd.to_datetime(getattr(dr, "test_start"))
        test_end = pd.to_datetime(getattr(dr, "test_end"))

        train_mask = (base_df["Date"] >= train_start) & (base_df["Date"] <= train_end)
        test_mask = (base_df["Date"] >= test_start) & (base_df["Date"] <= test_end)

        self.base_df = base_df
        self.train_indices = np.where(train_mask.values)[0].astype(int)
        self.test_indices = np.where(test_mask.values)[0].astype(int)

        if len(self.train_indices) < 10:
            raise ValueError(f"Train data too short: {len(self.train_indices)}. Check DateRange.")

        # state
        self.mode = "train"  # "train" or "test"
        self.ptr = 0
        self._indices: np.ndarray = self.train_indices

        self.position = 0.0
        self.prev_pnl = 0.0
        self.prev_action = 0.0
        self.equity = float(self.env_cfg.initial_cash)

        # buy & hold equity from base_df if available, else compute fallback
        if "bh_equity" in base_df.columns and base_df["bh_equity"].notna().any():
            self.bh_equity_series = base_df["bh_equity"].astype(float).values
        else:
            if "Close" in base_df.columns:
                close = base_df["Close"].astype(float).values
                first = close[0] if len(close) > 0 else 1.0
                units = float(self.env_cfg.initial_cash) / (first if first > 0 else 1.0)
                self.bh_equity_series = units * close
            else:
                self.bh_equity_series = np.full(len(base_df), np.nan)

        self.episode_log: List[Dict[str, Any]] = []

        # cache builder refs
        self._build_observation_24d = build_observation_24d
        self._AGENT_NAMES = AGENT_NAMES

    def _reset_df(self, *args, **kwargs) -> np.ndarray:
        # legacy: reset(mode="train"|"test")
        mode = None
        if len(args) >= 1 and isinstance(args[0], str):
            mode = args[0]
        if "mode" in kwargs and isinstance(kwargs["mode"], str):
            mode = kwargs["mode"]
        if mode is None:
            mode = "train"

        mode = str(mode).lower().strip()
        if mode not in ("train", "test"):
            raise ValueError(f"mode must be 'train' or 'test'. got {mode}")

        self.mode = mode
        self._indices = self.train_indices if mode == "train" else self.test_indices
        if len(self._indices) < 2:
            raise ValueError(f"Not enough data in mode={mode}. indices={len(self._indices)}")

        self.ptr = 0
        self.position = 0.0
        self.prev_pnl = 0.0
        self.prev_action = 0.0
        self.equity = float(self.env_cfg.initial_cash)
        self.episode_log = []

        idx = int(self._indices[self.ptr])
        obs = self._build_observation_24d(
            base_df=self.base_df,
            idx=idx,
            agent_name=self.agent_name,
            prev_pnl=float(self.prev_pnl),
            prev_action=float(self.prev_action),
            agent_names=self._AGENT_NAMES,
        )
        return obs.astype(np.float32)

    def _step_df(self, action: Union[float, np.ndarray]) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if isinstance(action, np.ndarray):
            ratio = float(action.reshape(-1)[0])
        else:
            ratio = float(action)

        ratio = float(np.clip(ratio, -abs(self.env_cfg.max_leverage), abs(self.env_cfg.max_leverage)))

        idx = int(self._indices[self.ptr])
        row = self.base_df.iloc[idx]

        daily_ret = float(row["ret"]) if "ret" in self.base_df.columns else 0.0

        cost_pct = float(self.env_cfg.trading_cost_bp) / 10_000.0
        trade_cost = cost_pct * abs(ratio - self.position)

        pnl_rate = ratio * daily_ret - trade_cost
        reward = (pnl_rate - float(self.env_cfg.risk_penalty) * (ratio ** 2)) * float(self.env_cfg.reward_scale)

        self.equity *= (1.0 + pnl_rate)

        bh_equity = float(self.bh_equity_series[idx]) if idx < len(self.bh_equity_series) else float("nan")

        info: Dict[str, Any] = {
            "t": int(self.ptr),
            "date": row["Date"],
            "daily_ret": float(daily_ret),
            "position": float(ratio),
            "prev_position": float(self.position),
            "trade_cost": float(trade_cost),
            "pnl_rate": float(pnl_rate),
            "reward": float(reward),
            "equity": float(self.equity),
            "bh_equity": float(bh_equity),
        }

        if "Close" in self.base_df.columns:
            info["price"] = float(row["Close"])

        self.episode_log.append(info)

        self.position = ratio
        self.prev_pnl = float(pnl_rate)
        self.prev_action = float(ratio)

        self.ptr += 1
        done = self.ptr >= (len(self._indices) - 1)
        if done:
            next_obs = np.zeros(24, dtype=np.float32)
        else:
            next_idx = int(self._indices[self.ptr])
            next_obs = self._build_observation_24d(
                base_df=self.base_df,
                idx=next_idx,
                agent_name=self.agent_name,
                prev_pnl=float(self.prev_pnl),
                prev_action=float(self.prev_action),
                agent_names=self._AGENT_NAMES,
            ).astype(np.float32)

        return next_obs, float(reward), bool(done), info

    # =========================================================
    # Reporting
    # =========================================================

    def get_episode_log_dataframe(self) -> pd.DataFrame:
        if not self.episode_log:
            return pd.DataFrame()
        return pd.DataFrame(self.episode_log)

    def compute_episode_stats(self) -> RLEpisodeStats:
        df = self.get_episode_log_dataframe()
        if df.empty or "equity" not in df.columns:
            return RLEpisodeStats(
                final_equity=float(self.env_cfg.initial_cash),
                total_return=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                num_steps=0,
            )

        eq = df["equity"].astype(float).values
        final_equity = float(eq[-1])
        total_return = float(final_equity / float(self.env_cfg.initial_cash) - 1.0)

        hwm = np.maximum.accumulate(eq)
        dd = eq / hwm - 1.0
        max_dd = float(np.min(dd)) if len(dd) else 0.0

        if "pnl_rate" in df.columns:
            r = df["pnl_rate"].astype(float).values
        elif "reward" in df.columns:
            r = df["reward"].astype(float).values
        else:
            r = np.zeros(len(eq), dtype=float)

        if np.std(r) == 0.0 or len(r) < 2:
            sharpe = 0.0
        else:
            sharpe = float(np.mean(r) / np.std(r) * np.sqrt(252))

        return RLEpisodeStats(
            final_equity=final_equity,
            total_return=total_return,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            num_steps=int(len(eq)),
        )
