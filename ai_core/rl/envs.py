from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from ai_core.types.common import DateRange
from ai_core.config.rl_config import EnvConfig


@dataclass
class _MergedData:
    df: pd.DataFrame
    numeric_cols_for_obs: List[str]
    agent_action_cols: Dict[str, Optional[str]]


class MultiAgentTradingEnv:
    """
    StepF 用マルチエージェント環境（実体は 1次元連続アクションのトレーディング環境）。

    役割
    ----
    - prices_df / features_df / preds_df / envelope_events を Date でマージし、
      1 日あたり 1 行のデータフレームを作る
    - 観測ベクトルは「Date を除いた数値列」をまとめたもの
    - action_ratio ∈ [-1, 1] を受け取り、
      日次リターンとの掛け算で PnL / Equity を更新
    - Buy & Hold の Equity も併せて管理
    - 各ステップで「Date と XSR/LSTM/FED の代理アクション（予測変化量ベース）」を記録し、
      get_action_log_dataframe() で返す
    - 評価エピソード中の「Date / Close / Position / Reward / Equity / BH_Equity」を
      get_episode_log_dataframe() で返す
    """

    def __init__(
        self,
        symbol: str,
        date_range: DateRange,
        prices_df: pd.DataFrame,
        features_df: pd.DataFrame,
        preds_df: pd.DataFrame,
        envelope_events: Dict[str, pd.DataFrame],
        env_config: EnvConfig,
        agent_names: List[str],
        agent_policy_paths: Dict[str, Any],
        agent_weights: Dict[str, float],
    ) -> None:
        self.symbol = symbol
        self.date_range = date_range
        self.env_config = env_config

        # コピーして内部で使う
        self._prices_df = prices_df.copy()
        self._features_df = features_df.copy()
        self._preds_df = preds_df.copy()
        self._envelope_events = {k: v.copy() for k, v in envelope_events.items()}

        self.agent_names = list(agent_names)
        self.agent_policy_paths = dict(agent_policy_paths)
        self.agent_weights = dict(agent_weights)

        # 日付列を datetime64[ns] に揃える
        for df in (self._prices_df, self._features_df, self._preds_df):
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])

        for df in self._envelope_events.values():
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])

        # すべてをマージして 1 日 1 行のテーブルを作成
        merged = self._merge_all_sources()

        self._data = merged.df
        self._numeric_cols_for_obs = merged.numeric_cols_for_obs
        self._agent_action_cols = merged.agent_action_cols

        # エピソード用の状態
        self._idx: int = 0
        self.equity: float = 1.0
        self.bh_equity: float = 1.0
        self.position: float = 0.0

        # ログ
        self._episode_log: List[Dict[str, Any]] = []
        self._action_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # マージ処理
    # ------------------------------------------------------------------

    def _merge_all_sources(self) -> _MergedData:
        # prices: 必須
        if "Date" not in self._prices_df.columns:
            raise ValueError("prices_df に Date 列がありません。")

        prices = self._prices_df.copy()
        prices["Date"] = pd.to_datetime(prices["Date"])
        prices = prices.sort_values("Date").reset_index(drop=True)

        base_cols = ["Date"]
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in prices.columns:
                base_cols.append(col)
        prices = prices[base_cols]

        df = prices

        # features をマージ
        if not self._features_df.empty and "Date" in self._features_df.columns:
            feat = self._features_df.copy()
            feat["Date"] = pd.to_datetime(feat["Date"])
            df = df.merge(feat, on="Date", how="left", suffixes=("", "_feat"))

        # preds をマージ
        if not self._preds_df.empty and "Date" in self._preds_df.columns:
            preds = self._preds_df.copy()
            preds["Date"] = pd.to_datetime(preds["Date"])
            df = df.merge(preds, on="Date", how="left", suffixes=("", "_pred"))

        # envelope（とりあえず xsr のみ利用。無ければスキップ）
        env_xsr = self._envelope_events.get("xsr")
        if env_xsr is not None and not env_xsr.empty and "Date" in env_xsr.columns:
            env_df = env_xsr.copy()
            env_df["Date"] = pd.to_datetime(env_df["Date"])
            df = df.merge(env_df, on="Date", how="left", suffixes=("", "_env"))

        # 日次リターン
        if "Close" not in df.columns:
            raise ValueError("マージ後のデータに Close 列がありません。")

        df = df.sort_values("Date").reset_index(drop=True)
        df["Close_next"] = df["Close"].shift(-1)
        df["Return"] = (df["Close_next"] - df["Close"]) / df["Close"]
        df["Return"] = df["Return"].fillna(0.0)

        # エージェントごとの「予測変化量」列を探して、アクションの代理指標にする
        agent_action_cols: Dict[str, Optional[str]] = {}
        for agent in self.agent_names:
            cand_cols: List[str] = []
            low_agent = agent.lower()
            for col in df.columns:
                if low_agent in col.lower() and "date" not in col.lower():
                    # 数値列のみ対象
                    if np.issubdtype(df[col].dtype, np.number):
                        cand_cols.append(col)
            if cand_cols:
                base_col = cand_cols[0]
                diff_col = f"__d_pred_{agent}"
                df[diff_col] = df[base_col].diff().fillna(0.0)
                agent_action_cols[agent] = diff_col
            else:
                agent_action_cols[agent] = None

        # 観測に使う数値列（Date / Close_next は除外）
        numeric_cols_for_obs: List[str] = []
        for col in df.columns:
            if col == "Date":
                continue
            if not np.issubdtype(df[col].dtype, np.number):
                continue
            if col in ("Close_next",):
                continue
            numeric_cols_for_obs.append(col)

        return _MergedData(
            df=df,
            numeric_cols_for_obs=numeric_cols_for_obs,
            agent_action_cols=agent_action_cols,
        )

    # ------------------------------------------------------------------
    # 公開インターフェース
    # ------------------------------------------------------------------

    def reset(self, *, evaluation: bool = False) -> np.ndarray:
        """
        環境をリセットして最初の観測を返す。

        evaluation=True のときは、
        ログ（エピソードログ・アクションログ）をクリアしてから開始する。
        """
        self._idx = 0
        self.equity = 1.0
        self.bh_equity = 1.0
        self.position = 0.0

        if evaluation:
            self._episode_log.clear()
            self._action_log.clear()

        return self._build_observation(self._idx)

    def step(self, action_ratio: float) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        1 日分のステップを進める。

        Parameters
        ----------
        action_ratio : float
            [-1, 1] の範囲のポジション比率。
            正: ロング、負: ショート（SOXS）という解釈を想定。
        """
        # 安全のためクリップ
        ratio = float(np.clip(action_ratio, -1.0, 1.0))

        # 終端チェック
        if self._idx >= len(self._data) - 1:
            # これ以上進めないので、報酬0で終了扱い
            obs = self._build_observation(self._idx)
            return obs, 0.0, True, {}

        row = self._data.iloc[self._idx]
        current_close = float(row["Close"])
        daily_ret = float(row["Return"])

        # 報酬：ポジション比率 × 日次リターン
        reward = ratio * daily_ret

        # Equity 更新
        self.equity *= (1.0 + reward)
        # Buy & Hold（常に1倍ロング）も更新
        self.bh_equity *= (1.0 + daily_ret)

        # ポジション更新
        self.position = ratio

        # 日付
        current_date = pd.to_datetime(row["Date"])

        # アクションログ用のエージェント別行動（予測変化量ベースの代理）
        agent_actions: Dict[str, float] = {}
        for agent in self.agent_names:
            col = self._agent_action_cols.get(agent)
            if col is not None and col in self._data.columns:
                val = float(self._data.iloc[self._idx][col])
                # スケールを抑えるため tanh を通す
                agent_actions[agent] = float(np.tanh(val))
            else:
                agent_actions[agent] = 0.0

        # アクションログに1行追加
        action_row: Dict[str, Any] = {"Date": current_date}
        for agent, a_val in agent_actions.items():
            action_row[f"a_{agent}"] = a_val
        self._action_log.append(action_row)

        # エピソードログにも追加
        self._episode_log.append(
            {
                "Date": current_date,
                "Close": current_close,
                "Position": self.position,
                "Reward": reward,
                "Equity": self.equity,
                "BH_Equity": self.bh_equity,
            }
        )

        # 次のインデックスへ
        self._idx += 1
        done = self._idx >= len(self._data) - 1
        next_obs = self._build_observation(self._idx)

        info: Dict[str, Any] = {
            "date": current_date,
            "equity": self.equity,
            "bh_equity": self.bh_equity,
            "position": self.position,
            "agent_actions": agent_actions,
        }

        return next_obs, reward, done, info

    # ------------------------------------------------------------------
    # 観測ベクトル生成
    # ------------------------------------------------------------------

    def _build_observation(self, idx: int) -> np.ndarray:
        """
        観測ベクトルを生成する。

        現状は「Date を除いた数値列（Close_next を除く）」をまとめたもの。
        24次元に限定せず、列数に応じて自動的に次元数が決まる。
        """
        if len(self._data) == 0:
            return np.zeros(1, dtype=np.float32)

        idx_clamped = int(np.clip(idx, 0, len(self._data) - 1))
        row = self._data.iloc[idx_clamped]

        values: List[float] = []
        for col in self._numeric_cols_for_obs:
            val = row[col]
            if pd.isna(val):
                values.append(0.0)
            else:
                values.append(float(val))

        if not values:
            values = [0.0]

        return np.asarray(values, dtype=np.float32)

    # ------------------------------------------------------------------
    # ログ取得
    # ------------------------------------------------------------------

    def get_episode_log_dataframe(self) -> pd.DataFrame:
        """
        評価エピソード中に蓄積した日次ログを DataFrame で返す。
        """
        if not self._episode_log:
            return pd.DataFrame()
        df = pd.DataFrame(self._episode_log)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
        return df

    def get_action_log_dataframe(self) -> pd.DataFrame:
        """
        学習・評価中に蓄積した「Date + a_xxx」形式のアクションログを返す。
        """
        if not self._action_log:
            return pd.DataFrame()
        df = pd.DataFrame(self._action_log)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
        return df
