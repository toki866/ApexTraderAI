from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
from torch import nn

from ai_core.rl.types import RLRunResult
from ai_core.rl.envs import MultiAgentTradingEnv
from ai_core.config.rl_config import RLMARLConfig


@dataclass
class _PPOHyperParams:
    """
    PPO のハイパーパラメータ集。

    RLMARLConfig から上書きされることを想定。
    """
    # 1エポックあたりのステップ数（ロールアウト長）
    rollout_len: int = 256
    # 学習エポック数（= ロールアウト→更新の回数）
    num_epochs: int = 50
    # PPO の内部更新で何回サブエポックを回すか
    ppo_epochs: int = 10
    # 1サブエポックあたりのミニバッチサイズ
    batch_size: int = 64

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    lr: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    hidden_dim: int = 64


class _MARLActorCritic(nn.Module):
    """
    連続アクション（1次元：ポジション比率）の Actor-Critic ネットワーク。

    入力: 観測ベクトル obs (obs_dim)
    出力:
      - mu: 行動の平均 (1次元)
      - log_std: 行動の log 標準偏差 (1次元, 学習可能パラメータを展開)
      - value: 状態価値 V(s)
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.obs_dim = obs_dim

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.mu_head = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(1))
        self.v_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        obs : torch.Tensor
            obs: (..., obs_dim)

        Returns
        -------
        mu : torch.Tensor
            形状: (..., )
        log_std : torch.Tensor
            形状: (..., )
        value : torch.Tensor
            形状: (..., )
        """
        x = self.actor(obs)
        v = self.critic(obs)

        mu = self.mu_head(x).squeeze(-1)
        log_std = self.log_std.expand_as(mu)
        value = self.v_head(v).squeeze(-1)
        return mu, log_std, value


class MARLAlgo:
    """
    StepF 用 PPO（MARL）アルゴリズム実装。

    - MultiAgentTradingEnv を対象とした 1次元連続アクション PPO
    - 学習後に評価エピソードを1本走らせ、Equity / 日次ログ / Metrics を CSV 出力
    - 環境側の get_action_log_dataframe() からアクションヒートマップ用 CSV も出力
    """

    def __init__(
        self,
        env: MultiAgentTradingEnv,
        config: RLMARLConfig,
        use_gpu: bool,
        output_dir: Path,
        symbol: str,
    ) -> None:
        self.env = env
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.symbol = symbol

        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )

        # 観測次元を推定
        obs_sample = self.env.reset(evaluation=False)
        if isinstance(obs_sample, np.ndarray):
            obs_dim = int(obs_sample.shape[-1])
        else:
            obs_dim = 1

        # ハイパーパラメータを config から解決
        self.hparams = self._resolve_hparams(config)

        # Actor-Critic ネットワーク
        self.ac = _MARLActorCritic(
            obs_dim=obs_dim,
            hidden_dim=self.hparams.hidden_dim,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=self.hparams.lr)

    # ------------------------------------------------------------------
    # 内部ユーティリティ
    # ------------------------------------------------------------------

    def _to_tensor(self, arr: np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        numpy -> torch.Tensor 変換＋device 乗せ helper。
        """
        if isinstance(arr, torch.Tensor):
            return arr.to(self.device)
        t = torch.from_numpy(np.asarray(arr, dtype=np.float32))
        return t.to(self.device)

    @staticmethod
    def _resolve_hparams(cfg: RLMARLConfig) -> _PPOHyperParams:
        hp = _PPOHyperParams()
        # cfg が dataclass でもクラスでも、属性があれば上書き
        for field in hp.__dataclass_fields__.keys():
            if hasattr(cfg, field):
                setattr(hp, field, getattr(cfg, field))
        return hp

    # ------------------------------------------------------------------
    # メイン入口
    # ------------------------------------------------------------------

    def train_and_evaluate(
        self,
        cancel_event: Optional[object] = None,
    ) -> RLRunResult:
        """
        PPO 学習＋評価＋CSV出力。

        cancel_event がセットされた場合、
        学習ループを中断して「キャンセル終了」として結果を返す。
        """
        cancelled = False

        # 学習ループ
        for epoch in range(self.hparams.num_epochs):
            if getattr(cancel_event, "is_set", lambda: False)():
                cancelled = True
                break

            batch = self._collect_rollout()
            self._ppo_update(batch)

        # 学習が途中でキャンセルされた場合でも、評価は一度実行する
        eval_result = self._evaluate_and_save()

        if cancelled:
            eval_result.message = (
                "[CANCELLED] " + eval_result.message
            )

        return eval_result

    # ------------------------------------------------------------------
    # ロールアウト収集
    # ------------------------------------------------------------------

    def _collect_rollout(self) -> dict:
        """
        1エピソード分のロールアウトを収集し、GAE を計算して返す。

        MultiAgentTradingEnv は 1エピソード = 学習/テスト期間全日分 という前提で、
        env.num_steps ぶんを1回で取り切ることを想定。
        """
        obs_buf: List[np.ndarray] = []
        act_buf: List[float] = []
        logprob_buf: List[float] = []
        value_buf: List[float] = []
        rew_buf: List[float] = []
        done_buf: List[bool] = []

        env = self.env
        obs = env.reset(evaluation=False)

        # マージ後の日数に応じて自動的に終わる想定
        while True:
            obs_tensor = self._to_tensor(obs)
            with torch.no_grad():
                mu, log_std, value = self.ac(obs_tensor)
                dist = torch.distributions.Normal(mu, log_std.exp())
                action_tensor = dist.sample()
                log_prob_tensor = dist.log_prob(action_tensor)

            action = float(action_tensor.squeeze().cpu().numpy())
            log_prob = float(log_prob_tensor.squeeze().cpu().numpy())
            value_val = float(value.squeeze().cpu().numpy())

            next_obs, reward, done, info = env.step(action)

            obs_buf.append(np.asarray(obs, dtype=np.float32))
            act_buf.append(action)
            logprob_buf.append(log_prob)
            value_buf.append(value_val)
            rew_buf.append(float(reward))
            done_buf.append(bool(done))

            obs = next_obs
            if done:
                break

        # 最終状態の価値を bootstrap 用に推定
        with torch.no_grad():
            last_obs_tensor = self._to_tensor(obs)
            _, _, last_v_tensor = self.ac(last_obs_tensor)
            last_v = float(last_v_tensor.squeeze().cpu().numpy())

        rewards = np.asarray(rew_buf, dtype=np.float32)
        values = np.asarray(value_buf, dtype=np.float32)
        dones = np.asarray(done_buf, dtype=np.float32)

        T = len(rewards)
        values_ext = np.concatenate([values, np.array([last_v], dtype=np.float32)])

        # GAE-Lambda
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.hparams.gamma * values_ext[t + 1] * mask - values_ext[t]
            gae = delta + self.hparams.gamma * self.hparams.gae_lambda * mask * gae
            advantages[t] = gae
        returns = advantages + values

        # advantage の標準化
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        batch = {
            "obs": np.asarray(obs_buf, dtype=np.float32),
            "actions": np.asarray(act_buf, dtype=np.float32),
            "logprobs": np.asarray(logprob_buf, dtype=np.float32),
            "values": values,
            "advantages": advantages,
            "returns": returns,
        }
        return batch

    # ------------------------------------------------------------------
    # PPO 更新
    # ------------------------------------------------------------------

    def _ppo_update(self, batch: dict) -> None:
        """
        収集したロールアウトに対して PPO 更新を行う。
        """
        obs = self._to_tensor(batch["obs"])
        actions = self._to_tensor(batch["actions"]).view(-1)
        old_logprobs = self._to_tensor(batch["logprobs"])
        advantages = self._to_tensor(batch["advantages"])
        returns = self._to_tensor(batch["returns"])

        num_samples = obs.shape[0]
        indices = np.arange(num_samples)

        for _ in range(self.hparams.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, num_samples, self.hparams.batch_size):
                end = start + self.hparams.batch_size
                mb_idx = indices[start:end]
                mb_idx_t = torch.from_numpy(mb_idx).to(self.device)

                mb_obs = obs[mb_idx_t]
                mb_actions = actions[mb_idx_t]
                mb_old_logprobs = old_logprobs[mb_idx_t]
                mb_advantages = advantages[mb_idx_t]
                mb_returns = returns[mb_idx_t]

                mu, log_std, values = self.ac(mb_obs)
                dist = torch.distributions.Normal(mu, log_std.exp())
                new_logprobs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # ratio = πθ(a|s) / πθ_old(a|s)
                ratio = torch.exp(new_logprobs - mb_old_logprobs)

                # PPO clip objective
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.hparams.clip_range,
                    1.0 + self.hparams.clip_range,
                ) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # value loss
                value_loss = nn.functional.mse_loss(values, mb_returns)

                # total loss
                loss = (
                    policy_loss
                    + self.hparams.value_coef * value_loss
                    - self.hparams.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.hparams.max_grad_norm)
                self.optimizer.step()

    # ------------------------------------------------------------------
    # 評価＋CSV出力
    # ------------------------------------------------------------------

    def _evaluate_and_save(self) -> RLRunResult:
        """
        学習済みポリシーで 1 エピソードを評価し、各種CSVを保存して RLRunResult を返す。
        """
        env = self.env
        obs = env.reset(evaluation=True)

        while True:
            obs_tensor = self._to_tensor(obs)
            with torch.no_grad():
                mu, log_std, value = self.ac(obs_tensor)
                # 評価時は平均値で deterministic に行動
                action_tensor = torch.tanh(mu)
            action = float(action_tensor.squeeze().cpu().numpy())

            next_obs, reward, done, info = env.step(action)
            obs = next_obs
            if done:
                break

        # 日次ログを取得
        log_df = env.get_episode_log_dataframe()
        if log_df is None or log_df.empty:
            # ログが無い場合は結果は0で success=False
            return RLRunResult(
                success=False,
                message="StepF evaluation failed: episode log is empty.",
                final_return=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                bh_final_return=0.0,
                policy_path=self.output_dir / f"policy_stepF_MARL_{self.symbol}.npz",
                equity_curve_path=self.output_dir / f"stepF_equity_{self.symbol}.csv",
                daily_log_path=self.output_dir / f"stepF_daily_log_{self.symbol}.csv",
                metrics_csv_path=self.output_dir / f"stepF_rl_metrics_{self.symbol}.csv",
            )

        # メトリクス計算
        equity = log_df["Equity"].astype(float).values
        bh_equity = log_df["BH_Equity"].astype(float).values

        initial_equity = equity[0]
        final_equity = equity[-1]
        final_return = float(final_equity / initial_equity - 1.0)

        initial_bh = bh_equity[0]
        final_bh = bh_equity[-1]
        bh_final = float(final_bh / initial_bh - 1.0)

        max_dd = self._max_drawdown(equity)
        sharpe = self._sharpe_ratio_from_equity(equity)

        # ポリシー保存（nn.Module の state_dict を npz に格納）
        policy_path = self.output_dir / f"policy_stepF_MARL_{self.symbol}.npz"
        self._save_policy(policy_path)

        # Equity カーブ CSV
        equity_df = log_df[["Date", "Equity", "BH_Equity"]].copy()
        equity_path = self.output_dir / f"stepF_equity_{self.symbol}.csv"
        equity_df.to_csv(equity_path, index=False)

        # 日次ログ CSV
        daily_log_path = self.output_dir / f"stepF_daily_log_{self.symbol}.csv"
        log_df.to_csv(daily_log_path, index=False)

        # メトリクス CSV
        metrics = {
            "final_return": final_return,
            "max_drawdown": max_dd,
            "sharpe_ratio": sharpe,
            "bh_final_return": bh_final,
        }
        metrics_df = pd.DataFrame([metrics])
        metrics_path = self.output_dir / f"stepF_rl_metrics_{self.symbol}.csv"
        metrics_df.to_csv(metrics_path, index=False)

        # アクションヒートマップ用 CSV（あれば）
        try:
            actions_df = env.get_action_log_dataframe()
            if actions_df is not None and not actions_df.empty:
                actions_path = self.output_dir / f"stepF_actions_{self.symbol}.csv"
                actions_df.to_csv(actions_path, index=False)
        except Exception:
            # 無くても致命的ではないので握りつぶす
            pass

        return RLRunResult(
            success=True,
            message=f"StepF MARL training & evaluation finished for symbol={self.symbol}",
            final_return=final_return,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            bh_final_return=bh_final,
            policy_path=policy_path,
            equity_curve_path=equity_path,
            daily_log_path=daily_log_path,
            metrics_csv_path=metrics_path,
        )

    # ------------------------------------------------------------------
    # ポリシー保存
    # ------------------------------------------------------------------

    def _save_policy(self, path: Path) -> None:
        """
        ポリシー（ActorCritic の state_dict）を numpy の npz として保存する。

        将来の LivePolicyRunner / 自動売買エンジンでロードして使うことを想定。
        """
        state_dict = self.ac.state_dict()
        # state_dict は {name: tensor} なので、tensor を numpy 配列に変換して保存
        np_state = {k: v.detach().cpu().numpy() for k, v in state_dict.items()}
        np.savez(path, **np_state)

    # ------------------------------------------------------------------
    # ヘルパー（最大DD / Sharpe）
    # ------------------------------------------------------------------

    @staticmethod
    def _max_drawdown(equity: np.ndarray) -> float:
        peak = equity[0]
        max_dd = 0.0
        for v in equity:
            if v > peak:
                peak = v
            dd = (v - peak) / peak
            if dd < max_dd:
                max_dd = dd
        return float(max_dd)

    @staticmethod
    def _sharpe_ratio_from_equity(
        equity: np.ndarray,
        risk_free: float = 0.0,
        trading_days: int = 252,
    ) -> float:
        if len(equity) < 2:
            return 0.0
        ret = equity[1:] / equity[:-1] - 1.0
        excess = ret - risk_free / trading_days
        std = excess.std(ddof=1)
        if std <= 0:
            return 0.0
        # 年率換算
        return float(excess.mean() / std * np.sqrt(trading_days))
