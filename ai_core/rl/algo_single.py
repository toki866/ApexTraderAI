from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ai_core.rl.env_single import RLSingleEnv


@dataclass
class RLSingleTrainResult:
    """
    単体RL（StepE）の学習結果サマリ。
    StepEService から StepEResult を組み立てるときの中間オブジェクトを想定。
    """
    ok: bool
    message: str = ""
    metrics: Dict[str, Any] | None = None
    artifacts: Dict[str, Path] | None = None


class _SingleActorCritic(nn.Module):
    """
    単体エージェント用の Actor-Critic ネットワーク（PPO 用）。
    連続アクション（1次元：ポジション比率 [-1, 1]）を想定。
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        # シンプルな 2層 MLP
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        # Actor: 平均(μ)とログ分散(log σ)
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(1))  # 学習可能パラメータ
        # Critic: 価値関数 V(s)
        self.v_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        obs : Tensor, shape (B, obs_dim) or (obs_dim,)

        Returns
        -------
        mu : Tensor, shape (B,)
        log_std : Tensor, shape (B,)
        value : Tensor, shape (B,)
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        x = self.shared(obs)
        mu = self.mu_head(x).squeeze(-1)
        log_std = self.log_std.expand_as(mu)
        value = self.v_head(x).squeeze(-1)
        return mu, log_std, value


class RLSingleAlgo:
    """
    単体エージェント用 PPO アルゴリズム実装。

    - env: RLSingleEnv（1銘柄 / 1エージェント）
    - PPO Actor-Critic で policy を学習
    - 学習後、評価1エピソードを回して日次ログを取得
    - ポリシーとログ・メトリクスを出力ディレクトリへ保存

    Stable-Baselines3 には依存せず、シンプルな PyTorch 実装。
    """

    def __init__(
        self,
        env: RLSingleEnv,
        config: Optional[Any] = None,
        *,
        agent_name: str = "xsr",
        output_dir: str | Path | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        """
        Parameters
        ----------
        env : RLSingleEnv
            単体RL環境。StepEService 側で DateRange / EnvConfig / RLSingleConfig などから生成したもの。
        config : Any, optional
            RLSingleConfig 相当の設定オブジェクト。
            以下の属性があれば優先的に使用し、なければデフォルト値を使う：
              - total_updates (int)        : PPO のアップデート回数
              - rollout_horizon (int)      : 1アップデートあたりの最大ステップ数（デフォルト: env.num_steps）
              - batch_size (int)
              - num_epochs (int)
              - gamma (float)
              - gae_lambda (float)
              - clip_range (float)
              - lr (float)
              - value_coef (float)
              - entropy_coef (float)
              - max_grad_norm (float)
              - hidden_dim (int)
        agent_name : str, default "xsr"
            ログ・ファイル名に埋め込むエージェント識別子。
        output_dir : str | Path, optional
            出力先ディレクトリ。None の場合は ./output/stepE_single/{agent_name} を使う。
        device : str | torch.device, optional
            "cpu" / "cuda" 等。None の場合は自動判定。
        """
        self.env = env
        self.agent_name = agent_name
        self.config = config

        # Device 判定
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 観測次元
        if getattr(env, "obs_array", None) is not None and len(env.obs_array) > 0:
            obs_sample = np.asarray(env.obs_array[0], dtype=np.float32)
            if obs_sample.ndim == 0:
                obs_dim = 1
            else:
                obs_dim = int(obs_sample.shape[-1])
        else:
            obs_dim = 1

        # Actor-Critic ネット
        hidden_dim = int(getattr(config, "hidden_dim", 64)) if config is not None else 64
        self.model = _SingleActorCritic(obs_dim=obs_dim, hidden_dim=hidden_dim).to(self.device)

        # ハイパーパラメータ（config から取れれば取る）
        self.total_updates: int = int(getattr(config, "total_updates", 50)) if config is not None else 50

        default_horizon = getattr(env, "num_steps", None)
        if default_horizon is None or default_horizon <= 0:
            default_horizon = 256
        self.rollout_horizon: int = (
            int(getattr(config, "rollout_horizon", default_horizon)) if config is not None else default_horizon
        )

        self.batch_size: int = int(getattr(config, "batch_size", 64)) if config is not None else 64
        self.num_epochs: int = int(getattr(config, "num_epochs", 10)) if config is not None else 10

        self.gamma: float = float(getattr(config, "gamma", 0.99)) if config is not None else 0.99
        self.gae_lambda: float = float(getattr(config, "gae_lambda", 0.95)) if config is not None else 0.95
        self.clip_range: float = float(getattr(config, "clip_range", 0.2)) if config is not None else 0.2
        self.lr: float = float(getattr(config, "lr", 3e-4)) if config is not None else 3e-4
        self.value_coef: float = float(getattr(config, "value_coef", 0.5)) if config is not None else 0.5
        self.entropy_coef: float = float(getattr(config, "entropy_coef", 0.01)) if config is not None else 0.01
        self.max_grad_norm: float = float(getattr(config, "max_grad_norm", 0.5)) if config is not None else 0.5

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # 出力ディレクトリ
        if output_dir is None:
            self.output_dir = Path("output") / "stepE_single" / agent_name
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 学習後に act() で使うフラグ（実際はチェックには使ってないが、状態として保持）
        self._trained: bool = False

    # ------------------------------------------------------------------
    # ユーティリティ
    # ------------------------------------------------------------------

    def _to_tensor(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        """numpy配列 or Tensor を、そのまま PyTorch Tensor へ変換して device に載せるヘルパー。"""
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        arr = np.asarray(x, dtype=np.float32)
        t = torch.from_numpy(arr)
        return t.to(self.device)

    # ------------------------------------------------------------------
    # PPO ロールアウト収集 & 更新
    # ------------------------------------------------------------------

    def _collect_rollout(self) -> Dict[str, np.ndarray]:
        """
        1エピソード分のロールアウトを収集し、GAE を計算して返す。

        Returns
        -------
        dict
            { "obs", "actions", "logprobs", "values", "advantages", "returns" } を含む。
        """
        obs_buf: list[np.ndarray] = []
        act_buf: list[float] = []
        logprob_buf: list[float] = []
        value_buf: list[float] = []
        reward_buf: list[float] = []
        done_buf: list[bool] = []

        env = self.env
        obs = env.reset(evaluation=False)

        # env.num_steps を上限に 1エピソード回す
        max_steps = getattr(env, "num_steps", None)
        if max_steps is None or max_steps <= 0:
            max_steps = self.rollout_horizon

        for _ in range(max_steps):
            obs_tensor = self._to_tensor(obs)
            with torch.no_grad():
                mu, log_std, value = self.model(obs_tensor)
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
            reward_buf.append(float(reward))
            done_buf.append(bool(done))

            obs = next_obs
            if done:
                break

        # 最終状態の価値を bootstrap 用に推定
        with torch.no_grad():
            last_obs_tensor = self._to_tensor(obs)
            _, _, last_value_tensor = self.model(last_obs_tensor)
            last_value = float(last_value_tensor.squeeze().cpu().numpy())

        # GAE-Lambda を計算
        rewards = np.asarray(reward_buf, dtype=np.float32)
        values = np.asarray(value_buf, dtype=np.float32)
        dones = np.asarray(done_buf, dtype=np.float32)

        T = len(rewards)
        values_ext = np.concatenate([values, np.array([last_value], dtype=np.float32)])

        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values_ext[t + 1] * mask - values_ext[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
        returns = advantages + values

        # 標準化
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        return {
            "obs": np.asarray(obs_buf, dtype=np.float32),
            "actions": np.asarray(act_buf, dtype=np.float32),
            "logprobs": np.asarray(logprob_buf, dtype=np.float32),
            "values": values,
            "advantages": advantages,
            "returns": returns,
        }

    def _ppo_update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        収集したロールアウトに対して PPO 更新を行い、平均損失などを返す。
        """
        obs = self._to_tensor(batch["obs"])
        actions = self._to_tensor(batch["actions"]).view(-1)  # shape (N,)
        old_logprobs = self._to_tensor(batch["logprobs"])
        advantages = self._to_tensor(batch["advantages"])
        returns = self._to_tensor(batch["returns"])

        num_samples = obs.shape[0]
        batch_indices = np.arange(num_samples)

        pi_loss_list: list[float] = []
        v_loss_list: list[float] = []
        entropy_list: list[float] = []

        for _ in range(self.num_epochs):
            np.random.shuffle(batch_indices)
            for start in range(0, num_samples, self.batch_size):
                end = start + self.batch_size
                mb_idx = batch_indices[start:end]
                mb_idx_t = torch.from_numpy(mb_idx).to(self.device)

                mb_obs = obs[mb_idx_t]
                mb_actions = actions[mb_idx_t]
                mb_old_logprobs = old_logprobs[mb_idx_t]
                mb_advantages = advantages[mb_idx_t]
                mb_returns = returns[mb_idx_t]

                mu, log_std, values = self.model(mb_obs)
                dist = torch.distributions.Normal(mu, log_std.exp())
                new_logprobs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # ratio = πθ(a|s) / πθ_old(a|s)
                ratio = torch.exp(new_logprobs - mb_old_logprobs)

                # PPO clip objective
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # value loss
                value_loss = nn.functional.mse_loss(values, mb_returns)

                # total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                pi_loss_list.append(float(policy_loss.detach().cpu().numpy()))
                v_loss_list.append(float(value_loss.detach().cpu().numpy()))
                entropy_list.append(float(entropy.detach().cpu().numpy()))

        stats = {
            "policy_loss": float(np.mean(pi_loss_list)) if pi_loss_list else 0.0,
            "value_loss": float(np.mean(v_loss_list)) if v_loss_list else 0.0,
            "entropy": float(np.mean(entropy_list)) if entropy_list else 0.0,
        }
        return stats

    # ------------------------------------------------------------------
    # 公開 API
    # ------------------------------------------------------------------

    def train(self) -> RLSingleTrainResult:
        """
        PPO で単体RLエージェントを学習し、RLSingleTrainResult を返す。
        """
        all_update_stats: list[Dict[str, float]] = []

        for update_idx in range(self.total_updates):
            batch = self._collect_rollout()
            stats = self._ppo_update(batch)
            all_update_stats.append(stats)

        self._trained = True

        # 学習後、評価モードで 1 エピソード回して日次ログを取得
        eval_log_df, eval_metrics = self._run_evaluation_episode()

        # ファイル出力
        artifacts: Dict[str, Path] = {}

        # ポリシー保存
        policy_path = self.output_dir / f"policy_stepE_{self.agent_name}.pt"
        torch.save(self.model.state_dict(), policy_path)
        artifacts["policy_path"] = policy_path

        # 日次ログ保存
        if eval_log_df is not None:
            daily_log_path = self.output_dir / f"stepE_daily_log_{self.agent_name}.csv"
            eval_log_df.to_csv(daily_log_path, index=False)
            artifacts["daily_log_path"] = daily_log_path

            # Equity カーブだけ抜き出したCSV
            if "Date" in eval_log_df.columns and "Equity" in eval_log_df.columns:
                equity_curve = eval_log_df[["Date", "Equity"]].copy()
                equity_curve_path = self.output_dir / f"stepE_equity_{self.agent_name}.csv"
                equity_curve.to_csv(equity_curve_path, index=False)
                artifacts["equity_curve_path"] = equity_curve_path

        # 学習過程の平均損失
        if all_update_stats:
            mean_policy_loss = float(np.mean([s["policy_loss"] for s in all_update_stats]))
            mean_value_loss = float(np.mean([s["value_loss"] for s in all_update_stats]))
            mean_entropy = float(np.mean([s["entropy"] for s in all_update_stats]))
        else:
            mean_policy_loss = mean_value_loss = mean_entropy = 0.0

        eval_metrics["mean_policy_loss"] = mean_policy_loss
        eval_metrics["mean_value_loss"] = mean_value_loss
        eval_metrics["mean_entropy"] = mean_entropy

        # メトリクスCSV
        import pandas as pd  # 遅延インポート
        metrics_path = self.output_dir / f"stepE_rl_metrics_{self.agent_name}.csv"
        pd.DataFrame([eval_metrics]).to_csv(metrics_path, index=False)
        artifacts["metrics_csv_path"] = metrics_path

        return RLSingleTrainResult(
            ok=True,
            message=f"StepE single training finished for agent={self.agent_name}",
            metrics=eval_metrics,
            artifacts=artifacts,
        )

    def _run_evaluation_episode(self):
        """
        学習済みポリシーで 1 エピソードを評価し、日次ログ DataFrame と
        最終メトリクスを返す。
        """
        import pandas as pd

        env = self.env
        obs = env.reset(evaluation=True)

        while True:
            obs_tensor = self._to_tensor(obs)
            with torch.no_grad():
                mu, log_std, value = self.model(obs_tensor)
                # 評価時は確率分布の平均値（deterministic action）を使う
                action_tensor = torch.tanh(mu)
            action = float(action_tensor.squeeze().cpu().numpy())
            next_obs, reward, done, info = env.step(action)
            obs = next_obs
            if done:
                break

        # 日次ログ取得
        log_df = env.get_episode_log_dataframe()
        if log_df is None or log_df.empty:
            metrics = {
                "final_return": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
            }
            return None, metrics

        # メトリクス計算
        equity = log_df["Equity"].astype(float).values
        if len(equity) == 0:
            final_return = 0.0
            max_drawdown = 0.0
            sharpe_ratio = 0.0
        else:
            initial_equity = equity[0]
            final_equity = equity[-1]
            final_return = float(final_equity / initial_equity - 1.0)

            # 最大ドローダウン
            peak = equity[0]
            dd_list = []
            for v in equity:
                peak = max(peak, v)
                dd = (v / peak) - 1.0
                dd_list.append(dd)
            max_drawdown = float(min(dd_list)) if dd_list else 0.0

            # 日次リターン系列から Sharpe 比
            ret_series = equity[1:] / equity[:-1] - 1.0
            if len(ret_series) > 1 and ret_series.std() > 1e-8:
                sharpe_ratio = float(ret_series.mean() / ret_series.std() * np.sqrt(252.0))
            else:
                sharpe_ratio = 0.0

        metrics = {
            "final_return": final_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
        }

        return log_df, metrics

    def act(self, obs: np.ndarray) -> float:
        """
        単体ポリシーからアクション（ポジション比率）を出す。

        Parameters
        ----------
        obs : np.ndarray
            観測ベクトル（1次元配列を想定）。

        Returns
        -------
        float
            アクション比率（-1.0 ～ +1.0 付近）。
        """
        obs_tensor = self._to_tensor(obs)
        with torch.no_grad():
            mu, log_std, value = self.model(obs_tensor)
            action_tensor = torch.tanh(mu)  # 決定的ポリシー
        action = float(action_tensor.squeeze().cpu().numpy())
        return float(np.clip(action, -1.0, 1.0))

    def load_policy(self, path: str | Path) -> None:
        """
        保存済みのポリシーをロードする。
        """
        state_dict = torch.load(str(path), map_location=self.device)
        self.model.load_state_dict(state_dict)
        self._trained = True
