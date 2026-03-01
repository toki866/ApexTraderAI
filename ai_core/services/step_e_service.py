# ai_core/services/step_e_service.py
# -*- coding: utf-8 -*-
"""
StepE Service (Independent RL agents)

This version is designed to run multiple StepE agents independently (10 agents etc.)
using StepD' embeddings as the primary observation stream.

What changed vs "pos=0" issue
----------------------------
Previously, if StepD' embeddings existed only for *test* dates (e.g. 62 rows),
then the merged dataframe during training had missing embeddings for most train dates.
Those were filled with zeros, and the baseline policy tended to output pos=0.

This service expects StepD' embeddings exported for "all" (train+test), e.g.:
  stepDprime_bnf_h01_SOXL_embeddings_all.csv

Then it trains an independent policy for each agent on the train split and evaluates on the test split.

Outputs
-------
output/stepE/<mode>/
  stepE_equity_<agent>_<SYMBOL>.csv          (test only)
  stepE_daily_log_<agent>_<SYMBOL>.csv       (train+test)
  stepE_summary_<agent>_<SYMBOL>.json
  models/stepE_<agent>_<SYMBOL>.pt

Notes
-----
- This is a lightweight "differentiable policy gradient" (diffPG) approach:
  it directly optimizes policy parameters to maximize cumulative log equity on the train split.
- Positions are continuous in [-1, +1] via tanh().
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
from torch import nn

from ai_core.utils.metrics_utils import compute_split_metrics


# ---------------------------
# Config
# ---------------------------

@dataclass
class StepEConfig:
    agent: str
    output_root: str = "output"
    obs_profile: str = "A"  # A/B/C/D
    seed: Optional[int] = 42
    verbose: bool = True

    # Trading
    trade_cost_bps: float = 5.0  # cost per unit position change (bps)
    pos_limit: float = 1.0

    # StepD' (Prime embeddings)
    use_stepd_prime: bool = False
    use_dprime_state: bool = False
    dprime_state_variant: str = ""  # bnf / all_features / mix
    dprime_profile: str = ""
    dprime_sources: str = ""      # e.g. "bnf" or "bnf,all_features"
    dprime_horizons: str = "1"    # e.g. "1" or "1,2,3"
    dprime_join: str = "concat"   # concat only (for now)

    # Policy training (diffPG)
    policy_kind: str = "diffpg"
    hidden_dim: int = 64
    lr: float = 1e-3
    epochs: int = 120
    patience: int = 15
    val_ratio: float = 0.2
    weight_decay: float = 1e-4
    pos_l2: float = 1e-3
    smooth_abs_eps: float = 1e-6
    device: str = "auto"

    # PPO training
    ppo_total_timesteps: int = 600_000
    ppo_n_steps: int = 2048
    ppo_batch_size: int = 256
    ppo_n_epochs: int = 20
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95
    ppo_ent_coef: float = 0.0
    ppo_clip_range: float = 0.2

    # Pair-trade mode (SOXL long when ratio>0, SOXS long when ratio<0)
    pair_trade: bool = True
    long_symbol: str = "SOXL"
    short_symbol: str = "SOXS"


# ---------------------------
# Policy (continuous position)
# ---------------------------

class DiffPolicyNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns mu (unbounded)
        return self.net(x).squeeze(-1)


# ---------------------------
# Service
# ---------------------------

class StepEService:
    _OBS_FORBIDDEN_EXACT = {
        "r_soxl_next",
        "r_soxs_next",
        "cc_ret_next",
        "reward_next",
        "ret",
        "equity",
        "gross",
        "cost",
        "market_ret",
        "abs_ratio",
        "underlying",
        "Position",
        "Action",
    }

    def __init__(self, app_config):
        """
        app_config is expected to have:
          - output_root (str)
          - stepE (list[StepEConfig])
        """
        self.app_config = app_config

    def run(self, date_range, symbol: str, agents: Optional[List[str]] = None, mode: Optional[str] = None):
        mode = str(mode or getattr(date_range, "mode", None) or "sim").strip().lower()
        if mode in {"ops", "prod", "production", "real"}:
            mode = "live"

        raw_cfgs = getattr(self.app_config, "stepE", None)
        if raw_cfgs is None:
            cfgs: List[StepEConfig] = []
        elif isinstance(raw_cfgs, (list, tuple)):
            cfgs = list(raw_cfgs)
        else:
            cfgs = [raw_cfgs]
        if agents:
            cfgs = [c for c in cfgs if c.agent in set(agents)]

        if not cfgs:
            print("[StepE] WARN: No StepE configs to run. Skipping StepE.")
            return {
                "skipped": True,
                "reason": "no_stepE_configs",
                "symbol": symbol,
                "mode": mode,
            }

        for cfg in cfgs:
            self._run_one(cfg, date_range=date_range, symbol=symbol, mode=mode)

        return {
            "skipped": False,
            "configs_run": len(cfgs),
            "symbol": symbol,
            "mode": mode,
        }

    # -----------------------
    # Main per-agent run
    # -----------------------

    def _run_one(self, cfg: StepEConfig, date_range, symbol: str, mode: str) -> None:
        out_root = Path(cfg.output_root or getattr(self.app_config, "output_root", "output"))
        out_dir = out_root / "stepE" / mode
        model_dir = out_dir / "models"
        out_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        cfg.seed = 42 if getattr(cfg, "seed", None) is None else int(cfg.seed)
        cfg.device = str(getattr(cfg, "device", "auto") or "auto")

        if cfg.verbose:
            print(f"[StepE] agent={cfg.agent} mode={mode} profile={cfg.obs_profile} use_stepd_prime={cfg.use_stepd_prime} seed={cfg.seed} device={cfg.device}")

        # Load & merge inputs (train+test)
        df_all, used_manifest = self._merge_inputs(cfg, out_root=out_root, mode=mode, symbol=symbol)

        # Split bounds
        train_start = pd.to_datetime(getattr(date_range, "train_start"))
        train_end = pd.to_datetime(getattr(date_range, "train_end"))
        test_start = pd.to_datetime(getattr(date_range, "test_start"))
        test_end = pd.to_datetime(getattr(date_range, "test_end"))

        df_all = df_all.sort_values("Date").reset_index(drop=True)
        df_train = df_all[(df_all["Date"] >= train_start) & (df_all["Date"] <= train_end)].copy()
        df_test = df_all[(df_all["Date"] >= test_start) & (df_all["Date"] <= test_end)].copy()

        if len(df_train) < 200:
            raise RuntimeError(f"Train split too short: {len(df_train)} rows. Check StepA split summary.")
        if len(df_test) < 10:
            raise RuntimeError(f"Test split too short: {len(df_test)} rows. Check StepA split summary.")

        # For dprime agents, ensure embeddings are not missing in train

        if cfg.use_stepd_prime:
            # A more reliable check: count non-zero on dprime columns (after fillna)
            dp = [c for c in df_train.columns if c.startswith("dprime_") and "_emb_" in c]
            if dp:
                nonzero_ratio = float((df_train[dp].abs().sum(axis=1) > 1e-9).mean())
                if nonzero_ratio < 0.5:
                    match_ratio = float(used_manifest.get("stepD_prime_match_ratio", 0.0))
                    if match_ratio < 0.10:
                        raise RuntimeError(
                            f"StepD' embeddings merge mismatch (match_ratio={match_ratio:.3f}). "
                            f"Generate embeddings with export-split all (stepDprime_*_embeddings_all.csv) first."
                        )
                    # If the dates merged fine but values are small/zero, don't hard-fail; keep running.
                    print(
                        f"[StepE] WARN StepD' embeddings are mostly zero in TRAIN (nonzero_ratio={nonzero_ratio:.3f}, "
                        f"match_ratio={match_ratio:.3f}). Continuing anyway."
                    )

        # Observation columns
        obs_cols = self._select_obs_columns(cfg.obs_profile, df_all=df_all)
        if cfg.use_stepd_prime:
            obs_cols = self._prepend_dprime_cols(df_all=df_all) + [c for c in obs_cols if not c.startswith("dprime_")]

        forbidden_hit = self._find_forbidden_obs_cols(obs_cols)
        if forbidden_hit:
            raise RuntimeError(f"[LEAK_GUARD] forbidden columns in obs: {forbidden_hit}")

        # Ensure columns exist
        obs_cols = [c for c in obs_cols if c in df_all.columns]
        if not obs_cols:
            raise RuntimeError("No observation columns selected.")
        if cfg.verbose:
            print(f"[StepE] obs_cols_count={len(obs_cols)}")
            print(f"[StepE] obs_cols(first5)={obs_cols[:5]}")
            print("[LEAK_GUARD] OK forbidden_hit=0")

        # Prepare tensors
        X_train, r_soxl_train, r_soxs_train, dates_train = self._build_obs_and_returns(df_train, obs_cols)
        X_test, r_soxl_test, r_soxs_test, dates_test = self._build_obs_and_returns(df_test, obs_cols)

        # Standardize based on train
        mu = X_train.mean(axis=0)
        sd = X_train.std(axis=0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        X_train_s = (X_train - mu) / sd
        X_test_s = (X_test - mu) / sd

        policy_kind = str(getattr(cfg, "policy_kind", "diffpg") or "diffpg").strip().lower()
        if policy_kind not in {"diffpg", "ppo"}:
            raise ValueError(f"Unsupported StepE policy_kind: {cfg.policy_kind}")
        if policy_kind == "ppo":
            self._train_and_eval_ppo(
                cfg=cfg,
                symbol=symbol,
                mode=mode,
                out_dir=out_dir,
                model_dir=model_dir,
                obs_cols=obs_cols,
                mu=mu,
                sd=sd,
                dates_train=dates_train,
                dates_test=dates_test,
                r_soxl_train=r_soxl_train,
                r_soxs_train=r_soxs_train,
                r_soxl_test=r_soxl_test,
                r_soxs_test=r_soxs_test,
                X_train_s=X_train_s,
                X_test_s=X_test_s,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                rows_train=len(df_train),
                rows_test=len(df_test),
            )
            return

        # Train policy (diffPG)
        device_name = str(cfg.device).strip().lower()
        if device_name in ("", "none", "auto"):
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)
        torch.manual_seed(int(cfg.seed))
        np.random.seed(int(cfg.seed))

        # include pos_prev as an extra input feature (stateful)
        X_train_t = torch.tensor(X_train_s, dtype=torch.float32, device=device)
        X_test_t = torch.tensor(X_test_s, dtype=torch.float32, device=device)
        r_soxl_train_t = torch.tensor(r_soxl_train, dtype=torch.float32, device=device)
        r_soxl_test_t = torch.tensor(r_soxl_test, dtype=torch.float32, device=device)
        r_soxs_train_t = torch.tensor(r_soxs_train, dtype=torch.float32, device=device)
        r_soxs_test_t = torch.tensor(r_soxs_test, dtype=torch.float32, device=device)

        net = DiffPolicyNet(in_dim=X_train_t.shape[1] + 1, hidden_dim=int(cfg.hidden_dim)).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

        best_val = float("inf")
        best_state = None
        wait = 0

        n = X_train_t.shape[0]
        n_val = max(10, int(cfg.val_ratio * n))
        n_fit = max(50, n - n_val)

        def _smooth_abs(x: torch.Tensor) -> torch.Tensor:
            return torch.sqrt(x * x + float(cfg.smooth_abs_eps))

        def _rollout(
            net_: DiffPolicyNet,
            X_: torch.Tensor,
            r_soxl_: torch.Tensor,
            r_soxs_: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Sequential rollout with differentiable pos_prev dependency.
            Returns:
              reward_next: (T,) daily net returns
              pos:     (T,) positions
              cost:    (T,) transaction cost component
            """
            T = X_.shape[0]
            pos_prev = torch.zeros((), device=X_.device)
            pos_list = []
            reward_list = []
            gross_list = []
            cost_list = []
            cost_k = float(cfg.trade_cost_bps) * 1e-4
            for t in range(T):
                xt = torch.cat([X_[t], pos_prev.unsqueeze(0)], dim=0)
                mu_t = net_(xt)
                pos_t = torch.tanh(mu_t) * float(cfg.pos_limit)
                gross = torch.relu(pos_t) * r_soxl_[t] + torch.relu(-pos_t) * r_soxs_[t]
                # transaction cost on position change
                cost = cost_k * _smooth_abs(pos_t - pos_prev)
                reward_next = gross - cost - float(cfg.pos_l2) * (pos_t * pos_t)
                pos_list.append(pos_t)
                reward_list.append(reward_next)
                gross_list.append(gross)
                cost_list.append(cost)
                pos_prev = pos_t
            pos = torch.stack(pos_list, dim=0)
            reward_next = torch.stack(reward_list, dim=0)
            gross = torch.stack(gross_list, dim=0)
            cost = torch.stack(cost_list, dim=0)
            return reward_next, pos, cost, gross

        for ep in range(1, int(cfg.epochs) + 1):
            net.train()
            opt.zero_grad()

            ret_fit, _, _, _ = _rollout(
                net,
                X_train_t[:n_fit],
                r_soxl_train_t[:n_fit],
                r_soxs_train_t[:n_fit],
            )
            # maximize log equity: sum log(1+ret)
            obj = -torch.log1p(ret_fit).mean()
            obj.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

            # validation
            net.eval()
            with torch.no_grad():
                ret_val, _, _, _ = _rollout(
                    net,
                    X_train_t[n_fit:],
                    r_soxl_train_t[n_fit:],
                    r_soxs_train_t[n_fit:],
                )
                val_loss = float((-torch.log1p(ret_val).mean()).item())

            if cfg.verbose:
                print(f"[StepE] ep={ep:03d} fit_loss={float(obj.item()):.6f} val_loss={val_loss:.6f}")

            if val_loss + 1e-8 < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= int(cfg.patience):
                    if cfg.verbose:
                        print(f"[StepE] early stop at ep={ep}")
                    break

        if best_state is not None:
            net.load_state_dict(best_state)

        # Evaluate on full train + test for daily log
        net.eval()
        with torch.no_grad():
            ret_tr_full, pos_tr_full, cost_tr_full, gross_tr_full = _rollout(net, X_train_t, r_soxl_train_t, r_soxs_train_t)
            ret_te_full, pos_te_full, cost_te_full, gross_te_full = _rollout(net, X_test_t, r_soxl_test_t, r_soxs_test_t)

        reward_next_arr = torch.cat([ret_tr_full, ret_te_full], dim=0).cpu().numpy().astype(float)
        pos_arr = torch.cat([pos_tr_full, pos_te_full], dim=0).cpu().numpy().astype(float)
        cost_arr = torch.cat([cost_tr_full, cost_te_full], dim=0).cpu().numpy().astype(float)
        gross_arr = torch.cat([gross_tr_full, gross_te_full], dim=0).cpu().numpy().astype(float)
        cc_soxl = np.concatenate([r_soxl_train, r_soxl_test]).astype(float)
        cc_soxs = np.concatenate([r_soxs_train, r_soxs_test]).astype(float)

        df_log = pd.DataFrame({
            "Date": pd.to_datetime(list(dates_train) + list(dates_test)),
            "Split": (["train"] * len(dates_train)) + (["test"] * len(dates_test)),
            "pos": pos_arr,
            "reward_next": reward_next_arr,
            "r_soxl_next": cc_soxl,
            "r_soxs_next": cc_soxs,
            "cost": cost_arr,
        })
        df_log["ratio"] = df_log["pos"]
        df_log["abs_ratio"] = df_log["ratio"].abs()
        df_log["market_ret"] = np.where(df_log["ratio"] > 1e-8, df_log["r_soxl_next"] * df_log["ratio"], np.where(df_log["ratio"] < -1e-8, df_log["r_soxs_next"] * (-df_log["ratio"]), 0.0))
        df_log["cc_ret_next"] = np.where(df_log["ratio"] > 1e-8, df_log["r_soxl_next"], np.where(df_log["ratio"] < -1e-8, df_log["r_soxs_next"], 0.0))
        df_log["underlying"] = np.where(df_log["ratio"] > 1e-8, "SOXL", np.where(df_log["ratio"] < -1e-8, "SOXS", "NONE"))
        df_log["gross"] = gross_arr
        df_log["ret"] = df_log["reward_next"].shift(1).fillna(0.0)
        df_log["equity"] = (1.0 + df_log["ret"]).cumprod()

        # Also provide Action/Position columns for StepF compatibility
        df_log["Position"] = df_log["pos"]
        df_log["Action"] = np.where(df_log["pos"] > 0.15, 1, np.where(df_log["pos"] < -0.15, -1, 0))

        # Test-only equity file (kept small)
        df_eq = df_log[df_log["Split"] == "test"][["Date", "pos", "ret", "equity", "reward_next"]].copy().reset_index(drop=True)
        # Reset equity at the start of the test window (initial capital = 1.0)
        if len(df_eq) > 0:
            df_eq["equity"] = (1.0 + df_eq["ret"].astype(float)).cumprod()

        eq_path = out_dir / f"stepE_equity_{cfg.agent}_{symbol}.csv"
        df_eq.to_csv(eq_path, index=False)

        log_path = out_dir / f"stepE_daily_log_{cfg.agent}_{symbol}.csv"
        df_log.to_csv(log_path, index=False)

        # Summary metrics (test) using metrics_summary.csv-aligned definitions.
        # Read back from saved daily log to guarantee summary uses persisted data.
        df_log_for_summary = pd.read_csv(log_path)
        metrics = compute_split_metrics(df_log_for_summary, split="test", equity_col="equity", ret_col="ret")
        legacy_metrics = {
            "test_return_pct": metrics["total_return_pct"],
            "test_sharpe": metrics["sharpe"],
            "test_max_dd": metrics["max_dd_pct"],
            "total_return": metrics["total_return_pct"] / 100.0 if np.isfinite(metrics["total_return_pct"]) else float("nan"),
            "cagr": metrics["cagr_pct"] / 100.0 if np.isfinite(metrics["cagr_pct"]) else float("nan"),
            "max_drawdown": metrics["max_dd_pct"],
        }

        summary = {
            "agent": cfg.agent,
            "mode": mode,
            "symbol": symbol,
            "train_start": str(train_start.date()),
            "train_end": str(train_end.date()),
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "rows_train": int(len(df_train)),
            "rows_test": int(len(df_test)),
            **metrics,
            **legacy_metrics,
        }

        summ_path = out_dir / f"stepE_summary_{cfg.agent}_{symbol}.json"
        summ_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        # Save model + scaler
        mdl_path = model_dir / f"stepE_{cfg.agent}_{symbol}.pt"
        torch.save({
            "cfg": asdict(cfg),
            "obs_cols": obs_cols,
            "mu": mu.astype(np.float32),
            "sd": sd.astype(np.float32),
            "state_dict": net.state_dict(),
        }, mdl_path)

        if cfg.verbose:
            print(f"[StepE] wrote equity={eq_path}")
            print(f"[StepE] wrote daily_log={log_path}")
            print(f"[StepE] wrote summary={summ_path}")
            print(f"[StepE] wrote model={mdl_path}")

    def _train_and_eval_ppo(
        self,
        *,
        cfg: StepEConfig,
        symbol: str,
        mode: str,
        out_dir: Path,
        model_dir: Path,
        obs_cols: List[str],
        mu: np.ndarray,
        sd: np.ndarray,
        dates_train: np.ndarray,
        dates_test: np.ndarray,
        r_soxl_train: np.ndarray,
        r_soxs_train: np.ndarray,
        r_soxl_test: np.ndarray,
        r_soxs_test: np.ndarray,
        X_train_s: np.ndarray,
        X_test_s: np.ndarray,
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
        test_start: pd.Timestamp,
        test_end: pd.Timestamp,
        rows_train: int,
        rows_test: int,
    ) -> None:
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv
        except Exception as e:
            raise RuntimeError("PPO policy_kind requires stable-baselines3 to be installed") from e

        from ai_core.rl.daily_pair_trading_env import DailyPairTradingEnv

        seed = int(cfg.seed or 42)
        np.random.seed(seed)
        torch.manual_seed(seed)

        train_env = DailyPairTradingEnv(
            X=X_train_s,
            r_soxl_next=r_soxl_train,
            r_soxs_next=r_soxs_train,
            trade_cost_bps=float(cfg.trade_cost_bps),
            pos_limit=float(cfg.pos_limit),
            pos_l2=float(cfg.pos_l2),
        )
        test_env = DailyPairTradingEnv(
            X=X_test_s,
            r_soxl_next=r_soxl_test,
            r_soxs_next=r_soxs_test,
            trade_cost_bps=float(cfg.trade_cost_bps),
            pos_limit=float(cfg.pos_limit),
            pos_l2=float(cfg.pos_l2),
        )

        vec_env = DummyVecEnv([lambda: train_env])
        vec_env.seed(seed)

        device_name = str(cfg.device).strip().lower()
        if device_name in ("", "none", "auto"):
            device_name = "cuda" if torch.cuda.is_available() else "cpu"

        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=float(cfg.lr),
            n_steps=int(cfg.ppo_n_steps),
            batch_size=int(cfg.ppo_batch_size),
            n_epochs=int(cfg.ppo_n_epochs),
            gamma=float(cfg.ppo_gamma),
            gae_lambda=float(cfg.ppo_gae_lambda),
            ent_coef=float(cfg.ppo_ent_coef),
            clip_range=float(cfg.ppo_clip_range),
            verbose=1 if cfg.verbose else 0,
            seed=seed,
            device=device_name,
        )
        model.learn(total_timesteps=int(cfg.ppo_total_timesteps))

        def _rollout_env(env: DailyPairTradingEnv):
            obs, _ = env.reset(seed=seed)
            rewards, pos, costs, gross = [], [], [], []
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                action = np.asarray(action, dtype=np.float32).reshape(-1)
                obs, reward, terminated, truncated, info = env.step(action)
                rewards.append(float(reward))
                pos.append(float(info.get("pos", 0.0)))
                costs.append(float(info.get("cost", 0.0)))
                gross.append(float(info.get("gross", 0.0)))
                done = bool(terminated or truncated)
            return np.asarray(rewards), np.asarray(pos), np.asarray(costs), np.asarray(gross)

        ret_tr_full, pos_tr_full, cost_tr_full, gross_tr_full = _rollout_env(train_env)
        ret_te_full, pos_te_full, cost_te_full, gross_te_full = _rollout_env(test_env)

        reward_next_arr = np.concatenate([ret_tr_full, ret_te_full]).astype(float)
        pos_arr = np.concatenate([pos_tr_full, pos_te_full]).astype(float)
        cost_arr = np.concatenate([cost_tr_full, cost_te_full]).astype(float)
        gross_arr = np.concatenate([gross_tr_full, gross_te_full]).astype(float)
        cc_soxl = np.concatenate([r_soxl_train, r_soxl_test]).astype(float)
        cc_soxs = np.concatenate([r_soxs_train, r_soxs_test]).astype(float)

        df_log = pd.DataFrame({
            "Date": pd.to_datetime(list(dates_train) + list(dates_test)),
            "Split": (["train"] * len(dates_train)) + (["test"] * len(dates_test)),
            "pos": pos_arr,
            "reward_next": reward_next_arr,
            "r_soxl_next": cc_soxl,
            "r_soxs_next": cc_soxs,
            "cost": cost_arr,
        })
        df_log["ratio"] = df_log["pos"]
        df_log["abs_ratio"] = df_log["ratio"].abs()
        df_log["market_ret"] = np.where(df_log["ratio"] > 1e-8, df_log["r_soxl_next"] * df_log["ratio"], np.where(df_log["ratio"] < -1e-8, df_log["r_soxs_next"] * (-df_log["ratio"]), 0.0))
        df_log["cc_ret_next"] = np.where(df_log["ratio"] > 1e-8, df_log["r_soxl_next"], np.where(df_log["ratio"] < -1e-8, df_log["r_soxs_next"], 0.0))
        df_log["underlying"] = np.where(df_log["ratio"] > 1e-8, "SOXL", np.where(df_log["ratio"] < -1e-8, "SOXS", "NONE"))
        df_log["gross"] = gross_arr
        df_log["ret"] = df_log["reward_next"].shift(1).fillna(0.0)
        df_log["equity"] = (1.0 + df_log["ret"]).cumprod()
        df_log["Position"] = df_log["pos"]
        df_log["Action"] = np.where(df_log["pos"] > 0.15, 1, np.where(df_log["pos"] < -0.15, -1, 0))

        df_eq = df_log[df_log["Split"] == "test"][["Date", "pos", "ret", "equity", "reward_next"]].copy().reset_index(drop=True)
        if len(df_eq) > 0:
            df_eq["equity"] = (1.0 + df_eq["ret"].astype(float)).cumprod()

        eq_path = out_dir / f"stepE_equity_{cfg.agent}_{symbol}.csv"
        log_path = out_dir / f"stepE_daily_log_{cfg.agent}_{symbol}.csv"
        summ_path = out_dir / f"stepE_summary_{cfg.agent}_{symbol}.json"
        mdl_zip_path = model_dir / f"stepE_{cfg.agent}_{symbol}_ppo.zip"
        mdl_pt_path = model_dir / f"stepE_{cfg.agent}_{symbol}.pt"

        df_eq.to_csv(eq_path, index=False)
        df_log.to_csv(log_path, index=False)

        df_log_for_summary = pd.read_csv(log_path)
        metrics = compute_split_metrics(df_log_for_summary, split="test", equity_col="equity", ret_col="ret")
        legacy_metrics = {
            "test_return_pct": metrics["total_return_pct"],
            "test_sharpe": metrics["sharpe"],
            "test_max_dd": metrics["max_dd_pct"],
            "total_return": metrics["total_return_pct"] / 100.0 if np.isfinite(metrics["total_return_pct"]) else float("nan"),
            "cagr": metrics["cagr_pct"] / 100.0 if np.isfinite(metrics["cagr_pct"]) else float("nan"),
            "max_drawdown": metrics["max_dd_pct"],
        }
        summary = {
            "agent": cfg.agent,
            "mode": mode,
            "symbol": symbol,
            "train_start": str(train_start.date()),
            "train_end": str(train_end.date()),
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "rows_train": int(rows_train),
            "rows_test": int(rows_test),
            **metrics,
            **legacy_metrics,
        }
        summ_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        model.save(str(mdl_zip_path))
        torch.save({
            "cfg": asdict(cfg),
            "obs_cols": obs_cols,
            "mu": mu.astype(np.float32),
            "sd": sd.astype(np.float32),
            "policy_kind": "ppo",
            "sb3_model_path": str(mdl_zip_path),
        }, mdl_pt_path)

        if cfg.verbose:
            print(f"[StepE] wrote equity={eq_path}")
            print(f"[StepE] wrote daily_log={log_path}")
            print(f"[StepE] wrote summary={summ_path}")
            print(f"[StepE] wrote model(zip)={mdl_zip_path}")
            print(f"[StepE] wrote model(pt)={mdl_pt_path}")

    # -----------------------
    # Load & merge
    # -----------------------

    def _merge_inputs(self, cfg: StepEConfig, out_root: Path, mode: str, symbol: str) -> tuple[pd.DataFrame, dict[str, object]]:
        df_prices = self._load_stepA_prices(out_root, mode, symbol)

        if cfg.use_dprime_state:
            df_state = self._load_stepD_prime_state(cfg, out_root=out_root, mode=mode, symbol=symbol)
            df = df_state.merge(
                df_prices[["Date", "Open", "High", "Low", "Close", "Volume", "price_exec"]],
                on="Date",
                how="left",
            )
        else:
            df = df_prices.copy()

        used_manifest: dict[str, object] = {}

        if not cfg.use_dprime_state:
            # merge periodic and tech if available
            for stem in ["stepA_periodic", "stepA_tech"]:
                p_train = out_root / "stepA" / mode / f"{stem}_train_{symbol}.csv"
                p_test = out_root / "stepA" / mode / f"{stem}_test_{symbol}.csv"
                if p_train.exists() and p_test.exists():
                    dft = pd.concat([pd.read_csv(p_train), pd.read_csv(p_test)], axis=0, ignore_index=True)
                    dft["Date"] = pd.to_datetime(dft["Date"], errors="coerce")
                    df = df.merge(dft, on="Date", how="left")

        # Ensure core features exist (Gap, ATR_norm, oc_ret)
        df = self._ensure_core_features(df)

        # StepD' embeddings
        if cfg.use_stepd_prime:
            self._ensure_dprime_sources(cfg)
            try:
                dprime_df = self._load_stepD_prime_embeddings(cfg, out_root=out_root, mode=mode, symbol=symbol)
            except Exception as e:
                print(f"[StepE] WARN: StepD' embeddings unavailable ({e}). Falling back to use_stepd_prime=False and continue.")
                cfg.use_stepd_prime = False
                dprime_df = None

            if dprime_df is not None:
                # Normalize dates defensively (time-of-day differences can cause a full join miss)
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
                dprime_df["Date"] = pd.to_datetime(dprime_df["Date"], errors="coerce").dt.normalize()

                dprime_dates = dprime_df["Date"].dropna().unique()
                match_ratio = float(df["Date"].isin(dprime_dates).mean()) if len(dprime_dates) > 0 else 0.0
                used_manifest["stepD_prime_match_ratio"] = match_ratio
                if match_ratio < 0.10:
                    # If this happens, all embeddings will be NaN after merge then become zeros.
                    df_min, df_max = df["Date"].min(), df["Date"].max()
                    dp_min, dp_max = dprime_df["Date"].min(), dprime_df["Date"].max()
                    raise RuntimeError(
                        "StepD' merge mismatch (match_ratio={:.3f}). "
                        "df Date range={}..{}, dprime Date range={}..{}. "
                        "Check the embeddings file Date column/format.".format(match_ratio, df_min, df_max, dp_min, dp_max)
                    )

                df = df.merge(dprime_df, on="Date", how="left")
                # missing embeddings -> 0 (warmup rows are expected to be missing)
                dp_cols = [c for c in df.columns if c.startswith("dprime_") and "_emb_" in c]
                if dp_cols:
                    df[dp_cols] = df[dp_cols].fillna(0.0)

        # Pair-trade daily next returns (D->D+1 confirmed, no leakage)
        pair_trade_enabled = bool(cfg.pair_trade) and symbol.upper() in {str(cfg.long_symbol).upper(), str(cfg.short_symbol).upper(), "SOXL", "SOXS"}
        if pair_trade_enabled:
            soxl_px = self._load_stepA_price_exec(out_root=out_root, mode=mode, symbol=str(cfg.long_symbol or "SOXL"))
            soxs_px = self._load_stepA_price_exec(out_root=out_root, mode=mode, symbol=str(cfg.short_symbol or "SOXS"))
            if soxl_px.empty or soxs_px.empty:
                print("[StepE] WARN: Pair-trade requested but SOXL/SOXS prices are missing. Fallback to single-symbol mode.")
                pair_trade_enabled = False
            else:
                pair = soxl_px.rename(columns={"price_exec": "price_soxl"}).merge(
                    soxs_px.rename(columns={"price_exec": "price_soxs"}),
                    on="Date",
                    how="inner",
                ).sort_values("Date").reset_index(drop=True)
                pair["r_soxl_next"] = (pair["price_soxl"].shift(-1) / pair["price_soxl"] - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                pair["r_soxs_next"] = (pair["price_soxs"].shift(-1) / pair["price_soxs"] - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                df = df.merge(pair[["Date", "r_soxl_next", "r_soxs_next"]], on="Date", how="left")
                df[["r_soxl_next", "r_soxs_next"]] = df[["r_soxl_next", "r_soxs_next"]].fillna(0.0)

        if not pair_trade_enabled:
            px = pd.to_numeric(df.get("price_exec", np.nan), errors="coerce")
            r_base = (px.shift(-1) / px - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            df["r_soxl_next"] = r_base.astype(float)
            df["r_soxs_next"] = r_base.astype(float)

        # backward-compatible baseline column name
        df["cc_ret_next"] = pd.to_numeric(df["r_soxl_next"], errors="coerce").fillna(0.0)

        df = df.sort_values("Date").reset_index(drop=True)
        return df, used_manifest

    def _load_stepA_prices(self, out_root: Path, mode: str, symbol: str) -> pd.DataFrame:
        df = self._load_symbol_prices(out_root=out_root, mode=mode, symbol=symbol, required=True)
        return df.sort_values("Date").reset_index(drop=True)

    def _load_stepA_price_exec(self, out_root: Path, mode: str, symbol: str) -> pd.DataFrame:
        df = self._load_symbol_prices(out_root=out_root, mode=mode, symbol=symbol, required=False)
        if df.empty:
            return pd.DataFrame(columns=["Date", "price_exec"])
        out = df[["Date", "price_exec"]].copy()
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.normalize()
        out["price_exec"] = pd.to_numeric(out["price_exec"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        return out.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)

    def _price_col_name(self, df: pd.DataFrame) -> str:
        for c in ("P_eff", "Close_eff", "price_eff", "close_eff", "Close"):
            if c in df.columns:
                return c
        raise KeyError("No usable price column found. Expected one of P_eff/Close_eff/Close.")

    def _load_symbol_prices(self, out_root: Path, mode: str, symbol: str, required: bool = True) -> pd.DataFrame:
        base = out_root / "stepA" / mode
        p_tr = base / f"stepA_prices_train_{symbol}.csv"
        p_te = base / f"stepA_prices_test_{symbol}.csv"
        frames = []
        if p_tr.exists() and p_te.exists():
            frames = [pd.read_csv(p_tr), pd.read_csv(p_te)]
        else:
            roots = []
            for attr in ("data_dir", "data_root"):
                v = getattr(self.app_config, attr, None)
                if v:
                    roots.append(Path(v))
            for root in roots:
                cand = root / f"prices_{symbol}.csv"
                if cand.exists():
                    frames = [pd.read_csv(cand)]
                    break
        if not frames:
            if required:
                raise FileNotFoundError(f"Missing prices for {symbol} in StepA output and app_config data roots")
            return pd.DataFrame(columns=["Date", "price_exec"])

        df = pd.concat(frames, axis=0, ignore_index=True)
        if "Date" not in df.columns:
            raise KeyError(f"prices_{symbol}.csv missing Date column")
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
        pcol = self._price_col_name(df)
        df["price_exec"] = pd.to_numeric(df[pcol], errors="coerce").replace([np.inf, -np.inf], np.nan)
        return df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)

    def _load_stepD_prime_state(self, cfg: StepEConfig, out_root: Path, mode: str, symbol: str) -> pd.DataFrame:
        profile = str(getattr(cfg, "dprime_profile", "") or "").strip()
        variant = str(cfg.dprime_state_variant or "").strip()
        if not profile and variant:
            # compatibility mapping for old variant-only configs
            profile = f"dprime_{variant}_h02"
        if not profile:
            raise ValueError("dprime_profile is empty while use_dprime_state=True")

        p_tr = out_root / "stepDprime" / mode / f"stepDprime_state_train_{profile}_{symbol}.csv"
        p_te = out_root / "stepDprime" / mode / f"stepDprime_state_test_{profile}_{symbol}.csv"
        if not (p_tr.exists() and p_te.exists()):
            legacy_base = out_root / "stepD_prime" / mode
            p_tr = legacy_base / f"stepDprime_state_{profile}_{symbol}_train.csv"
            p_te = legacy_base / f"stepDprime_state_{profile}_{symbol}_test.csv"
        if not (p_tr.exists() and p_te.exists()):
            raise FileNotFoundError(f"Missing StepD' state CSVs: {p_tr} / {p_te}")

        df = pd.concat([pd.read_csv(p_tr), pd.read_csv(p_te)], axis=0, ignore_index=True)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
        return df.sort_values("Date").reset_index(drop=True)

    def _ensure_core_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        # Gap
        if "Gap" not in out.columns and "gap" not in out.columns:
            close_prev = out["Close"].astype(float).shift(1)
            out["Gap"] = out["Open"].astype(float) / close_prev.replace(0, np.nan) - 1.0
        elif "gap" in out.columns and "Gap" not in out.columns:
            out["Gap"] = out["gap"]

        # ATR_norm
        if "ATR_norm" not in out.columns and "atr_norm" not in out.columns and "atr_norm_14" not in out.columns:
            # simple ATR_norm(14)
            high = out["High"].astype(float)
            low = out["Low"].astype(float)
            close_prev = out["Close"].astype(float).shift(1)
            tr = pd.concat([(high - low), (high - close_prev).abs(), (low - close_prev).abs()], axis=1).max(axis=1)
            atr = tr.rolling(14, min_periods=14).mean()
            out["ATR_norm"] = (atr / close_prev.replace(0, np.nan))
        elif "atr_norm_14" in out.columns and "ATR_norm" not in out.columns:
            out["ATR_norm"] = out["atr_norm_14"]
        elif "atr_norm" in out.columns and "ATR_norm" not in out.columns:
            out["ATR_norm"] = out["atr_norm"]

        # Open->Close ret
        if "oc_ret" not in out.columns:
            out["oc_ret"] = out["Close"].astype(float) / out["Open"].astype(float).replace(0, np.nan) - 1.0

        # sanitize
        for c in ["Gap", "ATR_norm", "oc_ret"]:
            out[c] = out[c].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

        return out

    def _load_stepD_prime_embeddings(self, cfg: StepEConfig, out_root: Path, mode: str, symbol: str) -> pd.DataFrame:
        """
        Load embeddings with profile-first resolution. If profile is missing, fallback to
        source/horizon style for backward compatibility.
        """
        base = out_root / "stepD_prime" / mode / "embeddings"

        profile = str(getattr(cfg, "dprime_profile", "") or "").strip()
        merged: Optional[pd.DataFrame] = None

        def _read_one(path: Path, prefix: str) -> pd.DataFrame:
            df = pd.read_csv(path)
            date_col = next((c for c in df.columns if str(c).strip().lower() == "date"), df.columns[0])
            if date_col != "Date":
                df = df.rename(columns={date_col: "Date"})
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
            emb_cols = sorted([c for c in df.columns if str(c).startswith("emb_")])
            if not emb_cols:
                raise RuntimeError(f"{path} has no emb_* columns")
            sub = df[["Date"] + emb_cols].copy()
            sub = sub.rename(columns={c: f"dprime_{prefix}_emb_{i:03d}" for i, c in enumerate(emb_cols)})
            return sub

        if profile:
            p_all = base / f"stepDprime_{profile}_{symbol}_embeddings_all.csv"
            if p_all.exists():
                return _read_one(p_all, profile).sort_values("Date").reset_index(drop=True)
            p_tr = base / f"stepDprime_{profile}_{symbol}_embeddings_train.csv"
            p_te = base / f"stepDprime_{profile}_{symbol}_embeddings_test.csv"
            if p_tr.exists() and p_te.exists():
                d0 = _read_one(p_tr, profile)
                d1 = _read_one(p_te, profile)
                return pd.concat([d0, d1], axis=0, ignore_index=True).sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)

        sources = [s.strip() for s in (cfg.dprime_sources or "").split(",") if s.strip()]
        if not sources:
            inferred = self._infer_dprime_source_from_agent(str(getattr(cfg, "agent", "") or ""))
            if inferred:
                sources = [inferred]

        horizon_tokens = [str(x).strip() for x in (cfg.dprime_horizons or "1").split(",") if str(x).strip()]
        horizons = [self._normalize_horizon_token(tok) for tok in horizon_tokens]
        if not sources:
            raise ValueError(
                "dprime_profile/dprime_sources are empty while use_stepd_prime=True, "
                "and source inference from agent name failed"
            )

        for src in sources:
            for hh in horizons:
                cands = [
                    base / f"stepDprime_{src}_{hh}_{symbol}_embeddings_all.csv",
                    base / f"stepDprime_{src}_{hh}_{symbol}_embeddings.csv",
                    base / f"stepDprime_{src}_{hh}_{symbol}_embeddings_test.csv",
                    base / f"stepDprime_{src}_{hh}_{symbol}_embeddings_train.csv",
                ]
                p = next((q for q in cands if q.exists()), None)
                if p is None:
                    raise FileNotFoundError(f"Missing StepD' embeddings: expected one of {cands[0].name} ... under {base}")
                sub = _read_one(p, f"{src}_{hh}")
                merged = sub if merged is None else merged.merge(sub, on="Date", how="outer")

        assert merged is not None
        merged = merged.sort_values("Date").reset_index(drop=True)
        return merged

    def _normalize_horizon_token(self, token: str) -> str:
        t = str(token or "").strip().lower()
        if not t:
            return "h01"
        if t.isdigit():
            return f"h{int(t):02d}"
        if t.startswith("h") and t[1:].isdigit():
            return f"h{int(t[1:]):02d}"
        if t == "3scale":
            return "3scale"
        raise ValueError(f"invalid dprime_horizons token: {token}")

    def _infer_dprime_source_from_agent(self, agent: str) -> str:
        a = str(agent or "").lower()
        if "all_features" in a:
            return "all_features"
        if "mix" in a:
            return "mix"
        if "bnf" in a:
            return "bnf"
        return ""

    # -----------------------
    # Observation & returns
    # -----------------------

    def _prepend_dprime_cols(self, df_all: pd.DataFrame) -> List[str]:
        cols = [c for c in df_all.columns if c.startswith("dprime_") and "_emb_" in c]
        return sorted(cols)

    def _select_obs_columns(self, profile: str, df_all: pd.DataFrame) -> List[str]:
        p = str(profile).upper().strip()

        dprime_cols = self._prepend_dprime_cols(df_all)

        base = dprime_cols + ["Gap", "ATR_norm"]

        if p == "A":
            return base

        # Add scale-free price stats
        extra_price = []
        for c in ["oc_ret", "ret_1", "ret_5", "ret_20", "body_ratio", "range_atr", "vol_log_ratio_20"]:
            if c in df_all.columns:
                extra_price.append(c)

        if p == "B":
            return base + extra_price

        # Add some tech and volume signals if present
        extra = []
        for c in ["RSI", "RSI_14", "MACD_hist", "dev_z_25", "bnf_score", "vol_z_20"]:
            if c in df_all.columns:
                extra.append(c)

        if p == "C":
            return base + extra_price + extra

        # D = "all in" (excluding raw Close scale)
        all_num = [c for c in df_all.columns if c not in {"Date"} and pd.api.types.is_numeric_dtype(df_all[c])]
        # Leak guard: drop label/target-like columns if they exist
        bad_keys = ("label", "available", "target", "y_")
        all_num = [c for c in all_num if not any(k in str(c).lower() for k in bad_keys)]
        all_num = [c for c in all_num if not self._is_forbidden_obs_column(c)]
        # Avoid raw scale columns (Open/High/Low/Close/Volume)
        drop = {"Open", "High", "Low", "Close", "Volume"}
        all_num = [c for c in all_num if c not in drop]
        # Put dprime first
        out = dprime_cols + [c for c in all_num if c not in dprime_cols]
        return out

    def _is_forbidden_obs_column(self, col: str) -> bool:
        c = str(col)
        cl = c.lower()
        if c in self._OBS_FORBIDDEN_EXACT:
            return True
        return "_next" in cl or cl.endswith("_next")

    def _find_forbidden_obs_cols(self, cols: List[str]) -> List[str]:
        return [c for c in cols if self._is_forbidden_obs_column(c)]

    def _build_obs_and_returns(self, df: pd.DataFrame, obs_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        df2 = df.copy()
        df2 = df2.sort_values("Date").reset_index(drop=True)

        X = df2[obs_cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        r_soxl = pd.to_numeric(df2.get("r_soxl_next", np.nan), errors="coerce")
        r_soxs = pd.to_numeric(df2.get("r_soxs_next", np.nan), errors="coerce")
        r_soxl = r_soxl.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        r_soxs = r_soxs.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        dates = df2["Date"].to_numpy()
        return X, r_soxl, r_soxs, dates

    def _ensure_dprime_sources(self, cfg: StepEConfig) -> None:
        if not bool(getattr(cfg, "use_stepd_prime", False)):
            return
        if str(getattr(cfg, "dprime_sources", "") or "").strip():
            return
        agent = str(getattr(cfg, "agent", "") or "").lower()
        inferred = "mamba"
        if "bnf" in agent:
            inferred = "bnf"
        elif "mix" in agent:
            inferred = "mix"
        elif "all_features" in agent:
            inferred = "all_features"
        cfg.dprime_sources = inferred
        print(f"[StepE] WARN: dprime_sources was empty with use_stepd_prime=True. Inferred dprime_sources='{inferred}' from agent.")
