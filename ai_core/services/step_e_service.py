# ai_core/services/step_e_service.py
# -*- coding: utf-8 -*-
"""
StepE Service (Independent PPO agents)

This service runs multiple StepE agents independently while keeping StepE's role
limited to expert evaluation / candidate generation. Final candidate selection and
integration remain StepF responsibility.

Policy support
--------------
- StepE is PPO-only.
- PPO training failures are treated as StepE failures; there is no automatic
  fallback to another policy implementation.

StepD' embedding expectation
----------------------------
If StepD' embeddings exist only for *test* dates, most train rows merge with
missing embeddings and the policy can degenerate toward neutral positions.
This service therefore expects StepD' embeddings exported for the full
train+test span, for example:
  stepDprime_bnf_h01_SOXL_embeddings_all.csv

Outputs
-------
output/stepE/<mode>/
  stepE_equity_<agent>_<SYMBOL>.csv          (test only)
  stepE_daily_log_<agent>_<SYMBOL>.csv       (train+test)
  stepE_summary_<agent>_<SYMBOL>.json
  models/stepE_<agent>_<SYMBOL>.pt

Notes
-----
- PPO is the only supported policy path for current operation.
- Positions are continuous in [-1, +1] via tanh().
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch

from ai_core.utils.metrics_utils import compute_split_metrics
from ai_core.utils.timing_logger import TimingLogger
from ai_core.utils.pipeline_artifact_utils import resolve_stepdprime_root


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

    # PPO training
    policy_kind: str = "ppo"
    lr: float = 3e-4
    ppo_lr: Optional[float] = None
    device: str = "auto"
    pos_l2: float = 1e-3
    ppo_total_timesteps: int = 80_000
    ppo_n_steps: int = 2048
    ppo_batch_size: int = 512
    ppo_n_epochs: int = 2
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95
    ppo_ent_coef: float = 0.0
    ppo_clip_range: float = 0.2
    max_parallel_agents: int = 1

    # Pair-trade mode (SOXL long when ratio>0, SOXS long when ratio<0)
    pair_trade: bool = True
    long_symbol: str = "SOXL"
    short_symbol: str = "SOXS"

    def __post_init__(self) -> None:
        self.policy_kind = str(self.policy_kind or "ppo").strip().lower()
        if self.policy_kind != "ppo":
            raise ValueError(f"Unsupported StepE policy_kind: {self.policy_kind}. StepE is PPO-only.")
        self.device = str(self.device or "auto")
        self.max_parallel_agents = max(1, int(self.max_parallel_agents or 1))


def _training_config_summary(cfg: StepEConfig, *, device: str) -> Dict[str, object]:
    ppo_lr = float(getattr(cfg, "ppo_lr", None) or getattr(cfg, "lr", 3e-4) or 3e-4)
    return {
        "policy_kind": "ppo",
        "policy_role": "only_supported_policy",
        "ppo_lr": ppo_lr,
        "ppo_total_timesteps": int(getattr(cfg, "ppo_total_timesteps", 0) or 0),
        "ppo_n_epochs": int(getattr(cfg, "ppo_n_epochs", 0) or 0),
        "ppo_n_steps": int(getattr(cfg, "ppo_n_steps", 0) or 0),
        "ppo_batch_size": int(getattr(cfg, "ppo_batch_size", 0) or 0),
        "max_parallel_agents": int(getattr(cfg, "max_parallel_agents", 1) or 1),
        "device": str(device),
    }


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
        self._merge_cache: Dict[Tuple[object, ...], tuple[pd.DataFrame, dict[str, object], dict[str, object]]] = {}
        self._merge_cache_inflight: Dict[Tuple[object, ...], threading.Event] = {}
        self._merge_cache_errors: Dict[Tuple[object, ...], BaseException] = {}
        self._merge_cache_lock = threading.Lock()

    def _timing(self) -> TimingLogger:
        t = getattr(self.app_config, "_timing_logger", None)
        return t if isinstance(t, TimingLogger) else TimingLogger.disabled()

    @staticmethod
    def _resolve_device(requested: str) -> tuple[str, List[str]]:
        req = str(requested or "auto").strip().lower()
        warnings: List[str] = []
        if req in ("", "none", "auto"):
            actual = "cuda" if torch.cuda.is_available() else "cpu"
            if actual == "cpu":
                warnings.append("auto_device_resolved_to_cpu")
            return actual, warnings
        if req.startswith("cuda") and not torch.cuda.is_available():
            warnings.append("cuda_requested_but_unavailable_fallback_cpu")
            return "cpu", warnings
        return req, warnings

    @staticmethod
    def _device_payload(*, requested: str, actual: str, model_loaded: bool = False) -> Dict[str, object]:
        return {
            "device_requested": str(requested),
            "device_execution": str(actual),
            "device_model_loaded": bool(model_loaded),
        }

    @staticmethod
    def _input_feature_summary(obs_cols: List[str]) -> Dict[str, object]:
        return {
            "obs_cols_count": int(len(obs_cols)),
            "obs_cols_preview": list(obs_cols[:12]),
            "obs_cols_signature": "|".join(obs_cols),
        }

    @staticmethod
    def _evaluate_stepdprime_join_quality(used_manifest: Dict[str, object]) -> Dict[str, object]:
        match_ratio = float(used_manifest.get("stepD_prime_match_ratio", 1.0) or 0.0)
        nonzero_ratio = float(used_manifest.get("stepD_prime_nonzero_ratio", 1.0) or 0.0)
        nan_fill_count = int(used_manifest.get("stepD_prime_nan_fill_count", 0) or 0)
        failure_reasons: List[str] = []
        if match_ratio < 0.95:
            failure_reasons.append(f"embedding_join_ratio_below_threshold:{match_ratio:.3f}")
        if nonzero_ratio < 0.80:
            failure_reasons.append(f"embedding_nonzero_ratio_below_threshold:{nonzero_ratio:.3f}")
        if nan_fill_count > 0:
            failure_reasons.append(f"dprime_nan_fill_detected:{nan_fill_count}")
        status = "PASS" if not failure_reasons else "FAIL"
        return {
            "stepdprime_join_status": status,
            "reuse_eligible": status == "PASS",
            "failure_reason": ";".join(failure_reasons),
            "embedding_join_ratio": match_ratio,
            "embedding_nonzero_ratio": nonzero_ratio,
            "nan_fill_count": nan_fill_count,
        }

    def _write_stepdprime_failure_artifacts(
        self,
        *,
        out_root: Path,
        mode: str,
        cfg: StepEConfig,
        symbol: str,
        quality: Dict[str, object],
        rows_train: int,
        rows_test: int,
    ) -> None:
        out_dir = out_root / "stepE" / mode
        out_dir.mkdir(parents=True, exist_ok=True)
        audit_dir = out_root / "audit" / mode
        audit_dir.mkdir(parents=True, exist_ok=True)
        summary_payload = {
            "agent": cfg.agent,
            "mode": mode,
            "symbol": symbol,
            "rows_train": int(rows_train),
            "rows_test": int(rows_test),
            **quality,
        }
        (out_dir / f"stepE_summary_{cfg.agent}_{symbol}.json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (audit_dir / f"stepE_audit_{cfg.agent}_{symbol}.json").write_text(json.dumps({"status": "FAIL", **summary_payload}, ensure_ascii=False, indent=2), encoding="utf-8")

    def run_agent(self, cfg: StepEConfig, *, date_range, symbol: str, mode: Optional[str] = None) -> Dict[str, object]:
        resolved_mode = str(mode or getattr(date_range, "mode", None) or "sim").strip().lower()
        if resolved_mode in {"ops", "prod", "production", "real"}:
            resolved_mode = "live"
        self._run_one(
            cfg,
            date_range=date_range,
            symbol=symbol,
            mode=resolved_mode,
            execution_context={
                "requested_parallel_agents": int(getattr(cfg, "max_parallel_agents", 1) or 1),
                "max_parallel_agents": int(getattr(cfg, "max_parallel_agents", 1) or 1),
                "effective_parallel_agents": 1,
                "parallelism_warning": "",
            },
        )
        return {"agent": str(cfg.agent), "status": "done", "mode": resolved_mode, "symbol": symbol}

    def run(self, date_range, symbol: str, agents: Optional[List[str]] = None, mode: Optional[str] = None):
        timing = self._timing()
        with timing.stage("stepE.total"):
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

            requested_parallel_agents = max(int(getattr(cfg, "max_parallel_agents", 1) or 1) for cfg in cfgs)
            max_parallel_agents = min(
                2,
                max(1, requested_parallel_agents),
            )
            parallelism_warnings: List[str] = []
            if max_parallel_agents < requested_parallel_agents:
                warning = "max_parallel_agents_capped_at_2_for_safety"
                parallelism_warnings.append(warning)
                print("[StepE][PARALLEL_WARN] max_parallel_agents capped at 2 for safety.")

            requested_devices = [str(getattr(cfg, "device", "auto") or "auto") for cfg in cfgs]
            if max_parallel_agents > 1 and any(dev.startswith("cuda") or dev == "auto" for dev in requested_devices):
                warning = "cuda_parallel_agents_gt_1_may_oversubscribe_gpu"
                parallelism_warnings.append(warning)
                print(
                    "[StepE][PARALLEL_WARN] max_parallel_agents>1 may oversubscribe a single GPU. "
                    "Keep StepE parallelism conservative on CUDA runs."
                )

            effective_parallel_agents = min(max_parallel_agents, len(cfgs))
            parallelism_warning = ";".join(dict.fromkeys(parallelism_warnings))
            run_context = {
                "requested_parallel_agents": int(requested_parallel_agents),
                "max_parallel_agents": int(max_parallel_agents),
                "effective_parallel_agents": int(effective_parallel_agents),
                "parallelism_warning": parallelism_warning,
            }
            print(
                f"[StepE][PARALLEL] requested_parallel_agents={requested_parallel_agents} "
                f"effective_parallel_agents={effective_parallel_agents} "
                f"parallelism_warning={(parallelism_warning or '(none)')}"
            )

            if max_parallel_agents == 1 or len(cfgs) == 1:
                for cfg in cfgs:
                    self._run_one(cfg, date_range=date_range, symbol=symbol, mode=mode, execution_context=run_context)
            else:
                print(f"[StepE] running {len(cfgs)} agents with max_parallel_agents={max_parallel_agents}")
                with ThreadPoolExecutor(max_workers=max_parallel_agents, thread_name_prefix="stepE-agent") as executor:
                    futures = [
                        executor.submit(self._run_one, cfg, date_range=date_range, symbol=symbol, mode=mode, execution_context=run_context)
                        for cfg in cfgs
                    ]
                    done, not_done = wait(futures, return_when=FIRST_EXCEPTION)
                    first_error = next((future.exception() for future in done if future.exception() is not None), None)
                    if first_error is not None:
                        for future in not_done:
                            future.cancel()
                        raise first_error
                    for future in futures:
                        future.result()

            return {
                "skipped": False,
                "configs_run": len(cfgs),
                "symbol": symbol,
                "mode": mode,
                "requested_parallel_agents": requested_parallel_agents,
                "max_parallel_agents": max_parallel_agents,
                "effective_parallel_agents": effective_parallel_agents,
                "parallelism_warning": parallelism_warning,
            }

    # -----------------------
    # Main per-agent run
    # -----------------------

    def _run_one(self, cfg: StepEConfig, date_range, symbol: str, mode: str, execution_context: Optional[Dict[str, object]] = None) -> None:
        timing = self._timing()
        execution_context = dict(execution_context or {})
        with timing.stage("stepE.agent.total", agent_id=str(cfg.agent), meta={"stage_group": "stepE_agent", "mode": str(mode), "agent_kind": "expert", "expert_name": str(cfg.agent)}):
            out_root = Path(cfg.output_root or getattr(self.app_config, "output_root", "output"))
            out_dir = out_root / "stepE" / mode
            model_dir = out_dir / "models"
            out_dir.mkdir(parents=True, exist_ok=True)
            model_dir.mkdir(parents=True, exist_ok=True)

            cfg.seed = 42 if getattr(cfg, "seed", None) is None else int(cfg.seed)
            cfg.device = str(getattr(cfg, "device", "auto") or "auto")
            cfg.policy_kind = str(getattr(cfg, "policy_kind", "ppo") or "ppo").strip().lower()
            if cfg.policy_kind != "ppo":
                raise ValueError(f"Unsupported StepE policy_kind: {cfg.policy_kind}. StepE is PPO-only.")
            actual_device_name, device_warnings = self._resolve_device(cfg.device)
            used_manifest: dict[str, object] = {}

            if cfg.verbose:
                print(
                    f"[StepE] agent={cfg.agent} mode={mode} profile={cfg.obs_profile} use_stepd_prime={cfg.use_stepd_prime} "
                    f"seed={cfg.seed} device_requested={cfg.device} device_execution={actual_device_name} "
                    f"policy_kind=ppo policy_role=only_supported_policy"
                )
            for warning in device_warnings:
                print(f"[StepE][DEVICE_WARN] agent={cfg.agent} warning={warning}")

        # Load & merge inputs (train+test)
        with timing.stage("stepE.agent.merge_inputs", agent_id=str(cfg.agent), meta={"agent_kind": "expert", "expert_name": str(cfg.agent)}):
            df_all, used_manifest, merge_cache_info = self._get_shared_merge_inputs(
                cfg,
                date_range=date_range,
                out_root=out_root,
                mode=mode,
                symbol=symbol,
            )

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
                used_manifest["stepD_prime_nonzero_ratio"] = nonzero_ratio

        stepdprime_quality = self._evaluate_stepdprime_join_quality(used_manifest) if cfg.use_stepd_prime else {
            "stepdprime_join_status": "PASS",
            "reuse_eligible": True,
            "failure_reason": "",
            "embedding_join_ratio": 1.0,
            "embedding_nonzero_ratio": 1.0,
            "nan_fill_count": 0,
        }
        if stepdprime_quality["stepdprime_join_status"] != "PASS":
            self._write_stepdprime_failure_artifacts(
                out_root=out_root,
                mode=mode,
                cfg=cfg,
                symbol=symbol,
                quality=stepdprime_quality,
                rows_train=len(df_train),
                rows_test=len(df_test),
            )
            raise RuntimeError(f"StepD' quality gate failed: {stepdprime_quality['failure_reason']}")

        # Observation columns
        with timing.stage("stepE.agent.select_obs", agent_id=str(cfg.agent), meta={"agent_kind": "expert", "expert_name": str(cfg.agent)}):
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
        feature_summary = self._input_feature_summary(obs_cols)

        # Prepare tensors
        X_train, r_soxl_train, r_soxs_train, dates_train = self._build_obs_and_returns(df_train, obs_cols)
        X_test, r_soxl_test, r_soxs_test, dates_test = self._build_obs_and_returns(df_test, obs_cols)

        # Standardize based on train
        mu = X_train.mean(axis=0)
        sd = X_train.std(axis=0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        X_train_s = (X_train - mu) / sd
        X_test_s = (X_test - mu) / sd

        with timing.stage("stepE.agent.train", agent_id=str(cfg.agent), meta={"policy": "ppo", "agent_kind": "expert", "expert_name": str(cfg.agent)}):
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
                used_manifest=used_manifest,
                device_warnings=device_warnings,
                resolved_device_name=actual_device_name,
                feature_summary=feature_summary,
                execution_context=execution_context,
                merge_cache_info=merge_cache_info,
            )
        timing.emit("stepE.agent.eval_and_save", elapsed_ms=0.0, agent_id=str(cfg.agent), meta={"policy": "ppo", "agent_kind": "expert", "expert_name": str(cfg.agent)})

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
        used_manifest: Dict[str, object],
        device_warnings: List[str],
        resolved_device_name: str,
        feature_summary: Dict[str, object],
        execution_context: Dict[str, object],
        merge_cache_info: Dict[str, object],
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
        stepdprime_quality = self._evaluate_stepdprime_join_quality(used_manifest) if cfg.use_stepd_prime else {
            "stepdprime_join_status": "PASS",
            "reuse_eligible": True,
            "failure_reason": "",
            "embedding_join_ratio": 1.0,
            "embedding_nonzero_ratio": 1.0,
            "nan_fill_count": 0,
        }

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

        # NOTE:
        # DummyVecEnv is still used here because SB3 PPO expects a VecEnv wrapper,
        # but this is intentionally a single-environment configuration:
        # DummyVecEnv([lambda: train_env]) does not introduce environment parallelism.
        # StepE wall-time reduction in this change comes from PPO-only simplification,
        # lighter PPO defaults, and agent-level parallel execution instead.
        vec_env = DummyVecEnv([lambda: train_env])
        vec_env.seed(seed)

        device_name = resolved_device_name
        learning_rate = float(getattr(cfg, "ppo_lr", None) or getattr(cfg, "lr", 3e-4) or 3e-4)

        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=learning_rate,
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
        df_log["source_device"] = device_name
        df_log["device_requested"] = str(cfg.device)
        df_log["embedding_join_ratio"] = float(used_manifest.get("stepD_prime_match_ratio", 1.0))
        df_log["embedding_nonzero_ratio"] = float(used_manifest.get("stepD_prime_nonzero_ratio", 1.0))
        df_log["nan_fill_count"] = int(used_manifest.get("stepD_prime_nan_fill_count", 0))
        df_log["obs_cols_count"] = int(feature_summary["obs_cols_count"])
        df_log["obs_cols_signature"] = str(feature_summary["obs_cols_signature"])
        df_log["input_feature_summary"] = json.dumps(feature_summary, ensure_ascii=False)
        df_log["requested_parallel_agents"] = int(execution_context.get("requested_parallel_agents", getattr(cfg, "max_parallel_agents", 1)) or 1)
        df_log["max_parallel_agents"] = int(execution_context.get("max_parallel_agents", getattr(cfg, "max_parallel_agents", 1)) or 1)
        df_log["effective_parallel_agents"] = int(execution_context.get("effective_parallel_agents", 1) or 1)
        df_log["parallelism_warning"] = str(execution_context.get("parallelism_warning", "") or "")
        df_log["merge_inputs_cache_hit"] = bool(merge_cache_info.get("merge_cache_hit", False))
        df_log["merge_cache_key"] = str(merge_cache_info.get("merge_cache_key", "") or "")
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
            **_training_config_summary(cfg, device=device_name),
            **self._device_payload(requested=cfg.device, actual=device_name, model_loaded=True),
            **metrics,
            **legacy_metrics,
            **feature_summary,
            **stepdprime_quality,
            "warnings": list(dict.fromkeys([*device_warnings, *list(used_manifest.get("warnings", []) or [])])),
            "degraded": bool(used_manifest.get("degraded", False)),
            "stepdprime_root": str(used_manifest.get("stepDprime_root", "")),
            "stepdprime_legacy_read": bool(used_manifest.get("stepDprime_legacy_read", False)),
            "requested_parallel_agents": int(execution_context.get("requested_parallel_agents", getattr(cfg, "max_parallel_agents", 1)) or 1),
            "max_parallel_agents": int(execution_context.get("max_parallel_agents", getattr(cfg, "max_parallel_agents", 1)) or 1),
            "effective_parallel_agents": int(execution_context.get("effective_parallel_agents", 1) or 1),
            "parallelism_warning": str(execution_context.get("parallelism_warning", "") or ""),
            "merge_inputs_cache_hit": bool(merge_cache_info.get("merge_cache_hit", False)),
            "merge_cache_key": str(merge_cache_info.get("merge_cache_key", "") or ""),
        }
        summ_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        model.save(str(mdl_zip_path))
        torch.save({
            "cfg": asdict(cfg),
            "obs_cols": obs_cols,
            "mu": mu.astype(np.float32),
            "sd": sd.astype(np.float32),
            "policy_kind": "ppo",
            "ppo_lr": learning_rate,
            "sb3_model_path": str(mdl_zip_path),
        }, mdl_pt_path)

        if cfg.verbose:
            print(f"[StepE] wrote equity={eq_path}")
            print(f"[StepE] wrote daily_log={log_path}")
            print(f"[StepE] wrote summary={summ_path}")
            print(f"[StepE] wrote model(zip)={mdl_zip_path}")
            print(f"[StepE] wrote model(pt)={mdl_pt_path}")
        audit_dir = Path(cfg.output_root or "output") / "audit" / mode
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit_payload = {
            "agent": cfg.agent,
            "symbol": symbol,
            "mode": mode,
            **self._device_payload(requested=cfg.device, actual=device_name, model_loaded=True),
            **feature_summary,
            **stepdprime_quality,
            "all_zero_input_detected": bool(np.all(np.abs(X_train_s) < 1e-12)),
            "constant_output_detected": bool(float(np.std(pos_arr)) < 1e-9),
            "warnings": list(dict.fromkeys([*device_warnings, *list(used_manifest.get("warnings", []) or [])])),
            "requested_parallel_agents": int(execution_context.get("requested_parallel_agents", getattr(cfg, "max_parallel_agents", 1)) or 1),
            "max_parallel_agents": int(execution_context.get("max_parallel_agents", getattr(cfg, "max_parallel_agents", 1)) or 1),
            "effective_parallel_agents": int(execution_context.get("effective_parallel_agents", 1) or 1),
            "parallelism_warning": str(execution_context.get("parallelism_warning", "") or ""),
            "merge_inputs_cache_hit": bool(merge_cache_info.get("merge_cache_hit", False)),
            "merge_cache_key": str(merge_cache_info.get("merge_cache_key", "") or ""),
            "status": "PASS",
        }
        (audit_dir / f"stepE_audit_{cfg.agent}_{symbol}.json").write_text(json.dumps(audit_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # -----------------------
    # Load & merge
    # -----------------------

    def _merge_cache_key(self, cfg: StepEConfig, *, date_range, out_root: Path, mode: str, symbol: str) -> Tuple[object, ...]:
        return (
            str(out_root.resolve()),
            str(mode),
            str(symbol),
            str(pd.to_datetime(getattr(date_range, "train_start")).date()),
            str(pd.to_datetime(getattr(date_range, "train_end")).date()),
            str(pd.to_datetime(getattr(date_range, "test_start")).date()),
            str(pd.to_datetime(getattr(date_range, "test_end")).date()),
            bool(getattr(cfg, "use_dprime_state", False)),
            bool(getattr(cfg, "use_stepd_prime", False)),
            str(getattr(cfg, "dprime_state_variant", "") or ""),
            str(getattr(cfg, "dprime_profile", "") or ""),
            str(getattr(cfg, "dprime_sources", "") or ""),
            str(getattr(cfg, "dprime_horizons", "") or ""),
            str(getattr(cfg, "dprime_join", "") or ""),
            bool(getattr(cfg, "pair_trade", True)),
            str(getattr(cfg, "long_symbol", "SOXL") or "SOXL"),
            str(getattr(cfg, "short_symbol", "SOXS") or "SOXS"),
        )

    def _build_stepdprime_shared_context(self, *, cfg: StepEConfig, out_root: Path, mode: str) -> dict[str, object]:
        needs_stepdprime = bool(getattr(cfg, "use_dprime_state", False) or getattr(cfg, "use_stepd_prime", False))
        shared_context: dict[str, object] = {
            "stepDprime_root": "",
            "stepDprime_root_resolved": False,
            "stepDprime_legacy_read": False,
            "stepDprime_warnings": [],
        }
        if not needs_stepdprime:
            return shared_context
        stepd_root, warnings, legacy_read = resolve_stepdprime_root(out_root, mode)
        shared_context.update(
            {
                "stepDprime_root": str(stepd_root),
                "stepDprime_root_resolved": True,
                "stepDprime_legacy_read": bool(legacy_read),
                "stepDprime_warnings": list(dict.fromkeys(str(w) for w in warnings if str(w))),
            }
        )
        return shared_context

    def _get_shared_merge_inputs(
        self,
        cfg: StepEConfig,
        *,
        date_range,
        out_root: Path,
        mode: str,
        symbol: str,
    ) -> tuple[pd.DataFrame, dict[str, object], Dict[str, object]]:
        cache_key = self._merge_cache_key(cfg, date_range=date_range, out_root=out_root, mode=mode, symbol=symbol)
        should_compute = False
        wait_event = None
        with self._merge_cache_lock:
            cached = self._merge_cache.get(cache_key)
            if cached is None:
                wait_event = self._merge_cache_inflight.get(cache_key)
                if wait_event is None:
                    wait_event = threading.Event()
                    self._merge_cache_inflight[cache_key] = wait_event
                    self._merge_cache_errors.pop(cache_key, None)
                    should_compute = True
        cache_hit = cached is not None or not should_compute
        if should_compute:
            try:
                shared_context = self._build_stepdprime_shared_context(cfg=cfg, out_root=out_root, mode=mode)
                df_all, used_manifest = self._merge_inputs(cfg, out_root=out_root, mode=mode, symbol=symbol, shared_context=shared_context)
                with self._merge_cache_lock:
                    cached = self._merge_cache.get(cache_key)
                    if cached is None:
                        cached = (df_all, used_manifest, shared_context)
                        self._merge_cache[cache_key] = cached
            except BaseException as exc:
                with self._merge_cache_lock:
                    self._merge_cache_errors[cache_key] = exc
                raise
            finally:
                with self._merge_cache_lock:
                    wait_event = self._merge_cache_inflight.pop(cache_key, wait_event)
                    if wait_event is not None:
                        wait_event.set()
        elif cached is None:
            assert wait_event is not None
            wait_event.wait()
            with self._merge_cache_lock:
                cached = self._merge_cache.get(cache_key)
                cached_error = self._merge_cache_errors.get(cache_key)
            if cached is None:
                if cached_error is not None:
                    raise RuntimeError(f"StepE shared merge_inputs failed for cache_key={cache_key}") from cached_error
                raise RuntimeError(f"StepE shared merge_inputs missing cache entry for cache_key={cache_key}")
        cache_info = {
            "merge_cache_hit": bool(cache_hit),
            "merge_cache_key": "|".join(str(part) for part in cache_key),
        }
        print(
            f"[StepE][MERGE_CACHE] agent={cfg.agent} cache_hit={str(cache_hit).lower()} "
            f"mode={mode} symbol={symbol} cache_key={cache_info['merge_cache_key']}"
        )
        shared_df_all, shared_manifest, _shared_context = cached
        return shared_df_all.copy(deep=False), dict(shared_manifest), cache_info

    def _merge_inputs(self, cfg: StepEConfig, out_root: Path, mode: str, symbol: str, *, shared_context: Optional[dict[str, object]] = None) -> tuple[pd.DataFrame, dict[str, object]]:
        df_prices = self._load_stepA_prices(out_root, mode, symbol)
        used_manifest: dict[str, object] = {"warnings": [], "degraded": False}
        shared_context = dict(shared_context or {})
        if shared_context:
            used_manifest["stepDprime_root"] = str(shared_context.get("stepDprime_root", "") or "")
            used_manifest["stepDprime_legacy_read"] = bool(shared_context.get("stepDprime_legacy_read", False))
            used_manifest["warnings"] = list(dict.fromkeys([*list(used_manifest.get("warnings", []) or []), *list(shared_context.get("stepDprime_warnings", []) or [])]))

        if cfg.use_dprime_state:
            df_state = self._load_stepD_prime_state(cfg, out_root=out_root, mode=mode, symbol=symbol, shared_context=shared_context)
            df = df_state.merge(
                df_prices[["Date", "Open", "High", "Low", "Close", "Volume", "price_exec"]],
                on="Date",
                how="left",
            )
        else:
            df = df_prices.copy()

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
            dprime_df = self._load_stepD_prime_embeddings(cfg, out_root=out_root, mode=mode, symbol=symbol, used_manifest=used_manifest, shared_context=shared_context)
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
            dprime_df["Date"] = pd.to_datetime(dprime_df["Date"], errors="coerce").dt.normalize()

            dprime_dates = dprime_df["Date"].dropna().unique()
            match_ratio = float(df["Date"].isin(dprime_dates).mean()) if len(dprime_dates) > 0 else 0.0
            used_manifest["stepD_prime_match_ratio"] = match_ratio
            if match_ratio < 0.95:
                df_min, df_max = df["Date"].min(), df["Date"].max()
                dp_min, dp_max = dprime_df["Date"].min(), dprime_df["Date"].max()
                raise RuntimeError(
                    "StepD' merge mismatch: "
                    f"match_ratio={match_ratio:.3f} df={df_min}..{df_max} dprime={dp_min}..{dp_max}"
                )

            df = df.merge(dprime_df, on="Date", how="left")
            dp_cols = [c for c in df.columns if c.startswith("dprime_") and "_emb_" in c]
            if dp_cols:
                nan_fill_count = int(df[dp_cols].isna().sum().sum())
                used_manifest["stepD_prime_nan_fill_count"] = nan_fill_count
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

    def _load_stepD_prime_state(self, cfg: StepEConfig, out_root: Path, mode: str, symbol: str, *, shared_context: Optional[dict[str, object]] = None) -> pd.DataFrame:
        profile = str(getattr(cfg, "dprime_profile", "") or "").strip()
        variant = str(cfg.dprime_state_variant or "").strip()
        if not profile and variant:
            # compatibility mapping for old variant-only configs
            profile = f"dprime_{variant}_h02"
        if not profile:
            raise ValueError("dprime_profile is empty while use_dprime_state=True")

        shared_context = dict(shared_context or {})
        stepd_root = Path(shared_context.get("stepDprime_root", "") or "") if shared_context.get("stepDprime_root") else None
        warnings = list(shared_context.get("stepDprime_warnings", []) or [])
        if stepd_root is None:
            stepd_root, warnings, _legacy_read = resolve_stepdprime_root(out_root, mode)
        for warning in warnings:
            print(f"[StepE][STEPDPRIME_WARN] {warning}")
        p_tr = stepd_root / f"stepDprime_state_train_{profile}_{symbol}.csv"
        p_te = stepd_root / f"stepDprime_state_test_{profile}_{symbol}.csv"
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

    def _load_stepD_prime_embeddings(self, cfg: StepEConfig, out_root: Path, mode: str, symbol: str, used_manifest: Optional[dict[str, object]] = None, *, shared_context: Optional[dict[str, object]] = None) -> pd.DataFrame:
        """
        Load embeddings with profile-first resolution. If profile is missing, fallback to
        source/horizon style for backward compatibility.
        """
        shared_context = dict(shared_context or {})
        stepd_root = Path(shared_context.get("stepDprime_root", "") or "") if shared_context.get("stepDprime_root") else None
        warnings = list(shared_context.get("stepDprime_warnings", []) or [])
        legacy_read = bool(shared_context.get("stepDprime_legacy_read", False))
        if stepd_root is None:
            stepd_root, warnings, legacy_read = resolve_stepdprime_root(out_root, mode)
        if used_manifest is not None:
            used_manifest["stepDprime_root"] = str(stepd_root)
            used_manifest["stepDprime_legacy_read"] = bool(legacy_read)
            used_manifest["warnings"] = list(dict.fromkeys([*list(used_manifest.get("warnings", []) or []), *warnings]))
        for warning in warnings:
            print(f"[StepE][STEPDPRIME_WARN] {warning}")
        base = stepd_root / "embeddings"

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
