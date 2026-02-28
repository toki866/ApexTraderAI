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

        # Ensure columns exist
        obs_cols = [c for c in obs_cols if c in df_all.columns]
        if not obs_cols:
            raise RuntimeError("No observation columns selected.")
        if cfg.verbose:
            print(f"[StepE] obs_cols(first5)={obs_cols[:5]}")

        # Prepare tensors
        X_train, r_soxl_train, r_soxs_train, cc_ret_train, dates_train = self._build_obs_and_returns(df_train, obs_cols)
        X_test, r_soxl_test, r_soxs_test, cc_ret_test, dates_test = self._build_obs_and_returns(df_test, obs_cols)

        # Standardize based on train
        mu = X_train.mean(axis=0)
        sd = X_train.std(axis=0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        X_train_s = (X_train - mu) / sd
        X_test_s = (X_test - mu) / sd

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
            "cc_ret_next_soxl": cc_soxl,
            "cc_ret_next_soxs": cc_soxs,
            "cost": cost_arr,
        })
        df_log["ratio"] = df_log["pos"]
        df_log["abs_ratio"] = df_log["ratio"].abs()
        df_log["cc_ret_next"] = np.where(df_log["ratio"] > 1e-8, df_log["cc_ret_next_soxl"], np.where(df_log["ratio"] < -1e-8, df_log["cc_ret_next_soxs"], 0.0))
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

        # Summary metrics (test)
        metrics = self._compute_metrics(df_eq)

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

    # -----------------------
    # Load & merge
    # -----------------------

    def _merge_inputs(self, cfg: StepEConfig, out_root: Path, mode: str, symbol: str) -> tuple[pd.DataFrame, dict[str, object]]:
        df_prices = self._load_stepA_prices(out_root, mode, symbol)
        df_soxs = self._load_symbol_prices(out_root=out_root, mode=mode, symbol="SOXS", required=False)
        if df_soxs.empty and symbol.upper() == "SOXS":
            df_soxs = df_prices[["Date", "price_exec"]].rename(columns={"price_exec": "price_exec_soxs"})
        elif not df_soxs.empty:
            df_soxs = df_soxs[["Date", "price_exec"]].rename(columns={"price_exec": "price_exec_soxs"})

        if cfg.use_dprime_state:
            df_state = self._load_stepD_prime_state(cfg, out_root=out_root, mode=mode, symbol=symbol)
            df = df_state.merge(
                df_prices[["Date", "Open", "High", "Low", "Close", "Volume", "price_exec"]],
                on="Date",
                how="left",
            )
        else:
            df = df_prices.copy()

        if not df_soxs.empty:
            df = df.merge(df_soxs, on="Date", how="left")
        else:
            df["price_exec_soxs"] = np.nan
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
            dprime_df = self._load_stepD_prime_embeddings(cfg, out_root=out_root, mode=mode, symbol=symbol)

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

        df = df.sort_values("Date").reset_index(drop=True)
        return df, used_manifest

    def _load_stepA_prices(self, out_root: Path, mode: str, symbol: str) -> pd.DataFrame:
        df = self._load_symbol_prices(out_root=out_root, mode=mode, symbol=symbol, required=True)
        return df.sort_values("Date").reset_index(drop=True)

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
        # Avoid raw scale columns (Open/High/Low/Close/Volume)
        drop = {"Open", "High", "Low", "Close", "Volume"}
        all_num = [c for c in all_num if c not in drop]
        # Put dprime first
        out = dprime_cols + [c for c in all_num if c not in dprime_cols]
        return out

    def _build_obs_and_returns(self, df: pd.DataFrame, obs_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        df2 = df.copy()
        df2 = df2.sort_values("Date").reset_index(drop=True)

        X = df2[obs_cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        p_soxl = pd.to_numeric(df2["price_exec"], errors="coerce")
        p_soxs = pd.to_numeric(df2.get("price_exec_soxs", np.nan), errors="coerce")
        r_soxl = (p_soxl.shift(-1) / p_soxl - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        r_soxs = (p_soxs.shift(-1) / p_soxs - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        cc_ret_next = r_soxl.copy()
        dates = df2["Date"].to_numpy()
        return X, r_soxl, r_soxs, cc_ret_next, dates

    # -----------------------
    # Metrics
    # -----------------------

    def _compute_metrics(self, df_eq: pd.DataFrame) -> Dict[str, float]:
        if df_eq is None or len(df_eq) == 0:
            return {
                "test_days": 0,
                "end_capital_from_initial": float("nan"),
                "profit_from_initial": float("nan"),
                "test_return_pct": float("nan"),
                "test_sharpe": float("nan"),
                "test_max_dd": float("nan"),
                "total_return": float("nan"),
                "cagr": float("nan"),
                "sharpe": float("nan"),
                "max_drawdown": float("nan"),
                "num_trades": 0.0,
            }

        # Equity should already be reset to start at 1.0 on the test window.
        if "equity" in df_eq.columns and pd.api.types.is_numeric_dtype(df_eq["equity"]):
            eq = df_eq["equity"].astype(float).to_numpy()
        else:
            r = df_eq["ret"].astype(float).to_numpy()
            eq = np.cumprod(1.0 + r)

        end_capital = float(eq[-1])
        profit = float(end_capital - 1.0)
        test_return_pct = float(profit * 100.0)

        # Sharpe (annualized)
        r = df_eq["ret"].astype(float).to_numpy()
        mu = float(np.mean(r)) if len(r) else 0.0
        sd = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
        sharpe = float((mu / (sd + 1e-12)) * np.sqrt(252.0)) if len(r) > 1 else float("nan")

        # Max drawdown
        peak = np.maximum.accumulate(eq)
        dd = eq / np.where(peak == 0, 1.0, peak) - 1.0
        max_drawdown = float(np.min(dd)) if len(dd) else float("nan")

        # CAGR over the test window (annualized using 252 trading days)
        days = max(1, len(eq))
        cagr = float(end_capital ** (252.0 / days) - 1.0) if end_capital > 0 else float("nan")

        num_trades = float((df_eq["Action"] != 0).sum()) if "Action" in df_eq.columns else float("nan")

        return {
            "test_days": int(len(df_eq)),
            "end_capital_from_initial": end_capital,
            "profit_from_initial": profit,
            "test_return_pct": test_return_pct,
            "test_sharpe": sharpe,
            "test_max_dd": max_drawdown,

            # Backward-compatible keys
            "total_return": profit,
            "cagr": cagr,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "num_trades": num_trades,
        }
