# ai_core/services/step_f_service.py
# -*- coding: utf-8 -*-
"""
StepF Service (MARL-like gate over StepE agents)

This StepF does NOT depend on ai_core/marl or ai_core/common folders.
It learns a small "gating" network that combines multiple StepE agents
into a single portfolio position.

Inputs
------
- StepE daily logs:
    output/stepE/<mode>/stepE_daily_log_<agent>_<SYMBOL>.csv
  The daily log must include:
    Date, Position (or pos), and ideally Split=train/test (not mandatory if dates are used)

- StepA prices:
    output/stepA/<mode>/stepA_prices_train_<SYMBOL>.csv
    output/stepA/<mode>/stepA_prices_test_<SYMBOL>.csv

Outputs
-------
output/stepF/<mode>/
  stepF_equity_marl_<SYMBOL>.csv
  stepF_daily_log_marl_<SYMBOL>.csv
  stepF_summary_marl_<SYMBOL>.json
  models/stepF_marl_gate_<SYMBOL>.pt
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
from torch import nn


@dataclass
class StepFConfig:
    output_root: str = "output"
    agents: str = ""  # comma-separated list of StepE agent names
    seed: int = 42
    verbose: bool = True

    # training
    hidden_dim: int = 64
    lr: float = 1e-3
    epochs: int = 120
    patience: int = 15
    weight_decay: float = 1e-4
    smooth_abs_eps: float = 1e-6

    # trading
    trade_cost_bps: float = 5.0
    pos_l2: float = 1e-3
    pos_limit: float = 1.0

    device: str = "auto"

    # optional context (StepDPrime state)
    use_context: bool = False
    context_variant: str = "mix"  # bnf/all_features/mix
    context_profile: str = "minimal"  # minimal / all


class GateNet(nn.Module):
    def __init__(self, n_agents: int, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.n_agents = int(n_agents)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_agents),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # logits


class StepFService:
    def __init__(self, app_config):
        self.app_config = app_config

    def run(self, date_range, symbol: str, mode: Optional[str] = None) -> None:
        mode = mode or getattr(date_range, "mode", None) or "sim"
        cfg: StepFConfig = getattr(self.app_config, "stepF", None)
        if cfg is None:
            raise ValueError("app_config.stepF (StepFConfig) is missing.")
        self._run_one(cfg, date_range=date_range, symbol=symbol, mode=mode)

    def _run_one(self, cfg: StepFConfig, date_range, symbol: str, mode: str) -> None:
        out_root = Path(cfg.output_root or getattr(self.app_config, "output_root", "output"))
        out_dir = out_root / "stepF" / mode
        model_dir = out_dir / "models"
        out_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        agents = [a.strip() for a in (cfg.agents or "").split(",") if a.strip()]
        if not agents:
            raise ValueError("StepFConfig.agents is empty. Provide comma-separated StepE agent names.")

        if cfg.verbose:
            print(f"[StepF] mode={mode} symbol={symbol} agents={len(agents)}")

        # load prices (train+test)
        df_prices = self._load_stepA_prices(out_root, mode, symbol)
        df_prices = df_prices.sort_values("Date").reset_index(drop=True)
        df_prices["oc_ret"] = df_prices["Close"].astype(float) / df_prices["Open"].astype(float).replace(0, np.nan) - 1.0
        df_prices["oc_ret"] = df_prices["oc_ret"].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        # Return must be based on next-day Close (Close_eff fallback not available here)
        _ce = df_prices["Close"].astype(float)
        df_prices["cc_ret_next"] = (_ce.shift(-1) / _ce - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

        train_start = pd.to_datetime(getattr(date_range, "train_start"))
        train_end = pd.to_datetime(getattr(date_range, "train_end"))
        test_start = pd.to_datetime(getattr(date_range, "test_start"))
        test_end = pd.to_datetime(getattr(date_range, "test_end"))

        df_prices = df_prices[(df_prices["Date"] >= train_start) & (df_prices["Date"] <= test_end)].copy()

        # load stepE positions
        pos_df = None
        for a in agents:
            df = self._load_stepE_daily_log(out_root, mode, symbol, a)
            pos_col = "Position" if "Position" in df.columns else ("pos" if "pos" in df.columns else None)
            if pos_col is None:
                raise ValueError(f"StepE daily log for {a} missing Position/pos column")
            df = df[["Date", pos_col]].copy()
            df = df.rename(columns={pos_col: f"pos_{a}"})
            pos_df = df if pos_df is None else pos_df.merge(df, on="Date", how="outer")

        assert pos_df is not None
        df_all = df_prices.merge(pos_df, on="Date", how="inner").sort_values("Date").reset_index(drop=True)

        context_cols: List[str] = []
        if bool(getattr(cfg, "use_context", False)):
            df_ctx = self._load_stepd_prime_context(out_root, mode, symbol, variant=str(getattr(cfg, "context_variant", "mix") or "mix"))
            if not df_ctx.empty:
                df_all = df_all.merge(df_ctx, on="Date", how="left")
                context_cols = self._select_context_columns(df_all, profile=str(getattr(cfg, "context_profile", "minimal") or "minimal"))
                for c in context_cols:
                    df_all[c] = pd.to_numeric(df_all[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

        feature_cols = [f"pos_{a}" for a in agents] + context_cols

        # ensure no missing
        for a in agents:
            c = f"pos_{a}"
            df_all[c] = df_all[c].astype(float).fillna(0.0)

        # split
        df_train = df_all[(df_all["Date"] >= train_start) & (df_all["Date"] <= train_end)].copy().reset_index(drop=True)
        df_test = df_all[(df_all["Date"] >= test_start) & (df_all["Date"] <= test_end)].copy().reset_index(drop=True)

        if len(df_train) < 200:
            raise RuntimeError(f"Train rows too small for StepF: {len(df_train)}")
        if len(df_test) < 10:
            raise RuntimeError(f"Test rows too small for StepF: {len(df_test)}")

        # tensors
        seed = int(42 if getattr(cfg, "seed", None) is None else cfg.seed)
        device_name = str(getattr(cfg, "device", "auto") or "auto").strip().lower()
        if device_name in ("", "none", "auto"):
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)
        torch.manual_seed(seed)
        np.random.seed(seed)

        X_train = torch.tensor(df_train[feature_cols].to_numpy(dtype=np.float32), device=device)
        X_test = torch.tensor(df_test[feature_cols].to_numpy(dtype=np.float32), device=device)

        r_train = torch.tensor(df_train["cc_ret_next"].to_numpy(dtype=np.float32), device=device)
        r_test = torch.tensor(df_test["cc_ret_next"].to_numpy(dtype=np.float32), device=device)

        gate = GateNet(n_agents=len(agents), in_dim=len(feature_cols), hidden_dim=int(cfg.hidden_dim)).to(device)
        opt = torch.optim.AdamW(gate.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

        best = float("inf")
        best_state = None
        wait = 0

        n = X_train.shape[0]
        n_val = max(10, int(0.2 * n))
        n_fit = max(50, n - n_val)

        def _smooth_abs(x: torch.Tensor) -> torch.Tensor:
            return torch.sqrt(x * x + float(cfg.smooth_abs_eps))

        def _rollout(g: GateNet, X_: torch.Tensor, r_: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            T = X_.shape[0]
            pos_prev = torch.zeros((), device=X_.device)
            pos_list = []
            w_list = []
            ret_list = []
            cost_k = float(cfg.trade_cost_bps) * 1e-4
            for t in range(T):
                logits = g(X_[t])
                w = torch.softmax(logits, dim=0)  # (N,)
                pos_t = torch.sum(w * X_[t])  # combine agent positions
                pos_t = torch.clamp(pos_t, -float(cfg.pos_limit), float(cfg.pos_limit))
                cost = cost_k * _smooth_abs(pos_t - pos_prev)
                ret_net = pos_t * r_[t] - cost - float(cfg.pos_l2) * (pos_t * pos_t)
                pos_list.append(pos_t)
                w_list.append(w)
                ret_list.append(ret_net)
                pos_prev = pos_t
            pos = torch.stack(pos_list, dim=0)
            W = torch.stack(w_list, dim=0)
            ret_net = torch.stack(ret_list, dim=0)
            return ret_net, pos, W

        for ep in range(1, int(cfg.epochs) + 1):
            gate.train()
            opt.zero_grad()
            ret_fit, _, _ = _rollout(gate, X_train[:n_fit], r_train[:n_fit])
            loss = -torch.log1p(ret_fit).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.0)
            opt.step()

            gate.eval()
            with torch.no_grad():
                ret_val, _, _ = _rollout(gate, X_train[n_fit:], r_train[n_fit:])
                val_loss = float((-torch.log1p(ret_val).mean()).item())

            if cfg.verbose:
                print(f"[StepF] ep={ep:03d} fit_loss={float(loss.item()):.6f} val_loss={val_loss:.6f}")

            if val_loss + 1e-8 < best:
                best = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in gate.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= int(cfg.patience):
                    if cfg.verbose:
                        print(f"[StepF] early stop at ep={ep}")
                    break

        if best_state is not None:
            gate.load_state_dict(best_state)

        # evaluate full train+test for daily log
        gate.eval()
        with torch.no_grad():
            ret_tr, pos_tr, W_tr = _rollout(gate, X_train, r_train)
            ret_te, pos_te, W_te = _rollout(gate, X_test, r_test)

        # build logs
        df_log = pd.DataFrame({
            "Date": pd.to_datetime(list(df_train["Date"].to_numpy()) + list(df_test["Date"].to_numpy())),
            "Split": (["train"] * len(df_train)) + (["test"] * len(df_test)),
            "pos": torch.cat([pos_tr, pos_te], dim=0).cpu().numpy().astype(float),
            "ret": torch.cat([ret_tr, ret_te], dim=0).cpu().numpy().astype(float),
        })
        df_log["equity"] = (1.0 + df_log["ret"]).cumprod()

        # add weights columns
        W = torch.cat([W_tr, W_te], dim=0).cpu().numpy()
        for i, a in enumerate(agents):
            df_log[f"w_{a}"] = W[:, i].astype(float)

        df_eq = df_log[df_log["Split"] == "test"][["Date", "pos", "ret", "equity"]].copy().reset_index(drop=True)

        eq_path = out_dir / f"stepF_equity_marl_{symbol}.csv"
        log_path = out_dir / f"stepF_daily_log_marl_{symbol}.csv"
        df_eq.to_csv(eq_path, index=False)
        df_log.to_csv(log_path, index=False)

        metrics = self._compute_metrics(df_eq)

        summary = {
            "mode": mode,
            "symbol": symbol,
            "agents": agents,
            "train_start": str(train_start.date()),
            "train_end": str(train_end.date()),
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "rows_train": int(len(df_train)),
            "rows_test": int(len(df_test)),
            "n_context": int(len(context_cols)),
            "context_columns": context_cols,
            **metrics,
        }
        summ_path = out_dir / f"stepF_summary_marl_{symbol}.json"
        summ_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        mdl_path = model_dir / f"stepF_marl_gate_{symbol}.pt"
        torch.save({
            "cfg": asdict(cfg),
            "agents": agents,
            "state_dict": gate.state_dict(),
        }, mdl_path)

        if cfg.verbose:
            print(f"[StepF] wrote equity={eq_path}")
            print(f"[StepF] wrote daily_log={log_path}")
            print(f"[StepF] wrote summary={summ_path}")
            print(f"[StepF] wrote model={mdl_path}")

    # -----------------------
    # IO
    # -----------------------

    def _load_stepA_prices(self, out_root: Path, mode: str, symbol: str) -> pd.DataFrame:
        base = out_root / "stepA" / mode
        p_tr = base / f"stepA_prices_train_{symbol}.csv"
        p_te = base / f"stepA_prices_test_{symbol}.csv"
        if not (p_tr.exists() and p_te.exists()):
            raise FileNotFoundError(f"Missing StepA prices: {p_tr} / {p_te}")
        df = pd.concat([pd.read_csv(p_tr), pd.read_csv(p_te)], axis=0, ignore_index=True)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        return df

    def _load_stepE_daily_log(self, out_root: Path, mode: str, symbol: str, agent: str) -> pd.DataFrame:
        p = out_root / "stepE" / mode / f"stepE_daily_log_{agent}_{symbol}.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing StepE daily log: {p}")
        df = pd.read_csv(p)
        if "Date" not in df.columns:
            raise KeyError(f"{p} missing Date")
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        if "Position" in df.columns:
            pos = df["Position"]
        elif "pos" in df.columns:
            pos = df["pos"]
        else:
            raise KeyError(f"{p} missing Position/pos column")

        out = pd.DataFrame({"Date": df["Date"], "Position": pos.astype(float)})
        return out.sort_values("Date").reset_index(drop=True)

    def _load_stepd_prime_context(self, out_root: Path, mode: str, symbol: str, variant: str) -> pd.DataFrame:
        base = out_root / "stepD_prime" / mode
        paths = [
            base / f"stepDprime_state_{variant}_{symbol}_train.csv",
            base / f"stepDprime_state_{variant}_{symbol}_test.csv",
        ]
        frames: List[pd.DataFrame] = []
        for p in paths:
            if not p.exists():
                continue
            df = pd.read_csv(p)
            if "Date" not in df.columns:
                continue
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            frames.append(df)

        if not frames:
            return pd.DataFrame(columns=["Date"])

        out = pd.concat(frames, axis=0, ignore_index=True)
        out = out.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
        return out.reset_index(drop=True)

    def _select_context_columns(self, df: pd.DataFrame, profile: str) -> List[str]:
        minimal_candidates = [
            "gap", "Gap", "atr_norm", "ATR_norm", "ret_1", "ret_5", "ret_20", "range_atr", "body_ratio",
            "vol_log_ratio_20", "dev_z_25", "bnf_score", "RSI", "MACD_hist",
        ]
        reserved = {"Date", "Open", "High", "Low", "Close", "Volume", "oc_ret", "cc_ret_next", "Split"}

        p = str(profile or "minimal").strip().lower()
        if p == "all":
            out = []
            for c in df.columns:
                if c in reserved or c.startswith("pos_"):
                    continue
                if pd.api.types.is_numeric_dtype(df[c]):
                    out.append(c)
            return out

        return [c for c in minimal_candidates if c in df.columns]

    # -----------------------
    # Metrics
    # -----------------------

    def _compute_metrics(self, df_eq: pd.DataFrame) -> Dict[str, float]:
        ret = df_eq["ret"].astype(float).to_numpy()
        eq = df_eq["equity"].astype(float).to_numpy()
        if len(ret) == 0:
            return {}

        total_return = float(eq[-1] - 1.0)
        mean = float(np.mean(ret))
        std = float(np.std(ret, ddof=0))
        sharpe = float((mean / (std + 1e-12)) * math.sqrt(252.0))

        peak = np.maximum.accumulate(eq)
        dd = (eq / np.where(peak == 0, 1.0, peak)) - 1.0
        max_dd = float(np.min(dd))

        pos = df_eq["pos"].astype(float).to_numpy()
        trades = int(np.sum(np.abs(np.diff(pos)) > 1e-9))

        years = max(1e-6, len(ret) / 252.0)
        cagr = float(eq[-1] ** (1.0 / years) - 1.0)

        return {
            "total_return": total_return,
            "cagr": cagr,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "num_trades": float(trades),
        }
