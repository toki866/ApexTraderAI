from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ai_core.config.app_config import AppConfig
from ai_core.config.step_b_config import StepBConfig, WaveletMambaTrainConfig
from ai_core.services.step_b_mamba_runner import run_stepB_mamba
from ai_core.types.step_b_types import StepBResult


class StepBService:
    """Mamba-only StepB service."""

    def __init__(self, app_config: AppConfig) -> None:
        self.app_config = app_config

    def _resolve_run_mode(self, cfg: StepBConfig) -> str:
        m = str(getattr(cfg.mamba, "mode", "sim") or "sim").strip().lower()
        if m in ("ops", "prod", "production"):
            return "live"
        return "sim" if m not in ("sim", "live") else m

    def _out_dir(self, run_mode: str) -> Path:
        out_root = Path(getattr(getattr(self.app_config, "data", None), "output_root", getattr(self.app_config, "output_root", "output")))
        p = out_root / "stepB" / run_mode
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _load_stepa_df(self, symbol: str, run_mode: str, kind: str) -> pd.DataFrame:
        out_root = Path(getattr(getattr(self.app_config, "data", None), "output_root", getattr(self.app_config, "output_root", "output")))
        mode_dir = out_root / "stepA" / run_mode

        if run_mode in ("sim", "live"):
            train_p = mode_dir / f"stepA_{kind}_train_{symbol}.csv"
            test_p = mode_dir / f"stepA_{kind}_test_{symbol}.csv"
            if not train_p.exists() or train_p.stat().st_size <= 0:
                raise FileNotFoundError(f"Missing StepA train {kind} CSV for {symbol}: {train_p}")
            if not test_p.exists() or test_p.stat().st_size <= 0:
                raise FileNotFoundError(f"Missing StepA test {kind} CSV for {symbol}: {test_p}")

            tr = pd.read_csv(train_p)
            te = pd.read_csv(test_p)
            if "Date" in tr.columns:
                tr["Date"] = pd.to_datetime(tr["Date"], errors="coerce")
            if "Date" in te.columns:
                te["Date"] = pd.to_datetime(te["Date"], errors="coerce")
            out = pd.concat([tr, te], axis=0, ignore_index=True)
            if "Date" in out.columns:
                out = out.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
            return out.reset_index(drop=True)

        display_p = mode_dir / f"stepA_{kind}_{symbol}.csv"
        if display_p.exists() and display_p.stat().st_size > 0:
            return pd.read_csv(display_p)
        raise FileNotFoundError(f"Missing StepA {kind} CSV for {symbol}: {display_p}")

    def _write_pred_time_all(self, symbol: str, run_mode: str, mamba_result) -> Path:
        stepb_dir = self._out_dir(run_mode)
        out_path = stepb_dir / f"stepB_pred_time_all_{symbol}.csv"

        pred_path = None
        for k in ("pred_close", "pred_close_path", "output_path"):
            v = getattr(mamba_result, k, None)
            if v:
                pred_path = Path(v)
                break
        if pred_path is None:
            artifacts = getattr(mamba_result, "artifacts", {}) or {}
            for key in ("pred_close", "pred_close_path", "output_path"):
                v = artifacts.get(key)
                if v:
                    pred_path = Path(v)
                    break
        if pred_path is None or not pred_path.exists():
            raise FileNotFoundError("StepB Mamba output path not found")

        df = pd.read_csv(pred_path)
        if "Date" not in df.columns:
            raise ValueError("StepB Mamba output must include Date")

        mamba_col = next((c for c in df.columns if c.lower() == "pred_close_mamba"), None)
        if mamba_col is None:
            raise ValueError("StepB Mamba output must include Pred_Close_MAMBA")

        out_df = df[["Date", mamba_col]].copy()
        out_df = out_df.rename(columns={mamba_col: "Pred_Close_MAMBA"})
        out_df.to_csv(out_path, index=False, encoding="utf-8")
        return out_path

    def run(self, config: StepBConfig | None = None, *args, **kwargs) -> StepBResult:
        stepb_config: Optional[StepBConfig] = config if isinstance(config, StepBConfig) else None
        if stepb_config is None:
            for arg in args:
                if isinstance(arg, StepBConfig):
                    stepb_config = arg
                    break
        if stepb_config is None and isinstance(kwargs.get("config"), StepBConfig):
            stepb_config = kwargs["config"]
        if stepb_config is None:
            raise TypeError("StepBService.run requires StepBConfig")

        cfg_all = StepBConfig.from_any(stepb_config)
        if not cfg_all.enabled_agents():
            raise ValueError("No agents enabled in StepBConfig (mamba disabled).")

        symbol = cfg_all.symbol
        run_mode = self._resolve_run_mode(cfg_all)
        prices_df = self._load_stepa_df(symbol, run_mode, "prices")
        tech_df = self._load_stepa_df(symbol, run_mode, "tech")
        periodic_df = self._load_stepa_df(symbol, run_mode, "periodic")
        features_df = tech_df.merge(periodic_df, on="Date", how="inner") if "Date" in tech_df.columns and "Date" in periodic_df.columns else tech_df

        mamba_res = run_stepB_mamba(
            app_config=self.app_config,
            symbol=symbol,
            prices_df=prices_df,
            features_df=features_df,
            cfg=cfg_all.mamba,
        )
        pred_time_all_path = self._write_pred_time_all(symbol, run_mode, mamba_res)

        return StepBResult(
            success=bool(getattr(mamba_res, "success", True)),
            message=str(getattr(mamba_res, "message", "")),
            out_dir=str(self._out_dir(run_mode)),
            pred_time_all_path=str(pred_time_all_path),
            agent_results={"mamba": mamba_res},
        )
