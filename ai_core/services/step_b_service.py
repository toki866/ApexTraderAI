from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional

import pandas as pd

from ai_core.config.app_config import AppConfig
from ai_core.config.step_b_config import StepBConfig, WaveletMambaTrainConfig
from ai_core.services.step_b_mamba_runner import rollout_periodic_h1_future, run_stepB_mamba
from ai_core.types.step_b_types import StepBResult


class StepBService:
    """Mamba-only StepB service."""

    STEPB_PRED_TIME_ALL_COLUMNS = ("Date", "Pred_Close_MAMBA")

    def __init__(self, app_config: AppConfig) -> None:
        self.app_config = app_config

    def _resolve_run_mode(self, cfg: StepBConfig) -> str:
        m = str(getattr(cfg.mamba, "mode", "sim") or "sim").strip().lower()
        if m in ("ops", "prod", "production"):
            return "live"
        return "sim" if m not in ("sim", "live") else m

    def _out_root(self) -> Path:
        return Path(getattr(getattr(self.app_config, "data", None), "output_root", getattr(self.app_config, "output_root", "output")))

    def _out_dir(self, run_mode: str) -> Path:
        p = self._out_root() / "stepB" / run_mode
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _mode_dir(self, run_mode: str) -> Path:
        return self._out_root() / "stepA" / run_mode

    def _load_stepa_split_df(self, symbol: str, run_mode: str, kind: str, split: str) -> pd.DataFrame:
        p = self._mode_dir(run_mode) / f"stepA_{kind}_{split}_{symbol}.csv"
        if not p.exists() or p.stat().st_size <= 0:
            raise FileNotFoundError(f"Missing StepA {split} {kind} CSV for {symbol}: {p}")
        df = pd.read_csv(p)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        return df

    def _load_stepa_df(self, symbol: str, run_mode: str, kind: str) -> pd.DataFrame:
        if run_mode in ("sim", "live"):
            tr = self._load_stepa_split_df(symbol, run_mode, kind, "train")
            te = self._load_stepa_split_df(symbol, run_mode, kind, "test")
            out = pd.concat([tr, te], axis=0, ignore_index=True)
            if "Date" in out.columns:
                out = out.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
            return out.reset_index(drop=True)

        display_p = self._mode_dir(run_mode) / f"stepA_{kind}_{symbol}.csv"
        if display_p.exists() and display_p.stat().st_size > 0:
            return pd.read_csv(display_p)
        raise FileNotFoundError(f"Missing StepA {kind} CSV for {symbol}: {display_p}")

    def _load_stepa_future_periodic(self, symbol: str, run_mode: str) -> pd.DataFrame:
        p = self._mode_dir(run_mode) / f"stepA_periodic_future_{symbol}.csv"
        if not p.exists() or p.stat().st_size <= 0:
            raise FileNotFoundError(f"Missing StepA future periodic CSV for {symbol}: {p}")
        df = pd.read_csv(p)
        if "Date" not in df.columns:
            raise ValueError(f"stepA_periodic_future_{symbol}.csv must include Date")
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        return df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    def _force_spec(self, cfg: WaveletMambaTrainConfig) -> WaveletMambaTrainConfig:
        return replace(
            cfg,
            lookback_days=128,
            horizons="1,5,10,20",
            periodic_snapshot_horizons=(1, 5, 10, 20),
            periodic_endpoints=(1, 5, 10, 20),
        )

    def _write_pred_time_all(self, symbol: str, run_mode: str, mamba_result) -> Path:
        stepb_dir = self._out_dir(run_mode)
        out_path = stepb_dir / f"stepB_pred_time_all_{symbol}.csv"

        pred_path = None
        csv_paths = getattr(mamba_result, "csv_paths", None)
        if isinstance(csv_paths, dict):
            for key in ("pred_close", "pred_close_path", "output_path"):
                value = csv_paths.get(key)
                if value:
                    pred_path = Path(value)
                    break

        iter_csv_preview = []
        if pred_path is None and hasattr(mamba_result, "iter_csv_paths"):
            try:
                iter_paths = list(mamba_result.iter_csv_paths() or [])
            except Exception:
                iter_paths = []
            iter_csv_preview = [str(p) for p in iter_paths[:5]]
            for value in iter_paths:
                p = Path(value)
                if "pred_close" in p.name.lower():
                    pred_path = p
                    break

        if pred_path is None:
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
            csv_paths_keys = list(csv_paths.keys()) if isinstance(csv_paths, dict) else []
            debug = {
                "mamba_result_type": type(mamba_result).__name__,
                "has_csv_paths": csv_paths is not None,
                "has_artifacts": hasattr(mamba_result, "artifacts"),
                "has_iter_csv_paths": hasattr(mamba_result, "iter_csv_paths"),
                "csv_paths_keys": csv_paths_keys,
                "iter_csv_paths_preview": iter_csv_preview,
            }
            raise FileNotFoundError(f"StepB Mamba output path not found: {debug}")

        df = pd.read_csv(pred_path)
        if "Date" not in df.columns:
            raise ValueError("StepB Mamba output must include Date")

        mamba_col = next((c for c in df.columns if c == "Pred_Close_MAMBA"), None)
        if mamba_col is None:
            mamba_col = next((c for c in df.columns if c.lower() == "pred_close_mamba"), None)
        if mamba_col is None:
            raise ValueError("StepB Mamba output must include Pred_Close_MAMBA")

        out_df = df[["Date", mamba_col]].copy()
        out_df = out_df.rename(columns={mamba_col: "Pred_Close_MAMBA"})
        out_df = out_df[list(self.STEPB_PRED_TIME_ALL_COLUMNS)]
        out_df.to_csv(out_path, index=False, encoding="utf-8")
        return out_path

    def _write_live_nextday(self, symbol: str, run_mode: str) -> Optional[Path]:
        daily_dir = self._out_dir(run_mode) / "daily"
        files = sorted(daily_dir.glob(f"stepB_daily_pred_mamba_h01_{symbol}_*.csv"))
        if not files:
            return None
        src = files[-1]
        df = pd.read_csv(src)
        if "step_ahead_bdays" not in df.columns:
            return None
        row = df.loc[pd.to_numeric(df["step_ahead_bdays"], errors="coerce") == 1].head(1)
        if row.empty:
            return None
        out = row[["Date_anchor", "Date_target", "Pred_Close"]].copy()
        out_path = self._out_dir(run_mode) / f"stepB_pred_nextday_mamba_{symbol}.csv"
        out.to_csv(out_path, index=False, encoding="utf-8")
        return out_path

    def _write_live_future_periodic(self, symbol: str, cfg: WaveletMambaTrainConfig, prices_test_df: pd.DataFrame, periodic_df: pd.DataFrame, run_mode: str) -> Optional[Path]:
        prices = prices_test_df.copy()
        if "Date" in prices.columns:
            prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce")
            prices = prices.dropna(subset=["Date"]).sort_values("Date")
        if prices.empty or "Close" not in prices.columns:
            return None
        anchor_close = float(pd.to_numeric(prices["Close"], errors="coerce").dropna().iloc[-1])

        future_per = self._load_stepa_future_periodic(symbol, run_mode)
        fut_df = rollout_periodic_h1_future(
            app_config=self.app_config,
            symbol=symbol,
            periodic_history_df=periodic_df,
            periodic_future_df=future_per,
            cfg=cfg,
            anchor_close=anchor_close,
            horizon_days=63,
        )
        out_path = self._out_dir(run_mode) / f"stepB_pred_future_mamba_periodic_{symbol}.csv"
        fut_df.to_csv(out_path, index=False, encoding="utf-8")
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
        prices_test_df = self._load_stepa_split_df(symbol, run_mode, "prices", "test")
        tech_df = self._load_stepa_df(symbol, run_mode, "tech")
        periodic_df = self._load_stepa_df(symbol, run_mode, "periodic")
        features_df = tech_df.merge(periodic_df, on="Date", how="inner") if "Date" in tech_df.columns and "Date" in periodic_df.columns else tech_df

        forced_cfg = self._force_spec(cfg_all.mamba)
        full_cfg = replace(forced_cfg, variant="full", periodic_output_tag="mamba_periodic", enable_periodic_snapshots=True)
        periodic_cfg = replace(forced_cfg, variant="periodic", periodic_output_tag="mamba_periodic", enable_periodic_snapshots=True)

        full_res = run_stepB_mamba(
            app_config=self.app_config,
            symbol=symbol,
            prices_df=prices_df,
            features_df=features_df,
            cfg=full_cfg,
        )
        periodic_res = run_stepB_mamba(
            app_config=self.app_config,
            symbol=symbol,
            prices_df=prices_df,
            features_df=periodic_df,
            cfg=periodic_cfg,
        )
        pred_time_all_path = self._write_pred_time_all(symbol, run_mode, full_res)

        info = {}
        if run_mode == "live":
            nextday = self._write_live_nextday(symbol, run_mode)
            future = self._write_live_future_periodic(symbol, periodic_cfg, prices_test_df, periodic_df, run_mode)
            if nextday is not None:
                info["pred_nextday_mamba_path"] = str(nextday)
            if future is not None:
                info["pred_future_mamba_periodic_path"] = str(future)

        return StepBResult(
            success=bool(getattr(full_res, "success", True) and getattr(periodic_res, "success", True)),
            message=f"{getattr(full_res, 'message', '')} / {getattr(periodic_res, 'message', '')}",
            out_dir=str(self._out_dir(run_mode)),
            pred_time_all_path=str(pred_time_all_path),
            agent_results={"mamba": full_res, "mamba_periodic": periodic_res},
            info=info,
        )
