from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional

import pandas as pd

from ai_core.config.app_config import AppConfig
from ai_core.config.step_b_config import StepBConfig, WaveletMambaTrainConfig
from ai_core.services.step_b_mamba_runner import rollout_periodic_h1_future, run_stepB_mamba
from ai_core.types.step_b_types import StepBResult
from ai_core.utils.timing_logger import TimingLogger


class StepBService:
    """Mamba-only StepB service."""

    STEPB_PRED_TIME_ALL_COLUMNS = ("Date", "Pred_Close_MAMBA", "pred_close_mamba")

    def __init__(self, app_config: AppConfig) -> None:
        self.app_config = app_config

    def _timing(self) -> TimingLogger:
        t = getattr(self.app_config, "_timing_logger", None)
        return t if isinstance(t, TimingLogger) else TimingLogger.disabled()

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

        # Prefer Date_target when available. The runner can emit both Date (anchor) and
        # Date_target (prediction target); coverage against StepA test dates must use target dates.
        date_col = "Date_target" if "Date_target" in df.columns else None
        if date_col is None and "Date" in df.columns:
            date_col = "Date"
        if date_col is None:
            raise ValueError("StepB Mamba output must include Date or Date_target")

        mamba_col = next((c for c in df.columns if c == "Pred_Close_MAMBA"), None)
        if mamba_col is None:
            mamba_col = next((c for c in df.columns if c.lower() == "pred_close_mamba"), None)
        if mamba_col is None:
            mamba_col = next((c for c in df.columns if c == "Pred_Close_MAMBA_h01"), None)
        if mamba_col is None:
            mamba_col = next((c for c in df.columns if c.lower() == "pred_close_mamba_h01"), None)
        if mamba_col is None:
            raise ValueError("StepB Mamba output must include Pred_Close_MAMBA/Pred_Close_MAMBA_h01")

        def _normalize_date_series(values: pd.Series) -> pd.Series:
            # timezone-safe normalize: parse in UTC, drop timezone, then normalize to date boundary.
            parsed = pd.to_datetime(values, errors="coerce", utc=True)
            return parsed.dt.tz_localize(None).dt.normalize()

        out_df = df[[date_col, mamba_col]].copy().rename(columns={date_col: "Date"})
        out_df["Date"] = _normalize_date_series(out_df["Date"])
        out_df = out_df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
        out_df = out_df.rename(columns={mamba_col: "Pred_Close_MAMBA"})

        # Align to StepA test dates so downstream evaluators see explicit test-window coverage.
        _aligned_to_test = False
        align_reason = ""
        # Stage-1 fallback: canonical StepA test split alignment.
        try:
            test_df = self._load_stepa_split_df(symbol, run_mode, "prices", "test")
            if "Date" in test_df.columns:
                test_dates = test_df[["Date"]].copy()
                test_dates["Date"] = _normalize_date_series(test_dates["Date"])
                test_dates = test_dates.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
                out_df = test_dates.merge(out_df, on="Date", how="left")
                _aligned_to_test = True
                align_reason = "stepA_test_split"
        except Exception as e:
            align_reason = f"stage1_failed:{type(e).__name__}"

        # Stage-2 fallback: align to the StepA combined prices calendar when split test load failed.
        if not _aligned_to_test:
            try:
                all_prices = self._load_stepa_df(symbol, run_mode, "prices")
                if "Date" in all_prices.columns:
                    all_dates = all_prices[["Date"]].copy()
                    all_dates["Date"] = _normalize_date_series(all_dates["Date"])
                    all_dates = all_dates.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
                    # Keep prediction window only to avoid unbounded expansion.
                    if not out_df.empty:
                        min_pred = out_df["Date"].min()
                        all_dates = all_dates.loc[all_dates["Date"] >= min_pred]
                    out_df = all_dates.merge(out_df, on="Date", how="left")
                    _aligned_to_test = True
                    align_reason = "stepA_combined_prices"
            except Exception as e:
                align_reason = f"{align_reason};stage2_failed:{type(e).__name__}" if align_reason else f"stage2_failed:{type(e).__name__}"

        print(
            f"[StepB:pred_time_all] symbol={symbol} mode={run_mode} "
            f"aligned_to_test={_aligned_to_test} reason={align_reason or 'raw_stepB_dates'} rows={len(out_df)}"
        )

        out_df["pred_close_mamba"] = pd.to_numeric(
            out_df["Pred_Close_MAMBA"].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )
        out_df["Pred_Close_MAMBA"] = out_df["pred_close_mamba"]
        out_df = out_df[list(self.STEPB_PRED_TIME_ALL_COLUMNS)]

        coverage = float(out_df["pred_close_mamba"].notna().mean()) if len(out_df) > 0 else 0.0
        if len(out_df) > 0 and coverage <= 0.0:
            # Fallback: rebuild from daily H=1 target-date files when pred_close alignment produced no
            # valid test-window predictions. This preserves strict "coverage must be > 0" behavior while
            # avoiding false negatives caused by endpoint CSV date mismatches.
            daily_dir = self._out_dir(run_mode) / "daily"
            daily_files = sorted(daily_dir.glob(f"stepB_daily_pred_mamba_h01_{symbol}_*.csv"))
            rows = []
            for fp in daily_files:
                try:
                    dfd = pd.read_csv(fp)
                except Exception:
                    continue
                if {"Date_target", "Pred_Close"}.issubset(dfd.columns):
                    sub = dfd[["Date_target", "Pred_Close"]].copy().rename(columns={"Date_target": "Date", "Pred_Close": "Pred_Close_MAMBA"})
                    rows.append(sub)
            if rows:
                rebuilt = pd.concat(rows, axis=0, ignore_index=True)
                rebuilt["Date"] = pd.to_datetime(rebuilt["Date"], errors="coerce").dt.normalize()
                rebuilt = rebuilt.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
                rebuilt["pred_close_mamba"] = pd.to_numeric(rebuilt["Pred_Close_MAMBA"], errors="coerce")
                rebuilt["Pred_Close_MAMBA"] = rebuilt["pred_close_mamba"]
                if "Date" in out_df.columns and len(out_df) > 0:
                    rebuilt = out_df[["Date"]].merge(rebuilt[["Date", "Pred_Close_MAMBA", "pred_close_mamba"]], on="Date", how="left")
                out_df = rebuilt[list(self.STEPB_PRED_TIME_ALL_COLUMNS)]
                coverage = float(out_df["pred_close_mamba"].notna().mean()) if len(out_df) > 0 else 0.0

        if len(out_df) == 0 or coverage <= 0.0:
            raise ValueError(f"StepB pred_time_all invalid test coverage: rows={len(out_df)} coverage_ratio_over_test={coverage:.4f}")

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

        timing = self._timing()
        with timing.stage("stepB.total"):
            cfg_all = StepBConfig.from_any(stepb_config)
            if not cfg_all.enabled_agents():
                raise ValueError("No agents enabled in StepBConfig (mamba disabled).")

            symbol = cfg_all.symbol
            run_mode = self._resolve_run_mode(cfg_all)
            with timing.stage("stepB.load_stepA_inputs"):
                prices_df = self._load_stepa_df(symbol, run_mode, "prices")
                prices_test_df = self._load_stepa_split_df(symbol, run_mode, "prices", "test")
                tech_df = self._load_stepa_df(symbol, run_mode, "tech")
                periodic_df = self._load_stepa_df(symbol, run_mode, "periodic")
                features_df = tech_df.merge(periodic_df, on="Date", how="inner") if "Date" in tech_df.columns and "Date" in periodic_df.columns else tech_df

            forced_cfg = self._force_spec(cfg_all.mamba)
            full_cfg = replace(forced_cfg, variant="full", periodic_output_tag="mamba_periodic", enable_periodic_snapshots=True)
            periodic_cfg = replace(forced_cfg, variant="periodic", periodic_output_tag="mamba_periodic", enable_periodic_snapshots=True)

            with timing.stage("stepB.full.run"):
                full_res = run_stepB_mamba(
                    app_config=self.app_config,
                    symbol=symbol,
                    prices_df=prices_df,
                    features_df=features_df,
                    cfg=full_cfg,
                    timing_logger=timing,
                    timing_stage_prefix="stepB.full",
                )

            with timing.stage("stepB.periodic.run"):
                periodic_res = run_stepB_mamba(
                    app_config=self.app_config,
                    symbol=symbol,
                    prices_df=prices_df,
                    features_df=periodic_df,
                    cfg=periodic_cfg,
                    timing_logger=timing,
                    timing_stage_prefix="stepB.periodic",
                )
            with timing.stage("stepB.write_pred_time_all"):
                pred_time_all_path = self._write_pred_time_all(symbol, run_mode, full_res)

            info = {}
            nextday = None
            future = None
            if run_mode == "live":
                with timing.stage("stepB.write_live_nextday"):
                    nextday = self._write_live_nextday(symbol, run_mode)
                with timing.stage("stepB.write_live_future"):
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
