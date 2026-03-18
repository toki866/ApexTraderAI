from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional

import pandas as pd
import torch

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

    def _is_repo_cache_path(self, p: Path) -> bool:
        norm = str(p).replace("\\", "/").lower()
        return "/apex_repo_cache/" in norm and "/output" in norm

    def _timing(self) -> TimingLogger:
        t = getattr(self.app_config, "_timing_logger", None)
        return t if isinstance(t, TimingLogger) else TimingLogger.disabled()

    def _resolve_run_mode(self, cfg: StepBConfig) -> str:
        m = str(getattr(cfg.mamba, "mode", "sim") or "sim").strip().lower()
        if m in ("ops", "prod", "production"):
            return "live"
        return "sim" if m not in ("sim", "live") else m

    def _out_root(self) -> Path:
        candidate = Path(getattr(getattr(self.app_config, "data", None), "output_root", getattr(self.app_config, "output_root", "output")))
        if self._is_repo_cache_path(candidate):
            canonical = Path("C:/work/apex_work/output") if sys.platform.startswith("win") else Path("/mnt/c/work/apex_work/output")
            print(
                f"[StepB:output_root:FAIL] repo_cache_output_root_detected={candidate} canonical_output_base={canonical}",
                file=sys.stderr,
            )
            raise RuntimeError(
                "StepBService output_root points to repo cache output path; expected apex_work/output canonical root. "
                f"detected={candidate} canonical_base={canonical}"
            )
        return candidate

    def _out_dir(self, run_mode: str) -> Path:
        p = self._out_root() / "stepB" / run_mode
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _mode_dir(self, run_mode: str) -> Path:
        p = self._out_root() / "stepA" / run_mode
        if self._is_repo_cache_path(p):
            print(f"[StepB:mode_dir:FAIL] repo_cache_mode_dir_detected={p}", file=sys.stderr)
            raise RuntimeError(f"StepBService mode_dir points to repo cache output path: {p}")
        return p

    def _actual_device(self, *agent_results) -> str:
        for result in agent_results:
            candidate = getattr(result, "device_execution", None)
            if candidate:
                return str(candidate)
            info = getattr(result, "info", None)
            if isinstance(info, dict) and info.get("device_execution"):
                return str(info["device_execution"])
        return "cuda" if torch.cuda.is_available() else "cpu"

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
        pred_source = ""

        def _raise_stepb_fail(reason: str, message: str) -> None:
            tag = f"STEPB_FAIL_REASON={reason}"
            print(f"[StepB:pred_time_all:FAIL] {tag} {message}", file=sys.stderr)
            raise ValueError(f"{tag} {message}")

        def _normalize_date_series(values: pd.Series) -> pd.Series:
            parsed = pd.to_datetime(values, errors="coerce", utc=True)
            return parsed.dt.tz_localize(None).dt.normalize()

        def _extract_raw_pred_df(df_src: pd.DataFrame, src_name: str) -> tuple[pd.DataFrame, str]:
            date_col_local = "Date_target" if "Date_target" in df_src.columns else ("Date" if "Date" in df_src.columns else "")
            if not date_col_local:
                raise ValueError(f"pred source {src_name} missing Date/Date_target")

            mamba_col_local = next((c for c in df_src.columns if c == "Pred_Close_MAMBA"), None)
            if mamba_col_local is None:
                mamba_col_local = next((c for c in df_src.columns if c.lower() == "pred_close_mamba"), None)
            if mamba_col_local is None:
                mamba_col_local = next((c for c in df_src.columns if c == "Pred_Close_MAMBA_h01"), None)
            if mamba_col_local is None:
                mamba_col_local = next((c for c in df_src.columns if c.lower() == "pred_close_mamba_h01"), None)
            if mamba_col_local is None:
                mamba_col_local = next((c for c in df_src.columns if c == "Pred_Close"), None)
            if mamba_col_local is None:
                raise ValueError(f"pred source {src_name} missing Pred_Close_MAMBA/Pred_Close_MAMBA_h01/Pred_Close")

            pred_df = df_src[[date_col_local, mamba_col_local]].copy().rename(columns={date_col_local: "Date", mamba_col_local: "Pred_Close_MAMBA"})
            pred_df["Date"] = _normalize_date_series(pred_df["Date"])
            pred_df = pred_df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
            return pred_df, date_col_local

        csv_paths = getattr(mamba_result, "csv_paths", None)
        if isinstance(csv_paths, dict):
            for key in ("pred_close", "pred_close_path", "output_path"):
                value = csv_paths.get(key)
                if value:
                    pred_path = Path(value)
                    pred_source = f"csv_paths:{key}"
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
                    pred_source = "iter_csv_paths:pred_close_match"
                    break

        if pred_path is None:
            for k in ("pred_close", "pred_close_path", "output_path"):
                v = getattr(mamba_result, k, None)
                if v:
                    pred_path = Path(v)
                    pred_source = f"attr:{k}"
                    break
        if pred_path is None:
            artifacts = getattr(mamba_result, "artifacts", {}) or {}
            for key in ("pred_close", "pred_close_path", "output_path"):
                v = artifacts.get(key)
                if v:
                    pred_path = Path(v)
                    pred_source = f"artifacts:{key}"
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
        raw_pred_rows = int(len(df))
        chosen_date_col = ""
        try:
            raw_pred_df, chosen_date_col = _extract_raw_pred_df(df, pred_source or str(pred_path))
        except Exception as first_exc:
            fallback_pred_path = None
            fallback_pred_source = ""
            if isinstance(csv_paths, dict) and csv_paths.get("pred_path"):
                fallback_pred_path = Path(csv_paths["pred_path"])
                fallback_pred_source = "csv_paths:pred_path"
            elif getattr(mamba_result, "pred_close_path", None):
                fallback_pred_path = Path(getattr(mamba_result, "pred_close_path"))
                fallback_pred_source = "attr:pred_close_path"
            if fallback_pred_path is None or not fallback_pred_path.exists():
                _raise_stepb_fail("invalid_pred_source_csv", f"pred_close/pred_path load failed: {type(first_exc).__name__}:{first_exc}")
            df = pd.read_csv(fallback_pred_path)
            raw_pred_rows = int(len(df))
            raw_pred_df, chosen_date_col = _extract_raw_pred_df(df, fallback_pred_source or str(fallback_pred_path))
            pred_path = fallback_pred_path
            pred_source = fallback_pred_source or pred_source

        raw_non_null = int(pd.to_numeric(raw_pred_df["Pred_Close_MAMBA"], errors="coerce").notna().sum())
        raw_min_date = ""
        raw_max_date = ""
        if not raw_pred_df.empty:
            raw_min_date = str(raw_pred_df["Date"].min().date())
            raw_max_date = str(raw_pred_df["Date"].max().date())

        first_pred_date = raw_pred_df["Date"].min() if not raw_pred_df.empty else pd.NaT
        last_pred_date = raw_pred_df["Date"].max() if not raw_pred_df.empty else pd.NaT

        _aligned_to_test = False
        align_reason = ""
        stage1_error = ""
        stage2_error = ""
        test_dates = pd.DataFrame(columns=["Date"])
        try:
            test_df = self._load_stepa_split_df(symbol, run_mode, "prices", "test")
            if "Date" in test_df.columns:
                test_dates = test_df[["Date"]].copy()
                test_dates["Date"] = _normalize_date_series(test_dates["Date"])
                test_dates = test_dates.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
                out_df = test_dates.merge(raw_pred_df, on="Date", how="left")
                _aligned_to_test = True
                align_reason = "stepA_test_split"
        except Exception as e:
            stage1_error = f"stage1_failed:{type(e).__name__}:{e}"

        if not _aligned_to_test and run_mode != "sim":
            try:
                all_prices = self._load_stepa_df(symbol, run_mode, "prices")
                if "Date" in all_prices.columns:
                    all_dates = all_prices[["Date"]].copy()
                    all_dates["Date"] = _normalize_date_series(all_dates["Date"])
                    all_dates = all_dates.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
                    if not raw_pred_df.empty:
                        min_pred = raw_pred_df["Date"].min()
                        all_dates = all_dates.loc[all_dates["Date"] >= min_pred]
                    out_df = all_dates.merge(raw_pred_df, on="Date", how="left")
                    _aligned_to_test = True
                    align_reason = "stepA_combined_prices"
            except Exception as e:
                stage2_error = f"stage2_failed:{type(e).__name__}:{e}"

        if not _aligned_to_test:
            out_df = raw_pred_df.copy()
            if run_mode == "sim":
                align_reason = stage1_error or "stepA_test_split_unavailable"
            else:
                align_reason = ";".join(v for v in (stage1_error, stage2_error) if v) or "raw_stepB_dates"

        if run_mode == "sim":
            if test_dates.empty:
                _raise_stepb_fail("missing_stepA_test_split", f"StepA test split load failure: reason={align_reason or stage1_error or 'unknown'}")
            out_df = test_dates.merge(raw_pred_df, on="Date", how="left")
            _aligned_to_test = True
            if not align_reason:
                align_reason = "stepA_test_split"

        out_df["pred_close_mamba"] = pd.to_numeric(out_df["Pred_Close_MAMBA"].astype(str).str.replace(",", "", regex=False), errors="coerce")
        out_df["Pred_Close_MAMBA"] = out_df["pred_close_mamba"]
        out_df = out_df[list(self.STEPB_PRED_TIME_ALL_COLUMNS)]

        non_null_rows = int(out_df["pred_close_mamba"].notna().sum()) if len(out_df) > 0 else 0
        coverage = (float(non_null_rows) / float(len(out_df))) if len(out_df) > 0 else 0.0
        fallback_file_count = 0
        fallback_rebuilt_non_null_over_test = 0
        prediction_source_used = pred_source or "pred_close_csv"

        if len(out_df) > 0 and coverage <= 0.0:
            daily_dir = self._out_dir(run_mode) / "daily"
            daily_files = sorted(daily_dir.glob(f"stepB_daily_pred_mamba_h01_{symbol}_*.csv"))
            fallback_file_count = len(daily_files)
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
                if run_mode == "sim" and not test_dates.empty:
                    rebuilt = test_dates.merge(rebuilt[["Date", "Pred_Close_MAMBA", "pred_close_mamba"]], on="Date", how="left")
                    align_reason = f"{align_reason};daily_h1_rebuild_test_split" if align_reason else "daily_h1_rebuild_test_split"
                elif "Date" in out_df.columns and len(out_df) > 0:
                    rebuilt = out_df[["Date"]].merge(rebuilt[["Date", "Pred_Close_MAMBA", "pred_close_mamba"]], on="Date", how="left")
                    align_reason = f"{align_reason};daily_h1_rebuild" if align_reason else "daily_h1_rebuild"
                out_df = rebuilt[list(self.STEPB_PRED_TIME_ALL_COLUMNS)]
                non_null_rows = int(out_df["pred_close_mamba"].notna().sum()) if len(out_df) > 0 else 0
                fallback_rebuilt_non_null_over_test = non_null_rows
                coverage = (float(non_null_rows) / float(len(out_df))) if len(out_df) > 0 else 0.0
                prediction_source_used = "daily_h1_rebuild_fallback"

        merged_test_rows = int(len(out_df))
        if run_mode == "sim" and not test_dates.empty and merged_test_rows != len(test_dates):
            _raise_stepb_fail("merged_test_rows_mismatch", f"merged_test_rows mismatch: merged_test_rows={merged_test_rows} test_rows={len(test_dates)}")

        if run_mode == "sim" and not test_dates.empty:
            first_test_dt = test_dates["Date"].min()
            test_date_set = set(test_dates["Date"].tolist())
            out_date_set = set(out_df["Date"].dropna().tolist()) if "Date" in out_df.columns else set()
            unexpected_dates = sorted(out_date_set - test_date_set)
            train_side_dates = out_df.loc[out_df["Date"] < first_test_dt, "Date"] if "Date" in out_df.columns else pd.Series(dtype="datetime64[ns]")
            if unexpected_dates or not train_side_dates.empty:
                _raise_stepb_fail(
                    "train_side_dates_detected",
                    "non-test/train-side dates mixed before final write: "
                    f"unexpected_dates_count={len(unexpected_dates)} train_side_dates_count={int(train_side_dates.shape[0])} "
                    f"first_test_date={str(first_test_dt.date()) if not pd.isna(first_test_dt) else ''} "
                    f"min_out_date={str(out_df['Date'].min().date()) if 'Date' in out_df.columns and not out_df.empty else ''}",
                )

        diag = {
            "symbol": symbol,
            "run_mode": run_mode,
            "pred_source": pred_source or "unknown",
            "pred_path": str(pred_path),
            "chosen_date_col": chosen_date_col,
            "raw_pred_rows": raw_pred_rows,
            "raw_min_date": raw_min_date,
            "raw_max_date": raw_max_date,
            "first_pred_date": "" if pd.isna(first_pred_date) else str(pd.Timestamp(first_pred_date).date()),
            "last_pred_date": "" if pd.isna(last_pred_date) else str(pd.Timestamp(last_pred_date).date()),
            "first_test_date": "" if test_dates.empty else str(test_dates["Date"].min().date()),
            "last_test_date": "" if test_dates.empty else str(test_dates["Date"].max().date()),
            "merged_test_rows": merged_test_rows,
            "non_null_rows_over_test": int(non_null_rows),
            "coverage_ratio_over_test": float(coverage),
            "align_reason": align_reason or "raw_stepB_dates",
            "prediction_source_used": prediction_source_used,
            "fallback_file_count": int(fallback_file_count),
            "fallback_rebuilt_non_null_over_test": int(fallback_rebuilt_non_null_over_test),
        }
        print(
            "[StepB:pred_time_all:checkpoint] "
            f"pred_source={diag['pred_source']} pred_path={diag['pred_path']} chosen_date_col={diag['chosen_date_col']} "
            f"raw_pred_rows={diag['raw_pred_rows']} raw_min_date={diag['raw_min_date']} raw_max_date={diag['raw_max_date']} "
            f"first_pred_date={diag['first_pred_date']} last_pred_date={diag['last_pred_date']} "
            f"first_test_date={diag['first_test_date']} last_test_date={diag['last_test_date']} merged_test_rows={diag['merged_test_rows']} "
            f"non_null_rows_over_test={diag['non_null_rows_over_test']} coverage_ratio_over_test={diag['coverage_ratio_over_test']:.4f} "
            f"align_reason={diag['align_reason']} fallback_file_count={diag['fallback_file_count']} "
            f"fallback_rebuilt_non_null_over_test={diag['fallback_rebuilt_non_null_over_test']}"
        )
        print(
            "[StepB:pred_time_all:stats] "
            f"pred_close_rows={raw_pred_rows} pred_close_first_date={raw_min_date} pred_close_last_date={raw_max_date} "
            f"test_split_rows={len(test_dates)} overlap_rows={non_null_rows}"
        )

        if len(out_df) == 0 or coverage <= 0.0:
            _raise_stepb_fail("coverage_zero_after_test_alignment", f"coverage<=0.0 after test alignment: {diag}")

        try:
            out_df.to_csv(out_path, index=False, encoding="utf-8")
        except Exception as write_exc:
            _raise_stepb_fail("missing_pred_time_all", f"write_failed:{type(write_exc).__name__}:{write_exc} path={out_path}")
        if (not out_path.exists()) or out_path.stat().st_size <= 0:
            _raise_stepb_fail("missing_pred_time_all", f"write_succeeded_but_file_missing_or_empty path={out_path}")
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
            print("[StepB] load_stepa_inputs begin")
            with timing.stage("stepB.load_stepA_inputs"):
                prices_df = self._load_stepa_df(symbol, run_mode, "prices")
                prices_test_df = self._load_stepa_split_df(symbol, run_mode, "prices", "test")
                tech_df = self._load_stepa_df(symbol, run_mode, "tech")
                periodic_df = self._load_stepa_df(symbol, run_mode, "periodic")
                features_df = tech_df.merge(periodic_df, on="Date", how="inner") if "Date" in tech_df.columns and "Date" in periodic_df.columns else tech_df
            print("[StepB] load_stepa_inputs ok")

            forced_cfg = self._force_spec(cfg_all.mamba)
            full_cfg = replace(
                forced_cfg,
                variant="full",
                mode=run_mode,
                train_start=getattr(cfg_all.mamba, "train_start", None),
                train_end=getattr(cfg_all.mamba, "train_end", None),
                test_start=getattr(cfg_all.mamba, "test_start", None),
                test_end=getattr(cfg_all.mamba, "test_end", None),
                date_range=getattr(cfg_all, "date_range", None),
                use_wavelet=bool(getattr(cfg_all.mamba, "use_wavelet", True)),
                periodic_use_wavelet=bool(getattr(cfg_all.mamba, "periodic_use_wavelet", False)),
                periodic_output_tag="mamba_periodic",
                enable_periodic_snapshots=True,
            )
            periodic_cfg = replace(
                forced_cfg,
                variant="periodic",
                mode=run_mode,
                train_start=getattr(cfg_all.mamba, "train_start", None),
                train_end=getattr(cfg_all.mamba, "train_end", None),
                test_start=getattr(cfg_all.mamba, "test_start", None),
                test_end=getattr(cfg_all.mamba, "test_end", None),
                date_range=getattr(cfg_all, "date_range", None),
                use_wavelet=bool(getattr(cfg_all.mamba, "use_wavelet", True)),
                periodic_use_wavelet=bool(getattr(cfg_all.mamba, "periodic_use_wavelet", False)),
                periodic_output_tag="mamba_periodic",
                enable_periodic_snapshots=True,
            )

            print("[StepB] full.run begin")
            with timing.stage("stepB.full.run"):
                with timing.stage("stepB.train_full"):
                    pass
                with timing.stage("stepB.predict_full"):
                    full_res = run_stepB_mamba(
                        app_config=self.app_config,
                        symbol=symbol,
                        prices_df=prices_df,
                        features_df=features_df,
                        cfg=full_cfg,
                        timing_logger=timing,
                        timing_stage_prefix="stepB.full",
                    )
            print("[StepB] full.run ok")

            print("[StepB] periodic.run begin")
            with timing.stage("stepB.periodic.run"):
                with timing.stage("stepB.train_periodic"):
                    pass
                with timing.stage("stepB.predict_periodic"):
                    periodic_res = run_stepB_mamba(
                        app_config=self.app_config,
                        symbol=symbol,
                        prices_df=prices_df,
                        features_df=periodic_df,
                        cfg=periodic_cfg,
                        timing_logger=timing,
                        timing_stage_prefix="stepB.periodic",
                    )
            print("[StepB] periodic.run ok")
            print("[StepB] write_pred_time_all begin")
            try:
                with timing.stage("stepB.write_outputs"):
                    with timing.stage("stepB.write_pred_time_all"):
                        pred_time_all_path = self._write_pred_time_all(symbol, run_mode, full_res)
                print("[StepB] write_pred_time_all ok")
            except Exception as write_exc:
                reason = "missing_pred_time_all"
                msg = str(write_exc)
                if "STEPB_FAIL_REASON=" in msg:
                    reason = msg.split("STEPB_FAIL_REASON=", 1)[1].split()[0].strip()
                print(f"[StepB] write_pred_time_all fail reason={reason}", file=sys.stderr)
                print(f"STEPB_FAIL_REASON={reason}", file=sys.stderr)
                raise

            print("[StepB] ensure_pred_time_all begin")
            if (not Path(pred_time_all_path).exists()) or Path(pred_time_all_path).stat().st_size <= 0:
                print("[StepB] ensure_pred_time_all fail reason=missing_pred_time_all", file=sys.stderr)
                print("STEPB_FAIL_REASON=missing_pred_time_all", file=sys.stderr)
                raise FileNotFoundError(f"STEPB_FAIL_REASON=missing_pred_time_all path={pred_time_all_path}")
            print("[StepB] ensure_pred_time_all ok")

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
            stepb_dir = self._out_dir(run_mode)
            feature_contract = {
                "prices_rows": int(len(prices_df)),
                "prices_test_rows": int(len(prices_test_df)),
                "tech_rows": int(len(tech_df)),
                "periodic_rows": int(len(periodic_df)),
                "full_feature_dim": int(len([c for c in features_df.columns if c != "Date"])),
                "periodic_feature_dim": int(len([c for c in periodic_df.columns if c != "Date"])),
                "full_feature_columns_preview": [c for c in features_df.columns if c != "Date"][:12],
                "periodic_feature_columns_preview": [c for c in periodic_df.columns if c != "Date"][:12],
            }
            if feature_contract["full_feature_dim"] <= 0 or feature_contract["periodic_feature_dim"] <= 0:
                raise RuntimeError(f"StepB feature_dim mismatch: {feature_contract}")
            summary = {
                "symbol": symbol,
                "mode": run_mode,
                "output_root": str(self._out_root()),
                "device_execution": self._actual_device(full_res, periodic_res),
                "pred_time_all_path": str(pred_time_all_path),
                "prediction_paths": {
                    "full": str(getattr(full_res, "pred_close_path", "")),
                    "periodic": str(getattr(periodic_res, "pred_close_path", "")),
                },
                "feature_contract": feature_contract,
                "model_variants": {
                    "full": "fusion",
                    "periodic": "periodic_only",
                },
            }
            (stepb_dir / f"stepB_summary_{symbol}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            audit_dir = self._out_root() / "audit" / run_mode
            audit_dir.mkdir(parents=True, exist_ok=True)
            (audit_dir / f"stepB_audit_{symbol}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        return StepBResult(
            success=bool(getattr(full_res, "success", True) and getattr(periodic_res, "success", True)),
            message=f"{getattr(full_res, 'message', '')} / {getattr(periodic_res, 'message', '')}",
            out_dir=str(self._out_dir(run_mode)),
            pred_time_all_path=str(pred_time_all_path),
            agent_results={"mamba": full_res, "mamba_periodic": periodic_res},
            info=info,
        )
