from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from ai_core.config.app_config import AppConfig
from ai_core.config.step_b_config import StepBConfig
from ai_core.services.step_b_mamba_runner import run_stepB_mamba
from ai_core.types.step_b_types import StepBResult


class StepBService:
    """Mamba-only StepB service."""

    STEPB_PRED_TIME_ALL_COLUMNS = ("Date", "Pred_Close_MAMBA")

    def __init__(self, app_config: AppConfig) -> None:
        self.app_config = app_config

    def _resolve_run_mode(self, cfg: StepBConfig) -> str:
        raw_mode = str(getattr(cfg.mamba, "mode", "") or getattr(cfg, "run_mode", "sim") or "sim").strip().lower()
        if raw_mode in ("ops", "prod", "production"):
            return "live"
        if raw_mode in ("sim", "live", "display"):
            return raw_mode
        return "sim"

    def _out_dir(self, run_mode: str) -> Path:
        out_root = Path(getattr(getattr(self.app_config, "data", None), "output_root", getattr(self.app_config, "output_root", "output")))
        p = out_root / "stepB" / run_mode
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _stepa_dir(self, run_mode: str) -> Path:
        out_root = Path(getattr(getattr(self.app_config, "data", None), "output_root", getattr(self.app_config, "output_root", "output")))
        return out_root / "stepA" / run_mode

    def _normalize_date_df(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "Date" in out.columns:
            out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
            out = out.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
        return out.reset_index(drop=True)

    def _load_stepa_df(self, symbol: str, run_mode: str, kind: str) -> pd.DataFrame:
        mode_dir = self._stepa_dir(run_mode)

        if run_mode in ("sim", "live"):
            train_p = mode_dir / f"stepA_{kind}_train_{symbol}.csv"
            test_p = mode_dir / f"stepA_{kind}_test_{symbol}.csv"
            if not train_p.exists() or train_p.stat().st_size <= 0:
                raise FileNotFoundError(f"Missing StepA train {kind} CSV for {symbol}: {train_p}")
            if not test_p.exists() or test_p.stat().st_size <= 0:
                raise FileNotFoundError(f"Missing StepA test {kind} CSV for {symbol}: {test_p}")
            tr = pd.read_csv(train_p)
            te = pd.read_csv(test_p)
            return self._normalize_date_df(pd.concat([tr, te], axis=0, ignore_index=True))

        display_p = mode_dir / f"stepA_{kind}_{symbol}.csv"
        if display_p.exists() and display_p.stat().st_size > 0:
            return self._normalize_date_df(pd.read_csv(display_p))
        raise FileNotFoundError(f"Missing StepA {kind} CSV for {symbol}: {display_p}")

    def _result_csv_path(self, result: Any, key: str) -> Path:
        csv_paths = getattr(result, "csv_paths", {}) or {}
        value = csv_paths.get(key)
        if value:
            return Path(value)
        raise FileNotFoundError(f"StepB result missing csv_paths['{key}']")

    def _write_pred_time_all(self, symbol: str, run_mode: str, mamba_result: Any) -> Path:
        stepb_dir = self._out_dir(run_mode)
        out_path = stepb_dir / f"stepB_pred_time_all_{symbol}.csv"

        pred_path = self._result_csv_path(mamba_result, "pred_close")
        if not pred_path.exists():
            raise FileNotFoundError(f"pred_close path missing: {pred_path}")

        df = pd.read_csv(pred_path)
        if "Date" not in df.columns:
            raise ValueError("StepB Mamba output must include Date")

        mamba_col = next((c for c in df.columns if c == "Pred_Close_MAMBA"), None)
        if mamba_col is None:
            mamba_col = next((c for c in df.columns if c.lower() == "pred_close_mamba"), None)
        if mamba_col is None:
            raise ValueError("StepB Mamba output must include Pred_Close_MAMBA")

        out_df = df[["Date", mamba_col]].copy().rename(columns={mamba_col: "Pred_Close_MAMBA"})
        out_df = out_df[list(self.STEPB_PRED_TIME_ALL_COLUMNS)]
        out_df.to_csv(out_path, index=False, encoding="utf-8")
        return out_path

    def _write_pred_time_variant(
        self,
        symbol: str,
        run_mode: str,
        variant: str,
        pred_close_path: Path,
        prices_df: pd.DataFrame,
    ) -> Path:
        if not pred_close_path.exists():
            raise FileNotFoundError(f"Missing pred_close for {variant}: {pred_close_path}")
        src = pd.read_csv(pred_close_path)
        col_prefix = "Pred_Close_MAMBA" if variant == "full" else "Pred_Close_MAMBA_PERIODIC"
        out = pd.DataFrame({
            "Date": src["Date"],
            "Close_pred_h1": src[f"{col_prefix}_h01"],
            "Close_pred_h5": src[f"{col_prefix}_h05"],
            "Close_pred_h10": src[f"{col_prefix}_h10"],
            "Close_pred_h20": src[f"{col_prefix}_h20"],
        })
        if "Date" in prices_df.columns and "Close" in prices_df.columns:
            t = prices_df[["Date", "Close"]].copy()
            t["Date"] = pd.to_datetime(t["Date"], errors="coerce")
            out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
            out = out.merge(t, on="Date", how="left").rename(columns={"Close": "Close_true"})
            out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")

        out_path = self._out_dir(run_mode) / f"stepB_pred_time_{variant}_{symbol}.csv"
        out.to_csv(out_path, index=False, encoding="utf-8")
        return out_path

    def _read_stepa_split_summary(self, symbol: str, run_mode: str) -> Dict[str, Any]:
        path = self._stepa_dir(run_mode) / f"stepA_split_summary_{symbol}.csv"
        if not path.exists() or path.stat().st_size <= 0:
            return {}
        df = pd.read_csv(path)
        if set(["key", "value"]).issubset(df.columns):
            return {str(r["key"]): r["value"] for _, r in df.iterrows()}
        return {}

    def _write_split_summary(
        self,
        symbol: str,
        run_mode: str,
        cfg_all: StepBConfig,
        stepa_summary: Dict[str, Any],
    ) -> Path:
        train_start = stepa_summary.get("train_start", getattr(cfg_all.mamba, "train_start", ""))
        train_end = stepa_summary.get("train_end", getattr(cfg_all.mamba, "train_end", ""))
        test_start = stepa_summary.get("test_start", getattr(cfg_all.mamba, "test_start", ""))
        test_end = stepa_summary.get("test_end", getattr(cfg_all.mamba, "test_end", ""))
        horizons = [1, 5, 10, 20]
        rows = []
        for variant, fset in (("full", "periodic+tech+prices"), ("periodic", "periodic44")):
            rows.append({
                "mode": run_mode,
                "symbol": symbol,
                "variant": variant,
                "backend": getattr(cfg_all.mamba, "backend", "mamba"),
                "lookback": int(getattr(cfg_all.mamba, "lookback_days", 128)),
                "horizons": ",".join(str(h) for h in horizons),
                "target_mode": getattr(cfg_all.mamba, "target_mode", "close" if variant == "full" else "logret"),
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "feature_set": fset,
                "seed": int(getattr(cfg_all.mamba, "seed", 42)),
                "epochs": int(getattr(cfg_all.mamba, "epochs", getattr(cfg_all.mamba, "num_epochs", 0))),
                "hidden_dim": int(getattr(cfg_all.mamba, "hidden_dim", 0)),
                "num_layers": int(getattr(cfg_all.mamba, "num_layers", 0)),
            })
        out = pd.DataFrame(rows)
        path = self._out_dir(run_mode) / f"stepB_split_summary_{symbol}.csv"
        out.to_csv(path, index=False, encoding="utf-8")
        return path

    def _write_live_nextday_full(self, symbol: str, run_mode: str, full_result: Any) -> Path:
        pred_path = self._result_csv_path(full_result, "pred_path")
        df = pd.read_csv(pred_path)
        if len(df) <= 0:
            raise RuntimeError("stepB_pred_path_mamba is empty")
        row = df.iloc[-1]
        anchor_date = pd.to_datetime(row["Date_anchor"], errors="coerce")
        if pd.isna(anchor_date):
            raise ValueError("Date_anchor missing in stepB_pred_path_mamba")
        out = pd.DataFrame([
            {
                "AnchorDate": anchor_date.strftime("%Y-%m-%d"),
                "NextDate": (anchor_date + pd.tseries.offsets.BDay(1)).strftime("%Y-%m-%d"),
                "Close_pred_h1": float(row["Pred_Close_t_plus_01"]),
            }
        ])
        out_path = self._out_dir(run_mode) / f"stepB_pred_nextday_full_{symbol}.csv"
        out.to_csv(out_path, index=False, encoding="utf-8")
        return out_path

    def _validate_live_periodic_future(self, symbol: str, run_mode: str, periodic_result: Any, required_rows: int = 63) -> Path:
        src = self._result_csv_path(periodic_result, "pred_future_periodic")
        if not src.exists():
            raise FileNotFoundError(f"Missing periodic future prediction file: {src}")
        df = pd.read_csv(src)
        if len(df) < required_rows:
            raise RuntimeError(f"stepB_pred_future_periodic insufficient rows: {len(df)} < {required_rows}")
        dst = self._out_dir(run_mode) / f"stepB_pred_future_periodic_{symbol}.csv"
        if src.resolve() != dst.resolve():
            df.to_csv(dst, index=False, encoding="utf-8")
        return dst

    def _write_rollout63_full(self, symbol: str, run_mode: str, pred_time_full_path: Path, stepa_summary: Dict[str, Any]) -> Path:
        test_start = pd.to_datetime(stepa_summary.get("test_start"), errors="coerce")
        test_end = pd.to_datetime(stepa_summary.get("test_end"), errors="coerce")
        if pd.isna(test_start) or pd.isna(test_end):
            raise RuntimeError("stepA_split_summary test_start/test_end required for rollout63")
        full_df = pd.read_csv(pred_time_full_path)
        full_df["Date"] = pd.to_datetime(full_df["Date"], errors="coerce")
        sliced = full_df[(full_df["Date"] >= test_start) & (full_df["Date"] <= test_end)].copy()
        out = sliced[["Date", "Close_pred_h1"]].rename(columns={"Close_pred_h1": "Close_pred"})
        out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
        out_path = self._out_dir(run_mode) / f"stepB_rollout63_full_{symbol}.csv"
        out.to_csv(out_path, index=False, encoding="utf-8")
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

        features_full = tech_df.merge(periodic_df, on="Date", how="inner")
        cfg_full = deepcopy(cfg_all.mamba)
        cfg_full.variant = "full"

        cfg_periodic = deepcopy(cfg_all.mamba)
        cfg_periodic.variant = "periodic"
        cfg_periodic.periodic_snapshot_horizons = (1, 5, 10, 20)

        full_result = run_stepB_mamba(self.app_config, symbol, prices_df, features_full, cfg_full)
        periodic_result = run_stepB_mamba(self.app_config, symbol, prices_df, periodic_df, cfg_periodic)
        if not bool(getattr(full_result, "success", False)) or not bool(getattr(periodic_result, "success", False)):
            raise RuntimeError("StepB full/periodic both must succeed")

        pred_time_all_path = self._write_pred_time_all(symbol, run_mode, full_result)
        pred_time_full_path = self._write_pred_time_variant(symbol, run_mode, "full", self._result_csv_path(full_result, "pred_close"), prices_df)
        pred_time_periodic_path = self._write_pred_time_variant(symbol, run_mode, "periodic", self._result_csv_path(periodic_result, "pred_close"), prices_df)

        stepa_summary = self._read_stepa_split_summary(symbol, run_mode)
        split_summary_path = self._write_split_summary(symbol, run_mode, cfg_all, stepa_summary)

        pred_nextday_full_path = ""
        pred_future_periodic_path = ""
        rollout63_full_path = ""

        if run_mode == "live":
            pred_nextday_full_path = str(self._write_live_nextday_full(symbol, run_mode, full_result))
            pred_future_periodic_path = str(self._validate_live_periodic_future(symbol, run_mode, periodic_result, required_rows=int(getattr(cfg_all.mamba, "live_future_bdays", 63))))

        if run_mode == "sim" and bool(getattr(cfg_all.mamba, "sim_rollout63", False)):
            rollout63_full_path = str(self._write_rollout63_full(symbol, run_mode, pred_time_full_path, stepa_summary))

        return StepBResult(
            success=True,
            message="mamba ok (full+periodic)",
            out_dir=str(self._out_dir(run_mode)),
            pred_time_all_path=str(pred_time_all_path),
            split_summary_path=str(split_summary_path),
            pred_time_full_path=str(pred_time_full_path),
            pred_time_periodic_path=str(pred_time_periodic_path),
            pred_future_periodic_path=pred_future_periodic_path,
            pred_nextday_full_path=pred_nextday_full_path,
            rollout63_full_path=rollout63_full_path,
            agent_results={"mamba_full": full_result, "mamba_periodic": periodic_result},
        )
