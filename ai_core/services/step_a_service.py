# -*- coding: utf-8 -*-
"""
StepAService (v7: SAFE - NO combined stepA_features_*.csv)

Purpose
-------
You requested to delete (stop generating) any combined feature CSV:
  - stepA_features_train_<SYMBOL>.csv
  - stepA_features_test_<SYMBOL>.csv
  - stepA_features_<SYMBOL>.csv
because these contain (periodic + technical + price) together and can be
accidentally used for training.

This v7 keeps:
- separated outputs only: prices / technicals(+BNF) / periodic
- sim/live/display folder rules preserved
- legacy combined files are purged automatically in each mode folder

Folders
-------
Save to: <output_root>/stepA/<mode>/   (mode in {sim, live, display})

Outputs (sim/live)
------------------
- stepA_prices_train_<SYMBOL>.csv
- stepA_prices_test_<SYMBOL>.csv
- stepA_tech_train_<SYMBOL>.csv
- stepA_tech_test_<SYMBOL>.csv
- stepA_periodic_train_<SYMBOL>.csv
- stepA_periodic_test_<SYMBOL>.csv
- stepA_split_summary_<SYMBOL>.csv

Outputs (display)
-----------------
- stepA_prices_<SYMBOL>.csv      (observed only: train+test)
- stepA_tech_<SYMBOL>.csv        (observed only: train+test)
- stepA_periodic_<SYMBOL>.csv    (observed + optional future periodic rows)
- stepA_split_summary_<SYMBOL>.csv

Notes
-----
- Date is kept as a column (NOT index).
- Periodic features are generated on US trading sessions when possible (holiday-aware).
  If no market calendar package is installed, it falls back to weekdays (Mon-Fri).
- live/display can extend periodic generation up to (latest_date + future_months),
  but prices/tech outputs remain observed-only (no NaN future rows).

Compatibility
-------------
- Periodic column names are now **explicit** (still 44 columns total):
  per_cal_* (calendar), per_astro_* (moon/solstice), per_planet_* (retro flag/speed), per_h2_*/per_h3_* (harmonics)
- Technical column names remain: Gap, ATR_norm, RSI, MACD, MACD_signal
  BNF: BNF_RVOL20, BNF_VolZ20, BNF_Return1, BNF_BodyPct, BNF_RangePct,
       BNF_DivDownVolUp, BNF_EnergyFade, BNF_PanicScore
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import math
import os
import numpy as np
import pandas as pd

from ai_core.utils.paths import get_repo_root


@dataclass
class _Cfg:
    output_root: str = "output"
    data_dir: str = "data"
    data_root: str = "data"


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _to_dt(v: Any) -> Optional[pd.Timestamp]:
    if v is None:
        return None
    try:
        return pd.to_datetime(v)
    except Exception:
        return None


def _norm_date(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.to_datetime(ts).normalize()


def snap_prev_by_prices(date: Any, available_dates_sorted: Any) -> pd.Timestamp:
    """Snap to previous available trading date (floor). If none, use min."""
    if available_dates_sorted is None or len(available_dates_sorted) == 0:
        return _norm_date(pd.to_datetime(date))
    ad = pd.Series(pd.to_datetime(available_dates_sorted, errors="coerce")).dropna().sort_values().drop_duplicates().reset_index(drop=True)
    if len(ad) == 0:
        return _norm_date(pd.to_datetime(date))
    d = _norm_date(pd.to_datetime(date))
    idx = ad.searchsorted(d, side="right") - 1
    if idx < 0:
        return _norm_date(pd.to_datetime(ad.iloc[0]))
    return _norm_date(pd.to_datetime(ad.iloc[int(idx)]))


class StepAService:
    def __init__(self, app_config: Any = None, /, **kwargs: Any):
        cfg = kwargs.get("app_config", app_config)
        if isinstance(cfg, dict) and "app_config" in cfg:
            cfg = cfg.get("app_config")
        cfg_data_dir = _get_attr(cfg, "data_dir", None)
        if cfg_data_dir is None:
            cfg_data_dir = _get_attr(cfg, "data_root", None)
        if cfg_data_dir is None:
            cfg_data = _get_attr(cfg, "data", None)
            cfg_data_dir = _get_attr(cfg_data, "data_dir", None)
        if cfg_data_dir is None:
            cfg_data = _get_attr(cfg, "data", None)
            cfg_data_dir = _get_attr(cfg_data, "data_root", None)
        resolved_data_dir = str(cfg_data_dir if cfg_data_dir is not None else kwargs.get("data_dir", kwargs.get("data_root", "data")))
        self.cfg = _Cfg(
            output_root=str(_get_attr(cfg, "output_root", kwargs.get("output_root", "output"))),
            data_dir=resolved_data_dir,
            data_root=resolved_data_dir,
        )

    # -------------------------
    # Public API
    # -------------------------
    def run(self, symbol: str, date_range: Any = None, **kwargs: Any) -> Dict[str, Any]:
        # Resolve mode (support legacy "ops" as alias to "live")
        mode_raw = self._resolve_mode(date_range=date_range, kwargs=kwargs)
        mode = self._normalize_mode(mode_raw)

        out_dir_mode = Path(self.cfg.output_root) / "stepA" / mode
        out_dir_mode.mkdir(parents=True, exist_ok=True)

        # Purge legacy combined files to prevent accidental training
        self._purge_combined_files(out_dir_mode, symbol)

        src_csv = self._resolve_src_csv(symbol=symbol, date_range=date_range, kwargs=kwargs)
        resolved_data_dir = self._configured_data_root(kwargs=kwargs).resolve()
        print(f"[StepA] data_dir={resolved_data_dir} src_csv={src_csv.resolve()}")
        if not src_csv.exists():
            data_dir_abs = resolved_data_dir
            configured_data_dir = (
                kwargs.get("data_dir", None)
                or kwargs.get("data_root", None)
                or _get_attr(self.cfg, "data_dir", None)
                or _get_attr(self.cfg, "data_root", None)
            )
            searched = self._list_price_csv_candidates(symbol=symbol, date_range=date_range, kwargs=kwargs)
            searched_text = "\n".join(f"  - {p.resolve()}" for p in searched)
            raise FileNotFoundError(
                "StepA: missing source price CSV.\n"
                f"expected_abs_path={src_csv.resolve()}\n"
                f"config_data_dir={configured_data_dir}\n"
                f"data_dir_abs={data_dir_abs}\n"
                "searched_abs_paths:\n"
                f"{searched_text}\n"
                "resolution: run tools/prepare_data.py with the SAME --data-dir passed to tools/run_pipeline.py, "
                "or review --data-dir/AppConfig.data_dir so StepA reads the correct location"
            )

        df = pd.read_csv(src_csv)
        df = self._normalize_prices(df)

        # Optional live pseudo daily bar row
        pseudo_daily_row = kwargs.get("pseudo_daily_row", None)
        if pseudo_daily_row is not None:
            df = self._apply_pseudo_daily_row(df, pseudo_daily_row)

        # Split spec
        split = self._resolve_split(date_range=date_range, kwargs=kwargs, available_dates_sorted=df["Date"])
        test_start_input = split.get("test_start_input", None)
        test_start = split["test_start"]
        train_start = split["train_start"]
        train_end = split["train_end"]
        test_end = split["test_end"]
        train_years = split["train_years"]
        test_months = split["test_months"]

        # Periodic controls
        periodic_start = str(_get_attr(date_range, "periodic_start", kwargs.get("periodic_start", "2014-01-04")))
        future_months = int(self._get_int(date_range, kwargs, "future_months", default=3))
        include_future = self._resolve_include_future(mode=mode, date_range=date_range, kwargs=kwargs)

        # Build FULL features (causal) on observed prices (in-memory only; never saved as combined)
        df_prices_full = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        df_feat_full = self._build_features(df_prices_full, periodic_start=periodic_start)

        prices_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        per_cols = [c for c in df_feat_full.columns if c.startswith("per_")]
        tech_cols = [c for c in df_feat_full.columns if (c not in prices_cols and c not in per_cols and c != "Date")]

        # -------------------------
        # sim/live: split-only outputs (separated)
        # -------------------------
        if mode in ("sim", "live"):
            df_feat_train, df_feat_test_obs = self._split_by_windows(
                df_feat_full,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )

            # periodic test (observed + optional future rows)
            df_periodic_test = (
                df_feat_test_obs[["Date"] + per_cols].copy() if per_cols else df_feat_test_obs[["Date"]].copy()
            )
            future_end = None
            if mode == "live" and include_future and future_months > 0 and len(df_feat_test_obs) > 0:
                last_obs = pd.to_datetime(df_feat_full["Date"].max()).normalize()
                future_end = _norm_date(last_obs + pd.DateOffset(months=future_months))
                df_periodic_test = self._append_future_periodic_rows_only(
                    df_periodic_test,
                    periodic_start=periodic_start,
                    last_obs=last_obs,
                    future_end=future_end,
                )

            # separated outputs
            df_prices_train = df_feat_train[prices_cols].copy()
            df_prices_test = df_feat_test_obs[prices_cols].copy()

            df_periodic_train = (
                df_feat_train[["Date"] + per_cols].copy() if per_cols else df_feat_train[["Date"]].copy()
            )

            df_tech_train = df_feat_train[["Date"] + tech_cols].copy() if tech_cols else df_feat_train[["Date"]].copy()
            df_tech_test = df_feat_test_obs[["Date"] + tech_cols].copy() if tech_cols else df_feat_test_obs[["Date"]].copy()

            prices_train_path = out_dir_mode / f"stepA_prices_train_{symbol}.csv"
            prices_test_path = out_dir_mode / f"stepA_prices_test_{symbol}.csv"
            periodic_train_path = out_dir_mode / f"stepA_periodic_train_{symbol}.csv"
            periodic_test_path = out_dir_mode / f"stepA_periodic_test_{symbol}.csv"
            tech_train_path = out_dir_mode / f"stepA_tech_train_{symbol}.csv"
            tech_test_path = out_dir_mode / f"stepA_tech_test_{symbol}.csv"
            summary_path = out_dir_mode / f"stepA_split_summary_{symbol}.csv"

            df_prices_train.to_csv(prices_train_path, index=False)
            df_prices_test.to_csv(prices_test_path, index=False)
            df_periodic_train.to_csv(periodic_train_path, index=False)
            df_periodic_test.to_csv(periodic_test_path, index=False)
            df_tech_train.to_csv(tech_train_path, index=False)
            df_tech_test.to_csv(tech_test_path, index=False)

            # Daily snapshots (per test-date) for "run-each-day" operation
            # Creates: output/stepA/<mode>/daily/stepA_(prices|periodic|tech|daily_features)_{symbol}_YYYY_MM_DD.csv
            daily_lb = int(os.getenv("STEPA_DAILY_LOOKBACK", str(kwargs.get("mamba_lookback", 30))))
            self._write_daily_snapshots(
                out_dir_mode=out_dir_mode,
                symbol=symbol,
                df_daily_source=df_feat_test_obs,
                prices_cols=prices_cols,
                per_cols=per_cols,
                tech_cols=tech_cols,
                scope="test",
                df_full_for_window=df_feat_full,
                periodic_start=periodic_start,
                lookback=daily_lb,
                future_months=future_months,
            )

            # Global future periodic file for periodic-only (Date + per_*), beyond test_end
            if str(os.getenv("STEPA_WRITE_PERIODIC_FUTURE", "1")).lower() not in ("0", "false", "no"):
                anchor_date = test_end if test_end is not None else pd.to_datetime(df_feat_test_obs["Date"].max())
                self._write_periodic_future_file(
                    out_dir_mode=out_dir_mode,
                    symbol=symbol,
                    periodic_start=periodic_start,
                    anchor_date=anchor_date,
                    future_months=future_months,
                    per_cols_ref=per_cols,
                )


            summary_df = self._build_split_summary_csv(
                df_full=df_feat_full,
                df_train=df_feat_train,
                df_test=df_feat_test_obs,  # observed window for reporting
                test_start_input=test_start_input,
                test_start=test_start,
                train_start=train_start,
                train_end=train_end,
                test_end=test_end,
                train_years=train_years,
                test_months=test_months,
                mode=mode,
                mode_raw=mode_raw,
                periodic_start=periodic_start,
                include_future_periodic=bool(include_future),
                future_months=future_months,
                future_end=future_end,
            )
            summary_df.to_csv(summary_path, index=False)

            out: Dict[str, Any] = {
                "mode": mode,
                "mode_raw": str(mode_raw),
                "output_root": str(Path(self.cfg.output_root).resolve()),
                "src_csv": str(src_csv),
                "stepA_prices_train": str(prices_train_path),
                "stepA_prices_test": str(prices_test_path),
                "stepA_periodic_train": str(periodic_train_path),
                "stepA_periodic_test": str(periodic_test_path),
                "stepA_tech_train": str(tech_train_path),
                "stepA_tech_test": str(tech_test_path),
                "stepA_split_summary": str(summary_path),
            }
            if future_end is not None:
                out["future_end"] = str(pd.to_datetime(future_end).date())
            return out

        # -------------------------
        # display: observed-only prices/tech + periodic (optional future)
        # -------------------------
        if mode == "display":
            # observed window for display (train_start..test_end)
            base_obs = self._select_observed_display_window(
                df_feat_full=df_feat_full,
                train_start=train_start,
                test_end=test_end,
            )

            # prices/tech observed-only
            disp_prices_path = out_dir_mode / f"stepA_prices_{symbol}.csv"
            disp_tech_path = out_dir_mode / f"stepA_tech_{symbol}.csv"
            base_obs[prices_cols].to_csv(disp_prices_path, index=False)
            (base_obs[["Date"] + tech_cols] if tech_cols else base_obs[["Date"]]).to_csv(disp_tech_path, index=False)

            # periodic: observed dates + optional future
            last_obs = pd.to_datetime(df_feat_full["Date"].max()).normalize() if len(df_feat_full) else None
            future_end = None
            if include_future and future_months > 0 and last_obs is not None and not pd.isna(last_obs):
                future_end = _norm_date(last_obs + pd.DateOffset(months=future_months))

            disp_periodic = (base_obs[["Date"] + per_cols].copy() if per_cols else base_obs[["Date"]].copy())
            if future_end is not None:
                disp_periodic = self._append_future_periodic_rows_only(
                    disp_periodic,
                    periodic_start=periodic_start,
                    last_obs=last_obs,
                    future_end=future_end,
                )
            disp_periodic_path = out_dir_mode / f"stepA_periodic_{symbol}.csv"
            disp_periodic.to_csv(disp_periodic_path, index=False)

            summary_path = out_dir_mode / f"stepA_split_summary_{symbol}.csv"
            summary_df = self._build_split_summary_csv(
                df_full=df_feat_full,
                df_train=df_feat_full.iloc[0:0].copy(),
                df_test=base_obs,
                test_start_input=test_start_input,
                test_start=test_start,
                train_start=train_start,
                train_end=train_end,
                test_end=test_end,
                train_years=train_years,
                test_months=test_months,
                mode=mode,
                mode_raw=mode_raw,
                periodic_start=periodic_start,
                include_future_periodic=bool(include_future),
                future_months=future_months,
                future_end=future_end,
            )
            summary_df.to_csv(summary_path, index=False)

            return {
                "mode": mode,
                "mode_raw": str(mode_raw),
                "output_root": str(Path(self.cfg.output_root).resolve()),
                "src_csv": str(src_csv),
                "stepA_prices": str(disp_prices_path),
                "stepA_periodic": str(disp_periodic_path),
                "stepA_tech": str(disp_tech_path),
                "stepA_split_summary": str(summary_path),
                **({"future_end": str(pd.to_datetime(future_end).date())} if future_end is not None else {}),
            }

        # Should never reach
        raise RuntimeError(f"StepA: unreachable mode={mode}")

    def _resolve_src_csv(self, symbol: str, date_range: Any, kwargs: Dict[str, Any]) -> Path:
        """Resolve source CSV from configured data_dir first, then fallback to repo_root/data."""
        candidates = self._list_price_csv_candidates(symbol=symbol, date_range=date_range, kwargs=kwargs)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _list_price_csv_candidates(self, symbol: str, date_range: Any, kwargs: Dict[str, Any]) -> List[Path]:
        filename = f"prices_{symbol}.csv"
        repo_root = get_repo_root().resolve()
        candidates: List[Path] = []

        cfg_data_dir = self._configured_data_dir_raw(kwargs=kwargs)
        if cfg_data_dir:
            data_dir_path = Path(cfg_data_dir).expanduser()
            if data_dir_path.is_absolute():
                candidates.append((data_dir_path / filename).resolve())
            else:
                candidates.append((repo_root / data_dir_path / filename).resolve())
                candidates.append((Path.cwd() / data_dir_path / filename).resolve())

        candidates.append((repo_root / "data" / filename).resolve())

        unique: List[Path] = []
        seen: set[Path] = set()
        for p in candidates:
            if p in seen:
                continue
            seen.add(p)
            unique.append(p)
        return unique

    def _configured_data_dir_raw(self, kwargs: Optional[Dict[str, Any]] = None) -> Optional[str]:
        runtime_data_dir = None
        if kwargs:
            runtime_data_dir = kwargs.get("data_dir", None) or kwargs.get("data_root", None)
        cfg_data_dir = runtime_data_dir or _get_attr(self.cfg, "data_dir", None) or _get_attr(self.cfg, "data_root", None)
        if cfg_data_dir is None:
            return None
        cfg_data_dir_str = str(cfg_data_dir).strip()
        return cfg_data_dir_str or None

    def _configured_data_root(self, kwargs: Optional[Dict[str, Any]] = None) -> Path:
        cfg_data_dir = self._configured_data_dir_raw(kwargs=kwargs)
        if cfg_data_dir:
            data_dir_path = Path(cfg_data_dir).expanduser()
            if data_dir_path.is_absolute():
                return data_dir_path.resolve()
            return (get_repo_root() / data_dir_path).resolve()
        return (get_repo_root() / "data").resolve()

    # -------------------------
    # Internals: purge combined files
    # -------------------------
    def _purge_combined_files(self, out_dir: Path, symbol: str) -> None:
        # Remove any combined feature files under this mode folder.
        patterns = [
            f"stepA_features_{symbol}.csv",
            f"stepA_features_train_{symbol}.csv",
            f"stepA_features_test_{symbol}.csv",
            f"stepA_features*_{symbol}.csv",
        ]
        for pat in patterns:
            for p in out_dir.glob(pat):
                try:
                    p.unlink()
                except Exception:
                    pass

    # -------------------------
    # Internals: mode / split
    # -------------------------
    def _resolve_mode(self, date_range: Any, kwargs: Dict[str, Any]) -> str:
        for key in ("mode", "stepA_mode", "run_mode"):
            if key in kwargs and kwargs[key] is not None:
                return str(kwargs[key]).lower()
        for key in ("mode", "stepA_mode", "run_mode"):
            v = _get_attr(date_range, key, None)
            if v is not None:
                return str(v).lower()
        v = _get_attr(self.cfg, "stepA_mode", None)
        if v is not None:
            return str(v).lower()
        return "sim"

    def _normalize_mode(self, mode: str) -> str:
        m = str(mode).lower().strip()
        if m == "ops":
            return "live"
        if m not in ("sim", "live", "display"):
            raise ValueError(
                f"StepA: invalid mode={mode}. expected sim/live/display (ops is accepted as alias for live)"
            )
        return m

    def _get_int(self, date_range: Any, kwargs: Dict[str, Any], key: str, default: int) -> int:
        v = kwargs.get(key, None)
        if v is None:
            v = _get_attr(date_range, key, None)
        if v is None:
            return int(default)
        try:
            return int(v)
        except Exception:
            return int(default)

    def _resolve_split(self, date_range: Any, kwargs: Dict[str, Any], available_dates_sorted: Any = None) -> Dict[str, Any]:
        test_start = _to_dt(kwargs.get("test_start", None))
        if test_start is None:
            test_start = _to_dt(_get_attr(date_range, "test_start", None))

        train_start = _to_dt(kwargs.get("train_start", None))
        train_end = _to_dt(kwargs.get("train_end", None))
        test_end = _to_dt(kwargs.get("test_end", None))

        if train_start is None:
            train_start = _to_dt(_get_attr(date_range, "train_start", None))
        if train_end is None:
            train_end = _to_dt(_get_attr(date_range, "train_end", None))
        if test_end is None:
            test_end = _to_dt(_get_attr(date_range, "test_end", None))

        train_years = self._get_int(date_range, kwargs, "train_years", default=8)
        test_months = self._get_int(date_range, kwargs, "test_months", default=3)

        if available_dates_sorted is None:
            available_dates_sorted = []
        available_dates_sorted = pd.Series(pd.to_datetime(available_dates_sorted, errors="coerce")).dropna().sort_values().drop_duplicates().reset_index(drop=True)
        if len(available_dates_sorted) == 0:
            return {
                "test_start_input": None if test_start is None else _norm_date(test_start),
                "test_start": None,
                "train_start": None,
                "train_end": None,
                "test_end": None,
                "train_years": train_years,
                "test_months": test_months,
            }

        dmin = _norm_date(available_dates_sorted.iloc[0])
        dmax = _norm_date(available_dates_sorted.iloc[-1])

        if test_start is None:
            test_start_input = _norm_date(dmax - pd.DateOffset(months=test_months))
        else:
            test_start_input = _norm_date(test_start)

        test_start = snap_prev_by_prices(test_start_input, available_dates_sorted)

        train_start_raw = _norm_date(train_start) if train_start is not None else _norm_date(test_start - pd.DateOffset(years=train_years))
        train_end_raw = _norm_date(train_end) if train_end is not None else _norm_date(test_start - pd.Timedelta(days=1))
        test_end_raw = _norm_date(test_end) if test_end is not None else _norm_date(test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1))

        train_start = snap_prev_by_prices(train_start_raw, available_dates_sorted)
        train_end = snap_prev_by_prices(train_end_raw, available_dates_sorted)
        test_end = snap_prev_by_prices(test_end_raw, available_dates_sorted)

        train_start = min(max(train_start, dmin), dmax)
        train_end = min(max(train_end, dmin), dmax)
        test_start = min(max(test_start, dmin), dmax)
        test_end = min(max(test_end, dmin), dmax)

        return {
            "test_start_input": test_start_input,
            "test_start": _norm_date(test_start),
            "train_start": _norm_date(train_start),
            "train_end": _norm_date(train_end),
            "test_end": _norm_date(test_end),
            "train_years": train_years,
            "test_months": test_months,
        }

    def _resolve_include_future(self, mode: str, date_range: Any, kwargs: Dict[str, Any]) -> bool:
        v = kwargs.get("include_future_periodic", None)
        if v is None:
            v = _get_attr(date_range, "include_future_periodic", None)
        if v is not None:
            return bool(v)
        return mode in ("live", "display")

    # -------------------------
    # Internals: normalize / pseudo row
    # -------------------------
    def _normalize_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Date" not in df.columns:
            if "Unnamed: 0" in df.columns:
                df = df.rename(columns={"Unnamed: 0": "Date"})
            else:
                raise ValueError(f"StepA: source CSV missing Date column. columns={list(df.columns)}")

        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"StepA: source CSV missing columns: {missing}. columns={list(df.columns)}")

        out = df.copy()
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.normalize()
        if out["Date"].isna().any():
            bad = out[out["Date"].isna()].head(5)
            raise ValueError(f"StepA: failed to parse Date in source CSV. sample_bad_rows=\n{bad}")

        out = out.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)

        for c in required:
            out[c] = pd.to_numeric(out[c], errors="coerce")

        if out[required].isna().any().any():
            out[required] = out[required].ffill().bfill()

        if out[required].isna().any().any():
            raise ValueError("StepA: still has NaNs after ffill/bfill in required OHLCV columns")

        return out

    def _apply_pseudo_daily_row(self, df: pd.DataFrame, row: Any) -> pd.DataFrame:
        if isinstance(row, pd.Series):
            row = row.to_dict()
        if not isinstance(row, dict):
            raise ValueError("pseudo_daily_row must be a dict-like object")

        required = ["Date", "Open", "High", "Low", "Close", "Volume"]
        for c in required:
            if c not in row:
                raise ValueError(f"pseudo_daily_row missing key: {c}")

        d = pd.to_datetime(row["Date"], errors="coerce")
        if pd.isna(d):
            raise ValueError("pseudo_daily_row has invalid Date")
        d = d.normalize()

        upd = {
            "Date": d,
            "Open": float(row["Open"]),
            "High": float(row["High"]),
            "Low": float(row["Low"]),
            "Close": float(row["Close"]),
            "Volume": float(row["Volume"]),
        }

        out = df.copy()
        mask = out["Date"] == d
        if mask.any():
            idx = out.index[mask][0]
            for k, v in upd.items():
                out.at[idx, k] = v
        else:
            out = pd.concat([out, pd.DataFrame([upd])], ignore_index=True)

        out = out.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
        return out

    # -------------------------
    # Internals: features
    # -------------------------
    def _build_features(self, df_prices: pd.DataFrame, periodic_start: str) -> pd.DataFrame:
        df_feat = df_prices.copy()

        # Technicals (causal)
        df_feat["Gap"] = self._gap(df_prices)
        df_feat["ATR_norm"] = self._atr_norm(df_prices, period=14)
        df_feat["RSI"] = self._rsi(df_prices["Close"], period=14)
        macd, macd_sig = self._macd(df_prices["Close"], fast=12, slow=26, signal=9)
        df_feat["MACD"] = macd
        df_feat["MACD_signal"] = macd_sig

        # BNF-inspired features (causal)
        bnf = self._bnf_features(df_prices)
        df_feat = pd.concat([df_feat, bnf], axis=1)

        # Periodic by Date (no leakage) — anchor at periodic_start for phase consistency
        end_date = pd.to_datetime(df_feat["Date"].max()).normalize()
        per = self._periodic_features_by_date(pd.to_datetime(periodic_start), end_date)
        df_feat = df_feat.merge(per, on="Date", how="left")

        per_cols = [c for c in df_feat.columns if c.startswith("per_")]
        if per_cols:
            df_feat[per_cols] = df_feat[per_cols].ffill().bfill()

        return df_feat

    def _split_by_windows(
        self,
        df: pd.DataFrame,
        train_start: Optional[pd.Timestamp],
        train_end: Optional[pd.Timestamp],
        test_start: Optional[pd.Timestamp],
        test_end: Optional[pd.Timestamp],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if test_start is None or train_start is None or train_end is None or test_end is None:
            return df.iloc[0:0].copy(), df.iloc[0:0].copy()

        d = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
        train_mask = (d >= train_start) & (d <= train_end)
        test_mask = (d >= test_start) & (d <= test_end)

        return df.loc[train_mask].copy(), df.loc[test_mask].copy()

    def _select_observed_display_window(
        self,
        df_feat_full: pd.DataFrame,
        train_start: Optional[pd.Timestamp],
        test_end: Optional[pd.Timestamp],
    ) -> pd.DataFrame:
        if train_start is None or test_end is None:
            return df_feat_full.copy()
        d = pd.to_datetime(df_feat_full["Date"], errors="coerce").dt.normalize()
        return df_feat_full.loc[(d >= train_start) & (d <= test_end)].copy()

    def _append_future_periodic_rows_only(
        self,
        df_periodic: pd.DataFrame,
        periodic_start: str,
        last_obs: pd.Timestamp,
        future_end: pd.Timestamp,
    ) -> pd.DataFrame:
        df_periodic = df_periodic.copy()
        last_obs = pd.to_datetime(last_obs).normalize()
        future_end = pd.to_datetime(future_end).normalize()
        if future_end <= last_obs:
            return df_periodic

        future_dates = self._get_us_trading_days(last_obs + pd.Timedelta(days=1), future_end)
        if len(future_dates) == 0:
            return df_periodic

        per = self._periodic_features_by_date(pd.to_datetime(periodic_start), future_end)
        fut = pd.DataFrame({"Date": future_dates.normalize()})
        fut = fut.merge(per, on="Date", how="left")

        # Keep only Date + periodic columns
        per_cols = [c for c in fut.columns if c.startswith("per_")]
        fut = fut[["Date"] + per_cols]

        out = pd.concat([df_periodic, fut], ignore_index=True)
        out = out.drop_duplicates(subset=["Date"], keep="first").sort_values("Date").reset_index(drop=True)

        # Fill periodic if gaps (shouldn't)
        per_cols2 = [c for c in out.columns if c.startswith("per_")]
        if per_cols2:
            out[per_cols2] = out[per_cols2].ffill().bfill()

        return out

    # -------------------------
    # Internals: split summary CSV
    # -------------------------
    def _build_split_summary_csv(
        self,
        df_full: pd.DataFrame,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        test_start_input: Optional[pd.Timestamp],
        test_start: Optional[pd.Timestamp],
        train_start: Optional[pd.Timestamp],
        train_end: Optional[pd.Timestamp],
        test_end: Optional[pd.Timestamp],
        train_years: int,
        test_months: int,
        mode: str,
        mode_raw: str,
        periodic_start: str,
        include_future_periodic: bool,
        future_months: int,
        future_end: Optional[pd.Timestamp],
    ) -> pd.DataFrame:
        def _minmax(df_: pd.DataFrame) -> Tuple[int, Optional[str], Optional[str]]:
            if df_ is None or len(df_) == 0:
                return 0, None, None
            d = pd.to_datetime(df_["Date"], errors="coerce").dt.normalize()
            return int(len(df_)), str(d.min().date()), str(d.max().date())

        full_rows, full_min, full_max = _minmax(df_full)
        tr_rows, tr_min, tr_max = _minmax(df_train)
        te_rows, te_min, te_max = _minmax(df_test)

        rows: List[Dict[str, Any]] = [
            {"key": "mode", "value": mode},
            {"key": "mode_raw", "value": str(mode_raw)},
            {"key": "periodic_start", "value": str(periodic_start)},
            {"key": "include_future_periodic", "value": bool(include_future_periodic)},
            {"key": "future_months", "value": int(future_months)},
            {"key": "future_end", "value": None if future_end is None else str(pd.to_datetime(future_end).date())},
            {"key": "train_years", "value": int(train_years)},
            {"key": "test_months", "value": int(test_months)},
            {"key": "test_start_input", "value": None if test_start_input is None else str(test_start_input.date())},
            {"key": "train_start", "value": None if train_start is None else str(train_start.date())},
            {"key": "train_end", "value": None if train_end is None else str(train_end.date())},
            {"key": "test_start", "value": None if test_start is None else str(test_start.date())},
            {"key": "test_end", "value": None if test_end is None else str(test_end.date())},
            {"key": "full_rows", "value": full_rows},
            {"key": "full_min_date", "value": full_min},
            {"key": "full_max_date", "value": full_max},
            {"key": "train_rows", "value": tr_rows},
            {"key": "train_min_date", "value": tr_min},
            {"key": "train_max_date", "value": tr_max},
            {"key": "test_rows", "value": te_rows},
            {"key": "test_min_date", "value": te_min},
            {"key": "test_max_date", "value": te_max},
        ]
        return pd.DataFrame(rows)

    # -------------------------
    # Internals: indicators
    # -------------------------
    def _gap(self, df_prices: pd.DataFrame) -> pd.Series:
        open_ = pd.to_numeric(df_prices["Open"], errors="coerce").astype(float)
        prev_close = pd.to_numeric(df_prices["Close"], errors="coerce").astype(float).shift(1)
        gap = (open_ / prev_close.replace(0.0, math.nan)) - 1.0
        return gap.replace([math.inf, -math.inf], math.nan).fillna(0.0)

    def _atr_norm(self, df_prices: pd.DataFrame, period: int = 14) -> pd.Series:
        high = pd.to_numeric(df_prices["High"], errors="coerce").astype(float)
        low = pd.to_numeric(df_prices["Low"], errors="coerce").astype(float)
        close = pd.to_numeric(df_prices["Close"], errors="coerce").astype(float)

        prev_close = close.shift(1)
        tr_components = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        )
        tr = tr_components.max(axis=1)
        atr14 = tr.rolling(window=period, min_periods=period).mean()
        atr_norm = atr14 / prev_close.replace(0.0, math.nan)
        return atr_norm.replace([math.inf, -math.inf], math.nan).fillna(0.0)

    def _rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
        rs = avg_gain / (avg_loss.replace(0.0, math.nan))
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.fillna(50.0)

    def _macd(self, close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        return macd.fillna(0.0), macd_signal.fillna(0.0)

    def _bnf_features(self, df_prices: pd.DataFrame) -> pd.DataFrame:
        open_ = pd.to_numeric(df_prices["Open"], errors="coerce").astype(float)
        high = pd.to_numeric(df_prices["High"], errors="coerce").astype(float)
        low = pd.to_numeric(df_prices["Low"], errors="coerce").astype(float)
        close = pd.to_numeric(df_prices["Close"], errors="coerce").astype(float)
        vol = pd.to_numeric(df_prices["Volume"], errors="coerce").astype(float)

        ret1 = close.pct_change().fillna(0.0)

        vol_mean20 = vol.rolling(20, min_periods=1).mean()
        vol_std20 = vol.rolling(20, min_periods=1).std().replace(0.0, math.nan)

        rvol20 = (vol / vol_mean20.replace(0.0, math.nan)).replace([math.inf, -math.inf], math.nan).fillna(1.0)
        volz20 = ((vol - vol_mean20) / vol_std20).replace([math.inf, -math.inf], math.nan).fillna(0.0)

        body_pct = ((close - open_) / open_.replace(0.0, math.nan)).replace([math.inf, -math.inf], math.nan).fillna(0.0)
        range_pct = ((high - low) / open_.replace(0.0, math.nan)).replace([math.inf, -math.inf], math.nan).fillna(0.0)

        div_down_volup = ((ret1 < 0).astype(float) * (rvol20 > 1.0).astype(float)).astype(float)
        energy_fade = ((ret1 > 0).astype(float) * (rvol20.diff().fillna(0.0) < 0).astype(float)).astype(float)

        panic_score = (-ret1).clip(lower=0.0) * volz20.clip(lower=0.0)

        return pd.DataFrame(
            {
                "BNF_RVOL20": rvol20.astype(float),
                "BNF_VolZ20": volz20.astype(float),
                "BNF_Return1": ret1.astype(float),
                "BNF_BodyPct": body_pct.astype(float),
                "BNF_RangePct": range_pct.astype(float),
                "BNF_DivDownVolUp": div_down_volup,
                "BNF_EnergyFade": energy_fade,
                "BNF_PanicScore": panic_score.astype(float),
            }
        )


    # -------------------------
    # Internals: daily snapshots (per-date CSVs)
    # -------------------------
    def _write_daily_snapshots(
        self,
        *,
        out_dir_mode: Path,
        symbol: str,
        df_daily_source: pd.DataFrame,
        prices_cols: List[str],
        per_cols: List[str],
        tech_cols: List[str],
        scope: str = "test",
        df_full_for_window: Optional[pd.DataFrame] = None,
        periodic_start: str = "2014-01-04",
        lookback: int = 30,
        future_months: int = 3,
    ) -> None:
        """Write per-day snapshot CSVs under output/stepA/<mode>/daily/.

        Always writes 1-row (as-of-day) snapshot CSVs for each observed date in df_daily_source:
          - stepA_prices_{symbol}_YYYY_MM_DD.csv
          - stepA_periodic_{symbol}_YYYY_MM_DD.csv
          - stepA_tech_{symbol}_YYYY_MM_DD.csv
          - stepA_daily_features_{symbol}_YYYY_MM_DD.csv  (1-row combined snapshot)

        Additionally (walk-forward / sequence models), also writes:
          - stepA_daily_window_features_{symbol}_YYYY_MM_DD_lb{lookback}.csv
              * last {lookback} rows up to the date (inclusive) from FULL history (df_full_for_window)
          - stepA_periodic_future_{symbol}_YYYY_MM_DD_m{future_months}.csv
              * future periodic-only inputs from next trading day to (date + future_months)

        Also writes:
          - stepA_daily_manifest_{symbol}.csv at out_dir_mode
            (new columns are appended but older readers remain compatible)
        """
        try:
            if df_daily_source is None or len(df_daily_source) == 0:
                return
            if "Date" not in df_daily_source.columns:
                return

            daily_dir = out_dir_mode / "daily"
            daily_dir.mkdir(parents=True, exist_ok=True)

            # lookback / future months
            try:
                lookback_eff = int(lookback)
            except Exception:
                lookback_eff = 30
            lookback_eff = max(1, lookback_eff)

            try:
                future_months_eff = int(future_months)
            except Exception:
                future_months_eff = 3
            future_months_eff = max(0, future_months_eff)

            # Prepare observed daily rows (typically test dates)
            dts = pd.to_datetime(df_daily_source["Date"], errors="coerce")
            work = df_daily_source.copy()
            work["__dt"] = dts
            work = work.dropna(subset=["__dt"]).sort_values("__dt").reset_index(drop=True)
            if len(work) == 0:
                return

            # Prepare FULL history for window slicing
            full_sorted: Optional[pd.DataFrame] = None
            if df_full_for_window is not None and len(df_full_for_window) > 0 and "Date" in df_full_for_window.columns:
                full_sorted = df_full_for_window.copy()
                full_sorted["__dt"] = pd.to_datetime(full_sorted["Date"], errors="coerce")
                full_sorted = full_sorted.dropna(subset=["__dt"]).sort_values("__dt").reset_index(drop=True)

            # Column sets (safe guard)
            prices_cols_eff = [c for c in prices_cols if c in work.columns]
            if "Date" not in prices_cols_eff and "Date" in work.columns:
                prices_cols_eff = ["Date"] + prices_cols_eff

            per_cols_eff = [c for c in (["Date"] + list(per_cols)) if c in work.columns]
            tech_cols_eff = [c for c in (["Date"] + list(tech_cols)) if c in work.columns]

            combined_cols = ["Date"] + [c for c in prices_cols_eff if c != "Date"]
            combined_cols += [c for c in tech_cols_eff if c != "Date"]
            combined_cols += [c for c in per_cols_eff if c != "Date"]
            combined_cols = [c for c in combined_cols if c in work.columns]

            manifest_rows: List[Dict[str, Any]] = []

            for _, row in work.iterrows():
                dt = pd.to_datetime(row["__dt"]).normalize()
                tag = dt.strftime("%Y_%m_%d")

                # 1-row snapshots
                df_prices_1 = pd.DataFrame([{c: row[c] for c in prices_cols_eff if c in work.columns}])
                df_periodic_1 = pd.DataFrame([{c: row[c] for c in per_cols_eff if c in work.columns}])
                df_tech_1 = pd.DataFrame([{c: row[c] for c in tech_cols_eff if c in work.columns}])
                df_combined_1 = pd.DataFrame([{c: row[c] for c in combined_cols if c in work.columns}])

                p_prices = daily_dir / f"stepA_prices_{symbol}_{tag}.csv"
                p_periodic = daily_dir / f"stepA_periodic_{symbol}_{tag}.csv"
                p_tech = daily_dir / f"stepA_tech_{symbol}_{tag}.csv"
                p_combined = daily_dir / f"stepA_daily_features_{symbol}_{tag}.csv"

                df_prices_1.to_csv(p_prices, index=False)
                df_periodic_1.to_csv(p_periodic, index=False)
                df_tech_1.to_csv(p_tech, index=False)
                df_combined_1.to_csv(p_combined, index=False)

                # Window snapshot: last lookback_eff rows from FULL history up to dt
                p_window = ""
                if full_sorted is not None and lookback_eff >= 2:
                    win = full_sorted.loc[full_sorted["__dt"] <= dt].tail(lookback_eff)
                    if len(win) > 0:
                        win = win.sort_values("__dt").reset_index(drop=True)
                        win_cols = [c for c in combined_cols if c in win.columns]
                        if "Date" not in win_cols and "Date" in win.columns:
                            win_cols = ["Date"] + win_cols
                        win_out = win[win_cols].copy()
                        p_window_path = daily_dir / f"stepA_daily_window_features_{symbol}_{tag}_lb{lookback_eff}.csv"
                        win_out.to_csv(p_window_path, index=False)
                        p_window = str(p_window_path)

                # Future periodic-only snapshot: dt+1 .. dt+future_months_eff
                p_future = ""
                if future_months_eff > 0:
                    future_end = _norm_date(dt + pd.DateOffset(months=future_months_eff))
                    future_dates = self._get_us_trading_days(dt + pd.Timedelta(days=1), future_end)
                    if len(future_dates) > 0:
                        per = self._periodic_features_by_date(pd.to_datetime(periodic_start), future_end)
                        fut = pd.DataFrame({"Date": pd.to_datetime(future_dates).normalize()})
                        fut = fut.merge(per, on="Date", how="left")
                        per_cols2 = [c for c in fut.columns if c.startswith("per_")]
                        fut = fut[["Date"] + per_cols2]
                        p_future_path = daily_dir / f"stepA_periodic_future_{symbol}_{tag}_m{future_months_eff}.csv"
                        fut.to_csv(p_future_path, index=False)
                        p_future = str(p_future_path)

                manifest_rows.append(
                    {
                        "Date": dt.strftime("%Y-%m-%d"),
                        "scope": str(scope),
                        "prices_path": str(p_prices),
                        "periodic_path": str(p_periodic),
                        "tech_path": str(p_tech),
                        "features_path": str(p_combined),
                        "window_features_path": p_window,
                        "periodic_future_path": p_future,
                    }
                )

            manifest_path = out_dir_mode / f"stepA_daily_manifest_{symbol}.csv"
            pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

        except Exception:
            # Daily snapshots are auxiliary; do not fail StepA on snapshot generation errors.
            return


    def _write_periodic_future_file(
        self,
        *,
        out_dir_mode: Path,
        symbol: str,
        periodic_start: str,
        anchor_date: pd.Timestamp,
        future_months: int,
        per_cols_ref: Optional[List[str]] = None,
    ) -> Optional[Path]:
        """Write global future periodic CSV: stepA_periodic_future_<SYMBOL>.csv.

        Contains only Date + per_* columns for the horizon (anchor_date+1 .. anchor_date+future_months).
        """
        try:
            try:
                future_months_eff = int(future_months)
            except Exception:
                future_months_eff = 3
            if future_months_eff <= 0:
                return None

            anchor = pd.to_datetime(anchor_date).normalize()
            future_end = _norm_date(anchor + pd.DateOffset(months=future_months_eff))
            future_dates = self._get_us_trading_days(anchor + pd.Timedelta(days=1), future_end)
            if len(future_dates) == 0:
                return None

            per = self._periodic_features_by_date(pd.to_datetime(periodic_start), future_end)
            fut = pd.DataFrame({"Date": pd.to_datetime(future_dates).normalize()})
            fut = fut.merge(per, on="Date", how="left")

            per_cols = [c for c in fut.columns if c.startswith("per_")]
            if per_cols_ref:
                ordered = [c for c in per_cols_ref if c in per_cols]
                rest = [c for c in per_cols if c not in ordered]
                per_cols = ordered + rest

            fut = fut[["Date"] + per_cols]
            out_path = out_dir_mode / f"stepA_periodic_future_{symbol}.csv"
            fut.to_csv(out_path, index=False)
            return out_path
        except Exception:
            return None

    def _get_us_trading_days(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
        start = pd.Timestamp(start).normalize()
        end = pd.Timestamp(end).normalize()
        if end < start:
            return pd.DatetimeIndex([])
        # 1) exchange_calendars (best)
        try:
            import exchange_calendars as ecals  # type: ignore
            cal = ecals.get_calendar("XNYS")
            sessions = cal.sessions_in_range(start, end)
            return pd.DatetimeIndex(sessions).tz_localize(None).normalize()
        except Exception:
            pass
        # 2) pandas_market_calendars
        try:
            import pandas_market_calendars as mcal  # type: ignore
            cal = mcal.get_calendar("NYSE")
            sched = cal.schedule(start_date=start, end_date=end)
            sessions = sched.index
            return pd.DatetimeIndex(sessions).tz_localize(None).normalize()
        except Exception:
            pass
        # 3) fallback (weekdays-only; holidays are NOT excluded)
        return pd.bdate_range(start, end).normalize()

    @staticmethod
    def _julian_date_from_datetimeindex(dti: pd.DatetimeIndex) -> np.ndarray:
        # JD at 00:00 UTC for each date (naive dates treated as UTC)
        ns = dti.view("int64").astype(np.float64)
        return ns / (86400.0 * 1e9) + 2440587.5

    @staticmethod
    def _wrap_pi(x: np.ndarray) -> np.ndarray:
        return (x + np.pi) % (2.0 * np.pi) - np.pi

    @staticmethod
    def _kepler_E(M: np.ndarray, e: np.ndarray, iters: int = 7) -> np.ndarray:
        # Solve Kepler's equation: E - e sin(E) = M  (M,E in radians)
        E = M + e * np.sin(M) * (1.0 + e * np.cos(M))
        for _ in range(iters):
            E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E) + 1e-12)
        return E

    @classmethod
    def _planet_heliocentric_ecliptic_xyz(cls, d_days: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Lightweight orbital elements (Paul Schlyter style; J2000-based), returns heliocentric ecliptic coords in AU.
        # d_days: days since J2000.0
        deg = np.pi / 180.0

        def elems(plan: str):
            if plan == "mercury":
                N = (48.3313 + 3.24587e-5 * d_days) * deg
                i = (7.0047 + 5.00e-8 * d_days) * deg
                w = (29.1241 + 1.01444e-5 * d_days) * deg
                a = 0.387098
                e = 0.205635 + 5.59e-10 * d_days
                M = (168.6562 + 4.0923344368 * d_days) * deg
                period = 87.969
            elif plan == "venus":
                N = (76.6799 + 2.46590e-5 * d_days) * deg
                i = (3.3946 + 2.75e-8 * d_days) * deg
                w = (54.8910 + 1.38374e-5 * d_days) * deg
                a = 0.723330
                e = 0.006773 - 1.302e-9 * d_days
                M = (48.0052 + 1.6021302244 * d_days) * deg
                period = 224.701
            elif plan == "earth":
                N = (0.0 + 0.0 * d_days) * deg
                i = (0.0 + 0.0 * d_days) * deg
                w = (282.9404 + 4.70935e-5 * d_days) * deg
                a = 1.0
                e = 0.016709 - 1.151e-9 * d_days
                M = (356.0470 + 0.9856002585 * d_days) * deg
                period = 365.256
            elif plan == "mars":
                N = (49.5574 + 2.11081e-5 * d_days) * deg
                i = (1.8497 - 1.78e-8 * d_days) * deg
                w = (286.5016 + 2.92961e-5 * d_days) * deg
                a = 1.523688
                e = 0.093405 + 2.516e-9 * d_days
                M = (18.6021 + 0.5240207766 * d_days) * deg
                period = 686.980
            elif plan == "jupiter":
                N = (100.4542 + 2.76854e-5 * d_days) * deg
                i = (1.3030 - 1.557e-7 * d_days) * deg
                w = (273.8777 + 1.64505e-5 * d_days) * deg
                a = 5.20256
                e = 0.048498 + 4.469e-9 * d_days
                M = (19.8950 + 0.0830853001 * d_days) * deg
                period = 4332.589
            elif plan == "saturn":
                N = (113.6634 + 2.38980e-5 * d_days) * deg
                i = (2.4886 - 1.081e-7 * d_days) * deg
                w = (339.3939 + 2.97661e-5 * d_days) * deg
                a = 9.55475
                e = 0.055546 - 9.499e-9 * d_days
                M = (316.9670 + 0.0334442282 * d_days) * deg
                period = 10759.22
            elif plan == "uranus":
                N = (74.0005 + 1.3978e-5 * d_days) * deg
                i = (0.7733 + 1.9e-8 * d_days) * deg
                w = (96.6612 + 3.0565e-5 * d_days) * deg
                a = 19.18171 - 1.55e-8 * d_days
                e = 0.047318 + 7.45e-9 * d_days
                M = (142.5905 + 0.011725806 * d_days) * deg
                period = 30688.5
            elif plan == "neptune":
                N = (131.7806 + 3.0173e-5 * d_days) * deg
                i = (1.7700 - 2.55e-7 * d_days) * deg
                w = (272.8461 - 6.027e-6 * d_days) * deg
                a = 30.05826 + 3.313e-8 * d_days
                e = 0.008606 + 2.15e-9 * d_days
                M = (260.2471 + 0.005995147 * d_days) * deg
                period = 60182.0
            elif plan == "pluto":
                # very rough (Pluto is hard; this is "good enough" for a deterministic feature)
                N = (110.30347 + 0.0 * d_days) * deg
                i = (17.14175 + 0.0 * d_days) * deg
                w = (113.76329 + 0.0 * d_days) * deg
                a = 39.48168677
                e = 0.24880766 + 0.0 * d_days
                M = (14.53 + 0.0039757 * d_days) * deg
                period = 90560.0
            else:
                raise ValueError(plan)
            return N, i, w, a, e, M, period

        plan = name.lower()
        N, inc, w, a, e, M, _period = elems(plan)
        e = np.clip(e, 0.0, 0.999999)
        M = cls._wrap_pi(M)

        E = cls._kepler_E(M, e)
        xv = a * (np.cos(E) - e)
        yv = a * (np.sqrt(1.0 - e * e) * np.sin(E))
        v = np.arctan2(yv, xv)
        r = np.sqrt(xv * xv + yv * yv)

        vw = v + w
        cosN, sinN = np.cos(N), np.sin(N)
        cosi, sini = np.cos(inc), np.sin(inc)
        cosvw, sinvw = np.cos(vw), np.sin(vw)

        xh = r * (cosN * cosvw - sinN * sinvw * cosi)
        yh = r * (sinN * cosvw + cosN * sinvw * cosi)
        zh = r * (sinvw * sini)
        return xh, yh, zh

    @classmethod
    def _planet_geocentric_longitude(cls, d_days: np.ndarray, planet: str) -> np.ndarray:
        # Geocentric ecliptic longitude in radians (naive approximation)
        xe, ye, ze = cls._planet_heliocentric_ecliptic_xyz(d_days, "earth")
        xp, yp, zp = cls._planet_heliocentric_ecliptic_xyz(d_days, planet)
        xg = xp - xe
        yg = yp - ye
        lon = np.arctan2(yg, xg)
        return lon

    @staticmethod
    def _planet_mean_motion_rad_per_day(planet: str) -> float:
        periods = {
            "mercury": 87.969,
            "venus": 224.701,
            "mars": 686.980,
            "jupiter": 4332.589,
            "saturn": 10759.22,
            "uranus": 30688.5,
            "neptune": 60182.0,
            "pluto": 90560.0,
        }
        p = periods[planet]
        return 2.0 * np.pi / p

    def _compute_planet_retro_features(self, dates: pd.DatetimeIndex) -> dict[str, np.ndarray]:
        # returns {col_name: array}
        # Compute retrograde flag and normalized signed speed for each planet
        dates = pd.DatetimeIndex(dates).normalize()
        jd = self._julian_date_from_datetimeindex(dates)
        d = jd - 2451545.0  # days since J2000.0
        out: dict[str, np.ndarray] = {}
        planets = ["mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto"]

        # Use the actual time axis (d) so holiday gaps don't distort the derivative.
        for p in planets:
            lon = self._planet_geocentric_longitude(d, p)
            lon_u = np.unwrap(lon)
            speed = np.gradient(lon_u, d)  # rad/day
            retro = (speed < 0.0).astype(np.float32)
            mm = float(self._planet_mean_motion_rad_per_day(p))
            speed_norm = np.clip(speed / (mm + 1e-12), -2.0, 2.0).astype(np.float32)

            out[f"per_planet_{p}_retro_flag"] = retro
            out[f"per_planet_{p}_speed"] = speed_norm
        return out

    def _periodic_features_by_date(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        *,
        dates: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Create 44 periodic features for a given date span.

        If `dates` is provided, features are generated only for those dates (recommended).
        Otherwise, dates are generated using a US trading calendar when available.
        """
        start = pd.Timestamp(start).normalize()
        end = pd.Timestamp(end).normalize()

        if dates is not None:
            dti = pd.DatetimeIndex(pd.to_datetime(dates)).tz_localize(None).normalize().unique().sort_values()
        else:
            dti = self._get_us_trading_days(start, end)

        if len(dti) == 0:
            cols = ["Date"]
            return pd.DataFrame(columns=cols)

        # --- base phase angles (7) ---
        epoch = pd.Timestamp("2000-01-01")
        delta_days = (dti - epoch).days.astype(np.float64)

        two_pi = 2.0 * np.pi
        # calendar-based cycles
        phase_month30 = two_pi * (delta_days / 30.0)
        phase_quarter90 = two_pi * (delta_days / 90.0)
        phase_year365 = two_pi * (delta_days / 365.0)
        phase_halfyear182 = two_pi * (delta_days / 182.0)

        # astronomy-based cycles (still date-based)
        phase_moon29 = two_pi * (delta_days / 29.53)
        phase_solstice91 = two_pi * (delta_days / 91.25)

        # trading-week cycle (holiday-aware): index within each calendar week based on trading sessions only
        df_tmp = pd.DataFrame({"Date": dti})
        # Week key = Monday of that week
        week_key = df_tmp["Date"] - pd.to_timedelta(df_tmp["Date"].dt.weekday, unit="D")
        df_tmp["_wk"] = week_key.dt.normalize()
        df_tmp["_idx_in_wk"] = df_tmp.groupby("_wk").cumcount().astype(np.float64)
        phase_wk_bday = two_pi * (df_tmp["_idx_in_wk"].to_numpy() / 5.0)

        # --- sin/cos of base phases (10 + 4) ---
        data: dict[str, np.ndarray] = {}
        data["Date"] = dti

        def add_sincos(prefix: str, phase: np.ndarray):
            data[f"{prefix}_sin"] = np.sin(phase).astype(np.float32)
            data[f"{prefix}_cos"] = np.cos(phase).astype(np.float32)

        add_sincos("per_cal_wk_bday", phase_wk_bday)
        add_sincos("per_cal_month30", phase_month30)
        add_sincos("per_cal_quarter90", phase_quarter90)
        add_sincos("per_cal_year365", phase_year365)
        add_sincos("per_cal_halfyear182", phase_halfyear182)

        add_sincos("per_astro_moon29_53", phase_moon29)
        add_sincos("per_astro_solstice91_25", phase_solstice91)

        # --- planet retro features (16) ---
        planet_feats = self._compute_planet_retro_features(dti)
        data.update(planet_feats)

        # --- harmonics (14): sin(2*phase), sin(3*phase) for each base phase (7) ---
        base_phases = [
            ("cal_wk_bday", phase_wk_bday),
            ("cal_month30", phase_month30),
            ("cal_quarter90", phase_quarter90),
            ("cal_year365", phase_year365),
            ("cal_halfyear182", phase_halfyear182),
            ("astro_moon29_53", phase_moon29),
            ("astro_solstice91_25", phase_solstice91),
        ]
        for name, ph in base_phases:
            data[f"per_h2_{name}"] = np.sin(2.0 * ph).astype(np.float32)
            data[f"per_h3_{name}"] = np.sin(3.0 * ph).astype(np.float32)

        df = pd.DataFrame(data)

        # Column order: Date, cal (10), astro (4), planets (16), harmonics (14)
        ordered_cols = [
            "Date",
            "per_cal_wk_bday_sin",
            "per_cal_wk_bday_cos",
            "per_cal_month30_sin",
            "per_cal_month30_cos",
            "per_cal_quarter90_sin",
            "per_cal_quarter90_cos",
            "per_cal_year365_sin",
            "per_cal_year365_cos",
            "per_cal_halfyear182_sin",
            "per_cal_halfyear182_cos",
            "per_astro_moon29_53_sin",
            "per_astro_moon29_53_cos",
            "per_astro_solstice91_25_sin",
            "per_astro_solstice91_25_cos",
            "per_planet_mercury_retro_flag",
            "per_planet_mercury_speed",
            "per_planet_venus_retro_flag",
            "per_planet_venus_speed",
            "per_planet_mars_retro_flag",
            "per_planet_mars_speed",
            "per_planet_jupiter_retro_flag",
            "per_planet_jupiter_speed",
            "per_planet_saturn_retro_flag",
            "per_planet_saturn_speed",
            "per_planet_uranus_retro_flag",
            "per_planet_uranus_speed",
            "per_planet_neptune_retro_flag",
            "per_planet_neptune_speed",
            "per_planet_pluto_retro_flag",
            "per_planet_pluto_speed",
            "per_h2_cal_wk_bday",
            "per_h3_cal_wk_bday",
            "per_h2_cal_month30",
            "per_h3_cal_month30",
            "per_h2_cal_quarter90",
            "per_h3_cal_quarter90",
            "per_h2_cal_year365",
            "per_h3_cal_year365",
            "per_h2_cal_halfyear182",
            "per_h3_cal_halfyear182",
            "per_h2_astro_moon29_53",
            "per_h3_astro_moon29_53",
            "per_h2_astro_solstice91_25",
            "per_h3_astro_solstice91_25",
        ]
        ordered_cols = [c for c in ordered_cols if c in df.columns]
        df = df[ordered_cols]
        return df
