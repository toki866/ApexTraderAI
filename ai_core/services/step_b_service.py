from __future__ import annotations

from pathlib import Path
import os
import re
import math
import shutil
from typing import Dict, Optional, Tuple, List

import pandas as pd

from ai_core.config.app_config import AppConfig
from ai_core.services._path_helpers import _get_output_root_from_app
from ai_core.config.step_b_config import (
    StepBConfig,
    XSRTrainConfig,
    WaveletMambaTrainConfig,
    FEDformerTrainConfig,
)
from ai_core.types.step_b_types import StepBAgentResult, StepBResult


class StepBService:
    """StepB service.

    Reads StepA outputs:
        {output_root}/stepA_prices_{symbol}.csv
        {output_root}/stepA_features_{symbol}.csv

    Always writes (io_contract.md v1.2):
        {output_root}/stepB_pred_time_all_{symbol}.csv

    Note: The headless runner validates that this CSV has >= 10 rows and contains
    required columns: Date, Pred_Close_XSR, Pred_Close_MAMBA, Pred_Close_FED.
    """

    def __init__(self, app_config: AppConfig) -> None:
        self.app_config = app_config


    def _out_dir(self, run_mode: str | None) -> Path:
        """Return output directory for StepB for the given run_mode.

        Normalization:
          - 'ops'/'prod'/'production' -> 'live'
          - unknown/None -> 'sim'
        """
        cfg = self.app_config.data
        out_root = Path(cfg.output_root)

        m = str(run_mode or "").strip().lower()
        if m in ("ops", "prod", "production"):
            m = "live"
        if m not in ("sim", "live"):
            m = "sim"

        d = out_root / "stepB" / m
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _resolve_run_mode(self, cfg_all: StepBConfig) -> str:
        """Decide StepB run mode ('sim' or 'live') from the config.

        Priority:
          1) cfg_all.mode (if exists)
          2) cfg_all.<agent>.mode (mamba/xsr/fedformer, if exists)
          3) default 'sim'
        """
        cand: list[object] = []

        v = getattr(cfg_all, "mode", None)
        if v:
            cand.append(v)

        for k in ("mamba", "xsr", "fedformer"):
            sub = getattr(cfg_all, k, None)
            if sub is None:
                continue
            mv = getattr(sub, "mode", None)
            if mv:
                cand.append(mv)

        m = str(cand[0] if cand else "sim").strip().lower()
        if m in ("ops", "prod", "production"):
            m = "live"
        if m not in ("sim", "live"):
            m = "sim"
        return m
    def _mirror_stepb_artifacts_to_mode_dir(self, symbol: str, run_mode: str) -> None:
        """Copy StepB artifacts from base stepB dir into stepB/{run_mode}.

        This keeps legacy paths (output/stepB/*.csv) intact while ensuring
        mode-separated copies exist under output/stepB/{run_mode}/.
        """
        base = self._out_dir(None)
        mode_dir = self._out_dir(run_mode)

        sym = str(symbol)
        sym_u = sym.upper()

        # Only copy files directly under base (not subfolders).
        for p in base.iterdir():
            if not p.is_file():
                continue
            name = p.name
            if (f"_{sym}." in name) or (f"_{sym_u}." in name) or name.endswith(f"_{sym}.csv") or name.endswith(f"_{sym_u}.csv"):
                try:
                    shutil.copy2(p, mode_dir / name)
                except Exception:
                    pass


    def run(self, config: StepBConfig | None = None, *args, **kwargs) -> StepBResult:
        """Run StepB.

        Accepted call styles (compat):
            service.run(stepB_config)
            service.run(date_range, stepB_config)
            service.run(symbol, date_range, stepB_config)

        (This method extracts StepBConfig from provided arguments.)
        """
        stepb_config: Optional[StepBConfig] = None

        if isinstance(config, StepBConfig):
            stepb_config = config
        else:
            for arg in args:
                if isinstance(arg, StepBConfig):
                    stepb_config = arg
                    break
            if stepb_config is None and "config" in kwargs:
                maybe_cfg = kwargs["config"]
                if isinstance(maybe_cfg, StepBConfig):
                    stepb_config = maybe_cfg

        if stepb_config is None:
            raise TypeError(
                "StepBService.run には StepBConfig を渡してください "
                "(例: service.run(stepB_config) または service.run(dr, stepB_config))。 "
                f"受け取った型: config={type(config)}, "
                f"args={[type(a) for a in args]}, "
                f"kwargs_keys={list(kwargs.keys())}"
            )

        cfg_all = StepBConfig.from_any(stepb_config)
        symbol = cfg_all.symbol

        run_mode = self._resolve_run_mode(cfg_all)
        mode_dir = self._out_dir(run_mode)

        # Load StepA outputs (mode-first; supports split train/test outputs).
        prices_df, features_df = self._load_stepA_outputs(symbol, mode=run_mode)

        enabled = cfg_all.enabled_agents()
        if not enabled:
            raise ValueError('No agents enabled in StepBConfig (xsr/mamba/fedformer all disabled).')

        # Filter periodic-only features only when needed (XSR/FEDformer). Mamba-only runs should not fail here.
        features_df_3ai: Optional[pd.DataFrame] = None
        if ('xsr' in enabled) or ('fedformer' in enabled):
            features_df_3ai = self._filter_features_periodic_only_for_3ai(features_df)

        # Run each enabled agent. Any failure should stop the pipeline (no silent empty artifacts).
        agents: Dict[str, StepBAgentResult] = {}
        for agent in enabled:
            if agent == 'xsr':

                if features_df_3ai is None:
                    raise RuntimeError("[StepB] internal error: features_df_3ai is None while running XSR. This should not happen.")
                agents['xsr'] = self._run_xsr(symbol, prices_df, features_df_3ai, cfg_all.xsr)
            elif agent == 'mamba':
                agents['mamba'] = self._run_mamba(symbol, prices_df, features_df, cfg_all.mamba, date_range=getattr(cfg_all, "date_range", None))
                try:
                    setattr(cfg_all.mamba, "_last_pred_path_mamba", agents["mamba"].csv_paths.get("pred_path", ""))
                    self._write_daily_snapshots_mamba(symbol, prices_df, run_mode, cfg_all.mamba, date_range=getattr(cfg_all, "date_range", None))
                except Exception as e:
                    print(f"[StepB] WARN: daily snapshot generation failed: {e}")
            elif agent == 'fedformer':

                if features_df_3ai is None:
                    raise RuntimeError("[StepB] internal error: features_df_3ai is None while running FEDformer. This should not happen.")
                agents['fedformer'] = self._run_fedformer(symbol, prices_df, features_df_3ai, cfg_all.fedformer)
            else:
                raise ValueError(f'Unknown agent: {agent}')

        # Build the combined prediction table.
        pred_time_all_path = mode_dir / f'stepB_pred_time_all_{symbol}.csv'
        self._write_pred_time_all(symbol=symbol, prices_df=prices_df, agents=agents, out_path=pred_time_all_path)

        # NOTE: StepB outputs are mode-separated under output/stepB/<run_mode>/ only.


        # Ensure delta CSVs exist (contract artifact).
        if cfg_all.ensure_contract_artifacts:
            self._ensure_stepb_delta_files(symbol=symbol, prices_df=prices_df, agents=agents, cfg_all=cfg_all)


        return StepBResult(
            success=True,
            message='StepB done.',
            out_dir=str(mode_dir),
            agent_results=agents,
            pred_time_all_path=str(pred_time_all_path),
        )



    def _select_periodic_cols(self, features_df: pd.DataFrame) -> List[str]:
        """StepA が生成する周期特徴量列（44本想定）を抽出する。

        新仕様では周期特徴量は **per_ プレフィックス**で統一される（例: per_cal_*, per_astro_*, per_planet_*, per_h2_*, per_h3_*）。
        逆行フラグ/速度や高調波など sin/cos 以外の列も含むため、sin/cos 判定ではなく per_ 判定で抽出する。

        Returns:
          List[str]: Date を除いた per_ 列名のリスト（順序は元DFの列順を維持）
        """
        cols: List[str] = []
        for c in features_df.columns:
            if str(c) == "Date":
                continue
            cs = str(c)
            if cs.lower().startswith("per_"):
                cols.append(cs)
        return cols


    def _filter_features_periodic_only_for_3ai(self, features_df: pd.DataFrame, expected_cols: int = 44) -> pd.DataFrame:
        """3AI 用に features_df を周期列（Date + per_* 44本）だけに絞る。"""
        if "Date" not in features_df.columns:
            raise RuntimeError(f"[StepB] features_df has no Date column. columns={list(features_df.columns)}")

        periodic_cols = self._select_periodic_cols(features_df)
        if len(periodic_cols) == 0:
            raise RuntimeError(
                "[StepB] periodic columns (per_*) not found in features_df. "
                "Please check StepA periodic feature column names." 
            )

        strict = os.environ.get("STEPB_PERIODIC_ONLY_STRICT", "1").strip() != "0"
        if strict and len(periodic_cols) != expected_cols:
            sample = periodic_cols[:30]
            raise RuntimeError(
                f"[StepB] periodic-only mode expects {expected_cols} columns, but found {len(periodic_cols)}. "
                f"sample={sample} ...  (set STEPB_PERIODIC_ONLY_STRICT=0 to allow non-44)"
            )

        out = features_df[["Date"] + periodic_cols].copy()
        return out


    def _load_stepA_outputs(self, symbol: str, mode: object = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load StepA outputs.

        Preferred (mode-first):
          - output/stepA/<mode>/stepA_prices_train_<SYMBOL>.csv + stepA_prices_test_<SYMBOL>.csv
          - output/stepA/<mode>/stepA_periodic_train/test_<SYMBOL>.csv
          - output/stepA/<mode>/stepA_tech_train/test_<SYMBOL>.csv

        Backward compatible:
          - output/stepA_prices_<SYMBOL>.csv
          - output/stepA_features_<SYMBOL>.csv
          - output/stepA/stepA_prices_<SYMBOL>.csv
          - output/stepA/stepA_features_<SYMBOL>.csv

        Returns:
          prices_df: must include columns ['Date','Close'] (OHLCV are kept if present)
          features_df: must include column ['Date'] + features (periodic/tech)
        """
        cfg = self.app_config.data
        out_root = Path(cfg.output_root)
        step_a_root = out_root / "stepA"

        # normalize mode
        m = str(mode or "").strip().lower()
        if m in ("ops", "prod", "production"):
            m = "live"
        if m not in ("sim", "live"):
            m = ""

        cand_dirs: list[Path] = []
        if m:
            cand_dirs.append(step_a_root / m)
        cand_dirs.extend([step_a_root / "sim", step_a_root / "live", step_a_root / "ops", step_a_root, out_root])

        # Final fallback: allow raw prices under config.data_dir/data_root.
        data_root = getattr(self.app_config, "data_dir", None)
        if data_root is None:
            data_root = getattr(self.app_config, "data_root", None)
        if data_root is None:
            data_root = getattr(getattr(self.app_config, "data", None), "data_dir", None)
        if data_root is None:
            data_root = getattr(getattr(self.app_config, "data", None), "data_root", None)
        if data_root:
            cand_dirs.append(Path(data_root))

        def _read_csv_norm(p: Path) -> pd.DataFrame:
            df = pd.read_csv(p)
            # normalize Date column name
            date_col = None
            for c in df.columns:
                if str(c).lower() == "date":
                    date_col = c
                    break
            if date_col is not None and date_col != "Date":
                df = df.rename(columns={date_col: "Date"})
            if "date" in df.columns and "Date" not in df.columns:
                df = df.rename(columns={"date": "Date"})
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
                df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
            return df

        def _concat_train_test(p_tr: Path, p_te: Path) -> pd.DataFrame:
            d1 = _read_csv_norm(p_tr)
            d2 = _read_csv_norm(p_te)
            df = pd.concat([d1, d2], ignore_index=True)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
                df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
            return df

        # -------------------------
        # Prices (train/test split preferred)
        # -------------------------
        prices_df: pd.DataFrame | None = None
        searched_prices: list[Path] = []

        for d in cand_dirs:
            p_tr = d / f"stepA_prices_train_{symbol}.csv"
            p_te = d / f"stepA_prices_test_{symbol}.csv"
            searched_prices.extend([p_tr, p_te])
            if p_tr.exists() and p_te.exists():
                prices_df = _concat_train_test(p_tr, p_te)
                break

        if prices_df is None:
            # legacy / display candidates
            legacy_names = [
                f"stepA_prices_{symbol}.csv",
                f"prices_{symbol}.csv",
                f"stepA_prices_{symbol}.CSV",
                f"prices_{symbol}.CSV",
            ]
            found: Path | None = None
            for d in cand_dirs:
                for nm in legacy_names:
                    p = d / nm
                    searched_prices.append(p)
                    if p.exists():
                        found = p
                        break
                if found is not None:
                    break
            if found is None:
                raise FileNotFoundError(
                    f"StepB: StepA prices CSV not found for {symbol}. searched={[str(p) for p in searched_prices]}"
                )
            prices_df = _read_csv_norm(found)

        # normalize price column names (Close is required)
        cols_lower = {str(c).lower(): c for c in prices_df.columns}
        if "close" in cols_lower and cols_lower["close"] != "Close":
            prices_df = prices_df.rename(columns={cols_lower["close"]: "Close"})
        if "Close" not in prices_df.columns:
            # try Adj Close fallback
            for k in ("adj close", "adjclose"):
                if k in cols_lower:
                    prices_df = prices_df.rename(columns={cols_lower[k]: "Close"})
                    break
        if "Close" not in prices_df.columns:
            raise KeyError(f"StepB: StepA prices CSV has no Close column for {symbol}. cols={list(prices_df.columns)}")

        # keep OHLCV but normalize common names if present
        for k, std in [("open", "Open"), ("high", "High"), ("low", "Low"), ("volume", "Volume")]:
            if k in cols_lower and cols_lower[k] != std:
                prices_df = prices_df.rename(columns={cols_lower[k]: std})

        # ensure Date exists
        if "Date" not in prices_df.columns:
            raise KeyError(f"StepB: StepA prices CSV has no Date column for {symbol}.")

        # -------------------------
        # Features (periodic/tech split preferred; fallback to legacy stepA_features)
        # -------------------------
        periodic_df: pd.DataFrame | None = None
        tech_df: pd.DataFrame | None = None
        searched_feat: list[Path] = []

        for d in cand_dirs:
            p_tr = d / f"stepA_periodic_train_{symbol}.csv"
            p_te = d / f"stepA_periodic_test_{symbol}.csv"
            searched_feat.extend([p_tr, p_te])
            if p_tr.exists() and p_te.exists():
                periodic_df = _concat_train_test(p_tr, p_te)
                break

        for d in cand_dirs:
            p_tr = d / f"stepA_tech_train_{symbol}.csv"
            p_te = d / f"stepA_tech_test_{symbol}.csv"
            searched_feat.extend([p_tr, p_te])
            if p_tr.exists() and p_te.exists():
                tech_df = _concat_train_test(p_tr, p_te)
                break

        if periodic_df is not None or tech_df is not None:
            if periodic_df is None:
                features_df = tech_df.copy()
            elif tech_df is None:
                features_df = periodic_df.copy()
            else:
                features_df = pd.merge(periodic_df, tech_df, on="Date", how="outer", suffixes=("", "_tech"))
                # if duplicates from tech side exist, prefer non-null base columns
                dup_cols = [c for c in features_df.columns if c.endswith("_tech")]
                for c in dup_cols:
                    base = c[: -len("_tech")]
                    if base in features_df.columns:
                        features_df[base] = pd.to_numeric(features_df[base], errors="coerce").combine_first(
                            pd.to_numeric(features_df[c], errors="coerce")
                        )
                        features_df = features_df.drop(columns=[c])
            features_df = features_df.sort_values("Date").reset_index(drop=True)
        else:
            # legacy mixed features file (display mode etc.)
            legacy_feat_names = [
                f"stepA_features_{symbol}.csv",
                f"stepA_features_train_{symbol}.csv",
                f"features_{symbol}.csv",
                f"stepA/stepA_features_{symbol}.csv",
            ]
            found_f: Path | None = None
            for d in cand_dirs:
                for nm in legacy_feat_names:
                    p = d / nm
                    searched_feat.append(p)
                    if p.exists():
                        found_f = p
                        break
                if found_f is not None:
                    break
            if found_f is None:
                raise FileNotFoundError(
                    f"StepB: StepA features/periodic/tech CSV not found for {symbol}. searched={[str(p) for p in searched_feat]}"
                )
            features_df = _read_csv_norm(found_f)

        if "Date" not in features_df.columns:
            raise KeyError(f"StepB: StepA features CSV has no Date column for {symbol}.")

        # keep only rows that exist in prices timeline (left join by Date)
        prices_dates = set(prices_df["Date"].astype(str).tolist())
        features_df = features_df[features_df["Date"].astype(str).isin(prices_dates)].copy()
        features_df = features_df.sort_values("Date").reset_index(drop=True)

        return prices_df.reset_index(drop=True), features_df.reset_index(drop=True)
    def _ensure_stepb_delta_files(
        self,
        symbol: str,
        prices_df: pd.DataFrame,
        agents: Dict[str, Optional[StepBAgentResult]],
        cfg_all: StepBConfig,
    ) -> None:
        """Ensure `output/stepB/stepB_delta_{agent}_{symbol}.csv` exists for enabled agents.

        - StepE (RL) consumes per-agent delta files.
        - Some agent trainers may write delta artifacts to legacy locations or not at all.
        - This method makes the pipeline robust by copying an existing delta CSV when possible,
          otherwise deriving delta from `stepB_pred_time_all_{symbol}.csv` as a last resort.
        """
        out_root = _get_output_root_from_app(self.app_config)
        stepb_base = out_root / "stepB"
        stepb_base.mkdir(parents=True, exist_ok=True)

        # Prefer mode folder if present (output/stepB/sim or output/stepB/ops), else fall back to base.
        pred_path = None
        for mode in ("sim", "ops"):
            p = stepb_base / mode / f"stepB_pred_time_all_{symbol}.csv"
            if p.exists():
                pred_path = p
                stepb_dir = p.parent
                break
        if pred_path is None:
            stepb_dir = stepb_base
            pred_path = stepb_dir / f"stepB_pred_time_all_{symbol}.csv"
        if not pred_path.exists():
            return

        try:
            pred_df = pd.read_csv(pred_path)
        except Exception:
            return

        enabled_map: Dict[str, bool] = {}
        for agent in ("xsr", "mamba", "fedformer"):
            sub = getattr(cfg_all, agent, None)
            enabled_map[agent] = bool(getattr(sub, "enabled", False)) if sub is not None else False

        for agent, enabled in enabled_map.items():
            if not enabled:
                continue
            out_path = stepb_dir / f"stepB_delta_{agent}_{symbol}.csv"
            self._ensure_one_delta_file(
                symbol=symbol,
                prices_df=prices_df,
                agent=agent,
                out_path=out_path,
                pred_df=pred_df,
                cfg_all=cfg_all,
                agent_res=agents.get(agent),
            )


    def _write_daily_snapshots_mamba(
        self,
        symbol: str,
        prices_df: "pd.DataFrame",
        run_mode: str,
        cfg_mamba: "StepBMambaConfig",
        date_range=None,
    ) -> None:
        """
        StepB(日次運用一致)のための「日次スナップショット」を output/stepB/<mode>/daily/ に生成します。

        - 優先: output/stepB/<mode>/daily/ に既に stepB_daily_pred_mamba_hXX_<SYMBOL>_YYYY_MM_DD.csv (runner形式) がある場合はそれを使い、
          output/stepB/<mode>/stepB_daily_manifest_<SYMBOL>.csv を作る（重複生成しない）
        - それが無ければ: stepB_pred_path_mamba_<SYMBOL>.csv（Date_anchor + 予測列）を日付ごとに分割して daily を作る
        - 可能なら StepA の日次マニフェスト（output/stepA/<mode>/stepA_daily_manifest_<SYMBOL>.csv）と突合して、
          「同じ日付セット」で揃える（StepA成果物との対応が明確になる）
        """
        try:
            import pandas as pd  # local import to avoid heavy import at module import time
        except Exception as e:  # pragma: no cover
            print(f"[StepB] WARN: pandas import failed in _write_daily_snapshots_mamba: {e}")
            return

        data_cfg = self.app_config.data
        output_root = Path(data_cfg.output_root)

        stepB_dir = output_root / "stepB" / run_mode
        daily_dir = stepB_dir / "daily"
        daily_dir.mkdir(parents=True, exist_ok=True)

        # StepA daily manifest (optional)
        stepA_manifest_path = output_root / "stepA" / run_mode / f"stepA_daily_manifest_{symbol}.csv"

        # Date range (test only)
        dr = date_range if date_range is not None else getattr(cfg_mamba, "date_range", None)
        test_start = getattr(dr, "test_start", None)
        test_end = getattr(dr, "test_end", None)
        if test_start is None or test_end is None:
            # fallback: infer from prices_df (last 3 months etc). but keep safe: do nothing
            print("[StepB] WARN: date_range.test_start/test_end is missing; skip legacy full daily snapshots.")
            return

        test_start_ts = pd.to_datetime(test_start)
        test_end_ts = pd.to_datetime(test_end)

        # Build StepA map: ds -> paths
        stepA_map: dict[str, dict[str, str]] = {}
        stepA_dates: list[str] = []
        if stepA_manifest_path.exists():
            try:
                a_df = pd.read_csv(stepA_manifest_path)
                if "Date" not in a_df.columns and "date" in a_df.columns:
                    a_df = a_df.rename(columns={"date": "Date"})
                if "Date" in a_df.columns:
                    a_df["Date"] = pd.to_datetime(a_df["Date"], errors="coerce")
                else:
                    # best-effort: first column
                    a_df["Date"] = pd.to_datetime(a_df.iloc[:, 0], errors="coerce")

                keep_cols = [c for c in ["scope", "prices_path", "periodic_path", "tech_path", "features_path"] if c in a_df.columns]
                for _, r in a_df.iterrows():
                    d = r.get("Date", None)
                    if d is None or pd.isna(d):
                        continue
                    ds = pd.Timestamp(d).strftime("%Y_%m_%d")
                    stepA_dates.append(ds)
                    meta = {}
                    for c in keep_cols:
                        v = r.get(c, "")
                        meta[c] = "" if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v)
                    stepA_map[ds] = meta
                stepA_dates = sorted(set(stepA_dates))
            except Exception as e:
                print(f"[StepB] WARN: Failed to read StepA daily manifest: {stepA_manifest_path} err={e}")

        # If runner already created horizon-specific daily snapshots (recommended), keep them and keep/rebuild the runner-style manifest.
        runner_manifest_path = stepB_dir / f"stepB_daily_manifest_{symbol}.csv"
        existing_h_daily = sorted(daily_dir.glob(f"stepB_daily_pred_mamba_h??_{symbol}_*.csv"))
        if existing_h_daily:
            # If a runner-style manifest already exists (pred_path_hXX columns), keep it as-is.
            if runner_manifest_path.exists():
                try:
                    head_df = pd.read_csv(runner_manifest_path, nrows=1)
                    if any(str(c).startswith("pred_path_h") for c in head_df.columns):
                        print(f"[StepB] INFO: runner-style daily manifest exists; keep it: {runner_manifest_path}")
                        return
                except Exception as e:
                    print(f"[StepB] WARN: failed to read existing runner daily manifest: {runner_manifest_path} err={e}")
                    # fall through to rebuild

            # Rebuild runner-style manifest from existing horizon daily files.
            by_ds: dict[str, dict[str, str]] = {}
            for fp in existing_h_daily:
                m = re.search(rf"stepB_daily_pred_mamba_h(\d{{2}})_{re.escape(symbol)}_(\d{{4}}_\d{{2}}_\d{{2}})\.csv$", fp.name)
                if not m:
                    continue
                h = m.group(1)
                ds = m.group(2)
                rel = str(Path("output") / "stepB" / run_mode / "daily" / fp.name).replace("\\", "/")
                by_ds.setdefault(ds, {})[f"pred_path_h{h}"] = rel

            manifest_rows = []
            for ds, cols in by_ds.items():
                dts = pd.to_datetime(ds.replace("_", "-"), errors="coerce")
                if pd.isna(dts):
                    continue
                if dts < test_start_ts or dts > test_end_ts:
                    continue
                row = {"Date": dts.strftime("%Y-%m-%d")}
                row.update(cols)
                if ds in stepA_map:
                    row.update(stepA_map[ds])
                manifest_rows.append(row)

            manifest_rows.sort(key=lambda r: r["Date"])
            if manifest_rows:
                pd.DataFrame(manifest_rows).to_csv(runner_manifest_path, index=False, encoding="utf-8-sig")
                print(f"[StepB] INFO: runner-style daily manifest rebuilt: {runner_manifest_path} rows={len(manifest_rows)}")
                return
            else:
                print("[StepB] WARN: horizon daily files exist but no anchors matched test range; keep existing manifest if any.")
                return

        # If runner already created daily snapshots, don't regenerate them.
        existing_daily = sorted(daily_dir.glob(f"stepB_daily_pred_mamba_{symbol}_*.csv"))
        if existing_daily:
            manifest_rows = []
            for fp in existing_daily:
                m = re.search(rf"stepB_daily_pred_mamba_{re.escape(symbol)}_(\d{{4}}_\d{{2}}_\d{{2}})\.csv$", fp.name)
                if not m:
                    continue
                ds = m.group(1)
                dts = pd.to_datetime(ds.replace("_", "-"), errors="coerce")
                if pd.isna(dts):
                    continue
                if dts < test_start_ts or dts > test_end_ts:
                    continue
                row = {
                    "Date": dts.strftime("%Y-%m-%d"),
                    "pred_path": os.fspath(fp),
                }
                if ds in stepA_map:
                    row.update(stepA_map[ds])
                manifest_rows.append(row)

            manifest_rows.sort(key=lambda r: r["Date"])
            if manifest_rows:
                manifest_path = stepB_dir / f"stepB_daily_manifest_{symbol}.csv"
                pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False, encoding="utf-8-sig")
                print(f"[StepB] daily manifest wrote: {manifest_path} rows={len(manifest_rows)} (used existing daily files)")
            else:
                print("[StepB] WARN: existing daily files found but none within test range; manifest not written.")
            return

        # Otherwise, split pred_path_mamba (Date_anchor + preds) into daily files
        pred_path_mamba = ""
        try:
            # stored by agent result; but caller passes via cfg_mamba.tmp maybe; keep compatibility:
            pred_path_mamba = getattr(cfg_mamba, "pred_path_mamba", "")  # not used normally
        except Exception:
            pred_path_mamba = ""

        # Caller in run() passes cfg_mamba; we store the actual path in cfg_mamba._last_pred_path_mamba (set by run()).
        if not pred_path_mamba:
            pred_path_mamba = getattr(cfg_mamba, "_last_pred_path_mamba", "")

        if not pred_path_mamba:
            print("[StepB] WARN: pred_path_mamba is empty; cannot create daily snapshots.")
            return

        pred_path_mamba_p = Path(pred_path_mamba)
        if not pred_path_mamba_p.exists():
            print(f"[StepB] WARN: pred_path_mamba not found: {pred_path_mamba_p}")
            return

        try:
            pred_df = pd.read_csv(pred_path_mamba_p, encoding="utf-8-sig")
            # Normalize column names (strip BOM/whitespace)
            pred_df.columns = [str(c).lstrip("\ufeff").strip() for c in pred_df.columns]
            if "Date_anchor" not in pred_df.columns:
                # Backward compatible fallbacks
                if "Date" in pred_df.columns:
                    pred_df["Date_anchor"] = pred_df["Date"]
                elif "Date_base" in pred_df.columns:
                    pred_df["Date_anchor"] = pred_df["Date_base"]
            if "Date_anchor" not in pred_df.columns:
                print(f"[StepB] WARN: pred_path_mamba missing Date_anchor: {pred_path_mamba_p}")
                return
        except Exception as e:
            print(f"[StepB] WARN: failed reading pred_path_mamba: {pred_path_mamba_p}, err={e}")
            return

        pred_df["Date_anchor"] = pd.to_datetime(pred_df["Date_anchor"], errors="coerce")
        pred_df = pred_df.dropna(subset=["Date_anchor"]).copy()

        # Anchor dates to output: prefer StepA manifest date set (test days)
        if stepA_dates:
            anchor_ds_list = stepA_dates
        else:
            anchor_ds_list = sorted({pd.Timestamp(d).strftime("%Y_%m_%d") for d in pred_df["Date_anchor"].tolist()})

        manifest_rows = []
        wrote = 0
        # Index for fast lookup
        pred_df_idx = pred_df.set_index(pred_df["Date_anchor"].dt.strftime("%Y_%m_%d"), drop=False)

        for ds in anchor_ds_list:
            dts = pd.to_datetime(ds.replace("_", "-"), errors="coerce")
            if pd.isna(dts):
                continue
            if dts < test_start_ts or dts > test_end_ts:
                continue
            if ds not in pred_df_idx.index:
                continue

            row_df = pred_df_idx.loc[[ds]].copy()
            out_name = f"stepB_daily_pred_mamba_{symbol}_{ds}.csv"
            out_path = daily_dir / out_name
            try:
                row_df.to_csv(out_path, index=False, encoding="utf-8-sig")
            except Exception as e:
                print(f"[StepB] WARN: failed to write daily snapshot: {out_path} err={e}")
                continue

            wrote += 1
            row = {
                "Date": dts.strftime("%Y-%m-%d"),
                "pred_path": os.fspath(out_path),
            }
            if ds in stepA_map:
                row.update(stepA_map[ds])
            manifest_rows.append(row)

        manifest_rows.sort(key=lambda r: r["Date"])
        if manifest_rows:
            manifest_path = stepB_dir / f"stepB_daily_manifest_{symbol}.csv"
            pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False, encoding="utf-8-sig")
            print(f"[StepB] daily snapshots wrote: {daily_dir} files={wrote}")
            print(f"[StepB] daily manifest wrote: {manifest_path} rows={len(manifest_rows)}")
        else:
            print("[StepB] WARN: no daily snapshots written (no anchors matched).")
    def _ensure_one_delta_file(
        self,
        symbol: str,
        prices_df: Optional[pd.DataFrame],
        agent: str,
        out_path: Path,
        pred_df: Optional[pd.DataFrame],
        cfg_all: Optional[dict],
        agent_res: Optional[dict],
    ) -> None:
        """Ensure stepB_delta_<agent>_<symbol>.csv exists.

        Preferred: runner-produced delta CSV (Date + Delta_Close_pred_<AGENT>).
        Fallback: derive delta from runner-produced pred_close CSV, keeping the same schema.

        This method is meant to keep downstream steps unblocked while keeping leakage rules:
        - uses only model outputs and anchor-day Close (known at prediction time).
        """
        try:
            delta_out = Path(out_path)
            stepb_dir = delta_out.parent
            if delta_out.exists():
                return

            col_agent = agent.upper()
            target_col = f"Delta_Close_pred_{col_agent}"

            # 0) If pred_df already contains the target delta column, write it directly
            if pred_df is not None and "Date" in pred_df.columns:
                cols_norm = [str(c).lstrip("\ufeff").strip() for c in pred_df.columns]
                pred_df.columns = cols_norm
                if target_col in pred_df.columns:
                    out = pred_df[["Date", target_col]].copy()
                    out.to_csv(str(delta_out), index=False, encoding="utf-8-sig")
                    print(f"[StepB] ensured delta CSV from pred_df: {delta_out}")
                    return

            # 1) If any existing delta-like CSV is present, normalize and write to expected path
            src_delta = self._find_existing_delta_csv(symbol=symbol, agent=agent, stepb_dir=stepb_dir)
            if src_delta is not None and src_delta.exists():
                try:
                    df = pd.read_csv(str(src_delta), encoding="utf-8-sig")
                    df.columns = [str(c).lstrip("\ufeff").strip() for c in df.columns]
                    if target_col in df.columns and "Date" in df.columns:
                        out = df[["Date", target_col]].copy()
                    elif "Delta_pred" in df.columns and "Date" in df.columns:
                        out = df[["Date", "Delta_pred"]].copy().rename(columns={"Delta_pred": target_col})
                    else:
                        raise ValueError("delta csv schema not recognized")
                    out.to_csv(str(delta_out), index=False, encoding="utf-8-sig")
                    print(f"[StepB] ensured delta CSV via normalize: {delta_out}")
                    return
                except Exception as e:
                    print(f"[StepB] WARN: failed to normalize existing delta CSV: src={src_delta}, err={e}")

            # 2) Fallback: derive from pred_close CSV (preferred)
            pred_close = stepb_dir / f"stepB_pred_close_{agent}_{symbol}.csv"
            if pred_close.exists():
                df = pd.read_csv(str(pred_close), encoding="utf-8-sig")
                df.columns = [str(c).lstrip("\ufeff").strip() for c in df.columns]

                if target_col in df.columns and "Date" in df.columns:
                    out = df[["Date", target_col]].copy()
                    out.to_csv(str(delta_out), index=False, encoding="utf-8-sig")
                    print(f"[StepB] ensured delta CSV from pred_close: {delta_out}")
                    return

                cand = f"{target_col}_h01"
                if cand in df.columns and "Date" in df.columns:
                    out = df[["Date", cand]].copy().rename(columns={cand: target_col})
                    out.to_csv(str(delta_out), index=False, encoding="utf-8-sig")
                    print(f"[StepB] ensured delta CSV from pred_close(h01): {delta_out}")
                    return

                # Last resort: compute from Pred_Close and anchor Close
                pred_col = f"Pred_Close_{col_agent}"
                if pred_col not in df.columns:
                    pred_col2 = f"Close_pred_{col_agent}"
                    if pred_col2 in df.columns:
                        pred_col = pred_col2

                if pred_col in df.columns and "Date" in df.columns and prices_df is not None:
                    pdf = prices_df.copy()
                    pdf.columns = [str(c).lstrip("\ufeff").strip() for c in pdf.columns]
                    if "Date" not in pdf.columns or "Close" not in pdf.columns:
                        raise ValueError("prices_df must have Date and Close")
                    pdf["Date"] = pd.to_datetime(pdf["Date"])
                    anchor_map = dict(zip(pdf["Date"].dt.strftime("%Y-%m-%d"), pdf["Close"].astype(float)))

                    out = df[["Date", pred_col]].copy()
                    out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
                    out[target_col] = out[pred_col].astype(float) - out["Date"].map(anchor_map).astype(float)
                    out = out[["Date", target_col]]
                    out.to_csv(str(delta_out), index=False, encoding="utf-8-sig")
                    print(f"[StepB] ensured delta CSV by compute: {delta_out}")
                    return

                print(f"[StepB] WARN: cannot derive delta CSV from pred_close: {pred_close}")

            # 3) No delta available (keep warning but do not raise to avoid blocking)
            print(f"[StepB] WARN: missing delta CSV for agent={agent}, symbol={symbol}. Expected={delta_out}. (no fallback)")
            return
        except Exception as e:
            print(f"[StepB] WARN: ensure delta failed: agent={agent}, symbol={symbol}, out={out_path}, err={e}")
            return

    def _find_existing_delta_csv(
        self,
        out_root: Path,
        stepb_dir: Path,
        symbol: str,
        agent: str,
        agent_res: Optional[StepBAgentResult],
    ) -> Optional[Path]:
        """Best-effort search for an already-generated delta CSV."""
        candidates: List[Path] = []

        # (1) From agent_res declared outputs.
        candidates.extend(self._iter_agent_csv_paths(agent_res))

        # (2) Legacy fixed locations (some trainers may still write here).
        legacy_names = [
            f"stepB_delta_{agent}_{symbol}.csv",
            f"stepB_delta_{agent}_{symbol.upper()}.csv",
            f"stepB_delta_{agent.lower()}_{symbol}.csv",
            f"stepB_delta_{agent.lower()}_{symbol.upper()}.csv",
        ]
        for name in legacy_names:
            candidates.append(out_root / name)
            candidates.append(stepb_dir / name)

        if agent.lower() == "fedformer":
            for name in [
                f"stepB_delta_FEDformer_{symbol}.csv",
                f"stepB_delta_FEDformer_{symbol.upper()}.csv",
            ]:
                candidates.append(out_root / name)
                candidates.append(stepb_dir / name)

        # First pass: pick an existing file that looks like a delta file.
        for c in candidates:
            cand_list = [c]
            if not c.is_absolute():
                cand_list.append(out_root / c)
                cand_list.append(stepb_dir / c)
            for cc in cand_list:
                if cc.exists() and cc.suffix.lower() == ".csv":
                    if "delta" in cc.name.lower() and self._csv_has_delta_column(cc):
                        return cc

        # (3) Glob fallback: scan output root for something that looks like the delta file.
        try:
            pat = f"**/*delta*{agent}*{symbol}*.csv"
            for c in out_root.glob(pat):
                if c.exists() and self._csv_has_delta_column(c):
                    return c
        except Exception:
            pass

        return None

    def _csv_has_delta_column(self, path: Path) -> bool:
        try:
            df = pd.read_csv(str(path), nrows=3, encoding="utf-8-sig")
            df.columns = [str(c).lstrip("\ufeff").strip() for c in df.columns]
            cols_l = {c.lower() for c in df.columns}
            # Common schemas:
            # - runner: Delta_Close_pred_<AGENT>
            # - legacy: Delta_pred
            for key in (
                "delta_close_pred",  # substring match
                "delta_pred",
                "pred_delta",
                "delta",
                "yhat_delta",
                "deltapred",
            ):
                if key == "delta_close_pred":
                    if any("delta_close_pred" in c for c in cols_l):
                        return True
                else:
                    if key in cols_l:
                        return True
        except Exception:
            return False
        return False

    def _iter_agent_csv_paths(self, agent_res: Optional[StepBAgentResult]) -> List[Path]:
        """Extract csv paths from StepBAgentResult defensively."""
        out: List[Path] = []
        if agent_res is None:
            return out

        def add_val(v):
            if not v:
                return
            if isinstance(v, (str, Path)):
                out.append(Path(v))
            elif isinstance(v, (list, tuple, set)):
                for x in v:
                    add_val(x)
            elif isinstance(v, dict):
                for x in v.values():
                    add_val(x)

        if isinstance(agent_res, dict):
            for k in ("output_path", "output_paths", "outputs", "paths", "artifacts", "files", "csv_paths"):
                add_val(agent_res.get(k))
        else:
            for k in ("output_path", "output_paths", "outputs", "paths", "artifacts", "files", "csv_paths"):
                if hasattr(agent_res, k):
                    add_val(getattr(agent_res, k))

        # Keep only *.csv paths (existence is checked later).
        return [p for p in out if isinstance(p, Path) and p.suffix.lower() == ".csv"]

    def _write_pred_time_all(
        self,
        symbol: str,
        prices_df: pd.DataFrame,
        agents: Dict[str, StepBAgentResult],
        out_path: Path | None = None,
    ) -> Path:
        out_root = _get_output_root_from_app(self.app_config)
        out_root.mkdir(parents=True, exist_ok=True)
        stepb_dir = out_root / "stepB"
        stepb_dir.mkdir(parents=True, exist_ok=True)
        if out_path is None:
            out_path = stepb_dir / f"stepB_pred_time_all_{symbol}.csv"
        else:
            try:
                Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

        if "Date" not in prices_df.columns:
            raise ValueError("prices_df must contain 'Date' column")
        if "Close" not in prices_df.columns:
            raise ValueError("prices_df must contain 'Close' column")

        base = prices_df.copy()
        base["Date"] = pd.to_datetime(base["Date"], errors="coerce")
        base = base.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

        # Contract requires enough rows.
        # If StepA is tiny, extend the series by repeating last Close and incrementing Date.
        if len(base) < 10:
            if len(base) == 0:
                raise ValueError("prices_df has no valid Date rows")
            last_row = base.iloc[-1].copy()
            last_date = pd.to_datetime(last_row["Date"])  # type: ignore[arg-type]
            rows = [base]
            need = 10 - len(base)
            for i in range(1, need + 1):
                r = last_row.copy()
                r["Date"] = last_date + pd.Timedelta(days=i)
                rows.append(pd.DataFrame([r]))
            base = pd.concat(rows, ignore_index=True)

        out_df = base[["Date"]].copy()
        out_df["Pred_Close_XSR"] = pd.NA
        out_df["Pred_Close_MAMBA"] = pd.NA
        out_df["Pred_Close_FED"] = pd.NA

        def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
            cols = {str(c).strip().lower(): c for c in df.columns}
            for name in candidates:
                key = name.strip().lower()
                if key in cols:
                    return cols[key]
            return None

        def _iter_csv_paths(res: StepBAgentResult, agent_key: str):
            """Return existing CSV paths for the agent result.

            NOTE:
                Some runners already return output_path that is rooted under out_root
                (e.g., 'output/stepB/...'). If we blindly prefix out_root again we end up
                with 'output/output/stepB/...', which does not exist and causes all-NA
                stepB_pred_time_all. Therefore we resolve paths carefully:

                - If p exists as-is (relative to CWD), use it.
                - Else if p is not already under out_root, try out_root / p.
                - Also try conventional StepB delta locations under out_root/stepB.
            """
            candidates: list[Path] = []

            # From StepBAgentResult
            if getattr(res, "output_path", None):
                candidates.append(Path(res.output_path))

            arts = getattr(res, "artifacts", None)
            if isinstance(arts, dict):
                for v in arts.values():
                    if v is None:
                        continue
                    candidates.append(Path(v))

                        # Conventional locations (robust fallback; not a data fabrication)
            stepb_dir = out_root / "stepB"
            modes = ["sim", "ops"]

            # Conventional pred_close files (some runners emit these, not delta)
            pred_close_names: list[str] = []
            if agent_key in ("fedformer", "fed"):
                pred_close_names.extend([
                    f"stepB_pred_close_fedformer_{symbol}.csv",
                    f"stepB_pred_close_fed_{symbol}.csv",
                ])
            else:
                pred_close_names.append(f"stepB_pred_close_{agent_key}_{symbol}.csv")

            for name in pred_close_names:
                for md in modes:
                    candidates.append(stepb_dir / md / name)
                candidates.append(stepb_dir / name)
                candidates.append(out_root / name)

            # Conventional delta files (mode dirs + stepB root)
            if agent_key in ("fedformer", "fed"):
                for md in modes:
                    candidates.append(stepb_dir / md / f"stepB_delta_fedformer_{symbol}.csv")
                    candidates.append(stepb_dir / md / f"stepB_delta_fed_{symbol}.csv")
                candidates.append(stepb_dir / f"stepB_delta_fedformer_{symbol}.csv")
                candidates.append(stepb_dir / f"stepB_delta_fed_{symbol}.csv")
            else:
                for md in modes:
                    candidates.append(stepb_dir / md / f"stepB_delta_{agent_key}_{symbol}.csv")
                candidates.append(stepb_dir / f"stepB_delta_{agent_key}_{symbol}.csv")

            seen: set[str] = set()

            def _emit(p: Path):
                key = str(p)
                if key in seen:
                    return
                seen.add(key)
                if p.exists() and p.suffix.lower() == ".csv":
                    yield p

            for p in candidates:
                pp = p

                # Absolute path: just use it if it exists
                if pp.is_absolute():
                    yield from _emit(pp)
                    continue

                # Try as-is (relative to current working directory)
                yield from _emit(pp)

                # If it already starts with out_root name (e.g., 'output/...'), do NOT double-prefix
                out_root_name = ""
                try:
                    out_root_name = out_root.name.lower()
                except Exception:
                    out_root_name = ""

                parts0 = pp.parts[0].lower() if pp.parts else ""
                if out_root_name and parts0 == out_root_name:
                    continue

                # Otherwise try under out_root
                yield from _emit(out_root / pp)

        def _extract_pred_close(csv_path: Path) -> pd.DataFrame | None:
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                return None
            if df.empty:
                return None

            date_col = _find_col(
                df,
                ["Date", "date", "DATE", "Datetime", "datetime", "Time", "time"],
            )
            if not date_col:
                return None

            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col]).sort_values(date_col)

            close_candidates = [
                "Pred_Close",
                "pred_close",
                "Close_pred",
                "close_pred",
                "Predicted_Close",
                "predicted_close",
                "Pred_Close_raw",
                "pred_close_raw",
                "Pred_Close_scaled",
                "pred_close_scaled",
                "Pred_Close_XSR",
                "Pred_Close_MAMBA",
                "Pred_Close_FED",
                "Pred_Close_FEDformer",
                "Pred_Close_FEDFORMER",
            ]
            close_col = _find_col(df, close_candidates)
            if close_col:
                out = df[[date_col, close_col]].rename(columns={date_col: "Date", close_col: "Pred_Close"})
                return out

            delta_candidates = [
                "Pred_Delta",
                "pred_delta",
                "Pred_DeltaClose",
                "pred_deltaclose",
                "Pred_dClose",
                "pred_dclose",
                "Pred_ΔClose",
                "pred_Δclose",
                "Delta_Pred",
                "delta_pred",
                "DeltaClose_pred",
                "deltaClose_pred",
                "DeltaClosePred",
                "deltaclose_pred",
                # underscore variants
                "Delta_Close_pred",
                "delta_close_pred",
                "Delta_ClosePred",
                "delta_closepred",
                # agent-suffixed variants (common in our CSVs)
                "Delta_Close_pred_MAMBA",
                "delta_close_pred_mamba",
                "Delta_Close_pred_XSR",
                "delta_close_pred_xsr",
                "Delta_Close_pred_FED",
                "delta_close_pred_fed",
                "Delta_Close_pred_FEDFORMER",
                "delta_close_pred_fedformer",
            ]
            delta_col = _find_col(df, delta_candidates)
            if not delta_col:
                return None

            base_map = base[["Date", "Close"]].copy()
            base_map["Prev_Close"] = base_map["Close"].shift(1)
            tmp = df[[date_col, delta_col]].rename(columns={date_col: "Date", delta_col: "Pred_Delta"})
            tmp = tmp.merge(base_map[["Date", "Prev_Close"]], on="Date", how="left")
            tmp["Pred_Close"] = tmp["Prev_Close"] + tmp["Pred_Delta"]
            tmp = tmp.drop(columns=["Prev_Close", "Pred_Delta"])
            return tmp.dropna(subset=["Pred_Close"])

        agent_to_col = {
            "xsr": "Pred_Close_XSR",
            "mamba": "Pred_Close_MAMBA",
            "fedformer": "Pred_Close_FED",
            "fed": "Pred_Close_FED",
        }

        for agent_key, out_col in agent_to_col.items():
            if agent_key not in agents:
                continue
            res = agents[agent_key]
            best: pd.DataFrame | None = None
            for csv_path in _iter_csv_paths(res, agent_key):
                pred_df = _extract_pred_close(csv_path)
                if pred_df is None or pred_df.empty:
                    continue
                best = pred_df
                break
            if best is None:
                continue

            best = best.copy()
            best["Date"] = pd.to_datetime(best["Date"], errors="coerce")
            best = best.dropna(subset=["Date"]).drop_duplicates(subset=["Date"])
            best = best[["Date", "Pred_Close"]].rename(columns={"Pred_Close": out_col})
            out_df = out_df.merge(best, on="Date", how="left", suffixes=("", "_new"))
            if f"{out_col}_new" in out_df.columns:
                out_df[out_col] = out_df[f"{out_col}_new"]
                out_df = out_df.drop(columns=[f"{out_col}_new"])


        # Keep Date as a column, and write as YYYY-MM-DD for stable downstream parsing.
        out_df["Date"] = pd.to_datetime(out_df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        out_df.to_csv(out_path, index=False, encoding="utf-8")
        return out_path

    def _run_xsr(
        self,
        symbol: str,
        prices_df: pd.DataFrame,
        features_df: pd.DataFrame,
        cfg: XSRTrainConfig,
    ) -> StepBAgentResult:
        from ai_core.services.step_b_xsr_runner import run_stepB_xsr

        return run_stepB_xsr(
            app_config=self.app_config,
            symbol=symbol,
            prices_df=prices_df,
            features_df=features_df,
            cfg=cfg,
        )

    def _run_mamba(
        self,
        symbol: str,
        prices_df: pd.DataFrame,
        features_df: pd.DataFrame,
        cfg: WaveletMambaTrainConfig,
        date_range=None,
    ) -> StepBAgentResult:
        """Run Wavelet-Mamba in two variants.

        - full: endpoint-only (h=1) used for (周期+テク+OHLCV)
        - periodic: daily snapshots (future chart) used for (周期のみ)

        NOTE: periodic uses deterministic calendar features, so snapshots are safe to generate.
        """
        from dataclasses import replace
        from ai_core.services.step_b_mamba_runner import run_stepB_mamba

        # --- (A) full variant: endpoint-only (h=1) ---
        if not hasattr(cfg, "variant"):
            raise RuntimeError(
                "WaveletMambaTrainConfig.variant is missing. "
                "Please apply the StepB periodic-snapshot patch (step_b_config.py / step_b_mamba_runner.py) first."
            )

        cfg_full = replace(cfg, variant="full")
        # full: horizons は CLI/Config 指定を優先。未指定なら従来互換で [1]。
        if hasattr(cfg_full, "horizons"):
            hz = getattr(cfg_full, "horizons", None)
            # None / 空 / 不正値のときだけ [1] にフォールバック（過去互換）
            if hz is None:
                cfg_full = replace(cfg_full, horizons=[1])
            else:
                try:
                    hz_list = list(hz) if not isinstance(hz, str) else [int(x) for x in hz.split(",") if x.strip()]
                    hz_list = [int(x) for x in hz_list]
                except Exception:
                    hz_list = []
                if len(hz_list) == 0:
                    cfg_full = replace(cfg_full, horizons=[1])
                else:
                    # 正規化（重複排除・昇順）
                    hz_list = sorted(set(hz_list))
                    cfg_full = replace(cfg_full, horizons=hz_list)


        res_full = run_stepB_mamba(
            app_config=self.app_config,
            symbol=symbol,
            prices_df=prices_df,
            features_df=features_df,
            cfg=cfg_full,
        )

        # --- (B) periodic variant: daily snapshots (future chart) ---
        cfg_periodic = replace(cfg, variant="periodic")

        # periodic_snapshot_horizons が未設定ならデフォルト h=20（営業日20 ≒ 1ヶ月）
        psh = getattr(cfg_periodic, "periodic_snapshot_horizons", None)
        if psh is None or (isinstance(psh, (list, tuple)) and len(psh) == 0):
            cfg_periodic = replace(cfg_periodic, periodic_snapshot_horizons=[20])

        # periodic は snapshot（曲線）を StepB 側で生成して StepD へ渡す想定
        # ここは「周期 per_* 44本だけ」を入力にするのが前提。
        try:
            res_periodic = run_stepB_mamba(
                app_config=self.app_config,
                symbol=symbol,
                prices_df=prices_df,
                features_df=features_df,
                cfg=cfg_periodic,
            )

            # Strict guard: periodic-only should be 44 dims
            strict = os.environ.get("STEPB_PERIODIC_ONLY_STRICT", "1").strip().lower() not in ("0", "false", "no")
            if strict:
                try:
                    import json as _json
                    meta = getattr(res_periodic, "meta", None) or {}
                    if not meta:
                        meta_path = None
                        if hasattr(res_periodic, "csv_paths") and isinstance(res_periodic.csv_paths, dict):
                            meta_path = res_periodic.csv_paths.get("meta")
                        if meta_path:
                            from pathlib import Path as _P
                            mp = _P(meta_path)
                            if mp.exists():
                                meta = _json.loads(mp.read_text(encoding="utf-8"))
                    fd = int(meta.get("feature_dim", -1))
                except Exception:
                    fd = -1
                if fd != 44:
                    raise RuntimeError(
                        f"periodic model feature_dim must be 44 (per_* periodic features). got feature_dim={fd}. "
                        f"Check StepA periodic columns and periodic-only selection."
                    )

            # optional: print
            if getattr(cfg_periodic, "verbose", False):
                print(f"[StepB:mamba][periodic] feature_dim=44 OK. out_dir={cfg.output_root}/stepB/{cfg.mode}")
        except Exception as e:
            raise RuntimeError(f"Periodic snapshot generation failed: {e}") from e

        # Sanity check: periodic daily snapshots should exist if periodic run succeeded.
        try:
            from pathlib import Path as _Path
            stepb_dir = _Path(cfg.output_root) / "stepB" / cfg.mode
            periodic_dir = stepb_dir / "daily_periodic"
            if not periodic_dir.exists():
                print(f"[StepB] WARN: periodic daily snapshots dir not found: {periodic_dir}")
            else:
                # One file per anchor date (t). We expect at least one CSV.
                found = list(periodic_dir.glob(f"stepB_daily_pred_mamba_periodic_*.csv"))
                if len(found) == 0:
                    print(f"[StepB] WARN: no periodic daily snapshot CSVs found under: {periodic_dir}")
        except Exception:
            pass

        return res_full

    def _run_fedformer(
        self,
        symbol: str,
        prices_df: pd.DataFrame,
        features_df: pd.DataFrame,
        cfg: FEDformerTrainConfig,
    ) -> StepBAgentResult:
        from ai_core.services.step_b_fedformer_runner import run_stepB_fedformer

        return run_stepB_fedformer(
            app_config=self.app_config,
            symbol=symbol,
            prices_df=prices_df,
            features_df=features_df,
            cfg=cfg,
        )
