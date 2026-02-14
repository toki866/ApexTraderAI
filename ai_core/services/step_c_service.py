from __future__ import annotations

"""ai_core.services.step_c_service

StepC: TimeRecon + ScaleCalib

目的
- StepB の予測系列（Pred_Close_*）を読み込み、価格（Close）と日付で整合
- 必要なら「ラグ補正（StepB-LagCalib の結果）」を適用
- 学習期間末尾（デフォルト252日）だけを使って a,b を推定し、
  Pred_Close_scaled_* を生成（価格スケールに揃える）

本ファイルの重要ポイント
- 成果物は run_mode 別に分離して保存する：
    - sim: output/stepC/sim/stepC_pred_time_all_{symbol}.csv
    - ops: output/stepC/ops/stepC_pred_time_all_{symbol}.csv

- 入力（StepA / StepB）も mode-first で探索する。

- Mamba マルチホライズン対応
    StepB mamba runner が出力する
      stepB_pred_close_mamba_{symbol}.csv
    に含まれる Pred_Close_MAMBA_hXX (XX=02,05,10...) を取り込み、
    各ホライズンごとに独立に ScaleCalib を行い、
      Pred_Close_scaled_MAMBA_hXX
    を出力する。

注意
- Close 代入などのダミーはしない。
- StepC は「予測のある日付」を基準に出力する。
  価格（Close）が存在しない将来日付は Close が NaN になるが、
  Pred_Close_scaled_* は a,b を適用して出力する（ops用途）。
"""

from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

from ai_core.utils.paths import resolve_repo_path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ai_core.config.app_config import AppConfig
from ai_core.types.common import DateRange, StepResult


@dataclass
class StepCConfig:
    """設定情報（StepC: TimeRecon + ScaleCalib）。"""

    app_config: AppConfig
    symbol: str
    date_range: DateRange
    calib_window_days: int = 252
    models: List[str] = field(default_factory=lambda: ["xsr", "mamba", "fedformer"])
    lag_days: Dict[str, int] = field(default_factory=dict)
    raw_column_map: Optional[Dict[str, str]] = None
    write_legacy_root_copy: bool = False


@dataclass
class StepCResult(StepResult):
    """StepC（TimeRecon + ScaleCalib）の結果。"""

    pred_all_path: Optional[Path] = None
    xsr_path: Optional[Path] = None
    lstm_path: Optional[Path] = None  # 互換: mamba をここに載せる
    fed_path: Optional[Path] = None


class StepCService:
    """StepC: TimeRecon + ScaleCalib を担当するサービスクラス。"""

    def __init__(
        self,
        app_config: AppConfig,
        calib_window_days: int = 252,
        models: Optional[List[str]] = None,
        lag_days: Optional[Dict[str, int]] = None,
        raw_column_map: Optional[Dict[str, str]] = None,
        write_legacy_root_copy: bool = False,
    ) -> None:
        self.app_config = app_config
        self._cached_output_root = None  # type: ignore
        self.calib_window_days = int(calib_window_days)
        self.models = models or ["xsr", "mamba", "fedformer"]
        self.lag_days = lag_days or {}
        self.raw_column_map = raw_column_map
        self.write_legacy_root_copy = bool(write_legacy_root_copy)

    # ========= 公開メソッド =========


    def _get_output_root(self) -> Path:
        """Best-effort resolver for output_root used by daily snapshot writers."""
        try:
            v = getattr(self, "_cached_output_root", None)
            if v is not None:
                return Path(v)
        except Exception:
            pass
        data_cfg = getattr(self.app_config, "data", None)
        if data_cfg is not None:
            out = getattr(data_cfg, "output_root", None)
            if out:
                return Path(out)
        return resolve_repo_path("output")

    def run(self, symbol: str, date_range: DateRange) -> StepCResult:
        config = StepCConfig(
            app_config=self.app_config,
            symbol=symbol,
            date_range=date_range,
            calib_window_days=self.calib_window_days,
            models=list(self.models),
            lag_days=dict(self.lag_days),
            raw_column_map=self.raw_column_map,
            write_legacy_root_copy=self.write_legacy_root_copy,
        )
        return self._run_with_config(config)

    # ========= 内部実装 =========

    def _run_with_config(self, config: StepCConfig) -> StepCResult:
        try:
            data_cfg = config.app_config.data
            symbol = config.symbol
            dr = config.date_range

            output_root = Path(data_cfg.output_root)

            # StepB op mode (sim/ops) detection
            stepb_mode_raw = (
                getattr(dr, "stepB_mode", None)
                or getattr(dr, "mamba_mode", None)
                or getattr(dr, "op_mode", None)
                or getattr(dr, "mode", None)
                or getattr(config.app_config, "op_mode", None)
                or getattr(config.app_config, "stepB_mode", None)
            )
            stepb_mode = self._normalize_mode(stepb_mode_raw)

            step_a_dir = output_root / "stepA"
            step_b_dir = output_root / "stepB"
            step_a_dir.mkdir(parents=True, exist_ok=True)
            step_b_dir.mkdir(parents=True, exist_ok=True)

            # 1) 価格（Close）
            price_df, price_path = self._load_price_df(step_a_dir, symbol, mode=stepb_mode)
            print(f"[StepC] price_path={price_path}")

            # 2) StepB の予測（pred_time_all + mamba pred_close を統合）
            pred_df, pred_path = self._load_stepb_pred_df(step_b_dir, symbol, mode=stepb_mode)

            # Determine mode from actual StepB input location if possible
            mode_final = None
            try:
                pn = pred_path.parent.name
                if pn in ("sim", "ops"):
                    mode_final = pn
            except Exception:
                mode_final = None
            if mode_final is None:
                mode_final = stepb_mode or "sim"

            step_c_dir = output_root / "stepC" / mode_final
            step_c_dir.mkdir(parents=True, exist_ok=True)

            # 3) マージ + date_range クリップ（train_start〜test_end）
            merged_df = self._merge_and_clip(price_df=price_df, pred_df=pred_df, date_range=dr)

            # 4) モデルごとにスケールキャリブ（a,b推定）し、標準列名を作る
            raw_col_map = self._build_raw_column_map(config=config, pred_df=merged_df)
            calibrations: Dict[str, Dict[str, object]] = {}

            for model_key in config.models:
                mk = str(model_key).strip().lower()
                raw_col = raw_col_map.get(model_key) or raw_col_map.get(mk)
                if not raw_col or raw_col not in merged_df.columns:
                    continue

                # ラグ補正（任意）
                lag = int(config.lag_days.get(model_key, config.lag_days.get(mk, 0)) or 0)

                # a,b 推定（学習末尾 calib_window_days 日だけ）
                calib_mask = self._build_calib_mask(
                    dates=merged_df["date"],
                    train_start=dr.train_start,
                    train_end=dr.train_end,
                    calib_window_days=int(config.calib_window_days),
                )

                series_raw = pd.to_numeric(merged_df[raw_col], errors="coerce")
                if lag != 0:
                    series_raw = series_raw.shift(-lag)

                nn_calib = int(series_raw.loc[calib_mask].notna().sum())
                if nn_calib < 20:
                    print(f"[StepC] skip calib model={mk} raw_col={raw_col} lag={lag} not_enough_nonnull_in_calib nn={nn_calib}")
                    continue

                close_calib = pd.to_numeric(merged_df.loc[calib_mask, "close"], errors="coerce")
                raw_calib = series_raw.loc[calib_mask]
                a, b = self._fit_scale_params(y_true=close_calib, y_pred=raw_calib)

                suffix = self._agent_suffix(mk)

                raw_out = f"Pred_Close_{suffix}"
                scaled_out = f"Pred_Close_scaled_{suffix}"

                # raw output
                merged_df[raw_out] = series_raw
                # scaled output
                merged_df[scaled_out] = a * series_raw + b

                calibrations[mk] = {
                    "raw_column": str(raw_col),
                    "raw_output": raw_out,
                    "scaled_output": scaled_out,
                    "a": float(a),
                    "b": float(b),
                    "lag_days": int(lag),
                    "calib_rows": int(calib_mask.sum()),
                }

                # ---- Mamba multi-horizon ----
                if mk in ("mamba", "lstm"):
                    mh_cols = self._find_mamba_horizon_cols(merged_df)
                    for hcol in mh_cols:
                        s_h = pd.to_numeric(merged_df[hcol], errors="coerce")
                        if lag != 0:
                            s_h = s_h.shift(-lag)

                        nn_h = int(s_h.loc[calib_mask].notna().sum())
                        if nn_h < 20:
                            print(f"[StepC] skip mamba horizon hcol={hcol} lag={lag} not_enough_nonnull_in_calib nn={nn_h}")
                            continue

                        raw_calib_h = s_h.loc[calib_mask]
                        a_h, b_h = self._fit_scale_params(y_true=close_calib, y_pred=raw_calib_h)

                        # keep raw as-is
                        merged_df[hcol] = s_h

                        # scaled
                        scaled_h = self._scaled_horizon_name(hcol)
                        merged_df[scaled_h] = a_h * s_h + b_h

                        key = f"mamba:{hcol}"
                        calibrations[key] = {
                            "raw_column": str(hcol),
                            "raw_output": str(hcol),
                            "scaled_output": scaled_h,
                            "a": float(a_h),
                            "b": float(b_h),
                            "lag_days": int(lag),
                            "calib_rows": int(calib_mask.sum()),
                        }

            # 5) 出力整形
            out_df = self._build_output_df(merged_df=merged_df, raw_col_map=raw_col_map)

            # 6) 保存
            out_path = step_c_dir / f"stepC_pred_time_all_{symbol}.csv"
            out_df.to_csv(out_path, index=False, encoding="utf-8")

            legacy_path = output_root / f"stepC_pred_time_all_{symbol}.csv"
            if config.write_legacy_root_copy:
                out_df.to_csv(legacy_path, index=False, encoding="utf-8")

            # Optional: write daily snapshot CSVs if StepA daily outputs exist
            try:
                self._write_daily_snapshots_from_stepA(out_df=out_df, symbol=symbol, mode=mode_final, step_c_dir=step_c_dir)
            except Exception as e:
                print(f"[StepC] daily snapshots skipped: {e}")

            result = StepCResult(
                success=True,
                message=f"StepC finished for symbol={symbol}",
                details={
                    "input_price_path": str(price_path),
                    "input_pred_path": str(pred_path),
                    "mode": str(mode_final),
                    "output_path": str(out_path),
                    "legacy_output_path": str(legacy_path) if config.write_legacy_root_copy else None,
                    "calibrations": calibrations,
                },
                pred_all_path=out_path,
            )

            if any(str(m).strip().lower() == "xsr" for m in config.models):
                result.xsr_path = out_path
            if any(str(m).strip().lower() in ("mamba", "lstm") for m in config.models):
                result.lstm_path = out_path
            if any(str(m).strip().lower() in ("fed", "fedformer", "fed_former") for m in config.models):
                result.fed_path = out_path

            return result

        except Exception as e:  # noqa: BLE001
            return StepCResult(
                success=False,
                message=f"StepC failed for symbol={config.symbol}: {e}",
                details={"error": repr(e)},
            )

    # ========= 内部ヘルパー =========

    @staticmethod
    def _normalize_mode(mode: object) -> str | None:
        """Normalize mode string to 'sim' or 'ops'."""
        m = str(mode or "").strip().lower()
        if not m:
            return None
        if m in {"sim", "simulation", "backtest", "test"}:
            return "sim"
        if m in {"ops", "op", "live", "prod", "production", "real"}:
            return "ops"
        if m.startswith("sim"):
            return "sim"
        if m.startswith("ops") or m.startswith("live") or m.startswith("prod"):
            return "ops"
        return None

    @staticmethod
    def _normalize_date_column(df: pd.DataFrame) -> pd.DataFrame:
        date_col = None
        for c in df.columns:
            if str(c).lower() == "date":
                date_col = c
                break
        if date_col is None:
            raise KeyError("No date column found (expected 'date' or 'Date').")
        if date_col != "date":
            df = df.rename(columns={date_col: "date"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        return df

    @staticmethod
    def _load_price_df(step_a_dir: Path, symbol: str, mode: object = None) -> Tuple[pd.DataFrame, Path]:

        """StepA の価格 CSV を読み込んで date/close に正規化（mode-first）。

        Preferred (sim/live):
          - stepA_prices_train_<SYMBOL>.csv + stepA_prices_test_<SYMBOL>.csv

        Fallback (display/legacy):
          - stepA_prices_<SYMBOL>.csv
          - prices_<SYMBOL>.csv
        """
        root_dir = step_a_dir.parent
        mode_norm = StepCService._normalize_mode(mode)

        cand_dirs: List[Path] = []
        if mode_norm:
            cand_dirs.append(step_a_dir / mode_norm)
        for m in ("sim", "live", "ops"):
            d = step_a_dir / m
            if d not in cand_dirs:
                cand_dirs.append(d)
        cand_dirs.append(step_a_dir)
        cand_dirs.append(root_dir)

        # 1) Try train/test split first
        train_path: Optional[Path] = None
        test_path: Optional[Path] = None
        for d in cand_dirs:
            p_tr = d / f"stepA_prices_train_{symbol}.csv"
            p_te = d / f"stepA_prices_test_{symbol}.csv"
            if train_path is None and p_tr.exists():
                train_path = p_tr
            if test_path is None and p_te.exists():
                test_path = p_te
            if train_path is not None and test_path is not None:
                break

        if train_path is not None and test_path is not None:
            df_tr = pd.read_csv(train_path)
            df_te = pd.read_csv(test_path)
            df_tr = StepCService._normalize_date_column(df_tr)
            df_te = StepCService._normalize_date_column(df_te)
            df = pd.concat([df_tr, df_te], ignore_index=True).sort_values("date").reset_index(drop=True)
            used_path = train_path
        else:
            # 2) Fallback to legacy single file
            candidates: List[Path] = []
            for d in cand_dirs:
                candidates.append(d / f"stepA_prices_{symbol}.csv")
                candidates.append(d / f"prices_{symbol}.csv")

            price_path: Optional[Path] = None
            for p in candidates:
                if p.exists():
                    price_path = p
                    break
            if price_path is None:
                tried = []
                if train_path is not None:
                    tried.append(str(train_path))
                if test_path is not None:
                    tried.append(str(test_path))
                tried.extend(str(c) for c in candidates)
                raise FileNotFoundError(
                    f"StepA price CSV not found for symbol={symbol} (tried: {', '.join(tried)})"
                )

            df = pd.read_csv(price_path)
            df = StepCService._normalize_date_column(df)
            used_path = price_path

        # normalize close column
        close_col = None
        for c in df.columns:
            if str(c).lower() == "close":
                close_col = c
                break
        if close_col is None:
            raise KeyError(f"Price CSV {used_path} has no Close column.")
        if close_col != "close":
            df = df.rename(columns={close_col: "close"})

        return df[["date", "close"]].copy(), Path(used_path)


    @staticmethod
    def _load_stepb_pred_df(step_b_dir: Path, symbol: str, mode: object = None) -> Tuple[pd.DataFrame, Path]:
        """StepB の予測 CSV を読み込み、mamba pred_close を統合する。"""
        root_dir = step_b_dir.parent
        mode_norm = StepCService._normalize_mode(mode)

        # Search order: explicit mode -> sim/ops -> stepB root -> output root
        dir_candidates: List[Path] = []
        if mode_norm:
            dir_candidates.append(step_b_dir / mode_norm)
        for m in ("sim", "live", "ops"):
            d = step_b_dir / m
            if d not in dir_candidates:
                dir_candidates.append(d)
        dir_candidates.append(step_b_dir)
        dir_candidates.append(root_dir)

        # Main pred_time_all candidates
        pred_time_all_candidates: List[Path] = []
        for d in dir_candidates:
            pred_time_all_candidates.append(d / f"stepB_pred_time_all_{symbol}.csv")
            pred_time_all_candidates.append(d / f"stepB_pred_all_{symbol}.csv")

        pred_time_all_path: Optional[Path] = None
        for p in pred_time_all_candidates:
            if p.exists():
                pred_time_all_path = p
                break

        # Mamba detail file candidates (multi-horizon)
        mamba_candidates: List[Path] = []
        for d in dir_candidates:
            mamba_candidates.append(d / f"stepB_pred_close_mamba_{symbol}.csv")
            mamba_candidates.append(d / f"stepB_pred_close_lstm_{symbol}.csv")
            mamba_candidates.append(d / f"stepB_pred_close_mamba_{symbol}.CSV")

        mamba_path: Optional[Path] = None
        for p in mamba_candidates:
            if p.exists():
                mamba_path = p
                break

        if pred_time_all_path is None and mamba_path is None:
            tried = pred_time_all_candidates + mamba_candidates
            raise FileNotFoundError(
                f"StepB prediction CSV not found for symbol={symbol} (tried: {', '.join(str(c) for c in tried)})"
            )

        # Read base
        if pred_time_all_path is not None:
            base_df = pd.read_csv(pred_time_all_path)
            base_df = StepCService._normalize_date_column(base_df)
            used_path = pred_time_all_path
        else:
            base_df = pd.DataFrame({"date": []})
            used_path = mamba_path  # type: ignore[assignment]

        # Merge mamba detail if present
        if mamba_path is not None:
            df_m = pd.read_csv(mamba_path)
            df_m = StepCService._normalize_date_column(df_m)

            # Ensure columns are in expected case
            # We keep all columns; but ensure Pred_Close_MAMBA exists if possible.
            cols_lower = {str(c).lower(): c for c in df_m.columns}
            if "pred_close_mamba" not in cols_lower:
                # try common alternatives
                for cand in ("Pred_Close", "Close_pred", "pred_close", "close_pred"):
                    if cand.lower() in cols_lower:
                        df_m = df_m.rename(columns={cols_lower[cand.lower()]: "Pred_Close_MAMBA"})
                        break

            if len(base_df) == 0:
                merged = df_m
            else:
                merged = pd.merge(base_df, df_m, on="date", how="outer", suffixes=("", "_mamba"))

                # If both have Pred_Close_MAMBA, prefer the mamba file
                if "Pred_Close_MAMBA_mamba" in merged.columns:
                    merged["Pred_Close_MAMBA"] = merged["Pred_Close_MAMBA_mamba"].combine_first(merged.get("Pred_Close_MAMBA"))
                    merged = merged.drop(columns=["Pred_Close_MAMBA_mamba"])

                # Drop other duplicate *_mamba columns by preferring base (or combine_first)
                for c in list(merged.columns):
                    if c.endswith("_mamba"):
                        base_name = c[: -len("_mamba")]
                        if base_name in merged.columns:
                            merged[base_name] = pd.to_numeric(merged[base_name], errors="coerce").combine_first(
                                pd.to_numeric(merged[c], errors="coerce")
                            )
                            merged = merged.drop(columns=[c])

            merged = merged.sort_values("date").reset_index(drop=True)
            return merged, used_path

        return base_df.sort_values("date").reset_index(drop=True), used_path

    @staticmethod
    def _merge_and_clip(price_df: pd.DataFrame, pred_df: pd.DataFrame, date_range: DateRange) -> pd.DataFrame:
        """価格と予測を日付でマージし、train_start〜test_end にクリップ。

        - 予測日付を基準に残す（how='left' on pred_df）
        - Close が存在しない将来日付は NaN のまま残す（ops用途）
        """
        if "date" not in pred_df.columns:
            raise ValueError("StepC: pred_df has no 'date' column.")
        merged = pd.merge(pred_df, price_df, on="date", how="left", sort=True)

        start = pd.Timestamp(date_range.train_start)
        end = pd.Timestamp(date_range.test_end)

        merged = merged.sort_values("date")
        mask = (merged["date"] >= start) & (merged["date"] <= end)
        return merged.loc[mask].reset_index(drop=True)

    def _build_raw_column_map(self, config: StepCConfig, pred_df: pd.DataFrame) -> Dict[str, str]:
        """model_key -> StepB側の raw 予測列名を推定する。"""
        col_lut = {str(c).lower(): str(c) for c in pred_df.columns}

        def _pick(candidates: List[str]) -> Optional[str]:
            for c in candidates:
                if c in pred_df.columns:
                    return c
                lc = str(c).lower()
                if lc in col_lut:
                    return col_lut[lc]
            return None

        out: Dict[str, str] = {}
        for model_key in config.models:
            mk = str(model_key).strip().lower()

            override = None
            if isinstance(config.raw_column_map, dict):
                override = config.raw_column_map.get(model_key) or config.raw_column_map.get(mk)
            if override:
                chosen = _pick([str(override)])
                if chosen:
                    out[model_key] = chosen
                continue

            if mk in ("xsr",):
                chosen = _pick(["Pred_Close_XSR", "pred_close_xsr", "pred_xsr", "xsr_pred"])
            elif mk in ("mamba", "lstm"):
                chosen = _pick(["Pred_Close_MAMBA", "Pred_Close_LSTM", "pred_close_mamba", "pred_close_lstm", "lstm_pred"])
            elif mk in ("fed", "fedformer", "fed_former"):
                chosen = _pick(["Pred_Close_FED", "Pred_Close_FEDFORMER", "pred_close_fedformer", "pred_close_fed", "fed_pred"])
            else:
                chosen = None

            if chosen:
                out[model_key] = chosen

        return out

    @staticmethod
    def _build_calib_mask(dates: pd.Series, train_start, train_end, calib_window_days: int) -> pd.Series:
        train_start_ts = pd.Timestamp(train_start)
        train_end_ts = pd.Timestamp(train_end)
        calib_start_ts = max(train_start_ts, train_end_ts - timedelta(days=int(calib_window_days) - 1))
        return (dates >= calib_start_ts) & (dates <= train_end_ts)

    @staticmethod
    def _fit_scale_params(y_true: pd.Series, y_pred: pd.Series) -> Tuple[float, float]:
        mask = y_true.notna() & y_pred.notna()
        if int(mask.sum()) < 2:
            return 1.0, 0.0

        y = y_true.loc[mask]
        x = y_pred.loc[mask]

        mean_y = float(y.mean())
        mean_x = float(x.mean())
        std_y = float(y.std())
        std_x = float(x.std())

        if std_x > 0 and std_y > 0:
            a = std_y / std_x
            b = mean_y - a * mean_x
        else:
            a = 1.0
            b = mean_y - mean_x

        return float(a), float(b)

    @staticmethod
    def _agent_suffix(mk: str) -> str:
        mk = (mk or "").strip().lower()
        if mk in ("xsr",):
            return "XSR"
        if mk in ("mamba", "lstm"):
            return "MAMBA"
        if mk in ("fed", "fedformer", "fed_former"):
            return "FED"
        return mk.upper() if mk else "MODEL"

    @staticmethod
    def _find_mamba_horizon_cols(df: pd.DataFrame) -> List[str]:
        cols: List[str] = []
        for c in df.columns:
            s = str(c)
            if s.startswith("Pred_Close_MAMBA_h"):
                cols.append(s)
        # stable sort by numeric suffix if possible
        def _key(x: str):
            try:
                return int(x.split("_h")[-1])
            except Exception:
                return 10_000
        cols.sort(key=_key)
        return cols

    @staticmethod
    def _scaled_horizon_name(raw_hcol: str) -> str:
        # raw_hcol: Pred_Close_MAMBA_hXX
        return raw_hcol.replace("Pred_Close_", "Pred_Close_scaled_", 1)

    @staticmethod
    def _build_output_df(merged_df: pd.DataFrame, raw_col_map: Dict[str, str]) -> pd.DataFrame:
        """StepC出力DFを組み立てる。

        Downstream (StepD/StepE) が期待する列:
          - Date
          - Close
          - Pred_Close_* / Pred_Close_scaled_* (if available)
        """
        if "date" not in merged_df.columns:
            raise ValueError("StepC: merged_df has no 'date' column.")
        if "close" not in merged_df.columns:
            # close may be missing only if StepA missing; treat as error
            raise ValueError("StepC: merged_df has no 'close' column.")

        keep_base = ["date", "close"]
        keep_extra: List[str] = []

        def _add(col: str) -> None:
            if col in merged_df.columns and col not in keep_base and col not in keep_extra:
                keep_extra.append(col)

        # Prefer canonical names first
        for col in merged_df.columns:
            sc = str(col)
            if sc.startswith("Pred_Close_scaled_") or sc.startswith("Pred_Close_"):
                _add(sc)

        # Include raw cols referenced by raw_col_map (debug use)
        for _, raw_col in (raw_col_map or {}).items():
            _add(str(raw_col))

        cols = keep_base + keep_extra
        out = merged_df.loc[:, [c for c in cols if c in merged_df.columns]].copy()
        out = out.rename(columns={"date": "Date", "close": "Close"})
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        return out

    def _write_daily_snapshots_from_stepA(self, out_df: pd.DataFrame, symbol: str, mode: str, step_c_dir: Path) -> None:
        """Create StepC daily snapshot files (1 row per day) for StepA's test dates, if StepA daily outputs exist."""
        if out_df is None or len(out_df) == 0:
            return

        output_root = self._get_output_root()
        step_a_dir = output_root / "stepA" / mode
        dates = self._collect_stepA_daily_dates(step_a_dir=step_a_dir, symbol=symbol)
        if not dates:
            return

        if "Date" not in out_df.columns:
            return

        df = out_df.copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

        daily_dir = step_c_dir / "daily"
        daily_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for dt in dates:
            date_str = dt.strftime("%Y-%m-%d")
            file_date = dt.strftime("%Y_%m_%d")
            one = df.loc[df["Date"] == date_str]
            if one.empty:
                continue
            out_path = daily_dir / f"stepC_daily_pred_time_all_{symbol}_{file_date}.csv"
            one.to_csv(out_path, index=False, encoding="utf-8")
            rows.append({"Date": date_str, "stepC_daily_path": str(out_path.as_posix())})

        if rows:
            manifest_path = step_c_dir / f"stepC_daily_manifest_{symbol}.csv"
            pd.DataFrame(rows).to_csv(manifest_path, index=False, encoding="utf-8")
            print(f"[StepC] wrote daily snapshots: {len(rows)} -> {manifest_path}")

    def _collect_stepA_daily_dates(self, step_a_dir: Path, symbol: str) -> List[pd.Timestamp]:
        """Collect dates from StepA daily manifest if present, else from filenames in StepA daily folder."""
        manifest = step_a_dir / f"stepA_daily_manifest_{symbol}.csv"
        if manifest.exists():
            try:
                mdf = pd.read_csv(manifest)
                for col in ("Date", "date", "DATE"):
                    if col in mdf.columns:
                        dts = pd.to_datetime(mdf[col], errors="coerce").dropna().drop_duplicates().sort_values()
                        return [pd.Timestamp(x).normalize() for x in dts.tolist()]
            except Exception:
                pass

        daily_dir = step_a_dir / "daily"
        if not daily_dir.exists():
            return []

        import re as _re
        pat = _re.compile(rf"{_re.escape(symbol)}_(\d{{4}})_(\d{{2}})_(\d{{2}})\.csv$")
        dates: List[pd.Timestamp] = []
        for p in daily_dir.glob(f"*{symbol}_*.csv"):
            m = pat.search(p.name)
            if not m:
                continue
            y, mo, d = map(int, m.groups())
            try:
                dates.append(pd.Timestamp(year=y, month=mo, day=d).normalize())
            except Exception:
                continue
        return sorted(set(dates))
