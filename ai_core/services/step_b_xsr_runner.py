from __future__ import annotations

from typing import Dict

import pandas as pd
import os

import re

def _stepb_select_periodic_cols(df: pd.DataFrame) -> list[str]:
    """周期 sin/cos 列（44本想定）を抽出する。

    判定ルール:
      - 列名を非英数字で分割した tokens に 'sin' または 'cos' が含まれる列
      - 'Date' は除外

    例:
      - 'wday_sin', 'wday_cos'
      - 'period_12_sin', 'period_12_cos'
      - 'p03_sin', 'p03_cos'
    """
    cols: list[str] = []
    for c in df.columns:
        if str(c) == "Date":
            continue
        tokens = re.split(r"[^A-Za-z0-9]+", str(c).lower())
        if ("sin" in tokens) or ("cos" in tokens):
            cols.append(str(c))
    return cols


def _stepb_filter_periodic_only(df: pd.DataFrame, expected_cols: int = 44) -> pd.DataFrame:
    """features_df を周期 sin/cos 列だけに絞る（Date + periodic）。"""
    if "Date" not in df.columns:
        raise RuntimeError(f"[StepB] features_df has no Date column. columns={list(df.columns)}")

    periodic_cols = _stepb_select_periodic_cols(df)
    if len(periodic_cols) == 0:
        raise RuntimeError(
            "[StepB] periodic sin/cos columns not found in features_df. "
            "Please check StepA feature column names."
        )

    strict = os.environ.get("STEPB_PERIODIC_ONLY_STRICT", "1").strip() != "0"
    if strict and len(periodic_cols) != expected_cols:
        sample = periodic_cols[:30]
        raise RuntimeError(
            f"[StepB] periodic-only mode expects {expected_cols} columns, but found {len(periodic_cols)}. "
            f"sample={sample} ...  (set STEPB_PERIODIC_ONLY_STRICT=0 to allow non-44)"
        )

    out = df[["Date"] + periodic_cols].copy()
    return out


from ai_core.config.app_config import AppConfig
from ai_core.config.step_b_config import XSRTrainConfig
from ai_core.models.xsr_model import XSRModel, XSRConfig
from ai_core.services.step_b_path_utils import _get_output_root_from_app
from ai_core.types.step_b_types import StepBAgentResult


def run_stepB_xsr(
    app_config: AppConfig,
    symbol: str,
    prices_df: pd.DataFrame,
    features_df: pd.DataFrame,
    cfg: XSRTrainConfig,
) -> StepBAgentResult:
    """XSR モデルの学習と ΔClose 予測 CSV 出力を行う。

    出力:
        {output_root}/stepB/stepB_delta_xsr_{symbol}.csv

    期待される列:
        - Date
        - Delta_pred
        - Pred_Close_raw

    Notes
    -----
    - 以前このファイルは `_get_output_root_from_app` を重複 import していたため、
      正本（ai_core.services.step_b_path_utils）に統一した。
    """
    output_root = _get_output_root_from_app(app_config)
    stepb_dir = output_root / "stepB"
    stepb_dir.mkdir(parents=True, exist_ok=True)

    out_path = stepb_dir / f"stepB_delta_xsr_{symbol}.csv"

    # 特徴量列（Date 以外を全部使うシンプル仕様）
    features_df = _stepb_filter_periodic_only(features_df, expected_cols=44)

    # 特徴量列（周期 sin/cos 44本のみ）
    feature_cols = [c for c in features_df.columns if c != "Date"]

    # XSRConfig を構築
    # NOTE: older XSRTrainConfig may not have fft_bins; do not crash.
    fft_bins = getattr(cfg, 'fft_bins', None)
    if fft_bins is None:
        # Default chosen to match typical FFT resolution. Prefer defining cfg.fft_bins explicitly.
        fft_bins = 256
        print('[StepB][XSR] WARN: XSRTrainConfig.fft_bins is missing; using default fft_bins=256')
    xsr_cfg = XSRConfig(
        l2_reg=cfg.l2_reg,
        max_lag_days=cfg.max_lag_days,
        use_phase_lag=cfg.use_phase_lag,
        max_freq=cfg.max_freq,
        fft_bins=fft_bins,
        use_price_scale=cfg.use_price_scale,
        random_state=cfg.random_state,
    )

    model = XSRModel(config=xsr_cfg)

    # 学習 + history 取得
    history = model.fit(
        prices_df=prices_df,
        features_df=features_df,
        feature_cols=feature_cols,
        date_range=cfg.date_range,
    )

    # 予測（DataFrame: Date, Delta_pred, Pred_Close_raw）
    pred_df = model.predict_delta(
        prices_df=prices_df,
        features_df=features_df,
        feature_cols=feature_cols,
        date_range=cfg.date_range,
    )

    # CSV 出力
    pred_df.to_csv(out_path, index=False)

    # メトリクス（history から主な値だけ拾っておく）
    metrics: Dict[str, float] = {}
    if isinstance(history, dict):
        for key in ("train_loss_last", "val_loss_last", "best_val_loss", "lag_days"):
            if key in history and history[key] is not None:
                try:
                    metrics[key] = float(history[key])
                except Exception:
                    pass

    return StepBAgentResult(
        agent_name="xsr",
        output_path=out_path,
        metrics=metrics,
    )