from __future__ import annotations

from typing import Dict, Any

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
from ai_core.config.step_b_config import FEDformerTrainConfig
from ai_core.models.fedformer_model import FEDformerModel, FEDformerConfig, FEDformerResult
from ai_core.services.step_b_path_utils import _get_output_root_from_app
from ai_core.types.step_b_types import StepBAgentResult


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


def _maybe_set(cfg_obj: Any, cfg_name: str, target_obj: Any, target_name: str) -> None:
    v = _get_attr(cfg_obj, cfg_name, None)
    if v is None:
        return
    try:
        setattr(target_obj, target_name, v)
    except Exception:
        return


def run_stepB_fedformer(
    app_config: AppConfig,
    symbol: str,
    prices_df: pd.DataFrame,
    features_df: pd.DataFrame,
    cfg: FEDformerTrainConfig,
) -> StepBAgentResult:
    """FEDformer モデルで ΔClose 予測を行い、CSV を出力して返す。

    出力:
        {output_root}/stepB/stepB_delta_fedformer_{symbol}.csv

    期待される列:
        - Date
        - Delta_pred
        - Pred_Close_raw
    """
    # 3AI は周期 sin/cos 44本のみで学習させる（テクニカル系のリークを遮断）
    features_df = _stepb_filter_periodic_only(features_df, expected_cols=44)

    fcfg = FEDformerConfig()

    _maybe_set(cfg, "seq_len", fcfg, "seq_len")
    _maybe_set(cfg, "d_model", fcfg, "d_model")

    # n_layers 互換（cfg が e_layers を持つ場合も想定）
    if _get_attr(cfg, "n_layers", None) is not None:
        _maybe_set(cfg, "n_layers", fcfg, "n_layers")
    else:
        _maybe_set(cfg, "e_layers", fcfg, "n_layers")

    _maybe_set(cfg, "d_ff", fcfg, "d_ff")
    _maybe_set(cfg, "dropout", fcfg, "dropout")
    _maybe_set(cfg, "batch_size", fcfg, "batch_size")
    _maybe_set(cfg, "num_epochs", fcfg, "num_epochs")
    _maybe_set(cfg, "lr", fcfg, "lr")
    _maybe_set(cfg, "weight_decay", fcfg, "weight_decay")
    _maybe_set(cfg, "device", fcfg, "device")

    model = FEDformerModel(config=fcfg)

    # fit / predict（モデルAPI準拠）
    fit_result = model.fit(
        df_features=features_df,
        df_prices=prices_df,
        date_range=cfg.date_range,
    )  # FEDformerResult

    pred_delta = model.predict(
        df_features=features_df,
        df_prices=prices_df,
        date_range=cfg.date_range,
    )  # pd.Series (index=Date, values=ΔClose_pred)

    # ---- write CSV
    out_root = _get_output_root_from_app(app_config)
    stepb_dir = out_root / "stepB"
    stepb_dir.mkdir(parents=True, exist_ok=True)
    out_path = stepb_dir / f"stepB_delta_fedformer_{symbol}.csv"

    df_out = pred_delta.reset_index()
    if df_out.shape[1] >= 2:
        df_out.columns = ["Date", "Delta_pred"]
    else:
        df_out = df_out.rename(columns={df_out.columns[0]: "Date"})
        df_out["Delta_pred"] = pd.NA

    df_out["Date"] = pd.to_datetime(df_out["Date"], errors="coerce")
    df_out["Delta_pred"] = pd.to_numeric(df_out["Delta_pred"], errors="coerce")

    # Prev_Close から Pred_Close_raw を復元
    p = prices_df[["Date", "Close"]].copy()
    p["Date"] = pd.to_datetime(p["Date"], errors="coerce")
    p = p.dropna(subset=["Date"]).drop_duplicates(subset=["Date"]).sort_values("Date")
    p["Prev_Close"] = p["Close"].astype(float).shift(1)
    p = p[["Date", "Prev_Close"]]

    df_out = df_out.merge(p, on="Date", how="left")
    df_out["Pred_Close_raw"] = df_out["Prev_Close"] + df_out["Delta_pred"]
    df_out = df_out.drop(columns=["Prev_Close"])

    df_out = df_out.dropna(subset=["Date"])
    df_out = df_out.sort_values("Date")
    df_out["Date"] = df_out["Date"].dt.strftime("%Y-%m-%d")
    df_out.to_csv(out_path, index=False)

    # ---- metrics
    metrics: Dict[str, float] = {}
    if isinstance(fit_result, FEDformerResult):
        if fit_result.best_val_loss is not None:
            metrics["best_val_loss"] = float(fit_result.best_val_loss)
        metrics["num_epochs_trained"] = float(getattr(fit_result, "num_epochs_trained", 0))
        if fit_result.train_loss_history:
            metrics["train_loss_last"] = float(fit_result.train_loss_history[-1])
        if fit_result.val_loss_history:
            metrics["val_loss_last"] = float(fit_result.val_loss_history[-1])

    return StepBAgentResult(
        agent_name="fedformer",
        output_path=out_path,
        metrics=metrics,
    )