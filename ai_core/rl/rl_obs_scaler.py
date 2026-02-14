from __future__ import annotations

"""
ai_core.rl.rl_obs_scaler
========================

StepE / StepF 用の「RL 観測ベクトル」スケール調整ユーティリティ。

目的
----
- 価格 / 出来高 / テクニカル指標 / Envelope特徴 / PnL / Action など、
  スケールの違う特徴量を 1 本の観測ベクトル（例: 24 次元）にまとめて RL に渡す前に、
  学習期間の統計量を使ってスケールを揃える。
- 内部では ai_core.features.scaler_utils.FeatureScaler を利用し、
  「学習期間で fit → 学習 / テスト / 将来データを transform」の流れを共通化する。

想定する使い方（StepEService 内など）
------------------------------------
1. 観測ベクトルを DataFrame として組み立てる（行=日付、列=観測成分）。
   - 例: columns = [
         "price_open", "price_high", "price_low", "price_close", "volume",
         "rsi_14", "macd", "gap",
         "env_dP_pct", ...,
         "pnl_prev", "action_prev",
     ]

2. DateRange(train_start, train_end, test_end) に基づいて
   学習期間のマスクを作る。

3. RLObservationScaler を生成し、学習期間データで fit する。

   scaler = RLObservationScaler()  # デフォルトは標準化（mean=0, std=1）で Date 列は除外
   scaler.fit(obs_df.loc[train_mask])

4. 学習 + テスト + 将来分を transform してから Env に渡す。

   obs_scaled = scaler.transform(obs_df)

5. 必要なら save() / load() でスケーラを永続化し、
   実運用時に再利用する。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd

from ai_core.features.scaler_utils import (
    FeatureScalerConfig,
    FeatureScaler,
)


# ======================================================================
# RL 観測用スケーラの設定
# ======================================================================


@dataclass
class RLObservationScalerConfig:
    """
    RL 観測ベクトル用スケーラの設定。

    Attributes
    ----------
    method : str
        FeatureScaler と同じスケーリング方法。
        - "standard": (x - mean) / std
        - "minmax":   (x - min) / (max - min) を feature_range に線形変換

    feature_range : tuple[float, float]
        method="minmax" のときの出力レンジ。

    exclude_columns : Optional[List[str]]
        スケール調整から除外する列名リスト。
        - 例: ["Date"] など。
        - None の場合は除外なし。

        Action_prev（-1,0,1）などもスケール対象に含めても学習上は問題ないが、
        離散値の意味を残したい場合は exclude_columns に追加してもよい。

    ignore_non_numeric : bool
        True の場合、非数値列（object, category, datetime など）は自動的に
        スケール対象から外す。
    """

    method: str = "standard"
    feature_range: tuple[float, float] = (-1.0, 1.0)
    exclude_columns: Optional[List[str]] = None
    ignore_non_numeric: bool = True

    def to_feature_scaler_config(self) -> FeatureScalerConfig:
        """
        内部で利用する FeatureScalerConfig に変換する。
        """
        return FeatureScalerConfig(
            method=self.method,
            feature_range=self.feature_range,
            exclude_columns=self.exclude_columns,
            ignore_non_numeric=self.ignore_non_numeric,
        )


# ======================================================================
# RL 観測ベクトル用スケーラ本体
# ======================================================================


class RLObservationScaler:
    """
    RL 観測ベクトル（StepE / StepF の obs_t）用スケーラ。

    役割
    ----
    - 学習期間の観測 DataFrame から統計量（mean/std or min/max）を推定。
    - その統計量を使って学習・テスト・将来分の観測をスケール調整。
    - FeatureScaler を内部に持ち、RL 用の名前・保存方法を提供する薄いラッパ。

    想定する DataFrame の例
    -----------------------
    index : Date（DatetimeIndex）
    columns : [
        "Open", "High", "Low", "Close", "Volume",           # 価格・出来高
        "RSI_14", "MACD", "Gap",                           # テクニカル
        "env_dP_pct", "env_theta_norm", ... ,              # Envelope 幾何特徴 11次元
        "AI_xsr_dclose", "AI_mamba_dclose", "AI_fed_dclose", # 3AI ΔClose予測
        "PnL_prev", "Action_prev",                         # 前日の履歴
    ]
    """

    def __init__(self, config: Optional[RLObservationScalerConfig] = None) -> None:
        # デフォルト設定
        if config is None:
            config = RLObservationScalerConfig(
                method="standard",
                feature_range=(-1.0, 1.0),
                exclude_columns=["Date"],  # Date 列があれば除外
                ignore_non_numeric=True,
            )
        self.config = config
        self._feature_scaler = FeatureScaler(config.to_feature_scaler_config())

    # ------------------------------------------------------------------
    # 学習・適用
    # ------------------------------------------------------------------

    def fit(self, df_obs_train: pd.DataFrame) -> "RLObservationScaler":
        """
        学習期間の観測 DataFrame に対してスケーラを fit する。

        Parameters
        ----------
        df_obs_train : pd.DataFrame
            学習期間の観測ベクトル（行=日付、列=観測成分）。

        Returns
        -------
        self : RLObservationScaler
        """
        self._feature_scaler.fit(df_obs_train)
        return self

    def transform(self, df_obs: pd.DataFrame) -> pd.DataFrame:
        """
        学習済みスケーラを使って任意の観測 DataFrame をスケール調整する。

        Parameters
        ----------
        df_obs : pd.DataFrame
            スケール調整したい観測ベクトル。
            学習期間・テスト期間・将来分など、どの期間でもよい。

        Returns
        -------
        df_scaled : pd.DataFrame
            スケール済み観測ベクトル。
        """
        return self._feature_scaler.transform(df_obs)

    def fit_transform(self, df_obs_train: pd.DataFrame) -> pd.DataFrame:
        """
        学習 & 変換をまとめて行うヘルパー。

        Parameters
        ----------
        df_obs_train : pd.DataFrame
            学習期間の観測ベクトル。

        Returns
        -------
        df_scaled_train : pd.DataFrame
            スケール済みの学習期間観測。
        """
        return self._feature_scaler.fit(df_obs_train).transform(df_obs_train)

    def inverse_transform(self, df_scaled: pd.DataFrame) -> pd.DataFrame:
        """
        スケール済み観測を元のスケールに戻す。

        デバッグや可視化（元スケールでの分布比較など）に使うことを想定。
        """
        return self._feature_scaler.inverse_transform(df_scaled)

    # ------------------------------------------------------------------
    # save / load（実運用向け）
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        スケーラ全体を辞書形式にシリアライズする。

        - RLObservationScalerConfig
        - 内部の FeatureScaler の状態
        """
        return {
            "config": {
                "method": self.config.method,
                "feature_range": list(self.config.feature_range),
                "exclude_columns": self.config.exclude_columns,
                "ignore_non_numeric": self.config.ignore_non_numeric,
            },
            "feature_scaler": self._feature_scaler.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RLObservationScaler":
        """
        to_dict() で保存した辞書から RLObservationScaler を復元する。
        """
        cfg_data = data.get("config", {})
        config = RLObservationScalerConfig(
            method=cfg_data.get("method", "standard"),
            feature_range=tuple(cfg_data.get("feature_range", (-1.0, 1.0))),
            exclude_columns=cfg_data.get("exclude_columns", None),
            ignore_non_numeric=cfg_data.get("ignore_non_numeric", True),
        )
        obj = cls(config=config)
        obj._feature_scaler = FeatureScaler.from_dict(data["feature_scaler"])
        return obj

    def save(self, path: Path) -> None:
        """
        JSON ファイルとしてディスクに保存する。

        Parameters
        ----------
        path : Path
            保存先のパス（例: output/rl/obs_scaler_SOXL.json）
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._feature_scaler.save(path)  # FeatureScaler.save は config+state を JSON 化して保存

    @classmethod
    def load(cls, path: Path) -> "RLObservationScaler":
        """
        save() で保存したファイルから RLObservationScaler を復元する。

        Note
        ----
        FeatureScaler.save/load は config と state を丸ごと扱うので、
        RLObservationScalerConfig はファイルには含まれない。
        ここでは、保存されている FeatureScaler の config をそのまま使う。
        """
        path = Path(path)
        feature_scaler = FeatureScaler.load(path)
        # FeatureScalerConfig はファイル側に含まれているのでそれを流用する
        fs_config = feature_scaler.config
        config = RLObservationScalerConfig(
            method=fs_config.method,
            feature_range=fs_config.feature_range,
            exclude_columns=fs_config.exclude_columns,
            ignore_non_numeric=fs_config.ignore_non_numeric,
        )
        obj = cls(config=config)
        obj._feature_scaler = feature_scaler
        return obj

    # ------------------------------------------------------------------
    # 学習・テスト一括ユーティリティ
    # ------------------------------------------------------------------

    def fit_on_train_and_transform_all(
        self,
        df_obs_all: pd.DataFrame,
        train_mask: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        便利関数：
        - train_mask=True の行で fit
        - 全行を transform
        - ついでに「学習部分」「非学習部分」を分けて返す

        Parameters
        ----------
        df_obs_all : pd.DataFrame
            全期間（学習＋テスト＋将来）の観測ベクトル。
        train_mask : pd.Series
            index を df_obs_all.index に合わせた bool マスク。
            True の行を学習期間として扱う。

        Returns
        -------
        df_scaled_all : pd.DataFrame
            全期間のスケール済み観測。
        df_scaled_train : pd.DataFrame
            学習期間部分だけを抜き出したスケール済み観測。
        """
        df_train = df_obs_all.loc[train_mask]
        self.fit(df_train)
        df_scaled_all = self.transform(df_obs_all)
        df_scaled_train = df_scaled_all.loc[train_mask]
        return df_scaled_all, df_scaled_train
