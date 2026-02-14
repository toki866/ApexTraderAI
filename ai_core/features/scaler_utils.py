from __future__ import annotations

"""
特徴量スケール調整ユーティリティ
================================

学習用 DataFrame に対してスケーラを fit し、そのパラメータ
(平均・標準偏差 / min-max) を使って任意の DataFrame を
スケール調整するための共通モジュール。

想定用途
--------
- StepA/StepB など「AIに入力する特徴量」をすべて同じルールでスケール調整する。
- 学習期間だけで fit して、テスト期間・将来データには transform のみ適用する。

使い方（例）
------------
>>> import pandas as pd
>>> from pathlib import Path
>>> from ai_core.features.scaler_utils import FeatureScalerConfig, FeatureScaler
>>>
>>> df_train = pd.read_csv("stepA_features_train.csv", parse_dates=["Date"])
>>> df_test  = pd.read_csv("stepA_features_test.csv", parse_dates=["Date"])
>>>
>>> config = FeatureScalerConfig(
...     method="standard",
...     exclude_columns=["Date", "Close", "DeltaClose"],
... )
>>> scaler = FeatureScaler(config)
>>> scaler.fit(df_train)
>>>
>>> df_train_scaled = scaler.transform(df_train)
>>> df_test_scaled  = scaler.transform(df_test)
>>>
>>> # 後で再利用したい場合は保存
>>> scaler.save(Path("output/feature_scaler.json"))

依存
----
- numpy
- pandas
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd


# ============================================================================
# 1. 設定クラス
# ============================================================================


@dataclass
class FeatureScalerConfig:
    """
    特徴量スケール調整の設定。

    Parameters
    ----------
    method : str
        スケール方法。現在は "standard" または "minmax" をサポート。
        - "standard": (x - mean) / std
        - "minmax":   (x - min) / (max - min) を feature_range に線形変換

    feature_range : tuple[float, float]
        method="minmax" のときの出力レンジ (min, max)。

    exclude_columns : Optional[List[str]]
        スケール調整から除外する列名リスト。
        例: ["Date", "Close", "DeltaClose"] など。
        None の場合は除外なし。

    ignore_non_numeric : bool
        True の場合、非数値列（object, category, datetime など）は
        自動的にスケール対象から外す。
    """

    method: str = "standard"
    feature_range: tuple[float, float] = (0.0, 1.0)
    exclude_columns: Optional[List[str]] = None
    ignore_non_numeric: bool = True


@dataclass
class FeatureScalerState:
    """
    スケーラの学習済みパラメータを保持するクラス。

    Attributes
    ----------
    columns : List[str]
        スケール対象にした列名のリスト。

    method : str
        "standard" または "minmax"。

    params : Dict[str, Dict[str, float]]
        列ごとのパラメータ辞書。
        method="standard" の場合:
            params[col] = {"mean": float, "std": float}
        method="minmax" の場合:
            params[col] = {"min": float, "max": float}

    feature_range : tuple[float, float]
        minmax の変換先レンジ。
    """

    columns: List[str]
    method: str
    params: Dict[str, Dict[str, float]]
    feature_range: tuple[float, float]


# ============================================================================
# 2. メインのスケーラクラス
# ============================================================================


class FeatureScaler:
    """
    pandas.DataFrame 用の特徴量スケーラ。

    - fit(): 学習用 DataFrame から列ごとの統計値を計算
    - transform(): 任意の DataFrame に対してスケール調整を適用
    - inverse_transform(): スケール調整を元に戻す（必要な場合）

    注意
    ----
    - スケーラは「列名」で紐づける。transform() する DataFrame に
      学習時と同じ列が存在しない場合、その列は無視される。
    - 新しい列（学習時になかった列）が存在しても、その列はそのまま残す。
    """

    def __init__(self, config: FeatureScalerConfig) -> None:
        self.config = config
        self.state: Optional[FeatureScalerState] = None

    # ------------------------------------------------------------------
    # 内部ヘルパー
    # ------------------------------------------------------------------

    def _select_target_columns(self, df: pd.DataFrame) -> List[str]:
        """スケール対象にする列名リストを決定する。"""
        cols = list(df.columns)

        # 除外列を取り除く
        if self.config.exclude_columns:
            exclude_set = set(self.config.exclude_columns)
            cols = [c for c in cols if c not in exclude_set]

        # 非数値を除外
        if self.config.ignore_non_numeric:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cols = [c for c in cols if c in numeric_cols]

        return cols

    # ------------------------------------------------------------------
    # 公開メソッド: fit / transform / inverse_transform
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "FeatureScaler":
        """
        DataFrame に対してスケーラをフィットする。

        Parameters
        ----------
        df : pd.DataFrame
            学習用の特徴量 DataFrame。

        Returns
        -------
        self : FeatureScaler
        """
        method = self.config.method.lower()
        if method not in ("standard", "minmax"):
            raise ValueError(f"Unsupported method: {self.config.method}")

        target_cols = self._select_target_columns(df)
        params: Dict[str, Dict[str, float]] = {}

        if method == "standard":
            for col in target_cols:
                series = pd.to_numeric(df[col], errors="coerce")
                mean = float(series.mean())
                std = float(series.std(ddof=0))
                if std == 0.0 or not np.isfinite(std):
                    # 分散ゼロ/NaN の場合はダミー値を入れておく（transform時にそのまま0になる）
                    std = 1.0
                params[col] = {"mean": mean, "std": std}
        else:  # minmax
            for col in target_cols:
                series = pd.to_numeric(df[col], errors="coerce")
                vmin = float(series.min())
                vmax = float(series.max())
                if vmax == vmin or not np.isfinite(vmin) or not np.isfinite(vmax):
                    # 範囲ゼロ/NaN の場合はデフォルト値
                    vmin, vmax = 0.0, 1.0
                params[col] = {"min": vmin, "max": vmax}

        self.state = FeatureScalerState(
            columns=target_cols,
            method=method,
            params=params,
            feature_range=self.config.feature_range,
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        学習済みスケーラを用いて DataFrame をスケール調整する。

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        df_out : pd.DataFrame
            スケール調整された DataFrame。
            非対象列は元のまま残る。
        """
        if self.state is None:
            raise RuntimeError("FeatureScaler is not fitted. Call fit() first.")

        df_out = df.copy()
        method = self.state.method
        feature_min, feature_max = self.state.feature_range
        col_set = set(self.state.columns)

        for col in df_out.columns:
            if col not in col_set:
                continue
            series = pd.to_numeric(df_out[col], errors="coerce")
            # 欠損はとりあえずそのまま残す（必要ならfillnaは別途）
            mask = series.notna()
            values = series[mask].to_numpy(dtype=np.float64)

            param = self.state.params.get(col, None)
            if param is None:
                # 安全のためスキップ
                continue

            if method == "standard":
                mean = param["mean"]
                std = param["std"]
                scaled = (values - mean) / std
            else:
                vmin = param["min"]
                vmax = param["max"]
                denom = vmax - vmin
                if denom == 0.0:
                    # ありえないが防御的
                    scaled = np.zeros_like(values)
                else:
                    scaled_01 = (values - vmin) / denom
                    scaled = feature_min + (feature_max - feature_min) * scaled_01

            series_scaled = series.copy()
            series_scaled[mask] = scaled
            df_out[col] = series_scaled

        return df_out

    def inverse_transform(self, df_scaled: pd.DataFrame) -> pd.DataFrame:
        """
        スケール調整された DataFrame を元のスケールに戻す。

        Parameters
        ----------
        df_scaled : pd.DataFrame

        Returns
        -------
        df_out : pd.DataFrame
            逆変換された DataFrame。
        """
        if self.state is None:
            raise RuntimeError("FeatureScaler is not fitted. Call fit() first.")

        df_out = df_scaled.copy()
        method = self.state.method
        feature_min, feature_max = self.state.feature_range
        col_set = set(self.state.columns)

        for col in df_out.columns:
            if col not in col_set:
                continue
            series = pd.to_numeric(df_out[col], errors="coerce")
            mask = series.notna()
            values = series[mask].to_numpy(dtype=np.float64)

            param = self.state.params.get(col, None)
            if param is None:
                continue

            if method == "standard":
                mean = param["mean"]
                std = param["std"]
                original = values * std + mean
            else:
                vmin = param["min"]
                vmax = param["max"]
                denom = feature_max - feature_min
                if denom == 0.0:
                    original = np.full_like(values, vmin)
                else:
                    scaled_01 = (values - feature_min) / denom
                    original = vmin + (vmax - vmin) * scaled_01

            series_orig = series.copy()
            series_orig[mask] = original
            df_out[col] = series_orig

        return df_out

    # ------------------------------------------------------------------
    # 状態の保存 / 復元
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """スケーラ状態を辞書にシリアライズして返す。"""
        if self.state is None:
            raise RuntimeError("FeatureScaler is not fitted.")
        return {
            "config": asdict(self.config),
            "state": {
                "columns": self.state.columns,
                "method": self.state.method,
                "params": self.state.params,
                "feature_range": list(self.state.feature_range),
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureScaler":
        """辞書から FeatureScaler を復元する。"""
        cfg_dict = data["config"]
        state_dict = data["state"]

        config = FeatureScalerConfig(**cfg_dict)
        scaler = cls(config)

        scaler.state = FeatureScalerState(
            columns=list(state_dict["columns"]),
            method=str(state_dict["method"]),
            params=dict(state_dict["params"]),
            feature_range=tuple(state_dict["feature_range"]),
        )
        return scaler

    def save(self, path: Path) -> None:
        """スケーラ状態を JSON 形式で保存する。"""
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "FeatureScaler":
        """JSON ファイルから FeatureScaler を読み込む。"""
        import json

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


# ============================================================================
# 3. おまけ：学習・テストをまとめて処理するヘルパー
# ============================================================================


def fit_transform_train_test(
    df_train: pd.DataFrame,
    df_test: Optional[pd.DataFrame],
    config: Optional[FeatureScalerConfig] = None,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], FeatureScaler]:
    """
    学習データでスケーラを fit し、学習・テスト両方を一括変換するヘルパー。

    Parameters
    ----------
    df_train : pd.DataFrame
        学習用 DataFrame。
    df_test : Optional[pd.DataFrame]
        テスト用 DataFrame。None の場合は学習データのみ変換。
    config : Optional[FeatureScalerConfig]
        スケーラ設定。None の場合はデフォルト設定を使用。

    Returns
    -------
    df_train_scaled : pd.DataFrame
    df_test_scaled : Optional[pd.DataFrame]
    scaler : FeatureScaler
        学習済みスケーラ。
    """
    if config is None:
        config = FeatureScalerConfig()

    scaler = FeatureScaler(config)
    scaler.fit(df_train)

    df_train_scaled = scaler.transform(df_train)
    df_test_scaled = scaler.transform(df_test) if df_test is not None else None

    return df_train_scaled, df_test_scaled, scaler
