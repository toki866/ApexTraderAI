from __future__ import annotations

"""
ai_core.features.wavelet_features
---------------------------------
Wavelet 多スケール特徴量生成モジュール。

- WaveletFeatureConfig   : Wavelet フロントエンド用の設定
- WaveletFeatureExtractor: DataFrame の数値列に対して Wavelet detail 系列を追加するクラス

依存:
    - numpy
    - pandas
    - pywt (pip install pywavelets)

パフォーマンス注意:
    - DataFrame に列を1本ずつ追加すると、pandas の内部ブロックが断片化して
      PerformanceWarning（highly fragmented）が出たり、極端に遅くなります。
    - 本実装では「追加列を dict に集めて最後に concat する」方式にして、
      断片化を避けます（速度が大幅に改善します）。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pywt


# ======================================================================
# 1. Config
# ======================================================================


@dataclass
class WaveletFeatureConfig:
    """Wavelet 特徴抽出の設定。

    Parameters
    ----------
    wavelet_type : str
        使用する Wavelet の種類（例: "db4", "haar" など）。
    wavelet_levels : int
        最大何段までの多段 DWT を使うか。
    min_length : int
        Wavelet を適用するための最小系列長。これより短い場合は特徴を付与しない。
    target_columns : Optional[List[str]]
        Wavelet 特徴を付与する対象列名。None の場合は「数値列すべて」。
    prefix : str
        生成される列名のプレフィックス。
    mode : str
        pywt の境界処理モード。
    """

    wavelet_type: str = "db4"
    wavelet_levels: int = 3
    min_length: int = 4
    target_columns: Optional[List[str]] = None
    prefix: str = ""
    mode: str = "periodization"


# ======================================================================
# 2. Wavelet Feature Extractor
# ======================================================================


class WaveletFeatureExtractor:
    """StepA の周期特徴などから Wavelet 多スケール特徴を生成するクラス。

    各数値列ごとに pywt.wavedec で多段DWTを行い、
    各レベルの「detail成分」を時間方向に再構成した系列を特徴量として追加する。
    """

    def __init__(self, config: WaveletFeatureConfig) -> None:
        self.config = config
        self.feature_cols_: List[str] = []
        # Wavelet オブジェクトは列ごとに再生成せず、インスタンスで共有する
        self._wave = pywt.Wavelet(self.config.wavelet_type)

    # ---- 内部ヘルパー -------------------------------------------------

    def _wavelet_details_per_series(self, x: np.ndarray) -> np.ndarray:
        """1次元系列 x に対して、Wavelet detail 成分の再構成系列を返す。

        Returns
        -------
        details : np.ndarray
            shape (T, L_eff)
            L_eff = 実際に適用できたレベル数（<= config.wavelet_levels）
        """
        x = np.asarray(x, dtype=np.float32)
        T = len(x)
        if T < self.config.min_length:
            # 短すぎる場合は特徴を付与しない
            return np.zeros((T, 0), dtype=np.float32)

        # 利用可能な最大レベル
        max_level = pywt.dwt_max_level(T, self._wave.dec_len)
        level = min(self.config.wavelet_levels, max_level)
        if level <= 0:
            return np.zeros((T, 0), dtype=np.float32)

        coeffs = pywt.wavedec(x, self._wave, level=level, mode=self.config.mode)
        # coeffs: [cA_L, cD_L, ..., cD_1]
        details_list: List[np.ndarray] = []

        # detail成分の再構成（各レベル）
        for j in range(1, level + 1):
            # j段目 detail 成分だけを残して再構成
            coeffs_j = [np.zeros_like(c) for c in coeffs]
            coeffs_j[-j] = coeffs[-j]
            rec = pywt.waverec(coeffs_j, self._wave, mode=self.config.mode)

            # waverec の長さを T に合わせる
            if len(rec) > T:
                rec = rec[:T]
            elif len(rec) < T:
                # 長さが足りない場合は末尾を伸ばす
                rec = np.pad(rec, (0, T - len(rec)), mode="edge")

            details_list.append(rec.astype(np.float32))

        if not details_list:
            return np.zeros((T, 0), dtype=np.float32)

        details = np.stack(details_list, axis=1)  # (T, level)
        return details

    # ---- 公開メソッド -------------------------------------------------

    def fit(self, df_feat: pd.DataFrame) -> "WaveletFeatureExtractor":
        """学習用特徴量に基づいて抽出器をフィットする。

        v1 では数値列のリストを保持するのみ。
        """
        df = df_feat
        if self.config.target_columns is not None:
            self.feature_cols_ = list(self.config.target_columns)
        else:
            self.feature_cols_ = df.select_dtypes(include=[np.number]).columns.tolist()
        return self

    def transform(self, df_feat: pd.DataFrame) -> pd.DataFrame:
        """特徴量 DataFrame に Wavelet 多スケール特徴列を追加して返す。

        重要:
            - pandas の断片化を避けるため、新規列は dict に集めて最後に concat する。
        """
        if not self.feature_cols_:
            # fit 前で呼ばれた場合は、その場で推定
            self.fit(df_feat)

        # 入力をコピー（呼び出し元の DataFrame を汚さない）
        df = df_feat.copy()
        index = df.index
        prefix = self.config.prefix

        # 追加列をここに集約して最後に concat する（断片化対策）
        new_cols: Dict[str, np.ndarray] = {}

        for col in self.feature_cols_:
            if col not in df.columns:
                continue

            x = df[col].to_numpy(dtype=np.float32, copy=False)
            details = self._wavelet_details_per_series(x)  # (T, L_eff)
            T, L_eff = details.shape
            if L_eff == 0:
                continue

            for l in range(L_eff):
                new_col = f"{prefix}{col}_wd{l+1}" if prefix else f"{col}_wd{l+1}"
                # numpy 配列をそのまま保持（最後に DataFrame 化）
                new_cols[new_col] = details[:, l]

        if new_cols:
            df_new = pd.DataFrame(new_cols, index=index)
            df = pd.concat([df, df_new], axis=1)

        # index を維持
        df.index = index
        return df

    def fit_transform(self, df_feat: pd.DataFrame) -> pd.DataFrame:
        """学習＆変換をまとめて行うヘルパー。"""
        self.fit(df_feat)
        return self.transform(df_feat)
