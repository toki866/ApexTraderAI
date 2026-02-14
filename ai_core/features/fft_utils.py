from __future__ import annotations

"""
FFT ユーティリティモジュール
============================

株価時系列などの 1 次元系列に対して FFT ベースの処理を提供する。

主な機能
--------
- estimate_phase_lag_freq:
    実系列 y_true と予測系列 y_pred の「位相差 vs 周波数」から、
    整数日ラグ（または実数ラグ）を推定する。

- rfft_1d / irfft_1d:
    numpy の rfft / irfft をラップした簡易関数。

想定用途
--------
- XSRModel 内部のラグ推定ロジックを、このモジュールの
  estimate_phase_lag_freq() に置き換えることで、
  「FFT 前処理」をモデル本体から分離する。

依存
----
- numpy
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# ============================================================================
# 1. 汎用 FFT ラッパ
# ============================================================================


def rfft_1d(x: np.ndarray, d: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    1次元実数系列に対して rFFT を実行し、(freqs, spectrum) を返す。

    Parameters
    ----------
    x : np.ndarray
        shape (T,) の実数系列。
    d : float, default 1.0
        サンプル間隔。日足なら 1.0（日）で良い。

    Returns
    -------
    freqs : np.ndarray
        rFFT に対応する周波数軸。shape (T_rfft,)。
    spectrum : np.ndarray
        複素数スペクトル。shape (T_rfft,)。
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=d)
    return freqs, spectrum


def irfft_1d(spectrum: np.ndarray, n: int, d: float = 1.0) -> np.ndarray:
    """
    rFFT スペクトルから時間領域系列を再構成する。

    Parameters
    ----------
    spectrum : np.ndarray
        np.fft.rfft で得られた複素数スペクトル。
    n : int
        再構成する時間軸の長さ。
    d : float, default 1.0
        サンプル間隔。ここでは通常 1.0 固定で良い。

    Returns
    -------
    x_rec : np.ndarray
        shape (n,) の実数系列。
    """
    x_rec = np.fft.irfft(spectrum, n=n)
    return x_rec


# ============================================================================
# 2. ラグ推定（位相 vs 周波数の傾きから求める）
# ============================================================================


@dataclass
class FFTLagConfig:
    """
    FFT ベースのラグ推定用設定。

    Parameters
    ----------
    sample_spacing : float
        サンプル間隔（時間単位）。日足なら 1.0（日）で良い。

    min_freq : float
        傾き推定に使う最小周波数 [同じ時間単位^-1]。
        0 に近すぎる周波数はノイズやトレンドの影響が大きいので、
        0.0 よりほんの少し大きい値を指定することを推奨。

    max_freq : Optional[float]
        傾き推定に使う最大周波数。None の場合は Nyquist まで。

    max_abs_lag : Optional[float]
        推定されたラグの絶対値の上限。None の場合は制限なし。
        単位は sample_spacing と同じ（例: 日）。

    use_unwrap : bool
        位相差を np.unwrap で連続化するかどうか。

    weight_by_power : bool
        周波数ごとのパワーを重みとして線形回帰を行うかどうか。

    return_float_lag : bool
        True の場合は実数ラグを返し、False の場合は round() して整数にする。
    """

    sample_spacing: float = 1.0
    min_freq: float = 0.0
    max_freq: Optional[float] = None
    max_abs_lag: Optional[float] = None
    use_unwrap: bool = True
    weight_by_power: bool = True
    return_float_lag: bool = True


@dataclass
class FFTLagResult:
    """
    FFT ベースのラグ推定結果。

    Attributes
    ----------
    lag : float
        推定されたラグ。単位は config.sample_spacing（例: 日）。
        config.return_float_lag=False のときは整数値。

    slope : float
        位相差 φ(f) を f に対して重み付き線形回帰したときの傾き。
        近似的に φ(f) ≒ intercept + slope * f となる。

    intercept : float
        線形回帰の切片。

    used_freqs : np.ndarray
        回帰に実際に使用した周波数配列。

    phase_diff : np.ndarray
        used_freqs に対応する位相差（ラジアン）。unwrap 済みかどうかは config に依存。
    """

    lag: float
    slope: float
    intercept: float
    used_freqs: np.ndarray
    phase_diff: np.ndarray


def estimate_phase_lag_freq(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    config: Optional[FFTLagConfig] = None,
) -> FFTLagResult:
    """
    周波数領域の位相差から時間ラグを推定する。

    概要
    ----
    連続時間信号 x(t) と、それを τ だけシフトした x(t - τ) は、
    周波数領域で位相が以下のように異なる:

        X_shift(f) = X(f) * exp(-j 2π f τ)

    よって、X_shift と X の位相差 φ(f) は

        φ(f) = arg(X_shift(f)) - arg(X(f)) ≒ -2π f τ

    となるので、φ(f) を f に対して線形回帰したときの傾き slope から

        τ ≒ - slope / (2π)

    としてラグを推定できる。

    Parameters
    ----------
    y_true : np.ndarray
        実測系列。shape (T,)。
    y_pred : np.ndarray
        予測系列（ラグがあると想定）。shape (T,)。
    config : Optional[FFTLagConfig]
        ラグ推定の設定。None の場合はデフォルト設定を使用。

    Returns
    -------
    result : FFTLagResult
        推定ラグ、回帰係数、使用した周波数・位相差など。
    """
    if config is None:
        config = FFTLagConfig()

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true.shape {y_true.shape} != y_pred.shape {y_pred.shape}")

    n = y_true.shape[0]
    if n < 4:
        raise ValueError("系列長が短すぎます（4 サンプル未満）")

    # DC成分を除去しておくと、長期トレンドの影響を少し減らせる
    y_true_centered = y_true - np.nanmean(y_true)
    y_pred_centered = y_pred - np.nanmean(y_pred)

    # 欠損がある場合は簡易的に線形補間
    if np.any(~np.isfinite(y_true_centered)) or np.any(~np.isfinite(y_pred_centered)):
        idx = np.arange(n)

        def _interp_nan(a: np.ndarray) -> np.ndarray:
            mask = np.isfinite(a)
            if mask.sum() < 2:
                # 全部 NaN など極端な場合はゼロにしておく
                return np.nan_to_num(a, nan=0.0)
            return np.interp(idx, idx[mask], a[mask])

        y_true_centered = _interp_nan(y_true_centered)
        y_pred_centered = _interp_nan(y_pred_centered)

    # rFFT
    freqs, Y_true = rfft_1d(y_true_centered, d=config.sample_spacing)
    _, Y_pred = rfft_1d(y_pred_centered, d=config.sample_spacing)

    # rFFT の周波数は [0, f_Nyquist]。0 Hz (DC) はラグ推定に使わない。
    # min_freq, max_freq でフィルタリング。
    f = freqs
    if config.max_freq is None:
        freq_mask = f > max(config.min_freq, 0.0)
    else:
        freq_mask = (f > max(config.min_freq, 0.0)) & (f <= config.max_freq)

    # 最低でも2点以上必要
    if freq_mask.sum() < 2:
        raise ValueError("ラグ推定に使える周波数ポイントが足りません。min_freq/max_freq を見直してください。")

    f_sel = f[freq_mask]
    phase_true = np.angle(Y_true[freq_mask])
    phase_pred = np.angle(Y_pred[freq_mask])

    # 位相差を計算（予測 - 実測）
    phase_diff = phase_pred - phase_true

    # unwrap して位相のジャンプを滑らかにする
    if config.use_unwrap:
        phase_diff = np.unwrap(phase_diff)

    # 重み（パワー）を計算
    if config.weight_by_power:
        power_true = np.abs(Y_true[freq_mask]) ** 2
        power_pred = np.abs(Y_pred[freq_mask]) ** 2
        weights = power_true + power_pred
        # 全部ゼロになるケースを避ける
        if np.all(weights <= 0):
            weights = None
    else:
        weights = None

    # 重み付き線形回帰で φ(f) ≒ intercept + slope * f をフィット
    # 正規方程式 (X^T W X)^{-1} X^T W y を使う
    X = np.vstack([np.ones_like(f_sel), f_sel]).T  # shape (N, 2)
    y = phase_diff  # shape (N,)

    if weights is not None:
        w = weights
        # 対角 W を展開せず、X^T W X を手計算
        WX = X * w[:, None]
        XT_WX = X.T @ WX
        XT_Wy = X.T @ (w * y)
    else:
        XT_WX = X.T @ X
        XT_Wy = X.T @ y

    # 2x2 なので安全に逆行列を取る
    try:
        beta = np.linalg.solve(XT_WX, XT_Wy)
    except np.linalg.LinAlgError:
        # 特異行列などの場合は、単純な最小二乗にフォールバック
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    intercept, slope = beta  # φ(f) ≒ intercept + slope * f

    # φ(f) ≒ -2π f τ  より、 τ ≒ - slope / (2π)
    lag = -slope / (2.0 * np.pi)

    # ラグの制限
    if config.max_abs_lag is not None:
        max_lag = float(config.max_abs_lag)
        if lag > max_lag:
            lag = max_lag
        elif lag < -max_lag:
            lag = -max_lag

    # 整数ラグにしたい場合は round
    if not config.return_float_lag:
        lag = float(round(lag))

    return FFTLagResult(
        lag=lag,
        slope=float(slope),
        intercept=float(intercept),
        used_freqs=f_sel,
        phase_diff=phase_diff,
    )
