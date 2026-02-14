from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Dict, Any

import numpy as np
import pandas as pd

# FFT ベースのラグ推定を外部ユーティリティに委譲
# （ai_core/features/fft_utils.py を別途作成している前提）
from ai_core.features.fft_utils import FFTLagConfig, estimate_phase_lag_freq


@dataclass
class XSRConfig:
    """
    XSR モデル用の設定。

    - l2_reg        : ΔClose に対する Ridge 回帰の L2 正則化係数
    - max_lag_days  : 位相特性から求めたラグを何日まで許容するか（クリップ用）
    - use_phase_lag : True のときだけ、周波数領域でラグ推定を行う

    v0.6 拡張:
    - max_freq        : ラグ推定で使う周波数の上限（None なら自動）
    - fft_bins        : FFT ビン数（将来拡張用）
    - use_price_scale : True のとき価格スケール前提（将来拡張用）
    - random_state    : 乱数シード（将来拡張用）
    """

    l2_reg: float = 1e-3
    max_lag_days: int = 10
    use_phase_lag: bool = True

    # v0.6 での拡張項目（今はプレースホルダ的に保持しておく）
    max_freq: Optional[float] = None
    fft_bins: int = 512
    use_price_scale: bool = True
    random_state: Optional[int] = None

    # FFT ラグ推定の詳細設定
    # dataclass ではミュータブルなデフォルトは default_factory を使う
    fft_lag_config: FFTLagConfig = field(default_factory=FFTLagConfig)


class XSRModel:
    """
    周期特徴 → ΔClose を予測するシンプルな XSR モデル。

    - Ridge 回帰による ΔClose 予測
    - 教師データは「学習期間の ΔClose」
    - 位相特性から一定日数のラグを推定し、Pred_Close_raw をラグ補正する

    ※ ローパス平滑・移動平均・余計なフィルタは一切かけない。
    """

    def __init__(
        self,
        config: Optional[XSRConfig] = None,
    ) -> None:
        self.config = config or XSRConfig()

        # 学習結果
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

        # ラグ推定結果（日数）
        self.lag_days_: int = 0

        # 学習履歴など
        self.history_: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 1) StepBService 向けの dataset 構築ユーティリティ
    # ------------------------------------------------------------------
    @staticmethod
    def build_dataset(
        df_features: pd.DataFrame,
        df_prices: pd.DataFrame,
        date_range: Any,
        feature_cols: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """
        StepBService 用の簡易 dataset ラッパ。

        Parameters
        ----------
        df_features : DataFrame
            StepA で作成した特徴量テーブル（Date 列を含む）
        df_prices : DataFrame
            価格テーブル（Date, Close を含む）
        date_range : 任意
            ai_core.types.common.DateRange 相当のオブジェクト想定。
            属性 train_start / train_end を持っている前提。

        Returns
        -------
        dict
            {
                "X_train": ndarray (N, D),
                "y_train": ndarray (N,),
                "dates"  : ndarray (N,),
                "close"  : ndarray (N,),
            }
        """
        if feature_cols is None:
            feature_cols = [c for c in df_features.columns if c.lower() != "date"]

        # 日付で join
        # features 側と prices 側の両方に Close 列があると、pd.merge で
        # Close_x / Close_y に分かれてしまう。そのため、prices 側の Close
        # だけ "Close_price" にリネームしてから結合する。
        f = df_features.copy()
        p = df_prices[["Date", "Close"]].copy()
        p = p.rename(columns={"Close": "Close_price"})

        merged = pd.merge(f, p, on="Date", how="inner")
        merged = merged.sort_values("Date").reset_index(drop=True)

        # 学習期間でフィルタ
        mask = (merged["Date"] >= pd.Timestamp(date_range.train_start)) & (
            merged["Date"] <= pd.Timestamp(date_range.train_end)
        )
        merged = merged.loc[mask].reset_index(drop=True)

        if merged.empty:
            raise ValueError("XSRModel.build_dataset: 学習期間内のデータが空です。")

        # ΔClose を作成（prices 側の Close_price を使用）
        close = merged["Close_price"].to_numpy(dtype=float)
        delta = np.diff(close, prepend=close[0])

        X = merged[feature_cols].to_numpy(dtype=float)
        y = delta

        dates = merged["Date"].to_numpy()

        return {
            "X_train": X,
            "y_train": y,
            "dates": dates,
            "close": close,
        }

    # ------------------------------------------------------------------
    # 2) 学習
    # ------------------------------------------------------------------
    def fit(
        self,
        prices_df: pd.DataFrame,
        features_df: pd.DataFrame,
        feature_cols: Optional[Sequence[str]],
        date_range: Any,
    ) -> Dict[str, Any]:
        """
        XSR モデルを学習し、history を返す。

        Parameters
        ----------
        prices_df : DataFrame
        features_df : DataFrame
        feature_cols : list[str] or None
        date_range : 任意
            ai_core.types.common.DateRange 相当。

        Returns
        -------
        dict
            {
                "coef": ndarray,
                "intercept": float,
                "lag_days": int,
                "lag_info": dict or object,
                "final_train_loss": float,
                "best_val_loss": float or nan,
                ...
            }
        """
        ds = self.build_dataset(
            df_features=features_df,
            df_prices=prices_df,
            date_range=date_range,
            feature_cols=feature_cols,
        )

        X = ds["X_train"]
        y = ds["y_train"]

        # 標準化
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True)
        std[std == 0.0] = 1.0
        X_std = (X - mean) / std

        # Ridge 回帰 (L2)
        lam = float(self.config.l2_reg)
        n_features = X_std.shape[1]
        A = X_std.T @ X_std + lam * np.eye(n_features)
        b = X_std.T @ y
        coef = np.linalg.solve(A, b)

        intercept = float(y.mean() - X_std.mean(axis=0) @ coef)

        self.coef_ = coef
        self.intercept_ = intercept

        # 学習データでの予測 ΔClose
        y_hat = X_std @ coef + intercept

        # ラグ推定（ΔClose 実値 vs ΔClose 予測）
        if self.config.use_phase_lag:
            lag_info = estimate_phase_lag_freq(
                y,                 # actual
                y_hat,             # predicted
                self.config.fft_lag_config,
            )

            # FFTLagResult / dict / その他 どれでも落ちないように柔軟に読む
            if hasattr(lag_info, "lag_days"):
                raw_lag = getattr(lag_info, "lag_days")
            elif hasattr(lag_info, "lag"):
                raw_lag = getattr(lag_info, "lag")
            elif isinstance(lag_info, dict):
                raw_lag = lag_info.get("lag_days", lag_info.get("lag", 0))
            else:
                raw_lag = 0

            try:
                raw_lag_val = float(raw_lag)
            except Exception:
                raw_lag_val = 0.0

            lag_days = int(
                np.clip(
                    raw_lag_val,
                    -self.config.max_lag_days,
                    self.config.max_lag_days,
                )
            )
        else:
            lag_info = {"lag": 0}
            lag_days = 0

        self.lag_days_ = lag_days

        # 学習損失なども保持（StepB メトリクス用にキー名を合わせる）
        train_mse = float(np.mean((y_hat - y) ** 2))

        self.history_ = {
            "train_mse": train_mse,
            "train_loss_last": train_mse,
            "val_loss_last": None,
            "best_val_loss": np.nan,
            "lag_info": lag_info,
            "lag_days": lag_days,
        }

        return {
            "coef": coef,
            "intercept": intercept,
            "lag_days": lag_days,
            "lag_info": lag_info,
            "final_train_loss": train_mse,
            "best_val_loss": np.nan,
        }

    # ------------------------------------------------------------------
    # 3) 予測 ΔClose → Pred_Close_raw
    # ------------------------------------------------------------------
    def predict_delta(
        self,
        prices_df: pd.DataFrame,
        features_df: pd.DataFrame,
        feature_cols: Optional[Sequence[str]],
        date_range: Any,
    ) -> pd.DataFrame:
        """
        学習済みモデルを使って ΔClose を予測し、日足で累積して Pred_Close_raw を返す。

        Returns
        -------
        DataFrame
            列: ["Date", "Delta_pred", "Pred_Close_raw"]
        """
        if self.coef_ is None:
            raise RuntimeError("XSRModel.predict_delta: モデルがまだ学習されていません。")

        if feature_cols is None:
            feature_cols = [c for c in features_df.columns if c.lower() != "date"]

        # 日付で join
        # ここも build_dataset と同様に、prices 側の Close を "Close_price"
        # へリネームしてから結合する。
        f = features_df.copy()
        p = prices_df[["Date", "Close"]].copy()
        p = p.rename(columns={"Close": "Close_price"})

        merged = pd.merge(f, p, on="Date", how="inner")
        merged = merged.sort_values("Date").reset_index(drop=True)

        # 予測対象期間は「学習期間の直後〜テスト期間末」など、呼び出し側で制御
        # ここでは単純に date_range.train_start 以降を全部使う。
        mask = merged["Date"] >= pd.Timestamp(date_range.train_start)
        merged = merged.loc[mask].reset_index(drop=True)

        if merged.empty:
            raise ValueError("XSRModel.predict_delta: 予測対象期間のデータが空です。")

        X_all = merged[feature_cols].to_numpy(dtype=float)
        dates_all = merged["Date"].to_numpy()
        # 価格系列は prices 側の Close_price を使う
        close_all = merged["Close_price"].to_numpy(dtype=float)

        # v0.6 簡易実装: 予測期間 ≒ 学習期間のスケールが大きく変わらない仮定で再標準化
        mean = X_all.mean(axis=0, keepdims=True)
        std = X_all.std(axis=0, keepdims=True)
        std[std == 0.0] = 1.0
        X_std_all = (X_all - mean) / std

        y_hat_all = X_std_all @ self.coef_ + self.intercept_

        # ΔClose 予測を累積して Pred_Close_raw を計算
        pred_close = np.empty_like(close_all)
        pred_close[0] = close_all[0]
        for t in range(1, len(close_all)):
            pred_close[t] = pred_close[t - 1] + y_hat_all[t]

        # lag_days_ を反映させる（ラグ分だけシフトさせる）
        lag = int(self.lag_days_)
        if lag != 0:
            # 正のラグ: 予測が遅れている → 予測系列を前に詰める
            if lag > 0:
                pred_close_shifted = np.empty_like(pred_close)
                pred_close_shifted[:-lag] = pred_close[lag:]
                pred_close_shifted[-lag:] = pred_close[-1]
            else:
                # 負のラグ: 予測が先行している → 予測系列を後ろにずらす
                lag_abs = -lag
                pred_close_shifted = np.empty_like(pred_close)
                pred_close_shifted[lag_abs:] = pred_close[:-lag_abs]
                pred_close_shifted[:lag_abs] = pred_close[0]
            pred_close = pred_close_shifted

        out = pd.DataFrame(
            {
                "Date": dates_all,
                "Delta_pred": y_hat_all,
                "Pred_Close_raw": pred_close,
            }
        )
        return out
