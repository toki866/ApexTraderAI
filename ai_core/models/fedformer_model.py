"""
fedformer_model.py

FEDformer ベースの株価予測モデル。

- 入力: StepA で生成した周期 sin/cos 特徴量 (例: 44 本)
- 出力: ΔClose(t) = Close(t) - Close(t-1) の 1 日先予測
- 用途:
    - StepB: 学習 (fit) / 予測 (predict)
    - StepC: ΔClose_pred から Close_pred_raw を復元し、ScaleCalib を実施

構造のポイント:
    - SeriesDecompose: 移動平均による trend / seasonal 分解
    - FourierBlock: 周波数ドメインで top-K 周波数を選択して強調
    - FEDformerLayer: seasonal 系列に FourierBlock と FFN を適用
    - FEDformerNet: 複数レイヤを積んだ encoder 型モデル
    - FEDformerModel: Dataset / DataLoader と学習ループをラップした高レベル API

この実装は、元論文「FEDformer: Frequency Enhanced Decomposed Transformer」
(https://arxiv.org/abs/2201.12740) の簡略版であり、株価時系列向けに調整している。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

import math
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from ai_core.types.common import DateRange, StepResult


# =========================================================
# ユーティリティ: シンプルな標準化スケーラ
# =========================================================


class SimpleStandardScaler:
    """
    特徴量行列 X に対する簡易標準化器。

    - fit(X): mean_, scale_ を推定
    - transform(X): (X - mean_) / scale_
    - inverse_transform(X_scaled): X_scaled * scale_ + mean_
    """

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> None:
        self.mean_ = X.mean(axis=0, keepdims=True)
        self.scale_ = X.std(axis=0, keepdims=True) + 1e-8

    def transform(self, X: np.ndarray) -> np.ndarray:
        # ★ BUG FIX: self.mean → self.mean_
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler has not been fitted yet.")
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler has not been fitted yet.")
        return X_scaled * self.scale_ + self.mean_


# =========================================================
# Series Decompose Block
# =========================================================


class SeriesDecompose(nn.Module):
    """
    シンプルな移動平均による trend / seasonal 分解ブロック。

    入力:
        x: Tensor, shape (B, L, C)

    出力:
        seasonal_part, trend_part
    """

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.avg_pool = nn.AvgPool1d(
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
            count_include_pad=False,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, L, C) -> (B, C, L)
        x_perm = x.permute(0, 2, 1)
        trend = self.avg_pool(x_perm)
        trend = trend.permute(0, 2, 1)
        seasonal = x - trend
        return seasonal, trend


# =========================================================
# Fourier Block (簡略版)
# =========================================================


class FourierBlock(nn.Module):
    """
    周波数ドメインで top-K 周波数成分を選択して強調するブロック（簡略版）。

    入力 x: (B, L, C) の seasonal 系列。
    """

    def __init__(self, seq_len: int, k: int = 32):
        super().__init__()
        self.seq_len = seq_len
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        B, L, C = x.shape
        # FFT -> (B, C, L)
        x_freq = torch.fft.rfft(x.permute(0, 2, 1), dim=-1)
        # 振幅が大きい top-k 周波数のみ残す
        amplitudes = torch.abs(x_freq)
        # 周波数軸 dim=-1
        topk = torch.topk(amplitudes, k=min(self.k, amplitudes.shape[-1]), dim=-1)
        mask = torch.zeros_like(x_freq, dtype=torch.bool)
        mask.scatter_(-1, topk.indices, True)
        x_freq_filtered = torch.where(mask, x_freq, torch.zeros_like(x_freq))
        # 逆 FFT
        x_time = torch.fft.irfft(x_freq_filtered, n=L, dim=-1)
        # (B, C, L) -> (B, L, C)
        return x_time.permute(0, 2, 1)


# =========================================================
# FEDformer Layer / Net
# =========================================================


class FEDformerLayer(nn.Module):
    """
    1 層分の FEDformer レイヤ。

    - seasonal 系列に FourierBlock を適用し、FFN で変換。
    - trend 系列は残差のような役割で足し戻す。
    """

    def __init__(self, seq_len: int, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fourier = FourierBlock(seq_len=seq_len, k=32)
        self.proj_in = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        seasonal: torch.Tensor,
        trend: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        seasonal : Tensor
            形状 (B, L, D) の seasonal 系列。
        trend : Tensor
            形状 (B, L, D) の trend 系列。

        Returns
        -------
        seasonal_out, trend_out : Tensor
        """
        # 1) seasonal に FourierBlock
        s = self.proj_in(seasonal)
        s2 = self.fourier(s)
        s = s + self.dropout(s2)
        s = self.norm1(s)

        # 2) FFN
        s2 = self.ffn(s)
        s = s + self.dropout(s2)
        s = self.norm2(s)

        # trend は恒等的に pass
        t = trend
        return s, t


class FEDformerNet(nn.Module):
    """
    FEDformer ベースの encoder ネットワーク。

    入力:
        x: Tensor, shape (B, L, D)

    出力:
        y: Tensor, shape (B, out_dim)
    """

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        n_layers: int,
        d_ff: int,
        out_dim: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        self.input_proj = nn.Linear(d_model, d_model)
        self.decomp = SeriesDecompose(kernel_size=25)
        self.layers = nn.ModuleList(
            [
                FEDformerLayer(
                    seq_len=seq_len,
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        # seq_len の最後の hidden を使って ΔClose を予測
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor
            形状 (B, L, D) の入力特徴量列。

        Returns
        -------
        y : Tensor
            形状 (B, out_dim) の予測値（ΔClose）。
        """
        # 入力プロジェクション
        h = self.input_proj(x)
        seasonal, trend = self.decomp(h)
        for layer in self.layers:
            seasonal, trend = layer(seasonal, trend)
        # 最終 hidden
        h_out = self.norm(seasonal)
        last_hidden = h_out[:, -1, :]  # (B, D)
        y = self.head(last_hidden)  # (B, 1)
        return y


# =========================================================
# Dataset
# =========================================================


class FedformerDataset(Dataset):
    """
    Fedformer 用のシーケンス Dataset。

    - 入力: 連続した特徴量配列 X (T, input_size) と ΔClose 配列 y (T,)
    - seq_len 日ぶんを 1 サンプルとして切り出し、seq_len 日目の ΔClose をターゲットとする。
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        super().__init__()
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.X) - self.seq_len + 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        idx ~ idx+seq_len-1 を 1 シーケンスとして取り出し、最後の日の y をターゲットにする。
        """
        start = idx
        end = idx + self.seq_len
        x_seq = self.X[start:end]  # (seq_len, input_size)
        y_target = self.y[end - 1]  # スライス末尾の ΔClose
        return (
            torch.from_numpy(x_seq).float(),
            torch.tensor(y_target, dtype=torch.float32),
        )


# =========================================================
# Config / Result dataclasses
# =========================================================


@dataclass
class FEDformerConfig:
    """
    FEDformerModel のハイパーパラメータ設定。
    """

    seq_len: int = 64
    d_model: int = 64
    n_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.1
    batch_size: int = 64
    num_epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-5
    device: str = "cpu"  # "cuda" or "cpu"

    # 学習時の DateRange は StepBConfig から渡される想定
    # ここでは placeholder として持たない。


@dataclass
class FEDformerResult(StepResult):
    """
    StepB 用: FEDformer 学習の結果を格納する dataclass。
    """

    best_val_loss: Optional[float] = None
    num_epochs_trained: int = 0
    train_loss_history: Optional[List[float]] = None
    val_loss_history: Optional[List[float]] = None


# =========================================================
# FEDformer Model: 高レベル API
# =========================================================


class FEDformerModel:
    """
    StepB 用の高レベル FEDformer モデルクラス。

    - fit(df_features, df_prices, date_range) で学習
    - predict(df_features, df_prices, date_range) で ΔClose 予測
    """

    def __init__(self, config: FEDformerConfig):
        self.config = config
        # Normalize device string. We accept "auto" to mean:
        #   - "cuda" if available, otherwise "cpu".
        # Also accept "gpu" as an alias for CUDA.
        dev = (getattr(config, "device", None) or "").strip()
        dev_lower = dev.lower()
        if dev_lower in ("", "auto"):
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        elif dev_lower in ("gpu", "cuda"):
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        # Leave strings like "cuda:0", "cpu", "mps" etc. as-is.
        self.device = torch.device(dev)
        self.net: Optional[FEDformerNet] = None
        self.scaler = SimpleStandardScaler()
        self.feature_columns: Optional[List[str]] = None

        # 乱数シードを固定（再現性確保）
        self._set_seed(42)

    # -----------------------------------------------------
    # 内部ユーティリティ
    # -----------------------------------------------------

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # -----------------------------------------------------
    # ネットワーク構築
    # -----------------------------------------------------

    def build(self, input_size: int) -> None:
        """
        入力次元が確定したタイミングでネットワークを構築する。
        """
        if self.net is not None:
            return
        cfg = self.config
        self.net = FEDformerNet(
            seq_len=cfg.seq_len,
            d_model=input_size,
            n_layers=cfg.n_layers,
            d_ff=cfg.d_ff,
            out_dim=1,
            dropout=cfg.dropout,
        ).to(self.device)

    # -----------------------------------------------------
    # データ前処理
    # -----------------------------------------------------

    def _prepare_train_arrays(
        self,
        df_features: pd.DataFrame,
        df_prices: pd.DataFrame,
        date_range: DateRange,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        学習期間の特徴量行列とターゲット配列を作成する内部ヘルパー。

        Returns
        -------
        features_train : np.ndarray
            形状 (T_train, input_size) の特徴量行列。
        targets_train : np.ndarray
            形状 (T_train,) の ΔClose 配列。
        """
        cfg = self.config

        df_feat = df_features.copy()
        df_px = df_prices.copy()

        # 特徴量側に Close が入っていると join 時に列名が衝突するので削除しておく
        if "Close" in df_feat.columns:
            df_feat = df_feat.drop(columns=["Close"])

        # Date 列を index に揃える
        if "Date" in df_feat.columns:
            df_feat = df_feat.set_index("Date")
        if "date" in df_feat.columns:
            df_feat = df_feat.set_index("date")

        if "Date" in df_px.columns:
            df_px = df_px.set_index("Date")
        if "date" in df_px.columns:
            df_px = df_px.set_index("date")

        df_feat = df_feat.sort_index()
        df_px = df_px.sort_index()

        # join して学習期間でフィルタ
        df_all = df_feat.join(df_px[["Close"]], how="inner")
        mask = (df_all.index >= date_range.train_start) & (
            df_all.index <= date_range.train_end
        )
        df_all = df_all.loc[mask]

        if len(df_all) <= cfg.seq_len:
            raise ValueError(
                f"Training data is too short for seq_len={cfg.seq_len}. "
                f"T={len(df_all)}"
            )

        # 特徴量カラム名（Date を除いたすべて）を保存
        self.feature_columns = [
            c for c in df_features.columns if c.lower() != "date"
        ]

        # X: 特徴量行列, y: Close からの ΔClose
        features = df_all[self.feature_columns].values.astype(np.float32)
        close = df_all["Close"].astype(float)
        delta_close = self._make_delta_close(close)
        # diff により先頭が NaN になるので 1 行ずつずらす
        delta_close = delta_close.iloc[1:]
        features = features[1:, :]

        return features, delta_close.values.astype(np.float32)

    def _make_delta_close(self, close: pd.Series) -> pd.Series:
        """
        Close 価格系列から ΔClose(t) = Close(t) - Close(t-1) を計算する。
        """
        close = close.astype(float)
        delta = close.diff()
        return delta

    # -----------------------------------------------------
    # 学習・予測 API
    # -----------------------------------------------------

    def fit(
        self,
        df_features: pd.DataFrame,
        df_prices: pd.DataFrame,
        date_range: DateRange,
    ) -> FEDformerResult:
        """
        StepB 用の学習メイン関数。
        """
        cfg = self.config

        # 学習データ生成
        features_train, targets_train = self._prepare_train_arrays(
            df_features, df_prices, date_range
        )

        if len(features_train) <= cfg.seq_len:
            raise ValueError(
                f"Training data is too short for seq_len={cfg.seq_len}. "
                f"T={len(features_train)}"
            )

        # 特徴量の標準化
        self.scaler.fit(features_train)
        features_scaled = self.scaler.transform(features_train)

        input_size = features_scaled.shape[1]
        self.build(input_size=input_size)
        assert self.net is not None

        dataset = FedformerDataset(
            X=features_scaled,
            y=targets_train,
            seq_len=cfg.seq_len,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=False,
        )

        # optimizer, loss
        optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        criterion = nn.MSELoss()

        train_loss_history: List[float] = []
        val_loss_history: List[float] = []  # 簡略版: 今回は train loss のみ使う

        best_val_loss = float("inf")

        self.net.train()
        for epoch in range(cfg.num_epochs):
            epoch_loss_sum = 0.0
            n_batches = 0
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                y_pred = self.net(X_batch).squeeze(-1)  # (B,)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss_sum += float(loss.item())
                n_batches += 1

            epoch_loss = epoch_loss_sum / max(1, n_batches)
            train_loss_history.append(epoch_loss)
            # 今回は val を用意していないので train_loss をそのまま best_val_loss とみなす
            if epoch_loss < best_val_loss:
                best_val_loss = epoch_loss

        result = FEDformerResult(
            success=True,
            message="FEDformer training finished.",
            best_val_loss=best_val_loss if best_val_loss < float("inf") else None,
            num_epochs_trained=len(train_loss_history),
            train_loss_history=train_loss_history,
            val_loss_history=val_loss_history,
        )
        return result

    def predict(
        self,
        df_features: pd.DataFrame,
        df_prices: pd.DataFrame,
        date_range: DateRange,
    ) -> pd.Series:
        """
        学習済みモデルを用いて、指定された期間に対する ΔClose 予測を返す。

        - date_range.train_start ～ date_range.test_end などの期間を対象に、
          seq_len 日ぶんの特徴量から 1 日先の ΔClose を予測する。
        - 予測結果は Date index を持つ pd.Series として返す。

        Returns
        -------
        pd.Series
            index: Date
            values: ΔClose_pred
        """
        if self.net is None or self.feature_columns is None:
            raise RuntimeError(
                "Model has not been trained yet. Call fit() before predict()."
            )
        cfg = self.config

        # 対象期間の特徴量・価格を join
        df_feat = df_features.copy()
        df_px = df_prices.copy()

        # 特徴量側に Close が入っていると join 時に列名が衝突するので削除しておく
        if "Close" in df_feat.columns:
            df_feat = df_feat.drop(columns=["Close"])

        # Date 列を index に揃える
        if "Date" in df_feat.columns:
            df_feat = df_feat.set_index("Date")
        if "date" in df_feat.columns:
            df_feat = df_feat.set_index("date")

        if "Date" in df_px.columns:
            df_px = df_px.set_index("Date")
        if "date" in df_px.columns:
            df_px = df_px.set_index("date")

        df_feat = df_feat.sort_index()
        df_px = df_px.sort_index()

        # join して期間フィルタ
        df_all = df_feat.join(df_px[["Close"]], how="inner")
        mask = (df_all.index >= date_range.train_start) & (
            df_all.index <= date_range.test_end
        )
        df_all = df_all.loc[mask]

        if len(df_all) <= cfg.seq_len:
            raise ValueError(
                f"Data is too short for prediction with seq_len={cfg.seq_len}."
            )

        # 特徴量と Close を分離
        feature_cols = (
            self.feature_columns
            if self.feature_columns
            else [c for c in df_features.columns if c.lower() != "date"]
        )
        X_all = df_all[feature_cols].values.astype(np.float32)
        close_all = df_all["Close"].astype(float)

        # ΔClose 計算
        delta_close = self._make_delta_close(close_all)
        delta_close = delta_close.iloc[1:]
        X_all = X_all[1:, :]

        # 標準化
        X_scaled = self.scaler.transform(X_all)

        # 予測ループ
        self.net.eval()
        preds: List[float] = []
        pred_dates: List[pd.Timestamp] = []

        with torch.no_grad():
            for end_idx in range(cfg.seq_len - 1, len(df_all)):
                start_idx = end_idx - (cfg.seq_len - 1)
                x_seq = X_scaled[start_idx : end_idx + 1]  # (seq_len, input_size)

                x_tensor = torch.from_numpy(x_seq).unsqueeze(0).to(
                    self.device
                )  # (1, L, input)
                y_pred = self.net(x_tensor).cpu().numpy().ravel()[0]

                date_label = df_all.index[end_idx]
                preds.append(float(y_pred))
                pred_dates.append(date_label)

        pred_series = pd.Series(data=preds, index=pd.Index(pred_dates, name="Date"))
        pred_series.name = "DeltaClose_pred_FED"
        return pred_series

    def save(self, path: Path) -> None:
        """
        モデルパラメータ・Config・スケーラ・特徴量カラム名をまとめて保存する。
        """
        payload = {
            "config": self.config.__dict__,
            "state_dict": self.net.state_dict() if self.net is not None else None,
            "scaler_mean": self.scaler.mean_,
            "scaler_scale": self.scaler.scale_,
            "feature_columns": self.feature_columns,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path) -> "FEDformerModel":
        """
        save() で保存したファイルから FEDformerModel を復元する。
        """
        payload = torch.load(path, map_location="cpu")

        cfg_dict = payload.get("config", {})
        config = FEDformerConfig(**cfg_dict)
        model = cls(config)

        state_dict = payload["state_dict"]
        model.build(input_size=cfg_dict.get("d_model", config.d_model))
        assert model.net is not None
        model.net.load_state_dict(state_dict)

        scaler = SimpleStandardScaler()
        scaler.mean_ = (
            None if payload.get("scaler_mean") is None else payload["scaler_mean"]
        )
        scaler.scale_ = (
            None if payload.get("scaler_scale") is None else payload["scaler_scale"]
        )
        model.scaler = scaler

        model.feature_columns = payload.get("feature_columns", None)

        return model
