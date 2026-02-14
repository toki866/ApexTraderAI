from __future__ import annotations

"""
ai_core.models.mamba_model
--------------------------
Wavelet-Mamba エージェント用モデルクラス群。

- WaveletMambaConfig : ハイパーパラメータ／設定
- MambaBlock         : SSM/Mamba系の1Dブロック
- WaveletMambaNet    : PyTorch ネットワーク本体
- WaveletMambaModel  : 学習・推論・保存をまとめた高レベルラッパ

依存:
    - numpy
    - pandas
    - torch
    - ai_core.features.wavelet_features (WaveletFeatureConfig, WaveletFeatureExtractor)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from ai_core.types.common import DateRange
from ai_core.features.wavelet_features import (
    WaveletFeatureConfig,
    WaveletFeatureExtractor,
)


# ======================================================================
# 1. Config
# ======================================================================


@dataclass
class WaveletMambaConfig:
    """
    Wavelet-Mamba モデル全体の設定。
    """

    # 基本情報
    symbol: str
    seq_len: int = 256
    horizon: int = 1  # v1では1ステップ先ΔClose予測を想定

    # モデル構造
    hidden_dim: int = 96
    num_layers: int = 4
    dropout: float = 0.1
    activation: str = "gelu"
    use_layer_norm: bool = True

    # 学習設定
    batch_size: int = 64
    num_epochs: int = 60
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    loss_type: str = "mse"  # "mse" | "huber"

    # Wavelet フロントエンド
    wavelet_type: str = "db4"
    wavelet_levels: int = 3

    # 実行環境
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    preset: Optional[str] = None  # "cpu_light" | "gpu_standard" | "gpu_heavy" など


# ======================================================================
# 2. Dataset
# ======================================================================


class SequenceDataset(Dataset):
    """
    単一系列用の時系列 Dataset。

    - 入力: Wavelet 付き特徴（feat_array; shape (N, D)）
    - 出力: ΔClose（close_array; shape (N,)）

    seq_len, horizon を指定して (L, D) -> (horizon) のペアを作る。
    """

    def __init__(
        self,
        feat_array: np.ndarray,
        close_array: np.ndarray,
        seq_len: int,
        horizon: int,
    ) -> None:
        super().__init__()
        self.feat_array = feat_array.astype(np.float32)
        self.close_array = close_array.astype(np.float32)
        self.seq_len = int(seq_len)
        self.horizon = int(horizon)

        if self.feat_array.shape[0] != self.close_array.shape[0]:
            raise ValueError("feat_array と close_array の長さが一致していません。")

    def __len__(self) -> int:
        return max(0, self.feat_array.shape[0] - self.seq_len - self.horizon + 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        x : Tensor
            shape (L, D)
        y : Tensor
            shape (horizon,)
        """
        start = idx
        end = idx + self.seq_len
        y_start = end
        y_end = end + self.horizon

        x = self.feat_array[start:end]  # (L, D)
        close_seg = self.close_array[y_start:y_end]  # (horizon,)
        close_prev = self.close_array[end - 1]
        y = close_seg - close_prev  # ΔClose へ変換

        return torch.from_numpy(x), torch.from_numpy(y)


# ======================================================================
# 3. MambaBlock & Net
# ======================================================================


class MambaBlock(nn.Module):
    """
    SSM/Mamba 風の 1D ブロック（簡易版）。
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            shape (B, L, H)

        Returns
        -------
        torch.Tensor
            shape (B, L, H)
        """
        residual = x
        x = x.transpose(1, 2)  # (B, H, L)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (B, L, H)
        x = self.act(x)
        x = self.dropout(x)
        x = self.norm(x + residual)
        return x


class WaveletMambaNet(nn.Module):
    """
    Wavelet 特徴 + Mamba ブロックによる ΔClose 予測ネットワーク。
    """

    def __init__(self, input_dim: int, config: WaveletMambaConfig) -> None:
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.hidden_dim = config.hidden_dim
        self.horizon = config.horizon

        self.input_proj = nn.Linear(input_dim, self.hidden_dim)
        self.blocks = nn.ModuleList(
            [
                MambaBlock(
                    hidden_dim=self.hidden_dim,
                    dropout=config.dropout,
                    use_layer_norm=config.use_layer_norm,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.output_proj = nn.Linear(self.hidden_dim, self.horizon)

        if config.activation.lower() == "gelu":
            self.act = nn.GELU()
        elif config.activation.lower() == "relu":
            self.act = nn.ReLU()
        else:
            self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            shape (B, L, D)  （Wavelet 付き特徴）

        Returns
        -------
        y : torch.Tensor
            shape (B, horizon)
        """
        b, l, d = x.shape
        x = self.input_proj(x)  # (B, L, H)
        x = self.act(x)
        for block in self.blocks:
            x = block(x)  # (B, L, H)
        x_last = x[:, -1, :]  # (B, H)
        y = self.output_proj(x_last)  # (B, horizon)
        return y


# ======================================================================
# 4. WaveletMambaModel 高レベルラッパ
# ======================================================================


class WaveletMambaModel:
    """
    Wavelet-Mamba エージェント用の高レベルラッパ。

    - WaveletFeatureExtractor で特徴を拡張
    - WaveletMambaNet で ΔClose を予測
    - fit / predict / save / load を提供
    """

    def __init__(self, config: WaveletMambaConfig) -> None:
        self.config = config

        wf_config = WaveletFeatureConfig(
    	wavelet_type=config.wavelet_type,
    	wavelet_levels=config.wavelet_levels,
    	min_length=4,          # お好みで調整可（デフォルト4でもOK）
    	target_columns=None,   # None → 数値列すべてに Wavelet 特徴を付与
    	prefix="",             # 必要なら "wm_" などに変更
    	mode="periodization",  # wavelet_features.py のデフォルトと合わせる
	)

        self.wavelet_extractor = WaveletFeatureExtractor(wf_config)

        self.net: Optional[WaveletMambaNet] = None
        self.device: torch.device = self._select_device()
        self.criterion: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self._input_dim: Optional[int] = None  # save/load 用

    # ------------------------------------------------------------------
    # 公開メソッド
    # ------------------------------------------------------------------

    def build(self, input_dim: int) -> None:
        """入力次元が決まった段階でネットワークを構築する。"""
        self._input_dim = input_dim
        self.net = WaveletMambaNet(input_dim=input_dim, config=self.config).to(self.device)

        if self.config.loss_type.lower() == "huber":
            self.criterion = nn.SmoothL1Loss()
        else:
            self.criterion = nn.MSELoss()

        if self.config.optimizer.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.net.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam(
                self.net.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

    
    def build_dataset(
        self,
        df_features: pd.DataFrame,
        df_prices: pd.DataFrame,
        date_range: DateRange,
        valid_ratio: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        StepBService 互換のためのユーティリティ。

        StepBService からは

            train_data = model.build_dataset(
                df_features=features_df,
                df_prices=prices_df,
                date_range=cfg.date_range,
            )
            history = model.fit(**train_data)

        のように呼び出される想定なので、WaveletMambaModel.fit の
        引数名に合わせて辞書を組み立てて返す。

        Parameters
        ----------
        df_features : pd.DataFrame
            StepA で生成した特徴量 DataFrame。
        df_prices : pd.DataFrame
            OHLCV を含む価格 DataFrame。
        date_range : DateRange
            学習／テストに用いる DateRange。
        valid_ratio : float, optional
            検証データ比率。未指定の場合は 0.1 を用いる。

        Returns
        -------
        dict
            WaveletMambaModel.fit(**dict) でそのまま展開できる辞書。
        """
        return {
            "df_feat": df_features,
            "df_price": df_prices,
            "date_range": date_range,
            "valid_ratio": valid_ratio if valid_ratio is not None else 0.1,
        }

    def fit(
        self,
        df_feat: pd.DataFrame,
        df_price: pd.DataFrame,
        date_range: DateRange,
        valid_ratio: float = 0.1,
    ) -> Dict[str, Any]:
        """
        学習を実行するメインメソッド。

        StepBService からは

            train_data = model.build_dataset(...)
            history = model.fit(**train_data)

        のように呼び出され、戻り値の `history` は StepBService 側の
        _run_mamba でそのまま metrics に展開される想定。

        Returns
        -------
        Dict[str, Any]
            以下のキーを含む辞書:

            - best_train_loss : float
            - best_val_loss   : float
            - final_train_loss: float
            - final_val_loss  : float
            - lag_days        : float  (Mamba では 0 固定)
            - n_train_samples : float
            - n_valid_samples : float
            - train_loss_history : List[float]
            - valid_loss_history : List[float]
            - model_path         : str  (現状は空文字のまま)
        """
        # Date列をインデックスに揃える
        feat = df_feat.copy()
        price = df_price.copy()

        if "Date" in feat.columns:
            feat["Date"] = pd.to_datetime(feat["Date"])
            feat = feat.set_index("Date")
        if "Date" in price.columns:
            price["Date"] = pd.to_datetime(price["Date"])
            price = price.set_index("Date")

        # 学習期間のスライス
        train_start = pd.to_datetime(date_range.train_start)
        train_end = pd.to_datetime(date_range.train_end)
        mask = (feat.index >= train_start) & (feat.index <= train_end)
        feat_train = feat.loc[mask].sort_index()
        price_train = price.loc[mask].sort_index()

        # インデックスを揃える
        common_idx = feat_train.index.intersection(price_train.index)
        feat_train = feat_train.loc[common_idx]
        price_train = price_train.loc[common_idx]

        if len(feat_train) <= self.config.seq_len + 1:
            raise ValueError(
                "学習用データが少なすぎます。seq_len や学習期間を見直してください。"
            )

        # Wavelet 特徴の付与
        feat_train_aug = self.wavelet_extractor.fit_transform(feat_train)
        feat_array = feat_train_aug.select_dtypes(include=[np.number]).to_numpy(
            dtype=np.float32
        )

        # Close 配列
        if "Close" not in price_train.columns:
            raise KeyError("df_price に 'Close' 列が必要です。")
        close_array = price_train["Close"].to_numpy(dtype=np.float32)

        # Dataset 作成
        dataset = SequenceDataset(
            feat_array=feat_array,
            close_array=close_array,
            seq_len=self.config.seq_len,
            horizon=self.config.horizon,
        )

        # train/valid 分割（末尾を valid にする）
        n_total = len(dataset)
        n_valid = max(1, int(n_total * valid_ratio))
        n_train = n_total - n_valid
        if n_train <= 0:
            raise ValueError(
                "検証データ比率が大きすぎて学習データが残りません。valid_ratio を下げてください。"
            )

        train_dataset, valid_dataset = torch.utils.data.random_split(
            dataset,
            [n_train, n_valid],
            generator=torch.Generator().manual_seed(42),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False,
        )

        # ネットワーク構築
        input_dim = feat_array.shape[1]
        if self.net is None or self._input_dim != input_dim:
            self.build(input_dim=input_dim)

        assert self.net is not None
        assert self.criterion is not None
        assert self.optimizer is not None

        self.net.train()
        best_valid_loss = float("inf")
        train_history: List[float] = []
        valid_history: List[float] = []

        for epoch in range(self.config.num_epochs):
            # ---- train ----
            self.net.train()
            train_loss_sum = 0.0
            n_train_batches = 0
            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()
                pred = self.net(x)
                loss = self.criterion(pred, y)
                loss.backward()
                self.optimizer.step()

                train_loss_sum += float(loss.item())
                n_train_batches += 1

            train_loss = train_loss_sum / max(1, n_train_batches)
            train_history.append(train_loss)

            # ---- valid ----
            self.net.eval()
            valid_loss_sum = 0.0
            n_valid_batches = 0
            with torch.no_grad():
                for x, y in valid_loader:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    pred = self.net(x)
                    loss = self.criterion(pred, y)
                    valid_loss_sum += float(loss.item())
                    n_valid_batches += 1

            valid_loss = valid_loss_sum / max(1, n_valid_batches)
            valid_history.append(valid_loss)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss

        best_train_loss = float(min(train_history)) if train_history else float("nan")
        final_train_loss = float(train_history[-1]) if train_history else float("nan")
        final_val_loss = float(valid_history[-1]) if valid_history else float("nan")

        metrics: Dict[str, Any] = {
            "best_train_loss": best_train_loss,
            "best_val_loss": float(best_valid_loss),
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            "lag_days": 0.0,
            "n_train_samples": float(len(train_dataset)),
            "n_valid_samples": float(len(valid_dataset)),
            "train_loss_history": train_history,
            "valid_loss_history": valid_history,
            "model_path": "",
        }

        return metrics

    def predict(
        self,
        df_feat: pd.DataFrame,
        df_price: pd.DataFrame,
        date_range: DateRange,
    ) -> pd.Series:
        """
        学習済みモデルを使って ΔClose を予測する。

        Returns
        -------
        pred_delta : pd.Series
            index = Date, values = ΔClose 予測値（horizon=1 を前提）
        """
        if self.net is None:
            raise RuntimeError("Model is not built/loaded. Call fit() or load() first.")

        self.net.eval()

        feat = df_feat.copy()
        price = df_price.copy()

        if "Date" in feat.columns:
            feat["Date"] = pd.to_datetime(feat["Date"])
            feat = feat.set_index("Date")
        if "Date" in price.columns:
            price["Date"] = pd.to_datetime(price["Date"])
            price = price.set_index("Date")

        # 学習＋テスト期間全体をカバー（train_start〜test_end を Timestamp 化してから比較）
        start = pd.to_datetime(date_range.train_start)
        end = pd.to_datetime(date_range.test_end)
        mask = (feat.index >= start) & (feat.index <= end)
        feat_all = feat.loc[mask].sort_index()
        price_all = price.loc[mask].sort_index()

        common_idx = feat_all.index.intersection(price_all.index)
        feat_all = feat_all.loc[common_idx]
        price_all = price_all.loc[common_idx]

        dates = feat_all.index.to_list()

        # Wavelet 特徴（学習済み extractor を使用）
        feat_all_aug = self.wavelet_extractor.transform(feat_all)
        feat_array = feat_all_aug.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)

        # Close 配列
        if "Close" not in price_all.columns:
            raise KeyError("df_price に 'Close' 列が必要です。")
        close_array = price_all["Close"].to_numpy(dtype=np.float32)

        dataset = SequenceDataset(
            feat_array=feat_array,
            close_array=close_array,
            seq_len=self.config.seq_len,
            horizon=self.config.horizon,
        )

        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False,
        )

        preds_list: List[np.ndarray] = []
        self.net.to(self.device)

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                pred = self.net(x)  # (B, horizon)
                if self.config.horizon == 1:
                    pred_np = pred[:, 0].cpu().numpy()
                else:
                    # horizon > 1 の場合はとりあえず1日目だけ
                    pred_np = pred[:, 0].cpu().numpy()
                preds_list.append(pred_np)

        if not preds_list:
            raise ValueError("No data to predict. Check date_range or source data.")

        preds = np.concatenate(preds_list, axis=0)
        # シーケンスの最後の位置に対応する日付を利用
        valid_dates = dates[self.config.seq_len : self.config.seq_len + len(preds)]

        pred_series = pd.Series(preds, index=pd.to_datetime(valid_dates), name="Pred_DeltaClose_Mamba")
        return pred_series

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """
        モデルと設定を保存する。

        Parameters
        ----------
        path : Path
            保存先のファイルパス（.pt 想定）
        """
        if self.net is None:
            raise RuntimeError("No model to save. Call fit() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "config": self.config.__dict__,
            "state_dict": self.net.state_dict(),
            "input_dim": self._input_dim,
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: Path) -> "WaveletMambaModel":
        """
        save() したファイルから WaveletMambaModel を復元する。
        """
        path = Path(path)
        state = torch.load(path, map_location="cpu")

        config_dict = state["config"]
        config = WaveletMambaConfig(**config_dict)

        model = cls(config=config)
        input_dim = state.get("input_dim", None)
        if input_dim is None:
            raise RuntimeError("Saved model does not contain input_dim information.")

        model.build(input_dim=input_dim)
        assert model.net is not None
        model.net.load_state_dict(state["state_dict"])
        model.net.to(model.device)
        return model

    # ------------------------------------------------------------------
    # 内部ユーティリティ
    # ------------------------------------------------------------------

    def _select_device(self) -> torch.device:
        """
        config.device / preset に応じて torch.device を決定する。
        """
        if self.config.device == "cpu":
            return torch.device("cpu")
        if self.config.device == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")

        # "auto" の場合
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
