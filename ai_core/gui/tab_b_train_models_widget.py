# tab_b_train_models_widget.py

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QLabel,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QCheckBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QMessageBox,
)

from ai_core.types.common import DateRange
from ai_core.config.step_b_config import StepBConfig
from ai_core.models.mamba_model import WaveletMambaConfig
from ai_core.models.fedformer_model import FEDformerConfig
from ai_core.services.step_b_service import StepBResult, StepBAgentMetrics

# tab_b_train_models_widget.py 冒頭あたり
from backend.backend_controller import BackendController



class TabB_TrainModelsQtWidget(QWidget):
    """
    StepB（学習タブ）の GUI。

    - 上: 共通ハイパラ + Wavelet-Mamba 専用 + FEDformer 専用フォーム
    - 中: 「StepB: 全モデル学習開始」ボタン
    - 下: 学習結果テーブル（XSR / Mamba / FED の train/val loss など）
    """

    def __init__(
        self,
        backend: BackendController,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.backend = backend

        self._build_ui()
        self._connect_signals()

    # =====================================================
    # UI 構築
    # =====================================================

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # 1) ハイパラフォーム群
        self._build_hyperparam_form(layout)

        # 2) 実行ボタン
        btn_layout = QHBoxLayout()
        self.btn_run_stepB = QPushButton("StepB: 全モデル学習開始", self)
        btn_layout.addWidget(self.btn_run_stepB)
        btn_layout.addStretch(1)
        layout.addLayout(btn_layout)

        # 3) 結果テーブル
        self.table_result = QTableWidget(self)
        headers = [
            "Agent",              # xsr / mamba / fedformer
            "Model Type",         # "XSR" / "Wavelet-Mamba" / "FEDformer"
            "Train Loss (last)",
            "Val Loss (last)",
            "Best Val Loss",
            "Epochs",
            "Train Samples",
            "Valid Samples",
            "Lag (days)",         # XSR のみ有効、他は "-"
            "Status / Message",
        ]
        self.table_result.setColumnCount(len(headers))
        self.table_result.setHorizontalHeaderLabels(headers)
        self.table_result.horizontalHeader().setStretchLastSection(True)
        self.table_result.verticalHeader().setVisible(False)
        self.table_result.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table_result.setSelectionBehavior(QTableWidget.SelectRows)

        layout.addWidget(self.table_result)
        self.setLayout(layout)

    def _build_hyperparam_form(self, parent_layout: QVBoxLayout) -> None:
        """
        共通ハイパラ + Wavelet-Mamba + FEDformer 専用ハイパラフォーム。
        """

        # ---------------- 共通 ----------------
        box_common = QGroupBox("共通ハイパーパラメータ", self)
        form_common = QFormLayout(box_common)

        # preset
        self.combo_preset = QComboBox(box_common)
        self.combo_preset.addItems(["custom", "cpu_light", "gpu_standard", "gpu_heavy"])
        self.combo_preset.setCurrentText("gpu_standard")

        # device
        self.combo_device = QComboBox(box_common)
        self.combo_device.addItems(["auto", "cpu", "cuda"])
        self.combo_device.setCurrentText("auto")

        # seq_len
        self.spin_seq_len = QSpinBox(box_common)
        self.spin_seq_len.setRange(16, 4096)
        self.spin_seq_len.setValue(256)

        # horizon
        self.spin_horizon = QSpinBox(box_common)
        self.spin_horizon.setRange(1, 64)
        self.spin_horizon.setValue(1)

        # batch_size
        self.spin_batch_size = QSpinBox(box_common)
        self.spin_batch_size.setRange(1, 4096)
        self.spin_batch_size.setValue(64)

        # num_epochs
        self.spin_num_epochs = QSpinBox(box_common)
        self.spin_num_epochs.setRange(1, 1000)
        self.spin_num_epochs.setValue(60)

        # learning_rate
        self.spin_learning_rate = QDoubleSpinBox(box_common)
        self.spin_learning_rate.setDecimals(6)
        self.spin_learning_rate.setRange(1e-6, 1.0)
        self.spin_learning_rate.setSingleStep(1e-4)
        self.spin_learning_rate.setValue(1e-3)

        # weight_decay
        self.spin_weight_decay = QDoubleSpinBox(box_common)
        self.spin_weight_decay.setDecimals(6)
        self.spin_weight_decay.setRange(0.0, 1.0)
        self.spin_weight_decay.setSingleStep(1e-5)
        self.spin_weight_decay.setValue(1e-4)

        # モデル有効/無効
        self.chk_use_xsr = QCheckBox("XSR 有効", box_common)
        self.chk_use_xsr.setChecked(True)
        self.chk_use_mamba = QCheckBox("Wavelet-Mamba 有効", box_common)
        self.chk_use_mamba.setChecked(True)
        self.chk_use_fed = QCheckBox("FEDformer 有効", box_common)
        self.chk_use_fed.setChecked(True)

        form_common.addRow(QLabel("Preset:", box_common), self.combo_preset)
        form_common.addRow(QLabel("Device:", box_common), self.combo_device)
        form_common.addRow(QLabel("seq_len:", box_common), self.spin_seq_len)
        form_common.addRow(QLabel("horizon:", box_common), self.spin_horizon)
        form_common.addRow(QLabel("batch_size:", box_common), self.spin_batch_size)
        form_common.addRow(QLabel("num_epochs:", box_common), self.spin_num_epochs)
        form_common.addRow(QLabel("learning_rate:", box_common), self.spin_learning_rate)
        form_common.addRow(QLabel("weight_decay:", box_common), self.spin_weight_decay)
        form_common.addRow(self.chk_use_xsr)
        form_common.addRow(self.chk_use_mamba)
        form_common.addRow(self.chk_use_fed)

        parent_layout.addWidget(box_common)

        # ---------------- Wavelet-Mamba 専用 ----------------
        box_mamba = QGroupBox("Wavelet-Mamba 専用ハイパラ", self)
        form_mamba = QFormLayout(box_mamba)

        self.spin_mamba_hidden_dim = QSpinBox(box_mamba)
        self.spin_mamba_hidden_dim.setRange(8, 2048)
        self.spin_mamba_hidden_dim.setValue(96)

        self.spin_mamba_num_layers = QSpinBox(box_mamba)
        self.spin_mamba_num_layers.setRange(1, 32)
        self.spin_mamba_num_layers.setValue(4)

        self.spin_mamba_dropout = QDoubleSpinBox(box_mamba)
        self.spin_mamba_dropout.setDecimals(3)
        self.spin_mamba_dropout.setRange(0.0, 0.9)
        self.spin_mamba_dropout.setSingleStep(0.01)
        self.spin_mamba_dropout.setValue(0.1)

        self.spin_mamba_wavelet_levels = QSpinBox(box_mamba)
        self.spin_mamba_wavelet_levels.setRange(1, 10)
        self.spin_mamba_wavelet_levels.setValue(3)

        form_mamba.addRow(QLabel("hidden_dim:", box_mamba), self.spin_mamba_hidden_dim)
        form_mamba.addRow(QLabel("num_layers:", box_mamba), self.spin_mamba_num_layers)
        form_mamba.addRow(QLabel("dropout:", box_mamba), self.spin_mamba_dropout)
        form_mamba.addRow(QLabel("wavelet_levels:", box_mamba), self.spin_mamba_wavelet_levels)

        parent_layout.addWidget(box_mamba)

        # ---------------- FEDformer 専用 ----------------
        box_fed = QGroupBox("FEDformer 専用ハイパラ", self)
        form_fed = QFormLayout(box_fed)

        self.spin_fed_d_model = QSpinBox(box_fed)
        self.spin_fed_d_model.setRange(8, 4096)
        self.spin_fed_d_model.setValue(128)

        self.spin_fed_n_heads = QSpinBox(box_fed)
        self.spin_fed_n_heads.setRange(1, 32)
        self.spin_fed_n_heads.setValue(4)

        self.spin_fed_e_layers = QSpinBox(box_fed)
        self.spin_fed_e_layers.setRange(1, 16)
        self.spin_fed_e_layers.setValue(3)

        self.spin_fed_dropout = QDoubleSpinBox(box_fed)
        self.spin_fed_dropout.setDecimals(3)
        self.spin_fed_dropout.setRange(0.0, 0.9)
        self.spin_fed_dropout.setSingleStep(0.01)
        self.spin_fed_dropout.setValue(0.1)

        self.spin_fed_modes = QSpinBox(box_fed)
        self.spin_fed_modes.setRange(1, 512)
        self.spin_fed_modes.setValue(32)

        self.combo_fed_mode_select = QComboBox(box_fed)
        self.combo_fed_mode_select.addItems(["topk", "low", "high"])
        self.combo_fed_mode_select.setCurrentText("topk")

        form_fed.addRow(QLabel("d_model:", box_fed), self.spin_fed_d_model)
        form_fed.addRow(QLabel("n_heads:", box_fed), self.spin_fed_n_heads)
        form_fed.addRow(QLabel("encoder_layers:", box_fed), self.spin_fed_e_layers)
        form_fed.addRow(QLabel("dropout:", box_fed), self.spin_fed_dropout)
        form_fed.addRow(QLabel("freq_top_k (modes):", box_fed), self.spin_fed_modes)
        form_fed.addRow(QLabel("mode_select:", box_fed), self.combo_fed_mode_select)

        parent_layout.addWidget(box_fed)

    # =====================================================
    # シグナル接続
    # =====================================================

    def _connect_signals(self) -> None:
        # 実行ボタン
        self.btn_run_stepB.clicked.connect(self._on_click_run_stepB)

        # BackendController からの通知
        self.backend.stepB_started.connect(self._on_stepB_started)
        self.backend.stepB_finished.connect(self._on_stepB_finished)
        self.backend.stepB_failed.connect(self._on_stepB_failed)

        # preset 変更に応じてざっくり値をセット
        self.combo_preset.currentTextChanged.connect(self._on_preset_changed)

    # =====================================================
    # preset の簡易切替
    # =====================================================

    def _on_preset_changed(self, text: str) -> None:
        """
        preset = cpu_light / gpu_standard / gpu_heavy に応じて
        代表的な値をざっくり設定する。
        """
        if text == "cpu_light":
            self.spin_seq_len.setValue(128)
            self.spin_batch_size.setValue(32)
            self.spin_num_epochs.setValue(40)
            self.spin_learning_rate.setValue(1e-3)
            self.spin_weight_decay.setValue(1e-4)

            self.spin_mamba_hidden_dim.setValue(64)
            self.spin_mamba_num_layers.setValue(3)

            self.spin_fed_d_model.setValue(64)
            self.spin_fed_n_heads.setValue(2)
            self.spin_fed_e_layers.setValue(2)
            self.spin_fed_modes.setValue(16)

        elif text == "gpu_standard":
            self.spin_seq_len.setValue(256)
            self.spin_batch_size.setValue(64)
            self.spin_num_epochs.setValue(60)
            self.spin_learning_rate.setValue(1e-3)
            self.spin_weight_decay.setValue(1e-4)

            self.spin_mamba_hidden_dim.setValue(96)
            self.spin_mamba_num_layers.setValue(4)

            self.spin_fed_d_model.setValue(128)
            self.spin_fed_n_heads.setValue(4)
            self.spin_fed_e_layers.setValue(3)
            self.spin_fed_modes.setValue(32)

        elif text == "gpu_heavy":
            self.spin_seq_len.setValue(512)
            self.spin_batch_size.setValue(64)
            self.spin_num_epochs.setValue(80)
            self.spin_learning_rate.setValue(5e-4)
            self.spin_weight_decay.setValue(1e-4)

            self.spin_mamba_hidden_dim.setValue(128)
            self.spin_mamba_num_layers.setValue(6)
            self.spin_mamba_dropout.setValue(0.12)

            self.spin_fed_d_model.setValue(192)
            self.spin_fed_n_heads.setValue(8)
            self.spin_fed_e_layers.setValue(4)
            self.spin_fed_modes.setValue(48)

        else:
            # custom の場合は何もしない
            pass

    # =====================================================
    # StepB 起動ボタン
    # =====================================================

    def _on_click_run_stepB(self) -> None:
        try:
            config = self._collect_stepB_config()
        except Exception as e:
            QMessageBox.critical(self, "StepB Config Error", str(e))
            return

        symbol = self.backend.app_context.current_symbol
        date_range: DateRange = self.backend.app_context.date_range

        # ボタン連打防止
        self.btn_run_stepB.setEnabled(False)

        # BackendController に依頼
        self.backend.start_stepB_training(
            symbol=symbol,
            date_range=date_range,
            config=config,
        )

    # =====================================================
    # BackendController からの通知
    # =====================================================

    def _on_stepB_started(self, symbol: str) -> None:
        # 必要ならステータスバーなどに反映
        # 例: self.status_label.setText(f"StepB started for {symbol}")
        pass

    def _on_stepB_finished(self, result: StepBResult) -> None:
        self.btn_run_stepB.setEnabled(True)
        self.update_result_table(result)

    def _on_stepB_failed(self, message: str) -> None:
        self.btn_run_stepB.setEnabled(True)
        QMessageBox.critical(self, "StepB Error", message)

    # =====================================================
    # StepBConfig 生成（ハイパラフォーム → Config）
    # =====================================================

    def _collect_stepB_config(self) -> StepBConfig:
        """
        現在のフォーム値から StepBConfig / WaveletMambaConfig / FEDformerConfig を組み立てる。
        """
        preset = self.combo_preset.currentText()
        device = self.combo_device.currentText()

        seq_len = self.spin_seq_len.value()
        horizon = self.spin_horizon.value()
        batch_size = self.spin_batch_size.value()
        num_epochs = self.spin_num_epochs.value()
        learning_rate = float(self.spin_learning_rate.value())
        weight_decay = float(self.spin_weight_decay.value())

        use_xsr = self.chk_use_xsr.isChecked()
        use_mamba = self.chk_use_mamba.isChecked()
        use_fed = self.chk_use_fed.isChecked()

        if not (use_xsr or use_mamba or use_fed):
            raise ValueError("少なくとも1つのモデル（XSR / Wavelet-Mamba / FEDformer）を有効にしてください。")

        symbol = self.backend.app_context.current_symbol

        # Wavelet-Mamba 用 Config
        mamba_cfg = WaveletMambaConfig(
            symbol=symbol,
            seq_len=seq_len,
            horizon=horizon,
            hidden_dim=self.spin_mamba_hidden_dim.value(),
            num_layers=self.spin_mamba_num_layers.value(),
            dropout=float(self.spin_mamba_dropout.value()),
            wavelet_levels=self.spin_mamba_wavelet_levels.value(),
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device,
            preset=preset if preset != "custom" else None,
        )

        # FEDformer 用 Config
        fed_cfg = FEDformerConfig(
            input_size=44,  # 周期 44 本前提。変えるなら AppConfig 等から取得する。
            seq_len=seq_len,
            d_model=self.spin_fed_d_model.value(),
            n_heads=self.spin_fed_n_heads.value(),
            e_layers=self.spin_fed_e_layers.value(),
            d_ff=self.spin_fed_d_model.value() * 2,
            dropout=float(self.spin_fed_dropout.value()),
            freq_top_k=self.spin_fed_modes.value(),
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate * 0.3,  # Transformer は少し低めでもよい例
            weight_decay=weight_decay,
            device=device,
        )

        # StepBConfig 本体
        cfg = StepBConfig(
            seq_len=seq_len,
            horizon=horizon,
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device,
            preset=preset if preset != "custom" else None,
            use_xsr=use_xsr,
            use_mamba=use_mamba,
            use_fedformer=use_fed,
            mamba_config=mamba_cfg,
            fed_config=fed_cfg,
        )
        return cfg

    # =====================================================
    # 結果テーブル更新ロジック
    # =====================================================

    def update_result_table(self, stepB_result: StepBResult) -> None:
        """
        StepBResult.metrics_by_agent を表形式にして表示する。

        - XSR / Wavelet-Mamba / FEDformer を 1 行ずつ
        - None や NaN は "-" で表示
        """
        metrics_dict = getattr(stepB_result, "metrics_by_agent", {}) or {}
        agents = list(metrics_dict.keys())

        self.table_result.setRowCount(len(agents))

        for row, agent_key in enumerate(agents):
            metrics: StepBAgentMetrics = metrics_dict[agent_key]

            def set_item(col: int, text: str) -> None:
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                self.table_result.setItem(row, col, item)

            # 0: Agent
            agent_name = getattr(metrics, "agent", None) or agent_key
            set_item(0, agent_name)

            # 1: Model Type
            set_item(1, getattr(metrics, "model_type", "-"))

            # 2: Train Loss (last)
            set_item(2, self._format_float(getattr(metrics, "train_loss_last", None)))

            # 3: Val Loss (last)
            set_item(3, self._format_float(getattr(metrics, "val_loss_last", None)))

            # 4: Best Val Loss
            set_item(4, self._format_float(getattr(metrics, "best_val_loss", None)))

            # 5: Epochs
            epochs = getattr(metrics, "num_epochs_trained", None)
            set_item(5, str(epochs) if epochs is not None else "-")

            # 6: Train Samples
            n_train = getattr(metrics, "n_train_samples", None)
            set_item(6, str(n_train) if n_train is not None else "-")

            # 7: Valid Samples
            n_valid = getattr(metrics, "n_valid_samples", None)
            set_item(7, str(n_valid) if n_valid is not None else "-")

            # 8: Lag (days)
            lag = getattr(metrics, "lag_days", None)
            if lag in (None, 0):
                set_item(8, "-")
            else:
                set_item(8, str(lag))

            # 9: Status / Message
            status = getattr(metrics, "status", "ok")
            message = getattr(metrics, "message", "")
            text = status if not message else f"{status}: {message}"
            set_item(9, text)

        self.table_result.resizeColumnsToContents()

    # =====================================================
    # 小ヘルパー
    # =====================================================

    @staticmethod
    def _format_float(v, ndigits: int = 6) -> str:
        if v is None:
            return "-"
        try:
            return f"{float(v):.{ndigits}f}"
        except Exception:
            return "-"
