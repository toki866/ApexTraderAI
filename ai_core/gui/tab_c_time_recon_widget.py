from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QMessageBox,
    QGroupBox,
)

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from ai_core.config.app_config import AppConfig
from ai_core.types.common import DateRange
from ai_core.services.step_c_service import StepCConfig, StepCResult
from backend.backend_controller import BackendController


class TabCTimeReconWidget(QWidget):
    """
    StepC（TimeRecon + ScaleCalib）の結果を表示するタブ。

    - 「StepC 実行」ボタンから BackendController.start_stepC_training を呼び出す
    - 完了後、stepC_pred_time_all_{symbol}.csv を読み込んで
      実株価（Close）＋予測（*_scaled）をグラフ表示
    - 下部に calibration 情報（a, b, lag_days など）をテーブル表示
    """

    def __init__(
        self,
        app_config: AppConfig,
        backend: BackendController,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self.app_config = app_config
        self.backend = backend

        self.symbol: str = "SOXL"
        self.date_range: Optional[DateRange] = None

        # UI 構築
        self._init_ui()

        # Backend のシグナルを接続
        self.backend.stepC_finished.connect(self.on_stepC_finished)

    # ==============================
    # UI 初期化
    # ==============================

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        # --------- 上部コントロールバー ---------
        control_box = QGroupBox("StepC 設定 & 実行")
        control_layout = QHBoxLayout(control_box)

        self.lbl_symbol = QLabel(f"Symbol: {self.symbol}")
        self.lbl_symbol.setMinimumWidth(120)

        self.lbl_date_range = QLabel("DateRange: (未設定)")
        self.lbl_date_range.setMinimumWidth(260)

        # キャリブレーション期間（日数）
        self.spin_calib_days = QSpinBox()
        self.spin_calib_days.setRange(10, 2000)
        self.spin_calib_days.setValue(252)
        self.spin_calib_days.setSingleStep(10)

        lbl_calib = QLabel("Calib window (days):")
        lbl_calib.setBuddy(self.spin_calib_days)

        self.btn_run = QPushButton("StepC 実行")
        self.btn_run.clicked.connect(self.on_run_clicked)

        control_layout.addWidget(self.lbl_symbol)
        control_layout.addWidget(self.lbl_date_range)
        control_layout.addStretch()
        control_layout.addWidget(lbl_calib)
        control_layout.addWidget(self.spin_calib_days)
        control_layout.addWidget(self.btn_run)

        # --------- 中央：グラフエリア ---------
        fig = Figure(figsize=(8, 5))
        self.ax = fig.add_subplot(111)
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Price")

        self.canvas = FigureCanvas(fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        graph_layout = QVBoxLayout()
        graph_layout.addWidget(self.toolbar)
        graph_layout.addWidget(self.canvas)

        graph_box = QGroupBox("実株価 + 予測（scaled）")
        graph_box.setLayout(graph_layout)

        # --------- 下部：キャリブテーブル ---------
        self.table_calib = QTableWidget()
        self.table_calib.setColumnCount(7)
        self.table_calib.setHorizontalHeaderLabels(
            ["Model", "Raw Col", "Scaled Col", "a", "b", "LagDays", "N_calib"]
        )
        self.table_calib.horizontalHeader().setStretchLastSection(True)

        calib_box = QGroupBox("Calibration 情報")
        calib_layout = QVBoxLayout(calib_box)
        calib_layout.addWidget(self.table_calib)

        # --------- 全体配置 ---------
        main_layout.addWidget(control_box)
        main_layout.addWidget(graph_box, stretch=3)
        main_layout.addWidget(calib_box, stretch=1)

    # ==============================
    # 外部からの設定
    # ==============================

    def set_symbol_and_range(self, symbol: str, date_range: DateRange) -> None:
        """
        MainWindow などから銘柄と DateRange をセットする。
        """
        self.symbol = symbol
        self.date_range = date_range
        self.lbl_symbol.setText(f"Symbol: {symbol}")
        self.lbl_date_range.setText(
            f"DateRange: {date_range.train_start} ～ {date_range.test_end}"
        )

    # ==============================
    # StepC 実行ボタン
    # ==============================

    @Slot()
    def on_run_clicked(self) -> None:
        if self.date_range is None:
            QMessageBox.warning(
                self,
                "StepC 実行エラー",
                "DateRange が未設定です。\n先にヘッダなどから期間をセットしてください。",
            )
            return

        calib_days = int(self.spin_calib_days.value())

        # StepCConfig を組み立て（models はとりあえず 3AI 固定）
        config = StepCConfig(
            app_config=self.app_config,
            symbol=self.symbol,
            date_range=self.date_range,
            calib_window_days=calib_days,
            models=["xsr", "lstm", "fed"],
            lag_days={},        # 必要なら将来 Tab から渡せるように拡張
            raw_column_map=None,  # デフォルトの列名マッピングを使用
        )

        # ボタン連打防止
        self.btn_run.setEnabled(False)
        self.btn_run.setText("実行中...")

        self.backend.start_stepC_training(config)

    # ==============================
    # StepC 完了シグナル
    # ==============================

    @Slot(StepCResult)
    def on_stepC_finished(self, result: StepCResult) -> None:
        # ボタン復帰
        self.btn_run.setEnabled(True)
        self.btn_run.setText("StepC 実行")

        if not result.success:
            msg = result.message or "StepC に失敗しました。"
            QMessageBox.critical(self, "StepC エラー", msg)
            return

        # CSV パスを取得
        pred_path: Optional[Path] = result.pred_all_path
        if pred_path is None or not Path(pred_path).exists():
            QMessageBox.warning(
                self,
                "StepC 結果なし",
                f"出力 CSV が見つかりません:\n{pred_path}",
            )
            return

        # CSV 読み込み & グラフ更新
        try:
            df = pd.read_csv(pred_path)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            self._update_graph(df)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(
                self,
                "CSV 読込エラー",
                f"StepC 出力の読み込みに失敗しました:\n{e}",
            )
            return

        # キャリブ情報テーブル更新
        calib_info: Dict[str, Any] = {}
        if result.details and "calibrations" in result.details:
            calib_info = result.details.get("calibrations", {})  # type: ignore[assignment]

        self._update_calib_table(calib_info)

    # ==============================
    # グラフ更新：Close + 予測（scaled）
    # ==============================

    def _update_graph(self, df: pd.DataFrame) -> None:
        self.ax.clear()

        if "date" not in df.columns:
            QMessageBox.warning(self, "グラフ更新エラー", "CSV に date 列がありません。")
            return

        dates = df["date"]

        # 実株価
        if "close" in df.columns:
            self.ax.plot(dates, df["close"], label="Close（実株価）", linewidth=1.5)

        # スケール後の予測（*_scaled）を重ねる
        for col in df.columns:
            if not col.startswith("pred_"):
                continue
            if not col.endswith("_scaled"):
                # 生の予測はここでは描かず、scaled 系だけ表示
                continue
            self.ax.plot(dates, df[col], label=col, alpha=0.8)

        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Price")
        self.ax.legend()
        self.ax.grid(True)

        self.canvas.draw()

    # ==============================
    # キャリブ情報テーブル更新
    # ==============================

    def _update_calib_table(self, calib_info: Dict[str, Any]) -> None:
        """
        calib_info は StepCResult.details["calibrations"] を想定。
        例:
        {
          "xsr": {
             "raw_column": "pred_close_xsr",
             "scaled_column": "pred_close_xsr_scaled",
             "a": 1.23,
             "b": 0.45,
             "lag_days": 0,
             "num_calib_points": 252,
          },
          ...
        }
        """
        self.table_calib.setRowCount(0)

        if not calib_info:
            return

        for row_idx, (model_key, info) in enumerate(calib_info.items()):
            self.table_calib.insertRow(row_idx)

            raw_col = str(info.get("raw_column", ""))
            scaled_col = str(info.get("scaled_column", ""))
            a = f'{info.get("a", "")}'
            b = f'{info.get("b", "")}'
            lag = f'{info.get("lag_days", "")}'
            npts = f'{info.get("num_calib_points", "")}'

            self.table_calib.setItem(row_idx, 0, QTableWidgetItem(model_key))
            self.table_calib.setItem(row_idx, 1, QTableWidgetItem(raw_col))
            self.table_calib.setItem(row_idx, 2, QTableWidgetItem(scaled_col))
            self.table_calib.setItem(row_idx, 3, QTableWidgetItem(a))
            self.table_calib.setItem(row_idx, 4, QTableWidgetItem(b))
            self.table_calib.setItem(row_idx, 5, QTableWidgetItem(lag))
            self.table_calib.setItem(row_idx, 6, QTableWidgetItem(npts))

        self.table_calib.resizeColumnsToContents()
