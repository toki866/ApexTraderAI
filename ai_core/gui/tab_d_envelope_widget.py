# gui/tab_d_envelope_widget.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QComboBox,
    QCheckBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ai_core.config.app_config import AppConfig
from ai_core.types.common import DateRange
from ai_core.services.step_d_service import StepDResult


@dataclass
class TabDContext:
    """
    TabD に渡す簡易コンテキスト。
    必要に応じて BackendController などを追加してOK。
    """
    app_config: AppConfig
    date_range: DateRange


class TabD_EnvelopeWidget(QWidget):
    """
    StepD（Envelope + Events）可視化タブ。

    - 左上: Close + Pred + Envelope + マーカーのグラフ
    - 左下: Eventテーブル（UP/DOWN＋幾何特徴）
    - 右側: コントロールパネル（銘柄・モデル選択・表示オプション）
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._context: Optional[TabDContext] = None
        self._symbol: Optional[str] = None
        self._date_range: Optional[DateRange] = None

        self._current_agent: str = "xsr"

        # データキャッシュ
        self._envelope_df: Optional[pd.DataFrame] = None
        self._events_df: Optional[pd.DataFrame] = None

        # UI 構築
        self._build_ui()

    # ============================
    # 外部から呼ばれるAPI
    # ============================
    def set_context(self, context: TabDContext) -> None:
        """
        AppConfig / DateRange などを受け取る。
        """
        self._context = context

    def set_symbol_and_range(self, symbol: str, date_range: DateRange) -> None:
        """
        ヘッダなどから銘柄・期間が変更されたときに呼ぶ。
        """
        self._symbol = symbol
        self._date_range = date_range
        self.label_symbol.setText(f"銘柄: {symbol}")
        self.label_range.setText(
            f"期間: {date_range.train_start}〜{date_range.test_end}"
        )

    def update_from_stepD_result(self, result: StepDResult) -> None:
        """
        Backend側で StepDService が完了したときに呼び出してもらう想定。

        現在選択中の agent について、Envelope / Events を読み込んで再描画。
        """
        if not self._symbol or not self._context:
            return

        agent = self._current_agent
        env_path = result.envelope_paths.get(agent)
        evt_path = result.event_paths.get(agent)

        if env_path is None or not Path(env_path).exists():
            # まだ実行されていない or 対象エージェントなし
            self._envelope_df = None
            self._events_df = None
            self._clear_plot()
            self._clear_table()
            return

        self._load_data_and_refresh(env_path, evt_path)

    # ============================
    # UI 構築
    # ============================
    def _build_ui(self) -> None:
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # 左側: グラフ + テーブル
        left_widget = QWidget(self)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(5)

        # --- グラフ（matplotlib） ---
        self.fig = Figure(figsize=(8, 4))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)

        left_layout.addWidget(self.canvas, stretch=3)

        # --- イベントテーブル ---
        self.table_events = QTableWidget(self)
        self.table_events.setColumnCount(0)
        self.table_events.setRowCount(0)
        self.table_events.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_events.verticalHeader().setVisible(False)
        self.table_events.setSelectionBehavior(QTableWidget.SelectRows)
        self.table_events.cellClicked.connect(self._on_event_row_clicked)

        left_layout.addWidget(self.table_events, stretch=2)

        main_layout.addWidget(left_widget, stretch=3)

        # 右側: コントロールパネル
        right_widget = QWidget(self)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(5)

        # --- 銘柄・期間表示 ---
        group_info = QGroupBox("対象情報", self)
        info_layout = QVBoxLayout(group_info)

        self.label_symbol = QLabel("銘柄: -", group_info)
        self.label_range = QLabel("期間: -", group_info)

        info_layout.addWidget(self.label_symbol)
        info_layout.addWidget(self.label_range)
        right_layout.addWidget(group_info)

        # --- モデル選択・ Envelope生成系 ---
        group_model = QGroupBox("モデル・処理", self)
        model_layout = QVBoxLayout(group_model)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("モデル:", group_model))
        self.combo_agent = QComboBox(group_model)
        self.combo_agent.addItems(["xsr", "lstm", "fed"])
        self.combo_agent.currentTextChanged.connect(self._on_agent_changed)
        row1.addWidget(self.combo_agent)
        model_layout.addLayout(row1)

        self.btn_refresh = QPushButton("再描画 / 再読み込み", group_model)
        self.btn_refresh.clicked.connect(self._on_refresh_clicked)
        model_layout.addWidget(self.btn_refresh)

        self.btn_open_folder = QPushButton("stepDフォルダを開く", group_model)
        self.btn_open_folder.clicked.connect(self._on_open_folder_clicked)
        model_layout.addWidget(self.btn_open_folder)

        right_layout.addWidget(group_model)

        # --- 表示オプション ---
        group_view = QGroupBox("表示オプション", self)
        view_layout = QVBoxLayout(group_view)

        self.chk_show_close = QCheckBox("実Closeを表示", group_view)
        self.chk_show_close.setChecked(True)
        self.chk_show_close.stateChanged.connect(self._update_plot_only)

        self.chk_show_pred = QCheckBox("予測値を表示", group_view)
        self.chk_show_pred.setChecked(True)
        self.chk_show_pred.stateChanged.connect(self._update_plot_only)

        self.chk_show_env = QCheckBox("Envelopeを表示", group_view)
        self.chk_show_env.setChecked(True)
        self.chk_show_env.stateChanged.connect(self._update_plot_only)

        self.chk_show_markers = QCheckBox("Top/Bottomマーカーを表示", group_view)
        self.chk_show_markers.setChecked(True)
        self.chk_show_markers.stateChanged.connect(self._update_plot_only)

        view_layout.addWidget(self.chk_show_close)
        view_layout.addWidget(self.chk_show_pred)
        view_layout.addWidget(self.chk_show_env)
        view_layout.addWidget(self.chk_show_markers)

        right_layout.addWidget(group_view)

        # スペーサ
        right_layout.addStretch(1)

        main_layout.addWidget(right_widget, stretch=1)

    # ============================
    # データ読み込み＋描画
    # ============================
    def _on_agent_changed(self, agent: str) -> None:
        self._current_agent = agent
        self._on_refresh_clicked()

    @Slot()
    def _on_refresh_clicked(self) -> None:
        if not self._context or not self._symbol:
            return

        # StepDService.run をここで直接叩かず、既に生成済みの CSV を読む想定。
        # （StepA〜D一括実行時に StepDResult から呼んでもらう方が自然）
        step_d_dir = self._context.app_config.paths.output_root / "stepD"
        env_path = step_d_dir / f"stepD_envelope_{self._current_agent}_{self._symbol}.csv"
        evt_path = step_d_dir / f"stepD_events_{self._current_agent}_{self._symbol}.csv"

        if env_path.exists():
            self._load_data_and_refresh(env_path, evt_path if evt_path.exists() else None)
        else:
            # ファイルがなければ表示をクリア
            self._envelope_df = None
            self._events_df = None
            self._clear_plot()
            self._clear_table()

    def _load_data_and_refresh(self, env_path: Path, evt_path: Optional[Path]) -> None:
        # Envelope / Events 読み込み
        self._envelope_df = pd.read_csv(env_path)
        if "Date" in self._envelope_df.columns:
            self._envelope_df["Date"] = pd.to_datetime(self._envelope_df["Date"])

        if evt_path is not None and evt_path.exists():
            self._events_df = pd.read_csv(evt_path, parse_dates=["start_date", "end_date"])
        else:
            self._events_df = None

        self._update_plot_only()
        self._update_events_table()

    def _clear_plot(self) -> None:
        self.ax.clear()
        self.ax.set_title("StepD Envelope")
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Price")
        self.canvas.draw_idle()

    def _clear_table(self) -> None:
        self.table_events.setRowCount(0)
        self.table_events.setColumnCount(0)

    def _update_plot_only(self) -> None:
        if self._envelope_df is None:
            self._clear_plot()
            return

        df = self._envelope_df
        agent_upper = self._current_agent.upper()
        pred_col = f"Pred_Close_{agent_upper}"
        top_col = f"Env_Top_{agent_upper}"
        bottom_col = f"Env_Bottom_{agent_upper}"

        self.ax.clear()

        x = df["Date"]

        # 実Close
        if self.chk_show_close.isChecked() and "Close" in df.columns:
            self.ax.plot(x, df["Close"], label="Close")

        # 予測値
        if self.chk_show_pred.isChecked() and pred_col in df.columns:
            self.ax.plot(x, df[pred_col], label=f"Pred_{agent_upper}")

        # Envelope
        if self.chk_show_env.isChecked():
            if top_col in df.columns:
                self.ax.plot(x, df[top_col], label=f"Env_Top_{agent_upper}")
            if bottom_col in df.columns:
                self.ax.plot(x, df[bottom_col], label=f"Env_Bottom_{agent_upper}")

        # Top/Bottomマーカー
        if self.chk_show_markers.isChecked():
            if "Top_flag" in df.columns and top_col in df.columns:
                mask_top = df["Top_flag"] == 1
                self.ax.scatter(
                    df.loc[mask_top, "Date"],
                    df.loc[mask_top, top_col],
                    marker="^",
                    s=40,
                    label="Top",
                )
            if "Bottom_flag" in df.columns and bottom_col in df.columns:
                mask_bottom = df["Bottom_flag"] == 1
                self.ax.scatter(
                    df.loc[mask_bottom, "Date"],
                    df.loc[mask_bottom, bottom_col],
                    marker="v",
                    s=40,
                    label="Bottom",
                )

        self.ax.set_title(f"Envelope & Events ({self._symbol}, {agent_upper})")
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Price")
        self.ax.grid(True)
        self.ax.legend(loc="best")
        self.fig.autofmt_xdate()

        self.canvas.draw_idle()

    # ============================
    # イベントテーブル
    # ============================
    def _update_events_table(self) -> None:
        if self._events_df is None or self._events_df.empty:
            self._clear_table()
            return

        df = self._events_df.copy()

        # 表示するカラムをある程度絞る（必要に応じて増やせる）
        display_cols = [
            "event_id",
            "direction",
            "start_date",
            "end_date",
            "duration_days",
            "delta_p_pct",
            "theta_norm",
            "L",
            "D_norm",
        ]
        display_cols = [c for c in display_cols if c in df.columns]

        self.table_events.setColumnCount(len(display_cols))
        self.table_events.setRowCount(len(df))
        self.table_events.setHorizontalHeaderLabels(display_cols)

        for row_idx, (_, row) in enumerate(df.iterrows()):
            for col_idx, col_name in enumerate(display_cols):
                val = row[col_name]
                if isinstance(val, pd.Timestamp):
                    text = val.strftime("%Y-%m-%d")
                else:
                    text = str(val)
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                self.table_events.setItem(row_idx, col_idx, item)

        self.table_events.resizeRowsToContents()
        self.table_events.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    @Slot(int, int)
    def _on_event_row_clicked(self, row: int, col: int) -> None:
        if self._events_df is None or self._events_df.empty:
            return
        if row < 0 or row >= len(self._events_df):
            return

        evt = self._events_df.iloc[row]
        start_date = evt["start_date"]
        end_date = evt["end_date"]

        # グラフ上で該当区間をハイライト
        self._highlight_event_interval(start_date, end_date)

    def _highlight_event_interval(self, start_date, end_date) -> None:
        if self._envelope_df is None:
            return

        # まず通常の描画をしてからハイライト
        self._update_plot_only()

        # 区間ハイライト
        self.ax.axvspan(start_date, end_date, alpha=0.15)

        # ズーム（必要なら）
        df = self._envelope_df
        mask = (df["Date"] >= start_date) & (df["Date"] <= end_date)
        if mask.any():
            x_min = df.loc[mask, "Date"].min()
            x_max = df.loc[mask, "Date"].max()
            self.ax.set_xlim(x_min, x_max)

        self.canvas.draw_idle()

    # ============================
    # その他
    # ============================
    @Slot()
    def _on_open_folder_clicked(self) -> None:
        """
        stepD 出力フォルダをOS側で開く（Windows想定）。
        """
        if not self._context:
            return
        step_d_dir = self._context.app_config.paths.output_root / "stepD"
        step_d_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Windows
            import os
            os.startfile(str(step_d_dir))
        except Exception:
            # 他OSの場合は特に何もしない（必要なら subprocess で open / xdg-open 等）
            pass
