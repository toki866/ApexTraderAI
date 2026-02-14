# gui/tab_f_marl_widget.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableView,
    QHeaderView,
    QMessageBox,
    QGroupBox,
)

from ai_core.config.app_config import AppConfig
from ai_core.types.common import DateRange
from ai_core.services.step_f_service import StepFConfig, StepFResult
from backend.backend_controller import BackendController


@dataclass
class AgentWeightWidgets:
    xsr_edit: QLineEdit
    lstm_edit: QLineEdit
    fed_edit: QLineEdit


class TabF_MARLWidget(QWidget):
    """
    StepF（MARL）の GUI タブ。

    - MARL 学習開始
    - （将来用）キャンセル
    - Equity カーブ（MARL vs Buy & Hold）
    - 価格＋Buy/Sellマーカー
    - 日次ログテーブル
    - アクションヒートマップ（XSR / LSTM / FED の日次行動）
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

        self.symbol: str = ""
        self.date_range: Optional[DateRange] = None

        self.current_equity_df: Optional[pd.DataFrame] = None
        self.current_daily_log_df: Optional[pd.DataFrame] = None
        self.current_actions_df: Optional[pd.DataFrame] = None

        self._build_ui()

    # ------------------------------------------------------
    # UI 構築
    # ------------------------------------------------------

    def _build_ui(self) -> None:
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(6)

        # 左側：チャート＋テーブル
        left_layout = QVBoxLayout()
        left_layout.setSpacing(6)

        # 価格＋Buy/Sell
        self.fig_price = Figure(figsize=(5, 3))
        self.ax_price = self.fig_price.add_subplot(111)
        self.canvas_price = FigureCanvas(self.fig_price)
        self.ax_price.set_title("Price + Buy/Sell Markers")
        self.ax_price.set_xlabel("Date")
        self.ax_price.set_ylabel("Price")

        # Equity
        self.fig_equity = Figure(figsize=(5, 3))
        self.ax_equity = self.fig_equity.add_subplot(111)
        self.canvas_equity = FigureCanvas(self.fig_equity)
        self.ax_equity.set_title("Equity Curve (MARL vs Buy & Hold)")
        self.ax_equity.set_xlabel("Date")
        self.ax_equity.set_ylabel("Equity")

        # アクションヒートマップ
        self.fig_actions = Figure(figsize=(5, 3))
        self.ax_actions = self.fig_actions.add_subplot(111)
        self.canvas_actions = FigureCanvas(self.fig_actions)
        self.ax_actions.set_title("Agent Actions Heatmap")
        self.ax_actions.set_xlabel("Agent")
        self.ax_actions.set_ylabel("Date")

        left_layout.addWidget(self.canvas_price, stretch=1)
        left_layout.addWidget(self.canvas_equity, stretch=1)
        left_layout.addWidget(self.canvas_actions, stretch=1)

        # 日次ログテーブル
        self.daily_log_view = QTableView()
        self.daily_log_view.horizontalHeader().setStretchLastSection(True)
        self.daily_log_view.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        left_layout.addWidget(QLabel("日次ログ"), stretch=0)
        left_layout.addWidget(self.daily_log_view, stretch=1)

        # 右側：フォーム
        right_layout = QVBoxLayout()
        right_layout.setSpacing(6)

        # 銘柄・期間
        info_group = QGroupBox("対象銘柄・期間")
        info_form = QFormLayout()
        self.lbl_symbol = QLabel("-")
        self.lbl_date_range = QLabel("-")
        info_form.addRow("Symbol:", self.lbl_symbol)
        info_form.addRow("期間:", self.lbl_date_range)
        info_group.setLayout(info_form)

        # エージェント重み
        weight_group = QGroupBox("初期エージェント重み")
        weight_form = QFormLayout()
        self.weight_xsr_edit = QLineEdit("1.0")
        self.weight_lstm_edit = QLineEdit("1.0")
        self.weight_fed_edit = QLineEdit("1.0")
        for edit in (self.weight_xsr_edit, self.weight_lstm_edit, self.weight_fed_edit):
            edit.setMaximumWidth(80)
        weight_form.addRow("XSR:", self.weight_xsr_edit)
        weight_form.addRow("LSTM:", self.weight_lstm_edit)
        weight_form.addRow("FED:", self.weight_fed_edit)
        weight_group.setLayout(weight_form)

        self.agent_weight_widgets = AgentWeightWidgets(
            xsr_edit=self.weight_xsr_edit,
            lstm_edit=self.weight_lstm_edit,
            fed_edit=self.weight_fed_edit,
        )

        # ボタン
        btn_group = QGroupBox("操作")
        btn_layout = QVBoxLayout()
        self.btn_run_marl = QPushButton("MARL 学習開始 (StepF)")
        self.btn_cancel = QPushButton("キャンセル")
        self.btn_cancel.setEnabled(False)

        self.btn_run_marl.clicked.connect(self._on_click_run_marl)
        self.btn_cancel.clicked.connect(self._on_click_cancel)

        btn_layout.addWidget(self.btn_run_marl)
        btn_layout.addWidget(self.btn_cancel)
        btn_group.setLayout(btn_layout)

        right_layout.addWidget(info_group)
        right_layout.addWidget(weight_group)
        right_layout.addWidget(btn_group)
        right_layout.addStretch(1)

        main_layout.addLayout(left_layout, stretch=3)
        main_layout.addLayout(right_layout, stretch=1)

    # ------------------------------------------------------
    # MainWindow から
    # ------------------------------------------------------

    def set_symbol_and_range(self, symbol: str, date_range: DateRange) -> None:
        """
        MainWindow / AppContext から現在の銘柄・期間を受け取る。
        """
        self.symbol = symbol
        self.date_range = date_range
        self.lbl_symbol.setText(symbol)
        self.lbl_date_range.setText(f"{date_range.train_start}〜{date_range.test_end}")

        # Backend 側にも伝搬
        self.backend.set_symbol_and_range(symbol, date_range)

    # ------------------------------------------------------
    # コンフィグ生成
    # ------------------------------------------------------

    def _parse_float(self, edit: QLineEdit, default: float) -> float:
        text = edit.text().strip()
        if not text:
            return default
        try:
            return float(text)
        except ValueError:
            return default

    def _collect_agent_weights(self) -> Dict[str, float]:
        w_xsr = self._parse_float(self.agent_weight_widgets.xsr_edit, 1.0)
        w_lstm = self._parse_float(self.agent_weight_widgets.lstm_edit, 1.0)
        w_fed = self._parse_float(self.agent_weight_widgets.fed_edit, 1.0)
        return {"xsr": w_xsr, "lstm": w_lstm, "fed": w_fed}

    def build_config_from_form(self) -> StepFConfig:
        """
        現在のフォーム内容から StepFConfig を生成する。

        - symbol / date_range は TabF が持っている現在値を使う
        - env_config / marl_config / use_gpu は app_config.rl から取得
        - agent_weights はフォームの値
        """
        if not self.symbol or self.date_range is None:
            raise RuntimeError("TabF: symbol / date_range が設定されていません。")

        # AppConfig.rl から EnvConfig / RLMARLConfig を取得
        env_cfg = self.app_config.rl.env_config
        marl_cfg = self.app_config.rl.marl_config

        cfg = StepFConfig(
            symbol=self.symbol,
            date_range=self.date_range,
            env_config=env_cfg,
            marl_config=marl_cfg,
            use_gpu=self.app_config.rl.use_gpu,
            agent_names=["xsr", "lstm", "fed"],
            agent_weights=self._collect_agent_weights(),
        )
        return cfg

    # ------------------------------------------------------
    # ボタン
    # ------------------------------------------------------

    def _on_click_run_marl(self) -> None:
        try:
            cfg = self.build_config_from_form()
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "StepF エラー", f"設定生成に失敗しました:\n{e}")
            return

        self.btn_run_marl.setEnabled(False)
        self.btn_cancel.setEnabled(True)

        # BackendController.start_stepF_training にコールバックを渡す
        self.backend.start_stepF_training(
            cfg,
            on_finished=self.update_stepF_result,
            on_error=self._on_stepF_error,
        )

    def _on_click_cancel(self) -> None:
        """
        将来用キャンセルボタン。
        現時点で BackendController に request_cancel が無ければ何もしない。
        """
        if hasattr(self.backend, "request_cancel"):
            self.backend.request_cancel("stepF")  # type: ignore[call-arg]
            self.btn_cancel.setEnabled(False)
            QMessageBox.information(self, "StepF キャンセル", "StepF にキャンセル要求を送りました。")
        else:
            QMessageBox.information(self, "StepF", "キャンセル機能はまだ未実装です。")

    # ------------------------------------------------------
    # Backend からのコールバック
    # ------------------------------------------------------

    def _on_stepF_error(self, msg: str) -> None:
        self.btn_run_marl.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        QMessageBox.critical(self, "StepF エラー", msg)

    # ------------------------------------------------------
    # 結果反映
    # ------------------------------------------------------

    def update_stepF_result(self, result: StepFResult) -> None:
        """
        BackendController.start_stepF_training の on_finished から呼ばれる。
        StepFResult を元にグラフ＋テーブルを更新。
        """
        self.btn_run_marl.setEnabled(True)
        self.btn_cancel.setEnabled(False)

        # キャンセル or エラーの場合
        if not result.success:
            msg = result.message or "StepF が失敗しました。"
            lower = msg.lower()
            if "cancel" in lower or "キャンセル" in lower:
                QMessageBox.information(self, "StepF キャンセル終了", msg)
            else:
                QMessageBox.critical(self, "StepF エラー", msg)
            return

        # --- CSV 読み込み ---
        equity_df: Optional[pd.DataFrame] = None
        daily_df: Optional[pd.DataFrame] = None
        actions_df: Optional[pd.DataFrame] = None

        try:
            if result.equity_curve_path is not None:
                equity_df = pd.read_csv(result.equity_curve_path)
                if "Date" in equity_df.columns:
                    equity_df["Date"] = pd.to_datetime(equity_df["Date"])
        except Exception as e:  # noqa: BLE001
            print(f"[TabF] Equity CSV 読み込み失敗: {e}")

        try:
            if result.daily_log_path is not None:
                daily_df = pd.read_csv(result.daily_log_path)
                if "Date" in daily_df.columns:
                    daily_df["Date"] = pd.to_datetime(daily_df["Date"])
        except Exception as e:  # noqa: BLE001
            print(f"[TabF] 日次ログ CSV 読み込み失敗: {e}")

        # StepF アクションCSV（固定パス output_root/stepF/stepF_actions_{symbol}.csv）
        try:
            output_root = Path(self.app_config.paths.output_root)
            actions_path = output_root / "stepF" / f"stepF_actions_{self.symbol}.csv"
            if actions_path.exists():
                actions_df = pd.read_csv(actions_path)
                if "Date" in actions_df.columns:
                    actions_df["Date"] = pd.to_datetime(actions_df["Date"])
            else:
                print(f"[TabF] action CSV が見つかりません: {actions_path}")
        except Exception as e:  # noqa: BLE001
            print(f"[TabF] action CSV 読み込み失敗: {e}")

        self.current_equity_df = equity_df
        self.current_daily_log_df = daily_df
        self.current_actions_df = actions_df

        self._plot_equity_curve()
        self._plot_price_with_markers()
        self._plot_action_heatmap()

        if daily_df is not None:
            self._set_table_from_dataframe(daily_df)

    # ------------------------------------------------------
    # グラフ（Equity）
    # ------------------------------------------------------

    def _plot_equity_curve(self) -> None:
        self.ax_equity.clear()
        self.ax_equity.set_title("Equity Curve (MARL vs Buy & Hold)")
        self.ax_equity.set_xlabel("Date")
        self.ax_equity.set_ylabel("Equity")

        df = self.current_equity_df
        if df is None or df.empty:
            self.ax_equity.text(
                0.5,
                0.5,
                "Equity データがありません",
                transform=self.ax_equity.transAxes,
                ha="center",
                va="center",
            )
            self.canvas_equity.draw()
            return

        if "Date" not in df.columns:
            self.ax_equity.text(
                0.5,
                0.5,
                "Equity CSV に Date 列がありません",
                transform=self.ax_equity.transAxes,
                ha="center",
                va="center",
            )
            self.canvas_equity.draw()
            return

        x = df["Date"]

        # MARL 本体
        for col in ["Equity", "equity"]:
            if col in df.columns:
                self.ax_equity.plot(x, df[col], label="MARL", linewidth=1.2)
                break

        # Buy & Hold 相当
        for col in ["BH_Equity", "bh_equity", "buy_and_hold"]:
            if col in df.columns:
                self.ax_equity.plot(
                    x, df[col], linestyle="--", label="Buy & Hold", linewidth=1.0
                )
                break

        self.ax_equity.legend()
        self.ax_equity.grid(True)
        self.fig_equity.autofmt_xdate()
        self.canvas_equity.draw()

    # ------------------------------------------------------
    # グラフ（価格＋Buy/Sell）
    # ------------------------------------------------------

    def _plot_price_with_markers(self) -> None:
        self.ax_price.clear()
        self.ax_price.set_title("Price + Buy/Sell Markers")
        self.ax_price.set_xlabel("Date")
        self.ax_price.set_ylabel("Price")

        df = self.current_daily_log_df
        if df is None or df.empty:
            self.ax_price.text(
                0.5,
                0.5,
                "日次ログがありません",
                transform=self.ax_price.transAxes,
                ha="center",
                va="center",
            )
            self.canvas_price.draw()
            return

        if "Date" not in df.columns:
            self.ax_price.text(
                0.5,
                0.5,
                "日次ログに Date 列がありません",
                transform=self.ax_price.transAxes,
                ha="center",
                va="center",
            )
            self.canvas_price.draw()
            return

        price_col = None
        for col in ["Close", "close", "Price", "price"]:
            if col in df.columns:
                price_col = col
                break

        if price_col is None:
            self.ax_price.text(
                0.5,
                0.5,
                "日次ログに価格列(Close/Price)が無いため価格チャートを描画できません",
                transform=self.ax_price.transAxes,
                ha="center",
                va="center",
                wrap=True,
            )
            self.canvas_price.draw()
            return

        pos_col = None
        for col in ["Position", "position", "Pos", "pos", "Ratio", "ratio"]:
            if col in df.columns:
                pos_col = col
                break

        dates = df["Date"]
        prices = df[price_col].astype(float)

        self.ax_price.plot(dates, prices, linewidth=1.2, label="Price")

        if pos_col is not None:
            pos = df[pos_col].astype(float).fillna(0.0).to_numpy()
            prev_pos = np.roll(pos, 1)
            prev_pos[0] = 0.0

            buy_mask = (pos > 0.0) & (prev_pos <= 0.0)
            sell_mask = (pos <= 0.0) & (prev_pos > 0.0)

            buy_dates = dates[buy_mask]
            buy_prices = prices[buy_mask]
            sell_dates = dates[sell_mask]
            sell_prices = prices[sell_mask]

            if not buy_dates.empty:
                self.ax_price.scatter(
                    buy_dates,
                    buy_prices,
                    marker="^",
                    s=40,
                    label="Buy",
                    zorder=5,
                )
            if not sell_dates.empty:
                self.ax_price.scatter(
                    sell_dates,
                    sell_prices,
                    marker="v",
                    s=40,
                    label="Sell",
                    zorder=5,
                )

        self.ax_price.legend()
        self.ax_price.grid(True)
        self.fig_price.autofmt_xdate()
        self.canvas_price.draw()

    # ------------------------------------------------------
    # グラフ（アクションヒートマップ）
    # ------------------------------------------------------

    def _plot_action_heatmap(self) -> None:
        self.ax_actions.clear()
        self.ax_actions.set_title("Agent Actions Heatmap")
        self.ax_actions.set_xlabel("Agent")
        self.ax_actions.set_ylabel("Date")

        df = self.current_actions_df
        if df is None or df.empty:
            self.ax_actions.text(
                0.5,
                0.5,
                "アクションログがありません",
                transform=self.ax_actions.transAxes,
                ha="center",
                va="center",
            )
            self.canvas_actions.draw()
            return

        if "Date" not in df.columns:
            self.ax_actions.text(
                0.5,
                0.5,
                "アクションCSVに Date 列がありません",
                transform=self.ax_actions.transAxes,
                ha="center",
                va="center",
            )
            self.canvas_actions.draw()
            return

        # エージェント列を抽出（a_xsr, a_lstm, a_fed）
        agent_cols = [c for c in df.columns if c.startswith("a_")]
        if not agent_cols:
            self.ax_actions.text(
                0.5,
                0.5,
                "a_xxx 列がありません（a_xsr / a_lstm / a_fed を想定）",
                transform=self.ax_actions.transAxes,
                ha="center",
                va="center",
            )
            self.canvas_actions.draw()
            return

        # Date でソート
        df_sorted = df.sort_values("Date").reset_index(drop=True)
        dates = df_sorted["Date"]

        # [-1, +1] のアクション行列（shape: [日数, エージェント数]）
        mat = df_sorted[agent_cols].astype(float).to_numpy()

        # imshow で描画（横：エージェント、縦：日）
        im = self.ax_actions.imshow(
            mat,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
        )

        # x 軸の目盛りにエージェント名（a_を外す）
        x_ticks = np.arange(len(agent_cols))
        x_labels = [c.replace("a_", "") for c in agent_cols]
        self.ax_actions.set_xticks(x_ticks)
        self.ax_actions.set_xticklabels(x_labels, rotation=45, ha="right")

        # y 軸ラベルに Date を間引き表示
        num_days = len(dates)
        if num_days <= 20:
            y_ticks = np.arange(num_days)
        else:
            step = max(1, num_days // 10)
            y_ticks = np.arange(0, num_days, step)

        y_labels = [dates.iloc[i].strftime("%Y-%m-%d") for i in y_ticks]
        self.ax_actions.set_yticks(y_ticks)
        self.ax_actions.set_yticklabels(y_labels)

        self.fig_actions.colorbar(im, ax=self.ax_actions, label="Action (-1 ~ +1)")

        self.fig_actions.tight_layout()
        self.canvas_actions.draw()

    # ------------------------------------------------------
    # テーブル
    # ------------------------------------------------------

    def _set_table_from_dataframe(self, df: pd.DataFrame) -> None:
        model = QStandardItemModel()
        model.setColumnCount(len(df.columns))
        model.setRowCount(len(df))

        for j, col in enumerate(df.columns):
            model.setHeaderData(j, Qt.Orientation.Horizontal, col)

        for i in range(len(df)):
            for j, col in enumerate(df.columns):
                val = df.iloc[i, j]
                if isinstance(val, (float, np.floating)):
                    text = f"{val:.6g}"
                else:
                    text = str(val)
                item = QStandardItem(text)
                model.setItem(i, j, item)

        self.daily_log_view.setModel(model)
        self.daily_log_view.resizeColumnsToContents()
