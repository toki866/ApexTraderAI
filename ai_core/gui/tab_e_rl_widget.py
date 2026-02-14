from __future__ import annotations

from dataclasses import replace, fields
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QFormLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QGroupBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QPlainTextEdit,
    QProgressBar,
    QMessageBox,
)

from ai_core.config.app_config import AppConfig
from ai_core.config.rl_config import RLSingleConfig
from ai_core.types.common import DateRange
from ai_core.services.step_e_service import StepEConfig, StepEResult
from ai_core.rl.types import RLRunResult
from backend.backend_controller import BackendController


class TabE_RLWidget(QWidget):
    """
    StepE（単体RL / SB3-PPO）の GUI タブ。

    - 銘柄・期間の表示
    - エージェント選択（XSR / Mamba(LSTM) / FED）
    - RLハイパーパラメータ入力フォーム
    - 「学習開始」ボタン → BackendController.start_stepE_training を呼び出し
    - 学習結果（資金曲線・日次ログ・指標）の表示
    """

    def __init__(
        self,
        app_config: AppConfig,
        backend: BackendController,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._app_config = app_config
        self._backend = backend

        # MainWindow 側から渡されるコンテキスト
        self.symbol: Optional[str] = None
        self.date_range: Optional[DateRange] = None

        # UI 構築
        self._setup_ui()

        # Backend シグナル接続
        if hasattr(self._backend, "stepE_finished"):
            # type: ignore[arg-type]
            self._backend.stepE_finished.connect(self.update_stepE_result)
        if hasattr(self._backend, "log_message"):
            # type: ignore[arg-type]
            self._backend.log_message.connect(self._on_log_message)

    # ======================================================
    # UI 構築
    # ======================================================

    def _setup_ui(self) -> None:
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        # 左：グラフ＋日次ログ＋メトリクス
        left_layout = QVBoxLayout()
        left_layout.setSpacing(6)

        # 対象銘柄・期間
        info_group = QGroupBox("対象銘柄・期間")
        info_grid = QGridLayout()
        info_group.setLayout(info_grid)

        info_grid.addWidget(QLabel("銘柄:"), 0, 0)
        self.lbl_symbol = QLabel("―")
        info_grid.addWidget(self.lbl_symbol, 0, 1)

        info_grid.addWidget(QLabel("期間:"), 1, 0)
        self.lbl_date_range = QLabel("―")
        info_grid.addWidget(self.lbl_date_range, 1, 1)

        left_layout.addWidget(info_group, stretch=0)

        # 資金曲線グラフ
        fig_group = QGroupBox("資金曲線")
        fig_layout = QVBoxLayout()
        fig_group.setLayout(fig_layout)

        self.fig = Figure(figsize=(5, 3))
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        fig_layout.addWidget(self.toolbar)
        fig_layout.addWidget(self.canvas, stretch=1)

        left_layout.addWidget(fig_group, stretch=2)

        # 日次ログテーブル
        log_group = QGroupBox("日次ログ")
        log_layout = QVBoxLayout()
        log_group.setLayout(log_layout)

        self.table_daily_log = QTableWidget()
        self.table_daily_log.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table_daily_log.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table_daily_log.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        header = self.table_daily_log.horizontalHeader()
        header.setStretchLastSection(True)
        log_layout.addWidget(self.table_daily_log)

        left_layout.addWidget(log_group, stretch=2)

        # メトリクス・ログ
        metrics_group = QGroupBox("指標・ログ")
        metrics_layout = QVBoxLayout()
        metrics_group.setLayout(metrics_layout)

        self.txt_metrics = QPlainTextEdit()
        self.txt_metrics.setReadOnly(True)
        metrics_layout.addWidget(self.txt_metrics)

        left_layout.addWidget(metrics_group, stretch=1)

        main_layout.addLayout(left_layout, stretch=3)

        # 右：設定フォーム＋ボタン
        right_layout = QVBoxLayout()
        right_layout.setSpacing(6)

        # RL エンジン選択
        engine_group = QGroupBox("RLエンジン")
        engine_form = QFormLayout()
        engine_group.setLayout(engine_form)

        self.combo_engine = QComboBox()
        self.combo_engine.addItem("内部 RL (RLSingleAlgo)", userData="internal")
        self.combo_engine.addItem("SB3 PPO (SOXL/SOXS)", userData="sb3")
        engine_form.addRow("エンジン:", self.combo_engine)

        # エージェント選択
        self.combo_agent = QComboBox()
        self.combo_agent.addItem("XSR", userData="xsr")
        self.combo_agent.addItem("Mamba(LSTM)", userData="lstm")
        self.combo_agent.addItem("FEDformer", userData="fed")
        self.combo_agent.addItem("3エージェントすべて", userData="all")
        engine_form.addRow("エージェント:", self.combo_agent)

        # デバイス
        self.combo_device = QComboBox()
        self.combo_device.addItem("CPU", userData="cpu")
        self.combo_device.addItem("GPU", userData="gpu")
        engine_form.addRow("デバイス:", self.combo_device)

        # SB3 用詳細
        self.combo_env_agent = QComboBox()
        self.combo_env_agent.addItem("XSR", userData="xsr")
        self.combo_env_agent.addItem("Mamba(LSTM)", userData="lstm")
        self.combo_env_agent.addItem("FEDformer", userData="fed")
        engine_form.addRow("Envelope元AI (SB3):", self.combo_env_agent)

        self.spin_ppo_timesteps = QSpinBox()
        self.spin_ppo_timesteps.setRange(10_000, 10_000_000)
        self.spin_ppo_timesteps.setSingleStep(10_000)
        self.spin_ppo_timesteps.setValue(200_000)
        engine_form.addRow("PPO total_timesteps:", self.spin_ppo_timesteps)

        right_layout.addWidget(engine_group)

        # 内部RL ハイパーパラメータ
        hyper_group = QGroupBox("内部RLハイパーパラメータ")
        hyper_layout = QGridLayout()
        hyper_group.setLayout(hyper_layout)

        default_rl: RLSingleConfig = self._app_config.rl.single_config

        # total_timesteps
        hyper_layout.addWidget(QLabel("Total timesteps:"), 0, 0)
        self.spin_total_timesteps = QSpinBox()
        self.spin_total_timesteps.setRange(1_000, 10_000_000)
        self.spin_total_timesteps.setSingleStep(10_000)
        self.spin_total_timesteps.setValue(
            int(getattr(default_rl, "total_timesteps", 200_000))
        )
        hyper_layout.addWidget(self.spin_total_timesteps, 0, 1)

        # learning_rate
        hyper_layout.addWidget(QLabel("Learning rate:"), 1, 0)
        self.spin_learning_rate = QDoubleSpinBox()
        self.spin_learning_rate.setDecimals(6)
        self.spin_learning_rate.setRange(1e-6, 1e-1)
        self.spin_learning_rate.setSingleStep(1e-5)
        self.spin_learning_rate.setValue(
            float(getattr(default_rl, "learning_rate", 1e-4))
        )
        hyper_layout.addWidget(self.spin_learning_rate, 1, 1)

        # gamma
        hyper_layout.addWidget(QLabel("Gamma:"), 2, 0)
        self.spin_gamma = QDoubleSpinBox()
        self.spin_gamma.setDecimals(3)
        self.spin_gamma.setRange(0.80, 0.999)
        self.spin_gamma.setSingleStep(0.001)
        self.spin_gamma.setValue(float(getattr(default_rl, "gamma", 0.99)))
        hyper_layout.addWidget(self.spin_gamma, 2, 1)

        right_layout.addWidget(hyper_group)

        # 実行ボタン
        btn_layout = QHBoxLayout()
        self.btn_run = QPushButton("学習開始")
        self.btn_cancel = QPushButton("キャンセル")
        btn_layout.addWidget(self.btn_run)
        btn_layout.addWidget(self.btn_cancel)
        right_layout.addLayout(btn_layout)

        # 進捗バー
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # 不定進捗
        self.progress_bar.setVisible(False)
        right_layout.addWidget(self.progress_bar)

        right_layout.addStretch(1)

        main_layout.addLayout(right_layout, stretch=2)

        # シグナル接続
        self.btn_run.clicked.connect(self._on_click_run)
        self.btn_cancel.clicked.connect(self._on_click_cancel)

    # ======================================================
    # MainWindow からの銘柄・期間設定
    # ======================================================

    def set_symbol_and_range(self, symbol: str, date_range: DateRange) -> None:
        self.symbol = symbol
        self.date_range = date_range
        self.lbl_symbol.setText(symbol)
        self.lbl_date_range.setText(
            f"{date_range.train_start}〜{date_range.test_end}"
        )

    # ======================================================
    # Config 構築
    # ======================================================

    def build_config_from_form(self) -> StepEConfig:
        """
        UI の内容から StepEConfig を組み立てる。
        AppConfig.rl.single_config もここで更新する。
        """
        if self.symbol is None or self.date_range is None:
            raise RuntimeError("TabE: symbol / date_range が設定されていません。")

        # エージェントリスト
        agent_key = self.combo_agent.currentData()
        if agent_key == "all":
            agents = ["xsr", "lstm", "fed"]
        else:
            agents = [str(agent_key)]

        # デバイス
        use_gpu = self.combo_device.currentData() == "gpu"

        # 内部 RL のハイパラを AppConfig に反映
        single_cfg: RLSingleConfig = self._app_config.rl.single_config
        # dataclasses.fields を用いて、存在するフィールドだけ更新（安全側）
        field_names = {f.name for f in fields(RLSingleConfig)}
        kwargs = {}
        if "total_timesteps" in field_names:
            kwargs["total_timesteps"] = int(self.spin_total_timesteps.value())
        if "learning_rate" in field_names:
            kwargs["learning_rate"] = float(self.spin_learning_rate.value())
        if "gamma" in field_names:
            kwargs["gamma"] = float(self.spin_gamma.value())
        if kwargs:
            try:
                self._app_config.rl.single_config = replace(single_cfg, **kwargs)
            except TypeError:
                # 何かおかしくても致命的ではないので、そのまま続行
                pass

        # StepEConfig 用の基本 kwargs
        cfg_kwargs: Dict[str, object] = {
            "symbol": self.symbol,
            "date_range": self.date_range,
            "agents": agents,
            "use_gpu": use_gpu,
        }

        # StepEConfig が SB3-PPO フィールドを持っている場合だけ追加
        se_field_names = {f.name for f in fields(StepEConfig)}
        engine = self.combo_engine.currentData()
        use_sb3 = engine == "sb3"

        if "use_sb3_ppo" in se_field_names:
            cfg_kwargs["use_sb3_ppo"] = use_sb3
        if use_sb3:
            # SB3 関連パラメータ
            if "symbol_short" in se_field_names:
                cfg_kwargs["symbol_short"] = "SOXS"  # 現状固定
            if "ppo_total_timesteps" in se_field_names:
                cfg_kwargs["ppo_total_timesteps"] = int(self.spin_ppo_timesteps.value())
            if "ppo_agent_names" in se_field_names:
                cfg_kwargs["ppo_agent_names"] = agents
            if "ppo_env_agent" in se_field_names:
                cfg_kwargs["ppo_env_agent"] = str(self.combo_env_agent.currentData() or "xsr")

        cfg = StepEConfig(**cfg_kwargs)  # type: ignore[arg-type]
        return cfg

    # ======================================================
    # ボタンハンドラ
    # ======================================================

    def _on_click_run(self) -> None:
        try:
            cfg = self.build_config_from_form()
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(
                self,
                "StepE 実行エラー",
                f"設定の取得中にエラーが発生しました:\n{type(e).__name__}: {e}",
            )
            return

        # 進捗バー ON、ボタン無効化
        self.progress_bar.setVisible(True)
        self.btn_run.setEnabled(False)

        # BackendController へ依頼
        self._backend.start_stepE_training(cfg)

    def _on_click_cancel(self) -> None:
        # Backend 側にキャンセル要求を送る（実装されていれば有効）
        if hasattr(self._backend, "request_cancel"):
            try:
                self._backend.request_cancel("stepE")
            except Exception:
                # 失敗しても UI 側では何もしない
                pass

    # ======================================================
    # 結果反映
    # ======================================================

    def update_stepE_result(self, result: StepEResult) -> None:
        """
        BackendController.stepE_finished から呼ばれる。
        グラフ・テーブル・メトリクス欄を更新する。
        """
        # 進捗バー OFF、ボタン再有効化
        self.progress_bar.setVisible(False)
        self.btn_run.setEnabled(True)

        # メトリクス表示
        lines: list[str] = []
        lines.append(f"success: {result.success}")
        lines.append(f"message: {result.message}")
        if self.symbol:
            lines.append(f"symbol: {self.symbol}")
        if self.date_range:
            lines.append(
                f"train: {self.date_range.train_start}〜{self.date_range.train_end}, "
                f"test: {self.date_range.test_start}〜{self.date_range.test_end}"
            )

        # 既存内部 RL 結果（エージェント別）
        if hasattr(result, "results_by_agent") and result.results_by_agent:
            lines.append("")
            lines.append("[内部RL結果]")
            for agent, rl_res in result.results_by_agent.items():
                if not isinstance(rl_res, RLRunResult):
                    continue
                lines.append(f"  Agent: {agent}")
                lines.append(
                    "    final_return={:.4f}, max_drawdown={:.4f}, "
                    "sharpe={:.4f}, bh_final={}".format(
                        rl_res.final_return,
                        rl_res.max_drawdown,
                        rl_res.sharpe_ratio,
                        rl_res.bh_final_return,
                    )
                )

        # SB3-PPO 結果（あれば）
        train_res = getattr(result, "ppo_train_result", None)
        test_res = getattr(result, "ppo_test_result", None)
        if train_res is not None:
            lines.append("")
            lines.append("[SB3-PPO Train]")
            for k, v in train_res.metrics.items():
                if isinstance(v, (float, int, np.floating)):
                    lines.append(f"  {k}: {float(v):.6g}")
                else:
                    lines.append(f"  {k}: {v}")
        if test_res is not None:
            lines.append("")
            lines.append("[SB3-PPO Test]")
            for k, v in test_res.metrics.items():
                if isinstance(v, (float, int, np.floating)):
                    lines.append(f"  {k}: {float(v):.6g}")
                else:
                    lines.append(f"  {k}: {v}")

        self._append_metrics_text("\n".join(lines), clear=True)

        # グラフと日次ログも更新
        self._update_equity_plot(result)
        self._update_daily_log(result)

    # ------------------------------------------------------
    # グラフ更新
    # ------------------------------------------------------

    def _update_equity_plot(self, result: StepEResult) -> None:
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity")
        ax.grid(True)

        curves: Dict[str, pd.DataFrame] = {}

        # 内部 RL の equity（最初のエージェントだけ表示）
        if hasattr(result, "results_by_agent") and result.results_by_agent:
            first_agent = sorted(result.results_by_agent.keys())[0]
            rl_res = result.results_by_agent[first_agent]
            if isinstance(rl_res, RLRunResult):
                path = rl_res.equity_curve_path
                if isinstance(path, Path) and path.is_file():
                    try:
                        df = pd.read_csv(path)
                        curves[f"RL ({first_agent})"] = df
                    except Exception:
                        pass

        # SB3-PPO train/test（DataFrame そのものを持っている想定）
        train_res = getattr(result, "ppo_train_result", None)
        if train_res is not None and hasattr(train_res, "equity_curve"):
            df = train_res.equity_curve
            if isinstance(df, pd.DataFrame):
                curves["PPO train"] = df

        test_res = getattr(result, "ppo_test_result", None)
        if test_res is not None and hasattr(test_res, "equity_curve"):
            df = test_res.equity_curve
            if isinstance(df, pd.DataFrame):
                curves["PPO test"] = df

        has_any = False
        for label, df in curves.items():
            if df is None or df.empty:
                continue
            if "Date" not in df.columns or "Equity" not in df.columns:
                continue
            x = pd.to_datetime(df["Date"])
            y = df["Equity"].astype(float)
            ax.plot(x, y, label=label)
            has_any = True

        if has_any:
            ax.legend(loc="best")

        self.canvas.draw_idle()

    # ------------------------------------------------------
    # 日次ログテーブル更新
    # ------------------------------------------------------

    def _update_daily_log(self, result: StepEResult) -> None:
        # 内部 RL の最初のエージェントの日次ログだけ表示（SB3-PPO は省略）
        csv_path: Optional[Path] = None

        if hasattr(result, "results_by_agent") and result.results_by_agent:
            first_agent = sorted(result.results_by_agent.keys())[0]
            rl_res = result.results_by_agent[first_agent]
            if isinstance(rl_res, RLRunResult):
                csv_path = rl_res.daily_log_path

        if csv_path is None or not isinstance(csv_path, Path) or not csv_path.is_file():
            # ログが無い場合はテーブルをクリア
            self.table_daily_log.clear()
            self.table_daily_log.setRowCount(0)
            self.table_daily_log.setColumnCount(0)
            return

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:  # noqa: BLE001
            self._append_metrics_text(
                f"[TabE] 日次ログCSVの読み込みに失敗しました: {csv_path} ({e})"
            )
            return

        self.table_daily_log.setRowCount(len(df))
        self.table_daily_log.setColumnCount(len(df.columns))
        self.table_daily_log.setHorizontalHeaderLabels(list(df.columns))

        for i in range(len(df)):
            for j, col in enumerate(df.columns):
                val = df.iloc[i, j]
                if isinstance(val, (float, int, np.floating)):
                    text = f"{float(val):.6g}"
                else:
                    text = str(val)
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table_daily_log.setItem(i, j, item)

        header = self.table_daily_log.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

    # ------------------------------------------------------
    # メトリクス欄への追記・ログ受信
    # ------------------------------------------------------

    def _append_metrics_text(self, text: str, clear: bool = False) -> None:
        if clear:
            base = ""
        else:
            base = self.txt_metrics.toPlainText()
            if base:
                base += "\n"
        new_text = base + text
        self.txt_metrics.setPlainText(new_text)
        # カーソルを末尾へ移動
        cursor = self.txt_metrics.textCursor()
        cursor.movePosition(cursor.End)
        self.txt_metrics.setTextCursor(cursor)

    def _on_log_message(self, message: str) -> None:
        """
        BackendController.log_message を受けて、指標欄に軽く追記する。
        StepE 以外のログも混ざる可能性があるが、そのまま流す。
        """
        self._append_metrics_text(message)
