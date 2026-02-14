# ai_core/gui/main_window.py
from __future__ import annotations

"""
MainWindow（Qt / PySide6）

このファイルは「importゲート」を通すために、次を満たすように修正してある。

- すべての import を `ai_core.*` の絶対importに統一（`backend.*` や `gui.*` を使わない）
- TabA のファイル名 / クラス名が環境によって揺れても、MainWindow 自体は import できるように
  - import / 生成に失敗したタブは「エラー表示タブ」に置き換える（GUIが落ちない）
- BackendController 側に StepA〜D / start_full_pipeline が未実装でも import できるように
  - シグナル接続は hasattr でガード
  - 「A→F全ステップ実行」ボタンは backend の API が揃っている時だけ実行し、無い場合は警告を出す

注意:
- TabA〜F の実体は、あなたの実装ファイル（ai_core/gui/tab_*.py）に依存する。
- このファイルだけで “全機能” を完成させるのではなく、importゲートを確実に通すことを優先している。
"""

from dataclasses import dataclass
import importlib
from typing import Any, Optional, Sequence, Tuple, Type

from PySide6.QtWidgets import (
    QComboBox,
    QDateEdit,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Signal

from ai_core.config.app_config import AppConfig
from ai_core.types.common import DateRange
from ai_core.backend.backend_controller import BackendController


# =====================================================
# タブの安全ロード
# =====================================================

@dataclass(frozen=True)
class _TabSpec:
    """タブの候補（module, class）"""
    module: str
    cls: str


class _ErrorTabWidget(QWidget):
    """タブのimport/生成に失敗した時に表示する代替タブ。"""

    def __init__(self, title: str, details: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        layout.addWidget(QLabel(f"❌ {title} の読み込みに失敗しました"))
        msg = QPlainTextEdit()
        msg.setReadOnly(True)
        msg.setPlainText(details)
        layout.addWidget(msg, stretch=1)


def _try_import_tab(specs: Sequence[_TabSpec]) -> Tuple[Optional[Type[QWidget]], str]:
    """
    specs の順番で import し、最初に見つかった QWidget 派生クラスを返す。
    失敗した場合は (None, error_text) を返す。
    """
    errors: list[str] = []
    for sp in specs:
        try:
            mod = importlib.import_module(sp.module)
            obj = getattr(mod, sp.cls, None)
            if obj is None:
                errors.append(f"- {sp.module}:{sp.cls} が見つかりません")
                continue
            if not isinstance(obj, type):
                errors.append(f"- {sp.module}:{sp.cls} は class ではありません（type={type(obj).__name__}）")
                continue
            return obj, ""
        except Exception as e:  # noqa: BLE001
            errors.append(f"- {sp.module}:{sp.cls} import失敗: {type(e).__name__}: {e}")
    return None, "\n".join(errors)


def _safe_create_tab(title: str, specs: Sequence[_TabSpec], app_config: AppConfig, backend: BackendController) -> QWidget:
    cls, err = _try_import_tab(specs)
    if cls is None:
        return _ErrorTabWidget(title, err)
    try:
        # タブのコンストラクタ引数は「(app_config, backend)」に揃えるのが基本方針
        return cls(app_config, backend)  # type: ignore[misc]
    except Exception as e:  # noqa: BLE001
        return _ErrorTabWidget(title, f"コンストラクタで例外: {type(e).__name__}: {e}")


# =====================================================
# ヘッダウィジェット
# =====================================================

class HeaderWidget(QGroupBox):
    """
    画面上部の共通ヘッダ。

    - 銘柄選択
    - テスト開始日
    - 「A→F 全ステップ実行」ボタン
    """

    symbolChanged = Signal(str)
    testStartDateChanged = Signal(object)  # datetime.date
    runAllStepsRequested = Signal()

    def __init__(self, app_config: AppConfig, parent: Optional[QWidget] = None) -> None:
        super().__init__("設定・実行", parent)

        self.app_config = app_config

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        symbol_box = QGroupBox("Symbol")
        symbol_layout = QHBoxLayout()
        self.cmb_symbol = QComboBox()
        for sym in self.app_config.data.symbol_list:
            self.cmb_symbol.addItem(sym)
        self.cmb_symbol.currentTextChanged.connect(self.symbolChanged)
        symbol_layout.addWidget(self.cmb_symbol)
        symbol_box.setLayout(symbol_layout)

        date_box = QGroupBox("テスト開始日")
        date_layout = QHBoxLayout()
        self.date_test_start = QDateEdit()
        self.date_test_start.setCalendarPopup(True)
        self.date_test_start.setDate(self.app_config.data.default_test_start_qdate())
        self.date_test_start.dateChanged.connect(self._on_date_changed)
        date_layout.addWidget(self.date_test_start)
        date_box.setLayout(date_layout)

        self.btn_run_all = QPushButton("A→F 全ステップ実行")
        self.btn_run_all.clicked.connect(self._on_click_run_all)

        layout.addWidget(symbol_box)
        layout.addWidget(date_box)
        layout.addWidget(self.btn_run_all)
        layout.addStretch(1)

    def _on_date_changed(self, qdate) -> None:
        self.testStartDateChanged.emit(qdate.toPython())

    def _on_click_run_all(self) -> None:
        self.runAllStepsRequested.emit()

    def set_run_all_enabled(self, enabled: bool) -> None:
        self.btn_run_all.setEnabled(enabled)


# =====================================================
# MainWindow
# =====================================================

class MainWindow(QMainWindow):
    """
    投資AIシステム GUI のメインウィンドウ。

    - 上: HeaderWidget
    - 中央: TabA〜TabF
    - 下: ログビュー + ステータスバー
    """

    def __init__(self, app_config: AppConfig, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self.app_config = app_config
        self.backend = BackendController(app_config)

        self.current_symbol: str = self.app_config.data.symbol_list[0]
        self.current_date_range: Optional[DateRange] = None

        self._build_ui()
        self._connect_backend_signals()

        self._update_date_range_from_header()

        self.setWindowTitle("SOXL RL GUI (StepA〜F)")
        self.resize(1200, 800)

    def _build_ui(self) -> None:
        central = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        self.header = HeaderWidget(self.app_config)
        self.header.symbolChanged.connect(self._on_symbol_changed)
        self.header.testStartDateChanged.connect(self._on_test_start_changed)
        self.header.runAllStepsRequested.connect(self._on_run_all_steps_requested)

        self.tabs = QTabWidget()

        self.tabA = _safe_create_tab(
            "StepA",
            specs=[
                _TabSpec("ai_core.gui.tab_a_stepa", "TabA"),
                _TabSpec("ai_core.gui.tab_a_stepa", "TabA_StepAWidget"),
                _TabSpec("ai_core.gui.tab_a_features_widget", "TabA_FeaturesWidget"),
            ],
            app_config=self.app_config,
            backend=self.backend,
        )
        self.tabB = _safe_create_tab(
            "StepB",
            specs=[
                _TabSpec("ai_core.gui.tab_b_train_models_widget", "TabB_TrainModelsWidget"),
                _TabSpec("ai_core.gui.tab_b_stepb", "TabB"),
            ],
            app_config=self.app_config,
            backend=self.backend,
        )
        self.tabC = _safe_create_tab(
            "StepC",
            specs=[
                _TabSpec("ai_core.gui.tab_c_time_recon_widget", "TabC_TimeReconWidget"),
                _TabSpec("ai_core.gui.tab_c_stepc", "TabC"),
            ],
            app_config=self.app_config,
            backend=self.backend,
        )
        self.tabD = _safe_create_tab(
            "StepD",
            specs=[
                _TabSpec("ai_core.gui.tab_d_envelope_widget", "TabD_EnvelopeWidget"),
                _TabSpec("ai_core.gui.tab_d_stepd", "TabD"),
            ],
            app_config=self.app_config,
            backend=self.backend,
        )
        self.tabE = _safe_create_tab(
            "StepE",
            specs=[
                _TabSpec("ai_core.gui.tab_e_rl_widget", "TabE_RLWidget"),
                _TabSpec("ai_core.gui.tab_e_stepe", "TabE"),
            ],
            app_config=self.app_config,
            backend=self.backend,
        )
        self.tabF = _safe_create_tab(
            "StepF",
            specs=[
                _TabSpec("ai_core.gui.tab_f_marl_widget", "TabF_MARLWidget"),
                _TabSpec("ai_core.gui.tab_f_stepf", "TabF"),
            ],
            app_config=self.app_config,
            backend=self.backend,
        )

        self.tabs.addTab(self.tabA, "StepA 特徴量")
        self.tabs.addTab(self.tabB, "StepB 学習")
        self.tabs.addTab(self.tabC, "StepC TimeRecon")
        self.tabs.addTab(self.tabD, "StepD Envelope")
        self.tabs.addTab(self.tabE, "StepE RL")
        self.tabs.addTab(self.tabF, "StepF MARL")

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)

        main_layout.addWidget(self.header)
        main_layout.addWidget(self.tabs, stretch=1)
        main_layout.addWidget(QLabel("ログ"))
        main_layout.addWidget(self.log_view)

        self.setCentralWidget(central)

        self.status = QStatusBar()
        self.setStatusBar(self.status)

    def _connect_backend_signals(self) -> None:
        if hasattr(self.backend, "log_message"):
            self.backend.log_message.connect(self._append_log)

        if hasattr(self.backend, "stepE_finished") and hasattr(self.tabE, "update_stepE_result"):
            self.backend.stepE_finished.connect(self.tabE.update_stepE_result)
        if hasattr(self.backend, "stepF_finished") and hasattr(self.tabF, "update_stepF_result"):
            self.backend.stepF_finished.connect(self.tabF.update_stepF_result)

        # StepA〜D などは backend 実装があるときだけ接続
        if hasattr(self.backend, "stepA_finished") and hasattr(self.tabA, "on_stepA_finished"):
            self.backend.stepA_finished.connect(self.tabA.on_stepA_finished)  # type: ignore[attr-defined]
        if hasattr(self.backend, "stepB_finished") and hasattr(self.tabB, "update_result_table"):
            self.backend.stepB_finished.connect(self.tabB.update_result_table)  # type: ignore[attr-defined]
        if hasattr(self.backend, "stepC_finished") and hasattr(self.tabC, "on_stepC_finished"):
            self.backend.stepC_finished.connect(self.tabC.on_stepC_finished)  # type: ignore[attr-defined]
        if hasattr(self.backend, "stepD_finished") and hasattr(self.tabD, "on_stepD_finished"):
            self.backend.stepD_finished.connect(self.tabD.on_stepD_finished)  # type: ignore[attr-defined]

    def _on_symbol_changed(self, symbol: str) -> None:
        self.current_symbol = symbol
        self._update_tabs_symbol_and_range()

    def _on_test_start_changed(self, test_start_date) -> None:
        from datetime import timedelta

        train_start = test_start_date.replace(year=test_start_date.year - 8)
        train_end = test_start_date - timedelta(days=1)
        test_end = test_start_date + timedelta(days=90)

        self.current_date_range = DateRange(
            train_start=train_start,
            train_end=train_end,
            test_start=test_start_date,
            test_end=test_end,
        )
        self._update_tabs_symbol_and_range()

    def _update_date_range_from_header(self) -> None:
        qdate = self.header.date_test_start.date()
        self._on_test_start_changed(qdate.toPython())

    def _update_tabs_symbol_and_range(self) -> None:
        if self.current_date_range is None:
            return

        dr = self.current_date_range
        sym = self.current_symbol

        if hasattr(self.backend, "set_symbol_and_range"):
            self.backend.set_symbol_and_range(sym, dr)

        for tab in (self.tabA, self.tabB, self.tabC, self.tabD, self.tabE, self.tabF):
            if hasattr(tab, "set_symbol_and_range"):
                try:
                    tab.set_symbol_and_range(sym, dr)
                except Exception as e:  # noqa: BLE001
                    self._append_log(f"[UI] set_symbol_and_range 失敗: {type(e).__name__}: {e}")

    def _on_run_all_steps_requested(self) -> None:
        if self.current_date_range is None:
            QMessageBox.warning(self, "A→F 実行エラー", "DateRange が未設定です。テスト開始日を確認してください。")
            return

        if not hasattr(self.backend, "start_full_pipeline"):
            QMessageBox.information(
                self,
                "A→F 全ステップ実行",
                "BackendController.start_full_pipeline が未実装です。\n"
                "現状は各タブの実行ボタン（または StepE/StepF の個別実行）を使ってください。",
            )
            return

        sym = self.current_symbol
        dr = self.current_date_range

        # backend 側に FullPipelineConfig がある前提（無ければエラー表示）
        try:
            backend_mod = importlib.import_module("ai_core.backend.backend_controller")
            FullPipelineConfig = getattr(backend_mod, "FullPipelineConfig", None)  # noqa: N806
            if FullPipelineConfig is None:
                raise RuntimeError("FullPipelineConfig が ai_core.backend.backend_controller に存在しません。")

            stepB_cfg = self.tabB.build_config_from_form() if hasattr(self.tabB, "build_config_from_form") else None
            stepC_cfg = self.tabC.build_config_from_form() if hasattr(self.tabC, "build_config_from_form") else None
            stepD_cfg = self.tabD.build_config_from_form() if hasattr(self.tabD, "build_config_from_form") else None
            stepE_cfg = self.tabE.build_config_from_form() if hasattr(self.tabE, "build_config_from_form") else None
            stepF_cfg = self.tabF.build_config_from_form() if hasattr(self.tabF, "build_config_from_form") else None

            pipeline_cfg = FullPipelineConfig(
                symbol=sym,
                date_range=dr,
                stepB_config=stepB_cfg,
                stepC_config=stepC_cfg,
                stepD_config=stepD_cfg,
                stepE_config=stepE_cfg,
                stepF_config=stepF_cfg,
            )
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "A→F 実行エラー", f"FullPipelineConfig の生成に失敗しました:\n{e}")
            return

        self.header.set_run_all_enabled(False)
        self._append_log("[UI] A→F 全ステップ実行をリクエストしました。")

        self.backend.start_full_pipeline(pipeline_cfg)

        if hasattr(self.backend, "stepF_finished"):
            self.backend.stepF_finished.connect(self._on_pipeline_finished_once)

    def _on_pipeline_finished_once(self, _result: Any) -> None:
        try:
            if hasattr(self.backend, "stepF_finished"):
                self.backend.stepF_finished.disconnect(self._on_pipeline_finished_once)
        except Exception:
            pass
        self.header.set_run_all_enabled(True)
        self._append_log("[UI] A→F 全ステップ実行が終了しました。")

    def _append_log(self, msg: str) -> None:
        self.log_view.appendPlainText(msg)
        self.status.showMessage(msg, 5000)
