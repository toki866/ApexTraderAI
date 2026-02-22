# gui/tab_b_train_models.py

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import dataclass
from typing import Optional

from ai_core.types.common import DateRange
from ai_core.config.step_b_config import StepBConfig
from ai_core.config.step_b_config import WaveletMambaTrainConfig
from ai_core.services.step_b_service import StepBResult, StepBAgentResult
from backend.backend_controller import BackendController


@dataclass
class TabBState:
    """
    TabB 内部で保持する簡易状態（必要なら拡張）。
    """
    current_result: Optional[StepBResult] = None
    is_running: bool = False


class TabB_TrainModelsWidget(ttk.Frame):
    """
    StepB (モデル学習) 用のタブ。

    - 左側: モデル・ハイパーパラメータ設定パネル
    - 右側: 結果テーブル（XSR / Mamba / FEDformer 別の学習結果）
    """

    def __init__(
        self,
        master,
        backend: BackendController,
        symbol_getter,
        date_range_getter,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        backend : BackendController
            start_stepB_training(...) を呼ぶための裏方。
        symbol_getter : Callable[[], str]
            現在ヘッダなどで選択されている銘柄コードを返す関数。
        date_range_getter : Callable[[], DateRange]
            現在の学習/テスト DateRange を返す関数。
        """
        super().__init__(master, **kwargs)
        self.backend = backend
        self.symbol_getter = symbol_getter
        self.date_range_getter = date_range_getter

        self.state = TabBState()

        self._build_widgets()
        self._layout_widgets()
        self._configure_treeview_columns()

    # ------------------------------------------------------------------
    # UI 構築
    # ------------------------------------------------------------------

    def _build_widgets(self) -> None:
        # 左: 設定パネル
        self.left_frame = ttk.Frame(self)
        self.chk_xsr_var = tk.BooleanVar(value=True)
        self.chk_mamba_var = tk.BooleanVar(value=True)
        self.chk_fed_var = tk.BooleanVar(value=True)

        self.chk_xsr = ttk.Checkbutton(self.left_frame, text="XSR", variable=self.chk_xsr_var)
        self.chk_mamba = ttk.Checkbutton(self.left_frame, text="Wavelet-Mamba", variable=self.chk_mamba_var)
        self.chk_fed = ttk.Checkbutton(self.left_frame, text="FEDformer", variable=self.chk_fed_var)

        # （ここに seq_len などの詳細設定パネルを足していく想定）
        # 例: self.entry_seq_len_mamba, self.entry_d_model_fed など

        self.btn_run = ttk.Button(self.left_frame, text="StepB 学習開始", command=self._on_click_run)
        self.lbl_status = ttk.Label(self.left_frame, text="待機中")

        # 右: 結果テーブル
        self.right_frame = ttk.Frame(self)
        self.tree = ttk.Treeview(
            self.right_frame,
            columns=(
                "agent",
                "best_train_loss",
                "best_val_loss",
                "final_train_loss",
                "final_val_loss",
                "lag_days",
                "n_train_samples",
                "n_valid_samples",
                "delta_csv_path",
                "model_path",
                "message",
            ),
            show="headings",
            height=8,
        )

        self.scroll_y = ttk.Scrollbar(self.right_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=self.scroll_y.set)

    def _layout_widgets(self) -> None:
        # 左フレーム
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.left_frame.columnconfigure(0, weight=1)

        row = 0
        self.chk_xsr.grid(row=row, column=0, sticky="w")
        row += 1
        self.chk_mamba.grid(row=row, column=0, sticky="w")
        row += 1
        self.chk_fed.grid(row=row, column=0, sticky="w")
        row += 1

        ttk.Separator(self.left_frame, orient="horizontal").grid(
            row=row, column=0, sticky="ew", pady=4
        )
        row += 1

        self.btn_run.grid(row=row, column=0, sticky="ew", pady=4)
        row += 1
        self.lbl_status.grid(row=row, column=0, sticky="w")
        row += 1

        # 右フレーム
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.right_frame.rowconfigure(0, weight=1)
        self.right_frame.columnconfigure(0, weight=1)

        self.tree.grid(row=0, column=0, sticky="nsew")
        self.scroll_y.grid(row=0, column=1, sticky="ns")

        # 全体の伸縮
        self.rowconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

    def _configure_treeview_columns(self) -> None:
        col_defs = {
            "agent": ("Agent", 100),
            "best_train_loss": ("Best Train Loss", 120),
            "best_val_loss": ("Best Val Loss", 120),
            "final_train_loss": ("Final Train", 120),
            "final_val_loss": ("Final Val", 120),
            "lag_days": ("Lag Days", 80),
            "n_train_samples": ("Train N", 80),
            "n_valid_samples": ("Valid N", 80),
            "delta_csv_path": ("ΔClose CSV", 200),
            "model_path": ("Model Path", 200),
            "message": ("Message", 200),
        }
        for col, (text, width) in col_defs.items():
            self.tree.heading(col, text=text)
            self.tree.column(col, width=width, anchor="center")

    # ------------------------------------------------------------------
    # イベントハンドラ
    # ------------------------------------------------------------------

    def _on_click_run(self) -> None:
        if self.state.is_running:
            return

        symbol = self.symbol_getter()
        date_range = self.date_range_getter()

        # GUI から StepBConfig を組み立てる（v1 は最小限）
        stepb_config = StepBConfig(
            symbol=symbol,
            date_range=date_range,
            train_wavelet_mamba=self.chk_mamba_var.get(),
            wavelet_mamba_config=WaveletMambaTrainConfig(),
        )

        self._set_running(True)
        self.lbl_status.config(text="StepB 実行中...")

        # Backend にジョブを依頼
        self.backend.start_stepB_training(
            stepb_config=stepb_config,
            on_finished=self._on_stepB_finished,
            on_error=self._on_stepB_error,
        )

    def _on_stepB_finished(self, result: StepBResult) -> None:
        self._set_running(False)
        self.state.current_result = result

        if result.success:
            self.lbl_status.config(text="StepB 完了")
        else:
            self.lbl_status.config(text="StepB 完了（エラーあり）")

        # 結果テーブルを更新
        self._update_result_table(result)

    def _on_stepB_error(self, exc: BaseException) -> None:
        self._set_running(False)
        self.lbl_status.config(text="StepB エラー")
        messagebox.showerror("StepB エラー", f"StepB 実行中にエラーが発生しました:\n{exc}")

    # ------------------------------------------------------------------
    # 内部ヘルパー
    # ------------------------------------------------------------------

    def _set_running(self, flag: bool) -> None:
        self.state.is_running = flag
        if flag:
            self.btn_run.config(state="disabled")
        else:
            self.btn_run.config(state="normal")

    def _update_result_table(self, result: StepBResult) -> None:
        """
        StepBResult → Treeview 行へ反映する。
        """
        # 既存行クリア
        for item_id in self.tree.get_children():
            self.tree.delete(item_id)

        if not result.agents:
            return

        for agent_name, agent_res in result.agents.items():
            assert isinstance(agent_res, StepBAgentResult)
            m = agent_res.metrics or {}
            a = agent_res.artifacts or {}

            values = [
                agent_name,
                m.get("best_train_loss"),
                m.get("best_val_loss"),
                m.get("final_train_loss"),
                m.get("final_val_loss"),
                m.get("lag_days"),
                m.get("n_train_samples"),
                m.get("n_valid_samples"),
                a.get("delta_csv_path", ""),
                a.get("model_path", ""),
                agent_res.message,
            ]
            self.tree.insert("", "end", values=values)
