from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)

from ai_core.services.step_f_service import StepFService, StepFConfig
from ai_core.gui._tab_utils import _parse_ymd, _ensure_datetime, _numeric_columns

if TYPE_CHECKING:
    # 実行時には使われないので循環 import にはならない
    from main_v0_6 import MainApp


# =====================
# ヘルパ関数
# =====================

# NOTE: TabA〜TabF 共通ヘルパは ai_core.gui._tab_utils に集約。

# ======================
# TabF: StepF (MARL Aggregation / Actions Heatmap)
# ======================


class TabF(ttk.Frame):
    """
    StepF タブ（MARL集約）

    機能:
    - Run StepF MARL:
        StepFService + StepFConfig を用いて、XSR/LSTM/FED のアクションを集約する
        MARL 学習・評価を実行。

    - Load actions CSV:
        StepF が出力した actions CSV（例: stepF_actions_SOXL.csv）を読み込み、
        アクションヒートマップとして可視化。

    - Load MARL equity CSV:
        StepF が出力した MARL の equity カーブ CSV を読み込み、グラフ表示。

    - Plot MARL equity + actions heatmap:
        上段に MARL 資金曲線、下段に 3エージェントのアクションヒートマップ
        （行=エージェント, 列=日付）を描画。
    """

    def __init__(self, parent: tk.Widget, app: "MainApp"):
        super().__init__(parent)
        self.app = app

        self.info_var = tk.StringVar(value="No StepF result loaded.")
        self.yscale_var = tk.StringVar(value="linear")

        # 読み込んだ結果
        self.equity_df: pd.DataFrame | None = None
        self.actions_df: pd.DataFrame | None = None

        # Matplotlib 図（上: equity, 下: actions heatmap）
        self.fig, (self.ax_eq, self.ax_heat) = plt.subplots(
            2, 1, figsize=(8, 6), sharex=True
        )
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)

        self._build_ui()

    # --------------------
    # UI 構築
    # --------------------
    def _build_ui(self) -> None:
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        left = ttk.Frame(top)
        left.pack(side=tk.LEFT)

        ttk.Button(
            left,
            text="Run StepF MARL",
            command=self.run_stepf,
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            left,
            text="Load actions CSV",
            command=self.load_actions_csv,
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            left,
            text="Load MARL equity CSV",
            command=self.load_equity_csv,
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            left,
            text="Plot equity + actions",
            command=self.plot_equity_and_actions,
        ).pack(side=tk.LEFT, padx=4)

        right = ttk.Frame(top)
        right.pack(side=tk.RIGHT)

        ttk.Label(right, text="Y-scale (equity):").pack(side=tk.LEFT)
        ttk.Radiobutton(
            right,
            text="Linear",
            value="linear",
            variable=self.yscale_var,
            command=self.plot_equity_and_actions,
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            right,
            text="Log",
            value="log",
            variable=self.yscale_var,
            command=self.plot_equity_and_actions,
        ).pack(side=tk.LEFT)

        ttk.Label(self, textvariable=self.info_var).pack(
            side=tk.TOP, anchor="w", padx=10, pady=2
        )

        fig_frame = ttk.Frame(self)
        fig_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(in_=fig_frame, side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, fig_frame)
        toolbar.update()

        # 初期表示
        self.ax_eq.set_title("StepF: MARL Equity")
        self.ax_eq.set_ylabel("Equity")
        self.ax_eq.grid(True)

        self.ax_heat.set_title("StepF: Actions Heatmap")
        self.ax_heat.set_ylabel("Agent")
        self.ax_heat.set_xlabel("Time")

        self.canvas.draw()

    # --------------------
    # StepF 実行
    # --------------------
    def run_stepf(self) -> None:
        """
        StepFService + StepFConfig を用いて MARL 集約を実行する。

        ※ StepFConfig / StepFService.run の引数仕様は ai_core 側に依存。
          ここでは「StepFConfig(symbol=symbol, date_range=dr)」→「svc.run(cfg)」
          という最小パターンで呼び出す。
        """
        dr = self.app.build_date_range()
        if dr is None:
            messagebox.showerror("Error", "DateRange を構成できませんでした。")
            return

        symbol = (self.app.symbol_var.get() or "").strip()
        if not symbol:
            messagebox.showerror("Error", "Symbol が空です。ヘッダで Symbol を指定してください。")
            return

        app_cfg = self.app.build_app_config()

        try:
            cfg = StepFConfig(symbol=symbol, date_range=dr)
        except TypeError as e:
            messagebox.showerror(
                "Error",
                "StepFConfig(symbol, date_range) の生成に失敗しました。\n"
                f"実装側のシグネチャに合わせて修正が必要です。\n\n{e}",
            )
            self.app.log(f"[StepF] ERROR creating StepFConfig: {e}")
            return

        try:
            svc = StepFService(app_cfg)
            result = svc.run(cfg)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Error", f"StepF MARL failed:\n{e}")
            self.app.log(f"[StepF] ERROR: {e}")
            return

        if not getattr(result, "success", True):
            msg = getattr(result, "message", "StepF failed.")
            messagebox.showerror("Error", f"StepF failed: {msg}")
            self.app.log(f"[StepF] FAILED: {msg}")
            self.info_var.set(f"StepF failed: {msg}")
            return

        msg = getattr(result, "message", "StepF finished.")
        self.app.log(f"[StepF] {msg}")
        self.app.set_status("StepF finished.")
        self.info_var.set(msg)

        artifacts = getattr(result, "artifacts", {}) or {}

        # equity
        eq_path = artifacts.get("equity_curve_path")
        if eq_path:
            self._load_equity_from_path(Path(eq_path))

        # actions
        actions_path = artifacts.get("actions_csv_path")
        if actions_path:
            self._load_actions_from_path(Path(actions_path))

        self.plot_equity_and_actions()

    # --------------------
    # actions CSV 読み込み
    # --------------------
    def load_actions_csv(self) -> None:
        sym = (self.app.symbol_var.get() or "").strip() or "SOXL"
        default = self.app.output_root / f"stepF_actions_{sym}.csv"
        initdir = str(default.parent) if default.exists() else str(self.app.output_root)

        p = filedialog.askopenfilename(
            title="Select StepF actions CSV",
            initialdir=initdir,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not p:
            return

        self._load_actions_from_path(Path(p))
        self.plot_equity_and_actions()

    def _load_actions_from_path(self, path: Path) -> None:
        try:
            df = pd.read_csv(path)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df = df.dropna(subset=["Date"]).sort_values("Date")
            self.actions_df = df
            self.app.log(f"[StepF] Loaded actions CSV: {path}")
            cols = [c for c in df.columns]
            self.info_var.set(
                f"Actions CSV loaded: {path.name} (rows={len(df)}, cols={cols})"
            )
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Error", f"Failed to load actions CSV:\n{e}")
            self.app.log(f"[StepF] ERROR reading actions CSV {path}: {e}")

    # --------------------
    # MARL equity CSV 読み込み
    # --------------------
    def load_equity_csv(self) -> None:
        sym = (self.app.symbol_var.get() or "").strip() or "SOXL"
        default = self.app.output_root / f"stepF_equity_{sym}.csv"
        initdir = str(default.parent) if default.exists() else str(self.app.output_root)

        p = filedialog.askopenfilename(
            title="Select StepF equity CSV",
            initialdir=initdir,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not p:
            return

        self._load_equity_from_path(Path(p))
        self.plot_equity_and_actions()

    def _load_equity_from_path(self, path: Path) -> None:
        try:
            df = pd.read_csv(path)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df = df.dropna(subset=["Date"]).sort_values("Date")
            self.equity_df = df
            self.app.log(f"[StepF] Loaded MARL equity CSV: {path}")
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Error", f"Failed to load MARL equity CSV:\n{e}")
            self.app.log(f"[StepF] ERROR reading MARL equity CSV {path}: {e}")

    # --------------------
    # MARL equity + actions ヒートマップ描画
    # --------------------
    def plot_equity_and_actions(self) -> None:
        """
        上段: MARL Equity
        下段: アクションヒートマップ（行=エージェント, 列=日付）
        """
        self.ax_eq.clear()
        self.ax_heat.clear()

        # --- Equity ---
        if self.equity_df is not None and not self.equity_df.empty:
            df_eq = self.equity_df.copy()
            if "Date" in df_eq.columns:
                df_eq["Date"] = _ensure_datetime(df_eq["Date"])
                df_eq = df_eq.dropna(subset=["Date"]).sort_values("Date")
                x_eq = df_eq["Date"]
                x_label = "Date"
            else:
                df_eq = df_eq.reset_index(drop=True)
                x_eq = df_eq.index
                x_label = "Step"

            num_cols_eq = _numeric_columns(df_eq, exclude=["Date"])
            if num_cols_eq:
                y_col = num_cols_eq[0]
                y_eq = df_eq[y_col]
                self.ax_eq.plot(x_eq, y_eq, label=y_col)
                self.ax_eq.legend()
            else:
                self.ax_eq.text(0.5, 0.5, "No numeric column in equity CSV", ha="center")

            self.ax_eq.set_title("StepF: MARL Equity")
            self.ax_eq.set_ylabel("Equity")
            self.ax_eq.grid(True)

            try:
                self.ax_eq.set_yscale(self.yscale_var.get())
            except ValueError:
                self.ax_eq.set_yscale("linear")
                self.yscale_var.set("linear")
        else:
            self.ax_eq.text(0.5, 0.5, "No MARL equity loaded", ha="center")
            self.ax_eq.set_title("StepF: MARL Equity (no data)")
            self.ax_eq.set_ylabel("Equity")
            self.ax_eq.grid(True)
            x_label = "Time"

        # --- Actions Heatmap ---
        if self.actions_df is not None and not self.actions_df.empty:
            df_act = self.actions_df.copy()
            if "Date" in df_act.columns:
                df_act["Date"] = _ensure_datetime(df_act["Date"])
                df_act = df_act.dropna(subset=["Date"]).sort_values("Date")
                dates = df_act["Date"]
            else:
                df_act = df_act.reset_index(drop=True)
                dates = df_act.index

            num_cols_act = _numeric_columns(df_act, exclude=["Date"])
            if num_cols_act:
                # アクション行列 shape = (num_steps, num_agents)
                mat = df_act[num_cols_act].to_numpy(dtype=float)  # [T, A]
                if mat.ndim == 1:
                    mat = mat.reshape(-1, 1)

                # imshow 用に transpose ([A, T] にする)
                img = self.ax_heat.imshow(
                    mat.T,
                    aspect="auto",
                    origin="lower",
                )

                # Y 軸にエージェント名
                self.ax_heat.set_yticks(np.arange(len(num_cols_act)))
                self.ax_heat.set_yticklabels(num_cols_act)

                # X 軸のラベリング（間引き）
                n_steps = mat.shape[0]
                if n_steps > 0:
                    step = max(1, n_steps // 10)
                    idx = np.arange(0, n_steps, step)
                    if isinstance(dates.iloc[0], (pd.Timestamp, datetime)):
                        labels = [dates.iloc[i].strftime("%Y-%m-%d") for i in idx]
                    else:
                        labels = [str(dates.iloc[i]) for i in idx]
                    self.ax_heat.set_xticks(idx)
                    self.ax_heat.set_xticklabels(labels, rotation=45, ha="right")

                self.fig.colorbar(img, ax=self.ax_heat, orientation="vertical")
                self.ax_heat.set_title("StepF: Actions Heatmap")
                self.ax_heat.set_ylabel("Agent")
                self.ax_heat.set_xlabel(x_label)
            else:
                self.ax_heat.text(0.5, 0.5, "No numeric action columns", ha="center")
                self.ax_heat.set_title("StepF: Actions Heatmap (no numeric columns)")
        else:
            self.ax_heat.text(0.5, 0.5, "No actions CSV loaded", ha="center")
            self.ax_heat.set_title("StepF: Actions Heatmap (no data)")
            self.ax_heat.set_ylabel("Agent")
            self.ax_heat.set_xlabel(x_label)

        self.fig.tight_layout()
        self.canvas.draw()

        self.app.log(
            "[TabF] Plotted MARL equity + actions "
            f"(equity_rows={len(self.equity_df) if self.equity_df is not None else 0}, "
            f"actions_rows={len(self.actions_df) if self.actions_df is not None else 0})"
        )