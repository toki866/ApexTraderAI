from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)

from ai_core.services.step_e_service import StepEService, StepEConfig
from ai_core.gui._tab_utils import _parse_ymd, _ensure_datetime, _numeric_columns

if TYPE_CHECKING:
    # 実行時には使われないので循環 import にはならない
    from main_v0_6 import MainApp


# =====================
# ヘルパ関数
# =====================

# NOTE: TabA〜TabF 共通ヘルパは ai_core.gui._tab_utils に集約。

# ======================
# TabE: StepE (Single-Agent RL)
# ======================


class TabE(ttk.Frame):
    """
    StepE タブ（単体RL学習）

    機能:
    - Run StepE RL training:
        StepEService + StepEConfig を用いて XSR/LSTM/FED など単体エージェントの
        RL 学習を実行する（実際のエージェント選択は StepEConfig/AppConfig 側で制御）。

    - Load equity CSV:
        StepE が出力した equity カーブ CSV を読み込み、グラフ表示。

    - Load daily log CSV:
        StepE が出力した daily_log CSV を読み込み、テーブル表示。

    - Plot equity curve:
        equity_df を元に資金曲線を描画。
    """

    def __init__(self, parent: tk.Widget, app: "MainApp"):
        super().__init__(parent)
        self.app = app

        self.info_var = tk.StringVar(value="No StepE result loaded.")
        self.yscale_var = tk.StringVar(value="linear")

        # 読み込んだ RL 結果
        self.equity_df: pd.DataFrame | None = None
        self.daily_df: pd.DataFrame | None = None

        # Matplotlib 図
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)

        # 日次ログ用 Treeview
        self.tree: ttk.Treeview | None = None

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
            text="Run StepE RL training",
            command=self.run_stepe,
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            left,
            text="Load equity CSV",
            command=self.load_equity_csv,
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            left,
            text="Load daily log CSV",
            command=self.load_daily_log_csv,
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            left,
            text="Plot equity curve",
            command=self.plot_equity,
        ).pack(side=tk.LEFT, padx=4)

        right = ttk.Frame(top)
        right.pack(side=tk.RIGHT)

        ttk.Label(right, text="Y-scale:").pack(side=tk.LEFT)
        ttk.Radiobutton(
            right,
            text="Linear",
            value="linear",
            variable=self.yscale_var,
            command=self.plot_equity,
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            right,
            text="Log",
            value="log",
            variable=self.yscale_var,
            command=self.plot_equity,
        ).pack(side=tk.LEFT)

        ttk.Label(self, textvariable=self.info_var).pack(
            side=tk.TOP, anchor="w", padx=10, pady=2
        )

        # グラフエリア
        fig_frame = ttk.Frame(self)
        fig_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(in_=fig_frame, side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, fig_frame)
        toolbar.update()

        # 日次ログテーブル
        table_frame = ttk.LabelFrame(self, text="StepE daily log")
        table_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.tree = ttk.Treeview(table_frame, columns=(), show="headings", height=8)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=vsb.set)

        # 初期グラフ
        self.ax.set_title("StepE: RL Equity")
        self.ax.set_xlabel("Date / Step")
        self.ax.set_ylabel("Equity")
        self.canvas.draw()

    # --------------------
    # StepE 実行
    # --------------------
    def run_stepe(self) -> None:
        """
        StepEService + StepEConfig を用いて RL 学習を実行する。

        ※ StepEConfig / StepEService.run の引数仕様は ai_core 側の実装に依存する。
          ここでは「StepEConfig(symbol=symbol, date_range=dr)」→「svc.run(cfg)」
          という最小パターンで呼び出す。
        """
        if self.app.prices_df is None:
            messagebox.showinfo(
                "Info",
                "まず TabA で StepA を実行（または StepA prices CSV を読み込み）してください。",
            )
            return

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
            cfg = StepEConfig(symbol=symbol, date_range=dr)
        except TypeError as e:
            # StepEConfig のシグネチャが異なる場合は、その旨を表示して終了。
            messagebox.showerror(
                "Error",
                "StepEConfig(symbol, date_range) の生成に失敗しました。\n"
                f"実装側のシグネチャに合わせて修正が必要です。\n\n{e}",
            )
            self.app.log(f"[StepE] ERROR creating StepEConfig: {e}")
            return

        try:
            svc = StepEService(app_cfg)
            result = svc.run(cfg)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Error", f"StepE RL training failed:\n{e}")
            self.app.log(f"[StepE] ERROR: {e}")
            return

        # 成否判定
        if not getattr(result, "success", True):
            msg = getattr(result, "message", "StepE failed.")
            messagebox.showerror("Error", f"StepE failed: {msg}")
            self.app.log(f"[StepE] FAILED: {msg}")
            self.info_var.set(f"StepE failed: {msg}")
            return

        msg = getattr(result, "message", "StepE finished.")
        self.app.log(f"[StepE] {msg}")
        self.app.set_status("StepE finished.")
        self.info_var.set(msg)

        # artifacts から equity/daily_log CSV を読み込み
        artifacts = getattr(result, "artifacts", {}) or {}

        # equity curve
        equity_path = artifacts.get("equity_curve_path")
        if equity_path:
            self._load_equity_from_path(Path(equity_path))

        # daily log
        daily_path = artifacts.get("daily_log_path")
        if daily_path:
            self._load_daily_from_path(Path(daily_path))

        self.plot_equity()
        self._refresh_daily_table()

    # --------------------
    # equity CSVの読み込み
    # --------------------
    def load_equity_csv(self) -> None:
        """
        StepE の equity カーブ CSV を手動で選択して読み込む。
        """
        sym = (self.app.symbol_var.get() or "").strip() or "SOXL"
        default = self.app.output_root / f"stepE_equity_XSR_{sym}.csv"
        initdir = str(default.parent) if default.exists() else str(self.app.output_root)

        p = filedialog.askopenfilename(
            title="Select StepE equity CSV",
            initialdir=initdir,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not p:
            return

        self._load_equity_from_path(Path(p))
        self.plot_equity()

    def _load_equity_from_path(self, path: Path) -> None:
        try:
            df = pd.read_csv(path)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df = df.dropna(subset=["Date"]).sort_values("Date")
            self.equity_df = df
            self.app.log(f"[StepE] Loaded equity CSV: {path}")
            cols = [c for c in df.columns]
            self.info_var.set(f"Equity CSV loaded: {path.name} (rows={len(df)}, cols={cols})")
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Error", f"Failed to load equity CSV:\n{e}")
            self.app.log(f"[StepE] ERROR reading equity CSV {path}: {e}")

    # --------------------
    # daily_log CSVの読み込み
    # --------------------
    def load_daily_log_csv(self) -> None:
        """
        StepE の daily_log CSV を手動で選択して読み込む。
        """
        sym = (self.app.symbol_var.get() or "").strip() or "SOXL"
        default = self.app.output_root / f"daily_log_XSR_{sym}.csv"
        initdir = str(default.parent) if default.exists() else str(self.app.output_root)

        p = filedialog.askopenfilename(
            title="Select StepE daily log CSV",
            initialdir=initdir,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not p:
            return

        self._load_daily_from_path(Path(p))
        self._refresh_daily_table()

    def _load_daily_from_path(self, path: Path) -> None:
        try:
            df = pd.read_csv(path)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            self.daily_df = df
            self.app.log(f"[StepE] Loaded daily log CSV: {path}")
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Error", f"Failed to load daily log CSV:\n{e}")
            self.app.log(f"[StepE] ERROR reading daily log CSV {path}: {e}")

    # --------------------
    # Equityカーブ描画
    # --------------------
    def plot_equity(self) -> None:
        df = self.equity_df
        if df is None or df.empty:
            # まだ何も読み込まれていない場合は、prices_df があれば Close を表示しておく
            prices = self.app.prices_df
            if prices is None:
                return
            p = prices.copy()
            if "Date" not in p.columns or "Close" not in p.columns:
                return
            p["Date"] = _ensure_datetime(p["Date"])
            p = p.dropna(subset=["Date"]).sort_values("Date")

            self.ax.clear()
            self.ax.plot(p["Date"], p["Close"], label="Close (no RL yet)")
            self.ax.set_title("StepE: RL Equity (no equity CSV yet)")
            self.ax.set_xlabel("Date")
            self.ax.set_ylabel("Value")
            self.ax.grid(True)
            self.fig.autofmt_xdate()
            self.canvas.draw()
            return

        data = df.copy()
        x_col: str
        y_col: str

        # X軸は Date があればそれを使う
        if "Date" in data.columns:
            data["Date"] = _ensure_datetime(data["Date"])
            data = data.dropna(subset=["Date"]).sort_values("Date")
            x_col = "Date"
        else:
            # Date が無ければ index をステップとみなす
            data = data.reset_index(drop=True)
            x_col = "step"

        # Y軸は数値列のうち最初の1本を equity とみなす
        num_cols = _numeric_columns(data, exclude=[x_col])
        if not num_cols:
            messagebox.showwarning(
                "Warning",
                "equity CSV に数値列が見つかりませんでした。",
            )
            return
        y_col = num_cols[0]

        self.ax.clear()
        x = data[x_col]
        y = data[y_col]
        self.ax.plot(x, y, label=y_col)

        self.ax.set_title("StepE: RL Equity")
        self.ax.set_xlabel(x_col)
        self.ax.set_ylabel(y_col)
        self.ax.grid(True)
        self.ax.legend()

        try:
            self.ax.set_yscale(self.yscale_var.get())
        except ValueError:
            self.ax.set_yscale("linear")
            self.yscale_var.set("linear")

        self.fig.autofmt_xdate()
        self.canvas.draw()

        self.app.log(
            f"[TabE] Plotted equity curve from {y_col} "
            f"(yscale={self.yscale_var.get()}, rows={len(data)})"
        )

    # --------------------
    # 日次ログテーブル更新
    # --------------------
    def _refresh_daily_table(self) -> None:
        if self.tree is None:
            return

        df = self.daily_df
        if df is None or df.empty:
            self.tree.delete(*self.tree.get_children())
            self.tree["columns"] = ()
            return

        cols = list(df.columns)
        max_cols = 12
        if len(cols) > max_cols:
            cols = cols[:max_cols]

        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = cols

        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=100, anchor="center")

        max_rows = 500
        for _, row in df.head(max_rows).iterrows():
            values = [row.get(c, "") for c in cols]
            values = [str(v) if not isinstance(v, (int, float, str)) else v for v in values]
            self.tree.insert("", tk.END, values=values)

        self.app.log(
            f"[TabE] Daily log table refreshed. rows={min(len(df), max_rows)}, cols={cols}"
        )