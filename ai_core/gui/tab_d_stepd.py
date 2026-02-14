from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, List

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)

from ai_core.services.step_d_service import StepDService
from ai_core.gui._tab_utils import _parse_ymd, _ensure_datetime, _numeric_columns

if TYPE_CHECKING:
    # 実行時には使われないので循環 import にはならない
    from main_v0_6 import MainApp


# =====================
# ヘルパ関数
# =====================

# NOTE: TabA〜TabF 共通ヘルパは ai_core.gui._tab_utils に集約。

# ======================
# TabD: StepD (Envelope)
# ======================


class TabD(ttk.Frame):
    """
    StepD タブ（Envelope + イベント）

    機能:
    - Run StepD:
        StepDService を実行して Envelope/イベント CSV を生成。

    - Load Envelope CSV:
        StepD が出力した「包絡線 + 価格」の CSV を読み込み。
        ※ Date, Close, Env_Top, Env_Bottom などを想定（列名は柔軟に対応）

    - Load Events CSV:
        StepD が出力したイベント一覧 CSV を読み込み。
        Treeview でイベントを一覧表示。

    - Plot Envelope:
        実際の株価 + 包絡線 (Top/Bottom) を重ねて描画。
    """

    def __init__(self, parent: tk.Widget, app: "MainApp"):
        super().__init__(parent)
        self.app = app

        self.info_var = tk.StringVar(value="No StepD data loaded.")
        self.yscale_var = tk.StringVar(value="linear")

        # Matplotlib 図
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)

        # イベント表示用 Treeview
        self.tree: ttk.Treeview | None = None

        self._build_ui()

    # --------------------
    # UI 構築
    # --------------------
    def _build_ui(self) -> None:
        # 上部コントロール
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        left = ttk.Frame(top)
        left.pack(side=tk.LEFT)

        ttk.Button(
            left,
            text="Run StepD (Envelope)",
            command=self.run_stepd,
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            left,
            text="Load Envelope CSV",
            command=self.load_envelope_csv,
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            left,
            text="Load Events CSV",
            command=self.load_events_csv,
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            left,
            text="Plot Envelope",
            command=self.plot_envelope,
        ).pack(side=tk.LEFT, padx=4)

        right = ttk.Frame(top)
        right.pack(side=tk.RIGHT)

        ttk.Label(right, text="Y-scale:").pack(side=tk.LEFT)
        ttk.Radiobutton(
            right,
            text="Linear",
            value="linear",
            variable=self.yscale_var,
            command=self.plot_envelope,
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            right,
            text="Log",
            value="log",
            variable=self.yscale_var,
            command=self.plot_envelope,
        ).pack(side=tk.LEFT)

        # 情報ラベル
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

        # イベント表エリア
        table_frame = ttk.LabelFrame(self, text="Events")
        table_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.tree = ttk.Treeview(table_frame, columns=(), show="headings", height=8)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=vsb.set)

        # 初期状態
        self.ax.set_title("StepD: Envelope (Close + Env_Top/Env_Bottom)")
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Price")
        self.canvas.draw()

    # --------------------
    # StepD 実行
    # --------------------
    def run_stepd(self) -> None:
        """
        StepDService を呼び出して Envelope/イベントを生成。
        """
        # StepA が終わっているか確認
        if self.app.prices_df is None:
            messagebox.showinfo(
                "Info",
                "まず TabA で StepA を実行（または StepA prices CSV を読み込み）してください。",
            )
            return

        # DateRange
        dr = self.app.build_date_range()
        if dr is None:
            messagebox.showerror("Error", "DateRange を構成できませんでした。")
            return

        symbol = (self.app.symbol_var.get() or "").strip()
        if not symbol:
            messagebox.showerror("Error", "Symbol が空です。ヘッダで Symbol を指定してください。")
            return

        app_cfg = self.app.build_app_config()

        svc = StepDService(app_cfg)

        result = None
        error_msgs: list[str] = []

        # StepDService.run のシグネチャが不明なのでパターンを試す（StepC と同じ方針）
        # パターン1: run(symbol, date_range)
        try:
            result = svc.run(symbol, dr)
        except TypeError as e1:
            error_msgs.append(f"run(symbol, date_range) -> {e1!r}")
        except Exception as e1:  # noqa: BLE001
            messagebox.showerror("Error", f"StepD failed:\n{e1}")
            self.app.log(f"[StepD] ERROR: {e1}")
            return

        if result is None:
            # パターン2: run(date_range) だけ
            try:
                result = svc.run(dr)  # type: ignore[arg-type]
            except TypeError as e2:
                error_msgs.append(f"run(date_range) -> {e2!r}")
            except Exception as e2:  # noqa: BLE001
                messagebox.showerror("Error", f"StepD failed:\n{e2}")
                self.app.log(f"[StepD] ERROR: {e2}")
                return

        if result is None:
            msg = "StepDService.run の呼び出しに失敗しました。\n" + "\n".join(error_msgs)
            messagebox.showerror("Error", msg)
            self.app.log(f"[StepD] ERROR: {msg}")
            return

        # 成否判定
        if not getattr(result, "success", True):
            msg = getattr(result, "message", "StepD failed.")
            messagebox.showerror("Error", f"StepD failed: {msg}")
            self.app.log(f"[StepD] FAILED: {msg}")
            return

        msg = getattr(result, "message", "StepD finished.")
        self.app.log(f"[StepD] {msg}")
        self.info_var.set(f"StepD finished: {msg}")
        self.app.set_status("StepD finished.")

        # 自動的に envelope/events CSV を探して読み込んでみる（見つからなければ何もしない）
        self._auto_load_outputs(symbol)

        # グラフ更新
        self.plot_envelope()
        self._refresh_events_table()

    def _auto_load_outputs(self, symbol: str) -> None:
        """
        出力ディレクトリから StepD の Envelope/イベント CSV を自動で探して読み込む。
        - envelope: 名前に 'envelope' が含まれてかつ symbol を含む CSV
        - events  : 名前に 'events'   が含まれてかつ symbol を含む CSV
        """
        out_root: Path = self.app.output_root
        if not out_root.exists():
            return

        # Envelope CSV 探索
        env_candidates = list(out_root.glob(f"*envelope*{symbol}*.csv"))
        if not env_candidates:
            # stepD_envelope_{symbol}.csv などの名前を想定して fallback
            env_candidates = list(out_root.glob(f"stepD_*{symbol}*.csv"))
            env_candidates = [p for p in env_candidates if "events" not in p.name.lower()]

        if env_candidates:
            env_path = sorted(env_candidates)[0]
            try:
                df_env = pd.read_csv(env_path)
                if "Date" in df_env.columns:
                    df_env["Date"] = pd.to_datetime(df_env["Date"], errors="coerce")
                    df_env = df_env.dropna(subset=["Date"]).sort_values("Date")
                self.app.envelope_df = df_env  # type: ignore[attr-defined]
                self.app.log(f"[StepD] Auto-loaded envelope CSV: {env_path}")
                self.info_var.set(
                    f"StepD envelope loaded: {env_path.name} (rows={len(df_env)})"
                )
            except Exception as e:  # noqa: BLE001
                self.app.log(f"[StepD] Failed to auto-load envelope CSV: {env_path} / {e}")

        # Events CSV 探索
        ev_candidates = list(out_root.glob(f"*events*{symbol}*.csv"))
        if not ev_candidates:
            ev_candidates = list(out_root.glob(f"stepD_events*{symbol}*.csv"))

        if ev_candidates:
            ev_path = sorted(ev_candidates)[0]
            try:
                df_ev = pd.read_csv(ev_path)
                self.app.events_df = df_ev  # type: ignore[attr-defined]
                self.app.log(f"[StepD] Auto-loaded events CSV: {ev_path}")
            except Exception as e:  # noqa: BLE001
                self.app.log(f"[StepD] Failed to auto-load events CSV: {ev_path} / {e}")

    # --------------------
    # Envelope CSV の手動読み込み
    # --------------------
    def load_envelope_csv(self) -> None:
        sym = (self.app.symbol_var.get() or "").strip() or "SOXL"
        default = self.app.output_root / f"stepD_envelope_{sym}.csv"
        initdir = str(default.parent) if default.exists() else str(self.app.output_root)

        p = filedialog.askopenfilename(
            title="Select StepD envelope CSV",
            initialdir=initdir,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not p:
            return

        env_path = Path(p)
        try:
            df_env = pd.read_csv(env_path)
            if "Date" not in df_env.columns:
                raise ValueError("Envelope CSV must contain 'Date' column.")
            df_env["Date"] = pd.to_datetime(df_env["Date"], errors="coerce")
            df_env = df_env.dropna(subset=["Date"]).sort_values("Date")
        except Exception as e:  # noqa: BLE001
            messagebox.showerror(
                "Error",
                f"Envelope CSV の読み込みに失敗しました。\n{e}",
            )
            self.app.log(f"[StepD] ERROR reading envelope CSV: {e}")
            return

        self.app.envelope_df = df_env  # type: ignore[attr-defined]
        self.info_var.set(f"Loaded envelope CSV: {env_path.name} (rows={len(df_env)})")
        self.app.log(f"[StepD] Loaded envelope CSV: {env_path}")

        self.plot_envelope()

    # --------------------
    # Events CSV の手動読み込み
    # --------------------
    def load_events_csv(self) -> None:
        sym = (self.app.symbol_var.get() or "").strip() or "SOXL"
        default = self.app.output_root / f"stepD_events_{sym}.csv"
        initdir = str(default.parent) if default.exists() else str(self.app.output_root)

        p = filedialog.askopenfilename(
            title="Select StepD events CSV",
            initialdir=initdir,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not p:
            return

        ev_path = Path(p)
        try:
            df_ev = pd.read_csv(ev_path)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror(
                "Error",
                f"Events CSV の読み込みに失敗しました。\n{e}",
            )
            self.app.log(f"[StepD] ERROR reading events CSV: {e}")
            return

        self.app.events_df = df_ev  # type: ignore[attr-defined]
        self.app.log(f"[StepD] Loaded events CSV: {ev_path}")
        self._refresh_events_table()

    # --------------------
    # Envelope 描画
    # --------------------
    def plot_envelope(self) -> None:
        """
        実際の株価 + Envelope(Top/Bottom) を描画。
        - envelope_df があれば、そこにある Close / Env_* を優先
        - 無ければ、prices_df の Close だけ描画
        """
        prices = self.app.prices_df
        env = getattr(self.app, "envelope_df", None)

        if prices is None:
            return

        # 実株価
        df_price = prices.copy()
        if "Date" not in df_price.columns or "Close" not in df_price.columns:
            messagebox.showerror(
                "Error",
                "prices_df に 'Date' または 'Close' 列がありません。",
            )
            return
        df_price["Date"] = _ensure_datetime(df_price["Date"])
        df_price = df_price.dropna(subset=["Date"]).sort_values("Date")

        # Envelope データ（あれば）
        df_env = None
        if env is not None:
            df_env = env.copy()
            if "Date" in df_env.columns:
                df_env["Date"] = _ensure_datetime(df_env["Date"])
                df_env = df_env.dropna(subset=["Date"]).sort_values("Date")

        self.ax.clear()

        # 価格ライン（Envelope 側に Close があればそちらを使用）
        if df_env is not None and "Close" in df_env.columns:
            self.ax.plot(df_env["Date"], df_env["Close"], label="Close")
            base_dates = df_env["Date"]
        else:
            self.ax.plot(df_price["Date"], df_price["Close"], label="Close")
            base_dates = df_price["Date"]

        # Envelope ライン
        if df_env is not None:
            # Env_* 系の列を自動検出（例: Env_Top, Env_Bottom）
            env_cols = [c for c in df_env.columns if c.lower().startswith("env")]
            if not env_cols:
                # 'Top' 'Bottom' だけのケースも考慮
                cand = [c for c in df_env.columns if c.lower() in ("top", "bottom")]
                env_cols = cand

            for col in env_cols:
                self.ax.plot(df_env["Date"], df_env[col], label=col)

        self.ax.set_title("StepD: Envelope (Close + Env)")
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Price")
        self.ax.grid(True)

        # Y軸スケール
        try:
            self.ax.set_yscale(self.yscale_var.get())
        except ValueError:
            self.ax.set_yscale("linear")
            self.yscale_var.set("linear")

        self.fig.autofmt_xdate()
        self.canvas.draw()

        self.app.log(
            f"[TabD] Plotted Envelope. yscale={self.yscale_var.get()}, "
            f"env_cols={env_cols if env is not None else []}"
        )

    # --------------------
    # イベント一覧の Treeview 更新
    # --------------------
    def _refresh_events_table(self) -> None:
        if self.tree is None:
            return

        df_ev = getattr(self.app, "events_df", None)
        if df_ev is None or df_ev.empty:
            # テーブルをクリア
            self.tree.delete(*self.tree.get_children())
            self.tree["columns"] = ()
            return

        # 列設定（最大 12 列くらいまで表示、残りは無視）
        cols = list(df_ev.columns)
        max_cols = 12
        if len(cols) > max_cols:
            cols = cols[:max_cols]

        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = cols

        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=100, anchor="center")

        # 行追加（大きすぎると重いので 500 行までに制限）
        max_rows = 500
        for _, row in df_ev.head(max_rows).iterrows():
            values = [row.get(c, "") for c in cols]
            # numpy型はそのまま表示できるよう str にする
            values = [str(v) if not isinstance(v, (int, float, str)) else v for v in values]
            self.tree.insert("", tk.END, values=values)

        self.app.log(
            f"[TabD] Events table refreshed. rows={min(len(df_ev), max_rows)}, cols={cols}"
        )