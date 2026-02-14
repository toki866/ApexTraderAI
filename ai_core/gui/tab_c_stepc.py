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

from ai_core.services.step_c_service import StepCService
from ai_core.gui._tab_utils import _parse_ymd, _ensure_datetime, _numeric_columns

if TYPE_CHECKING:
    # 型ヒント用（実行時には使われないので循環 import にならない）
    from main_v0_6 import MainApp


# =====================
# ヘルパ関数
# =====================

# NOTE: TabA〜TabF 共通ヘルパは ai_core.gui._tab_utils に集約。

# ======================
# TabC: StepC (TimeRecon + ScaleCalib)
# ======================


class TabC(ttk.Frame):
    """
    StepC タブ

    - Run StepC:
        StepCService を呼び出し、TimeRecon + ScaleCalib を実行して
        stepC_pred_time_all_{symbol}.csv などを生成する想定。

    - Load StepC predictions CSV:
        生成済みの StepC 予測 CSV を手動で選択して読み込む。

    - Plot Close + Predictions:
        TabA で読み込んだ実際の株価 (Close) と、
        StepC の予測列（Pred_* 系）を同じグラフに重ねて表示。
    """

    def __init__(self, parent: tk.Widget, app: "MainApp"):
        super().__init__(parent)
        self.app = app

        self.info_var = tk.StringVar(value="No StepC result loaded.")
        self.yscale_var = tk.StringVar(value="linear")

        # Matplotlib Figure
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)

        self._build_ui()

    # --------------------
    # UI 構築
    # --------------------
    def _build_ui(self) -> None:
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # 左側：ボタン群
        left = ttk.Frame(top)
        left.pack(side=tk.LEFT)

        ttk.Button(
            left,
            text="Run StepC (TimeRecon + ScaleCalib)",
            command=self.run_stepc,
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            left,
            text="Load StepC predictions CSV",
            command=self.load_stepc_csv,
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            left,
            text="Plot Close + Predictions",
            command=self.plot_close_and_predictions,
        ).pack(side=tk.LEFT, padx=4)

        # 右側：Yスケール切り替え
        right = ttk.Frame(top)
        right.pack(side=tk.RIGHT)

        ttk.Label(right, text="Y-scale:").pack(side=tk.LEFT)
        ttk.Radiobutton(
            right,
            text="Linear",
            value="linear",
            variable=self.yscale_var,
            command=self.plot_close_and_predictions,
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            right,
            text="Log",
            value="log",
            variable=self.yscale_var,
            command=self.plot_close_and_predictions,
        ).pack(side=tk.LEFT)

        # 情報ラベル
        ttk.Label(self, textvariable=self.info_var).pack(
            side=tk.TOP, anchor="w", padx=10, pady=2
        )

        # グラフ領域
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, self)
        toolbar.update()

        # 初期状態は空グラフ
        self.ax.set_title("StepC: TimeRecon + ScaleCalib")
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Price")
        self.canvas.draw()

    # --------------------
    # StepC 実行
    # --------------------
    def run_stepc(self) -> None:
        """
        StepCService を実行して TimeRecon + ScaleCalib を行う。

        StepCService.run(...) のシグネチャは環境によって違う可能性があるので、
        まず run(symbol, date_range) を試し、ダメなら run(date_range) などに
        フォールバックしている。
        """
        # StepA（価格）が読み込まれているか確認
        if self.app.prices_df is None:
            messagebox.showinfo(
                "Info",
                "まず TabA で StepA を実行（または StepA prices CSV を読み込み）してください。",
            )
            return

        # DateRange（MainApp 側で構成）
        dr = self.app.build_date_range()
        if dr is None:
            messagebox.showerror("Error", "DateRange を構成できませんでした。")
            return

        symbol = (self.app.symbol_var.get() or "").strip()
        if not symbol:
            messagebox.showerror("Error", "Symbol が空です。ヘッダで Symbol を指定してください。")
            return

        app_cfg = self.app.build_app_config()

        svc = StepCService(app_cfg)

        # StepCService.run のシグネチャが不確定なので try/except で対応
        result = None
        error_msgs: list[str] = []

        # パターン1: run(symbol, date_range)
        try:
            result = svc.run(symbol, dr)
        except TypeError as e1:
            error_msgs.append(f"run(symbol, date_range) -> {e1!r}")
        except Exception as e1:  # noqa: BLE001
            messagebox.showerror("Error", f"StepC failed:\n{e1}")
            self.app.log(f"[StepC] ERROR: {e1}")
            return

        # パターン1で成功したらそのまま続行
        if result is None:
            # パターン2: run(date_range) だけ
            try:
                result = svc.run(dr)  # type: ignore[arg-type]
            except TypeError as e2:
                error_msgs.append(f"run(date_range) -> {e2!r}")
            except Exception as e2:  # noqa: BLE001
                messagebox.showerror("Error", f"StepC failed:\n{e2}")
                self.app.log(f"[StepC] ERROR: {e2}")
                return

        if result is None:
            # ここまで来るとシグネチャがまったく違うということなので、ログを吐いて終了
            msg = (
                "StepCService.run の呼び出しに失敗しました。\n"
                + "\n".join(error_msgs)
            )
            messagebox.showerror("Error", msg)
            self.app.log(f"[StepC] ERROR: {msg}")
            return

        # 成否判定
        if not getattr(result, "success", True):
            msg = getattr(result, "message", "StepC failed.")
            messagebox.showerror("Error", f"StepC failed: {msg}")
            self.app.log(f"[StepC] FAILED: {msg}")
            return

        # 予測CSVパス
        pred_path: Path | None = None
        if hasattr(result, "pred_all_path") and getattr(result, "pred_all_path") is not None:
            pred_path = Path(result.pred_all_path)

        # fallback: 仕様どおりのデフォルトパスを推測
        if pred_path is None:
            pred_path = self.app.output_root / f"stepC_pred_time_all_{symbol}.csv"

        if not pred_path.exists():
            messagebox.showwarning(
                "Warning",
                f"StepC は成功しましたが、予測CSVが見つかりませんでした。\n"
                f"期待パス: {pred_path}",
            )
            self.app.log(
                f"[StepC] SUCCESS but prediction CSV not found. expected={pred_path}"
            )
            self.info_var.set("StepC finished, but prediction CSV not found.")
            return

        # CSV 読み込み
        try:
            df_pred = pd.read_csv(pred_path)
            if "Date" not in df_pred.columns:
                raise ValueError("prediction CSV must contain 'Date' column.")
            df_pred["Date"] = pd.to_datetime(df_pred["Date"], errors="coerce")
            df_pred = df_pred.dropna(subset=["Date"]).sort_values("Date")
        except Exception as e:  # noqa: BLE001
            messagebox.showerror(
                "Error",
                f"StepC prediction CSV の読み込みに失敗しました。\n{e}",
            )
            self.app.log(f"[StepC] ERROR reading prediction CSV: {e}")
            return

        # アプリ共通状態に保存
        self.app.pred_df = df_pred

        cols = [c for c in df_pred.columns if c != "Date"]
        self.info_var.set(
            f"Loaded StepC predictions: {pred_path.name} (rows={len(df_pred)}, cols={len(cols)})"
        )
        self.app.log(
            f"[StepC] Loaded predictions from {pred_path} (rows={len(df_pred)}, cols={cols})"
        )

        # グラフ更新
        self.plot_close_and_predictions()

    # --------------------
    # 既存 StepC 予測 CSV の読み込み
    # --------------------
    def load_stepc_csv(self) -> None:
        """
        すでに生成済みの stepC_pred_time_all_{symbol}.csv などを選んで読み込む。
        """
        sym = (self.app.symbol_var.get() or "").strip() or "SOXL"
        default = self.app.output_root / f"stepC_pred_time_all_{sym}.csv"
        initdir = str(default.parent) if default.exists() else str(self.app.output_root)

        p = filedialog.askopenfilename(
            title="Select StepC prediction CSV",
            initialdir=initdir,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not p:
            return

        pred_path = Path(p)
        try:
            df_pred = pd.read_csv(pred_path)
            if "Date" not in df_pred.columns:
                raise ValueError("prediction CSV must contain 'Date' column.")
            df_pred["Date"] = pd.to_datetime(df_pred["Date"], errors="coerce")
            df_pred = df_pred.dropna(subset=["Date"]).sort_values("Date")
        except Exception as e:  # noqa: BLE001
            messagebox.showerror(
                "Error",
                f"StepC prediction CSV の読み込みに失敗しました。\n{e}",
            )
            self.app.log(f"[StepC] ERROR reading prediction CSV: {e}")
            return

        self.app.pred_df = df_pred
        cols = [c for c in df_pred.columns if c != "Date"]
        self.info_var.set(
            f"Loaded StepC predictions: {pred_path.name} (rows={len(df_pred)}, cols={len(cols)})"
        )
        self.app.log(
            f"[StepC] Loaded predictions from {pred_path} (rows={len(df_pred)}, cols={cols})"
        )

        self.plot_close_and_predictions()

    # --------------------
    # 実株価 + StepC予測の描画
    # --------------------
    def plot_close_and_predictions(self) -> None:
        """
        TabA で読み込んだ実際の Close と、StepC 予測列を重ねて描画。
        """
        prices = self.app.prices_df
        preds = self.app.pred_df

        if prices is None:
            # StepA 未実行
            return

        # 実株価
        p = prices.copy()
        if "Date" not in p.columns or "Close" not in p.columns:
            messagebox.showerror("Error", "prices_df に 'Date' または 'Close' 列がありません。")
            return
        p["Date"] = _ensure_datetime(p["Date"])
        p = p.dropna(subset=["Date"]).sort_values("Date")

        self.ax.clear()
        self.ax.plot(p["Date"], p["Close"], label="Close (actual)")

        # 予測があれば追加
        if preds is not None and not preds.empty:
            df_pred = preds.copy()
            if "Date" not in df_pred.columns:
                messagebox.showerror(
                    "Error", "StepC prediction DataFrame に 'Date' 列がありません。"
                )
                return
            df_pred["Date"] = _ensure_datetime(df_pred["Date"])
            df_pred = df_pred.dropna(subset=["Date"]).sort_values("Date")

            pred_cols = _numeric_columns(df_pred, exclude=["Date"])
            for col in pred_cols:
                self.ax.plot(df_pred["Date"], df_pred[col], label=col)

        self.ax.set_title("StepC: Close + Predictions")
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Price")
        self.ax.grid(True)

        # Y 軸スケール
        try:
            self.ax.set_yscale(self.yscale_var.get())
        except ValueError:
            # log で 0 以下があるとエラーになるので、その場合は linear に戻す
            self.ax.set_yscale("linear")
            self.yscale_var.set("linear")

        self.fig.autofmt_xdate()
        self.canvas.draw()

        self.app.log(
            "[TabC] Plotted Close + StepC predictions "
            f"(yscale={self.yscale_var.get()}, "
            f"pred_cols={list(preds.columns) if preds is not None else []})"
        )