from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import pandas as pd

from ai_core.services.step_b_service import StepBService, StepBResult
from ai_core.config.step_b_config import StepBConfig
from ai_core.gui._tab_utils import _parse_ymd, _ensure_datetime, _numeric_columns

if TYPE_CHECKING:
    # 型ヒント用。実行時には使われないので循環 import にはならない
    from main_v0_6 import MainApp


# =====================
# ヘルパ関数
# =====================

# NOTE: TabA〜TabF 共通ヘルパは ai_core.gui._tab_utils に集約。

# ======================
# TabB: StepB
# ======================


class TabB(ttk.Frame):
    """
    StepB タブ（学習＆予測）

    - 「Recalculate split」
        ヘッダの TrainEnd を使って Train/Test の行数を計算して表示

    - 「Run StepB training」
        StepBService.run(date_range, config) を実行
        ※config は ai_core.config.step_b_config.StepBConfig をそのまま使用
    """

    def __init__(self, parent: tk.Widget, app: "MainApp"):
        super().__init__(parent)
        self.app = app

        self.summary_var = tk.StringVar(value="Train/Test rows: - / -")

        # StepB agents enable flags (GUI)
        # - These flags are mapped to StepBConfig.train_* and StepBConfig.<agent>.enabled.
        # - If all are OFF, StepBService will refuse to run.
        self.train_xsr_var = tk.BooleanVar(value=True)
        self.train_mamba_var = tk.BooleanVar(value=True)
        self.train_fedformer_var = tk.BooleanVar(value=True)

        self._build()

    # --------------------
    # GUI 構築
    # --------------------
    def _build(self) -> None:
        # 上部ボタン群
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        ttk.Label(top, text="TrainEnd (YYYY-MM-DD):").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.app.train_end_var, width=12).pack(
            side=tk.LEFT, padx=4
        )

        # Agent enable toggles (mapped to StepBConfig)
        ttk.Label(top, text="Models:").pack(side=tk.LEFT, padx=(10, 4))
        ttk.Checkbutton(top, text="XSR", variable=self.train_xsr_var, onvalue=True, offvalue=False).pack(side=tk.LEFT)
        ttk.Checkbutton(top, text="Wavelet-Mamba", variable=self.train_mamba_var, onvalue=True, offvalue=False).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Checkbutton(top, text="FEDformer", variable=self.train_fedformer_var, onvalue=True, offvalue=False).pack(side=tk.LEFT, padx=(4, 0))

        ttk.Button(
            top,
            text="Recalculate split",
            command=self.recalculate_split,
        ).pack(side=tk.LEFT, padx=4)

        ttk.Button(
            top,
            text="Run StepB training",
            command=self.run_stepb,
        ).pack(side=tk.LEFT, padx=8)

        # 説明ラベル
        ttk.Label(
            self,
            text="※現時点では、StepBConfig の初期値どおりの設定で学習・予測します。",
        ).pack(side=tk.TOP, anchor="w", padx=10, pady=4)

        # サマリ表示
        ttk.Label(self, textvariable=self.summary_var).pack(
            side=tk.TOP, anchor="w", padx=10, pady=4
        )

    # --------------------
    # Train/Test 分割計算
    # --------------------
    # --------------------
    # ログ出力ヘルパ
    # --------------------
    def _append_log(self, msg: str) -> None:
        """TabB 内部のログ出力。

        MainApp 側のログAPIは実装揺れがあるため、
        まず `app.log(msg)` を試し、無ければ `print` にフォールバックする。
        """
        try:
            fn = getattr(self.app, 'log', None)
            if callable(fn):
                fn(msg)
                return
        except Exception:
            pass
        print(msg)

    def recalculate_split(self) -> None:
        """
        ヘッダ部の TrainEnd を使って、prices_df を
        Train/Test に分けて行数を表示する。
        """
        df = self.app.prices_df
        if df is None:
            messagebox.showinfo("Info", "まず TabA で StepA を実行 (または CSV を読み込み) してください。")
            return

        if "Date" not in df.columns:
            messagebox.showerror("Error", "prices_df に 'Date' 列がありません。")
            return

        # TrainEnd を取得（未入力なら最終日）
        train_end_input = (self.app.train_end_var.get() or "").strip()
        if train_end_input:
            try:
                train_end = datetime.strptime(train_end_input, "%Y-%m-%d").date()
            except ValueError:
                messagebox.showerror("Error", "TrainEnd は YYYY-MM-DD 形式で入力してください。")
                return
        else:
            # 未入力ならデータの最終日を採用
            dates = _ensure_datetime(df["Date"])
            train_end = dates.max().date()
            self.app.train_end_var.set(train_end.strftime("%Y-%m-%d"))

        dates = _ensure_datetime(df["Date"])
        mask_train = dates <= pd.Timestamp(train_end)
        n_train = int(mask_train.sum())
        n_test = int((~mask_train).sum())

        self.summary_var.set(f"Train/Test rows: {n_train} / {n_test}")
        self.app.log(
            f"[TabB] Train/Test split recalculated. Train={n_train}, Test={n_test} (train_end={train_end})"
        )

        # TestStart/TestEnd もついでに補完
        if n_test > 0:
            test_start = (train_end + timedelta(days=1))
            test_end = dates[~mask_train].max().date()
            self.app.test_start_var.set(test_start.strftime("%Y-%m-%d"))
            self.app.test_end_var.set(test_end.strftime("%Y-%m-%d"))

    # --------------------
    # StepB 実行
    # --------------------
    def run_stepb(self) -> None:
        """
        StepBService.run(...) を実行する。
        """
        # StepA が終わっているか確認
        if self.app.prices_df is None:
            messagebox.showinfo("Info", "まず TabA で StepA を実行 (または CSV を読み込み) してください。")
            return

        # DateRange（MainApp で組み立てる想定）
        dr = self.app.build_date_range()
        if dr is None:
            messagebox.showerror("Error", "DateRange を構成できませんでした。（StepA のデータが未設定？）")
            return

        symbol = (self.app.symbol_var.get() or "").strip()
        if not symbol:
            messagebox.showerror("Error", "Symbol が空です。ヘッダで Symbol を指定してください。")
            return

        # AppConfig を組み立て
        app_cfg = self.app.build_app_config()

        # StepBConfig を組み立て（GUI の有効フラグを反映）
        train_xsr = bool(self.train_xsr_var.get())
        train_mamba = bool(self.train_mamba_var.get())
        train_fedformer = bool(self.train_fedformer_var.get())
        self._append_log(f"[StepB] Selected models: xsr={train_xsr}, mamba={train_mamba}, fedformer={train_fedformer}")
        if not (train_xsr or train_mamba or train_fedformer):
            self._append_log("[StepB] FAILED: No agents enabled in GUI (XSR/Mamba/FEDformer all OFF).")
            messagebox.showerror(
                "StepB",
                "No agents enabled.\n\nModels のチェックで、少なくとも 1 つを ON にしてください。",
            )
            return

        try:
            cfg = StepBConfig(
                symbol=symbol,
                date_range=dr,
                train_xsr=train_xsr,
                train_wavelet_mamba=train_mamba,
                train_fedformer=train_fedformer,
            )
        except TypeError:
            # 互換性のため: 旧 StepBConfig でも動くように best-effort で属性を埋める
            try:
                cfg = StepBConfig(symbol=symbol, date_range=dr)  # type: ignore[arg-type]
            except TypeError:
                cfg = StepBConfig(symbol=symbol)  # type: ignore[arg-type]
            if hasattr(cfg, "train_xsr"):
                setattr(cfg, "train_xsr", train_xsr)
            if hasattr(cfg, "train_wavelet_mamba"):
                setattr(cfg, "train_wavelet_mamba", train_mamba)
            if hasattr(cfg, "train_fedformer"):
                setattr(cfg, "train_fedformer", train_fedformer)

        # StepBService が参照する <agent>.enabled を強制同期
        
        # StepBService は cfg.{xsr,mamba,fedformer}.enabled を参照するため、GUI のチェックを同期
        for _name, _flag in (("xsr", train_xsr), ("mamba", train_mamba), ("fedformer", train_fedformer)):
            _obj = getattr(cfg, _name, None)
            if _obj is not None and hasattr(_obj, "enabled"):
                try:
                    _obj.enabled = bool(_flag)  # type: ignore[attr-defined]
                except Exception:
                    pass


        # StepB 実行
        try:
            svc = StepBService(app_cfg)
            # 以前のエラーから推測すると run(date_range, config) 形式
            # （もし run(symbol, date_range, config) だったりしたら、
            #  そのときの TypeError を見て修正する）
            result: StepBResult = svc.run(dr, cfg)
        except TypeError as e:
            # シグネチャが違った場合のフォールバック（symbol を追加して再試行）
            try:
                svc = StepBService(app_cfg)
                result = svc.run(symbol, dr, cfg)  # type: ignore[assignment]
            except Exception as e2:  # noqa: BLE001
                messagebox.showerror(
                    "Error",
                    f"StepB training failed (run signature mismatch):\n{e}\n\nSecond trial:\n{e2}",
                )
                self.app.log(f"[StepB] ERROR (run signature mismatch): {e} / {e2}")
                return
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Error", f"StepB training failed:\n{e}")
            self.app.log(f"[StepB] ERROR: {e}")
            return

        # 成功・失敗判定
        if not getattr(result, "success", True):
            msg = getattr(result, "message", "StepB failed.")
            messagebox.showerror("Error", f"StepB failed: {msg}")
            self.app.log(f"[StepB] FAILED: {msg}")
            return

        msg = getattr(result, "message", "StepB finished.")
        self.app.log(f"[StepB] {msg}")
        self.app.set_status("StepB training finished.")