from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)

from ai_core.gui._tab_utils import _parse_ymd


def _clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw OHLCV prices DataFrame.

    This GUI tab expects a CSV like data/prices_{sym}.csv with at least:
    - Date, Close
    and preferably OHLCV columns: Open, High, Low, Close, Volume.

    The function:
    - Ensures a 'Date' column exists and is datetime64[ns]
    - Sorts by Date, drops duplicates
    - Coerces numeric columns to float (Volume kept numeric)
    - Forward/back fills small gaps

    Raises:
        ValueError: if required columns are missing.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    out = df.copy()

    # Normalize Date column
    if "Date" not in out.columns:
        # Try common alternatives or first column
        for alt in ("date", "DATE", "Datetime", "datetime", "Time", "time"):
            if alt in out.columns:
                out = out.rename(columns={alt: "Date"})
                break
    if "Date" not in out.columns:
        # If the first column looks like dates, use it
        if len(out.columns) >= 1:
            out = out.rename(columns={out.columns[0]: "Date"})
        else:
            raise ValueError("No columns found in prices CSV")

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"]).copy()

    # Basic required columns
    if "Close" not in out.columns:
        raise ValueError("Missing required column: Close")

    # Coerce numeric columns
    numeric_candidates = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in out.columns]
    for c in numeric_candidates:
        # Remove commas/whitespace in strings
        if out[c].dtype == object:
            out[c] = out[c].astype(str).str.replace(",", "", regex=False).str.strip()
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Sort / de-duplicate
    out = out.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)

    # Fill small gaps in price columns
    price_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close"] if c in out.columns]
    if price_cols:
        out[price_cols] = out[price_cols].ffill()  # no bfill (avoid future fill)

    # Volume: fill missing with 0 (safer) then ensure non-negative
    if "Volume" in out.columns:
        out["Volume"] = out["Volume"].fillna(0)
        out.loc[out["Volume"] < 0, "Volume"] = 0

    return out

if TYPE_CHECKING:
    # 型ヒント用。実行時には使われないので循環importになりません。
    from main_v0_6 import MainApp


# ======================
# ヘルパ関数
# ======================

# NOTE:
# - _parse_ymd は ai_core.gui._tab_utils の正本を使用します（重複防止）。

# TabA: StepA
# ======================


class TabA(ttk.Frame):
    """
    StepA タブ

    - Run StepA (prices+features)
        data/prices_{sym}.csv を読み込み
        → 価格CSV と「特徴量CSV」を output に保存
          * output/stepA_prices_{sym}.csv
          * output/stepA_features_{sym}.csv
        （features には OHLCV に加えて
          Gap / RSI / MACD / MACD_signal / 周期sin/cos(44本)
         を含める）

    - Load StepA prices CSV
        上記 CSV を手動で選んで読み込み

    - Plot Close
        日付範囲と Y軸スケール（Linear/Log）を指定して Close を描画
    """

    def __init__(self, parent: tk.Widget, app: "MainApp"):
        super().__init__(parent)
        self.app = app

        self.from_var = tk.StringVar(value="")
        self.to_var = tk.StringVar(value="")
        self.yscale_var = tk.StringVar(value="linear")

        self.info_var = tk.StringVar(value="No file loaded")

        # Matplotlib Figure
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)

        self._build()

    # --------------------
    # GUI 構築
    # --------------------
    def _build(self) -> None:
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # 左：ボタン類
        left = ttk.Frame(top)
        left.pack(side=tk.LEFT)

        ttk.Button(
            left,
            text="Run StepA (prices+features)",
            command=self.run_stepa,
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            left,
            text="Load StepA prices CSV",
            command=self.load_prices,
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            left,
            text="Plot Close",
            command=self.plot_prices,
        ).pack(side=tk.LEFT, padx=4)

        # 右：日付範囲＋Y軸スケール
        right = ttk.Frame(top)
        right.pack(side=tk.RIGHT)

        ttk.Label(right, text="From:").pack(side=tk.LEFT)
        ttk.Entry(right, textvariable=self.from_var, width=12).pack(
            side=tk.LEFT, padx=2
        )

        ttk.Label(right, text="To:").pack(side=tk.LEFT)
        ttk.Entry(right, textvariable=self.to_var, width=12).pack(
            side=tk.LEFT, padx=2
        )

        ttk.Label(right, text="Y-scale:").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Radiobutton(
            right,
            text="Linear",
            value="linear",
            variable=self.yscale_var,
            command=self.plot_prices,
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            right,
            text="Log",
            value="log",
            variable=self.yscale_var,
            command=self.plot_prices,
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

    # --------------------
    # StepA 実行
    # --------------------
    
def run_stepa(self) -> None:
    """Run StepAService using the canonical implementation (split outputs).

    This GUI tab no longer re-implements feature engineering. It:
    1) Loads and cleans the selected raw prices CSV
    2) Writes it to data/prices_{sym}.csv (with a timestamped backup if overwriting)
    3) Calls ai_core.services.step_a_service.StepAService.run(sym, date_range)
    4) Loads the generated display prices CSV for plotting
    """
    import inspect
    import os
    import shutil
    from types import SimpleNamespace

    sym = self.app.symbol_var.get().strip() or "SOXL"

    # Raw prices path (default: data/prices_{sym}.csv)
    raw_default = Path("data") / f"prices_{sym}.csv"
    if raw_default.exists():
        raw_path = raw_default
    else:
        p = filedialog.askopenfilename(
            title=f"Select raw prices CSV for {sym}",
            initialdir=".",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not p:
            return
        raw_path = Path(p)

    # Load & clean
    try:
        df_raw = pd.read_csv(raw_path)
        df_clean = _clean_prices(df_raw)
    except Exception as e:  # noqa: BLE001
        messagebox.showerror("Error", f"Failed to load raw prices:\n{e}")
        return

    # Ensure canonical location for StepAService: data/prices_{sym}.csv
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    canonical = data_dir / f"prices_{sym}.csv"

    try:
        if canonical.exists() and canonical.resolve() != raw_path.resolve():
            # backup existing canonical
            old_dir = data_dir / "old"
            old_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = old_dir / f"prices_{sym}_{ts}.csv"
            shutil.copy2(canonical, backup)
        # write cleaned canonical (ensures Date parsing & ffill-only)
        df_clean.to_csv(canonical, index=False)
    except Exception as e:  # noqa: BLE001
        messagebox.showerror("Error", f"Failed to prepare canonical prices CSV:\n{e}")
        return

    # Build a best-effort DateRange:
    # - Prefer app.date_range if present
    date_range = getattr(self.app, "date_range", None) or getattr(self.app, "current_date_range", None)
    if date_range is None:
        try:
            from ai_core.types.common import DateRange
            # Defaults consistent with your headless runner
            train_years = int(getattr(getattr(self.app, "train_years_var", None), "get", lambda: 8)())
            test_months = int(getattr(getattr(self.app, "test_months_var", None), "get", lambda: 3)())
            test_start_str = getattr(getattr(self.app, "test_start_var", None), "get", lambda: "")().strip() or None
            ts = pd.to_datetime(test_start_str) if test_start_str else pd.to_datetime(df_clean["Date"].iloc[-1]) - pd.DateOffset(months=test_months)
            train_start = ts - pd.DateOffset(years=train_years)
            train_end = ts - pd.Timedelta(days=1)
            test_end = ts + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)

            # future_end: GUI input (if provided) else env FUTURE_END_DATE else test_end
            future_end_str = getattr(getattr(self.app, "future_end_var", None), "get", lambda: "")().strip()
            future_end_str = future_end_str or os.environ.get("FUTURE_END_DATE") or os.environ.get("FUTURE_END")
            future_end = pd.to_datetime(future_end_str) if future_end_str else test_end

            date_range = DateRange(train_start=train_start, train_end=train_end, test_start=ts, test_end=test_end)
            # attach future_end if possible
            try:
                setattr(date_range, "future_end", future_end)
            except Exception:
                pass
        except Exception:
            date_range = None  # StepAService can run without it in many versions

    # Prepare config for StepAService
    app_cfg = getattr(self.app, "app_config", None) or getattr(self.app, "config", None)
    if app_cfg is None:
        # minimal config shim expected by StepAService (__init__ extracts data_root/output_root)
        app_cfg = SimpleNamespace(data_root=Path("data"), output_root=self.app.output_root)

    # Run StepAService
    try:
        from ai_core.services.step_a_service import StepAService
        svc = StepAService(app_cfg)

        fn = getattr(svc, "run", None)
        if not callable(fn):
            raise RuntimeError("StepAService.run not found")

        # best-effort call (symbol, date_range)
        sig = inspect.signature(fn)
        kwargs = {}
        if "symbol" in sig.parameters:
            kwargs["symbol"] = sym
        elif "sym" in sig.parameters:
            kwargs["sym"] = sym

        if "date_range" in sig.parameters:
            kwargs["date_range"] = date_range

        fn(**kwargs)  # type: ignore[arg-type]
    except Exception as e:  # noqa: BLE001
        messagebox.showerror("Error", f"StepAService failed:\n{e}")
        return

    # Load generated display prices for plotting
    out_root: Path = self.app.output_root
    display_prices = out_root / "stepA" / "display" / "prices.csv"
    legacy_prices = out_root / f"stepA_prices_{sym}.csv"
    try:
        if display_prices.exists():
            df_out = _clean_prices(pd.read_csv(display_prices))
        elif legacy_prices.exists():
            df_out = _clean_prices(pd.read_csv(legacy_prices))
        else:
            df_out = df_clean
    except Exception:
        df_out = df_clean

    self.app.prices_df = df_out
    self.info_var.set(f"StepA done. Loaded prices rows={len(df_out)}")
    self.app.log(f"[StepA] Ran StepAService for {sym}. display={display_prices.exists()} legacy={legacy_prices.exists()}")

    self.plot_prices()

    def load_prices(self) -> None:
        sym = self.app.symbol_var.get().strip() or "SOXL"
        default = self.app.output_root / "stepA" / "display" / "prices.csv"
        if not default.exists():
            default = self.app.output_root / f"stepA_prices_{sym}.csv"
        initdir = str(default.parent) if default.exists() else "."

        p = filedialog.askopenfilename(
            title="Select StepA prices CSV",
            initialdir=initdir,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not p:
            return

        try:
            df = _clean_prices(pd.read_csv(p))
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Error", f"Failed to load prices CSV:\n{e}")
            return

        self.app.prices_df = df
        self.info_var.set(f"Loaded: {Path(p).name} (rows={len(df)})")
        self.app.log(f"[TabA] Loaded prices: {p}")

        self.plot_prices()

    # --------------------
    # グラフ描画
    # --------------------
    def plot_prices(self) -> None:
        df = self.app.prices_df
        if df is None:
            # 何もなければ何もしない（エラーにはしない）
            return

        data = df.copy()
        if "Date" not in data.columns:
            messagebox.showerror("Error", "Prices CSV must contain 'Date' column.")
            return

        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        data = data.dropna(subset=["Date"]).sort_values("Date")

        d_from = _parse_ymd(self.from_var.get())
        d_to = _parse_ymd(self.to_var.get())

        if d_from is not None:
            data = data[data["Date"] >= d_from]
        if d_to is not None:
            data = data[data["Date"] <= d_to]

        if data.empty:
            messagebox.showinfo("Info", "No rows in the selected date range.")
            return

        self.ax.clear()
        self.ax.plot(data["Date"], data["Close"], label="Close")

        self.ax.set_title("StepA: Close")
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Price")
        self.ax.grid(True)
        self.ax.legend()

        # Y 軸スケール
        try:
            self.ax.set_yscale(self.yscale_var.get())
        except ValueError:
            # log で 0 以下があるとエラーになるので、その場合は linear のまま
            self.ax.set_yscale("linear")
            self.yscale_var.set("linear")

        self.fig.autofmt_xdate()
        self.canvas.draw()

        self.app.log(
            f"[TabA] Plotted Close. "
            f"range=({d_from.date() if d_from else 'min'} ~ "
            f"{d_to.date() if d_to else 'max'}), "
            f"yscale={self.yscale_var.get()}"
        )
