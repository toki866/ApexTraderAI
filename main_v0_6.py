import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from pathlib import Path
from datetime import datetime, date, timedelta

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import RangeSlider
import matplotlib.dates as mdates

# ai_core 側サービス
from ai_core.config.app_config import AppConfig, DataConfig
from ai_core.types.common import DateRange
from ai_core.services.step_b_service import StepBService, StepBConfig, StepBResult
from ai_core.services.step_c_service import StepCService
from ai_core.services.step_d_service import StepDService

# 分割したタブクラス
from ai_core.gui.tab_a_stepa import TabA
from ai_core.gui.tab_b_stepb import TabB
from ai_core.gui.tab_c_stepc import TabC
from ai_core.gui.tab_d_stepd import TabD
from ai_core.gui.tab_e_stepe import TabE
from ai_core.gui.tab_f_stepf import TabF


# ======================
# ヘルパ関数
# ======================

def parse_ymd(s: str):
    s = (s or "").strip()
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None


def in_range_mask(dt_series: pd.Series, d_from: date | None, d_to: date | None):
    m = pd.Series(True, index=dt_series.index)
    if d_from:
        m &= (dt_series.dt.date >= d_from)
    if d_to:
        m &= (dt_series.dt.date <= d_to)
    return m


def to_num_dates(series: pd.Series) -> pd.Series:
    """datetime64 -> matplotlib float days"""
    if not pd.api.types.is_datetime64_any_dtype(series):
        series = pd.to_datetime(series)
    return mdates.date2num(series.dt.to_pydatetime())


# ======================
# MainApp
# ======================

class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SOXL RL GUI (Tkinter v0.6)")
        self.geometry("1200x820")

        self.output_root = Path("output")
        self.symbol_var = tk.StringVar(value="SOXL")
        self.config_path_var = tk.StringVar(value="")
        self.train_end_var = tk.StringVar(value="2024-12-31")

        self.prices_df: pd.DataFrame | None = None
        self.pred_df: pd.DataFrame | None = None
        self.daily_df: pd.DataFrame | None = None

        self._build()

    # --------------------
    # AppConfig / DateRange
    # --------------------
    def build_app_config(self) -> AppConfig:
        """
        GUIの設定から AppConfig を構成。
        - YAML があれば load_from_yaml
        - 失敗したら data/output 直書き
        さらに互換レイヤとして cfg.data / cfg.paths の両方を必ず用意する。
        """
        cfg_path = (self.config_path_var.get() or "").strip()
        cfg: AppConfig | None = None

        # 1) YAML
        if cfg_path:
            try:
                cfg = AppConfig.load_from_yaml(cfg_path)
                self.log(f"[Config] Loaded YAML: {cfg_path}")
            except Exception as e:  # noqa: BLE001
                messagebox.showwarning(
                    "Config",
                    f"Failed to load YAML config ({cfg_path}).\n"
                    f"Exception: {e}\n\n"
                    "Fallback to default data/output paths.",
                )
                cfg = None

        # 2) fallback
        if cfg is None:
            data_root = Path("data")
            output_root = self.output_root
            symbols = ["SOXL", "SOXS", "SPY"]
            cfg = AppConfig(
                data=DataConfig(
                    data_root=data_root,
                    output_root=output_root,
                    symbols=symbols,
                )
            )

        # 3) data_root / output_root / symbols を抽出
        data_root: Path | None = None
        output_root: Path | None = None
        symbols: list[str] | None = None

        # data セクション優先
        if hasattr(cfg, "data") and getattr(cfg, "data") is not None:
            d = cfg.data
            data_root = Path(getattr(d, "data_root", data_root))
            output_root = Path(getattr(d, "output_root", output_root))
            symbols = list(getattr(d, "symbols", symbols or []))

        # paths セクション（旧仕様互換）
        if hasattr(cfg, "paths") and getattr(cfg, "paths") is not None:
            p = cfg.paths
            data_root = Path(getattr(p, "data_root", data_root))
            output_root = Path(getattr(p, "output_root", output_root))

        # デフォルト補完
        if data_root is None:
            data_root = Path("data")
        if output_root is None:
            output_root = self.output_root
        if symbols is None:
            symbols = ["SOXL", "SOXS", "SPY"]

        # 4) cfg.data が無ければ追加
        if not hasattr(cfg, "data") or cfg.data is None:
            cfg.data = DataConfig(
                data_root=data_root,
                output_root=output_root,
                symbols=symbols,
            )

        # 5) cfg.paths が無ければ追加（StepB/StepC/StepD 互換用）
        if not hasattr(cfg, "paths") or getattr(cfg, "paths") is None:

            class _Paths:
                def __init__(self, data_root: Path, output_root: Path):
                    self.data_root = Path(data_root)
                    self.output_root = Path(output_root)

            cfg.paths = _Paths(data_root, output_root)

        return cfg

    def build_date_range(self) -> DateRange | None:
        """prices_df + train_end_var から DateRange を構成"""
        df = self.prices_df
        if df is None or "Date" not in df.columns:
            return None

        d = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(d["Date"]):
            d["Date"] = pd.to_datetime(d["Date"])
        d = d.sort_values("Date")

        start = d["Date"].dt.date.min()
        end = d["Date"].dt.date.max()

        train_end_str = (self.train_end_var.get() or "").strip()
        te = parse_ymd(train_end_str)
        if te is None:
            te = end

        train_start = start
        train_end = min(te, end)
        test_start = train_end + timedelta(days=1)
        test_end = end

        return DateRange(
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )

    # --------------------
    # GUI build
    # --------------------
    def _build(self):
        # ---- header ----
        header = ttk.Frame(self)
        header.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        ttk.Label(header, text="Config YAML:").pack(side=tk.LEFT, padx=(0, 2))
        ttk.Entry(header, textvariable=self.config_path_var, width=40).pack(
            side=tk.LEFT, padx=(0, 2)
        )
        ttk.Button(header, text="Browse...", command=self._browse_config).pack(
            side=tk.LEFT, padx=2
        )

        ttk.Label(header, text="Symbol:").pack(side=tk.LEFT, padx=(18, 2))
        ttk.Combobox(
            header,
            textvariable=self.symbol_var,
            values=["SOXL", "SOXS", "SPY"],
            width=8,
        ).pack(side=tk.LEFT)

        ttk.Label(header, text="Train end (YYYY-MM-DD):").pack(
            side=tk.LEFT, padx=(18, 2)
        )
        ttk.Entry(header, textvariable=self.train_end_var, width=12).pack(
            side=tk.LEFT
        )

        ttk.Button(
            header,
            text="Run A→F",
            command=lambda: self.log("Run A→F clicked (stub)"),
        ).pack(side=tk.RIGHT, padx=5)

        # ---- notebook (tabs) ----
        self.nb = ttk.Notebook(self)
        self.nb.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.tab_a = TabA(self.nb, self)
        self.nb.add(self.tab_a, text="A: Features / Prices")

        self.tab_b = TabB(self.nb, self)
        self.nb.add(self.tab_b, text="B: Train/Test + StepB")

        self.tab_c = TabC(self.nb, self)
        self.nb.add(self.tab_c, text="C: Long-term Chart")

        self.tab_d = TabD(self.nb, self)
        self.nb.add(self.tab_d, text="D: Envelope")

        self.tab_e = TabE(self.nb, self)
        self.nb.add(self.tab_e, text="E: RL Single (Markers)")

        self.tab_f = TabF(self.nb, self)
        self.nb.add(self.tab_f, text="F: MARL")

        # ---- bottom (log + status) ----
        bottom = ttk.Frame(self)
        bottom.pack(side=tk.BOTTOM, fill=tk.X)
        self.log_text = ScrolledText(bottom, height=6, state="disabled")
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=3)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(bottom, textvariable=self.status_var, width=30).pack(
            side=tk.RIGHT, padx=5, pady=3
        )

    # --------------------
    # misc
    # --------------------
    def _browse_config(self):
        p = filedialog.askopenfilename(
            title="Select config YAML",
            filetypes=[("YAML files", "*.yml *.yaml"), ("All files", "*.*")],
        )
        if p:
            self.config_path_var.set(p)
            self.log(f"Config selected: {p}")

    def log(self, msg: str):
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")

    def set_status(self, msg: str):
        self.status_var.set(msg)


# ======================
# Tab A (StepA)
# ======================

# Tabs (TabA–TabF) are implemented in gui/tab_*.py


# ======================
# entry
# ======================

if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
