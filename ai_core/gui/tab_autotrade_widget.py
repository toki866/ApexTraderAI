from __future__ import annotations

import csv
import os
import sys
import threading
import subprocess
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import tkinter as tk
from tkinter import ttk, messagebox

try:
    import yaml
except ImportError:
    yaml = None  # config編集機能はyamlがあれば有効


# ============================
# 設定用データクラス
# ============================


@dataclass
class AutoTradePaths:
    """
    自動売買モニタ用のパス設定。

    ※ プロジェクト構成に合わせて必要なら後で調整してください。
    """
    app_root: Path = Path(".")                       # プロジェクトルート
    config_path: Path = Path("config.yaml")          # live_trading.enabled など
    daily_log_path: Path = Path("output/daily_log.csv")
    auto_trader_log_path: Path = Path("output/auto_trader.log")
    flags_path: Path = Path("output/auto_trader_flags.txt")
    emergency_stop_path: Path = Path("runtime/EMERGENCY_STOP")
    logs_dir: Path = Path("output")
    auto_trader_script: Path = Path("auto_trader.py")  # ローカルテストで実行するスクリプト


# ============================
# ユーティリティ
# ============================


def _safe_read_text(path: Path, default: str = "") -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return default
    except Exception:
        return default


def _parse_last_log_time(log_text: str) -> Optional[str]:
    """
    auto_trader.log 全文から、最後のログ行っぽいものを拾って日時を返す。
    ログフォーマットは環境依存なので、ここでは「最後の非空行」をそのまま返す。
    """
    if not log_text:
        return None
    lines = [ln.strip() for ln in log_text.splitlines() if ln.strip()]
    if not lines:
        return None
    return lines[-1]


def _parse_flags_file(flags_text: str) -> Dict[str, str]:
    """
    flagsファイルを "key=value" フォーマット想定で簡易パース。
    例:
        date=2025-11-27
        morning_done=1
        close_done=0
    """
    result: Dict[str, str] = {}
    for line in flags_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            result[k.strip()] = v.strip()
    return result


def _calc_summary_from_daily_log(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    daily_log.csv の DictRow 群から、合計PnL%、勝率、最大DDなどをざっくり計算。
    カラム名は想定：
      - date
      - pnl_pct
      - equity
      - action (BUY/SELL/HOLD)
      - mode (sim/paper/real) など
    ※ カラム存在しない場合は可能な範囲で計算。
    """
    if not rows:
        return {
            "num_trades": 0,
            "win_rate": None,
            "total_pnl_pct": None,
            "max_drawdown": None,
        }

    # PnL%
    pnl_list: List[float] = []
    # トレード判定用：action列がBUY or SELLならトレードとみなす
    wins = 0
    losses = 0

    # Equity曲線用
    equity_list: List[float] = []

    for r in rows:
        # pnl_pct
        if "pnl_pct" in r and r["pnl_pct"] not in ("", None):
            try:
                pnl = float(r["pnl_pct"])
                pnl_list.append(pnl)
                if r.get("action", "").upper() in ("BUY", "SELL"):
                    if pnl > 0:
                        wins += 1
                    elif pnl < 0:
                        losses += 1
            except ValueError:
                pass

        # equity
        if "equity" in r and r["equity"] not in ("", None):
            try:
                eq = float(r["equity"])
                equity_list.append(eq)
            except ValueError:
                pass

    num_trades = wins + losses
    win_rate = None
    if num_trades > 0:
        win_rate = wins / num_trades * 100.0

    total_pnl = sum(pnl_list) if pnl_list else None

    # 最大ドローダウン（かなり簡易な計算）
    max_dd = None
    if equity_list:
        peak = equity_list[0]
        dd_list: List[float] = []
        for eq in equity_list:
            if eq > peak:
                peak = eq
            dd = (eq - peak) / peak if peak != 0 else 0.0
            dd_list.append(dd)
        if dd_list:
            max_dd = min(dd_list) * 100.0  # ％表示に変換

    return {
        "num_trades": num_trades,
        "win_rate": win_rate,
        "total_pnl_pct": total_pnl,
        "max_drawdown": max_dd,
    }


# ============================
# メインクラス：TabAutoTradeWidget
# ============================


class TabAutoTradeWidget(ttk.Frame):
    """
    自動売買モニタ用タブ。

    主な機能:
        - システム選択（将来複数システム対応を想定）
        - 現在モード表示 (sim / ibkr_paper / ibkr_real など)
        - 本番ON/OFF（config.yaml の live_trading.enabled 操作）
        - 緊急停止 (EMERGENCY_STOPファイル) のON/OFF
        - ローカルSimテスト（日付範囲指定で auto_trader.py をサブプロセス実行）
        - auto_trader.log / flags / daily_log.csv を読み込んでステータス表示
        - daily_log のテーブル表示＋簡易サマリ
    """

    def __init__(
        self,
        master: tk.Misc,
        paths: Optional[AutoTradePaths] = None,
        systems: Optional[List[str]] = None,
        logger: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(master, **kwargs)

        self.paths = paths or AutoTradePaths()
        self.logger = logger

        # 将来複数システム想定。とりあえず1つだけ使う場合は ["SOXL_RL_v1"] など。
        self.systems: List[str] = systems or ["SOXL_RL_v1"]
        self.current_system_var = tk.StringVar(value=self.systems[0])

        # 本番ON/OFFの状態（config.yaml を読むことで初期化）
        self.live_enabled_var = tk.BooleanVar(value=False)
        # 緊急停止
        self.emergency_active_var = tk.BooleanVar(value=False)

        # モード表示（configの broker.mode 想定）
        self.broker_mode_var = tk.StringVar(value="N/A")

        # auto_trader.log 最終行表示
        self.last_run_var = tk.StringVar(value="N/A")

        # flags（朝/引けフロー）
        self.morning_status_var = tk.StringVar(value="N/A")
        self.close_status_var = tk.StringVar(value="N/A")

        # サマリ表示
        self.summary_trades_var = tk.StringVar(value="-")
        self.summary_win_rate_var = tk.StringVar(value="-")
        self.summary_pnl_var = tk.StringVar(value="-")
        self.summary_dd_var = tk.StringVar(value="-")

        # ローカルテスト用日付
        today_str = date.today().isoformat()
        self.local_start_date_var = tk.StringVar(value=today_str)
        self.local_end_date_var = tk.StringVar(value=today_str)

        # UI構築
        self._build_ui()

        # 初期状態のロード
        self.reload_all_status()

    # ------------------------------
    # UI構築
    # ------------------------------
    def _build_ui(self) -> None:
        # 全体は縦3分割：上=ヘッダ、中=操作、下=結果（ログテーブル＋サマリ）
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        # ========= 上段：システム・モード・本番ON/OFF =========
        header_frame = ttk.LabelFrame(self, text="システム / モード / 本番ON-OFF")
        header_frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=4)

        # システム選択
        ttk.Label(header_frame, text="システム:").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        system_combo = ttk.Combobox(
            header_frame,
            textvariable=self.current_system_var,
            values=self.systems,
            state="readonly",
            width=20,
        )
        system_combo.grid(row=0, column=1, sticky="w", padx=4, pady=2)
        system_combo.bind("<<ComboboxSelected>>", self._on_system_changed)

        # ブローカーモード表示
        ttk.Label(header_frame, text="モード:").grid(row=0, column=2, sticky="w", padx=8, pady=2)
        ttk.Label(header_frame, textvariable=self.broker_mode_var).grid(
            row=0, column=3, sticky="w", padx=4, pady=2
        )

        # 本番ON/OFF
        live_chk = ttk.Checkbutton(
            header_frame,
            text="本番トレード ON/OFF (config.yaml)",
            variable=self.live_enabled_var,
            command=self.on_toggle_live_enabled,
        )
        live_chk.grid(row=1, column=0, columnspan=2, sticky="w", padx=4, pady=2)

        # 緊急停止
        emergency_chk = ttk.Checkbutton(
            header_frame,
            text="緊急停止 (EMERGENCY_STOP)",
            variable=self.emergency_active_var,
            command=self.on_toggle_emergency_stop,
        )
        emergency_chk.grid(row=1, column=2, columnspan=2, sticky="w", padx=4, pady=2)

        # 最終実行ログ
        ttk.Label(header_frame, text="最終 auto_trader 実行ログ:").grid(
            row=2, column=0, sticky="w", padx=4, pady=2
        )
        ttk.Label(header_frame, textvariable=self.last_run_var).grid(
            row=2, column=1, columnspan=3, sticky="w", padx=4, pady=2
        )

        # 朝/引けフロー
        ttk.Label(header_frame, text="朝フロー:").grid(row=3, column=0, sticky="w", padx=4, pady=2)
        ttk.Label(header_frame, textvariable=self.morning_status_var).grid(
            row=3, column=1, sticky="w", padx=4, pady=2
        )
        ttk.Label(header_frame, text="引けフロー:").grid(row=3, column=2, sticky="w", padx=4, pady=2)
        ttk.Label(header_frame, textvariable=self.close_status_var).grid(
            row=3, column=3, sticky="w", padx=4, pady=2
        )

        for col in range(4):
            header_frame.columnconfigure(col, weight=1)

        # ========= 中段：ローカルテスト＆操作ボタン =========
        control_frame = ttk.LabelFrame(self, text="ローカルテスト / 操作")
        control_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)

        # ローカルテスト日付入力
        ttk.Label(control_frame, text="ローカルSimテスト日付範囲:").grid(
            row=0, column=0, sticky="w", padx=4, pady=2
        )

        ttk.Label(control_frame, text="開始日:").grid(row=1, column=0, sticky="e", padx=4, pady=2)
        start_entry = ttk.Entry(control_frame, textvariable=self.local_start_date_var, width=12)
        start_entry.grid(row=1, column=1, sticky="w", padx=4, pady=2)

        ttk.Label(control_frame, text="終了日:").grid(row=1, column=2, sticky="e", padx=4, pady=2)
        end_entry = ttk.Entry(control_frame, textvariable=self.local_end_date_var, width=12)
        end_entry.grid(row=1, column=3, sticky="w", padx=4, pady=2)

        # ローカルテスト実行ボタン
        run_local_btn = ttk.Button(
            control_frame,
            text="ローカルSimテスト実行 (auto_trader.py)",
            command=self.on_click_run_local_test,
        )
        run_local_btn.grid(row=2, column=0, columnspan=2, sticky="w", padx=4, pady=4)

        # リロードボタン
        reload_btn = ttk.Button(
            control_frame,
            text="ステータス再読込",
            command=self.reload_all_status,
        )
        reload_btn.grid(row=2, column=2, sticky="e", padx=4, pady=4)

        # ログフォルダを開くボタン
        open_logs_btn = ttk.Button(
            control_frame,
            text="ログフォルダを開く",
            command=self.on_open_logs_dir,
        )
        open_logs_btn.grid(row=2, column=3, sticky="e", padx=4, pady=4)

        for col in range(4):
            control_frame.columnconfigure(col, weight=1)

        # ========= 下段：日次ログテーブル＋サマリ =========
        bottom_frame = ttk.Frame(self)
        bottom_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=4)
        bottom_frame.columnconfigure(0, weight=3)
        bottom_frame.columnconfigure(1, weight=1)
        bottom_frame.rowconfigure(0, weight=1)

        # 左：Treeview
        log_frame = ttk.LabelFrame(bottom_frame, text="日次ログ (daily_log.csv)")
        log_frame.grid(row=0, column=0, sticky="nsew", padx=4, pady=2)
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        columns = (
            "date",
            "symbol",
            "mode",
            "action",
            "qty",
            "open_price",
            "close_price",
            "pnl_pct",
            "equity",
        )
        self.log_tree = ttk.Treeview(
            log_frame,
            columns=columns,
            show="headings",
            height=12,
        )
        for col in columns:
            self.log_tree.heading(col, text=col)
            self.log_tree.column(col, width=80, anchor="center")

        vsb = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_tree.yview)
        self.log_tree.configure(yscrollcommand=vsb.set)

        self.log_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        # 右：サマリ
        summary_frame = ttk.LabelFrame(bottom_frame, text="サマリ / メトリクス")
        summary_frame.grid(row=0, column=1, sticky="nsew", padx=4, pady=2)

        ttk.Label(summary_frame, text="トレード数:").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        ttk.Label(summary_frame, textvariable=self.summary_trades_var).grid(
            row=0, column=1, sticky="w", padx=4, pady=2
        )

        ttk.Label(summary_frame, text="勝率(%):").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        ttk.Label(summary_frame, textvariable=self.summary_win_rate_var).grid(
            row=1, column=1, sticky="w", padx=4, pady=2
        )

        ttk.Label(summary_frame, text="合計PnL(%):").grid(row=2, column=0, sticky="w", padx=4, pady=2)
        ttk.Label(summary_frame, textvariable=self.summary_pnl_var).grid(
            row=2, column=1, sticky="w", padx=4, pady=2
        )

        ttk.Label(summary_frame, text="最大DD(%):").grid(row=3, column=0, sticky="w", padx=4, pady=2)
        ttk.Label(summary_frame, textvariable=self.summary_dd_var).grid(
            row=3, column=1, sticky="w", padx=4, pady=2
        )

        for col in range(2):
            summary_frame.columnconfigure(col, weight=1)

    # ------------------------------
    # イベントハンドラ
    # ------------------------------
    def _on_system_changed(self, event: Any) -> None:
        # 将来、system別のログ/設定に分かれる想定。
        # 現時点では単にステータス再読込するだけ。
        self._log(f"System changed to: {self.current_system_var.get()}")
        self.reload_all_status()

    def on_toggle_live_enabled(self) -> None:
        """
        本番ON/OFFチェックボックスが変更されたときに呼ばれる。
        config.yaml の live_trading.enabled を書き換える前提。
        """
        enabled = self.live_enabled_var.get()
        if yaml is None:
            messagebox.showerror("Error", "PyYAML がインポートできません。'pip install pyyaml' を実行してください。")
            # チェック状態を元に戻しておく
            self.live_enabled_var.set(not enabled)
            return

        cfg_path = self.paths.config_path
        try:
            if cfg_path.exists():
                data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            else:
                data = {}

            # ネストは例: live_trading: { enabled: bool, broker_mode: "ibkr_real" } を想定
            live_cfg = data.get("live_trading", {})
            live_cfg["enabled"] = bool(enabled)
            data["live_trading"] = live_cfg

            cfg_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
            self._log(f"Updated live_trading.enabled = {enabled} in {cfg_path}")

            if enabled:
                messagebox.showinfo("本番ON", "live_trading.enabled を True にしました。\n本番モードに注意してください。")
            else:
                messagebox.showinfo("本番OFF", "live_trading.enabled を False にしました。")
        except Exception as e:
            messagebox.showerror("Error", f"config.yaml の更新に失敗しました:\n{e}")
            # 失敗した場合は状態を元に戻す
            self.live_enabled_var.set(not enabled)
            return

        # 状態再読込
        self.reload_config_status()

    def on_toggle_emergency_stop(self) -> None:
        """
        緊急停止のON/OFF。EMERGENCY_STOPファイルを作成/削除する。
        """
        active = self.emergency_active_var.get()
        path = self.paths.emergency_stop_path
        try:
            if active:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("EMERGENCY STOP\n", encoding="utf-8")
                self._log(f"Created EMERGENCY_STOP at {path}")
                messagebox.showwarning("緊急停止ON", "EMERGENCY_STOP ファイルを作成しました。自動売買は停止状態になります。")
            else:
                if path.exists():
                    path.unlink()
                    self._log(f"Removed EMERGENCY_STOP at {path}")
                messagebox.showinfo("緊急停止OFF", "EMERGENCY_STOP を解除しました。")
        except Exception as e:
            messagebox.showerror("Error", f"EMERGENCY_STOP の切り替えに失敗しました:\n{e}")
            # 失敗した場合は状態を元に戻す
            self.emergency_active_var.set(not active)
            return

    def on_click_run_local_test(self) -> None:
        """
        ローカルSimテスト実行ボタン。
        auto_trader.py をサブプロセスで呼び出す前提。
        """
        start_str = self.local_start_date_var.get().strip()
        end_str = self.local_end_date_var.get().strip()

        try:
            start_date = datetime.strptime(start_str, "%Y-%m-%d").date()
            end_date = datetime.strptime(end_str, "%Y-%m-%d").date()
        except ValueError:
            messagebox.showerror("日付エラー", "開始日・終了日は YYYY-MM-DD 形式で入力してください。")
            return

        if end_date < start_date:
            messagebox.showerror("日付エラー", "終了日は開始日以降の日付にしてください。")
            return

        # 確認
        if not messagebox.askyesno(
            "ローカルSimテスト",
            f"{start_date}〜{end_date} の範囲で\nauto_trader.py (Simモード) を実行しますか？",
        ):
            return

        # 別スレッドで実行（UIをブロックしない）
        t = threading.Thread(
            target=self._run_local_test_subprocess,
            args=(start_date, end_date),
            daemon=True,
        )
        t.start()

    def on_open_logs_dir(self) -> None:
        """
        ログフォルダをエクスプローラ等で開く。
        """
        log_dir = self.paths.logs_dir
        if not log_dir.exists():
            messagebox.showerror("Error", f"ログフォルダが存在しません: {log_dir}")
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(log_dir))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(log_dir)])
            else:
                subprocess.Popen(["xdg-open", str(log_dir)])
        except Exception as e:
            messagebox.showerror("Error", f"ログフォルダを開けませんでした:\n{e}")

    # ------------------------------
    # 内部処理：ローカルテスト実行
    # ------------------------------
    def _run_local_test_subprocess(self, start_date: date, end_date: date) -> None:
        """
        auto_trader.py をサブプロセスで実行する。
        実行完了後にステータス再読込を行う。
        """
        script = self.paths.auto_trader_script
        if not script.exists():
            self._log(f"auto_trader.py not found: {script}")
            self._call_in_main_thread(
                lambda: messagebox.showerror("Error", f"auto_trader.py が見つかりません:\n{script}")
            )
            return

        cmd = [
            sys.executable,
            str(script),
            "--mode",
            "sim",
            "--start-date",
            start_date.isoformat(),
            "--end-date",
            end_date.isoformat(),
        ]

        self._log(f"Running local sim test: {' '.join(cmd)}")

        def run_and_notify() -> None:
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(self.paths.app_root),
                    capture_output=True,
                    text=True,
                )
                if proc.returncode != 0:
                    msg = f"auto_trader.py 実行中にエラーが発生しました。\n\nreturncode={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
                    self._log(msg)
                    self._call_in_main_thread(lambda: messagebox.showerror("Error", msg))
                else:
                    self._log("auto_trader.py (Sim) 実行完了")
                    self._call_in_main_thread(
                        lambda: messagebox.showinfo(
                            "ローカルSimテスト完了",
                            "auto_trader.py (Sim) の実行が完了しました。\nログと日次結果を更新します。",
                        )
                    )
                    # ステータス再読込
                    self._call_in_main_thread(self.reload_all_status)
            except Exception as e:
                msg = f"auto_trader.py 実行に失敗しました:\n{e}"
                self._log(msg)
                self._call_in_main_thread(lambda: messagebox.showerror("Error", msg))

        # 既にスレッド内なので、そのまま実行
        run_and_notify()

    def _call_in_main_thread(self, func) -> None:
        """
        別スレッドからUI更新を安全に行うためのヘルパ。
        """
        self.after(0, func)

    # ------------------------------
    # ステータス再読込
    # ------------------------------
    def reload_all_status(self) -> None:
        """
        config / EMERGENCY_STOP / flags / auto_trader.log / daily_log をまとめて再読込。
        """
        self.reload_config_status()
        self.reload_emergency_status()
        self.reload_flags_status()
        self.reload_last_run_status()
        self.reload_daily_log()

    def reload_config_status(self) -> None:
        """
        config.yaml から live_trading.enabled と broker.mode を読んで反映。
        """
        if yaml is None:
            # PyYAMLがない場合は読み取りも諦める
            self.broker_mode_var.set("N/A (pyyamlなし)")
            return

        cfg_path = self.paths.config_path
        if not cfg_path.exists():
            self.broker_mode_var.set("N/A (configなし)")
            self.live_enabled_var.set(False)
            return

        try:
            data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except Exception as e:
            self._log(f"Failed to read config.yaml: {e}")
            self.broker_mode_var.set("読み込みエラー")
            return

        live_cfg = data.get("live_trading", {})
        enabled = bool(live_cfg.get("enabled", False))
        self.live_enabled_var.set(enabled)

        broker_mode = data.get("broker", {}).get("mode") or live_cfg.get("broker_mode") or "N/A"
        self.broker_mode_var.set(str(broker_mode))

    def reload_emergency_status(self) -> None:
        """
        EMERGENCY_STOP ファイルの有無で状態更新。
        """
        path = self.paths.emergency_stop_path
        self.emergency_active_var.set(path.exists())

    def reload_flags_status(self) -> None:
        """
        auto_trader_flags.txt を読み、朝/引けフロー状態を更新。
        """
        text = _safe_read_text(self.paths.flags_path)
        if not text:
            self.morning_status_var.set("N/A")
            self.close_status_var.set("N/A")
            return

        flags = _parse_flags_file(text)
        today = date.today().isoformat()
        flag_date = flags.get("date", "")

        def _fmt_status(key: str) -> str:
            v = flags.get(key, "")
            if not v:
                return "未実行"
            if v in ("1", "true", "True"):
                return "実行済"
            return v

        if flag_date == today:
            self.morning_status_var.set(_fmt_status("morning_done"))
            self.close_status_var.set(_fmt_status("close_done"))
        else:
            self.morning_status_var.set(f"別日 ({flag_date})")
            self.close_status_var.set(f"別日 ({flag_date})")

    def reload_last_run_status(self) -> None:
        """
        auto_trader.log の最終行を last_run_var に表示。
        """
        text = _safe_read_text(self.paths.auto_trader_log_path)
        last = _parse_last_log_time(text)
        self.last_run_var.set(last or "ログなし")

    def reload_daily_log(self) -> None:
        """
        daily_log.csv を読み込んで Treeview に反映し、サマリを計算。
        """
        path = self.paths.daily_log_path
        rows: List[Dict[str, Any]] = []

        if not path.exists():
            # テーブルクリアだけして終わり
            for i in self.log_tree.get_children():
                self.log_tree.delete(i)
            self.summary_trades_var.set("-")
            self.summary_win_rate_var.set("-")
            self.summary_pnl_var.set("-")
            self.summary_dd_var.set("-")
            return

        try:
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows.append(r)
        except Exception as e:
            self._log(f"Failed to read daily_log: {e}")
            return

        # 一旦全部削除
        for i in self.log_tree.get_children():
            self.log_tree.delete(i)

        # 新しい行を追加（最新が下に来るよう、そのまま順に）
        for r in rows[-500:]:  # 最新500件だけ表示
            values = [
                r.get("date", ""),
                r.get("symbol", ""),
                r.get("mode", ""),
                r.get("action", ""),
                r.get("qty", ""),
                r.get("open_price", ""),
                r.get("close_price", ""),
                r.get("pnl_pct", ""),
                r.get("equity", ""),
            ]
            self.log_tree.insert("", "end", values=values)

        # サマリ計算
        summary = _calc_summary_from_daily_log(rows)
        self.summary_trades_var.set(str(summary.get("num_trades", "-")))

        win_rate = summary.get("win_rate")
        self.summary_win_rate_var.set(f"{win_rate:.2f}" if isinstance(win_rate, float) else "-")

        total_pnl = summary.get("total_pnl_pct")
        self.summary_pnl_var.set(f"{total_pnl:.2f}" if isinstance(total_pnl, (float, int)) else "-")

        max_dd = summary.get("max_drawdown")
        self.summary_dd_var.set(f"{max_dd:.2f}" if isinstance(max_dd, (float, int)) else "-")

    # ------------------------------
    # ログ出力
    # ------------------------------
    def _log(self, msg: str) -> None:
        if self.logger is not None:
            try:
                self.logger.info(f"[TabAutoTrade] {msg}")
                return
            except Exception:
                pass
        # fallback
        print(f"[TabAutoTrade] {msg}")
