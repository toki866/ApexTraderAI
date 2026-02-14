# -*- coding: utf-8 -*-
"""
Profit summary utility for SOXL_RL_GUI.

Reads:
- output/stepE/<mode>/stepE_daily_log_*_<SYMBOL>.csv
- output/stepF/<mode>/stepF_equity_marl_<SYMBOL>.csv (optional)
- output/stepA/<mode>/stepA_prices_test_<SYMBOL>.csv (optional)

Computes test-split performance (default) and writes:
- output/stepE/<mode>/stepE_profit_summary_<SYMBOL>.csv

Usage (Windows CMD):
  cd /d C:\path\to\soxl_rl_gui
  python tools\check_profit_summary.py --symbol SOXL --mode sim
  python tools\check_profit_summary.py --symbol SOXL --mode sim --initial-capital 1000000

Notes:
- Profit is reported as return% (scale-free). If you want currency,
  set --initial-capital (e.g., JPY) and it will compute end capital.
"""
from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Metrics:
    n: int
    start: float
    end: float
    profit: float
    return_pct: float
    max_drawdown: float
    sharpe: float


def _safe_to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _compute_metrics_from_equity(df: pd.DataFrame, equity_col: str = "equity", date_col: str = "Date") -> Optional[Metrics]:
    d = df.copy()
    if date_col in d.columns:
        d[date_col] = _safe_to_datetime(d[date_col])
        d = d.sort_values(date_col)
    if equity_col not in d.columns:
        return None
    eq = pd.to_numeric(d[equity_col], errors="coerce").to_numpy(dtype=float)
    eq = eq[np.isfinite(eq)]
    if eq.size < 2:
        return None

    start = float(eq[0])
    end = float(eq[-1])
    profit = end - start
    return_pct = (end / start - 1.0) if start != 0 else float("nan")

    daily = np.diff(eq) / eq[:-1]
    daily = daily[np.isfinite(daily)]
    if daily.size >= 2 and float(np.std(daily, ddof=1)) > 0:
        sharpe = float(np.mean(daily) / np.std(daily, ddof=1) * np.sqrt(252))
    else:
        sharpe = float("nan")

    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    max_dd = float(np.nanmin(dd)) if dd.size > 0 else float("nan")

    return Metrics(
        n=int(eq.size),
        start=start,
        end=end,
        profit=profit,
        return_pct=float(return_pct),
        max_drawdown=max_dd,
        sharpe=sharpe,
    )


def _compute_metrics_from_daily_log(df: pd.DataFrame, split: str = "test") -> Optional[Metrics]:
    d = df.copy()
    if "Date" in d.columns:
        d["Date"] = _safe_to_datetime(d["Date"])
        d = d.sort_values("Date")

    if split != "all" and "Split" in d.columns:
        d = d[d["Split"] == split]

    return _compute_metrics_from_equity(d, equity_col="equity", date_col="Date")


def _fmt_pct(x: float) -> str:
    if x is None or not np.isfinite(x):
        return ""
    return f"{x*100:.1f}%"


def _fmt_num(x: float) -> str:
    if x is None or not np.isfinite(x):
        return ""
    return f"{x:.2f}"


def _find_stepE_daily_logs(output_root: str, mode: str, symbol: str) -> List[str]:
    d = os.path.join(output_root, "stepE", mode)
    if not os.path.isdir(d):
        return []
    files = []
    for fn in os.listdir(d):
        if fn.startswith("stepE_daily_log_") and fn.endswith(f"_{symbol}.csv"):
            files.append(os.path.join(d, fn))
    files.sort()
    return files


def _agent_from_daily_log_path(p: str, symbol: str) -> str:
    fn = os.path.basename(p)
    # stepE_daily_log_<agent>_<SYMBOL>.csv
    m = re.match(rf"stepE_daily_log_(.+)_{re.escape(symbol)}\.csv$", fn)
    return m.group(1) if m else fn


def _find_stepF_equity(output_root: str, mode: str, symbol: str) -> Optional[str]:
    p = os.path.join(output_root, "stepF", mode, f"stepF_equity_marl_{symbol}.csv")
    return p if os.path.isfile(p) else None


def _find_stepA_prices_test(output_root: str, mode: str, symbol: str) -> Optional[str]:
    p = os.path.join(output_root, "stepA", mode, f"stepA_prices_test_{symbol}.csv")
    return p if os.path.isfile(p) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--mode", required=True, choices=["sim", "live", "ops", "display"])
    ap.add_argument("--output-root", default="output")
    ap.add_argument("--split", default="test", choices=["train", "test", "all"])
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--initial-capital", type=float, default=1.0, help="Capital for currency conversion (e.g., 1000000 for JPY).")
    args = ap.parse_args()

    symbol = args.symbol
    mode = args.mode
    output_root = args.output_root

    # StepA baseline
    pA = _find_stepA_prices_test(output_root, mode, symbol)
    if pA:
        dfA = pd.read_csv(pA)
        if "Close" in dfA.columns and len(dfA) >= 2:
            soxl_ret = float(dfA["Close"].iloc[-1] / dfA["Close"].iloc[0] - 1.0)
            print(f"[Baseline] {symbol} buy&hold (test): {_fmt_pct(soxl_ret)}  (days={len(dfA)})  ({dfA['Date'].iloc[0]} -> {dfA['Date'].iloc[-1]})")
        else:
            print(f"[Baseline] found but unexpected columns: {pA}")
    else:
        print("[Baseline] StepA prices_test not found (skip).")

    # StepE agents
    logs = _find_stepE_daily_logs(output_root, mode, symbol)
    if not logs:
        print("[StepE] daily logs not found. Did you run StepE?")
    else:
        rows = []
        date_range = None
        for p in logs:
            agent = _agent_from_daily_log_path(p, symbol)
            df = pd.read_csv(p)
            met = _compute_metrics_from_daily_log(df, split=args.split)
            if met is None:
                continue
            # capture test date range if possible
            if date_range is None and "Date" in df.columns:
                dd = df.copy()
                dd["Date"] = _safe_to_datetime(dd["Date"])
                if args.split != "all" and "Split" in dd.columns:
                    dd = dd[dd["Split"] == args.split]
                dd = dd.dropna(subset=["Date"])
                if len(dd) >= 2:
                    date_range = (str(dd["Date"].iloc[0].date()), str(dd["Date"].iloc[-1].date()), len(dd))
            rows.append({
                "agent": agent,
                f"{args.split}_days": met.n,
                f"{args.split}_return_pct": met.return_pct,
                f"{args.split}_max_dd": met.max_drawdown,
                f"{args.split}_sharpe": met.sharpe,
                "end_capital_from_initial": args.initial_capital * (1.0 + met.return_pct) if np.isfinite(met.return_pct) else np.nan,
                "profit_from_initial": args.initial_capital * met.return_pct if np.isfinite(met.return_pct) else np.nan,
            })

        if rows:
            out_df = pd.DataFrame(rows).sort_values(f"{args.split}_return_pct", ascending=False)
            out_path = os.path.join(output_root, "stepE", mode, f"stepE_profit_summary_{symbol}.csv")
            out_df.to_csv(out_path, index=False, encoding="utf-8")
            print(f"[StepE] wrote -> {out_path}")

            if date_range:
                print(f"[StepE] split={args.split} date_range={date_range[0]} -> {date_range[1]} (days={date_range[2]})")

            print("\n[StepE] Top results:")
            show = out_df.head(args.topk).copy()
            show[f"{args.split}_return_pct"] = show[f"{args.split}_return_pct"].map(_fmt_pct)
            show[f"{args.split}_max_dd"] = show[f"{args.split}_max_dd"].map(_fmt_pct)
            show[f"{args.split}_sharpe"] = show[f"{args.split}_sharpe"].map(_fmt_num)
            # pretty print without wide truncation
            with pd.option_context("display.max_rows", 200, "display.max_columns", 200, "display.width", 160):
                print(show[["agent", f"{args.split}_days", f"{args.split}_return_pct", f"{args.split}_max_dd", f"{args.split}_sharpe", "end_capital_from_initial", "profit_from_initial"]])
        else:
            print("[StepE] metrics could not be computed from logs (no usable equity series).")

    # StepF MARL equity
    pF = _find_stepF_equity(output_root, mode, symbol)
    if pF:
        dfF = pd.read_csv(pF)
        metF = _compute_metrics_from_equity(dfF, equity_col="equity", date_col="Date")
        if metF:
            end_cap = args.initial_capital * (1.0 + metF.return_pct) if np.isfinite(metF.return_pct) else np.nan
            profit = args.initial_capital * metF.return_pct if np.isfinite(metF.return_pct) else np.nan
            print("\n[StepF] MARL gate:")
            print(f"  days={metF.n} return={_fmt_pct(metF.return_pct)} maxDD={_fmt_pct(metF.max_drawdown)} sharpe={_fmt_num(metF.sharpe)}")
            print(f"  initial_capital={args.initial_capital:.2f} -> end={end_cap:.2f} (profit={profit:.2f})")
        else:
            print("[StepF] found but metrics could not be computed:", pF)
    else:
        print("[StepF] equity not found (skip).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
