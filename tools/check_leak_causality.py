#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/check_leak_causality.py

Purpose
-------
Given StepE daily logs (output/stepE/<mode>/stepE_daily_log_*_<SYMBOL>.csv) and StepA prices_test,
this script evaluates whether the profit accounting is consistent with a no-leak daily workflow.

Key idea
--------
Your intended (no-leak) rule:
  - You may use intraday (t_eff) information on day D to decide position pos(D),
  - but the profit MUST be realized on the next trading day:
      reward(D+1) = pos(D) * (P_eff(D+1)/P_eff(D) - 1) - cost - penalty
  - equivalently in "reward-day logging" form:
      reward(D) = pos_held_over(D-1->D) * (P_eff(D)/P_eff(D-1) - 1)

However, many pipelines *log* the held position and action on the reward-day row, which is fine.
This tool helps you infer what "Date" and "pos" mean in your logs by comparing multiple hypotheses.

What it does
------------
1) Loads StepA prices_test and builds:
     ret_back[t] = Close[t]/Close[t-1]-1  (aligned to Date[t])
     ret_fwd[t]  = Close[t+1]/Close[t]-1  (aligned to Date[t])
2) For each StepE daily log, tests 6 hypotheses:
     back/fwd x pos shift (t-1, t, t+1)
   and prints MAE/corr AND the final return for each hypothesis.
3) If Action column exists, prints:
     corr(Action[t], ret_back[t])  vs  corr(Action[t], ret_fwd[t])
   Large corr with ret_back can be a red flag, but is NOT a proof alone.
4) Searches for "prime/features" files under output/** excluding stepE/stepA
   and scans for suspicious column names:
     REALCLOSE, Close_true, target, label, future, y_, leak

Usage
-----
  python tools\\check_leak_causality.py --symbol SOXL --mode sim
  python tools\\check_leak_causality.py --symbol SOXL --mode sim --agent dprime_bnf_3scale

Notes
-----
- This tool cannot *prove* no-leak without also verifying what data was used to compute Action/pos.
  It provides strong evidence and points you to the exact file/row semantics you need to confirm.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd


# ---------- helpers ----------
def _find_all(patterns: List[str], recursive: bool = False) -> List[str]:
    out: List[str] = []
    for pat in patterns:
        out.extend(glob.glob(pat, recursive=recursive))
    return sorted(set(out))


def _detect_date_col(cols: List[str]) -> Optional[str]:
    for c in cols:
        if c.lower() in ("date", "datetime", "timestamp", "time"):
            return c
    for c in cols:
        if "date" in c.lower():
            return c
    return None


def _detect_close_col(cols: List[str]) -> Optional[str]:
    for key in ("close_eff", "closeeff", "p_eff", "peff"):
        for c in cols:
            if c.lower() == key:
                return c
    for c in cols:
        if c.lower() == "close":
            return c
    for c in cols:
        if "close" in c.lower():
            return c
    return None


def _detect_pos_col(cols: List[str]) -> Optional[str]:
    for c in cols:
        if c.lower() in ("pos", "position", "pos_ratio", "ratio", "weight", "w"):
            return c
    for c in cols:
        if "pos" in c.lower():
            return c
    return None


def _detect_action_col(cols: List[str]) -> Optional[str]:
    for c in cols:
        if c.lower() in ("action", "act", "a"):
            return c
    for c in cols:
        if "action" in c.lower():
            return c
    return None


def _detect_ret_col(cols: List[str]) -> Optional[str]:
    for c in cols:
        if c.lower() in ("portfolio_ret", "daily_ret", "ret", "reward"):
            return c
    for c in cols:
        cl = c.lower()
        if any(k in cl for k in ("portfolio_ret", "daily_ret", "ret", "reward", "pnl")):
            return c
    return None


def _corr(a: pd.Series, b: pd.Series) -> float:
    m = ~(a.isna() | b.isna())
    if int(m.sum()) < 10:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _equity_from_returns(r: pd.Series) -> Tuple[float, float]:
    """
    Returns:
      final_multiple (start=1)
      max_drawdown (negative value)
    """
    rr = r.fillna(0.0).astype(float)
    eq = (1.0 + rr).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(eq.iloc[-1]), float(dd.min())


@dataclass
class PricesRets:
    path: str
    close_col: str
    df: pd.DataFrame  # Date, ret_back, ret_fwd


def load_prices_test(output_root: str, mode: str, symbol: str) -> Optional[PricesRets]:
    pats = [
        os.path.join(output_root, "stepA", mode, f"*prices*test*{symbol}*.csv"),
        os.path.join(output_root, "stepA", mode, f"*prices_test*{symbol}*.csv"),
        os.path.join(output_root, "stepA", mode, f"*Prices*test*{symbol}*.csv"),
    ]
    hits = _find_all(pats)
    if not hits:
        return None
    p = hits[0]
    df = pd.read_csv(p, engine="python")
    dcol = _detect_date_col(list(df.columns))
    ccol = _detect_close_col(list(df.columns))
    if not (dcol and ccol):
        return None
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df = df.dropna(subset=[dcol]).sort_values(dcol).reset_index(drop=True)
    close = pd.to_numeric(df[ccol], errors="coerce")
    ret_back = close.pct_change()
    ret_fwd = close.shift(-1) / close - 1.0
    out = pd.DataFrame({"Date": df[dcol], "ret_back": ret_back, "ret_fwd": ret_fwd})
    return PricesRets(path=p, close_col=ccol, df=out)


def eval_hypotheses(merged: pd.DataFrame, pos_col: str, y_col: str) -> List[Dict[str, float]]:
    """
    merged must contain: Date, ret_back, ret_fwd, pos_col, y_col
    """
    y = pd.to_numeric(merged[y_col], errors="coerce")
    pos = pd.to_numeric(merged[pos_col], errors="coerce")
    rb = pd.to_numeric(merged["ret_back"], errors="coerce")
    rf = pd.to_numeric(merged["ret_fwd"], errors="coerce")

    hyps = [
        ("back: pos[t-1]*ret_back[t]", rb, +1),
        ("back: pos[t]*ret_back[t]",   rb,  0),
        ("back: pos[t+1]*ret_back[t]", rb, -1),
        ("fwd: pos[t-1]*ret_fwd[t]",   rf, +1),
        ("fwd: pos[t]*ret_fwd[t]",     rf,  0),
        ("fwd: pos[t+1]*ret_fwd[t]",   rf, -1),
    ]

    rows: List[Dict[str, float]] = []
    for name, r, sh in hyps:
        x = pos.shift(sh) * r
        m = ~(x.isna() | y.isna())
        n = int(m.sum())
        mae = float((y[m] - x[m]).abs().mean()) if n > 0 else float("inf")
        corr = float(np.corrcoef(y[m], x[m])[0, 1]) if n >= 2 else float("nan")
        final_mult, maxdd = _equity_from_returns(x)
        rows.append({
            "name": name,
            "mae": mae,
            "corr": corr,
            "n": n,
            "final_multiple": final_mult,
            "max_dd": maxdd,
        })
    rows.sort(key=lambda d: d["mae"])
    return rows


def scan_prime_feature_files(output_root: str, mode: str, symbol: str, agent: str = "", limit: int = 50) -> List[Tuple[str, List[str]]]:
    """
    Search output_root recursively for prime/features files EXCLUDING stepE and stepA.
    """
    bad_pat = re.compile(r"(realclose|close_true|future|label|target|y_|leak)", re.I)
    results: List[Tuple[str, List[str]]] = []

    for root, dirs, files in os.walk(output_root):
        # exclude stepE and stepA trees (we only want feature inputs)
        rel_root = os.path.relpath(root, output_root).replace("\\", "/")
        if rel_root.startswith("stepE/") or rel_root.startswith("stepA/"):
            continue

        for fn in files:
            if not fn.lower().endswith(".csv"):
                continue
            low = fn.lower()
            if symbol.lower() not in low:
                continue
            if "equity" in low or "daily_log" in low or "profit_summary" in low:
                continue
            if not any(k in low for k in ("prime", "dprime", "feature")):
                continue
            if agent and agent.lower() not in low:
                continue

            path = os.path.join(root, fn)
            try:
                df = pd.read_csv(path, nrows=5, engine="python")
                bad_cols = [c for c in df.columns if bad_pat.search(str(c))]
                results.append((os.path.relpath(path, output_root), bad_cols))
            except Exception as e:
                results.append((os.path.relpath(path, output_root), [f"__READ_ERROR__ {e}"]))

            if len(results) >= limit:
                return results

    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--mode", default="sim")
    ap.add_argument("--output-root", default="output")
    ap.add_argument("--agent", default="")
    args = ap.parse_args()

    symbol = args.symbol
    mode = args.mode
    out_root = args.output_root
    agent = args.agent.strip()

    print(f"[causality] symbol={symbol} mode={mode} output_root={out_root} agent_filter={agent or '(none)'}")

    prices = load_prices_test(out_root, mode, symbol)
    if prices is None:
        print("[fatal] StepA prices_test not found or missing Date/Close columns.")
        return
    print(f"[StepA] prices_test -> {prices.path} (close_col={prices.close_col}) rows={len(prices.df)}")

    log_pats = [
        os.path.join(out_root, "stepE", mode, f"stepE_daily_log_*{symbol}*.csv"),
        os.path.join(out_root, "stepE", mode, f"*daily*log*{symbol}*.csv"),
    ]
    logs = _find_all(log_pats)
    if agent:
        logs = [p for p in logs if agent in os.path.basename(p)]
    if not logs:
        print("[fatal] StepE daily logs not found.")
        return

    print(f"[StepE] logs: {len(logs)}")
    for p in logs:
        df = pd.read_csv(p, engine="python")
        cols = list(df.columns)
        dcol = _detect_date_col(cols)
        pcol = _detect_pos_col(cols)
        ycol = _detect_ret_col(cols)
        acol = _detect_action_col(cols)
        if not (dcol and pcol and ycol):
            print(f"  - {os.path.basename(p)} | skip (need Date+pos+ret). detected Date={dcol} pos={pcol} ret={ycol}")
            continue

        df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
        df = df.dropna(subset=[dcol]).sort_values(dcol)
        if "Split" in df.columns:
            df = df[df["Split"] == "test"].copy()

        df = df.rename(columns={dcol: "Date"})
        merged = df.merge(prices.df, on="Date", how="left")

        hyps = eval_hypotheses(merged, pcol, ycol)
        best = hyps[0]
        print(f"  - {os.path.basename(p)} | best={best['name']} MAE={best['mae']:.6g} corr={best['corr']:.4f} n={int(best['n'])} | final_multiple(best)={best['final_multiple']:.4f} maxDD(best)={best['max_dd']:.2%}")

        # show all hypotheses final_multiple quickly (top 6 already sorted by MAE)
        for h in hyps:
            print(f"      {h['name']:<26} | MAE={h['mae']:.6g} corr={h['corr']:.4f} final={h['final_multiple']:.4f} maxDD={h['max_dd']:.2%}")

        if acol:
            act = pd.to_numeric(merged[acol], errors="coerce")
            rb = pd.to_numeric(merged["ret_back"], errors="coerce")
            rf = pd.to_numeric(merged["ret_fwd"], errors="coerce")
            print(f"      corr(Action, ret_back)={_corr(act, rb):.4f} ; corr(Action, ret_fwd)={_corr(act, rf):.4f}")

    scans = scan_prime_feature_files(out_root, mode, symbol, agent=agent)
    if not scans:
        print("[inputs] prime/features scan: not found (searches output/** excluding stepE/stepA).")
        print("         If your prime/features are stored outside output/, run with --output-root <path>.")
    else:
        print("[inputs] suspicious column scan in prime/features CSV (REALCLOSE/Close_true/target/future/label/leak)")
        any_bad = False
        for rel, bad in scans:
            if bad:
                any_bad = True
                print(f"  - {rel} -> suspicious: {bad}")
        if not any_bad:
            print("  (none) suspicious column names detected in scanned prime/features inputs.")

    print("[done]")


if __name__ == "__main__":
    main()
