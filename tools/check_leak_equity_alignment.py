#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/check_leak_equity_alignment.py

Stronger leak/alignment check using StepE *equity* as the ground truth.

Why:
  - daily_log 'ret' may not be exactly the same return used to update equity.
  - profit summaries are derived from stepE_equity_*.csv.
  - So we should align equity-derived returns to price returns and positions.

This script:
  1) Loads StepA prices_test to compute:
        ret_back[t] = Close[t]/Close[t-1]-1  (aligned to Date[t])
        ret_fwd[t]  = Close[t+1]/Close[t]-1  (aligned to Date[t])
  2) For each agent:
        - loads output/stepE/<mode>/stepE_equity_<agent>_<SYMBOL>.csv
        - loads output/stepE/<mode>/stepE_daily_log_<agent>_<SYMBOL>.csv
        - merges on Date
        - computes eq_ret[t] = equity[t]/equity[t-1]-1 (aligned to Date[t])
        - tests 6 hypotheses:
              back/fwd × pos shift (t-1,t,t+1)
          by MAE/corr against eq_ret.
  3) Prints action-vs-pos shift hint and basic sanity checks.

Interpretation guide:
  - If best is back: pos[t]*ret_back[t]
        then the equity updates behave like "reward-day logging":
            eq_ret at Date[t] corresponds to price move (t-1 -> t)
            multiplied by the position recorded on the same row.
        This is NOT automatically a leak if that position was decided at t-1 (t_eff).
        But if you intended Date to mean decision day, it's a red flag.
  - If best is fwd: pos[t]*ret_fwd[t]
        this matches your intended "decision-day logging" style:
            decide at day t, profit realized t -> t+1

Usage:
  python tools\\check_leak_equity_alignment.py --symbol SOXL --mode sim
  python tools\\check_leak_equity_alignment.py --symbol SOXL --mode sim --agent dprime_bnf_3scale
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def _find_all(patterns: List[str]) -> List[str]:
    out: List[str] = []
    for pat in patterns:
        out.extend(glob.glob(pat))
    return sorted(set(out))


def _find_one(patterns: List[str]) -> Optional[str]:
    hits = _find_all(patterns)
    return hits[0] if hits else None


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


def _detect_equity_col(cols: List[str]) -> Optional[str]:
    for c in cols:
        if c.lower() in ("equity", "capital", "nav", "portfolio_value", "value"):
            return c
    for c in cols:
        if "equity" in c.lower() or "capital" in c.lower() or "nav" in c.lower():
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


def _corr(a: pd.Series, b: pd.Series) -> float:
    m = ~(a.isna() | b.isna())
    if int(m.sum()) < 10:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


@dataclass
class PricesRets:
    path: str
    close_col: str
    df: pd.DataFrame  # Date, ret_back, ret_fwd


def load_prices_test(output_root: str, mode: str, symbol: str) -> Optional[PricesRets]:
    patterns = [
        os.path.join(output_root, "stepA", mode, f"*prices*test*{symbol}*.csv"),
        os.path.join(output_root, "stepA", mode, f"*prices_test*{symbol}*.csv"),
    ]
    p = _find_one(patterns)
    if not p:
        return None
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


def score(y: pd.Series, pos: pd.Series, r: pd.Series, shift: int, name: str) -> Tuple[str, float, float, int]:
    x = pos.shift(shift) * r
    m = ~(x.isna() | y.isna())
    n = int(m.sum())
    if n < 10:
        return (name, float("inf"), float("nan"), n)
    mae = float((y[m] - x[m]).abs().mean())
    corr = float(np.corrcoef(y[m], x[m])[0, 1]) if n >= 2 else float("nan")
    return (name, mae, corr, n)


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
    agent_filter = args.agent.strip()

    print(f"[equity-align] symbol={symbol} mode={mode} output_root={out_root} agent_filter={agent_filter or '(none)'}")

    prices = load_prices_test(out_root, mode, symbol)
    if prices is None:
        print("[fatal] StepA prices_test not found.")
        return
    print(f"[StepA] prices_test -> {prices.path} (close_col={prices.close_col}) rows={len(prices.df)}")

    equity_files = _find_all([
        os.path.join(out_root, "stepE", mode, f"stepE_equity_*_{symbol}.csv"),
        os.path.join(out_root, "stepE", mode, f"*equity*{symbol}*.csv"),
    ])
    log_files = _find_all([
        os.path.join(out_root, "stepE", mode, f"stepE_daily_log_*_{symbol}.csv"),
        os.path.join(out_root, "stepE", mode, f"*daily*log*{symbol}*.csv"),
    ])

    if agent_filter:
        equity_files = [p for p in equity_files if agent_filter in os.path.basename(p)]
        log_files = [p for p in log_files if agent_filter in os.path.basename(p)]

    if not equity_files:
        print("[fatal] stepE equity files not found.")
        return
    if not log_files:
        print("[fatal] stepE daily logs not found.")
        return

    # Map agent key by filename substring between 'equity_' and _SYMBOL
    def agent_key_from_equity(fn: str) -> str:
        base = os.path.basename(fn)
        m = base.split(f"_{symbol}.csv")[0]
        # stepE_equity_<agent>_<SYMBOL>.csv
        if m.startswith("stepE_equity_"):
            return m[len("stepE_equity_"):].rsplit("_", 1)[0]
        return m

    def agent_key_from_log(fn: str) -> str:
        base = os.path.basename(fn)
        m = base.split(f"_{symbol}.csv")[0]
        # stepE_daily_log_<agent>_<SYMBOL>.csv
        if m.startswith("stepE_daily_log_"):
            return m[len("stepE_daily_log_"):].rsplit("_", 1)[0]
        return m

    eq_map = {agent_key_from_equity(p): p for p in equity_files}
    log_map = {agent_key_from_log(p): p for p in log_files}

    agents = sorted(set(eq_map.keys()) & set(log_map.keys()))
    print(f"[StepE] matched agents: {len(agents)}")

    for ag in agents:
        eq_path = eq_map[ag]
        lg_path = log_map[ag]

        eq = pd.read_csv(eq_path, engine="python")
        lg = pd.read_csv(lg_path, engine="python")

        eq_dcol = _detect_date_col(list(eq.columns))
        eq_ecol = _detect_equity_col(list(eq.columns))
        lg_dcol = _detect_date_col(list(lg.columns))
        lg_pcol = _detect_pos_col(list(lg.columns))
        lg_acol = _detect_action_col(list(lg.columns))

        if not (eq_dcol and eq_ecol and lg_dcol and lg_pcol):
            print(f"  - {ag} | skip (need Date/equity and Date/pos)")
            continue

        eq[eq_dcol] = pd.to_datetime(eq[eq_dcol], errors="coerce")
        lg[lg_dcol] = pd.to_datetime(lg[lg_dcol], errors="coerce")
        eq = eq.dropna(subset=[eq_dcol]).sort_values(eq_dcol).rename(columns={eq_dcol: "Date"})
        lg = lg.dropna(subset=[lg_dcol]).sort_values(lg_dcol).rename(columns={lg_dcol: "Date"})
        if "Split" in lg.columns:
            lg = lg[lg["Split"] == "test"].copy()

        eq_val = pd.to_numeric(eq[eq_ecol], errors="coerce")
        eq_ret = eq_val.pct_change()
        eq2 = pd.DataFrame({"Date": eq["Date"], "eq_ret": eq_ret, "equity": eq_val})

        merged = lg.merge(eq2, on="Date", how="inner").merge(prices.df, on="Date", how="left")
        y = pd.to_numeric(merged["eq_ret"], errors="coerce")
        pos = pd.to_numeric(merged[lg_pcol], errors="coerce")
        rb = pd.to_numeric(merged["ret_back"], errors="coerce")
        rf = pd.to_numeric(merged["ret_fwd"], errors="coerce")

        tests = [
            score(y, pos, rb, +1, "back: pos[t-1]*ret_back[t]"),
            score(y, pos, rb,  0, "back: pos[t]*ret_back[t]"),
            score(y, pos, rb, -1, "back: pos[t+1]*ret_back[t]"),
            score(y, pos, rf, +1, "fwd: pos[t-1]*ret_fwd[t]"),
            score(y, pos, rf,  0, "fwd: pos[t]*ret_fwd[t]"),
            score(y, pos, rf, -1, "fwd: pos[t+1]*ret_fwd[t]"),
        ]
        tests_sorted = sorted(tests, key=lambda t: t[1])
        best = tests_sorted[0]

        # equity multiple sanity
        eq_multiple = float(eq_val.dropna().iloc[-1] / eq_val.dropna().iloc[0]) if eq_val.dropna().shape[0] >= 2 else float("nan")

        print(f"  - {ag} | equity_multiple={eq_multiple:.4f} | best={best[0]} (MAE={best[1]:.6g}, corr={best[2]:.4f}, n={best[3]}) | pos_col={lg_pcol} action_col={lg_acol or '(none)'}")
        # print next 2
        for nxt in tests_sorted[1:3]:
            print(f"      next: {nxt[0]} (MAE={nxt[1]:.6g}, corr={nxt[2]:.4f}, n={nxt[3]})")

        if lg_acol:
            act = pd.to_numeric(merged[lg_acol], errors="coerce")
            print(f"      hint: corr(action[t],pos[t])={_corr(act,pos):.4f} ; corr(action[t],pos[t+1])={_corr(act,pos.shift(-1)):.4f}")

    print("[done]")


if __name__ == "__main__":
    main()
