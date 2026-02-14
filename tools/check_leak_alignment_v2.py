#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/check_leak_alignment_v2.py

Leak / time-shift diagnostics for StepE results (output-only).

It disambiguates BOTH of these common logging conventions:
  (A) Date = reward date (end of interval), ret = Close[t]/Close[t-1]-1
      -> should align with the position that was held over (t-1 -> t).
  (B) Date = decision date (start of interval), ret = Close[t+1]/Close[t]-1
      -> should align with the position decided at t (held over t -> t+1).

This script tests 6 hypotheses:
  - return type: back (t-1 -> t)  OR  fwd (t -> t+1)
  - position shift: pos[t-1], pos[t], pos[t+1]

It prints which hypothesis best explains your logged ret/reward (lowest MAE),
plus correlation and MAE for the top candidates.

Also scans StepD/prime feature CSVs for suspicious column names:
  REALCLOSE, Close_true, target, label, future, y_, leak ...

Usage (Windows CMD at repo root):
  python tools\check_leak_alignment_v2.py --symbol SOXL --mode sim
  python tools\check_leak_alignment_v2.py --symbol SOXL --mode sim --agent dprime_bnf_3scale
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass
from typing import List, Optional

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


def _detect_pos_col(cols: List[str]) -> Optional[str]:
    for c in cols:
        if c.lower() in ("pos", "position", "pos_ratio", "ratio", "weight", "w"):
            return c
    for c in cols:
        if "pos" in c.lower():
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


def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, engine="python")


@dataclass
class PricesRets:
    path: str
    close_col: str
    df: pd.DataFrame  # Date, ret_back, ret_fwd


def load_prices_test(output_root: str, mode: str, symbol: str) -> Optional[PricesRets]:
    patterns = [
        os.path.join(output_root, "stepA", mode, f"*prices*test*{symbol}*.csv"),
        os.path.join(output_root, "stepA", mode, f"*prices_test*{symbol}*.csv"),
        os.path.join(output_root, "stepA", mode, f"*Prices*test*{symbol}*.csv"),
    ]
    p = _find_one(patterns)
    if not p:
        return None
    df = _read_csv(p)
    dcol = _detect_date_col(list(df.columns))
    if not dcol:
        return None
    close_col = _detect_close_col(list(df.columns))
    if not close_col:
        return None

    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df = df.dropna(subset=[dcol]).sort_values(dcol).reset_index(drop=True)

    close = pd.to_numeric(df[close_col], errors="coerce")
    ret_back = close.pct_change()               # close[t]/close[t-1]-1 aligned to Date[t]
    ret_fwd = close.shift(-1) / close - 1.0     # close[t+1]/close[t]-1 aligned to Date[t]

    out = pd.DataFrame({"Date": df[dcol], "ret_back": ret_back, "ret_fwd": ret_fwd})
    return PricesRets(path=p, close_col=close_col, df=out)


@dataclass
class HypScore:
    name: str
    mae: float
    corr: float
    n: int


def _score(y: pd.Series, pos: pd.Series, r: pd.Series, pos_shift: int, label: str) -> HypScore:
    x = pos.shift(pos_shift) * r
    m = ~(x.isna() | y.isna())
    n = int(m.sum())
    if n < 10:
        return HypScore(name=label, mae=float("inf"), corr=float("nan"), n=n)
    mae = float((y[m] - x[m]).abs().mean())
    corr = float(np.corrcoef(y[m], x[m])[0, 1]) if n >= 2 else float("nan")
    return HypScore(name=label, mae=mae, corr=corr, n=n)


def analyze_log(log_csv: str, prices: PricesRets, split_filter: str = "test"):
    df = _read_csv(log_csv)
    cols = list(df.columns)
    dcol = _detect_date_col(cols)
    pcol = _detect_pos_col(cols)
    ycol = _detect_ret_col(cols)
    if not (dcol and pcol and ycol):
        return None

    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df = df.dropna(subset=[dcol]).sort_values(dcol)
    if split_filter is not None and "Split" in df.columns:
        df = df[df["Split"] == split_filter].copy()

    df = df.rename(columns={dcol: "Date"})
    merged = df.merge(prices.df, on="Date", how="left")

    y = pd.to_numeric(merged[ycol], errors="coerce")
    pos = pd.to_numeric(merged[pcol], errors="coerce")
    r_back = pd.to_numeric(merged["ret_back"], errors="coerce")
    r_fwd = pd.to_numeric(merged["ret_fwd"], errors="coerce")

    scores = []
    scores.append(_score(y, pos, r_back, +1, "back: pos[t-1]*ret_back[t]"))
    scores.append(_score(y, pos, r_back,  0, "back: pos[t]*ret_back[t]"))
    scores.append(_score(y, pos, r_back, -1, "back: pos[t+1]*ret_back[t]"))
    scores.append(_score(y, pos, r_fwd, +1, "fwd: pos[t-1]*ret_fwd[t]"))
    scores.append(_score(y, pos, r_fwd,  0, "fwd: pos[t]*ret_fwd[t]"))
    scores.append(_score(y, pos, r_fwd, -1, "fwd: pos[t+1]*ret_fwd[t]"))

    return (os.path.basename(log_csv), pcol, ycol, scores)


def scan_prime_features(output_root: str, mode: str, symbol: str, limit: int = 200):
    patterns = [
        os.path.join(output_root, "stepD", mode, f"*{symbol}*.csv"),
        os.path.join(output_root, "stepD", f"*{mode}*", f"*{symbol}*.csv"),
        os.path.join(output_root, "stepD_prime", mode, f"*{symbol}*.csv"),
        os.path.join(output_root, "stepDPrime", mode, f"*{symbol}*.csv"),
        os.path.join(output_root, "stepD*", mode, f"*{symbol}*.csv"),
        os.path.join(output_root, "**", mode, f"*prime*{symbol}*.csv"),
        os.path.join(output_root, "**", mode, f"*dprime*{symbol}*.csv"),
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))
    files = sorted(set(files))

    pref = [f for f in files if re.search(r"(prime|dprime|feature)", os.path.basename(f), re.I)]
    files = pref if pref else files

    pat = re.compile(r"(realclose|close_true|future|label|target|y_|leak)", re.I)
    out = []
    for f in files[:limit]:
        try:
            df = _read_csv(f, nrows=5)
            bad = [str(c) for c in df.columns if pat.search(str(c))]
            out.append((os.path.basename(f), bad))
        except Exception as e:
            out.append((os.path.basename(f), [f"__READ_ERROR__ {e}"]))
    return out


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

    print(f"[check-v2] symbol={symbol} mode={mode} output_root={out_root} agent_filter={agent or '(none)'}")

    prices = load_prices_test(out_root, mode, symbol)
    if prices is None:
        print("[warn] StepA prices_test not found or missing Date/Close columns.")
        print("       expected: output/stepA/<mode>/*prices*test*<SYMBOL>*.csv")
        return
    print(f"[StepA] prices_test -> {prices.path} (close_col={prices.close_col}) rows={len(prices.df)}")

    log_files = _find_all([
        os.path.join(out_root, "stepE", mode, f"stepE_daily_log_*{symbol}*.csv"),
        os.path.join(out_root, "stepE", mode, f"*daily*log*{symbol}*.csv"),
        os.path.join(out_root, "stepE", mode, f"*trades*{symbol}*.csv"),
        os.path.join(out_root, "stepE", mode, f"*actions*{symbol}*.csv"),
    ])
    if agent:
        log_files = [p for p in log_files if agent in os.path.basename(p)]
    log_files = sorted(log_files)

    if not log_files:
        print("[warn] StepE daily/trade/action log not found.")
    else:
        print(f"[StepE] logs: {len(log_files)} (testing 6 hypotheses: back/fwd x pos shifts)")
        for f in log_files[:50]:
            res = analyze_log(f, prices, split_filter="test")
            if res is None:
                df = _read_csv(f)
                cols = list(df.columns)
                print(f"  - {os.path.basename(f)} | skip (need Date+pos+ret/reward). detected: Date={_detect_date_col(cols)} pos={_detect_pos_col(cols)} ret={_detect_ret_col(cols)}")
                continue
            fname, pcol, ycol, scores = res
            best = min(scores, key=lambda s: s.mae)
            print(f"  - {fname} | best={best.name} (MAE={best.mae:.6g}, corr={best.corr:.4f}, n={best.n}) | cols: pos={pcol} y={ycol}")
            top3 = sorted(scores, key=lambda s: s.mae)[:3]
            for s in top3[1:]:
                print(f"      next: {s.name} (MAE={s.mae:.6g}, corr={s.corr:.4f}, n={s.n})")

    scans = scan_prime_features(out_root, mode, symbol)
    if not scans:
        print("[warn] prime/features CSV not found by extended patterns.")
    else:
        print("[StepD] suspicious column scan (REALCLOSE/Close_true/target/future/label/leak)")
        any_bad = False
        for name, bad in scans:
            if bad:
                any_bad = True
                print(f"  - {name} -> suspicious: {bad}")
        if not any_bad:
            print("  (none) suspicious column names detected in scanned files.")

    print("[done]")


if __name__ == "__main__":
    main()
