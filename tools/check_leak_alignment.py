#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/check_leak_alignment.py

Leak / time-shift diagnostics for StepE results (output-only).

It helps answer:
  "Is the huge return (e.g., +900%) explained without leakage?"
  Specifically, it checks whether your daily reward/return is aligned to:
    reward(D) computed from pos(D-1) * price_ret(D-1->D)  (NO same-day profit)
  versus:
    reward(D) computed from pos(D)   * price_ret(D-1->D)  (POTENTIAL same-day profit/time-shift)
  or even:
    reward(D) computed from pos(D+1) * price_ret(D-1->D)  (future; highly suspicious)

What it does:
  1) Equity sanity: daily return max/min + extreme counts (test split if present).
  2) Alignment test (if StepE daily log exists with pos and ret/reward):
       compares the 3 hypotheses above (MAE + correlation).
  3) Prime feature scan (StepD): flags suspicious column names that suggest realized/future labels
       (REALCLOSE, Close_true, target, label, y_, future, leak).
  4) Optional "prefix invariance" helper: compare same-date rows between two CSVs and report diffs.

Usage (Windows CMD at repo root):
  python tools\check_leak_alignment.py --symbol SOXL --mode sim
  python tools\check_leak_alignment.py --symbol SOXL --mode sim --agent dprime_bnf_3scale
  python tools\check_leak_alignment.py --symbol SOXL --mode sim --output-root output
  python tools\check_leak_alignment.py --compare-csv A.csv --compare-csv2 B.csv --compare-on Date --head 5

Notes:
  - If you don't have a StepE daily/trade log that includes pos + ret/reward,
    this script cannot *prove* no-leak from outputs alone. It will tell you what's missing.
  - For your "t_eff = close-10min" design, the gold-standard proof is:
      reward uses pos_prev and P_eff(D-1)->P_eff(D).
    If your logs include P_eff_prev/P_eff and pos_prev, this script will directly verify it.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

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


def _detect_equity_col(cols: List[str]) -> Optional[str]:
    prefs = ("equity", "capital", "portfolio_value", "value", "nav")
    for c in cols:
        cl = c.lower()
        if cl in prefs or any(cl.startswith(p) for p in prefs):
            return c
    return None


def _detect_close_col(cols: List[str]) -> Optional[str]:
    # Prefer Close_eff if exists, then Close
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
        if c.lower() in ("pos_prev", "posprev", "position_prev", "prev_pos"):
            return c  # special: pos_prev exists; handled separately
    for c in cols:
        if c.lower() in ("pos", "position", "pos_ratio", "ratio", "weight", "w"):
            return c
    for c in cols:
        if "pos" in c.lower():
            return c
    return None


def _detect_pos_prev_col(cols: List[str]) -> Optional[str]:
    for c in cols:
        if c.lower() in ("pos_prev", "posprev", "position_prev", "prev_pos"):
            return c
    return None


def _detect_ret_or_reward_col(cols: List[str]) -> Optional[str]:
    for c in cols:
        if c.lower() in ("portfolio_ret", "daily_ret", "ret", "reward"):
            return c
    for c in cols:
        cl = c.lower()
        if any(k in cl for k in ("portfolio_ret", "daily_ret", "ret", "reward", "pnl")):
            return c
    return None


def _detect_price_prev_now(cols: List[str]) -> Tuple[Optional[str], Optional[str]]:
    prev_keys = ("price_prev", "p_prev", "p_eff_prev", "peff_prev", "prev_price")
    now_keys  = ("price_now", "p_now", "p_eff", "peff", "price", "p")
    prev = None
    now = None
    for c in cols:
        if c.lower() in prev_keys:
            prev = c
            break
    for c in cols:
        if c.lower() in now_keys:
            now = c
            break
    return prev, now


def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, engine="python")


@dataclass
class EquitySummary:
    path: str
    rows: int
    equity_col: str
    ret_max: float
    ret_min: float
    gt30: int
    gt50: int
    ltm30: int


def summarize_equity(path: str, split_filter: str = "test") -> EquitySummary:
    df = _read_csv(path)
    cols = list(df.columns)
    dcol = _detect_date_col(cols)
    if not dcol:
        raise ValueError(f"[equity] Date列が見つかりません: {path}")
    ecol = _detect_equity_col(cols) or cols[-1]
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df = df.dropna(subset=[dcol]).sort_values(dcol)
    if split_filter is not None and "Split" in df.columns:
        df = df[df["Split"] == split_filter].copy()
    eq = pd.to_numeric(df[ecol], errors="coerce")
    r = eq.pct_change().dropna()
    if len(r) == 0:
        ret_max = float("nan")
        ret_min = float("nan")
        gt30 = gt50 = ltm30 = 0
    else:
        ret_max = float(r.max())
        ret_min = float(r.min())
        gt30 = int((r > 0.3).sum())
        gt50 = int((r > 0.5).sum())
        ltm30 = int((r < -0.3).sum())
    return EquitySummary(path=path, rows=int(len(df)), equity_col=str(ecol),
                        ret_max=ret_max, ret_min=ret_min, gt30=gt30, gt50=gt50, ltm30=ltm30)


@dataclass
class PricesRet:
    path: str
    close_col: str
    df: pd.DataFrame  # Date, raw_ret


def load_prices_test(output_root: str, mode: str, symbol: str) -> Optional[PricesRet]:
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
    df = df.dropna(subset=[dcol]).sort_values(dcol)
    close = pd.to_numeric(df[close_col], errors="coerce")
    out = pd.DataFrame({"Date": df[dcol], "raw_ret": close.pct_change()})
    return PricesRet(path=p, close_col=close_col, df=out)


@dataclass
class AlignResult:
    file: str
    best: str
    best_mae: float
    best_corr: float
    n: int
    mae_pos_tminus1: float
    corr_pos_tminus1: float
    mae_pos_t: float
    corr_pos_t: float
    mae_pos_tplus1: float
    corr_pos_tplus1: float
    used_ret_col: str
    used_pos_col: str
    used_date_col: str
    used_split: str


def _score_alignment(y: pd.Series, pos: pd.Series, price_ret: pd.Series, pos_shift: int) -> Tuple[float, float, int]:
    x = pos.shift(pos_shift) * price_ret
    m = ~(x.isna() | y.isna())
    n = int(m.sum())
    if n < 10:
        return (float("inf"), float("nan"), n)
    mae = float((y[m] - x[m]).abs().mean())
    corr = float(np.corrcoef(y[m], x[m])[0, 1]) if n >= 2 else float("nan")
    return (mae, corr, n)


def alignment_test(log_csv: str, prices_ret: PricesRet, split_filter: str = "test") -> Optional[AlignResult]:
    df = _read_csv(log_csv)
    cols = list(df.columns)
    dcol = _detect_date_col(cols)
    pos_col = _detect_pos_col(cols)
    ret_col = _detect_ret_or_reward_col(cols)
    if not (dcol and pos_col and ret_col):
        return None

    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df = df.dropna(subset=[dcol]).sort_values(dcol)
    used_split = split_filter
    if split_filter is not None and "Split" in df.columns:
        df = df[df["Split"] == split_filter].copy()
    df = df.rename(columns={dcol: "Date"})
    merged = df.merge(prices_ret.df, on="Date", how="left")

    y = pd.to_numeric(merged[ret_col], errors="coerce")
    pos = pd.to_numeric(merged[pos_col], errors="coerce")
    price_ret = pd.to_numeric(merged["raw_ret"], errors="coerce")

    maeA, corrA, nA = _score_alignment(y, pos, price_ret, pos_shift=+1)  # pos[t-1]
    maeB, corrB, nB = _score_alignment(y, pos, price_ret, pos_shift=0)   # pos[t]
    maeC, corrC, nC = _score_alignment(y, pos, price_ret, pos_shift=-1)  # pos[t+1]

    choices = [("pos[t-1]", maeA, corrA, nA), ("pos[t]", maeB, corrB, nB), ("pos[t+1]", maeC, corrC, nC)]
    best = min(choices, key=lambda it: it[1])

    return AlignResult(
        file=os.path.basename(log_csv),
        best=best[0],
        best_mae=float(best[1]),
        best_corr=float(best[2]) if not np.isnan(best[2]) else float("nan"),
        n=int(best[3]),
        mae_pos_tminus1=float(maeA),
        corr_pos_tminus1=float(corrA) if not np.isnan(corrA) else float("nan"),
        mae_pos_t=float(maeB),
        corr_pos_t=float(corrB) if not np.isnan(corrB) else float("nan"),
        mae_pos_tplus1=float(maeC),
        corr_pos_tplus1=float(corrC) if not np.isnan(corrC) else float("nan"),
        used_ret_col=ret_col,
        used_pos_col=pos_col,
        used_date_col=dcol,
        used_split=used_split,
    )


@dataclass
class RewardAudit:
    file: str
    n: int
    mae_vs_pos_prev: float
    mae_vs_pos_now: float
    best: str
    used_cols: Dict[str, str]


def reward_audit_if_possible(log_csv: str, split_filter: str = "test") -> Optional[RewardAudit]:
    df = _read_csv(log_csv)
    cols = list(df.columns)
    dcol = _detect_date_col(cols)
    pos_col = _detect_pos_col(cols)
    pos_prev_col = _detect_pos_prev_col(cols)
    ret_col = _detect_ret_or_reward_col(cols)
    price_prev_col, price_now_col = _detect_price_prev_now(cols)

    if not (dcol and pos_col and ret_col and price_prev_col and price_now_col):
        return None

    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df = df.dropna(subset=[dcol]).sort_values(dcol)
    if split_filter is not None and "Split" in df.columns:
        df = df[df["Split"] == split_filter].copy()

    reward = pd.to_numeric(df[ret_col], errors="coerce")
    pos_now = pd.to_numeric(df[pos_col], errors="coerce")
    price_prev = pd.to_numeric(df[price_prev_col], errors="coerce")
    price_now = pd.to_numeric(df[price_now_col], errors="coerce")
    price_ret = (price_now / price_prev) - 1.0

    if pos_prev_col:
        pos_prev = pd.to_numeric(df[pos_prev_col], errors="coerce")
        pos_prev_used = pos_prev_col
    else:
        pos_prev = pos_now.shift(1)
        pos_prev_used = "(derived: shift(pos))"

    exp_prev = pos_prev * price_ret
    exp_now = pos_now * price_ret

    m_prev = ~(exp_prev.isna() | reward.isna())
    m_now = ~(exp_now.isna() | reward.isna())
    n = int(min(m_prev.sum(), m_now.sum()))
    if n < 10:
        return None

    mae_prev = float((reward[m_prev] - exp_prev[m_prev]).abs().mean())
    mae_now = float((reward[m_now] - exp_now[m_now]).abs().mean())
    best = "pos_prev" if mae_prev < mae_now else "pos_now"

    return RewardAudit(
        file=os.path.basename(log_csv),
        n=n,
        mae_vs_pos_prev=mae_prev,
        mae_vs_pos_now=mae_now,
        best=best,
        used_cols={
            "Date": dcol,
            "pos": pos_col,
            "pos_prev": pos_prev_used,
            "reward/ret": ret_col,
            "price_prev": price_prev_col,
            "price_now": price_now_col,
        },
    )


def scan_prime_features(output_root: str, mode: str, symbol: str, limit: int = 50) -> List[Tuple[str, List[str]]]:
    patterns = [
        os.path.join(output_root, "stepD", mode, f"*{symbol}*.csv"),
        os.path.join(output_root, "stepD", mode, f"*prime*{symbol}*.csv"),
        os.path.join(output_root, "stepD", mode, f"*dprime*{symbol}*.csv"),
    ]
    files = _find_all(patterns)
    files2 = [f for f in files if re.search(r"(prime|feature|dprime)", os.path.basename(f), re.I)]
    files = files2 if files2 else files

    suspicious = []
    pat = re.compile(r"(realclose|close_true|future|label|target|y_|leak)", re.I)
    for f in files[:limit]:
        try:
            df = _read_csv(f, nrows=5)
            cols = list(df.columns)
            bad = [c for c in cols if pat.search(c)]
            suspicious.append((os.path.basename(f), bad))
        except Exception as e:
            suspicious.append((os.path.basename(f), [f"__READ_ERROR__ {e}"]))
    return suspicious


def compare_csv(csv1: str, csv2: str, on: str, head: int = 10) -> None:
    a = _read_csv(csv1)
    b = _read_csv(csv2)
    if on not in a.columns or on not in b.columns:
        raise ValueError(f"compare-on '{on}' must exist in both CSVs.")
    a[on] = pd.to_datetime(a[on], errors="coerce")
    b[on] = pd.to_datetime(b[on], errors="coerce")
    a = a.dropna(subset=[on]).sort_values(on)
    b = b.dropna(subset=[on]).sort_values(on)

    common = pd.merge(a, b, on=on, how="inner", suffixes=("_A", "_B"))
    if len(common) == 0:
        print("[compare] no common rows on", on)
        return

    diffs = []
    for col in a.columns:
        if col == on:
            continue
        colA = col + "_A"
        colB = col + "_B"
        if colA in common.columns and colB in common.columns:
            x = pd.to_numeric(common[colA], errors="coerce")
            y = pd.to_numeric(common[colB], errors="coerce")
            m = ~(x.isna() | y.isna())
            if m.sum() == 0:
                continue
            max_abs = float((x[m] - y[m]).abs().max())
            mean_abs = float((x[m] - y[m]).abs().mean())
            if max_abs > 0:
                diffs.append((col, mean_abs, max_abs))

    diffs.sort(key=lambda t: (-t[2], -t[1]))
    print(f"[compare] common_rows={len(common)} on={on}")
    print("[compare] top diffs (mean_abs, max_abs):")
    for col, mean_abs, max_abs in diffs[:head]:
        print(f"  - {col}: mean_abs={mean_abs:.6g} max_abs={max_abs:.6g}")
    if not diffs:
        print("  (none) numeric columns are identical (or non-numeric).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="", help="e.g., SOXL")
    ap.add_argument("--mode", default="sim", help="sim/live/display")
    ap.add_argument("--output-root", default="output")
    ap.add_argument("--agent", default="", help="filter by agent substring, e.g., dprime_bnf_3scale")

    ap.add_argument("--compare-csv", default="", help="CSV path A (optional)")
    ap.add_argument("--compare-csv2", default="", help="CSV path B (optional)")
    ap.add_argument("--compare-on", default="Date", help="join key for compare (default Date)")
    ap.add_argument("--head", type=int, default=10, help="top diff columns to show in compare summary")

    args = ap.parse_args()

    if args.compare_csv and args.compare_csv2:
        compare_csv(args.compare_csv, args.compare_csv2, on=args.compare_on, head=args.head)
        return

    symbol = args.symbol.strip()
    if not symbol:
        raise SystemExit("--symbol is required unless --compare-csv/--compare-csv2 are used.")

    mode = args.mode.strip()
    out_root = args.output_root.strip()
    agent = args.agent.strip()

    print(f"[check] symbol={symbol} mode={mode} output_root={out_root} agent_filter={agent or '(none)'}")

    eq_files = _find_all([
        os.path.join(out_root, "stepE", mode, f"stepE_equity_*{symbol}*.csv"),
        os.path.join(out_root, "stepE", mode, f"*equity*{symbol}*.csv"),
    ])
    if agent:
        eq_files = [p for p in eq_files if agent in os.path.basename(p)]

    if not eq_files:
        print("[warn] StepE equity CSV not found.")
    else:
        print(f"[StepE] equity files: {len(eq_files)} (test split if available)")
        for p in eq_files:
            s = summarize_equity(p, split_filter="test")
            print(f"  - {os.path.basename(p)} | rows={s.rows} | col={s.equity_col} | "
                  f"ret_max={s.ret_max:.4f} ret_min={s.ret_min:.4f} | >30%={s.gt30} >50%={s.gt50} < -30%={s.ltm30}")

    prices = load_prices_test(out_root, mode, symbol)
    if prices is None:
        print("[warn] StepA prices_test not found or missing Date/Close columns.")
        print("       expected: output/stepA/<mode>/*prices*test*<SYMBOL>*.csv")
    else:
        print(f"[StepA] prices_test -> {prices.path} (close_col={prices.close_col}) rows={len(prices.df)}")

    log_files = _find_all([
        os.path.join(out_root, "stepE", mode, f"stepE_daily_log_*{symbol}*.csv"),
        os.path.join(out_root, "stepE", mode, f"*daily*log*{symbol}*.csv"),
        os.path.join(out_root, "stepE", mode, f"*trades*{symbol}*.csv"),
        os.path.join(out_root, "stepE", mode, f"*actions*{symbol}*.csv"),
    ])
    if agent:
        log_files = [p for p in log_files if agent in os.path.basename(p)]

    if not log_files:
        print("[warn] StepE daily/trade/action log not found. Cannot prove pos-vs-reward alignment from outputs alone.")
        print("       Recommended: export a daily log with Date,pos_prev,pos,price_prev,price_now,ret/reward,Split.")
    else:
        print(f"[StepE] candidate logs: {len(log_files)}")
        for f in log_files[:30]:
            audit = reward_audit_if_possible(f, split_filter="test")
            if audit is not None:
                used = ", ".join([f"{k}={v}" for k, v in audit.used_cols.items()])
                print(f"  - {os.path.basename(f)} | RewardAudit best={audit.best} "
                      f"(MAE prev={audit.mae_vs_pos_prev:.6g}, MAE now={audit.mae_vs_pos_now:.6g}, n={audit.n})")
                print(f"    used: {used}")

        if prices is not None:
            for f in log_files[:30]:
                res = alignment_test(f, prices, split_filter="test")
                if res is None:
                    df = _read_csv(f)
                    cols = list(df.columns)
                    print(f"  - {os.path.basename(f)} | skip (need Date+pos+ret/reward). "
                          f"detected: Date={_detect_date_col(cols)} pos={_detect_pos_col(cols)} ret={_detect_ret_or_reward_col(cols)}")
                    continue
                print(f"  - {res.file} | Align best={res.best} (MAE={res.best_mae:.6g}, corr={res.best_corr:.4f}, n={res.n}) "
                      f"| MAE pos[t-1]={res.mae_pos_tminus1:.6g} pos[t]={res.mae_pos_t:.6g} pos[t+1]={res.mae_pos_tplus1:.6g} "
                      f"| cols: Date={res.used_date_col} pos={res.used_pos_col} ret={res.used_ret_col}")

    scans = scan_prime_features(out_root, mode, symbol)
    if not scans:
        print("[warn] StepD prime/features CSV not found (or patterns didn't match).")
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
