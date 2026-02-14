#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Generate --agent-csv arguments for tools/run_router_bandit_backtest.py

Why this exists
- If you write placeholders like a4=... then Python tries to open a file literally named "..." and crashes.
- This tool scans output\stepE\<mode>\ for StepE daily logs and prints ready-to-paste --agent-csv lines.

Usage (Windows CMD)
  python tools\gen_agent_csv_args.py --dir output\stepE\sim --symbol SOXL --top 10

Tips
- By default PRIMARY logs are excluded (often not an "agent"). Use --include-primary to include them.
- Output includes:
  (1) Ranked candidates
  (2) Multi-line caret-continued args (for CMD)
  (3) One-line args
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List, Tuple

def _safe_import_pandas():
    try:
        import pandas as pd  # type: ignore
        return pd
    except Exception:
        print("[ERROR] pandas is required. Try: pip install -U pandas", file=sys.stderr)
        raise

def _to_num(series):
    pd = _safe_import_pandas()
    return pd.to_numeric(series, errors="coerce")

def _score_daily_log(p: Path) -> float:
    """
    Lightweight score to rank agent logs:
    Prefer: equity_end / equity_start on Split=='test'
    Fallback: prod(1 + reward_next/ret/ret_next) on Split=='test'
    """
    import numpy as np  # type: ignore
    pd = _safe_import_pandas()
    df = pd.read_csv(p)

    d = df
    if "Split" in df.columns:
        m = df["Split"].astype(str).str.lower().eq("test")
        if m.any():
            d = df.loc[m].copy()

    if "equity" in d.columns:
        e = _to_num(d["equity"]).dropna()
        if len(e) >= 2:
            e0 = float(e.iloc[0])
            e1 = float(e.iloc[-1])
            if abs(e0) < 1e-12:
                e0 = 1e-12
            return e1 / e0

    for col in ("reward_next", "ret", "ret_next", "reward", "pnl"):
        if col in d.columns:
            r = _to_num(d[col]).fillna(0.0).astype(float).values
            return float(np.prod(1.0 + r))

    return float("nan")

def _win_path(p: Path) -> str:
    return str(p).replace("/", "\\")

def _unique_csv_files(root: Path, pattern: str) -> List[Path]:
    # Ensure a single glob pass and deduplicate paths.
    pat = pattern
    if not pat.lower().endswith(".csv"):
        pat = pat + ".csv"
    files = [p for p in root.glob(pat) if p.is_file() and p.suffix.lower() == ".csv"]
    # Use resolved absolute path for stable deduplication, then keep original Path.
    uniq = {}
    for p in files:
        try:
            k = str(p.resolve())
        except Exception:
            k = str(p)
        uniq[k] = p
    return sorted(uniq.values(), key=lambda x: x.name.lower())

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help=r"Directory containing StepE daily logs (e.g., output\stepE\sim)")
    ap.add_argument("--symbol", default="", help="Symbol filter (e.g., SOXL). If empty, no filter.")
    ap.add_argument("--glob", default="stepE_daily_log_*", help="Glob pattern inside --dir (without .csv is OK)")
    ap.add_argument("--top", type=int, default=10, help="How many agents to emit")
    ap.add_argument("--name-prefix", default="a", help="Agent name prefix (default: a -> a1,a2,...)")
    ap.add_argument("--start-index", type=int, default=1, help="Starting index for agent names (default: 1)")
    ap.add_argument("--include-primary", action="store_true", help="Include *PRIMARY* logs as candidates")
    args = ap.parse_args()

    root = Path(args.dir)
    if not root.exists():
        print(f"[ERROR] dir not found: {root}")
        return 2

    files = _unique_csv_files(root, args.glob)

    if args.symbol:
        sym = args.symbol.upper()
        files = [p for p in files if sym in p.name.upper()]

    if not args.include_primary:
        files = [p for p in files if "PRIMARY" not in p.name.upper()]

    if not files:
        print("[ERROR] No candidate CSV files found.")
        print(r"Try: dir output\stepE\sim\stepE_daily_log_*_SOXL.csv /b")
        return 3

    scored: List[Tuple[Path, float]] = []
    for p in files:
        try:
            s = _score_daily_log(p)
        except Exception:
            s = float("nan")
        scored.append((p, s))

    # Sort: nan last, otherwise descending
    scored.sort(key=lambda t: (-t[1]) if (t[1] == t[1]) else 1e99)

    topn = scored[: max(1, int(args.top))]

    print("[INFO] Candidates ranked (higher is better):")
    for i, (p, s) in enumerate(topn, 1):
        print(f"  {i:2d}. score={s:.6g}  {p.name}")

    start = int(args.start_index)
    names = [f"{args.name_prefix}{start + i}" for i in range(len(topn))]

    print("\n[PASTE] Windows CMD multi-line (caret continued):")
    for i, ((p, _), name) in enumerate(zip(topn, names), 1):
        tail = " ^" if i < len(topn) else ""
        print(f"  --agent-csv {name}={_win_path(p)}{tail}")

    print("\n[PASTE] One-line:")
    oneline = " ".join([f'--agent-csv {name}={_win_path(p)}' for (p, _), name in zip(topn, names)])
    print(oneline)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
