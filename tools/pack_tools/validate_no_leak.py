#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
validate_no_leak.py

Heuristic leak checks for StepA features (bfill-like) and StepD code (center=True/bfill).

Usage:
  python tools/validate_no_leak.py --features-csv output/stepA/stepA_features_SOXL.csv --stepa-code ai_core/services/step_a_service.py --stepd-code ai_core/services/step_d_service.py
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _ok(msg: str) -> None:
    print(f"[ OK ] {msg}")


def check_stepa_features(features_csv: Path) -> int:
    df = pd.read_csv(features_csv)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    bad = 0

    if "Gap" in df.columns and len(df) > 0:
        g0 = df.loc[0, "Gap"]
        if pd.notna(g0):
            bad += 1
            _warn(f"StepA: Gap first row is not NaN (value={g0}). Likely bfill() -> future leakage risk.")
        else:
            _ok("StepA: Gap first row is NaN (good).")

    if "RSI" in df.columns and len(df) >= 14:
        head = df["RSI"].iloc[:13]
        if head.notna().any():
            bad += 1
            _warn("StepA: RSI has non-NaN in first 13 rows (RSI14 should be NaN). Strong bfill() leak signal.")
        else:
            _ok("StepA: RSI first 13 rows are NaN (good).")

    return bad


def check_code_for_patterns(code_path: Path, patterns: List[Tuple[str, str]]) -> int:
    if not code_path.exists():
        _warn(f"code not found: {code_path}")
        return 0
    text = code_path.read_text(encoding="utf-8", errors="ignore")
    bad = 0
    for label, pat in patterns:
        if pat in text:
            bad += 1
            _warn(f"{code_path.name}: found '{pat}' ({label})")
    if bad == 0:
        _ok(f"{code_path.name}: no dangerous patterns found (checked patterns only).")

    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-csv", required=True)
    ap.add_argument("--stepa-code", default=None)
    ap.add_argument("--stepd-code", default=None)
    args = ap.parse_args()

    bad_total = 0
    bad_total += check_stepa_features(Path(args.features_csv))

    if args.stepa_code:
        bad_total += check_code_for_patterns(Path(args.stepa_code), [
            ("future back-fill", ".bfill("),
            ("double fill", ".ffill().bfill("),
        ])

    if args.stepd_code:
        bad_total += check_code_for_patterns(Path(args.stepd_code), [
            ("non-causal rolling", "center=True"),
            ("future back-fill", ".bfill("),
            ("double fill", ".ffill().bfill("),
        ])

    print("--------------------------------------------------")
    if bad_total == 0:
        print("[PASS] No critical leak patterns detected by heuristics.")
    else:
        print(f"[FAIL] Detected {bad_total} leak risk indicator(s). Fix recommended.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
