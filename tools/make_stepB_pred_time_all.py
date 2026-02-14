#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tools/make_stepB_pred_time_all.py

Create output/stepB_pred_time_all_{symbol}.csv from StepA prices and StepB agent outputs.

Why
---
Windows CMD makes multi-line `python -c` fragile. This script is a stable, reproducible way to
generate the io_contract artifact without "Closeコピー" (dummy fallback).

Inputs
------
- output/stepA_prices_{symbol}.csv
- output/stepB/stepB_delta_xsr_{symbol}.csv
- output/stepB/stepB_delta_mamba_{symbol}.csv
- output/stepB/stepB_delta_fedformer_{symbol}.csv (optional by default)

For each StepB file, this script uses:
- Pred_Close_raw (preferred) OR
- Pred_Close OR
- Delta_pred (or legacy delta columns) + Prev_Close -> Pred_Close_*

Supported legacy delta column names
-----------------------------------
Delta_pred, DeltaClose_pred, DeltaClosePred, dClose_pred, dClosePred, Pred_Delta, Delta

Output
------
- output/stepB_pred_time_all_{symbol}.csv
  columns: Date, Pred_Close_XSR, Pred_Close_MAMBA, Pred_Close_FED

Behavior
--------
- By default, if FEDformer file is missing, Pred_Close_FED will be NaN (NOT copied from Close).
- Use --require-fedformer to hard-fail if fedformer file is missing.

Usage
-----
python tools\\make_stepB_pred_time_all.py --symbol SOXL
python tools\\make_stepB_pred_time_all.py --symbol SOXL --require-fedformer
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd


def _read_prices(stepa_prices_path: Path) -> pd.DataFrame:
    if not stepa_prices_path.exists():
        raise FileNotFoundError(f"StepA prices not found: {stepa_prices_path}")
    prices = pd.read_csv(stepa_prices_path)
    if "Date" not in prices.columns or "Close" not in prices.columns:
        raise ValueError(f"StepA prices must contain Date, Close. cols={list(prices.columns)}")
    prices = prices.copy()
    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce")
    prices = prices.dropna(subset=["Date"]).drop_duplicates(subset=["Date"]).sort_values("Date")
    prices["Close"] = pd.to_numeric(prices["Close"], errors="coerce")
    return prices


def _find_stepb_file(output_root: Path, name: str) -> Path | None:
    c1 = output_root / "stepB" / name
    if c1.exists():
        return c1
    c2 = output_root / name
    if c2.exists():
        return c2
    return None


def _pick_first_existing(output_root: Path, names: list[str]) -> Path | None:
    for n in names:
        p = _find_stepb_file(output_root, n)
        if p is not None:
            return p
    return None


def _extract_pred_close(stepb_path: Path, base: pd.DataFrame, out_col: str) -> pd.DataFrame:
    df = pd.read_csv(stepb_path)
    if df.empty:
        raise ValueError(f"StepB file is empty: {stepb_path}")
    if "Date" not in df.columns:
        raise ValueError(f"StepB file has no Date column: {stepb_path} cols={list(df.columns)}")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).drop_duplicates(subset=["Date"]).sort_values("Date")

    # direct pred close
    for c in ("Pred_Close_raw", "Pred_Close", "Close_pred", "close_pred"):
        if c in df.columns:
            y = df[["Date", c]].rename(columns={c: out_col})
            y[out_col] = pd.to_numeric(y[out_col], errors="coerce")
            return y.dropna(subset=[out_col])

    # delta-based reconstruction (support legacy column names)
    delta_candidates = [
        "Delta_pred",
        "DeltaClose_pred",
        "DeltaClosePred",
        "dClose_pred",
        "dClosePred",
        "Pred_Delta",
        "Delta",
        "Delta_Pred",
    ]
    delta_col = next((c for c in delta_candidates if c in df.columns), None)
    if delta_col is None:
        raise ValueError(
            f"StepB file has no Pred_Close_raw/Pred_Close nor delta columns: {stepb_path} cols={list(df.columns)}"
        )

    y = df[["Date", delta_col]].copy()
    y[delta_col] = pd.to_numeric(y[delta_col], errors="coerce")
    y = y.dropna(subset=[delta_col])
    y = y.merge(base[["Date", "Prev_Close"]], on="Date", how="left").dropna(subset=["Prev_Close"])
    y[out_col] = y["Prev_Close"] + y[delta_col]
    return y[["Date", out_col]].dropna(subset=[out_col])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SOXL")
    ap.add_argument("--output-root", default="output")
    ap.add_argument("--require-fedformer", action="store_true")
    args = ap.parse_args()

    sym = args.symbol
    output_root = Path(args.output_root)

    prices = _read_prices(output_root / f"stepA_prices_{sym}.csv")
    base = prices[["Date", "Close"]].copy()
    base["Prev_Close"] = base["Close"].astype(float).shift(1)

    res = base[["Date"]].copy()

    # XSR
    p = _pick_first_existing(output_root, [f"stepB_delta_xsr_{sym}.csv"])
    if p is None:
        raise FileNotFoundError(f"Missing XSR stepB file: output/stepB/stepB_delta_xsr_{sym}.csv")
    res = res.merge(_extract_pred_close(p, base, "Pred_Close_XSR"), on="Date", how="left")

    # Mamba
    p = _pick_first_existing(output_root, [f"stepB_delta_mamba_{sym}.csv"])
    if p is None:
        raise FileNotFoundError(f"Missing Mamba stepB file: output/stepB/stepB_delta_mamba_{sym}.csv")
    res = res.merge(_extract_pred_close(p, base, "Pred_Close_MAMBA"), on="Date", how="left")

    # FEDformer (optional by default)
    p = _pick_first_existing(output_root, [f"stepB_delta_fedformer_{sym}.csv"])
    if p is None:
        if args.require_fedformer:
            raise FileNotFoundError(f"Missing FEDformer stepB file: output/stepB/stepB_delta_fedformer_{sym}.csv")
        res["Pred_Close_FED"] = pd.NA
        print("[WARN] FEDformer file missing. Pred_Close_FED will be NaN (no dummy fill).", file=sys.stderr)
    else:
        res = res.merge(_extract_pred_close(p, base, "Pred_Close_FED"), on="Date", how="left")

    # Minimal sanity
    if len(res) < 10:
        raise RuntimeError(f"Too few rows in output (rows={len(res)}).")

    # Write
    out_path = output_root / f"stepB_pred_time_all_{sym}.csv"
    res = res.sort_values("Date")
    res["Date"] = res["Date"].dt.strftime("%Y-%m-%d")
    res.to_csv(out_path, index=False, encoding="utf-8")
    print(f"WROTE {out_path} rows={len(res)} cols={list(res.columns)}")

    # Diagnostics: dummy check vs Close
    a = prices.copy()
    a["Date"] = a["Date"].dt.strftime("%Y-%m-%d")
    m = pd.read_csv(out_path).merge(a[["Date", "Close"]], on="Date", how="left")
    for c in ["Pred_Close_XSR", "Pred_Close_MAMBA", "Pred_Close_FED"]:
        if c not in m.columns:
            continue
        vv = pd.to_numeric(m[c], errors="coerce")
        cc = pd.to_numeric(m["Close"], errors="coerce")
        mask = vv.notna() & cc.notna()
        if int(mask.sum()) == 0:
            print(f"[INFO] {c}: no numeric values (all NaN).")
            continue
        same = float((vv[mask] == cc[mask]).mean())
        print(f"[CHECK] {c}: same_as_close_ratio={same:.6f} (1.0 means dummy/copy)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        raise
