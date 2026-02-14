#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tools/run_stepB_fedformer_only.py

Run FEDformer StepB runner directly and write:
  output/stepB/stepB_delta_fedformer_{symbol}.csv

This is useful when StepB GUI/service isn't triggering FEDformer or when you want
a focused repro with a visible traceback (no swallowing).

It builds a minimal "date_range-like" object from StepA price dates:
  train_start = min(Date)
  train_end   = max(Date)
  test_start  = min(Date)
  test_end    = max(Date)

FEDformerModel internally expects attributes:
  date_range.train_start, date_range.train_end, date_range.test_end

Usage
-----
python tools\\run_stepB_fedformer_only.py --symbol SOXL
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Ensure repo root is on sys.path so `import ai_core` works even when running from tools/ directory.
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from ai_core.services.step_b_fedformer_runner import run_stepB_fedformer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SOXL")
    ap.add_argument("--output-root", default="output")
    args = ap.parse_args()

    sym = args.symbol
    out = Path(args.output_root)

    prices_path = out / f"stepA_prices_{sym}.csv"
    feats_path = out / f"stepA_features_{sym}.csv"
    if not prices_path.exists():
        raise FileNotFoundError(f"Missing {prices_path}")
    if not feats_path.exists():
        raise FileNotFoundError(f"Missing {feats_path}")

    prices = pd.read_csv(prices_path)
    feats = pd.read_csv(feats_path)

    if "Date" not in prices.columns:
        raise ValueError(f"StepA prices has no Date column. cols={list(prices.columns)}")

    dt = pd.to_datetime(prices["Date"], errors="coerce").dropna()
    if dt.empty:
        raise ValueError("Could not parse any dates from StepA prices.")
    d0 = dt.min()
    d1 = dt.max()

    # Minimal date_range-like object compatible with FEDformerModel
    dr = SimpleNamespace(train_start=d0, train_end=d1, test_start=d0, test_end=d1)

    # Minimal cfg-like object (runner reads attributes if present)
    cfg = SimpleNamespace(date_range=dr)

    # app_config can be None; step_b_path_utils will fallback to "output"
    res = run_stepB_fedformer(app_config=None, symbol=sym, prices_df=prices, features_df=feats, cfg=cfg)
    print("OK:", res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
