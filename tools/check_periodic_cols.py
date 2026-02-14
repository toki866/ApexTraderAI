from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd


def select_periodic_cols(columns: list[str]) -> list[str]:
    cols: list[str] = []
    for c in columns:
        if str(c) == "Date":
            continue
        tokens = re.split(r"[^A-Za-z0-9]+", str(c).lower())
        if ("sin" in tokens) or ("cos" in tokens):
            cols.append(str(c))
    return cols


def main() -> int:
    ap = argparse.ArgumentParser(description="Check StepA periodic sin/cos columns (expected 44).")
    ap.add_argument("--output-root", required=True, help="output folder (e.g., output)")
    ap.add_argument("--symbol", required=True, help="symbol (e.g., SOXL)")
    ap.add_argument("--expected", type=int, default=44, help="expected number of periodic columns (default: 44)")
    ap.add_argument("--show", action="store_true", help="print periodic column names")
    args = ap.parse_args()

    out_root = Path(args.output_root)
    sym = args.symbol

    candidates = [
        out_root / f"stepA_features_{sym}.csv",
        out_root / "stepA" / f"stepA_features_{sym}.csv",
    ]
    feat_path = None
    for p in candidates:
        if p.exists():
            feat_path = p
            break
    if feat_path is None:
        print(f"[check] StepA features CSV not found. searched={candidates}")
        return 2

    df = pd.read_csv(feat_path, nrows=5)
    cols = [str(c).strip() for c in df.columns]
    periodic = select_periodic_cols(cols)

    strict = os.environ.get("STEPB_PERIODIC_ONLY_STRICT", "1").strip() != "0"
    print(f"[check] features_path={feat_path}")
    print(f"[check] STEPB_PERIODIC_ONLY_STRICT={os.environ.get('STEPB_PERIODIC_ONLY_STRICT', '(unset)')} (strict={strict})")
    print(f"[check] periodic_cols_count={len(periodic)} expected={args.expected}")

    if len(periodic) != args.expected:
        print("[check] ❌ NOT MATCHED: periodic cols count differs from expected.")
        if args.show:
            for c in periodic:
                print(c)
        return 1

    print("[check] ✅ OK: periodic cols count matches expected.")
    if args.show:
        for c in periodic:
            print(c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
