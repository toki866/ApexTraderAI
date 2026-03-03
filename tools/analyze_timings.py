from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output")
    ap.add_argument("--buffer", type=float, default=0.25)
    args = ap.parse_args()

    root = Path(args.root)
    files = list(root.glob("**/timings.csv"))
    if not files:
        print("No timings.csv found")
        return 0

    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    if df.empty:
        print("timings.csv files are empty")
        return 0

    totals = df.groupby(["run_id", "retrain", "branch_id"], as_index=False)["elapsed_sec"].sum()
    for (retrain, branch), g in totals.groupby(["retrain", "branch_id"]):
        p95 = g["elapsed_sec"].quantile(0.95)
        p50 = g["elapsed_sec"].quantile(0.50)
        mx = g["elapsed_sec"].max()
        rec = p95 * (1.0 + float(args.buffer))
        print(f"retrain_{retrain} {branch}: p50={p50:.2f}s p95={p95:.2f}s max={mx:.2f}s -> recommended={rec:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
