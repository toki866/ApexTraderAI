# -*- coding: utf-8 -*-
import argparse
import glob
import os

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SOXL")
    ap.add_argument("--mode", default="sim")
    ap.add_argument("--output-root", default="output")
    args = ap.parse_args()

    pattern = os.path.join(args.output_root, "stepE", args.mode, f"stepE_daily_log_*_{args.symbol}.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"No files matched: {pattern}")

    rows = []
    for p in paths:
        df = pd.read_csv(p)
        if not {"Date", "Split", "equity"}.issubset(df.columns):
            continue

        df["Date"] = pd.to_datetime(df["Date"])
        t = df[df["Split"].eq("test")].copy().sort_values("Date")
        if t.empty:
            continue

        e0 = float(t["equity"].iloc[0])
        e1 = float(t["equity"].iloc[-1])
        agent = os.path.basename(p).replace("stepE_daily_log_", "").replace(f"_{args.symbol}.csv", "")

        rows.append(
            (
                agent,
                int(len(t)),
                str(t["Date"].iloc[0].date()),
                str(t["Date"].iloc[-1].date()),
                (e1 / e0 - 1.0) * 100.0,
                e1 - e0,
            )
        )

    out = pd.DataFrame(
        rows,
        columns=["agent", "test_days", "test_start", "test_end", "test_return_pct", "test_profit_abs"],
    ).sort_values("test_return_pct", ascending=False)

    if out.empty:
        raise SystemExit("No valid rows (no test data found).")

    disp = out.copy()
    disp["test_return_pct"] = disp["test_return_pct"].map(lambda x: f"{x: .3f}%")
    disp["test_profit_abs"] = disp["test_profit_abs"].map(lambda x: f"{x:,.3f}")
    print(disp.to_string(index=False))

    out_path = os.path.join(args.output_root, "stepE", args.mode, f"stepE_test_summary_{args.symbol}.csv")
    out.to_csv(out_path, index=False)
    print("")
    print("wrote", out_path)
    print("positive_agents=", int((out["test_return_pct"] > 0).sum()), "/", len(out))


if __name__ == "__main__":
    main()
