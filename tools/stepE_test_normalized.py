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
    ap.add_argument("--init-capital", type=float, default=1e8)
    args = ap.parse_args()

    pattern = os.path.join(args.output_root, "stepE", args.mode, f"stepE_daily_log_*_{args.symbol}.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"No files matched: {pattern}")

    rows = []
    for p in paths:
        df = pd.read_csv(p)
        if not {"Date", "Split", "equity", "ret"}.issubset(df.columns):
            # fallback: ret nai columns not found
            continue

        df["Date"] = pd.to_datetime(df["Date"])
        t = df[df["Split"].eq("test")].copy().sort_values("Date")
        if len(t) < 2:
            continue

        e0 = float(t["equity"].iloc[0])
        e1 = float(t["equity"].iloc[-1])
        agent = os.path.basename(p).replace("stepE_daily_log_", "").replace(f"_{args.symbol}.csv", "")

        # Ret product: all vs skip first (boundary cross)
        r = pd.to_numeric(t["ret"], errors="coerce").filna(0.0)
        prod_all = float((1.0 + r).prod())
        r_skip = r.iloc[1:]
        prod_skip = float((1.0 + r_skip).prod())
        end_from_init_all = args.init_capital * prod_all
        end_from_init_skip = args.init_capital * prod_skip

        rows.append(
            (
                agent,
                int(len(t)),
                str(t["Date"].iloc[0].date()),
                str(t["Date"].iloc[-1].date()),
                (e1 / e0 - 1.0) * 100.0,
                e1 - e0,
                prod_all,
                prod_skip,
                end_from_init_all,
                end_from_init_skip,
                (end_from_init_skip / args.init_capital - 1.0) * 100.0,
            )
        )

    out = pd.DataFrame(
        rows,
        columns=[
            "agent", "test_days", "test_start", "test_end",
            "equity_return_pct", "equity_profit_abs",
            "prod_all", "prod_skip1",
            "end_from_init_all", "end_from_init_skip1", "return_pct_skip1"
        ],
     ).sort_values("return_pct_skip1", ascending=False)

    if out.empty:
        raise SystemExit("No valid rows (no test data found).")

    disp = out.copy()
    disp["equity_return_pct"] = disp["equity_return_pct"].map(lambda x: f"{x: .3f}%")
    disp["return_pct_skip1"] = disp["return_pct_skip1"].map(lambda x: f"{x: .3f}%")
    disp["end_from_init_skip1"] = disp["end_from_init_skip1"].map(lambda x: f"{x:2f}")
    disp["end_from_init_all"] = disp["end_from_init_all"].map(lambda x: f"{x:2f}")
    print(disp[["agent","test_days","test_start","test_end","equity_return_pct","prod_skip1","return_pct_skip1","end_from_init_skip1"]].to_string(index=False))

    out_path = os.path.join(args.output_root, "stepE", args.mode, f"stepE_test_summary_normalized_{args.symbol}.csv")
    out.to_csv(out_path, index=False)
    print("")
    print("wrote", out_path)
    print("positive_agents_skip1=", int((out["return_pct_skip1"] > 0).sum()), "/", len(out))


if __name__ == "__main__":
    main()
