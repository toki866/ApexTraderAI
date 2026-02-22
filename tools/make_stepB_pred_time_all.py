#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def _read_prices(stepa_prices_path: Path) -> pd.DataFrame:
    prices = pd.read_csv(stepa_prices_path)
    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce")
    prices = prices.dropna(subset=["Date"]).drop_duplicates(subset=["Date"]).sort_values("Date")
    prices["Close"] = pd.to_numeric(prices["Close"], errors="coerce")
    return prices


def _pick_stepb_mamba_file(output_root: Path, sym: str) -> Path:
    for p in [
        output_root / "stepB" / f"stepB_pred_close_mamba_{sym}.csv",
        output_root / f"stepB_pred_close_mamba_{sym}.csv",
        output_root / "stepB" / f"stepB_delta_mamba_{sym}.csv",
    ]:
        if p.exists() and p.stat().st_size > 0:
            return p
    raise FileNotFoundError(f"Missing Mamba StepB file for {sym}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SOXL")
    ap.add_argument("--output-root", default="output")
    args = ap.parse_args()

    sym = args.symbol
    output_root = Path(args.output_root)
    prices = _read_prices(output_root / f"stepA_prices_{sym}.csv")
    base = prices[["Date", "Close"]].copy()
    base["Prev_Close"] = base["Close"].shift(1)

    mamba = pd.read_csv(_pick_stepb_mamba_file(output_root, sym))
    mamba["Date"] = pd.to_datetime(mamba["Date"], errors="coerce")

    mamba_col = next((c for c in mamba.columns if c.lower() == "pred_close_mamba"), None)
    if mamba_col is None:
        delta_col = next((c for c in ["Delta_Close_pred_MAMBA", "Delta_pred", "Delta"] if c in mamba.columns), None)
        if delta_col is None:
            raise ValueError("Mamba file must contain Pred_Close_MAMBA or a delta column")
        m = mamba[["Date", delta_col]].merge(base[["Date", "Prev_Close"]], on="Date", how="left")
        m["Pred_Close_MAMBA"] = pd.to_numeric(m[delta_col], errors="coerce") + pd.to_numeric(m["Prev_Close"], errors="coerce")
        out = m[["Date", "Pred_Close_MAMBA"]]
    else:
        out = mamba[["Date", mamba_col]].rename(columns={mamba_col: "Pred_Close_MAMBA"})

    out = out.dropna(subset=["Date"]).sort_values("Date")
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    out_path = output_root / f"stepB_pred_time_all_{sym}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"WROTE {out_path} cols={list(out.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
