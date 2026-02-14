#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_packs_from_raw.py

Create a clean packs structure:
  packs/ops, packs/train, packs/test, packs/display

Inputs are auto-discovered under --input-root:
- stepA_prices_{symbol}.csv
- stepA_features_{symbol}.csv
Optional:
- stepB_pred_time_all_{symbol}.csv
- stepD_events_*_{symbol}.csv
- stepD_envelope_daily_*_{symbol}.csv

This script is intentionally conservative: train/test outputs are *split first* and
written to separate folders so downstream code can avoid reading display files.

Usage:
  python tools/build_packs_from_raw.py --symbol SOXL --input-root output --run-id SOXL_20260115_073000 --test-start 2022-01-03 --test-end 2022-03-31 --train-years 8 --out-root output/runs/SOXL_20260115_073000/packs
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

from build_split_dates import compute_split  # same folder


PRICE_COLS = ["Date", "Open", "High", "Low", "Close", "Volume"]
TECH_HINTS = ["Gap", "RSI", "MACD", "MACD_signal"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_file(input_root: Path, pattern: str) -> Optional[Path]:
    candidates = []
    for sub in ["stepA", "stepB", "stepC", "stepD", ""]:
        base = input_root / sub if sub else input_root
        candidates.extend(list(base.glob(pattern)))
    return candidates[0] if candidates else None


def norm_date(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" not in df.columns:
        for c in df.columns:
            if str(c).lower() in ("date", "datetime", "time", "timestamp"):
                df = df.rename(columns={c: "Date"})
                break
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    return df


def split_by_dates(df: pd.DataFrame, split_df: pd.DataFrame, which: str) -> pd.DataFrame:
    if df.empty:
        return df
    sdf = split_df[split_df["split"] == which][["Date"]].copy()
    return df.merge(sdf, on="Date", how="inner")


def detect_periodic_cols(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        cl = str(c).lower()
        if cl.startswith("sin_k") or cl.startswith("cos_k"):
            cols.append(c)
    return cols


def detect_technical_cols(df: pd.DataFrame) -> List[str]:
    periodic = set(detect_periodic_cols(df))
    price = set([c for c in PRICE_COLS if c in df.columns])
    tech = [c for c in df.columns if c not in periodic and c not in price and c != "Date"]
    ordered = [h for h in TECH_HINTS if h in tech] + [c for c in tech if c not in TECH_HINTS]
    return ordered


def make_labels(prices_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    df = prices_df.copy()
    close = pd.to_numeric(df["Close"], errors="coerce")
    if mode == "delta_close":
        y = close.shift(-1) - close
        name = "y_delta_close_1d"
    elif mode == "ret1":
        y = (close.shift(-1) / close) - 1.0
        name = "y_ret_1d"
    else:
        raise SystemExit(f"ERROR: unknown label mode: {mode}")
    out = pd.DataFrame({"Date": df["Date"], name: y}).dropna().reset_index(drop=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--input-root", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--split-csv", default=None)
    ap.add_argument("--test-start", default=None)
    ap.add_argument("--test-end", default=None)
    ap.add_argument("--test-months", type=int, default=3)
    ap.add_argument("--train-years", type=int, default=None)
    ap.add_argument("--label", choices=["delta_close", "ret1"], default="delta_close")

    args = ap.parse_args()

    symbol = str(args.symbol)
    input_root = Path(args.input_root)
    out_root = Path(args.out_root)

    ops_dir = out_root / "ops"
    train_dir = out_root / "train"
    test_dir = out_root / "test"
    display_dir = out_root / "display"
    for d in [ops_dir, train_dir, test_dir, display_dir]:
        d.mkdir(parents=True, exist_ok=True)

    prices_path = find_file(input_root, f"stepA_prices_{symbol}.csv")
    feats_path = find_file(input_root, f"stepA_features_{symbol}.csv")
    if not prices_path or not feats_path:
        raise SystemExit(f"ERROR: StepA files not found under {input_root}. Need stepA_prices/stepA_features for {symbol}.")

    df_prices = norm_date(pd.read_csv(prices_path))
    df_feats = norm_date(pd.read_csv(feats_path))

    periodic_cols = detect_periodic_cols(df_feats)
    tech_cols = detect_technical_cols(df_feats)

    prices_full = df_prices[[c for c in PRICE_COLS if c in df_prices.columns]].copy()
    periodic_full = df_feats[["Date"] + periodic_cols].copy() if periodic_cols else pd.DataFrame({"Date": df_feats["Date"]})

    # split_dates
    if args.split_csv:
        split_df = pd.read_csv(Path(args.split_csv))
    else:
        split = compute_split(
            test_start=pd.to_datetime(args.test_start) if args.test_start else None,
            test_end=pd.to_datetime(args.test_end) if args.test_end else None,
            train_years=args.train_years,
            test_months=args.test_months,
            train_start=None,
            train_end=None,
        )
        dr = pd.bdate_range(start=split.train_start, end=split.test_end)
        split_df = pd.DataFrame({"Date": dr})
        split_df["split"] = "ignore"
        split_df.loc[(split_df["Date"] >= split.train_start) & (split_df["Date"] <= split.train_end), "split"] = "train"
        split_df.loc[(split_df["Date"] >= split.test_start) & (split_df["Date"] <= split.test_end), "split"] = "test"
        split_df["train_start"] = split.train_start.strftime("%Y-%m-%d")
        split_df["train_end"] = split.train_end.strftime("%Y-%m-%d")
        split_df["test_start"] = split.test_start.strftime("%Y-%m-%d")
        split_df["test_end"] = split.test_end.strftime("%Y-%m-%d")
        split_df["Date"] = split_df["Date"].dt.strftime("%Y-%m-%d")
        split_df.to_csv(ops_dir / "split_dates.csv", index=False, encoding="utf-8-sig")

    train_prices = split_by_dates(prices_full, split_df, "train")
    test_prices = split_by_dates(prices_full, split_df, "test")

    feats_light = df_feats[["Date"] + [c for c in (tech_cols + periodic_cols) if c in df_feats.columns]].copy()
    train_feats = split_by_dates(feats_light, split_df, "train")
    test_feats = split_by_dates(feats_light, split_df, "test")

    train_tech = train_feats[["Date"] + [c for c in tech_cols if c in train_feats.columns]].copy()
    test_tech = test_feats[["Date"] + [c for c in tech_cols if c in test_feats.columns]].copy()
    train_periodic = train_feats[["Date"] + [c for c in periodic_cols if c in train_feats.columns]].copy()
    test_periodic = test_feats[["Date"] + [c for c in periodic_cols if c in test_feats.columns]].copy()

    train_label = make_labels(train_prices, mode=args.label)
    test_label = make_labels(test_prices, mode=args.label)

    train_prices.to_csv(train_dir / "train_prices.csv", index=False, encoding="utf-8-sig")
    train_tech.to_csv(train_dir / "train_technical.csv", index=False, encoding="utf-8-sig")
    train_periodic.to_csv(train_dir / "train_periodic.csv", index=False, encoding="utf-8-sig")
    train_label.to_csv(train_dir / "train_label.csv", index=False, encoding="utf-8-sig")

    test_prices.to_csv(test_dir / "test_prices.csv", index=False, encoding="utf-8-sig")
    test_tech.to_csv(test_dir / "test_technical.csv", index=False, encoding="utf-8-sig")
    test_periodic.to_csv(test_dir / "test_periodic.csv", index=False, encoding="utf-8-sig")
    test_label.to_csv(test_dir / "test_label.csv", index=False, encoding="utf-8-sig")

    prices_full.to_csv(display_dir / "prices_display_full.csv", index=False, encoding="utf-8-sig")
    periodic_full.to_csv(display_dir / "periodic_full.csv", index=False, encoding="utf-8-sig")

    # optional display: pred/envelope
    pred_path = find_file(input_root, f"stepB_pred_time_all_{symbol}.csv")
    if pred_path and pred_path.exists():
        norm_date(pd.read_csv(pred_path)).to_csv(display_dir / "pred_time_all.csv", index=False, encoding="utf-8-sig")

    # copy any stepD csvs if found
    ev = find_file(input_root, f"stepD_events_*_{symbol}.csv")
    if ev and ev.exists():
        shutil.copy2(ev, display_dir / ev.name.replace("stepD_events_", "envelope_events_"))
    daily = find_file(input_root, f"stepD_envelope_daily_*_{symbol}.csv")
    if daily and daily.exists():
        shutil.copy2(daily, display_dir / daily.name.replace("stepD_envelope_daily_", "envelope_daily_"))

    # ops manifest + hashes
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta = [{
        "run_id": args.run_id,
        "created_at": now,
        "symbol": symbol,
        "input_root": str(input_root),
        "prices_path": str(prices_path),
        "features_path": str(feats_path),
        "train_rows": int(len(train_prices)),
        "test_rows": int(len(test_prices)),
        "min_date": str(prices_full["Date"].min()),
        "max_date": str(prices_full["Date"].max()),
        "label": args.label,
    }]
    pd.DataFrame(meta).to_csv(ops_dir / "run_manifest.csv", index=False, encoding="utf-8-sig")

    rows = []
    for fp in [prices_path, feats_path]:
        rows.append({"file": str(fp), "sha256": sha256_file(fp), "bytes": fp.stat().st_size})
    if pred_path and pred_path.exists():
        rows.append({"file": str(pred_path), "sha256": sha256_file(pred_path), "bytes": pred_path.stat().st_size})
    pd.DataFrame(rows).to_csv(ops_dir / "data_hashes.csv", index=False, encoding="utf-8-sig")

    print("[OK] packs created at:", out_root)


if __name__ == "__main__":
    main()
