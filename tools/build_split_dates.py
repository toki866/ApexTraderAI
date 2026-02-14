#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_split_dates.py

Create packs/ops/split_dates.csv (the single source of truth for train/test boundary).

Rules (compatible with your project spec):
- If train_start/train_end are provided, use them.
- Otherwise, require test_start and train_years:
    train_start = (test_start - train_years years)
    train_end   = (test_start - 1 business day)
- test_end can be given directly, or computed from test_months (default 3).
- Dates are aligned to the available trading dates if --prices-csv is provided.

Output: split_dates.csv with columns:
  Date, split, train_start, train_end, test_start, test_end
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


def _parse_date(s: Optional[str]) -> Optional[pd.Timestamp]:
    if not s:
        return None
    return pd.to_datetime(s, errors="raise")


def _bday_prev(d: pd.Timestamp) -> pd.Timestamp:
    return (pd.bdate_range(end=d, periods=2)[0]).normalize()


def _bday_add_months(start: pd.Timestamp, months: int) -> pd.Timestamp:
    target = (start + pd.DateOffset(months=months)).normalize()
    return target


def _align_to_prices(d: pd.Timestamp, dates: pd.Series, how: str) -> pd.Timestamp:
    dates = pd.to_datetime(dates).dropna().sort_values().reset_index(drop=True)
    if len(dates) == 0:
        return d.normalize()
    d = d.normalize()
    if how == "ceil":
        m = dates[dates >= d]
        return (m.iloc[0] if len(m) else dates.iloc[-1]).normalize()
    if how == "floor":
        m = dates[dates <= d]
        return (m.iloc[-1] if len(m) else dates.iloc[0]).normalize()
    return d.normalize()


@dataclass
class Split:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def compute_split(
    test_start: Optional[pd.Timestamp],
    test_end: Optional[pd.Timestamp],
    train_years: Optional[int],
    test_months: int,
    train_start: Optional[pd.Timestamp],
    train_end: Optional[pd.Timestamp],
) -> Split:
    if train_start is not None and train_end is not None and test_start is not None and test_end is not None:
        return Split(train_start.normalize(), train_end.normalize(), test_start.normalize(), test_end.normalize())

    if test_start is None:
        raise SystemExit("ERROR: --test-start is required unless you provide --train-start/--train-end and --test-start/--test-end.")

    if train_years is None or train_years <= 0:
        raise SystemExit("ERROR: --train-years is required (positive int) unless --train-start/--train-end are provided.")

    train_start2 = (test_start - pd.DateOffset(years=int(train_years))).normalize()
    train_end2 = _bday_prev(test_start.normalize())
    test_end2 = test_end.normalize() if test_end is not None else _bday_add_months(test_start.normalize(), int(test_months))

    if train_start is not None:
        train_start2 = train_start.normalize()
    if train_end is not None:
        train_end2 = train_end.normalize()

    return Split(train_start2, train_end2, test_start.normalize(), test_end2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SOXL")
    ap.add_argument("--test-start")
    ap.add_argument("--test-end")
    ap.add_argument("--test-months", type=int, default=3)
    ap.add_argument("--train-years", type=int, default=None)
    ap.add_argument("--train-start")
    ap.add_argument("--train-end")
    ap.add_argument("--prices-csv", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ts_test_start = _parse_date(args.test_start)
    ts_test_end = _parse_date(args.test_end)
    ts_train_start = _parse_date(args.train_start)
    ts_train_end = _parse_date(args.train_end)

    split = compute_split(
        test_start=ts_test_start,
        test_end=ts_test_end,
        train_years=args.train_years,
        test_months=args.test_months,
        train_start=ts_train_start,
        train_end=ts_train_end,
    )

    if args.prices_csv:
        p = Path(args.prices_csv)
        if p.exists():
            dfp = pd.read_csv(p)
            if "Date" not in dfp.columns:
                raise SystemExit(f"ERROR: prices-csv has no Date column: {p}")
            dates = pd.to_datetime(dfp["Date"], errors="coerce")
            split = Split(
                train_start=_align_to_prices(split.train_start, dates, "ceil"),
                train_end=_align_to_prices(split.train_end, dates, "floor"),
                test_start=_align_to_prices(split.test_start, dates, "ceil"),
                test_end=_align_to_prices(split.test_end, dates, "floor"),
            )

    dr = pd.bdate_range(start=split.train_start, end=split.test_end)
    out = pd.DataFrame({"Date": dr})
    out["split"] = "ignore"
    out.loc[(out["Date"] >= split.train_start) & (out["Date"] <= split.train_end), "split"] = "train"
    out.loc[(out["Date"] >= split.test_start) & (out["Date"] <= split.test_end), "split"] = "test"
    out["train_start"] = split.train_start.strftime("%Y-%m-%d")
    out["train_end"] = split.train_end.strftime("%Y-%m-%d")
    out["test_start"] = split.test_start.strftime("%Y-%m-%d")
    out["test_end"] = split.test_end.strftime("%Y-%m-%d")
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] wrote split_dates: {out_path} rows={len(out)}")
    print(f"  train: {split.train_start.date()} .. {split.train_end.date()}")
    print(f"  test : {split.test_start.date()} .. {split.test_end.date()}")


if __name__ == "__main__":
    main()
