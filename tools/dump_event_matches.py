#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tools/dump_event_matches.py

Dump matches between pred events (from StepD on Pred_Close_* series) and real events
(from StepD on REALCLOSE) within a date window.

Why this exists
---------------
When you evaluate in a limited window (e.g. test period), "overlap" filtering can
introduce a harmless "boundary fragment" event:
  - an event that started before date-from but overlaps the window
This can show up as unmatched_pred/unmatched_real even though the in-window segmentation
matches well.

To avoid that, this tool supports:
  --filter-mode within
      Keep ONLY events fully contained in [date-from, date-to]
      (StartDate >= date-from AND EndDate <= date-to)
This is recommended for robust matching-quality evaluation.

Matching
--------
Greedy 1-to-1 matching by highest overlap score, with constraints:
  - overlap_score >= --min-overlap
  - abs(pred_mid - real_mid) <= --max-mid-diff-days
  - optional same direction: --require-same-dir

Overlap metrics:
  - iou      : intersection / union
  - pred_cov : intersection / pred_len
  - real_cov : intersection / real_len

Usage examples
--------------
python tools\\dump_event_matches.py ^
  --pred-events output\\stepD\\stepD_events_MAMBA_SOXL.csv ^
  --real-events output\\stepD\\stepD_events_REALCLOSE_SOXL_rbw20_gap10.csv ^
  --date-from 2020-03-01 --date-to 2020-06-30 ^
  --filter-mode within ^
  --overlap-metric iou --min-overlap 0.8 ^
  --max-mid-diff-days 10 --require-same-dir

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import pandas as pd


@dataclass(frozen=True)
class Interval:
    start: pd.Timestamp
    end: pd.Timestamp

    def clip(self, lo: pd.Timestamp, hi: pd.Timestamp) -> "Interval":
        s = self.start if self.start >= lo else lo
        e = self.end if self.end <= hi else hi
        if e < s:
            # empty interval
            return Interval(start=s, end=s)
        return Interval(start=s, end=e)

    @property
    def mid(self) -> pd.Timestamp:
        # midpoint in time
        return self.start + (self.end - self.start) / 2

    def len_days_inclusive(self) -> int:
        # inclusive day count (calendar days)
        return int((self.end - self.start).days) + 1

    def intersect_len(self, other: "Interval") -> int:
        s = max(self.start, other.start)
        e = min(self.end, other.end)
        if e < s:
            return 0
        return int((e - s).days) + 1

    def iou(self, other: "Interval") -> float:
        inter = self.intersect_len(other)
        if inter <= 0:
            return 0.0
        a = self.len_days_inclusive()
        b = other.len_days_inclusive()
        union = a + b - inter
        return float(inter) / float(union) if union > 0 else 0.0

    def cov_pred(self, other: "Interval") -> float:
        inter = self.intersect_len(other)
        a = self.len_days_inclusive()
        return float(inter) / float(a) if a > 0 else 0.0

    def cov_real(self, other: "Interval") -> float:
        inter = self.intersect_len(other)
        b = other.len_days_inclusive()
        return float(inter) / float(b) if b > 0 else 0.0


def _parse_dt(s: Optional[str]) -> Optional[pd.Timestamp]:
    if s is None or str(s).strip() == "":
        return None
    return pd.to_datetime(s)


def _load_events(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # accept some variants
    for c in ["StartDate", "start", "start_date", "startDate"]:
        if c in df.columns and "StartDate" not in df.columns:
            df = df.rename(columns={c: "StartDate"})
            break
    for c in ["EndDate", "end", "end_date", "endDate"]:
        if c in df.columns and "EndDate" not in df.columns:
            df = df.rename(columns={c: "EndDate"})
            break
    if "StartDate" not in df.columns or "EndDate" not in df.columns:
        raise ValueError(f"Missing StartDate/EndDate in {path}")
    df["StartDate"] = pd.to_datetime(df["StartDate"])
    df["EndDate"] = pd.to_datetime(df["EndDate"])
    if "Direction" not in df.columns:
        # try derive direction from sign of DeltaP% if present
        if "DeltaP%" in df.columns:
            df["Direction"] = df["DeltaP%"].apply(lambda x: "UP" if float(x) >= 0 else "DOWN")
        else:
            df["Direction"] = "NA"
    else:
        df["Direction"] = df["Direction"].astype(str).str.upper()
    df = df.sort_values(["StartDate", "EndDate"]).reset_index(drop=True)
    return df


def _filter_events(
    df: pd.DataFrame,
    date_from: Optional[pd.Timestamp],
    date_to: Optional[pd.Timestamp],
    mode: str,
    clip_to_window: bool,
) -> pd.DataFrame:
    if date_from is None and date_to is None:
        return df.copy()

    lo = date_from or df["StartDate"].min()
    hi = date_to or df["EndDate"].max()

    out = df.copy()

    if mode == "overlap":
        # keep anything that overlaps window
        m = (out["EndDate"] >= lo) & (out["StartDate"] <= hi)
        out = out.loc[m].copy()
        if clip_to_window:
            out["StartDate"] = out["StartDate"].clip(lower=lo)
            out["EndDate"] = out["EndDate"].clip(upper=hi)
            # ensure Start<=End
            out.loc[out["EndDate"] < out["StartDate"], "EndDate"] = out["StartDate"]
    elif mode == "within":
        # keep ONLY events fully within window
        m = (out["StartDate"] >= lo) & (out["EndDate"] <= hi)
        out = out.loc[m].copy()
        # clip doesn't matter here; keep as-is
    else:
        raise ValueError(f"Unknown filter mode: {mode}")

    out = out.sort_values(["StartDate", "EndDate"]).reset_index(drop=True)
    return out


def _overlap_score(a: Interval, b: Interval, metric: str) -> float:
    if metric == "iou":
        return a.iou(b)
    if metric == "pred_cov":
        return a.cov_pred(b)
    if metric == "real_cov":
        return a.cov_real(b)
    raise ValueError(f"Unknown overlap metric: {metric}")


def _match_greedy(
    pred: pd.DataFrame,
    real: pd.DataFrame,
    overlap_metric: str,
    min_overlap: float,
    max_mid_diff_days: int,
    require_same_dir: bool,
    allow_reuse_real: bool,
) -> Tuple[pd.DataFrame, List[int], List[int]]:
    # Precompute intervals
    pred_iv = [Interval(r.StartDate, r.EndDate) for r in pred.itertuples(index=False)]
    real_iv = [Interval(r.StartDate, r.EndDate) for r in real.itertuples(index=False)]

    candidates: List[Tuple[float, int, int, int, int]] = []
    # tuple: (-score, mid_abs_days, pred_i, real_j, lag_days)
    for i, pi in enumerate(pred_iv):
        pdir = str(pred.loc[i, "Direction"]).upper()
        for j, rj in enumerate(real_iv):
            rdir = str(real.loc[j, "Direction"]).upper()
            if require_same_dir and pdir != rdir:
                continue
            # mid difference constraint
            lag_days = int((pi.mid - rj.mid).days)
            mid_abs = abs(lag_days)
            if mid_abs > max_mid_diff_days:
                continue
            score = _overlap_score(pi, rj, overlap_metric)
            if score < min_overlap:
                continue
            candidates.append((-score, mid_abs, i, j, lag_days))

    # sort by score desc, then smaller mid_abs, then stable
    candidates.sort()

    used_pred = set()
    used_real = set()
    rows = []
    for neg_score, mid_abs, i, j, lag_days in candidates:
        if i in used_pred:
            continue
        if (not allow_reuse_real) and (j in used_real):
            continue
        used_pred.add(i)
        used_real.add(j)
        rows.append({
            "pred_i": i,
            "real_j": j,
            "score": float(-neg_score),
            "mid_abs_diff_days": int(mid_abs),
            "lag_days": int(lag_days),
            "pred_dir": str(pred.loc[i, "Direction"]).upper(),
            "real_dir": str(real.loc[j, "Direction"]).upper(),
            "pred_start": pred.loc[i, "StartDate"].date().isoformat(),
            "pred_end": pred.loc[i, "EndDate"].date().isoformat(),
            "real_start": real.loc[j, "StartDate"].date().isoformat(),
            "real_end": real.loc[j, "EndDate"].date().isoformat(),
        })

    matches = pd.DataFrame(rows)
    unmatched_pred = [i for i in range(len(pred)) if i not in used_pred]
    unmatched_real = [j for j in range(len(real)) if j not in used_real]
    return matches, unmatched_pred, unmatched_real


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-events", required=True, help="Pred events CSV path (StepD events from Pred_Close)")
    ap.add_argument("--real-events", required=True, help="Real events CSV path (StepD events from REALCLOSE)")
    ap.add_argument("--date-from", default=None, help="Window start (YYYY-MM-DD)")
    ap.add_argument("--date-to", default=None, help="Window end (YYYY-MM-DD)")
    ap.add_argument("--filter-mode", default="overlap", choices=["overlap", "within"], help="How to filter events by window")
    ap.add_argument("--clip-to-window", action="store_true", help="(overlap mode) Clip event boundaries to the window")
    ap.add_argument("--overlap-metric", default="iou", choices=["iou", "pred_cov", "real_cov"], help="Overlap scoring metric")
    ap.add_argument("--min-overlap", type=float, default=0.8, help="Minimum overlap score to consider a match")
    ap.add_argument("--max-mid-diff-days", type=int, default=10, help="Max abs(midpoint lag) allowed")
    ap.add_argument("--require-same-dir", action="store_true", help="Require Direction match (UP/DOWN)")
    ap.add_argument("--allow-reuse-real", action="store_true", help="Allow matching multiple pred to same real (not recommended)")

    args = ap.parse_args()

    pred_path = Path(args.pred_events)
    real_path = Path(args.real_events)
    if not pred_path.exists():
        raise SystemExit(f"pred-events not found: {pred_path}")
    if not real_path.exists():
        raise SystemExit(f"real-events not found: {real_path}")

    date_from = _parse_dt(args.date_from)
    date_to = _parse_dt(args.date_to)

    pred = _load_events(pred_path)
    real = _load_events(real_path)

    pred_f = _filter_events(pred, date_from, date_to, mode=args.filter_mode, clip_to_window=args.clip_to_window)
    real_f = _filter_events(real, date_from, date_to, mode=args.filter_mode, clip_to_window=args.clip_to_window)

    print(f"[filter] mode={args.filter_mode} date_from={date_from} date_to={date_to} clip_to_window={args.clip_to_window}")
    print(f"[events] pred={len(pred_f)} real={len(real_f)} overlap_metric={args.overlap_metric} min_overlap={args.min_overlap} "
          f"max_mid_diff_days={args.max_mid_diff_days} require_same_dir={args.require_same_dir} allow_reuse_real={args.allow_reuse_real}")
    print("")

    matches, unmatched_pred, unmatched_real = _match_greedy(
        pred_f,
        real_f,
        overlap_metric=args.overlap_metric,
        min_overlap=float(args.min_overlap),
        max_mid_diff_days=int(args.max_mid_diff_days),
        require_same_dir=bool(args.require_same_dir),
        allow_reuse_real=bool(args.allow_reuse_real),
    )

    if len(matches) > 0:
        print("[matches]")
        print(matches.to_string(index=False))
    else:
        print("[matches] none")

    if unmatched_pred:
        print("")
        print(f"[unmatched_pred] count={len(unmatched_pred)}")
        cols = [c for c in ["Direction", "StartDate", "EndDate", "Duration"] if c in pred_f.columns]
        print(pred_f.loc[unmatched_pred, cols].to_string(index=False))
    else:
        print("")
        print("[unmatched_pred] none")

    if unmatched_real:
        print("")
        print(f"[unmatched_real] count={len(unmatched_real)}")
        cols = [c for c in ["Direction", "StartDate", "EndDate", "Duration"] if c in real_f.columns]
        print(real_f.loc[unmatched_real, cols].to_string(index=False))
    else:
        print("")
        print("[unmatched_real] none")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
