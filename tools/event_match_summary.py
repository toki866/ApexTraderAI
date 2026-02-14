#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tools/event_match_summary.py

Purpose
-------
Summarize "envelope event compression" match quality between predicted events (3AI)
and REALCLOSE events within a specified window.

This is a standalone implementation (does NOT import compare_envelope_events.py),
so it remains stable even if compare_envelope_events.py output format changes.

Inputs (default paths)
----------------------
- REAL events:
    {output_root}/stepD/stepD_events_REALCLOSE_{symbol}.csv
- Pred events per agent:
    {output_root}/stepD/stepD_events_{AGENT}_{symbol}.csv
    where AGENT is one of: XSR, MAMBA, FED (you can pass any list via --agents)

Event CSV required columns
--------------------------
- Direction
- StartDate
- EndDate

Comparison protocol (defaults)
------------------------------
- filter_mode = "overlap"  (keep events overlapping [date_from, date_to])
- clip_to_window = True   (clip intervals to the window before matching)
- overlap_metric = "iou"  (intersection-over-union over calendar days)
- require_same_dir = True
- allow_reuse_real = False
- max_mid_diff_days = 10
- min_overlap = 0.8

Outputs
-------
- Prints a summary table to stdout
- Writes a CSV summary to:
    {output_root}/stepD/event_match_summary_{symbol}_{date_from}_{date_to}.csv

Notes
-----
- Duration/overlap are computed in *calendar days* based on StartDate/EndDate.
  (Trading-day IoU would require a trading calendar; for quick sanity this is enough.)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import pandas as pd
import numpy as np


@dataclass(frozen=True)
class Event:
    idx: int
    direction: str
    start: pd.Timestamp
    end: pd.Timestamp


def _load_events(path: str) -> List[Event]:
    df = pd.read_csv(path)

    # Column detection (keep strict but friendly)
    if "StartDate" not in df.columns or "EndDate" not in df.columns or "Direction" not in df.columns:
        raise SystemExit(
            f"[ERR] Event CSV missing required columns. "
            f"Need Direction/StartDate/EndDate. path={path} cols={list(df.columns)}"
        )

    df["StartDate"] = pd.to_datetime(df["StartDate"])
    df["EndDate"] = pd.to_datetime(df["EndDate"])

    out: List[Event] = []
    for i, row in df.iterrows():
        out.append(
            Event(
                idx=int(i),
                direction=str(row["Direction"]).strip().upper(),
                start=row["StartDate"],
                end=row["EndDate"],
            )
        )
    return out


def _clip_event(e: Event, win_start: pd.Timestamp, win_end: pd.Timestamp) -> Optional[Event]:
    s = max(e.start, win_start)
    t = min(e.end, win_end)
    if t < s:
        return None
    return Event(idx=e.idx, direction=e.direction, start=s, end=t)


def _mid(e: Event) -> pd.Timestamp:
    return e.start + (e.end - e.start) / 2


def _overlap_days(a: Event, b: Event) -> int:
    s = max(a.start, b.start)
    t = min(a.end, b.end)
    if t < s:
        return 0
    return int((t - s).days)


def _union_days(a: Event, b: Event) -> int:
    s = min(a.start, b.start)
    t = max(a.end, b.end)
    return max(0, int((t - s).days))


def _score(a: Event, b: Event, metric: str) -> float:
    inter = _overlap_days(a, b)
    if inter <= 0:
        return 0.0

    if metric == "iou":
        uni = _union_days(a, b)
        return 0.0 if uni <= 0 else inter / uni

    if metric == "pred_cov":
        denom = max(1, int((a.end - a.start).days))
        return inter / denom

    if metric == "real_cov":
        denom = max(1, int((b.end - b.start).days))
        return inter / denom

    raise SystemExit(f"[ERR] Unknown overlap_metric: {metric} (use iou/pred_cov/real_cov)")


def _filter_events(
    events: List[Event],
    win_start: pd.Timestamp,
    win_end: pd.Timestamp,
    mode: str,
    clip_to_window: bool,
) -> List[Event]:
    if mode not in ("overlap", "mid"):
        raise SystemExit(f"[ERR] filter_mode must be overlap or mid. got={mode}")

    out: List[Event] = []
    for e in events:
        if mode == "overlap":
            if e.end < win_start or e.start > win_end:
                continue
            if clip_to_window:
                ce = _clip_event(e, win_start, win_end)
                if ce is None:
                    continue
                out.append(ce)
            else:
                out.append(e)
        else:  # mid
            m = _mid(e)
            if win_start <= m <= win_end:
                if clip_to_window:
                    ce = _clip_event(e, win_start, win_end)
                    if ce is None:
                        continue
                    out.append(ce)
                else:
                    out.append(e)

    return out


def _match_events(
    pred: List[Event],
    real: List[Event],
    max_mid_diff_days: int,
    min_overlap: float,
    metric: str,
    require_same_dir: bool,
    allow_reuse_real: bool,
) -> Tuple[List[Tuple[int, int, float, int]], int, int]:
    """
    Returns:
      matches: list of (pred_i, real_j, score, lag_days)
      matched_pred_count
      matched_real_unique_count
    """
    if not pred or not real:
        return [], 0, 0

    pred_mid = [_mid(e) for e in pred]
    real_mid = [_mid(e) for e in real]

    candidates: List[Tuple[int, int, float, int]] = []
    for pi, pe in enumerate(pred):
        for rj, re in enumerate(real):
            if require_same_dir and pe.direction != re.direction:
                continue
            lag = int((pred_mid[pi] - real_mid[rj]).days)
            if abs(lag) > max_mid_diff_days:
                continue
            sc = _score(pe, re, metric)
            if sc < min_overlap:
                continue
            candidates.append((pi, rj, sc, lag))

    # Greedy assignment: best score first, then smaller mid diff
    candidates.sort(key=lambda x: (-x[2], abs(x[3])))

    used_pred = set()
    used_real = set()
    matches: List[Tuple[int, int, float, int]] = []

    for pi, rj, sc, lag in candidates:
        if pi in used_pred:
            continue
        if (not allow_reuse_real) and (rj in used_real):
            continue
        used_pred.add(pi)
        used_real.add(rj)
        matches.append((pi, rj, sc, lag))

    return matches, len(used_pred), len(used_real)


def _fmt_float(x: float) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "nan"
    return f"{x:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--symbol", required=True)

    ap.add_argument("--real-events", default=None, help="optional explicit path to REAL events CSV")
    ap.add_argument("--agents", default="MAMBA,FED,XSR", help="comma-separated agent names")

    ap.add_argument("--date-from", required=True)
    ap.add_argument("--date-to", required=True)
    ap.add_argument("--filter-mode", default="overlap", choices=["overlap", "mid"])
    ap.add_argument("--clip-to-window", action="store_true", default=False)

    ap.add_argument("--overlap-metric", default="iou", choices=["iou", "pred_cov", "real_cov"])
    ap.add_argument("--max-mid-diff-days", type=int, default=10)
    ap.add_argument("--min-overlap", type=float, default=0.8)
    ap.add_argument("--require-same-dir", action="store_true", default=False)
    ap.add_argument("--allow-reuse-real", action="store_true", default=False)

    ap.add_argument("--show-window-events", action="store_true", default=False)
    ap.add_argument("--show-matches", type=int, default=0, help="print first N matches per agent")
    ap.add_argument("--out-csv", default=None)

    args = ap.parse_args()

    win_start = pd.to_datetime(args.date_from)
    win_end = pd.to_datetime(args.date_to)

    out_root = args.output_root
    sym = args.symbol
    stepd_dir = os.path.join(out_root, "stepD")

    real_path = args.real_events or os.path.join(stepd_dir, f"stepD_events_REALCLOSE_{sym}.csv")
    if not os.path.exists(real_path):
        raise SystemExit(f"[ERR] real-events not found: {real_path}\n"
                         f"Run: python tools/generate_realclose_events.py --output-root {out_root} --symbol {sym} ...")

    agents = [a.strip() for a in args.agents.split(",") if a.strip()]
    if not agents:
        raise SystemExit("[ERR] --agents is empty")

    real_all = _load_events(real_path)
    real = _filter_events(real_all, win_start, win_end, mode=args.filter_mode, clip_to_window=args.clip_to_window)

    if args.show_window_events:
        print(f"[REAL] window events count={len(real)} path={real_path}")
        for e in real:
            print(f"  {e.direction:>4} {e.start.date()} -> {e.end.date()}  dur={(e.end-e.start).days}")

    rows: List[Dict[str, object]] = []
    for agent in agents:
        pred_path = os.path.join(stepd_dir, f"stepD_events_{agent.upper()}_{sym}.csv")
        if not os.path.exists(pred_path):
            # try lower-case agent name
            pred_path2 = os.path.join(stepd_dir, f"stepD_events_{agent.lower()}_{sym}.csv")
            if os.path.exists(pred_path2):
                pred_path = pred_path2
            else:
                print(f"[WARN] pred-events not found for agent={agent}: {pred_path}")
                rows.append({
                    "agent": agent,
                    "pred_events": 0,
                    "real_events": len(real),
                    "matched_pred": 0,
                    "matched_real_unique": 0,
                    "precision": np.nan,
                    "recall_unique": np.nan,
                    "lag_mean": np.nan,
                    "lag_median": np.nan,
                    "overlap_mean": np.nan,
                    "pred_path": pred_path,
                })
                continue

        pred_all = _load_events(pred_path)
        pred = _filter_events(pred_all, win_start, win_end, mode=args.filter_mode, clip_to_window=args.clip_to_window)

        if args.show_window_events:
            print(f"[{agent}] window events count={len(pred)} path={pred_path}")
            for e in pred:
                print(f"  {e.direction:>4} {e.start.date()} -> {e.end.date()}  dur={(e.end-e.start).days}")

        matches, matched_pred, matched_real_unique = _match_events(
            pred=pred,
            real=real,
            max_mid_diff_days=args.max_mid_diff_days,
            min_overlap=args.min_overlap,
            metric=args.overlap_metric,
            require_same_dir=args.require_same_dir,
            allow_reuse_real=args.allow_reuse_real,
        )

        precision = (matched_pred / len(pred)) if len(pred) else np.nan
        recall_unique = (matched_real_unique / len(real)) if len(real) else np.nan
        lags = [m[3] for m in matches]
        scores = [m[2] for m in matches]
        lag_mean = float(np.mean(lags)) if lags else np.nan
        lag_median = float(np.median(lags)) if lags else np.nan
        overlap_mean = float(np.mean(scores)) if scores else np.nan

        if args.show_matches > 0:
            print(f"[{agent}] matches (show first {args.show_matches})")
            for (pi, rj, sc, lag) in matches[: args.show_matches]:
                pe = pred[pi]
                re = real[rj]
                print(f"  sc={sc:.3f} lag={lag:+d}d  P:{pe.direction} {pe.start.date()}->{pe.end.date()}  "
                      f"R:{re.direction} {re.start.date()}->{re.end.date()}")

        rows.append({
            "agent": agent,
            "pred_events": len(pred),
            "real_events": len(real),
            "matched_pred": matched_pred,
            "matched_real_unique": matched_real_unique,
            "precision": precision,
            "recall_unique": recall_unique,
            "lag_mean": lag_mean,
            "lag_median": lag_median,
            "overlap_mean": overlap_mean,
            "pred_path": pred_path,
        })

    out_df = pd.DataFrame(rows)
    # Pretty print
    print("\n[event-match summary]")
    cols = ["agent","pred_events","real_events","matched_pred","matched_real_unique",
            "precision","recall_unique","lag_mean","lag_median","overlap_mean"]
    # format floats for printing
    print(out_df[cols].to_string(index=False, justify="left", float_format=lambda x: f"{x:.3f}"))

    out_csv = args.out_csv or os.path.join(stepd_dir, f"event_match_summary_{sym}_{args.date_from}_{args.date_to}.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"\n[wrote] {out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
