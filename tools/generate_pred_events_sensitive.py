# -*- coding: utf-8 -*-
"""
generate_pred_events_sensitive.py

Purpose
-------
Generate StepD-style envelope events from predicted Close series for ONE agent,
with knobs to make the turning-point detection more/less sensitive.

Why this exists
---------------
Some agents can produce smoother predicted series and therefore
fewer turning points with the default StepD settings. This tool lets you
experiment with smaller RBW (rolling window), smaller min-gap, and optional
smoothing, and writes files in the same naming convention as StepD outputs:

  output/stepD/stepD_events_{AGENT}_{SYMBOL}.csv
  output/stepD/stepD_envelope_daily_{agent_lower}_{SYMBOL}.csv

Key parameters
--------------
- --rbw:
    Rolling window size used to detect local maxima/minima.
    Smaller => more sensitive (more turning points).
- --min-gap:
    Minimum days between adjacent turning points. Smaller => more sensitive.
- --min-move-pct:
    Minimum abs move% between turning points. Set 0.0 to disable.
- --smooth:
    Rolling-mean smoothing window applied BEFORE turning point detection.
    Set 1 to disable smoothing.

I/O
---
Input:
  {output_root}/stepB/stepB_pred_time_all_{symbol}.csv
    - Date
    - Pred_Close_MAMBA

Output:
  {output_root}/stepD/stepD_events_{AGENT}_{symbol}.csv
  {output_root}/stepD/stepD_envelope_daily_{agent_lower}_{symbol}.csv

Notes
-----
- Events ALWAYS cover the whole date range (head + between turning points + tail).
- This is a standalone tool for analysis/experimentation.
  If you want this behavior in the pipeline, patch StepDService similarly.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd


@dataclass
class StepDEvent:
    Direction: str
    StartDate: str
    EndDate: str
    Duration: int
    DeltaP_pct: float
    Theta_norm: float
    Top_pct: float
    Bottom_pct: float
    D_norm: float
    DeltaP: float
    D: float
    L: float
    Theta_deg: float
    Top_abs: float
    Bottom_abs: float


def _pick_pred_col(agent: str) -> str:
    a = agent.upper()
    if a in ("MAMBA", "LSTM"):
        return "Pred_Close_MAMBA"
    raise ValueError(f"Unsupported agent '{agent}'. Use MAMBA.")


def _compress_bool_runs(mask: np.ndarray) -> List[int]:
    """
    Convert runs of True into one representative index (middle of each run).
    """
    if mask.size == 0:
        return []
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    out: List[int] = []
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
            continue
        mid = (start + prev) // 2
        out.append(int(mid))
        start = i
        prev = i
    mid = (start + prev) // 2
    out.append(int(mid))
    return out


def _enforce_alternation(tps: List[Tuple[int, str]], values: np.ndarray) -> List[Tuple[int, str]]:
    """
    Ensure TP types alternate: top/bottom/top/bottom...
    If duplicates occur, keep the more extreme one.
    """
    if len(tps) <= 1:
        return tps
    out: List[Tuple[int, str]] = []
    for i, typ in tps:
        if not out:
            out.append((i, typ))
            continue
        pi, ptyp = out[-1]
        if typ != ptyp:
            out.append((i, typ))
            continue
        # same type -> keep more extreme
        if typ == "top":
            if float(values[i]) >= float(values[pi]):
                out[-1] = (i, typ)
        else:
            if float(values[i]) <= float(values[pi]):
                out[-1] = (i, typ)
    return out


def _apply_min_gap(
    tps: List[Tuple[int, str]],
    dates: pd.Series,
    values: np.ndarray,
    min_gap_days: int,
) -> List[Tuple[int, str]]:
    if min_gap_days <= 0 or len(tps) <= 1:
        return tps
    dates_ts = pd.to_datetime(dates).reset_index(drop=True)
    out: List[Tuple[int, str]] = []
    for i, typ in tps:
        if not out:
            out.append((i, typ))
            continue
        pi, ptyp = out[-1]
        try:
            gap = int((dates_ts.iloc[i] - dates_ts.iloc[pi]).days)
        except Exception:
            gap = min_gap_days
        if gap < min_gap_days:
            if typ == ptyp:
                # keep the more extreme
                if typ == "top":
                    if float(values[i]) >= float(values[pi]):
                        out[-1] = (i, typ)
                else:
                    if float(values[i]) <= float(values[pi]):
                        out[-1] = (i, typ)
            else:
                # different types but too close -> keep later (stable, simple)
                out[-1] = (i, typ)
            continue
        out.append((i, typ))
    return out


def _apply_min_move_pct(
    tps: List[Tuple[int, str]],
    values: np.ndarray,
    min_move_pct: float,
) -> List[Tuple[int, str]]:
    if min_move_pct <= 0.0 or len(tps) <= 1:
        return tps
    out: List[Tuple[int, str]] = [tps[0]]
    for i, typ in tps[1:]:
        pi, _ = out[-1]
        p0 = float(values[pi])
        p1 = float(values[i])
        if abs(p0) < 1e-12:
            move = abs(p1 - p0)
        else:
            move = abs((p1 - p0) / p0 * 100.0)
        if move < float(min_move_pct):
            continue
        out.append((i, typ))
    return out


def _make_event(dates: pd.Series, values: np.ndarray, i0: int, i1: int, rbw: int) -> StepDEvent:
    d0 = pd.to_datetime(dates.iloc[i0]).date()
    d1 = pd.to_datetime(dates.iloc[i1]).date()
    p0 = float(values[i0])
    p1 = float(values[i1])

    direction = "UP" if p1 >= p0 else "DOWN"
    duration = max(1, int((d1 - d0).days))
    delta_p = p1 - p0
    delta_p_pct = (delta_p / max(1e-12, abs(p0))) * 100.0

    top_abs = max(p0, p1)
    bottom_abs = min(p0, p1)
    mid = (top_abs + bottom_abs) / 2.0
    top_pct = ((top_abs - mid) / max(1e-12, abs(mid))) * 100.0
    bottom_pct = ((mid - bottom_abs) / max(1e-12, abs(mid))) * 100.0

    d_abs = float(duration)
    l_abs = float(abs(delta_p))
    d_norm = float(duration) / float(max(1, int(rbw)))

    theta_deg = math.degrees(math.atan2(delta_p, float(duration)))
    theta_norm = theta_deg / 90.0

    return StepDEvent(
        Direction=direction,
        StartDate=str(d0),
        EndDate=str(d1),
        Duration=int(duration),
        DeltaP_pct=float(delta_p_pct),
        Theta_norm=float(theta_norm),
        Top_pct=float(top_pct),
        Bottom_pct=float(bottom_pct),
        D_norm=float(d_norm),
        DeltaP=float(delta_p),
        D=float(d_abs),
        L=float(l_abs),
        Theta_deg=float(theta_deg),
        Top_abs=float(top_abs),
        Bottom_abs=float(bottom_abs),
    )


def _build_events(
    df: pd.DataFrame,
    series_col: str,
    *,
    rbw: int,
    min_gap: int,
    min_move_pct: float,
    smooth: int,
) -> Tuple[List[StepDEvent], pd.DataFrame, List[Tuple[int, str]]]:
    dates = df["Date"]
    s = pd.to_numeric(df[series_col], errors="coerce").astype(float).ffill().bfill()
    if smooth > 1:
        s = s.rolling(window=int(smooth), center=True, min_periods=1).mean()

    values = s.to_numpy()
    n = int(len(values))
    if n < 2:
        return [], pd.DataFrame(), []

    window = max(int(rbw), 3)
    roll_max = s.rolling(window=window, center=True, min_periods=1).max()
    roll_min = s.rolling(window=window, center=True, min_periods=1).min()
    is_top = (s >= roll_max - 1e-12).to_numpy()
    is_bot = (s <= roll_min + 1e-12).to_numpy()

    top_idx = _compress_bool_runs(is_top)
    bot_idx = _compress_bool_runs(is_bot)

    tps: List[Tuple[int, str]] = [(i, "top") for i in top_idx] + [(i, "bottom") for i in bot_idx]
    tps = sorted(set(tps), key=lambda x: x[0])
    tps = _enforce_alternation(tps, values)

    if min_gap > 0 and len(tps) >= 2:
        tps = _apply_min_gap(tps, dates, values, int(min_gap))
        tps = _enforce_alternation(tps, values)

    if min_move_pct > 0.0 and len(tps) >= 2:
        tps = _apply_min_move_pct(tps, values, float(min_move_pct))
        tps = _enforce_alternation(tps, values)

    events: List[StepDEvent] = []

    # If not enough turning points, produce one event
    if len(tps) < 2:
        ev = _make_event(dates, values, 0, n - 1, rbw=window)
        events = [ev]
        daily = _events_to_daily(df, events)
        return events, daily, tps

    # head coverage
    first_i = tps[0][0]
    if first_i > 0:
        events.append(_make_event(dates, values, 0, first_i, rbw=window))

    # between tps
    for (i0, _), (i1, _) in zip(tps[:-1], tps[1:]):
        if i1 <= i0:
            continue
        events.append(_make_event(dates, values, i0, i1, rbw=window))

    # tail coverage
    last_i = tps[-1][0]
    if last_i < n - 1:
        events.append(_make_event(dates, values, last_i, n - 1, rbw=window))

    daily = _events_to_daily(df, events)
    return events, daily, tps


def _events_to_daily(df: pd.DataFrame, events: List[StepDEvent]) -> pd.DataFrame:
    """
    Expand event-level features into daily rows by forward-filling.
    This matches the idea used in StepDService.
    """
    base = pd.DataFrame({"Date": pd.to_datetime(df["Date"])})
    out = base.copy()
    if not events:
        # still write with Date only (caller might prefer)
        return out

    # Prepare event DataFrame
    ev_df = pd.DataFrame([asdict(e) for e in events])
    ev_df["StartDate"] = pd.to_datetime(ev_df["StartDate"])
    ev_df["EndDate"] = pd.to_datetime(ev_df["EndDate"])

    # Assign event features to each date by overlap, then forward fill.
    # We do a simple scan because event count is small.
    feature_cols = [c for c in ev_df.columns if c not in ("StartDate", "EndDate")]
    for col in feature_cols:
        out[col] = np.nan

    dates = out["Date"].to_numpy()
    for _, row in ev_df.iterrows():
        m = (dates >= row["StartDate"].to_datetime64()) & (dates <= row["EndDate"].to_datetime64())
        for col in feature_cols:
            out.loc[m, col] = row[col]

    # Fill any gaps (head/tail) consistently
    out[feature_cols] = out[feature_cols].ffill().bfill()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", required=True, help="output root (e.g., output)")
    ap.add_argument("--symbol", required=True, help="symbol (e.g., SOXL)")
    ap.add_argument("--agent", required=True, help="agent: MAMBA")
    ap.add_argument("--rbw", type=int, default=10)
    ap.add_argument("--min-gap", type=int, default=3)
    ap.add_argument("--min-move-pct", type=float, default=0.0)
    ap.add_argument("--smooth", type=int, default=1)
    ap.add_argument("--pred-col", default="", help="override predicted column name")
    args = ap.parse_args()

    output_root = Path(args.output_root)
    sym = args.symbol
    agent = args.agent.upper()

    pred_csv = output_root / "stepB" / f"stepB_pred_time_all_{sym}.csv"
    if not pred_csv.exists():
        raise SystemExit(f"[ERROR] pred_csv not found: {pred_csv}")

    df = pd.read_csv(pred_csv)
    if "Date" not in df.columns:
        raise SystemExit(f"[ERROR] Date column not found in {pred_csv}")
    df["Date"] = pd.to_datetime(df["Date"])

    series_col = args.pred_col.strip() or _pick_pred_col(agent)
    if series_col not in df.columns:
        raise SystemExit(f"[ERROR] series_col '{series_col}' not found. Available: {list(df.columns)}")

    rbw = max(int(args.rbw), 3)
    min_gap = max(int(args.min_gap), 0)
    min_move_pct = float(args.min_move_pct)
    smooth = max(int(args.smooth), 1)

    print(f"[in] pred_csv={pred_csv}")
    print(f"[in] pred_col={series_col} rows={len(df)}")
    print(f"[params] rbw={rbw} min_gap={min_gap} min_move_pct={min_move_pct} smooth={smooth}")

    events, daily, tps = _build_events(
        df,
        series_col,
        rbw=rbw,
        min_gap=min_gap,
        min_move_pct=min_move_pct,
        smooth=smooth,
    )

    if tps:
        print(f"[turning_points] n={len(tps)} first={tps[0][0]}:{tps[0][1].upper()} last={tps[-1][0]}:{tps[-1][1].upper()}")
    else:
        print("[turning_points] n=0")

    out_dir = output_root / "stepD"
    out_dir.mkdir(parents=True, exist_ok=True)

    events_path = out_dir / f"stepD_events_{agent}_{sym}.csv"
    daily_path = out_dir / f"stepD_envelope_daily_{agent.lower()}_{sym}.csv"

    ev_df = pd.DataFrame([asdict(e) for e in events])
    ev_df.to_csv(events_path, index=False)

    daily.to_csv(daily_path, index=False)

    print(f"[events] rows={len(ev_df)} -> {events_path}")
    print(f"[daily] rows={len(daily)} -> {daily_path}")
    print("[wrote] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
