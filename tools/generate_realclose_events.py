from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def _find_prices_csv(output_root: Path, symbol: str) -> Path:
    cands = [
        output_root / "stepA" / f"stepA_prices_{symbol}.csv",
        output_root / f"stepA_prices_{symbol}.csv",
    ]
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError(f"StepA prices CSV not found. searched={cands}")


def _rolling_extrema_indices(close: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return boolean masks (is_top, is_bottom) using centered rolling max/min."""
    s = pd.Series(close)
    roll_max = s.rolling(window=window, center=True, min_periods=window).max().to_numpy()
    roll_min = s.rolling(window=window, center=True, min_periods=window).min().to_numpy()
    is_top = np.isfinite(roll_max) & (close == roll_max)
    is_bottom = np.isfinite(roll_min) & (close == roll_min)
    return is_top, is_bottom


def _pick_alternating_turning_points(
    close: np.ndarray,
    is_top: np.ndarray,
    is_bottom: np.ndarray,
    min_gap: int,
) -> List[Tuple[int, str]]:
    """
    Build an alternating sequence of turning points: TOP/BOTTOM/TOP/...
    Resolve ties by taking the most extreme within a local cluster.
    """
    candidates: List[Tuple[int, str]] = []
    for i in range(len(close)):
        if is_top[i]:
            candidates.append((i, "TOP"))
        if is_bottom[i]:
            candidates.append((i, "BOTTOM"))
    candidates.sort(key=lambda x: x[0])
    if not candidates:
        return []

    # reduce dense clusters: keep most extreme per type within min_gap
    filtered: List[Tuple[int, str]] = []
    last_kept = -10**9
    for i, typ in candidates:
        if not filtered:
            filtered.append((i, typ))
            last_kept = i
            continue
        if i - last_kept < min_gap:
            pi, ptyp = filtered[-1]
            if ptyp == typ:
                if typ == "TOP":
                    if close[i] > close[pi]:
                        filtered[-1] = (i, typ)
                        last_kept = i
                else:
                    if close[i] < close[pi]:
                        filtered[-1] = (i, typ)
                        last_kept = i
            else:
                # different type too close: skip later candidate
                continue
        else:
            filtered.append((i, typ))
            last_kept = i

    # enforce alternation and keep most extreme when same type repeats
    turning: List[Tuple[int, str]] = []
    for i, typ in filtered:
        if not turning:
            turning.append((i, typ))
            continue
        pi, ptyp = turning[-1]
        if ptyp == typ:
            if typ == "TOP":
                if close[i] > close[pi]:
                    turning[-1] = (i, typ)
            else:
                if close[i] < close[pi]:
                    turning[-1] = (i, typ)
        else:
            turning.append((i, typ))

    return turning


def _build_events(
    dates: np.ndarray,
    close: np.ndarray,
    turning: List[Tuple[int, str]],
    rbw: int,
) -> pd.DataFrame:
    rows = []
    for k in range(len(turning) - 1):
        i0, t0 = turning[k]
        i1, t1 = turning[k + 1]
        if i1 <= i0:
            continue

        start_date = pd.to_datetime(dates[i0])
        end_date = pd.to_datetime(dates[i1])
        start_p = float(close[i0])
        end_p = float(close[i1])

        if t0 == "BOTTOM" and t1 == "TOP":
            direction = "UP"
        elif t0 == "TOP" and t1 == "BOTTOM":
            direction = "DOWN"
        else:
            direction = "UP" if end_p >= start_p else "DOWN"

        dp = end_p - start_p
        dp_pct = (dp / start_p) * 100.0 if start_p != 0 else np.nan

        L = int(i1 - i0)  # bar distance
        duration = L
        d_norm = float(L / rbw) if rbw > 0 else np.nan

        theta_rad = np.arctan2(dp, max(L, 1))
        theta_deg = float(theta_rad * 180.0 / np.pi)
        theta_norm = float(theta_deg / 90.0)

        top_abs = max(start_p, end_p)
        bottom_abs = min(start_p, end_p)
        top_pct = (top_abs / start_p) * 100.0 if start_p != 0 else np.nan
        bottom_pct = (bottom_abs / start_p) * 100.0 if start_p != 0 else np.nan

        rows.append({
            "EventId": k,
            "Direction": direction,
            "StartDate": start_date,
            "EndDate": end_date,
            "StartAbs": start_p,
            "EndAbs": end_p,
            "Duration": duration,
            "L": L,
            "ΔP": dp,
            "DeltaP%": dp_pct,
            "theta_deg": theta_deg,
            "theta_norm": theta_norm,
            "D_norm": d_norm,
            "Top_abs": top_abs,
            "Bottom_abs": bottom_abs,
            "Top%": top_pct,
            "Bottom%": bottom_pct,
            "StartType": t0,
            "EndType": t1,
        })

    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate REALCLOSE envelope-like events CSV from StepA prices (Close).")
    ap.add_argument("--output-root", required=True, help="e.g., output")
    ap.add_argument("--symbol", required=True, help="e.g., SOXL")
    ap.add_argument("--rbw", type=int, default=20, help="rolling window (RBW) for extrema detection (default: 20)")
    ap.add_argument("--min-gap", type=int, default=10, help="min gap (bars) between turning points (default: 10)")
    ap.add_argument("--out", default="", help="output events csv path (default: output/stepD/stepD_events_REALCLOSE_{sym}.csv)")
    args = ap.parse_args()

    out_root = Path(args.output_root)
    sym = args.symbol
    rbw = int(args.rbw)
    min_gap = int(args.min_gap)

    prices_path = _find_prices_csv(out_root, sym)
    df = pd.read_csv(prices_path)
    if "Date" not in df.columns or "Close" not in df.columns:
        raise RuntimeError(f"prices csv must contain Date and Close. columns={list(df.columns)}")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    dates = df["Date"].to_numpy()
    close = df["Close"].astype(float).to_numpy()

    is_top, is_bottom = _rolling_extrema_indices(close, window=rbw)
    turning = _pick_alternating_turning_points(close, is_top, is_bottom, min_gap=min_gap)
    # --- ensure tail coverage (append an anchor at the last row) ---
    # Without this, the final segment after the last turning point is dropped,
    # which can make REALCLOSE events end early (e.g., stopping at the last TOP/BOTTOM date).
    n = len(close)
    if len(turning) >= 1 and turning[-1][0] != (n - 1):
        # Keep the same turning-point type for the tail anchor; direction is inferred by price slope
        turning = list(turning) + [(n - 1, turning[-1][1])]
        turning = tuple(turning)
    if len(turning) < 2:
        raise RuntimeError(f"Not enough turning points found. turning_points={len(turning)}. Try smaller rbw/min-gap.")

    events = _build_events(dates, close, turning, rbw=rbw)

    out_path = Path(args.out) if args.out else (out_root / "stepD" / f"stepD_events_REALCLOSE_{sym}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(out_path, index=False)
    print(f"[wrote] {out_path} rows={len(events)} rbw={rbw} min_gap={min_gap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())