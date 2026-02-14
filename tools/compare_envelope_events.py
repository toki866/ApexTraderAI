from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def _find_col_regex(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
    for c in df.columns:
        cl = str(c).lower()
        for pat in patterns:
            if re.search(pat, cl):
                return c
    return None


def _parse_date_col(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_datetime(df[col], errors="coerce")


def _normalize_direction(s: pd.Series) -> pd.Series:
    def norm_one(x):
        if pd.isna(x):
            return np.nan
        t = str(x).strip().lower()
        if t in ("up", "u", "long", "buy", "1", "+1", "true"):
            return "UP"
        if t in ("down", "d", "short", "sell", "-1", "minus1", "false"):
            return "DOWN"
        if "上" in t:
            return "UP"
        if "下" in t:
            return "DOWN"
        return np.nan

    return s.apply(norm_one)


def _load_events(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"events csv is empty: {path}")

    dir_col = _find_col(df, ["Direction", "Dir", "Type", "Side"]) or _find_col_regex(df, [r"dir", r"direction", r"side", r"type"])
    start_col = _find_col(df, ["StartDate", "Start", "BeginDate", "Begin"]) or _find_col_regex(df, [r"start", r"begin"])
    end_col = _find_col(df, ["EndDate", "End", "FinishDate", "Finish"]) or _find_col_regex(df, [r"end", r"finish"])

    if start_col is None or end_col is None:
        raise RuntimeError(f"Cannot find start/end date columns in {path}. columns={list(df.columns)}")

    out = df.copy()
    out["_start"] = _parse_date_col(out, start_col)
    out["_end"] = _parse_date_col(out, end_col)
    if dir_col is not None:
        out["_dir"] = _normalize_direction(out[dir_col])
    else:
        out["_dir"] = np.nan

    out = out.dropna(subset=["_start", "_end"]).sort_values(["_start", "_end"]).reset_index(drop=True)
    out["_mid"] = out["_start"] + (out["_end"] - out["_start"]) / 2

    for cand in ["Duration", "L", "D_norm", "DeltaP%", "ΔP%", "ΔP", "theta_norm", "θ_norm", "theta_deg", "θ_deg"]:
        if cand in out.columns:
            out[cand] = pd.to_numeric(out[cand], errors="coerce")

    return out


def _filter_events_by_range(df: pd.DataFrame, date_from: Optional[pd.Timestamp], date_to: Optional[pd.Timestamp], mode: str) -> pd.DataFrame:
    if date_from is None and date_to is None:
        return df

    if date_from is None:
        date_from = pd.Timestamp.min
    if date_to is None:
        date_to = pd.Timestamp.max

    if mode == "overlap":
        m = (df["_end"] >= date_from) & (df["_start"] <= date_to)
    elif mode == "mid":
        m = (df["_mid"] >= date_from) & (df["_mid"] <= date_to)
    else:
        raise ValueError(f"Unknown filter mode: {mode}")
    return df.loc[m].copy().reset_index(drop=True)


def _clip_interval(a0, a1, date_from, date_to):
    if date_from is not None:
        a0 = max(a0, date_from)
    if date_to is not None:
        a1 = min(a1, date_to)
    return a0, a1


def _mid_of_interval(a0, a1):
    return a0 + (a1 - a0) / 2


def _overlap_score(a0, a1, b0, b1, metric: str) -> float:
    """
    metric:
      - iou: intersection / union
      - pred_cov: intersection / len(pred interval)   (a is pred)
      - real_cov: intersection / len(real interval)   (b is real)
    """
    inter0 = max(a0, b0)
    inter1 = min(a1, b1)
    if inter1 <= inter0:
        return 0.0
    inter = (inter1 - inter0).days
    if inter <= 0:
        return 0.0

    if metric == "iou":
        union0 = min(a0, b0)
        union1 = max(a1, b1)
        union = (union1 - union0).days
        if union <= 0:
            return 0.0
        return float(inter / union)

    a_len = max((a1 - a0).days, 1)
    b_len = max((b1 - b0).days, 1)

    if metric == "pred_cov":
        return float(inter / a_len)
    if metric == "real_cov":
        return float(inter / b_len)

    raise ValueError(f"Unknown overlap metric: {metric}")


def _match_best_for_each_pred(
    pred: pd.DataFrame,
    real: pd.DataFrame,
    max_mid_diff_days: int,
    min_overlap: float,
    overlap_metric: str,
    require_same_dir: bool,
    allow_reuse_real: bool,
    clip_to_window: bool,
    date_from: Optional[pd.Timestamp],
    date_to: Optional[pd.Timestamp],
) -> pd.DataFrame:
    """
    v4 change:
      - When clip_to_window is enabled AND a date window is provided, midpoint-difference gating is computed
        on *clipped* midpoints, not raw midpoints.
    """
    used_real = np.zeros(len(real), dtype=bool)
    rows = []

    for i, prow in pred.iterrows():
        p0, p1 = prow["_start"], prow["_end"]
        if clip_to_window and (date_from is not None or date_to is not None):
            p0, p1 = _clip_interval(p0, p1, date_from, date_to)
        p_mid = _mid_of_interval(p0, p1)

        diffs = []
        for _, rrow in real.iterrows():
            r0, r1 = rrow["_start"], rrow["_end"]
            if clip_to_window and (date_from is not None or date_to is not None):
                r0, r1 = _clip_interval(r0, r1, date_from, date_to)
            r_mid = _mid_of_interval(r0, r1)
            diffs.append(abs((r_mid - p_mid).days))
        diffs = pd.Series(diffs, index=real.index, dtype=float)

        if not allow_reuse_real:
            diffs[used_real] = np.inf

        cand_idx = np.where(diffs.values <= max_mid_diff_days)[0]
        if cand_idx.size == 0:
            rows.append((i, None, np.nan, np.nan, np.nan))
            continue

        if require_same_dir and pd.notna(prow["_dir"]):
            cand_idx = [j for j in cand_idx if real.loc[j, "_dir"] == prow["_dir"]]
            if len(cand_idx) == 0:
                rows.append((i, None, np.nan, np.nan, np.nan))
                continue

        best_j = None
        best_score = -1.0
        best_diff = np.inf
        best_ov = 0.0
        best_lag = np.nan

        for j in cand_idx:
            a0, a1 = prow["_start"], prow["_end"]
            b0, b1 = real.loc[j, "_start"], real.loc[j, "_end"]

            if clip_to_window and (date_from is not None or date_to is not None):
                a0, a1 = _clip_interval(a0, a1, date_from, date_to)
                b0, b1 = _clip_interval(b0, b1, date_from, date_to)
                if a1 <= a0 or b1 <= b0:
                    continue

            ov = _overlap_score(a0, a1, b0, b1, metric=overlap_metric)
            if ov < min_overlap:
                continue

            a_mid = _mid_of_interval(a0, a1)
            b_mid = _mid_of_interval(b0, b1)
            d = abs((b_mid - a_mid).days)
            score = ov - 0.001 * d
            if score > best_score:
                best_score = score
                best_j = j
                best_diff = d
                best_ov = ov
                best_lag = int((a_mid - b_mid).days)

        if best_j is None:
            rows.append((i, None, np.nan, np.nan, np.nan))
            continue

        if not allow_reuse_real:
            used_real[best_j] = True
        rows.append((i, int(best_j), int(best_diff), float(best_ov), int(best_lag)))

    return pd.DataFrame(rows, columns=["pred_i", "real_j", "mid_abs_diff_days", "overlap_score", "lag_days"])


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    m = ~(a.isna() | b.isna())
    if m.sum() < 10:
        return float("nan")
    return float(np.corrcoef(a[m].values, b[m].values)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare StepD envelope event tables: predicted vs real close (with optional date filtering).")
    ap.add_argument("--pred-events", required=True)
    ap.add_argument("--real-events", required=True)
    ap.add_argument("--max-mid-diff-days", type=int, default=5)
    ap.add_argument("--min-iou", type=float, default=0.0, help="minimum overlap threshold (name kept for compatibility).")
    ap.add_argument("--overlap-metric", choices=["iou", "pred_cov", "real_cov"], default="iou")
    ap.add_argument("--require-same-dir", action="store_true")
    ap.add_argument("--allow-reuse-real", action="store_true")
    ap.add_argument("--clip-to-window", action="store_true")
    ap.add_argument("--date-from", default="")
    ap.add_argument("--date-to", default="")
    ap.add_argument("--filter-mode", choices=["overlap", "mid"], default="overlap")
    ap.add_argument("--out-csv", default="")
    args = ap.parse_args()

    pred = _load_events(Path(args.pred_events))
    real = _load_events(Path(args.real_events))

    date_from = pd.to_datetime(args.date_from, errors="coerce") if args.date_from else None
    date_to = pd.to_datetime(args.date_to, errors="coerce") if args.date_to else None

    pred_f = _filter_events_by_range(pred, date_from, date_to, mode=args.filter_mode)
    real_f = _filter_events_by_range(real, date_from, date_to, mode=args.filter_mode)

    if (date_from is not None) or (date_to is not None):
        print(f"[filter] mode={args.filter_mode} date_from={date_from} date_to={date_to}")
        print(f"[filter] pred {len(pred)} -> {len(pred_f)} ; real {len(real)} -> {len(real_f)}")

    mdf = _match_best_for_each_pred(
        pred_f, real_f,
        max_mid_diff_days=args.max_mid_diff_days,
        min_overlap=args.min_iou,
        overlap_metric=args.overlap_metric,
        require_same_dir=args.require_same_dir,
        allow_reuse_real=args.allow_reuse_real,
        clip_to_window=args.clip_to_window,
        date_from=date_from,
        date_to=date_to,
    )

    matched_pred = int(mdf["real_j"].notna().sum())
    matched_real_unique = int(pd.Series(mdf.loc[mdf["real_j"].notna(), "real_j"].astype(int)).nunique()) if matched_pred else 0

    precision = matched_pred / max(len(pred_f), 1)
    recall_unique = matched_real_unique / max(len(real_f), 1)

    lags = mdf.loc[mdf["real_j"].notna(), "lag_days"].astype(int)
    lag_med = float(np.median(lags)) if len(lags) else float("nan")
    lag_mean = float(np.mean(lags)) if len(lags) else float("nan")

    print(f"[events] pred={len(pred_f)} real={len(real_f)} matched_pred={matched_pred} matched_real_unique={matched_real_unique}")
    print(f"[match] precision={precision:.3f} recall_unique={recall_unique:.3f} (max_mid_diff_days={args.max_mid_diff_days}, min_overlap={args.min_iou}, overlap_metric={args.overlap_metric}, require_same_dir={args.require_same_dir}, allow_reuse_real={args.allow_reuse_real}, clip_to_window={args.clip_to_window})")
    print(f"[lag] mean={lag_mean:.3f} days, median={lag_med:.3f} days (pred_mid - real_mid)")

    if matched_pred == 0:
        print("\n[matches] none")
        return 0

    merged_rows: List[Dict[str, Any]] = []
    for _, row in mdf.dropna(subset=["real_j"]).iterrows():
        i = int(row["pred_i"])
        j = int(row["real_j"])
        merged_rows.append({
            "pred_i": i,
            "real_j": j,
            "lag_days": int(row["lag_days"]),
            "mid_abs_diff_days": int(row["mid_abs_diff_days"]),
            "overlap_score": float(row["overlap_score"]),
            "pred_dir": pred_f.loc[i, "_dir"],
            "real_dir": real_f.loc[j, "_dir"],
            "pred_start": pred_f.loc[i, "_start"],
            "pred_end": pred_f.loc[i, "_end"],
            "real_start": real_f.loc[j, "_start"],
            "real_end": real_f.loc[j, "_end"],
        })
    pairs = pd.DataFrame(merged_rows)

    if args.out_csv:
        outp = Path(args.out_csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        pairs.to_csv(outp, index=False)
        print(f"[wrote] {outp}")
    else:
        print("\n[matches head]")
        print(pairs.head(30).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
