#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diagnose "where the router loses" using ONLY existing router daily logs.

It produces:
  - router_diag_groups_all.csv        : all (phase, chosen_agent) group stats
  - router_diag_groups_topK_loss.csv  : worst K groups by loss contribution (sum of reward_next)
  - router_diag_rows_topK_loss.csv    : rows (dates) belonging to those topK loss groups
  - router_diag_worst_dates.csv       : overall worst N dates by reward_next
  - router_diag_group_<...>.csv       : per-group detailed rows (sorted by reward_next asc)

Typical usage (Windows CMD):
  python tools\\diagnose_router_misselection.py ^
    --router-log output\\stepF\\sim\\router_bandit\\router_daily_log_bandit_testA_SOXL.csv ^
    --top-k 10

You can pass multiple logs:
  python tools\\diagnose_router_misselection.py --router-log fileA.csv fileB.csv --top-k 10

Notes:
- This tool does NOT compute counterfactual "what-if you had chosen another agent",
  because router dynamics include hold constraints and state dependence.
  Instead, it pinpoints the realized loss contributors by (phase, chosen_agent),
  and lists the exact dates + features (agreement/size/ratio/etc.) for inspection.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List

import pandas as pd


def _infer_eval_label(path: Path) -> str:
    name = path.name.lower()
    if "train" in name:
        return "train"
    if "testa" in name or "test_a" in name or "test-a" in name:
        return "testA"
    if "testb" in name or "test_b" in name or "test-b" in name:
        return "testB"
    if "test" in name:
        return "test"
    return "unknown"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" not in df.columns:
        for cand in ["date", "DATE", "DateTime", "datetime", "timestamp", "Timestamp"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "Date"})
                break
    if "phase" not in df.columns:
        for cand in ["Phase", "PHASE"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "phase"})
                break
    if "chosen_agent" not in df.columns:
        for cand in ["ChosenAgent", "agent", "chosen", "CHOSEN_AGENT"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "chosen_agent"})
                break
    if "reward_next" not in df.columns:
        for cand in ["reward", "Reward", "pnl", "PnL", "rewardNext", "reward_nextday"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "reward_next"})
                break
    if "ret_next" not in df.columns:
        for cand in ["ret", "Ret", "return_next", "cc_next", "retNext"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "ret_next"})
                break
    return df


def _to_numeric(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _ensure_required(df: pd.DataFrame) -> None:
    missing = [c for c in ["Date", "phase", "chosen_agent", "reward_next"] if c not in df.columns]
    if missing:
        raise SystemExit(
            f"[ERROR] router-log missing required columns: {missing}\n"
            f"       available columns: {list(df.columns)}"
        )


def _load_router_logs(paths: List[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            raise SystemExit(f"[ERROR] file not found: {p}")
        df = pd.read_csv(path)
        df = _normalize_columns(df)
        _ensure_required(df)
        df["eval"] = _infer_eval_label(path)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    out["Date"] = out["Date"].astype(str)
    out["chosen_agent"] = out["chosen_agent"].astype(str)
    out["phase"] = pd.to_numeric(out["phase"], errors="coerce")

    _to_numeric(
        out,
        [
            "reward_next",
            "ret_next",
            "size",
            "ratio",
            "agent_ratio",
            "agreement_dist",
            "agreement_label",
            "equity",
            "action_chosen",
            "hold_counter",
            "pos_prev",
            "action_prev",
        ],
    )
    return out


def _safe_filename(s: str) -> str:
    s = re.sub(r"[^\w\-.]+", "_", s)
    return s[:150]


def _group_stats(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if "ratio" in df.columns:
        df = df.copy()
        df["_abs_ratio"] = df["ratio"].abs()
    else:
        df = df.copy()
        df["_abs_ratio"] = pd.NA

    agg = {"reward_next": ["count", "mean", "std", "sum", "min", "max"]}
    if "ret_next" in df.columns:
        agg["ret_next"] = ["mean", "std", "sum", "min", "max"]
    if "size" in df.columns:
        agg["size"] = ["mean", "std", "min", "max"]
    agg["_abs_ratio"] = ["mean", "std", "min", "max"]

    g = df.groupby(group_cols, dropna=False).agg(agg)
    g.columns = ["_".join([c for c in col if c]) for col in g.columns.to_flat_index()]
    g = g.reset_index()

    g = g.rename(columns={"reward_next_sum": "loss_contrib_sum_reward"})
    g = g.rename(
        columns={
            "reward_next_count": "count",
            "reward_next_mean": "mean_reward",
            "reward_next_std": "std_reward",
            "reward_next_min": "min_reward",
            "reward_next_max": "max_reward",
        }
    )
    return g


def _select_topk_loss_groups(stats: pd.DataFrame, top_k: int, min_count: int) -> pd.DataFrame:
    s = stats.copy()
    if "count" in s.columns:
        s = s[s["count"].fillna(0).astype(int) >= int(min_count)].copy()

    s = s.sort_values("loss_contrib_sum_reward", ascending=True)
    neg = s[s["loss_contrib_sum_reward"] < 0].copy()
    if len(neg) == 0:
        neg = s.copy()
    return neg.head(int(top_k)).reset_index(drop=True)


def _build_group_key(row: pd.Series, group_cols: List[str]) -> str:
    return "|".join(f"{c}={row.get(c)}" for c in group_cols)


def run(args: argparse.Namespace) -> int:
    df = _load_router_logs(args.router_log)

    group_cols = list(args.group_cols)
    for c in group_cols:
        if c not in df.columns:
            raise SystemExit(f"[ERROR] group column '{c}' not in router-log. cols={list(df.columns)}")

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.router_log[0]).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = _group_stats(df, group_cols=group_cols)
    stats_all_path = out_dir / "router_diag_groups_all.csv"
    stats.to_csv(stats_all_path, index=False, encoding="utf-8-sig")

    top = _select_topk_loss_groups(stats, top_k=args.top_k, min_count=args.min_count)
    top_path = out_dir / "router_diag_groups_topK_loss.csv"
    top.to_csv(top_path, index=False, encoding="utf-8-sig")

    top_keys = set(_build_group_key(r, group_cols) for _, r in top.iterrows())
    df["_group_key"] = df.apply(lambda r: _build_group_key(r, group_cols), axis=1)
    rows_top = df[df["_group_key"].isin(top_keys)].copy()

    preferred_cols = [
        "Date",
        "eval",
        "phase",
        "chosen_agent",
        "action_chosen",
        "reward_next",
        "ret_next",
        "agreement_dist",
        "agreement_label",
        "size",
        "agent_ratio",
        "ratio",
        "pos_prev",
        "action_prev",
        "hold_counter",
        "equity",
        "_group_key",
    ]
    cols = [c for c in preferred_cols if c in rows_top.columns]

    if args.include_extra_cols:
        extra = [c for c in rows_top.columns if c not in cols]
        cols = cols + extra

    rows_top = rows_top.sort_values(["_group_key", "reward_next", "Date"], ascending=[True, True, True])
    rows_top_path = out_dir / "router_diag_rows_topK_loss.csv"
    rows_top[cols].to_csv(rows_top_path, index=False, encoding="utf-8-sig")

    worst = df.sort_values(["reward_next", "Date"], ascending=[True, True]).head(int(args.worst_n)).copy()
    worst_path = out_dir / "router_diag_worst_dates.csv"
    worst_cols = [c for c in cols if c in worst.columns]
    worst[worst_cols].to_csv(worst_path, index=False, encoding="utf-8-sig")

    per_dir = out_dir / "router_diag_groups"
    per_dir.mkdir(parents=True, exist_ok=True)
    for _, gr in top.iterrows():
        key = _build_group_key(gr, group_cols)
        sub = df[df["_group_key"] == key].copy().sort_values(["reward_next", "Date"], ascending=[True, True])
        fname = _safe_filename(f"router_diag_group_{key}.csv")
        sub_path = per_dir / fname
        sub_cols = [c for c in cols if c in sub.columns]
        sub[sub_cols].to_csv(sub_path, index=False, encoding="utf-8-sig")

    print("[router_diag] OK")
    print(f"  loaded_rows: {len(df)}  files: {len(args.router_log)}")
    print(f"  out_dir: {out_dir}")
    print(f"  wrote: {stats_all_path.name}")
    print(f"  wrote: {top_path.name}")
    print(f"  wrote: {rows_top_path.name}")
    print(f"  wrote: {worst_path.name}")
    print(f"  wrote per-group: {per_dir}\\*.csv ({len(top)} files)")
    print("")
    print("[router_diag] topK loss groups (by sum reward_next):")
    show_cols = group_cols + ["count", "mean_reward", "loss_contrib_sum_reward"]
    show_cols = [c for c in show_cols if c in top.columns]
    print(top[show_cols].to_string(index=False))

    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--router-log",
        nargs="+",
        required=True,
        help="Path(s) to router_daily_log_*.csv (e.g., router_daily_log_bandit_testA_SOXL.csv).",
    )
    ap.add_argument(
        "--out-dir",
        default="",
        help="Output directory. Default: same directory as the first --router-log.",
    )
    ap.add_argument("--top-k", type=int, default=10, help="Top K worst (loss contributing) groups.")
    ap.add_argument("--min-count", type=int, default=1, help="Ignore groups with count < min-count.")
    ap.add_argument(
        "--group-cols",
        nargs="+",
        default=["phase", "chosen_agent"],
        help="Grouping columns (default: phase chosen_agent).",
    )
    ap.add_argument("--worst-n", type=int, default=50, help="Number of worst dates to export.")
    ap.add_argument(
        "--include-extra-cols",
        action="store_true",
        help="Include all extra columns from router-log in the exported rows files.",
    )
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
