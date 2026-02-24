#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

DATE_CANDIDATES = ["Date", "date", "Datetime", "datetime", "timestamp", "Timestamp"]
SPLIT_CANDIDATES = ["Split", "split", "dataset", "Dataset"]
EQUITY_CANDIDATES = ["equity", "Equity", "portfolio_value", "PortfolioValue", "account_value"]
RET_CANDIDATES = ["ret", "return", "daily_return", "reward_next", "cc_next", "pnl_ret", "strategy_return"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Create public ONE_TAP markdown + metrics CSV from run outputs.")
    ap.add_argument("--output-root", default="output")
    ap.add_argument("--mode", default="sim")
    ap.add_argument("--symbols", default="SOXL,SOXS")
    ap.add_argument("--test-start", default="")
    ap.add_argument("--train-years", default="")
    ap.add_argument("--test-months", default="")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--sha", default="")
    ap.add_argument("--error-log", default="")
    ap.add_argument("--out-dir", default="public_report")
    ap.add_argument("--error-tail-lines", type=int, default=80)
    return ap.parse_args()


def find_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        lc = str(c).lower()
        if lc in lower:
            return lower[lc]
    return None


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def compute_metrics(df: pd.DataFrame, step: str, mode: str, source_csv: Path) -> dict:
    d = df.copy()
    date_col = find_column(d, DATE_CANDIDATES)
    if date_col is not None:
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
        d = d.sort_values(date_col)

    split_col = find_column(d, SPLIT_CANDIDATES)
    if split_col is not None:
        split_values = d[split_col].astype(str).str.strip().str.lower()
        if (split_values == "test").any():
            d = d[split_values == "test"]

    eq_col = find_column(d, EQUITY_CANDIDATES)
    ret_col = find_column(d, RET_CANDIDATES)
    equity = to_num(d[eq_col]).dropna().to_numpy(dtype=float) if eq_col else np.array([], dtype=float)
    rets = to_num(d[ret_col]).dropna().to_numpy(dtype=float) if ret_col else np.array([], dtype=float)

    if rets.size == 0 and equity.size >= 2:
        rets = np.diff(equity) / np.where(equity[:-1] == 0, np.nan, equity[:-1])
        rets = rets[np.isfinite(rets)]

    test_days = int(len(d))
    total_return_pct = np.nan
    max_drawdown = np.nan
    if equity.size >= 2 and np.isfinite(equity[0]) and equity[0] != 0:
        total_return_pct = float(equity[-1] / equity[0] - 1.0)
        peak = np.maximum.accumulate(equity)
        dd = equity / np.where(peak == 0, np.nan, peak) - 1.0
        if dd.size > 0 and np.isfinite(dd).any():
            max_drawdown = float(np.nanmin(dd))

    sharpe = np.nan
    avg_ret = np.nan
    std_ret = np.nan
    win_rate = np.nan
    if rets.size > 0:
        avg_ret = float(np.nanmean(rets))
        std_ret = float(np.nanstd(rets, ddof=1)) if rets.size > 1 else np.nan
        win_rate = float(np.nanmean(rets > 0))
        if rets.size > 1 and np.isfinite(std_ret) and std_ret > 0:
            sharpe = float(avg_ret / std_ret * math.sqrt(252.0))

    return {
        "step": step,
        "mode": mode,
        "source_csv": source_csv.as_posix(),
        "test_days": test_days,
        "total_return_pct": total_return_pct,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "avg_ret": avg_ret,
        "std_ret": std_ret,
    }


def discover_csvs(output_root: Path, step: str, mode: str, patterns: list[str]) -> list[Path]:
    results: list[Path] = []
    mode_dir = output_root / step / mode
    search_roots = [mode_dir] if mode_dir.exists() else []
    fallback = output_root / step
    if fallback.exists() and fallback not in search_roots:
        search_roots.append(fallback)
    for root in search_roots:
        for pat in patterns:
            results.extend(root.rglob(pat))
    uniq = sorted({p.resolve() for p in results if p.is_file()})
    return [Path(p) for p in uniq]


def to_md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no data)"
    cols = [str(c) for c in df.columns]
    rows = [["" if pd.isna(v) else str(v) for v in row] for row in df.to_numpy()]
    widths = [len(c) for c in cols]
    for r in rows:
        for i, v in enumerate(r):
            widths[i] = max(widths[i], len(v))

    def line(vals: list[str]) -> str:
        return "| " + " | ".join(v.ljust(widths[i]) for i, v in enumerate(vals)) + " |"

    out = [line(cols), "| " + " | ".join("-" * w for w in widths) + " |"]
    out.extend(line(r) for r in rows)
    return "\n".join(out)


def read_error_tail(path: str, n: int) -> str:
    if not path:
        return "No errors detected"
    p = Path(path)
    if not p.exists():
        return "No errors detected"
    try:
        lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return "No errors detected"
    if not lines:
        return "No errors detected"
    tail = lines[-n:]
    return "\n".join(tail).strip() or "No errors detected"


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    listed_files: list[str] = []

    stepe_files = discover_csvs(output_root, "stepE", args.mode, ["stepE_daily_log_*.csv"])
    for f in stepe_files:
        listed_files.append(f"StepE: {f.as_posix()}")
        try:
            rows.append(compute_metrics(pd.read_csv(f), "StepE", args.mode, f))
        except Exception as exc:
            rows.append({"step": "StepE", "mode": args.mode, "source_csv": f.as_posix(), "error": str(exc)})

    stepf_files = discover_csvs(output_root, "stepF", args.mode, ["stepF_*.csv", "*equity*.csv"])
    if not stepf_files:
        rows.append({"step": "StepF", "mode": args.mode, "source_csv": "not found", "status": "not found"})
    else:
        for f in stepf_files:
            listed_files.append(f"StepF: {f.as_posix()}")
            try:
                rows.append(compute_metrics(pd.read_csv(f), "StepF", args.mode, f))
            except Exception as exc:
                rows.append({"step": "StepF", "mode": args.mode, "source_csv": f.as_posix(), "error": str(exc)})

    metrics_df = pd.DataFrame(rows)
    metrics_path = out_dir / "metrics_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)

    md_lines = [
        "# ONE_TAP Public Run Report",
        "",
        f"- run_id: {args.run_id or '(unknown)'}",
        f"- sha: {args.sha or '(unknown)'}",
        f"- mode: {args.mode}",
        f"- symbols: {args.symbols}",
        f"- test_start: {args.test_start}",
        f"- train_years: {args.train_years}",
        f"- test_months: {args.test_months}",
        f"- output_root: {output_root.as_posix()}",
        "",
    ]

    for step_name in ["StepE", "StepF"]:
        step_df = metrics_df[metrics_df.get("step", "") == step_name].copy() if not metrics_df.empty else pd.DataFrame()
        md_lines.append(f"## {step_name} metrics")
        if step_df.empty:
            md_lines.append("not found")
        else:
            ordered = [c for c in ["step", "mode", "source_csv", "test_days", "total_return_pct", "max_drawdown", "sharpe", "win_rate", "avg_ret", "std_ret", "status", "error"] if c in step_df.columns]
            md_lines.append(to_md_table(step_df[ordered].fillna("")))
        md_lines.append("")

    md_lines.append("## Key CSV files")
    if listed_files:
        md_lines.extend([f"- {x}" for x in listed_files])
    else:
        md_lines.append("- No target CSV files found")
    md_lines.append("")

    md_lines.append("## Error summary")
    md_lines.append("```text")
    md_lines.append(read_error_tail(args.error_log, args.error_tail_lines))
    md_lines.append("```")

    (out_dir / "ONE_TAP.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
