#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

DATE_CANDIDATES = ["Date", "date", "Datetime", "datetime", "timestamp", "Timestamp"]
SPLIT_CANDIDATES = ["Split", "split", "dataset", "Dataset"]
RET_CANDIDATES = ["ret", "reward_next", "return", "daily_return", "pnl_ret", "strategy_return"]
EQUITY_CANDIDATES = ["equity", "Equity", "portfolio_value", "PortfolioValue", "account_value"]
RATIO_CANDIDATES = ["ratio", "position", "weight", "signal"]

FIXED_COLUMNS = [
    "step",
    "mode",
    "source_csv",
    "symbol",
    "test_days",
    "total_return_pct",
    "cagr_pct",
    "sharpe",
    "max_dd_pct",
    "win_rate",
    "avg_daily_ret",
    "vol_annual",
    "num_trades",
    "avg_hold_days",
    "status",
    "reason",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate metrics_summary.csv/.md from output step CSV files.")
    p.add_argument("--output-root", default="output")
    p.add_argument("--mode", default="sim")
    p.add_argument("--test-start", default="")
    return p.parse_args()


def pick_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    col_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        lc = c.lower()
        if lc in col_map:
            return col_map[lc]
    return None


def parse_symbol(path: Path) -> str:
    stem = path.stem
    return stem.split("_")[-1] if "_" in stem else ""


def find_files(output_root: Path, mode: str) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    for p in sorted((output_root / "stepE" / mode).glob("stepE_daily_log_*_*.csv")):
        found.append(("StepE", p))

    step_f_patterns = ["*router*csv", "*daily_log*csv", "*results*csv"]
    step_f_dir = output_root / "stepF" / mode
    for pat in step_f_patterns:
        for p in sorted(step_f_dir.glob(pat)):
            found.append(("StepF", p))

    for p in sorted((output_root / "stepA" / mode).glob("stepA_prices_test_*.csv")):
        found.append(("StepA", p))
    return found


def filter_test_split(df: pd.DataFrame, test_start: str) -> tuple[pd.DataFrame, str]:
    d = df.copy()
    split_col = pick_col(d, SPLIT_CANDIDATES)
    date_col = pick_col(d, DATE_CANDIDATES)

    if date_col:
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
        d = d.sort_values(date_col)

    if split_col is not None:
        split_vals = d[split_col].astype(str).str.strip().str.lower()
        if (split_vals == "test").any():
            return d[split_vals == "test"].copy(), "split=test"

    if test_start and date_col:
        ts = pd.to_datetime(test_start, errors="coerce")
        if pd.notna(ts):
            return d[d[date_col] >= ts].copy(), f"date>={test_start}"

    return d, "all_rows"


def safe_float(v: float) -> float | None:
    return None if not np.isfinite(v) else float(v)


def compute_avg_hold_days(ratio: pd.Series) -> float | None:
    vals = pd.to_numeric(ratio, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    mask = np.abs(vals) > 0.01
    if mask.size == 0 or not mask.any():
        return None
    lengths: list[int] = []
    run = 0
    for m in mask:
        if m:
            run += 1
        elif run > 0:
            lengths.append(run)
            run = 0
    if run > 0:
        lengths.append(run)
    if not lengths:
        return None
    return float(np.mean(lengths))


def compute_metrics(step: str, mode: str, path: Path, df: pd.DataFrame, test_start: str) -> dict[str, object]:
    out: dict[str, object] = {c: None for c in FIXED_COLUMNS}
    out.update({"step": step, "mode": mode, "source_csv": path.as_posix(), "symbol": parse_symbol(path), "status": "ok"})

    d, basis = filter_test_split(df, test_start)
    out["reason"] = basis

    ret_col = pick_col(d, RET_CANDIDATES)
    eq_col = pick_col(d, EQUITY_CANDIDATES)
    ratio_col = pick_col(d, RATIO_CANDIDATES)

    if d.empty:
        out["status"] = "no_rows"
        out["reason"] = f"{basis}; empty"
        return out

    rets = pd.to_numeric(d[ret_col], errors="coerce") if ret_col else pd.Series(dtype=float)
    equity = pd.to_numeric(d[eq_col], errors="coerce") if eq_col else pd.Series(dtype=float)

    if (rets.empty or rets.dropna().empty) and equity.notna().sum() >= 2:
        eq = equity.dropna().to_numpy(dtype=float)
        derived = np.diff(eq) / np.where(eq[:-1] == 0, np.nan, eq[:-1])
        rets = pd.Series(derived)
        if ret_col is None:
            out["reason"] = f"{basis}; ret=derived_from_equity"

    rets = pd.to_numeric(rets, errors="coerce").dropna()
    eq_clean = pd.to_numeric(equity, errors="coerce").dropna()

    out["test_days"] = int(len(d))
    if eq_clean.size >= 2 and eq_clean.iloc[0] != 0:
        total_ret = eq_clean.iloc[-1] / eq_clean.iloc[0] - 1.0
        out["total_return_pct"] = safe_float(total_ret * 100.0)
        peak = np.maximum.accumulate(eq_clean.to_numpy(dtype=float))
        dd = eq_clean.to_numpy(dtype=float) / np.where(peak == 0, np.nan, peak) - 1.0
        out["max_dd_pct"] = safe_float(np.nanmin(dd) * 100.0) if np.isfinite(dd).any() else None

    if rets.size > 0:
        mean_ret = float(np.nanmean(rets.to_numpy(dtype=float)))
        out["avg_daily_ret"] = safe_float(mean_ret)
        out["win_rate"] = safe_float(float(np.nanmean(rets.to_numpy(dtype=float) > 0)))
        if rets.size > 1:
            std = float(np.nanstd(rets.to_numpy(dtype=float), ddof=1))
            if std > 0 and np.isfinite(std):
                out["sharpe"] = safe_float(mean_ret / std * math.sqrt(252.0))
            out["vol_annual"] = safe_float(std * math.sqrt(252.0))

    if out.get("total_return_pct") is not None and out.get("test_days") and int(out["test_days"]) > 0:
        years = int(out["test_days"]) / 252.0
        total = float(out["total_return_pct"]) / 100.0
        if years > 0 and (1.0 + total) > 0:
            out["cagr_pct"] = safe_float(((1.0 + total) ** (1.0 / years) - 1.0) * 100.0)

    if ratio_col:
        ratio_num = pd.to_numeric(d[ratio_col], errors="coerce").fillna(0.0)
        out["num_trades"] = int((ratio_num.abs() > 0.01).sum())
        out["avg_hold_days"] = safe_float(compute_avg_hold_days(ratio_num))

    return out


def to_markdown_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    rows = [["" if pd.isna(v) else str(v) for v in row] for row in df.to_numpy()]
    widths = [len(c) for c in cols]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(val))

    def mk(vals: list[str]) -> str:
        return "| " + " | ".join(vals[i].ljust(widths[i]) for i in range(len(vals))) + " |"

    out = [mk(cols), "| " + " | ".join("-" * w for w in widths) + " |"]
    out.extend(mk(r) for r in rows)
    return "\n".join(out)


def make_plot(df: pd.DataFrame, path: Path, title: str, test_start: str) -> None:
    if plt is None:
        return
    date_col = pick_col(df, DATE_CANDIDATES)
    eq_col = pick_col(df, EQUITY_CANDIDATES)
    if not eq_col:
        return

    d, _ = filter_test_split(df, test_start)
    eq = pd.to_numeric(d[eq_col], errors="coerce")
    mask = eq.notna()
    if mask.sum() < 2:
        return

    x = pd.to_datetime(d[date_col], errors="coerce") if date_col else np.arange(mask.sum())
    x = x[mask] if date_col else x
    y = eq[mask]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(x, y, linewidth=1.3)
    ax.set_title(title)
    ax.set_xlabel("Date" if date_col else "Index")
    ax.set_ylabel(eq_col)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    report_dir = Path("reports")
    plots_dir = report_dir / "plots"
    report_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    files = find_files(output_root, args.mode)

    for step, path in files:
        try:
            df = pd.read_csv(path)
            row = compute_metrics(step, args.mode, path, df, args.test_start)
            rows.append(row)
            make_plot(df, plots_dir / f"{step.lower()}_{path.stem}.png", f"{step} {path.name}", args.test_start)
        except Exception as exc:
            rows.append(
                {
                    "step": step,
                    "mode": args.mode,
                    "source_csv": path.as_posix(),
                    "symbol": parse_symbol(path),
                    "status": "error",
                    "reason": str(exc),
                }
            )

    metrics_df = pd.DataFrame(rows)
    if metrics_df.empty:
        metrics_df = pd.DataFrame(columns=FIXED_COLUMNS)
    for c in FIXED_COLUMNS:
        if c not in metrics_df.columns:
            metrics_df[c] = None
    metrics_df = metrics_df[FIXED_COLUMNS]
    metrics_df.to_csv(report_dir / "metrics_summary.csv", index=False)

    if rows:
        md = "# Metrics Summary\n\n" + to_markdown_table(metrics_df.fillna("")) + "\n"
    else:
        md = "# Metrics Summary\n\nNo evaluation CSV found.\n\n- Searched: stepE_daily_log, stepF router/daily_log/results, stepA_prices_test\n"
    (report_dir / "metrics_summary.md").write_text(md, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
