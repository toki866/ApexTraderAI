#!/usr/bin/env python3
"""Best-effort evaluator for StepA/StepE CSV outputs.

The script NEVER raises a hard failure for workflow usage:
- Missing files/columns are reported as SKIP.
- Process exit code is always 0.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import traceback
from typing import Any

import numpy as np
import pandas as pd


def _find_first(pattern: str) -> str | None:
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else None


def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _parse_date(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    if date_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    return out


def _parse_key_value_summary(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {}

    keys = ["mode", "train_start", "train_end", "test_start", "test_end", "train_days", "test_days"]
    lower_cols = {c.lower(): c for c in df.columns}

    # Table-style summary with named columns.
    found = {k: df[lower_cols[k]].iloc[0] for k in keys if k in lower_cols}
    if found:
        return {k: (None if pd.isna(v) else str(v)) for k, v in found.items()}

    # key/value style summary.
    if len(df.columns) >= 2:
        out: dict[str, Any] = {}
        c0, c1 = df.columns[0], df.columns[1]
        for _, row in df.iterrows():
            k = str(row.get(c0, "")).strip().lower()
            v = row.get(c1)
            if k in keys:
                out[k] = None if pd.isna(v) else str(v)
        return out
    return {}


def _detect_pred_cols(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for c in df.columns:
        lc = c.lower()
        if "pred_" in lc or lc.endswith("_pred") or "close_pred" in lc or "delta_close_pred" in lc:
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)
    return cols


def _can_compare_close_scale(pred_col: str) -> bool:
    lc = pred_col.lower()
    if "delta" in lc:
        return False
    return "close" in lc or "pred_" in lc


def _calc_metrics(true_s: pd.Series, pred_s: pd.Series) -> dict[str, Any]:
    pair = pd.DataFrame({"t": pd.to_numeric(true_s, errors="coerce"), "p": pd.to_numeric(pred_s, errors="coerce")}).dropna()
    if pair.empty:
        return {"mae": None, "rmse": None, "corr": None, "dir_acc": None, "n_eval": 0}

    err = pair["p"] - pair["t"]
    dp = pair["p"].diff()
    dt = pair["t"].diff()
    valid_dir = pd.DataFrame({"dp": dp, "dt": dt}).dropna()
    dir_acc = None
    if not valid_dir.empty:
        dir_acc = float((np.sign(valid_dir["dp"]) == np.sign(valid_dir["dt"])).mean())

    corr = pair["t"].corr(pair["p"]) if len(pair) >= 2 else None
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt(np.square(err).mean())),
        "corr": (None if pd.isna(corr) else float(corr)),
        "dir_acc": dir_acc,
        "n_eval": int(len(pair)),
    }


def _status(non_null_ratio: float | None, coverage_ratio: float | None, pred_cols_found: bool) -> str:
    if not pred_cols_found:
        return "BAD"
    nn = non_null_ratio if non_null_ratio is not None else 0.0
    cov = coverage_ratio if coverage_ratio is not None else 0.0
    if nn >= 0.90 and cov >= 0.90:
        return "OK"
    if nn < 0.50:
        return "BAD"
    return "WARN"


def evaluate(output_root: str, mode: str, symbol: str) -> dict[str, Any]:
    report: dict[str, Any] = {
        "output_root": output_root,
        "mode": mode,
        "symbol": symbol,
        "stepA": {"status": "SKIP", "summary": "not evaluated", "details": {}},
        "stepB": {"status": "SKIP", "summary": "not evaluated", "rows": []},
        "stepE": {"status": "SKIP", "summary": "not evaluated", "rows": []},
    }

    # StepA
    prices_path = _find_first(os.path.join(output_root, "stepA", "*", f"stepA_prices_test_{symbol}.csv"))
    split_path = _find_first(os.path.join(output_root, "stepA", "*", f"stepA_split_summary_{symbol}.csv"))
    stepa_prices: pd.DataFrame | None = None

    try:
        if not prices_path:
            report["stepA"] = {"status": "SKIP", "summary": "stepA_prices_test file missing", "details": {}}
        else:
            px = _parse_date(_read_csv(prices_path), "Date")
            stepa_prices = px
            d = {
                "path": prices_path,
                "rows": int(len(px)),
                "cols": int(len(px.columns)),
                "date_start": str(px["Date"].dropna().min().date()) if "Date" in px.columns and px["Date"].notna().any() else None,
                "date_end": str(px["Date"].dropna().max().date()) if "Date" in px.columns and px["Date"].notna().any() else None,
                "date_monotonic_increasing": bool(px["Date"].is_monotonic_increasing) if "Date" in px.columns else None,
                "date_duplicates": int(px["Date"].duplicated().sum()) if "Date" in px.columns else None,
                "ohlcv_missing": {c: int(px[c].isna().sum()) for c in ["Open", "High", "Low", "Close", "Volume"] if c in px.columns},
            }
            report["stepA"] = {"status": "OK", "summary": "prices_test evaluated", "details": d}

        split_info: dict[str, Any] = {}
        if split_path:
            split_info = _parse_key_value_summary(_read_csv(split_path))
        else:
            split_info = {"status": "SKIP", "reason": "stepA_split_summary file missing"}
        report["stepA"]["split_summary_path"] = split_path
        report["stepA"]["split_summary"] = split_info
    except Exception as exc:
        report["stepA"] = {"status": "SKIP", "summary": f"exception: {exc}", "details": {"traceback": traceback.format_exc(limit=2)}}

    # StepB
    split_b_path = _find_first(os.path.join(output_root, "stepB", mode, f"stepB_split_summary_{symbol}.csv"))
    patterns = [
        f"stepB_pred_time_all_{symbol}.csv",
        f"stepB_pred_close_mamba_{symbol}.csv",
        f"stepB_pred_close_mamba_periodic_{symbol}.csv",
        f"stepB_pred_path_mamba_{symbol}.csv",
        f"stepB_pred_path_mamba_periodic_{symbol}.csv",
        f"stepB_pred_*_{symbol}.csv",
    ]

    files: list[str] = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(output_root, "stepB", mode, p)))
    files = sorted(set(files))

    if not files:
        report["stepB"] = {
            "status": "SKIP",
            "summary": "no stepB prediction files found",
            "split_summary_path": split_b_path,
            "rows": [],
        }
        return report

    stepb_rows: list[dict[str, Any]] = []
    for fpath in files:
        try:
            df = _read_csv(fpath)
            date_col = "Date" if "Date" in df.columns else ("Date_anchor" if "Date_anchor" in df.columns else None)
            if date_col:
                df = _parse_date(df, date_col)
            pred_cols = _detect_pred_cols(df)

            if not pred_cols:
                stepb_rows.append(
                    {
                        "file": os.path.basename(fpath),
                        "pred_col": None,
                        "rows": int(len(df)),
                        "non_null_count": 0,
                        "non_null_ratio": 0.0,
                        "first_valid_date": None,
                        "last_valid_date": None,
                        "coverage_ratio": 0.0,
                        "mae": None,
                        "rmse": None,
                        "corr": None,
                        "dir_acc": None,
                        "status": "BAD",
                        "reason": "prediction columns not found",
                    }
                )
                continue

            for col in pred_cols:
                total_rows = len(df)
                nn = int(df[col].notna().sum())
                nn_ratio = float(nn / total_rows) if total_rows else 0.0

                first_valid = None
                last_valid = None
                if date_col and nn > 0:
                    valid_dates = df.loc[df[col].notna(), date_col].dropna()
                    if not valid_dates.empty:
                        first_valid = str(valid_dates.iloc[0].date())
                        last_valid = str(valid_dates.iloc[-1].date())

                coverage = None
                merged: pd.DataFrame | None = None
                if stepa_prices is not None and "Date" in stepa_prices.columns and date_col:
                    left = stepa_prices[["Date"]].copy()
                    right = df[[date_col, col]].copy().rename(columns={date_col: "Date"})
                    merged = left.merge(right, on="Date", how="left")
                    coverage = float(merged[col].notna().mean()) if len(merged) else None

                mae = rmse = corr = dir_acc = None
                if _can_compare_close_scale(col):
                    true_s = None
                    if "Close_true" in df.columns:
                        true_s = df["Close_true"]
                        pred_s = df[col]
                    elif merged is not None and stepa_prices is not None and "Close" in stepa_prices.columns:
                        merged_true = merged.merge(stepa_prices[["Date", "Close"]], on="Date", how="left")
                        true_s = merged_true["Close"]
                        pred_s = merged_true[col]
                    else:
                        pred_s = None

                    if true_s is not None and pred_s is not None:
                        mm = _calc_metrics(true_s, pred_s)
                        mae, rmse, corr, dir_acc = mm["mae"], mm["rmse"], mm["corr"], mm["dir_acc"]

                stepb_rows.append(
                    {
                        "file": os.path.basename(fpath),
                        "pred_col": col,
                        "rows": int(total_rows),
                        "non_null_count": nn,
                        "non_null_ratio": nn_ratio,
                        "first_valid_date": first_valid,
                        "last_valid_date": last_valid,
                        "coverage_ratio": coverage,
                        "mae": mae,
                        "rmse": rmse,
                        "corr": corr,
                        "dir_acc": dir_acc,
                        "status": _status(nn_ratio, coverage, True),
                        "reason": None,
                    }
                )
        except Exception as exc:
            stepb_rows.append(
                {
                    "file": os.path.basename(fpath),
                    "pred_col": None,
                    "rows": 0,
                    "non_null_count": 0,
                    "non_null_ratio": None,
                    "first_valid_date": None,
                    "last_valid_date": None,
                    "coverage_ratio": None,
                    "mae": None,
                    "rmse": None,
                    "corr": None,
                    "dir_acc": None,
                    "status": "SKIP",
                    "reason": str(exc),
                }
            )

    overall = "OK" if any(r["status"] == "OK" for r in stepb_rows) else "WARN"
    report["stepB"] = {
        "status": overall,
        "summary": "stepB files evaluated",
        "split_summary_path": split_b_path,
        "rows": stepb_rows,
    }

    # StepE
    selected_mode_logs = sorted(glob.glob(os.path.join(output_root, "stepE", mode, "stepE_daily_log_*.csv")))
    step_e_daily_logs = list(selected_mode_logs)
    summary = "stepE daily logs evaluated"
    status_on_empty = "WARN"
    if not step_e_daily_logs:
        step_e_daily_logs = sorted(glob.glob(os.path.join(output_root, "stepE", "*", "stepE_daily_log_*.csv")))
        if step_e_daily_logs:
            summary = f"stepE daily logs found in other modes (requested_mode={mode})"
        else:
            summary = "no stepE daily logs found"

    if not step_e_daily_logs:
        report["stepE"] = {
            "status": status_on_empty,
            "summary": summary,
            "rows": [],
        }
    else:
        rows: list[dict[str, Any]] = []
        for fpath in step_e_daily_logs:
            try:
                df = _read_csv(fpath)
                rows.append({
                    "file": os.path.basename(fpath),
                    "rows": int(len(df)),
                    "columns": list(df.columns),
                    "status": "OK" if len(df) > 0 else "WARN",
                })
            except Exception as exc:
                rows.append({
                    "file": os.path.basename(fpath),
                    "rows": 0,
                    "columns": [],
                    "status": "SKIP",
                    "reason": str(exc),
                })

        report["stepE"] = {
            "status": "OK" if any(r.get("status") == "OK" for r in rows) else "WARN",
            "summary": summary,
            "rows": rows,
        }

    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# EVAL_REPORT")
    lines.append("")
    lines.append(f"- output_root: `{report.get('output_root')}`")
    lines.append(f"- mode: `{report.get('mode')}`")
    lines.append(f"- symbol: `{report.get('symbol')}`")
    lines.append("")

    stepa = report.get("stepA", {})
    lines.append("## StepA summary")
    lines.append(f"- status: **{stepa.get('status')}**")
    lines.append(f"- summary: {stepa.get('summary')}")
    d = stepa.get("details", {})
    if d:
        lines.append("- prices_test:")
        for k in ["rows", "cols", "date_start", "date_end", "date_monotonic_increasing", "date_duplicates"]:
            lines.append(f"  - {k}: {d.get(k)}")
        if d.get("ohlcv_missing"):
            lines.append("  - ohlcv_missing:")
            for k, v in d["ohlcv_missing"].items():
                lines.append(f"    - {k}: {v}")
    lines.append(f"- split_summary_path: {stepa.get('split_summary_path')}")
    lines.append(f"- split_summary: {json.dumps(stepa.get('split_summary', {}), ensure_ascii=False)}")
    lines.append("")

    stepb = report.get("stepB", {})
    lines.append("## StepB file-by-file")
    lines.append(f"- status: **{stepb.get('status')}**")
    lines.append(f"- summary: {stepb.get('summary')}")
    lines.append("")
    lines.append("| file | pred_col | rows | non_null_ratio | first_valid | last_valid | coverage_ratio | MAE | corr | dir_acc | status |")
    lines.append("|---|---|---:|---:|---|---|---:|---:|---:|---:|---|")
    for row in stepb.get("rows", []):
        lines.append(
            f"| {row.get('file')} | {row.get('pred_col')} | {row.get('rows')} | {row.get('non_null_ratio')} | "
            f"{row.get('first_valid_date')} | {row.get('last_valid_date')} | {row.get('coverage_ratio')} | "
            f"{row.get('mae')} | {row.get('corr')} | {row.get('dir_acc')} | {row.get('status')} |"
        )
        if row.get("reason"):
            lines.append(f"- note ({row.get('file')}): {row.get('reason')}")

    lines.append("")
    step_e = report.get("stepE", {})
    lines.append("## StepE daily logs")
    lines.append(f"- status: **{step_e.get('status')}**")
    lines.append(f"- summary: {step_e.get('summary')}")
    for row in step_e.get("rows", []):
        lines.append(f"- {row.get('file')}: rows={row.get('rows')} status={row.get('status')}")

    lines.append("")
    lines.append("Status rule: OK if non_null_ratio>=0.90 and coverage_ratio>=0.90; WARN otherwise; BAD if non_null_ratio<0.50 or pred cols missing.")
    lines.append("Best-effort mode: this evaluator writes SKIP/notes and always exits 0.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--mode", default="sim")
    ap.add_argument("--symbol", default="SOXL")
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_md)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)

    report = evaluate(args.output_root, args.mode, args.symbol)
    md = render_markdown(report)

    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(md)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
