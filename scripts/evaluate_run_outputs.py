#!/usr/bin/env python3
"""Best-effort evaluator for StepA/StepB/StepE/StepF outputs.

Design goals:
- Never raise uncaught exceptions (workflow-safe).
- Always exit code 0.
- Write detailed markdown/json plus compact summary for issue posting.
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

MAX_SUMMARY_LINES = 200
MAX_SUMMARY_CHARS = 15000
MAX_LIST_ITEMS = 12


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


def _to_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        f = float(v)
        if np.isnan(f):
            return None
        return f
    except Exception:
        return None


def _to_int(v: Any) -> int | None:
    f = _to_float(v)
    return int(f) if f is not None else None


def _fmt(v: Any, nd: int = 4) -> str:
    if v is None:
        return "NA"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _status_level(s: str) -> int:
    return {"OK": 0, "WARN": 1, "BAD": 2}.get(s, 1)


def _calc_metrics(true_s: pd.Series, pred_s: pd.Series) -> dict[str, Any]:
    pair = pd.DataFrame({"t": pd.to_numeric(true_s, errors="coerce"), "p": pd.to_numeric(pred_s, errors="coerce")}).dropna()
    if pair.empty:
        return {"mae": None, "corr": None, "n_eval": 0}

    err = pair["p"] - pair["t"]
    corr = pair["t"].corr(pair["p"]) if len(pair) >= 2 else None
    return {
        "mae": float(np.abs(err).mean()),
        "corr": (None if pd.isna(corr) else float(corr)),
        "n_eval": int(len(pair)),
    }


def evaluate(output_root: str, mode: str, symbol: str) -> dict[str, Any]:
    report: dict[str, Any] = {
        "output_root": output_root,
        "mode": mode,
        "symbol": symbol,
        "stepA": {"status": "SKIP", "summary": "not evaluated", "details": {}},
        "stepB": {"status": "SKIP", "summary": "not evaluated", "rows": []},
        "stepE": {"status": "SKIP", "summary": "not evaluated", "rows": []},
        "stepF": {"status": "SKIP", "summary": "not evaluated", "rows": []},
        "overall_status": "WARN",
    }

    stepa_prices: pd.DataFrame | None = None

    # StepA
    try:
        prices_path = _find_first(os.path.join(output_root, "stepA", "*", f"stepA_prices_test_{symbol}.csv"))
        if not prices_path:
            report["stepA"] = {"status": "SKIP", "summary": "stepA_prices_test file missing", "details": {}}
        else:
            px = _parse_date(_read_csv(prices_path), "Date")
            stepa_prices = px
            d = {
                "path": prices_path,
                "test_rows": int(len(px)),
                "test_date_start": str(px["Date"].dropna().min().date()) if "Date" in px.columns and px["Date"].notna().any() else None,
                "test_date_end": str(px["Date"].dropna().max().date()) if "Date" in px.columns and px["Date"].notna().any() else None,
                "missing_ohlcv_count": int(
                    sum(int(px[c].isna().sum()) for c in ["Open", "High", "Low", "Close", "Volume"] if c in px.columns)
                ),
                "ohlcv_missing": {c: int(px[c].isna().sum()) for c in ["Open", "High", "Low", "Close", "Volume"] if c in px.columns},
            }
            report["stepA"] = {"status": "OK", "summary": "prices_test evaluated", "details": d}
    except Exception as exc:
        report["stepA"] = {"status": "SKIP", "summary": f"exception: {exc}", "details": {"traceback": traceback.format_exc(limit=2)}}

    # StepB
    try:
        patterns = [
            f"stepB_pred_time_all_{symbol}.csv",
            f"stepB_pred_close_*_{symbol}.csv",
            f"stepB_pred_path_*_{symbol}.csv",
            f"stepB_pred_*_{symbol}.csv",
        ]
        files: list[str] = []
        for p in patterns:
            files.extend(glob.glob(os.path.join(output_root, "stepB", mode, p)))
        files = sorted(set(files))

        if not files:
            report["stepB"] = {"status": "SKIP", "summary": "no stepB prediction files found", "rows": [], "files": []}
        else:
            rows: list[dict[str, Any]] = []
            key_cols = {"pred_close_mamba"}
            for fpath in files:
                try:
                    df = _read_csv(fpath)
                    date_col = "Date" if "Date" in df.columns else ("Date_anchor" if "Date_anchor" in df.columns else None)
                    if date_col:
                        df = _parse_date(df, date_col)

                    pred_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and ("pred_" in c.lower() or c.lower().endswith("_pred"))]
                    picked = [c for c in pred_cols if c.lower() in key_cols]
                    if not picked:
                        picked = pred_cols[:8]

                    for col in picked:
                        total_rows = len(df)
                        nn = int(df[col].notna().sum())
                        nn_ratio = float(nn / total_rows) if total_rows else 0.0
                        first_valid_date = None
                        if nn > 0 and date_col:
                            valid_dates = df.loc[df[col].notna(), date_col].dropna()
                            if not valid_dates.empty:
                                first_valid_date = str(valid_dates.iloc[0].date())

                        coverage_ratio = None
                        true_series = None
                        pred_series = None
                        if stepa_prices is not None and "Date" in stepa_prices.columns and date_col:
                            merged = stepa_prices[["Date", "Close"]].merge(
                                df[[date_col, col]].rename(columns={date_col: "Date"}), on="Date", how="left"
                            )
                            coverage_ratio = float(merged[col].notna().mean()) if len(merged) else None
                            true_series = merged["Close"]
                            pred_series = merged[col]
                        elif "Close_true" in df.columns:
                            true_series = df["Close_true"]
                            pred_series = df[col]

                        mae = corr = None
                        if true_series is not None and pred_series is not None:
                            mm = _calc_metrics(true_series, pred_series)
                            mae, corr = mm["mae"], mm["corr"]

                        status = "OK"
                        if nn_ratio < 0.5:
                            status = "BAD"
                        elif nn_ratio < 0.9 or (coverage_ratio is not None and coverage_ratio < 0.9):
                            status = "WARN"

                        rows.append(
                            {
                                "file": os.path.basename(fpath),
                                "pred_col": col,
                                "non_null_ratio": nn_ratio,
                                "first_valid_date": first_valid_date,
                                "coverage_ratio_over_test": coverage_ratio,
                                "mae": mae,
                                "corr": corr,
                                "status": status,
                            }
                        )
                except Exception as exc:
                    rows.append({"file": os.path.basename(fpath), "pred_col": None, "status": "SKIP", "reason": str(exc)})

            stepb_status = "SKIP" if not rows else max((r.get("status", "WARN") for r in rows), key=_status_level)
            report["stepB"] = {
                "status": stepb_status,
                "summary": "stepB files evaluated",
                "rows": rows,
                "files": [os.path.basename(p) for p in files],
            }
    except Exception as exc:
        report["stepB"] = {"status": "SKIP", "summary": f"exception: {exc}", "rows": []}

    # StepE
    try:
        step_e_logs = sorted(glob.glob(os.path.join(output_root, "stepE", mode, "stepE_daily_log_*.csv")))
        if not step_e_logs:
            step_e_logs = sorted(glob.glob(os.path.join(output_root, "stepE", "*", "stepE_daily_log_*.csv")))
        if not step_e_logs:
            report["stepE"] = {"status": "SKIP", "summary": "stepE_daily_log missing", "rows": []}
        else:
            rows = []
            for fpath in step_e_logs:
                try:
                    df = _read_csv(fpath)
                    row = {
                        "file": os.path.basename(fpath),
                        "test_days": int(len(df)),
                        "equity_multiple": _to_float(df["equity_multiple"].iloc[-1]) if "equity_multiple" in df.columns and len(df) else None,
                        "max_dd": _to_float(df["max_dd"].iloc[-1]) if "max_dd" in df.columns and len(df) else None,
                        "sharpe": _to_float(df["sharpe"].iloc[-1]) if "sharpe" in df.columns and len(df) else None,
                        "mean_reward": _to_float(df["reward"].mean()) if "reward" in df.columns and len(df) else None,
                        "status": "OK" if len(df) > 0 else "WARN",
                    }
                    rows.append(row)
                except Exception as exc:
                    rows.append({"file": os.path.basename(fpath), "status": "SKIP", "reason": str(exc)})
            report["stepE"] = {
                "status": "OK" if any(r.get("status") == "OK" for r in rows) else "WARN",
                "summary": "stepE daily logs evaluated",
                "rows": rows,
            }
    except Exception as exc:
        report["stepE"] = {"status": "SKIP", "summary": f"exception: {exc}", "rows": []}

    # StepF
    try:
        step_f_logs = sorted(glob.glob(os.path.join(output_root, "stepF", mode, "stepF_equity_marl*.csv")))
        if not step_f_logs:
            step_f_logs = sorted(glob.glob(os.path.join(output_root, "stepF", "*", "stepF_equity_marl*.csv")))
        if not step_f_logs:
            report["stepF"] = {"status": "SKIP", "summary": "stepF_equity_marl missing", "rows": []}
        else:
            rows = []
            for fpath in step_f_logs:
                try:
                    df = _read_csv(fpath)
                    row = {
                        "file": os.path.basename(fpath),
                        "test_days": int(len(df)),
                        "total_return": _to_float(df["total_return"].iloc[-1]) if "total_return" in df.columns and len(df) else None,
                        "max_drawdown": _to_float(df["max_drawdown"].iloc[-1]) if "max_drawdown" in df.columns and len(df) else None,
                        "sharpe": _to_float(df["sharpe"].iloc[-1]) if "sharpe" in df.columns and len(df) else None,
                        "status": "OK" if len(df) > 0 else "WARN",
                    }
                    rows.append(row)
                except Exception as exc:
                    rows.append({"file": os.path.basename(fpath), "status": "SKIP", "reason": str(exc)})
            report["stepF"] = {
                "status": "OK" if any(r.get("status") == "OK" for r in rows) else "WARN",
                "summary": "stepF equity logs evaluated",
                "rows": rows,
            }
    except Exception as exc:
        report["stepF"] = {"status": "SKIP", "summary": f"exception: {exc}", "rows": []}

    statuses = [report["stepA"]["status"], report["stepB"]["status"], report["stepE"]["status"], report["stepF"]["status"]]
    report["overall_status"] = "BAD" if "BAD" in statuses else ("WARN" if "WARN" in statuses or "SKIP" in statuses else "OK")
    return report


def render_markdown(report: dict[str, Any]) -> str:
    return "\n".join([
        "# EVAL_REPORT",
        "",
        f"- output_root: `{report.get('output_root')}`",
        f"- mode: `{report.get('mode')}`",
        f"- symbol: `{report.get('symbol')}`",
        f"- overall_status: **{report.get('overall_status')}**",
        "",
        "## Raw JSON",
        "```json",
        json.dumps(report, ensure_ascii=False, indent=2),
        "```",
        "",
        "Best-effort mode: this evaluator writes SKIP/notes and always exits 0.",
    ])


def render_summary(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"status={report.get('overall_status', 'WARN')}")

    stepa = report.get("stepA", {})
    ad = stepa.get("details", {})
    lines.append("StepA:")
    if stepa.get("status") == "SKIP":
        lines.append(f"  SKIP: {stepa.get('summary')}")
    else:
        lines.append(f"  test_rows={_fmt(ad.get('test_rows'))}")
        lines.append(f"  test_date_start={_fmt(ad.get('test_date_start'))} test_date_end={_fmt(ad.get('test_date_end'))}")
        lines.append(f"  missing_ohlcv_count={_fmt(ad.get('missing_ohlcv_count'))}")

    stepb = report.get("stepB", {})
    lines.append("StepB:")
    if stepb.get("status") == "SKIP":
        lines.append(f"  SKIP: {stepb.get('summary')}")
    else:
        files = stepb.get("files", [])
        show_files = files[:MAX_LIST_ITEMS]
        lines.append(f"  prediction_files_found={len(files)}")
        for fn in show_files:
            lines.append(f"    - {fn}")
        if len(files) > len(show_files):
            lines.append(f"    ... +{len(files) - len(show_files)} more")

        for r in stepb.get("rows", [])[:MAX_LIST_ITEMS * 2]:
            lines.append(
                "  "
                + f"{r.get('file')}::{r.get('pred_col')} nn_ratio={_fmt(r.get('non_null_ratio'))} "
                + f"first_valid_date={_fmt(r.get('first_valid_date'))} coverage_ratio_over_test={_fmt(r.get('coverage_ratio_over_test'))} "
                + f"mae={_fmt(r.get('mae'))} corr={_fmt(r.get('corr'))}"
            )

    stepe = report.get("stepE", {})
    lines.append("StepE:")
    if stepe.get("status") == "SKIP":
        lines.append(f"  SKIP: {stepe.get('summary')}")
    else:
        for r in stepe.get("rows", [])[:MAX_LIST_ITEMS]:
            lines.append(
                f"  {r.get('file')} test_days={_fmt(r.get('test_days'))} equity_multiple={_fmt(r.get('equity_multiple'))} "
                f"max_dd={_fmt(r.get('max_dd'))} sharpe={_fmt(r.get('sharpe'))} mean_reward={_fmt(r.get('mean_reward'))}"
            )

    stepf = report.get("stepF", {})
    lines.append("StepF:")
    if stepf.get("status") == "SKIP":
        lines.append(f"  SKIP: {stepf.get('summary')}")
    else:
        for r in stepf.get("rows", [])[:MAX_LIST_ITEMS]:
            lines.append(
                f"  {r.get('file')} test_days={_fmt(r.get('test_days'))} total_return={_fmt(r.get('total_return'))} "
                f"max_drawdown={_fmt(r.get('max_drawdown'))} sharpe={_fmt(r.get('sharpe'))}"
            )

    text = "\n".join(lines)
    sliced = text[:MAX_SUMMARY_CHARS]
    out_lines = sliced.splitlines()[:MAX_SUMMARY_LINES]
    return "\n".join(out_lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--mode", default="sim")
    ap.add_argument("--symbol", default="SOXL")
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-summary", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_md)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_summary)), exist_ok=True)

    try:
        report = evaluate(args.output_root, args.mode, args.symbol)
        md = render_markdown(report)
        summary = render_summary(report)
    except Exception as exc:
        report = {
            "output_root": args.output_root,
            "mode": args.mode,
            "symbol": args.symbol,
            "overall_status": "WARN",
            "error": str(exc),
            "traceback": traceback.format_exc(limit=4),
        }
        md = "# EVAL_REPORT\n\nEvaluator failed but continued in best-effort mode.\n\n```\n" + report["traceback"] + "\n```\n"
        summary = f"status=WARN\nStepA:\n  SKIP: evaluator exception={exc}\nStepB:\n  SKIP: evaluator exception\nStepE:\n  SKIP: evaluator exception\nStepF:\n  SKIP: evaluator exception"

    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(md)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(args.out_summary, "w", encoding="utf-8") as f:
        f.write(summary)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
    raise SystemExit(0)
