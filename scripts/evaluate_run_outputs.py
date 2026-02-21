#!/usr/bin/env python3
"""Best-effort evaluator for StepA/StepB/StepE run outputs.

Always exits with code 0 and writes markdown/json reports.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import traceback
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SectionResult:
    status: str
    summary: str
    details: dict[str, Any]


def _find_first(pattern: str) -> str | None:
    paths = sorted(glob.glob(pattern))
    return paths[0] if paths else None


def _find_all(patterns: list[str]) -> list[str]:
    out: list[str] = []
    for p in patterns:
        out.extend(glob.glob(p))
    return sorted(set(out))


def _safe_read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _to_datetime_col(df: pd.DataFrame, col: str = "Date") -> tuple[pd.DataFrame, dict[str, Any]]:
    info: dict[str, Any] = {}
    if col not in df.columns:
        info["date_column_present"] = False
        return df, info
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce")
    info["date_column_present"] = True
    info["date_parse_nat"] = int(out[col].isna().sum())
    return out, info


def _date_checks(df: pd.DataFrame, col: str = "Date") -> dict[str, Any]:
    if col not in df.columns:
        return {
            "date_column_present": False,
            "date_monotonic_increasing": False,
            "date_duplicates": None,
            "date_missing_days": None,
            "start": None,
            "end": None,
        }
    s = df[col]
    dt = s.dropna().sort_values()
    miss = None
    if len(dt) >= 2:
        full = pd.date_range(dt.iloc[0], dt.iloc[-1], freq="D")
        miss = int(len(full.difference(pd.DatetimeIndex(dt.unique()))))
    return {
        "date_column_present": True,
        "date_monotonic_increasing": bool(s.is_monotonic_increasing),
        "date_duplicates": int(s.duplicated().sum()),
        "date_missing_days": miss,
        "start": str(dt.iloc[0].date()) if len(dt) else None,
        "end": str(dt.iloc[-1].date()) if len(dt) else None,
    }


def _ohlcv_missing(df: pd.DataFrame) -> dict[str, int]:
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return {c: int(df[c].isna().sum()) for c in cols}


def _metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, Any]:
    z = pd.DataFrame({"t": y_true, "p": y_pred}).dropna()
    if z.empty:
        return {"n": 0, "mae": None, "rmse": None, "corr": None, "dir_acc": None}
    e = z["p"] - z["t"]
    d_pred = z["p"].diff()
    d_true = z["t"].diff()
    valid = pd.DataFrame({"dp": d_pred, "dt": d_true}).dropna()
    dir_acc = None
    if not valid.empty:
        dir_acc = float((np.sign(valid["dp"]) == np.sign(valid["dt"])).mean())
    return {
        "n": int(len(z)),
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(np.square(e)))),
        "corr": float(z["t"].corr(z["p"])) if len(z) >= 2 else None,
        "dir_acc": dir_acc,
    }


def evaluate_step_a(output_root: str, symbol: str) -> SectionResult:
    try:
        p_summary = _find_first(os.path.join(output_root, "stepA", "*", f"stepA_split_summary_{symbol}.csv"))
        p_test = _find_first(os.path.join(output_root, "stepA", "*", f"stepA_prices_test_{symbol}.csv"))
        if not p_summary or not p_test:
            miss = []
            if not p_summary:
                miss.append("stepA_split_summary")
            if not p_test:
                miss.append("stepA_prices_test")
            return SectionResult("SKIP", f"missing files: {', '.join(miss)}", {"missing": miss})

        s = _safe_read_csv(p_summary)
        t = _safe_read_csv(p_test)
        t, parse_info = _to_datetime_col(t, "Date")

        split_test_start = None
        split_test_end = None
        if {"test_start", "test_end"}.issubset(s.columns) and len(s) > 0:
            split_test_start = str(pd.to_datetime(s["test_start"].iloc[0], errors="coerce").date())
            split_test_end = str(pd.to_datetime(s["test_end"].iloc[0], errors="coerce").date())

        dck = _date_checks(t, "Date")
        align_ok = None
        if split_test_start and split_test_end and dck["start"] and dck["end"]:
            align_ok = bool(split_test_start == dck["start"] and split_test_end == dck["end"])

        details = {
            "paths": {"split_summary": p_summary, "prices_test": p_test},
            "split_test_start": split_test_start,
            "split_test_end": split_test_end,
            "test_rows": int(len(t)),
            "test_cols": int(len(t.columns)),
            "date_parse": parse_info,
            "date_checks": dck,
            "ohlcv_missing": _ohlcv_missing(t),
            "split_vs_test_alignment_ok": align_ok,
        }
        status = "OK" if align_ok is not False else "SUSPECT"
        return SectionResult(status, "evaluated stepA", details)
    except Exception as e:
        return SectionResult("SKIP", f"exception: {e}", {"traceback": traceback.format_exc(limit=2)})


def _detect_pred_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        lc = c.lower()
        if "pred" in lc or lc.startswith("close_pred"):
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)
    return cols


def evaluate_step_b(output_root: str, symbol: str, stepa_test_df: pd.DataFrame | None) -> SectionResult:
    try:
        p_summary = _find_first(os.path.join(output_root, "stepB", "*", f"stepB_split_summary_{symbol}.csv"))
        pred_files = _find_all(
            [
                os.path.join(output_root, "stepB", "*", f"stepB_pred_time_*_{symbol}.csv"),
                os.path.join(output_root, "stepB", "*", f"stepB_pred_*_{symbol}.csv"),
            ]
        )
        if not pred_files:
            return SectionResult("SKIP", "no stepB prediction files found", {"pred_files": []})

        summary_obj: dict[str, Any] = {
            "split_summary_path": p_summary,
            "prediction_files": [],
        }
        for pf in pred_files:
            item: dict[str, Any] = {"path": pf}
            try:
                df = _safe_read_csv(pf)
                df, parse_info = _to_datetime_col(df)
                item["date_parse"] = parse_info
                item["rows"] = int(len(df))
                item["cols"] = int(len(df.columns))
                item["date_checks"] = _date_checks(df)
                pred_cols = _detect_pred_cols(df)
                item["pred_cols"] = pred_cols
                item["pred_nan"] = {c: int(df[c].isna().sum()) for c in pred_cols}

                true_col = None
                for c in ["Close_true", "close_true", "Close"]:
                    if c in df.columns:
                        true_col = c
                        break

                merged = df
                if true_col is None and stepa_test_df is not None and "Date" in df.columns and "Date" in stepa_test_df.columns:
                    rhs = stepa_test_df[["Date", "Close"]].copy()
                    rhs = rhs.rename(columns={"Close": "Close_true_from_stepA"})
                    merged = df.merge(rhs, on="Date", how="left")
                    true_col = "Close_true_from_stepA"
                    item["close_true_source"] = "stepA_prices_test"
                elif true_col is not None:
                    item["close_true_source"] = true_col

                item["metrics"] = {}
                for pc in pred_cols:
                    if true_col is not None and true_col in merged.columns:
                        item["metrics"][pc] = _metrics(merged[true_col], merged[pc])
                    else:
                        item["metrics"][pc] = {
                            "n": 0,
                            "mae": None,
                            "rmse": None,
                            "corr": None,
                            "dir_acc": None,
                            "note": "Close_true unavailable",
                        }
            except Exception as ie:
                item["status"] = "SKIP"
                item["reason"] = str(ie)
            summary_obj["prediction_files"].append(item)

        any_metrics = any(bool(x.get("metrics")) for x in summary_obj["prediction_files"])
        status = "OK" if any_metrics else "SUSPECT"
        return SectionResult(status, "evaluated stepB prediction files", summary_obj)
    except Exception as e:
        return SectionResult("SKIP", f"exception: {e}", {"traceback": traceback.format_exc(limit=2)})


def _max_drawdown(equity: pd.Series) -> float | None:
    x = pd.to_numeric(equity, errors="coerce").dropna()
    if x.empty:
        return None
    run_max = x.cummax()
    dd = x / run_max - 1.0
    return float(dd.min())


def _sharpe(ret: pd.Series) -> float | None:
    r = pd.to_numeric(ret, errors="coerce").dropna()
    if len(r) < 2:
        return None
    sd = float(r.std(ddof=1))
    if sd == 0:
        return None
    return float((r.mean() / sd) * math.sqrt(252.0))


def _alignment_check(df: pd.DataFrame, price_df: pd.DataFrame | None) -> dict[str, Any]:
    needed = ["Date", "pos", "ret"]
    missing = [c for c in needed if c not in df.columns]
    if price_df is None:
        return {"status": "SKIP", "reason": "stepA test prices unavailable"}
    if missing:
        return {"status": "SKIP", "reason": f"missing columns: {missing}"}

    px_col = "Close_eff" if "Close_eff" in price_df.columns else ("Close" if "Close" in price_df.columns else None)
    if px_col is None:
        return {"status": "SKIP", "reason": "Close/Close_eff unavailable in stepA test prices"}

    p = price_df[["Date", px_col]].copy().rename(columns={px_col: "px"})
    p = p.sort_values("Date")
    p["cc_next"] = p["px"] / p["px"].shift(1) - 1.0

    m = df.merge(p[["Date", "cc_next"]], on="Date", how="left")
    m["implied_ret"] = pd.to_numeric(m["pos"], errors="coerce").shift(1) * pd.to_numeric(m["cc_next"], errors="coerce")

    if "cost" in m.columns:
        m["implied_ret"] = m["implied_ret"] - pd.to_numeric(m["cost"], errors="coerce")

    out: dict[str, Any] = {"status": "OK"}
    out["ret_vs_implied"] = _metrics(pd.to_numeric(m["ret"], errors="coerce"), pd.to_numeric(m["implied_ret"], errors="coerce"))

    if "reward" in m.columns:
        implied_reward = m["implied_ret"].copy()
        if "penalty" in m.columns:
            implied_reward = implied_reward - pd.to_numeric(m["penalty"], errors="coerce")
        out["reward_vs_implied"] = _metrics(pd.to_numeric(m["reward"], errors="coerce"), pd.to_numeric(implied_reward, errors="coerce"))

    ret_corr = out["ret_vs_implied"].get("corr")
    ret_mae = out["ret_vs_implied"].get("mae")
    ok = (ret_corr is not None and ret_corr >= 0.9) or (ret_mae is not None and ret_mae <= 1e-4)
    out["alignment_judgement"] = "OK" if ok else "SUSPECT"
    return out


def evaluate_step_e(output_root: str, mode: str, symbol: str, stepa_test_df: pd.DataFrame | None) -> SectionResult:
    try:
        paths = _find_all([os.path.join(output_root, "stepE", mode, f"stepE_daily_log_*_{symbol}.csv")])
        if not paths:
            return SectionResult("SKIP", "no stepE daily log files found", {"files": []})

        all_items: list[dict[str, Any]] = []
        for p in paths:
            item: dict[str, Any] = {"path": p}
            try:
                df = _safe_read_csv(p)
                df, parse_info = _to_datetime_col(df)
                item["date_parse"] = parse_info
                if "Date" in df.columns:
                    df = df.sort_values("Date")

                t = df[df["Split"].astype(str).str.lower().eq("test")].copy() if "Split" in df.columns else df.copy()
                if t.empty:
                    item["status"] = "SKIP"
                    item["reason"] = "no test rows"
                    all_items.append(item)
                    continue

                eq_col = "equity" if "equity" in t.columns else None
                ret_col = "ret" if "ret" in t.columns else None
                rw_col = "reward" if "reward" in t.columns else None

                item["test_days"] = int(len(t))
                item["test_start"] = str(t["Date"].dropna().iloc[0].date()) if "Date" in t.columns and t["Date"].notna().any() else None
                item["test_end"] = str(t["Date"].dropna().iloc[-1].date()) if "Date" in t.columns and t["Date"].notna().any() else None

                if eq_col:
                    eqs = pd.to_numeric(t[eq_col], errors="coerce").dropna()
                    if not eqs.empty:
                        item["equity_start"] = float(eqs.iloc[0])
                        item["equity_end"] = float(eqs.iloc[-1])
                        item["multiple"] = float(eqs.iloc[-1] / eqs.iloc[0]) if eqs.iloc[0] != 0 else None
                        item["max_drawdown"] = _max_drawdown(eqs)

                if ret_col:
                    item["sharpe"] = _sharpe(t[ret_col])

                if rw_col:
                    rw = pd.to_numeric(t[rw_col], errors="coerce").dropna()
                    item["reward_stats"] = {
                        "mean": float(rw.mean()) if not rw.empty else None,
                        "std": float(rw.std(ddof=1)) if len(rw) >= 2 else None,
                        "min": float(rw.min()) if not rw.empty else None,
                        "max": float(rw.max()) if not rw.empty else None,
                    }

                item["alignment"] = _alignment_check(t, stepa_test_df)
                item["status"] = "OK"
            except Exception as ie:
                item["status"] = "SKIP"
                item["reason"] = str(ie)
            all_items.append(item)

        has_ok = any(x.get("status") == "OK" for x in all_items)
        status = "OK" if has_ok else "SKIP"
        return SectionResult(status, "evaluated stepE logs", {"files": all_items})
    except Exception as e:
        return SectionResult("SKIP", f"exception: {e}", {"traceback": traceback.format_exc(limit=2)})


def _render_markdown(step_a: SectionResult, step_b: SectionResult, step_e: SectionResult, output_root: str, mode: str, symbol: str) -> str:
    lines: list[str] = []
    lines.append("# Run Output Evaluation Report")
    lines.append("")
    lines.append(f"- output_root: `{output_root}`")
    lines.append(f"- mode: `{mode}`")
    lines.append(f"- symbol: `{symbol}`")
    lines.append("")

    lines.append("## StepA Evaluation")
    lines.append(f"**Status: {step_a.status}** - {step_a.summary}")
    if step_a.status != "SKIP":
        d = step_a.details
        ck = d.get("date_checks", {})
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| test_rows | {d.get('test_rows')} |")
        lines.append(f"| test_cols | {d.get('test_cols')} |")
        lines.append(f"| test_start | {ck.get('start')} |")
        lines.append(f"| test_end | {ck.get('end')} |")
        lines.append(f"| date_monotonic_increasing | {ck.get('date_monotonic_increasing')} |")
        lines.append(f"| date_duplicates | {ck.get('date_duplicates')} |")
        lines.append(f"| date_missing_days | {ck.get('date_missing_days')} |")
        lines.append(f"| split_vs_test_alignment_ok | {d.get('split_vs_test_alignment_ok')} |")
        lines.append("")
        lines.append("OHLCV missing counts:")
        for k, v in d.get("ohlcv_missing", {}).items():
            lines.append(f"- {k}: {v}")
    else:
        lines.append("")
        lines.append(f"SKIP reason: {step_a.summary}")

    lines.append("")
    lines.append("## StepB Evaluation")
    lines.append(f"**Status: {step_b.status}** - {step_b.summary}")
    if step_b.status == "SKIP":
        lines.append("")
        lines.append(f"SKIP reason: {step_b.summary}")
    else:
        for i, it in enumerate(step_b.details.get("prediction_files", []), 1):
            lines.append("")
            lines.append(f"### StepB file {i}")
            lines.append(f"- path: `{it.get('path')}`")
            lines.append(f"- rows/cols: {it.get('rows')}/{it.get('cols')}")
            dck = it.get("date_checks", {})
            lines.append(f"- date_monotonic: {dck.get('date_monotonic_increasing')}, duplicates: {dck.get('date_duplicates')}")
            if it.get("status") == "SKIP":
                lines.append(f"- SKIP: {it.get('reason')}")
                continue
            lines.append(f"- pred_cols: {', '.join(it.get('pred_cols', [])) or '(none)'}")
            pred_nan = it.get("pred_nan", {})
            if pred_nan:
                lines.append("- pred NaN:")
                for c, n in pred_nan.items():
                    lines.append(f"  - {c}: {n}")
            metrics = it.get("metrics", {})
            if metrics:
                lines.append("")
                lines.append("| pred_col | n | MAE | RMSE | corr | dir_acc |")
                lines.append("|---|---:|---:|---:|---:|---:|")
                for c, m in metrics.items():
                    lines.append(
                        f"| {c} | {m.get('n')} | {m.get('mae')} | {m.get('rmse')} | {m.get('corr')} | {m.get('dir_acc')} |"
                    )

    lines.append("")
    lines.append("## StepE Reward/Return/Equity Evaluation")
    lines.append(f"**Status: {step_e.status}** - {step_e.summary}")
    if step_e.status == "SKIP":
        lines.append("")
        lines.append(f"SKIP reason: {step_e.summary}")
    else:
        for i, it in enumerate(step_e.details.get("files", []), 1):
            lines.append("")
            lines.append(f"### StepE file {i}")
            lines.append(f"- path: `{it.get('path')}`")
            lines.append(f"- status: {it.get('status')} {it.get('reason') or ''}")
            if it.get("status") != "OK":
                continue
            lines.append("| Metric | Value |")
            lines.append("|---|---|")
            for k in ["test_days", "test_start", "test_end", "equity_start", "equity_end", "multiple", "max_drawdown", "sharpe"]:
                if k in it:
                    lines.append(f"| {k} | {it.get(k)} |")
            if "reward_stats" in it:
                rs = it["reward_stats"]
                lines.append(f"| reward_mean | {rs.get('mean')} |")
                lines.append(f"| reward_std | {rs.get('std')} |")
                lines.append(f"| reward_min | {rs.get('min')} |")
                lines.append(f"| reward_max | {rs.get('max')} |")
            ali = it.get("alignment", {})
            lines.append("")
            lines.append(f"- alignment: **{ali.get('alignment_judgement', ali.get('status'))}**")
            if ali.get("status") == "SKIP":
                lines.append(f"- alignment SKIP reason: {ali.get('reason')}")
            else:
                rv = ali.get("ret_vs_implied", {})
                lines.append(f"- ret_vs_implied corr={rv.get('corr')} mae={rv.get('mae')}")
                rw = ali.get("reward_vs_implied")
                if rw:
                    lines.append(f"- reward_vs_implied corr={rw.get('corr')} mae={rw.get('mae')}")

    lines.append("")
    lines.append("---")
    lines.append("best-effort evaluator: errors in individual sections are reported as SKIP; process exit code is always 0.")
    return "\n".join(lines) + "\n"


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

    stepa_test_df: pd.DataFrame | None = None
    p_test = _find_first(os.path.join(args.output_root, "stepA", "*", f"stepA_prices_test_{args.symbol}.csv"))
    if p_test:
        try:
            stepa_test_df = _safe_read_csv(p_test)
            stepa_test_df, _ = _to_datetime_col(stepa_test_df)
        except Exception:
            stepa_test_df = None

    step_a = evaluate_step_a(args.output_root, args.symbol)
    step_b = evaluate_step_b(args.output_root, args.symbol, stepa_test_df)
    step_e = evaluate_step_e(args.output_root, args.mode, args.symbol, stepa_test_df)

    md = _render_markdown(step_a, step_b, step_e, args.output_root, args.mode, args.symbol)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(md)

    payload = {
        "output_root": args.output_root,
        "mode": args.mode,
        "symbol": args.symbol,
        "stepA": {"status": step_a.status, "summary": step_a.summary, "details": step_a.details},
        "stepB": {"status": step_b.status, "summary": step_b.summary, "details": step_b.details},
        "stepE": {"status": step_e.status, "summary": step_e.summary, "details": step_e.details},
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Best effort contract: never fail pipeline for evaluator issues.
        traceback.print_exc()
