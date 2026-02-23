#!/usr/bin/env python3
"""Best-effort evaluator for StepA/StepB/D'/StepE/StepF outputs.

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
import re
import traceback
from typing import Any

import importlib
import importlib.util

import numpy as np
import pandas as pd

_MPL_SPEC = importlib.util.find_spec("matplotlib")
MPL_AVAILABLE = _MPL_SPEC is not None
if MPL_AVAILABLE:
    matplotlib = importlib.import_module("matplotlib")
    matplotlib.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")
else:
    matplotlib = None
    plt = None

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


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _calc_equity_metrics(df: pd.DataFrame, equity_col: str, ret_col: str, split_col: str | None) -> tuple[dict[str, Any], str | None]:
    work = df.copy()
    if split_col is None:
        reason = "Split missing: evaluated all rows as test"
        df_test = work
    else:
        df_test = work[work[split_col] == "test"]
        reason = None

    if df_test.empty:
        return {
            "test_days": 0,
            "equity_multiple": None,
            "max_dd": None,
            "mean_ret": None,
            "std_ret": None,
            "sharpe": None,
        }, "no test rows after split filter"

    eq = pd.to_numeric(df_test[equity_col], errors="coerce")
    rets = pd.to_numeric(df_test[ret_col], errors="coerce")
    eq = eq[np.isfinite(eq)]
    rets = rets[np.isfinite(rets)]
    if len(eq) < 1:
        return {
            "test_days": int(len(df_test)),
            "equity_multiple": None,
            "max_dd": None,
            "mean_ret": None,
            "std_ret": None,
            "sharpe": None,
        }, "equity has no numeric rows"

    eq_start, eq_end = float(eq.iloc[0]), float(eq.iloc[-1])
    eq_multiple = None if eq_start == 0 else (eq_end / eq_start)

    peak = eq.cummax()
    dd = eq / peak - 1.0
    max_dd = float(dd.min()) if len(dd) else None

    mean_ret = float(rets.mean()) if len(rets) >= 1 else None
    std_ret = float(rets.std(ddof=1)) if len(rets) >= 2 else None

    sharpe = None
    sharpe_reason = None
    if len(rets) >= 2:
        if std_ret is not None and std_ret > 0:
            sharpe = float((mean_ret or 0.0) / std_ret * np.sqrt(252.0))
        else:
            sharpe_reason = "sharpe NA: ret std is 0"
    else:
        sharpe_reason = "sharpe NA: ret rows < 2"

    reason_out = reason
    if sharpe_reason:
        reason_out = f"{reason_out}; {sharpe_reason}" if reason_out else sharpe_reason

    return {
        "test_days": int(len(df_test)),
        "equity_multiple": _to_float(eq_multiple),
        "max_dd": _to_float(max_dd),
        "mean_ret": _to_float(mean_ret),
        "std_ret": _to_float(std_ret),
        "sharpe": _to_float(sharpe),
    }, reason_out


def _calc_diversity(pos_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(pos_rows) < 2:
        return {
            "status": "SKIP",
            "summary": "need >=2 agents with numeric test position rows",
            "max_corr": None,
            "max_match_ratio": None,
            "pairs_over_0_9999": 0,
            "all_pairs": 0,
            "identical_all_agents": False,
        }

    pair_corrs: list[float] = []
    pair_match_ratios: list[float] = []
    over_09999 = 0
    valid_pairs = 0
    all_corr_one = True
    all_match_one = True

    for i in range(len(pos_rows)):
        for j in range(i + 1, len(pos_rows)):
            left = pos_rows[i]
            right = pos_rows[j]
            merged = left["series"].to_frame("a").join(right["series"].to_frame("b"), how="inner").dropna()
            if merged.empty:
                continue
            valid_pairs += 1
            if len(merged) < 2:
                corr = 1.0 if (merged["a"] == merged["b"]).all() else 0.0
            else:
                std_a = float(merged["a"].std(ddof=0))
                std_b = float(merged["b"].std(ddof=0))
                if std_a == 0.0 or std_b == 0.0:
                    corr = 1.0 if (merged["a"] == merged["b"]).all() else 0.0
                else:
                    corr = merged["a"].corr(merged["b"])
                    if corr is None or pd.isna(corr):
                        corr = 1.0 if (merged["a"] == merged["b"]).all() else 0.0
            corr = float(corr)
            pair_corrs.append(corr)
            if corr > 0.9999:
                over_09999 += 1
            if corr < 1.0:
                all_corr_one = False

            match_ratio = float((merged["a"] == merged["b"]).mean())
            pair_match_ratios.append(match_ratio)
            if match_ratio < 1.0:
                all_match_one = False

    if valid_pairs == 0:
        return {
            "status": "SKIP",
            "summary": "no overlapping numeric test position rows across agents",
            "max_corr": None,
            "max_match_ratio": None,
            "pairs_over_0_9999": 0,
            "all_pairs": 0,
            "identical_all_agents": False,
        }

    identical_all = all_corr_one and all_match_one
    high_corr_ratio = over_09999 / valid_pairs
    status = "OK"
    summary = "agent positions look diverse"
    if identical_all:
        status = "BAD"
        summary = "all agent test positions are identical"
    elif over_09999 >= 3 or high_corr_ratio >= 0.5:
        status = "WARN"
        summary = "many agent pairs are near-identical (corr>0.9999)"

    return {
        "status": status,
        "summary": summary,
        "max_corr": _to_float(max(pair_corrs) if pair_corrs else None),
        "max_match_ratio": _to_float(max(pair_match_ratios) if pair_match_ratios else None),
        "pairs_over_0_9999": int(over_09999),
        "all_pairs": int(valid_pairs),
        "identical_all_agents": bool(identical_all),
    }


def _save_empty_plot_notice(path: str, title: str, message: str) -> None:
    if not MPL_AVAILABLE or plt is None:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _generate_plots(output_root: str, mode: str, symbol: str, report: dict[str, Any], out_dir: str) -> list[str]:
    notes: list[str] = []
    plots: list[dict[str, Any]] = []
    os.makedirs(out_dir, exist_ok=True)

    if not MPL_AVAILABLE or plt is None:
        notes.append("PLOT backend unavailable: matplotlib is not installed")
        report["plots"] = {"items": plots, "notes": notes}
        return notes

    def record_plot(name: str, reason: str | None = None) -> str:
        path = os.path.join(out_dir, name)
        plots.append({"name": name, "path": path, "exists": os.path.exists(path), "reason": reason})
        return path

    # StepE equity overlay
    try:
        step_e_logs = sorted(glob.glob(os.path.join(output_root, "stepE", mode, f"stepE_daily_log_*_{symbol}.csv")))
        if not step_e_logs:
            step_e_logs = sorted(glob.glob(os.path.join(output_root, "stepE", "*", f"stepE_daily_log_*_{symbol}.csv")))
        path = record_plot("equity_stepE_topN.png")
        if not step_e_logs:
            _save_empty_plot_notice(path, "StepE topN equity", "stepE_daily_log files were not found")
            notes.append("PLOT StepE topN: stepE_daily_log files were not found")
        else:
            rows = report.get("stepE", {}).get("rows", [])
            ranked = [r for r in rows if r.get("equity_multiple") is not None]
            ranked = sorted(ranked, key=lambda x: float(x.get("equity_multiple") or -1), reverse=True)[:10]
            if not ranked:
                _save_empty_plot_notice(path, "StepE topN equity", "No StepE rows with numeric equity_multiple")
                notes.append("PLOT StepE topN: no numeric StepE equity_multiple rows")
            else:
                fig, ax = plt.subplots(figsize=(12, 6))
                plotted = 0
                for row in ranked:
                    agent = row.get("agent")
                    if not agent:
                        continue
                    hit = [x for x in step_e_logs if f"_{agent}_{symbol}.csv" in os.path.basename(x)]
                    if not hit:
                        continue
                    df = _read_csv(hit[0])
                    split_col = _pick_col(df, ["Split"])
                    equity_col = _pick_col(df, ["equity", "Equity"])
                    if equity_col is None:
                        continue
                    if split_col and split_col in df.columns:
                        df = df[df[split_col] == "test"]
                    eq = pd.to_numeric(df[equity_col], errors="coerce").dropna().reset_index(drop=True)
                    if eq.empty:
                        continue
                    ax.plot(eq.values, linewidth=1.5, label=str(agent))
                    plotted += 1
                if plotted == 0:
                    _save_empty_plot_notice(path, "StepE topN equity", "No plottable StepE test equity series")
                    notes.append("PLOT StepE topN: no plottable StepE test equity series")
                else:
                    ax.set_title("StepE topN test equity")
                    ax.set_xlabel("Test step")
                    ax.set_ylabel("Equity")
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc="best", fontsize=8)
                    fig.tight_layout()
                    fig.savefig(path, dpi=120)
                    plt.close(fig)
    except Exception as exc:
        notes.append(f"PLOT StepE topN exception: {exc}")

    # StepF equity
    try:
        path = record_plot("equity_stepF.png")
        step_f_logs = sorted(glob.glob(os.path.join(output_root, "stepF", mode, f"stepF_equity_marl_{symbol}.csv")))
        if not step_f_logs:
            step_f_logs = sorted(glob.glob(os.path.join(output_root, "stepF", "*", f"stepF_equity_marl_{symbol}.csv")))
        if not step_f_logs:
            _save_empty_plot_notice(path, "StepF equity", "stepF_equity_marl file was not found")
            notes.append("PLOT StepF equity: stepF_equity_marl file not found")
        else:
            df = _read_csv(step_f_logs[0])
            split_col = _pick_col(df, ["Split"])
            equity_col = _pick_col(df, ["equity", "Equity"])
            if equity_col is None:
                _save_empty_plot_notice(path, "StepF equity", "equity column missing")
                notes.append("PLOT StepF equity: equity column missing")
            else:
                if split_col and split_col in df.columns:
                    df = df[df[split_col] == "test"]
                eq = pd.to_numeric(df[equity_col], errors="coerce").dropna().reset_index(drop=True)
                if eq.empty:
                    _save_empty_plot_notice(path, "StepF equity", "No numeric StepF test equity rows")
                    notes.append("PLOT StepF equity: no numeric test equity rows")
                else:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(eq.values, color="black", linewidth=2.0, label="StepF")
                    ax.set_title("StepF test equity")
                    ax.set_xlabel("Test step")
                    ax.set_ylabel("Equity")
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc="best")
                    fig.tight_layout()
                    fig.savefig(path, dpi=120)
                    plt.close(fig)
    except Exception as exc:
        notes.append(f"PLOT StepF equity exception: {exc}")

    # StepE returns bar
    try:
        path = record_plot("bar_stepE_return.png")
        rows = [r for r in report.get("stepE", {}).get("rows", []) if r.get("equity_multiple") is not None and r.get("agent")]
        if not rows:
            _save_empty_plot_notice(path, "StepE equity multiple", "No numeric StepE equity_multiple rows")
            notes.append("PLOT StepE return bar: no numeric StepE rows")
        else:
            rows = sorted(rows, key=lambda x: float(x.get("equity_multiple") or -1), reverse=True)[:10]
            fig, ax = plt.subplots(figsize=(12, 6))
            agents = [str(r.get("agent")) for r in rows]
            values = [float(r.get("equity_multiple") or 0.0) for r in rows]
            ax.bar(agents, values, color="tab:blue")
            ax.set_title("StepE equity multiple (topN)")
            ax.set_xlabel("Agent")
            ax.set_ylabel("Equity multiple")
            ax.tick_params(axis="x", rotation=30)
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(path, dpi=120)
            plt.close(fig)
    except Exception as exc:
        notes.append(f"PLOT StepE return bar exception: {exc}")

    # StepE DD vs return scatter and StepF comparison note
    try:
        path = record_plot("scatter_stepE_dd_vs_ret.png")
        rows = [
            r
            for r in report.get("stepE", {}).get("rows", [])
            if r.get("equity_multiple") is not None and r.get("max_dd") is not None and r.get("agent")
        ]
        if not rows:
            _save_empty_plot_notice(path, "StepE max_dd vs return", "No numeric StepE (max_dd, equity_multiple) pairs")
            notes.append("PLOT StepE scatter: no numeric StepE DD/return pairs")
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            x = [float(r.get("max_dd")) for r in rows]
            y = [float(r.get("equity_multiple")) for r in rows]
            ax.scatter(x, y, color="tab:green", alpha=0.8)
            for r in rows[:10]:
                ax.annotate(str(r.get("agent")), (float(r.get("max_dd")), float(r.get("equity_multiple"))), fontsize=8)
            ax.set_title("StepE max drawdown vs equity multiple")
            ax.set_xlabel("max_dd")
            ax.set_ylabel("equity_multiple")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(path, dpi=120)
            plt.close(fig)

        stepf_rows = [r for r in report.get("stepF", {}).get("rows", []) if r.get("equity_multiple") is not None]
        best_stepe = max((float(r.get("equity_multiple")) for r in report.get("stepE", {}).get("rows", []) if r.get("equity_multiple") is not None), default=None)
        stepf_best = max((float(r.get("equity_multiple")) for r in stepf_rows), default=None)
        if best_stepe is not None and stepf_best is not None:
            relation = "勝ってる" if stepf_best >= best_stepe else "負けてる"
            notes.append(f"StepF_vs_best_StepE: StepF({stepf_best:.4f}) は best StepE({best_stepe:.4f}) に{relation}")
        else:
            notes.append("StepF_vs_best_StepE: 比較に必要な equity_multiple が不足")
    except Exception as exc:
        notes.append(f"PLOT StepE scatter exception: {exc}")

    for item in plots:
        item["exists"] = os.path.exists(item.get("path", ""))
    report["plots"] = {"items": plots, "notes": notes}
    return notes




def _collect_dprime_artifacts(output_root: str, mode: str, symbol: str) -> dict[str, Any]:
    base = os.path.join(output_root, "stepD_prime", mode)
    state_patterns = [
        os.path.join(base, f"stepDprime_state_*_{symbol}_train.csv"),
        os.path.join(base, f"stepDprime_state_*_{symbol}_test.csv"),
    ]
    emb_patterns = [
        os.path.join(base, "embeddings", f"stepDprime_*_{symbol}_embeddings*.csv"),
    ]

    state_files: list[str] = []
    emb_files: list[str] = []
    for pat in state_patterns:
        state_files.extend(sorted(glob.glob(pat)))
    for pat in emb_patterns:
        emb_files.extend(sorted(glob.glob(pat)))

    state_files = sorted(set(state_files))
    emb_files = sorted(set(emb_files))

    has_state = len(state_files) > 0
    has_embeddings = len(emb_files) > 0
    status = "OK" if has_state and has_embeddings else "WARN"
    missing = []
    if not has_state:
        missing.append("state")
    if not has_embeddings:
        missing.append("embeddings")
    summary = "D' state/embeddings found" if not missing else f"D' missing: {', '.join(missing)}"

    return {
        "status": status,
        "summary": summary,
        "details": {
            "state_count": len(state_files),
            "embeddings_count": len(emb_files),
            "state_files": [os.path.basename(x) for x in state_files],
            "embeddings_files": [os.path.basename(x) for x in emb_files],
            "searched": state_patterns + emb_patterns,
        },
    }

def evaluate(output_root: str, mode: str, symbol: str) -> dict[str, Any]:
    report: dict[str, Any] = {
        "output_root": output_root,
        "mode": mode,
        "symbol": symbol,
        "stepA": {"status": "SKIP", "summary": "not evaluated", "details": {}},
        "stepB": {"status": "SKIP", "summary": "not evaluated", "rows": []},
        "dprime": {"status": "SKIP", "summary": "not evaluated", "details": {}},
        "stepE": {"status": "SKIP", "summary": "not evaluated", "rows": []},
        "stepF": {"status": "SKIP", "summary": "not evaluated", "rows": []},
        "diversity": {"status": "SKIP", "summary": "not evaluated"},
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
            f"stepB_pred_close_mamba_{symbol}.csv",
            f"stepB_pred_path_mamba_{symbol}.csv",
            f"stepB_pred_close_wavelet_mamba_{symbol}.csv",
            f"stepB_pred_path_wavelet_mamba_{symbol}.csv",
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
            ignore_tokens = ("xsr", "fed")
            for fpath in files:
                try:
                    df = _read_csv(fpath)
                    date_col = "Date" if "Date" in df.columns else ("Date_anchor" if "Date_anchor" in df.columns else None)
                    if date_col:
                        df = _parse_date(df, date_col)

                    pred_cols = [
                        c
                        for c in df.columns
                        if pd.api.types.is_numeric_dtype(df[c])
                        and ("pred_" in c.lower() or c.lower().endswith("_pred"))
                        and not any(token in c.lower() for token in ignore_tokens)
                    ]
                    picked = [c for c in pred_cols if c.lower() in key_cols]
                    if not picked:
                        # StepB phase-1: evaluate only MAMBA-like prediction columns.
                        picked = [c for c in pred_cols if "mamba" in c.lower()]

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

    # D' (StepD prime artifacts only)
    try:
        report["dprime"] = _collect_dprime_artifacts(output_root=output_root, mode=mode, symbol=symbol)
    except Exception as exc:
        report["dprime"] = {"status": "WARN", "summary": f"exception: {exc}", "details": {}}

    # StepE
    try:
        step_e_logs = sorted(glob.glob(os.path.join(output_root, "stepE", mode, f"stepE_daily_log_*_{symbol}.csv")))
        if not step_e_logs:
            step_e_logs = sorted(glob.glob(os.path.join(output_root, "stepE", "*", f"stepE_daily_log_*_{symbol}.csv")))
        if not step_e_logs:
            report["stepE"] = {"status": "SKIP", "summary": "stepE_daily_log missing", "rows": []}
        else:
            rows = []
            pos_rows: list[dict[str, Any]] = []
            for fpath in step_e_logs:
                try:
                    df = _read_csv(fpath)
                    fname = os.path.basename(fpath)
                    m = re.match(rf"stepE_daily_log_(.+)_{re.escape(symbol)}\.csv$", fname)
                    agent = m.group(1) if m else fname
                    pos_col = _pick_col(df, ["pos", "Position"])
                    ret_col = _pick_col(df, ["ret"])
                    equity_col = _pick_col(df, ["equity"])
                    split_col = _pick_col(df, ["Split"])

                    missing_reasons = []
                    if pos_col is None:
                        missing_reasons.append("pos/Position missing")
                    if ret_col is None:
                        missing_reasons.append("ret missing")
                    if equity_col is None:
                        missing_reasons.append("equity missing")
                    if split_col is None:
                        missing_reasons.append("Split missing")
                    if missing_reasons:
                        rows.append({"file": fname, "agent": agent, "status": "SKIP", "reason": ", ".join(missing_reasons)})
                        continue

                    metrics, reason = _calc_equity_metrics(df, equity_col=equity_col, ret_col=ret_col, split_col=split_col)
                    row = {
                        "file": fname,
                        "agent": agent,
                        **metrics,
                        "status": "OK" if metrics["test_days"] > 0 else "WARN",
                    }
                    if reason:
                        row["note"] = reason
                    rows.append(row)

                    test_pos = df[df[split_col] == "test"][pos_col] if split_col in df.columns else pd.Series(dtype=float)
                    test_pos = pd.to_numeric(test_pos, errors="coerce").dropna().reset_index(drop=True)
                    if not test_pos.empty:
                        pos_rows.append({"agent": agent, "series": test_pos})
                except Exception as exc:
                    rows.append({"file": os.path.basename(fpath), "status": "SKIP", "reason": str(exc)})

            report["diversity"] = _calc_diversity(pos_rows)
            rows = sorted(rows, key=lambda r: (0 if r.get("status") == "OK" else 1, str(r.get("agent", ""))))
            report["stepE"] = {
                "status": "OK" if any(r.get("status") == "OK" for r in rows) else "SKIP",
                "summary": "stepE daily logs evaluated",
                "rows": rows,
            }
    except Exception as exc:
        report["stepE"] = {"status": "SKIP", "summary": f"exception: {exc}", "rows": []}

    # StepF
    try:
        step_f_logs = sorted(glob.glob(os.path.join(output_root, "stepF", mode, f"stepF_equity_marl_{symbol}.csv")))
        if not step_f_logs:
            step_f_logs = sorted(glob.glob(os.path.join(output_root, "stepF", "*", f"stepF_equity_marl_{symbol}.csv")))
        if not step_f_logs:
            report["stepF"] = {"status": "SKIP", "summary": "stepF_equity_marl missing", "rows": []}
        else:
            rows = []
            for fpath in step_f_logs:
                try:
                    df = _read_csv(fpath)
                    equity_col = _pick_col(df, ["equity", "Equity"])
                    if equity_col is None:
                        rows.append({"file": os.path.basename(fpath), "status": "SKIP", "reason": "equity/Equity missing"})
                        continue
                    ret_col = _pick_col(df, ["ret"])
                    note = None
                    if ret_col is None:
                        work = df.copy()
                        work["_ret_eval"] = pd.to_numeric(work[equity_col], errors="coerce").pct_change()
                        df = work
                        ret_col = "_ret_eval"
                        note = "ret missing: computed from equity pct_change"
                    split_col = _pick_col(df, ["Split"])
                    metrics, reason = _calc_equity_metrics(df, equity_col=equity_col, ret_col=ret_col, split_col=split_col)
                    row = {
                        "file": os.path.basename(fpath),
                        **metrics,
                        "status": "OK" if metrics["test_days"] > 0 else "WARN",
                    }
                    merged_note = "; ".join([x for x in [note, reason] if x])
                    if merged_note:
                        row["note"] = merged_note
                    rows.append(row)
                except Exception as exc:
                    rows.append({"file": os.path.basename(fpath), "status": "SKIP", "reason": str(exc)})
            report["stepF"] = {
                "status": "OK" if any(r.get("status") == "OK" for r in rows) else "SKIP",
                "summary": "stepF equity logs evaluated",
                "rows": rows,
            }
    except Exception as exc:
        report["stepF"] = {"status": "SKIP", "summary": f"exception: {exc}", "rows": []}

    statuses = [
        report["stepA"]["status"],
        report["stepB"]["status"],
        report["dprime"]["status"],
        report["stepE"]["status"],
        report["stepF"]["status"],
        report.get("diversity", {}).get("status", "SKIP"),
    ]
    report["overall_status"] = "BAD" if "BAD" in statuses else ("WARN" if "WARN" in statuses or "SKIP" in statuses else "OK")
    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# EVAL_REPORT",
        "",
        f"- output_root: `{report.get('output_root')}`",
        f"- mode: `{report.get('mode')}`",
        f"- symbol: `{report.get('symbol')}`",
        f"- overall_status: **{report.get('overall_status')}**",
        "",
        "## D' (stepD_prime) artifacts",
    ]
    dprime = report.get("dprime", {})
    ddet = dprime.get("details", {})
    lines.extend([
        f"- status: **{dprime.get('status', 'SKIP')}**",
        f"- summary: {dprime.get('summary', 'NA')}",
        f"- state_count: {_fmt(ddet.get('state_count'))}",
        f"- embeddings_count: {_fmt(ddet.get('embeddings_count'))}",
    ])

    lines.extend([
        "",
        "## StepE agents table",
    ])

    stepe_rows = report.get("stepE", {}).get("rows", [])
    if stepe_rows:
        lines.extend([
            "| agent | file | test_days | equity_multiple | max_dd | mean_ret | std_ret | sharpe | note | status |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
        ])
        for r in stepe_rows:
            lines.append(
                f"| {r.get('agent', 'NA')} | {r.get('file', 'NA')} | {_fmt(r.get('test_days'))} | {_fmt(r.get('equity_multiple'))} | "
                f"{_fmt(r.get('max_dd'))} | {_fmt(r.get('mean_ret'))} | {_fmt(r.get('std_ret'))} | {_fmt(r.get('sharpe'))} | "
                f"{r.get('note', r.get('reason', ''))} | {r.get('status', 'NA')} |"
            )
    else:
        lines.append(f"- SKIP: {report.get('stepE', {}).get('summary')}")

    lines.extend([
        "",
        "## StepF router summary",
    ])
    stepf_rows = report.get("stepF", {}).get("rows", [])
    if stepf_rows:
        lines.extend([
            "| file | test_days | equity_multiple | max_dd | mean_ret | std_ret | sharpe | note | status |",
            "|---|---:|---:|---:|---:|---:|---:|---|---|",
        ])
        for r in stepf_rows:
            lines.append(
                f"| {r.get('file', 'NA')} | {_fmt(r.get('test_days'))} | {_fmt(r.get('equity_multiple'))} | {_fmt(r.get('max_dd'))} | "
                f"{_fmt(r.get('mean_ret'))} | {_fmt(r.get('std_ret'))} | {_fmt(r.get('sharpe'))} | {r.get('note', r.get('reason', ''))} | {r.get('status', 'NA')} |"
            )
    else:
        lines.append(f"- SKIP: {report.get('stepF', {}).get('summary')}")

    div = report.get("diversity", {})
    lines.extend([
        "",
        "## Diversity",
        f"- status: **{div.get('status', 'SKIP')}**",
        f"- summary: {div.get('summary', 'NA')}",
        f"- max_corr: {_fmt(div.get('max_corr'))}",
        f"- max_match_ratio: {_fmt(div.get('max_match_ratio'))}",
        f"- pairs_over_0_9999: {_fmt(div.get('pairs_over_0_9999'))} / {_fmt(div.get('all_pairs'))}",
        f"- identical_all_agents: {_fmt(div.get('identical_all_agents'))}",
    ])

    plots = report.get("plots", {})
    plot_items = plots.get("items", []) if isinstance(plots, dict) else []
    lines.extend([
        "",
        "## PLOTS",
    ])
    if plot_items:
        for item in plot_items:
            nm = item.get("name", "plot")
            lines.append(f"- [{nm}](./{nm})")
    else:
        lines.append("- (no plots)")
    for note in (plots.get("notes", []) if isinstance(plots, dict) else []):
        lines.append(f"  - note: {note}")

    lines.extend([
        "",
        "## Raw JSON",
        "```json",
        json.dumps(report, ensure_ascii=False, indent=2),
        "```",
        "",
        "Best-effort mode: this evaluator writes SKIP/notes and always exits 0.",
    ])
    return "\n".join(lines)


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

    dprime = report.get("dprime", {})
    ddet = dprime.get("details", {})
    lines.append("DPrime:")
    lines.append(f"  status={dprime.get('status', 'SKIP')} summary={dprime.get('summary', 'NA')}")
    lines.append(f"  state_count={_fmt(ddet.get('state_count'))} embeddings_count={_fmt(ddet.get('embeddings_count'))}")

    stepe = report.get("stepE", {})
    lines.append("StepE:")
    if stepe.get("status") == "SKIP":
        lines.append(f"  SKIP: {stepe.get('summary')}")
    else:
        for r in stepe.get("rows", [])[:MAX_LIST_ITEMS]:
            lines.append(
                f"  {r.get('agent', 'NA')}::{r.get('file')} test_days={_fmt(r.get('test_days'))} equity_multiple={_fmt(r.get('equity_multiple'))} "
                f"max_dd={_fmt(r.get('max_dd'))} mean_ret={_fmt(r.get('mean_ret'))} std_ret={_fmt(r.get('std_ret'))} sharpe={_fmt(r.get('sharpe'))}"
            )
            if r.get("reason"):
                lines.append(f"    reason={r.get('reason')}")
            if r.get("note"):
                lines.append(f"    note={r.get('note')}")

    stepf = report.get("stepF", {})
    lines.append("StepF:")
    if stepf.get("status") == "SKIP":
        lines.append(f"  SKIP: {stepf.get('summary')}")
    else:
        for r in stepf.get("rows", [])[:MAX_LIST_ITEMS]:
            lines.append(
                f"  {r.get('file')} test_days={_fmt(r.get('test_days'))} equity_multiple={_fmt(r.get('equity_multiple'))} "
                f"max_dd={_fmt(r.get('max_dd'))} mean_ret={_fmt(r.get('mean_ret'))} std_ret={_fmt(r.get('std_ret'))} sharpe={_fmt(r.get('sharpe'))}"
            )
            if r.get("reason"):
                lines.append(f"    reason={r.get('reason')}")
            if r.get("note"):
                lines.append(f"    note={r.get('note')}")

    div = report.get("diversity", {})
    lines.append("Diversity:")
    lines.append(
        "  "
        + f"status={div.get('status', 'SKIP')} max_corr={_fmt(div.get('max_corr'))} "
        + f"max_match_ratio={_fmt(div.get('max_match_ratio'))} "
        + f"pairs_over_0_9999={_fmt(div.get('pairs_over_0_9999'))}/{_fmt(div.get('all_pairs'))} "
        + f"identical_all_agents={_fmt(div.get('identical_all_agents'))}"
    )
    lines.append(f"  summary={div.get('summary', 'NA')}")

    plots = report.get("plots", {}) if isinstance(report.get("plots", {}), dict) else {}
    lines.append("PLOTS:")
    for item in plots.get("items", []):
        exists = "yes" if item.get("exists") else "no"
        reason = item.get("reason") or ""
        lines.append(f"  {item.get('name')} exists={exists} {reason}".rstrip())
    for note in plots.get("notes", []):
        lines.append(f"  note={note}")

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
        _generate_plots(
            output_root=args.output_root,
            mode=args.mode,
            symbol=args.symbol,
            report=report,
            out_dir=os.path.dirname(os.path.abspath(args.out_md)),
        )
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
