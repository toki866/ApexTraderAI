#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATE_CANDIDATES = ["Date", "date", "Datetime", "datetime", "timestamp", "Timestamp"]
SPLIT_CANDIDATES = ["Split", "split", "dataset", "Dataset"]
EQUITY_CANDIDATES = ["equity", "Equity", "portfolio_value", "PortfolioValue", "account_value"]
RET_CANDIDATES = ["ret", "return", "daily_return", "pnl_ret", "strategy_return"]


@dataclass
class MetricRow:
    agent: str
    n: int
    total_return_equity: float
    total_return_prod: float
    max_drawdown: float
    sharpe: float


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate web-viewable run report from Step A-F artifacts.")
    ap.add_argument("--output-root", default="output", help="Root folder containing stepA/stepB/... outputs")
    ap.add_argument("--mode", default="sim", help="One mode or comma-separated modes (sim/live/display)")
    ap.add_argument("--symbol", required=True, help="Target symbol, e.g. SOXL")
    ap.add_argument("--out-dir", required=True, help="Output report directory")
    return ap.parse_args()


def parse_modes(mode_arg: str) -> List[str]:
    modes = [m.strip() for m in mode_arg.split(",") if m.strip()]
    if not modes:
        return ["sim"]
    return modes


def ensure_dirs(out_dir: Path) -> Tuple[Path, Path]:
    plots = out_dir / "plots"
    tables = out_dir / "tables"
    plots.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    return plots, tables


def find_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        lc = c.lower()
        if lc in lower:
            return lower[lc]
    return None


def find_date_column(df: pd.DataFrame) -> Optional[str]:
    return find_column(df, DATE_CANDIDATES)


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    dc = find_date_column(d)
    if dc:
        d[dc] = pd.to_datetime(d[dc], errors="coerce")
        d = d.sort_values(dc)
    return d


def to_md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    cols = [str(c) for c in df.columns]
    rows = [["" if pd.isna(v) else str(v) for v in r] for r in df.to_numpy()]
    widths = [len(c) for c in cols]
    for r in rows:
        for i, v in enumerate(r):
            widths[i] = max(widths[i], len(v))

    def fmt_row(values: List[str]) -> str:
        return "| " + " | ".join(v.ljust(widths[i]) for i, v in enumerate(values)) + " |"

    out = [fmt_row(cols), "| " + " | ".join("-" * w for w in widths) + " |"]
    out.extend(fmt_row(r) for r in rows)
    return "\n".join(out)


def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def maybe_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def step_a_summary(df: pd.DataFrame) -> Dict[str, object]:
    d = parse_dates(df)
    date_col = find_date_column(d)
    close_col = find_column(d, ["Close", "close"])
    vol_col = find_column(d, ["Volume", "volume"])

    start = ""
    end = ""
    if date_col and d[date_col].notna().any():
        x = d[date_col].dropna()
        start = str(x.iloc[0].date())
        end = str(x.iloc[-1].date())

    close = maybe_num(d[close_col]) if close_col else pd.Series(dtype=float)
    vol = maybe_num(d[vol_col]) if vol_col else pd.Series(dtype=float)

    return {
        "rows": int(len(d)),
        "date_start": start,
        "date_end": end,
        "missing_total": int(d.isna().sum().sum()),
        "close_min": float(close.min()) if close.size and close.notna().any() else np.nan,
        "close_max": float(close.max()) if close.size and close.notna().any() else np.nan,
        "volume_min": float(vol.min()) if vol.size and vol.notna().any() else np.nan,
        "volume_max": float(vol.max()) if vol.size and vol.notna().any() else np.nan,
    }


def save_step_a_plot(train_df: Optional[pd.DataFrame], test_df: Optional[pd.DataFrame], out_path: Path, symbol: str, mode: str) -> bool:
    if train_df is None and test_df is None:
        return False
    fig, ax1 = plt.subplots(figsize=(10, 4.5))

    plotted = False
    for label, df, color in [("train", train_df, "tab:blue"), ("test", test_df, "tab:orange")]:
        if df is None:
            continue
        d = parse_dates(df)
        date_col = find_date_column(d)
        close_col = find_column(d, ["Close", "close"])
        if date_col and close_col:
            x = d[date_col]
            y = maybe_num(d[close_col])
            ax1.plot(x, y, label=f"Close ({label})", color=color)
            plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax1.set_title(f"StepA Close {symbol} [{mode}]")
    ax1.set_ylabel("Close")
    ax1.legend(loc="best")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def extract_true_pred(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    d = parse_dates(df)
    date_col = find_date_column(d)
    if not date_col:
        return None
    lower_map = {c.lower(): c for c in d.columns}

    true_col = None
    for c in ["close_true", "y_true", "target", "close"]:
        if c in lower_map:
            true_col = lower_map[c]
            break

    pred_col = None
    for c in ["close_pred_h1", "pred_h1", "y_pred_h1", "close_pred", "pred"]:
        if c in lower_map:
            pred_col = lower_map[c]
            break
    if pred_col is None:
        for k, v in lower_map.items():
            if "pred" in k and ("h1" in k or "close" in k):
                pred_col = v
                break

    if true_col is None or pred_col is None:
        return None

    out = pd.DataFrame({
        "Date": pd.to_datetime(d[date_col], errors="coerce"),
        "Close_true": maybe_num(d[true_col]),
        "Close_pred_h1": maybe_num(d[pred_col]),
    }).dropna(subset=["Date"])

    split_col = find_column(d, SPLIT_CANDIDATES)
    if split_col:
        out["Split"] = d[split_col].astype(str).values
    return out


def compute_mae_rmse(y_true: pd.Series, y_pred: pd.Series) -> Tuple[float, float, int]:
    mask = np.isfinite(y_true.to_numpy(dtype=float)) & np.isfinite(y_pred.to_numpy(dtype=float))
    if mask.sum() == 0:
        return (np.nan, np.nan, 0)
    a = y_true.to_numpy(dtype=float)[mask]
    b = y_pred.to_numpy(dtype=float)[mask]
    err = b - a
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    return mae, rmse, int(mask.sum())


def save_step_b_plot(df: pd.DataFrame, out_path: Path, symbol: str, mode: str) -> bool:
    if df.empty:
        return False
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(df["Date"], df["Close_true"], label="Close_true", color="tab:blue")
    ax.plot(df["Date"], df["Close_pred_h1"], label="Close_pred_h1", color="tab:red", alpha=0.8)
    ax.set_title(f"StepB Prediction vs Truth {symbol} [{mode}]")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def compute_drawdown(eq: np.ndarray) -> float:
    if eq.size == 0:
        return np.nan
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    return float(np.nanmin(dd)) if dd.size else np.nan


def compute_sharpe(ret: np.ndarray) -> float:
    ret = ret[np.isfinite(ret)]
    if ret.size < 2:
        return np.nan
    s = float(np.std(ret, ddof=1))
    if s <= 0:
        return np.nan
    return float(np.mean(ret) / s * math.sqrt(252.0))


def metric_from_df(df: pd.DataFrame, agent: str) -> Optional[MetricRow]:
    d = parse_dates(df)
    split_col = find_column(d, SPLIT_CANDIDATES)
    if split_col:
        d = d[d[split_col].astype(str).str.lower() == "test"]
    if d.empty:
        return None

    eq_col = find_column(d, EQUITY_CANDIDATES)
    eq = np.array([], dtype=float)
    total_eq = np.nan
    mdd = np.nan

    if eq_col:
        eq = maybe_num(d[eq_col]).to_numpy(dtype=float)
        eq = eq[np.isfinite(eq)]
        if eq.size >= 2 and eq[0] != 0:
            total_eq = float(eq[-1] / eq[0] - 1.0)
            mdd = compute_drawdown(eq)

    ret_col = find_column(d, RET_CANDIDATES)
    if ret_col:
        ret = maybe_num(d[ret_col]).to_numpy(dtype=float)
    elif eq.size >= 2:
        ret = np.diff(eq) / eq[:-1]
    else:
        ret = np.array([], dtype=float)

    total_prod = np.nan
    if ret.size:
        good = ret[np.isfinite(ret)]
        if good.size:
            total_prod = float(np.prod(1.0 + good) - 1.0)

    sharpe = compute_sharpe(ret)
    n = int(len(d))
    return MetricRow(agent=agent, n=n, total_return_equity=total_eq, total_return_prod=total_prod, max_drawdown=mdd, sharpe=sharpe)


def extract_agent_name(path: Path, symbol: str) -> str:
    m = re.match(rf"stepE_daily_log_(.+)_{re.escape(symbol)}\.csv$", path.name)
    return m.group(1) if m else path.stem


def save_equity_plot(curves: List[Tuple[str, pd.DataFrame]], out_path: Path, title: str, top_k: int = 5) -> bool:
    ranked: List[Tuple[str, float, pd.DataFrame]] = []
    for name, df in curves:
        met = metric_from_df(df, name)
        if met is None:
            continue
        score = met.total_return_equity
        if not np.isfinite(score):
            score = met.total_return_prod
        ranked.append((name, score if np.isfinite(score) else -1e9, df))

    if not ranked:
        return False

    ranked.sort(key=lambda x: x[1], reverse=True)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    plotted = 0
    for name, _, df in ranked[:top_k]:
        d = parse_dates(df)
        date_col = find_date_column(d)
        eq_col = find_column(d, EQUITY_CANDIDATES)
        split_col = find_column(d, SPLIT_CANDIDATES)
        if split_col:
            d = d[d[split_col].astype(str).str.lower() == "test"]
        if date_col and eq_col and not d.empty:
            ax.plot(d[date_col], maybe_num(d[eq_col]), label=name)
            plotted += 1

    if plotted == 0:
        plt.close(fig)
        return False

    ax.set_title(title)
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def html_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p>(empty)</p>"
    h = ["<table>", "<thead><tr>"]
    for c in df.columns:
        h.append(f"<th>{html.escape(str(c))}</th>")
    h.append("</tr></thead><tbody>")
    for _, row in df.iterrows():
        h.append("<tr>")
        for c in df.columns:
            v = row[c]
            txt = "" if pd.isna(v) else str(v)
            h.append(f"<td>{html.escape(txt)}</td>")
        h.append("</tr>")
    h.append("</tbody></table>")
    return "".join(h)


def process_mode(output_root: Path, mode: str, symbol: str, out_dir: Path, plots_dir: Path, tables_dir: Path) -> Tuple[str, str]:
    md: List[str] = [f"## Mode: {mode}", ""]
    html_parts: List[str] = [f"<h2>Mode: {html.escape(mode)}</h2>"]
    read_paths: List[Path] = []
    missing_notes: List[str] = []

    # --- StepA ---
    md.append("### StepA")
    html_parts.append("<h3>StepA</h3>")
    stepA_dir = output_root / "stepA" / mode
    a_rows = []
    train_df = None
    test_df = None
    for split in ["train", "test"]:
        patterns = [
            stepA_dir / f"stepA_prices_{split}_{symbol}.csv",
            stepA_dir / f"stepA_tech_{split}_{symbol}.csv",
            stepA_dir / f"stepA_periodic_{split}_{symbol}.csv",
            stepA_dir / f"stepA_split_summary_{split}_{symbol}.csv",
            stepA_dir / f"stepA_split_summary_{symbol}.csv",
        ]
        p = stepA_dir / f"stepA_prices_{split}_{symbol}.csv"
        if p.exists():
            read_paths.append(p)
            df = safe_read_csv(p)
            if df is not None:
                if split == "train":
                    train_df = df
                else:
                    test_df = df
                s = step_a_summary(df)
                s["split"] = split
                a_rows.append(s)
            else:
                missing_notes.append(f"StepA {split}: failed reading {p.as_posix()}")
        else:
            missing_notes.append(f"StepA {split}: missing {p.as_posix()}")
        for cand in patterns[1:]:
            if cand.exists():
                read_paths.append(cand)

    a_df = pd.DataFrame(a_rows, columns=["split", "date_start", "date_end", "rows", "missing_total", "close_min", "close_max", "volume_min", "volume_max"])
    a_csv = tables_dir / f"{mode}_stepA_summary.csv"
    a_df.to_csv(a_csv, index=False)

    md.append(f"- table: [tables/{a_csv.name}](tables/{a_csv.name})")
    md.append(to_md_table(a_df))
    html_parts.append(f"<p>table: <a href='tables/{a_csv.name}'>tables/{a_csv.name}</a></p>")
    html_parts.append(html_table(a_df))

    a_plot = plots_dir / f"{mode}_stepA_close.png"
    if save_step_a_plot(train_df, test_df, a_plot, symbol, mode):
        md.append(f"- plot: [plots/{a_plot.name}](plots/{a_plot.name})")
        md.append(f"![stepA close {mode}](plots/{a_plot.name})")
        html_parts.append(f"<p><a href='plots/{a_plot.name}'>plots/{a_plot.name}</a></p><img src='plots/{a_plot.name}' alt='stepA close {mode}'/> ")
    else:
        missing_notes.append(f"StepA plot skipped for mode={mode}: Close/date data unavailable.")

    # --- StepB ---
    md.append("\n### StepB")
    html_parts.append("<h3>StepB</h3>")
    stepB_dir = output_root / "stepB" / mode
    b_candidates = sorted(stepB_dir.glob(f"stepB_pred_time_*_{symbol}.csv"))
    if not b_candidates:
        missing_notes.append(f"StepB: no files matched {stepB_dir.as_posix()}/stepB_pred_time_*_{symbol}.csv")
        b_df = pd.DataFrame(columns=["source_file", "n", "mae", "rmse"])
    else:
        b_rows = []
        plot_done = False
        for p in b_candidates:
            read_paths.append(p)
            df = safe_read_csv(p)
            if df is None:
                missing_notes.append(f"StepB: failed reading {p.as_posix()}")
                continue
            tp = extract_true_pred(df)
            if tp is None or tp.empty:
                missing_notes.append(f"StepB: usable true/pred columns not found in {p.as_posix()}")
                continue
            if "Split" in tp.columns:
                tp_test = tp[tp["Split"].astype(str).str.lower() == "test"].copy()
                if tp_test.empty:
                    tp_test = tp
            else:
                tp_test = tp
            mae, rmse, n = compute_mae_rmse(tp_test["Close_true"], tp_test["Close_pred_h1"])
            b_rows.append({"source_file": p.name, "n": n, "mae": mae, "rmse": rmse})
            if not plot_done:
                b_plot = plots_dir / f"{mode}_stepB_pred_vs_true.png"
                if save_step_b_plot(tp_test, b_plot, symbol, mode):
                    md.append(f"- plot: [plots/{b_plot.name}](plots/{b_plot.name})")
                    md.append(f"![stepB pred {mode}](plots/{b_plot.name})")
                    html_parts.append(f"<p><a href='plots/{b_plot.name}'>plots/{b_plot.name}</a></p><img src='plots/{b_plot.name}' alt='stepB pred {mode}'/> ")
                    plot_done = True
        b_df = pd.DataFrame(b_rows, columns=["source_file", "n", "mae", "rmse"])

    b_csv = tables_dir / f"{mode}_stepB_metrics.csv"
    b_df.to_csv(b_csv, index=False)
    md.append(f"- table: [tables/{b_csv.name}](tables/{b_csv.name})")
    md.append(to_md_table(b_df))
    html_parts.append(f"<p>table: <a href='tables/{b_csv.name}'>tables/{b_csv.name}</a></p>")
    html_parts.append(html_table(b_df))

    split_summary = stepB_dir / f"stepB_split_summary_{symbol}.csv"
    if split_summary.exists():
        read_paths.append(split_summary)

    # --- StepDprime ---
    md.append("\n### StepDprime")
    html_parts.append("<h3>StepDprime</h3>")
    stepD_dir = output_root / "stepDprime" / mode
    d_files = sorted(stepD_dir.glob("*.csv")) if stepD_dir.exists() else []
    d_rows = []
    if d_files:
        for p in d_files:
            if "train" in p.name.lower() or "test" in p.name.lower():
                read_paths.append(p)
                df = safe_read_csv(p)
                if df is None:
                    continue
                d_rows.append({"file": p.name, "rows": len(df), "cols": len(df.columns)})
    else:
        missing_notes.append(f"StepDprime: directory missing or empty -> {stepD_dir.as_posix()}")

    d_df = pd.DataFrame(d_rows, columns=["file", "rows", "cols"])
    d_csv = tables_dir / f"{mode}_stepDprime_files.csv"
    d_df.to_csv(d_csv, index=False)
    md.append(f"- table: [tables/{d_csv.name}](tables/{d_csv.name})")
    md.append(to_md_table(d_df))
    html_parts.append(f"<p>table: <a href='tables/{d_csv.name}'>tables/{d_csv.name}</a></p>")
    html_parts.append(html_table(d_df))

    # --- StepE ---
    md.append("\n### StepE")
    html_parts.append("<h3>StepE</h3>")
    stepE_dir = output_root / "stepE" / mode
    e_files = sorted(stepE_dir.glob(f"stepE_daily_log_*_{symbol}.csv")) if stepE_dir.exists() else []
    e_metrics: List[MetricRow] = []
    e_curves: List[Tuple[str, pd.DataFrame]] = []
    for p in e_files:
        read_paths.append(p)
        df = safe_read_csv(p)
        if df is None:
            continue
        agent = extract_agent_name(p, symbol)
        met = metric_from_df(df, agent)
        if met:
            e_metrics.append(met)
            e_curves.append((agent, df))

    if not e_files:
        missing_notes.append(f"StepE: no files matched {stepE_dir.as_posix()}/stepE_daily_log_*_{symbol}.csv")

    e_df = pd.DataFrame([
        {
            "agent": m.agent,
            "n_test_rows": m.n,
            "total_return_equity": m.total_return_equity,
            "total_return_prod": m.total_return_prod,
            "max_drawdown": m.max_drawdown,
            "sharpe": m.sharpe,
        }
        for m in e_metrics
    ])
    if not e_df.empty:
        e_df = e_df.sort_values("total_return_equity", ascending=False)

    e_csv = tables_dir / f"{mode}_stepE_metrics.csv"
    e_df.to_csv(e_csv, index=False)
    md.append(f"- table: [tables/{e_csv.name}](tables/{e_csv.name})")
    md.append(to_md_table(e_df))
    html_parts.append(f"<p>table: <a href='tables/{e_csv.name}'>tables/{e_csv.name}</a></p>")
    html_parts.append(html_table(e_df))

    e_plot = plots_dir / f"{mode}_stepE_equity_top.png"
    if save_equity_plot(e_curves, e_plot, f"StepE Equity Curves (top agents) {symbol} [{mode}]", top_k=5):
        md.append(f"- plot: [plots/{e_plot.name}](plots/{e_plot.name})")
        md.append(f"![stepE equity {mode}](plots/{e_plot.name})")
        html_parts.append(f"<p><a href='plots/{e_plot.name}'>plots/{e_plot.name}</a></p><img src='plots/{e_plot.name}' alt='stepE equity {mode}'/> ")
    else:
        missing_notes.append(f"StepE plot skipped for mode={mode}: no usable test equity curves.")

    # --- StepF ---
    md.append("\n### StepF")
    html_parts.append("<h3>StepF</h3>")
    stepF_dir = output_root / "stepF" / mode
    f_candidates = []
    if stepF_dir.exists():
        for pat in [f"*router*{symbol}*.csv", f"*phase2_state*{symbol}*.csv", f"*eval*{symbol}*.csv", "*.csv"]:
            f_candidates.extend(stepF_dir.glob(pat))
    f_candidates = sorted(set(f_candidates))

    f_metric_rows = []
    f_curves: List[Tuple[str, pd.DataFrame]] = []
    for p in f_candidates:
        read_paths.append(p)
        df = safe_read_csv(p)
        if df is None or df.empty:
            continue
        met = metric_from_df(df, p.stem)
        if met:
            f_metric_rows.append({
                "source": p.name,
                "n_test_rows": met.n,
                "total_return_equity": met.total_return_equity,
                "total_return_prod": met.total_return_prod,
                "max_drawdown": met.max_drawdown,
                "sharpe": met.sharpe,
            })
            f_curves.append((p.stem, df))

    if not f_candidates:
        missing_notes.append(f"StepF: directory missing or no csv -> {stepF_dir.as_posix()}")

    f_df = pd.DataFrame(f_metric_rows, columns=["source", "n_test_rows", "total_return_equity", "total_return_prod", "max_drawdown", "sharpe"])
    if not f_df.empty:
        f_df = f_df.sort_values("total_return_equity", ascending=False)

    f_csv = tables_dir / f"{mode}_stepF_metrics.csv"
    f_df.to_csv(f_csv, index=False)
    md.append(f"- table: [tables/{f_csv.name}](tables/{f_csv.name})")
    md.append(to_md_table(f_df))
    html_parts.append(f"<p>table: <a href='tables/{f_csv.name}'>tables/{f_csv.name}</a></p>")
    html_parts.append(html_table(f_df))

    f_plot = plots_dir / f"{mode}_stepF_equity_top.png"
    if save_equity_plot(f_curves, f_plot, f"StepF Router/Phase2 Equity Curves {symbol} [{mode}]", top_k=3):
        md.append(f"- plot: [plots/{f_plot.name}](plots/{f_plot.name})")
        md.append(f"![stepF equity {mode}](plots/{f_plot.name})")
        html_parts.append(f"<p><a href='plots/{f_plot.name}'>plots/{f_plot.name}</a></p><img src='plots/{f_plot.name}' alt='stepF equity {mode}'/> ")
    else:
        missing_notes.append(f"StepF plot skipped for mode={mode}: no usable equity series from router/phase2/eval CSVs.")

    # Read/missing sections
    md.append("\n### Read files")
    if read_paths:
        for p in sorted(set(read_paths)):
            rel = p.as_posix()
            md.append(f"- `{rel}`")
    else:
        md.append("- (none)")

    md.append("\n### Missing / skipped")
    if missing_notes:
        for n in missing_notes:
            md.append(f"- {n}")
    else:
        md.append("- none")

    html_parts.append("<h3>Read files</h3>")
    if read_paths:
        html_parts.append("<ul>" + "".join(f"<li><code>{html.escape(p.as_posix())}</code></li>" for p in sorted(set(read_paths))) + "</ul>")
    else:
        html_parts.append("<p>(none)</p>")
    html_parts.append("<h3>Missing / skipped</h3>")
    if missing_notes:
        html_parts.append("<ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in missing_notes) + "</ul>")
    else:
        html_parts.append("<p>none</p>")

    return "\n".join(md), "\n".join(html_parts)


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir, tables_dir = ensure_dirs(out_dir)

    modes = parse_modes(args.mode)
    md_lines = [
        "# Run Report",
        "",
        f"- symbol: **{args.symbol}**",
        f"- output_root: `{output_root.as_posix()}`",
        f"- mode(s): `{', '.join(modes)}`",
        "",
        "This report is generated by `tools/generate_run_report.py` and is designed to be readable on GitHub Web.",
        "",
    ]

    html_sections = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>Run Report</title>",
        "<style>body{font-family:Arial,sans-serif;max-width:1200px;margin:20px auto;padding:0 16px;}table{border-collapse:collapse;margin:8px 0 16px;}th,td{border:1px solid #ccc;padding:6px 8px;font-size:13px;}img{max-width:100%;height:auto;border:1px solid #ddd;}code{background:#f6f8fa;padding:1px 4px;border-radius:3px;}</style>",
        "</head><body>",
        "<h1>Run Report</h1>",
        f"<p><strong>symbol:</strong> {html.escape(args.symbol)}<br><strong>output_root:</strong> <code>{html.escape(output_root.as_posix())}</code><br><strong>mode(s):</strong> <code>{html.escape(', '.join(modes))}</code></p>",
    ]

    for mode in modes:
        md_sec, html_sec = process_mode(output_root, mode, args.symbol, out_dir, plots_dir, tables_dir)
        md_lines.append(md_sec)
        md_lines.append("")
        html_sections.append(html_sec)

    html_sections.append("</body></html>")

    (out_dir / "index.md").write_text("\n".join(md_lines), encoding="utf-8")
    (out_dir / "index.html").write_text("\n".join(html_sections), encoding="utf-8")
    print(f"[OK] Report generated: {(out_dir / 'index.md').as_posix()}")
    print(f"[OK] Report generated: {(out_dir / 'index.html').as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
