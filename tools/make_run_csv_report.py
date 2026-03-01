#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DATE_CANDIDATES = ["Date", "date", "Datetime", "datetime", "timestamp", "Timestamp"]
PREFERRED_PATTERNS = [
    r"^(close|open|high|low)$",
    r"^volume$",
    r"^close_pred",
    r"^close_true$",
    r"^equity$",
    r"^ret",
    r"^reward",
    r"ratio",
    r"^pos",
]
STEP_ORDER = ["stepA", "stepB", "stepDPRIME", "stepE", "stepF"]
STEP_SORT_KEY = {name.lower(): idx for idx, name in enumerate(STEP_ORDER)}
METRIC_COLUMNS = [
    "step",
    "type",
    "scope",
    "csv",
    "agent",
    "split",
    "n_days",
    "start_date",
    "end_date",
    "total_return_pct",
    "cagr_pct",
    "vol_ann_pct",
    "sharpe",
    "sortino",
    "max_drawdown_pct",
    "win_rate_pct",
    "avg_daily_ret_pct",
    "best_day_pct",
    "worst_day_pct",
    "turnover",
    "avg_cost",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build markdown/html/png report from mode CSV outputs.")
    p.add_argument("--output-root", required=True)
    p.add_argument("--mode", choices=["sim", "live"], required=True)
    p.add_argument("--report-dir", required=True)
    p.add_argument("--max-files", type=int, default=200)
    p.add_argument("--max-cols", type=int, default=8)
    return p.parse_args()


def find_date_col(df: pd.DataFrame) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for c in DATE_CANDIDATES:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def iter_mode_csv(output_root: Path, mode: str) -> Iterable[Path]:
    for p in sorted(output_root.rglob("*.csv")):
        if mode.lower() in [part.lower() for part in p.parts]:
            yield p


def classify_eval_target(rel: Path, cols: list[str]) -> str | None:
    path_l = str(rel).replace("\\", "/").lower()
    name_l = rel.name.lower()
    if "/stepe/" in path_l or "stepe_daily_log" in name_l:
        return "StepE"
    if "/stepf/" in path_l or any(k in name_l for k in ["router", "bandit", "phase2"]):
        return "StepF"
    colset = {c.lower() for c in cols}
    if {"equity", "ret", "ratio"} & colset:
        return "Other"
    return None


def first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    lower = {str(c).lower(): str(c) for c in df.columns}
    for n in names:
        if n in df.columns:
            return n
        n_l = n.lower()
        if n_l in lower:
            return lower[n_l]
    return None


def infer_step(rel: Path) -> str:
    for part in rel.parts:
        l = part.lower()
        if l in {"stepa", "stepb", "stepc", "stepdprime", "stepe", "stepf"}:
            return part
    return rel.parts[0] if rel.parts else "unknown"


def step_sort_value(step: str) -> tuple[int, str]:
    l = step.lower()
    return (STEP_SORT_KEY.get(l, 999), l)


def infer_agent_from_stepe(rel: Path) -> str:
    name = rel.stem
    m = re.search(r"stepe_daily_log_(.+)$", name, flags=re.IGNORECASE)
    if not m:
        return ""
    agent = m.group(1)
    agent = re.sub(r"_(sim|live)$", "", agent, flags=re.IGNORECASE)
    agent = re.sub(r"_(SOXL|SOXS)$", "", agent, flags=re.IGNORECASE)
    return agent


def make_equity_from_ret(ret: pd.Series) -> pd.Series:
    clean = pd.to_numeric(ret, errors="coerce").fillna(0.0)
    return (1.0 + clean).cumprod()


def compute_metrics(eval_df: pd.DataFrame, date_col: str | None, rel: Path, target_type: str, step: str) -> dict[str, object] | None:
    ret_col = first_existing(eval_df, ["ret", "return", "daily_ret", "daily_return", "pnl", "reward"])
    equity_col = first_existing(eval_df, ["equity", "portfolio_value", "balance", "nav"])
    ratio_col = first_existing(eval_df, ["ratio", "position_ratio", "alloc_ratio", "weight"])
    cost_col = first_existing(eval_df, ["cost", "fee", "commission"])

    if not ret_col and not equity_col:
        return None

    df = eval_df.copy()
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        if df[date_col].notna().any():
            df = df.sort_values(date_col)

    if ret_col:
        ret = pd.to_numeric(df[ret_col], errors="coerce")
    elif equity_col:
        eq_tmp = pd.to_numeric(df[equity_col], errors="coerce")
        ret = eq_tmp.pct_change()
    else:
        ret = pd.Series(dtype=float)

    if equity_col:
        equity = pd.to_numeric(df[equity_col], errors="coerce")
    else:
        equity = make_equity_from_ret(ret)

    data = pd.DataFrame({"ret": ret, "equity": equity})
    if date_col and date_col in df.columns:
        data["date"] = pd.to_datetime(df[date_col], errors="coerce")

    data = data.dropna(subset=["equity"]).reset_index(drop=True)
    if data.empty:
        return None

    if "ret" in data.columns:
        data["ret"] = pd.to_numeric(data["ret"], errors="coerce").fillna(0.0)

    n_days = int(len(data))
    r = data["ret"]
    eq_first = float(data["equity"].iloc[0])
    eq_last = float(data["equity"].iloc[-1])
    if equity_col:
        if math.isclose(eq_first, 0.0):
            total_return = float("nan")
        else:
            total_return = (eq_last / eq_first) - 1.0
    else:
        total_return = float((1.0 + r).prod() - 1.0)

    cagr = float("nan")
    if n_days >= 2 and eq_first > 0 and eq_last > 0:
        years = n_days / 252.0
        if years > 0:
            cagr = (eq_last / eq_first) ** (1.0 / years) - 1.0

    vol = float(r.std(ddof=0)) * math.sqrt(252.0)
    r_std = float(r.std(ddof=0))
    sharpe = float("nan") if r_std == 0 else float(r.mean()) / r_std * math.sqrt(252.0)

    neg = r[r < 0]
    neg_std = float(neg.std(ddof=0)) if len(neg) > 0 else 0.0
    sortino = float("nan") if neg_std == 0 else float(r.mean()) / neg_std * math.sqrt(252.0)

    roll_max = data["equity"].cummax()
    dd = (data["equity"] / roll_max) - 1.0
    max_dd = float(dd.min()) if dd.notna().any() else float("nan")

    start_date = ""
    end_date = ""
    if "date" in data.columns and data["date"].notna().any():
        start_date = str(data["date"].min().date())
        end_date = str(data["date"].max().date())

    turnover = float("nan")
    if ratio_col:
        ratio = pd.to_numeric(df[ratio_col], errors="coerce")
        turnover = float(ratio.diff().abs().mean() / 2.0)

    avg_cost = float("nan")
    if cost_col:
        avg_cost = float(pd.to_numeric(df[cost_col], errors="coerce").mean())

    return {
        "step": step,
        "type": target_type,
        "scope": "evaluation",
        "csv": str(rel).replace("\\", "/"),
        "agent": infer_agent_from_stepe(rel) if target_type == "StepE" else "",
        "split": "",
        "n_days": n_days,
        "start_date": start_date,
        "end_date": end_date,
        "total_return_pct": total_return * 100.0,
        "cagr_pct": cagr * 100.0,
        "vol_ann_pct": vol * 100.0,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown_pct": max_dd * 100.0,
        "win_rate_pct": float((r > 0).mean()) * 100.0,
        "avg_daily_ret_pct": float(r.mean()) * 100.0,
        "best_day_pct": float(r.max()) * 100.0,
        "worst_day_pct": float(r.min()) * 100.0,
        "turnover": turnover,
        "avg_cost": avg_cost,
        "_equity": data["equity"],
    }


def metric_md_table(df: pd.DataFrame, cols: list[str] | None = None) -> str:
    if df.empty:
        return "no evaluatable csv found"
    use_cols = cols if cols else [c for c in METRIC_COLUMNS if c in df.columns]
    table = df[use_cols].copy()
    for c in table.columns:
        if pd.api.types.is_float_dtype(table[c]):
            table[c] = table[c].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    try:
        return table.to_markdown(index=False)
    except Exception:
        return "```\n" + table.to_string(index=False) + "\n```"


def safe_key(rel: Path) -> str:
    s = str(rel).replace("\\", "__").replace("/", "__")
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)


def choose_plot_cols(df: pd.DataFrame, max_cols: int) -> list[str]:
    numeric = []
    for c in df.columns:
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.notna().sum() > 1:
            numeric.append(c)
    if not numeric:
        return []
    ranked: list[str] = []
    seen = set()
    lower_map = {c.lower(): c for c in numeric}
    for pat in PREFERRED_PATTERNS:
        rx = re.compile(pat, re.IGNORECASE)
        for k, orig in lower_map.items():
            if rx.search(k) and orig not in seen:
                ranked.append(orig)
                seen.add(orig)
                if len(ranked) >= max_cols:
                    return ranked
    for c in numeric:
        if c not in seen:
            ranked.append(c)
            if len(ranked) >= max_cols:
                break
    return ranked


def df_to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    safe = df.fillna("")
    try:
        return safe.to_markdown(index=False)
    except Exception:
        return "```\n" + safe.to_string(index=False) + "\n```"


def write_html_from_md(md_text: str, out_html: Path) -> None:
    body = html.escape(md_text)
    out_html.write_text(
        "<html><head><meta charset='utf-8'><title>CSV report</title></head>"
        "<body><pre>" + body + "</pre></body></html>",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    report_dir = Path(args.report_dir)
    tables_dir = report_dir / "tables"
    plots_dir = report_dir / "plots"
    metrics_dir = report_dir / "metrics"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    csvs = list(iter_mode_csv(output_root, args.mode))[: max(1, args.max_files)]
    index_lines = [f"# CSV report ({args.mode})", ""]

    if not csvs:
        index_lines.extend(["no csv found", ""])
    
    grouped: dict[str, list[dict[str, str]]] = {}
    metrics_rows: list[dict[str, object]] = []
    metrics_warnings: list[str] = []

    for csv_path in csvs:
        rel = csv_path.relative_to(output_root)
        key = safe_key(rel)
        step = infer_step(rel)
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            (tables_dir / f"{key}.md").write_text(f"# {rel}\n\nread error: {e}\n", encoding="utf-8")
            metrics_warnings.append(f"- {rel}: read error: {e}")
            continue

        date_col = find_date_col(df)
        date_min = ""
        date_max = ""
        if date_col:
            d = pd.to_datetime(df[date_col], errors="coerce")
            if d.notna().any():
                date_min = str(d.min())
                date_max = str(d.max())
                df = df.copy()
                df[date_col] = d

        table_md = [
            f"# {rel}",
            "",
            f"- rows: {len(df)}",
            f"- missing_cells: {int(df.isna().sum().sum())}",
            f"- date_min: {date_min or 'n/a'}",
            f"- date_max: {date_max or 'n/a'}",
            f"- columns: {', '.join(str(c) for c in df.columns)}",
            "",
            "## head(20)",
            "",
            df_to_md(df.head(20)),
            "",
        ]
        table_rel = f"tables/{key}.md"
        (report_dir / table_rel).write_text("\n".join(table_md), encoding="utf-8")

        plot_links = []
        if date_col:
            plot_cols = choose_plot_cols(df, max_cols=max(1, args.max_cols))
            x = pd.to_datetime(df[date_col], errors="coerce")
            for col in plot_cols:
                y = pd.to_numeric(df[col], errors="coerce")
                if y.notna().sum() < 2:
                    continue
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(x, y, linewidth=1.0)
                ax.set_title(f"{rel} :: {col}")
                ax.grid(alpha=0.3)
                fig.tight_layout()
                png_name = f"{key}__{re.sub(r'[^A-Za-z0-9_.-]', '_', str(col))}.png"
                out_png = plots_dir / png_name
                fig.savefig(out_png, dpi=130)
                plt.close(fig)
                plot_links.append(f"plots/{png_name}")

        grouped.setdefault(step, []).append(
            {"csv": str(rel).replace('\\', '/'), "table": table_rel, "plot": plot_links[0] if plot_links else ""}
        )

        target_type = classify_eval_target(rel, [str(c) for c in df.columns])
        if not target_type:
            continue

        split_col = first_existing(df, ["Split", "split", "dataset", "phase"])
        eval_sets: list[tuple[str, pd.DataFrame]] = []
        if split_col:
            split_norm = df[split_col].astype(str).str.lower().str.strip()
            if (split_norm == "test").any():
                eval_sets.append(("test", df.loc[split_norm == "test"].copy()))
            else:
                eval_sets.append(("all", df.copy()))
        else:
            eval_sets.append(("all", df.copy()))

        for split_name, sub_df in eval_sets:
            try:
                item = compute_metrics(sub_df, date_col=date_col, rel=rel, target_type=target_type, step=step)
            except Exception as e:
                metrics_warnings.append(f"- {rel} [{split_name}]: metrics error: {e}")
                continue
            if not item:
                continue
            item["split"] = split_name
            metrics_rows.append(item)

    metrics_df = pd.DataFrame([{k: v for k, v in row.items() if not k.startswith("_")} for row in metrics_rows])
    if not metrics_df.empty:
        for c in METRIC_COLUMNS:
            if c not in metrics_df.columns:
                metrics_df[c] = ""
        metrics_df = metrics_df[METRIC_COLUMNS]
        metrics_df = metrics_df.sort_values(by=["type", "step", "split", "sharpe"], ascending=[True, True, True, False], na_position="last")
    metrics_csv = metrics_dir / "metrics_summary.csv"
    metrics_md = metrics_dir / "metrics_summary.md"
    by_step_md = metrics_dir / "metrics_by_step.md"

    if metrics_df.empty:
        metrics_csv.write_text("", encoding="utf-8")
        metrics_md.write_text("# Metrics summary\n\nno evaluatable csv found\n", encoding="utf-8")
        by_step_md.write_text("# Metrics by step\n\nno evaluatable csv found\n", encoding="utf-8")
    else:
        metrics_df.to_csv(metrics_csv, index=False)
        top_cols = ["step", "agent", "csv", "split", "sharpe", "total_return_pct", "max_drawdown_pct", "n_days"]
        stepe_test = metrics_df[(metrics_df["type"] == "StepE") & (metrics_df["split"] == "test")]
        stepf_test = metrics_df[(metrics_df["type"] == "StepF") & (metrics_df["split"] == "test")]
        mlines = ["# Metrics summary", "", "## StepE(test) top by Sharpe", "", metric_md_table(stepe_test.sort_values("sharpe", ascending=False).head(10), top_cols), "", "## StepE(test) top by total_return_pct", "", metric_md_table(stepe_test.sort_values("total_return_pct", ascending=False).head(10), top_cols), "", "## StepF(test) top by Sharpe", "", metric_md_table(stepf_test.sort_values("sharpe", ascending=False).head(10), top_cols), "", "## StepF(test) top by total_return_pct", "", metric_md_table(stepf_test.sort_values("total_return_pct", ascending=False).head(10), top_cols), "", "## All metrics", "", metric_md_table(metrics_df)]
        if metrics_warnings:
            mlines.extend(["", "## Warnings", "", *metrics_warnings])
        metrics_md.write_text("\n".join(mlines), encoding="utf-8")

        by_step_lines = ["# Metrics by step", ""]
        for step_name in sorted(metrics_df["step"].astype(str).unique(), key=step_sort_value):
            s = metrics_df[metrics_df["step"] == step_name]
            by_step_lines.extend([f"## {step_name}", "", metric_md_table(s), ""])
        by_step_md.write_text("\n".join(by_step_lines), encoding="utf-8")

    top_equity_png = metrics_dir / "top_equity_curves.png"
    equity_candidates = [m for m in metrics_rows if m.get("split") == "test" and isinstance(m.get("_equity"), pd.Series)]
    equity_candidates = sorted(
        equity_candidates,
        key=lambda m: (
            float(m.get("sharpe")) if pd.notna(m.get("sharpe")) else -1e9,
            float(m.get("total_return_pct")) if pd.notna(m.get("total_return_pct")) else -1e9,
        ),
        reverse=True,
    )
    if equity_candidates:
        fig, ax = plt.subplots(figsize=(10, 5))
        for item in equity_candidates[:5]:
            eq = pd.to_numeric(item["_equity"], errors="coerce")
            if eq.notna().sum() < 2:
                continue
            label = f"{item.get('type','')}|{Path(str(item.get('csv',''))).name}"
            if item.get("agent"):
                label = f"{item['agent']} ({item.get('type','')})"
            ax.plot(eq.reset_index(drop=True), linewidth=1.1, label=label[:80])
        if ax.has_data():
            ax.set_title("Top Equity Curves (test)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, loc="best")
            fig.tight_layout()
            fig.savefig(top_equity_png, dpi=130)
        plt.close(fig)

    index_lines.extend([
        "## Metrics summary",
        "",
        "- [metrics_summary](metrics/metrics_summary.md)",
        "- [metrics_by_step](metrics/metrics_by_step.md)",
        "",
    ])
    ms_text = metrics_md.read_text(encoding="utf-8") if metrics_md.exists() else ""
    for heading in ["## StepE(test) top by Sharpe", "## StepF(test) top by Sharpe"]:
        if heading in ms_text:
            chunk = ms_text.split(heading, 1)[1]
            end_pos = len(chunk)
            for token in ["\n## StepE(test) top by total_return_pct", "\n## StepF(test) top by total_return_pct", "\n## All metrics"]:
                i = chunk.find(token)
                if i >= 0:
                    end_pos = min(end_pos, i)
            preview = chunk[:end_pos].strip()
            index_lines.extend([heading, "", preview, ""])

    index_lines.append("## CSV list")
    index_lines.append("")
    for step in sorted(grouped.keys()):
        index_lines.append(f"### {step}")
        index_lines.append("")
        index_lines.append(f"- metrics: [view](metrics/metrics_by_step.md#{step.lower()})")
        for item in grouped[step]:
            index_lines.append(f"- `{item['csv']}`: [table]({item['table']})")
            if item["plot"]:
                index_lines.append(f"  ")
                index_lines.append(f"  ![]({item['plot']})")
        index_lines.append("")

    index_md = report_dir / "index.md"
    index_md.write_text("\n".join(index_lines), encoding="utf-8")
    write_html_from_md("\n".join(index_lines), report_dir / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
