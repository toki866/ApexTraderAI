#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
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
    return df.fillna("").to_markdown(index=False)


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
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    csvs = list(iter_mode_csv(output_root, args.mode))[: max(1, args.max_files)]
    index_lines = [f"# CSV report ({args.mode})", ""]

    if not csvs:
        index_lines.extend(["no csv found", ""])
    
    grouped: dict[str, list[dict[str, str]]] = {}

    for csv_path in csvs:
        rel = csv_path.relative_to(output_root)
        key = safe_key(rel)
        step = rel.parts[0] if len(rel.parts) > 0 else "unknown"
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            (tables_dir / f"{key}.md").write_text(f"# {rel}\n\nread error: {e}\n", encoding="utf-8")
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

    index_lines.append("## CSV list")
    index_lines.append("")
    for step in sorted(grouped.keys()):
        index_lines.append(f"### {step}")
        index_lines.append("")
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
