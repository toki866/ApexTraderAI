from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _agg(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=keys + ["count", "mean_ms", "p50_ms", "p95_ms", "max_ms"])
    g = df.groupby(keys, dropna=False)["elapsed_ms"]
    out = g.agg(count="count", mean_ms="mean", max_ms="max").reset_index()
    out["p50_ms"] = g.quantile(0.50).values
    out["p95_ms"] = g.quantile(0.95).values
    return out[keys + ["count", "mean_ms", "p50_ms", "p95_ms", "max_ms"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", default="output")
    ap.add_argument("--mode", default="sim")
    args = ap.parse_args()

    timing_dir = Path(args.output_root) / "timing" / str(args.mode)
    events_path = timing_dir / "timing_events.jsonl"
    if not events_path.exists():
        raise FileNotFoundError(f"timing events not found: {events_path}")

    df = pd.read_json(events_path, lines=True)
    if "agent_id" not in df.columns:
        df["agent_id"] = ""
    df["agent_id"] = df["agent_id"].fillna("").astype(str)

    summary = _agg(df, ["branch_id", "stage", "agent_id"])
    summary_path = timing_dir / "timing_summary.csv"
    summary.to_csv(summary_path, index=False)

    branch_df = df[df["stage"] == "branch.total"].copy()
    branch_summary = _agg(branch_df, ["branch_id"])
    branch_summary_path = timing_dir / "timing_branch_summary.csv"
    branch_summary.to_csv(branch_summary_path, index=False)

    print(f"[timing] wrote {summary_path}")
    print(f"[timing] wrote {branch_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
