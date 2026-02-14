# -*- coding: utf-8 -*-
"""
tools/build_phase2_state.py

prices + agent_csv から phase2_state_<SYMBOL>.csv を作る簡易ジェネレータ。

本来 phase2_state は StepF Phase2 の成果物（regime/trend/phase + agreement指標）だが、
現状ジェネレータ未整備/未実行のケースがあるため、後段の router_bandit backtest を止めない目的。

出力列（router_bandit/io_utils.load_phase2_state の必須列）:
  Date, regime_cluster, trend_cluster, phase, agreement_dist, agreement_label

実行例:
python tools\\build_phase2_state.py ^
  --symbol SOXL --mode sim --output-root output ^
  --prices-soxl-train output\\stepA\\sim\\stepA_prices_train_SOXL.csv ^
  --prices-soxl-test  output\\stepA\\sim\\stepA_prices_test_SOXL.csv ^
  --prices-soxs-train output\\stepA\\sim\\stepA_prices_train_SOXS.csv ^
  --prices-soxs-test  output\\stepA\\sim\\stepA_prices_test_SOXS.csv ^
  --test-start 2022-01-03 --train-years 8 --test-months 3 ^
  --agent-csv a1=output\\stepE\\sim\\stepE_daily_log_dprime_bnf_h02_SOXL.csv

"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve()
REPO_ROOT = _THIS.parents[1]
TOOLS_DIR = _THIS.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from router_bandit.io_utils import load_prices_close, load_agent_ratio_series, parse_kv_list
from router_bandit.backtest_runner import compute_train_test_ranges


def _concat_prices(train_path: Optional[str], test_path: Optional[str], single_path: Optional[str], label: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    if single_path:
        frames.append(load_prices_close(single_path, label=label))
    else:
        if train_path:
            frames.append(load_prices_close(train_path, label=label))
        if test_path:
            frames.append(load_prices_close(test_path, label=label))
    if not frames:
        raise SystemExit(f"ERROR: prices for {label} が指定されていません。")
    df = pd.concat(frames, axis=0, ignore_index=True)
    df = df.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df


def _bin_by_quantiles(x: np.ndarray, qs: Tuple[float, float]) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    q0, q1 = np.nanpercentile(x, [qs[0], qs[1]])
    out = np.zeros_like(x, dtype=int)
    out[x > q0] = 1
    out[x > q1] = 2
    out[np.isnan(x)] = 1
    return out


def build_phase2(
    px_soxl: pd.DataFrame,
    agent_series: Dict[str, pd.DataFrame],
    test_end: str,
    px_soxs: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    dates = pd.Series(px_soxl["Date"]).astype(str)
    if px_soxs is not None:
        soxs_set = set(pd.Series(px_soxs["Date"]).astype(str).tolist())
        dates = dates[dates.isin(soxs_set)]
    dates = dates.sort_values().reset_index(drop=True)

    df = pd.DataFrame({"Date": dates})
    df = df.merge(px_soxl[["Date", "Close_SOXL"]], on="Date", how="left")
    df["Close_SOXL"] = df["Close_SOXL"].astype(float)
    df["logret"] = np.log(df["Close_SOXL"]).diff()

    vol = df["logret"].rolling(20, min_periods=5).std().to_numpy(dtype=float)
    trd = df["logret"].rolling(20, min_periods=5).mean().to_numpy(dtype=float)

    df["regime_cluster"] = _bin_by_quantiles(vol, (33.0, 66.0))
    df["trend_cluster"] = _bin_by_quantiles(trd, (33.0, 66.0))
    df["phase"] = (df["trend_cluster"].astype(int) * 3 + df["regime_cluster"].astype(int)).astype(int)

    wide = df[["Date"]].copy()
    for name, s in agent_series.items():
        tmp = s.rename(columns={"agent_ratio": f"ratio_{name}"})
        wide = wide.merge(tmp[["Date", f"ratio_{name}"]], on="Date", how="left")
    ratio_cols = [c for c in wide.columns if c.startswith("ratio_")]

    mean_ratio = wide[ratio_cols].mean(axis=1, skipna=True).fillna(0.0).astype(float).to_numpy()
    mean_ratio = np.clip(mean_ratio, -1.0, 1.0)
    df["agreement_label"] = mean_ratio
    df["agreement_dist"] = np.abs(mean_ratio)

    df = df[df["Date"] <= str(test_end)].copy().reset_index(drop=True)

    df["regime_cluster"] = df["regime_cluster"].astype(int)
    df["trend_cluster"] = df["trend_cluster"].astype(int)
    df["phase"] = df["phase"].astype(int)
    df["agreement_dist"] = df["agreement_dist"].astype(float)
    df["agreement_label"] = df["agreement_label"].astype(float)

    return df[["Date", "regime_cluster", "trend_cluster", "phase", "agreement_dist", "agreement_label"]].copy()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build phase2_state CSV from prices + agent ratios")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--mode", default="sim", choices=["sim", "live", "ops", "display"])
    ap.add_argument("--output-root", default="output")

    ap.add_argument("--prices-soxl", default=None)
    ap.add_argument("--prices-soxs", default=None)
    ap.add_argument("--prices-soxl-train", default=None)
    ap.add_argument("--prices-soxl-test", default=None)
    ap.add_argument("--prices-soxs-train", default=None)
    ap.add_argument("--prices-soxs-test", default=None)

    ap.add_argument("--test-start", required=True)
    ap.add_argument("--train-years", required=True, type=int)
    ap.add_argument("--test-months", required=True, type=int)

    ap.add_argument("--out", default="", help="Optional output CSV path (default: output/stepF/<mode>/phase2_state_<SYMBOL>.csv)")

    ap.add_argument(
        "--agent-csv",
        action="append",
        default=[],
        type=str,
        help="Repeatable: name=path_to_csv_with_daily_ratio",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if len(args.agent_csv) < 1:
        raise SystemExit("ERROR: --agent-csv name=path を少なくとも1つ指定してください。")

    agent_paths = parse_kv_list(args.agent_csv)

    train_start, test_end = compute_train_test_ranges(args.test_start, args.train_years, args.test_months)

    px_soxl = _concat_prices(args.prices_soxl_train, args.prices_soxl_test, args.prices_soxl, label="SOXL")
    px_soxs = None
    if args.prices_soxs_train or args.prices_soxs_test or args.prices_soxs:
        px_soxs = _concat_prices(args.prices_soxs_train, args.prices_soxs_test, args.prices_soxs, label="SOXS")

    agent_series = {k: load_agent_ratio_series(v) for k, v in agent_paths.items()}

    phase2 = build_phase2(px_soxl, agent_series, test_end=test_end, px_soxs=px_soxs)

    out_path = Path(args.out) if args.out else (Path(args.output_root) / "stepF" / args.mode / f"phase2_state_{args.symbol}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    phase2.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] wrote phase2_state -> {out_path}")
    print(f"range: {phase2['Date'].min()} .. {phase2['Date'].max()}  rows={len(phase2)}")


if __name__ == "__main__":
    main()
