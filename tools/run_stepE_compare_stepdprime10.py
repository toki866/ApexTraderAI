# tools/run_stepE_compare_stepdprime10.py
# -*- coding: utf-8 -*-
"""
Run StepE 10 times (10 independent RL agents) using StepD' embedding streams.

This script assumes you have already generated StepD' embeddings with:
  python tools/run_stepd_prime_features.py --symbol SOXL --mode sim --sources bnf,all_features --scales 1,2,3 --fit-split train --export-split all

Expected embedding CSVs (prefer "all" export):
  output/stepD_prime/<mode>/embeddings/
    stepDprime_bnf_h01_<SYMBOL>_embeddings_all.csv
    stepDprime_bnf_h02_<SYMBOL>_embeddings_all.csv
    stepDprime_bnf_h03_<SYMBOL>_embeddings_all.csv
    stepDprime_all_features_h01_<SYMBOL>_embeddings_all.csv
    stepDprime_all_features_h02_<SYMBOL>_embeddings_all.csv
    stepDprime_all_features_h03_<SYMBOL>_embeddings_all.csv

Legacy (test-only) names without "_all" are also accepted, but those will NOT have train rows, so RL may degenerate.

Outputs:
  output/stepE/<mode>/
    stepE_equity_<agent>_<SYMBOL>.csv
    stepE_daily_log_<agent>_<SYMBOL>.csv
    stepE_summary_<agent>_<SYMBOL>.json
    stepE_compare10_<SYMBOL>.csv
"""

from __future__ import annotations

import sys
import os
from pathlib import Path as _Path

# Ensure repo root is on sys.path so `import ai_core` works when running as `python tools/<script>.py`.
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
# Also force CWD to repo root so relative paths like 'output/...' are stable.
try:
    os.chdir(_REPO_ROOT)
except Exception:
    pass


import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import pandas as pd


def _read_split_summary(output_root: Path, mode: str, symbol: str) -> Dict[str, str]:
    base = output_root / "stepA" / mode
    p = base / f"stepA_split_summary_{symbol}.csv"
    if not p.exists():
        cand = list(base.glob(f"*split*summary*{symbol}*.csv"))
        if cand:
            p = cand[0]
    if not p.exists():
        raise FileNotFoundError(f"Split summary not found under {base}")

    df = pd.read_csv(p)

    # common format: key,value
    if "key" in df.columns and "value" in df.columns:
        m = dict(zip(df["key"].astype(str), df["value"].astype(str)))
        return m

    # alternative wide format
    m: Dict[str, str] = {}
    for c in df.columns:
        if len(df) == 1:
            m[c] = str(df.iloc[0][c])
    return m


def _parse_date(s: str) -> str:
    s = str(s).strip()
    # allow YYYY-MM-DD already
    return s[:10]


def _resolve_embed_path(output_root: Path, mode: str, symbol: str, src: str, h: int) -> Path:
    """Resolve StepD' embedding CSV path.

    Preferred location:
      <output_root>/stepD_prime/<mode>/embeddings/stepDprime_<src>_h01_<SYMBOL>_embeddings_all.csv

    We also apply a fallback search (rglob) so the runner remains robust even if the working
    directory or output_root differs.
    """
    hh = f"h{int(h):02d}"
    base = output_root / "stepD_prime" / mode / "embeddings"
    cands = [
        base / f"stepDprime_{src}_{hh}_{symbol}_embeddings_all.csv",
        base / f"stepDprime_{src}_{hh}_{symbol}_embeddings.csv",  # legacy (often test-only)
        base / f"stepDprime_{src}_{hh}_{symbol}_embeddings_test.csv",
        base / f"stepDprime_{src}_{hh}_{symbol}_embeddings_train.csv",
    ]
    for p in cands:
        if p.exists():
            return p

    # Fallback: search anywhere under output_root (in case of unexpected CWD / path layout).
    pats = [
        f"stepDprime_{src}_{hh}_{symbol}_embeddings_all.csv",
        f"stepDprime_{src}_{hh}_{symbol}_embeddings.csv",
        f"stepDprime_{src}_{hh}_{symbol}_embeddings_test.csv",
        f"stepDprime_{src}_{hh}_{symbol}_embeddings_train.csv",
        f"stepDprime_{src}_{hh}_*{symbol}*_embeddings*.csv",
    ]
    for pat in pats:
        found = list(output_root.rglob(pat))
        if found:
            return sorted(found)[0]

    return cands[0]  # return first expected for error message


def _check_required_embeddings(output_root: Path, mode: str, symbol: str, sources: List[str], horizons: List[int]) -> Tuple[bool, List[str]]:
    missing = []
    for src in sources:
        for h in horizons:
            p = _resolve_embed_path(output_root, mode, symbol, src, h)
            if not p.exists():
                missing.append(str(p))
    return (len(missing) == 0), missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--mode", default="sim", choices=["sim", "live", "ops", "display"])
    ap.add_argument("--output-root", default="output")
    ap.add_argument("--profile", default="A", choices=["A", "B", "C", "D"])
    ap.add_argument("--trade-cost-bps", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    symbol = args.symbol
    mode = args.mode
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = _REPO_ROOT / output_root
    split = _read_split_summary(output_root, mode, symbol)

    # Minimal date_range object that StepEService understands
    date_range = SimpleNamespace(
        train_start=_parse_date(split.get("train_start", split.get("train_start_date", split.get("train_start_dt", "")))),
        train_end=_parse_date(split.get("train_end", split.get("train_end_date", split.get("train_end_dt", "")))),
        test_start=_parse_date(split.get("test_start", split.get("test_start_date", split.get("test_start_dt", "")))),
        test_end=_parse_date(split.get("test_end", split.get("test_end_date", split.get("test_end_dt", "")))),
        mode=mode,
    )

    # 10 independent agents based on available StepD' streams
    # (No mamba files required here.)
    runs: List[Tuple[str, str, List[str], List[int], str]] = [
        ("dprime_bnf_h01", "stepdprime", ["bnf"], [1], "concat"),
        ("dprime_bnf_h02", "stepdprime", ["bnf"], [2], "concat"),
        ("dprime_bnf_h03", "stepdprime", ["bnf"], [3], "concat"),
        ("dprime_all_features_h01", "stepdprime", ["all_features"], [1], "concat"),
        ("dprime_all_features_h02", "stepdprime", ["all_features"], [2], "concat"),
        ("dprime_all_features_h03", "stepdprime", ["all_features"], [3], "concat"),
        ("dprime_bnf_3scale", "stepdprime", ["bnf"], [1, 2, 3], "concat"),
        ("dprime_all_features_3scale", "stepdprime", ["all_features"], [1, 2, 3], "concat"),
        ("dprime_mix_h01", "stepdprime", ["bnf", "all_features"], [1], "concat"),
        ("dprime_mix_3scale", "stepdprime", ["bnf", "all_features"], [1, 2, 3], "concat"),
    ]

    from ai_core.services.step_e_service import StepEService, StepEConfig

    configs: List[StepEConfig] = []
    runs_used: List[str] = []

    for agent, kind, sources, hs, join in runs:
        ok, missing = _check_required_embeddings(output_root, mode, symbol, sources, hs)
        if not ok:
            print(f"[compare10] SKIP agent={agent} missing={len(missing)} e.g. {missing[0]}")
            continue

        cfg = StepEConfig(
            agent=agent,
            output_root=str(output_root),
            obs_profile=args.profile,
            seed=int(args.seed),
            verbose=True,
            trade_cost_bps=float(args.trade_cost_bps),

            # StepD' settings
            use_stepd_prime=True,
            dprime_sources=",".join(sources),
            dprime_horizons=",".join(str(int(x)) for x in hs),
            dprime_join=join,

            # RL hyperparams (lightweight)
            policy_kind="diffpg",
            hidden_dim=64,
            lr=1e-3,
            epochs=120,
            patience=15,
            val_ratio=0.2,
            weight_decay=1e-4,
            pos_l2=1e-3,
            smooth_abs_eps=1e-6,
        )
        configs.append(cfg)
        runs_used.append(agent)

    if not configs:
        raise SystemExit("[compare10] No runnable agents. Generate StepD' embeddings first (export-split all).")

    app_config = SimpleNamespace(output_root=str(output_root), stepE=configs)
    svc = StepEService(app_config)

    svc.run(date_range=date_range, symbol=symbol, agents=runs_used)

    # Summarize results
    rows = []
    for agent in runs_used:
        summ = output_root / "stepE" / mode / f"stepE_summary_{agent}_{symbol}.json"
        if summ.exists():
            try:
                d = json.loads(summ.read_text(encoding="utf-8"))
                rows.append(d)
            except Exception:
                pass

    out_dir = output_root / "stepE" / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / f"stepE_compare10_{symbol}.csv"
    if rows:
        df = pd.DataFrame(rows)
        # keep a compact useful subset if present
        keep = [c for c in [
            "agent", "mode", "test_start", "test_end", "rows_test",
            "total_return", "cagr", "sharpe", "max_drawdown", "num_trades"
        ] if c in df.columns]
        if keep:
            df = df[keep]
        df.to_csv(out_csv, index=False)
        print(f"[compare10] wrote -> {out_csv} rows={len(df)}")
    else:
        print("[compare10] No summaries found to aggregate.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
