# tools/run_stepF_marl_gate.py
# -*- coding: utf-8 -*-
"""
Convenience runner for StepF gating.

Example:
  python tools/run_stepF_marl_gate.py --symbol SOXL --mode sim --agents dprime_bnf_h01,dprime_bnf_h02,dprime_bnf_h03,dprime_all_features_h01,dprime_all_features_h02,dprime_all_features_h03,dprime_bnf_3scale,dprime_all_features_3scale,dprime_mix_h01,dprime_mix_3scale

It will read StepA split_summary to determine train/test dates.
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
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

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
    if "key" in df.columns and "value" in df.columns:
        return dict(zip(df["key"].astype(str), df["value"].astype(str)))
    out: Dict[str, str] = {}
    if len(df) == 1:
        for c in df.columns:
            out[c] = str(df.iloc[0][c])
    return out


def _parse_date(s: str) -> str:
    s = str(s).strip()
    return s[:10]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--mode", default="sim", choices=["sim", "live", "ops", "display"])
    ap.add_argument("--output-root", default="output")
    ap.add_argument("--agents", required=True, help="comma-separated StepE agent names")
    ap.add_argument("--trade-cost-bps", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    symbol = args.symbol
    mode = args.mode
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = _REPO_ROOT / output_root
    split = _read_split_summary(output_root, mode, symbol)
    date_range = SimpleNamespace(
        train_start=_parse_date(split.get("train_start", split.get("train_start_date", ""))),
        train_end=_parse_date(split.get("train_end", split.get("train_end_date", ""))),
        test_start=_parse_date(split.get("test_start", split.get("test_start_date", ""))),
        test_end=_parse_date(split.get("test_end", split.get("test_end_date", ""))),
        mode=mode,
    )

    from ai_core.services.step_f_service import StepFService, StepFConfig
    from types import SimpleNamespace as SN

    cfg = StepFConfig(
        output_root=str(output_root),
        agents=str(args.agents),
        seed=int(args.seed),
        trade_cost_bps=float(args.trade_cost_bps),
        verbose=True,
    )
    app_config = SN(output_root=str(output_root), stepF=cfg)
    svc = StepFService(app_config)
    svc.run(date_range=date_range, symbol=symbol, mode=mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
