#!/usr/bin/env python
from __future__ import annotations

import argparse
from datetime import datetime

import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _default_date(mode: str) -> str:
    m = str(mode or "sim").lower()
    if m == "live":
        return str(pd.Timestamp.now(tz="America/New_York").date())
    return str(datetime.utcnow().date())


def main() -> None:
    ap = argparse.ArgumentParser(description="Run StepF close-pre two-stage router")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--mode", default="sim", choices=["sim", "live", "ops", "prod"])
    ap.add_argument("--date", default=None, help="YYYY-MM-DD")
    ap.add_argument("--output-root", default="output")
    ap.add_argument("--stage0-topk", type=int, default=3)
    ap.add_argument("--fit-window-days", type=int, default=504)
    ap.add_argument("--pca-n-components", type=int, default=30)
    ap.add_argument("--hdbscan-min-cluster-size", type=int, default=30)
    ap.add_argument("--hdbscan-min-samples", type=int, default=10)
    ap.add_argument("--safe-set", type=str, default="dprime_bnf_h01,dprime_all_features_h01")
    ap.add_argument("--topK-agents-per-regime", type=int, default=3)
    ap.add_argument("--dry-run", type=int, default=1)
    args = ap.parse_args()

    from ai_core.live.stepf_two_stage_router import run_close_pre

    target_date = args.date or _default_date(args.mode)
    cfg = {
        "stage0_topk": args.stage0_topk,
        "fit_window_days": args.fit_window_days,
        "pca_n_components": args.pca_n_components,
        "hdbscan_min_cluster_size": args.hdbscan_min_cluster_size,
        "hdbscan_min_samples": args.hdbscan_min_samples,
        "safe_set": args.safe_set,
        "topK_agents_per_regime": args.topK_agents_per_regime,
    }
    decision = run_close_pre(
        symbol=args.symbol,
        mode=args.mode,
        target_date=target_date,
        output_root=args.output_root,
        config=cfg,
    )
    print(
        f"[StepF close-pre] symbol={args.symbol} mode={decision['mode']} date={decision['target_date']} "
        f"regime_id={decision['stage1']['regime_id_final']} ratio_final={decision['ratio_final']:.6f} "
        f"agents_inferred={len(decision['ratios'])} total_sec={decision['timing']['total_sec']:.3f}"
    )


if __name__ == "__main__":
    main()
