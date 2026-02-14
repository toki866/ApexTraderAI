# -*- coding: utf-8 -*-
'''
tools/run_stepd_prime.py

CLI runner for StepD' (Transformer summarizer) to produce embeddings from StepB predicted paths.

IMPORTANT:
  - When running as a script (python tools\run_stepd_prime.py), Python's import path may not include the repo root.
    This file bootstraps the repo root into sys.path so that `import ai_core...` works reliably.
'''

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is in sys.path (so `import ai_core` works even when sys.path[0] == tools/)
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ai_core.services.step_dprime_service import StepDPrimeConfig, StepDPrimeService  # noqa: E402


def _csv_ints(s: str):
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def _csv_strs(s: str):
    return tuple(x.strip() for x in s.split(",") if x.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--mode", default="sim")
    ap.add_argument("--output-root", default="output")
    ap.add_argument("--sources", default="mamba_periodic,mamba", help="comma separated")
    ap.add_argument("--horizons", default="1,5,10,20", help="comma separated ints")

    ap.add_argument("--max-len", type=int, default=20)
    ap.add_argument("--embed-dim", type=int, default=32)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=int, default=1e-3)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--early-stop", type=int, default=5)

    ap.add_argument("--fit-split", default="available", choices=["train", "test", "train+test", "available"])
    ap.add_argument("--device", default="auto", help="auto/cpu/cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--export-labels-in-embeddings", action="store_true",
                    help="DEBUG: include label columns in embeddings CSV (default: off)")
    ap.add_argument("--export-meta-mode", default="none", choices=["none", "labels"],
                    help="Optional: write labels to <stepD_prime>/<mode>/meta/ (default: none)")
    args = ap.parse_args()

    cfg = StepDPrimeConfig(
        symbol=args.symbol,
        mode=args.mode,
        output_root=args.output_root,
        sources=_csv_strs(args.sources),
        horizons=_csv_ints(args.horizons),
        max_len=args.max_len,
        embedding_dim=args.embed_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=float(args.weight_decay),
        val_ratio=args.val_ratio,
        early_stop_patience=args.early_stop,
        fit_split=args.fit_split,
        device=args.device,
        seed=args.seed,
        verbose=not args.quiet,
    )

    svc = StepDPrimeService()
    svc.run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())