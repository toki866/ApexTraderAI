from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ai_core.services.step_dprime_service import StepDPrimeConfig, StepDPrimeService, _PROFILES


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--mode", default="sim")
    ap.add_argument("--output-root", default="output")
    ap.add_argument("--profiles", default="all", help="all or comma list")
    ap.add_argument("--l-past", type=int, default=63)
    ap.add_argument("--pred-k", type=int, default=20)
    ap.add_argument("--z-past-dim", type=int, default=32)
    ap.add_argument("--z-pred-dim", type=int, default=32)
    ap.add_argument("--z-state-dim", type=int, default=64)
    ap.add_argument("--encoder-type", choices=["legacy", "transformer"], default="legacy")
    ap.add_argument("--transformer-d-model", type=int, default=64)
    ap.add_argument("--transformer-nhead", type=int, default=4)
    ap.add_argument("--transformer-num-layers", type=int, default=2)
    ap.add_argument("--transformer-ff-dim", type=int, default=128)
    ap.add_argument("--transformer-dropout", type=float, default=0.1)
    ap.add_argument("--transformer-epochs", type=int, default=4)
    ap.add_argument("--transformer-batch-size", type=int, default=64)
    ap.add_argument("--transformer-lr", type=float, default=1e-3)
    ap.add_argument("--transformer-mask-ratio", type=float, default=0.15)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.profiles.strip().lower() == "all":
        profiles = _PROFILES
    else:
        profiles = tuple([x.strip() for x in args.profiles.split(",") if x.strip()])

    cfg = StepDPrimeConfig(
        symbol=args.symbol,
        mode=args.mode,
        output_root=args.output_root,
        profiles=profiles,
        l_past=args.l_past,
        pred_k=args.pred_k,
        z_past_dim=args.z_past_dim,
        z_pred_dim=args.z_pred_dim,
        z_state_dim=args.z_state_dim,
        encoder_type=args.encoder_type,
        transformer_d_model=args.transformer_d_model,
        transformer_nhead=args.transformer_nhead,
        transformer_num_layers=args.transformer_num_layers,
        transformer_ff_dim=args.transformer_ff_dim,
        transformer_dropout=args.transformer_dropout,
        transformer_epochs=args.transformer_epochs,
        transformer_batch_size=args.transformer_batch_size,
        transformer_lr=args.transformer_lr,
        transformer_mask_ratio=args.transformer_mask_ratio,
        verbose=not args.quiet,
    )
    StepDPrimeService().run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
