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
        verbose=not args.quiet,
    )
    StepDPrimeService().run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
