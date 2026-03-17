from __future__ import annotations

import argparse

from ai_core.services.step_dprime_service import StepDPrimeConfig, StepDPrimeService


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbol', required=True)
    ap.add_argument('--profile', required=True)
    ap.add_argument('--mode', default='sim')
    ap.add_argument('--output-root', default='output')
    args = ap.parse_args()
    cfg = StepDPrimeConfig(symbol=args.symbol, mode=args.mode, output_root=args.output_root)
    out = StepDPrimeService().run_final_profile(cfg, args.profile)
    print(out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
