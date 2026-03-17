from __future__ import annotations

import argparse
from types import SimpleNamespace

from ai_core.services.step_e_service import StepEConfig, StepEService


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--agent', required=True)
    ap.add_argument('--symbol', required=True)
    ap.add_argument('--mode', default='sim')
    ap.add_argument('--output-root', default='output')
    ap.add_argument('--train-start', required=True)
    ap.add_argument('--train-end', required=True)
    ap.add_argument('--test-start', required=True)
    ap.add_argument('--test-end', required=True)
    args = ap.parse_args()

    cfg = StepEConfig(agent=args.agent, output_root=args.output_root)
    app_cfg = SimpleNamespace(output_root=args.output_root, stepE=[cfg])
    date_range = SimpleNamespace(
        mode=args.mode,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
    )
    svc = StepEService(app_cfg)
    print(svc.run_agent(cfg, date_range=date_range, symbol=args.symbol, mode=args.mode))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
