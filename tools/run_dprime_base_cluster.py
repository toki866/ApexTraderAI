from __future__ import annotations

import argparse
from pathlib import Path

from ai_core.services.step_dprime_service import StepDPrimeService
from tools.run_pipeline import _apply_config_output_root, _build_stepdprime_config, _get_app_config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbol', required=True)
    ap.add_argument('--mode', default='sim')
    ap.add_argument('--output-root', default='output')
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out = Path(args.output_root).expanduser()
    if not out.is_absolute():
        out = (repo_root / out).resolve()

    app_cfg = _apply_config_output_root(_get_app_config(repo_root), out)
    cfg = _build_stepdprime_config(app_cfg, args.symbol, args.mode, output_root=str(out))
    result = StepDPrimeService().run_base_cluster(cfg)
    print(result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
