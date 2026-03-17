from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ai_core.services.step_e_service import StepEConfig, StepEService
from tools.run_pipeline import _apply_config_output_root, _get_app_config


def _clone_cfg_from_existing(raw: Any, *, output_root: str) -> StepEConfig:
    if isinstance(raw, StepEConfig):
        data = dict(raw.__dict__)
    elif isinstance(raw, dict):
        data = dict(raw)
    else:
        data = {k: getattr(raw, k) for k in StepEConfig.__dataclass_fields__.keys() if hasattr(raw, k)}
    data["output_root"] = str(output_root)
    filtered = {k: v for k, v in data.items() if k in StepEConfig.__dataclass_fields__}
    return StepEConfig(**filtered)


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

    repo_root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = (repo_root / output_root).resolve()

    app_cfg = _get_app_config(repo_root)
    app_cfg = _apply_config_output_root(app_cfg, output_root)

    raw_cfgs = app_cfg.get("stepE") if isinstance(app_cfg, dict) else getattr(app_cfg, "stepE", None)
    cfgs = list(raw_cfgs) if isinstance(raw_cfgs, (list, tuple)) else ([raw_cfgs] if raw_cfgs else [])

    selected = None
    for raw in cfgs:
        if raw is None:
            continue
        agent_name = raw.get("agent") if isinstance(raw, dict) else getattr(raw, "agent", None)
        if str(agent_name or "") == str(args.agent):
            selected = raw
            break
    if selected is None:
        raise RuntimeError(f"StepE config not found for agent={args.agent}")

    cfg = _clone_cfg_from_existing(selected, output_root=str(output_root))
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
