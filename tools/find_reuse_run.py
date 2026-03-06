#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI tool: scan WORK_ROOT for a prior run matching the given signature.

Prints the output_root path of the best matching run (if found), or nothing.
Exit code is always 0 to avoid BAT for-loop errors.

Called from run_all_local_then_copy.bat (via WSL) when REUSE_OUTPUT=1.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow import from repo root (tools/ is on sys.path when run via run_with_python.py,
# but also support direct invocation).
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.run_manifest import build_run_signature, find_latest_matching_run  # noqa: E402


def _parse_int_list(s: str):
    out = []
    for part in (s or "").split(","):
        part = part.strip()
        if part:
            try:
                out.append(int(part))
            except ValueError:
                pass
    return tuple(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Find a matching prior run in scan_root.")
    ap.add_argument("--scan-root", required=True, help="Root directory to scan (WORK_ROOT).")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--mode", default="sim")
    ap.add_argument("--test-start", dest="test_start", default="")
    ap.add_argument("--train-years", dest="train_years", type=int, default=8)
    ap.add_argument("--test-months", dest="test_months", type=int, default=3)
    ap.add_argument("--steps", default="A,B,C,DPRIME,E,F")
    ap.add_argument("--enable-mamba", dest="enable_mamba", type=int, default=0)
    ap.add_argument("--enable-mamba-periodic", dest="enable_mamba_periodic", type=int, default=0)
    ap.add_argument("--mamba-lookback", dest="mamba_lookback", type=int, default=None)
    ap.add_argument("--mamba-horizons", dest="mamba_horizons", default="")
    ap.add_argument("--stepe-agents", dest="stepe_agents", default="")
    args = ap.parse_args()

    steps_parsed = tuple(
        s.strip().upper() for s in args.steps.split(",") if s.strip()
    )
    # Normalise DPRIME alias
    steps_parsed = tuple("DPRIME" if s in ("D", "DPRIME") else s for s in steps_parsed)

    stepe_agents_parsed = None
    if args.stepe_agents.strip():
        stepe_agents_parsed = tuple(
            a.strip() for a in args.stepe_agents.split(",") if a.strip()
        )

    sig = build_run_signature(
        symbol=args.symbol,
        mode=args.mode,
        test_start=args.test_start,
        train_years=args.train_years,
        test_months=args.test_months,
        steps=steps_parsed,
        enable_mamba=bool(args.enable_mamba),
        enable_mamba_periodic=bool(args.enable_mamba_periodic),
        mamba_lookback=args.mamba_lookback,
        mamba_horizons=_parse_int_list(args.mamba_horizons),
        stepe_agents=stepe_agents_parsed,
    )

    scan_root = Path(args.scan_root)
    result = find_latest_matching_run(scan_root, sig)
    if result is not None:
        print(str(result))

    return 0


if __name__ == "__main__":
    sys.exit(main())
