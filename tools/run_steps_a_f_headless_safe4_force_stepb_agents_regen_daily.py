# -*- coding: utf-8 -*-
"""Runner wrapper to keep StepB daily outputs fresh and consistent.

Why this exists:
- The StepB implementation may keep previously generated daily outputs when a daily manifest already exists.
- If you re-run with changed settings, that can mix old daily outputs with new summary outputs.

This wrapper:
1) Computes a signature from CLI args and environment.
2) If changed (or --always-regenerate-daily), purges StepB daily outputs.
3) Runs the original headless runner script.
4) Repairs daily outputs from pred_path summaries to guarantee consistency.
5) Runs sanity check and exits non-zero on failure.

Note: This does NOT feed predictions back into model inputs.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from stepb_daily_cache_utils import (
    build_signature_from_args,
    load_signature,
    purge_stepb_daily_outputs,
    repair_stepb_daily_from_pred_path,
    run_sanity_check,
    write_signature,
)


def _repo_root_from_this_file() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--test-start", required=True)
    p.add_argument("--train-years", type=int, required=True)
    p.add_argument("--test-months", type=int, required=True)
    p.add_argument("--steps", required=True)

    p.add_argument("--enable-mamba", action="store_true")
    p.add_argument("--enable-mamba-periodic", action="store_true")
    p.add_argument("--mamba-mode", default="sim")
    p.add_argument("--mamba-lookback", type=int, default=30)
    p.add_argument("--mamba-horizons", default="1")

    p.add_argument("--always-regenerate-daily", action="store_true")
    return p


def main() -> int:
    args = _build_arg_parser().parse_args()

    repo_root = _repo_root_from_this_file()
    stepb_dir = repo_root / "output" / "stepB" / args.mamba_mode

    sig = build_signature_from_args(
        symbol=args.symbol,
        mode=args.mamba_mode,
        lookback=args.mamba_lookback,
        horizons_csv=args.mamba_horizons,
        enable_mamba=bool(args.enable_mamba),
        enable_mamba_periodic=bool(args.enable_mamba_periodic),
    )

    prev = load_signature(stepb_dir, args.symbol)
    prev_hash = (prev or {}).get("hash") if isinstance(prev, dict) else None
    cur_hash = sig.stable_hash()

    need_purge = args.always_regenerate_daily or (prev_hash != cur_hash)

    if need_purge:
        reason = "forced" if args.always_regenerate_daily else "signature changed"
        print(f"[regen_daily] StepB daily purge: {reason}")
        removed = purge_stepb_daily_outputs(stepb_dir, args.symbol)
        for p in removed:
            print(f"[regen_daily] removed: {p}")

    # Run original runner
    runner = repo_root / "run_steps_a_f_headless_safe4_force_stepb_agents.py"
    if not runner.exists():
        print(f"[regen_daily] ERROR: base runner not found: {runner}")
        return 2

    cmd = [
        sys.executable,
        str(runner),
        "--symbol",
        args.symbol,
        "--test-start",
        args.test_start,
        "--train-years",
        str(args.train_years),
        "--test-months",
        str(args.test_months),
        "--steps",
        args.steps,
    ]

    if args.enable_mamba:
        cmd.append("--enable-mamba")
    if args.enable_mamba_periodic:
        cmd.append("--enable-mamba-periodic")

    cmd += [
        "--mamba-mode",
        args.mamba_mode,
        "--mamba-lookback",
        str(args.mamba_lookback),
        "--mamba-horizons",
        args.mamba_horizons,
    ]

    print("[regen_daily] running:", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"[regen_daily] base runner failed: rc={r.returncode}")
        return r.returncode

    # Repair then sanity check
    log_lines = []
    repair_stepb_daily_from_pred_path(repo_root, args.symbol, args.mamba_mode, log_lines)
    for line in log_lines:
        print(line)

    report_path = stepb_dir / f"stepB_daily_sanity_report_{args.symbol}.txt"
    passed = run_sanity_check(repo_root, args.symbol, args.mamba_mode, report_path)
    print(f"[regen_daily] wrote sanity report: {report_path}")

    # Write signature only if we have consistent outputs.
    if passed:
        write_signature(stepb_dir, args.symbol, sig)
        print("[regen_daily] SANITY CHECK PASSED")
        return 0
    else:
        print("[regen_daily] SANITY CHECK FAILED. See report above.")
        return 10


if __name__ == "__main__":
    raise SystemExit(main())
