from __future__ import annotations

import argparse
import subprocess
import sys
import uuid
from pathlib import Path

OFFICIAL_AGENTS = [
    "dprime_bnf_h01",
    "dprime_bnf_h02",
    "dprime_bnf_3scale",
    "dprime_mix_h01",
    "dprime_mix_h02",
    "dprime_mix_3scale",
    "dprime_all_features_h01",
    "dprime_all_features_h02",
    "dprime_all_features_h03",
    "dprime_all_features_3scale",
]


def _run(cmd: list[str]) -> None:
    print("[measure]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SOXL")
    ap.add_argument("--test-start", required=True)
    ap.add_argument("--train-years", type=int, default=8)
    ap.add_argument("--test-months", type=int, default=3)
    ap.add_argument("--output-root", default="output")
    ap.add_argument("--clear-timing", type=int, default=0, choices=[0, 1])
    args = ap.parse_args()

    py = sys.executable
    rp = Path("tools/run_pipeline.py")

    branches: list[tuple[str, str, str | None]] = [
        ("A_only_prep", "A", None),
        ("AB", "A,B", None),
        ("B_only_warm", "B", None),
        ("ABC", "A,B,C", None),
        ("ABCDPRIME", "A,B,C,DPRIME", None),
        ("DPRIME_only_warm", "DPRIME", None),
        ("E_all_warm", "E", None),
    ]
    branches += [(f"E_one_{a}", "E", a) for a in OFFICIAL_AGENTS]
    branches += [
        ("F_only_warm", "F", None),
        ("DEF_all", "DPRIME,E,F", None),
        ("FULL_A_to_F", "A,B,C,DPRIME,E,F", None),
    ]

    first = True
    for branch_id, steps, one_agent in branches:
        run_id = f"{branch_id}_{uuid.uuid4().hex[:8]}"
        cmd = [
            py,
            str(rp),
            "--symbol",
            args.symbol,
            "--test-start",
            args.test_start,
            "--train-years",
            str(args.train_years),
            "--test-months",
            str(args.test_months),
            "--output-root",
            args.output_root,
            "--mode",
            "sim",
            "--steps",
            steps,
            "--timing",
            "1",
            "--run-id",
            run_id,
            "--branch-id",
            branch_id,
            "--execution-mode",
            "sequential",
            "--clear-timing",
            str(args.clear_timing if first else 0),
        ]
        if one_agent:
            cmd += ["--stepe-agents", one_agent]
        _run(cmd)
        first = False

    _run([py, "tools/summarize_timing.py", "--output-root", args.output_root, "--mode", "sim"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
