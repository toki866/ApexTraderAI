#!/usr/bin/env python3
"""Run a Python script with sys.executable using argv execution (shell=False)."""
from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path


def _format_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def main() -> int:
    if len(sys.argv) < 2:
        print("[ERROR] Usage: run_with_python.py <script> [args...]", file=sys.stderr)
        return 2

    script = sys.argv[1]
    if not Path(script).exists():
        print(f"[ERROR] Script not found: {script}", file=sys.stderr)
        return 2

    cmd = [sys.executable, script, *sys.argv[2:]]
    print(f"[CMD] {_format_cmd(cmd)}")
    completed = subprocess.run(cmd, shell=False, check=False)
    print(f"[RC] {completed.returncode}")
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
