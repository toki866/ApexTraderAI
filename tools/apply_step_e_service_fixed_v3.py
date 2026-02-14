# -*- coding: utf-8 -*-
"""
Apply a fixed StepE service file (step_e_service.py) into the repository, with an automatic backup.

Usage:
  python tools\apply_step_e_service_fixed_v3.py

What it does:
  - copies tools\_payload_stepE\step_e_service.py into ai_core\services\step_e_service.py
  - creates a timestamped backup under old\step_e_service_old_YYYYMMDD_HHMMSS_XX.py
  - runs `python -m py_compile` on the updated file

This script is safe to run multiple times.
"""
from __future__ import annotations

from pathlib import Path
import datetime
import shutil
import subprocess
import sys

def _ts() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    payload = repo / "tools" / "_payload_stepE" / "step_e_service.py"
    dst = repo / "ai_core" / "services" / "step_e_service.py"
    old_dir = repo / "old"
    old_dir.mkdir(parents=True, exist_ok=True)

    if not payload.exists():
        print(f"[apply] ERROR: payload not found: {payload}")
        return 2
    if not dst.exists():
        print(f"[apply] ERROR: destination not found: {dst}")
        return 3

    stamp = _ts()
    for k in range(1, 100):
        backup = old_dir / f"step_e_service_old_{stamp}_{k:02d}.py"
        if not backup.exists():
            break
    shutil.copy2(dst, backup)
    shutil.copy2(payload, dst)
    print(f"[apply] UPDATED: {dst}")
    print(f"[apply] backup -> {backup}")

    # compile check
    cmd = [sys.executable, "-m", "py_compile", str(dst)]
    p = subprocess.run(cmd, cwd=str(repo))
    if p.returncode != 0:
        print("[apply] ERROR: py_compile failed.")
        return p.returncode

    print("[apply] OK: py_compile passed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
