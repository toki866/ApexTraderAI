# -*- coding: utf-8 -*-
"""
Apply the patched StepE service (step_e_service.py) safely.

Usage:
  python tools\apply_step_e_service_fixed.py

- Backs up current ai_core/services/step_e_service.py to ./old/
- Overwrites it with the patched version included in this ZIP.
"""
from __future__ import annotations

from pathlib import Path
import datetime
import shutil


def _timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    repo = _repo_root()
    src = repo / "ai_core" / "services" / "step_e_service.py"
    patched = repo / "ai_core" / "services" / "step_e_service.py.patched_in_zip"

    if not patched.exists():
        raise SystemExit(f"[apply] ERROR: patched file not found: {patched}")
    if not src.exists():
        raise SystemExit(f"[apply] ERROR: target not found: {src}")

    old_dir = repo / "old"
    old_dir.mkdir(parents=True, exist_ok=True)
    backup = old_dir / f"step_e_service_old_{_timestamp()}_01.py"
    shutil.copy2(src, backup)

    shutil.copy2(patched, src)
    print(f"[apply] UPDATED: {src}")
    print(f"[apply] backup -> {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
