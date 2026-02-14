#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Apply patch files (StepE/StepF) with automatic backups into ./old/

Usage:
  python tools/apply_patch_stepE_stepF_v4c.py
"""
from __future__ import annotations

import shutil
from pathlib import Path
from datetime import datetime

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _backup(src: Path, old_dir: Path) -> Path:
    old_dir.mkdir(parents=True, exist_ok=True)
    dst = old_dir / f"{src.stem}_old_{_timestamp()}{src.suffix}"
    shutil.copy2(src, dst)
    return dst

def main() -> int:
    repo = _repo_root()
    patch_root = repo / "patch_files"
    old_dir = repo / "old"

    targets = [
        (patch_root / "ai_core" / "services" / "step_e_service.py", repo / "ai_core" / "services" / "step_e_service.py"),
        (patch_root / "ai_core" / "services" / "step_f_service.py", repo / "ai_core" / "services" / "step_f_service.py"),
    ]

    for src, dst in targets:
        if not src.exists():
            raise FileNotFoundError(f"Patch file not found: {src}")
        if not dst.exists():
            raise FileNotFoundError(f"Target file not found: {dst}")
        b = _backup(dst, old_dir)
        print(f"[apply] backup -> {b}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"[apply] UPDATED: {dst}")

    print("[apply] DONE")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
