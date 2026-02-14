#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fix StepE: _ensure_core_features() returning a list / referencing df_all (NameError),
which can turn a DataFrame into a list and later crash with:
  TypeError: list indices must be integers or slices, not str

This script replaces the whole _ensure_core_features method with a robust
DataFrame-returning implementation, and keeps a timestamped backup under ./old/.
"""

from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(8):
        if (cur / "ai_core").exists() and (cur / "tools").exists():
            return cur
        cur = cur.parent
    # fallback: assume tools/ under repo root
    return start.resolve().parent


def _backup(src: Path, repo_root: Path) -> Path:
    old_dir = repo_root / "old"
    old_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = old_dir / f"step_e_service_old_{ts}_01.py"
    shutil.copy2(src, dst)
    return dst


def _build_new_method(indent: str) -> str:
    # Build with stable, minimal dependencies.
    # Uses globals().get("_SUSPICIOUS_COL_PAT") if present to drop leak-like columns.
    base = r