# -*- coding: utf-8 -*-
# fix_step_c_indent.py
#
# Purpose:
#   Auto-fix an IndentationError in ai_core/services/step_c_service.py, especially when
#   `def _build_raw_column_map(...):` is accidentally indented (or de-indented) and import fails.
#
# Usage (Windows/cmd):
#   cd /d C:\Users\ss-to\OneDrive\デスクトップ\Python\soxl_rl_gui
#   conda activate soxl_gui
#   python tools\fix_step_c_indent.py
#
# Behavior:
#   1) Create a backup under ai_core/services/old/
#      e.g. ai_core/services/old/step_c_service_old_YYYYMMDD_HHMMSS.py
#   2) Find the line that matches: def _build_raw_column_map(
#   3) Assume it should be a method inside class StepCService and set its indent to 4 spaces
#      (if class StepCService is not found above, it will set indent to 0).
#   4) Shift the whole function block (until next def/class at same-or-lower indent) by the same delta.
#
# Notes:
#   - This script targets the most common single-point indentation breakage.
#   - If you have multiple indentation issues, you may need additional manual fixes.

from __future__ import annotations

import re
import sys
from pathlib import Path
from datetime import datetime

TARGET_DEFAULT = Path("ai_core") / "services" / "step_c_service.py"

def _backup(src: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    old_dir = src.parent / "old"
    old_dir.mkdir(parents=True, exist_ok=True)
    dst = old_dir / f"step_c_service_old_{ts}.py"
    dst.write_text(src.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
    return dst

def _indent_len(s: str) -> int:
    return len(s) - len(s.lstrip(" "))

def _is_block_boundary(line: str) -> bool:
    stripped = line.lstrip(" ")
    return stripped.startswith("def ") or stripped.startswith("class ")

def fix_file(path: Path) -> int:
    if not path.exists():
        print(f"[fix_step_c_indent] NOT FOUND: {path}")
        return 2

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines(True)

    # Find target def
    pat = re.compile(r"^\\s*def\\s+_build_raw_column_map\\s*\\(")
    idx = None
    for i, ln in enumerate(lines):
        if pat.match(ln):
            idx = i
            break
    if idx is None:
        print("[fix_step_c_indent] target def not found: def _build_raw_column_map(")
        return 1

    # Find nearest preceding class StepCService
    class_idx = None
    class_pat = re.compile(r"^\\s*class\\s+StepCService\\b")
    for j in range(idx, -1, -1):
        if class_pat.match(lines[j]):
            class_idx = j
            break

    desired_indent = 4 if class_idx is not None else 0
    cur_indent = _indent_len(lines[idx])

    if cur_indent == desired_indent:
        print(f"[fix_step_c_indent] already ok (indent={cur_indent})")
        return 0

    shift = desired_indent - cur_indent  # + adds spaces, - removes spaces
    print(f"[fix_step_c_indent] fixing indent: {cur_indent} -> {desired_indent} (shift {shift})")

    # Determine block range: from def line until next def/class at indent <= cur_indent
    start = idx
    end = idx + 1
    base_indent = cur_indent
    n = len(lines)

    while end < n:
        ln = lines[end]
        if ln.strip() == "":
            end += 1
            continue
        ind = _indent_len(ln)
        if ind <= base_indent and _is_block_boundary(ln):
            break
        end += 1

    # Apply shift to def block
    for k in range(start, end):
        ln = lines[k]
        if ln.strip() == "":
            continue
        if shift > 0:
            lines[k] = (" " * shift) + ln
        else:
            remove = min(-shift, _indent_len(ln))
            lines[k] = ln[remove:]

    path.write_text("".join(lines), encoding="utf-8")
    print(f"[fix_step_c_indent] wrote: {path} (block lines {start+1}-{end})")
    return 0

def main() -> int:
    p = Path(sys.argv[1]) if len(sys.argv) >= 2 else TARGET_DEFAULT

    if not p.exists():
        print(f"[fix_step_c_indent] target file missing: {p}")
        return 2

    backup = _backup(p)
    print(f"[fix_step_c_indent] backup: {backup}")

    return fix_file(p)

if __name__ == "__main__":
    raise SystemExit(main())
