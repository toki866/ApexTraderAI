# -*- coding: utf-8 -*-
"""
Fix StepE NameError: df_all is not defined inside _ensure_core_features().

This script:
- edits ai_core/services/step_e_service.py in-place
- makes a backup under ./old/ with timestamped filename
- inserts `df_all = df` at the top of _ensure_core_features() if missing

Usage:
  python tools/fix_step_e_df_all_alias.py
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import List, Tuple


def _repo_root_from_here() -> Path:
    # tools/ -> repo root
    here = Path(__file__).resolve()
    return here.parent.parent


def _next_backup_path(old_dir: Path, stem: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i in range(1, 100):
        p = old_dir / f"{stem}_old_{ts}_{i:02d}.py"
        if not p.exists():
            return p
    raise RuntimeError("Could not allocate backup name (too many collisions).")


def _find_func_block(lines: List[str], func_name: str) -> Tuple[int, int]:
    """
    Return (start_idx, end_idx) of a method block (0-based, end exclusive).
    Finds `def func_name(` with any indentation.
    End is next line that starts with the same indentation + 'def ' (method) or EOF.
    """
    import re
    pat = re.compile(rf"^(\s*)def\s+{re.escape(func_name)}\s*\(")
    start = None
    indent = ""
    for i, line in enumerate(lines):
        m = pat.match(line)
        if m:
            start = i
            indent = m.group(1)
            break
    if start is None:
        raise SystemExit(f"[fix] ERROR: function '{func_name}' not found.")
    # find end: next def at same indent
    end = len(lines)
    pat_next = re.compile(rf"^{re.escape(indent)}def\s+")
    for j in range(start + 1, len(lines)):
        if pat_next.match(lines[j]):
            end = j
            break
    return start, end


def _has_df_all_alias(block_lines: List[str]) -> bool:
    for l in block_lines[:50]:
        s = l.strip().replace(" ", "")
        if s.startswith("df_all=df"):
            return True
    return False


def main() -> int:
    root = _repo_root_from_here()
    target = root / "ai_core" / "services" / "step_e_service.py"
    if not target.exists():
        raise SystemExit(f"[fix] ERROR: not found: {target}")

    text = target.read_text(encoding="utf-8")
    lines = text.splitlines(True)  # keep EOL

    start, end = _find_func_block(lines, "_ensure_core_features")
    block = [l.rstrip("\n") for l in lines[start:end]]

    if _has_df_all_alias(block):
        print(f"[fix] OK: df_all alias already present. No changes made.")
        return 0

    # Insert after def line and optional docstring
    k = start + 1

    # skip blank lines
    while k < end and lines[k].strip() == "":
        k += 1

    triple_dq = '"' * 3
    triple_sq = "'" * 3

    # docstring right after def
    if k < end and (lines[k].lstrip().startswith(triple_dq) or lines[k].lstrip().startswith(triple_sq)):
        quote = triple_dq if lines[k].lstrip().startswith(triple_dq) else triple_sq
        k += 1
        while k < end and quote not in lines[k]:
            k += 1
        if k < end:
            k += 1  # include closing docstring line

    # skip blank lines after docstring
    while k < end and lines[k].strip() == "":
        k += 1

    def_indent = lines[start][:len(lines[start]) - len(lines[start].lstrip(" "))]
    inner_indent = def_indent + "    "
    if k < end:
        # use indentation of first statement if available
        inner_indent = lines[k][:len(lines[k]) - len(lines[k].lstrip(" "))] or inner_indent

    insert_line = f"{inner_indent}df_all = df  # alias to avoid NameError (df_all referenced)\n"

    new_lines = lines[:k] + [insert_line] + lines[k:]

    # backup
    old_dir = root / "old"
    old_dir.mkdir(parents=True, exist_ok=True)
    backup = _next_backup_path(old_dir, "step_e_service")
    backup.write_text(text, encoding="utf-8")

    target.write_text("".join(new_lines), encoding="utf-8")

    print(f"[fix] UPDATED: inserted 'df_all = df' into {target}")
    print(f"[fix] backup -> {backup}")
    print(f"[fix] inserted_at_line={k+1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
