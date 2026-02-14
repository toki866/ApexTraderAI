# -*- coding: utf-8 -*-
"""
fix_step_e_import_re_before_pat.py

Purpose:
  Fix StepE import order so that 'import re' is guaranteed to appear BEFORE
  any top-level usage like:
      _SUSPICIOUS_COL_PAT = re.compile(...)

This prevents:
  NameError: name 're' is not defined

Behavior:
  - Edits: ai_core/services/step_e_service.py
  - Creates a backup in: old/step_e_service_old_YYYYMMDD_HHMMSS_XX.py
  - If 'import re' exists but is located AFTER the first top-level re.compile usage,
    it is moved up into the import section.
  - If 'import re' does not exist, it is inserted into the import section.

Usage:
  python tools\\fix_step_e_import_re_before_pat.py
"""
from __future__ import annotations

import datetime
import shutil
from pathlib import Path
import re as _re


def _timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _backup(src: Path, old_dir: Path) -> Path:
    old_dir.mkdir(parents=True, exist_ok=True)
    ts = _timestamp()
    for i in range(1, 100):
        out = old_dir / f"step_e_service_old_{ts}_{i:02d}.py"
        if not out.exists():
            shutil.copy2(src, out)
            return out
    raise RuntimeError("backup rotation exhausted")


def _find_first_use_line(lines: list[str]) -> int | None:
    # Look for common top-level usage patterns.
    for i, line in enumerate(lines):
        s = line.lstrip()
        if s.startswith("#"):
            continue
        if "re.compile" in line:
            return i
        if "_SUSPICIOUS_COL_PAT" in line and "compile" in line:
            return i
    return None


def _find_import_re_line(lines: list[str]) -> int | None:
    pat = _re.compile(r"^\s*import\s+re(\s+as\s+\w+)?\s*$")
    for i, line in enumerate(lines):
        if pat.match(line):
            return i
    return None


def _find_import_insertion_point(lines: list[str]) -> int:
    """Choose a safe insertion point within the top import block."""
    i = 0
    n = len(lines)

    # Skip shebang/encoding
    while i < n and (lines[i].startswith("#!") or "coding" in lines[i]):
        i += 1

    # Skip module docstring if present
    q3_sng = "'" * 3
    q3_dbl = '"' * 3
    stripped = lines[i].lstrip() if i < n else ""
    if stripped.startswith(q3_sng) or stripped.startswith(q3_dbl):
        q = stripped[:3]
        i += 1
        while i < n and q not in lines[i]:
            i += 1
        if i < n:
            i += 1  # include closing quote line

    last_import = None
    import_pat = _re.compile(r"^\s*(import|from)\s+")
    while i < n:
        line = lines[i]
        if import_pat.match(line):
            last_import = i
            i += 1
            continue
        # allow blank lines between import statements early on
        if line.strip() == "" and last_import is not None:
            i += 1
            continue
        break

    if last_import is None:
        return 0
    return last_import + 1


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "ai_core" / "services" / "step_e_service.py"
    if not target.exists():
        print(f"[fix] ERROR: not found: {target}")
        return 2

    raw = target.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()

    first_use = _find_first_use_line(lines)
    if first_use is None:
        print("[fix] WARN: no obvious top-level re.compile usage found; nothing to do.")
        return 0

    import_re = _find_import_re_line(lines)
    need_move = (import_re is None) or (import_re > first_use)

    if not need_move:
        print(f"[fix] OK: 'import re' is already before first use (import_re_line={import_re+1}, first_use_line={first_use+1}).")
        return 0

    backup_path = _backup(target, repo_root / "old")
    print(f"[fix] backup -> {backup_path}")

    # Remove any existing standalone 'import re' lines (we'll re-insert in the right place)
    new_lines: list[str] = []
    removed = 0
    pat = _re.compile(r"^\s*import\s+re(\s+as\s+\w+)?\s*$")
    for line in lines:
        if pat.match(line):
            removed += 1
            continue
        new_lines.append(line)

    ins_at = _find_import_insertion_point(new_lines)
    new_lines.insert(ins_at, "import re")
    target.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    print(f"[fix] UPDATED: inserted 'import re' before first use (first_use_line={first_use+1}).")
    if removed:
        print(f"[fix] NOTE: removed {removed} existing 'import re' line(s) and re-inserted at import block.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
