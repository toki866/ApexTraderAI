# -*- coding: utf-8 -*-
"""
Fix StepE service import error: ensure `import re` exists before `_SUSPICIOUS_COL_PAT = re.compile(...)`.

Usage:
  python tools\fix_step_e_add_import_re.py

- Creates a backup under ./old/ before modifying.
- Idempotent: if `import re` already exists, no changes.
"""
from __future__ import annotations

from pathlib import Path
import datetime
import shutil
import ast


def _timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _backup(src: Path, old_dir: Path) -> Path:
    old_dir.mkdir(parents=True, exist_ok=True)
    out = old_dir / f"step_e_service_old_{_timestamp()}_01.py"
    shutil.copy2(src, out)
    return out


def _has_import_re(lines: list[str]) -> bool:
    for ln in lines:
        s = ln.strip()
        if s == "import re" or s.startswith("import re,") or s.endswith(", re") or s.endswith(",re"):
            return True
        if s.startswith("from re import"):
            return True
    return False


def _find_insert_lineno(text: str) -> int:
    """
    Return 1-based line number after which we insert `import re`.
    We insert after the initial consecutive import block (right after module docstring if present).
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        # If the file itself is broken, do the safest: insert near the top after the first non-empty line.
        lines = text.splitlines(True)
        for i, ln in enumerate(lines, start=1):
            if ln.strip():
                return i
        return 1

    body = list(tree.body)
    if not body:
        return 1

    idx = 0
    # skip module docstring
    if isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
        idx = 1

    last_import_end = 0
    for node in body[idx:]:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            end = getattr(node, "end_lineno", None) or getattr(node, "lineno", 1)
            if end > last_import_end:
                last_import_end = end
        else:
            break

    if last_import_end <= 0:
        # no import block detected; insert after docstring if exists else at line 1
        if idx == 1:
            end = getattr(body[0], "end_lineno", None) or getattr(body[0], "lineno", 1)
            return end
        return 1

    return last_import_end


def main() -> int:
    repo = _repo_root()
    target = repo / "ai_core" / "services" / "step_e_service.py"
    if not target.exists():
        raise SystemExit(f"[fix] ERROR: not found: {target}")

    text = target.read_text(encoding="utf-8")
    lines = text.splitlines(True)

    if _has_import_re(lines):
        print("[fix] OK: 'import re' already present. No changes made.")
        return 0

    old_dir = repo / "old"
    b = _backup(target, old_dir)

    insert_after = _find_insert_lineno(text)  # 1-based
    insert_idx = max(0, min(len(lines), insert_after))  # index in list after that line

    lines.insert(insert_idx, "import re\n")

    target.write_text("".join(lines), encoding="utf-8")
    print(f"[fix] UPDATED: inserted 'import re' into {target}")
    print(f"[fix] backup -> {b}")
    print(f"[fix] inserted_after_line={insert_after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
