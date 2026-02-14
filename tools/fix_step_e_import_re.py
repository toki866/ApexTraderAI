# tools/fix_step_e_import_re.py
# -*- coding: utf-8 -*-
"""
Fix NameError: name 're' is not defined in ai_core/services/step_e_service.py

Why this happens:
- Some versions of step_e_service.py refer to `re.compile(...)` at module import time
  before `import re` is executed (or `import re` is missing entirely).

What this fixer does:
- Parses the target module and checks whether `re` is imported *before* the first
  module-level use of `re`.
- If not, it inserts `import re` near the top (after module docstring and any
  `from __future__ import ...` lines).
- Creates a timestamped backup under ./old/ before editing.

Usage:
  python tools/fix_step_e_import_re.py
"""

from __future__ import annotations

import ast
import datetime as _dt
from pathlib import Path
from typing import Optional, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_text(path: Path) -> str:
    # Prefer UTF-8 with BOM handling; fall back to cp932 for JP Windows if needed.
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    # last resort
    return path.read_text(errors="replace")


def _write_text(path: Path, text: str) -> None:
    # Keep UTF-8-SIG to be friendly for Excel/JP Windows tools, consistent with repo style.
    path.write_text(text, encoding="utf-8-sig", newline="\n")


def _is_import_re(node: ast.AST) -> bool:
    if isinstance(node, ast.Import):
        return any(a.name == "re" for a in node.names)
    if isinstance(node, ast.ImportFrom):
        # from re import ...
        return node.module == "re"
    return False


class _ReUseVisitor(ast.NodeVisitor):
    """
    Detect `re` usage in a node, but DO NOT descend into function/class bodies.
    """

    def __init__(self) -> None:
        self.found = False

    def visit_Name(self, node: ast.Name) -> None:
        if node.id == "re":
            self.found = True

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return


def _uses_re_at_module_eval(node: ast.AST) -> bool:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return False
    v = _ReUseVisitor()
    v.visit(node)
    return v.found


def _find_docstring_and_future_insert_line(mod: ast.Module) -> int:
    """
    Returns 1-based line number after which we can safely insert imports:
    - after module docstring (if exists)
    - after any `from __future__ import ...` lines (which must be first)
    If nothing, returns 0 (meaning insert at top before line 1).
    """
    insert_after = 0
    body = list(mod.body)

    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
        if isinstance(body[0].value.value, str):
            insert_after = int(getattr(body[0], "end_lineno", None) or body[0].lineno or 0)

    idx = 1 if insert_after > 0 else 0
    while idx < len(body):
        n = body[idx]
        if isinstance(n, ast.ImportFrom) and n.module == "__future__":
            insert_after = int(getattr(n, "end_lineno", None) or n.lineno or insert_after)
            idx += 1
            continue
        break

    return insert_after


def _analyze_order(mod: ast.Module) -> Tuple[Optional[int], Optional[int]]:
    """
    Returns (first_re_use_line, first_import_re_line) considering only module-level executed statements.
    """
    first_use: Optional[int] = None
    first_import: Optional[int] = None

    for node in mod.body:
        if _is_import_re(node):
            if first_import is None:
                first_import = int(getattr(node, "lineno", None) or 0) or None
            continue

        if first_import is None and first_use is None and _uses_re_at_module_eval(node):
            first_use = int(getattr(node, "lineno", None) or 0) or None

        if first_import is not None and first_use is not None:
            break

    return first_use, first_import


def _backup(src: Path) -> Path:
    root = _repo_root()
    old_dir = root / "old"
    old_dir.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = old_dir / f"step_e_service_old_{ts}_01.py"
    dst.write_bytes(src.read_bytes())
    return dst


def main() -> int:
    root = _repo_root()
    target = root / "ai_core" / "services" / "step_e_service.py"
    if not target.exists():
        print(f"[fix] ERROR: target not found: {target}")
        return 2

    text = _read_text(target)

    try:
        mod = ast.parse(text)
    except SyntaxError as e:
        print(f"[fix] ERROR: cannot parse {target}: {e}")
        print("[fix] No changes made.")
        return 3

    first_use, first_import = _analyze_order(mod)

    if first_use is None:
        print(f"[fix] OK: no module-level `re` usage detected. No changes made. (import_re_line={first_import})")
        return 0

    if first_import is not None and first_import < first_use:
        print(f"[fix] OK: `re` is imported before first use. (first_import={first_import}, first_use={first_use})")
        return 0

    insert_after = _find_docstring_and_future_insert_line(mod)  # 1-based
    lines = text.splitlines()

    insert_idx = insert_after  # 0-based insertion index (after N lines)
    ins_line = "import re"

    # Avoid inserting duplicates at insertion point
    if insert_idx > 0 and lines[insert_idx - 1].strip() == ins_line:
        print(f"[fix] OK: already has '{ins_line}' at insertion point. No changes made.")
        return 0
    if insert_idx < len(lines) and lines[insert_idx].strip() == ins_line:
        print(f"[fix] OK: already has '{ins_line}' at insertion point. No changes made.")
        return 0

    backup_path = _backup(target)

    lines.insert(insert_idx, ins_line)
    new_text = "\n".join(lines) + "\n"
    _write_text(target, new_text)

    print(f"[fix] UPDATED: inserted '{ins_line}' into {target}")
    print(f"[fix] backup -> {backup_path}")
    print(f"[fix] first_use_line={first_use}, first_import_re_line={first_import}, inserted_after_line={insert_after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
