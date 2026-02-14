"""
Fix StepE: prevent StepD' label/label_available columns from being picked into RL obs (especially profile D).

- Inserts a "bad_keys" leak guard and excludes columns containing:
  label, available, target, y_
  when building the numeric column set for profile D.

This script:
- backs up the original file to ./old/
- edits ai_core/services/step_e_service.py in place
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


def _backup(src: Path, old_dir: Path) -> Path:
    old_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = old_dir / f"{src.stem}_old_{ts}{src.suffix}"
    dst.write_bytes(src.read_bytes())
    return dst


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "ai_core" / "services" / "step_e_service.py"
    if not target.exists():
        raise FileNotFoundError(f"Missing: {target}")

    txt = target.read_text(encoding="utf-8", errors="ignore")

    if "bad_keys = (\"label\", \"available\", \"target\", \"y_\")" in txt:
        print("[fix] already applied")
        return

    # Locate the profile D numeric column list comprehension
    # Example original line:
    #   all_num = [c for c in df_all.columns if c != "Date" and pd.api.types.is_numeric_dtype(df_all[c])]
    pat = re.compile(
        r'^(?P<indent>\s*)all_num\s*=\s*\[c\s+for\s+c\s+in\s+df_all\.columns\s+if\s+c\s*!=\s*"Date"\s+and\s+pd\.api\.types\.is_numeric_dtype\(df_all\[c\]\)\]\s*$',
        re.MULTILINE,
    )
    m = pat.search(txt)
    if not m:
        # slightly more permissive fallback
        pat2 = re.compile(r'^(?P<indent>\s*)all_num\s*=\s*\[c\s+for\s+c\s+in\s+df_all\.columns.*is_numeric_dtype\(df_all\[c\]\).*\]\s*$',
                          re.MULTILINE)
        m = pat2.search(txt)
    if not m:
        raise RuntimeError("Could not find profile D all_num line in step_e_service.py (file layout changed)")

    indent = m.group("indent")
    bad_line = indent + 'bad_keys = ("label", "available", "target", "y_")  # leak guard'
    new_all_num = indent + 'all_num = [c for c in df_all.columns if c not in {"Date"} and pd.api.types.is_numeric_dtype(df_all[c]) and not any(k in str(c).lower() for k in bad_keys)]'

    # Insert bad_keys line above the matched all_num line
    start = m.start()
    # find line start
    line_start = txt.rfind("\n", 0, start) + 1
    replacement = bad_line + "\n" + new_all_num
    # replace the whole matched line
    txt2 = txt[:line_start] + replacement + txt[m.end():]

    backup = _backup(target, repo_root / "old")
    target.write_text(txt2, encoding="utf-8")
    print(f"[fix] UPDATED: {target}")
    print(f"[fix] backup -> {backup}")


if __name__ == "__main__":
    main()
