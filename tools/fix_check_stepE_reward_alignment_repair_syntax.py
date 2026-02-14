#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Repair tools/check_stepE_reward_alignment.py if it was broken by escaped triple quotes (\"\"\").
Also ensures candidate ranking prefers corr then rmse then mae then n.
Creates a backup under ./old/ before overwriting.
"""
from __future__ import annotations
from pathlib import Path
import shutil
from datetime import datetime
import re

def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "tools" / "check_stepE_reward_alignment.py"
    if not target.exists():
        print(f"[fix] Not found: {target}")
        return 2

    txt = target.read_text(encoding="utf-8")

    # Backup
    old_dir = repo_root / "old"
    old_dir.mkdir(parents=True, exist_ok=True)
    backup = old_dir / f"check_stepE_reward_alignment_old_{_timestamp()}.py"
    shutil.copy2(target, backup)

    changed = False

    # 1) Fix escaped docstring triple quotes: \"\"\" -> """
    if '\\"\\\"\\\"\\\"' in txt or '\\"\\"\\"' in txt:
        txt2 = txt.replace('\\"\\"\\"', '"""')
        if txt2 != txt:
            txt = txt2
            changed = True

    # 2) Ensure sort key prefers corr (desc), then rmse (asc), then mae (asc), then n (desc)
    # Replace the body of _sort_key if present but different.
    pattern = r"def _sort_key\\(x: CandidateResult\\)\\s*->\\s*Tuple\\[float, float, float, float\\]:\\s*\\n(\\s+.+?\\n)\\s*results\\.sort\\(key=_sort_key\\)"
    m = re.search(pattern, txt, flags=re.S)
    if m:
        desired = (
            "def _sort_key(x: CandidateResult) -> Tuple[float, float, float, float]:\n"
            "        corr = x.corr if not pd.isna(x.corr) else -1.0\n"
            "        rmse = x.rmse if not pd.isna(x.rmse) else 1e9\n"
            "        mae = x.mae if not pd.isna(x.mae) else 1e9\n"
            "        return (-corr, rmse, mae, -float(x.n))\n\n"
            "    results.sort(key=_sort_key)"
        )
        current_block = txt[m.start():m.end()]
        if desired not in current_block:
            # replace just the function+sort line block
            txt = txt[:m.start()] + re.sub(pattern, desired, txt[m.start():m.end()], flags=re.S) + txt[m.end():]
            changed = True

    if changed:
        target.write_text(txt, encoding="utf-8")
        print(f"[fix] UPDATED: {target}")
    else:
        print(f"[fix] No changes needed: {target}")
    print(f"[fix] backup -> {backup}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
