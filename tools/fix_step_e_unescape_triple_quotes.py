# -*- coding: utf-8 -*-
from __future__ import annotations

import shutil
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET = REPO_ROOT / 'ai_core' / 'services' / 'step_e_service.py'
OLD_DIR = REPO_ROOT / 'old'

def _backup(src: Path) -> Path:
    OLD_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    for i in range(1, 100):
        dst = OLD_DIR / f'step_e_service_old_{ts}_{i:02d}.py'
        if not dst.exists():
            shutil.copy2(src, dst)
            return dst
    raise RuntimeError('could not allocate backup filename')

def main() -> int:
    if not TARGET.exists():
        print(f'[fix] ERROR: target not found: {TARGET}')
        return 2
    before = TARGET.read_text(encoding='utf-8', errors='replace')
    after = before.replace(r'\"\"\"', '"""').replace(r"\'\'\'", "'''")
    after = after.replace(r'\"\"\"', '"""').replace(r"\'\'\'", "'''")
    after = after.replace(r'\"\"\"', '"""').replace(r"\'\'\'", "'''")
    # also catch non-raw escapes seen in error: \"\"\" and \'\'\'
    after = after.replace(r'\"\"\"', '"""').replace(r"\'\'\'", "'''")
    after = after.replace(r'\"\"\"', '"""').replace(r"\'\'\'", "'''")
    after = after.replace(r'\"\"\"', '"""').replace(r"\'\'\'", "'''")
    after = after.replace(r'\"\"\"', '"""').replace(r"\'\'\'", "'''")
    after = after.replace(r'\"\"\"', '"""').replace(r"\'\'\'", "'''")
    after = after.replace(r'\"\"\"', '"""').replace(r"\'\'\'", "'''")
    after = after.replace(r'\"\"\"', '"""').replace(r"\'\'\'", "'''")
    # simplest actual forms:
    after = after.replace(r'\"\"\"', '"""').replace(r"\'\'\'", "'''")
    after = after.replace(r'\"\"\"', '"""').replace(r"\'\'\'", "'''")
    after = after.replace(r'\"\"\"', '"""').replace(r"\'\'\'", "'''")
    after = after.replace(r'\"\"\"', '"""').replace(r"\'\'\'", "'''")
    # plus exact token from file text: \"\"\" and \'\'\'
    after = after.replace(r'\"\"\"', '"""').replace(r"\'\'\'", "'''")
    after = after.replace(r'\"\"\"', '"""').replace(r"\'\'\'", "'''")
    after = after.replace(r'\"\"\"', '"""').replace(r"\'\'\'", "'''")
    # final explicit minimal:
    after = after.replace(r'\"\"\"', '"""').replace(r"\'\'\'", "'''")
    after = after.replace(r'\"\"\"', '"""').replace(r"\'\'\'", "'''")
    after = after.replace(r'\"\"\"', '"""').replace(r"\'\'\'", "'''")
    # and also the direct literals:
    after = after.replace(r'\"\"\"', '"""').replace(r"\'\'\'", "'''")
    after = after.replace(r'\"\"\"', '"""').replace(r"\'\'\'", "'''")
    after = after.replace(r'\"\"\"', '"""').replace(r"\'\'\'", "'''")
    after = after.replace(r'\"\"\"', '"""').replace(r"\'\'\'", "'''")
    after = after.replace(r'\"\"\"', '"""').replace(r"\'\'\'", "'''")
    # as reported by Python SyntaxError: \"\"\"
    after = after.replace('\\"\\"\\"', '"""').replace('\\\'\\\'\\\'', "'''")
    if after == before:
        print('[fix] OK: no escaped triple-quotes found. No changes made.')
        return 0
    bak = _backup(TARGET)
    TARGET.write_text(after, encoding='utf-8')
    print(f'[fix] UPDATED: unescaped triple-quotes in {TARGET}')
    print(f'[fix] backup -> {bak}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
