# -*- coding: utf-8 -*-
from __future__ import annotations

import re
import shutil
import py_compile
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

def _compile_ok(text: str) -> bool:
    try:
        compile(text, str(TARGET), 'exec')
        return True
    except SyntaxError:
        return False

def _latest_good_backup() -> Path | None:
    if not OLD_DIR.exists():
        return None
    cands = sorted(OLD_DIR.glob('step_e_service_old_*.py'), key=lambda p: p.stat().st_mtime, reverse=True)
    for fp in cands:
        try:
            t = fp.read_text(encoding='utf-8', errors='replace')
        except Exception:
            continue
        if _compile_ok(t):
            return fp
    return None

def _unescape_triple_quotes(text: str) -> str:
    text2 = text.replace(r'\"\"\"', '"""').replace(r"\'\'\'", "'''")
    text2 = text2.replace('\\"\\"\\"', '"""').replace('\\\'\\\'\\\'', "'''")
    return text2

def _ensure_import_re(text: str) -> str:
    if re.search(r'^\s*import\s+re\s*$', text, flags=re.M) or re.search(r'^\s*from\s+re\s+import\b', text, flags=re.M):
        return text
    lines = text.splitlines()
    insert_at = None
    for i, line in enumerate(lines[:160]):
        if line.startswith('import ') or line.startswith('from '):
            insert_at = i + 1
    if insert_at is None:
        insert_at = 0
    lines.insert(insert_at, 'import re')
    return '\n'.join(lines) + ('\n' if text.endswith('\n') else '')

def _patch_suspicious_pat(text: str) -> str:
    pat = re.compile(r'^\s*_SUSPICIOUS_COL_PAT\s*=\s*re\.compile\(r\".*?\",\s*re\.IGNORECASE\)\s*$', re.M)
    repl = '_SUSPICIOUS_COL_PAT = re.compile(r"(realclose|close_true|future|label_available|(^|[^a-zA-Z0-9])label($|[^a-zA-Z0-9])|(^|[^a-zA-Z0-9])target($|[^a-zA-Z0-9])|leak)", re.IGNORECASE)'
    if pat.search(text):
        return pat.sub(repl, text, count=1)
    if '_SUSPICIOUS_COL_PAT' not in text:
        lines = text.splitlines()
        insert_at = 0
        for i, line in enumerate(lines[:220]):
            if line.startswith('import ') or line.startswith('from '):
                insert_at = i + 1
        lines.insert(insert_at, repl)
        return '\n'.join(lines) + ('\n' if text.endswith('\n') else '')
    return text

def _replace_match_r_typo(text: str) -> str:
    return text.replace('match_r', 'match_ratio')

def _replace_build_obs_and_returns(text: str) -> str:
    lines = text.splitlines()
    idx = None
    indent = ''
    for i, line in enumerate(lines):
        if 'def _build_obs_and_returns' in line:
            idx = i
            indent = line[:len(line) - len(line.lstrip(' '))]
            break
    if idx is None:
        return text
    j = idx + 1
    while j < len(lines):
        if lines[j].startswith(indent + 'def ') and j > idx:
            break
        j += 1
    new_block = [
        indent + 'def _build_obs_and_returns(self, df: \"pd.DataFrame\", obs_cols: \"list[str]\"):',
        indent + '    ' + '"""Build X and forward returns yret (no-leak).',
        indent + '    Decision time: end-of-day (t_eff ~ Close).',
        indent + '    Reward uses Close(t+1)/Close(t)-1.',
        indent + '    Returns: (X, yret, dates)',
        indent + '    """',
        indent + '    df2 = df.copy()',
        indent + "    if 'Date' in df2.columns:",
        indent + "        df2['Date'] = pd.to_datetime(df2['Date'], errors='coerce').dt.normalize()",
        indent + "        dates = df2['Date'].to_numpy()",
        indent + '    else:',
        indent + '        dates = np.arange(len(df2), dtype=np.int64)',
        indent + '    X = df2[obs_cols].to_numpy(dtype=np.float32, copy=True)',
        indent + "    if 'Close' not in df2.columns:",
        indent + "        raise KeyError('Close column missing for yret computation')",
        indent + "    close = df2['Close'].astype(float).to_numpy()",
        indent + '    yret = np.zeros(len(close), dtype=np.float32)',
        indent + '    if len(close) >= 2:',
        indent + '        denom = np.where(np.abs(close[:-1]) < 1e-12, np.nan, close[:-1])',
        indent + '        yret[:-1] = (close[1:] / denom - 1.0).astype(np.float32)',
        indent + '        yret[-1] = 0.0',
        indent + '    yret = np.nan_to_num(yret, nan=0.0, posinf=0.0, neginf=0.0)',
        indent + '    return X, yret, dates',
    ]
    out_lines = lines[:idx] + new_block + lines[j:]
    return '\n'.join(out_lines) + ('\n' if text.endswith('\n') else '')

def _patch_core_features_ret_fwd1(text: str) -> str:
    if 'ret_fwd1' in text:
        return text
    lines = text.splitlines()
    start = None
    indent = ''
    for i, line in enumerate(lines):
        if 'def _ensure_core_features' in line:
            start = i
            indent = line[:len(line) - len(line.lstrip(' '))]
            break
    if start is None:
        return text
    ins = None
    for i in range(start, min(start + 700, len(lines))):
        if 'oc_ret' in lines[i] and ('out[' in lines[i] or 'out.' in lines[i]):
            ins = i + 1
            break
    if ins is None:
        for i in range(start, min(start + 800, len(lines))):
            if lines[i].startswith(indent + '    return '):
                ins = i
                break
    if ins is None:
        return text
    add = [
        indent + '        # Forward 1-day return (Close->next Close). Used as reward target (no-leak).',
        indent + "        if 'ret_fwd1' not in out.columns and 'Close' in out.columns:",
        indent + "            c = out['Close'].astype(float)",
        indent + "            denom = c.replace(0, float('nan'))",
        indent + "            out['ret_fwd1'] = (c.shift(-1) / denom - 1.0).astype(float)",
    ]
    out_lines = lines[:ins] + add + lines[ins:]
    return '\n'.join(out_lines) + ('\n' if text.endswith('\n') else '')

def main() -> int:
    if not TARGET.exists():
        print(f'[fix] ERROR: target not found: {TARGET}')
        return 2
    text = TARGET.read_text(encoding='utf-8', errors='replace')
    if not _compile_ok(text):
        text_u = _unescape_triple_quotes(text)
        if _compile_ok(text_u):
            bak = _backup(TARGET)
            TARGET.write_text(text_u, encoding='utf-8')
            print(f'[fix] UPDATED: unescaped triple-quotes in {TARGET}')
            print(f'[fix] backup -> {bak}')
            text = text_u
        else:
            good = _latest_good_backup()
            if good is None:
                print('[fix] ERROR: step_e_service.py has SyntaxError and no good backup found in old/')
                return 3
            bak = _backup(TARGET)
            shutil.copy2(good, TARGET)
            print(f'[fix] RESTORED: {TARGET} <- {good}')
            print(f'[fix] backup -> {bak}')
            text = TARGET.read_text(encoding='utf-8', errors='replace')
    text2 = text
    text2 = _unescape_triple_quotes(text2)
    text2 = _ensure_import_re(text2)
    text2 = _patch_suspicious_pat(text2)
    text2 = _replace_match_r_typo(text2)
    text2 = _patch_core_features_ret_fwd1(text2)
    text2 = _replace_build_obs_and_returns(text2)
    if text2 != text:
        bak2 = _backup(TARGET)
        TARGET.write_text(text2, encoding='utf-8')
        print(f'[fix] UPDATED: applied restore+no-leak patches to {TARGET}')
        print(f'[fix] backup -> {bak2}')
    else:
        print('[fix] OK: nothing to change (already patched).')
    try:
        py_compile.compile(str(TARGET), doraise=True)
        print('[fix] OK: py_compile passed.')
    except Exception as e:
        print('[fix] ERROR: py_compile failed:', e)
        return 4
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
