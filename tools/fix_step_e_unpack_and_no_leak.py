# -*- coding: utf-8 -*-
"""
Fix StepE issues:
  1) _build_obs_and_returns() must return (X, yret, dates)
  2) suspicious-column filter should not drop Pred_* columns
  3) add forward return ret_fwd1 for no-leak close->next-close reward (kept OUT of obs)
  4) minor typo fix if 'match_r' exists

Usage:
  python tools\fix_step_e_unpack_and_no_leak.py
"""
from __future__ import annotations

import re
from pathlib import Path
from datetime import datetime

def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _backup(src: Path, old_dir: Path) -> Path:
    old_dir.mkdir(parents=True, exist_ok=True)
    dst = old_dir / f"{src.stem}_old_{_timestamp()}_01{src.suffix}"
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return dst

def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "ai_core" / "services" / "step_e_service.py"
    if not target.exists():
        print(f"[fix] ERROR: not found: {target}")
        return 2

    original = target.read_text(encoding="utf-8")
    text = original

    # 0) Ensure "import re" exists at module top-level.
    if not re.search(r"^\s*import\s+re\s*$", text, re.M):
        lines = text.splitlines()
        insert_at = 0
        while insert_at < len(lines) and lines[insert_at].startswith("#"):
            insert_at += 1

        # skip module docstring if present (triple quotes)
        if insert_at < len(lines):
            s0 = lines[insert_at].lstrip()
            if s0.startswith('"""') or s0.startswith("'"*3):
                q = s0[:3]
                insert_at += 1
                while insert_at < len(lines) and q not in lines[insert_at]:
                    insert_at += 1
                if insert_at < len(lines):
                    insert_at += 1

        while insert_at < len(lines) and lines[insert_at].strip() == "":
            insert_at += 1
        lines.insert(insert_at, "import re")
        text = "\n".join(lines) + ("\n" if not text.endswith("\n") else "")
        print(f"[fix] inserted: import re (line~{insert_at+1})")

    # 1) Update suspicious regex to avoid dropping Pred_* columns (Pred_y_from_anchor etc).
    text2 = re.sub(
        r"_SUSPICIOUS_COL_PAT\s*=\s*re\.compile\([^\n]*\)\s*",
        r'_SUSPICIOUS_COL_PAT = re.compile(r"(realclose|close_true|label|leak)", re.IGNORECASE)\n',
        text,
        count=1,
        flags=re.M,
    )
    if text2 != text:
        text = text2
        print("[fix] UPDATED: _SUSPICIOUS_COL_PAT (avoid dropping Pred_* columns)")

    # 2) Replace _build_obs_and_returns() with a stable 3-return implementation.
    new_func = r"""
    def _build_obs_and_returns(self, df: pd.DataFrame, obs_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        \"\"\"Build observation matrix X and forward returns yret.

        IMPORTANT (no-leak):
          - If you decide at t_eff (close-10min) and hold overnight,
            the reward must be based on Close[t] -> Close[t+1] return.
          - We store it as ret_fwd1 and keep it OUT of obs columns.

        Fallback:
          - If ret_fwd1 is not present, uses oc_ret (Open->Close) for legacy behavior.
        \"\"\"
        if "Date" not in df.columns:
            raise ValueError("missing Date column")
        dates = pd.to_datetime(df["Date"], errors="coerce").dt.normalize().to_numpy(dtype="datetime64[ns]")

        ret_col = "ret_fwd1" if "ret_fwd1" in df.columns else "oc_ret"
        if ret_col not in df.columns:
            raise ValueError(f"missing return column: {ret_col}")

        X = df[obs_cols].astype(float).to_numpy(copy=True)
        yret = df[ret_col].astype(float).to_numpy(copy=True)

        if np.isnan(yret).any():
            yret = np.nan_to_num(yret, nan=0.0, posinf=0.0, neginf=0.0)

        return X, yret, dates
"""
    rx = re.compile(r"\n\s{4}def\s+_build_obs_and_returns\b[\s\S]*?(?=\n\s{4}def\s+|\Z)", re.M)
    if rx.search(text):
        text = rx.sub("\n" + new_func + "\n", text, count=1)
        print("[fix] UPDATED: _build_obs_and_returns() -> returns (X, yret, dates)")
    else:
        print("[fix] WARN: _build_obs_and_returns() block not found; skipped replacement.")

    # 3) Insert ret_fwd1 creation into _ensure_core_features (if missing).
    if "ret_fwd1" not in text:
        rx_oc = re.compile(r'(df\["oc_ret"\]\s*=\s*\(df\["Close"\]\s*/\s*df\["Open"\]\s*-\s*1\.0\)\s*)', re.M)
        m2 = rx_oc.search(text)
        if m2:
            ins = m2.group(1) + '\n        # Forward return for no-leak close->next-close reward (kept OUT of obs)\n        df["ret_fwd1"] = (df["Close"].shift(-1) / df["Close"] - 1.0)\n        df["ret_fwd1"] = df["ret_fwd1"].replace([np.inf, -np.inf], np.nan).fillna(0.0)\n'
            text = rx_oc.sub(ins, text, count=1)
            print("[fix] INSERTED: ret_fwd1 in _ensure_core_features()")
        else:
            print("[fix] WARN: could not auto-insert ret_fwd1 (oc_ret pattern not found).")

    # 4) Ensure _select_obs_columns excludes ret_fwd1 from candidates by adding to mandatory list.
    if "mandatory" in text and "ret_fwd1" not in text:
        rx_mand = re.compile(r"(mandatory\s*=\s*\[[^\]]*\])", re.M)
        m3 = rx_mand.search(text)
        if m3:
            block = m3.group(1)
            if "ret_fwd1" not in block:
                new_block = block[:-1] + ', "ret_fwd1"]'
                text = text.replace(block, new_block, 1)
                print('[fix] UPDATED: mandatory list includes "ret_fwd1" (excluded from obs)')

    # 5) Fix typo match_r -> match_ratio if present
    if "match_r" in text:
        text = text.replace("match_r =", "match_ratio =")
        text = text.replace("match_r <", "match_ratio <")
        text = text.replace("match_r)", "match_ratio)")
        print("[fix] UPDATED: typo 'match_r' -> 'match_ratio'")

    if text == original:
        print("[fix] OK: no changes made.")
        return 0

    backup = _backup(target, repo_root / "old")
    target.write_text(text, encoding="utf-8")
    print(f"[fix] UPDATED: {target}")
    print(f"[fix] backup -> {backup}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
