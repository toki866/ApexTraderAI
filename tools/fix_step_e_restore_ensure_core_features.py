# tools/fix_step_e_restore_ensure_core_features.py
# -*- coding: utf-8 -*-
"""
Fix StepE TypeError: "list indices must be integers or slices, not str" at df["Date"].

Root cause:
- ai_core/services/step_e_service.py の _ensure_core_features() が、誤って DataFrame を返さず
  "列名のリスト(list)" を返す状態になっている（例: out=[c for c in out if ...] のような処理が混入）

このスクリプトは以下を実施します：
1) step_e_service.py を old/ にバックアップ
2) StepEService._ensure_core_features() を「正しい DataFrame 返却版」に丸ごと置換
3) _SUSPICIOUS_COL_PAT が無い場合は追加（label/target/future 等を観測に入れない）
4) _run_one() の obs_cols フィルタに "suspicious column" 除外を追加

Usage:
  (base) cd <repo_root>
  python tools/fix_step_e_restore_ensure_core_features.py

Then rerun:
  python tools/run_stepE_compare_stepdprime10.py --symbol SOXL --mode sim
"""

from __future__ import annotations

import datetime as _dt
import os
import re
from pathlib import Path

def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def _backup(src: Path, repo_root: Path) -> Path:
    old_dir = repo_root / "old"
    old_dir.mkdir(parents=True, exist_ok=True)
    dst = old_dir / f"{src.stem}_old_{_timestamp()}_01{src.suffix}"
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return dst

def _find_block(lines: list[str], start_pat: re.Pattern, indent: str) -> tuple[int, int]:
    """
    Return (start_idx, end_idx_exclusive) for a method block.
    start_pat matches the def line. end is next line that starts with indent + 'def ' (same indent).
    """
    start = None
    for i, ln in enumerate(lines):
        if start_pat.search(ln):
            start = i
            break
    if start is None:
        return (-1, -1)

    # find next method at same indent
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith(indent + "def "):
            end = j
            break
    return (start, end)

def _ensure_suspicious_pat(text: str) -> str:
    # ensure "import re" exists
    if not re.search(r"(?m)^\s*import\s+re\s*$", text):
        # insert after first future/typing imports block or after module docstring
        lines = text.splitlines(True)
        insert_at = 0
        # skip shebang/encoding/docstring
        i = 0
        if i < len(lines) and lines[i].startswith("#!"):
            i += 1
        if i < len(lines) and "coding" in lines[i]:
            i += 1
        # docstring block
        if i < len(lines) and lines[i].lstrip().startswith('"""'):
            i += 1
            while i < len(lines) and '"""' not in lines[i]:
                i += 1
            if i < len(lines):
                i += 1
        # insert after that
        insert_at = i
        lines.insert(insert_at, "import re\n")
        text = "".join(lines)

    if not re.search(r"(?m)^\s*_SUSPICIOUS_COL_PAT\s*=", text):
        # insert before class StepEService:
        m = re.search(r"(?m)^\s*class\s+StepEService\s*:", text)
        if m:
            idx = m.start()
            ins = "\n# Suspicious columns that must NOT enter RL observation (leak-prone / labels / targets)\n" \
                  "_SUSPICIOUS_COL_PAT = re.compile(r\"(realclose|close_true|future|label|target|y_|leak)\", re.IGNORECASE)\n\n"
            text = text[:idx] + ins + text[idx:]
        else:
            # append at end of imports
            text = text + "\n_SUSPICIOUS_COL_PAT = re.compile(r\"(realclose|close_true|future|label|target|y_|leak)\", re.IGNORECASE)\n"
    return text

def _patch_run_one_obs_filter(text: str) -> str:
    """
    Replace:
      obs_cols = [c for c in obs_cols if c in df_all.columns]
    with:
      obs_cols = [c for c in obs_cols if c in df_all.columns and not _SUSPICIOUS_COL_PAT.search(c)]
    If already patched, do nothing.
    """
    if "and not _SUSPICIOUS_COL_PAT.search(c)" in text:
        return text

    pat = re.compile(r"(?m)^\s*obs_cols\s*=\s*\[c\s+for\s+c\s+in\s+obs_cols\s+if\s+c\s+in\s+df_all\.columns\s*\]\s*$")
    return pat.sub("        obs_cols = [c for c in obs_cols if c in df_all.columns and not _SUSPICIOUS_COL_PAT.search(c)]", text)

def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "ai_core" / "services" / "step_e_service.py"
    if not path.exists():
        raise FileNotFoundError(f"not found: {path}")

    src = path.read_text(encoding="utf-8")
    src = _ensure_suspicious_pat(src)

    lines = src.splitlines(True)

    # locate _ensure_core_features method inside StepEService
    start_pat = re.compile(r"^\s{4}def\s+_ensure_core_features\s*\(", re.M)
    s, e = _find_block(lines, start_pat=start_pat, indent="    ")
    if s < 0:
        raise RuntimeError("Could not find StepEService._ensure_core_features() block.")

    replacement = [
        "    def _ensure_core_features(self, df: pd.DataFrame) -> pd.DataFrame:\n",
        "        \"\"\"Ensure minimum core features exist and return a DataFrame.\n",
        "\n",
        "        Required:\n",
        "          - Gap      : Open/PrevClose - 1\n",
        "          - ATR_norm : ATR(14)/PrevClose\n",
        "          - oc_ret   : Close/Open - 1 (used as return label for RL)\n",
        "\n",
        "        IMPORTANT: This function MUST return a pandas.DataFrame.\n",
        "        \"\"\"\n",
        "        out = df.copy()\n",
        "\n",
        "        # Normalize Date\n",
        "        if \"Date\" in out.columns:\n",
        "            out[\"Date\"] = pd.to_datetime(out[\"Date\"], errors=\"coerce\").dt.normalize()\n",
        "\n",
        "        # Gap\n",
        "        if \"Gap\" not in out.columns:\n",
        "            if \"gap\" in out.columns:\n",
        "                out[\"Gap\"] = out[\"gap\"].astype(float)\n",
        "            else:\n",
        "                close_prev = out[\"Close\"].astype(float).shift(1)\n",
        "                out[\"Gap\"] = out[\"Open\"].astype(float) / close_prev.replace(0, np.nan) - 1.0\n",
        "\n",
        "        # ATR_norm (ATR14 / PrevClose)\n",
        "        if \"ATR_norm\" not in out.columns:\n",
        "            if \"atr_norm_14\" in out.columns:\n",
        "                out[\"ATR_norm\"] = out[\"atr_norm_14\"].astype(float)\n",
        "            elif \"atr_norm\" in out.columns:\n",
        "                out[\"ATR_norm\"] = out[\"atr_norm\"].astype(float)\n",
        "            else:\n",
        "                high = out[\"High\"].astype(float)\n",
        "                low = out[\"Low\"].astype(float)\n",
        "                close_prev = out[\"Close\"].astype(float).shift(1)\n",
        "                tr = pd.concat([\n",
        "                    (high - low),\n",
        "                    (high - close_prev).abs(),\n",
        "                    (low - close_prev).abs(),\n",
        "                ], axis=1).max(axis=1)\n",
        "                atr = tr.rolling(14, min_periods=14).mean()\n",
        "                out[\"ATR_norm\"] = atr / close_prev.replace(0, np.nan)\n",
        "\n",
        "        # Open->Close return (label for RL)\n",
        "        if \"oc_ret\" not in out.columns:\n",
        "            out[\"oc_ret\"] = out[\"Close\"].astype(float) / out[\"Open\"].astype(float).replace(0, np.nan) - 1.0\n",
        "\n",
        "        # Clean core columns\n",
        "        for c in [\"Gap\", \"ATR_norm\", \"oc_ret\"]:\n",
        "            if c in out.columns:\n",
        "                out[c] = out[c].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)\n",
        "\n",
        "        return out\n",
        "\n",
    ]

    new_lines = lines[:s] + replacement + lines[e:]
    new_text = "".join(new_lines)
    new_text = _patch_run_one_obs_filter(new_text)

    # write backup + overwrite
    bk = _backup(path, repo_root)
    path.write_text(new_text, encoding="utf-8")

    print(f"[fix] UPDATED: restored _ensure_core_features() in {path}")
    print(f"[fix] backup -> {bk}")

    # quick sanity: ensure function now returns DataFrame (static check only)
    if re.search(r"def\s+_ensure_core_features.*return\s+out", new_text, flags=re.S) is None:
        print("[fix] WARN: could not confirm 'return out' in _ensure_core_features (please inspect).")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
