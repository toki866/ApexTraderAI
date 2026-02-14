#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
StepD' embeddings 出力を「安全な最小列」にサニタイズするツール。

目的
- StepD' / StepB 側に存在し得る "label/target/y_/available/true/realclose/Date_target/Pred_y_from_anchor/target_mode" 等の列が、
  embeddings CSV に混ざって監査ノイズになるのを防ぐ。
- 事故（将来の変更で「全数値列を特徴量にする」等）を防ぐ安全弁として、embeddings 出力を最小化する。

動作
- output_root 配下から「dprime_*_emb_*」列を含む CSV を探索（symbol を含むファイル名を優先）
- 対象CSVについて、
    - Date列（大文字小文字違い許容）
    - embeddings列（正規表現: ^dprime_.*_emb_\d+$ または "dprime_" かつ "_emb_" を含む）
  だけを残して書き出す
- 既定では *sanitized.csv を別名で生成。--inplace を付けると上書き（old/ にバックアップ）

使い方例（Windows cmd）
  cd /d C:\path\to\soxl_rl_gui
  set "PYTHONPATH=%CD%"
  python tools\stepDprime_sanitize_embeddings.py --symbol SOXL --mode sim --output-root output
  python tools\stepDprime_sanitize_embeddings.py --symbol SOXL --mode sim --output-root output --inplace

注意
- StepE 側は embeddings 列だけ見ているので、出力を最小化しても基本的に壊れません。
- ただし、あなたの StepD' 実装が「追加のメタ列」を StepE が参照する設計になっている場合は、
  --inplace ではなく別名出力で様子見してください（デフォルトは別名出力）。
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd


EMB_RE = re.compile(r"^dprime_.*_emb_\d+$", re.IGNORECASE)
EMB_RAW_RE = re.compile(r"^emb_\d+$", re.IGNORECASE)

SUSPICIOUS_KEYS = (
    "label",
    "available",
    "target",
    "y_",
    "close_true",
    "price_true",
    "true_close",
    "realclose",
    "date_target",
    "pred_y_from_anchor",
    "target_mode",
    "future",
    "next",
    "t+1",
    "t1",
    "lead",
    "ahead",
)

PREFERRED_DIRS = (
    "stepD_prime",
    "stepDprime",
    "stepDPrime",
    "stepD",
)


@dataclass
class SanitizeResult:
    path_in: Path
    path_out: Path
    date_col: str
    ncols_in: int
    ncols_out: int
    dropped_cols: List[str]
    suspicious_dropped: List[str]


def _find_date_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if str(c).strip().lower() == "date":
            return c
    # fallback: contains "date"
    for c in df.columns:
        if "date" in str(c).strip().lower():
            return c
    return None


def _find_emb_cols(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []

    # Preferred: already-renamed dprime_*_emb_*
    for c in df.columns:
        s = str(c)
        if EMB_RE.match(s):
            cols.append(c)
    if cols:
        return cols

    # Next: raw emb_### columns (from run_stepd_prime_features.py)
    for c in df.columns:
        s = str(c)
        if EMB_RAW_RE.match(s):
            cols.append(c)
    if cols:
        return cols

    # Fallback: dprime_ + _emb_ を含む
    for c in df.columns:
        sl = str(c).lower()
        if sl.startswith("dprime_") and "_emb_" in sl:
            cols.append(c)

    return cols
    # fallback: dprime_ + _emb_ を含む
    for c in df.columns:
        sl = str(c).lower()
        if sl.startswith("dprime_") and "_emb_" in sl:
            cols.append(c)

    return cols


def _is_suspicious(col: str) -> bool:
    cl = col.lower()
    return any(k in cl for k in SUSPICIOUS_KEYS)


def _iter_candidate_csvs(output_root: Path, mode: str, symbol: str) -> List[Path]:
    # まず “ありそうな場所” を優先探索
    cands: List[Path] = []
    sym = symbol.upper().strip()

    for dname in PREFERRED_DIRS:
        base = output_root / dname / mode
        if base.exists():
            cands.extend(sorted(base.rglob(f"*{sym}*.csv")))

    # 見つからなければ output_root 全体から探す
    if not cands:
        cands = sorted(output_root.rglob(f"*{sym}*.csv"))

    # 重複排除
    seen = set()
    uniq: List[Path] = []
    for p in cands:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def sanitize_one(path_in: Path, out_dir: Path, inplace: bool, backup_dir: Path) -> Optional[SanitizeResult]:
    try:
        df = pd.read_csv(path_in)
    except Exception:
        return None

    date_col = _find_date_col(df)
    if date_col is None:
        return None

    emb_cols = _find_emb_cols(df)
    if not emb_cols:
        return None

    keep_cols = [date_col] + sorted(emb_cols, key=lambda x: str(x))
    dropped = [c for c in df.columns if c not in keep_cols]
    suspicious_dropped = [c for c in dropped if _is_suspicious(str(c))]

    df_out = df.loc[:, keep_cols].copy()

    if inplace:
        backup_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"{path_in.stem}_old_{ts}{path_in.suffix}"
        backup_path.write_bytes(path_in.read_bytes())
        path_out = path_in
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        path_out = out_dir / f"{path_in.stem}_sanitized{path_in.suffix}"

    df_out.to_csv(path_out, index=False)

    return SanitizeResult(
        path_in=path_in,
        path_out=path_out,
        date_col=str(date_col),
        ncols_in=len(df.columns),
        ncols_out=len(df_out.columns),
        dropped_cols=[str(c) for c in dropped],
        suspicious_dropped=[str(c) for c in suspicious_dropped],
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--mode", default="sim", choices=["sim", "live", "ops", "display"])
    ap.add_argument("--output-root", default="output")
    ap.add_argument("--out-dir", default=None, help="出力先（省略時は入力CSVと同じフォルダ）")
    ap.add_argument("--inplace", action="store_true", help="入力CSVを上書き（old/ にバックアップ）")
    ap.add_argument("--backup-dir", default="old", help="--inplace のバックアップ先ディレクトリ")
    ap.add_argument("--report", default=None, help="レポートtxt出力パス（省略時は標準出力）")
    args = ap.parse_args()

    output_root = Path(args.output_root)
    mode = args.mode
    symbol = args.symbol

    cands = _iter_candidate_csvs(output_root, mode, symbol)
    results: List[SanitizeResult] = []

    for p in cands:
        out_dir = Path(args.out_dir) if args.out_dir else p.parent
        r = sanitize_one(p, out_dir=out_dir, inplace=args.inplace, backup_dir=Path(args.backup_dir))
        if r is not None:
            results.append(r)

    lines: List[str] = []
    lines.append(f"[stepDprime_sanitize] symbol={symbol} mode={mode} output_root={output_root.as_posix()}")
    lines.append(f"candidates_scanned={len(cands)} sanitized={len(results)} inplace={args.inplace}")
    for r in results:
        lines.append("-" * 100)
        lines.append(f"in : {r.path_in.as_posix()}")
        lines.append(f"out: {r.path_out.as_posix()}")
        lines.append(f"date_col={r.date_col}  ncols_in={r.ncols_in} -> ncols_out={r.ncols_out}")
        if r.suspicious_dropped:
            lines.append(f"suspicious_dropped({len(r.suspicious_dropped)}): {r.suspicious_dropped}")
        else:
            lines.append("suspicious_dropped: []")

    text = "\n".join(lines)

    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
    else:
        print(text)

    # Exit code: 0 if at least one sanitized, else 2 (nothing found)
    return 0 if results else 2


if __name__ == "__main__":
    raise SystemExit(main())
