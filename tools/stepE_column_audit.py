#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/stepE_column_audit.py

StepE "runtime" column audit / guard tool.

Goal
----
Detect accidental data leakage by auditing columns present in *inputs that StepE might consume*
(e.g., StepD_prime embeddings/features, StepD outputs, StepB predictions, StepA features),
and write an audit log under output/stepE/<mode>/audit/.

This is designed to be run:
  A) just before running StepE (recommended, no code change), or
  B) called from StepE code as a guard (optional: see "Hook into StepE" below).

Usage
-----
  python tools\\stepE_column_audit.py --symbol SOXL --mode sim
  python tools\\stepE_column_audit.py --symbol SOXL --mode sim --strict

Options
-------
  --output-root  default: output
  --audit-dir    default: <output-root>/stepE/<mode>/audit
  --strict       exit non-zero if suspicious columns found
  --show-cols    show full column lists in console (can be long)
  --only-types   comma separated subset of: stepA,stepB,stepD,stepDprime,stepE
  --extra        extra CSV paths to audit (repeatable)

What is "suspicious"?
---------------------
By default, columns matching any of these patterns are flagged:
  - label / label_available
  - (close|price)_true
  - REALCLOSE
  - target / y / gt (ground-truth) like names
  - future / next / t+1 like names
  - leak

IMPORTANT:
  Having a 'label' column in a CSV is not automatically a leak, *IF* StepE drops it.
  But keeping it in the same file makes accidents easy. This tool makes it visible and
  can enforce a strict guard.

Hook into StepE (optional)
--------------------------
If you want StepE to always log columns during execution, add this near the start of StepE run:
    from tools.stepE_column_audit import audit_and_guard
    audit_and_guard(output_root=app_config.output_root, mode=mode, symbol=symbol,
                    strict=os.getenv("STEP_E_AUDIT_STRICT","0")=="1")

No other changes are required.

Exit codes
----------
  0 : OK (no suspicious columns, or --strict not used)
  2 : suspicious columns found and --strict
  3 : fatal errors (missing StepA prices etc are NOT fatal for this audit)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


SUSPICIOUS_PATTERNS: Sequence[Tuple[str, str]] = (
    ("label", r"(^|[^a-zA-Z0-9])label($|[^a-zA-Z0-9])"),
    ("label_available", r"label_available"),
    ("close_true", r"close_true|price_true|true_close|close_gt|gt_close"),
    ("realclose", r"realclose"),
    ("target", r"(^|[^a-zA-Z0-9])target($|[^a-zA-Z0-9])|(^|[^a-zA-Z0-9])y($|[^a-zA-Z0-9])"),
    ("future_next", r"future|next|t\+1|t1|lead|ahead"),
    ("leak", r"leak"),
)


DEFAULT_ONLY_TYPES = "stepA,stepB,stepD,stepDprime,stepE"


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_header(csv_path: Path) -> List[str]:
    # Use pandas directly; nrows=0 reads only header.
    df = pd.read_csv(csv_path, nrows=0, engine="python")
    return list(df.columns)


def _match_suspicious(cols: Sequence[str]) -> List[str]:
    hits: List[str] = []
    for c in cols:
        cl = str(c).lower()
        for name, pat in SUSPICIOUS_PATTERNS:
            if re.search(pat, cl, flags=re.IGNORECASE):
                hits.append(str(c))
                break
    return sorted(set(hits))


def _glob_many(patterns: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    for pat in patterns:
        out.extend(Path().glob(pat) if (("*" not in pat) and ("?" not in pat) and ("[" not in pat)) else Path().glob(pat))
    # The above isn't reliable for absolute patterns; use Path.glob only for relative.
    return out


def _glob_abs(patterns: Sequence[str]) -> List[Path]:
    # Robust absolute glob using glob module
    import glob

    files: List[Path] = []
    for pat in patterns:
        files.extend([Path(x) for x in glob.glob(pat, recursive=True)])
    # unique
    uniq: Dict[str, Path] = {}
    for p in files:
        uniq[str(p)] = p
    return sorted(uniq.values(), key=lambda x: str(x))


@dataclass
class AuditRow:
    file_type: str
    path: str
    ncols: int
    suspicious_cols: List[str]
    read_error: str = ""
    cols_preview: List[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "file_type": self.file_type,
            "path": self.path,
            "ncols": self.ncols,
            "has_suspicious": bool(self.suspicious_cols),
            "suspicious_cols": json.dumps(self.suspicious_cols, ensure_ascii=False),
            "read_error": self.read_error,
            "cols_preview": json.dumps(self.cols_preview or [], ensure_ascii=False),
        }


def _patterns_for_types(output_root: Path, mode: str, symbol: str, only_types: Sequence[str]) -> Dict[str, List[str]]:
    pats: Dict[str, List[str]] = {}

    if "stepA" in only_types:
        pats["stepA"] = [
            str(output_root / "stepA" / mode / f"*{symbol}*.csv"),
        ]
    if "stepB" in only_types:
        pats["stepB"] = [
            str(output_root / "stepB" / mode / "**" / f"*{symbol}*.csv"),
        ]
    if "stepD" in only_types:
        pats["stepD"] = [
            str(output_root / "stepD" / mode / f"*{symbol}*.csv"),
            str(output_root / "stepD" / mode / "**" / f"*{symbol}*.csv"),
        ]
    if "stepDprime" in only_types:
        pats["stepDprime"] = [
            str(output_root / "stepD_prime" / mode / "**" / f"*{symbol}*.csv"),
        ]
    if "stepE" in only_types:
        pats["stepE"] = [
            str(output_root / "stepE" / mode / f"*{symbol}*.csv"),
        ]

    return pats


def audit_columns(
    output_root: str,
    mode: str,
    symbol: str,
    audit_dir: Optional[str] = None,
    strict: bool = False,
    show_cols: bool = False,
    only_types: Sequence[str] = ("stepA", "stepB", "stepD", "stepDprime", "stepE"),
    extra_paths: Sequence[str] = (),
) -> Tuple[int, Path, Path]:
    out_root = Path(output_root)
    if audit_dir is None:
        audit_dir_path = out_root / "stepE" / mode / "audit"
    else:
        audit_dir_path = Path(audit_dir)

    _safe_mkdir(audit_dir_path)

    rows: List[AuditRow] = []
    patterns = _patterns_for_types(out_root, mode, symbol, list(only_types))

    total_files = 0
    for ftype, pats in patterns.items():
        files = _glob_abs(pats)
        # Reduce noise: ignore model weights and non-csv already filtered
        files = [p for p in files if p.is_file() and p.suffix.lower() == ".csv"]
        total_files += len(files)

        for p in files:
            try:
                cols = _read_header(p)
                susp = _match_suspicious(cols)
                preview = cols[:30]
                rows.append(AuditRow(file_type=ftype, path=str(p), ncols=len(cols), suspicious_cols=susp, cols_preview=preview))
                if show_cols:
                    print(f"[{ftype}] {p} cols({len(cols)}): {cols}")
                else:
                    print(f"[{ftype}] {p} cols={len(cols)} suspicious={len(susp)}" + (f" -> {susp}" if susp else ""))
            except Exception as e:
                rows.append(AuditRow(file_type=ftype, path=str(p), ncols=0, suspicious_cols=[], read_error=str(e), cols_preview=[]))
                print(f"[{ftype}] {p} READ_ERROR: {e}")

    # Extra user-provided paths
    for xp in extra_paths:
        p = Path(xp)
        ftype = "extra"
        if p.is_dir():
            candidates = sorted([x for x in p.rglob("*.csv") if x.is_file()])
        else:
            candidates = [p] if p.is_file() else []
        for cp in candidates:
            try:
                cols = _read_header(cp)
                susp = _match_suspicious(cols)
                preview = cols[:30]
                rows.append(AuditRow(file_type=ftype, path=str(cp), ncols=len(cols), suspicious_cols=susp, cols_preview=preview))
                if show_cols:
                    print(f"[{ftype}] {cp} cols({len(cols)}): {cols}")
                else:
                    print(f"[{ftype}] {cp} cols={len(cols)} suspicious={len(susp)}" + (f" -> {susp}" if susp else ""))
            except Exception as e:
                rows.append(AuditRow(file_type=ftype, path=str(cp), ncols=0, suspicious_cols=[], read_error=str(e), cols_preview=[]))
                print(f"[{ftype}] {cp} READ_ERROR: {e}")

    # Write reports
    tag = _now_tag()
    csv_path = audit_dir_path / f"stepE_column_audit_{symbol}_{mode}_{tag}.csv"
    txt_path = audit_dir_path / f"stepE_column_audit_{symbol}_{mode}_{tag}.txt"

    df = pd.DataFrame([r.to_dict() for r in rows])
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    suspicious_rows = df[df["has_suspicious"] == True]  # noqa: E712
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"StepE Column Audit Report\n")
        f.write(f"  symbol={symbol}\n  mode={mode}\n  output_root={out_root}\n")
        f.write(f"  scanned_files={len(rows)} (patterns_total={total_files}, extra={len(extra_paths)})\n")
        f.write("\nSuspicious patterns:\n")
        for name, pat in SUSPICIOUS_PATTERNS:
            f.write(f"  - {name}: /{pat}/\n")
        f.write("\nResults:\n")
        if suspicious_rows.shape[0] == 0:
            f.write("  OK: no suspicious columns found.\n")
        else:
            f.write(f"  WARNING: suspicious columns found in {suspicious_rows.shape[0]} files.\n\n")
            for _, r in suspicious_rows.iterrows():
                f.write(f"- {r['file_type']} {r['path']}\n")
                f.write(f"    suspicious_cols={r['suspicious_cols']}\n")
                if r.get("read_error"):
                    f.write(f"    read_error={r['read_error']}\n")
                f.write("\n")

        # Also list read errors
        err_rows = df[df["read_error"].astype(str) != ""]
        if err_rows.shape[0] > 0:
            f.write("\nRead errors:\n")
            for _, r in err_rows.iterrows():
                f.write(f"- {r['file_type']} {r['path']}\n")
                f.write(f"    error={r['read_error']}\n")

    print(f"\n[report] csv -> {csv_path}")
    print(f"[report] txt -> {txt_path}")

    if strict and suspicious_rows.shape[0] > 0:
        print("[strict] suspicious columns detected -> exit code 2")
        return (2, csv_path, txt_path)
    return (0, csv_path, txt_path)


def audit_and_guard(output_root: str, mode: str, symbol: str, strict: bool = False) -> None:
    """
    Small helper for calling from StepE code.
    Raises RuntimeError if strict and suspicious columns found.
    """
    code, csv_path, txt_path = audit_columns(output_root=output_root, mode=mode, symbol=symbol, strict=strict)
    if strict and code != 0:
        raise RuntimeError(f"StepE column audit FAILED (suspicious columns). See: {txt_path}")


def _parse_only_types(s: str) -> List[str]:
    parts = [x.strip() for x in s.split(",") if x.strip()]
    allowed = {"stepA", "stepB", "stepD", "stepDprime", "stepE"}
    out = [p for p in parts if p in allowed]
    return out if out else list(allowed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--mode", default="sim")
    ap.add_argument("--output-root", default="output")
    ap.add_argument("--audit-dir", default="")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--show-cols", action="store_true")
    ap.add_argument("--only-types", default=DEFAULT_ONLY_TYPES)
    ap.add_argument("--extra", action="append", default=[])
    args = ap.parse_args()

    only_types = _parse_only_types(args.only_types)
    audit_dir = args.audit_dir.strip() or None

    code, _, _ = audit_columns(
        output_root=args.output_root,
        mode=args.mode,
        symbol=args.symbol,
        audit_dir=audit_dir,
        strict=args.strict,
        show_cols=args.show_cols,
        only_types=only_types,
        extra_paths=args.extra,
    )
    return code


if __name__ == "__main__":
    sys.exit(main())
