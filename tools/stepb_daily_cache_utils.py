# -*- coding: utf-8 -*-
"""Utilities to keep StepB daily outputs consistent.

This project sometimes keeps previously generated StepB daily outputs when a daily manifest already exists.
That can cause inconsistencies when you re-run with different settings.

This module:
- Computes a lightweight signature of the current run configuration.
- Purges StepB daily outputs when the signature changes.
- Repairs daily outputs to match pred_path summaries.
- Runs a sanity check to ensure consistency.

No Windows-style paths are embedded in strings to avoid unicode escape issues.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class StepBSignature:
    symbol: str
    mode: str
    lookback: int
    horizons: Tuple[int, ...]
    enable_mamba: bool
    enable_mamba_periodic: bool
    target_mode_env: str

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "mode": self.mode,
            "lookback": self.lookback,
            "horizons": list(self.horizons),
            "enable_mamba": bool(self.enable_mamba),
            "enable_mamba_periodic": bool(self.enable_mamba_periodic),
            "target_mode_env": self.target_mode_env,
        }

    def stable_hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]


def _read_csv_rows(path: Path) -> List[dict]:
    """Read CSV into list-of-dicts, normalizing headers.

    Some CSVs may be written with UTF-8 BOM (sometimes even duplicated).
    We normalize keys by stripping leading BOM characters and whitespace.
    """
    if not path.exists():
        return []
    out: List[dict] = []
    # Use utf-8-sig to drop one BOM if present; we still strip any remaining BOM from headers.
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned = {}
            for k, v in row.items():
                if k is None:
                    continue
                kk = str(k).lstrip("﻿").strip()
                cleaned[kk] = v
            out.append(cleaned)
    return out


def _write_csv_rows(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _parse_int_list(s: str) -> Tuple[int, ...]:
    out: List[int] = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return tuple(out)


def build_signature_from_args(
    symbol: str,
    mode: str,
    lookback: int,
    horizons_csv: str,
    enable_mamba: bool,
    enable_mamba_periodic: bool,
) -> StepBSignature:
    target_mode_env = os.environ.get("STEPB_MAMBA_TARGET_MODE", "").strip()
    horizons = _parse_int_list(horizons_csv)
    if not horizons:
        horizons = (1,)
    return StepBSignature(
        symbol=symbol,
        mode=mode,
        lookback=int(lookback),
        horizons=horizons,
        enable_mamba=enable_mamba,
        enable_mamba_periodic=enable_mamba_periodic,
        target_mode_env=target_mode_env,
    )


def signature_path(stepb_dir: Path, symbol: str) -> Path:
    return stepb_dir / f".stepB_daily_signature_{symbol}.json"


def load_signature(stepb_dir: Path, symbol: str) -> Optional[dict]:
    p = signature_path(stepb_dir, symbol)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_signature(stepb_dir: Path, symbol: str, sig: StepBSignature) -> None:
    p = signature_path(stepb_dir, symbol)
    obj = {
        "created_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "sig": sig.to_dict(),
        "hash": sig.stable_hash(),
    }
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def purge_stepb_daily_outputs(stepb_dir: Path, symbol: str) -> List[Path]:
    """Remove StepB daily outputs that can be stale.

    Returns list of removed paths (for logging).
    """
    removed: List[Path] = []
    targets = [
        stepb_dir / "daily",
        stepb_dir / "daily_periodic",
        stepb_dir / f"stepB_daily_manifest_{symbol}.csv",
        stepb_dir / f"stepB_daily_manifest_periodic_{symbol}.csv",
    ]
    for t in targets:
        if t.is_dir():
            shutil.rmtree(t, ignore_errors=True)
            removed.append(t)
        elif t.exists():
            try:
                t.unlink()
            except Exception:
                pass
            removed.append(t)
    return removed


def _load_pred_path_map(pred_path_csv: Path, key_col: str, val_col: str) -> Dict[str, float]:
    if not pred_path_csv.exists():
        return {}
    rows = _read_csv_rows(pred_path_csv)
    out: Dict[str, float] = {}
    for r in rows:
        k = (r.get(key_col) or "").strip()
        v = r.get(val_col)
        if not k or v is None or v == "":
            continue
        try:
            out[k] = float(v)
        except Exception:
            continue
    return out


def _float_fmt(x: float) -> str:
    # Keep enough precision; pandas will parse fine.
    return f"{x:.10f}".rstrip("0").rstrip(".") if math.isfinite(x) else ""


def repair_stepb_daily_from_pred_path(repo_root: Path, symbol: str, mode: str, log_lines: List[str]) -> None:
    """Repair StepB daily outputs to match pred_path summaries.

    - For full mamba daily (h01): enforce Pred_Close == Pred_Close_t_plus_01
    - For periodic daily_periodic (h20): enforce step=20 Pred_Close == Pred_Close_t_plus_20
      and fill steps 1..20 with a smooth geometric path.

    This does NOT feed predictions back into model inputs. It only makes output files consistent.
    """
    stepb_dir = repo_root / "output" / "stepB" / mode
    if not stepb_dir.exists():
        log_lines.append(f"[repair] stepB dir not found: {stepb_dir}")
        return

    # Maps for expected values
    full_map = _load_pred_path_map(stepb_dir / f"stepB_pred_path_mamba_{symbol}.csv", "Date_anchor", "Pred_Close_t_plus_01")
    periodic_map = _load_pred_path_map(stepb_dir / f"stepB_pred_path_mamba_periodic_{symbol}.csv", "Date_anchor", "Pred_Close_t_plus_20")

    # Repair full daily h01 based on manifest (test days)
    man_full = stepb_dir / f"stepB_daily_manifest_{symbol}.csv"
    if man_full.exists():
        rows = _read_csv_rows(man_full)
        fixed = 0
        for r in rows:
            d = (r.get("Date") or "").strip()
            pred_path = (r.get("pred_path_h01") or "").strip()
            if not d or not pred_path:
                continue
            expected = full_map.get(d)
            if expected is None:
                continue
            fpath = repo_root / pred_path.replace("/", os.sep)
            if not fpath.exists():
                # Create minimal daily file if missing
                daily_rows = [{
                    "Date_anchor": d,
                    "Close_anchor": "",
                    "mode": mode,
                    "symbol": symbol,
                    "stepA_features_path": r.get("stepA_features_path", ""),
                    "step_ahead_bdays": "1",
                    "Date_target": "",
                    "Pred_y_from_anchor": _float_fmt(expected),
                    "target_mode": "close",
                    "Pred_Close": _float_fmt(expected),
                    "Pred_ret_from_anchor": "",
                    "horizon_model": "1",
                }]
                _write_csv_rows(
                    fpath,
                    daily_rows,
                    [
                        "Date_anchor",
                        "Close_anchor",
                        "mode",
                        "symbol",
                        "stepA_features_path",
                        "step_ahead_bdays",
                        "Date_target",
                        "Pred_y_from_anchor",
                        "target_mode",
                        "Pred_Close",
                        "Pred_ret_from_anchor",
                        "horizon_model",
                    ],
                )
                fixed += 1
                continue

            drows = _read_csv_rows(fpath)
            if not drows:
                continue
            # Expect 1 row
            row0 = drows[0]
            try:
                close_anchor = float(row0.get("Close_anchor", ""))
                pred_ret = expected / close_anchor - 1.0
            except Exception:
                pred_ret = float("nan")
            row0["Pred_Close"] = _float_fmt(expected)
            row0["Pred_y_from_anchor"] = _float_fmt(expected)
            row0["target_mode"] = "close"
            row0["horizon_model"] = row0.get("horizon_model", "1") or "1"
            row0["step_ahead_bdays"] = row0.get("step_ahead_bdays", "1") or "1"
            if math.isfinite(pred_ret):
                row0["Pred_ret_from_anchor"] = _float_fmt(pred_ret)
            drows[0] = row0
            _write_csv_rows(fpath, drows, list(drows[0].keys()))
            fixed += 1
        log_lines.append(f"[repair] full daily fixed/created: {fixed}")
    else:
        log_lines.append("[repair] full manifest not found; skip full daily repair")

    # Repair periodic daily_periodic based on manifest
    man_per = stepb_dir / f"stepB_daily_manifest_periodic_{symbol}.csv"
    if man_per.exists():
        rows = _read_csv_rows(man_per)
        fixed = 0
        skipped = 0
        for r in rows:
            d = (r.get("Date") or "").strip()
            pred_path = (r.get("pred_path_h20") or "").strip()
            if not d or not pred_path:
                continue
            expected20 = periodic_map.get(d)
            if expected20 is None:
                skipped += 1
                continue
            fpath = repo_root / pred_path.replace("/", os.sep)
            if not fpath.exists():
                skipped += 1
                continue
            drows = _read_csv_rows(fpath)
            if not drows:
                skipped += 1
                continue
            # Need Close_anchor
            try:
                close_anchor = float(drows[0].get("Close_anchor", ""))
                if close_anchor <= 0:
                    raise ValueError
            except Exception:
                skipped += 1
                continue
            # Build geometric path
            try:
                per_step_lr = math.log(expected20 / close_anchor) / 20.0
            except Exception:
                skipped += 1
                continue

            # Update rows preserving Date_target and step_ahead_bdays
            for rr in drows:
                try:
                    step = int(float(rr.get("step_ahead_bdays", "0")))
                except Exception:
                    step = 0
                if step < 1 or step > 20:
                    continue
                pred_close = close_anchor * math.exp(per_step_lr * step)
                rr["target_mode"] = "logret"
                rr["horizon_model"] = rr.get("horizon_model", "20") or "20"
                rr["Pred_y_from_anchor"] = _float_fmt(per_step_lr)
                rr["Pred_Close"] = _float_fmt(pred_close)
                rr["Pred_ret_from_anchor"] = _float_fmt(pred_close / close_anchor - 1.0)
            _write_csv_rows(fpath, drows, list(drows[0].keys()))
            fixed += 1
        log_lines.append(f"[repair] periodic daily fixed: {fixed}, skipped: {skipped}")
    else:
        log_lines.append("[repair] periodic manifest not found; skip periodic repair")


def run_sanity_check(repo_root: Path, symbol: str, mode: str, report_path: Path, tol: float = 1e-6) -> bool:
    """Check daily files match pred_path summaries for test days.

    Returns True if passed.
    """
    stepb_dir = repo_root / "output" / "stepB" / mode
    lines: List[str] = []

    full_map = _load_pred_path_map(stepb_dir / f"stepB_pred_path_mamba_{symbol}.csv", "Date_anchor", "Pred_Close_t_plus_01")
    per_map = _load_pred_path_map(stepb_dir / f"stepB_pred_path_mamba_periodic_{symbol}.csv", "Date_anchor", "Pred_Close_t_plus_20")

    ok = True

    def _cmp(a: float, b: float) -> bool:
        return abs(a - b) <= tol

    # full
    man_full = stepb_dir / f"stepB_daily_manifest_{symbol}.csv"
    if man_full.exists():
        rows = _read_csv_rows(man_full)
        for r in rows:
            d = (r.get("Date") or "").strip()
            pred_path = (r.get("pred_path_h01") or "").strip()
            if not d or not pred_path:
                continue
            expected = full_map.get(d)
            if expected is None:
                lines.append(f"FULL {d}: SKIP (no pred_path)")
                continue
            fpath = repo_root / pred_path.replace("/", os.sep)
            if not fpath.exists():
                ok = False
                lines.append(f"FULL {d}: FAIL (missing daily file)")
                continue
            drows = _read_csv_rows(fpath)
            if not drows:
                ok = False
                lines.append(f"FULL {d}: FAIL (empty daily file)")
                continue
            try:
                got = float(drows[0].get("Pred_Close", ""))
            except Exception:
                ok = False
                lines.append(f"FULL {d}: FAIL (invalid Pred_Close)")
                continue
            if not _cmp(got, expected):
                ok = False
                lines.append(f"FULL {d}: FAIL got={got} expected={expected}")
            else:
                lines.append(f"FULL {d}: OK")
    else:
        ok = False
        lines.append("FULL: manifest missing")

    # periodic step=20
    man_per = stepb_dir / f"stepB_daily_manifest_periodic_{symbol}.csv"
    if man_per.exists():
        rows = _read_csv_rows(man_per)
        for r in rows:
            d = (r.get("Date") or "").strip()
            pred_path = (r.get("pred_path_h20") or "").strip()
            if not d or not pred_path:
                continue
            expected = per_map.get(d)
            if expected is None:
                lines.append(f"PER {d}: SKIP (no pred_path)")
                continue
            fpath = repo_root / pred_path.replace("/", os.sep)
            if not fpath.exists():
                ok = False
                lines.append(f"PER {d}: FAIL (missing daily file)")
                continue
            drows = _read_csv_rows(fpath)
            if not drows:
                ok = False
                lines.append(f"PER {d}: FAIL (empty daily file)")
                continue
            # find step 20
            row20 = None
            for rr in drows:
                try:
                    step = int(float(rr.get("step_ahead_bdays", "0")))
                except Exception:
                    step = 0
                if step == 20:
                    row20 = rr
                    break
            if row20 is None:
                ok = False
                lines.append(f"PER {d}: FAIL (no step=20 row)")
                continue
            try:
                got = float(row20.get("Pred_Close", ""))
            except Exception:
                ok = False
                lines.append(f"PER {d}: FAIL (invalid Pred_Close)")
                continue
            if not _cmp(got, expected):
                ok = False
                lines.append(f"PER {d}: FAIL got={got} expected={expected}")
            else:
                lines.append(f"PER {d}: OK")
    else:
        ok = False
        lines.append("PER: manifest missing")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ok
