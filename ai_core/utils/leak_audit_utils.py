from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


EQ_TOL = 1e-10
EQ_ABS_TOL = 1e-12


def _require_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"missing required columns: {missing}")


def _equity_reconstruction_rel_err(eq: np.ndarray, ret: np.ndarray) -> float:
    if eq.size == 0:
        return 0.0
    eq_recon = np.cumprod(1.0 + ret)
    first_recon = float(eq_recon[0]) if eq_recon.size else 1.0
    first_eq = float(eq[0]) if eq.size else 1.0
    if abs(first_recon) <= 1e-18:
        scale = 0.0 if abs(first_eq) <= EQ_ABS_TOL else 1.0
    else:
        scale = first_eq / first_recon
    eq_recon = eq_recon * scale
    denom = np.maximum(np.maximum(np.abs(eq), np.abs(eq_recon)), EQ_ABS_TOL)
    rel = np.abs(eq_recon - eq) / denom
    return float(np.nanmax(rel)) if np.isfinite(rel).any() else 0.0


def audit_stepE_reward_alignment(df: pd.DataFrame, split: str = "test", tol: float = 1e-12) -> Dict[str, Any]:
    sub = df[df["Split"] == split].copy().reset_index(drop=True)
    _require_columns(sub, ["Date", "ret", "reward_next", "equity", "pos", "r_soxl_next", "r_soxs_next", "cost"])

    ret = pd.to_numeric(sub["ret"], errors="coerce").astype(float).to_numpy(dtype=float)
    reward_next = pd.to_numeric(sub["reward_next"], errors="coerce").astype(float).to_numpy(dtype=float)
    eq = pd.to_numeric(sub["equity"], errors="coerce").astype(float).to_numpy(dtype=float)

    if len(ret) > 1:
        diff = np.abs(ret[1:] - reward_next[:-1])
        max_abs = float(np.max(diff))
        idx = int(np.argmax(diff) + 1)
        max_date = str(sub["Date"].iloc[idx])
    else:
        max_abs = 0.0
        max_date = ""

    rel_err = _equity_reconstruction_rel_err(eq=eq, ret=ret)

    out: Dict[str, Any] = {
        "check": "stepE_reward_alignment",
        "split": split,
        "status": "PASS" if (max_abs <= tol and rel_err <= EQ_TOL) else "FAIL",
        "max_abs": max_abs,
        "max_date": max_date,
        "rel_err": rel_err,
        "rows": int(len(sub)),
        "tol": float(tol),
        "equity_tol": float(EQ_TOL),
        "note": "ret[t] == reward_next[t-1]; equity == cumprod(1+ret)",
    }

    if "underlying" in sub.columns:
        pos = pd.to_numeric(sub["pos"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        underlying = sub["underlying"].astype(str).str.upper().to_numpy()
        mask = np.abs(pos) > 1e-12
        if np.any(mask):
            expected = np.where(pos[mask] > 0.0, "SOXL", "SOXS")
            out["underlying_match_rate"] = float(np.mean(expected == underlying[mask]))
        else:
            out["underlying_match_rate"] = float("nan")

    return out


def audit_stepF_market_alignment(df: pd.DataFrame, split: str = "test", tol: float = 1e-12) -> Dict[str, Any]:
    sub = df[df["Split"] == split].copy().reset_index(drop=True)
    _require_columns(sub, ["Date", "ratio", "ret", "cost", "equity", "r_soxl", "r_soxs"])

    ratio = pd.to_numeric(sub["ratio"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    ret = pd.to_numeric(sub["ret"], errors="coerce").astype(float).to_numpy(dtype=float)
    cost = pd.to_numeric(sub["cost"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    r_soxl = pd.to_numeric(sub["r_soxl"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    r_soxs = pd.to_numeric(sub["r_soxs"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    eq = pd.to_numeric(sub["equity"], errors="coerce").astype(float).to_numpy(dtype=float)

    market = np.maximum(ratio, 0.0) * r_soxl + np.maximum(-ratio, 0.0) * r_soxs
    diff = np.abs(ret - (market - cost)) if len(ret) > 0 else np.array([0.0], dtype=float)
    max_abs = float(np.max(diff)) if diff.size > 0 else 0.0
    idx = int(np.argmax(diff)) if diff.size > 0 else 0
    max_date = str(sub["Date"].iloc[idx]) if len(sub) > 0 else ""

    rel_err = _equity_reconstruction_rel_err(eq=eq, ret=ret)

    return {
        "check": "stepF_market_alignment",
        "split": split,
        "status": "PASS" if (max_abs <= tol and rel_err <= EQ_TOL) else "FAIL",
        "max_abs": max_abs,
        "max_date": max_date,
        "rel_err": rel_err,
        "rows": int(len(sub)),
        "tol": float(tol),
        "equity_tol": float(EQ_TOL),
        "note": "ret == max(ratio,0)*r_soxl + max(-ratio,0)*r_soxs - cost; equity == cumprod(1+ret)",
    }


def write_audit_reports(audit_dir: Path, prefix: str, audit_dict: Dict[str, Any]) -> tuple[Path, Path]:
    audit_dir.mkdir(parents=True, exist_ok=True)
    json_path = audit_dir / f"{prefix}.json"
    csv_path = audit_dir / f"{prefix}.csv"

    json_path.write_text(json.dumps(audit_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{"key": str(k), "value": v} for k, v in audit_dict.items()]).to_csv(csv_path, index=False)
    return json_path, csv_path


# ---------------------------------------------------------------------------
# Per-step immediate audit helpers (called right after each step completes)
# ---------------------------------------------------------------------------

def audit_stepe_agent_now(
    output_root: Path,
    mode: str,
    symbol: str,
    agent: str,
    audit_root: Path,
) -> Dict[str, Any]:
    """Load StepE daily_log for *agent*, run reward-alignment audit, write reports.

    Returns the audit dict (keys: check, status, max_abs, max_date, rel_err, ...).
    Raises if the daily_log file is missing or unreadable.
    """
    log_path = Path(output_root) / "stepE" / mode / f"stepE_daily_log_{agent}_{symbol}.csv"
    df = pd.read_csv(log_path)
    audit = audit_stepE_reward_alignment(df, split="test", tol=1e-12)
    prefix = f"audit_stepE_{agent}_{symbol}"
    write_audit_reports(Path(audit_root), prefix, audit)
    return audit


def audit_stepf_now(
    output_root: Path,
    mode: str,
    symbol: str,
    audit_root: Path,
) -> Dict[str, Dict[str, Any]]:
    """Run StepF market-alignment audit for router + marl daily logs. Write reports.

    Returns dict keyed by name ("router", "marl"). Only includes names whose log
    file exists.
    """
    results: Dict[str, Dict[str, Any]] = {}
    for name in ("router", "marl"):
        p = Path(output_root) / "stepF" / mode / f"stepF_daily_log_{name}_{symbol}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        audit = audit_stepF_market_alignment(df, split="test", tol=1e-12)
        prefix = f"audit_stepF_{name}_{symbol}"
        write_audit_reports(Path(audit_root), prefix, audit)
        results[name] = audit
    return results
