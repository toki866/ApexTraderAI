from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


EQ_TOL = 1e-10


def _require_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"missing required columns: {missing}")


def _equity_reconstruction_rel_err(eq: np.ndarray, ret: np.ndarray) -> float:
    if eq.size == 0:
        return 0.0
    eq_recon = np.cumprod(1.0 + ret)
    denom = np.where(np.abs(eq) < 1e-18, np.nan, eq)
    rel = np.abs(eq_recon / denom - 1.0)
    if np.isnan(rel).all():
        return float("inf")
    return float(np.nanmax(rel))


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
