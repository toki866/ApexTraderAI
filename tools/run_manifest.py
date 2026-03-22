# -*- coding: utf-8 -*-
"""Run-level manifest and reuse helpers for ApexTraderAI.

Manages per-run manifest.json that tracks step completion status and enables
artifact-based reuse across runs (including post-cancel resume).

Usage in run_pipeline.py:
    from tools.run_manifest import (
        RunSignature, RunManifest,
        check_step_artifacts, check_stepe_agent_artifact,
        find_latest_matching_run,
    )
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ai_core.utils.manifest_path_utils import resolve_output_artifact_path

# ---------------------------------------------------------------------------
# Official StepE agent list (mirrors _OFFICIAL_STEPE_AGENTS in run_pipeline.py)
# ---------------------------------------------------------------------------
_OFFICIAL_STEPE_AGENTS: Tuple[str, ...] = (
    "dprime_bnf_h01",
    "dprime_bnf_h02",
    "dprime_bnf_3scale",
    "dprime_mix_h01",
    "dprime_mix_h02",
    "dprime_mix_3scale",
    "dprime_all_features_h01",
    "dprime_all_features_h02",
    "dprime_all_features_h03",
    "dprime_all_features_3scale",
)

_MANIFEST_FILENAME = "run_manifest.json"
_REUSE_SIGNATURE_FILENAME = "reuse_signature.json"
_SCHEMA_VERSION = 2
_CANONICAL_OUTPUT_BASE_WINDOWS = Path("C:/work/apex_work/output")
_CANONICAL_OUTPUT_BASE_WSL = Path("/mnt/c/work/apex_work/output")


def build_canonical_output_base() -> Path:
    """Return canonical output base path for current runtime.

    Canonical base is fixed to C:/work/apex_work/output on Windows and
    /mnt/c/work/apex_work/output on WSL/Linux.
    """
    if os.name == "nt":
        return _CANONICAL_OUTPUT_BASE_WINDOWS
    return _CANONICAL_OUTPUT_BASE_WSL


def build_canonical_output_root(base_output_root: Path, mode: str, symbol: str, test_start: str) -> Path:
    """Return canonical artifact root: output/<mode>/<symbol>/<test_start>/.

    base_output_root is the logical output base (typically repo/output).
    """
    ts = str(test_start or "").strip() or "unknown_test_start"
    return Path(base_output_root) / str(mode).strip().lower() / str(symbol).strip().upper() / ts


def resolve_canonical_output_root(mode: str, symbol: str, test_start: str) -> Path:
    """Return canonical output root using fixed canonical output base."""
    return build_canonical_output_root(build_canonical_output_base(), mode, symbol, test_start)


def required_outputs_for_step(step: str, symbol: str, mode: str) -> Tuple[str, ...]:
    s = step.upper()
    sym = str(symbol)
    if s == "A":
        return (
            f"stepA/{mode}/stepA_prices_train_{sym}.csv",
            f"stepA/{mode}/stepA_prices_test_{sym}.csv",
            f"stepA/{mode}/stepA_periodic_train_{sym}.csv",
            f"stepA/{mode}/stepA_periodic_test_{sym}.csv",
            f"stepA/{mode}/stepA_tech_train_{sym}.csv",
            f"stepA/{mode}/stepA_tech_test_{sym}.csv",
            f"stepA/{mode}/stepA_split_summary_{sym}.csv",
            f"stepA/{mode}/stepA_periodic_future_{sym}.csv",
            f"stepA/{mode}/stepA_daily_manifest_{sym}.csv",
            "split_summary.json",
        )
    if s == "B":
        return (f"stepB/{mode}/stepB_pred_time_all_{sym}.csv", "split_summary.json")
    if s == "C":
        return (f"stepC/{mode}/stepC_features_{sym}.csv", "split_summary.json")
    if s == "DPRIME":
        return (f"stepDprime/{mode}/stepDprime_state_test_dprime_all_features_h01_{sym}.csv", "split_summary.json")
    if s == "E":
        return (f"stepE/{mode}/stepE_daily_log_dprime_all_features_h01_{sym}.csv", "split_summary.json")
    if s == "F":
        return (
            f"stepF/{mode}/stepF_daily_log_router_{sym}.csv",
            f"stepF/{mode}/stepF_daily_log_marl_{sym}.csv",
            "split_summary.json",
        )
    return tuple()


def load_split_summary(output_root: Path) -> Optional[dict]:
    p = Path(output_root) / "split_summary.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else None
    except Exception:
        return None


def split_summary_matches(summary: dict, *, symbol: str, mode: str, test_start: str, train_years: int, test_months: int) -> bool:
    if not isinstance(summary, dict):
        return False
    if str(summary.get("symbol", "")).upper() != str(symbol).upper():
        return False
    if str(summary.get("mode", "")).lower() != str(mode).lower():
        return False
    if str(summary.get("test_start", "")) != str(test_start or ""):
        return False
    # Existing repo uses train_years/test_months on CLI; compare these when present.
    if "train_years" in summary and int(summary.get("train_years", -1)) != int(train_years):
        return False
    if "test_months" in summary and int(summary.get("test_months", -1)) != int(test_months):
        return False
    return True


def _csv_valid(path: Path, required_cols: Tuple[str, ...]) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    if df.empty:
        return False
    cols = set(df.columns)
    return all(c in cols for c in required_cols)


def _validate_stepa_daily_manifest(output_root: Path, symbol: str, mode: str) -> bool:
    manifest_path = Path(output_root) / f"stepA/{mode}/stepA_daily_manifest_{symbol}.csv"
    if not manifest_path.exists() or manifest_path.stat().st_size <= 0:
        return False
    try:
        manifest = pd.read_csv(manifest_path)
    except Exception:
        return False
    if manifest.empty:
        return False
    required_cols = ("prices_path", "periodic_path", "tech_path", "periodic_future_path")
    if not all(c in manifest.columns for c in required_cols):
        return False
    for c in required_cols:
        for raw in manifest[c].fillna("").astype(str):
            p = resolve_output_artifact_path(raw, canonical_output_root=output_root)
            if not raw.strip() or not p.exists():
                return False
    return True


def validate_step_outputs(step: str, output_root: Path, symbol: str, mode: str) -> Tuple[bool, str]:
    required = required_outputs_for_step(step, symbol, mode)
    for rel in required:
        p = Path(output_root) / rel
        if not p.exists():
            return False, "missing_required_outputs"
        if p.is_file() and p.stat().st_size <= 0:
            return False, "invalid_output_data"
    csv_rules = {
        "A": [(f"stepA/{mode}/stepA_prices_train_{symbol}.csv", ("Date", "Open", "High", "Low", "Close", "Volume"))],
        "B": [(f"stepB/{mode}/stepB_pred_time_all_{symbol}.csv", ("Date",))],
        "C": [(f"stepC/{mode}/stepC_features_{symbol}.csv", ("Date",))],
        "DPRIME": [(f"stepDprime/{mode}/stepDprime_state_test_dprime_all_features_h01_{symbol}.csv", ("Date",))],
        "E": [(f"stepE/{mode}/stepE_daily_log_dprime_all_features_h01_{symbol}.csv", ("Date",))],
        "F": [(f"stepF/{mode}/stepF_daily_log_router_{symbol}.csv", ("Date",))],
    }
    for rel, cols in csv_rules.get(step.upper(), []):
        p = Path(output_root) / rel
        if not _csv_valid(p, cols):
            return False, "invalid_output_data"
    if step.upper() == "A" and not _validate_stepa_daily_manifest(Path(output_root), symbol, mode):
        return False, "invalid_output_data"
    return True, "reuse"


# ---------------------------------------------------------------------------
# RunSignature
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunSignature:
    """Immutable run-condition signature used to match prior runs."""
    symbol: str
    mode: str
    test_start: str           # "YYYY-MM-DD" or "" if derived
    train_years: int
    test_months: int
    steps: Tuple[str, ...]    # sorted requested steps
    enable_mamba: bool
    enable_mamba_periodic: bool
    mamba_lookback: Optional[int]
    mamba_horizons: Tuple[int, ...]
    stepe_agents: Tuple[str, ...]  # sorted; empty = all official agents

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "mode": self.mode,
            "test_start": self.test_start,
            "train_years": int(self.train_years),
            "test_months": int(self.test_months),
            "steps": list(self.steps),
            "enable_mamba": bool(self.enable_mamba),
            "enable_mamba_periodic": bool(self.enable_mamba_periodic),
            "mamba_lookback": self.mamba_lookback,
            "mamba_horizons": list(self.mamba_horizons),
            "stepe_agents": list(self.stepe_agents),
        }

    def stable_hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]


@dataclass(frozen=True)
class StepASimpleReuseSignature:
    """Simplified StepA reuse signature.

    StepA only checks these keys:
    - mode
    - symbols
    - test_start
    - train_years
    - test_months
    """

    mode: str
    symbols: Tuple[str, ...]
    test_start: str
    train_years: int
    test_months: int


@dataclass(frozen=True)
class ReuseOutputSignature:
    """Stable key used to select an output root for reuse."""

    mode: str
    symbols: Tuple[str, ...]
    test_start_date: str
    train_years: int
    test_months: int
    feature_signature: str
    algorithm_signature: str
    parameter_signature: str

    def to_dict(self) -> dict:
        payload = {
            "mode": str(self.mode),
            "symbols": list(self.symbols),
            "test_start_date": str(self.test_start_date),
            "train_years": int(self.train_years),
            "test_months": int(self.test_months),
            "feature_signature": str(self.feature_signature),
            "algorithm_signature": str(self.algorithm_signature),
            "parameter_signature": str(self.parameter_signature),
        }
        payload["reuse_key_hash"] = stable_reuse_key_hash(payload)
        return payload


def stable_reuse_key_hash(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def build_reuse_output_signature(
    mode: str,
    symbols: Tuple[str, ...],
    test_start_date: str,
    train_years: int,
    test_months: int,
    feature_signature: str,
    algorithm_signature: str,
    parameter_signature: str,
) -> ReuseOutputSignature:
    return ReuseOutputSignature(
        mode=str(mode),
        symbols=_normalize_symbols(symbols),
        test_start_date=str(test_start_date or ""),
        train_years=int(train_years),
        test_months=int(test_months),
        feature_signature=str(feature_signature),
        algorithm_signature=str(algorithm_signature),
        parameter_signature=str(parameter_signature),
    )


def _load_reuse_signature(output_root: Path) -> Optional[dict]:
    sig_path = Path(output_root) / _REUSE_SIGNATURE_FILENAME
    if not sig_path.exists():
        return None
    try:
        data = json.loads(sig_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _extract_reuse_signature_from_manifest(output_root: Path) -> Optional[dict]:
    manifest_path = Path(output_root) / _MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    sig = data.get("signature", {}) if isinstance(data, dict) else {}
    if not isinstance(sig, dict):
        return None
    symbol_values = sig.get("symbols")
    if symbol_values is None:
        symbol_values = sig.get("symbol")
    payload = {
        "mode": str(sig.get("mode", "")),
        "symbols": list(_normalize_symbols(symbol_values)),
        "test_start_date": str(sig.get("test_start", "")),
        "train_years": int(sig.get("train_years", 0)),
        "test_months": int(sig.get("test_months", 0)),
        "feature_signature": "",
        "algorithm_signature": "",
        "parameter_signature": "",
    }
    payload["reuse_key_hash"] = stable_reuse_key_hash(payload)
    return payload


def find_matching_output_root(scan_root: Path, reuse_sig: ReuseOutputSignature) -> Optional[Path]:
    """Return canonical output root (scan_root is ignored for compatibility)."""
    symbols = _normalize_symbols(reuse_sig.symbols)
    if not symbols:
        return None
    candidate = resolve_canonical_output_root(reuse_sig.mode, symbols[0], reuse_sig.test_start_date)
    return candidate if candidate.exists() else None


def build_run_signature(
    symbol: str,
    mode: str,
    test_start: Optional[str],
    train_years: int,
    test_months: int,
    steps: Tuple[str, ...],
    enable_mamba: bool,
    enable_mamba_periodic: bool,
    mamba_lookback: Optional[int],
    mamba_horizons: Tuple[int, ...],
    stepe_agents: Optional[Tuple[str, ...]],
) -> RunSignature:
    """Build a RunSignature from pipeline arguments."""
    sorted_steps = tuple(sorted(set(steps)))
    effective_agents: Tuple[str, ...]
    if stepe_agents:
        effective_agents = tuple(sorted(set(stepe_agents)))
    else:
        effective_agents = ()  # empty = use all official agents
    return RunSignature(
        symbol=str(symbol),
        mode=str(mode),
        test_start=str(test_start or ""),
        train_years=int(train_years),
        test_months=int(test_months),
        steps=sorted_steps,
        enable_mamba=bool(enable_mamba),
        enable_mamba_periodic=bool(enable_mamba_periodic),
        mamba_lookback=int(mamba_lookback) if mamba_lookback is not None else None,
        mamba_horizons=tuple(sorted(set(mamba_horizons))) if mamba_horizons else (),
        stepe_agents=effective_agents,
    )


def _normalize_symbols(values: object) -> Tuple[str, ...]:
    """Normalize symbols to upper-case tuple preserving order and uniqueness."""
    if values is None:
        return ()

    if isinstance(values, str):
        raw_items = values.split(",")
    elif isinstance(values, (list, tuple, set)):
        raw_items = list(values)
    else:
        raw_items = [values]

    out: List[str] = []
    seen = set()
    for item in raw_items:
        sym = str(item).strip()
        if not sym:
            continue
        key = sym.upper()
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return tuple(out)


# ---------------------------------------------------------------------------
# Artifact presence checks
# ---------------------------------------------------------------------------

def check_step_artifacts(
    step: str,
    output_root: Path,
    symbol: str,
    mode: str,
    *,
    required_stepf_reward_modes: Optional[Tuple[str, ...]] = None,
    required_dprime_profiles: Optional[Tuple[str, ...]] = None,
) -> bool:
    """Return True if the key artifacts for *step* exist in output_root."""
    base = Path(output_root)
    step_upper = step.upper()

    if step_upper == "A":
        d = base / "stepA" / mode
        required = (
            f"stepA_prices_train_{symbol}.csv",
            f"stepA_prices_test_{symbol}.csv",
            f"stepA_periodic_train_{symbol}.csv",
            f"stepA_periodic_test_{symbol}.csv",
            f"stepA_tech_train_{symbol}.csv",
            f"stepA_tech_test_{symbol}.csv",
            f"stepA_split_summary_{symbol}.csv",
            f"stepA_periodic_future_{symbol}.csv",
            f"stepA_daily_manifest_{symbol}.csv",
            "split_summary.json",
        )
        if not all((base / name if name == "split_summary.json" else d / name).exists() for name in required):
            return False
        return _validate_stepa_daily_manifest(base, symbol, mode)

    if step_upper == "B":
        d = base / "stepB" / mode
        required = (
            f"stepB_pred_time_all_{symbol}.csv",
            f"stepB_pred_close_mamba_{symbol}.csv",
            f"stepB_pred_path_mamba_{symbol}.csv",
            f"stepB_pred_close_mamba_periodic_{symbol}.csv",
            f"stepB_pred_path_mamba_periodic_{symbol}.csv",
            f"stepB_summary_{symbol}.json",
        )
        return all((d / name).exists() for name in required) and (base / "audit" / mode / f"stepB_audit_{symbol}.json").exists()

    if step_upper == "C":
        d = base / "stepC" / mode
        if not d.exists():
            return False
        # Prefer explicit main-series artifacts when present, but keep historical fallback.
        preferred = (
            d / f"stepC_features_{symbol}.csv",
            d / f"stepC_state_{symbol}.csv",
        )
        if any(p.exists() for p in preferred):
            return True
        return any(d.glob(f"*{symbol}*.csv"))

    if step_upper == "DPRIME":
        d = base / "stepDprime" / mode
        emb = d / "embeddings"
        marker_dir = d / "pipeline_markers"
        profiles = tuple(str(p).strip() for p in (required_dprime_profiles or _OFFICIAL_STEPE_AGENTS) if str(p).strip())
        if not (d / f"stepDprime_base_meta_{symbol}.json").exists():
            return False
        if not (marker_dir / "DPrimeBaseCluster.READY.json").exists():
            return False
        for profile in profiles:
            # StepDPrime success is defined by state+embedding artifacts.
            # split_summary may be absent on some historical runs and should not
            # force a false-negative reuse/failure when core artifacts are complete.
            if not (d / f"stepDprime_state_train_{profile}_{symbol}.csv").exists():
                return False
            if not (d / f"stepDprime_state_test_{profile}_{symbol}.csv").exists():
                return False
            if not (emb / f"stepDprime_{profile}_{symbol}_embeddings_test.csv").exists():
                return False
            if not (emb / f"stepDprime_{profile}_{symbol}_embeddings_all.csv").exists():
                return False
            if not (d / f"stepDprime_profile_summary_{profile}_{symbol}.json").exists():
                return False
            if not (marker_dir / f"DPrimeFinal_{profile}.READY.json").exists():
                return False
        return True

    if step_upper == "E":
        d = base / "stepE" / mode
        model_dir = d / "models"
        return (
            d.exists()
            and any(d.glob(f"stepE_daily_log_*_{symbol}.csv"))
            and any(d.glob(f"stepE_summary_*_{symbol}.json"))
            and any((base / "audit" / mode).glob(f"stepE_audit_*_{symbol}.json"))
            and model_dir.exists()
            and (
                any(model_dir.glob(f"stepE_*_{symbol}.pt"))
                or any(model_dir.glob(f"stepE_*_{symbol}_ppo.zip"))
            )
        )

    if step_upper == "F":
        d = base / "stepF" / mode
        reward_modes = tuple(str(x).strip().lower() for x in (required_stepf_reward_modes or tuple()) if str(x).strip())
        primary_ok = (
            (d / f"stepF_equity_marl_{symbol}.csv").exists()
            and (d / f"stepF_daily_log_marl_{symbol}.csv").exists()
            and (d / f"stepF_daily_log_router_{symbol}.csv").exists()
            and (d / f"stepF_summary_router_{symbol}.json").exists()
            and (d / f"stepF_audit_router_{symbol}.json").exists()
            and (base / "audit" / mode / f"stepF_policy_compare_{symbol}.json").exists()
        )
        if len(reward_modes) > 1:
            primary_ok = primary_ok and (d / f"stepF_compare_reward_modes_{symbol}.json").exists() and (d / f"stepF_best_reward_mode_{symbol}.json").exists()
        if not primary_ok:
            return False
        for rm in reward_modes:
            rdir = d / f"reward_{rm}"
            if not _stepf_reward_mode_complete(rdir, symbol):
                return False
        return True

    return False


def check_stepe_agent_artifact(agent: str, output_root: Path, symbol: str, mode: str) -> bool:
    """Return True if required StepE artifacts for *agent* exist.

    Required minimum set:
    - stepE_daily_log_<agent>_<SYMBOL>.csv
    - stepE_summary_<agent>_<SYMBOL>.json
    - model artifact: .pt or _ppo.zip
    """
    base = Path(output_root) / "stepE" / mode
    daily = base / f"stepE_daily_log_{agent}_{symbol}.csv"
    summary = base / f"stepE_summary_{agent}_{symbol}.json"
    model_pt = base / "models" / f"stepE_{agent}_{symbol}.pt"
    model_zip = base / "models" / f"stepE_{agent}_{symbol}_ppo.zip"
    audit = Path(output_root) / "audit" / mode / f"stepE_audit_{agent}_{symbol}.json"
    return daily.exists() and summary.exists() and audit.exists() and (model_pt.exists() or model_zip.exists())


def _read_json_payload(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _stepf_reward_mode_complete(reward_dir: Path, symbol: str) -> bool:
    status_payload = _read_json_payload(reward_dir / "status.json")
    manifest_payload = _read_json_payload(reward_dir / "artifacts_manifest.json")
    if str(status_payload.get("status", "")).lower() != "complete":
        return False
    if not bool(status_payload.get("required_artifacts_present", False)):
        return False
    if not bool(manifest_payload.get("validation_passed", False)):
        return False
    required = (
        reward_dir / f"stepF_equity_marl_{symbol}.csv",
        reward_dir / f"stepF_daily_log_marl_{symbol}.csv",
        reward_dir / f"stepF_daily_log_router_{symbol}.csv",
        reward_dir / f"stepF_summary_router_{symbol}.json",
        reward_dir / f"stepF_audit_router_{symbol}.json",
        reward_dir / f"stepF_policy_compare_{symbol}.json",
    )
    return all(path.exists() and path.stat().st_size > 0 for path in required)


def read_stepe_quality_status(agent: str, output_root: Path, symbol: str, mode: str) -> Optional[str]:
    """Return normalized PASS/FAIL/WARN for a StepE agent from audit/summary payloads.

    Preference order:
    1) audit JSON's ``audit_status`` or ``status``
    2) summary JSON's ``audit_status`` or ``status``

    Returns ``None`` when no supported status key is present.
    """
    base = Path(output_root) / "stepE" / mode
    audit_path = Path(output_root) / "audit" / mode / f"stepE_audit_{agent}_{symbol}.json"
    summary_path = base / f"stepE_summary_{agent}_{symbol}.json"
    for meta_path in (audit_path, summary_path):
        payload = _read_json_payload(meta_path)
        for key in ("audit_status", "status"):
            raw = str(payload.get(key, "") or "").strip().upper()
            if raw:
                return raw
    return None


def read_stepf_quality_status(output_root: Path, symbol: str, mode: str) -> Optional[str]:
    """Return normalized PASS/WARN/FAIL for StepF based on persisted audit payloads."""
    audit_root = Path(output_root) / "audit" / mode
    mode_root = Path(output_root) / "stepF" / mode
    statuses: List[str] = []
    for meta_path in (
        audit_root / f"audit_stepF_router_{symbol}.json",
        audit_root / f"audit_stepF_marl_{symbol}.json",
        mode_root / f"stepF_audit_router_{symbol}.json",
        mode_root / f"stepF_summary_router_{symbol}.json",
    ):
        payload = _read_json_payload(meta_path)
        for key in ("audit_status", "status"):
            raw = str(payload.get(key, "") or "").strip().upper()
            if raw:
                statuses.append(raw)
                break
    normalized = [s for s in statuses if s in {"PASS", "WARN", "FAIL"}]
    if not normalized:
        return None
    if "FAIL" in normalized:
        return "FAIL"
    if "WARN" in normalized:
        return "WARN"
    return "PASS"


def reconcile_stepe_manifest_from_artifacts(
    requested_agents: List[str],
    *,
    output_root: Path,
    mode: str,
    symbol: str,
    manifest: Optional["RunManifest"] = None,
) -> Dict[str, Any]:
    """Reconcile stale StepE manifest state against on-disk artifacts.

    A StepE agent is considered complete only when:
    - daily log / summary / audit / model artifacts exist, and
    - quality status resolves to ``PASS``.

    When *manifest* is provided, stale per-agent FAIL/pending entries are self-healed
    to ``complete``+``PASS`` whenever artifacts and quality agree. The step-level
    ``E`` status is also updated to ``complete``/``PASS`` once all requested agents
    converge, otherwise downgraded to ``pending``/``FAIL``.
    """
    normalized_agents = list(dict.fromkeys(str(a).strip() for a in requested_agents if str(a).strip()))
    complete_agents: List[str] = []
    missing_agents: List[str] = []
    missing_detail: Dict[str, List[str]] = {}

    for agent in normalized_agents:
        artifact_ok = check_stepe_agent_artifact(agent, output_root, symbol, mode)
        quality_status = read_stepe_quality_status(agent, output_root, symbol, mode)
        quality_ok = str(quality_status or "").upper() == "PASS"
        manifest_ok = False
        if manifest is not None:
            manifest_ok = manifest.can_reuse_stepe_agent(agent)
            if artifact_ok and quality_ok and not manifest_ok:
                manifest.mark_stepe_agent_verified(
                    agent,
                    "complete",
                    artifacts_ok=True,
                    audit_status="PASS",
                    invalid_status="pending",
                )
                manifest_ok = manifest.can_reuse_stepe_agent(agent)

        if artifact_ok and quality_ok and (manifest is None or manifest_ok):
            complete_agents.append(agent)
            continue

        missing_agents.append(agent)
        detail: List[str] = []
        base = Path(output_root) / "stepE" / mode
        model_dir = base / "models"
        if not (base / f"stepE_daily_log_{agent}_{symbol}.csv").exists():
            detail.append("daily_log")
        if not (base / f"stepE_summary_{agent}_{symbol}.json").exists():
            detail.append("summary")
        if not (Path(output_root) / "audit" / mode / f"stepE_audit_{agent}_{symbol}.json").exists():
            detail.append("audit")
        if not (
            (model_dir / f"stepE_{agent}_{symbol}.pt").exists()
            or (model_dir / f"stepE_{agent}_{symbol}_ppo.zip").exists()
        ):
            detail.append("model")
        if not quality_ok:
            detail.append(f"quality:{quality_status or 'missing'}")
        if manifest is not None and not manifest_ok:
            detail.append("manifest")
        missing_detail[agent] = detail

    all_complete = len(complete_agents) == len(normalized_agents)
    if manifest is not None and normalized_agents:
        manifest.mark_step_verified(
            "E",
            "complete",
            artifacts_ok=all_complete,
            audit_status="PASS" if all_complete else "FAIL",
            invalid_status="pending",
        )

    return {
        "requested_agents": normalized_agents,
        "complete_agents": complete_agents,
        "missing_agents": missing_agents,
        "missing_detail": missing_detail,
        "all_complete": all_complete,
        "final_status": "complete" if all_complete else "partial",
        "return_code": 0 if all_complete else 1,
    }


def reconcile_stepf_manifest_from_artifacts(
    *,
    output_root: Path,
    mode: str,
    symbol: str,
    requested_reward_modes: Optional[List[str]] = None,
    manifest: Optional["RunManifest"] = None,
) -> Dict[str, Any]:
    """Reconcile stale StepF manifest state against on-disk outputs and audits."""
    requested = list(dict.fromkeys(str(m).strip().lower() for m in (requested_reward_modes or []) if str(m).strip()))
    mode_root = Path(output_root) / "stepF" / mode
    summary_path = mode_root / f"stepF_multi_mode_summary_{symbol}.json"
    summary_payload = _read_json_payload(summary_path)
    if not requested:
        requested = [str(x).strip().lower() for x in summary_payload.get("reward_modes", []) if str(x).strip()]

    success_modes = [str(x).strip().lower() for x in summary_payload.get("success_modes", []) if str(x).strip()]
    failed_modes = [str(x).strip().lower() for x in summary_payload.get("failed_modes", []) if str(x).strip()]
    reused_modes = [str(x).strip().lower() for x in summary_payload.get("reused_modes", []) if str(x).strip()]
    publish_completed = bool(summary_payload.get("publish_completed", False))
    artifacts_ok = check_step_artifacts("F", output_root, symbol, mode, required_stepf_reward_modes=tuple(requested))
    audit_status = read_stepf_quality_status(output_root, symbol, mode) or ("PASS" if artifacts_ok else "FAIL")

    inconsistencies: List[Dict[str, Any]] = []
    if artifacts_ok and requested and set(requested) - set(success_modes) - set(reused_modes):
        inconsistencies.append(
            {
                "kind": "structured_inconsistency",
                "code": "stepf_reward_mode_summary_missing_success",
                "requested_reward_modes": requested,
                "success_modes": success_modes,
                "reused_modes": reused_modes,
            }
        )
    if artifacts_ok and summary_payload and not publish_completed:
        inconsistencies.append(
            {
                "kind": "structured_inconsistency",
                "code": "stepf_publish_incomplete_with_artifacts_present",
                "summary_path": str(summary_path),
            }
        )
    if failed_modes and artifacts_ok:
        inconsistencies.append(
            {
                "kind": "structured_inconsistency",
                "code": "stepf_failed_modes_present_but_canonical_outputs_complete",
                "failed_modes": failed_modes,
            }
        )

    final_status = "complete" if artifacts_ok else ("failed" if failed_modes else "pending")

    if manifest is not None:
        step_data = manifest._data.setdefault("steps", {}).setdefault("F", {})
        step_data.update(
            {
                "reward_modes_requested": requested,
                "reward_modes_completed": success_modes,
                "reward_modes_failed": failed_modes,
                "reward_modes_reused": reused_modes,
                "publish_completed": publish_completed,
                "multi_mode_summary_path": str(summary_path) if summary_path.exists() else "",
                "structured_inconsistencies": inconsistencies,
            }
        )
        if final_status == "complete":
            manifest.mark_step_verified("F", "complete", artifacts_ok=True, audit_status=audit_status, invalid_status="pending")
        else:
            manifest.mark_step("F", final_status)
            manifest.mark_step_audit("F", audit_status)
            manifest.save()

    return {
        "requested_reward_modes": requested,
        "success_modes": success_modes,
        "failed_modes": failed_modes,
        "reused_modes": reused_modes,
        "publish_completed": publish_completed,
        "artifacts_ok": artifacts_ok,
        "audit_status": audit_status,
        "final_status": final_status,
        "structured_inconsistencies": inconsistencies,
        "summary_path": str(summary_path) if summary_path.exists() else "",
    }


# ---------------------------------------------------------------------------
# RunManifest
# ---------------------------------------------------------------------------

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class RunManifest:
    """Tracks per-step completion status for a single run directory."""

    def __init__(self, output_root: Path, data: dict) -> None:
        self._output_root = output_root
        self._data = data

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def load_or_create(
        cls,
        output_root: Path,
        sig: RunSignature,
        reuse_enabled: bool,
        force_rebuild: bool,
    ) -> "RunManifest":
        """Load existing manifest or create a fresh one.

        If force_rebuild=True, the step statuses are reset to "pending"
        regardless of what is on disk.
        """
        path = Path(output_root) / _MANIFEST_FILENAME
        existing: Optional[dict] = None

        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                existing = None

        # Validate schema and signature match
        if existing is not None:
            if existing.get("schema_version") == 1:
                # Migrate v1 → v2 in-place (non-destructive: only adds new fields)
                existing = cls._migrate_v1_to_v2(existing)
            if existing.get("schema_version") != _SCHEMA_VERSION:
                existing = None
            elif existing.get("signature_hash") != sig.stable_hash():
                existing = None  # Different conditions → start fresh

        if existing is None or force_rebuild:
            data = cls._fresh_data(sig, reuse_enabled, force_rebuild)
        else:
            data = cls._normalize_loaded_data(existing, sig)
            data["reuse_enabled"] = reuse_enabled
            data["updated_at"] = _utcnow_iso()

        manifest = cls(Path(output_root), data)
        manifest.save()
        return manifest

    @classmethod
    def _fresh_data(cls, sig: RunSignature, reuse_enabled: bool, force_rebuild: bool) -> dict:
        now = _utcnow_iso()
        agent_list = list(sig.stepe_agents) if sig.stepe_agents else list(_OFFICIAL_STEPE_AGENTS)
        def _step_shell() -> dict:
            return {"status": "pending", "completed_at": None, "elapsed_sec": None, "audit_status": None, "reused": False, "rebuilt": False, "degraded": False, "failed": False}
        return {
            "schema_version": _SCHEMA_VERSION,
            "signature": sig.to_dict(),
            "signature_hash": sig.stable_hash(),
            "created_at": now,
            "updated_at": now,
            "reuse_enabled": reuse_enabled,
            "force_rebuild": force_rebuild,
            "source_output_root": None,
            "run_id": None,
            "requested_steps": list(sig.steps),
            "steps": {
                "A": _step_shell(),
                "B": _step_shell(),
                "C": _step_shell(),
                "DPRIME": _step_shell(),
                "E": {
                    **_step_shell(),
                    "agents": {
                        a: {"status": "pending", "completed_at": None, "elapsed_sec": None, "audit_status": None}
                        for a in agent_list
                    },
                },
                "F": _step_shell(),
            },
        }

    @classmethod
    def _migrate_v1_to_v2(cls, data: dict) -> dict:
        """Add v2 fields (elapsed_sec, audit_status) to a v1 manifest in-place."""
        for step, info in (data.get("steps") or {}).items():
            if not isinstance(info, dict):
                continue
            info.setdefault("elapsed_sec", None)
            info.setdefault("audit_status", None)
            info.setdefault("reused", False)
            info.setdefault("rebuilt", False)
            info.setdefault("degraded", False)
            info.setdefault("failed", False)
            # StepE agents
            for _agent, ainfo in (info.get("agents") or {}).items():
                if isinstance(ainfo, dict):
                    ainfo.setdefault("elapsed_sec", None)
                    ainfo.setdefault("audit_status", None)
        data["schema_version"] = _SCHEMA_VERSION
        data.setdefault("source_output_root", None)
        data.setdefault("run_id", None)
        return data

    @classmethod
    def _normalize_loaded_data(cls, data: dict, sig: Optional[RunSignature] = None) -> dict:
        """Coerce nullable/missing manifest fields into safe containers.

        Older/partially-written manifests can contain None values for keys that are
        treated as dict/list by runtime logic. Normalize them to avoid NoneType
        iteration errors during reuse checks.
        """
        if not isinstance(data, dict):
            data = {}

        failure_field = "<unknown>"

        def _warn_default(field: str, default_repr: str) -> None:
            print(f"[manifest_init] field={field} was None -> {default_repr}", file=sys.stderr)

        def _warn_invalid(field: str, actual: object, default_repr: str) -> None:
            print(
                f"[manifest_init] field={field} had invalid type={type(actual).__name__} -> {default_repr}",
                file=sys.stderr,
            )

        def _as_dict(obj: Optional[dict], field: str) -> dict:
            if obj is None:
                _warn_default(field, "{}")
                return {}
            if isinstance(obj, dict):
                return obj
            _warn_invalid(field, obj, "{}")
            return {}

        def _as_list(obj: Optional[list], field: str) -> list:
            if obj is None:
                _warn_default(field, "[]")
                return []
            if isinstance(obj, list):
                return obj
            if isinstance(obj, tuple):
                print(f"[manifest_init] field={field} had type=tuple -> list", file=sys.stderr)
                return list(obj)
            if isinstance(obj, set):
                print(f"[manifest_init] field={field} had type=set -> list", file=sys.stderr)
                return list(obj)
            _warn_invalid(field, obj, "[]")
            return []

        try:
            failure_field = "artifacts"
            data["artifacts"] = _as_list(data.get("artifacts"), "artifacts")
            failure_field = "entries"
            data["entries"] = _as_list(data.get("entries"), "entries")
            failure_field = "history"
            data["history"] = _as_list(data.get("history"), "history")
            failure_field = "required_outputs"
            data["required_outputs"] = _as_list(data.get("required_outputs"), "required_outputs")
            failure_field = "signature"
            data["signature"] = _as_dict(data.get("signature"), "signature")
            failure_field = "requested_steps"
            data["requested_steps"] = _as_list(data.get("requested_steps"), "requested_steps")
            failure_field = "steps"
            data["steps"] = _as_dict(data.get("steps"), "steps")

            failure_field = "signature.steps"
            if data["signature"].get("steps") is None:
                _warn_default("signature.steps", "[]")
                data["signature"]["steps"] = []
            elif not isinstance(data["signature"].get("steps"), list):
                _warn_invalid("signature.steps", data["signature"].get("steps"), "[]")
                data["signature"]["steps"] = []

            failure_field = "signature.stepe_agents"
            if data["signature"].get("stepe_agents") is None:
                _warn_default("signature.stepe_agents", "[]")
                data["signature"]["stepe_agents"] = []
            elif not isinstance(data["signature"].get("stepe_agents"), list):
                _warn_invalid("signature.stepe_agents", data["signature"].get("stepe_agents"), "[]")
                data["signature"]["stepe_agents"] = []

            failure_field = "steps"
            default_steps = cls._fresh_data(sig, bool(data.get("reuse_enabled")), bool(data.get("force_rebuild")))["steps"]
            for step, fallback in default_steps.items():
                failure_field = f"steps.{step}"
                s = data["steps"].get(step)
                if s is None:
                    _warn_default(f"steps.{step}", "{}")
                    s = {}
                if not isinstance(s, dict):
                    _warn_invalid(f"steps.{step}", s, "{}")
                    s = {}
                for k, v in fallback.items():
                    failure_field = f"steps.{step}.{k}"
                    if s.get(k) is None and v is not None:
                        _default_repr = "{}" if isinstance(v, dict) else "[]" if isinstance(v, list) else repr(v)
                        _warn_default(f"steps.{step}.{k}", _default_repr)
                        s[k] = v

                # Legacy/custom keys used by some manifests.
                failure_field = f"steps.{step}.required_outputs"
                if s.get("required_outputs") is None:
                    _warn_default(f"steps.{step}.required_outputs", "[]")
                    s["required_outputs"] = []
                elif not isinstance(s.get("required_outputs"), list):
                    _warn_invalid(f"steps.{step}.required_outputs", s.get("required_outputs"), "[]")
                    s["required_outputs"] = []

                failure_field = f"steps.{step}.prior_outputs"
                if s.get("prior_outputs") is None:
                    _warn_default(f"steps.{step}.prior_outputs", "[]")
                    s["prior_outputs"] = []
                elif not isinstance(s.get("prior_outputs"), list):
                    _warn_invalid(f"steps.{step}.prior_outputs", s.get("prior_outputs"), "[]")
                    s["prior_outputs"] = []

                failure_field = f"steps.{step}.artifacts"
                if s.get("artifacts") is None:
                    _warn_default(f"steps.{step}.artifacts", "{}")
                    s["artifacts"] = {}
                elif not isinstance(s.get("artifacts"), dict):
                    _warn_invalid(f"steps.{step}.artifacts", s.get("artifacts"), "{}")
                    s["artifacts"] = {}

                if step == "E":
                    failure_field = "steps.E.agents"
                    agents = s.get("agents")
                    if agents is None:
                        _warn_default("steps.E.agents", "{}")
                        agents = {}
                    if not isinstance(agents, dict):
                        _warn_invalid("steps.E.agents", agents, "{}")
                        agents = {}
                    s["agents"] = agents
                data["steps"][step] = s
                s.setdefault("reused", False)
                s.setdefault("rebuilt", False)
                s.setdefault("degraded", False)
                s.setdefault("failed", False)

            failure_field = "requested_steps"
            if not data["requested_steps"] and sig is not None:
                data["requested_steps"] = list(sig.steps)

            return data
        except Exception:
            print(f"[manifest_init] failure_field={failure_field}", file=sys.stderr)
            raise

    # ------------------------------------------------------------------
    # Step status helpers
    # ------------------------------------------------------------------

    def _step_data(self, step: str) -> dict:
        return self._data["steps"].get(step.upper(), {})

    def step_status(self, step: str) -> str:
        return self._step_data(step).get("status", "pending")

    def can_reuse_step(self, step: str) -> bool:
        """Return True if manifest says this step completed previously.

        NOTE:
        This only answers the manifest-side question. Callers must still verify
        that the required artifacts exist before reusing the step.
        """
        s = self._step_data(step)
        return s.get("status") in ("complete", "reuse") and str(s.get("audit_status") or "") == "PASS"

    def mark_step(self, step: str, status: str) -> None:
        s = self._data["steps"].setdefault(step.upper(), {})
        s["status"] = status
        s["reused"] = status == "reuse"
        s["rebuilt"] = status == "complete"
        s["failed"] = status == "failed"
        if status in ("pending", "running"):
            s["reused"] = False
            s["rebuilt"] = False
            s["failed"] = False
            s["completed_at"] = None
            s["audit_status"] = None
        elif status == "failed":
            s["completed_at"] = None
        if status in ("complete", "reuse"):
            s["completed_at"] = _utcnow_iso()
        self._data["updated_at"] = _utcnow_iso()
        self.save()

    def mark_step_verified(
        self,
        step: str,
        status: str,
        *,
        artifacts_ok: bool,
        audit_status: Optional[str] = None,
        invalid_status: str = "pending",
    ) -> bool:
        """Persist a step status only when its artifact contract is satisfied.

        Returns True when the requested terminal status was accepted.
        When *status* is complete/reuse but artifacts are missing, the manifest is
        downgraded to *invalid_status* instead.
        """
        desired = str(status).strip().lower()
        accepted = True
        final_status = desired
        if desired in ("complete", "reuse") and not artifacts_ok:
            final_status = str(invalid_status).strip().lower() or "pending"
            accepted = False

        s = self._data["steps"].setdefault(step.upper(), {})
        s["status"] = final_status
        s["reused"] = final_status == "reuse"
        s["rebuilt"] = final_status == "complete"
        s["failed"] = final_status == "failed"
        if final_status in ("pending", "running"):
            s["reused"] = False
            s["rebuilt"] = False
            s["failed"] = False
            s["completed_at"] = None
            if audit_status is None:
                s["audit_status"] = None
        elif final_status == "failed":
            s["completed_at"] = None
        elif final_status in ("complete", "reuse"):
            s["completed_at"] = _utcnow_iso()
        if audit_status is not None:
            s["audit_status"] = str(audit_status)
        elif not accepted and final_status == "pending":
            s["audit_status"] = "FAIL"
        self._data["updated_at"] = _utcnow_iso()
        self.save()
        return accepted

    # ------------------------------------------------------------------
    # StepE per-agent helpers
    # ------------------------------------------------------------------

    def _agents_data(self) -> dict:
        return self._data["steps"].get("E", {}).get("agents", {})

    def stepe_agent_status(self, agent: str) -> str:
        return self._agents_data().get(agent, {}).get("status", "pending")

    def can_reuse_stepe_agent(self, agent: str) -> bool:
        info = self._agents_data().get(agent, {})
        return info.get("status") in ("complete", "reuse") and str(info.get("audit_status") or "") == "PASS"

    def mark_stepe_agent(self, agent: str, status: str) -> None:
        agents = self._data["steps"].setdefault("E", {}).setdefault("agents", {})
        agents.setdefault(agent, {})["status"] = status
        if status in ("pending", "running"):
            agents[agent]["completed_at"] = None
            agents[agent]["audit_status"] = None
        elif status == "failed":
            agents[agent]["completed_at"] = None
        elif status in ("complete", "reuse"):
            agents[agent]["completed_at"] = _utcnow_iso()
        self._data["updated_at"] = _utcnow_iso()
        self.save()

    def mark_stepe_agent_verified(
        self,
        agent: str,
        status: str,
        *,
        artifacts_ok: bool,
        audit_status: Optional[str] = None,
        invalid_status: str = "pending",
    ) -> bool:
        """Persist a StepE agent status only when its artifacts really exist."""
        desired = str(status).strip().lower()
        accepted = True
        final_status = desired
        if desired in ("complete", "reuse") and not artifacts_ok:
            final_status = str(invalid_status).strip().lower() or "pending"
            accepted = False

        agents = self._data["steps"].setdefault("E", {}).setdefault("agents", {})
        info = agents.setdefault(agent, {})
        info["status"] = final_status
        if final_status in ("pending", "running"):
            info["completed_at"] = None
            if audit_status is None:
                info["audit_status"] = None
        elif final_status == "failed":
            info["completed_at"] = None
        elif final_status in ("complete", "reuse"):
            info["completed_at"] = _utcnow_iso()
        if audit_status is not None:
            info["audit_status"] = str(audit_status)
        elif not accepted and final_status == "pending":
            info["audit_status"] = "FAIL"
        self._data["updated_at"] = _utcnow_iso()
        self.save()
        return accepted

    def mark_step_elapsed(self, step: str, elapsed_sec: float) -> None:
        s = self._data["steps"].setdefault(step.upper(), {})
        s["elapsed_sec"] = round(float(elapsed_sec), 3)
        self._data["updated_at"] = _utcnow_iso()
        self.save()

    def mark_step_audit(self, step: str, status: str) -> None:
        """Set audit_status for a step. status should be 'PASS', 'FAIL', or 'SKIP'."""
        s = self._data["steps"].setdefault(step.upper(), {})
        s["audit_status"] = str(status)
        self._data["updated_at"] = _utcnow_iso()
        self.save()

    def mark_stepe_agent_elapsed(self, agent: str, elapsed_sec: float) -> None:
        agents = self._data["steps"].setdefault("E", {}).setdefault("agents", {})
        agents.setdefault(agent, {})["elapsed_sec"] = round(float(elapsed_sec), 3)
        self._data["updated_at"] = _utcnow_iso()
        self.save()

    def mark_stepe_agent_audit(self, agent: str, status: str) -> None:
        """Set audit_status for a single StepE agent. status: 'PASS' / 'FAIL'."""
        agents = self._data["steps"].setdefault("E", {}).setdefault("agents", {})
        agents.setdefault(agent, {})["audit_status"] = str(status)
        self._data["updated_at"] = _utcnow_iso()
        self.save()

    def stepe_agent_audit_status(self, agent: str) -> Optional[str]:
        """Return audit_status for agent, or None if not yet set."""
        return self._agents_data().get(agent, {}).get("audit_status")

    def mark_source_output_root(self, path: str, run_id: Optional[str] = None) -> None:
        """Record the output_root path used by this pipeline run.

        When reuse is active, this is the prior run's output_root, not the new run_dir/output.
        Useful for lineage tracing and diagnostics.
        """
        self._data["source_output_root"] = str(path) if path else None
        if run_id is not None:
            self._data["run_id"] = str(run_id)
        self._data["updated_at"] = _utcnow_iso()
        self.save()

    def ensure_stepe_agents(self, all_agents: List[str]) -> None:
        """Ensure all_agents are represented in the manifest E.agents dict with stable order."""
        ordered_agents = list(dict.fromkeys(str(a).strip() for a in all_agents if str(a).strip()))
        agents = self._data["steps"].setdefault("E", {}).setdefault("agents", {})
        normalized: Dict[str, Dict[str, Any]] = {}
        for a in ordered_agents:
            normalized[a] = dict(agents.get(a) or {
                "status": "pending",
                "completed_at": None,
                "elapsed_sec": None,
                "audit_status": None,
            })
            normalized[a].setdefault("status", "pending")
            normalized[a].setdefault("completed_at", None)
            normalized[a].setdefault("elapsed_sec", None)
            normalized[a].setdefault("audit_status", None)

        for a, payload in agents.items():
            if a not in normalized:
                normalized[a] = payload

        self._data["steps"].setdefault("E", {})["agents"] = normalized
        self.save()

    def pending_stepe_agents(self, all_agents: List[str]) -> List[str]:
        """Return agents that are NOT yet complete (need to run)."""
        return [a for a in all_agents if not self.can_reuse_stepe_agent(a)]

    def complete_stepe_agents(self, all_agents: List[str]) -> List[str]:
        return [a for a in all_agents if self.can_reuse_stepe_agent(a)]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        path = self._output_root / _MANIFEST_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self._data, indent=2, ensure_ascii=False)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)

    # ------------------------------------------------------------------
    # Completed step count (for ranking during scan)
    # ------------------------------------------------------------------

    def completed_step_count(self) -> int:
        count = 0
        for step, info in self._data.get("steps", {}).items():
            if info.get("status") in ("complete", "reuse"):
                count += 1
        return count


# ---------------------------------------------------------------------------
# WORK_ROOT scanner
# ---------------------------------------------------------------------------

def find_latest_matching_run(scan_root: Path, sig: RunSignature) -> Optional[Path]:
    """Return canonical output root (scan_root is ignored for compatibility)."""
    candidate = resolve_canonical_output_root(sig.mode, sig.symbol, sig.test_start)
    return candidate if candidate.exists() else None


def find_latest_matching_stepa_simple_run(
    scan_root: Path,
    simple_sig: StepASimpleReuseSignature,
) -> Optional[Path]:
    """Return canonical output root usable for StepA simple reuse.

    scan_root is intentionally ignored to remove run_id-based scanning.
    """
    symbols = _normalize_symbols(simple_sig.symbols)
    if not symbols:
        return None
    output_root = resolve_canonical_output_root(simple_sig.mode, symbols[0], simple_sig.test_start)
    if not output_root.exists():
        return None
    if not all(check_step_artifacts("A", output_root, symbol=s, mode=simple_sig.mode) for s in symbols):
        return None
    return output_root


def diagnose_stepa_simple_reuse(simple_sig: StepASimpleReuseSignature) -> Dict[str, Any]:
    """Return detailed StepA simple-reuse diagnostics.

    reason values are constrained to:
    - missing_required_outputs
    - invalid_output_data
    - symbol_mismatch
    - mode_mismatch
    - split_mismatch
    - reuse
    """
    symbols = _normalize_symbols(simple_sig.symbols)
    symbol = symbols[0] if symbols else ""
    mode = str(simple_sig.mode)
    canonical_root = resolve_canonical_output_root(mode, symbol, simple_sig.test_start)
    checks: List[Dict[str, str]] = []

    required = [
        f"stepA/{mode}/stepA_prices_train_{symbol}.csv",
        f"stepA/{mode}/stepA_prices_test_{symbol}.csv",
        f"stepA/{mode}/stepA_periodic_train_{symbol}.csv",
        f"stepA/{mode}/stepA_periodic_test_{symbol}.csv",
        f"stepA/{mode}/stepA_tech_train_{symbol}.csv",
        f"stepA/{mode}/stepA_tech_test_{symbol}.csv",
        f"stepA/{mode}/stepA_split_summary_{symbol}.csv",
        f"stepA/{mode}/stepA_periodic_future_{symbol}.csv",
        f"stepA/{mode}/stepA_daily_manifest_{symbol}.csv",
        "split_summary.json",
    ]
    optional: List[str] = []

    for rel in required:
        p = canonical_root / rel
        checks.append({"path": rel, "status": "pass" if p.exists() else "fail", "required": "true"})
    for rel in optional:
        p = canonical_root / rel
        checks.append({"path": rel, "status": "pass" if p.exists() else "fail", "required": "false"})

    if not canonical_root.exists() or any(c["status"] == "fail" for c in checks if c["required"] == "true"):
        return {
            "matched": False,
            "reason": "missing_required_outputs",
            "canonical_output_root": str(canonical_root),
            "checks": checks,
            "stepa_reuse_path": str(canonical_root / "stepA" / mode),
            "evaluation_stepa_path": str(canonical_root / "stepA" / mode),
        }
    if not _validate_stepa_daily_manifest(canonical_root, symbol, mode):
        checks.append({"path": f"stepA/{mode}/stepA_daily_manifest_{symbol}.csv(manifest_paths)", "status": "fail", "required": "true"})
        return {
            "matched": False,
            "reason": "missing_required_outputs",
            "canonical_output_root": str(canonical_root),
            "checks": checks,
            "stepa_reuse_path": str(canonical_root / "stepA" / mode),
            "evaluation_stepa_path": str(canonical_root / "stepA" / mode),
        }

    summary = load_split_summary(canonical_root)
    if not isinstance(summary, dict):
        return {
            "matched": False,
            "reason": "invalid_output_data",
            "canonical_output_root": str(canonical_root),
            "checks": checks,
            "stepa_reuse_path": str(canonical_root / "stepA" / mode),
            "evaluation_stepa_path": str(canonical_root / "stepA" / mode),
        }
    if str(summary.get("symbol", "")).upper() != symbol.upper():
        reason = "symbol_mismatch"
    elif str(summary.get("mode", "")).lower() != mode.lower():
        reason = "mode_mismatch"
    elif not split_summary_matches(
        summary,
        symbol=symbol,
        mode=mode,
        test_start=str(simple_sig.test_start or ""),
        train_years=int(simple_sig.train_years),
        test_months=int(simple_sig.test_months),
    ):
        reason = "split_mismatch"
    else:
        prices_train = canonical_root / f"stepA/{mode}/stepA_prices_train_{symbol}.csv"
        prices_test = canonical_root / f"stepA/{mode}/stepA_prices_test_{symbol}.csv"
        csv_ok = _csv_valid(prices_train, ("Date", "Open", "High", "Low", "Close", "Volume")) and _csv_valid(
            prices_test, ("Date", "Open", "High", "Low", "Close", "Volume")
        )
        reason = "reuse" if csv_ok else "invalid_output_data"

    return {
        "matched": reason == "reuse",
        "reason": reason,
        "canonical_output_root": str(canonical_root),
        "checks": checks,
        "stepa_reuse_path": str(canonical_root / "stepA" / mode),
        "evaluation_stepa_path": str(canonical_root / "stepA" / mode),
    }
