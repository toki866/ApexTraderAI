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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
_SCHEMA_VERSION = 2


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


# ---------------------------------------------------------------------------
# Artifact presence checks
# ---------------------------------------------------------------------------

def check_step_artifacts(step: str, output_root: Path, symbol: str, mode: str) -> bool:
    """Return True if the key artifacts for *step* exist in output_root."""
    base = Path(output_root)
    step_upper = step.upper()

    if step_upper == "A":
        d = base / "stepA" / mode
        return (
            (d / f"stepA_prices_train_{symbol}.csv").exists()
            and (d / f"stepA_prices_test_{symbol}.csv").exists()
        )

    if step_upper == "B":
        d = base / "stepB" / mode
        return (d / f"stepB_pred_time_all_{symbol}.csv").exists()

    if step_upper == "C":
        d = base / "stepC" / mode
        if not d.exists():
            return False
        return any(d.glob(f"*{symbol}*.csv"))

    if step_upper == "DPRIME":
        d = base / "stepDprime" / mode
        for profile in _OFFICIAL_STEPE_AGENTS:
            if not (d / f"stepDprime_state_test_{profile}_{symbol}.csv").exists():
                return False
        return True

    if step_upper == "E":
        # Per-agent check is done separately; here we just verify the dir exists.
        d = base / "stepE" / mode
        return d.exists() and any(d.glob(f"stepE_daily_log_*_{symbol}.csv"))

    if step_upper == "F":
        d = base / "stepF" / mode
        return (
            (d / f"stepF_equity_marl_{symbol}.csv").exists()
            or (d / f"stepF_daily_log_marl_{symbol}.csv").exists()
        )

    return False


def check_stepe_agent_artifact(agent: str, output_root: Path, symbol: str, mode: str) -> bool:
    """Return True if the StepE daily_log artifact for *agent* exists."""
    p = Path(output_root) / "stepE" / mode / f"stepE_daily_log_{agent}_{symbol}.csv"
    return p.exists()


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
            data = existing
            data["reuse_enabled"] = reuse_enabled
            data["updated_at"] = _utcnow_iso()

        manifest = cls(Path(output_root), data)
        manifest.save()
        return manifest

    @classmethod
    def _fresh_data(cls, sig: RunSignature, reuse_enabled: bool, force_rebuild: bool) -> dict:
        now = _utcnow_iso()
        agent_list = list(sig.stepe_agents) if sig.stepe_agents else list(_OFFICIAL_STEPE_AGENTS)
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
                "A":      {"status": "pending", "completed_at": None, "elapsed_sec": None, "audit_status": None},
                "B":      {"status": "pending", "completed_at": None, "elapsed_sec": None, "audit_status": None},
                "C":      {"status": "pending", "completed_at": None, "elapsed_sec": None, "audit_status": None},
                "DPRIME": {"status": "pending", "completed_at": None, "elapsed_sec": None, "audit_status": None},
                "E": {
                    "status": "pending",
                    "completed_at": None,
                    "elapsed_sec": None,
                    "audit_status": None,
                    "agents": {
                        a: {"status": "pending", "completed_at": None, "elapsed_sec": None, "audit_status": None}
                        for a in agent_list
                    },
                },
                "F":      {"status": "pending", "completed_at": None, "elapsed_sec": None, "audit_status": None},
            },
        }

    @classmethod
    def _migrate_v1_to_v2(cls, data: dict) -> dict:
        """Add v2 fields (elapsed_sec, audit_status) to a v1 manifest in-place."""
        for step, info in data.get("steps", {}).items():
            if not isinstance(info, dict):
                continue
            info.setdefault("elapsed_sec", None)
            info.setdefault("audit_status", None)
            # StepE agents
            for _agent, ainfo in info.get("agents", {}).items():
                if isinstance(ainfo, dict):
                    ainfo.setdefault("elapsed_sec", None)
                    ainfo.setdefault("audit_status", None)
        data["schema_version"] = _SCHEMA_VERSION
        data.setdefault("source_output_root", None)
        data.setdefault("run_id", None)
        return data

    # ------------------------------------------------------------------
    # Step status helpers
    # ------------------------------------------------------------------

    def _step_data(self, step: str) -> dict:
        return self._data["steps"].get(step.upper(), {})

    def step_status(self, step: str) -> str:
        return self._step_data(step).get("status", "pending")

    def can_reuse_step(self, step: str) -> bool:
        """Return True if manifest says this step completed previously."""
        return self.step_status(step) in ("complete", "reuse")

    def mark_step(self, step: str, status: str) -> None:
        s = self._data["steps"].setdefault(step.upper(), {})
        s["status"] = status
        if status in ("complete", "reuse"):
            s["completed_at"] = _utcnow_iso()
        self._data["updated_at"] = _utcnow_iso()
        self.save()

    # ------------------------------------------------------------------
    # StepE per-agent helpers
    # ------------------------------------------------------------------

    def _agents_data(self) -> dict:
        return self._data["steps"].get("E", {}).get("agents", {})

    def stepe_agent_status(self, agent: str) -> str:
        return self._agents_data().get(agent, {}).get("status", "pending")

    def can_reuse_stepe_agent(self, agent: str) -> bool:
        return self.stepe_agent_status(agent) in ("complete", "reuse")

    def mark_stepe_agent(self, agent: str, status: str) -> None:
        agents = self._data["steps"].setdefault("E", {}).setdefault("agents", {})
        agents.setdefault(agent, {})["status"] = status
        if status in ("complete", "reuse"):
            agents[agent]["completed_at"] = _utcnow_iso()
        self._data["updated_at"] = _utcnow_iso()
        self.save()

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
        """Ensure all_agents are represented in the manifest E.agents dict."""
        agents = self._data["steps"].setdefault("E", {}).setdefault("agents", {})
        for a in all_agents:
            if a not in agents:
                agents[a] = {
                    "status": "pending",
                    "completed_at": None,
                    "elapsed_sec": None,
                    "audit_status": None,
                }
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
        path.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")

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
    """Scan <scan_root>/*/output/run_manifest.json for a matching prior run.

    Returns the output_root Path of the best match (most steps complete),
    or None if no match is found.

    "Match" = same signature_hash AND at least one step is "complete".
    """
    scan_root = Path(scan_root)
    if not scan_root.exists():
        return None

    target_hash = sig.stable_hash()
    best: Optional[Path] = None
    best_count = -1

    try:
        candidates = sorted(scan_root.iterdir(), key=lambda p: p.name, reverse=True)
    except Exception:
        return None

    for run_dir in candidates:
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "output" / _MANIFEST_FILENAME
        if not manifest_path.exists():
            continue
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("schema_version") != _SCHEMA_VERSION:
            continue
        if data.get("signature_hash") != target_hash:
            continue

        # Count completed steps
        count = sum(
            1 for v in data.get("steps", {}).values()
            if isinstance(v, dict) and v.get("status") in ("complete", "reuse")
        )
        if count > best_count:
            best_count = count
            best = run_dir / "output"

    return best if best_count > 0 else None
