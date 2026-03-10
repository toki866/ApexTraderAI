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

import pandas as pd

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


def build_canonical_output_root(base_output_root: Path, mode: str, symbol: str, test_start: str) -> Path:
    """Return canonical artifact root: output/<mode>/<symbol>/<test_start>/.

    base_output_root is the logical output base (typically repo/output).
    """
    ts = str(test_start or "").strip() or "unknown_test_start"
    return Path(base_output_root) / str(mode).strip().lower() / str(symbol).strip().upper() / ts


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
    """Find output root whose reuse_signature.json matches the requested key.

    Primary source: reuse_signature.json.
    Fallback: run_manifest.json-derived signature when reuse_signature.json is absent.
    """
    scan_root = Path(scan_root)
    if not scan_root.exists():
        return None

    target = reuse_sig.to_dict()
    try:
        candidates = sorted(scan_root.iterdir(), key=lambda p: p.name, reverse=True)
    except Exception:
        return None

    for run_dir in candidates:
        if not run_dir.is_dir():
            continue
        output_root = run_dir / "output"
        if not output_root.exists():
            continue
        candidate = _load_reuse_signature(output_root)
        if candidate is None:
            candidate = _extract_reuse_signature_from_manifest(output_root)
        if candidate is None:
            continue

        if (
            str(candidate.get("mode", "")) == target["mode"]
            and tuple(_normalize_symbols(candidate.get("symbols"))) == tuple(target["symbols"])
            and str(candidate.get("test_start_date", "")) == target["test_start_date"]
            and int(candidate.get("train_years", -1)) == target["train_years"]
            and int(candidate.get("test_months", -1)) == target["test_months"]
            and str(candidate.get("feature_signature", "")) == target["feature_signature"]
            and str(candidate.get("algorithm_signature", "")) == target["algorithm_signature"]
            and str(candidate.get("parameter_signature", "")) == target["parameter_signature"]
        ):
            return output_root
    return None


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

def check_step_artifacts(step: str, output_root: Path, symbol: str, mode: str) -> bool:
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
        )
        return all((d / name).exists() for name in required)

    if step_upper == "B":
        d = base / "stepB" / mode
        required = (
            f"stepB_pred_time_all_{symbol}.csv",
            f"stepB_pred_close_mamba_{symbol}.csv",
            f"stepB_pred_path_mamba_{symbol}.csv",
            f"stepB_pred_close_mamba_periodic_{symbol}.csv",
            f"stepB_pred_path_mamba_periodic_{symbol}.csv",
        )
        return all((d / name).exists() for name in required)

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
        for profile in _OFFICIAL_STEPE_AGENTS:
            # StepDPrime success is defined by state+embedding artifacts.
            # split_summary may be absent on some historical runs and should not
            # force a false-negative reuse/failure when core artifacts are complete.
            if not (d / f"stepDprime_state_test_{profile}_{symbol}.csv").exists():
                return False
            if not (emb / f"stepDprime_{profile}_{symbol}_embeddings_test.csv").exists():
                return False
        return True

    if step_upper == "E":
        d = base / "stepE" / mode
        model_dir = d / "models"
        return (
            d.exists()
            and any(d.glob(f"stepE_daily_log_*_{symbol}.csv"))
            and any(d.glob(f"stepE_summary_*_{symbol}.json"))
            and model_dir.exists()
            and (
                any(model_dir.glob(f"stepE_*_{symbol}.pt"))
                or any(model_dir.glob(f"stepE_*_{symbol}_ppo.zip"))
            )
        )

    if step_upper == "F":
        d = base / "stepF" / mode
        return (
            (d / f"stepF_equity_marl_{symbol}.csv").exists()
            and (d / f"stepF_daily_log_marl_{symbol}.csv").exists()
            and (d / f"stepF_daily_log_router_{symbol}.csv").exists()
            and (d / f"stepF_summary_router_{symbol}.json").exists()
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


def find_latest_matching_stepa_simple_run(
    scan_root: Path,
    simple_sig: StepASimpleReuseSignature,
) -> Optional[Path]:
    """Find latest prior run usable for StepA simple reuse.

    Candidate rules:
    - StepA artifacts must exist for every symbol in simple_sig.symbols.
    - If run_manifest.json is present *and readable*, require:
      steps.A.status in {complete, reuse} and signature keys match
      (mode, test_start, train_years, test_months).
    - If manifest is missing or broken, artifact-only fallback is allowed.
    """
    scan_root = Path(scan_root)
    if not scan_root.exists():
        return None

    try:
        candidates = sorted(scan_root.iterdir(), key=lambda p: p.name, reverse=True)
    except Exception:
        return None

    for run_dir in candidates:
        if not run_dir.is_dir():
            continue
        output_root = run_dir / "output"
        if not output_root.exists():
            continue

        # StepA artifacts must exist for every requested symbol.
        if not all(
            check_step_artifacts("A", output_root, symbol=s, mode=simple_sig.mode)
            for s in simple_sig.symbols
        ):
            continue

        manifest_path = output_root / _MANIFEST_FILENAME
        if not manifest_path.exists():
            return output_root

        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            # Broken manifest fallback is explicitly allowed.
            return output_root

        step_a_status = (
            data.get("steps", {}).get("A", {}).get("status")
            if isinstance(data, dict)
            else None
        )
        if step_a_status not in ("complete", "reuse"):
            continue

        sig = data.get("signature", {}) if isinstance(data, dict) else {}
        if not isinstance(sig, dict):
            continue

        if str(sig.get("mode", "")).strip().lower() != str(simple_sig.mode).strip().lower():
            continue
        if str(sig.get("test_start", "")).strip() != str(simple_sig.test_start).strip():
            continue
        if int(sig.get("train_years", -1)) != int(simple_sig.train_years):
            continue
        if int(sig.get("test_months", -1)) != int(simple_sig.test_months):
            continue

        # symbols may be stored as a single signature.symbol or list-like signature.symbols.
        # If absent/unreadable, artifact-only validation above remains authoritative.
        requested_symbols = _normalize_symbols(simple_sig.symbols)
        manifest_symbols = _normalize_symbols(sig.get("symbols"))
        if not manifest_symbols:
            manifest_symbols = _normalize_symbols(sig.get("symbol"))
        if manifest_symbols and manifest_symbols != requested_symbols:
            continue

        return output_root

    return None
