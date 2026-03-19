from __future__ import annotations

import json
from pathlib import Path

from tools.run_manifest import RunManifest, RunSignature


def _build_manifest(tmp_path: Path) -> RunManifest:
    sig = RunSignature(
        symbol="SOXL",
        mode="sim",
        test_start="2024-01-02",
        train_years=8,
        test_months=3,
        steps=("A", "B", "C", "DPRIME", "E", "F"),
        enable_mamba=True,
        enable_mamba_periodic=True,
        mamba_lookback=20,
        mamba_horizons=(1, 5, 20),
        stepe_agents=("dprime_all_features_h01",),
    )
    return RunManifest.load_or_create(tmp_path / "output", sig, reuse_enabled=True, force_rebuild=False)


def test_mark_step_verified_downgrades_to_pending_when_artifacts_missing(tmp_path: Path) -> None:
    manifest = _build_manifest(tmp_path)

    accepted = manifest.mark_step_verified("E", "complete", artifacts_ok=False, audit_status="FAIL", invalid_status="pending")

    data = json.loads((tmp_path / "output" / "run_manifest.json").read_text(encoding="utf-8"))
    assert accepted is False
    assert data["steps"]["E"]["status"] == "pending"
    assert data["steps"]["E"]["audit_status"] == "FAIL"
    assert data["steps"]["E"]["completed_at"] is None


def test_mark_stepe_agent_verified_requires_artifacts_for_reuse(tmp_path: Path) -> None:
    manifest = _build_manifest(tmp_path)

    accepted = manifest.mark_stepe_agent_verified(
        "dprime_all_features_h01",
        "reuse",
        artifacts_ok=False,
        audit_status="FAIL",
        invalid_status="pending",
    )

    data = json.loads((tmp_path / "output" / "run_manifest.json").read_text(encoding="utf-8"))
    agent = data["steps"]["E"]["agents"]["dprime_all_features_h01"]
    assert accepted is False
    assert agent["status"] == "pending"
    assert agent["audit_status"] == "FAIL"
    assert agent["completed_at"] is None
