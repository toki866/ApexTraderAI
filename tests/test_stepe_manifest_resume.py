from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import tools.run_manifest as rm
import tools.run_pipeline as rp
from tools.run_manifest import RunManifest, RunSignature, reconcile_stepe_manifest_from_artifacts


def _build_manifest(output_root: Path, agents: list[str]) -> RunManifest:
    sig = RunSignature(
        symbol="SOXL",
        mode="sim",
        test_start="2024-01-02",
        train_years=8,
        test_months=3,
        steps=("A", "B", "C", "DPRIME", "E", "F"),
        enable_mamba=False,
        enable_mamba_periodic=False,
        mamba_lookback=None,
        mamba_horizons=(),
        stepe_agents=tuple(agents),
    )
    manifest = RunManifest.load_or_create(output_root, sig, reuse_enabled=True, force_rebuild=False)
    manifest.ensure_stepe_agents(agents)
    for agent in agents:
        manifest.mark_stepe_agent_verified(agent, "complete", artifacts_ok=False, audit_status="FAIL", invalid_status="pending")
    manifest.mark_step_verified("E", "complete", artifacts_ok=False, audit_status="FAIL", invalid_status="pending")
    return manifest


def _write_stepe_artifacts(output_root: Path, agents: list[str], *, symbol: str = "SOXL", mode: str = "sim") -> None:
    step_e_root = output_root / "stepE" / mode
    model_root = step_e_root / "models"
    audit_root = output_root / "audit" / mode
    model_root.mkdir(parents=True, exist_ok=True)
    audit_root.mkdir(parents=True, exist_ok=True)
    for idx, agent in enumerate(agents, start=1):
        (step_e_root / f"stepE_daily_log_{agent}_{symbol}.csv").write_text(
            "Date,Split,ret,reward_next,equity,pos,r_soxl_next,r_soxs_next,cost\n"
            f"2024-01-{idx:02d},test,0.0,0.0,1.0,0.0,0.0,0.0,0.0\n",
            encoding="utf-8",
        )
        (step_e_root / f"stepE_equity_{agent}_{symbol}.csv").write_text(
            "Date,pos,ret,equity,reward_next\n"
            f"2024-01-{idx:02d},0.0,0.0,1.0,0.0\n",
            encoding="utf-8",
        )
        (step_e_root / f"stepE_summary_{agent}_{symbol}.json").write_text(
            json.dumps({"agent": agent, "status": "PASS", "audit_status": "PASS"}),
            encoding="utf-8",
        )
        (audit_root / f"stepE_audit_{agent}_{symbol}.json").write_text(
            json.dumps({"agent": agent, "status": "PASS", "audit_status": "PASS"}),
            encoding="utf-8",
        )
        (model_root / f"stepE_{agent}_{symbol}.pt").write_text("model", encoding="utf-8")


def test_reconcile_stepe_manifest_self_heals_stale_manifest(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    agents = ["agent_a", "agent_b"]
    manifest = _build_manifest(output_root, agents)
    _write_stepe_artifacts(output_root, agents)

    result = reconcile_stepe_manifest_from_artifacts(
        agents,
        output_root=output_root,
        mode="sim",
        symbol="SOXL",
        manifest=manifest,
    )

    data = json.loads((output_root / "run_manifest.json").read_text(encoding="utf-8"))
    assert result["complete_agents"] == agents
    assert result["missing_agents"] == []
    assert result["all_complete"] is True
    assert data["steps"]["E"]["status"] == "complete"
    assert data["steps"]["E"]["audit_status"] == "PASS"
    for agent in agents:
        assert data["steps"]["E"]["agents"][agent]["status"] == "complete"
        assert data["steps"]["E"]["agents"][agent]["audit_status"] == "PASS"


def test_main_reuse_resume_advances_from_stepe_to_stepf(tmp_path: Path, monkeypatch, capsys) -> None:
    output_root = tmp_path / "output"
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    for ticker in ("SOXL", "SOXS"):
        (data_root / f"prices_{ticker}.csv").write_text("Date,Close\n2024-01-02,100\n", encoding="utf-8")
    agents = ["agent_a", "agent_b"]
    manifest = _build_manifest(output_root, agents)
    _write_stepe_artifacts(output_root, agents)

    original_check_step_artifacts = rm.check_step_artifacts
    stepf_calls: list[str] = []

    def _fake_check_step_artifacts(step: str, output_root_arg: Path, symbol: str, mode: str, **kwargs):
        step_upper = str(step).upper()
        if step_upper in {"A", "B", "C", "DPRIME"}:
            return True
        if step_upper == "F":
            step_f_root = Path(output_root_arg) / "stepF" / mode
            return (
                (step_f_root / f"stepF_equity_marl_{symbol}.csv").exists()
                and (step_f_root / f"stepF_daily_log_marl_{symbol}.csv").exists()
                and (step_f_root / f"stepF_daily_log_router_{symbol}.csv").exists()
                and (step_f_root / f"stepF_summary_router_{symbol}.json").exists()
            )
        return original_check_step_artifacts(step, output_root_arg, symbol, mode, **kwargs)

    def _fake_get_app_config(repo_root: Path):
        return SimpleNamespace(
            output_root=str(output_root),
            data=SimpleNamespace(output_root=str(output_root), data_root=str(data_root)),
            stepE=[SimpleNamespace(agent=agent, output_root=str(output_root), max_parallel_agents=1, policy_kind="ppo") for agent in agents],
            stepF=SimpleNamespace(retrain="off", reward_mode="legacy", stepf_compare_reward_modes=False, stepf_reward_modes=""),
        )

    def _fake_date_range(*args, **kwargs):
        return SimpleNamespace(
            mode="sim",
            train_start="2023-01-01",
            train_end="2023-12-31",
            test_start="2024-01-02",
            test_end="2024-03-29",
            future_end="2024-03-29",
        )

    def _fake_run_step_generic(step_letter: str, app_config, symbol: str, date_range, prev_results):
        assert step_letter == "F"
        stepf_calls.append(step_letter)
        step_f_root = output_root / "stepF" / "sim"
        step_f_root.mkdir(parents=True, exist_ok=True)
        for name in ("router", "marl"):
            (step_f_root / f"stepF_daily_log_{name}_{symbol}.csv").write_text(
                "Date,Split,ratio,ret,cost,equity,r_soxl,r_soxs\n2024-01-02,test,1.0,0.0,0.0,1.0,0.0,0.0\n",
                encoding="utf-8",
            )
        (step_f_root / f"stepF_equity_marl_{symbol}.csv").write_text(
            "Date,ratio,ret,equity\n2024-01-02,1.0,0.0,1.0\n",
            encoding="utf-8",
        )
        (step_f_root / f"stepF_summary_router_{symbol}.json").write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
        return {"status": "ok"}

    monkeypatch.setattr(rm, "check_step_artifacts", _fake_check_step_artifacts)
    monkeypatch.setattr(rp, "_get_app_config", _fake_get_app_config)
    monkeypatch.setattr(rp, "_build_date_range", _fake_date_range)
    monkeypatch.setattr(rp, "_preflight_ticc_backend_if_needed", lambda steps: None)
    monkeypatch.setattr(rp, "_run_step_generic", _fake_run_step_generic)
    monkeypatch.setattr(rp, "_sync_root_split_summary_from_stepa", lambda **kwargs: None)
    monkeypatch.setattr(rp, "_check_and_repair_split_summary_before_stepb", lambda **kwargs: None)
    monkeypatch.setattr(rp, "validate_step_e_agent", lambda *args, **kwargs: [])
    monkeypatch.setattr(rp, "validate_step_f", lambda *args, **kwargs: [])
    monkeypatch.setattr(rp, "audit_stepf_now", lambda *args, **kwargs: {"router": {"status": "PASS"}, "marl": {"status": "PASS"}})
    monkeypatch.setattr(rp, "_run_leak_audits", lambda **kwargs: None)

    from ai_core.services.step_f_service import StepFService

    monkeypatch.setattr(
        StepFService,
        "evaluate_final_outputs",
        staticmethod(lambda **kwargs: {"return_code": 0, "errors": []}),
    )

    exit_code = rp.main(
        [
            "--symbol",
            "SOXL",
            "--steps",
            "A,B,C,DPRIME,E,F",
            "--test-start",
            "2024-01-02",
            "--train-years",
            "8",
            "--test-months",
            "3",
            "--mode",
            "sim",
            "--output-root",
            str(output_root),
            "--data-dir",
            str(data_root),
            "--auto-prepare-data",
            "0",
            "--timing",
            "1",
            "--reuse-output",
            "1",
            "--stepf-compare-reward-modes",
            "0",
            "--stepe-agents",
            ",".join(agents),
        ]
    )

    captured = capsys.readouterr()
    data = json.loads((output_root / "run_manifest.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert stepf_calls == ["F"]
    assert "StepE partial completion" not in captured.out
    assert "StepE partial completion" not in captured.err
    assert data["steps"]["E"]["status"] in {"complete", "reuse"}
    assert data["steps"]["E"]["audit_status"] == "PASS"
    assert data["steps"]["F"]["status"] == "complete"
    for agent in agents:
        assert data["steps"]["E"]["agents"][agent]["audit_status"] == "PASS"
        assert data["steps"]["E"]["agents"][agent]["status"] in {"complete", "reuse"}
