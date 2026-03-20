from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_core.services.step_f_service import StepFRouterConfig
from tools import run_pipeline as rp


def test_stepf_compare_partial_failure_marks_manifest_failed(monkeypatch, tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    cfg = SimpleNamespace(
        stepF=StepFRouterConfig(
            output_root=str(output_root),
            agents="a1,a2",
            reward_mode="legacy",
            stepf_compare_reward_modes=True,
            stepf_reward_modes="legacy,profit_basic",
        ),
        stepE=[],
        output_root=str(output_root),
        data=SimpleNamespace(output_root=str(tmp_path / "data")),
    )

    monkeypatch.setattr(rp, "_get_app_config", lambda repo_root: cfg)
    monkeypatch.setattr(rp, "_apply_config_output_root", lambda app_config, resolved_output_root: app_config)
    monkeypatch.setattr(
        rp,
        "_build_date_range",
        lambda *args, **kwargs: SimpleNamespace(
            mode="sim",
            train_start="2023-01-01",
            train_end="2023-12-31",
            test_start="2024-01-01",
            test_end="2024-03-31",
        ),
    )
    monkeypatch.setattr(rp, "_extract_stepe_agents_from_config", lambda app_config: ["a1"])
    monkeypatch.setattr(rp, "_preflight_ticc_backend_if_needed", lambda **kwargs: None)
    monkeypatch.setattr(rp, "_prepare_missing_data_if_needed", lambda **kwargs: None)

    stepe_root = output_root / "stepE" / "sim"
    audit_root = output_root / "audit" / "sim"
    models_root = stepe_root / "models"
    stepe_root.mkdir(parents=True, exist_ok=True)
    audit_root.mkdir(parents=True, exist_ok=True)
    models_root.mkdir(parents=True, exist_ok=True)
    (stepe_root / "stepE_daily_log_a1_SOXL.csv").write_text(
        "Date,Split,ratio,reward_next,equity,pos,r_soxl_next,r_soxs_next,cost\n"
        "2024-01-01,test,1,0.01,1.01,1,0.01,-0.01,0.0\n",
        encoding="utf-8",
    )
    (stepe_root / "stepE_summary_a1_SOXL.json").write_text('{"audit_status":"PASS"}', encoding="utf-8")
    (audit_root / "stepE_audit_a1_SOXL.json").write_text('{"status":"PASS"}', encoding="utf-8")
    (models_root / "stepE_a1_SOXL.pt").write_text("model", encoding="utf-8")

    def _run_step_generic(step_letter, app_config, symbol, date_range, prev_results):
        assert step_letter == "F"
        mode_root = output_root / "stepF" / "sim"
        legacy = mode_root / "reward_legacy"
        failed = mode_root / "reward_profit_basic"
        legacy.mkdir(parents=True, exist_ok=True)
        failed.mkdir(parents=True, exist_ok=True)
        for fname in (
            "stepF_equity_marl_SOXL.csv",
            "stepF_daily_log_marl_SOXL.csv",
            "stepF_daily_log_router_SOXL.csv",
            "stepF_summary_router_SOXL.json",
            "stepF_audit_router_SOXL.json",
            "stepF_policy_compare_SOXL.json",
        ):
            (legacy / fname).write_text('{"ok": true}' if fname.endswith(".json") else "Date,Split,ratio,ret,equity\n2024-01-01,test,1,0.01,1.01\n", encoding="utf-8")
        (legacy / "status.json").write_text('{"status":"complete","required_artifacts_present":true,"publish_ready":false}', encoding="utf-8")
        (legacy / "artifacts_manifest.json").write_text('{"validation_passed":true}', encoding="utf-8")
        (failed / "status.json").write_text('{"status":"failed","required_artifacts_present":false,"publish_ready":false}', encoding="utf-8")
        (failed / "artifacts_manifest.json").write_text('{"validation_passed":false}', encoding="utf-8")
        (failed / "traceback.txt").write_text("traceback", encoding="utf-8")
        summary = {
            "compare_enabled": True,
            "reward_modes": ["legacy", "profit_basic"],
            "success_modes": ["legacy"],
            "failed_modes": ["profit_basic"],
            "reused_modes": [],
            "missing_outputs": ["reward_profit_basic/stepF_equity_marl_SOXL.csv"],
            "publish_completed": False,
            "records": [
                {"mode": "legacy", "status": "COMPLETE"},
                {"mode": "profit_basic", "status": "FAIL"},
            ],
        }
        (mode_root / "stepF_multi_mode_summary_SOXL.json").write_text(json.dumps(summary), encoding="utf-8")
        raise RuntimeError("simulated partial failure")

    monkeypatch.setattr(rp, "_run_step_generic", _run_step_generic)

    with pytest.raises(RuntimeError, match="simulated partial failure"):
        rp.main(
            [
                "--symbol",
                "SOXL",
                "--steps",
                "F",
                "--mode",
                "sim",
                "--output-root",
                str(output_root),
                "--test-start",
                "2024-01-01",
                "--auto-prepare-data",
                "0",
                "--stepf-compare-reward-modes",
                "1",
                "--stepf-reward-modes",
                "legacy,profit_basic",
            ]
        )

    manifest = json.loads((output_root / "run_manifest.json").read_text(encoding="utf-8"))
    stepf = manifest["steps"]["F"]
    assert stepf["status"] == "failed"
    assert stepf["reward_modes_completed"] == ["legacy"]
    assert stepf["reward_modes_failed"] == ["profit_basic"]
    assert stepf["publish_completed"] is False
    assert stepf["status"] != "running"
