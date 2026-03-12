from __future__ import annotations

import json

import pandas as pd

from ai_core.services.step_f_service import StepFService


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_stepf_final_outputs_success_with_split_missing_note(tmp_path):
    base = tmp_path / "output" / "stepF" / "sim"
    _write_csv(
        base / "stepF_equity_marl_SOXL.csv",
        [{"Date": "2024-01-01", "ratio": 0.1, "ret": 0.01, "equity": 1.01}],
    )
    _write_csv(base / "stepF_daily_log_router_SOXL.csv", [{"Date": "2024-01-01"}])
    _write_csv(base / "stepF_daily_log_marl_SOXL.csv", [{"Date": "2024-01-01"}])
    (base / "stepF_summary_router_SOXL.json").write_text(
        json.dumps({"status": "OK", "note": "Split missing: evaluated all rows as test"}),
        encoding="utf-8",
    )

    out = StepFService.evaluate_final_outputs(tmp_path / "output", "sim", "SOXL")

    assert out["return_code"] == 0
    assert out["final_status"] in {"warn", "complete"}
    assert any("split_warning:" in w for w in out["warnings"])


def test_stepf_final_outputs_failure_when_equity_missing(tmp_path):
    out = StepFService.evaluate_final_outputs(tmp_path / "output", "sim", "SOXL")
    assert out["return_code"] == 1
    assert "missing_equity_csv" in out["errors"]


def test_stepf_final_outputs_failure_when_equity_empty(tmp_path):
    base = tmp_path / "output" / "stepF" / "sim"
    base.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=["Date", "ratio", "ret", "equity"]).to_csv(base / "stepF_equity_marl_SOXL.csv", index=False)

    out = StepFService.evaluate_final_outputs(tmp_path / "output", "sim", "SOXL")

    assert out["return_code"] == 1
    assert "equity_csv_empty" in out["errors"]


def test_stepf_final_outputs_failure_when_required_columns_missing(tmp_path):
    base = tmp_path / "output" / "stepF" / "sim"
    _write_csv(base / "stepF_equity_marl_SOXL.csv", [{"Date": "2024-01-01", "ret": 0.01}])

    out = StepFService.evaluate_final_outputs(tmp_path / "output", "sim", "SOXL")

    assert out["return_code"] == 1
    assert any(str(e).startswith("equity_csv_missing_cols:") for e in out["errors"])
