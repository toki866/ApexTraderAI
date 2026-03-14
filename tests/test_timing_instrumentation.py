from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ai_core.utils.timing_logger import TimingLogger


def _mk_logger(tmp_path: Path) -> TimingLogger:
    return TimingLogger(
        output_root=tmp_path / "out",
        mode="sim",
        run_id="r1",
        branch_id="b1",
        execution_mode="sequential",
        retrain="off",
        enabled=True,
        clear=True,
        run_type="full_rebuild",
        symbol="SOXL",
    )


def test_timing_enabled_creates_csv_and_jsonl(tmp_path: Path) -> None:
    timing = _mk_logger(tmp_path)
    assert timing.timings_csv_path.exists()
    assert timing.events_path.exists()


def test_all_steps_total_are_recorded_with_condition_columns(tmp_path: Path) -> None:
    timing = _mk_logger(tmp_path)

    for stage in (
        "stepA.total",
        "stepB.total",
        "stepC.total",
        "stepDPrime.total",
        "stepE.total",
        "stepF.total",
    ):
        with timing.stage(stage):
            pass

    df = pd.read_csv(timing.timings_csv_path)
    got = set(df["step"].astype(str).tolist())
    assert {"StepA", "StepB", "StepC", "StepDPrime", "StepE", "StepF"}.issubset(got)
    for col in [
        "run_type",
        "status",
        "symbol",
        "agent_kind",
        "reward_mode",
        "profile_name",
        "expert_name",
        "fallback_used",
        "reused_steps",
        "skipped",
        "critical_path_sec",
    ]:
        assert col in df.columns


def test_agent_kind_for_dprime_stepe_stepf(tmp_path: Path) -> None:
    timing = _mk_logger(tmp_path)

    with timing.stage("stepDPrimeRL.profile.loop", agent_id="dprime_all_features_h01", meta={"agent_kind": "profile", "profile_name": "dprime_all_features_h01"}):
        pass
    with timing.stage("stepE.agent.total", agent_id="dprime_mid_h01", meta={"agent_kind": "expert", "expert_name": "dprime_mid_h01"}):
        pass
    with timing.stage("stepF.reward_mode.total", agent_id="profit_basic", meta={"agent_kind": "reward_mode", "reward_mode": "profit_basic"}):
        pass

    df = pd.read_csv(timing.timings_csv_path)
    assert ((df["agent_id"] == "dprime_all_features_h01") & (df["agent_kind"] == "profile")).any()
    assert ((df["agent_id"] == "dprime_mid_h01") & (df["agent_kind"] == "expert")).any()
    assert ((df["agent_id"] == "profit_basic") & (df["agent_kind"] == "reward_mode")).any()


def test_failure_and_skipped_status_are_recorded(tmp_path: Path) -> None:
    timing = _mk_logger(tmp_path)

    try:
        with timing.stage("stepE.agent.train", agent_id="dprime_bnf_h01"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    timing.emit_instant(stage="stepB.total", status="skipped", meta={"skipped": True})

    df = pd.read_csv(timing.timings_csv_path)
    assert (df["status"] == "fail").any()
    assert (df["status"] == "skipped").any()


def test_summary_csvs_are_generated_with_p95_and_budgets(tmp_path: Path) -> None:
    timing = _mk_logger(tmp_path)

    with timing.stage("stepA.total"):
        pass
    with timing.stage("stepE.agent.train", agent_id="dprime_bnf_h01", meta={"agent_kind": "expert", "expert_name": "dprime_bnf_h01"}):
        pass
    with timing.stage("stepE.agent.eval", agent_id="dprime_bnf_h01", meta={"agent_kind": "expert", "expert_name": "dprime_bnf_h01"}):
        pass
    with timing.stage("stepF.reward_mode.total", agent_id="profit_basic", meta={"agent_kind": "reward_mode", "reward_mode": "profit_basic", "fallback_used": True}):
        pass

    timing.write_summaries()

    assert timing.summary_step_csv_path.exists()
    assert timing.summary_agent_csv_path.exists()
    assert timing.summary_branch_budget_csv_path.exists()
    assert timing.summary_live_start_budget_csv_path.exists()

    step_df = pd.read_csv(timing.summary_step_csv_path)
    agent_df = pd.read_csv(timing.summary_agent_csv_path)
    branch_df = pd.read_csv(timing.summary_branch_budget_csv_path)
    live_df = pd.read_csv(timing.summary_live_start_budget_csv_path)

    assert "elapsed_sec_p95" in step_df.columns
    assert "elapsed_sec_p95" in agent_df.columns
    assert "elapsed_p95_sec" in branch_df.columns
    assert "recommended_start_offset_sec" in live_df.columns


def test_events_meta_contains_new_context(tmp_path: Path) -> None:
    timing = _mk_logger(tmp_path)
    with timing.stage("stepF.reward_mode.total", agent_id="profit_basic", meta={"agent_kind": "reward_mode", "reward_mode": "profit_basic", "fallback_used": True}):
        pass

    lines = timing.events_path.read_text(encoding="utf-8").strip().splitlines()
    event = json.loads(lines[-1])
    meta = event.get("meta", {})
    for key in ["symbol", "run_type", "status", "agent_kind", "reward_mode", "profile_name", "expert_name", "fallback_used", "reused_steps", "skipped"]:
        assert key in meta
