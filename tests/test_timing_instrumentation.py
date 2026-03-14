from __future__ import annotations

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
    )


def test_timing_enabled_creates_csv_and_jsonl(tmp_path: Path) -> None:
    timing = _mk_logger(tmp_path)

    assert timing.timings_csv_path.exists()
    assert timing.events_path.exists()


def test_all_steps_total_are_recorded(tmp_path: Path) -> None:
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


def test_agent_id_is_recorded_for_dprime_stepe_stepf(tmp_path: Path) -> None:
    timing = _mk_logger(tmp_path)

    with timing.stage("stepDPrimeRL.profile.loop", agent_id="dprime_all_features_h01"):
        pass
    with timing.stage("stepE.agent.total", agent_id="dprime_mid_h01"):
        pass
    with timing.stage("stepF.reward_mode.total", agent_id="profit_basic"):
        pass

    df = pd.read_csv(timing.timings_csv_path)
    ids = set(df["agent_id"].astype(str).tolist())
    assert "dprime_all_features_h01" in ids
    assert "dprime_mid_h01" in ids
    assert "profit_basic" in ids


def test_exception_still_emits_elapsed(tmp_path: Path) -> None:
    timing = _mk_logger(tmp_path)

    try:
        with timing.stage("stepE.agent.train", agent_id="dprime_bnf_h01"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    df = pd.read_csv(timing.timings_csv_path)
    hit = df[df["section"].astype(str) == "agent.train"]
    assert not hit.empty
    assert float(hit["elapsed_sec"].iloc[-1]) >= 0.0


def test_summary_csvs_are_generated(tmp_path: Path) -> None:
    timing = _mk_logger(tmp_path)

    with timing.stage("stepA.total"):
        pass
    with timing.stage("stepE.agent.train", agent_id="dprime_bnf_h01"):
        pass
    with timing.stage("stepE.agent.eval", agent_id="dprime_bnf_h01"):
        pass

    timing.write_summaries()

    assert timing.summary_step_csv_path.exists()
    assert timing.summary_agent_csv_path.exists()

    step_df = pd.read_csv(timing.summary_step_csv_path)
    agent_df = pd.read_csv(timing.summary_agent_csv_path)

    assert {"step", "section", "count", "elapsed_sec_sum", "elapsed_sec_mean", "elapsed_sec_max"}.issubset(step_df.columns)
    assert {"step", "section", "agent_id", "count", "elapsed_sec_sum", "elapsed_sec_mean", "elapsed_sec_max"}.issubset(agent_df.columns)
    assert "dprime_bnf_h01" in set(agent_df["agent_id"].astype(str).tolist())
