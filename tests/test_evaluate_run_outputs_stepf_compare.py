from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    p = Path("scripts") / "evaluate_run_outputs.py"
    spec = importlib.util.spec_from_file_location("evaluate_run_outputs", p)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_step_e(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_stepf_compare_generates_table_and_regret(tmp_path: Path):
    mod = _load_module()
    out = tmp_path / "output"
    stepe = out / "stepE" / "sim"
    stepf = out / "stepF" / "sim"
    stepe.mkdir(parents=True, exist_ok=True)
    stepf.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    # Fixed best expert across full period
    ret_best = [0.02] * 10
    ret_mid = [0.005] * 10
    # Oracle can exceed fixed best by alternating spike expert
    ret_oracle_helper = [-0.02, 0.05] * 5

    def mk_rows(rets):
        eq = 1.0
        rows = []
        for d, r in zip(dates, rets):
            eq *= 1.0 + r
            rows.append({"Date": d.strftime("%Y-%m-%d"), "Split": "test", "pos": 1.0, "ratio": 1.0, "ret": r, "equity": eq})
        return rows

    _write_step_e(stepe / "stepE_daily_log_dprime_bnf_h01_SOXL.csv", mk_rows(ret_best))
    _write_step_e(stepe / "stepE_daily_log_dprime_mid_h01_SOXL.csv", mk_rows(ret_mid))
    _write_step_e(stepe / "stepE_daily_log_dprime_oracle_h01_SOXL.csv", mk_rows(ret_oracle_helper))

    # add 7 more simple experts (10 total)
    for i in range(7):
        _write_step_e(stepe / f"stepE_daily_log_agent{i}_SOXL.csv", mk_rows([0.001] * 10))

    # Current StepF underperforms fixed best
    eq = 1.0
    frows = []
    rrows = []
    for d in dates:
        r = 0.005
        eq *= 1.0 + r
        frows.append({"Date": d.strftime("%Y-%m-%d"), "Split": "test", "ratio": 1.0, "ret": r, "equity": eq})
        rrows.append({"Date": d.strftime("%Y-%m-%d"), "Split": "test", "ratio": 1.0, "ret": r, "equity": eq, "w_dprime_bnf_h01": 0.3, "w_dprime_oracle_h01": 0.7})

    pd.DataFrame(frows).to_csv(stepf / "stepF_equity_marl_SOXL.csv", index=False)
    pd.DataFrame(rrows).to_csv(stepf / "stepF_daily_log_router_SOXL.csv", index=False)

    # Reward-mode outputs for horizontal comparison
    reward_basic = stepf / "reward_profit_basic"
    reward_regret = stepf / "reward_profit_regret"
    reward_light = stepf / "reward_profit_light_risk"
    for ddir, r in [(reward_basic, 0.007), (reward_regret, 0.009), (reward_light, 0.008)]:
        ddir.mkdir(parents=True, exist_ok=True)
        eq = 1.0
        eq_rows = []
        router_rows = []
        for d in dates:
            eq *= 1.0 + r
            eq_rows.append({"Date": d.strftime("%Y-%m-%d"), "Split": "test", "ratio": 1.0, "ret": r, "equity": eq})
            router_rows.append({"Date": d.strftime("%Y-%m-%d"), "Split": "test", "ratio": 1.0, "ret": r, "equity": eq, "w_dprime_bnf_h01": 0.4, "w_dprime_oracle_h01": 0.6})
        pd.DataFrame(eq_rows).to_csv(ddir / "stepF_equity_marl_SOXL.csv", index=False)
        pd.DataFrame(router_rows).to_csv(ddir / "stepF_daily_log_router_SOXL.csv", index=False)

    rep = mod.evaluate(str(out), "sim", "SOXL")
    cmp = rep["stepF_compare"]
    assert cmp["status"] == "OK"
    row = cmp["row"]
    assert row["fixed_best_expert"] == "dprime_bnf_h01"
    assert float(row["regret_vs_fixed_best"]) > 0
    assert float(row["regret_vs_oracle"]) > 0
    assert float(row["oracle_equity_multiple"]) > float(row["fixed_best_equity_multiple"])

    reward_cmp = cmp["stepF_reward_compare"]
    assert reward_cmp["status"] == "OK"
    names = [r["name"] for r in reward_cmp["rows"]]
    assert "current_stepf" in names
    assert "reward_legacy" in names
    assert "reward_profit_basic" in names
    assert "reward_profit_regret" in names
    assert "reward_profit_light_risk" in names

    report_dir = tmp_path / "eval"
    mod._write_eval_tables(rep, str(report_dir))
    cmp_csv = pd.read_csv(report_dir / "EVAL_TABLE_stepF_compare.csv")
    assert "regret_vs_oracle" in cmp_csv.columns
    reward_cmp_csv = pd.read_csv(report_dir / "EVAL_TABLE_stepF_reward_compare.csv")
    assert "name" in reward_cmp_csv.columns


def test_stepf_compare_warn_when_stepe_missing(tmp_path: Path):
    mod = _load_module()
    out = tmp_path / "output"
    stepf = out / "stepF" / "sim"
    stepf.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"Date": "2024-01-01", "Split": "test", "ratio": 0.1, "ret": 0.01, "equity": 1.01}]).to_csv(
        stepf / "stepF_equity_marl_SOXL.csv", index=False
    )

    rep = mod.evaluate(str(out), "sim", "SOXL")
    cmp = rep["stepF_compare"]
    assert cmp["status"] == "WARN"
    assert "skipped" in str(cmp.get("summary", "")).lower()
