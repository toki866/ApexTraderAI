from __future__ import annotations

from types import SimpleNamespace
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_core.services import step_f_service as sf_mod
from ai_core.services.step_f_service import StepFRouterConfig, StepFService


def _base_merged() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Date": "2024-01-01",
                "Split": "test",
                "regime_id": 1,
                "r_soxl": 0.01,
                "r_soxs": -0.01,
                "ratio_a1": 1.0,
                "ret_a1": 0.01,
                "ratio_a2": 0.5,
                "ret_a2": 0.02,
            }
        ]
    )


def test_reward_mode_branching_changes_reward_value() -> None:
    svc = StepFService(app_config=SimpleNamespace())
    merged = _base_merged()
    edge = pd.DataFrame([{"regime_id": 1, "agent": "a1", "IR": 1.0}, {"regime_id": 1, "agent": "a2", "IR": 0.0}])
    allow = pd.DataFrame([{"regime_id": 1, "allowed_agents": "a1|a2"}])

    cfg_legacy = StepFRouterConfig(agents="a1,a2", reward_mode="legacy", ema_alpha=1.0, softmax_beta=100.0, trade_cost_bps=0.0)
    cfg_basic = StepFRouterConfig(agents="a1,a2", reward_mode="profit_basic", ema_alpha=1.0, softmax_beta=100.0, lambda_switch=0.1, lambda_churn=0.1, trade_cost_bps=0.0)
    cfg_regret = StepFRouterConfig(agents="a1,a2", reward_mode="profit_regret", ema_alpha=1.0, softmax_beta=100.0, lambda_regret=1.0, lambda_switch=0.0, trade_cost_bps=0.0)

    legacy = svc._run_router_sim(merged, ["a1", "a2"], edge, allow, ["a1", "a2"], cfg_legacy)
    basic = svc._run_router_sim(merged, ["a1", "a2"], edge, allow, ["a1", "a2"], cfg_basic)
    regret = svc._run_router_sim(merged, ["a1", "a2"], edge, allow, ["a1", "a2"], cfg_regret)

    assert float(legacy.loc[0, "ret"]) > float(basic.loc[0, "ret"])
    assert float(regret.loc[0, "ret"]) <= float(legacy.loc[0, "ret"])
    assert "ret_best_expert" in regret.columns


def test_profit_regret_oracle_used_for_reward_only_not_state() -> None:
    svc = StepFService(app_config=SimpleNamespace())
    merged = _base_merged()
    edge = pd.DataFrame([{"regime_id": 1, "agent": "a1", "IR": 1.0}, {"regime_id": 1, "agent": "a2", "IR": 0.0}])
    allow = pd.DataFrame([{"regime_id": 1, "allowed_agents": "a1|a2"}])
    cfg_regret = StepFRouterConfig(agents="a1,a2", reward_mode="profit_regret", ema_alpha=1.0, softmax_beta=100.0, trade_cost_bps=0.0)

    out = svc._run_router_sim(merged, ["a1", "a2"], edge, allow, ["a1", "a2"], cfg_regret)
    assert "oracle_expert" not in out.columns
    assert "oracle_ret" not in out.columns
    assert "ret_best_expert" in out.columns



def test_resolve_reward_modes_defaults_for_compare() -> None:
    cfg = StepFRouterConfig(stepf_compare_reward_modes=True, stepf_reward_modes="")
    enabled, modes = StepFService._resolve_reward_modes(cfg)
    assert enabled is True
    assert modes == ["legacy", "profit_basic", "profit_regret", "profit_light_risk"]

def test_reward_mode_outputs_are_separated(tmp_path) -> None:
    cfg = StepFRouterConfig(output_root=str(tmp_path / "out"), agents="a1,a2", reward_mode="profit_basic")
    app_config = SimpleNamespace(stepF=cfg, output_root=str(tmp_path / "out"))
    svc = StepFService(app_config=app_config)

    sf_mod.hdbscan = object()
    sf_mod.hdbscan_prediction = object()

    svc._load_stepa_price_tech = lambda out_root, mode, symbol: pd.DataFrame({"Date": pd.to_datetime(["2024-01-01"]), "price_exec": [100.0]})  # type: ignore[assignment]
    svc._load_stepe_logs = lambda out_root, mode, symbol, agents: {  # type: ignore[assignment]
        "a1": pd.DataFrame({"Date": pd.to_datetime(["2024-01-01"]), "Split": ["test"], "ratio": [1.0], "stepE_ret_for_stats": [0.01]}),
        "a2": pd.DataFrame({"Date": pd.to_datetime(["2024-01-01"]), "Split": ["test"], "ratio": [0.5], "stepE_ret_for_stats": [0.02]}),
    }
    svc._build_phase2_state = lambda **kwargs: pd.DataFrame({"Date": pd.to_datetime(["2024-01-01"]), "regime_id": [1], "confidence": [1.0]})  # type: ignore[assignment]
    svc._build_regime_edge_table = lambda merged, agents, cfg: pd.DataFrame([{"regime_id": 1, "agent": "a1", "IR": 1.0}, {"regime_id": 1, "agent": "a2", "IR": 0.0}])  # type: ignore[assignment]
    svc._build_allowlist = lambda edge_table, agents, safe_set, cfg: pd.DataFrame([{"regime_id": 1, "allowed_agents": "a1|a2"}])  # type: ignore[assignment]
    svc.evaluate_final_outputs = staticmethod(lambda **kwargs: {"return_code": 0})  # type: ignore[assignment]

    date_range = SimpleNamespace(mode="sim", train_start="2023-01-01", train_end="2023-12-31", test_start="2024-01-01", test_end="2024-12-31")
    svc.run(date_range, symbol="SOXL", mode="sim")

    reward_base = tmp_path / "out" / "stepF" / "sim" / "reward_profit_basic"
    assert (reward_base / "stepF_equity_marl_SOXL.csv").exists()
    assert (reward_base / "stepF_daily_log_router_SOXL.csv").exists()


def test_compare_mode_runs_all_reward_modes(tmp_path) -> None:
    cfg = StepFRouterConfig(
        output_root=str(tmp_path / "out"),
        agents="a1,a2",
        reward_mode="legacy",
        stepf_compare_reward_modes=True,
    )
    app_config = SimpleNamespace(stepF=cfg, output_root=str(tmp_path / "out"))
    svc = StepFService(app_config=app_config)

    sf_mod.hdbscan = object()
    sf_mod.hdbscan_prediction = object()

    svc._load_stepa_price_tech = lambda out_root, mode, symbol: pd.DataFrame({"Date": pd.to_datetime(["2024-01-01"]), "price_exec": [100.0]})  # type: ignore[assignment]
    svc._load_stepe_logs = lambda out_root, mode, symbol, agents: {  # type: ignore[assignment]
        "a1": pd.DataFrame({"Date": pd.to_datetime(["2024-01-01"]), "Split": ["test"], "ratio": [1.0], "stepE_ret_for_stats": [0.01]}),
        "a2": pd.DataFrame({"Date": pd.to_datetime(["2024-01-01"]), "Split": ["test"], "ratio": [0.5], "stepE_ret_for_stats": [0.02]}),
    }
    svc._build_phase2_state = lambda **kwargs: pd.DataFrame({"Date": pd.to_datetime(["2024-01-01"]), "regime_id": [1], "confidence": [1.0]})  # type: ignore[assignment]
    svc._build_regime_edge_table = lambda merged, agents, cfg: pd.DataFrame([{"regime_id": 1, "agent": "a1", "IR": 1.0}, {"regime_id": 1, "agent": "a2", "IR": 0.0}])  # type: ignore[assignment]
    svc._build_allowlist = lambda edge_table, agents, safe_set, cfg: pd.DataFrame([{"regime_id": 1, "allowed_agents": "a1|a2"}])  # type: ignore[assignment]
    svc.evaluate_final_outputs = staticmethod(lambda **kwargs: {"return_code": 0})  # type: ignore[assignment]

    date_range = SimpleNamespace(mode="sim", train_start="2023-01-01", train_end="2023-12-31", test_start="2024-01-01", test_end="2024-12-31")
    svc.run(date_range, symbol="SOXL", mode="sim")

    base = tmp_path / "out" / "stepF" / "sim"
    for mode_name in ["legacy", "profit_basic", "profit_regret", "profit_light_risk"]:
        reward_base = base / f"reward_{mode_name}"
        assert (reward_base / "stepF_equity_marl_SOXL.csv").exists()
        assert (reward_base / "stepF_daily_log_router_SOXL.csv").exists()
        assert (reward_base / "stepF_daily_log_marl_SOXL.csv").exists()
        assert (reward_base / "stepF_summary_router_SOXL.json").exists()


def test_compare_mode_continues_on_single_mode_failure_and_writes_traceback(tmp_path, capsys) -> None:
    cfg = StepFRouterConfig(
        output_root=str(tmp_path / "out"),
        agents="a1,a2",
        reward_mode="legacy",
        stepf_compare_reward_modes=True,
    )
    app_config = SimpleNamespace(stepF=cfg, output_root=str(tmp_path / "out"))
    svc = StepFService(app_config=app_config)

    sf_mod.hdbscan = object()
    sf_mod.hdbscan_prediction = object()

    svc._load_stepa_price_tech = lambda out_root, mode, symbol: pd.DataFrame({"Date": pd.to_datetime(["2024-01-01"]), "price_exec": [100.0]})  # type: ignore[assignment]
    svc._load_stepe_logs = lambda out_root, mode, symbol, agents: {  # type: ignore[assignment]
        "a1": pd.DataFrame({"Date": pd.to_datetime(["2024-01-01"]), "Split": ["test"], "ratio": [1.0], "stepE_ret_for_stats": [0.01]}),
        "a2": pd.DataFrame({"Date": pd.to_datetime(["2024-01-01"]), "Split": ["test"], "ratio": [0.5], "stepE_ret_for_stats": [0.02]}),
    }
    svc._build_phase2_state = lambda **kwargs: pd.DataFrame({"Date": pd.to_datetime(["2024-01-01"]), "regime_id": [1], "confidence": [1.0]})  # type: ignore[assignment]
    svc._build_regime_edge_table = lambda merged, agents, cfg: pd.DataFrame([{"regime_id": 1, "agent": "a1", "IR": 1.0}, {"regime_id": 1, "agent": "a2", "IR": 0.0}])  # type: ignore[assignment]
    svc._build_allowlist = lambda edge_table, agents, safe_set, cfg: pd.DataFrame([{"regime_id": 1, "allowed_agents": "a1|a2"}])  # type: ignore[assignment]
    svc.evaluate_final_outputs = staticmethod(lambda **kwargs: {"return_code": 0})  # type: ignore[assignment]

    original_run_router = svc._run_router

    def _run_router_with_one_failure(cfg, date_range, symbol, mode, persist_primary_outputs=True, data_cutoff=""):
        if cfg.reward_mode == "profit_regret":
            raise RuntimeError("intentional mode failure")
        return original_run_router(cfg, date_range, symbol, mode, persist_primary_outputs=persist_primary_outputs, data_cutoff=data_cutoff)

    svc._run_router = _run_router_with_one_failure  # type: ignore[assignment]

    date_range = SimpleNamespace(mode="sim", train_start="2023-01-01", train_end="2023-12-31", test_start="2024-01-01", test_end="2024-12-31")
    svc.run(date_range, symbol="SOXL", mode="sim")

    out = capsys.readouterr().out
    assert "[STEPF_MULTI] mode_start=legacy" in out
    assert "[STEPF_MULTI] mode_success=legacy" in out
    assert "[STEPF_MULTI] mode_fail=profit_regret" in out
    assert "Traceback (most recent call last)" in out

    base = tmp_path / "out" / "stepF" / "sim"
    assert (base / "reward_legacy" / "stepF_equity_marl_SOXL.csv").exists()
    assert (base / "reward_profit_basic" / "stepF_equity_marl_SOXL.csv").exists()
    assert (base / "reward_profit_regret" / "stepF_traceback_SOXL.log").exists()
    assert (base / "stepF_multi_mode_summary_SOXL.json").exists()


def test_compare_mode_all_fail_raises_and_records_summary(tmp_path) -> None:
    cfg = StepFRouterConfig(
        output_root=str(tmp_path / "out"),
        agents="a1,a2",
        reward_mode="legacy",
        stepf_compare_reward_modes=True,
    )
    app_config = SimpleNamespace(stepF=cfg, output_root=str(tmp_path / "out"))
    svc = StepFService(app_config=app_config)

    sf_mod.hdbscan = object()
    sf_mod.hdbscan_prediction = object()

    def _always_fail(*args, **kwargs):
        raise ValueError("all modes failed")

    svc._run_router = _always_fail  # type: ignore[assignment]

    date_range = SimpleNamespace(mode="sim", train_start="2023-01-01", train_end="2023-12-31", test_start="2024-01-01", test_end="2024-12-31")
    try:
        svc.run(date_range, symbol="SOXL", mode="sim")
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "failed for all modes" in str(exc)

    base = tmp_path / "out" / "stepF" / "sim"
    summary_path = base / "stepF_multi_mode_summary_SOXL.json"
    assert summary_path.exists()
    summary = __import__('json').loads(summary_path.read_text(encoding='utf-8'))
    assert len(summary.get("records", [])) == 4
    assert all(r.get("status") == "FAIL" for r in summary.get("records", []))
