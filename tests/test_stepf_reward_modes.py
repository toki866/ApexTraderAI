from __future__ import annotations

import json
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


def _build_stubbed_stepf_service(tmp_path: Path, *, compare: bool = True, reward_modes: str = "", reward_mode: str = "legacy") -> tuple[StepFService, SimpleNamespace]:
    cfg = StepFRouterConfig(
        output_root=str(tmp_path / "out"),
        agents="a1,a2",
        reward_mode=reward_mode,
        stepf_compare_reward_modes=compare,
        stepf_reward_modes=reward_modes,
    )
    app_config = SimpleNamespace(stepF=cfg, output_root=str(tmp_path / "out"))
    svc = StepFService(app_config=app_config)
    svc._load_stepa_price_tech = lambda out_root, mode, symbol: pd.DataFrame({"Date": pd.to_datetime(["2024-01-01"]), "price_exec": [100.0]})  # type: ignore[assignment]
    svc._load_stepe_logs = lambda step_e_root, symbol, agents: {  # type: ignore[assignment]
        "a1": pd.DataFrame({"Date": pd.to_datetime(["2024-01-01"]), "Split": ["test"], "ratio": [1.0], "stepE_ret_for_stats": [0.01]}),
        "a2": pd.DataFrame({"Date": pd.to_datetime(["2024-01-01"]), "Split": ["test"], "ratio": [0.5], "stepE_ret_for_stats": [0.02]}),
    }
    svc._build_phase2_state = lambda **kwargs: pd.DataFrame({"Date": pd.to_datetime(["2024-01-01"]), "regime_id": [1], "confidence": [1.0]})  # type: ignore[assignment]
    svc._build_regime_edge_table = lambda merged, agents, cfg: pd.DataFrame([{"regime_id": 1, "agent": "a1", "IR": 1.0}, {"regime_id": 1, "agent": "a2", "IR": 0.0}])  # type: ignore[assignment]
    svc._build_allowlist = lambda edge_table, agents, safe_set, cfg: pd.DataFrame([{"regime_id": 1, "allowed_agents": "a1|a2"}])  # type: ignore[assignment]
    svc.evaluate_final_outputs = staticmethod(lambda **kwargs: {"return_code": 0})  # type: ignore[assignment]
    date_range = SimpleNamespace(mode="sim", train_start="2023-01-01", train_end="2023-12-31", test_start="2024-01-01", test_end="2024-12-31")
    return svc, date_range


def test_reward_mode_branching_changes_reward_value() -> None:
    svc = StepFService(app_config=SimpleNamespace())
    merged = _base_merged()
    edge = pd.DataFrame([{"regime_id": 1, "agent": "a1", "IR": 1.0}, {"regime_id": 1, "agent": "a2", "IR": 0.0}])
    allow = pd.DataFrame([{"regime_id": 1, "allowed_agents": "a1|a2"}])

    cfg_legacy = StepFRouterConfig(agents="a1,a2", reward_mode="legacy", ema_alpha=1.0, softmax_beta=100.0, trade_cost_bps=0.0)
    cfg_basic = StepFRouterConfig(agents="a1,a2", reward_mode="profit_basic", ema_alpha=1.0, softmax_beta=100.0, lambda_switch=0.1, lambda_churn=0.1, trade_cost_bps=0.0)
    cfg_regret = StepFRouterConfig(agents="a1,a2", reward_mode="profit_regret", ema_alpha=1.0, softmax_beta=100.0, lambda_regret=1.0, lambda_switch=0.0, trade_cost_bps=0.0)

    legacy = svc._run_router_sim(merged, ["a1", "a2"], edge, allow, ["a1", "a2"], cfg_legacy, context_profiles={}, device_name="cpu")
    basic = svc._run_router_sim(merged, ["a1", "a2"], edge, allow, ["a1", "a2"], cfg_basic, context_profiles={}, device_name="cpu")
    regret = svc._run_router_sim(merged, ["a1", "a2"], edge, allow, ["a1", "a2"], cfg_regret, context_profiles={}, device_name="cpu")

    assert float(legacy.loc[0, "reward"]) > float(basic.loc[0, "reward"])
    assert float(regret.loc[0, "reward"]) <= float(legacy.loc[0, "reward"])
    assert "ret_best_expert" in regret.columns


def test_profit_regret_oracle_used_for_reward_only_not_state() -> None:
    svc = StepFService(app_config=SimpleNamespace())
    merged = _base_merged()
    edge = pd.DataFrame([{"regime_id": 1, "agent": "a1", "IR": 1.0}, {"regime_id": 1, "agent": "a2", "IR": 0.0}])
    allow = pd.DataFrame([{"regime_id": 1, "allowed_agents": "a1|a2"}])
    cfg_regret = StepFRouterConfig(agents="a1,a2", reward_mode="profit_regret", ema_alpha=1.0, softmax_beta=100.0, trade_cost_bps=0.0)

    out = svc._run_router_sim(merged, ["a1", "a2"], edge, allow, ["a1", "a2"], cfg_regret, context_profiles={}, device_name="cpu")
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

    svc._load_stepa_price_tech = lambda out_root, mode, symbol: pd.DataFrame({"Date": pd.to_datetime(["2024-01-01"]), "price_exec": [100.0]})  # type: ignore[assignment]
    svc._load_stepe_logs = lambda step_e_root, symbol, agents: {  # type: ignore[assignment]
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
    svc, date_range = _build_stubbed_stepf_service(tmp_path)
    svc.run(date_range, symbol="SOXL", mode="sim")

    base = tmp_path / "out" / "stepF" / "sim"
    for mode_name in ["legacy", "profit_basic", "profit_regret", "profit_light_risk"]:
        reward_base = base / f"reward_{mode_name}"
        assert (reward_base / "stepF_equity_marl_SOXL.csv").exists()
        assert (reward_base / "stepF_daily_log_router_SOXL.csv").exists()
        assert (reward_base / "stepF_daily_log_marl_SOXL.csv").exists()
        assert (reward_base / "stepF_summary_router_SOXL.json").exists()
        assert json.loads((reward_base / "status.json").read_text(encoding="utf-8"))["status"] == "complete"
        assert json.loads((reward_base / "artifacts_manifest.json").read_text(encoding="utf-8"))["validation_passed"] is True
    assert (base / "stepF_compare_reward_modes_SOXL.json").exists()
    assert (base / "stepF_best_reward_mode_SOXL.json").exists()


def test_compare_mode_stops_on_single_mode_failure_and_writes_traceback(tmp_path, capsys) -> None:
    svc, date_range = _build_stubbed_stepf_service(tmp_path)

    original_run_router = svc._run_router

    def _run_router_with_one_failure(cfg, date_range, symbol, mode, persist_primary_outputs=True, data_cutoff=""):
        if cfg.reward_mode == "profit_basic":
            raise RuntimeError("intentional mode failure")
        return original_run_router(cfg, date_range, symbol, mode, persist_primary_outputs=persist_primary_outputs, data_cutoff=data_cutoff)

    svc._run_router = _run_router_with_one_failure  # type: ignore[assignment]

    try:
        svc.run(date_range, symbol="SOXL", mode="sim")
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "intentional mode failure" in str(exc)

    out = capsys.readouterr().out
    assert "[STEPF_MULTI] mode_start=legacy" in out
    assert "[STEPF_MULTI] mode_success=legacy" in out
    assert "[STEPF_MULTI] mode_fail=profit_basic" in out
    assert "[ONE_TAP][STEPF_MULTI] mode_fail_traceback_begin=profit_basic" in out
    assert "Traceback (most recent call last)" in out

    base = tmp_path / "out" / "stepF" / "sim"
    assert (base / "reward_legacy" / "stepF_equity_marl_SOXL.csv").exists()
    assert json.loads((base / "reward_legacy" / "status.json").read_text(encoding="utf-8"))["status"] == "complete"
    failed_status = json.loads((base / "reward_profit_basic" / "status.json").read_text(encoding="utf-8"))
    assert failed_status["status"] == "failed"
    assert (base / "reward_profit_basic" / "traceback.txt").exists()
    assert not (base / "reward_profit_regret" / "stepF_equity_marl_SOXL.csv").exists()
    assert (base / "stepF_multi_mode_summary_SOXL.json").exists()
    assert not (base / "stepF_equity_marl_SOXL.csv").exists()
    assert not (base / "stepF_compare_reward_modes_SOXL.json").exists()


def test_compare_mode_all_fail_raises_and_records_summary(tmp_path) -> None:
    svc, date_range = _build_stubbed_stepf_service(tmp_path)

    def _always_fail(*args, **kwargs):
        raise ValueError("all modes failed")

    svc._run_router = _always_fail  # type: ignore[assignment]

    try:
        svc.run(date_range, symbol="SOXL", mode="sim")
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "failed for all modes" in str(exc)

    base = tmp_path / "out" / "stepF" / "sim"
    summary_path = base / "stepF_multi_mode_summary_SOXL.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    assert len(summary.get("records", [])) == 1
    assert all(r.get("status") == "FAIL" for r in summary.get("records", []))
    assert summary.get("success_modes") == []
    assert len(summary.get("failed_modes", [])) == 1
    assert len(summary.get("missing_outputs", [])) == 1


def test_compare_mode_first_failure_does_not_publish_primary_outputs(tmp_path) -> None:
    svc, date_range = _build_stubbed_stepf_service(tmp_path, reward_modes="profit_regret,legacy")

    original_run_router = svc._run_router

    def _run_router_with_first_failure(cfg, date_range, symbol, mode, persist_primary_outputs=True, data_cutoff=""):
        if cfg.reward_mode == "profit_regret":
            raise RuntimeError("intentional first failure")
        return original_run_router(cfg, date_range, symbol, mode, persist_primary_outputs=persist_primary_outputs, data_cutoff=data_cutoff)

    svc._run_router = _run_router_with_first_failure  # type: ignore[assignment]

    try:
        svc.run(date_range, symbol="SOXL", mode="sim")
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass

    base = tmp_path / "out" / "stepF" / "sim"
    assert not (base / "stepF_equity_marl_SOXL.csv").exists()
    assert not (base / "reward_legacy" / "stepF_equity_marl_SOXL.csv").exists()
    assert (base / "reward_profit_regret" / "traceback.txt").exists()


def test_stepf_rerun_reuses_completed_reward_mode_only(tmp_path) -> None:
    svc, date_range = _build_stubbed_stepf_service(tmp_path, reward_modes="legacy,profit_basic")
    call_counts: dict[str, int] = {}
    original_run_router = svc._run_router

    def _run_router_with_retry(cfg, date_range, symbol, mode, persist_primary_outputs=True, data_cutoff=""):
        call_counts[cfg.reward_mode] = call_counts.get(cfg.reward_mode, 0) + 1
        if cfg.reward_mode == "profit_basic" and call_counts[cfg.reward_mode] == 1:
            raise RuntimeError("profit_basic first-run failure")
        return original_run_router(cfg, date_range, symbol, mode, persist_primary_outputs=persist_primary_outputs, data_cutoff=data_cutoff)

    svc._run_router = _run_router_with_retry  # type: ignore[assignment]

    try:
        svc.run(date_range, symbol="SOXL", mode="sim")
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass

    svc.run(date_range, symbol="SOXL", mode="sim")

    assert call_counts["legacy"] == 1
    assert call_counts["profit_basic"] == 2
    base = tmp_path / "out" / "stepF" / "sim"
    summary = json.loads((base / "stepF_multi_mode_summary_SOXL.json").read_text(encoding="utf-8"))
    assert summary["reused_modes"] == ["legacy"]
    assert summary["success_modes"] == ["legacy", "profit_basic"]
    assert summary["publish_completed"] is True
    assert json.loads((base / "reward_legacy" / "status.json").read_text(encoding="utf-8"))["publish_ready"] is True
    assert (base / "stepF_equity_marl_SOXL.csv").exists()


def test_stepf_complete_requires_required_artifacts(tmp_path) -> None:
    reward_dir = tmp_path / "reward_legacy"
    reward_dir.mkdir(parents=True, exist_ok=True)
    (reward_dir / "stepF_equity_marl_SOXL.csv").write_text("Date,Split,ratio,ret,equity\n", encoding="utf-8")
    (reward_dir / "stepF_daily_log_marl_SOXL.csv").write_text("", encoding="utf-8")
    (reward_dir / "stepF_daily_log_router_SOXL.csv").write_text("Date,Split,ratio,ret,equity\n2024-01-01,test,1.0,0.01,1.01\n", encoding="utf-8")
    (reward_dir / "stepF_summary_router_SOXL.json").write_text("{}", encoding="utf-8")
    (reward_dir / "stepF_audit_router_SOXL.json").write_text("{}", encoding="utf-8")
    (reward_dir / "stepF_policy_compare_SOXL.json").write_text("{}", encoding="utf-8")

    manifest = StepFService._validate_reward_mode_artifacts(reward_dir, "SOXL")
    assert manifest["validation_passed"] is False
    assert "stepF_daily_log_marl_SOXL.csv" in manifest["invalid_artifacts"]
    assert "stepF_summary_router_SOXL.json" in manifest["invalid_artifacts"]


def test_stepf_no_canonical_publish_until_all_modes_complete(tmp_path) -> None:
    svc, date_range = _build_stubbed_stepf_service(tmp_path, reward_modes="legacy,profit_basic")
    original_run_router = svc._run_router

    def _run_router_partial(cfg, date_range, symbol, mode, persist_primary_outputs=True, data_cutoff=""):
        if cfg.reward_mode == "profit_basic":
            raise RuntimeError("stop before publish")
        return original_run_router(cfg, date_range, symbol, mode, persist_primary_outputs=persist_primary_outputs, data_cutoff=data_cutoff)

    svc._run_router = _run_router_partial  # type: ignore[assignment]

    try:
        svc.run(date_range, symbol="SOXL", mode="sim")
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass

    base = tmp_path / "out" / "stepF" / "sim"
    assert not (base / "stepF_equity_marl_SOXL.csv").exists()
    assert not (base / "stepF_compare_reward_modes_SOXL.json").exists()
    summary = json.loads((base / "stepF_multi_mode_summary_SOXL.json").read_text(encoding="utf-8"))
    assert summary["publish_completed"] is False


def test_stepf_wrapper_creates_stepf_sim_dir_before_config_validation(tmp_path, capsys) -> None:
    app_config = SimpleNamespace(stepF=None, output_root=str(tmp_path / "out"))
    svc = StepFService(app_config=app_config)
    date_range = SimpleNamespace(mode="sim")

    try:
        svc.run(date_range, symbol="SOXL", mode="sim")
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "app_config.stepF is missing" in str(exc)

    out = capsys.readouterr().out
    assert "[ONE_TAP][STEPF_ENTRY] begin" in out
    assert (tmp_path / "out" / "stepF").exists()
    assert (tmp_path / "out" / "stepF" / "sim").exists()


def test_stepf_wrapper_exception_logs_traceback_to_one_tap(tmp_path, capsys) -> None:
    cfg = StepFRouterConfig(
        output_root=str(tmp_path / "out"),
        agents="a1,a2",
        reward_mode="legacy",
        stepf_compare_reward_modes=True,
        stepf_reward_modes="legacy",
    )
    app_config = SimpleNamespace(stepF=cfg, output_root=str(tmp_path / "out"))
    svc = StepFService(app_config=app_config)

    svc._run_router = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))  # type: ignore[assignment]

    date_range = SimpleNamespace(mode="sim")
    try:
        svc.run(date_range, symbol="SOXL", mode="sim")
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "failed for all modes" in str(exc)

    out = capsys.readouterr().out
    assert "[ONE_TAP][STEPF_ENTRY] wrapper_traceback_begin" in out
    assert "Traceback (most recent call last)" in out
    assert "[ONE_TAP][STEPF_ENTRY] wrapper_stepf_dir_exists=true" in out


def _write_stepe_daily_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "Date": ["2024-01-01"],
            "Split": ["test"],
            "ratio": [1.0],
            "reward_next": [0.01],
        }
    ).to_csv(path, index=False)


def test_stepf_resolve_agents_from_stepe_daily_logs_10_agents(tmp_path, capsys) -> None:
    symbol = "SOXL"
    out_root = tmp_path / "out"
    stepe_root = out_root / "stepE" / "sim"
    expected_agents = [
        "dprime_all_features_3scale",
        "dprime_all_features_h01",
        "dprime_all_features_h02",
        "dprime_all_features_h03",
        "dprime_bnf_3scale",
        "dprime_bnf_h01",
        "dprime_bnf_h02",
        "dprime_mix_3scale",
        "dprime_mix_h01",
        "dprime_mix_h02",
    ]
    for agent in expected_agents:
        _write_stepe_daily_log(stepe_root / f"stepE_daily_log_{agent}_{symbol}.csv")

    svc = StepFService(app_config=SimpleNamespace())
    resolved, discovered, requested, selected_root = svc._resolve_agents(
        out_root=out_root,
        input_mode="sim",
        symbol=symbol,
        requested_agents_raw="",
    )

    assert len(discovered) == 10
    assert resolved == expected_agents
    assert requested == []
    assert selected_root == stepe_root.resolve()

    out = capsys.readouterr().out
    assert "[STEPF_AGENTS] begin" in out
    assert "[STEPF_AGENTS] candidate_stepE_roots=" in out
    assert f"[STEPF_AGENTS] selected_stepE_root={stepe_root.resolve()}" in out
    assert "[STEPF_AGENTS] discovered_daily_logs_count=10" in out
    assert "[STEPF_AGENTS] resolved_agents_count=10" in out


def test_stepf_resolve_agents_fallback_when_config_empty(tmp_path) -> None:
    symbol = "SOXL"
    out_root = tmp_path / "out"
    stepe_root = out_root / "stepE" / "sim"
    _write_stepe_daily_log(stepe_root / f"stepE_daily_log_dprime_bnf_h01_{symbol}.csv")
    _write_stepe_daily_log(stepe_root / f"stepE_daily_log_dprime_mix_h01_{symbol}.csv")

    svc = StepFService(app_config=SimpleNamespace())
    resolved, _, _, _ = svc._resolve_agents(
        out_root=out_root,
        input_mode="sim",
        symbol=symbol,
        requested_agents_raw="   ",
    )

    assert resolved == ["dprime_bnf_h01", "dprime_mix_h01"]


def test_stepf_extract_agent_name_handles_soxl_suffix() -> None:
    symbol = "SOXL"
    p = Path(f"stepE_daily_log_dprime_bnf_h01_{symbol}.csv")
    assert StepFService._extract_agent_name_from_daily_log(p, symbol) == "dprime_bnf_h01"


def test_stepf_resolve_agents_logs_details_when_empty(tmp_path, capsys) -> None:
    svc = StepFService(app_config=SimpleNamespace())
    resolved, discovered, requested, selected_root = svc._resolve_agents(
        out_root=tmp_path / "out",
        input_mode="sim",
        symbol="SOXL",
        requested_agents_raw="dprime_bnf_h01",
    )

    assert resolved == ["dprime_bnf_h01"]
    assert discovered == []
    assert requested == ["dprime_bnf_h01"]

    out = capsys.readouterr().out
    assert "[STEPF_AGENTS] input_stepE_root=" in out
    assert "[STEPF_AGENTS] requested_agents_raw=dprime_bnf_h01" in out
    assert "[STEPF_AGENTS] discovered_daily_logs_count=0" in out
    assert "[STEPF_AGENTS] resolved_agents_count=1" in out


def test_stepf_regression_empty_config_but_stepe_logs_exist_no_agents_empty_error(tmp_path) -> None:
    symbol = "SOXL"
    out_root = tmp_path / "out"
    stepe_root = out_root / "stepE" / "sim"
    _write_stepe_daily_log(stepe_root / f"stepE_daily_log_dprime_all_features_h01_{symbol}.csv")

    cfg = StepFRouterConfig(output_root=str(out_root), agents="", reward_mode="legacy")
    app_config = SimpleNamespace(stepF=cfg, output_root=str(out_root))
    svc = StepFService(app_config=app_config)

    svc._load_stepa_price_tech = lambda out_root, mode, symbol: pd.DataFrame({"Date": pd.to_datetime(["2024-01-01"]), "price_exec": [100.0]})  # type: ignore[assignment]
    svc._build_phase2_state = lambda **kwargs: pd.DataFrame({"Date": pd.to_datetime(["2024-01-01"]), "regime_id": [1], "confidence": [1.0]})  # type: ignore[assignment]
    svc._build_regime_edge_table = lambda merged, agents, cfg: pd.DataFrame([{"regime_id": 1, "agent": "dprime_all_features_h01", "IR": 1.0}])  # type: ignore[assignment]
    svc._build_allowlist = lambda edge_table, agents, safe_set, cfg: pd.DataFrame([{"regime_id": 1, "allowed_agents": "dprime_all_features_h01"}])  # type: ignore[assignment]
    svc.evaluate_final_outputs = staticmethod(lambda **kwargs: {"return_code": 0})  # type: ignore[assignment]

    date_range = SimpleNamespace(mode="sim", train_start="2023-01-01", train_end="2023-12-31", test_start="2024-01-01", test_end="2024-12-31")
    svc.run(date_range, symbol=symbol, mode="sim")

    assert (out_root / "stepF" / "sim" / "reward_legacy" / f"stepF_equity_marl_{symbol}.csv").exists()


def test_stepf_resolve_agents_logs_resolved_zero_when_no_request_and_no_logs(tmp_path, capsys) -> None:
    svc = StepFService(app_config=SimpleNamespace())
    resolved, discovered, requested, selected_root = svc._resolve_agents(
        out_root=tmp_path / "out",
        input_mode="sim",
        symbol="SOXL",
        requested_agents_raw="",
    )

    assert resolved == []
    assert discovered == []
    assert requested == []

    out = capsys.readouterr().out
    assert "[STEPF_AGENTS] discovered_daily_logs_count=0" in out
    assert "[STEPF_AGENTS] resolved_agents_count=0" in out


def test_stepf_resolve_agents_prefers_effective_output_root_and_avoids_repo_relative_default_output(tmp_path, capsys) -> None:
    symbol = "SOXL"
    output_root = tmp_path / "run_123" / "output"
    stepe_root = output_root / "stepE" / "sim"
    for i in range(10):
        _write_stepe_daily_log(stepe_root / f"stepE_daily_log_agent{i}_{symbol}.csv")

    svc = StepFService(app_config=SimpleNamespace(output_root="output"))
    resolved, discovered, requested, selected_root = svc._resolve_agents(
        out_root=output_root,
        input_mode="sim",
        symbol=symbol,
        requested_agents_raw="",
    )

    assert len(discovered) == 10
    assert len(resolved) == 10
    assert requested == []
    assert selected_root == stepe_root.resolve()
    repo_relative_candidate = (Path.cwd() / "output" / "stepE" / "sim").resolve()
    assert selected_root != repo_relative_candidate

    out = capsys.readouterr().out
    assert f"[STEPF_AGENTS] selected_stepE_root={stepe_root.resolve()}" in out
    assert "[STEPF_AGENTS] selected_stepE_root_exists=true" in out
    assert f"[STEPF_AGENTS] candidate_stepE_roots={stepe_root.resolve()}" in out
    assert str(repo_relative_candidate) not in out
    assert "[STEPF_AGENTS] discovered_daily_logs_count=10" in out
    assert "[STEPF_AGENTS] resolved_agents_count=10" in out


def test_stepf_resolve_output_root_prefers_effective_root_when_config_is_default_output(tmp_path) -> None:
    effective_root = tmp_path / "runs" / "20260101_000000" / "output"
    svc = StepFService(
        app_config=SimpleNamespace(
            output_root="output",
            effective_output_root=str(effective_root),
        )
    )

    resolved = svc._resolve_output_root("output")

    assert resolved == effective_root.resolve()
