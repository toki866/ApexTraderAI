from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_core.services import step_dprime_service as sds
from ai_core.utils.step_contract_utils import validate_step_dprime


def test_build_pred_from_stepb_accepts_path_mamba_without_pathseq(tmp_path: Path):
    stepb = tmp_path / "stepB"
    stepb.mkdir()
    pd.DataFrame(
        {
            "Date_anchor": ["2024-01-02", "2024-01-03"],
            "Pred_Close_t_plus_01": [10.0, 11.0],
            "Pred_Close_t_plus_05": [12.0, 13.0],
        }
    ).to_csv(stepb / "stepB_pred_path_mamba_SOXL.csv", index=False)

    df, meta = sds._build_pred_from_stepb(stepb, "SOXL", pred_k=5)

    assert meta["pred_source_mode"] == "path_mamba"
    assert meta["pred_available_horizons"] == [1, 5]
    assert "Date" in df.columns


def test_build_pred_from_stepb_accepts_time_all_without_pathseq(tmp_path: Path):
    stepb = tmp_path / "stepB"
    stepb.mkdir()
    pd.DataFrame(
        {
            "Date": ["2024-01-02", "2024-01-03"],
            "Pred_Close_MAMBA_h01": [10.0, 11.0],
            "Pred_Close_MAMBA_h05": [12.0, 13.0],
        }
    ).to_csv(stepb / "stepB_pred_time_all_SOXL.csv", index=False)

    df, meta = sds._build_pred_from_stepb(stepb, "SOXL", pred_k=5)

    assert meta["pred_source_mode"] == "time_all"
    assert "Pred_Close_t_plus_01" in df.columns
    assert "Pred_Close_t_plus_05" in df.columns


def test_dprime_rl_fills_missing_horizons_and_records_summary(tmp_path: Path):
    n = 40
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    data = pd.DataFrame({"Date": dates, "Close_anchor": np.linspace(100, 120, n), "gap_atr": 0.1, "ATR_norm": 0.2})
    for c in [
        "ret_1", "ret_5", "ret_20", "range_atr", "body_ratio", "body_atr", "upper_wick_ratio", "lower_wick_ratio",
        "Gap", "vol_log_ratio_20", "vol_chg", "dev_z_25", "bnf_score", "RSI", "MACD_hist", "macd_hist_delta",
        "macd_hist_cross_up", "clv", "distribution_day", "dist_count_25", "absorption_day", "cmf_20",
    ]:
        data[c] = 0.01

    cluster_daily = pd.DataFrame({"Date": dates, "cluster_id_raw20": 1, "cluster_id_stable": 1, "rare_flag_raw20": 0})

    stepb = tmp_path / "stepB"
    stepc = tmp_path / "stepC"
    stepd = tmp_path / "stepD"
    stepb.mkdir(); stepc.mkdir(); stepd.mkdir()
    pd.DataFrame(
        {
            "Date": [d.strftime("%Y-%m-%d") for d in dates],
            "Pred_Close_MAMBA_h01": np.linspace(100, 120, n),
            "Pred_Close_MAMBA_h05": np.linspace(101, 121, n),
        }
    ).to_csv(stepb / "stepB_pred_time_all_SOXL.csv", index=False)

    cfg = sds.StepDPrimeConfig(
        symbol="SOXL",
        pred_k=5,
        l_past=3,
        z_past_dim=2,
        z_pred_dim=2,
        profiles=("dprime_all_features_h03",),
    )
    split = {"train_start": "2024-01-01", "train_end": "2024-01-20", "test_start": "2024-01-21", "test_end": "2024-02-09"}

    out = sds.DPrimeRLService().run(
        cfg,
        timing=sds.TimingLogger.disabled(),
        data=data,
        split=split,
        stepb_dir=stepb,
        stepc_dir=stepc,
        stepd_dir=stepd,
        cluster_daily=cluster_daily,
    )

    assert out["pred_missing_horizons_filled"]["h02"] == "reuse_h01"
    assert out["pred_missing_horizons_filled"]["h03"] == "reuse_h01"
    assert (stepd / "stepDprime_state_test_dprime_all_features_h03_SOXL.csv").exists()
    assert (stepd / "embeddings" / "stepDprime_dprime_all_features_h03_SOXL_embeddings_all.csv").exists()


def test_add_rl_scalar_context_uses_volume_proxy_when_volume_missing():
    data = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=30, freq="D"),
            "Close_anchor": np.linspace(100, 120, 30),
            "ret_1": np.linspace(-0.02, 0.02, 30),
            "ret_20": np.linspace(-0.05, 0.05, 30),
            "gap_atr": 0.1,
            "ATR_norm": 0.2,
            "vol_log_ratio_20": np.linspace(0.1, 0.4, 30),
        }
    )

    out, diag = sds._add_rl_scalar_context(data)

    assert diag["volume_source"] == "vol_log_ratio_20"
    assert diag["volume_proxy_used"] is True
    assert "Volume" not in data.columns
    assert "ctx_turnover_20" in out.columns
    assert np.isfinite(out["ctx_turnover_20"]).all()


def test_dprime_rl_runs_without_raw_volume_column_and_writes_scalar_context_diagnostics(tmp_path: Path):
    n = 50
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    data = pd.DataFrame({"Date": dates, "Close_anchor": np.linspace(100, 125, n), "gap_atr": 0.1, "ATR_norm": 0.2})
    for c in [
        "ret_1", "ret_5", "ret_20", "range_atr", "body_ratio", "body_atr", "upper_wick_ratio", "lower_wick_ratio",
        "Gap", "vol_log_ratio_20", "vol_chg", "dev_z_25", "bnf_score", "RSI", "MACD_hist", "macd_hist_delta",
        "macd_hist_cross_up", "clv", "distribution_day", "dist_count_25", "absorption_day", "cmf_20",
    ]:
        data[c] = np.linspace(0.01, 0.05, n)
    data = data.drop(columns=["Volume"], errors="ignore")

    cluster_daily = pd.DataFrame({"Date": dates, "cluster_id_raw20": 1, "cluster_id_stable": 1, "rare_flag_raw20": 0})

    stepb = tmp_path / "stepB"
    stepc = tmp_path / "stepC"
    stepd = tmp_path / "stepD"
    stepb.mkdir(); stepc.mkdir(); stepd.mkdir()
    pd.DataFrame(
        {
            "Date": [d.strftime("%Y-%m-%d") for d in dates],
            "Pred_Close_MAMBA_h01": np.linspace(100, 125, n),
            "Pred_Close_MAMBA_h05": np.linspace(101, 126, n),
        }
    ).to_csv(stepb / "stepB_pred_time_all_SOXL.csv", index=False)

    cfg = sds.StepDPrimeConfig(
        symbol="SOXL",
        pred_k=5,
        l_past=3,
        z_past_dim=2,
        z_pred_dim=2,
        profiles=("dprime_bnf_h01",),
    )
    split = {"train_start": "2024-01-01", "train_end": "2024-01-25", "test_start": "2024-01-26", "test_end": "2024-02-19"}

    out = sds.DPrimeRLService().run(
        cfg,
        timing=sds.TimingLogger.disabled(),
        data=data,
        split=split,
        stepb_dir=stepb,
        stepc_dir=stepc,
        stepd_dir=stepd,
        cluster_daily=cluster_daily,
    )

    diag_path = stepd / "stepDprime_scalar_context_diagnostics_SOXL.json"
    assert out["profiles"]["dprime_bnf_h01"]["test"].endswith("stepDprime_state_test_dprime_bnf_h01_SOXL.csv")
    assert (stepd / "stepDprime_state_test_dprime_bnf_h01_SOXL.csv").exists()
    assert (stepd / "embeddings" / "stepDprime_dprime_bnf_h01_SOXL_embeddings_all.csv").exists()
    assert diag_path.exists()
    diag = json.loads(diag_path.read_text(encoding="utf-8"))
    assert diag["volume_source"] == "vol_log_ratio_20"
    assert diag["volume_proxy_used"] is True


def test_stepdprime_service_writes_traceback_log_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    mode = "sim"
    symbol = "SOXL"
    stepa = tmp_path / "output" / "stepA" / mode
    stepd = tmp_path / "output" / "stepDprime" / mode
    stepa.mkdir(parents=True)

    pd.DataFrame({"key": ["train_start", "train_end", "test_start", "test_end"], "value": ["2024-01-01", "2024-01-10", "2024-01-11", "2024-01-15"]}).to_csv(
        stepa / f"stepA_split_summary_{symbol}.csv", index=False
    )
    for stem in ("stepA_prices", "stepA_tech", "stepA_periodic"):
        pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=15, freq="D"), "Open": 1, "High": 1, "Low": 1, "Close": 1, "Volume": 1}).to_csv(
            stepa / f"{stem}_train_{symbol}.csv", index=False
        )
        pd.DataFrame({"Date": pd.date_range("2024-01-16", periods=5, freq="D"), "Open": 1, "High": 1, "Low": 1, "Close": 1, "Volume": 1}).to_csv(
            stepa / f"{stem}_test_{symbol}.csv", index=False
        )

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(sds.DPrimeClusterService, "run", _boom)

    with pytest.raises(RuntimeError):
        sds.StepDPrimeService().run(
            sds.StepDPrimeConfig(symbol=symbol, output_root=str(tmp_path / "output"), mode=mode)
        )

    tb = stepd / f"stepDprime_traceback_{symbol}.log"
    assert tb.exists()
    assert "RuntimeError: boom" in tb.read_text(encoding="utf-8")
    fs = stepd / f"stepDprime_failure_summary_{symbol}.json"
    assert fs.exists()


def test_stepdprime_service_postcheck_validates_required_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    mode = "sim"
    symbol = "SOXL"
    out_root = tmp_path / "output"
    stepa = out_root / "stepA" / mode
    stepb = out_root / "stepB" / mode
    stepa.mkdir(parents=True)
    stepb.mkdir(parents=True)

    pd.DataFrame({"key": ["train_start", "train_end", "test_start", "test_end"], "value": ["2024-01-01", "2024-01-10", "2024-01-11", "2024-01-15"]}).to_csv(
        stepa / f"stepA_split_summary_{symbol}.csv", index=False
    )
    for stem in ("stepA_prices", "stepA_tech", "stepA_periodic"):
        pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=15, freq="D"), "Open": 1, "High": 2, "Low": 1, "Close": 1, "Volume": 1}).to_csv(
            stepa / f"{stem}_train_{symbol}.csv", index=False
        )
        pd.DataFrame({"Date": pd.date_range("2024-01-16", periods=5, freq="D"), "Open": 1, "High": 2, "Low": 1, "Close": 1, "Volume": 1}).to_csv(
            stepa / f"{stem}_test_{symbol}.csv", index=False
        )

    pd.DataFrame({"Date_anchor": ["2024-01-11"], "Pred_Close_t_plus_01": [1.0], "Pred_Close_t_plus_20": [1.0]}).to_csv(
        stepb / f"stepB_pred_pathseq_mamba_h20_{symbol}.csv", index=False
    )

    def _cluster_ok(self, cfg, data, periodic, stepd_dir):
        pd.DataFrame({"Date": ["2024-01-11"], "cluster_id_raw20": [1], "cluster_id_stable": [1], "rare_flag_raw20": [0]}).to_csv(
            stepd_dir / f"stepDprime_cluster_daily_assign_{symbol}.csv", index=False
        )
        (stepd_dir / f"stepDprime_cluster_summary_{symbol}.json").write_text("{}", encoding="utf-8")
        return {"daily": pd.DataFrame({"Date": [pd.Timestamp("2024-01-11")], "cluster_id_raw20": [1], "cluster_id_stable": [1], "rare_flag_raw20": [0]}), "summary": {}}

    def _rl_bad(self, cfg, **kwargs):
        stepd_dir = kwargs["stepd_dir"]
        pd.DataFrame({"Date": ["2024-01-11"], "x": [1]}).to_csv(
            stepd_dir / f"stepDprime_state_test_dprime_all_features_h01_{symbol}.csv", index=False
        )
        return {"profiles": {}}

    monkeypatch.setattr(sds.DPrimeClusterService, "run", _cluster_ok)
    monkeypatch.setattr(sds.DPrimeRLService, "run", _rl_bad)

    with pytest.raises(FileNotFoundError):
        sds.StepDPrimeService().run(sds.StepDPrimeConfig(symbol=symbol, output_root=str(out_root), mode=mode))


def test_stepdprime_service_failure_prints_traceback_markers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    mode = "sim"
    symbol = "SOXL"
    stepa = tmp_path / "output" / "stepA" / mode
    stepa.mkdir(parents=True)

    pd.DataFrame({"key": ["train_start", "train_end", "test_start", "test_end"], "value": ["2024-01-01", "2024-01-10", "2024-01-11", "2024-01-15"]}).to_csv(
        stepa / f"stepA_split_summary_{symbol}.csv", index=False
    )
    for stem in ("stepA_prices", "stepA_tech", "stepA_periodic"):
        pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=15, freq="D"), "Open": 1, "High": 1, "Low": 1, "Close": 1, "Volume": 1}).to_csv(
            stepa / f"{stem}_train_{symbol}.csv", index=False
        )
        pd.DataFrame({"Date": pd.date_range("2024-01-16", periods=5, freq="D"), "Open": 1, "High": 1, "Low": 1, "Close": 1, "Volume": 1}).to_csv(
            stepa / f"{stem}_test_{symbol}.csv", index=False
        )

    def _boom(*args, **kwargs):
        raise RuntimeError("boom-marker")

    monkeypatch.setattr(sds.DPrimeClusterService, "run", _boom)

    with pytest.raises(RuntimeError):
        sds.StepDPrimeService().run(
            sds.StepDPrimeConfig(symbol=symbol, output_root=str(tmp_path / "output"), mode=mode)
        )

    captured = capsys.readouterr()
    assert "[STEPDPRIME_FAIL_TRACEBACK_BEGIN]" in captured.out
    assert "RuntimeError: boom-marker" in captured.out
    assert "[STEPDPRIME_FAIL_TRACEBACK_END]" in captured.out


def test_stepdprime_service_writes_ready_markers_and_audit_summaries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    mode = "sim"
    symbol = "SOXL"
    out_root = tmp_path / "output"
    stepa = out_root / "stepA" / mode
    stepb = out_root / "stepB" / mode
    stepa.mkdir(parents=True)
    stepb.mkdir(parents=True)

    pd.DataFrame({"key": ["train_start", "train_end", "test_start", "test_end"], "value": ["2024-01-01", "2024-01-10", "2024-01-11", "2024-01-15"]}).to_csv(
        stepa / f"stepA_split_summary_{symbol}.csv", index=False
    )
    for stem in ("stepA_prices", "stepA_tech", "stepA_periodic"):
        pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=15, freq="D"), "Open": 1, "High": 2, "Low": 1, "Close": 1, "Volume": 1}).to_csv(
            stepa / f"{stem}_train_{symbol}.csv", index=False
        )
        pd.DataFrame({"Date": pd.date_range("2024-01-16", periods=5, freq="D"), "Open": 1, "High": 2, "Low": 1, "Close": 1, "Volume": 1}).to_csv(
            stepa / f"{stem}_test_{symbol}.csv", index=False
        )

    pred_source = stepb / f"stepB_pred_time_all_{symbol}.csv"
    pd.DataFrame({"Date": ["2024-01-11"], "Pred_Close_MAMBA_h01": [1.0], "Pred_Close_MAMBA_h05": [1.0]}).to_csv(pred_source, index=False)

    def _cluster_ok(self, cfg, data, periodic, stepd_dir):
        pd.DataFrame({"Date": ["2024-01-11"], "cluster_id_raw20": [1], "cluster_id_stable": [1], "rare_flag_raw20": [0]}).to_csv(
            stepd_dir / f"stepDprime_cluster_daily_assign_{symbol}.csv", index=False
        )
        (stepd_dir / f"stepDprime_cluster_summary_{symbol}.json").write_text('{"status":"READY"}', encoding="utf-8")
        return {"daily": pd.DataFrame({"Date": [pd.Timestamp("2024-01-11")], "cluster_id_raw20": [1], "cluster_id_stable": [1], "rare_flag_raw20": [0]}), "summary": {"status": "READY"}}

    def _rl_ok(self, cfg, **kwargs):
        stepd_dir = kwargs["stepd_dir"]
        profile = cfg.profiles[0]
        pd.DataFrame({"Date": ["2024-01-11"], "x": [1]}).to_csv(stepd_dir / f"stepDprime_state_train_{profile}_{symbol}.csv", index=False)
        pd.DataFrame({"Date": ["2024-01-12"], "x": [1]}).to_csv(stepd_dir / f"stepDprime_state_test_{profile}_{symbol}.csv", index=False)
        emb_dir = stepd_dir / "embeddings"
        emb_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"Date": ["2024-01-11"], "emb_000": [1.0]}).to_csv(emb_dir / f"stepDprime_{profile}_{symbol}_embeddings_test.csv", index=False)
        pd.DataFrame({"Date": ["2024-01-11"], "emb_000": [1.0]}).to_csv(emb_dir / f"stepDprime_{profile}_{symbol}_embeddings_all.csv", index=False)
        pd.DataFrame(
            [
                {"key": "pred_source_file", "value": pred_source.as_posix()},
                {"key": "pred_source_mode", "value": "time_all"},
                {"key": "pred_available_horizons", "value": "1|5"},
            ]
        ).to_csv(stepd_dir / f"stepDprime_split_summary_{profile}_{symbol}.csv", index=False)
        return {"profiles": {profile: {"train": "ok", "test": "ok"}}}

    monkeypatch.setattr(sds.DPrimeClusterService, "run", _cluster_ok)
    monkeypatch.setattr(sds.DPrimeRLService, "run", _rl_ok)

    cfg = sds.StepDPrimeConfig(symbol=symbol, output_root=str(out_root), mode=mode, profiles=("dprime_all_features_h01",))
    result = sds.StepDPrimeService().run(cfg)

    stepd_dir = out_root / "stepDprime" / mode
    marker_dir = stepd_dir / "pipeline_markers"
    base_meta = json.loads((stepd_dir / f"stepDprime_base_meta_{symbol}.json").read_text(encoding="utf-8"))
    profile_summary = json.loads((stepd_dir / f"stepDprime_profile_summary_dprime_all_features_h01_{symbol}.json").read_text(encoding="utf-8"))

    assert result["dprime_cluster"]["status"] == "READY"
    assert base_meta["status"] == "READY"
    assert (marker_dir / "DPrimeBaseCluster.READY.json").exists()
    assert not (marker_dir / "DPrimeBaseCluster.RUNNING.json").exists()
    assert profile_summary["status"] == "READY"
    assert profile_summary["pred_source_selected"] == f"stepB/{mode}/stepB_pred_time_all_{symbol}.csv"
    assert profile_summary["pred_available_horizons"] == [1, 5]
    assert profile_summary["force_cpu"] is False
    assert profile_summary["gpu_execution_guard"] == "dprime_final_completes_before_stepe_agent_start"
    assert (marker_dir / f"DPrimeFinal_dprime_all_features_h01.READY.json").exists()
    assert not (marker_dir / f"DPrimeFinal_dprime_all_features_h01.RUNNING.json").exists()
    assert validate_step_dprime(out_root, mode, symbol, ("dprime_all_features_h01",)) == []


def test_stepdprime_service_failure_marks_failed_profile_and_cleans_running_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    mode = "sim"
    symbol = "SOXL"
    out_root = tmp_path / "output"
    stepa = out_root / "stepA" / mode
    stepa.mkdir(parents=True)

    pd.DataFrame({"key": ["train_start", "train_end", "test_start", "test_end"], "value": ["2024-01-01", "2024-01-10", "2024-01-11", "2024-01-15"]}).to_csv(
        stepa / f"stepA_split_summary_{symbol}.csv", index=False
    )
    for stem in ("stepA_prices", "stepA_tech", "stepA_periodic"):
        pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=15, freq="D"), "Open": 1, "High": 2, "Low": 1, "Close": 1, "Volume": 1}).to_csv(
            stepa / f"{stem}_train_{symbol}.csv", index=False
        )
        pd.DataFrame({"Date": pd.date_range("2024-01-16", periods=5, freq="D"), "Open": 1, "High": 2, "Low": 1, "Close": 1, "Volume": 1}).to_csv(
            stepa / f"{stem}_test_{symbol}.csv", index=False
        )

    def _cluster_ok(self, cfg, data, periodic, stepd_dir):
        pd.DataFrame({"Date": ["2024-01-11"], "cluster_id_raw20": [1], "cluster_id_stable": [1], "rare_flag_raw20": [0]}).to_csv(
            stepd_dir / f"stepDprime_cluster_daily_assign_{symbol}.csv", index=False
        )
        (stepd_dir / f"stepDprime_cluster_summary_{symbol}.json").write_text('{"status":"READY"}', encoding="utf-8")
        return {"daily": pd.DataFrame({"Date": [pd.Timestamp("2024-01-11")], "cluster_id_raw20": [1], "cluster_id_stable": [1], "rare_flag_raw20": [0]}), "summary": {"status": "READY"}}

    def _rl_boom(self, cfg, **kwargs):
        raise RuntimeError("profile-boom")

    monkeypatch.setattr(sds.DPrimeClusterService, "run", _cluster_ok)
    monkeypatch.setattr(sds.DPrimeRLService, "run", _rl_boom)

    cfg = sds.StepDPrimeConfig(symbol=symbol, output_root=str(out_root), mode=mode, profiles=("dprime_all_features_h01",))
    with pytest.raises(RuntimeError):
        sds.StepDPrimeService().run(cfg)

    stepd_dir = out_root / "stepDprime" / mode
    marker_dir = stepd_dir / "pipeline_markers"
    profile_summary = json.loads((stepd_dir / f"stepDprime_profile_summary_dprime_all_features_h01_{symbol}.json").read_text(encoding="utf-8"))
    failure_summary = json.loads((stepd_dir / f"stepDprime_failure_summary_{symbol}.json").read_text(encoding="utf-8"))

    assert profile_summary["status"] == "FAILED"
    assert profile_summary["error_type"] == "RuntimeError"
    assert (marker_dir / "DPrimeBaseCluster.READY.json").exists()
    assert (marker_dir / "DPrimeFinal_dprime_all_features_h01.FAILED.json").exists()
    assert not (marker_dir / "DPrimeFinal_dprime_all_features_h01.RUNNING.json").exists()
    assert failure_summary["status"] == "FAILED"
    assert failure_summary["traceback_path"] == f"stepDprime/{mode}/stepDprime_traceback_{symbol}.log"


def test_dprime_rl_duplicate_dates_are_handled(tmp_path: Path):
    n = 45
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    data = pd.DataFrame({"Date": list(dates) + [dates[-1]], "Close_anchor": list(np.linspace(100, 130, n)) + [130.0], "gap_atr": 0.1, "ATR_norm": 0.2})
    for c in [
        "ret_1", "ret_5", "ret_20", "range_atr", "body_ratio", "body_atr", "upper_wick_ratio", "lower_wick_ratio",
        "Gap", "vol_log_ratio_20", "vol_chg", "dev_z_25", "bnf_score", "RSI", "MACD_hist", "macd_hist_delta",
        "macd_hist_cross_up", "clv", "distribution_day", "dist_count_25", "absorption_day", "cmf_20",
    ]:
        data[c] = 0.01

    cluster_daily = pd.DataFrame({"Date": list(dates) + [dates[-1]], "cluster_id_raw20": 1, "cluster_id_stable": 1, "rare_flag_raw20": 0})

    stepb = tmp_path / "stepB"
    stepc = tmp_path / "stepC"
    stepd = tmp_path / "stepD"
    stepb.mkdir(); stepc.mkdir(); stepd.mkdir()
    pd.DataFrame(
        {
            "Date": [d.strftime("%Y-%m-%d") for d in dates],
            "Pred_Close_MAMBA_h01": np.linspace(100, 130, n),
            "Pred_Close_MAMBA_h05": np.linspace(101, 131, n),
        }
    ).to_csv(stepb / "stepB_pred_time_all_SOXL.csv", index=False)

    cfg = sds.StepDPrimeConfig(
        symbol="SOXL",
        pred_k=5,
        l_past=3,
        z_past_dim=2,
        z_pred_dim=2,
        profiles=("dprime_all_features_h03",),
    )
    split = {"train_start": "2024-01-01", "train_end": "2024-01-25", "test_start": "2024-01-26", "test_end": "2024-02-14"}

    out = sds.DPrimeRLService().run(
        cfg,
        timing=sds.TimingLogger.disabled(),
        data=data,
        split=split,
        stepb_dir=stepb,
        stepc_dir=stepc,
        stepd_dir=stepd,
        cluster_daily=cluster_daily,
    )
    assert out["profiles"]
    assert (stepd / "stepDprime_state_test_dprime_all_features_h03_SOXL.csv").exists()


def test_fit_scaler_handles_nonfinite_inputs():
    x = np.array(
        [
            [1.0, np.nan, np.inf, -np.inf],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 3.0, 4.0, 5.0],
        ],
        dtype=float,
    )
    mu, sd = sds._fit_scaler(x)
    assert np.isfinite(mu).all()
    assert np.isfinite(sd).all()

def test_stepdprime_config_rejects_legacy_encoder_type():
    with pytest.raises(ValueError, match="legacy"):
        sds.StepDPrimeConfig(symbol="SOXL", encoder_type="legacy")


def test_dprime_rl_writes_encoder_diagnostics_json(tmp_path: Path):
    n = 42
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    data = pd.DataFrame({"Date": dates, "Close_anchor": np.linspace(100, 120, n), "gap_atr": 0.1, "ATR_norm": 0.2})
    for c in [
        "ret_1", "ret_5", "ret_20", "range_atr", "body_ratio", "body_atr", "upper_wick_ratio", "lower_wick_ratio",
        "Gap", "vol_log_ratio_20", "vol_chg", "dev_z_25", "bnf_score", "RSI", "MACD_hist", "macd_hist_delta",
        "macd_hist_cross_up", "clv", "distribution_day", "dist_count_25", "absorption_day", "cmf_20",
    ]:
        data[c] = 0.01
    data.loc[data.index[5], "ret_1"] = np.nan
    data.loc[data.index[6], "ret_5"] = np.inf

    cluster_daily = pd.DataFrame({"Date": dates, "cluster_id_raw20": 1, "cluster_id_stable": 1, "rare_flag_raw20": 0})

    stepb = tmp_path / "stepB"
    stepc = tmp_path / "stepC"
    stepd = tmp_path / "stepD"
    stepb.mkdir(); stepc.mkdir(); stepd.mkdir()
    pd.DataFrame(
        {
            "Date": [d.strftime("%Y-%m-%d") for d in dates],
            "Pred_Close_MAMBA_h01": np.linspace(100, 120, n),
            "Pred_Close_MAMBA_h05": np.linspace(101, 121, n),
        }
    ).to_csv(stepb / "stepB_pred_time_all_SOXL.csv", index=False)

    cfg = sds.StepDPrimeConfig(symbol="SOXL", pred_k=5, l_past=3, z_past_dim=2, z_pred_dim=2, profiles=("dprime_all_features_h01",))
    split = {"train_start": "2024-01-01", "train_end": "2024-01-22", "test_start": "2024-01-23", "test_end": "2024-02-11"}

    sds.DPrimeRLService().run(
        cfg,
        timing=sds.TimingLogger.disabled(),
        data=data,
        split=split,
        stepb_dir=stepb,
        stepc_dir=stepc,
        stepd_dir=stepd,
        cluster_daily=cluster_daily,
    )

    diag_path = stepd / "stepDprime_encoder_diagnostics_dprime_all_features_h01_SOXL.json"
    assert diag_path.exists()
    diag = json.loads(diag_path.read_text(encoding="utf-8"))
    assert diag["profile"] == "dprime_all_features_h01"
    assert diag["encoder_type"] == "transformer"
    assert "past_seq_shape" in diag
    assert "epoch_losses" in diag
