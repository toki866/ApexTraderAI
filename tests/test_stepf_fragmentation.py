from __future__ import annotations

from pathlib import Path

from types import SimpleNamespace

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from ai_core.services import step_f_service as sf_mod
from ai_core.services.step_f_service import StepFRouterConfig, StepFService


class _FakeClusterer:
    def __init__(self, **kwargs):
        self.labels_ = np.array([], dtype=int)

    def fit(self, x_train_p):
        self.labels_ = np.zeros((x_train_p.shape[0],), dtype=int)
        return self


class _FakePred:
    @staticmethod
    def approximate_predict(clusterer, x_test_p):
        return np.zeros((x_test_p.shape[0],), dtype=int), np.ones((x_test_p.shape[0],), dtype=float)


class _FakeHdbscanModule:
    HDBSCAN = _FakeClusterer


class _FakeTICCClusterer:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._train_mode = "raw20" if int(kwargs.get("num_clusters", 0)) == 20 else "stable"
        _FakeTICCClusterer.calls.append(int(kwargs.get("num_clusters", 0)))

    def fit_predict_train(self, x_train):
        if self._train_mode == "raw20":
            labels = np.array([0, 0, 0, 1, 1, 2, 3, 4], dtype=int)
            if x_train.shape[0] <= len(labels):
                return labels[: x_train.shape[0]]
            return np.pad(labels, (0, x_train.shape[0] - len(labels)), mode="edge")
        return np.array([(i % max(1, int(self.kwargs.get("num_clusters", 1)))) for i in range(x_train.shape[0])], dtype=int)

    def predict_test(self, x_test):
        if self._train_mode == "raw20":
            labels = np.array([0, 3, 2], dtype=int)
        else:
            labels = np.array([1, 1, 2], dtype=int)
        labels = labels[: x_test.shape[0]] if x_test.shape[0] <= len(labels) else np.pad(labels, (0, x_test.shape[0] - len(labels)), mode="edge")
        return labels.astype(int), np.full((x_test.shape[0],), 0.8, dtype=float)


def test_build_phase2_state_uses_concat_and_logs_frame_stats(monkeypatch, capsys) -> None:
    svc = StepFService(app_config=SimpleNamespace())
    monkeypatch.setattr(sf_mod, "hdbscan", _FakeHdbscanModule)
    monkeypatch.setattr(sf_mod, "hdbscan_prediction", _FakePred)

    def _fake_base_features(price_tech: pd.DataFrame, tech: pd.DataFrame) -> pd.DataFrame:
        n = len(price_tech)
        return pd.DataFrame(
            {
                "Date": pd.to_datetime(price_tech["Date"]),
                "Gap": np.linspace(0.0, 1.0, n),
                "ATR_norm": np.linspace(1.0, 2.0, n),
                "ret_1": np.linspace(-0.1, 0.1, n),
            }
        )

    monkeypatch.setattr(sf_mod, "_compute_base_features", _fake_base_features)

    cfg = StepFRouterConfig(
        past_window_days=5,
        past_resample_len=4,
        pca_n_components=2,
        hdbscan_min_cluster_size=2,
        hdbscan_min_samples=1,
        use_z_pred=False,
    )
    dates = pd.date_range("2024-01-01", periods=12, freq="D")
    price_tech = pd.DataFrame(
        {
            "Date": dates,
            "price_exec": np.linspace(100.0, 112.0, len(dates)),
            "feat_a_tech": np.linspace(0.0, 2.0, len(dates)),
        }
    )
    date_range = SimpleNamespace(
        train_start="2024-01-01",
        train_end="2024-01-08",
        test_start="2024-01-09",
        test_end="2024-01-12",
    )

    out = svc._build_phase2_state(cfg=cfg, date_range=date_range, symbol="SOXL", mode="sim", out_root=Path("."), price_tech=price_tech)

    assert {"Date", "regime_id", "cluster_id_stable", "cluster_id_raw20", "rare_flag_raw20", "confidence_stable", "confidence_raw20", "confidence"}.issubset(set(out.columns))
    logs = capsys.readouterr().out
    assert "[STEPF_FRAME] before_feature_concat" in logs
    assert "[STEPF_FRAME] new_feature_cols=" in logs
    assert "[STEPF_FRAME] after_feature_concat" in logs
    assert "[STEPF_FRAME] applied_copy_defragment=true" in logs


def test_run_ignores_performance_warning_from_phase2_builder(monkeypatch, tmp_path) -> None:
    cfg = StepFRouterConfig(output_root=str(tmp_path / "out"), agents="a1", reward_mode="legacy")
    app_config = SimpleNamespace(stepF=cfg, output_root=str(tmp_path / "out"))
    svc = StepFService(app_config=app_config)

    monkeypatch.setattr(sf_mod, "hdbscan", object())
    monkeypatch.setattr(sf_mod, "hdbscan_prediction", object())

    svc._load_stepa_price_tech = lambda out_root, mode, symbol: pd.DataFrame({"Date": pd.to_datetime(["2024-01-01", "2024-01-02"]), "price_exec": [100.0, 101.0]})  # type: ignore[assignment]
    svc._load_stepe_logs = lambda step_e_root, symbol, agents: {  # type: ignore[assignment]
        "a1": pd.DataFrame({"Date": pd.to_datetime(["2024-01-01", "2024-01-02"]), "Split": ["train", "test"], "ratio": [1.0, 1.0], "stepE_ret_for_stats": [0.01, 0.01]})
    }

    def _warn_phase2(**kwargs):
        import warnings
        from pandas.errors import PerformanceWarning

        warnings.warn("DataFrame is highly fragmented", PerformanceWarning)
        return pd.DataFrame({"Date": pd.to_datetime(["2024-01-01", "2024-01-02"]), "regime_id": [1, 1], "confidence": [1.0, 1.0]})

    svc._build_phase2_state = _warn_phase2  # type: ignore[assignment]
    svc._build_regime_edge_table = lambda merged, agents, cfg: pd.DataFrame([{"regime_id": 1, "agent": "a1", "IR": 1.0}])  # type: ignore[assignment]
    svc._build_allowlist = lambda edge_table, agents, safe_set, cfg: pd.DataFrame([{"regime_id": 1, "allowed_agents": "a1"}])  # type: ignore[assignment]
    svc.evaluate_final_outputs = staticmethod(lambda **kwargs: {"return_code": 0})  # type: ignore[assignment]

    date_range = SimpleNamespace(mode="sim", train_start="2024-01-01", train_end="2024-01-01", test_start="2024-01-02", test_end="2024-01-02")
    result = svc.run(date_range, symbol="SOXL", mode="sim")
    assert "stepF_daily_log_router_SOXL.csv" in result.daily_log_path


def test_run_still_fails_on_real_exception(monkeypatch, tmp_path) -> None:
    cfg = StepFRouterConfig(output_root=str(tmp_path / "out"), agents="a1", reward_mode="legacy")
    app_config = SimpleNamespace(stepF=cfg, output_root=str(tmp_path / "out"))
    svc = StepFService(app_config=app_config)

    monkeypatch.setattr(sf_mod, "hdbscan", object())
    monkeypatch.setattr(sf_mod, "hdbscan_prediction", object())

    svc._load_stepa_price_tech = lambda out_root, mode, symbol: pd.DataFrame({"Date": pd.to_datetime(["2024-01-01", "2024-01-02"]), "price_exec": [100.0, 101.0]})  # type: ignore[assignment]
    svc._load_stepe_logs = lambda step_e_root, symbol, agents: {  # type: ignore[assignment]
        "a1": pd.DataFrame({"Date": pd.to_datetime(["2024-01-01", "2024-01-02"]), "Split": ["train", "test"], "ratio": [1.0, 1.0], "stepE_ret_for_stats": [0.01, 0.01]})
    }

    def _raise_phase2(**kwargs):
        raise ValueError("phase2 build failed")

    svc._build_phase2_state = _raise_phase2  # type: ignore[assignment]

    date_range = SimpleNamespace(mode="sim", train_start="2024-01-01", train_end="2024-01-01", test_start="2024-01-02", test_end="2024-01-02")
    with pytest.raises(RuntimeError, match="failed for all modes"):
        svc.run(date_range, symbol="SOXL", mode="sim")


def test_cluster_phase2_uses_real_ticc_route(monkeypatch) -> None:
    svc = StepFService(app_config=SimpleNamespace())
    _FakeTICCClusterer.calls = []
    monkeypatch.setattr(sf_mod, "TICCClusterer", _FakeTICCClusterer)
    cfg = StepFRouterConfig(clusterer_type="ticc_raw20_stable", clusterer_fallback_type="none", raw20_num_clusters=20, k_eff_min=12, cluster_share_min=0.2, cluster_mean_run_min=2.0)

    x_train = np.ones((8, 2), dtype=float)
    x_test = np.ones((3, 2), dtype=float)
    out, diag = svc._cluster_phase2(cfg=cfg, x_train=x_train, x_test=x_test)

    assert diag["clusterer_type_requested"] == "ticc_raw20_stable"
    assert diag["clusterer_type_used"] == "raw20_stable"
    assert diag["fallback_used"] is False
    assert diag["raw20_num_clusters_requested"] == 20
    assert diag["k_valid"] >= 1
    assert diag["k_eff"] >= 12
    assert _FakeTICCClusterer.calls[0] == 20
    assert _FakeTICCClusterer.calls[1] == diag["k_eff"]
    assert np.array_equal(out["train_main"], out["train_stable"])
    assert int(np.max(out["train_rare_raw20"])) == 1


def test_cluster_phase2_ticc_unavailable_falls_back_to_none(monkeypatch) -> None:
    svc = StepFService(app_config=SimpleNamespace())

    class _BrokenTICC:
        def __init__(self, **kwargs):
            raise RuntimeError("ticc backend missing")

    monkeypatch.setattr(sf_mod, "TICCClusterer", _BrokenTICC)
    cfg = StepFRouterConfig(clusterer_type="ticc_raw20_stable", clusterer_fallback_type="none")

    out, diag = svc._cluster_phase2(
        cfg=cfg,
        x_train=np.ones((4, 2), dtype=float),
        x_test=np.ones((2, 2), dtype=float),
    )

    assert diag["clusterer_type_requested"] == "ticc_raw20_stable"
    assert diag["clusterer_type_used"] == "none"
    assert diag["fallback_used"] is True
    assert out["train_main"].tolist() == [0, 0, 0, 0]
    assert out["test_main"].tolist() == [0, 0]
    assert out["test_conf_stable"].tolist() == [1.0, 1.0]


def test_cluster_phase2_none_returns_single_regime() -> None:
    svc = StepFService(app_config=SimpleNamespace())
    cfg = StepFRouterConfig(clusterer_type="none")

    out, diag = svc._cluster_phase2(
        cfg=cfg,
        x_train=np.ones((3, 2), dtype=float),
        x_test=np.ones((1, 2), dtype=float),
    )

    assert diag["clusterer_type_used"] == "none"
    assert out["train_main"].tolist() == [0, 0, 0]
    assert out["test_main"].tolist() == [0]
    assert out["test_conf_stable"].tolist() == [1.0]


def test_stepf_service_source_has_no_kmeans_placeholder() -> None:
    src = Path(sf_mod.__file__).read_text(encoding="utf-8")
    assert "KMeans" not in src

def test_build_phase2_state_sets_regime_id_from_stable(monkeypatch) -> None:
    svc = StepFService(app_config=SimpleNamespace())
    _FakeTICCClusterer.calls = []
    monkeypatch.setattr(sf_mod, "TICCClusterer", _FakeTICCClusterer)

    def _fake_base_features(price_tech: pd.DataFrame, tech: pd.DataFrame) -> pd.DataFrame:
        n = len(price_tech)
        return pd.DataFrame(
            {
                "Date": pd.to_datetime(price_tech["Date"]),
                "Gap": np.linspace(0.0, 1.0, n),
                "ATR_norm": np.linspace(1.0, 2.0, n),
                "ret_1": np.linspace(-0.1, 0.1, n),
            }
        )

    monkeypatch.setattr(sf_mod, "_compute_base_features", _fake_base_features)
    cfg = StepFRouterConfig(
        clusterer_type="ticc_raw20_stable",
        clusterer_fallback_type="none",
        raw20_num_clusters=20,
        k_eff_min=12,
        cluster_share_min=0.2,
        cluster_mean_run_min=2.0,
        past_window_days=5,
        past_resample_len=4,
        pca_n_components=2,
        use_z_pred=False,
    )
    dates = pd.date_range("2024-01-01", periods=12, freq="D")
    price_tech = pd.DataFrame({"Date": dates, "price_exec": np.linspace(100.0, 112.0, len(dates)), "feat_a_tech": np.linspace(0.0, 2.0, len(dates))})
    date_range = SimpleNamespace(train_start="2024-01-01", train_end="2024-01-08", test_start="2024-01-09", test_end="2024-01-12")

    out = svc._build_phase2_state(cfg=cfg, date_range=date_range, symbol="SOXL", mode="sim", out_root=Path("."), price_tech=price_tech)
    assert (out["regime_id"].astype(int) == out["cluster_id_stable"].astype(int)).all()
    assert "cluster_id_raw20" in out.columns
    assert "rare_flag_raw20" in out.columns
    assert set(out["rare_flag_raw20"].astype(int).unique().tolist()).issubset({0, 1})


def test_stepf_router_config_default_fallback_is_none() -> None:
    cfg = StepFRouterConfig()
    assert cfg.clusterer_type == "ticc_raw20_stable"
    assert cfg.clusterer_fallback_type == "none"


def test_cluster_phase2_hdbscan_no_defined_clusters_falls_back_to_none(monkeypatch) -> None:
    svc = StepFService(app_config=SimpleNamespace())

    class _NoClusterer:
        def __init__(self, **kwargs):
            self.labels_ = np.array([], dtype=int)

        def fit(self, x_train):
            self.labels_ = np.full((x_train.shape[0],), -1, dtype=int)
            return self

    class _NoClusterPred:
        called = 0

        @staticmethod
        def approximate_predict(clusterer, x_test):
            _NoClusterPred.called += 1
            return np.zeros((x_test.shape[0],), dtype=int), np.ones((x_test.shape[0],), dtype=float)

    class _NoClusterModule:
        HDBSCAN = _NoClusterer

    monkeypatch.setattr(sf_mod, "hdbscan", _NoClusterModule)
    monkeypatch.setattr(sf_mod, "hdbscan_prediction", _NoClusterPred)

    def _fake_base_features(price_tech: pd.DataFrame, tech: pd.DataFrame) -> pd.DataFrame:
        n = len(price_tech)
        return pd.DataFrame(
            {
                "Date": pd.to_datetime(price_tech["Date"]),
                "Gap": np.linspace(0.0, 1.0, n),
                "ATR_norm": np.linspace(1.0, 2.0, n),
                "ret_1": np.linspace(-0.1, 0.1, n),
            }
        )

    monkeypatch.setattr(sf_mod, "_compute_base_features", _fake_base_features)

    out, diag = svc._cluster_phase2(
        cfg=StepFRouterConfig(clusterer_type="hdbscan", clusterer_fallback_type="hdbscan", hdbscan_min_cluster_size=2, hdbscan_min_samples=1),
        x_train=np.ones((5, 2), dtype=float),
        x_test=np.ones((3, 2), dtype=float),
    )

    assert diag["clusterer_type_requested"] == "hdbscan"
    assert diag["clusterer_type_used"] == "none"
    assert diag["fallback_used"] is True
    assert diag["fallback_reason"] == "hdbscan_no_defined_clusters"
    assert diag["hdbscan_defined_cluster_count"] == 0
    assert diag["hdbscan_all_noise"] is True
    assert _NoClusterPred.called == 0
    assert out["test_main"].tolist() == [0, 0, 0]
    assert out["test_conf_stable"].tolist() == [1.0, 1.0, 1.0]


def test_run_completes_and_writes_equity_csv_with_none_fallback(monkeypatch, tmp_path) -> None:
    cfg = StepFRouterConfig(
        output_root=str(tmp_path / "out"),
        agents="a1",
        reward_mode="legacy",
        clusterer_type="hdbscan",
        clusterer_fallback_type="hdbscan",
        hdbscan_min_cluster_size=2,
        hdbscan_min_samples=1,
    )
    app_config = SimpleNamespace(stepF=cfg, output_root=str(tmp_path / "out"))
    svc = StepFService(app_config=app_config)

    class _NoClusterer:
        def __init__(self, **kwargs):
            self.labels_ = np.array([], dtype=int)

        def fit(self, x_train):
            self.labels_ = np.full((x_train.shape[0],), -1, dtype=int)
            return self

    class _NoClusterPred:
        @staticmethod
        def approximate_predict(clusterer, x_test):
            raise AssertionError("approximate_predict should not be called when no defined clusters")

    class _NoClusterModule:
        HDBSCAN = _NoClusterer

    monkeypatch.setattr(sf_mod, "hdbscan", _NoClusterModule)
    monkeypatch.setattr(sf_mod, "hdbscan_prediction", _NoClusterPred)

    svc._load_stepa_price_tech = lambda out_root, mode, symbol: pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "price_exec": [100.0, 101.0, 102.0],
        "feat_a_tech": [0.1, 0.2, 0.3],
    })  # type: ignore[assignment]
    svc._load_stepe_logs = lambda step_e_root, symbol, agents: {
        "a1": pd.DataFrame({
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "Split": ["train", "test", "test"],
            "ratio": [1.0, 1.0, 1.0],
            "stepE_ret_for_stats": [0.01, 0.01, 0.01],
        })
    }  # type: ignore[assignment]

    def _phase2_none(**kwargs):
        svc._last_cluster_diag = {
            "clusterer_type_requested": "hdbscan",
            "clusterer_type_used": "none",
            "fallback_type": "hdbscan",
            "fallback_used": True,
            "fallback_reason": "hdbscan_no_defined_clusters",
            "hdbscan_defined_cluster_count": 0,
            "hdbscan_all_noise": True,
        }
        return pd.DataFrame({
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "regime_id": [0, 0, 0],
            "cluster_id_stable": [0, 0, 0],
            "cluster_id_raw20": [0, 0, 0],
            "rare_flag_raw20": [0, 0, 0],
            "confidence_stable": [1.0, 1.0, 1.0],
            "confidence_raw20": [1.0, 1.0, 1.0],
            "cluster_id_main": [0, 0, 0],
            "confidence": [1.0, 1.0, 1.0],
        })

    svc._build_phase2_state = _phase2_none  # type: ignore[assignment]

    date_range = SimpleNamespace(mode="sim", train_start="2024-01-01", train_end="2024-01-01", test_start="2024-01-02", test_end="2024-01-03")
    result = svc.run(date_range, symbol="SOXL", mode="sim")

    assert Path(result.daily_log_path).exists()
    eq_path = tmp_path / "out" / "stepF" / "sim" / "stepF_equity_marl_SOXL.csv"
    assert eq_path.exists()
    import json

    summary = json.loads((tmp_path / "out" / "stepF" / "sim" / "stepF_summary_router_SOXL.json").read_text(encoding="utf-8"))
    assert summary["clusterer_type_used"] == "none"
    assert summary["fallback_reason"] == "hdbscan_no_defined_clusters"
