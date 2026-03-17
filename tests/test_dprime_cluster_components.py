from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_core.services.dprime_cluster_components import ClusterArtifactManager, ClusterMonthlyTrainer, ClusterRuntimeConfig


def _calendar10_features(periods: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=periods, freq="D"),
            "ret_1": [0.0, 1.0, 0.5, 1.5, 2.0, 1.8, 1.1, 0.8, 2.2, 2.4, 1.7, 0.2][:periods],
            "ret_5": [0.2] * periods,
            "ret_20": [0.3] * periods,
            "ATR_norm": [0.4] * periods,
            "gap_atr": [0.5] * periods,
            "vol_log_ratio_20": [0.6] * periods,
            "dev_z_25": [0.7] * periods,
            "body_ratio": [0.8] * periods,
            "per_cal_year365_sin": [0.1] * periods,
            "per_cal_year365_cos": [0.9] * periods,
        }
    )


def test_cluster_monthly_trainer_handles_existing_cluster_id_raw20_series_name(monkeypatch: pytest.MonkeyPatch):
    features = _calendar10_features(periods=12)
    cfg = ClusterRuntimeConfig(symbol="SOXL", cluster_raw_k=4, cluster_k_eff_min=2)

    captured = {}

    def _fake_fit_predict_train(self, x_train):
        captured["shape"] = list(x_train.shape)
        return [i % 2 for i in range(len(x_train))]

    monkeypatch.setattr("ai_core.clusterers.ticc_clusterer.TICCClusterer.fit_predict_train", _fake_fit_predict_train)
    monkeypatch.setattr(
        "ai_core.clusterers.ticc_clusterer.TICCClusterer.get_diagnostics",
        lambda self: {
            "backend_resolved_name": "fake",
            "backend_entrypoint_name": "fake.fit",
            "backend_entrypoint_kind": "method",
            "backend_api_candidates": ["fit_predict_train"],
            "backend_predict_methods": ["fit_predict_train"],
            "backend_methods": ["fit_predict_train"],
        },
    )

    out = ClusterMonthlyTrainer().train(features, cfg)

    assert "train_df" in out
    assert "cluster_id_raw20" in out["train_df"].columns
    assert len(out["train_df"]) == len(features)
    assert out["ticc_feature_set_name"] == "calendar10"
    assert out["ticc_feature_cols"] == [
        "ret_1",
        "ret_5",
        "ret_20",
        "ATR_norm",
        "gap_atr",
        "vol_log_ratio_20",
        "dev_z_25",
        "body_ratio",
        "per_cal_year365_sin",
        "per_cal_year365_cos",
    ]
    assert out["ticc_feature_count"] == 10
    assert out["ticc_train_shape"] == [len(features), 10]
    assert captured["shape"] == [len(features), 10]


def test_cluster_monthly_trainer_raises_when_calendar10_column_missing():
    features = _calendar10_features(periods=8).drop(columns=["per_cal_year365_cos"])
    cfg = ClusterRuntimeConfig(symbol="SOXL", cluster_raw_k=4)

    with pytest.raises(
        ValueError,
        match=r"Missing required TICC feature columns for feature_set=calendar10: \['per_cal_year365_cos'\]",
    ):
        ClusterMonthlyTrainer().train(features, cfg)


def test_cluster_monthly_trainer_allows_core8_when_explicitly_selected(monkeypatch: pytest.MonkeyPatch):
    features = _calendar10_features(periods=12)
    cfg = ClusterRuntimeConfig(symbol="SOXL", cluster_raw_k=4, cluster_k_eff_min=2)
    captured = {}

    def _fake_fit_predict_train(self, x_train):
        captured["shape"] = list(x_train.shape)
        return [i % 2 for i in range(len(x_train))]

    monkeypatch.setattr("ai_core.clusterers.ticc_clusterer.TICCClusterer.fit_predict_train", _fake_fit_predict_train)
    monkeypatch.setattr(
        "ai_core.clusterers.ticc_clusterer.TICCClusterer.get_diagnostics",
        lambda self: {
            "backend_resolved_name": "fake",
            "backend_entrypoint_name": "fake.fit",
            "backend_entrypoint_kind": "method",
            "backend_api_candidates": ["fit_predict_train"],
            "backend_predict_methods": ["fit_predict_train"],
            "backend_methods": ["fit_predict_train"],
        },
    )

    out = ClusterMonthlyTrainer().train(features, cfg, feature_set="core8")
    assert out["ticc_feature_set_name"] == "core8"
    assert out["ticc_feature_count"] == 8
    assert out["ticc_train_shape"] == [len(features), 8]
    assert captured["shape"] == [len(features), 8]


def test_cluster_artifact_manager_writes_ticc_feature_metadata(tmp_path: Path):
    symbol = "SOXL"
    mode = "sim"
    stepd_dir = tmp_path / "stepDprime" / mode
    stepd_dir.mkdir(parents=True)

    daily = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=3, freq="D"),
            "cluster_id_raw20": [0, 1, 1],
            "cluster_id_stable": [0, 1, 1],
            "rare_flag_raw20": [0, 0, 0],
            "year_month": ["2024-01", "2024-01", "2024-01"],
        }
    )
    monthly = {
        "status": "live",
        "note": "ok",
        "k_valid": 2,
        "k_eff": 12,
        "small_clusters": [],
        "stable_map": {0: 0, 1: 1},
        "ticc_feature_set_name": "calendar10",
        "ticc_feature_cols": [
            "ret_1",
            "ret_5",
            "ret_20",
            "ATR_norm",
            "gap_atr",
            "vol_log_ratio_20",
            "dev_z_25",
            "body_ratio",
            "per_cal_year365_sin",
            "per_cal_year365_cos",
        ],
        "ticc_feature_count": 10,
        "ticc_train_shape": [3, 10],
    }
    cfg = ClusterRuntimeConfig(symbol=symbol)

    ClusterArtifactManager(stepd_dir=stepd_dir, mode=mode, symbol=symbol).write(daily=daily, monthly=monthly, cfg=cfg)

    summary = pd.read_json(stepd_dir / "cluster" / mode / f"cluster_summary_{symbol}.json", typ="series")
    manifest = pd.read_json(stepd_dir / "cluster" / mode / f"cluster_feature_manifest_{symbol}.json", typ="series")
    assert summary["ticc_feature_set_name"] == "calendar10"
    assert len(summary["ticc_feature_cols"]) == 10
    assert int(summary["ticc_feature_count"]) == 10
    assert summary["ticc_train_shape"] == [3, 10]
    assert manifest["ticc_feature_set_name"] == "calendar10"
    assert len(manifest["ticc_feature_cols"]) == 10
    assert int(manifest["ticc_feature_count"]) == 10
    assert manifest["ticc_train_shape"] == [3, 10]
