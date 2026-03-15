from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_core.services.dprime_cluster_components import ClusterArtifactManager, ClusterMonthlyTrainer, ClusterRuntimeConfig


def test_cluster_monthly_trainer_handles_existing_cluster_id_raw20_series_name(monkeypatch: pytest.MonkeyPatch):
    features = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=12, freq="D"),
            "ret_1": [0.0, 1.0, 0.5, 1.5, 2.0, 1.8, 1.1, 0.8, 2.2, 2.4, 1.7, 0.2],
        }
    )
    cfg = ClusterRuntimeConfig(symbol="SOXL", cluster_raw_k=4)

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
    assert out["ticc_feature_cols"] == ["ret_1"]
    assert out["ticc_feature_count"] == 1
    assert out["ticc_train_shape"] == [len(features), 1]
    assert captured["shape"] == [len(features), 1]


def test_cluster_monthly_trainer_raises_when_ret1_missing():
    features = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=8, freq="D"),
            "ret_5": [0.1] * 8,
        }
    )
    cfg = ClusterRuntimeConfig(symbol="SOXL", cluster_raw_k=4)

    with pytest.raises(ValueError, match="TICC required feature missing: ret_1"):
        ClusterMonthlyTrainer().train(features, cfg)


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
        "ticc_feature_cols": ["ret_1"],
        "ticc_feature_count": 1,
        "ticc_train_shape": [3, 1],
    }
    cfg = ClusterRuntimeConfig(symbol=symbol)

    ClusterArtifactManager(stepd_dir=stepd_dir, mode=mode, symbol=symbol).write(daily=daily, monthly=monthly, cfg=cfg)

    summary = pd.read_json(stepd_dir / "cluster" / mode / f"cluster_summary_{symbol}.json", typ="series")
    manifest = pd.read_json(stepd_dir / "cluster" / mode / f"cluster_feature_manifest_{symbol}.json", typ="series")
    assert summary["ticc_feature_cols"] == ["ret_1"]
    assert int(summary["ticc_feature_count"]) == 1
    assert summary["ticc_train_shape"] == [3, 1]
    assert manifest["ticc_feature_cols"] == ["ret_1"]
    assert int(manifest["ticc_feature_count"]) == 1
    assert manifest["ticc_train_shape"] == [3, 1]
