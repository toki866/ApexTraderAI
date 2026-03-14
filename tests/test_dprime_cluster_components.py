from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_core.services.dprime_cluster_components import ClusterMonthlyTrainer, ClusterRuntimeConfig


def test_cluster_monthly_trainer_handles_existing_cluster_id_raw20_series_name():
    features = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=12, freq="D"),
            "cluster_short_signal": [0.0, 1.0, 0.5, 1.5, 2.0, 1.8, 1.1, 0.8, 2.2, 2.4, 1.7, 0.2],
        }
    )
    cfg = ClusterRuntimeConfig(symbol="SOXL", cluster_raw_k=4)

    out = ClusterMonthlyTrainer().train(features, cfg)

    assert "train_df" in out
    assert "cluster_id_raw20" in out["train_df"].columns
    assert len(out["train_df"]) == len(features)
