from __future__ import annotations

import pandas as pd

from ai_core.utils.leak_audit_utils import audit_stepE_reward_alignment, audit_stepF_market_alignment


def test_stepe_reward_alignment_zero_baseline_does_not_false_fail() -> None:
    df = pd.DataFrame(
        [
            {"Date": "2024-01-01", "Split": "test", "ret": 0.0, "reward_next": 0.0, "equity": 0.0, "pos": 0.0, "r_soxl_next": 0.0, "r_soxs_next": 0.0, "cost": 0.0},
            {"Date": "2024-01-02", "Split": "test", "ret": 0.0, "reward_next": 0.0, "equity": 0.0, "pos": 0.0, "r_soxl_next": 0.0, "r_soxs_next": 0.0, "cost": 0.0},
        ]
    )

    audit = audit_stepE_reward_alignment(df, split="test")

    assert audit["max_abs"] == 0.0
    assert audit["status"] == "PASS"


def test_stepf_market_alignment_uses_same_day_market_return() -> None:
    df = pd.DataFrame(
        [
            {"Date": "2024-01-01", "Split": "test", "ratio": 1.0, "ret": 0.01, "cost": 0.0, "equity": 1.01, "r_soxl": 0.01, "r_soxs": -0.01},
            {"Date": "2024-01-02", "Split": "test", "ratio": -0.5, "ret": 0.01, "cost": 0.0, "equity": 1.0201, "r_soxl": -0.02, "r_soxs": 0.02},
        ]
    )

    audit = audit_stepF_market_alignment(df, split="test")

    assert audit["max_abs"] == 0.0
    assert audit["status"] == "PASS"
