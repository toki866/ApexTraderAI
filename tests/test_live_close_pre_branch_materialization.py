from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.mark.skipif(not Path("output").exists(), reason="no output artifacts")
def test_live_close_pre_branch_materialization_smoke():
    pytest.importorskip("hdbscan")
    pytest.importorskip("torch")
    from ai_core.live.stepf_two_stage_router import run_close_pre

    symbol = "SOXL"
    mode = "sim"
    stepa_test = Path("output") / "stepA" / mode / f"stepA_prices_test_{symbol}.csv"
    if not stepa_test.exists():
        pytest.skip("missing StepA test artifact")

    df = pd.read_csv(stepa_test)
    if df.empty:
        pytest.skip("empty StepA test artifact")
    target_date = str(pd.to_datetime(df["Date"], errors="coerce").dropna().dt.normalize().iloc[-1].date())

    decision = run_close_pre(
        symbol=symbol,
        mode=mode,
        target_date=target_date,
        output_root="output",
        config={
            "fit_window_days": 252,
            "stage0_topk": 3,
            "safe_branches": "dprime_bnf_h01,dprime_all_features_h01",
            "refresh_stepb": False,
            "refresh_dprime": True,
        },
    )
    assert np.isfinite(float(decision["ratio_final"]))
    assert len(decision.get("ratios", {})) < 10

    for profile in decision.get("dprime", {}).get("executed_profiles", []):
        p = Path("output") / "stepDprime" / mode / f"stepDprime_state_test_{profile}_{symbol}.csv"
        assert p.exists(), f"missing {p}"

    out_dir = Path("output") / "stepF" / mode / "live_close_pre"
    day_tag = pd.to_datetime(target_date).strftime("%Y%m%d")
    decision_path = out_dir / f"decision_{symbol}_{day_tag}.json"
    csv_path = out_dir / f"decisions_{symbol}.csv"
    state_path = out_dir / f"state_{symbol}.json"
    assert decision_path.exists()
    assert csv_path.exists()
    assert state_path.exists()

    loaded = json.loads(decision_path.read_text(encoding="utf-8"))
    assert np.isfinite(float(loaded["ratio_final"]))
