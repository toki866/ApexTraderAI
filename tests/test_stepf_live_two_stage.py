from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _find_model_and_log(symbol: str = "SOXL", mode: str = "sim"):
    model_dir = Path("output") / "stepE" / mode / "models"
    if not model_dir.exists():
        return None, None, None
    for pt in sorted(model_dir.glob(f"stepE_*_{symbol}.pt")):
        agent = pt.stem[len("stepE_") : -len(f"_{symbol}")]
        logp = Path("output") / "stepE" / mode / f"stepE_daily_log_{agent}_{symbol}.csv"
        if logp.exists():
            return agent, pt, logp
    return None, None, None


def test_stepe_policy_matches_log_position():
    pytest.importorskip("torch")
    from ai_core.live.feature_store import FeatureStore
    from ai_core.live.step_e_policy import StepEPolicy

    agent, model_path, log_path = _find_model_and_log()
    if model_path is None:
        pytest.skip("no local StepE model/log artifacts")

    policy = StepEPolicy(model_path=model_path)
    fs = FeatureStore(output_root="output", mode="sim", symbol="SOXL")

    log_df = pd.read_csv(log_path)
    log_df["Date"] = pd.to_datetime(log_df["Date"], errors="coerce").dt.normalize()
    ratio_col = "ratio" if "ratio" in log_df.columns else ("pos" if "pos" in log_df.columns else "Position")
    assert ratio_col in log_df.columns

    sample = log_df.tail(10).copy()
    errs = []
    pos_prev = 0.0
    for r in sample.itertuples(index=False):
        obs = fs.get_row(getattr(r, "Date"), policy.obs_cols)
        pred = policy.predict(obs, pos_prev=pos_prev)
        truth = float(getattr(r, ratio_col))
        errs.append(abs(pred - truth))
        pos_prev = truth

    assert np.nanmean(errs) < 1e-2, f"agent={agent} mean_abs_err={np.nanmean(errs):.6f}"


def test_close_pre_smoke():
    pytest.importorskip("hdbscan")
    pytest.importorskip("torch")
    from ai_core.live.stepf_two_stage_router import run_close_pre

    stepa = Path("output") / "stepA" / "sim"
    if not stepa.exists():
        pytest.skip("no local stepA artifacts")
    p = stepa / "stepA_prices_test_SOXL.csv"
    if not p.exists():
        pytest.skip("missing stepA price test")
    df = pd.read_csv(p)
    if df.empty:
        pytest.skip("empty stepA price test")
    d = pd.to_datetime(df["Date"], errors="coerce").dropna().dt.normalize().iloc[0]

    decision = run_close_pre(symbol="SOXL", mode="sim", target_date=str(d.date()), output_root="output", config={"fit_window_days": 252})
    assert np.isfinite(float(decision["ratio_final"]))
    assert len(decision.get("ratios", {})) >= len(decision.get("stage1", {}).get("allowed_agents_final", []))
