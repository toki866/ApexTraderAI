# -*- coding: utf-8 -*-
"""Per-step artifact contract validators for ApexTraderAI.

Each function returns a list of missing / invalid artifact paths.
Empty list means all required artifacts are present (PASS).

Usage in run_pipeline.py:
    from ai_core.utils.step_contract_utils import (
        validate_step_a, validate_step_b, validate_step_c,
        validate_step_dprime, validate_step_e_agent, validate_step_f,
    )
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import List, Sequence

from ai_core.utils.manifest_path_utils import resolve_output_artifact_path


def validate_step_a(output_root: Path, symbol: str, mode: str) -> List[str]:
    """Return missing artifact paths for StepA complete output set."""
    base = Path(output_root) / "stepA" / mode
    missing: List[str] = []

    for name in (
        f"stepA_prices_train_{symbol}.csv",
        f"stepA_prices_test_{symbol}.csv",
        f"stepA_periodic_train_{symbol}.csv",
        f"stepA_periodic_test_{symbol}.csv",
        f"stepA_tech_train_{symbol}.csv",
        f"stepA_tech_test_{symbol}.csv",
        f"stepA_split_summary_{symbol}.csv",
        f"stepA_periodic_future_{symbol}.csv",
        f"stepA_daily_manifest_{symbol}.csv",
    ):
        p = base / name
        if not p.exists():
            missing.append(str(p))

    split_summary_json = Path(output_root) / "split_summary.json"
    if not split_summary_json.exists():
        missing.append(str(split_summary_json))

    manifest_path = base / f"stepA_daily_manifest_{symbol}.csv"
    if manifest_path.exists():
        try:
            manifest = pd.read_csv(manifest_path)
            if manifest.empty:
                missing.append(f"{manifest_path}::empty")
            required_cols = ("prices_path", "periodic_path", "tech_path", "periodic_future_path")
            for col in required_cols:
                if col not in manifest.columns:
                    missing.append(f"{manifest_path}::missing_column({col})")
                    continue
                for raw in manifest[col].fillna("").astype(str):
                    resolved = resolve_output_artifact_path(raw, canonical_output_root=output_root)
                    if not raw.strip() or not resolved.exists():
                        missing.append(f"{manifest_path}::missing_path({col})={raw}")
                        break
        except Exception as e:
            missing.append(f"{manifest_path}::read_error={type(e).__name__}:{e}")

    return missing


def validate_step_b(output_root: Path, symbol: str, mode: str) -> List[str]:
    """Return missing artifact paths for StepB.

    Hard required: pred_time_all (used by DPrime/StepE downstream).
    Soft required: pred_close, pred_path.
    """
    base = Path(output_root) / "stepB" / mode
    missing: List[str] = []

    for name in (
        f"stepB_pred_time_all_{symbol}.csv",
        f"stepB_pred_close_mamba_{symbol}.csv",
        f"stepB_pred_path_mamba_{symbol}.csv",
        f"stepB_pred_close_mamba_periodic_{symbol}.csv",
        f"stepB_pred_path_mamba_periodic_{symbol}.csv",
        f"stepB_summary_{symbol}.json",
    ):
        p = base / name
        if not p.exists():
            missing.append(str(p))
    audit_path = Path(output_root) / "audit" / mode / f"stepB_audit_{symbol}.json"
    if not audit_path.exists():
        missing.append(str(audit_path))

    periodic_close = base / f"stepB_pred_close_mamba_periodic_{symbol}.csv"
    if periodic_close.exists():
        periodic_path = base / f"stepB_pred_path_mamba_periodic_{symbol}.csv"
        if not periodic_path.exists():
            missing.append(str(periodic_path))

    pred_time_all = base / f"stepB_pred_time_all_{symbol}.csv"
    stepa_test = Path(output_root) / "stepA" / mode / f"stepA_prices_test_{symbol}.csv"
    if pred_time_all.exists() and stepa_test.exists():
        try:
            pred_df = pd.read_csv(pred_time_all)
            test_df = pd.read_csv(stepa_test)
            if "Date" not in pred_df.columns:
                missing.append(f"{pred_time_all}::missing_column(Date)")
            elif "Date" not in test_df.columns:
                missing.append(f"{stepa_test}::missing_column(Date)")
            else:
                pred_df["Date"] = pd.to_datetime(pred_df["Date"], errors="coerce").dt.normalize()
                test_df["Date"] = pd.to_datetime(test_df["Date"], errors="coerce").dt.normalize()
                pred_col = "Pred_Close_MAMBA" if "Pred_Close_MAMBA" in pred_df.columns else "pred_close_mamba"
                if pred_col not in pred_df.columns:
                    missing.append(f"{pred_time_all}::missing_prediction_column")
                else:
                    merged = test_df[["Date"]].dropna().drop_duplicates().merge(
                        pred_df[["Date", pred_col]].dropna(subset=["Date"]),
                        on="Date",
                        how="left",
                    )
                    if merged.empty:
                        missing.append(f"{pred_time_all}::empty_test_window")
                    else:
                        vals = pd.to_numeric(merged[pred_col], errors="coerce")
                        nn = vals.notna().sum()
                        coverage = float(nn) / float(len(merged)) if len(merged) > 0 else 0.0
                        if coverage <= 0.0:
                            missing.append(f"{pred_time_all}::coverage_ratio_over_test={coverage:.4f}")
                        if vals.notna().sum() == 0:
                            missing.append(f"{pred_time_all}::all_nan")
        except Exception as e:
            missing.append(f"{pred_time_all}::read_or_coverage_error={type(e).__name__}:{e}")

    return missing


def validate_step_c(output_root: Path, symbol: str, mode: str) -> List[str]:
    """Return missing artifact paths for StepC.

    At least one CSV containing the symbol name must exist.
    """
    base = Path(output_root) / "stepC" / mode
    if not base.exists():
        return [str(base)]

    found = list(base.glob(f"*{symbol}*.csv"))
    if not found:
        return [str(base / f"*{symbol}*.csv (no match)")]

    return []


def validate_step_dprime(
    output_root: Path,
    mode: str,
    symbol: str,
    agents: Sequence[str],
) -> List[str]:
    """Return missing artifact paths for StepDPrime.

    Requires state_test CSV for every agent in *agents*.
    This mirrors (and replaces) _validate_stepdprime_contract in run_pipeline.py.
    """
    base = Path(output_root) / "stepDprime" / mode
    missing: List[str] = []

    for profile in agents:
        p = base / f"stepDprime_state_test_{profile}_{symbol}.csv"
        if not p.exists():
            missing.append(str(p))
        emb = base / "embeddings" / f"stepDprime_{profile}_{symbol}_embeddings_test.csv"
        if not emb.exists():
            missing.append(str(emb))

    return missing


def validate_step_e_agent(
    output_root: Path,
    symbol: str,
    mode: str,
    agent: str,
) -> List[str]:
    """Return missing artifact paths for a single StepE agent."""
    base = Path(output_root) / "stepE" / mode
    daily = base / f"stepE_daily_log_{agent}_{symbol}.csv"
    summary = base / f"stepE_summary_{agent}_{symbol}.json"
    model_pt = base / "models" / f"stepE_{agent}_{symbol}.pt"
    model_zip = base / "models" / f"stepE_{agent}_{symbol}_ppo.zip"
    audit = Path(output_root) / "audit" / mode / f"stepE_audit_{agent}_{symbol}.json"
    if not daily.exists():
        return [str(daily)]
    missing = []
    if not summary.exists():
        missing.append(str(summary))
    if not audit.exists():
        missing.append(str(audit))
    if not model_pt.exists() and not model_zip.exists():
        missing.append(f"{model_pt} | {model_zip}")
    if missing:
        return missing
    return []


def validate_step_f(output_root: Path, symbol: str, mode: str) -> List[str]:
    """Return missing artifact paths for StepF.

    Requires equity_marl, daily_log_marl, and daily_log_router.
    """
    base = Path(output_root) / "stepF" / mode
    missing: List[str] = []

    for name in (
        f"stepF_equity_marl_{symbol}.csv",
        f"stepF_daily_log_marl_{symbol}.csv",
        f"stepF_daily_log_router_{symbol}.csv",
        f"stepF_summary_router_{symbol}.json",
        f"stepF_audit_router_{symbol}.json",
    ):
        p = base / name
        if not p.exists():
            missing.append(str(p))
    audit_compare = Path(output_root) / "audit" / mode / f"stepF_policy_compare_{symbol}.json"
    if not audit_compare.exists():
        missing.append(str(audit_compare))

    return missing
