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

from pathlib import Path
from typing import List, Sequence


def validate_step_a(output_root: Path, symbol: str, mode: str) -> List[str]:
    """Return missing artifact paths for StepA.

    Hard required: prices_train, prices_test.
    Soft required (warn-only, included in return list): periodic, tech, split_summary.
    """
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
    ):
        p = base / name
        if not p.exists():
            missing.append(str(p))

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
        f"stepB_pred_close_{symbol}.csv",
        f"stepB_pred_path_{symbol}.csv",
    ):
        p = base / name
        if not p.exists():
            missing.append(str(p))

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

    return missing


def validate_step_e_agent(
    output_root: Path,
    symbol: str,
    mode: str,
    agent: str,
) -> List[str]:
    """Return missing artifact paths for a single StepE agent."""
    p = Path(output_root) / "stepE" / mode / f"stepE_daily_log_{agent}_{symbol}.csv"
    if not p.exists():
        return [str(p)]
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
    ):
        p = base / name
        if not p.exists():
            missing.append(str(p))

    return missing
