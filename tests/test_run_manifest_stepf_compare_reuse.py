from __future__ import annotations

from pathlib import Path

from tools.run_manifest import check_step_artifacts


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")


def _write_stepf_primary(base: Path, symbol: str = "SOXL") -> None:
    _touch(base / f"stepF_equity_marl_{symbol}.csv")
    _touch(base / f"stepF_daily_log_marl_{symbol}.csv")
    _touch(base / f"stepF_daily_log_router_{symbol}.csv")
    _touch(base / f"stepF_summary_router_{symbol}.json")


def _write_reward_mode(base: Path, mode_name: str, symbol: str = "SOXL") -> None:
    rdir = base / f"reward_{mode_name}"
    _touch(rdir / f"stepF_equity_marl_{symbol}.csv")
    _touch(rdir / f"stepF_daily_log_marl_{symbol}.csv")
    _touch(rdir / f"stepF_daily_log_router_{symbol}.csv")
    _touch(rdir / f"stepF_summary_router_{symbol}.json")


def test_check_step_artifacts_stepf_compare_requires_all_modes(tmp_path: Path) -> None:
    out = tmp_path / "out" / "stepF" / "sim"
    _write_stepf_primary(out)
    _write_reward_mode(out, "legacy")

    modes = ("legacy", "profit_basic", "profit_regret", "profit_light_risk")
    assert not check_step_artifacts("F", tmp_path / "out", "SOXL", "sim", required_stepf_reward_modes=modes)

    _write_reward_mode(out, "profit_basic")
    _write_reward_mode(out, "profit_regret")
    _write_reward_mode(out, "profit_light_risk")
    assert check_step_artifacts("F", tmp_path / "out", "SOXL", "sim", required_stepf_reward_modes=modes)
