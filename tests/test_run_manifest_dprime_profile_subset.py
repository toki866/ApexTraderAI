from __future__ import annotations

from pathlib import Path

from tools.run_manifest import check_step_artifacts


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")


def _write_dprime_profile(base: Path, profile: str, symbol: str = "SOXL") -> None:
    emb = base / "embeddings"
    marker_dir = base / "pipeline_markers"
    _touch(base / f"stepDprime_state_train_{profile}_{symbol}.csv")
    _touch(base / f"stepDprime_state_test_{profile}_{symbol}.csv")
    _touch(emb / f"stepDprime_{profile}_{symbol}_embeddings_test.csv")
    _touch(emb / f"stepDprime_{profile}_{symbol}_embeddings_all.csv")
    _touch(base / f"stepDprime_profile_summary_{profile}_{symbol}.json")
    _touch(marker_dir / f"DPrimeFinal_{profile}.READY.json")


def test_check_step_artifacts_dprime_honors_requested_profile_subset(tmp_path: Path) -> None:
    out = tmp_path / "out" / "stepDprime" / "sim"
    marker_dir = out / "pipeline_markers"
    _touch(out / "stepDprime_base_meta_SOXL.json")
    _touch(marker_dir / "DPrimeBaseCluster.READY.json")
    _write_dprime_profile(out, "dprime_bnf_h01")

    assert check_step_artifacts(
        "DPRIME",
        tmp_path / "out",
        "SOXL",
        "sim",
        required_dprime_profiles=("dprime_bnf_h01",),
    )
    assert not check_step_artifacts("DPRIME", tmp_path / "out", "SOXL", "sim")
