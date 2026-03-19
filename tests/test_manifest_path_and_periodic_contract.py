from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from ai_core.services.step_b_service import StepBService
from ai_core.utils.pipeline_artifact_utils import normalize_output_artifact_path, resolve_output_artifact_path
from tools import run_manifest as run_manifest_module


def _touch_csv(path: Path, columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{c: 1 for c in (columns or ["Date"])}])
    if "Date" in df.columns:
        df["Date"] = "2024-01-02"
    df.to_csv(path, index=False)


def test_normalize_output_artifact_path_rewrites_runs_prefix() -> None:
    canonical = Path("/canonical/output")
    raw = "/tmp/runs/run-123/output/stepA/sim/daily/stepA_daily_features_SOXL_2024_01_02.csv"

    normalized = normalize_output_artifact_path(raw, canonical_output_root=canonical, prefer_relative=True)

    assert normalized == "stepA/sim/daily/stepA_daily_features_SOXL_2024_01_02.csv"
    assert resolve_output_artifact_path(normalized, output_root=canonical) == canonical / normalized


def test_validate_stepa_daily_manifest_accepts_relative_paths(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    stepa_dir = output_root / "stepA" / "sim"
    daily_dir = stepa_dir / "daily"
    for stem in ("prices", "periodic", "tech"):
        _touch_csv(daily_dir / f"stepA_{stem}_SOXL_2024_01_02.csv", ["Date", stem])
    _touch_csv(daily_dir / "stepA_periodic_future_SOXL_2024_01_02_m3.csv", ["Date", "per_cal_01"])
    manifest = pd.DataFrame(
        [
            {
                "Date": "2024-01-02",
                "prices_path": "stepA/sim/daily/stepA_prices_SOXL_2024_01_02.csv",
                "periodic_path": "stepA/sim/daily/stepA_periodic_SOXL_2024_01_02.csv",
                "tech_path": "stepA/sim/daily/stepA_tech_SOXL_2024_01_02.csv",
                "periodic_future_path": "stepA/sim/daily/stepA_periodic_future_SOXL_2024_01_02_m3.csv",
            }
        ]
    )
    stepa_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(stepa_dir / "stepA_daily_manifest_SOXL.csv", index=False)

    assert run_manifest_module._validate_stepa_daily_manifest(output_root, "SOXL", "sim")


def test_check_step_artifacts_dprime_accepts_legacy_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = tmp_path / "output" / "stepD_prime" / "sim"
    profile = "dprime_all_features_h01"
    monkeypatch.setattr(run_manifest_module, "_OFFICIAL_STEPE_AGENTS", (profile,))
    _touch_csv(base / f"stepDprime_state_test_{profile}_SOXL.csv", ["Date", "state"])
    _touch_csv(base / "embeddings" / f"stepDprime_{profile}_SOXL_embeddings_test.csv", ["Date", "emb_000"])

    assert run_manifest_module.check_step_artifacts("DPRIME", tmp_path / "output", "SOXL", "sim")


def test_stepb_periodic_only_features_extracts_exact_44_columns() -> None:
    service = StepBService(SimpleNamespace())
    data = {"Date": pd.date_range("2024-01-01", periods=3)}
    for idx in range(44):
        data[f"per_feature_{idx:02d}"] = [float(idx)] * 3
    data["noise_feature"] = [999.0] * 3
    df = pd.DataFrame(data)

    periodic_only = service._extract_periodic_only_features(df)

    assert list(periodic_only.columns) == ["Date", *[f"per_feature_{idx:02d}" for idx in range(44)]]


def test_stepb_periodic_only_features_fails_when_not_44_columns() -> None:
    service = StepBService(SimpleNamespace())
    data = {"Date": pd.date_range("2024-01-01", periods=3)}
    for idx in range(43):
        data[f"per_feature_{idx:02d}"] = [float(idx)] * 3
    df = pd.DataFrame(data)

    with pytest.raises(RuntimeError, match=r"feature_dim=43"):
        service._extract_periodic_only_features(df)
