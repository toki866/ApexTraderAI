from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from ai_core.services.step_e_service import StepEConfig, StepEService
from ai_core.services.stepdprime_path_utils import resolve_stepdprime_dir, stepdprime_read_candidates


def test_resolve_stepdprime_dir_prefers_canonical_for_write_and_legacy_for_read(tmp_path: Path) -> None:
    out_root = tmp_path / "output"
    legacy = out_root / "stepD_prime" / "sim"
    legacy.mkdir(parents=True)

    assert resolve_stepdprime_dir(out_root, "sim", for_write=True) == out_root / "stepDprime" / "sim"
    assert resolve_stepdprime_dir(out_root, "sim", for_write=False) == legacy
    assert stepdprime_read_candidates(out_root, "sim") == [out_root / "stepDprime" / "sim", legacy]


def test_stepe_embeddings_fallback_reads_legacy_dir(tmp_path: Path) -> None:
    out_root = tmp_path / "output"
    legacy_emb = out_root / "stepD_prime" / "sim" / "embeddings"
    legacy_emb.mkdir(parents=True)
    pd.DataFrame(
        {
            "Date": ["2024-01-02", "2024-01-03"],
            "emb_000": [0.1, 0.2],
            "emb_001": [0.3, 0.4],
        }
    ).to_csv(legacy_emb / "stepDprime_dprime_all_features_h01_SOXL_embeddings_all.csv", index=False)

    service = StepEService(SimpleNamespace())
    cfg = StepEConfig(agent="dprime_all_features_h01", use_stepd_prime=True, dprime_profile="dprime_all_features_h01")
    shared = service._build_stepdprime_shared_context(cfg=cfg, out_root=out_root, mode="sim")

    df = service._load_stepD_prime_embeddings(cfg, out_root=out_root, mode="sim", symbol="SOXL", shared_context=shared)

    assert shared["stepDprime_root"].endswith("stepD_prime/sim")
    assert shared["stepDprime_legacy_read"] is True
    assert list(df.columns) == ["Date", "dprime_dprime_all_features_h01_emb_000", "dprime_dprime_all_features_h01_emb_001"]


def test_stepe_embeddings_error_reports_checked_roots(tmp_path: Path) -> None:
    out_root = tmp_path / "output"
    service = StepEService(SimpleNamespace())
    cfg = StepEConfig(agent="dprime_all_features_h01", use_stepd_prime=True, dprime_profile="dprime_all_features_h01")

    with pytest.raises(FileNotFoundError) as exc_info:
        service._load_stepD_prime_embeddings(cfg, out_root=out_root, mode="sim", symbol="SOXL")

    message = str(exc_info.value)
    assert "checked_roots=" in message
    assert str(out_root / "stepDprime" / "sim") in message
    assert str(out_root / "stepD_prime" / "sim") in message
