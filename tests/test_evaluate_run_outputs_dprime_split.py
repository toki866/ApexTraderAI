from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    p = Path("scripts") / "evaluate_run_outputs.py"
    spec = importlib.util.spec_from_file_location("evaluate_run_outputs", p)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x\n", encoding="utf-8")


def test_dprime_split_rl_and_cluster_present(tmp_path: Path):
    mod = _load_module()
    base = tmp_path / "output" / "stepDprime" / "sim"
    _touch(base / "stepDprime_state_test_dprime_all_features_h01_SOXL.csv")
    _touch(base / "embeddings" / "stepDprime_cluster_SOXL_embeddings.csv")

    got = mod._collect_dprime_artifacts(str(tmp_path / "output"), "sim", "SOXL")
    d = got["details"]
    assert d["rl_status"] == "OK"
    assert d["cluster_status"] == "OK"
    assert d["rl_state_count"] == 1
    assert d["cluster_embeddings_count"] == 1


def test_dprime_split_rl_only(tmp_path: Path):
    mod = _load_module()
    base = tmp_path / "output" / "stepDprime" / "sim"
    _touch(base / "stepDprime_state_test_dprime_mix_h01_SOXL.csv")

    got = mod._collect_dprime_artifacts(str(tmp_path / "output"), "sim", "SOXL")
    d = got["details"]
    assert d["rl_status"] == "OK"
    assert d["cluster_status"] == "WARN"


def test_dprime_split_cluster_only(tmp_path: Path):
    mod = _load_module()
    base = tmp_path / "output" / "stepDprime" / "sim"
    _touch(base / "embeddings" / "stepDprime_cluster_SOXL_embeddings.csv")

    got = mod._collect_dprime_artifacts(str(tmp_path / "output"), "sim", "SOXL")
    d = got["details"]
    assert d["rl_status"] == "WARN"
    assert d["cluster_status"] == "OK"


def test_dprime_split_both_missing(tmp_path: Path):
    mod = _load_module()
    got = mod._collect_dprime_artifacts(str(tmp_path / "output"), "sim", "SOXL")
    d = got["details"]
    assert d["rl_status"] == "WARN"
    assert d["cluster_status"] == "WARN"
    assert d["rl_state_count"] == 0
    assert d["cluster_embeddings_count"] == 0


def test_dprime_split_legacy_stepdprime_case_insensitive_dir(tmp_path: Path):
    mod = _load_module()
    base = tmp_path / "output" / "stepDPrime" / "sim"
    _touch(base / "stepDprime_state_test_dprime_all_features_h01_SOXL.csv")

    got = mod._collect_dprime_artifacts(str(tmp_path / "output"), "sim", "SOXL")
    d = got["details"]
    assert d["rl_status"] == "OK"
    assert d["rl_state_count"] == 1
