from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


def _load_module():
    p = Path("scripts") / "evaluate_run_outputs.py"
    spec = importlib.util.spec_from_file_location("evaluate_run_outputs", p)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_stepf_split_from_csv_no_note(tmp_path: Path):
    mod = _load_module()
    out_root = tmp_path / "output"
    stepf = out_root / "stepF" / "sim"
    stepf.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {"Date": "2024-01-01", "Split": "train", "ret": 0.01, "equity": 1.01},
            {"Date": "2024-01-02", "Split": "test", "ret": 0.02, "equity": 1.03},
        ]
    ).to_csv(stepf / "stepF_equity_marl_SOXL.csv", index=False)

    report = mod.evaluate(str(out_root), "sim", "SOXL")
    row = report["stepF"]["rows"][0]
    assert row["split_source"] == "csv"
    assert row["test_days"] == 1
    assert "Split missing" not in row.get("note", "")


def test_stepf_split_from_summary_no_note(tmp_path: Path):
    mod = _load_module()
    out_root = tmp_path / "output"
    stepf = out_root / "stepF" / "sim"
    stepf.mkdir(parents=True, exist_ok=True)

    (out_root / "split_summary.json").write_text(json.dumps({"test_start": "2024-01-03"}), encoding="utf-8")
    pd.DataFrame(
        [
            {"Date": "2024-01-01", "ret": 0.01, "equity": 1.01},
            {"Date": "2024-01-03", "ret": 0.02, "equity": 1.03},
            {"Date": "2024-01-04", "ret": -0.01, "equity": 1.02},
        ]
    ).to_csv(stepf / "stepF_equity_marl_SOXL.csv", index=False)

    report = mod.evaluate(str(out_root), "sim", "SOXL")
    row = report["stepF"]["rows"][0]
    assert row["split_source"] == "summary"
    assert row["test_days"] == 2
    assert "Split missing" not in row.get("note", "")


def test_stepf_split_fallback_all_test_with_warning(tmp_path: Path):
    mod = _load_module()
    out_root = tmp_path / "output"
    stepf = out_root / "stepF" / "sim"
    stepf.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {"Date": None, "ret": 0.01, "equity": 1.01},
            {"Date": None, "ret": 0.02, "equity": 1.03},
        ]
    ).to_csv(stepf / "stepF_equity_marl_SOXL.csv", index=False)

    report = mod.evaluate(str(out_root), "sim", "SOXL")
    row = report["stepF"]["rows"][0]
    assert row["split_source"] == "fallback_all_test"
    assert row["test_days"] == 2
    assert "Split missing: evaluated all rows as test" in row.get("note", "")


def test_stepf_split_date_infer_without_warning(tmp_path: Path):
    mod = _load_module()
    out_root = tmp_path / "output"
    stepf = out_root / "stepF" / "sim"
    stepf.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {"Date": "2024-01-01", "ret": 0.0, "equity": 1.00},
            {"Date": "2024-01-02", "ret": 0.0, "equity": 1.00},
            {"Date": "2024-06-01", "ret": 0.01, "equity": 1.01},
            {"Date": "2024-06-02", "ret": 0.01, "equity": 1.02},
        ]
    ).to_csv(stepf / "stepF_equity_marl_SOXL.csv", index=False)

    report = mod.evaluate(str(out_root), "sim", "SOXL")
    row = report["stepF"]["rows"][0]
    assert row["split_source"] == "date_infer"
    assert row["test_days"] == 2
    assert "Split missing" not in row.get("note", "")
