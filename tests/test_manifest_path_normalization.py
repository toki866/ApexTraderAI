from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from ai_core.services.step_a_service import StepAService
from ai_core.utils.manifest_path_utils import resolve_output_artifact_path
from ai_core.utils.step_contract_utils import validate_step_a
from tools.run_manifest import build_canonical_output_root, resolve_canonical_output_root, validate_step_outputs
from tools.stepb_daily_cache_utils import repair_stepb_daily_from_pred_path


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_stepa_daily_manifest_writer_uses_output_relative_paths(tmp_path: Path) -> None:
    output_root = tmp_path / "runs" / "20260319_010203" / "output"
    service = StepAService(SimpleNamespace(output_root=str(output_root), data_dir=str(tmp_path / "data")))
    service._get_us_trading_days = lambda start, end: [pd.Timestamp("2024-01-03")]  # type: ignore[method-assign]
    service._periodic_features_by_date = lambda start, end: pd.DataFrame(  # type: ignore[method-assign]
        {"Date": [pd.Timestamp("2024-01-03")], "per_cycle": [1.0]}
    )

    out_dir_mode = output_root / "stepA" / "sim"
    out_dir_mode.mkdir(parents=True, exist_ok=True)
    df_daily = pd.DataFrame(
        [
            {
                "Date": "2024-01-02",
                "Open": 10.0,
                "High": 11.0,
                "Low": 9.5,
                "Close": 10.5,
                "Volume": 1000,
                "per_cycle": 0.5,
                "tech_alpha": 1.2,
            }
        ]
    )

    service._write_daily_snapshots(
        out_dir_mode=out_dir_mode,
        symbol="SOXL",
        df_daily_source=df_daily,
        prices_cols=["Date", "Open", "High", "Low", "Close", "Volume"],
        per_cols=["per_cycle"],
        tech_cols=["tech_alpha"],
        scope="test",
        df_full_for_window=df_daily,
        lookback=2,
        future_months=1,
    )

    manifest = pd.read_csv(out_dir_mode / "stepA_daily_manifest_SOXL.csv")
    row = manifest.iloc[0].to_dict()
    for col in (
        "prices_path",
        "periodic_path",
        "tech_path",
        "features_path",
        "window_features_path",
        "periodic_future_path",
    ):
        value = str(row[col])
        assert value.startswith("stepA/sim/")
        assert "runs/" not in value


def test_stepa_manifest_validation_accepts_legacy_run_paths(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    stepa_dir = output_root / "stepA" / "sim"
    daily_dir = stepa_dir / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)

    for name in (
        "stepA_prices_train_SOXL.csv",
        "stepA_prices_test_SOXL.csv",
        "stepA_periodic_train_SOXL.csv",
        "stepA_periodic_test_SOXL.csv",
        "stepA_tech_train_SOXL.csv",
        "stepA_tech_test_SOXL.csv",
        "stepA_split_summary_SOXL.csv",
        "stepA_periodic_future_SOXL.csv",
    ):
        _write_csv(stepa_dir / name, [{"Date": "2024-01-02", "Open": 1, "High": 1, "Low": 1, "Close": 1, "Volume": 1}])

    output_root.joinpath("split_summary.json").write_text("{}", encoding="utf-8")

    filenames = {
        "prices_path": "stepA_prices_SOXL_2024_01_02.csv",
        "periodic_path": "stepA_periodic_SOXL_2024_01_02.csv",
        "tech_path": "stepA_tech_SOXL_2024_01_02.csv",
        "periodic_future_path": "stepA_periodic_future_SOXL_2024_01_02_m1.csv",
    }
    for filename in filenames.values():
        _write_csv(daily_dir / filename, [{"Date": "2024-01-02", "value": 1}])

    legacy_root = tmp_path / "runs" / "old_run" / "output"
    manifest_row = {"Date": "2024-01-02", "scope": "test"}
    for col, filename in filenames.items():
        manifest_row[col] = str(legacy_root / "stepA" / "sim" / "daily" / filename)

    _write_csv(stepa_dir / "stepA_daily_manifest_SOXL.csv", [manifest_row])

    assert validate_step_outputs("A", output_root, "SOXL", "sim") == (True, "reuse")
    assert validate_step_a(output_root, "SOXL", "sim") == []
    assert resolve_output_artifact_path(
        manifest_row["prices_path"],
        canonical_output_root=output_root,
    ) == daily_dir / filenames["prices_path"]


def test_stepb_repair_resolves_relative_manifest_paths(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    stepb_dir = repo_root / "output" / "stepB" / "sim"
    stepb_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(
        stepb_dir / "stepB_pred_path_mamba_SOXL.csv",
        [{"Date_anchor": "2024-01-02", "Pred_Close_t_plus_01": 123.45}],
    )
    _write_csv(
        stepb_dir / "stepB_daily_manifest_SOXL.csv",
        [{"Date": "2024-01-02", "pred_path_h01": "stepB/sim/daily/stepB_daily_pred_mamba_h01_SOXL_2024_01_02.csv", "stepA_features_path": "stepA/sim/daily/stepA_daily_features_SOXL_2024_01_02.csv"}],
    )

    logs: list[str] = []
    repair_stepb_daily_from_pred_path(repo_root, "SOXL", "sim", logs)

    repaired = stepb_dir / "daily" / "stepB_daily_pred_mamba_h01_SOXL_2024_01_02.csv"
    assert repaired.exists()
    repaired_df = pd.read_csv(repaired)
    assert float(repaired_df.loc[0, "Pred_Close"]) == 123.45


def test_canonical_output_root_uses_test_start_without_generation_suffix() -> None:
    root = build_canonical_output_root(Path("C:/work/apex_work/output"), "sim", "SOXL", "2022-01-03")
    assert root.as_posix() == "C:/work/apex_work/output/sim/SOXL/2022-01-03"


def test_resolve_canonical_output_root_uses_fixed_apex_work_base() -> None:
    root = resolve_canonical_output_root("sim", "SOXL", "2022-01-03")
    assert root.as_posix().endswith("/work/apex_work/output/sim/SOXL/2022-01-03")
