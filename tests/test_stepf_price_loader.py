from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from ai_core.services.step_f_service import StepFService


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_load_stepa_price_tech_prefers_stepa_split(tmp_path):
    out_root = tmp_path / "output"
    base = out_root / "stepA" / "sim"
    _write_csv(
        base / "stepA_prices_train_SOXL.csv",
        [{"Date": "2022-01-03", "Close": 10.0}],
    )
    _write_csv(
        base / "stepA_prices_test_SOXL.csv",
        [{"Date": "2022-01-04", "Close": 11.0}],
    )
    _write_csv(base / "stepA_tech_train_SOXL.csv", [{"Date": "2022-01-03", "feat": 1.0}])
    _write_csv(base / "stepA_tech_test_SOXL.csv", [{"Date": "2022-01-04", "feat": 2.0}])

    svc = StepFService(app_config=SimpleNamespace(data_dir=str(tmp_path / "data")))
    df = svc._load_stepa_price_tech(out_root=out_root, mode="sim", symbol="SOXL")

    assert list(df["Date"].dt.strftime("%Y-%m-%d")) == ["2022-01-03", "2022-01-04"]
    assert list(df["price_exec"].astype(float)) == [10.0, 11.0]
    assert "feat" in df.columns


def test_load_stepa_price_tech_falls_back_to_raw_prices(tmp_path):
    out_root = tmp_path / "output"
    data_dir = tmp_path / "data"
    _write_csv(
        data_dir / "prices_SOXS.csv",
        [
            {"Date": "2022-01-03", "Close": 20.0},
            {"Date": "2022-01-04", "Close": 21.0},
        ],
    )

    svc = StepFService(app_config=SimpleNamespace(data_dir=str(data_dir)))
    df = svc._load_stepa_price_tech(out_root=out_root, mode="sim", symbol="SOXS")

    assert list(df["Date"].dt.strftime("%Y-%m-%d")) == ["2022-01-03", "2022-01-04"]
    assert list(df["price_exec"].astype(float)) == [20.0, 21.0]
