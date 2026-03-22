from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_core.utils.cluster_eval import run_cluster_evaluation
from scripts.evaluate_run_outputs import evaluate


def _cum_equity(ret: pd.Series) -> pd.Series:
    return (1.0 + pd.to_numeric(ret, errors="coerce").fillna(0.0)).cumprod()


def test_cluster_eval_subsystem_writes_canonical_artifacts(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    mode = "sim"
    symbol = "SOXL"
    stepd_root = output_root / "stepDprime" / mode
    cluster_root = stepd_root / "cluster" / mode
    stepe_root = output_root / "stepE" / mode
    stepf_root = output_root / "stepF" / mode
    for path in [cluster_root, stepe_root, stepf_root]:
        path.mkdir(parents=True, exist_ok=True)

    dates = pd.bdate_range("2024-01-02", periods=85)
    n = len(dates)
    base_ret = np.where(np.arange(n) < 30, 0.003, np.where(np.arange(n) < 60, -0.002, 0.0015))
    close = 100.0 * np.cumprod(1.0 + base_ret)
    volume = 1_000_000 + np.linspace(0, 200_000, n)
    base_df = pd.DataFrame(
        {
            "Date": dates,
            "Close_anchor": close,
            "Volume": volume,
            "ATR_norm": np.where(np.arange(n) < 30, 0.02, np.where(np.arange(n) < 60, 0.05, 0.03)),
            "ret_1": base_ret,
            "BNF_PanicScore": np.where(np.arange(n) < 60, 0.1, 1.5),
            "BNF_EnergyFade": np.where(np.arange(n) < 30, 0.2, 0.6),
            "BNF_DivDownVolUp": np.where(np.arange(n) < 60, 0.1, 0.9),
        }
    )
    base_df.to_csv(stepd_root / f"stepDprime_base_features_{symbol}.csv", index=False)

    stable = np.where(np.arange(n) < 30, 1, np.where(np.arange(n) < 60, 2, 3))
    raw20 = np.where(np.arange(n) < 30, 11, np.where(np.arange(n) < 60, 12, 19))
    rare = (raw20 == 19).astype(int)
    cluster_df = pd.DataFrame(
        {
            "Date": dates,
            "cluster_id_stable": stable,
            "cluster_id_raw20": raw20,
            "rare_flag_raw20": rare,
        }
    )
    cluster_df.to_csv(cluster_root / f"cluster_assignments_{symbol}.csv", index=False)
    (cluster_root / f"cluster_summary_{symbol}.json").write_text(
        json.dumps({"small_clusters": [19], "raw_k": 20, "k_valid": 3, "k_eff": 12}, ensure_ascii=False),
        encoding="utf-8",
    )

    split = np.where(np.arange(n) < 20, "train", "test")
    for agent in ["a1", "a2"]:
        if agent == "a1":
            ret = np.where(stable == 1, 0.004, np.where(stable == 2, -0.003, 0.0005))
            ratio = np.where(stable == 1, 0.9, 0.2)
        else:
            ret = np.where(stable == 2, 0.0045, np.where(stable == 1, -0.002, -0.0005))
            ratio = np.where(stable == 2, -0.8, -0.1)
        df = pd.DataFrame({"Date": dates, "Split": split, "ret": ret, "ratio": ratio})
        df["equity"] = _cum_equity(df["ret"])
        df.to_csv(stepe_root / f"stepE_daily_log_{agent}_{symbol}.csv", index=False)

    stepf_ret = np.where(stable == 1, 0.0035, np.where(stable == 2, 0.0038, -0.0002))
    stepf_ratio = np.where(stable == 1, 0.8, np.where(stable == 2, -0.75, 0.1))
    selected = np.where(stable == 1, "a1", np.where(stable == 2, "a2", "a1"))
    turnover = np.abs(pd.Series(stepf_ratio).diff()).fillna(0.0)
    router_df = pd.DataFrame(
        {
            "Date": dates,
            "Split": split,
            "ratio": stepf_ratio,
            "ret": stepf_ret,
            "equity": _cum_equity(pd.Series(stepf_ret)),
            "turnover": turnover,
            "selected_expert": selected,
            "cluster_id_stable": stable,
        }
    )
    router_df.to_csv(stepf_root / f"stepF_daily_log_router_{symbol}.csv", index=False)
    router_df[["Date", "Split", "ratio", "ret", "equity", "turnover"]].to_csv(stepf_root / f"stepF_equity_marl_{symbol}.csv", index=False)

    summary = run_cluster_evaluation(output_root=output_root, mode=mode, symbol=symbol)
    assert summary["status"] == "OK"
    out_dir = output_root / "audit" / "cluster_eval" / mode / symbol
    assert (out_dir / "cluster_profile_stable.csv").exists()
    assert (out_dir / "cluster_profile_raw20.csv").exists()
    assert (out_dir / "cluster_label_candidates.json").exists()
    assert (out_dir / "cluster_run_stats.csv").exists()
    assert (out_dir / "cluster_monthly_stability.csv").exists()
    assert (out_dir / "cluster_expert_effect.csv").exists()
    assert (out_dir / "cluster_stepf_effect.csv").exists()
    assert (out_dir / "cluster_rare_eval_raw20.csv").exists()
    assert (out_dir / "cluster_eval_summary.json").exists()

    labels = json.loads((out_dir / "cluster_label_candidates.json").read_text(encoding="utf-8"))
    assert any(item["cluster_source"] == "stable" for item in labels)

    expert_effect = pd.read_csv(out_dir / "cluster_expert_effect.csv")
    best_by_cluster = expert_effect.sort_values(["cluster_id", "mean_ret"], ascending=[True, False]).drop_duplicates("cluster_id")
    assert set(best_by_cluster["expert"]) >= {"a1", "a2"}

    report = evaluate(str(output_root), mode, symbol)
    assert report["cluster_eval"]["status"] == "OK"
    assert "stable_top_clusters" in report["cluster_eval"]
