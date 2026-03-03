from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_core.services.step_f_service import StepFRouterConfig, StepFService
from ai_core.utils.timing_logger import TimingLogger
from tools.run_pipeline import _get_app_config, _repo_root


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return ""


def _build_stepf_config(output_root: Path, input_mode: str, symbol: str) -> StepFRouterConfig:
    stepe_dir = output_root / "stepE" / input_mode
    agents = []
    for p in sorted(stepe_dir.glob(f"stepE_daily_log_*_{symbol}.csv")):
        agents.append(p.name[len("stepE_daily_log_") : -len(f"_{symbol}.csv")])
    if not agents:
        raise FileNotFoundError(f"No StepE logs found for symbol={symbol} under {stepe_dir}")
    safe = [a for a in ("dprime_bnf_h01", "dprime_all_features_h01", "dprime_mix_3scale") if a in agents]
    if len(safe) < 2:
        safe = agents[: min(2, len(agents))]
    return StepFRouterConfig(output_root=str(output_root), agents=",".join(agents), mode="live", safe_set=",".join(safe))


def _build_date_range_from_stepa(output_root: Path, mode: str, symbol: str) -> Any:
    p_tr = pd.read_csv(output_root / "stepA" / mode / f"stepA_prices_train_{symbol}.csv")
    p_te = pd.read_csv(output_root / "stepA" / mode / f"stepA_prices_test_{symbol}.csv")
    p_tr["Date"] = pd.to_datetime(p_tr["Date"], errors="coerce")
    p_te["Date"] = pd.to_datetime(p_te["Date"], errors="coerce")
    from types import SimpleNamespace

    return SimpleNamespace(
        train_start=p_tr["Date"].min().date().isoformat(),
        train_end=p_tr["Date"].max().date().isoformat(),
        test_start=p_te["Date"].min().date().isoformat(),
        test_end=p_te["Date"].max().date().isoformat(),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--mode", default="live")
    ap.add_argument("--retrain", choices=["on", "off"], default="off")
    ap.add_argument("--branch", default="default")
    ap.add_argument("--config", default="")
    ap.add_argument("--output-root", default="output")
    ap.add_argument("--input-mode", default="sim")
    ap.add_argument("--data-cutoff", default="")
    args = ap.parse_args()

    repo_root = _repo_root()
    app_config = _get_app_config(repo_root)

    output_root = Path(args.output_root)
    retrain = args.retrain
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{random.randint(1000,9999)}"
    run_dir = output_root / "stepF" / "live" / f"retrain_{retrain}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = _build_stepf_config(output_root=output_root, input_mode=args.input_mode, symbol=args.symbol)
    cfg.retrain = retrain
    cfg.branch_id = args.branch
    cfg.input_mode = args.input_mode
    setattr(app_config, "stepF", cfg)

    timing = TimingLogger(output_root=run_dir, mode="live", run_id=run_id, branch_id=args.branch, retrain=retrain, enabled=True, clear=True)
    setattr(app_config, "_timing_logger", timing)

    date_range = _build_date_range_from_stepa(output_root=output_root, mode=args.input_mode, symbol=args.symbol)
    svc = StepFService(app_config)

    with timing.stage("stepA.reference_check"):
        _ = output_root / "stepA" / args.input_mode
    with timing.stage("stepB.reference_check"):
        _ = output_root / "stepB"
    with timing.stage("stepC.reference_check"):
        _ = output_root / "stepC"
    with timing.stage("stepD.reference_check"):
        _ = output_root / "stepD_prime"
    with timing.stage("stepE.reference_check"):
        _ = output_root / "stepE" / args.input_mode
    with timing.stage("stepF.service_entry"):
        result = svc.run_live(date_range=date_range, symbol=args.symbol, retrain=retrain, branch_id=args.branch, data_cutoff=args.data_cutoff)

    timings_csv = run_dir / "timings.csv"
    total_sec = float(pd.read_csv(timings_csv)["elapsed_sec"].sum()) if timings_csv.exists() else 0.0
    commit = _git_commit(repo_root)
    if timings_csv.exists() and commit:
        tdf = pd.read_csv(timings_csv)
        tdf["git_commit"] = commit
        tdf.to_csv(timings_csv, index=False)

    summary = {
        "run_id": run_id,
        "mode": "live",
        "retrain": retrain,
        "branch_id": args.branch,
        "data_cutoff": args.data_cutoff,
        "symbol": args.symbol,
        "model_path": cfg.model_source_dir if retrain == "off" else str(run_dir / "router"),
        "timings_total_sec": total_sec,
        "output_dir": str(run_dir),
        "config_path": args.config,
        "ratio_csv": getattr(result, "ratio_path", ""),
        "daily_log": result.daily_log_path,
        "summary_json": result.summary_path,
        "target_period": {"test_start": date_range.test_start, "test_end": date_range.test_end},
    }
    run_summary_path = run_dir / "run_summary.json"
    run_summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
