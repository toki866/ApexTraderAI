#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import uuid
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

NY_TZ = ZoneInfo("America/New_York")
DEFAULT_DECISION_THRESHOLD = 0.05


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run one live orchestration cycle (pipeline + final decision + optional order).")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--trade-date", required=True)
    ap.add_argument("--output-root", default=None)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--steps", default="A,B,C,DPRIME,E,F")
    ap.add_argument("--reuse-output", type=int, default=1, choices=[0, 1])
    ap.add_argument("--force-rebuild", type=int, default=0, choices=[0, 1])
    ap.add_argument("--timing", type=int, default=1, choices=[0, 1])
    ap.add_argument("--order-mode", default="dry-run", choices=["dry-run", "paper", "live"])
    ap.add_argument("--execute-order", type=int, default=0, choices=[0, 1])
    ap.add_argument("--decision-window", default="close_pre")
    ap.add_argument("--test-start", default=None)
    ap.add_argument("--train-years", type=int, default=8)
    ap.add_argument("--test-months", type=int, default=3)
    ap.add_argument("--broker", default="auto")
    ap.add_argument("--decision-threshold", type=float, default=DEFAULT_DECISION_THRESHOLD)
    return ap.parse_args()


def _is_us_trading_day(target_date: date) -> bool:
    if target_date.weekday() >= 5:
        return False
    try:
        from pandas.tseries.holiday import USFederalHolidayCalendar

        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=target_date, end=target_date)
        return target_date not in {d.date() for d in holidays.to_pydatetime()}
    except Exception:
        return True


def check_market_window(now_ny: datetime, trade_date: str, decision_window: str) -> Dict[str, Any]:
    td = datetime.strptime(trade_date, "%Y-%m-%d").date()
    trading_day = _is_us_trading_day(td)
    market_open = datetime.combine(td, time(9, 30), tzinfo=NY_TZ)
    market_close = datetime.combine(td, time(16, 0), tzinfo=NY_TZ)

    if decision_window == "close_pre":
        start = datetime.combine(td, time(15, 45), tzinfo=NY_TZ)
        end = market_close
    else:
        start = market_open
        end = market_close

    within = trading_day and (start <= now_ny <= end)
    reason = "ok" if within else (
        "not_trading_day" if not trading_day else "outside_decision_window"
    )
    return {
        "trading_day": trading_day,
        "market_open": market_open.isoformat(),
        "within_decision_window": within,
        "now_ny": now_ny.isoformat(),
        "reason": reason,
    }


def resolve_live_output_root(symbol: str, trade_date: str, output_root: Optional[str]) -> Path:
    if output_root:
        return Path(output_root)
    return Path("output") / "live" / symbol.upper() / trade_date


def _log(msg: str, fh) -> None:
    line = f"[{datetime.now(tz=NY_TZ).isoformat()}] {msg}"
    print(line)
    fh.write(line + "\n")
    fh.flush()


def run_live_pipeline(args: argparse.Namespace, output_root: Path, log_fh) -> None:
    cmd = [
        sys.executable,
        "tools/run_pipeline.py",
        "--mode",
        "live",
        "--steps",
        args.steps,
        "--symbol",
        args.symbol.upper(),
        "--output-root",
        str(output_root),
        "--data-dir",
        args.data_dir,
        "--reuse-output",
        str(args.reuse_output),
        "--force-rebuild",
        str(args.force_rebuild),
        "--timing",
        str(args.timing),
        "--train-years",
        str(args.train_years),
        "--test-months",
        str(args.test_months),
    ]
    if args.test_start:
        cmd.extend(["--test-start", args.test_start])

    _log(f"run_pipeline command: {' '.join(cmd)}", log_fh)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        log_fh.write(line)
        log_fh.flush()
        print(line, end="")
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"run_pipeline failed with return code={rc}")


def _read_last_ratio_from_csv(path: Path) -> Tuple[float, Optional[str], Dict[str, Any]]:
    import pandas as pd

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"StepF artifact has no rows: {path}")

    ratio_col = next((c for c in ("ratio", "pos", "Position") if c in df.columns), None)
    if ratio_col is None:
        raise KeyError(f"No ratio column in {path}")

    date_col = "Date" if "Date" in df.columns else ("date" if "date" in df.columns else None)
    row = df.iloc[-1].to_dict()
    ratio = float(row.get(ratio_col, 0.0) or 0.0)
    ts = str(row.get(date_col)) if date_col else None
    return ratio, ts, row


def build_final_decision_from_stepf(
    output_root: Path,
    symbol: str,
    trade_date: str,
    decision_window: str,
    threshold: float,
) -> Tuple[Path, Dict[str, Any]]:
    stepf_live = output_root / "stepF" / "live"
    candidates: List[Path] = []
    patterns = [
        f"stepF_daily_log_router_{symbol}.csv",
        f"stepF_daily_log_marl_{symbol}.csv",
        f"reward_*/stepF_daily_log_router_{symbol}.csv",
        f"reward_*/stepF_daily_log_marl_{symbol}.csv",
    ]
    for p in patterns:
        candidates.extend(stepf_live.glob(p))
    if not candidates:
        raise FileNotFoundError(f"StepF output not found under {stepf_live}")

    source_path = max(candidates, key=lambda p: p.stat().st_mtime)
    ratio, decision_price_ts, last_row = _read_last_ratio_from_csv(source_path)

    if abs(ratio) <= threshold:
        side = "NO_TRADE"
        target_symbol = "NO_TRADE"
    elif ratio > 0:
        side = "BUY_SOXL"
        target_symbol = "SOXL"
    else:
        side = "BUY_SOXS"
        target_symbol = "SOXS"

    inverse_symbol = "SOXS" if target_symbol == "SOXL" else "SOXL"
    decision = {
        "run_id": f"live-{datetime.now(tz=NY_TZ).strftime('%Y%m%d_%H%M%S')}-{uuid.uuid4().hex[:8]}",
        "trade_date": trade_date,
        "mode": "live",
        "decision_window": decision_window,
        "base_symbol": symbol,
        "target_symbol": target_symbol,
        "inverse_symbol": inverse_symbol,
        "selected_agent": str(last_row.get("selected_agent", "unknown")),
        "selected_cluster": str(last_row.get("selected_cluster", "unknown")),
        "source_stepf_path": str(source_path),
        "ratio": float(ratio),
        "side": side,
        "position_policy": "single_leg_long_only",
        "decision_price_timestamp": decision_price_ts,
        "generated_at": datetime.now(tz=NY_TZ).isoformat(),
        "notes": {
            "threshold": float(threshold),
            "source_row_columns": sorted(list(last_row.keys())),
        },
    }

    out_path = stepf_live / f"final_decision_{symbol}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path, decision


def write_final_signal_csv(output_root: Path, symbol: str, decision: Dict[str, Any]) -> Path:
    out_path = output_root / "stepF" / "live" / f"final_signal_{symbol}.csv"
    fields = [
        "trade_date",
        "symbol",
        "ratio",
        "side",
        "target_symbol",
        "selected_agent",
        "selected_cluster",
        "run_id",
        "generated_at",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({
            "trade_date": decision["trade_date"],
            "symbol": symbol,
            "ratio": decision["ratio"],
            "side": decision["side"],
            "target_symbol": decision["target_symbol"],
            "selected_agent": decision["selected_agent"],
            "selected_cluster": decision["selected_cluster"],
            "run_id": decision["run_id"],
            "generated_at": decision["generated_at"],
        })
    return out_path


def _extract_step_statuses(output_root: Path, steps: str) -> Tuple[List[str], List[str], List[str]]:
    manifest_path = output_root / "run_manifest.json"
    requested = [s.strip().upper() for s in steps.split(",") if s.strip()]
    if not manifest_path.exists():
        return [], [], requested
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    step_data = data.get("steps", {}) if isinstance(data, dict) else {}
    executed, reused, skipped = [], [], []
    for step in requested:
        status = str((step_data.get(step, {}) or {}).get("status", "pending")).lower()
        if status == "complete":
            executed.append(step)
        elif status == "reuse":
            reused.append(step)
        else:
            skipped.append(step)
    return executed, reused, skipped


def maybe_execute_order(args: argparse.Namespace, decision_path: Path, output_root: Path, log_fh) -> Dict[str, Any]:
    if int(args.execute_order) != 1:
        return {"order_executed": False, "order_mode": args.order_mode, "message": "execute-order=0"}

    cmd = [
        sys.executable,
        "tools/order_executor.py",
        "--decision-file",
        str(decision_path),
        "--order-mode",
        args.order_mode,
        "--broker",
        args.broker,
        "--output-root",
        str(output_root),
        "--execute-order",
        str(args.execute_order),
    ]
    _log(f"order_executor command: {' '.join(cmd)}", log_fh)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.stdout:
        log_fh.write(res.stdout)
        print(res.stdout, end="")
    if res.stderr:
        log_fh.write(res.stderr)
        print(res.stderr, end="", file=sys.stderr)
    if res.returncode != 0:
        raise RuntimeError(f"order_executor failed rc={res.returncode}")
    return {"order_executed": True, "order_mode": args.order_mode, "message": "ok"}


def write_live_run_summary(output_root: Path, symbol: str, summary: Dict[str, Any]) -> Path:
    path = output_root / "logs" / "live" / f"live_run_summary_{symbol}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    symbol = args.symbol.upper()
    output_root = resolve_live_output_root(symbol, args.trade_date, args.output_root)
    log_path = output_root / "logs" / "live" / f"live_run_{args.trade_date}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    success = False
    error_summary = ""
    final_decision: Dict[str, Any] = {}
    order_result: Dict[str, Any] = {"order_executed": False, "order_mode": args.order_mode}
    executed_steps: List[str] = []
    reused_steps: List[str] = []
    skipped_steps: List[str] = []

    with log_path.open("a", encoding="utf-8") as log_fh:
        try:
            now_ny = datetime.now(tz=NY_TZ)
            market = check_market_window(now_ny, args.trade_date, args.decision_window)
            _log(f"market_window={json.dumps(market, ensure_ascii=False)}", log_fh)

            run_live_pipeline(args, output_root, log_fh)
            decision_path, final_decision = build_final_decision_from_stepf(
                output_root=output_root,
                symbol=symbol,
                trade_date=args.trade_date,
                decision_window=args.decision_window,
                threshold=args.decision_threshold,
            )
            write_final_signal_csv(output_root, symbol, final_decision)
            order_result = maybe_execute_order(args, decision_path, output_root, log_fh)
            executed_steps, reused_steps, skipped_steps = _extract_step_statuses(output_root, args.steps)
            success = True
        except Exception as exc:
            error_summary = str(exc)
            _log(f"[ERROR] {error_summary}", log_fh)

    summary = {
        "run_id": final_decision.get("run_id", f"live-{datetime.now(tz=NY_TZ).strftime('%Y%m%d_%H%M%S')}"),
        "trade_date": args.trade_date,
        "mode": "live",
        "executed_steps": executed_steps,
        "reused_steps": reused_steps,
        "skipped_steps": skipped_steps,
        "final_ratio": final_decision.get("ratio"),
        "final_side": final_decision.get("side"),
        "order_mode": args.order_mode,
        "order_executed": bool(order_result.get("order_executed", False)),
        "success": success,
        "error_summary": error_summary,
    }
    write_live_run_summary(output_root, symbol, summary)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
