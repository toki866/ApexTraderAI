#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from engine.broker_client import IBKRBrokerClient, IBKRSettings, SimBrokerClient
from tools.execute_from_decision import _effective_ratio


def _build_broker(name: str):
    n = str(name or "sim").strip().lower()
    if n == "sim":
        return SimBrokerClient()
    if n == "ibkr":
        return IBKRBrokerClient(IBKRSettings())
    raise ValueError(f"unsupported broker={name}")


def _get_position(broker, symbol: str) -> int:
    if hasattr(broker, "get_position"):
        return int(broker.get_position(symbol))
    return int(broker.get_current_position(symbol))


def main() -> None:
    ap = argparse.ArgumentParser(description="Reconcile state.last_ratio from broker positions")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--mode", default="live")
    ap.add_argument("--output-root", default="output")
    ap.add_argument("--max-position-shares", type=int, required=True)
    ap.add_argument("--broker", default="sim", choices=["sim", "ibkr"])
    a = ap.parse_args()

    out_root = Path(a.output_root)
    state_path = out_root / "stepF" / "live" / "live_close_pre" / f"state_{a.symbol.upper()}.json"
    logs_dir = out_root / "auto_trade" / "live"
    logs_dir.mkdir(parents=True, exist_ok=True)

    broker = _build_broker(a.broker)
    try:
        soxl = _get_position(broker, "SOXL")
        soxs = _get_position(broker, "SOXS")
        ratio = _effective_ratio(soxl, soxs, int(a.max_position_shares))

        prev = {}
        if state_path.exists():
            prev = json.loads(state_path.read_text(encoding="utf-8"))
        prev["last_ratio"] = float(ratio)
        prev["last_trading_day"] = str(pd.Timestamp.now(tz="America/New_York").date())
        prev["state_reconciled_from_broker"] = True
        prev["updated_at_utc"] = pd.Timestamp.utcnow().isoformat()
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(prev, ensure_ascii=False, indent=2), encoding="utf-8")

        log_path = logs_dir / f"reconcile_state_{a.symbol.upper()}.log"
        log_path.write_text(
            (
                f"timestamp_utc={pd.Timestamp.utcnow().isoformat()}\n"
                f"symbol={a.symbol.upper()} mode={a.mode} broker={a.broker}\n"
                f"positions soxl={soxl} soxs={soxs} max_position_shares={a.max_position_shares}\n"
                f"effective_ratio={ratio:.6f}\nstate_path={state_path}\n"
            ),
            encoding="utf-8",
        )
    finally:
        if hasattr(broker, "close"):
            broker.close()


if __name__ == "__main__":
    main()
