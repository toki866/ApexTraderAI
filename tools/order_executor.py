#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from abc import ABC, abstractmethod
from datetime import datetime, time
from math import floor
from pathlib import Path
from typing import Any, Dict, Tuple
from zoneinfo import ZoneInfo

NY_TZ = ZoneInfo("America/New_York")
VALID_SIDES = {"BUY_SOXL", "BUY_SOXS", "NO_TRADE"}
REQUIRED_DECISION_KEYS = {
    "run_id",
    "trade_date",
    "mode",
    "decision_window",
    "base_symbol",
    "target_symbol",
    "inverse_symbol",
    "selected_agent",
    "selected_cluster",
    "source_stepf_path",
    "ratio",
    "side",
    "position_policy",
    "decision_price_timestamp",
    "generated_at",
    "notes",
}


class BaseBrokerAdapter(ABC):
    @abstractmethod
    def submit_order(self, order_plan: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class DryRunBrokerAdapter(BaseBrokerAdapter):
    def submit_order(self, order_plan: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": "accepted",
            "broker_order_id": f"dryrun-{context['trade_date']}-{context['target_symbol']}",
            "filled_qty": int(order_plan.get("delta_qty", 0)),
            "message": "dry-run mode: no live order submitted",
            "submitted_at": datetime.now(tz=NY_TZ).isoformat(),
        }


class UnsupportedBrokerAdapter(BaseBrokerAdapter):
    def __init__(self, order_mode: str, broker: str) -> None:
        self.order_mode = order_mode
        self.broker = broker

    def submit_order(self, order_plan: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        raise RuntimeError(
            f"No adapter configured for order_mode={self.order_mode}, broker={self.broker}. Safe stop."
        )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Execute order from final_decision JSON.")
    ap.add_argument("--decision-file", required=True)
    ap.add_argument("--order-mode", required=True, choices=["dry-run", "paper", "live"])
    ap.add_argument("--broker", default="auto")
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--account-size", type=float, default=100000.0)
    ap.add_argument("--max-position-fraction", type=float, default=0.2)
    ap.add_argument("--execute-order", type=int, default=0, choices=[0, 1])
    ap.add_argument("--data-dir", default="data")
    return ap.parse_args()


def load_final_decision(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED_DECISION_KEYS if k not in data]
    if missing:
        raise ValueError(f"final_decision missing required keys: {missing}")
    return data


def validate_final_decision(decision: Dict[str, Any]) -> None:
    ratio = float(decision["ratio"])
    side = str(decision["side"])
    target_symbol = str(decision["target_symbol"])

    if not (-1.0 <= ratio <= 1.0):
        raise ValueError(f"ratio out of bounds [-1,1]: {ratio}")
    if side not in VALID_SIDES:
        raise ValueError(f"invalid side: {side}")

    expected_target = "NO_TRADE" if side == "NO_TRADE" else ("SOXL" if side == "BUY_SOXL" else "SOXS")
    if target_symbol != expected_target:
        raise ValueError(f"target_symbol mismatch. side={side} target_symbol={target_symbol}")


def load_position_state(path: Path, symbol: str) -> Dict[str, Any]:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    return {
        "symbol": symbol,
        "held_symbol": "NONE",
        "qty": 0,
        "updated_at": datetime.now(tz=NY_TZ).isoformat(),
    }


def _estimate_price(target_symbol: str, trade_date: str, data_dir: Path) -> Tuple[float, str]:
    if target_symbol == "NO_TRADE":
        return 1.0, "no-trade"
    csv_path = data_dir / f"prices_{target_symbol}.csv"
    if csv_path.exists():
        import pandas as pd

        df = pd.read_csv(csv_path)
        if "Date" in df.columns and "Close" in df.columns:
            d = df[df["Date"].astype(str) <= str(trade_date)]
            if not d.empty:
                px = float(d.iloc[-1]["Close"])
                if px > 0:
                    return px, f"data:{csv_path}"
    return 20.0, "fallback:20.0"


def compute_order_plan(
    decision: Dict[str, Any],
    order_mode: str,
    account_size: float,
    max_position_fraction: float,
    current_position: Dict[str, Any],
    data_dir: Path,
) -> Dict[str, Any]:
    ratio = float(decision["ratio"])
    target_symbol = str(decision["target_symbol"])
    side = str(decision["side"])
    current_qty = int(current_position.get("qty", 0) or 0)
    current_symbol = str(current_position.get("held_symbol", "NONE"))

    px, price_source = _estimate_price(target_symbol, decision["trade_date"], data_dir)
    target_notional = float(account_size) * float(max_position_fraction) * abs(ratio)
    target_qty = 0 if side == "NO_TRADE" else int(max(0, floor(target_notional / max(px, 1e-9))))

    close_existing_qty = current_qty if (current_symbol not in ("NONE", target_symbol) and current_qty > 0) else 0
    effective_current_qty = current_qty if current_symbol == target_symbol else 0
    delta_qty = target_qty - effective_current_qty

    return {
        "side": side,
        "target_symbol": target_symbol,
        "target_qty": int(target_qty),
        "current_qty": int(effective_current_qty),
        "delta_qty": int(delta_qty),
        "close_existing_qty": int(close_existing_qty),
        "notional_estimate": float(target_qty * px),
        "estimated_price": float(px),
        "price_source": price_source,
        "order_mode": order_mode,
    }


def preflight_order_checks(
    decision: Dict[str, Any],
    order_plan: Dict[str, Any],
    execute_order: int,
    order_mode: str,
    result_file: Path,
) -> Dict[str, Any]:
    td = datetime.strptime(decision["trade_date"], "%Y-%m-%d").date()
    now_ny = datetime.now(tz=NY_TZ)
    market_open = datetime.combine(td, time(9, 30), tzinfo=NY_TZ)
    market_close = datetime.combine(td, time(16, 0), tzinfo=NY_TZ)
    within_market = market_open <= now_ny <= market_close

    if order_mode not in {"dry-run", "paper", "live"}:
        raise ValueError(f"invalid order_mode={order_mode}")
    if int(order_plan["target_qty"]) < 0:
        raise ValueError("target_qty must be >= 0")
    if result_file.exists() and int(execute_order) == 1:
        raise RuntimeError(f"duplicate order blocked for same day: {result_file}")

    return {
        "within_market_hours": within_market,
        "execute_order": int(execute_order),
        "order_mode": order_mode,
        "decision_valid": True,
        "target_qty_non_negative": True,
    }


def _resolve_adapter(order_mode: str, broker: str) -> BaseBrokerAdapter:
    if order_mode == "dry-run":
        return DryRunBrokerAdapter()
    return UnsupportedBrokerAdapter(order_mode=order_mode, broker=broker)


def execute_order(
    adapter: BaseBrokerAdapter,
    decision: Dict[str, Any],
    order_plan: Dict[str, Any],
    execute_order_flag: int,
) -> Dict[str, Any]:
    if int(execute_order_flag) != 1:
        return {
            "status": "skipped",
            "message": "execute-order=0",
            "submitted_at": datetime.now(tz=NY_TZ).isoformat(),
        }
    if decision["side"] == "NO_TRADE" or order_plan["delta_qty"] == 0:
        return {
            "status": "no_order",
            "message": "NO_TRADE or zero delta_qty",
            "submitted_at": datetime.now(tz=NY_TZ).isoformat(),
        }
    return adapter.submit_order(order_plan, {
        "trade_date": decision["trade_date"],
        "target_symbol": decision["target_symbol"],
    })


def save_order_request(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_order_result(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_position_state(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_broker_status(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    decision_file = Path(args.decision_file)
    output_root = Path(args.output_root)

    decision = load_final_decision(decision_file)
    validate_final_decision(decision)

    symbol = str(decision["base_symbol"]).upper()
    trade_date = str(decision["trade_date"])

    orders_dir = output_root / "orders" / "live"
    broker_dir = output_root / "broker" / "live"
    order_request_path = orders_dir / f"order_request_{symbol}_{trade_date}.json"
    order_result_path = orders_dir / f"order_result_{symbol}_{trade_date}.json"
    position_state_path = broker_dir / f"position_state_{symbol}.json"
    broker_status_path = broker_dir / f"broker_status_{trade_date}.json"

    current_position = load_position_state(position_state_path, symbol)
    order_plan = compute_order_plan(
        decision=decision,
        order_mode=args.order_mode,
        account_size=args.account_size,
        max_position_fraction=args.max_position_fraction,
        current_position=current_position,
        data_dir=Path(args.data_dir),
    )
    preflight = preflight_order_checks(
        decision=decision,
        order_plan=order_plan,
        execute_order=args.execute_order,
        order_mode=args.order_mode,
        result_file=order_result_path,
    )

    req_payload = {
        "decision": decision,
        "order_plan": order_plan,
        "preflight": preflight,
        "created_at": datetime.now(tz=NY_TZ).isoformat(),
    }
    save_order_request(order_request_path, req_payload)

    adapter = _resolve_adapter(args.order_mode, args.broker)
    try:
        exec_result = execute_order(adapter, decision, order_plan, args.execute_order)
        success = exec_result.get("status") in {"accepted", "no_order", "skipped"}
        error = ""
    except Exception as exc:
        exec_result = {
            "status": "error",
            "message": str(exc),
            "submitted_at": datetime.now(tz=NY_TZ).isoformat(),
        }
        success = False
        error = str(exc)

    result_payload = {
        "order_mode": args.order_mode,
        "broker": args.broker,
        "success": success,
        "error": error,
        "execution": exec_result,
        "order_plan": order_plan,
        "trade_date": trade_date,
        "symbol": symbol,
        "run_id": decision["run_id"],
    }
    save_order_result(order_result_path, result_payload)

    if success and exec_result.get("status") in {"accepted", "no_order"}:
        if decision["side"] == "NO_TRADE":
            new_position = {"symbol": symbol, "held_symbol": "NONE", "qty": 0, "updated_at": datetime.now(tz=NY_TZ).isoformat()}
        else:
            new_position = {
                "symbol": symbol,
                "held_symbol": decision["target_symbol"],
                "qty": int(order_plan["target_qty"]),
                "updated_at": datetime.now(tz=NY_TZ).isoformat(),
            }
    else:
        new_position = current_position
    save_position_state(position_state_path, new_position)

    broker_payload = {
        "trade_date": trade_date,
        "order_mode": args.order_mode,
        "broker": args.broker,
        "adapter": adapter.__class__.__name__,
        "success": success,
        "updated_at": datetime.now(tz=NY_TZ).isoformat(),
    }
    save_broker_status(broker_status_path, broker_payload)

    print(json.dumps({
        "order_request": str(order_request_path),
        "order_result": str(order_result_path),
        "position_state": str(position_state_path),
        "broker_status": str(broker_status_path),
        "success": success,
    }, ensure_ascii=False))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
