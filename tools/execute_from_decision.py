#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from engine.broker_client import IBKRBrokerClient, IBKRSettings, SimBrokerClient


NY_TZ = "America/New_York"


@dataclass
class ExecConfig:
    symbol: str
    mode: str
    trading_day: str
    decision_path: Optional[str]
    output_root: str
    broker: str
    order_type: str
    safe_band_entry_pct: float
    safe_band_exit_pct: float
    neutral_threshold: float
    max_position_shares: int
    pos_limit: float
    dry_run: int
    force: int
    allow_outside_window: int


def _ny_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz=NY_TZ)


def _default_trading_day() -> str:
    return str(_ny_now().date())


def _resolve_day_token(day: str) -> Tuple[str, str]:
    dt = pd.to_datetime(day, errors="coerce")
    if pd.isna(dt):
        raise ValueError(f"invalid trading_day={day}")
    dt = dt.normalize()
    return dt.strftime("%Y-%m-%d"), dt.strftime("%Y%m%d")


def _decision_candidates(output_root: Path, symbol: str, day_dash: str, day_compact: str) -> List[Path]:
    base = output_root / "stepF" / "live" / "live_close_pre"
    return [
        base / f"decision_{symbol}_{day_dash}.json",
        base / f"decision_{symbol}_{day_compact}.json",
        base / f"decision_SOXL_{day_dash}.json",
        base / f"decision_SOXL_{day_compact}.json",
    ]


def _find_decision_path(cfg: ExecConfig) -> Path:
    if cfg.decision_path:
        return Path(cfg.decision_path)
    day_dash, day_compact = _resolve_day_token(cfg.trading_day)
    cands = _decision_candidates(Path(cfg.output_root), cfg.symbol, day_dash, day_compact)
    for p in cands:
        if p.exists() and p.is_file():
            return p
    raise FileNotFoundError(f"decision file not found. tried: {[str(p) for p in cands]}")


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _get_guard_val(decision: Dict[str, Any], key: str, default: Any) -> Any:
    guards = decision.get("guards", {}) or {}
    return guards.get(key, default)


def _in_ny_close_pre_window(ts: pd.Timestamp) -> bool:
    local = ts.tz_convert(NY_TZ)
    mins = local.hour * 60 + local.minute
    return (15 * 60 + 30) <= mins <= (16 * 60)


def _require_time_guard(allow_outside_window: bool) -> None:
    if allow_outside_window:
        return
    now = _ny_now()
    if not _in_ny_close_pre_window(now):
        raise RuntimeError(f"outside NY close-pre window: now={now}")


def _get_position(broker: Any, symbol: str) -> int:
    if hasattr(broker, "get_position"):
        return int(broker.get_position(symbol))
    if hasattr(broker, "get_current_position"):
        return int(broker.get_current_position(symbol))
    raise AttributeError("broker has no get_position/get_current_position")


def _get_last_price_safe(broker: Any, symbol: str) -> Optional[float]:
    try:
        return float(broker.get_last_price(symbol))
    except Exception:
        return None


def _place_order(
    broker: Any,
    symbol: str,
    side: str,
    qty: int,
    order_type: str,
    safe_band_entry_pct: float,
    safe_band_exit_pct: float,
) -> Dict[str, Any]:
    order_type = order_type.upper()
    if order_type == "MARKET":
        if hasattr(broker, "place_market_order"):
            order_id = broker.place_market_order(symbol=symbol, side=side, qty=qty)
        else:
            order_id = broker.submit_order(symbol=symbol, qty=qty, side=side)
        return {
            "symbol": symbol,
            "side": side,
            "qty": int(qty),
            "order_type": "MARKET",
            "limit_price": None,
            "order_id": str(order_id),
            "status": "submitted",
            "fills": [],
            "commission": None,
        }

    if order_type != "SAFE_LIMIT":
        raise ValueError(f"unsupported order_type={order_type}")

    last_price = float(broker.get_last_price(symbol))
    limit_price = (
        last_price * (1.0 + safe_band_entry_pct)
        if side == "BUY"
        else last_price * (1.0 - safe_band_exit_pct)
    )
    if hasattr(broker, "place_limit_order"):
        order_id = broker.place_limit_order(symbol=symbol, side=side, qty=qty, limit_price=limit_price)
    else:
        # SimBroker/IBKR compatibility fallback
        order_id = broker.submit_order(symbol=symbol, qty=qty, side=side)
    return {
        "symbol": symbol,
        "side": side,
        "qty": int(qty),
        "order_type": "SAFE_LIMIT",
        "limit_price": float(limit_price),
        "order_id": str(order_id),
        "status": "submitted",
        "fills": [],
        "commission": None,
    }


def _build_broker(name: str) -> Any:
    n = str(name or "sim").strip().lower()
    if n == "sim":
        return SimBrokerClient()
    if n == "ibkr":
        return IBKRBrokerClient(IBKRSettings())
    raise ValueError(f"unsupported broker={name}")


def _extract_ratio_final(decision: Dict[str, Any]) -> float:
    ratio = decision.get("ratio_final")
    if ratio is None:
        raise KeyError("decision.ratio_final is required")
    ratio = float(ratio)
    if not math.isfinite(ratio):
        raise ValueError("ratio_final must be finite")
    return ratio


def _extract_trading_day(decision: Dict[str, Any], fallback: str) -> str:
    v = decision.get("trading_day") or decision.get("target_date") or fallback
    return _resolve_day_token(str(v))[0]


def _safe_clip_ratio(ratio: float, pos_limit: float) -> float:
    return max(-abs(pos_limit), min(abs(pos_limit), ratio))


def _effective_ratio(soxl_shares: int, soxs_shares: int, max_position_shares: int) -> float:
    if max_position_shares <= 0:
        return 0.0
    if soxl_shares > 0 and soxs_shares > 0:
        raise RuntimeError("both SOXL and SOXS positions are positive; simultaneous holding is not allowed")
    if soxl_shares > 0:
        return float(soxl_shares) / float(max_position_shares)
    if soxs_shares > 0:
        return -float(soxs_shares) / float(max_position_shares)
    return 0.0


def _append_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _update_state(
    *,
    state_path: Path,
    trading_day: str,
    effective_ratio: float,
    trade_execution_path: Path,
    decision_path: Path,
    decision: Dict[str, Any],
) -> None:
    prev = {}
    if state_path.exists():
        try:
            prev = _load_json(state_path)
        except Exception:
            prev = {}
    next_state = dict(prev)
    next_state["last_trading_day"] = trading_day
    next_state["last_ratio"] = float(effective_ratio)
    next_state["last_trade_execution_path"] = str(trade_execution_path)
    next_state["last_decision_path"] = str(decision_path)
    if decision.get("stage1", {}).get("regime_id") is not None:
        next_state["last_regime_id"] = decision["stage1"]["regime_id"]
    next_state["updated_at_utc"] = pd.Timestamp.utcnow().isoformat()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(next_state, ensure_ascii=False, indent=2), encoding="utf-8")


def execute_from_decision(cfg: ExecConfig) -> int:
    output_root = Path(cfg.output_root)
    symbol = cfg.symbol.upper()
    day_dash, _ = _resolve_day_token(cfg.trading_day)
    trade_dir = output_root / "auto_trade" / "live"
    trade_path = trade_dir / f"trade_execution_{symbol}_{day_dash}.json"
    state_path = output_root / "stepF" / "live" / "live_close_pre" / f"state_{symbol}.json"

    record: Dict[str, Any] = {
        "symbol": symbol,
        "mode": cfg.mode,
        "created_at_utc": pd.Timestamp.utcnow().isoformat(),
        "trading_day": day_dash,
        "orders": [],
        "errors": [],
    }

    def _persist_failure() -> None:
        trade_dir.mkdir(parents=True, exist_ok=True)
        if trade_path.exists():
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            fail_path = trade_dir / f"trade_execution_{symbol}_{day_dash}_failed_{ts}.json"
            fail_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            trade_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    broker = None
    try:
        _require_time_guard(bool(cfg.allow_outside_window))
        decision_path = _find_decision_path(cfg)
        decision_bytes = decision_path.read_bytes()
        decision = json.loads(decision_bytes.decode("utf-8"))
        decision_hash = _sha256_bytes(decision_bytes)

        trading_day = _extract_trading_day(decision, fallback=day_dash)
        record["trading_day"] = trading_day
        if trading_day != day_dash:
            raise RuntimeError(f"stale guard failed: decision day={trading_day} cli day={day_dash}")

        if trade_path.exists() and not bool(cfg.force):
            raise RuntimeError(f"idempotency guard: trade execution exists: {trade_path}")

        ratio_final = _extract_ratio_final(decision)
        neutral_threshold = float(_get_guard_val(decision, "neutral_threshold", cfg.neutral_threshold))
        max_position_shares = int(_get_guard_val(decision, "max_position_shares", cfg.max_position_shares))
        pos_limit = float(_get_guard_val(decision, "pos_limit", cfg.pos_limit))
        ratio_final = _safe_clip_ratio(ratio_final, pos_limit)

        if abs(ratio_final) < neutral_threshold:
            target_long = 0
            target_inv = 0
        elif ratio_final > 0:
            target_long = int(round(abs(ratio_final) * max_position_shares))
            target_inv = 0
        else:
            target_long = 0
            target_inv = int(round(abs(ratio_final) * max_position_shares))

        broker = _build_broker(cfg.broker)
        decision_prices = decision.get("prices") or decision.get("pre_trade", {}).get("prices") or {}
        if hasattr(broker, "update_price") and isinstance(decision_prices, dict):
            for s in ("SOXL", "SOXS"):
                if s in decision_prices:
                    try:
                        broker.update_price(s, float(decision_prices[s]))
                    except Exception:
                        pass

        pre_soxl = _get_position(broker, "SOXL")
        pre_soxs = _get_position(broker, "SOXS")
        pre_prices = {
            "SOXL": _get_last_price_safe(broker, "SOXL"),
            "SOXS": _get_last_price_safe(broker, "SOXS"),
        }

        delta_long = target_long - pre_soxl
        delta_inv = target_inv - pre_soxs

        record.update(
            {
                "linked_decision": {
                    "decision_path": str(decision_path),
                    "decision_hash": decision_hash,
                    "decision_run_id": decision.get("run_id") or decision.get("decision_run_id"),
                },
                "pre_trade": {
                    "positions": {"SOXL": int(pre_soxl), "SOXS": int(pre_soxs)},
                    "prices": pre_prices,
                    "timestamp": _ny_now().isoformat(),
                },
                "target": {
                    "ratio_final": float(ratio_final),
                    "target_shares_long": int(target_long),
                    "target_shares_inverse": int(target_inv),
                    "neutral_threshold": neutral_threshold,
                    "max_position_shares": max_position_shares,
                    "pos_limit": pos_limit,
                },
                "idempotency_key": f"{trading_day}:{decision_hash}",
                "dry_run": bool(cfg.dry_run),
            }
        )

        order_plan: List[Tuple[str, str, int]] = []
        # SELL first (especially for reversals)
        if delta_long < 0:
            order_plan.append(("SOXL", "SELL", abs(delta_long)))
        if delta_inv < 0:
            order_plan.append(("SOXS", "SELL", abs(delta_inv)))
        if delta_long > 0:
            order_plan.append(("SOXL", "BUY", delta_long))
        if delta_inv > 0:
            order_plan.append(("SOXS", "BUY", delta_inv))

        if not bool(cfg.dry_run):
            for sym, side, qty in order_plan:
                od = _place_order(
                    broker=broker,
                    symbol=sym,
                    side=side,
                    qty=int(qty),
                    order_type=cfg.order_type,
                    safe_band_entry_pct=float(cfg.safe_band_entry_pct),
                    safe_band_exit_pct=float(cfg.safe_band_exit_pct),
                )
                record["orders"].append(od)
        else:
            for sym, side, qty in order_plan:
                record["orders"].append(
                    {
                        "symbol": sym,
                        "side": side,
                        "qty": int(qty),
                        "order_type": cfg.order_type.upper(),
                        "limit_price": None,
                        "order_id": None,
                        "status": "planned",
                        "fills": [],
                        "commission": None,
                    }
                )

        post_soxl = _get_position(broker, "SOXL")
        post_soxs = _get_position(broker, "SOXS")
        eff_ratio = _effective_ratio(post_soxl, post_soxs, max_position_shares)
        record["post_trade"] = {
            "positions_after": {"SOXL": int(post_soxl), "SOXS": int(post_soxs)},
            "effective_ratio_after": float(eff_ratio),
            "timestamp": _ny_now().isoformat(),
        }

        trade_dir.mkdir(parents=True, exist_ok=True)
        trade_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

        _append_csv(
            trade_dir / f"trade_executions_{symbol}.csv",
            {
                "trading_day": trading_day,
                "decision_hash": decision_hash,
                "ratio_final": float(ratio_final),
                "effective_ratio_after": float(eff_ratio),
                "dry_run": int(bool(cfg.dry_run)),
                "trade_execution_path": str(trade_path),
                "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
            },
        )
        for od in record["orders"]:
            _append_csv(
                trade_dir / f"orders_{symbol}.csv",
                {
                    "trading_day": trading_day,
                    "symbol": od.get("symbol"),
                    "side": od.get("side"),
                    "qty": od.get("qty"),
                    "order_type": od.get("order_type"),
                    "limit_price": od.get("limit_price"),
                    "order_id": od.get("order_id"),
                    "status": od.get("status"),
                    "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
                },
            )

        if not bool(cfg.dry_run):
            _update_state(
                state_path=state_path,
                trading_day=trading_day,
                effective_ratio=float(eff_ratio),
                trade_execution_path=trade_path,
                decision_path=decision_path,
                decision=decision,
            )

        return 0
    except Exception as e:
        record["errors"].append(
            {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
        )
        _persist_failure()
        return 1
    finally:
        if broker is not None and hasattr(broker, "close"):
            try:
                broker.close()
            except Exception:
                pass


def _parse_args() -> ExecConfig:
    ap = argparse.ArgumentParser(description="Execute live orders from StepF decision.json")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--mode", default="live")
    ap.add_argument("--trading-day", default=None)
    ap.add_argument("--decision-path", default=None)
    ap.add_argument("--output-root", default="output")
    ap.add_argument("--broker", default="sim", choices=["sim", "ibkr"])
    ap.add_argument("--order-type", default="SAFE_LIMIT", choices=["MARKET", "SAFE_LIMIT"])
    ap.add_argument("--safe-band-entry-pct", type=float, default=0.01)
    ap.add_argument("--safe-band-exit-pct", type=float, default=0.01)
    ap.add_argument("--neutral-threshold", type=float, default=0.1)
    ap.add_argument("--max-position-shares", type=int, default=100)
    ap.add_argument("--pos-limit", type=float, default=1.0)
    ap.add_argument("--dry-run", type=int, default=1)
    ap.add_argument("--force", type=int, default=0)
    ap.add_argument("--allow-outside-window", type=int, default=0)
    a = ap.parse_args()
    return ExecConfig(
        symbol=a.symbol,
        mode=a.mode,
        trading_day=a.trading_day or _default_trading_day(),
        decision_path=a.decision_path,
        output_root=a.output_root,
        broker=a.broker,
        order_type=a.order_type,
        safe_band_entry_pct=a.safe_band_entry_pct,
        safe_band_exit_pct=a.safe_band_exit_pct,
        neutral_threshold=a.neutral_threshold,
        max_position_shares=a.max_position_shares,
        pos_limit=a.pos_limit,
        dry_run=a.dry_run,
        force=a.force,
        allow_outside_window=a.allow_outside_window,
    )


def main() -> None:
    cfg = _parse_args()
    raise SystemExit(execute_from_decision(cfg))


if __name__ == "__main__":
    main()
