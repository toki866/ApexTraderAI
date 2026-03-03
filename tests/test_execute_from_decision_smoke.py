import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
from tools.execute_from_decision import ExecConfig, execute_from_decision


class DummyBroker:
    def __init__(self) -> None:
        self.positions = {"SOXL": 0, "SOXS": 0}
        self.prices = {"SOXL": 100.0, "SOXS": 50.0}

    def update_price(self, symbol: str, price: float) -> None:
        self.prices[symbol] = float(price)

    def get_current_position(self, symbol: str) -> int:
        return int(self.positions.get(symbol, 0))

    def get_last_price(self, symbol: str) -> float:
        return float(self.prices[symbol])

    def submit_order(self, symbol: str, qty: int, side: str) -> str:
        if side == "BUY":
            self.positions[symbol] += int(qty)
        else:
            self.positions[symbol] -= int(qty)
        return f"DUMMY-{symbol}-{side}-{qty}"

    def close(self) -> None:
        return None


def _write_decision(root: Path, day: str, ratio: float = 0.5) -> Path:
    p = root / "stepF" / "live" / "live_close_pre"
    p.mkdir(parents=True, exist_ok=True)
    d = {
        "symbol": "SOXL",
        "target_date": day,
        "ratio_final": ratio,
        "guards": {
            "neutral_threshold": 0.1,
            "max_position_shares": 100,
            "pos_limit": 1.0,
        },
        "prices": {"SOXL": 100.0, "SOXS": 50.0},
        "stage1": {"regime_id": 7},
    }
    dec_path = p / f"decision_SOXL_{day}.json"
    dec_path.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    return dec_path


def _cfg(root: Path, day: str, dry_run: int) -> ExecConfig:
    return ExecConfig(
        symbol="SOXL",
        mode="live",
        trading_day=day,
        decision_path=None,
        output_root=str(root),
        broker="sim",
        order_type="MARKET",
        safe_band_entry_pct=0.01,
        safe_band_exit_pct=0.01,
        neutral_threshold=0.1,
        max_position_shares=100,
        pos_limit=1.0,
        dry_run=dry_run,
        force=0,
        allow_outside_window=1,
    )


def test_dry_run_generates_trade_execution_and_no_state(monkeypatch, tmp_path: Path):
    day = "2026-01-05"
    _write_decision(tmp_path, day, ratio=0.4)
    monkeypatch.setattr("tools.execute_from_decision._build_broker", lambda _: DummyBroker())

    code = execute_from_decision(_cfg(tmp_path, day, dry_run=1))
    assert code == 0

    trade_path = tmp_path / "auto_trade" / "live" / f"trade_execution_SOXL_{day}.json"
    assert trade_path.exists()
    payload = json.loads(trade_path.read_text(encoding="utf-8"))
    assert payload["dry_run"] is True

    state_path = tmp_path / "stepF" / "live" / "live_close_pre" / "state_SOXL.json"
    assert not state_path.exists()


def test_idempotency_rejects_second_run(monkeypatch, tmp_path: Path):
    day = "2026-01-06"
    _write_decision(tmp_path, day, ratio=0.2)
    monkeypatch.setattr("tools.execute_from_decision._build_broker", lambda _: DummyBroker())

    code1 = execute_from_decision(_cfg(tmp_path, day, dry_run=1))
    assert code1 == 0
    code2 = execute_from_decision(_cfg(tmp_path, day, dry_run=1))
    assert code2 == 1


def test_execute_updates_state_with_effective_ratio(monkeypatch, tmp_path: Path):
    day = "2026-01-07"
    _write_decision(tmp_path, day, ratio=0.5)
    monkeypatch.setattr("tools.execute_from_decision._build_broker", lambda _: DummyBroker())

    code = execute_from_decision(_cfg(tmp_path, day, dry_run=0))
    assert code == 0

    state_path = tmp_path / "stepF" / "live" / "live_close_pre" / "state_SOXL.json"
    assert state_path.exists()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["last_trading_day"] == day
    assert abs(float(state["last_ratio"]) - 0.5) < 1e-9
