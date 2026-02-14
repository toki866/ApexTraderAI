from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Any

try:
    # IBKR 用（インストールされていない環境では ImportError になるが、その場合は IBKRBrokerClient を使わなければOK）
    from ib_insync import IB, Stock, MarketOrder, Trade
except ImportError:  # pragma: no cover - ib_insync未導入環境でもSimBrokerは動かしたい
    IB = None
    Stock = None
    MarketOrder = None
    Trade = None


# ==============================
# Broker モード種別
# ==============================


class BrokerMode(str, Enum):
    SIM = "sim"              # ローカル擬似口座
    IBKR_PAPER = "ibkr_paper"
    IBKR_REAL = "ibkr_real"


# ==============================
# 共通インターフェース
# ==============================


class BaseBrokerClient:
    """
    すべてのブローカー実装が従う共通インターフェース。

    - get_current_position(symbol): 現在の保有株数
    - get_cash(): 利用可能キャッシュ（USD想定）
    - submit_order(symbol, qty, side): 成行注文（BUY/SELL）
    - get_last_price(symbol): 現在価格（終値相当）
    - close(): 接続クローズ
    """

    def get_current_position(self, symbol: str) -> int:
        raise NotImplementedError

    def get_cash(self) -> float:
        raise NotImplementedError

    def submit_order(self, symbol: str, qty: int, side: str) -> str:
        raise NotImplementedError

    def get_last_price(self, symbol: str) -> float:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


# ==============================
# Sim（ローカル擬似口座）実装
# ==============================


@dataclass
class SimBrokerSettings:
    """
    ローカル擬似口座用の設定。

    Attributes
    ----------
    starting_cash : float
        初期キャッシュ（USD想定）
    base_currency : str
        通貨コード（通常 "USD"）
    """
    starting_cash: float = 100_000.0
    base_currency: str = "USD"


class SimBrokerClient(BaseBrokerClient):
    """
    シンプルなローカル擬似ブローカー。

    - 現ポジションとキャッシュをメモリ上で管理
    - 価格は update_price(symbol, price) で更新してもらう
    - submit_order では「現在価格 × 数量」でキャッシュを増減する
    """

    def __init__(self, settings: Optional[SimBrokerSettings] = None, logger: Any = None) -> None:
        self._settings = settings or SimBrokerSettings()
        self._logger = logger

        # symbol -> 保有株数
        self._positions: Dict[str, int] = {}
        # 現在価格（疑似用）
        self._last_prices: Dict[str, float] = {}
        # キャッシュ
        self._cash: float = float(self._settings.starting_cash)

    # ------------------------------
    # 内部ヘルパ
    # ------------------------------
    def _log(self, msg: str) -> None:
        if self._logger is not None:
            try:
                self._logger.info(f"[SimBroker] {msg}")
            except Exception:
                print(f"[SimBroker] {msg}")
        else:
            print(f"[SimBroker] {msg}")

    # 外部から日足やクローズ前価格を反映させるためのヘルパ
    def update_price(self, symbol: str, price: float) -> None:
        """
        擬似口座用に「現在価格」を更新する。

        DailyTradingOrchestratorなどが、その日の終値 or クローズ前価格を
        このメソッドでブローカーに教えてあげる想定。
        """
        self._last_prices[symbol] = float(price)
        self._log(f"Updated price: {symbol} = {price}")

    # ------------------------------
    # BaseBrokerClient 実装
    # ------------------------------
    def get_current_position(self, symbol: str) -> int:
        return int(self._positions.get(symbol, 0))

    def get_cash(self) -> float:
        return float(self._cash)

    def submit_order(self, symbol: str, qty: int, side: str) -> str:
        if qty <= 0:
            raise ValueError("qty must be positive")

        side = side.upper()
        if side not in ("BUY", "SELL"):
            raise ValueError("side must be 'BUY' or 'SELL'")

        if symbol not in self._last_prices:
            raise RuntimeError(
                f"Price for symbol={symbol} is not set. "
                f"Call SimBrokerClient.update_price(symbol, price) before submit_order."
            )

        price = self._last_prices[symbol]
        cost = price * qty

        pos_before = self._positions.get(symbol, 0)

        if side == "BUY":
            # シンプルに「買えるだけ買う」ロジック。
            # 厳密な証拠金チェックなどは別レイヤー（PnLCalculator等）で実施する前提。
            self._positions[symbol] = pos_before + qty
            self._cash -= cost
            self._log(f"BUY {symbol} x {qty} @ {price} -> position={self._positions[symbol]}, cash={self._cash:.2f}")
        else:  # SELL
            # ショート禁止とし、保有株数を超える売りはエラーにする。
            if qty > pos_before:
                raise RuntimeError(
                    f"Cannot SELL {qty} {symbol}; current position is {pos_before} (short selling disabled)."
                )
            self._positions[symbol] = pos_before - qty
            self._cash += cost
            self._log(f"SELL {symbol} x {qty} @ {price} -> position={self._positions[symbol]}, cash={self._cash:.2f}")

        # 擬似なので order_id は簡易的に生成
        order_id = f"SIM-{symbol}-{side}-{qty}"
        return order_id

    def get_last_price(self, symbol: str) -> float:
        if symbol not in self._last_prices:
            raise RuntimeError(
                f"Price for symbol={symbol} is not set. "
                f"Call SimBrokerClient.update_price(symbol, price) before get_last_price."
            )
        return float(self._last_prices[symbol])

    def close(self) -> None:
        # 擬似ブローカーなので特にクローズ処理なし
        self._log("SimBrokerClient closed.")


# ==============================
# IBKR 実口座 / Paper口座 用
# ==============================


@dataclass
class IBKRSettings:
    """
    IBKR 用接続設定。

    Attributes
    ----------
    host : str
        TWS / IB Gateway のホスト（通常 "127.0.0.1"）
    port : int
        ポート番号（paper: 7497, real: 7496 がデフォルト）
    client_id : int
        APIクライアントID（一意）
    account_id : str
        取引口座ID（paper なら DU****、real なら U****）
    exchange : str
        取引所（SMART が一般的）
    currency : str
        通貨コード（通常 "USD"）
    """
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1
    account_id: str = ""
    exchange: str = "SMART"
    currency: str = "USD"


class IBKRBrokerClient(BaseBrokerClient):
    """
    IBKR TWS / IB Gateway + ib_insync を用いた実ブローカー。

    - paper / real は IBKRSettings（port, account_id等）の差し替えで両対応
    - SOXL / SOXS に限らず任意のシンボルを扱える設計
    """

    def __init__(self, settings: IBKRSettings, logger: Any = None) -> None:
        if IB is None:
            raise ImportError(
                "ib_insync がインポートできませんでした。'pip install ib_insync' を実行してください。"
            )

        self._settings = settings
        self._logger = logger

        self._ib: IB = IB()
        self._connected: bool = False

    # ------------------------------
    # 内部ヘルパ
    # ------------------------------
    def _log(self, msg: str) -> None:
        if self._logger is not None:
            try:
                self._logger.info(f"[IBKR] {msg}")
            except Exception:
                print(f"[IBKR] {msg}")
        else:
            print(f"[IBKR] {msg}")

    def _ensure_connected(self) -> None:
        if self._connected:
            return
        self._log(
            f"Connecting to IBKR {self._settings.host}:{self._settings.port} "
            f"(clientId={self._settings.client_id})"
        )
        self._ib.connect(
            host=self._settings.host,
            port=self._settings.port,
            clientId=self._settings.client_id,
        )
        if not self._ib.isConnected():
            raise RuntimeError("Failed to connect to IBKR TWS/Gateway")
        self._connected = True
        self._log("Connected to IBKR")

    def _stock_contract(self, symbol: str):
        """
        指定シンボルの Stock コントラクトを生成。
        """
        if Stock is None:
            raise RuntimeError("ib_insync.Stock が利用できません。")
        return Stock(symbol, self._settings.exchange, self._settings.currency)

    # ------------------------------
    # BaseBrokerClient 実装
    # ------------------------------
    def get_current_position(self, symbol: str) -> int:
        self._ensure_connected()
        contract = self._stock_contract(symbol)

        positions = self._ib.positions()
        total = 0
        for pos in positions:
            if (
                pos.account == self._settings.account_id
                and pos.contract.symbol == contract.symbol
                and pos.contract.currency == contract.currency
            ):
                total += int(pos.position)

        # このシステムの仕様ではショート禁止なので、負ポジションは0扱い
        return max(total, 0)

    def get_cash(self) -> float:
        self._ensure_connected()
        # AccountSummaryからAvailableFunds（またはTotalCashValue）を取得
        summary = self._ib.accountSummary()
        cash = 0.0
        for item in summary:
            if (
                item.account == self._settings.account_id
                and item.tag in ("AvailableFunds", "TotalCashValue")
                and item.currency == self._settings.currency
            ):
                try:
                    cash = float(item.value)
                except ValueError:
                    cash = 0.0
                break
        return cash

    def submit_order(self, symbol: str, qty: int, side: str) -> str:
        """
        成行注文。日次運用（1日2回判定）なので MKT 注文のみ対応。
        """
        if qty <= 0:
            raise ValueError("qty must be positive")

        side = side.upper()
        if side not in ("BUY", "SELL"):
            raise ValueError("side must be 'BUY' or 'SELL'")

        self._ensure_connected()
        if MarketOrder is None:
            raise RuntimeError("ib_insync.MarketOrder が利用できません。")

        contract = self._stock_contract(symbol)
        order = MarketOrder(side, qty)

        self._log(f"Placing {side} MKT order: {symbol} x {qty}")
        trade: Trade = self._ib.placeOrder(contract, order)
        order_id = str(trade.order.orderId)
        self._log(f"Submitted order_id={order_id}")
        return order_id

    def get_last_price(self, symbol: str) -> float:
        """
        現在価格（last or close or bid/ask中間）を返す。
        """
        self._ensure_connected()
        contract = self._stock_contract(symbol)
        tickers = self._ib.reqTickers(contract)
        if not tickers:
            raise RuntimeError(f"No ticker data for symbol={symbol}")

        t = tickers[0]
        if t.last is not None:
            return float(t.last)
        if t.close is not None:
            return float(t.close)
        if t.bid is not None and t.ask is not None:
            return float((t.bid + t.ask) / 2.0)

        raise RuntimeError(f"No usable price data for symbol={symbol}")

    def close(self) -> None:
        if self._connected and self._ib is not None:
            self._log("Disconnecting from IBKR")
            self._ib.disconnect()
            self._connected = False
