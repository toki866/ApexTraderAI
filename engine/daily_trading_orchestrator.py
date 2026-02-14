from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Dict, Tuple

from engine.live_policy_runner import LivePolicyConfig, LivePolicyRunner
from engine.state_builder import StateBuilder
from engine.broker_client import BrokerClient
from engine.pnl_calculator import PnLCalculator
from engine.trade_logger import TradeLogger

logger = logging.getLogger(__name__)


# =========================
# 注文種別 Enum
# =========================

class OrderType(str, Enum):
    """
    実運用で使う注文種別。

    - MARKET: 純粋な成行注文
    - SAFE_LIMIT: 「成行に近い指値」(現在値 ± 安全バンド%) で出す
    - LIMIT: 将来拡張用の純粋な指値（v1では未使用でもOK）
    """

    MARKET = "MARKET"
    SAFE_LIMIT = "SAFE_LIMIT"
    LIMIT = "LIMIT"


# =========================
# オーケストレータ用設定
# =========================

@dataclass
class DailyTradingConfig:
    """
    DailyTradingOrchestrator の設定。

    Attributes
    ----------
    symbol_long : str
        ratio > 0 のときにロングする銘柄（例: "SOXL"）。
    symbol_inverse : str
        ratio < 0 のときにロングする銘柄（例: "SOXS"）。
    neutral_threshold : float
        |ratio| がこの値未満なら「ニュートラル（ノーポジ）」と判定する。
    max_position_shares : int
        ratio=±1.0 のときの最大ポジション株数。
        実際のターゲット株数は abs(ratio) * max_position_shares を丸めて決定する。
    order_type : OrderType
        注文種別（MARKET / SAFE_LIMIT / LIMIT）。
        v1では SAFE_LIMIT を推奨。
    safe_band_entry_pct : float
        SAFE_LIMIT のとき、買い注文の許容スリッページ率（例: 0.01 → +1%）。
    safe_band_exit_pct : float
        SAFE_LIMIT のとき、売り注文の許容スリッページ率（例: 0.01 → -1%）。
    flags_path : Path
        「当日のオープンフロー／クローズフローを実行済みか」を記録するフラグファイル。
        同じ日・同じフローを二重実行しないための仕組み。
    """

    symbol_long: str = "SOXL"
    symbol_inverse: str = "SOXS"
    neutral_threshold: float = 0.1
    max_position_shares: int = 100

    order_type: OrderType = OrderType.SAFE_LIMIT
    safe_band_entry_pct: float = 0.01
    safe_band_exit_pct: float = 0.01

    flags_path: Path = field(default_factory=lambda: Path("output/auto_trader_flags.txt"))


# =========================
# オーケストレータ本体
# =========================

class DailyTradingOrchestrator:
    """
    1営業日の「朝フロー（建玉）」と「引け前フロー（クローズ＋PnL計算）」をまとめるクラス。

    - 朝: RLポリシーから ratio∈[-1,1] を取得して SOXL / SOXS / ノーポジを決定
    - 引け前: 全ポジションをクローズして PnL を確定、ログ保存
    - 1日2回のフローが同日に二重実行されないよう、flags_path で制御
    """

    def __init__(
        self,
        trading_config: DailyTradingConfig,
        policy_config: LivePolicyConfig,
        broker: BrokerClient,
        state_builder: StateBuilder,
        pnl_calculator: PnLCalculator,
        trade_logger: TradeLogger,
    ) -> None:
        self.config = trading_config
        self.policy_runner = LivePolicyRunner.from_config(policy_config)
        self.broker = broker
        self.state_builder = state_builder
        self.pnl_calculator = pnl_calculator
        self.trade_logger = trade_logger

        # フラグファイルのディレクトリを事前に作成
        if self.config.flags_path.parent and not self.config.flags_path.parent.exists():
            self.config.flags_path.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # 公開メソッド
    # -------------------------

    def run_morning_open_flow(self, trading_date: date) -> None:
        """
        朝（オープン前後）のフロー：
        - 状態を構築
        - RLポリシーから ratio を取得
        - ターゲットポジションを計算
        - 必要な注文を発注
        """
        flow_name = "open"
        if self._is_flow_done(trading_date, flow_name):
            logger.info("[DailyTradingOrchestrator] %s flow already executed for %s. Skipping.", flow_name, trading_date)
            return

        logger.info("[DailyTradingOrchestrator] === Morning open flow start: %s ===", trading_date)

        # 1) オープン時点の状態を作る（StateBuilder 側の仕様に依存）
        obs = self.state_builder.build_morning_state(self.config.symbol_long, trading_date)
        logger.debug("[DailyTradingOrchestrator] Built state for open: shape=%s", getattr(obs, "shape", None))

        # 2) RL ポリシーから ratio ∈ [-1, 1] を取得
        tech_row = self.state_builder.get_tech_row(self.config.symbol_long, trading_date)
        ratio = self.policy_runner.safe_decide_position(obs, trading_date=trading_date, tech_row=tech_row)
        logger.info("[DailyTradingOrchestrator] Policy ratio for %s: %.4f", trading_date, ratio)

        # 3) ratio → ターゲット株数（SOXL / SOXS）に変換
        target_positions = self._decide_target_positions_from_ratio(ratio)
        logger.info(
            "[DailyTradingOrchestrator] Target positions (long=%s, inverse=%s) for ratio=%.4f",
            target_positions.get(self.config.symbol_long, 0),
            target_positions.get(self.config.symbol_inverse, 0),
            ratio,
        )

        # 4) 現在ポジションとの差分を埋めるように注文発注
        self._rebalance_positions(trading_date, ratio, target_positions)

        # 5) フロー実行済みフラグを記録
        self._mark_flow_done(trading_date, flow_name)

        logger.info("[DailyTradingOrchestrator] === Morning open flow finished: %s ===", trading_date)

    def run_close_pre_flow(self, trading_date: date) -> None:
        """
        引け前のフロー：
        - 現在保有している SOXL / SOXS をすべてクローズ
        - その日の PnL を計算
        - ログを保存
        """
        flow_name = "close"
        if self._is_flow_done(trading_date, flow_name):
            logger.info("[DailyTradingOrchestrator] %s flow already executed for %s. Skipping.", flow_name, trading_date)
            return

        logger.info("[DailyTradingOrchestrator] === Close-pre flow start: %s ===", trading_date)

        symbols = [self.config.symbol_long, self.config.symbol_inverse]
        close_prices: Dict[str, float] = {}
        positions_before_close: Dict[str, int] = {}

        # 1) 現在ポジションと終値近辺の価格を取得
        for symbol in symbols:
            pos = self.broker.get_position(symbol)
            price = self.broker.get_last_price(symbol)
            positions_before_close[symbol] = pos
            close_prices[symbol] = price
            logger.info(
                "[DailyTradingOrchestrator] Before close: symbol=%s, position=%d, last_price=%.4f",
                symbol,
                pos,
                price,
            )

        # 2) 全ポジションをクローズ（SELL）
        for symbol in symbols:
            qty = positions_before_close[symbol]
            if qty > 0:
                logger.info(
                    "[DailyTradingOrchestrator] Closing position: symbol=%s, qty=%d", symbol, qty
                )
                self._place_order(
                    trading_date=trading_date,
                    symbol=symbol,
                    side="SELL",
                    quantity=qty,
                    note="close_all",
                )

        # 3) PnL 計算（PnLCalculator 側の仕様に依存）
        #    必要なら state_builder や trade_logger が出したログを使って計算する。
        try:
            daily_pnl = self.pnl_calculator.calculate_daily_pnl(trading_date)
            logger.info(
                "[DailyTradingOrchestrator] Daily PnL for %s: %.2f", trading_date, daily_pnl
            )
        except Exception as e:
            logger.exception(
                "[DailyTradingOrchestrator] Failed to calculate daily PnL for %s: %s",
                trading_date,
                e,
            )

        # 4) フロー実行済みをマーク
        self._mark_flow_done(trading_date, flow_name)

        logger.info("[DailyTradingOrchestrator] === Close-pre flow finished: %s ===", trading_date)

    def run_full_day(self, trading_date: date) -> None:
        """
        その日の「朝フロー」と「引け前フロー」をまとめて実行する。
        （Task Scheduler からは基本 auto_trader.py 側で open/close を時間帯別に実行する想定だが、
         手動テスト用に 1関数で通せるようにしておく）
        """
        self.run_morning_open_flow(trading_date)
        self.run_close_pre_flow(trading_date)

    # -------------------------
    # ratio → position 決定
    # -------------------------

    def _decide_target_positions_from_ratio(self, ratio: float) -> Dict[str, int]:
        """
        RL の ratio ∈ [-1, 1] から SOXL / SOXS のターゲット株数を決める。

        ルール:
        - |ratio| < neutral_threshold → 完全ノーポジ（両銘柄 0株）
        - ratio > 0                → symbol_long を abs(ratio) * max_position_shares 株ロング
        - ratio < 0                → symbol_inverse を abs(ratio) * max_position_shares 株ロング
        """
        cfg = self.config
        targets: Dict[str, int] = {
            cfg.symbol_long: 0,
            cfg.symbol_inverse: 0,
        }

        if abs(ratio) < cfg.neutral_threshold:
            # ニュートラル：どちらも 0 株
            return targets

        # 最大ポジション株数に ratio の絶対値を掛けて丸める
        base_shares = int(round(abs(ratio) * cfg.max_position_shares))
        if base_shares <= 0:
            # 念のため下限1株
            base_shares = 1

        if ratio > 0:
            targets[cfg.symbol_long] = base_shares
            targets[cfg.symbol_inverse] = 0
        else:
            targets[cfg.symbol_long] = 0
            targets[cfg.symbol_inverse] = base_shares

        return targets

    # -------------------------
    # ポジションリバランス → 注文発注
    # -------------------------

    def _rebalance_positions(
        self,
        trading_date: date,
        ratio: float,
        target_positions: Dict[str, int],
    ) -> None:
        """
        現在のポジションとターゲットポジションの差分を埋めるように注文を発注する。
        """
        symbols = [self.config.symbol_long, self.config.symbol_inverse]

        for symbol in symbols:
            target_qty = target_positions.get(symbol, 0)
            current_qty = self.broker.get_position(symbol)
            delta = target_qty - current_qty

            if delta == 0:
                logger.info(
                    "[DailyTradingOrchestrator] No change for %s (current=%d, target=%d).",
                    symbol,
                    current_qty,
                    target_qty,
                )
                continue

            side = "BUY" if delta > 0 else "SELL"
            qty = abs(delta)

            logger.info(
                "[DailyTradingOrchestrator] Rebalance %s: side=%s, qty=%d (current=%d, target=%d)",
                symbol,
                side,
                qty,
                current_qty,
                target_qty,
            )

            self._place_order(
                trading_date=trading_date,
                symbol=symbol,
                side=side,
                quantity=qty,
                note=f"rebalance (ratio={ratio:.4f})",
            )

    # -------------------------
    # 注文発注（OrderType & SAFE_LIMIT）
    # -------------------------

    def _place_order(
        self,
        trading_date: date,
        symbol: str,
        side: str,
        quantity: int,
        note: str = "",
    ) -> Tuple[str, float]:
        """
        config.order_type に応じて BrokerClient に注文を飛ばす。

        Returns
        -------
        (order_id, used_limit_price)
        """
        if quantity <= 0:
            raise ValueError(f"quantity must be positive, got {quantity}")

        order_type = self.config.order_type
        order_id = ""
        used_limit_price: float = 0.0

        # 現在価格（SAFE_LIMIT / LIMIT 価格算出用）
        last_price = self.broker.get_last_price(symbol)

        if order_type == OrderType.MARKET:
            order_id = self.broker.place_market_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
            )
            used_limit_price = 0.0  # 成行なので 0 としておく
            logger.info(
                "[DailyTradingOrchestrator] Placed MARKET order: symbol=%s, side=%s, qty=%d, order_id=%s",
                symbol,
                side,
                quantity,
                order_id,
            )

        elif order_type == OrderType.SAFE_LIMIT:
            limit_price = self._calc_safe_limit_price(side=side, current_price=last_price)
            order_id = self.broker.place_limit_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                limit_price=limit_price,
                time_in_force="day",
            )
            used_limit_price = limit_price
            logger.info(
                "[DailyTradingOrchestrator] Placed SAFE_LIMIT order: symbol=%s, side=%s, qty=%d, limit=%.4f, order_id=%s",
                symbol,
                side,
                quantity,
                limit_price,
                order_id,
            )

        elif order_type == OrderType.LIMIT:
            # v1 では汎用 LIMIT は SAFE_LIMIT と同じロジックで出しておく。
            limit_price = self._calc_safe_limit_price(side=side, current_price=last_price)
            order_id = self.broker.place_limit_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                limit_price=limit_price,
                time_in_force="day",
            )
            used_limit_price = limit_price
            logger.info(
                "[DailyTradingOrchestrator] Placed LIMIT(order_type=LIMIT) order: symbol=%s, side=%s, qty=%d, limit=%.4f, order_id=%s",
                symbol,
                side,
                quantity,
                limit_price,
                order_id,
            )
        else:
            raise ValueError(f"Unsupported OrderType: {order_type}")

        # TradeLogger にも記録しておく
        try:
            self.trade_logger.log_order(
                trading_date=trading_date,
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type.value,
                price=last_price if used_limit_price == 0.0 else used_limit_price,
                note=note,
            )
        except Exception as e:
            logger.exception("[DailyTradingOrchestrator] Failed to log order: %s", e)

        return order_id, used_limit_price

    def _calc_safe_limit_price(self, side: str, current_price: float) -> float:
        """
        「成行＋指値の安全バンド」用の指値価格を計算する。

        BUY: 現在価格 × (1 + safe_band_entry_pct)
        SELL: 現在価格 × (1 - safe_band_exit_pct)
        """
        if current_price <= 0:
            raise ValueError(f"current_price must be positive, got {current_price}")

        cfg = self.config

        if side.upper() == "BUY":
            band = cfg.safe_band_entry_pct
            limit_price = current_price * (1.0 + band)
        elif side.upper() == "SELL":
            band = cfg.safe_band_exit_pct
            limit_price = current_price * (1.0 - band)
        else:
            raise ValueError(f"Unsupported side: {side}")

        # 必要ならティックサイズに丸める（ここではそのまま返す）
        return float(limit_price)

    # -------------------------
    # フロー実行フラグ管理
    # -------------------------

    def _flag_key(self, trading_date: date, flow_name: str) -> str:
        """
        フラグファイル内で使うキー文字列を生成。
        例: "2025-11-27_open", "2025-11-27_close"
        """
        return f"{trading_date.isoformat()}_{flow_name}"

    def _read_flags(self) -> Dict[str, bool]:
        """
        flags_path から実行済みフロー一覧を読み込む。
        """
        flags: Dict[str, bool] = {}
        if not self.config.flags_path.exists():
            return flags

        try:
            with self.config.flags_path.open("r", encoding="utf-8") as f:
                for line in f:
                    key = line.strip()
                    if key:
                        flags[key] = True
        except Exception as e:
            logger.exception("[DailyTradingOrchestrator] Failed to read flags file: %s", e)

        return flags

    def _write_flags(self, flags: Dict[str, bool]) -> None:
        """
        flags_path にフロー実行済み一覧を書き出す。
        （単純にキーを1行ずつ書く）
        """
        try:
            with self.config.flags_path.open("w", encoding="utf-8") as f:
                for key, done in flags.items():
                    if done:
                        f.write(key + "\n")
        except Exception as e:
            logger.exception("[DailyTradingOrchestrator] Failed to write flags file: %s", e)

    def _is_flow_done(self, trading_date: date, flow_name: str) -> bool:
        """
        指定日のフロー（open / close）が既に実行済みかどうかを確認。
        """
        key = self._flag_key(trading_date, flow_name)
        flags = self._read_flags()
        return flags.get(key, False)

    def _mark_flow_done(self, trading_date: date, flow_name: str) -> None:
        """
        指定日のフロー（open / close）を「実行済み」としてフラグファイルに記録。
        """
        key = self._flag_key(trading_date, flow_name)
        flags = self._read_flags()
        flags[key] = True
        self._write_flags(flags)
