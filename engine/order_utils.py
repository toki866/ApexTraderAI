from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Literal


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class SafeLimitOrderParams:
    symbol: str
    side: Side
    quantity: int
    limit_price: float
    time_in_force: str = "DAY"


def calc_safe_limit_price(
    side: Side,
    current_price: float,
    entry_band_pct: float = 0.01,
    exit_band_pct: float = 0.01,
) -> float:
    """
    成行に近い「安全バンド付き指値価格」を計算する。

    Parameters
    ----------
    side : Side
        BUY なら現在価格より上に指値（+band）、SELL なら下に指値（-band）。
    current_price : float
        ブローカーから取得した最新価格（bid/ask か mid_price など）。
    entry_band_pct : float
        建玉用の許容スリッページ（例: 0.01 → 1%）。
    exit_band_pct : float
        決済用の許容スリッページ（例: 0.01 → 1%）。

    Returns
    -------
    float
        指値価格。
    """
    if current_price <= 0:
        raise ValueError(f"current_price must be positive, got {current_price}")

    if side == Side.BUY:
        band = entry_band_pct
        limit_price = current_price * (1.0 + band)
    elif side == Side.SELL:
        band = exit_band_pct
        limit_price = current_price * (1.0 - band)
    else:
        raise ValueError(f"Unsupported side: {side}")

    # 必要ならここでティック単位に丸める（例：0.01ドル刻みなど）
    # tick = 0.01
    # limit_price = round(limit_price / tick) * tick

    return float(limit_price)


def build_safe_limit_order(
    symbol: str,
    side: Literal["BUY", "SELL"],
    quantity: int,
    current_price: float,
    entry_band_pct: float = 0.01,
    exit_band_pct: float = 0.01,
) -> SafeLimitOrderParams:
    """
    「成行＋指値の安全バンド」ポリシーで出す LIMIT 注文パラメータを構築する。

    Parameters
    ----------
    symbol : str
        銘柄シンボル（例: "SOXL"）。
    side : {"BUY", "SELL"}
        売買区分。
    quantity : int
        株数。正の整数。
    current_price : float
        最新価格。
    entry_band_pct : float
        建玉時の許容スリッページ（例: 0.01）。
    exit_band_pct : float
        決済時の許容スリッページ（例: 0.01）。

    Returns
    -------
    SafeLimitOrderParams
        BrokerClient.place_limit_order(...) にそのまま渡せるような注文情報。
    """
    if quantity <= 0:
        raise ValueError(f"quantity must be positive, got {quantity}")

    _side = Side(side)  # Enumに変換（バリデーション兼ねる）

    limit_price = calc_safe_limit_price(
        side=_side,
        current_price=current_price,
        entry_band_pct=entry_band_pct,
        exit_band_pct=exit_band_pct,
    )

    return SafeLimitOrderParams(
        symbol=symbol,
        side=_side,
        quantity=quantity,
        limit_price=limit_price,
        time_in_force="DAY",
    )
