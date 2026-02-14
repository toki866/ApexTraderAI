from __future__ import annotations

import logging
from dataclasses import dataclass

from engine.daily_trading_orchestrator import PnLCalculatorProtocol

logger = logging.getLogger(__name__)


@dataclass
class PnLCalculatorConfig:
    """
    PnLCalculator の設定クラス。

    Parameters
    ----------
    min_prev_equity : float
        PnL% 計算時に「前日エクイティがこれ未満なら 0 とみなす」ための閾値。
        0 に近すぎる値で割り算すると数値が暴れるのを防ぐため。
    """

    min_prev_equity: float = 1e-6


class PnLCalculator(PnLCalculatorProtocol):
    """
    前日エクイティと当日エクイティから 1日分の PnL を計算するクラス。

    定義
    ----
    pnl_abs = current_equity - prev_equity
    pnl_pct = pnl_abs / prev_equity       （ただし prev_equity が小さすぎる場合は 0.0）

    v1 ではこの単純定義で十分。
    将来的に「手数料」「スリッページ」「税金」などを入れたければ、
    compute_pnl 内にオプションで組み込んでいく。
    """

    def __init__(self, config: PnLCalculatorConfig | None = None) -> None:
        self.config = config or PnLCalculatorConfig()

    def compute_pnl(self, prev_equity: float, current_equity: float) -> tuple[float, float]:
        """
        前日エクイティと当日エクイティから PnL（金額・％）を計算する。

        Parameters
        ----------
        prev_equity : float
            前営業日引け時点のエクイティ（口座評価額）。
        current_equity : float
            当日引け時点（または effective close）のエクイティ。

        Returns
        -------
        pnl_abs : float
            当日の損益額（current_equity - prev_equity）。
        pnl_pct : float
            当日の損益率（pnl_abs / prev_equity）。
            prev_equity が小さすぎる場合（min_prev_equity 未満）は 0.0。
        """
        prev = float(prev_equity)
        current = float(current_equity)

        pnl_abs = current - prev

        if abs(prev) < self.config.min_prev_equity:
            # 初日など、前日エクイティがほぼ0（または未設定）の場合は
            # パーセントは 0 として扱う。
            pnl_pct = 0.0
            logger.warning(
                "PnLCalculator: prev_equity too small (%.8f). "
                "Set pnl_pct=0.0 (pnl_abs=%.4f, current=%.4f).",
                prev, pnl_abs, current,
            )
        else:
            pnl_pct = pnl_abs / prev

        logger.info(
            "PnLCalculator: prev_equity=%.4f, current_equity=%.4f, "
            "pnl_abs=%.4f, pnl_pct=%.6f",
            prev, current, pnl_abs, pnl_pct,
        )

        return pnl_abs, pnl_pct
