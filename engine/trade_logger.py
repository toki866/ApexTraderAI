from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from engine.daily_trading_orchestrator import TradeLoggerProtocol

logger = logging.getLogger(__name__)


@dataclass
class TradeLoggerConfig:
    """
    TradeLogger の設定。

    Parameters
    ----------
    log_root : Path
        日次ログCSVを保存するルートディレクトリ。
        例: Path("output")
    log_file_pattern : str
        銘柄ごとの日次ログCSVファイル名パターン。
        "{symbol}" を銘柄コードで置き換える。
        例: "daily_log_{symbol}.csv"
    """

    log_root: Path = Path("output")
    log_file_pattern: str = "daily_log_{symbol}.csv"


class TradeLogger(TradeLoggerProtocol):
    """
    日次ログ（朝の決定内容・引けの結果）をCSVに保存するクラス。

    1銘柄につき1つの CSV ファイル（daily_log_{symbol}.csv）を持つ。

    カラム構成（想定）
    -------------------
    - Date               : 日付 (datetime64)
    - Ratio              : 当日のRL出力（-1〜+1）
    - Action             : 前日ActionとしてStateBuilderで使う値（v1では Ratio と同じ）
    - Equity_morning_before : 朝フロー前のエクイティ
    - Equity_morning_after  : 朝フロー後のエクイティ
    - Equity_close       : 引け時点のエクイティ
    - PnL_abs            : 当日損益額
    - PnL_pct            : 当日損益率
    - Pos_SOXL_before    : 朝フロー前のSOXL保有株数
    - Pos_SOXS_before    : 朝フロー前のSOXS保有株数
    - Pos_SOXL_after     : 朝フロー後のSOXL保有株数
    - Pos_SOXS_after     : 朝フロー後のSOXS保有株数
    - Pos_SOXL_close     : 引け時点のSOXL保有株数
    - Pos_SOXS_close     : 引け時点のSOXS保有株数

    v1 では最低限、StateBuilder が参照するのは「Date / PnL_pct / Action」。
    それ以外は将来の分析やデバッグに役立てるための情報。
    """

    def __init__(self, symbol: str, config: Optional[TradeLoggerConfig] = None) -> None:
        """
        Parameters
        ----------
        symbol : str
            この TradeLogger が担当する銘柄コード（例: "SOXL"）。
            実際にはSOXL/SOXSペア運用だが、ログファイル名の識別用として使う。
        config : TradeLoggerConfig, optional
            ログ保存に関する設定。
        """
        self.symbol = symbol
        self.config = config or TradeLoggerConfig()

        self.log_root: Path = self.config.log_root
        self.log_file: Path = self.log_root / self.config.log_file_pattern.format(symbol=self.symbol)

        # ログディレクトリを作成
        self.log_root.mkdir(parents=True, exist_ok=True)

        # DataFrameキャッシュ（毎回読み書きでもいいが、v1は単純化してその都度読み直し）
        # ここではキャッシュは持たず、必要なときに読んで、その都度書き戻す。

    # =====================================================
    # TradeLoggerProtocol 実装
    # =====================================================

    def get_last_equity(self) -> Optional[float]:
        """
        直近営業日のエクイティを返す。

        優先順位：
        1. Equity_close があればそれを使う
        2. なければ Equity_morning_after を使う
        3. どちらも無い場合は None
        """
        df = self._load_log()
        if df.empty:
            return None

        df_sorted = df.sort_values("Date").reset_index(drop=True)
        row = df_sorted.iloc[-1]

        equity_close = row.get("Equity_close", np.nan)
        if pd.notna(equity_close):
            try:
                return float(equity_close)
            except (TypeError, ValueError):
                pass

        equity_morning_after = row.get("Equity_morning_after", np.nan)
        if pd.notna(equity_morning_after):
            try:
                return float(equity_morning_after)
            except (TypeError, ValueError):
                pass

        return None

    def log_morning_decision(
        self,
        trading_date: date,
        ratio: float,
        equity_before: float,
        equity_after: float,
        pos_soxl_before: int,
        pos_soxs_before: int,
        pos_soxl_after: int,
        pos_soxs_after: int,
    ) -> None:
        """
        朝フローの決定内容を CSV に記録する。

        - 当日の Ratio を Action として保存（StateBuilder が前日Actionとして使用）
        - PnL_abs / PnL_pct / Equity_close / Pos_*_close はこの時点では未定なので NaN
        """
        d = self._to_date(trading_date)
        df = self._load_log()

        # 既に同日の行があれば上書き、無ければ新規行
        idx = self._find_row_index_by_date(df, d)

        row_data = {
            "Date": pd.Timestamp(d),
            "Ratio": float(ratio),
            "Action": float(ratio),  # v1では Ratio をそのままActionにする
            "Equity_morning_before": float(equity_before),
            "Equity_morning_after": float(equity_after),
            # 引け情報はまだ未定
            "Equity_close": np.nan,
            "PnL_abs": np.nan,
            "PnL_pct": np.nan,
            # ポジション情報
            "Pos_SOXL_before": int(pos_soxl_before),
            "Pos_SOXS_before": int(pos_soxs_before),
            "Pos_SOXL_after": int(pos_soxl_after),
            "Pos_SOXS_after": int(pos_soxs_after),
            "Pos_SOXL_close": np.nan,
            "Pos_SOXS_close": np.nan,
        }

        df = self._upsert_row(df, idx, row_data)
        self._save_log(df)

        logger.info(
            "TradeLogger(%s): logged morning decision for %s "
            "(ratio=%.4f, equity_before=%.2f, equity_after=%.2f)",
            self.symbol,
            d,
            ratio,
            equity_before,
            equity_after,
        )

    def log_close_result(
        self,
        trading_date: date,
        equity_close: float,
        pnl_abs: float,
        pnl_pct: float,
        pos_soxl: int,
        pos_soxs: int,
    ) -> None:
        """
        引け時点のエクイティとPnL、ポジション情報を CSV に記録する。

        朝フローで既に同日行がある場合はそこを更新、
        無ければ新規行として作成する。
        """
        d = self._to_date(trading_date)
        df = self._load_log()

        idx = self._find_row_index_by_date(df, d)

        # 既存行がある場合はそれをベースにして更新、
        # 無い場合は新規行として必要カラムを埋める。
        if idx is not None:
            row = df.loc[idx].to_dict()
        else:
            row = {
                "Date": pd.Timestamp(d),
                "Ratio": np.nan,
                "Action": np.nan,
                "Equity_morning_before": np.nan,
                "Equity_morning_after": np.nan,
                "Pos_SOXL_before": np.nan,
                "Pos_SOXS_before": np.nan,
                "Pos_SOXL_after": np.nan,
                "Pos_SOXS_after": np.nan,
            }

        row.update(
            {
                "Date": pd.Timestamp(d),
                "Equity_close": float(equity_close),
                "PnL_abs": float(pnl_abs),
                "PnL_pct": float(pnl_pct),
                "Pos_SOXL_close": int(pos_soxl),
                "Pos_SOXS_close": int(pos_soxs),
            }
        )

        df = self._upsert_row(df, idx, row)
        self._save_log(df)

        logger.info(
            "TradeLogger(%s): logged close result for %s "
            "(equity_close=%.2f, pnl_abs=%.2f, pnl_pct=%.4f)",
            self.symbol,
            d,
            equity_close,
            pnl_abs,
            pnl_pct,
        )

    # =====================================================
    # 内部ヘルパー
    # =====================================================

    @staticmethod
    def _to_date(d: date | datetime) -> date:
        if isinstance(d, datetime):
            return d.date()
        return d

    def _load_log(self) -> pd.DataFrame:
        """
        ログCSVを読み込んで DataFrame を返す。
        存在しない場合は空のDataFrameを返す。
        """
        path = self.log_file
        if not path.exists():
            return pd.DataFrame(
                columns=[
                    "Date",
                    "Ratio",
                    "Action",
                    "Equity_morning_before",
                    "Equity_morning_after",
                    "Equity_close",
                    "PnL_abs",
                    "PnL_pct",
                    "Pos_SOXL_before",
                    "Pos_SOXS_before",
                    "Pos_SOXL_after",
                    "Pos_SOXS_after",
                    "Pos_SOXL_close",
                    "Pos_SOXS_close",
                ]
            )

        df = pd.read_csv(path)
        if "Date" not in df.columns:
            raise KeyError(f"'Date' column not found in log file: {path}")

        # Date を datetime64 に変換
        if not np.issubdtype(df["Date"].dtype, np.datetime64):
            df["Date"] = pd.to_datetime(df["Date"])

        return df

    def _save_log(self, df: pd.DataFrame) -> None:
        """
        DataFrame を CSV に保存する。
        """
        path = self.log_file
        path.parent.mkdir(parents=True, exist_ok=True)
        # Date列をISO形式で保存
        df_to_save = df.copy()
        if "Date" in df_to_save.columns:
            df_to_save["Date"] = pd.to_datetime(df_to_save["Date"]).dt.strftime("%Y-%m-%d")
        df_to_save.to_csv(path, index=False, encoding="utf-8")

    @staticmethod
    def _find_row_index_by_date(df: pd.DataFrame, d: date) -> Optional[int]:
        """
        DataFrame 内に指定日の行があればその index を返し、
        無ければ None を返す。
        """
        if df.empty:
            return None
        mask = pd.to_datetime(df["Date"]).dt.date == d
        if not mask.any():
            return None
        return int(df.loc[mask].index[0])

    @staticmethod
    def _upsert_row(df: pd.DataFrame, idx: Optional[int], row_data: dict) -> pd.DataFrame:
        """
        idx が指定されていればその行を上書き、
        None の場合は新規行として追加する。
        """
        # 全カラム集合を作り、DataFrameに反映
        all_cols = set(df.columns) | set(row_data.keys())
        df = df.reindex(columns=sorted(all_cols))

        if idx is None:
            # 新規行として追加
            new_row = pd.DataFrame([row_data], columns=df.columns)
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            # 既存行の上書き
            for k, v in row_data.items():
                df.at[idx, k] = v

        return df
