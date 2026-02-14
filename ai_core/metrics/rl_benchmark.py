# ai_core/metrics/rl_benchmark.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ai_core.types.common import DateRange, StepResult
from ai_core.gui._utils_dates import _ensure_datetime


@dataclass
class SegmentMetrics:
    segment: str          # "train" / "test" / "full"
    kind: str             # "rl" / "bh"
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    final_return: float
    max_drawdown: float
    sharpe_ratio: float

    def to_dict(self) -> Dict:
        return asdict(self)


def _calc_equity_metrics(equity: pd.Series, segment: str, kind: str) -> SegmentMetrics:
    equity = equity.dropna().sort_index()
    if equity.empty or len(equity) < 2:
        return SegmentMetrics(
            segment=segment,
            kind=kind,
            start_date=pd.NaT,
            end_date=pd.NaT,
            final_return=np.nan,
            max_drawdown=np.nan,
            sharpe_ratio=np.nan,
        )

    start_date = equity.index[0]
    end_date = equity.index[-1]

    # 最終リターン
    final_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)

    # 最大ドローダウン
    hwm = equity.cummax()
    dd = equity / hwm - 1.0
    max_dd = float(dd.min())

    # 日次シャープレシオ → 年率換算
    ret = equity.pct_change().dropna()
    if ret.std() == 0 or len(ret) < 2:
        sharpe = np.nan
    else:
        sharpe = float(ret.mean() / ret.std() * np.sqrt(252))

    return SegmentMetrics(
        segment=segment,
        kind=kind,
        start_date=start_date,
        end_date=end_date,
        final_return=final_return,
        max_drawdown=max_dd,
        sharpe_ratio=sharpe,
    )


def _simulate_buy_and_hold(
    prices: pd.DataFrame,
    date_col: str = "Date",
    close_col: str = "Close",
    initial_capital: float = 1.0,
) -> pd.DataFrame:
    """単純な Buy & Hold の資金曲線を生成。"""
    df = prices.copy()
    if date_col not in df.columns:
        cand = [c for c in df.columns if c.lower() == "date"]
        if cand:
            date_col = cand[0]
        else:
            raise KeyError(f"date_col '{date_col}' not found in prices")
    if close_col not in df.columns:
        cand = [c for c in df.columns if c.lower() == "close"]
        if cand:
            close_col = cand[0]
        else:
            raise KeyError(f"close_col '{close_col}' not found in prices")

    df[date_col] = _ensure_datetime(df[date_col])
    df = df.sort_values(date_col)
    df = df[[date_col, close_col]].dropna()

    if df.empty:
        raise ValueError("prices is empty in _simulate_buy_and_hold")

    first_price = df[close_col].iloc[0]
    if first_price <= 0:
        raise ValueError("first_price must be > 0 for buy & hold")

    # 初期資金を全額インする想定
    units = initial_capital / first_price
    df["equity_bh"] = units * df[close_col]

    return df.rename(columns={date_col: "date"})


def compute_train_test_and_bh_metrics(
    daily_log: pd.DataFrame,
    prices: pd.DataFrame,
    date_range: DateRange,
    initial_capital: float = 1.0,
    date_col_daily: str = "date",
    equity_col_daily: str = "equity",
    date_col_prices: str = "Date",
    close_col_prices: str = "Close",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Train/Test/Full × (RL / Buy & Hold) の指標をまとめて計算。"""
    dl = daily_log.copy()

    if date_col_daily not in dl.columns:
        raise KeyError(f"daily_log に '{date_col_daily}' 列がありません")
    if equity_col_daily not in dl.columns:
        raise KeyError(f"daily_log に '{equity_col_daily}' 列がありません")

    dl[date_col_daily] = _ensure_datetime(dl[date_col_daily])
    dl = dl.sort_values(date_col_daily)

    # RL の資金曲線
    rl_equity = dl.set_index(date_col_daily)[equity_col_daily]

    # Buy & Hold の資金曲線
    bh_equity_df = _simulate_buy_and_hold(
        prices,
        date_col=date_col_prices,
        close_col=close_col_prices,
        initial_capital=initial_capital,
    )
    bh_equity = bh_equity_df.set_index("date")["equity_bh"]

    # 日付範囲
    dr = date_range
    train_start = pd.to_datetime(dr.train_start)
    train_end   = pd.to_datetime(dr.train_end)
    test_start  = pd.to_datetime(dr.test_start)
    test_end    = pd.to_datetime(dr.test_end)

    metrics: List[SegmentMetrics] = []

    # full
    metrics.append(_calc_equity_metrics(rl_equity, "full", "rl"))
    metrics.append(_calc_equity_metrics(bh_equity, "full", "bh"))

    # train
    rl_train = rl_equity.loc[(rl_equity.index >= train_start) & (rl_equity.index <= train_end)]
    bh_train = bh_equity.loc[(bh_equity.index >= train_start) & (bh_equity.index <= train_end)]
    metrics.append(_calc_equity_metrics(rl_train, "train", "rl"))
    metrics.append(_calc_equity_metrics(bh_train, "train", "bh"))

    # test
    rl_test = rl_equity.loc[(rl_equity.index >= test_start) & (rl_equity.index <= test_end)]
    bh_test = bh_equity.loc[(bh_equity.index >= test_start) & (bh_equity.index <= test_end)]
    metrics.append(_calc_equity_metrics(rl_test, "test", "rl"))
    metrics.append(_calc_equity_metrics(bh_test, "test", "bh"))

    metrics_df = pd.DataFrame([m.to_dict() for m in metrics])
    return metrics_df, bh_equity_df


def enrich_step_result_with_benchmark(
    step_result: StepResult,
    prices: pd.DataFrame,
    date_range: DateRange,
    initial_capital: float = 1.0,
    date_col_daily: str = "date",
    equity_col_daily: str = "equity",
    date_col_prices: str = "Date",
    close_col_prices: str = "Close",
) -> None:
    """StepResult に train/test ＋ Buy & Hold の指標を追記する。

    前提:
        - step_result.artifacts["daily_log_path"] が存在していること
          （StepE / StepF が既に daily_log を CSV に保存している想定）

    ここで行うこと:
        - daily_log をロード
        - Train/Test/Full の RL vs Buy&Hold 指標を計算
        - 指標を CSV 保存
        - step_result.metrics / step_result.artifacts に追記
    """
    artifacts = step_result.artifacts
    metrics = step_result.metrics

    if "daily_log_path" not in artifacts:
        raise KeyError("step_result.artifacts に 'daily_log_path' がありません")

    daily_log_path = artifacts["daily_log_path"]
    dl = pd.read_csv(daily_log_path)

    metrics_df, bh_equity_df = compute_train_test_and_bh_metrics(
        daily_log=dl,
        prices=prices,
        date_range=date_range,
        initial_capital=initial_capital,
        date_col_daily=date_col_daily,
        equity_col_daily=equity_col_daily,
        date_col_prices=date_col_prices,
        close_col_prices=close_col_prices,
    )

    def _get_value(seg: str, kind: str, col: str) -> float:
        row = metrics_df[(metrics_df["segment"] == seg) & (metrics_df["kind"] == kind)]
        if row.empty:
            return float("nan")
        return float(row.iloc[0][col])

    rl_test_ret = _get_value("test", "rl", "final_return")
    bh_test_ret = _get_value("test", "bh", "final_return")

    metrics["rl_test_final_return"] = rl_test_ret
    metrics["bh_test_final_return"] = bh_test_ret
    metrics["rl_test_excess_return_vs_bh"] = rl_test_ret - bh_test_ret

    metrics["rl_train_final_return"] = _get_value("train", "rl", "final_return")
    metrics["bh_train_final_return"] = _get_value("train", "bh", "final_return")

    import os
    base_dir = os.path.dirname(daily_log_path)
    metrics_csv_path = os.path.join(base_dir, "rl_metrics_train_test_bh.csv")
    bh_equity_path   = os.path.join(base_dir, "bh_equity_curve.csv")

    metrics_df.to_csv(metrics_csv_path, index=False)
    bh_equity_df.to_csv(bh_equity_path, index=False)

    artifacts["metrics_train_test_bh_path"] = metrics_csv_path
    artifacts["bh_equity_path"] = bh_equity_path
