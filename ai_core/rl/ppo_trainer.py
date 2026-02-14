# ai_core/rl/ppo_trainer.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from ai_core.rl.trading_env import TradingEnv, TradingEnvConfig
from ai_core.rl.common_observation import (
    prepare_common_base_df,
    build_df_obs_24d_from_base_df,
)


@dataclass
class SingleRLTrainResult:
    """
    SB3 PPO による単体エージェント学習 or 評価の結果。

    Attributes
    ----------
    equity_curve : pd.DataFrame
        エクイティカーブ（Date, Equity）。
    metrics : Dict[str, float]
        主要指標。
        - final_return     : エージェントの最終リターン
        - max_drawdown     : エージェントの最大DD
        - sharpe_ratio     : エージェントのシャープレシオ
        - bh_final_return  : Buy & Hold の最終リターン
        - bh_max_drawdown  : Buy & Hold の最大DD
        - bh_sharpe_ratio  : Buy & Hold のシャープレシオ
    policy_path : Path
        学習済み PPO モデルの保存パス
        （評価のみの場合も、そのポリシーのパスをそのまま入れる）。
    """

    equity_curve: pd.DataFrame
    metrics: Dict[str, float]
    policy_path: Path


# ======================================================================
# 共通メトリクス計算ヘルパー
# ======================================================================


def _calc_metrics_from_equity(eq: pd.Series) -> Dict[str, float]:
    """
    エクイティ系列から final_return / max_drawdown / sharpe_ratio を計算する。

    eq : 累積エクイティ系列（例: [1.0, 1.01, 1.005, ...]）
    """
    eq = eq.astype(float)
    if len(eq) == 0:
        return {"final_return": 0.0, "max_drawdown": 0.0, "sharpe_ratio": 0.0}

    returns = eq.pct_change().fillna(0.0)
    final_return = float(eq.iloc[-1] - 1.0)

    # 最大ドローダウン
    cummax = eq.cummax()
    dd = (eq - cummax) / cummax
    max_dd = float(dd.min())

    # 単純な日次シャープレシオ
    std = float(returns.std())
    if std > 0:
        sharpe = float(np.sqrt(252.0) * returns.mean() / std)
    else:
        sharpe = 0.0

    return {
        "final_return": final_return,
        "max_drawdown": max_dd,
        "sharpe_ratio": sharpe,
    }


def _build_buy_and_hold_equity(
    price: pd.Series,
    initial_equity: float = 1.0,
) -> pd.Series:
    """
    単純な Buy & Hold (ロング側銘柄) のエクイティカーブを作成する。

    price : 価格系列（終値など）
    """
    price = price.astype(float)
    if len(price) == 0:
        return pd.Series(dtype=float)

    p0 = float(price.iloc[0])
    if p0 <= 0:
        # 異常時はフラットな 1.0 を返す
        return pd.Series([initial_equity] * len(price))

    ret = (price / p0).fillna(1.0)
    eq = initial_equity * ret
    return eq


def _subset_by_date_range(
    df_obs_24d: pd.DataFrame,
    dates: pd.Series,
    price_long: pd.Series,
    price_short: pd.Series,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    dates の start_date〜end_date の範囲で df_obs_24d / price_long / price_short を切り出す。
    """
    d0 = pd.to_datetime(start_date)
    d1 = pd.to_datetime(end_date)

    mask = (dates >= d0) & (dates <= d1)
    idx = np.where(mask.to_numpy())[0]

    if len(idx) == 0:
        raise ValueError(
            f"指定した期間にデータがありません: start={d0}, end={d1}"
        )

    df_sub = df_obs_24d.iloc[idx].reset_index(drop=True)
    dates_sub = dates.iloc[idx].reset_index(drop=True)
    pl_sub = price_long.iloc[idx].reset_index(drop=True)
    ps_sub = price_short.iloc[idx].reset_index(drop=True)

    return df_sub, dates_sub, pl_sub, ps_sub


# ======================================================================
# SOXL / SOXS ペア用の前処理ヘルパー
# ======================================================================


def _normalize_date(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """Date 列を datetime64[ns] に揃えてソートしなおす小ヘルパー。"""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out.sort_values(date_col, inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def prepare_ppo_inputs_for_soxl_soxs(
    prices_long: pd.DataFrame,
    prices_short: pd.DataFrame,
    features_long: pd.DataFrame,
    preds_long: pd.DataFrame,
    env_daily_long: pd.DataFrame,
    agent_names: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    SOXL/SOXS ペア用に PPO 学習向けの入力（base_df / df_obs_24d / 日付 / 価格2本）を準備する。

    Parameters
    ----------
    prices_long : pd.DataFrame
        ロング側銘柄（例: SOXL）の価格データ（少なくとも Date, Close を含む）。
        StepA: stepA_prices_SOXL.csv 相当。

    prices_short : pd.DataFrame
        ショート側銘柄（例: SOXS）の価格データ（少なくとも Date, Close を含む）。
        StepA: stepA_prices_SOXS.csv 相当。

    features_long : pd.DataFrame
        ロング側銘柄向け特徴量 DataFrame（少なくとも Date, RSI, MACD, Gap を含む）。
        StepA: stepA_features_SOXL.csv 相当。

    preds_long : pd.DataFrame
        ロング側銘柄向け 3AI Pred_Close 系列（少なくとも Date, Pred_Close_* を含む）。
        StepC: stepC_pred_time_all_SOXL.csv 相当。

    env_daily_long : pd.DataFrame
        ロング側銘柄向け日次 Envelope 幾何特徴（ENV_COLS 11本を含む）。
        StepD で出力した日次サマリ。

    agent_names : list[str]
        3AI の名前（例: ["xsr", "lstm", "fed"]）。

    Returns
    -------
    base_df : pd.DataFrame
        Date を含む共通のベース DataFrame。
    df_obs_24d : pd.DataFrame
        24次元観測ベクトルのみを持つ DataFrame。
        行数は base_df と同じ。
    dates : pd.Series
        観測に対応する日付系列（datetime64[ns]）。
    price_long : pd.Series
        ロング側銘柄の Close 価格系列（df_obs_24d と同じ長さ）。
    price_short : pd.Series
        ショート側銘柄の Close 価格系列（df_obs_24d と同じ長さ）。
    """
    # --- Date 正規化 ---
    prices_long = _normalize_date(prices_long, "Date")
    prices_short = _normalize_date(prices_short, "Date")
    features_long = _normalize_date(features_long, "Date")
    preds_long = _normalize_date(preds_long, "Date")
    env_daily_long = _normalize_date(env_daily_long, "Date")

    # --- ロング / ショートの共通営業日を決定 ---
    dates_long_set = set(prices_long["Date"])
    dates_short_set = set(prices_short["Date"])
    common_dates = sorted(dates_long_set & dates_short_set)
    if not common_dates:
        raise ValueError("ロング銘柄とショート銘柄の共通日付がありません。データを確認してください。")

    prices_long = prices_long[prices_long["Date"].isin(common_dates)].reset_index(drop=True)
    prices_short = prices_short[prices_short["Date"].isin(common_dates)].reset_index(drop=True)

    # --- ロング側で base_df を構築 ---
    base_df = prepare_common_base_df(
        prices_df=prices_long,
        features_df=features_long,
        preds_df=preds_long,
        envelope_daily_df=env_daily_long,
        agent_names=agent_names,
    )

    # base_df の Date にショート側の Close をそろえる
    short_close_by_date = (
        prices_short[["Date", "Close"]].set_index("Date").sort_index()
    )
    price_short = short_close_by_date.reindex(base_df["Date"])["Close"]

    if price_short.isna().any():
        raise ValueError(
            "base_df の日付に対応するショート銘柄 Close が存在しません。"
            "StepA の価格データを確認してください。"
        )

    # ロング側の Close は base_df からそのまま取る
    price_long = base_df["Close"].astype(float).reset_index(drop=True)
    price_short = price_short.astype(float).reset_index(drop=True)
    dates = base_df["Date"].reset_index(drop=True)

    # --- 24次元観測 df_obs_24d を構築 ---
    df_obs_24d = build_df_obs_24d_from_base_df(base_df, agent_names)

    return base_df, df_obs_24d, dates, price_long, price_short


# ======================================================================
# ① 全期間そのまま学習（従来版）
# ======================================================================


def train_single_agent_with_ppo(
    df_obs_24d: pd.DataFrame,
    dates: pd.Series,
    price_long: pd.Series,
    price_short: pd.Series,
    total_timesteps: int,
    policy_output_dir: Optional[Path] = None,
    env_config: Optional[TradingEnvConfig] = None,
) -> SingleRLTrainResult:
    """
    df_obs_24d（全期間）をそのまま使って PPO を学習し、
    同じ期間で評価する従来版ヘルパー。
    """
    if not (len(df_obs_24d) == len(dates) == len(price_long) == len(price_short)):
        raise ValueError("df_obs_24d / dates / price_long / price_short の長さが一致していません。")

    env_cfg = env_config or TradingEnvConfig()
    if policy_output_dir is None:
        policy_output_dir = Path("output") / "policies_sb3"

    # Gym 環境の構築
    env = TradingEnv(
        df_obs=df_obs_24d.reset_index(drop=True),
        price_long=price_long.reset_index(drop=True),
        price_short=price_short.reset_index(drop=True),
        config=env_cfg,
    )

    # PPO モデル学習
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log=None,
    )
    model.learn(total_timesteps=total_timesteps)

    # ポリシー保存
    policy_output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = policy_output_dir / "ppo_policy.zip"
    model.save(policy_path)

    # フル期間で評価
    eval_env = TradingEnv(
        df_obs=df_obs_24d.reset_index(drop=True),
        price_long=price_long.reset_index(drop=True),
        price_short=price_short.reset_index(drop=True),
        config=env_cfg,
    )
    obs, _ = eval_env.reset()

    equities: List[float] = [env_cfg.initial_equity]
    eq_dates: List[pd.Timestamp] = [pd.to_datetime(dates.iloc[0])]

    for i in range(len(df_obs_24d) - 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = eval_env.step(action)

        equities.append(float(info["equity"]))
        eq_dates.append(pd.to_datetime(dates.iloc[i + 1]))

        if done:
            break

    equity_df = pd.DataFrame({"Date": eq_dates, "Equity": equities})
    eq_series = pd.Series(equities, index=pd.to_datetime(eq_dates))
    metrics = _calc_metrics_from_equity(eq_series)

    # Buy & Hold (ロング銘柄) 側のメトリクス
    bh_eq = _build_buy_and_hold_equity(
        price=price_long.reset_index(drop=True),
        initial_equity=env_cfg.initial_equity,
    )
    bh_eq.index = pd.to_datetime(dates.reset_index(drop=True))
    bh_metrics = _calc_metrics_from_equity(bh_eq)

    metrics["bh_final_return"] = bh_metrics["final_return"]
    metrics["bh_max_drawdown"] = bh_metrics["max_drawdown"]
    metrics["bh_sharpe_ratio"] = bh_metrics["sharpe_ratio"]

    return SingleRLTrainResult(
        equity_curve=equity_df,
        metrics=metrics,
        policy_path=policy_path,
    )


# ======================================================================
# ② train_start〜train_end だけ学習する版（StepE 用）
# ======================================================================


def train_single_agent_with_ppo_on_range(
    df_obs_24d: pd.DataFrame,
    dates: pd.Series,
    price_long: pd.Series,
    price_short: pd.Series,
    train_start: str | pd.Timestamp,
    train_end: str | pd.Timestamp,
    total_timesteps: int,
    policy_output_dir: Optional[Path] = None,
    env_config: Optional[TradingEnvConfig] = None,
) -> SingleRLTrainResult:
    """
    学習期間（train_start〜train_end）だけを使って PPO を学習し、
    その学習期間内で評価するヘルパー。

    テスト期間の評価は evaluate_policy_on_range(...) で行う。
    StepEService.run_sb3_ppo_split から呼ばれることを想定。
    """
    if not (len(df_obs_24d) == len(dates) == len(price_long) == len(price_short)):
        raise ValueError("df_obs_24d / dates / price_long / price_short の長さが一致していません。")

    if policy_output_dir is None:
        policy_output_dir = Path("output") / "policies_sb3"

    # 学習期間でサブセット
    d0 = pd.to_datetime(train_start)
    d1 = pd.to_datetime(train_end)

    df_train, dates_train, price_long_train, price_short_train = _subset_by_date_range(
        df_obs_24d=df_obs_24d,
        dates=dates,
        price_long=price_long,
        price_short=price_short,
        start_date=d0,
        end_date=d1,
    )

    env_cfg = env_config or TradingEnvConfig()

    # 学習用Env
    env = TradingEnv(
        df_obs=df_train.reset_index(drop=True),
        price_long=price_long_train.reset_index(drop=True),
        price_short=price_short_train.reset_index(drop=True),
        config=env_cfg,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log=None,
    )
    model.learn(total_timesteps=total_timesteps)

    # ポリシー保存
    policy_output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = policy_output_dir / "ppo_policy.zip"
    model.save(policy_path)

    # 学習期間内で評価（学習データに対するパフォーマンス）
    eval_env = TradingEnv(
        df_obs=df_train.reset_index(drop=True),
        price_long=price_long_train.reset_index(drop=True),
        price_short=price_short_train.reset_index(drop=True),
        config=env_cfg,
    )
    obs, _ = eval_env.reset()

    equities: List[float] = [env_cfg.initial_equity]
    eq_dates: List[pd.Timestamp] = [pd.to_datetime(dates_train.iloc[0])]

    for i in range(len(df_train) - 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = eval_env.step(action)

        equities.append(float(info["equity"]))
        eq_dates.append(pd.to_datetime(dates_train.iloc[i + 1]))

        if done:
            break

    equity_df = pd.DataFrame({"Date": eq_dates, "Equity": equities})
    eq_series = pd.Series(equities, index=pd.to_datetime(eq_dates))
    metrics = _calc_metrics_from_equity(eq_series)

    # Buy & Hold (ロング銘柄) 側のメトリクス（学習期間内のみ）
    bh_eq = _build_buy_and_hold_equity(
        price=price_long_train.reset_index(drop=True),
        initial_equity=env_cfg.initial_equity,
    )
    bh_eq.index = pd.to_datetime(dates_train.reset_index(drop=True))
    bh_metrics = _calc_metrics_from_equity(bh_eq)

    metrics["bh_final_return"] = bh_metrics["final_return"]
    metrics["bh_max_drawdown"] = bh_metrics["max_drawdown"]
    metrics["bh_sharpe_ratio"] = bh_metrics["sharpe_ratio"]

    return SingleRLTrainResult(
        equity_curve=equity_df,
        metrics=metrics,
        policy_path=policy_path,
    )


# ======================================================================
# ③ 既存ポリシーを読み込んで、任意期間だけ評価する版
# ======================================================================


def evaluate_policy_on_range(
    policy_path: Path | str,
    df_obs_24d: pd.DataFrame,
    dates: pd.Series,
    price_long: pd.Series,
    price_short: pd.Series,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    env_config: Optional[TradingEnvConfig] = None,
) -> SingleRLTrainResult:
    """
    保存済み PPO ポリシーを読み込み、
    start_date〜end_date の期間だけシミュレーションして評価する。

    学習済みモデルは「train_single_agent_with_ppo_*」で作ったものを想定。
    """
    if not (len(df_obs_24d) == len(dates) == len(price_long) == len(price_short)):
        raise ValueError("df_obs_24d / dates / price_long / price_short の長さが一致していません。")

    d0 = pd.to_datetime(start_date)
    d1 = pd.to_datetime(end_date)

    df_eval, dates_eval, price_long_eval, price_short_eval = _subset_by_date_range(
        df_obs_24d=df_obs_24d,
        dates=dates,
        price_long=price_long,
        price_short=price_short,
        start_date=d0,
        end_date=d1,
    )

    env_cfg = env_config or TradingEnvConfig()

    env = TradingEnv(
        df_obs=df_eval.reset_index(drop=True),
        price_long=price_long_eval.reset_index(drop=True),
        price_short=price_short_eval.reset_index(drop=True),
        config=env_cfg,
    )

    policy_path = Path(policy_path)
    model = PPO.load(str(policy_path), env=env)

    obs, _ = env.reset()

    equities: List[float] = [env_cfg.initial_equity]
    eq_dates: List[pd.Timestamp] = [pd.to_datetime(dates_eval.iloc[0])]

    for i in range(len(df_eval) - 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        equities.append(float(info["equity"]))
        eq_dates.append(pd.to_datetime(dates_eval.iloc[i + 1]))

        if done:
            break

    equity_df = pd.DataFrame({"Date": eq_dates, "Equity": equities})
    eq_series = pd.Series(equities, index=pd.to_datetime(eq_dates))
    metrics = _calc_metrics_from_equity(eq_series)

    # Buy & Hold (ロング銘柄) 側のメトリクス（評価期間内）
    bh_eq = _build_buy_and_hold_equity(
        price=price_long_eval.reset_index(drop=True),
        initial_equity=env_cfg.initial_equity,
    )
    bh_eq.index = pd.to_datetime(dates_eval.reset_index(drop=True))
    bh_metrics = _calc_metrics_from_equity(bh_eq)

    metrics["bh_final_return"] = bh_metrics["final_return"]
    metrics["bh_max_drawdown"] = bh_metrics["max_drawdown"]
    metrics["bh_sharpe_ratio"] = bh_metrics["sharpe_ratio"]

    return SingleRLTrainResult(
        equity_curve=equity_df,
        metrics=metrics,
        policy_path=policy_path,
    )
