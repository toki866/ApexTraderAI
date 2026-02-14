from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

# Envelope 用 11次元の標準カラム名
ENV_COLS = [
    "env_dP_pct",
    "env_theta_norm",
    "env_top_pct",
    "env_bottom_pct",
    "env_D_norm",
    "env_dP",
    "env_D",
    "env_L",
    "env_theta_deg",
    "env_top_abs",
    "env_bottom_abs",
]


def find_pred_col(preds_df: pd.DataFrame, agent_name: str) -> str:
    """
    Pred_Close カラム名を推測するヘルパー。

    探索順:
      - Pred_Close_{AGENT_UPPER}
      - Pred_Close_{agent_name}
      - pred_close_{agent_name}

    いずれも存在しなければ KeyError。
    """
    upper = agent_name.upper()
    candidates = [
        f"Pred_Close_{upper}",
        f"Pred_Close_{agent_name}",
        f"pred_close_{agent_name}",
    ]
    for col in candidates:
        if col in preds_df.columns:
            return col
    raise KeyError(
        f"Pred_Close カラムが見つかりません: agent={agent_name}, "
        f"tried={candidates}"
    )


def prepare_common_base_df(
    prices_df: pd.DataFrame,
    features_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    envelope_daily_df: pd.DataFrame,
    agent_names: List[str],
) -> pd.DataFrame:
    """
    StepE / StepF / PPO で共通利用する「観測用 base_df」を構築する。

    base_df に含まれる主なカラム:
      - Date
      - ret, bh_equity
      - delta_pred_{agent}  （3AI ΔClose予測）
      - ENV_COLS            （Envelope 幾何特徴 11次元）
      - RSI_s1, MACD_s1, Gap_s1
      - Open_s1, High_s1, Low_s1, Close_s1, Volume_s1

    ここで *_s1 系はすべて shift(1) された値（= 前日までの情報）なので、
    観測に用いても未来リークしない。
    """
    # --- 入力をコピーして Date を datetime に統一 ---
    prices = prices_df.copy()
    prices["Date"] = pd.to_datetime(prices["Date"])

    feats = features_df.copy()
    feats["Date"] = pd.to_datetime(feats["Date"])

    preds = preds_df.copy()
    preds["Date"] = pd.to_datetime(preds["Date"])

    env_daily = envelope_daily_df.copy()
    env_daily["Date"] = pd.to_datetime(env_daily["Date"])

    # Envelope カラムが足りない場合は 0 埋め
    for col in ENV_COLS:
        if col not in env_daily.columns:
            env_daily[col] = 0.0

    # Pred_Close 系だけに絞る
    pred_cols = [c for c in preds.columns if c.startswith("Pred_Close")]
    preds_small = preds[["Date"] + pred_cols].copy()

    # 必要なテクニカルだけ抜き出す（余計な列は無視）
    tech_cols = [c for c in ["RSI", "MACD", "Gap"] if c in feats.columns]
    feats_small = feats[["Date"] + tech_cols].copy()

    # 順にマージ
    df = prices.merge(feats_small, on="Date", how="left")
    df = df.merge(preds_small, on="Date", how="left")
    df = df.merge(env_daily[["Date"] + ENV_COLS], on="Date", how="left")

    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # --- 日次リターン & Buy&Hold 資金曲線 ---
    df["ret"] = df["Close"].pct_change().fillna(0.0)
    df["bh_equity"] = (1.0 + df["ret"]).cumprod()

    # --- 価格の shift(1)（前日までの情報だけを観測で使う） ---
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[f"{col}_s1"] = df[col].shift(1)
        else:
            df[f"{col}_s1"] = np.nan

    # --- テクニカルの shift(1) ---
    if "RSI" in df.columns:
        df["RSI_s1"] = df["RSI"].shift(1)
    else:
        df["RSI_s1"] = np.nan

    if "MACD" in df.columns:
        df["MACD_s1"] = df["MACD"].shift(1)
    else:
        df["MACD_s1"] = np.nan

    if "Gap" in df.columns:
        df["Gap_s1"] = df["Gap"].shift(1)
    else:
        df["Gap_s1"] = np.nan

    # --- 3AI ΔClose予測（Pred_Close - Close_{t-1}） ---
    preds_by_date = preds_small.set_index("Date")
    close_prev = df["Close_s1"]

    for agent in agent_names:
        pred_col = find_pred_col(preds_by_date, agent)
        pred_series = preds_by_date[pred_col].reindex(df["Date"])
        delta = pred_series - close_prev
        delta = delta.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        df[f"delta_pred_{agent}"] = delta

    # 最初の行は Close_s1 が NaN になるので落とす
    mask = df["Close_s1"].notna()
    df = df.loc[mask].reset_index(drop=True)

    # 残った NaN には 0 を入れておく（RSI_s1 など）
    df = df.fillna(0.0)

    return df


def build_observation_24d(
    base_df: pd.DataFrame,
    idx: int,
    prev_pnl: float,
    prev_action: float,
    agent_names: List[str],
) -> np.ndarray:
    """
    base_df の idx 行から「24次元観測ベクトル」を構築する。

    ベクトル構成（順番固定）:

      1. 3AI ΔClose 予測
         [ delta_pred_{agent_names[0]},
           delta_pred_{agent_names[1]},
           delta_pred_{agent_names[2]} ]

      2. 11次元: Envelope 幾何特徴
         [ ENV_COLS 順に 11本
           (env_dP_pct, env_theta_norm, ..., env_bottom_abs) ]

      3. 3次元: テクニカル（前日までの情報）
         [ RSI_s1, MACD_s1, Gap_s1 ]

      4. 5次元: 価格（前日までの情報）
         [ Open_s1, High_s1, Low_s1, Close_s1, Volume_s1 ]

      5. 2次元: 履歴
         [ prev_pnl, prev_action ]

    合計 3 + 11 + 3 + 5 + 2 = 24 次元。
    """
    if len(agent_names) != 3:
        raise ValueError(
            f"build_observation_24d は agent_names を3つ想定していますが "
            f"{len(agent_names)} 個が渡されました: {agent_names}"
        )

    row = base_df.iloc[idx]

    vec: List[float] = []

    # 1. 3AI ΔClose予測
    for agent in agent_names:
        vec.append(float(row[f"delta_pred_{agent}"]))

    # 2. Envelope 11次元
    for col in ENV_COLS:
        vec.append(float(row[col]))

    # 3. テクニカル 3次元
    vec.append(float(row["RSI_s1"]))
    vec.append(float(row["MACD_s1"]))
    vec.append(float(row["Gap_s1"]))

    # 4. 価格 5次元
    vec.append(float(row["Open_s1"]))
    vec.append(float(row["High_s1"]))
    vec.append(float(row["Low_s1"]))
    vec.append(float(row["Close_s1"]))
    vec.append(float(row["Volume_s1"]))

    # 5. 履歴 2次元
    vec.append(float(prev_pnl))
    vec.append(float(prev_action))

    if len(vec) != 24:
        raise RuntimeError(f"観測ベクトルの次元が24ではありません: {len(vec)}")

    return np.asarray(vec, dtype=np.float32)


def build_df_obs_24d_from_base_df(
    base_df: pd.DataFrame,
    agent_names: List[str],
) -> pd.DataFrame:
    """
    StepE/StepF/PPO 共通仕様の 24次元観測を、
    base_df 全期間ぶんまとめて DataFrame 化するヘルパー。

    base_df は prepare_common_base_df(...) で構築されている想定。
    """
    if len(agent_names) != 3:
        raise ValueError(
            f"agent_names は3つを想定していますが {len(agent_names)} 個です: {agent_names}"
        )

    n = len(base_df)
    if n == 0:
        return pd.DataFrame()

    rows: List[np.ndarray] = []

    prev_pnl = 0.0
    prev_action = 0.0

    for idx in range(n):
        if idx == 0:
            prev_pnl = 0.0
            prev_action = 0.0
        else:
            # 単純な proxy:
            #   prev_pnl   : 前日のリターン（100%ロングとみなしたPnL%）
            #   prev_action: 前日のリターンの符号（上昇:+1, 下落:-1, フラット:0）
            ret_prev = float(base_df.iloc[idx - 1]["ret"])
            prev_pnl = ret_prev
            if ret_prev > 0:
                prev_action = 1.0
            elif ret_prev < 0:
                prev_action = -1.0
            else:
                prev_action = 0.0

        vec = build_observation_24d(
            base_df=base_df,
            idx=idx,
            prev_pnl=prev_pnl,
            prev_action=prev_action,
            agent_names=agent_names,
        )
        rows.append(vec)

    # 列名を 24次元仕様に合わせて構成
    col_names: List[str] = []
    # 3AI ΔClose 予測
    for agent in agent_names:
        col_names.append(f"delta_pred_{agent}")
    # Envelope 11本
    col_names.extend(ENV_COLS)
    # テクニカル 3本
    col_names.extend(["RSI_s1", "MACD_s1", "Gap_s1"])
    # 価格 5本
    col_names.extend(["Open_s1", "High_s1", "Low_s1", "Close_s1", "Volume_s1"])
    # 履歴 2本
    col_names.extend(["prev_pnl", "prev_action"])

    df_obs_24d = pd.DataFrame(rows, columns=col_names)
    return df_obs_24d
