# ai_core/rl/run_ppo_soxl_soxs.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ai_core.rl.ppo_trainer import (
    prepare_ppo_inputs_for_soxl_soxs,
    train_single_agent_with_ppo,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SOXL/SOXS ペアで PPO を学習・評価する実行スクリプト"
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="output",
        help="StepA/StepC/StepD の出力が入っているルートディレクトリ "
             "(例: output)",
    )
    parser.add_argument(
        "--stepA-dirname",
        type=str,
        default="stepA",
        help="StepA 出力ディレクトリ名 (data_root/stepA)",
    )
    parser.add_argument(
        "--stepC-dirname",
        type=str,
        default="stepC",
        help="StepC 出力ディレクトリ名 (data_root/stepC)",
    )
    parser.add_argument(
        "--stepD-dirname",
        type=str,
        default="stepD",
        help="StepD 出力ディレクトリ名 (data_root/stepD)",
    )

    parser.add_argument(
        "--symbol-long",
        type=str,
        default="SOXL",
        help="ロング側シンボル (デフォルト: SOXL)",
    )
    parser.add_argument(
        "--symbol-short",
        type=str,
        default="SOXS",
        help="ショート側シンボル (デフォルト: SOXS)",
    )

    parser.add_argument(
        "--env-agent",
        type=str,
        default="xsr",
        help="StepD の日次 Envelope ファイルに使うエージェント名 "
             "(例: xsr → stepD_events_xsr_SOXL_daily.csv)",
    )

    parser.add_argument(
        "--agents",
        type=str,
        nargs=3,
        default=["xsr", "lstm", "fed"],
        help="24次元観測で使用する 3AI の名前リスト "
             "(例: xsr lstm fed)",
    )

    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=200_000,
        help="PPO の total_timesteps",
    )

    parser.add_argument(
        "--policy-output-root",
        type=str,
        default="output/ppo_single",
        help="学習済み PPO ポリシーと結果を保存するルートディレクトリ",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root)
    stepA_dir = data_root / args.stepA_dirname
    stepC_dir = data_root / args.stepC_dirname
    stepD_dir = data_root / args.stepD_dirname

    symbol_long = args.symbol_long
    symbol_short = args.symbol_short
    env_agent = args.env_agent
    agent_names = args.agents

    # -------------------------------
    # 入力 CSV パスの組み立て
    # -------------------------------
    prices_soxl_path = stepA_dir / f"stepA_prices_{symbol_long}.csv"
    prices_soxs_path = stepA_dir / f"stepA_prices_{symbol_short}.csv"
    features_soxl_path = stepA_dir / f"stepA_features_{symbol_long}.csv"
    preds_soxl_path = stepC_dir / f"stepC_pred_time_all_{symbol_long}.csv"
    envelope_daily_path = (
        stepD_dir / f"stepD_events_{env_agent}_{symbol_long}_daily.csv"
    )

    # -------------------------------
    # CSV 読み込み
    # -------------------------------
    print("[INFO] Loading CSV files...")
    print(f"  prices_soxl   : {prices_soxl_path}")
    print(f"  prices_soxs   : {prices_soxs_path}")
    print(f"  features_soxl : {features_soxl_path}")
    print(f"  preds_soxl    : {preds_soxl_path}")
    print(f"  envelope_daily: {envelope_daily_path}")

    prices_soxl = pd.read_csv(prices_soxl_path)
    prices_soxs = pd.read_csv(prices_soxs_path)
    features_soxl = pd.read_csv(features_soxl_path)
    preds_soxl = pd.read_csv(preds_soxl_path)
    envelope_daily_soxl = pd.read_csv(envelope_daily_path)

    # -------------------------------
    # PPO 入力（df_obs_24d + 価格2本）を準備
    # -------------------------------
    print("[INFO] Preparing 24-dim observations and price series...")
    df_obs_24d, dates, price_long, price_short = prepare_ppo_inputs_for_soxl_soxs(
        prices_soxl=prices_soxl,
        prices_soxs=prices_soxs,
        features_soxl=features_soxl,
        preds_soxl=preds_soxl,
        envelope_daily_soxl=envelope_daily_soxl,
        agent_names=agent_names,
    )

    print(f"[INFO] df_obs_24d shape: {df_obs_24d.shape}")
    print(f"[INFO] dates range     : {dates.iloc[0]} 〜 {dates.iloc[-1]}")

    # -------------------------------
    # PPO 学習・評価
    # -------------------------------
    policy_output_root = Path(args.policy_output_root)
    policy_output_dir = policy_output_root / f"{symbol_long}_{symbol_short}"

    print("[INFO] Training PPO agent...")
    print(f"  total_timesteps: {args.total_timesteps}")
    print(f"  output_dir     : {policy_output_dir}")

    result = train_single_agent_with_ppo(
        df_obs_24d=df_obs_24d,
        dates=dates,
        price_long=price_long,
        price_short=price_short,
        total_timesteps=args.total_timesteps,
        policy_output_dir=policy_output_dir,
    )

    # -------------------------------
    # 結果を保存
    # -------------------------------
    policy_output_dir.mkdir(parents=True, exist_ok=True)

    equity_csv_path = policy_output_dir / "ppo_equity_curve.csv"
    result.equity_curve.to_csv(equity_csv_path, index=False)

    metrics_csv_path = policy_output_dir / "ppo_metrics.csv"
    pd.DataFrame([result.metrics]).to_csv(metrics_csv_path, index=False)

    print("[INFO] Training finished.")
    print(f"  Policy saved to     : {result.policy_path}")
    print(f"  Equity curve saved  : {equity_csv_path}")
    print(f"  Metrics saved       : {metrics_csv_path}")
    print("")
    print("---- Metrics ----")
    for k, v in result.metrics.items():
        print(f"{k:16s}: {v}")


if __name__ == "__main__":
    main()
