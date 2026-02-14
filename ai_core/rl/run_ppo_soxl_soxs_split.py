# ai_core/rl/run_ppo_soxl_soxs_split.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ai_core.rl.ppo_trainer import (
    prepare_ppo_inputs_for_soxl_soxs,
    train_single_agent_with_ppo_on_range,
    evaluate_policy_on_range,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SOXL/SOXS ペアで PPO を学習・評価（train/test 分離 or 既存ポリシー評価）"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["train_and_test", "test_only"],
        default="train_and_test",
        help="train_and_test: 学習＋テストを実行, test_only: 既存ポリシーでテストのみ実行",
    )

    # データルート
    parser.add_argument(
        "--data-root",
        type=str,
        default="output",
        help="StepA/StepC/StepD の出力が入っているルートディレクトリ (例: output)",
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

    # シンボル類
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

    # 3AI 名
    parser.add_argument(
        "--agents",
        type=str,
        nargs=3,
        default=["xsr", "lstm", "fed"],
        help="24次元観測で使用する 3AI の名前 (例: xsr lstm fed)",
    )

    # train/test 期間
    parser.add_argument(
        "--train-start",
        type=str,
        required=False,
        help="学習開始日 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--train-end",
        type=str,
        required=False,
        help="学習終了日 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--test-start",
        type=str,
        required=False,
        help="テスト開始日 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--test-end",
        type=str,
        required=False,
        help="テスト終了日 (YYYY-MM-DD)",
    )

    # PPO 設定
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=200_000,
        help="PPO の total_timesteps（train モード時のみ使用）",
    )

    # ポリシー入出力
    parser.add_argument(
        "--policy-output-root",
        type=str,
        default="output/ppo_single_split",
        help="学習済み PPO ポリシー＆結果を保存するルートディレクトリ",
    )
    parser.add_argument(
        "--policy-path",
        type=str,
        default="",
        help="既存ポリシーのパス（mode=test_only の場合必須）",
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
    # PPO 入力（df_obs_24d + 価格2本）を準備（全期間）
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
    # モード分岐
    # -------------------------------
    policy_output_root = Path(args.policy_output_root)
    out_dir = policy_output_root / f"{symbol_long}_{symbol_short}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "train_and_test":
        if not args.train_start or not args.train_end:
            raise SystemExit("mode=train_and_test では --train-start / --train-end が必須です。")
        if not args.test_start or not args.test_end:
            raise SystemExit("mode=train_and_test では --test-start / --test-end が必須です。")

        # 1) 学習
        print("[INFO] Training PPO agent on train range...")
        print(f"  train_start: {args.train_start}")
        print(f"  train_end  : {args.train_end}")
        print(f"  timesteps  : {args.total_timesteps}")
        print(f"  output_dir : {out_dir}")

        train_result = train_single_agent_with_ppo_on_range(
            df_obs_24d=df_obs_24d,
            dates=dates,
            price_long=price_long,
            price_short=price_short,
            train_start=args.train_start,
            train_end=args.train_end,
            total_timesteps=args.total_timesteps,
            policy_output_dir=out_dir,
        )

        # 2) テスト（学習したポリシーを test_start〜test_end だけ評価）
        print("[INFO] Evaluating trained PPO agent on test range...")
        print(f"  test_start: {args.test_start}")
        print(f"  test_end  : {args.test_end}")

        test_result = evaluate_policy_on_range(
            policy_path=train_result.policy_path,
            df_obs_24d=df_obs_24d,
            dates=dates,
            price_long=price_long,
            price_short=price_short,
            start_date=args.test_start,
            end_date=args.test_end,
        )

        # 3) 保存
        equity_train_path = out_dir / "ppo_equity_curve_train.csv"
        train_result.equity_curve.to_csv(equity_train_path, index=False)

        equity_test_path = out_dir / "ppo_equity_curve_test.csv"
        test_result.equity_curve.to_csv(equity_test_path, index=False)

        metrics_train_path = out_dir / "ppo_metrics_train.csv"
        pd.DataFrame([train_result.metrics]).to_csv(metrics_train_path, index=False)

        metrics_test_path = out_dir / "ppo_metrics_test.csv"
        pd.DataFrame([test_result.metrics]).to_csv(metrics_test_path, index=False)

        print("[INFO] Training & Test finished.")
        print(f"  Policy saved to           : {train_result.policy_path}")
        print(f"  Train equity curve saved  : {equity_train_path}")
        print(f"  Test  equity curve saved  : {equity_test_path}")
        print(f"  Train metrics saved       : {metrics_train_path}")
        print(f"  Test  metrics saved       : {metrics_test_path}")

        print("")
        print("---- Train Metrics ----")
        for k, v in train_result.metrics.items():
            print(f"{k:16s}: {v}")

        print("")
        print("---- Test Metrics ----")
        for k, v in test_result.metrics.items():
            print(f"{k:16s}: {v}")

    elif args.mode == "test_only":
        if not args.policy_path:
            raise SystemExit("mode=test_only では --policy-path が必須です。")
        if not args.test_start or not args.test_end:
            raise SystemExit("mode=test_only では --test-start / --test-end が必須です。")

        policy_path = Path(args.policy_path)

        print("[INFO] Evaluating existing PPO policy on test range...")
        print(f"  policy    : {policy_path}")
        print(f"  test_start: {args.test_start}")
        print(f"  test_end  : {args.test_end}")

        test_result = evaluate_policy_on_range(
            policy_path=policy_path,
            df_obs_24d=df_obs_24d,
            dates=dates,
            price_long=price_long,
            price_short=price_short,
            start_date=args.test_start,
            end_date=args.test_end,
        )

        equity_test_path = out_dir / "ppo_equity_curve_test_only.csv"
        test_result.equity_curve.to_csv(equity_test_path, index=False)

        metrics_test_path = out_dir / "ppo_metrics_test_only.csv"
        pd.DataFrame([test_result.metrics]).to_csv(metrics_test_path, index=False)

        print("[INFO] Test-only evaluation finished.")
        print(f"  Equity curve saved : {equity_test_path}")
        print(f"  Metrics saved      : {metrics_test_path}")

        print("")
        print("---- Test Metrics ----")
        for k, v in test_result.metrics.items():
            print(f"{k:16s}: {v}")


if __name__ == "__main__":
    main()
