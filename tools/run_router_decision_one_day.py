# -*- coding: utf-8 -*-
# run_router_decision_one_day.py
#
# Usage (Windows cmd):
#   python tools\run_router_decision_one_day.py --symbol SOXL --output-root output --mode sim --date 2022-01-03 ^
#     --router-table output\stepF\sim\router_table_SOXL.yaml ^
#     --router-log output\engine\router_log_SOXL.csv ^
#     --agent dprime_all_features_h01=output\policies\policy_stepE_dprime_all_features_h01_SOXL.zip ^
#     --agent dprime_all_features_h02=output\policies\policy_stepE_dprime_all_features_h02_SOXL.zip ^
#     --agent dprime_all_features_h03=output\policies\policy_stepE_dprime_all_features_h03_SOXL.zip
#
# Notes:
#   - This script does NOT place any orders. It only runs: StateBuilder -> LivePolicyRunner(router) -> prints ratio + router_info
#   - Agent keys MUST match router_table.yaml primary_agent values.
#   - Policy files can be .zip (SB3 PPO) or .npz (linear). If .zip, stable_baselines3 must be installed.

from __future__ import annotations

import argparse
from pathlib import Path
import datetime as dt

from engine.live_policy_runner import LivePolicyConfig, LivePolicyRunner
from engine.state_builder import StateBuilder, StateBuilderConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--output-root", default="output")
    p.add_argument("--mode", default="sim", choices=["sim", "live", "ops", "display"])
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--router-table", required=True, help="Path to router_table.yaml/json")
    p.add_argument("--router-min-hold-days", type=int, default=1)
    p.add_argument("--router-log", default="", help="Optional CSV path to append router decisions")
    p.add_argument("--obs-dim", type=int, default=24)
    p.add_argument(
        "--agent",
        action="append",
        default=[],
        help="Agent mapping 'agent_key=policy_path'. Repeatable.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    d = dt.date.fromisoformat(args.date)
    symbol = args.symbol
    output_root = Path(args.output_root)

    policy_paths: dict[str, Path] = {}
    agent_names: list[str] = []
    for item in args.agent:
        if "=" not in item:
            raise SystemExit(f"--agent must be in 'agent_key=policy_path' form. got: {item}")
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k or not v:
            raise SystemExit(f"Invalid --agent value: {item}")
        policy_paths[k] = Path(v)
        agent_names.append(k)

    if not agent_names:
        raise SystemExit("No agents provided. Add at least one --agent agent_key=policy_path")

    live_cfg = LivePolicyConfig(
        symbol=symbol,
        mode="router",
        agent_names=agent_names,
        policy_paths=policy_paths,
        router_table_path=Path(args.router_table),
        router_min_hold_days=int(args.router_min_hold_days),
        router_log_path=(Path(args.router_log) if args.router_log else None),
        obs_dim=int(args.obs_dim),
        log_decisions=False,
    )

    runner = LivePolicyRunner.from_config(live_cfg)

    sb_cfg = StateBuilderConfig(output_root=output_root, stepA_mode=args.mode)
    sb = StateBuilder(sb_cfg)

    obs = sb.build_morning_state(symbol, d)
    tech_row = sb.get_tech_row(symbol, d)

    ratio = runner.safe_decide_position(obs, tech_row=tech_row, trading_date=d)
    info = runner.get_last_router_info()

    print("date       =", d.isoformat())
    print("symbol     =", symbol)
    print("mode       =", args.mode)
    print("ratio      =", ratio)
    if info is None:
        print("router_info = None (mode is not router?)")
    else:
        for k in ["regime", "panic_score", "desired_agent", "selected_agent", "switched", "hold_days"]:
            print(f"{k:12} =", info.get(k))


if __name__ == "__main__":
    main()
