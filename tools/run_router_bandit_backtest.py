# -*- coding: utf-8 -*-
"""
tools/run_router_bandit_backtest.py

Bandit/ScoreTable Router backtest runner.

ポイント:
- phase2_state_* が無くても、prices + agent_csv から自動生成して backtest を回せます。
- StepA(sim/live) は prices が train/test 分離される運用なので、本スクリプトは
  --prices-soxl-train / --prices-soxl-test を受け取り、内部で結合します。
  （従来の --prices-soxl 1本指定も互換で残しています）

必要な phase2_state の列（router_bandit/io_utils.load_phase2_state の必須列）:
  Date, regime_cluster, trend_cluster, phase, agreement_dist, agreement_label
これらは Bandit のコンテキストベクトル（onehot + 連続値）と、サイズスケーリングに使われます。

実行例（推奨: train/test 分離入力）:
python tools\\run_router_bandit_backtest.py ^
  --symbol SOXL --mode sim --output-root output ^
  --prices-soxl-train output\\stepA\\sim\\stepA_prices_train_SOXL.csv ^
  --prices-soxl-test  output\\stepA\\sim\\stepA_prices_test_SOXL.csv ^
  --prices-soxs-train output\\stepA\\sim\\stepA_prices_train_SOXS.csv ^
  --prices-soxs-test  output\\stepA\\sim\\stepA_prices_test_SOXS.csv ^
  --test-start 2022-01-03 --train-years 8 --test-months 3 ^
  --router bandit --eval-mode both ^
  --trade-cost-bps 10 --min-hold-days 3 ^
  --agent-csv a1=output\\stepE\\sim\\stepE_daily_log_dprime_bnf_h02_SOXL.csv

（phase2_state を明示する場合）
  --phase2-state output\\stepF\\sim\\phase2_state_SOXL.csv

"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd

# ----------------------------
# sys.path fix (repo root / tools どちらからでも import)
# ----------------------------
_THIS = Path(__file__).resolve()
REPO_ROOT = _THIS.parents[1]
TOOLS_DIR = _THIS.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

try:
    import router_bandit  # noqa: F401
except ModuleNotFoundError as e:
    msg = (
        "[ERROR] 'router_bandit' が見つかりません。\n\n"
        "想定する配置は以下のどちらかです:\n"
        "  (A) <repo_root>/router_bandit/...\n"
        "  (B) <repo_root>/tools/router_bandit/...\n\n"
        "確認コマンド（repo_rootで実行）:\n"
        "  dir router_bandit\n"
        "  dir tools\\router_bandit\n\n"
        "ZIPを展開した場所を見直してください。"
    )
    raise ModuleNotFoundError(msg) from e

from router_bandit.io_utils import (
    load_phase2_state,
    load_prices_close,
    load_agent_ratio_series,
    build_returns_table,
    parse_kv_list,
)
from router_bandit.backtest_runner import (
    BacktestConfig,
    backtest_router,
    save_outputs,
    compute_train_test_ranges,
)


def _concat_prices(train_path: Optional[str], test_path: Optional[str], single_path: Optional[str], label: str) -> pd.DataFrame:
    """
    StepA(sim/live) の運用では prices_train / prices_test が分かれる。
    - train/test が両方あれば結合
    - 片方しか無ければそれだけ
    - single_path があればそれを使う（後方互換）
    """
    frames: List[pd.DataFrame] = []
    if single_path:
        frames.append(load_prices_close(single_path, label=label))
    else:
        if train_path:
            frames.append(load_prices_close(train_path, label=label))
        if test_path:
            frames.append(load_prices_close(test_path, label=label))
    if not frames:
        raise SystemExit(f"ERROR: prices for {label} が指定されていません。")
    df = pd.concat(frames, axis=0, ignore_index=True)
    df = df.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df


def _bin_by_quantiles(x: np.ndarray, qs: Tuple[float, float]) -> np.ndarray:
    """
    3-bin quantile discretization: <=q0 -> 0, <=q1 -> 1, else 2
    """
    x = np.asarray(x, dtype=float)
    q0, q1 = np.nanpercentile(x, [qs[0], qs[1]])
    out = np.zeros_like(x, dtype=int)
    out[x > q0] = 1
    out[x > q1] = 2
    # NaN => middle
    out[np.isnan(x)] = 1
    return out


def _auto_build_phase2_state(
    px_soxl: pd.DataFrame,
    agent_series: Dict[str, pd.DataFrame],
    test_start: str,
    test_end: str,
    px_soxs: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    phase2_state が無い場合の自動生成。
    本来は StepF Phase2 の成果物だが、現状ジェネレータが無い/未実行のケースを救う。

    生成方針（軽量・決定論的）:
    - regime_cluster: 20日ローリングの log-return vol を 3分位で離散化
    - trend_cluster : 20日ローリングの log-return mean を 3分位で離散化
    - phase         : (trend_cluster * 3 + regime_cluster) で 0..8
    - agreement_label: agents ratio の平均（[-1,1]想定）
    - agreement_dist : abs(agreement_label)（= 合意の強さ）
      ※ SizeScaler は agreement_dist が大きいほど size が増える設計。
    """
    # date universe: prices ベース（SOXSがあれば共通日のみ）
    dates = pd.Series(px_soxl["Date"]).astype(str)
    if px_soxs is not None:
        soxs_set = set(pd.Series(px_soxs["Date"]).astype(str).tolist())
        dates = dates[dates.isin(soxs_set)]
    dates = dates.sort_values().reset_index(drop=True)

    # close series aligned
    df = pd.DataFrame({"Date": dates})
    df = df.merge(px_soxl[["Date", "Close_SOXL"]], on="Date", how="left")
    df["Close_SOXL"] = df["Close_SOXL"].astype(float)

    # log returns
    df["logret"] = np.log(df["Close_SOXL"]).diff()

    vol = df["logret"].rolling(20, min_periods=5).std().to_numpy(dtype=float)
    trd = df["logret"].rolling(20, min_periods=5).mean().to_numpy(dtype=float)

    df["regime_cluster"] = _bin_by_quantiles(vol, (33.0, 66.0))
    df["trend_cluster"] = _bin_by_quantiles(trd, (33.0, 66.0))
    df["phase"] = (df["trend_cluster"].astype(int) * 3 + df["regime_cluster"].astype(int)).astype(int)

    # agreement from agent ratios
    # build wide table
    wide = df[["Date"]].copy()
    for name, s in agent_series.items():
        tmp = s.rename(columns={"agent_ratio": f"ratio_{name}"})
        wide = wide.merge(tmp[["Date", f"ratio_{name}"]], on="Date", how="left")
    ratio_cols = [c for c in wide.columns if c.startswith("ratio_")]

    # すべてNaNの日は0扱い
    mean_ratio = wide[ratio_cols].mean(axis=1, skipna=True).fillna(0.0).astype(float).to_numpy()
    mean_ratio = np.clip(mean_ratio, -1.0, 1.0)
    df["agreement_label"] = mean_ratio
    df["agreement_dist"] = np.abs(mean_ratio)

    # backtestの対象は train+test（ただし returns 計算で最後の日は落ちる）
    # ここでは test_end まで（含む）に絞っておく（それ以降は不要）
    df = df[df["Date"] <= str(test_end)].copy().reset_index(drop=True)

    # type normalize
    df["regime_cluster"] = df["regime_cluster"].astype(int)
    df["trend_cluster"] = df["trend_cluster"].astype(int)
    df["phase"] = df["phase"].astype(int)
    df["agreement_dist"] = df["agreement_dist"].astype(float)
    df["agreement_label"] = df["agreement_label"].astype(float)
    return df[["Date", "regime_cluster", "trend_cluster", "phase", "agreement_dist", "agreement_label"]].copy()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="StepF Phase4 Bandit Router backtest (sim)")

    ap.add_argument("--symbol", required=True, type=str)
    ap.add_argument("--mode", default="sim", choices=["sim", "live", "ops", "display"])
    ap.add_argument("--output-root", default="output", type=str)

    ap.add_argument("--phase2-state", default=None, type=str, help="Optional: phase2_state_*.csv (if missing -> auto-build)")

    # Backward compatible (single file)
    ap.add_argument("--prices-soxl", default=None, type=str)
    ap.add_argument("--prices-soxs", default=None, type=str)

    # Recommended (train/test split inputs)
    ap.add_argument("--prices-soxl-train", default=None, type=str)
    ap.add_argument("--prices-soxl-test", default=None, type=str)
    ap.add_argument("--prices-soxs-train", default=None, type=str)
    ap.add_argument("--prices-soxs-test", default=None, type=str)

    ap.add_argument("--test-start", required=True, type=str)
    ap.add_argument("--train-years", required=True, type=int)
    ap.add_argument("--test-months", required=True, type=int)

    ap.add_argument("--router", default="bandit", choices=["bandit", "scoretable"])
    ap.add_argument("--eval-mode", default="both", choices=["A", "B", "both", "a", "b"])

    ap.add_argument("--trade-cost-bps", default=10.0, type=float)
    ap.add_argument("--min-hold-days", default=0, type=int)
    ap.add_argument("--delta-threshold", default=0.0, type=float)

    ap.add_argument("--alpha", default=1.0, type=float)
    ap.add_argument("--ridge-lambda", default=1.0, type=float)

    ap.add_argument("--T0-quantiles", default="10,15,20,25,30", type=str)
    ap.add_argument("--T0-objective", default="sharpe", choices=["sharpe", "mean_minus_dd"])
    ap.add_argument("--T1-quantile", default=80.0, type=float)
    ap.add_argument("--dd-weight", default=0.5, type=float)

    ap.add_argument(
        "--agent-csv",
        action="append",
        default=[],
        type=str,
        help="Repeatable: name=path_to_csv_with_daily_ratio",
    )

    ap.add_argument("--save-auto-phase2", action="store_true", help="If auto-built, save CSV under output/stepF/<mode>/phase2_state_<SYMBOL>_auto.csv")
    ap.add_argument("--router-config-out", default=None, type=str, help="Write unified auto-router config YAML (default: output/stepF/<mode>/router_auto_<SYMBOL>.yaml)")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if len(args.agent_csv) < 1:
        raise SystemExit("ERROR: --agent-csv name=path を少なくとも1つ指定してください（推奨10本）。")

    agent_paths: Dict[str, str] = parse_kv_list(args.agent_csv)
    agent_names = list(agent_paths.keys())

    # ranges
    train_start, test_end = compute_train_test_ranges(args.test_start, args.train_years, args.test_months)

    # load prices (concat train/test if provided)
    px_soxl = _concat_prices(args.prices_soxl_train, args.prices_soxl_test, args.prices_soxl, label="SOXL")
    px_soxs = None
    if args.prices_soxs_train or args.prices_soxs_test or args.prices_soxs:
        px_soxs = _concat_prices(args.prices_soxs_train, args.prices_soxs_test, args.prices_soxs, label="SOXS")

    # load agent series early (for auto phase2)
    agent_series: Dict[str, pd.DataFrame] = {}
    for name, path in agent_paths.items():
        agent_series[name] = load_agent_ratio_series(path)

    # load or auto-build phase2
    phase2: pd.DataFrame
    auto_built = False
    if args.phase2_state:
        p = Path(args.phase2_state)
        if p.exists():
            phase2 = load_phase2_state(str(p))
        else:
            print(f"[WARN] phase2_state not found -> {p}")
            print("[WARN] fallback: auto-build phase2_state from prices + agent ratios")
            phase2 = _auto_build_phase2_state(px_soxl, agent_series, args.test_start, test_end, px_soxs=px_soxs)
            auto_built = True
    else:
        print("[INFO] no --phase2-state provided -> auto-build phase2_state from prices + agent ratios")
        phase2 = _auto_build_phase2_state(px_soxl, agent_series, args.test_start, test_end, px_soxs=px_soxs)
        auto_built = True

    if auto_built and args.save_auto_phase2:
        out_p = Path(args.output_root) / "stepF" / args.mode / f"phase2_state_{args.symbol}_auto.csv"
        out_p.parent.mkdir(parents=True, exist_ok=True)
        phase2.to_csv(out_p, index=False, encoding="utf-8")
        print(f"[OK] saved auto phase2_state -> {out_p}")

    # build returns table (phase2 provides Date + state columns)
    df_all = build_returns_table(phase2, px_soxl, px_soxs)

    # sanity: ensure we have enough train range
    dmin = str(df_all["Date"].min())
    if dmin > train_start:
        print(f"[WARN] prices/phase2 does not include full train range. train_start={train_start} but data_min={dmin}")
        print("[WARN] => training will use available earliest date instead. (To fix: pass prices_train+prices_test combined.)")
        train_start = dmin

    # Build agent ratio map by date (align to df_all dates)
    agent_ratio_by_date: Dict[str, Dict[str, float]] = {}
    for name, s in agent_series.items():
        merged = df_all[["Date"]].merge(s, on="Date", how="left")
        if merged["agent_ratio"].isna().any():
            miss = merged.loc[merged["agent_ratio"].isna(), "Date"].head(5).tolist()
            raise SystemExit(
                f"ERROR: agent '{name}' のratioが欠損です。例: {miss}\nCSVのDate範囲を確認してください。"
            )
        for d, r in zip(merged["Date"].tolist(), merged["agent_ratio"].tolist()):
            agent_ratio_by_date.setdefault(d, {})[name] = float(r)

    # config
    t0_qs = [float(x.strip()) for x in args.T0_quantiles.split(",") if x.strip()]
    cfg = BacktestConfig(
        symbol=args.symbol,
        mode=args.mode,
        output_root=args.output_root,
        router=args.router,
        eval_mode=args.eval_mode,
        trade_cost_bps=float(args.trade_cost_bps),
        min_hold_days=int(args.min_hold_days),
        delta_threshold=float(args.delta_threshold),
        alpha=float(args.alpha),
        ridge_lambda=float(args.ridge_lambda),
        t0_quantiles=t0_qs,
        t0_objective=args.T0_objective,
        t1_quantile=float(args.T1_quantile),
        dd_weight=float(args.dd_weight),
    )

    logs = backtest_router(
        df_all=df_all,
        train_start=train_start,
        test_start=args.test_start,
        test_end=test_end,
        agent_ratio_by_date=agent_ratio_by_date,
        agent_names=agent_names,
        cfg=cfg,
    )
    save_outputs(logs, cfg)

    out_dir = Path(cfg.output_root) / "stepF" / cfg.mode / "router_bandit"
    print(f"[OK] wrote outputs -> {out_dir}")

    # Write unified auto-router config (to replace router_table_*.yaml)
    try:
        cfg_out: Optional[Path]
        cfg_out = Path(args.router_config_out) if args.router_config_out else (Path(args.output_root) / "stepF" / args.mode / f"router_auto_{args.symbol}.yaml")
        cfg_out.parent.mkdir(parents=True, exist_ok=True)

        # phase2_state path used in this run
        # NOTE:
        # - If user provided --phase2-state and it exists -> use it.
        # - If user provided --phase2-state but it does NOT exist (common when pasting a placeholder),
        #   we still auto-build phase2_state; in that case, point router_auto config to the auto csv
        #   *if it exists* to avoid later confusion.
        auto_phase2_path = str(Path(args.output_root) / "stepF" / args.mode / f"phase2_state_{args.symbol}_auto.csv")

        phase2_used = args.phase2_state
        if phase2_used is not None:
            if not Path(phase2_used).exists():
                if Path(auto_phase2_path).exists():
                    print(f"[WARN] --phase2-state was provided but not found -> {phase2_used}")
                    print(f"[WARN] router_auto config will use auto phase2_state -> {auto_phase2_path}")
                    phase2_used = auto_phase2_path
        else:
            if args.save_auto_phase2 and Path(auto_phase2_path).exists():
                phase2_used = auto_phase2_path

        agent_csv_map = {k: str(v) for k, v in agent_paths.items()}

        policy_train_path = str(out_dir / "router_policy_train.pkl")
        policy_testB_final_path = str(out_dir / "router_policy_testB_final.pkl")
        meta_csv_path = str(out_dir / f"router_meta_bandit_{args.symbol}.csv")
        eval_csv_path = str(out_dir / f"router_eval_summary_bandit_{args.symbol}.csv")

        auto_cfg = {
            "version": 1,
            "router_type": "bandit",
            "symbol": args.symbol,
            "mode": args.mode,
            "output_root": str(Path(args.output_root)),
            "artifacts_dir": str(out_dir),
            "policy_train_path": policy_train_path,
            "policy_train": policy_train_path,
            "policy_testB_final_path": policy_testB_final_path if Path(policy_testB_final_path).exists() else None,
            "policy_testB_final": (policy_testB_final_path if Path(policy_testB_final_path).exists() else None),
            "meta_csv_path": meta_csv_path if Path(meta_csv_path).exists() else None,
            "meta_csv": (meta_csv_path if Path(meta_csv_path).exists() else None),
            "eval_summary_csv_path": eval_csv_path if Path(eval_csv_path).exists() else None,
            "eval_summary_csv": (eval_csv_path if Path(eval_csv_path).exists() else None),
            "phase2_state_path": phase2_used if (phase2_used and Path(phase2_used).exists()) else None,
            "phase2_state": (phase2_used if (phase2_used and Path(phase2_used).exists()) else None),
            "phase2_state_csv": (phase2_used if (phase2_used and Path(phase2_used).exists()) else None),
            "agent_names": list(agent_paths.keys()),
            "agent_csv_map": agent_csv_map,
            "agent_csv": agent_csv_map,
            "agents": agent_csv_map,
            "trade_cost_bps": float(args.trade_cost_bps),
            "min_hold_days": int(args.min_hold_days),
            "delta_threshold": float(args.delta_threshold),
            "notes": "Unified auto-router config generated by run_router_bandit_backtest.py (router_table_*.yaml is deprecated).",        }

        # YAML is preferred if PyYAML exists; otherwise fall back to JSON next to it.
        try:
            import yaml  # type: ignore

            with cfg_out.open("w", encoding="utf-8") as f:
                yaml.safe_dump(auto_cfg, f, sort_keys=False, allow_unicode=True)
            print(f"[OK] wrote router_auto config -> {cfg_out}")
        except Exception:
            import json

            json_path = cfg_out.with_suffix(".json")
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(auto_cfg, f, ensure_ascii=False, indent=2)
            print(f"[OK] wrote router_auto config -> {json_path}")
    except Exception as e:
        print(f"[WARN] failed to write router_auto config: {e}")


if __name__ == "__main__":
    main()
