#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
StepE の「実際に学習・推論へ渡している観測列 (obs columns)」をダンプするツール。

目的
- StepE の入力DF (StepA daily + StepD' embedding などを merge した df_all) から、
  StepEService._select_obs_columns(profile, df_all) が最終的に選ぶ列の一覧を確定させる。
- label / target / y_* / available など “リーク疑い列” が **存在していても選ばれていない** ことを、
  CSV/TXT で証拠として残す。

使い方（repo ルートからでも、どこからでもOK）
  python tools/stepE_dump_selected_obs_cols.py --symbol SOXL --mode sim --agent dprime_all_features_h01 --profile D

出力
- output/stepE/<mode>/stepE_selected_obs_cols_<agent>_<symbol>_profile<profile>.txt
- output/stepE/<mode>/stepE_selected_obs_cols_<agent>_<symbol>_profile<profile>.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import pandas as pd


BAD_KEYS = ("label", "available", "target", "y_")


def _repo_root() -> Path:
    # tools/ 直下のこのファイルから repo ルート (= 親の親) を推定
    return Path(__file__).resolve().parents[1]


def _ensure_import_path() -> None:
    rr = _repo_root()
    if str(rr) not in sys.path:
        sys.path.insert(0, str(rr))


def _infer_dprime_sources(agent: str) -> str:
    """
    agent 名から dprime_sources を推定する。
    例:
      dprime_all_features_h01  -> all_features
      dprime_bnf_h02           -> bnf
      dprime_mix_3scale        -> mix
    """
    a = str(agent).strip()
    m = re.match(r"^dprime_(.+?)(?:_h\d+|_3scale)?$", a)
    if m:
        return m.group(1)
    # fallback
    return "all_features"


def _infer_dprime_horizons(agent: str) -> str:
    """
    agent 名から dprime_horizons を推定する。
    例:
      dprime_*_h01 -> "1"
      dprime_*_h02 -> "2"
      dprime_*     -> "1"
    """
    a = str(agent).strip()
    m = re.search(r"_h(\d+)", a)
    if m:
        return str(int(m.group(1)))
    return "1"


def _find_suspicious(selected: List[str]) -> List[str]:
    out = []
    for c in selected:
        cl = str(c).lower()
        if any(k in cl for k in BAD_KEYS):
            out.append(c)
    return out


def _dump(out_dir: Path, agent: str, symbol: str, profile: str, selected: List[str]) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    txt = out_dir / f"stepE_selected_obs_cols_{agent}_{symbol}_profile{profile}.txt"
    csv = out_dir / f"stepE_selected_obs_cols_{agent}_{symbol}_profile{profile}.csv"

    suspicious = _find_suspicious(selected)

    lines = []
    lines.append(f"agent={agent}")
    lines.append(f"symbol={symbol}")
    lines.append(f"profile={profile}")
    lines.append(f"n_selected={len(selected)}")
    lines.append(f"n_suspicious={len(suspicious)}  keys={BAD_KEYS}")
    if suspicious:
        lines.append("suspicious_cols=" + ", ".join(suspicious))
    lines.append("")
    lines.append("selected_cols:")
    lines.extend(selected)

    txt.write_text("\n".join(lines), encoding="utf-8")

    df = pd.DataFrame({"selected_col": selected})
    df["is_suspicious"] = df["selected_col"].astype(str).str.lower().apply(lambda s: any(k in s for k in BAD_KEYS))
    df.to_csv(csv, index=False, encoding="utf-8")

    return txt, csv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--mode", default="sim", choices=["sim", "live", "ops", "display"])
    ap.add_argument("--output-root", default="output")
    ap.add_argument("--agent", required=True)
    ap.add_argument("--profile", default="D", choices=["A", "B", "C", "D"])
    ap.add_argument("--trade-cost-bps", type=float, default=10.0)
    ap.add_argument("--dprime-sources", default=None, help="Override dprime_sources (comma separated string).")
    ap.add_argument("--dprime-horizons", default=None, help="Override dprime_horizons (comma separated). Example: 1,5,10,20")
    args = ap.parse_args()

    _ensure_import_path()

    from ai_core.services.step_e_service import StepEService, StepEConfig  # noqa

    output_root = Path(args.output_root)
    mode = args.mode
    symbol = args.symbol
    agent = args.agent
    profile = args.profile

    dprime_sources = args.dprime_sources or _infer_dprime_sources(agent)
    dprime_horizons = args.dprime_horizons or _infer_dprime_horizons(agent)

    # StepEService が参照する最小限の app_config を用意
    cfg = StepEConfig(
        agent=agent,
        obs_profile=profile,
        trade_cost_bps=float(args.trade_cost_bps),
        # DPrime embedding を使う前提（agent 名が dprime_* の場合は必須）
        use_stepd_prime=True,
        dprime_sources=str(dprime_sources),
        dprime_horizons=str(dprime_horizons),
        dprime_join="concat",
    )
    app_config = SimpleNamespace(output_root=str(output_root), stepE=[cfg])

    svc = StepEService(app_config)

    df_all, used_manifest = svc._merge_inputs(cfg, out_root=output_root, mode=mode, symbol=symbol)
    selected = svc._select_obs_columns(profile=profile, df_all=df_all)

    # 画面出力（短め）
    suspicious = _find_suspicious(selected)
    print(f"[OK] merged df_all: rows={len(df_all)} cols={len(df_all.columns)}  used_manifest={used_manifest}")
    print(f"[OK] selected obs cols: n={len(selected)}  suspicious={len(suspicious)}")

    # Default: keep audits out of sim/live folders to avoid confusing them as pipeline artifacts
    out_dir = output_root / "stepE" / "display" / "audit" / mode
    txt, csv = _dump(out_dir, agent, symbol, profile, selected)
    print(f"[WROTE] {txt.as_posix()}")
    print(f"[WROTE] {csv.as_posix()}")

    if suspicious:
        print("[WARN] suspicious columns were selected. Please review:")
        for c in suspicious:
            print("  -", c)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
