#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_stepE_agents.py

StepE の出力（主に daily_log）を、2エージェント間で同一期間で比較するツール。

背景:
- StepE は output/stepE/ 配下のファイル名が固定のため、別期間で StepE を回すと
  metrics / equity ファイルが上書きされます。
- その状態で compare すると、"daily_log は一致しているのに metrics が違う" などが起きます。

このツールは **daily_log を主ソース** として、
- Position / Action の一致率
- Equity の最大差分
- ウィンドウ内の簡易メトリクス（total_return / max_drawdown / sharpe / win_rate）
を計算して表示します。

使い方例:
  python tools/compare_stepE_agents.py --output-root output --symbol SOXL \
    --agent-a mamba --agent-b mamba --date-from 2022-01-03 --date-to 2022-03-31 --show-diffs 50
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd


def agent_tag(agent: str) -> str:
    a = (agent or "").strip().lower()
    if a in ("mamba", "wavelet_mamba", "wav_mamba"):
        return "MAMBA"
    return a.upper()


def p_daily_log(output_root: Path, agent: str, symbol: str) -> Path:
    return output_root / "stepE" / f"daily_log_{agent.lower()}_{symbol}.csv"


def p_equity(output_root: Path, agent: str, symbol: str) -> Path:
    return output_root / "stepE" / f"stepE_equity_{agent_tag(agent)}_{symbol}.csv"


def p_metrics(output_root: Path, agent: str, symbol: str) -> Path:
    return output_root / "stepE" / f"stepE_rl_metrics_{agent_tag(agent)}_{symbol}.csv"


def parse_dt(s: Optional[str]) -> Optional[pd.Timestamp]:
    if not s:
        return None
    return pd.to_datetime(s, errors="coerce")


def ensure_dt_col(df: pd.DataFrame, col: str = "Date") -> pd.DataFrame:
    if col not in df.columns:
        return df
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def filter_by_date(df: pd.DataFrame, date_from: Optional[pd.Timestamp], date_to: Optional[pd.Timestamp], date_col: str = "Date") -> pd.DataFrame:
    if df.empty or date_col not in df.columns:
        return df
    out = ensure_dt_col(df, date_col)
    m = out[date_col].notna()
    if date_from is not None:
        m &= (out[date_col] >= date_from)
    if date_to is not None:
        m &= (out[date_col] <= date_to)
    return out.loc[m].sort_values(date_col).reset_index(drop=True)


def align_on_date(a: pd.DataFrame, b: pd.DataFrame, date_col: str = "Date") -> Tuple[pd.DataFrame, pd.DataFrame]:
    a = ensure_dt_col(a, date_col)
    b = ensure_dt_col(b, date_col)
    ca = set(a[date_col].dropna().tolist()) if date_col in a.columns else set()
    cb = set(b[date_col].dropna().tolist()) if date_col in b.columns else set()
    common = sorted(ca & cb)
    if not common:
        return a.iloc[0:0].copy(), b.iloc[0:0].copy()
    a2 = a[a[date_col].isin(common)].sort_values(date_col).reset_index(drop=True)
    b2 = b[b[date_col].isin(common)].sort_values(date_col).reset_index(drop=True)
    return a2, b2


def pick_equity_col(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return None
    for c in ("Equity", "equity", "Equity_norm", "equity_norm", "EquityRaw", "equity_raw"):
        if c in df.columns:
            return c
    return None


def pick_drawdown_col(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return None
    for c in ("Drawdown", "drawdown", "DD", "dd"):
        if c in df.columns:
            return c
    return None


def pick_pnl_col(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return None
    for c in ("PnL", "pnl", "Reward", "reward", "Ret", "ret", "Return", "return"):
        if c in df.columns:
            return c
    return None


def compute_metrics_from_daily_log(df: pd.DataFrame) -> dict:
    """Compute window metrics from daily_log rows (already filtered to the window).

    NOTE:
      - daily_log may contain a 'Drawdown' column computed against the *full history* peak.
        That is NOT suitable for "window-only" evaluation. We ignore it and recompute drawdown
        from the window's Equity series.

      - Sharpe is computed from equity percentage change (equity pct_change), which matches
        StepE metrics behavior in this project.
    """
    if df is None or len(df) == 0:
        return {}

    if "Equity" not in df.columns:
        return {}

    eq = pd.to_numeric(df["Equity"], errors="coerce").astype(float).ffill()
    if len(eq) == 0:
        return {}

    eq0 = float(eq.iloc[0]) if pd.notna(eq.iloc[0]) else 1.0
    if eq0 == 0.0:
        eq0 = 1.0

    eqn = (eq / eq0).astype(float)
    # equity returns inside the window
    rets = eqn.pct_change().fillna(0.0)

    total_return = float(eqn.iloc[-1] - 1.0)

    # window-only max drawdown
    peak = eqn.cummax()
    dd = (eqn / peak) - 1.0
    max_drawdown = float(dd.min()) if len(dd) else float("nan")

    mu = float(rets.mean()) if len(rets) else float("nan")
    sd = float(rets.std(ddof=1)) if len(rets) else float("nan")
    sharpe = float(mu / sd * np.sqrt(252.0)) if (sd and sd > 0.0) else float("nan")

    win_rate = float((rets > 0.0).mean()) if len(rets) else float("nan")
    trades = int(len(eqn))

    return {
        "days": int(len(eqn)),
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "trades": trades,
    }


def head_diffs(a: pd.DataFrame, b: pd.DataFrame, cols: List[str], n: int = 50) -> pd.DataFrame:
    if a.empty or b.empty:
        return pd.DataFrame()
    diffs = []
    for i in range(min(len(a), len(b))):
        row = {"Date": a.loc[i, "Date"]}
        changed = False
        for c in cols:
            va = a.loc[i, c] if c in a.columns else np.nan
            vb = b.loc[i, c] if c in b.columns else np.nan
            if (pd.isna(va) and pd.isna(vb)) or (va == vb):
                continue
            changed = True
            row[f"{c}_A"] = va
            row[f"{c}_B"] = vb
        if changed:
            diffs.append(row)
        if len(diffs) >= n:
            break
    return pd.DataFrame(diffs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--agent-a", required=True)
    ap.add_argument("--agent-b", required=True)
    ap.add_argument("--date-from", default=None)
    ap.add_argument("--date-to", default=None)
    ap.add_argument("--show-diffs", type=int, default=0, help="daily_log の差分行を最大 N 行表示（0で非表示）")
    ap.add_argument("--also-show-metrics-file", action="store_true", help="StepE が出力した metrics CSV も参考として表示")
    args = ap.parse_args()

    out_root = Path(args.output_root)
    sym = args.symbol
    a = args.agent_a
    b = args.agent_b
    d0 = parse_dt(args.date_from)
    d1 = parse_dt(args.date_to)

    # --- daily_log ---
    pa = p_daily_log(out_root, a, sym)
    pb = p_daily_log(out_root, b, sym)
    if not pa.exists():
        raise SystemExit(f"[error] missing daily_log A: {pa}")
    if not pb.exists():
        raise SystemExit(f"[error] missing daily_log B: {pb}")

    da = pd.read_csv(pa)
    db = pd.read_csv(pb)
    da = filter_by_date(da, d0, d1, "Date")
    db = filter_by_date(db, d0, d1, "Date")
    da, db = align_on_date(da, db, "Date")

    print(f"[daily_log] {a} rows={len(da)}  {b} rows={len(db)}  (common by Date)")
    if da.empty or db.empty:
        print("[daily_log] empty after filtering+align (date range mismatch or files not generated for this window).")
    else:
        # Position / Action comparison
        for col in ("Position", "Action"):
            if col in da.columns and col in db.columns:
                eq = (da[col].values == db[col].values)
                rate = float(eq.mean()) if len(eq) else float("nan")
                print(f"[daily_log] {col:<8} equal_rate={rate:.6f}  unequal={(~eq).sum()}")
            else:
                print(f"[daily_log] {col:<8} missing in one side")

        # Equity comparison (from daily_log)
        eq_col_a = pick_equity_col(da)
        eq_col_b = pick_equity_col(db)
        if eq_col_a and eq_col_b:
            ea = pd.to_numeric(da[eq_col_a], errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan).ffill()
            eb = pd.to_numeric(db[eq_col_b], errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan).ffill()
            if len(ea) and len(eb):
                max_abs = float((ea - eb).abs().max())
                print(f"[daily_log] Equity   max_abs_diff={max_abs:.6g}  end_A={float(ea.iloc[-1]):.6g}  end_B={float(eb.iloc[-1]):.6g}")
        else:
            print("[daily_log] Equity column not found in daily_log (Equity/equity/Equity_norm...).")

        if args.show_diffs and args.show_diffs > 0:
            diff_df = head_diffs(da, db, cols=["Position", "Action"], n=args.show_diffs)
            if diff_df.empty:
                print("[daily_log diffs] none (within the compared window)")
            else:
                print(f"[daily_log diffs] head (n<={args.show_diffs})")
                print(diff_df.to_string(index=False))

    # --- window metrics from daily_log ---
    ma = compute_metrics_from_daily_log(da)
    mb = compute_metrics_from_daily_log(db)
    print("\n[window metrics from daily_log] A")
    print(pd.DataFrame([{**ma, "agent": a, "agent_tag": agent_tag(a), "symbol": sym}]).to_string(index=False))
    print("\n[window metrics from daily_log] B")
    print(pd.DataFrame([{**mb, "agent": b, "agent_tag": agent_tag(b), "symbol": sym}]).to_string(index=False))

    # --- optional: show StepE metrics CSV (note: may be overwritten by other runs) ---
    if args.also_show_metrics_file:
        pma = p_metrics(out_root, a, sym)
        pmb = p_metrics(out_root, b, sym)
        if pma.exists():
            print("\n[metrics file] A (reference; may be overwritten)")
            print(pd.read_csv(pma).to_string(index=False))
        else:
            print(f"\n[metrics file] A missing: {pma}")
        if pmb.exists():
            print("\n[metrics file] B (reference; may be overwritten)")
            print(pd.read_csv(pmb).to_string(index=False))
        else:
            print(f"\n[metrics file] B missing: {pmb}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
