#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
StepE の daily log (Date/pos/ret) が「どのリターン定義」に一致しているかを監査します。

目的:
- ret が同日 (D) の値動きを使っていないか（=リーク疑い）を切り分ける
- ret が翌日 (D->D+1) の値動きで確定しているか（=タイミング安全）を確認する
- pos が pos(D) か pos(D-1) のどちらで掛けられているかも推定する

入力:
- output/stepE/<mode>/*.csv  (StepE daily log)
- output/stepA/<mode>/daily/stepA_daily_features_<SYMBOL>_YYYY_MM_DD.csv  (Open/Close を含む)

使い方（例）:
  python tools/check_stepE_reward_alignment.py --symbol SOXL --mode sim --trade-cost-bps 5
  python tools/check_stepE_reward_alignment.py --symbol SOXL --mode sim --log output/stepE/sim/stepE_daily_log_dprime_bnf_h01_SOXL.csv
  python tools/check_stepE_reward_alignment.py --symbol SOXL --mode sim --all

注意:
- cmd.exe のバックスラッシュ例はエスケープ警告になりやすいので、この docstring では / を使っています。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


LOG_GLOB_PATTERNS = [
    "stepE_daily_log_*_{symbol}.csv",
    "stepE_log_*_{symbol}.csv",
    "*stepE*log*{symbol}*.csv",
]


def _mode_candidates(output_root: Path, mode: str, symbol: str) -> List[Path]:
    base = output_root / "stepE" / mode
    cands: List[Path] = []
    for pat in LOG_GLOB_PATTERNS:
        for p in base.glob(pat.format(symbol=symbol)):
            if p.is_file():
                cands.append(p)
    # unique
    seen = set()
    uniq = []
    for p in cands:
        rp = str(p).replace("\\", "/")
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return sorted(uniq, key=lambda x: x.stat().st_mtime, reverse=True)


def _pick_default_log(cands: List[Path]) -> Path:
    # newest first already
    return cands[0]


def _find_col(df: pd.DataFrame, keys: List[str]) -> Optional[str]:
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}
    for k in keys:
        if k.lower() in lower:
            return lower[k.lower()]
    # fuzzy contains
    for k in keys:
        kl = k.lower()
        for c in cols:
            if kl in c.lower():
                return c
    return None


def _parse_date_series(s: pd.Series) -> pd.Series:
    # tolerate "YYYY-MM-DD" and "YYYY/MM/DD" and "YYYY_MM_DD"
    ss = s.astype(str).str.strip()
    ss = ss.str.replace("_", "-", regex=False).str.replace("/", "-", regex=False)
    return pd.to_datetime(ss, errors="coerce").dt.normalize()


def _load_log(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = _find_col(df, ["Date", "date", "dt", "Day"])
    if date_col is None:
        raise RuntimeError(f"Date列が見つかりません: {path}")
    df = df.copy()
    df["__Date"] = _parse_date_series(df[date_col])
    df = df.dropna(subset=["__Date"]).sort_values("__Date").reset_index(drop=True)

    pos_col = _find_col(df, ["pos", "position", "Pos", "action_pos", "portfolio_pos"])
    ret_col = _find_col(df, ["ret", "reward", "pnl", "return", "Ret", "r"])
    if pos_col is None:
        raise RuntimeError(f"pos列が見つかりません: {path}  columns={list(df.columns)}")
    if ret_col is None:
        raise RuntimeError(f"ret列が見つかりません: {path}  columns={list(df.columns)}")

    df["__pos"] = pd.to_numeric(df[pos_col], errors="coerce")
    df["__ret"] = pd.to_numeric(df[ret_col], errors="coerce")
    df = df.dropna(subset=["__pos", "__ret"]).reset_index(drop=True)
    return df


def _features_path(output_root: Path, mode: str, symbol: str, dt: pd.Timestamp) -> Path:
    d = dt.strftime("%Y_%m_%d")
    return output_root / "stepA" / mode / "daily" / f"stepA_daily_features_{symbol}_{d}.csv"


def _load_day_features(path: Path) -> Optional[pd.Series]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if len(df) == 0:
            return None
        return df.iloc[0]
    except Exception:
        return None


@dataclass
class CandidateResult:
    implied: str               # "ret/pos(D)" etc
    return_def: str            # "cc_same" etc
    n: int
    mae: float
    rmse: float
    corr: float


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    try:
        if a.nunique() < 2 or b.nunique() < 2:
            return float("nan")
        return float(a.corr(b))
    except Exception:
        return float("nan")


def _metrics(a: pd.Series, b: pd.Series) -> Tuple[int, float, float, float]:
    df = pd.concat([a, b], axis=1).dropna()
    n = len(df)
    if n == 0:
        return 0, float("nan"), float("nan"), float("nan")
    err = df.iloc[:, 0] - df.iloc[:, 1]
    mae = float(err.abs().mean())
    rmse = float((err * err).mean() ** 0.5)
    corr = _safe_corr(df.iloc[:, 0], df.iloc[:, 1])
    return n, mae, rmse, corr


def analyze_one(log_path: Path, output_root: Path, mode: str, symbol: str, trade_cost_bps: float) -> List[CandidateResult]:
    L = _load_log(log_path)

    # Build a unified table with Open/Close for D-1, D, D+1 where possible
    dates = list(L["__Date"])
    feat_cache: Dict[pd.Timestamp, Optional[pd.Series]] = {}
    def get_feat(dt: pd.Timestamp) -> Optional[pd.Series]:
        if dt in feat_cache:
            return feat_cache[dt]
        p = _features_path(output_root, mode, symbol, dt)
        s = _load_day_features(p)
        feat_cache[dt] = s
        return s

    opens = []
    closes = []
    close_prev = []
    open_next = []
    close_next = []
    has = []

    for dt in dates:
        f = get_feat(dt)
        f_prev = get_feat(dt - pd.Timedelta(days=1))
        f_next = get_feat(dt + pd.Timedelta(days=1))
        # We may miss prev/next due to weekends; use trading-day adjacency via available files:
        # We'll also try to infer prev/next by looking at sorted available log dates, not calendar day.
        opens.append(float(f["Open"]) if f is not None and "Open" in f else float("nan"))
        closes.append(float(f["Close"]) if f is not None and "Close" in f else float("nan"))
        close_prev.append(float(f_prev["Close"]) if f_prev is not None and "Close" in f_prev else float("nan"))
        open_next.append(float(f_next["Open"]) if f_next is not None and "Open" in f_next else float("nan"))
        close_next.append(float(f_next["Close"]) if f_next is not None and "Close" in f_next else float("nan"))
        has.append(f is not None)

    T = pd.DataFrame({
        "Date": dates,
        "Open": opens,
        "Close": closes,
        "Close_prev_cal": close_prev,
        "Open_next_cal": open_next,
        "Close_next_cal": close_next,
        "pos": L["__pos"].astype(float).values,
        "ret": L["__ret"].astype(float).values,
    })

    # Improve prev/next with *log-date adjacency* (trading days)
    T = T.sort_values("Date").reset_index(drop=True)
    T["Close_prev"] = T["Close"].shift(1)
    T["Open_next"] = T["Open"].shift(-1)
    T["Close_next"] = T["Close"].shift(-1)

    # Candidate return definitions
    eps = 1e-12
    T["oc_same"] = T["Close"] / T["Open"] - 1.0
    T["cc_same"] = T["Close"] / T["Close_prev"] - 1.0
    T["oc_next"] = T["Close_next"] / T["Open_next"] - 1.0
    T["cc_next"] = T["Close_next"] / T["Close"] - 1.0

    # Trade cost model (simple): cost = bps/10000 * |pos - prev_pos|
    bps = float(trade_cost_bps)
    T["pos_prev"] = T["pos"].shift(1)
    T["dpos"] = (T["pos"] - T["pos_prev"]).abs()
    T["cost"] = (bps / 10000.0) * T["dpos"].fillna(0.0)

    # Two implied-return hypotheses:
    #   H1: ret = pos(D) * r + cost_term
    #   H2: ret = pos(D-1) * r + cost_term
    # We test both with and without adding back cost (unknown sign conventions).
    def implied_series(div_by: str, add_cost: bool) -> pd.Series:
        denom = T[div_by].copy()
        denom = denom.where(denom.abs() > eps)
        numer = T["ret"] + (T["cost"] if add_cost else 0.0)
        return numer / denom

    implied_defs = [
        ("ret/pos(D) (no_cost_adj)", implied_series("pos", add_cost=False)),
        ("ret/pos(D) (add_cost)", implied_series("pos", add_cost=True)),
        ("ret/pos(D-1) (no_cost_adj)", implied_series("pos_prev", add_cost=False)),
        ("ret/pos(D-1) (add_cost)", implied_series("pos_prev", add_cost=True)),
    ]
    return_defs = ["oc_same", "cc_same", "oc_next", "cc_next"]

    results: List[CandidateResult] = []
    for iname, iser in implied_defs:
        for rname in return_defs:
            n, mae, rmse, corr = _metrics(iser, T[rname])
            results.append(CandidateResult(implied=iname, return_def=rname, n=n, mae=mae, rmse=rmse, corr=corr))

    # Sort:
    # - 1) higher corr is better
    # - 2) lower rmse is better
    # - 3) lower mae is better
    # - 4) larger n is better
    def _sort_key(x: CandidateResult) -> Tuple[float, float, float, float]:
        corr = x.corr if not pd.isna(x.corr) else -1.0
        rmse = x.rmse if not pd.isna(x.rmse) else 1e9
        mae = x.mae if not pd.isna(x.mae) else 1e9
        return (-corr, rmse, mae, -float(x.n))

    results.sort(key=_sort_key)
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--mode", default="sim", choices=["sim", "live", "ops", "display"])
    ap.add_argument("--output-root", default="output")
    ap.add_argument("--log", default=None, help="StepE log csv path. If omitted, newest candidate is auto-selected (or use --all).")
    ap.add_argument("--tag", default=None, help="Filter by substring in log filename (e.g., dprime_all_features_h01). Can be used with --all.")
    ap.add_argument("--all", action="store_true", help="Analyze all matching logs under output/stepE/<mode> for this symbol.")
    ap.add_argument("--trade-cost-bps", type=float, default=0.0)
    args = ap.parse_args()

    output_root = Path(args.output_root)
    mode = args.mode
    symbol = args.symbol

    if args.all and args.log:
        raise SystemExit("--all と --log は同時に指定できません。")

    if args.all:
        cands = _mode_candidates(output_root, mode, symbol)
        if args.tag:
            tl = args.tag.lower()
            cands = [p for p in cands if tl in p.name.lower()]
        if not cands:
            raise SystemExit(f"ログが見つかりません: {output_root/'stepE'/mode}")
        for p in cands:
            print("=" * 90)
            print(f"[LOG] {p.as_posix()}")
            results = analyze_one(p, output_root, mode, symbol, args.trade_cost_bps)
            top = results[0]
            print(f"Top match: implied='{top.implied}'  return='{top.return_def}'  n={top.n}  mae={top.mae:.6g}  rmse={top.rmse:.6g}  corr={top.corr:.6g}")
            print("Top 8 candidates:")
            for r in results[:8]:
                print(f"  n={r.n:4d}  mae={r.mae:.6g}  rmse={r.rmse:.6g}  corr={r.corr:.6g}  | {r.implied}  vs  {r.return_def}")
        return 0

    if args.log:
        log_path = Path(args.log)
        if not log_path.exists():
            raise SystemExit(f"--log が見つかりません: {log_path}")
    else:
        cands = _mode_candidates(output_root, mode, symbol)
        if args.tag:
            tl = args.tag.lower()
            cands = [p for p in cands if tl in p.name.lower()]
        if not cands:
            raise SystemExit(f"ログが見つかりません: {output_root/'stepE'/mode}")
        if args.tag and len(cands) > 1:
            msg = "\n".join([p.as_posix() for p in cands[:50]])
            raise SystemExit(f"複数のログ候補が見つかりました（tag={args.tag}）。--log で明示してください。\n" + msg)
        log_path = _pick_default_log(cands)
        if len(cands) > 1:
            print("[INFO] 複数候補があるため、最新のログを自動選択しました。別のログを見るには --log を指定するか --all を使ってください。")
            for p in cands[:10]:
                mark = "*" if p == log_path else " "
                print(f"  {mark} {p.as_posix()}")

    print(f"[LOG] {log_path.as_posix()}")
    results = analyze_one(log_path, output_root, mode, symbol, args.trade_cost_bps)
    top = results[0]
    print(f"Top match: implied='{top.implied}'  return='{top.return_def}'  n={top.n}  mae={top.mae:.6g}  rmse={top.rmse:.6g}  corr={top.corr:.6g}")
    print("Top 12 candidates:")
    for r in results[:12]:
        print(f"  n={r.n:4d}  mae={r.mae:.6g}  rmse={r.rmse:.6g}  corr={r.corr:.6g}  | {r.implied}  vs  {r.return_def}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
