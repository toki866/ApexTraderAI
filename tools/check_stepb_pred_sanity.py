from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def _read_prices(output_root: Path, symbol: str) -> pd.DataFrame:
    cands = [
        output_root / "stepA" / f"stepA_prices_{symbol}.csv",
        output_root / f"stepA_prices_{symbol}.csv",
    ]
    p = next((x for x in cands if x.exists()), None)
    if p is None:
        raise FileNotFoundError(f"StepA prices not found. searched={cands}")
    df = pd.read_csv(p)
    if "Date" not in df.columns or "Close" not in df.columns:
        raise RuntimeError(f"prices CSV must have Date and Close. columns={list(df.columns)}")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df[["Date", "Close"]]


def _read_stepb_preds(output_root: Path, symbol: str) -> pd.DataFrame:
    cands = [
        output_root / "stepB" / f"stepB_pred_time_all_{symbol}.csv",
        output_root / f"stepB_pred_time_all_{symbol}.csv",
    ]
    p = next((x for x in cands if x.exists()), None)
    if p is None:
        raise FileNotFoundError(f"StepB pred_time_all not found. searched={cands}")
    df = pd.read_csv(p)
    if "Date" not in df.columns:
        raise RuntimeError(f"pred CSV must have Date. columns={list(df.columns)}")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def _mae(a: pd.Series, b: pd.Series) -> float:
    x = (a - b).abs()
    return float(np.nanmean(x.values))


def _rmse(a: pd.Series, b: pd.Series) -> float:
    x = (a - b)
    return float(np.sqrt(np.nanmean((x.values) ** 2)))


def _corr(a: pd.Series, b: pd.Series) -> float:
    aa = a.astype(float)
    bb = b.astype(float)
    m = ~(aa.isna() | bb.isna())
    if m.sum() < 5:
        return float("nan")
    return float(np.corrcoef(aa[m].values, bb[m].values)[0, 1])


def eval_alignment(df: pd.DataFrame, pred_col: str, close_col: str = "Close") -> dict:
    """Return metrics for:
      - same_day: pred[t] vs close[t]
      - next_day: pred[t] vs close[t+1]
      - prev_close_copy: pred[t] vs close[t-1]
      - baseline_prev: close[t-1] vs close[t] (naive baseline)
    """
    out = {"pred_col": pred_col}

    pred = df[pred_col]
    close = df[close_col]

    close_prev = close.shift(1)
    close_next = close.shift(-1)

    # same-day
    out["n_same"] = int((~(pred.isna() | close.isna())).sum())
    out["mae_same"] = _mae(pred, close)
    out["rmse_same"] = _rmse(pred, close)
    out["corr_same"] = _corr(pred, close)

    # next-day (is pred actually for t+1?)
    out["n_next"] = int((~(pred.isna() | close_next.isna())).sum())
    out["mae_next"] = _mae(pred, close_next)
    out["rmse_next"] = _rmse(pred, close_next)
    out["corr_next"] = _corr(pred, close_next)

    # copy-check: pred[t] vs prev close
    out["n_prev"] = int((~(pred.isna() | close_prev.isna())).sum())
    out["mae_prevclose"] = _mae(pred, close_prev)
    out["rmse_prevclose"] = _rmse(pred, close_prev)
    out["corr_prevclose"] = _corr(pred, close_prev)

    # baseline prev close vs today close
    out["baseline_n"] = int((~(close.isna() | close_prev.isna())).sum())
    out["baseline_mae"] = _mae(close_prev, close)
    out["baseline_rmse"] = _rmse(close_prev, close)
    out["baseline_corr"] = _corr(close_prev, close)

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Sanity-check StepB predictions vs Close alignment and naive baseline.")
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--out-csv", default="", help="optional: write metrics table to this path")
    args = ap.parse_args()

    out_root = Path(args.output_root)
    sym = args.symbol

    prices = _read_prices(out_root, sym)
    preds = _read_stepb_preds(out_root, sym)

    df = prices.merge(preds, on="Date", how="left")

    pred_cols = [c for c in df.columns if c.startswith("Pred_Close_")]
    if not pred_cols:
        raise RuntimeError(f"No Pred_Close_* columns found. columns={list(df.columns)}")

    rows = [eval_alignment(df, c) for c in pred_cols]
    res = pd.DataFrame(rows)

    # Pretty print: show the most important columns first
    show_cols = [
        "pred_col",
        "n_same", "mae_same", "rmse_same", "corr_same",
        "n_next", "mae_next", "rmse_next", "corr_next",
        "mae_prevclose", "rmse_prevclose", "corr_prevclose",
        "baseline_mae", "baseline_rmse", "baseline_corr",
    ]
    for c in show_cols:
        if c not in res.columns:
            show_cols.remove(c)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 200)
    print(res[show_cols].to_string(index=False))

    # Quick decision hints
    print("\n[hint] If mae_same is unrealistically tiny (much smaller than baseline_mae), suspect same-day leakage/plotting.")
    print("[hint] If mae_next < mae_same, your prediction is more like t->t+1 (alignment might need shifting in the viewer).")
    print("[hint] If mae_prevclose is tiny, the model behaves like 'copy previous close'.")

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(out_path, index=False)
        print(f"\n[wrote] {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
