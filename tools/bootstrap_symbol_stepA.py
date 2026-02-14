# -*- coding: utf-8 -*-
"""
Bootstrap a missing symbol (e.g., SOXS) for this repo:
1) Ensure raw OHLCV exists at: <data_root>/prices_<SYMBOL>.csv
   - If missing (or --force-download), download from Yahoo Finance via yfinance.
2) Run StepAService for the symbol and generate:
   output/stepA/<mode>/stepA_prices_train_<SYMBOL>.csv
   output/stepA/<mode>/stepA_prices_test_<SYMBOL>.csv
   ... plus periodic/tech/split_summary

Why this exists
---------------
Your StepAService requires a source CSV at:
  Path(data_root) / f"prices_{symbol}.csv"
and raises FileNotFoundError if missing.
This script makes that file and runs StepA in one command. (No patches, no manual steps.)

Run (Windows cmd):
------------------
pip install -U yfinance pandas numpy

python tools\\bootstrap_symbol_stepA.py ^
  --symbol SOXS --mode sim --output-root output --data-root data ^
  --test-start 2022-01-03 --train-years 8 --test-months 3 ^
  --auto-install

Then your bandit backtest can use:
  output\\stepA\\sim\\stepA_prices_train_SOXS.csv
  output\\stepA\\sim\\stepA_prices_test_SOXS.csv

Notes
-----
- This script only downloads daily OHLCV. No features are generated here; StepA generates features.
- If you already have the raw CSV, just omit --force-download.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


def _repo_root_from_this_file() -> Path:
    this_ = Path(__file__).resolve()
    # tools/bootstrap_symbol_stepA.py -> repo root is parent of tools
    return this_.parents[1]


def _import_stepa(repo_root: Path):
    # Make imports robust regardless of where the user runs from.
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from ai_core.services.step_a_service import StepAService  # type: ignore
        return StepAService
    except Exception:
        # Fallback: try to locate a step_a_service.py file
        cand = list(repo_root.glob("**/step_a_service.py"))
        if cand:
            # add its parent folder
            if str(cand[0].parent) not in sys.path:
                sys.path.insert(0, str(cand[0].parent))
            from step_a_service import StepAService  # type: ignore
            return StepAService
        raise


def _compute_download_window(test_start: str, train_years: int, test_months: int) -> Tuple[str, str]:
    ts = pd.to_datetime(test_start).normalize()
    train_start = (ts - pd.DateOffset(years=int(train_years)) - pd.Timedelta(days=30)).normalize()
    test_end = (ts + pd.DateOffset(months=int(test_months)) + pd.Timedelta(days=30)).normalize()
    return train_start.strftime("%Y-%m-%d"), test_end.strftime("%Y-%m-%d")


def _ensure_yfinance(auto_install: bool) -> None:
    try:
        import yfinance  # noqa: F401
        return
    except Exception:
        if not auto_install:
            raise SystemExit(
                "ERROR: yfinance not installed.\n"
                "Run: pip install -U yfinance pandas numpy\n"
                "or rerun with --auto-install"
            )
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "yfinance", "pandas", "numpy"])


def _download_yf(ticker: str, start: str, end: str, auto_install: bool) -> pd.DataFrame:
    _ensure_yfinance(auto_install=auto_install)
    import yfinance as yf  # type: ignore

    df = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column",
    )
    if df is None or len(df) == 0:
        raise SystemExit(f"ERROR: No data downloaded for {ticker}. Check ticker/network/date range.")

    # Flatten possible MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        # common cases: ('Open', 'SOXS') or ('Open',)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    df = df.reset_index()
    # Normalize expected columns
    cols_map = {str(c).lower(): c for c in df.columns}
    need = ["date", "open", "high", "low", "close", "volume"]
    miss = [k for k in need if k not in cols_map]
    if miss:
        raise SystemExit(f"ERROR: yfinance result missing {miss}. got={list(df.columns)}")

    out = pd.DataFrame(
        {
            "Date": pd.to_datetime(df[cols_map["date"]]).dt.strftime("%Y-%m-%d"),
            "Open": pd.to_numeric(df[cols_map["open"]], errors="coerce"),
            "High": pd.to_numeric(df[cols_map["high"]], errors="coerce"),
            "Low": pd.to_numeric(df[cols_map["low"]], errors="coerce"),
            "Close": pd.to_numeric(df[cols_map["close"]], errors="coerce"),
            "Volume": pd.to_numeric(df[cols_map["volume"]], errors="coerce"),
        }
    )
    out = out.dropna(subset=["Date", "Open", "High", "Low", "Close"]).copy()
    out["Volume"] = out["Volume"].fillna(0).astype("int64")
    out = out.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)
    return out


def _write_raw_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    if len(df) == 0:
        raise SystemExit(f"ERROR: wrote empty csv: {path}")
    print(f"[OK] wrote raw prices: {path}  rows={len(df)}  [{df['Date'].iloc[0]} .. {df['Date'].iloc[-1]}]")


def _assert_file_nonempty(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"ERROR: expected output missing: {label}: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise SystemExit(f"ERROR: cannot read output csv: {label}: {path}\n{e}")
    if len(df) == 0:
        raise SystemExit(f"ERROR: output csv is empty: {label}: {path}")
    if "Date" not in df.columns:
        raise SystemExit(f"ERROR: output csv missing Date column: {label}: {path} cols={list(df.columns)}")
    print(f"[OK] verified output: {label}: {path}  rows={len(df)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True, type=str, help="Symbol, e.g., SOXS")
    ap.add_argument("--mode", default="sim", choices=["sim", "live", "display", "ops"], help="StepA mode (ops=live)")
    ap.add_argument("--output-root", default="output", type=str)
    ap.add_argument("--data-root", default="data", type=str)

    ap.add_argument("--test-start", required=True, type=str, help="YYYY-MM-DD")
    ap.add_argument("--train-years", required=True, type=int)
    ap.add_argument("--test-months", required=True, type=int)

    ap.add_argument("--download-start", default="", type=str, help="Override download start YYYY-MM-DD")
    ap.add_argument("--download-end", default="", type=str, help="Override download end YYYY-MM-DD")
    ap.add_argument("--force-download", action="store_true", help="Redownload even if raw CSV exists")
    ap.add_argument("--auto-install", action="store_true", help="Auto install yfinance/pandas/numpy if missing")
    args = ap.parse_args()

    repo_root = _repo_root_from_this_file()
    StepAService = _import_stepa(repo_root)

    symbol = args.symbol.strip().upper()
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)

    raw_path = data_root / f"prices_{symbol}.csv"

    # 1) ensure raw
    if (not raw_path.exists()) or args.force_download:
        if args.download_start and args.download_end:
            dl_start, dl_end = args.download_start, args.download_end
        else:
            dl_start, dl_end = _compute_download_window(args.test_start, args.train_years, args.test_months)

        print(f"[INFO] downloading {symbol} from Yahoo Finance: start={dl_start} end={dl_end}")
        df_raw = _download_yf(symbol, dl_start, dl_end, auto_install=args.auto_install)
        _write_raw_csv(df_raw, raw_path)
    else:
        print(f"[OK] raw prices already exist: {raw_path}")

    # 2) run StepA
    print("[INFO] running StepAService...")
    svc = StepAService(app_config={"output_root": str(output_root), "data_root": str(data_root)})
    res = svc.run(
        symbol=symbol,
        mode=("live" if args.mode == "ops" else args.mode),
        test_start=args.test_start,
        train_years=int(args.train_years),
        test_months=int(args.test_months),
    )

    # 3) verify key outputs
    out_dir = output_root / "stepA" / ("live" if args.mode == "ops" else args.mode)
    prices_train = out_dir / f"stepA_prices_train_{symbol}.csv"
    prices_test = out_dir / f"stepA_prices_test_{symbol}.csv"
    _assert_file_nonempty(prices_train, "stepA_prices_train")
    _assert_file_nonempty(prices_test, "stepA_prices_test")

    print("\n[OK] StepA completed.")
    print(f"  raw:   {raw_path}")
    print(f"  train: {prices_train}")
    print(f"  test:  {prices_test}")
    print("\n[INFO] StepA returned keys:")
    for k in sorted(res.keys()):
        print(f"  - {k}: {res[k]}")

if __name__ == "__main__":
    main()
