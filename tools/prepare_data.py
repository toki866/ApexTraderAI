#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd


class DataPreparationError(RuntimeError):
    """Raised when market data could not be downloaded/normalized."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_symbols(symbols_csv: str) -> List[str]:
    symbols = [s.strip().upper() for s in symbols_csv.split(",") if s.strip()]
    deduped: List[str] = []
    for s in symbols:
        if s not in deduped:
            deduped.append(s)
    if not deduped:
        raise DataPreparationError("No valid symbols were provided. Use --symbols like SOXL,SOXS.")
    return deduped


def _normalize_download_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df is None or df.empty:
        raise DataPreparationError(f"Yahoo Finance returned empty data for {symbol}.")

    work = df.copy()
    if isinstance(work.columns, pd.MultiIndex):
        required_cols = {"Open", "High", "Low", "Close", "Volume"}

        # yfinance can return (Ticker, Field) or (Field, Ticker) for a single ticker.
        # Prefer the level that contains OHLCV field names and flatten to that level.
        level_with_fields: Optional[int] = None
        for level in range(work.columns.nlevels):
            values = set(work.columns.get_level_values(level))
            if required_cols.issubset(values):
                level_with_fields = level
                break

        if level_with_fields is not None:
            work.columns = work.columns.get_level_values(level_with_fields)
        elif symbol in work.columns.get_level_values(0):
            work = work[symbol]
        elif symbol in work.columns.get_level_values(-1):
            work = work.xs(symbol, axis=1, level=-1)
        else:
            work.columns = work.columns.get_level_values(-1)

    missing = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c not in work.columns]
    if missing:
        raise DataPreparationError(
            f"Downloaded data for {symbol} is missing required columns: {missing}. "
            f"Available columns={list(work.columns)}"
        )

    out = work[["Open", "High", "Low", "Close", "Volume"]].copy()
    out = out.reset_index()

    date_col = "Date" if "Date" in out.columns else out.columns[0]
    out = out.rename(columns={date_col: "Date"})
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out = out.dropna(subset=["Date"]).drop_duplicates(subset=["Date"]).sort_values("Date")

    return out[["Date", "Open", "High", "Low", "Close", "Volume"]]


def fetch_symbol_prices(symbol: str, start: str, end: Optional[str] = None) -> pd.DataFrame:
    try:
        import yfinance as yf
    except Exception as exc:
        raise DataPreparationError(
            "Failed to import yfinance. Run `pip install -r requirements.txt` first."
        ) from exc

    end_eff = end
    if end:
        end_date = datetime.strptime(end, "%Y-%m-%d").date()
        end_eff = (end_date + timedelta(days=1)).isoformat()

    try:
        raw = yf.download(
            symbol,
            start=start,
            end=end_eff,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
            actions=False,
        )
    except Exception as exc:
        raise DataPreparationError(f"Failed to download {symbol} from Yahoo Finance: {type(exc).__name__}: {exc}") from exc

    return _normalize_download_df(raw, symbol)


def ensure_price_csvs(
    symbols: Sequence[str],
    start: str,
    end: Optional[str],
    data_dir: Path,
    force: bool = False,
) -> List[Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for symbol in symbols:
        target = data_dir / f"prices_{symbol}.csv"
        if target.exists() and not force:
            print(f"[prepare_data] skip existing: {target}")
            continue

        frame = fetch_symbol_prices(symbol, start=start, end=end)
        if frame.empty:
            raise DataPreparationError(f"Normalized dataframe is empty for {symbol}.")

        frame.to_csv(target, index=False)
        written.append(target)
        print(f"[prepare_data] wrote {target} ({len(frame)} rows)")

    return written


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Download OHLCV CSV files into data/prices_<SYMBOL>.csv")
    ap.add_argument("--symbols", required=True, help="Comma-separated symbols. Example: SOXL,SOXS")
    ap.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="End date YYYY-MM-DD (inclusive)")
    ap.add_argument("--data-dir", default=None, help="Output directory (default: <repo>/data)")
    ap.add_argument("--force", action="store_true", help="Overwrite existing files")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    symbols = _parse_symbols(args.symbols)
    data_dir = Path(args.data_dir) if args.data_dir else (_repo_root() / "data")

    try:
        ensure_price_csvs(
            symbols=symbols,
            start=args.start,
            end=args.end,
            data_dir=data_dir,
            force=bool(args.force),
        )
    except DataPreparationError as exc:
        raise SystemExit(f"[prepare_data] ERROR: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
