#!/usr/bin/env python
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from engine.state_builder import StateBuilder, StateBuilderConfig
from engine.live_policy_runner import RegimeDetector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--output-root", default="output")
    ap.add_argument("--mode", default=None, help="sim/ops/live/display (optional)")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD (trading day)")
    args = ap.parse_args()

    d = pd.to_datetime(args.date).date()

    sb = StateBuilder(StateBuilderConfig(output_root=Path(args.output_root), mode=args.mode))
    tech = sb.get_tech_row(args.symbol, d)

    det = RegimeDetector()
    regime = det.detect(tech)

    print("date       =", d)
    print("symbol     =", args.symbol)
    print("mode       =", args.mode)
    for c in [det.COL_DIV_DOWN_VOL_UP, det.COL_ENERGY_FADE]:
        print(f"{c:16s} =", tech.get(c, None))
    print("regime     =", regime)


if __name__ == "__main__":
    main()
