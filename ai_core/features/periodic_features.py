"""ai_core.features.periodic_features

Periodic (calendar) feature generator.

- Produces 44 features: sin_k01..sin_k22 and cos_k01..cos_k22.
- These are deterministic from the date only (no market data), so they are safe to use for
  future-date inference.

Anchor date:
- By default uses 2014-01-02 as t=0. This matches common SOXL datasets starting on 2014-01-02.

Periods:
- 22 periods (business-day counts) chosen as a reasonable multi-scale set.

This module is used by StepAService and may also be imported elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd


DEFAULT_ANCHOR_DATE = "2014-01-02"

# 22 periods -> 44 sin/cos features
DEFAULT_PERIODS_BDAYS = [
    5,
    10,
    20,
    30,
    45,
    60,
    90,
    120,
    180,
    252,
    360,
    504,
    720,
    1008,
    1260,
    1512,
    1764,
    2016,
    2268,
    2520,
    2772,
    3024,
]


@dataclass(frozen=True)
class PeriodicFeatureSpec:
    anchor_date: str = DEFAULT_ANCHOR_DATE
    periods_bdays: tuple[int, ...] = tuple(DEFAULT_PERIODS_BDAYS)


def build_periodic_features_df(
    dates: Iterable[pd.Timestamp | str],
    spec: Optional[PeriodicFeatureSpec] = None,
) -> pd.DataFrame:
    """Build periodic sin/cos features for the given dates.

    Args:
        dates: Iterable of dates (Timestamp or ISO strings). Order is preserved.
        spec: PeriodicFeatureSpec (anchor_date, periods_bdays).

    Returns:
        DataFrame with columns: sin_k01..sin_k22, cos_k01..cos_k22 (44 cols).
    """
    if spec is None:
        spec = PeriodicFeatureSpec()

    # normalize to Timestamp (naive) and keep order
    dt = pd.to_datetime(pd.Series(list(dates)), errors="raise")

    anchor = pd.Timestamp(spec.anchor_date)
    # Use business-day index relative to anchor. If dates contain non-business days,
    # this still yields a deterministic integer index (we are not filtering to bdays here).
    t = (dt - anchor).dt.days.astype(np.int64)

    out = {}
    periods = list(spec.periods_bdays)
    for i, p in enumerate(periods, start=1):
        k = f"k{i:02d}"
        w = 2.0 * np.pi / float(p)
        out[f"sin_{k}"] = np.sin(w * t)
        out[f"cos_{k}"] = np.cos(w * t)

    return pd.DataFrame(out)
