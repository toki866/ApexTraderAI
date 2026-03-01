from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np
import pandas as pd


def compute_split_metrics(
    df: pd.DataFrame,
    split: str = "test",
    equity_col: str = "equity",
    ret_col: str = "ret",
) -> Dict[str, Any]:
    """Compute split metrics with the same definitions as metrics_summary.csv."""
    sub = df[df["Split"] == split].copy()
    test_days = int(len(sub))

    out: Dict[str, Any] = {
        "test_days": test_days,
        "total_return_pct": float("nan"),
        "cagr_pct": float("nan"),
        "max_dd_pct": float("nan"),
        "sharpe": float("nan"),
        "win_rate": float("nan"),
        "avg_daily_ret": float("nan"),
        "vol_annual": float("nan"),
    }

    if sub.empty:
        return out

    eq_series = pd.to_numeric(sub[equity_col], errors="coerce").astype(float)
    if eq_series.notna().sum() >= 1:
        equity_start = float(eq_series.iloc[0])
        equity_end = float(eq_series.iloc[-1])
        if np.isfinite(equity_start) and np.isfinite(equity_end) and equity_start != 0.0:
            total_return = equity_end / equity_start - 1.0
            out["total_return_pct"] = float(total_return * 100.0)
            if test_days > 0 and (1.0 + total_return) > 0.0:
                out["cagr_pct"] = float(((1.0 + total_return) ** (252.0 / float(test_days)) - 1.0) * 100.0)

        eq = eq_series.to_numpy(dtype=float)
        peak = np.maximum.accumulate(eq)
        dd = eq / np.where(peak == 0.0, np.nan, peak) - 1.0
        if np.isfinite(dd).any():
            out["max_dd_pct"] = float(np.nanmin(dd) * 100.0)

    r = pd.to_numeric(sub[ret_col], errors="coerce").astype(float).to_numpy(dtype=float)
    if r.size > 0:
        mu = float(np.mean(r))
        out["avg_daily_ret"] = mu
        out["win_rate"] = float(np.mean(r > 0.0))

        sd = float(np.std(r, ddof=1)) if r.size > 1 else float("nan")
        if np.isfinite(sd):
            out["vol_annual"] = float(sd * math.sqrt(252.0))
            if sd > 0.0:
                out["sharpe"] = float((mu / sd) * math.sqrt(252.0))

    return out

