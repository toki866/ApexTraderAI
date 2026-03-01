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
    annualization: int = 252,
) -> Dict[str, Any]:
    """Compute split metrics aligned with diag-logs metrics_summary.csv definitions."""
    sub = df[df["Split"] == split].copy()
    test_days = int(len(sub))

    out: Dict[str, Any] = {
        "split": split,
        "test_days": test_days,
        "equity_start": float("nan"),
        "equity_end": float("nan"),
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

    eq = pd.to_numeric(sub[equity_col], errors="coerce").astype(float).to_numpy(dtype=float)
    if eq.size > 0 and np.isfinite(eq[0]) and np.isfinite(eq[-1]) and eq[0] != 0.0:
        equity_start = float(eq[0])
        equity_end = float(eq[-1])
        total_return = equity_end / equity_start - 1.0
        out["equity_start"] = equity_start
        out["equity_end"] = equity_end
        out["total_return_pct"] = float(total_return * 100.0)

        if test_days > 0 and (1.0 + total_return) > 0.0:
            out["cagr_pct"] = float(((1.0 + total_return) ** (float(annualization) / float(test_days)) - 1.0) * 100.0)

    if eq.size > 0:
        peak = np.maximum.accumulate(eq)
        dd = eq / np.where(peak == 0.0, np.nan, peak) - 1.0
        if np.isfinite(dd).any():
            out["max_dd_pct"] = float(np.nanmin(dd) * 100.0)

    r = pd.to_numeric(sub[ret_col], errors="coerce").astype(float).to_numpy(dtype=float)
    if r.size > 0:
        mu = float(np.mean(r))
        sd = float(np.std(r, ddof=1)) if r.size > 1 else float("nan")
        out["avg_daily_ret"] = mu
        out["win_rate"] = float(np.mean(r > 0.0))
        if np.isfinite(sd):
            out["vol_annual"] = float(sd * math.sqrt(float(annualization)))
            if sd > 0.0:
                out["sharpe"] = float((mu / sd) * math.sqrt(float(annualization)))

    return out
