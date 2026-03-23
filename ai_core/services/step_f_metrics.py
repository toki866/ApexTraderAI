from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252.0


@dataclass
class StrategySimulationResult:
    name: str
    frame: pd.DataFrame
    metrics: Dict[str, float]


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def max_drawdown(equity: Sequence[float]) -> float:
    arr = np.asarray(list(equity), dtype=float)
    if arr.size == 0:
        return 0.0
    peak = np.maximum.accumulate(arr)
    dd = arr / np.where(peak == 0.0, 1.0, peak) - 1.0
    return float(np.min(dd))


def summarize_returns(
    returns: Iterable[float],
    *,
    ratios: Iterable[float] | None = None,
    fixed_best_gap: float = float("nan"),
    oracle_gap: float = float("nan"),
    candidate_coverage: float = float("nan"),
    notes: str = "",
) -> Dict[str, float | str]:
    ret = pd.Series(list(returns), dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    equity = (1.0 + ret).cumprod()
    std = float(ret.std(ddof=0))
    sharpe = float(np.sqrt(TRADING_DAYS_PER_YEAR) * ret.mean() / std) if std > 0 else 0.0
    ratio_series = pd.Series(list(ratios), dtype=float) if ratios is not None else pd.Series(dtype=float)
    turnover = float(ratio_series.diff().abs().fillna(ratio_series.abs()).mean()) if not ratio_series.empty else 0.0
    return {
        "total_return": float(equity.iloc[-1] - 1.0) if not equity.empty else 0.0,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(equity.tolist()),
        "fixed_best_gap": _safe_float(fixed_best_gap, float("nan")),
        "oracle_gap": _safe_float(oracle_gap, float("nan")),
        "turnover_or_churn": turnover,
        "candidate_coverage": _safe_float(candidate_coverage, float("nan")),
        "notes": notes,
    }


def simulate_weighted_strategy(
    *,
    merged: pd.DataFrame,
    agents: Sequence[str],
    weight_rows: List[Dict[str, float]],
    trade_cost_bps: float,
    name: str,
) -> StrategySimulationResult:
    rows: List[Dict[str, float | str | pd.Timestamp]] = []
    prev_ratio = 0.0
    for idx, (_, row) in enumerate(merged.iterrows()):
        weights = weight_rows[idx] if idx < len(weight_rows) else {}
        ratio = 0.0
        for agent in agents:
            ratio += _safe_float(weights.get(agent, 0.0)) * _safe_float(row.get(f"ratio_{agent}", 0.0))
        ratio = float(np.clip(ratio, -1.0, 1.0))
        turnover = abs(ratio - prev_ratio)
        cost = float(trade_cost_bps) * 1e-4 * turnover
        pos_plus = max(ratio, 0.0)
        pos_minus = max(-ratio, 0.0)
        r_soxl = _safe_float(row.get("r_soxl", 0.0))
        r_soxs = _safe_float(row.get("r_soxs", 0.0))
        gross_ret = pos_plus * r_soxl + pos_minus * r_soxs
        net_ret = gross_ret - cost
        selected = max(weights, key=weights.get) if weights else ""
        rows.append(
            {
                "Date": row.get("Date"),
                "Split": row.get("Split", "test"),
                "ratio": ratio,
                "gross_ret": gross_ret,
                "ret": net_ret,
                "cost": cost,
                "turnover": turnover,
                "selected_expert": selected,
                "weights_json": pd.Series(weights, dtype=float).to_json(force_ascii=False),
            }
        )
        prev_ratio = ratio
    frame = pd.DataFrame(rows)
    frame["equity"] = (1.0 + pd.to_numeric(frame.get("ret", 0.0), errors="coerce").fillna(0.0)).cumprod() if not frame.empty else []
    metrics = summarize_returns(frame.get("ret", []), ratios=frame.get("ratio", []))
    return StrategySimulationResult(name=name, frame=frame, metrics={k: v for k, v in metrics.items() if isinstance(v, float)})


def build_daily_winner_table(merged: pd.DataFrame, agents: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, row in merged.iterrows():
        ranked: List[Tuple[str, float]] = []
        for agent in agents:
            ranked.append((agent, _safe_float(row.get(f"ret_{agent}", np.nan), float("nan"))))
        ranked = [(a, r) for a, r in ranked if np.isfinite(r)]
        ranked.sort(key=lambda item: item[1], reverse=True)
        top1_agent, top1_ret = ranked[0] if ranked else ("", 0.0)
        top2_agent, top2_ret = ranked[1] if len(ranked) > 1 else ("", 0.0)
        rows.append(
            {
                "Date": row.get("Date"),
                "Split": row.get("Split", "test"),
                "winner_expert": top1_agent,
                "top1_expert": top1_agent,
                "top2_expert": top2_agent,
                "winner_ret": top1_ret,
                "top1_ret": top1_ret,
                "top2_ret": top2_ret,
                "top1_top2_margin": float(top1_ret - top2_ret),
            }
        )
    return pd.DataFrame(rows)


def compute_fixed_best_agent(merged: pd.DataFrame, agents: Sequence[str], split: str = "train") -> str:
    subset = merged[merged.get("Split", "").astype(str).str.lower() == split.lower()].copy()
    scores: Dict[str, float] = {}
    for agent in agents:
        scores[agent] = float(pd.to_numeric(subset.get(f"ret_{agent}"), errors="coerce").mean())
    if not scores:
        return ""
    return max(scores, key=lambda key: (-np.inf if not np.isfinite(scores[key]) else scores[key]))


def score_rank_map(edge_table: pd.DataFrame, score_col: str = "IR_shrink") -> Dict[Tuple[int, str], Dict[str, float]]:
    value_col = score_col if score_col in edge_table.columns else ("IR" if "IR" in edge_table.columns else None)
    edge_col = "EV_shrink" if "EV_shrink" in edge_table.columns else ("EV" if "EV" in edge_table.columns else value_col)
    rank_map: Dict[Tuple[int, str], Dict[str, float]] = {}
    if value_col is None or edge_table.empty:
        return rank_map
    for regime_id, regime_df in edge_table.groupby("regime_id"):
        sorted_df = regime_df.sort_values([value_col, edge_col], ascending=[False, False]).reset_index(drop=True)
        for rank, r in enumerate(sorted_df.itertuples(index=False), start=1):
            rank_map[(int(regime_id), str(r.agent))] = {
                "rank": float(rank),
                "router_score": _safe_float(getattr(r, value_col, np.nan), float("nan")),
                "edge_score": _safe_float(getattr(r, edge_col, np.nan), float("nan")),
            }
    return rank_map


def parse_allowlist(allowlist: pd.DataFrame) -> Dict[int, List[str]]:
    out: Dict[int, List[str]] = {}
    if allowlist.empty:
        return out
    for row in allowlist.itertuples(index=False):
        out[int(row.regime_id)] = [a for a in str(row.allowed_agents).split("|") if a]
    return out


def candidate_coverage_rate(candidate_flags: Iterable[bool]) -> float:
    vals = pd.Series(list(candidate_flags), dtype=float)
    return float(vals.mean()) if not vals.empty else float("nan")
