from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from ai_core.services.step_f_metrics import (
    StrategySimulationResult,
    parse_allowlist,
    score_rank_map,
    simulate_weighted_strategy,
)
from ai_core.services.step_f_types import StepFAuditConfig


class StepFSoftRoutingService:
    """Small A/B experiments for softer router decisions on boundary days."""

    def __init__(self, config: StepFAuditConfig):
        self.config = config

    def run(
        self,
        *,
        merged: pd.DataFrame,
        daily: pd.DataFrame,
        edge_table: pd.DataFrame,
        allowlist: pd.DataFrame,
        agents: Sequence[str],
        output_path: str | Path,
    ) -> pd.DataFrame:
        output_path = Path(output_path)
        test_df = merged[merged["Split"].astype(str).str.lower() == "test"].copy().reset_index(drop=True)
        current_df = daily[daily["Split"].astype(str).str.lower() == "test"].copy().reset_index(drop=True)
        allow_map = parse_allowlist(allowlist)
        rank_map = score_rank_map(edge_table)
        boundary_mask = self._boundary_mask(test_df)

        rows: List[Dict[str, object]] = []
        rows.append(self._row_from_current(current_df, boundary_mask))

        hard_result = self._simulate_variant(test_df, agents, allow_map, rank_map, mode="hard_top1")
        rows.append(self._to_metrics_row(hard_result, current_df, boundary_mask, notes="score top1 hard routing"))

        mixture_result = self._simulate_variant(test_df, agents, allow_map, rank_map, mode="margin_top2_mixture")
        rows.append(self._to_metrics_row(mixture_result, current_df, boundary_mask, notes="top1-top2 margin threshold mixture"))

        normalized_result = self._simulate_variant(test_df, agents, allow_map, rank_map, mode="normalized_score_blend")
        rows.append(self._to_metrics_row(normalized_result, current_df, boundary_mask, notes="normalized score proportional blending"))

        out_df = pd.DataFrame(rows)
        out_df.to_csv(output_path, index=False)
        return out_df

    def _simulate_variant(
        self,
        test_df: pd.DataFrame,
        agents: Sequence[str],
        allow_map: Dict[int, List[str]],
        rank_map: Dict[tuple[int, str], Dict[str, float]],
        *,
        mode: str,
    ) -> StrategySimulationResult:
        weight_rows: List[Dict[str, float]] = []
        for _, row in test_df.iterrows():
            regime_id = int(pd.to_numeric(pd.Series([row.get("regime_id")]), errors="coerce").fillna(0).iloc[0])
            allowed = allow_map.get(regime_id, allow_map.get(-1, list(agents))) or list(agents)
            score_pairs = []
            for agent in allowed:
                score_pairs.append((agent, float(rank_map.get((regime_id, agent), {}).get("router_score", 0.0))))
            score_pairs.sort(key=lambda item: item[1], reverse=True)
            top_pairs = score_pairs[: max(2, int(self.config.soft_routing_max_candidates))]
            boundary = self._is_boundary_day(row)
            if mode == "hard_top1":
                weights = {top_pairs[0][0]: 1.0} if top_pairs else {}
            elif mode == "margin_top2_mixture":
                weights = self._margin_mixture(top_pairs, boundary)
            else:
                weights = self._normalized_blend(top_pairs, boundary)
            weight_rows.append(weights)
        return simulate_weighted_strategy(
            merged=test_df,
            agents=agents,
            weight_rows=weight_rows,
            trade_cost_bps=15.0,
            name=mode,
        )

    def _row_from_current(self, current_df: pd.DataFrame, boundary_mask: pd.Series) -> Dict[str, object]:
        total = float((1.0 + pd.to_numeric(current_df.get("ret"), errors="coerce").fillna(0.0)).prod() - 1.0) if not current_df.empty else 0.0
        boundary_improvement = 0.0
        return {
            "model_name": "current_stepf",
            "split": "test",
            "total_return": total,
            "sharpe": float(self._sharpe(current_df.get("ret", []))),
            "max_drawdown": float(self._max_drawdown(current_df.get("ret", []))),
            "fixed_best_gap": float("nan"),
            "oracle_gap": float("nan"),
            "boundary_day_improvement": boundary_improvement,
            "churn_or_switching": float(pd.to_numeric(current_df.get("ratio"), errors="coerce").diff().abs().fillna(0.0).mean()) if not current_df.empty else 0.0,
            "notes": "published/current StepF router",
        }

    def _to_metrics_row(self, result: StrategySimulationResult, current_df: pd.DataFrame, boundary_mask: pd.Series, *, notes: str) -> Dict[str, object]:
        current_ret = pd.to_numeric(current_df.get("ret"), errors="coerce").fillna(0.0).reset_index(drop=True)
        trial_ret = pd.to_numeric(result.frame.get("ret"), errors="coerce").fillna(0.0).reset_index(drop=True)
        improvement = float((trial_ret[boundary_mask].mean() - current_ret[boundary_mask].mean())) if len(trial_ret) == len(boundary_mask) and boundary_mask.any() else 0.0
        return {
            "model_name": result.name,
            "split": "test",
            "total_return": float(result.metrics.get("total_return", 0.0)),
            "sharpe": float(result.metrics.get("sharpe", 0.0)),
            "max_drawdown": float(result.metrics.get("max_drawdown", 0.0)),
            "fixed_best_gap": float("nan"),
            "oracle_gap": float("nan"),
            "boundary_day_improvement": improvement,
            "churn_or_switching": float(result.metrics.get("turnover_or_churn", 0.0)),
            "notes": notes,
        }

    def _boundary_mask(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series([self._is_boundary_day(row) for _, row in df.iterrows()], index=df.index)

    def _is_boundary_day(self, row: pd.Series) -> bool:
        confidence = float(pd.to_numeric(pd.Series([row.get("confidence_stable")]), errors="coerce").fillna(1.0).iloc[0])
        margin = float(pd.to_numeric(pd.Series([row.get("top1_top2_margin")]), errors="coerce").fillna(0.0).iloc[0])
        uncertainty = 1.0 - confidence
        return (
            abs(margin) <= float(self.config.soft_routing_margin_threshold)
            or confidence <= float(self.config.soft_routing_confidence_threshold)
            or uncertainty >= float(self.config.boundary_uncertainty_threshold)
        )

    def _margin_mixture(self, score_pairs: List[tuple[str, float]], boundary: bool) -> Dict[str, float]:
        if not score_pairs:
            return {}
        if len(score_pairs) == 1:
            return {score_pairs[0][0]: 1.0}
        margin = float(score_pairs[0][1] - score_pairs[1][1])
        if not boundary and margin > float(self.config.soft_routing_margin_threshold):
            return {score_pairs[0][0]: 1.0}
        threshold = max(float(self.config.soft_routing_margin_threshold), 1e-9)
        top1_weight = float(np.clip(0.5 + margin / (2.0 * threshold), float(self.config.soft_routing_top2_min_weight), 1.0 - float(self.config.soft_routing_top2_min_weight)))
        top2_weight = 1.0 - top1_weight
        return {score_pairs[0][0]: top1_weight, score_pairs[1][0]: top2_weight}

    def _normalized_blend(self, score_pairs: List[tuple[str, float]], boundary: bool) -> Dict[str, float]:
        if not score_pairs:
            return {}
        if not boundary:
            return {score_pairs[0][0]: 1.0}
        scores = np.asarray([pair[1] for pair in score_pairs], dtype=float)
        scores = scores - np.nanmax(scores)
        if self.config.soft_routing_normalization == "linear":
            shifted = scores - np.nanmin(scores)
            denom = shifted.sum()
            weights = shifted / denom if denom > 0 else np.ones(len(score_pairs)) / len(score_pairs)
        else:
            exps = np.exp(scores)
            weights = exps / exps.sum() if exps.sum() > 0 else np.ones(len(score_pairs)) / len(score_pairs)
        return {score_pairs[i][0]: float(weights[i]) for i in range(len(score_pairs))}

    @staticmethod
    def _sharpe(values: Sequence[float]) -> float:
        s = pd.Series(list(values), dtype=float).fillna(0.0)
        std = float(s.std(ddof=0))
        return float(np.sqrt(252.0) * s.mean() / std) if std > 0 else 0.0

    @staticmethod
    def _max_drawdown(values: Sequence[float]) -> float:
        s = pd.Series(list(values), dtype=float).fillna(0.0)
        eq = (1.0 + s).cumprod()
        peak = eq.cummax()
        dd = eq / peak - 1.0
        return float(dd.min()) if not dd.empty else 0.0
