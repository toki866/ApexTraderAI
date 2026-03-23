from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from ai_core.services.step_f_metrics import (
    StrategySimulationResult,
    build_daily_winner_table,
    parse_allowlist,
    score_rank_map,
    simulate_weighted_strategy,
)
from ai_core.services.step_f_types import StepFAuditConfig


class StepFFallbackService:
    """Small A/B experiments for calibrated candidate fallback policies."""

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
        winners = build_daily_winner_table(test_df, agents)
        test_df = test_df.merge(winners[["Date", "winner_expert", "top1_top2_margin"]], on="Date", how="left")
        allow_map = parse_allowlist(allowlist)
        rank_map = score_rank_map(edge_table)

        rows: List[Dict[str, object]] = []
        rows.append(self._current_row(current_df))
        hard_result, hard_meta = self._simulate_policy(test_df, agents, allow_map, rank_map, policy="hard_cut")
        rows.append(self._metrics_row(hard_result, hard_meta, current_df, notes="hard allowlist exclusion baseline"))
        penalty_result, penalty_meta = self._simulate_policy(test_df, agents, allow_map, rank_map, policy="penalty")
        rows.append(self._metrics_row(penalty_result, penalty_meta, current_df, notes="hard cut -> score penalty"))
        expand_result, expand_meta = self._simulate_policy(test_df, agents, allow_map, rank_map, policy="rare_expand")
        rows.append(self._metrics_row(expand_result, expand_meta, current_df, notes="rare/low-confidence candidate expansion"))

        out_df = pd.DataFrame(rows)
        out_df.to_csv(output_path, index=False)
        return out_df

    def _simulate_policy(
        self,
        test_df: pd.DataFrame,
        agents: Sequence[str],
        allow_map: Dict[int, List[str]],
        rank_map: Dict[tuple[int, str], Dict[str, float]],
        *,
        policy: str,
    ) -> tuple[StrategySimulationResult, Dict[str, float]]:
        weight_rows: List[Dict[str, float]] = []
        coverage: List[bool] = []
        rescued: List[bool] = []
        rare_rows: List[bool] = []
        for _, row in test_df.iterrows():
            regime_id = int(pd.to_numeric(pd.Series([row.get("regime_id")]), errors="coerce").fillna(0).iloc[0])
            allowed = allow_map.get(regime_id, allow_map.get(-1, list(agents))) or list(agents)
            score_pairs = [(agent, float(rank_map.get((regime_id, agent), {}).get("router_score", 0.0))) for agent in agents]
            score_pairs.sort(key=lambda item: item[1], reverse=True)
            confidence = float(pd.to_numeric(pd.Series([row.get("confidence_stable")]), errors="coerce").fillna(1.0).iloc[0])
            margin = float(pd.to_numeric(pd.Series([row.get("top1_top2_margin")]), errors="coerce").fillna(0.0).iloc[0])
            rare = int(pd.to_numeric(pd.Series([row.get("rare_flag_raw20")]), errors="coerce").fillna(0).iloc[0]) == 1
            winner = str(row.get("winner_expert", ""))
            rare_rows.append(rare)
            if policy == "hard_cut":
                candidates = [pair for pair in score_pairs if pair[0] in allowed] or score_pairs[:1]
            elif policy == "penalty":
                penalized = []
                for agent, score in score_pairs:
                    adj = score if agent in allowed else score - float(self.config.fallback_penalty)
                    penalized.append((agent, adj))
                candidates = sorted(penalized, key=lambda item: item[1], reverse=True)
            else:
                candidates = [pair for pair in score_pairs if pair[0] in allowed]
                if rare or confidence <= float(self.config.fallback_confidence_expansion_threshold) or abs(margin) <= float(self.config.fallback_margin_expansion_threshold):
                    expanded = [pair for pair in score_pairs if pair not in candidates][: int(self.config.fallback_rare_only_extra_candidates)]
                    candidates = candidates + expanded
                if not candidates:
                    candidates = score_pairs[:1]
            chosen = candidates[0][0] if candidates else ""
            weight_rows.append({chosen: 1.0} if chosen else {})
            in_candidate = winner in [agent for agent, _ in candidates] if winner else False
            coverage.append(in_candidate)
            rescued.append(winner not in allowed and in_candidate)
        result = simulate_weighted_strategy(merged=test_df, agents=agents, weight_rows=weight_rows, trade_cost_bps=15.0, name=policy)
        policy_ret = pd.to_numeric(result.frame.get("ret"), errors="coerce").fillna(0.0)
        meta = {
            "candidate_coverage": float(np.mean(coverage)) if coverage else 0.0,
            "fallback_induced_miss_rate": float(1.0 - np.mean(coverage)) if coverage else 0.0,
            "rescued_out_of_allowlist_winner_rate": float(np.mean(rescued)) if rescued else 0.0,
            "rare_only_improvement": float(policy_ret[pd.Series(rare_rows)].mean()) if rare_rows else 0.0,
            "churn_or_noise": float(pd.to_numeric(result.frame.get("ratio"), errors="coerce").diff().abs().fillna(0.0).mean()) if not result.frame.empty else 0.0,
        }
        return result, meta

    @staticmethod
    def _current_row(current_df: pd.DataFrame) -> Dict[str, object]:
        ret = pd.to_numeric(current_df.get("ret"), errors="coerce").fillna(0.0)
        eq = (1.0 + ret).cumprod()
        peak = eq.cummax()
        dd = eq / peak - 1.0
        return {
            "model_name": "current_stepf",
            "split": "test",
            "total_return": float(eq.iloc[-1] - 1.0) if not eq.empty else 0.0,
            "sharpe": float(np.sqrt(252.0) * ret.mean() / ret.std(ddof=0)) if len(ret) and float(ret.std(ddof=0)) > 0 else 0.0,
            "max_drawdown": float(dd.min()) if not dd.empty else 0.0,
            "candidate_coverage": float("nan"),
            "fallback_induced_miss_rate": float("nan"),
            "rescued_out_of_allowlist_winner_rate": float("nan"),
            "rare_only_improvement": float("nan"),
            "churn_or_noise_increase": float(pd.to_numeric(current_df.get("ratio"), errors="coerce").diff().abs().fillna(0.0).mean()) if not current_df.empty else 0.0,
            "notes": "published/current StepF router",
        }

    @staticmethod
    def _metrics_row(result: StrategySimulationResult, meta: Dict[str, float], current_df: pd.DataFrame, *, notes: str) -> Dict[str, object]:
        return {
            "model_name": result.name,
            "split": "test",
            "total_return": float(result.metrics.get("total_return", 0.0)),
            "sharpe": float(result.metrics.get("sharpe", 0.0)),
            "max_drawdown": float(result.metrics.get("max_drawdown", 0.0)),
            "candidate_coverage": float(meta.get("candidate_coverage", 0.0)),
            "fallback_induced_miss_rate": float(meta.get("fallback_induced_miss_rate", 0.0)),
            "rescued_out_of_allowlist_winner_rate": float(meta.get("rescued_out_of_allowlist_winner_rate", 0.0)),
            "rare_only_improvement": float(meta.get("rare_only_improvement", 0.0)),
            "churn_or_noise_increase": float(meta.get("churn_or_noise", 0.0)),
            "notes": notes,
        }
