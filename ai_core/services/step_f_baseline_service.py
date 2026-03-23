from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from ai_core.services.step_f_metrics import (
    StrategySimulationResult,
    compute_fixed_best_agent,
    score_rank_map,
    simulate_weighted_strategy,
    summarize_returns,
)
from ai_core.services.step_f_types import StepFAuditConfig


class StepFBaselineService:
    """Compare StepF against fixed/no-cluster/adaptive/oracle baselines under identical inputs."""

    def __init__(self, config: StepFAuditConfig):
        self.config = config

    def run(
        self,
        *,
        merged: pd.DataFrame,
        daily: pd.DataFrame,
        edge_table: pd.DataFrame,
        agents: Sequence[str],
        output_path: str | Path,
    ) -> pd.DataFrame:
        output_path = Path(output_path)
        test_df = merged[merged["Split"].astype(str).str.lower() == "test"].copy().reset_index(drop=True)
        current_df = daily[daily["Split"].astype(str).str.lower() == "test"].copy().reset_index(drop=True)
        fixed_agent = compute_fixed_best_agent(merged, agents, split="train")
        oracle_daily_best = self._daily_best_returns(test_df, agents)
        baseline_rows: List[Dict[str, object]] = []

        current_metrics = summarize_returns(
            current_df.get("ret", []),
            ratios=current_df.get("ratio", []),
            fixed_best_gap=float(self._strategy_total_return(current_df.get("ret", [])) - self._fixed_best_total_return(test_df, fixed_agent)),
            oracle_gap=float(self._oracle_total_return(oracle_daily_best) - self._strategy_total_return(current_df.get("ret", []))),
            candidate_coverage=float(self._candidate_coverage(current_df)),
            notes="published/current StepF router",
        )
        baseline_rows.append({"model_name": "current_stepf", "split": "test", **current_metrics})

        fixed_weights = [{fixed_agent: 1.0} for _ in range(len(test_df))] if fixed_agent else [{} for _ in range(len(test_df))]
        fixed_result = simulate_weighted_strategy(
            merged=test_df,
            agents=agents,
            weight_rows=fixed_weights,
            trade_cost_bps=15.0,
            name="fixed_best_stepE_expert",
        )
        baseline_rows.append(
            self._metrics_row(
                fixed_result,
                split="test",
                oracle_daily_best=oracle_daily_best,
                fixed_total=fixed_result.metrics.get("total_return", 0.0),
                notes=f"best StepE fixed expert ({fixed_agent})",
            )
        )

        simple_result = self._run_simple_no_cluster_router(test_df, merged, edge_table, agents)
        baseline_rows.append(self._metrics_row(simple_result, split="test", oracle_daily_best=oracle_daily_best, fixed_total=fixed_result.metrics.get("total_return", 0.0), notes="no-cluster simple router"))

        adaptive_result = self._run_adaptive_no_cluster_router(test_df, merged, agents)
        baseline_rows.append(self._metrics_row(adaptive_result, split="test", oracle_daily_best=oracle_daily_best, fixed_total=fixed_result.metrics.get("total_return", 0.0), notes=f"adaptive no-cluster meta-router ({self.config.no_cluster_baseline_kind})"))

        oracle_result = self._run_oracle_router(test_df, agents)
        baseline_rows.append(self._metrics_row(oracle_result, split="test", oracle_daily_best=oracle_daily_best, fixed_total=fixed_result.metrics.get("total_return", 0.0), notes="regime-wise oracle"))

        out_df = pd.DataFrame(baseline_rows)
        out_df.to_csv(output_path, index=False)
        return out_df

    def _run_simple_no_cluster_router(self, test_df: pd.DataFrame, merged: pd.DataFrame, edge_table: pd.DataFrame, agents: Sequence[str]) -> StrategySimulationResult:
        train_df = merged[merged["Split"].astype(str).str.lower() == "train"].copy()
        global_scores = {
            agent: float(pd.to_numeric(train_df.get(f"ret_{agent}"), errors="coerce").mean()) for agent in agents
        }
        values = np.asarray([global_scores.get(agent, 0.0) for agent in agents], dtype=float)
        if np.all(~np.isfinite(values)) or np.all(np.abs(values) < 1e-12):
            weights = np.ones(len(agents), dtype=float) / max(1, len(agents))
        else:
            centered = np.nan_to_num(values - np.nanmax(values), nan=0.0)
            exps = np.exp(centered)
            weights = exps / np.sum(exps)
        weight_rows = [{agent: float(weights[i]) for i, agent in enumerate(agents)} for _ in range(len(test_df))]
        return simulate_weighted_strategy(merged=test_df, agents=agents, weight_rows=weight_rows, trade_cost_bps=15.0, name="no_cluster_simple_router")

    def _run_adaptive_no_cluster_router(self, test_df: pd.DataFrame, merged: pd.DataFrame, agents: Sequence[str]) -> StrategySimulationResult:
        train_df = merged[merged["Split"].astype(str).str.lower() == "train"].copy()
        prior = np.asarray([float(pd.to_numeric(train_df.get(f"ret_{agent}"), errors="coerce").mean()) for agent in agents], dtype=float)
        prior = np.nan_to_num(prior, nan=0.0)
        wealth = np.exp(prior)
        weight_rows: List[Dict[str, float]] = []
        alpha = float(self.config.adaptive_baseline_alpha)
        eta = float(self.config.adaptive_baseline_eta)
        ewma = prior.copy()
        for _, row in test_df.iterrows():
            if wealth.sum() <= 0:
                weights = np.ones(len(agents), dtype=float) / max(1, len(agents))
            else:
                weights = wealth / wealth.sum()
            weight_rows.append({agent: float(weights[i]) for i, agent in enumerate(agents)})
            realized = np.asarray([float(pd.to_numeric(pd.Series([row.get(f"ret_{agent}")]), errors="coerce").fillna(0.0).iloc[0]) for agent in agents], dtype=float)
            ewma = alpha * realized + (1.0 - alpha) * ewma
            wealth = wealth * np.exp(eta * ewma)
        return simulate_weighted_strategy(merged=test_df, agents=agents, weight_rows=weight_rows, trade_cost_bps=15.0, name="no_cluster_adaptive_meta_router")

    def _run_oracle_router(self, test_df: pd.DataFrame, agents: Sequence[str]) -> StrategySimulationResult:
        weight_rows: List[Dict[str, float]] = []
        for _, row in test_df.iterrows():
            best_agent = max(
                agents,
                key=lambda agent: float(pd.to_numeric(pd.Series([row.get(f"ret_{agent}")]), errors="coerce").fillna(-np.inf).iloc[0]),
            )
            weight_rows.append({best_agent: 1.0})
        return simulate_weighted_strategy(merged=test_df, agents=agents, weight_rows=weight_rows, trade_cost_bps=15.0, name="regime_oracle")

    @staticmethod
    def _daily_best_returns(test_df: pd.DataFrame, agents: Sequence[str]) -> List[float]:
        out: List[float] = []
        for _, row in test_df.iterrows():
            vals = [float(pd.to_numeric(pd.Series([row.get(f"ret_{agent}")]), errors="coerce").fillna(0.0).iloc[0]) for agent in agents]
            out.append(max(vals) if vals else 0.0)
        return out

    @staticmethod
    def _strategy_total_return(returns: Sequence[float]) -> float:
        s = pd.Series(list(returns), dtype=float).fillna(0.0)
        return float((1.0 + s).prod() - 1.0) if not s.empty else 0.0

    def _oracle_total_return(self, oracle_daily_best: Sequence[float]) -> float:
        return self._strategy_total_return(oracle_daily_best)

    def _fixed_best_total_return(self, test_df: pd.DataFrame, fixed_agent: str) -> float:
        if not fixed_agent:
            return 0.0
        returns = pd.to_numeric(test_df.get(f"ret_{fixed_agent}"), errors="coerce").fillna(0.0)
        return self._strategy_total_return(returns.tolist())

    @staticmethod
    def _candidate_coverage(current_df: pd.DataFrame) -> float:
        if "winner_expert" in current_df.columns and "allowed_agents" in current_df.columns:
            flags = current_df.apply(lambda row: str(row.get("winner_expert", "")) in str(row.get("allowed_agents", "")).split("|"), axis=1)
            return float(flags.mean()) if len(flags) else float("nan")
        return float("nan")

    def _metrics_row(
        self,
        result: StrategySimulationResult,
        *,
        split: str,
        oracle_daily_best: Sequence[float],
        fixed_total: float,
        notes: str,
    ) -> Dict[str, object]:
        total = float(result.metrics.get("total_return", 0.0))
        return {
            "model_name": result.name,
            "split": split,
            "total_return": total,
            "sharpe": float(result.metrics.get("sharpe", 0.0)),
            "max_drawdown": float(result.metrics.get("max_drawdown", 0.0)),
            "fixed_best_gap": float(total - fixed_total),
            "oracle_gap": float(self._oracle_total_return(oracle_daily_best) - total),
            "turnover_or_churn": float(result.metrics.get("turnover_or_churn", 0.0)),
            "candidate_coverage": 1.0,
            "notes": notes,
        }
