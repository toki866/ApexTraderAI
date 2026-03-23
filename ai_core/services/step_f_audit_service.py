from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from ai_core.services.step_f_metrics import (
    build_daily_winner_table,
    candidate_coverage_rate,
    compute_fixed_best_agent,
    parse_allowlist,
    score_rank_map,
)
from ai_core.services.step_f_types import StepFAuditConfig


class StepFAuditService:
    """Generate StepF router-side audits from consumed cluster/regime assignments."""

    def __init__(self, config: StepFAuditConfig):
        self.config = config

    def run(
        self,
        *,
        merged: pd.DataFrame,
        daily: pd.DataFrame,
        cluster_context: pd.DataFrame,
        edge_table: pd.DataFrame,
        allowlist: pd.DataFrame,
        agents: Sequence[str],
        output_dir: str | Path,
        symbol: str,
        mode: str,
    ) -> Dict[str, str]:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        winners = build_daily_winner_table(merged, agents)
        daily_joined = daily.merge(winners, on=["Date", "Split"], how="left")
        quality = self._write_cluster_quality(
            cluster_context=cluster_context,
            merged=merged,
            winners=winners,
            edge_table=edge_table,
            output_dir=out_dir,
            agents=agents,
        )
        usage = self._write_cluster_usage(
            merged=merged,
            daily=daily_joined,
            edge_table=edge_table,
            allowlist=allowlist,
            output_dir=out_dir,
            agents=agents,
        )
        focus = self._write_focus_regime_audit(
            merged=merged,
            daily=daily_joined,
            edge_table=edge_table,
            allowlist=allowlist,
            output_dir=out_dir,
        )
        summary = {
            "symbol": symbol,
            "mode": mode,
            "cluster_quality_summary": quality,
            "cluster_usage_summary": usage,
            "focus_regime_summary": focus,
        }
        manifest_path = out_dir / "stepf_audit_manifest.json"
        manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "cluster_quality_summary": str(out_dir / "cluster_quality_summary.json"),
            "cluster_quality_by_regime": str(out_dir / "cluster_quality_by_regime.csv"),
            "cluster_bootstrap_stability": str(out_dir / "cluster_bootstrap_stability.csv"),
            "cluster_boundary_days": str(out_dir / "cluster_boundary_days.csv"),
            "cluster_usage_audit_daily": str(out_dir / "cluster_usage_audit_daily.csv"),
            "cluster_usage_audit_summary": str(out_dir / "cluster_usage_audit_summary.json"),
            "regime_deep_audit_daily": str(out_dir / "regime8_deep_audit_daily.csv"),
            "regime_deep_audit_summary": str(out_dir / "regime8_deep_audit_summary.md"),
            "failure_attribution_summary": str(out_dir / "stepf_failure_attribution_summary.md"),
            "audit_manifest": str(manifest_path),
        }

    def _boundary_flags(self, row: pd.Series) -> List[str]:
        flags: List[str] = []
        margin = float(pd.to_numeric(pd.Series([row.get("top1_top2_margin")]), errors="coerce").iloc[0] or 0.0)
        confidence = float(pd.to_numeric(pd.Series([row.get("confidence_stable")]), errors="coerce").iloc[0] or 0.0)
        uncertainty = float(pd.to_numeric(pd.Series([row.get("assignment_uncertainty")]), errors="coerce").iloc[0] or 0.0)
        if abs(margin) <= float(self.config.boundary_margin_threshold):
            flags.append("small_margin")
        if confidence <= float(self.config.boundary_confidence_threshold):
            flags.append("low_confidence")
        if uncertainty >= float(self.config.boundary_uncertainty_threshold):
            flags.append("high_uncertainty")
        stable = int(pd.to_numeric(pd.Series([row.get("regime_id")]), errors="coerce").fillna(0).iloc[0])
        raw20 = int(pd.to_numeric(pd.Series([row.get("cluster_id_raw20")]), errors="coerce").fillna(stable).iloc[0])
        if stable != raw20:
            flags.append("stable_raw20_mismatch")
        if int(pd.to_numeric(pd.Series([row.get("rare_flag_raw20")]), errors="coerce").fillna(0).iloc[0]) == 1:
            flags.append("rare_flag")
        return flags

    def _write_cluster_quality(
        self,
        *,
        cluster_context: pd.DataFrame,
        merged: pd.DataFrame,
        winners: pd.DataFrame,
        edge_table: pd.DataFrame,
        output_dir: Path,
        agents: Sequence[str],
    ) -> Dict[str, object]:
        quality_rows: List[Dict[str, object]] = []
        bootstrap_rows: List[Dict[str, object]] = []
        transition_matrix: Dict[str, Dict[str, int]] = {}
        fixed_best = compute_fixed_best_agent(merged, agents, split="train")
        merged_with_winner = merged.merge(winners, on=["Date", "Split"], how="left")
        merged_with_winner["assignment_uncertainty"] = 1.0 - pd.to_numeric(merged_with_winner.get("confidence_stable"), errors="coerce").fillna(1.0)
        boundary_records: List[Dict[str, object]] = []

        for split, split_df in merged_with_winner.groupby(merged_with_winner["Split"].astype(str).str.lower()):
            split_df = split_df.sort_values("Date").reset_index(drop=True)
            total = max(1, len(split_df))
            regime_counts = split_df["regime_id"].astype(int).value_counts().sort_index()
            transitions = split_df["regime_id"].astype(int).shift(1).astype("Int64").astype(str) + "->" + split_df["regime_id"].astype(int).astype(str)
            transition_counts = transitions[split_df["regime_id"].shift(1).notna()].value_counts().to_dict()
            transition_matrix[split] = {str(k): int(v) for k, v in transition_counts.items()}
            for regime_id, count in regime_counts.items():
                regime_df = split_df[split_df["regime_id"].astype(int) == int(regime_id)].copy()
                run_lengths = self._run_lengths(split_df["regime_id"].astype(int).tolist(), int(regime_id))
                train_rank_consistency = self._rank_consistency(merged, agents, int(regime_id))
                winner_separation = float(pd.to_numeric(regime_df.get("top1_top2_margin"), errors="coerce").mean()) if not regime_df.empty else 0.0
                fixed_series = pd.to_numeric(regime_df.get(f"ret_{fixed_best}"), errors="coerce") if fixed_best else pd.Series(dtype=float)
                oracle_series = pd.to_numeric(regime_df.get("top1_ret"), errors="coerce")
                fixed_best_gap = float(oracle_series.mean() - fixed_series.mean()) if fixed_best and not fixed_series.empty else float("nan")
                regime_best_mean = max(
                    [float(pd.to_numeric(regime_df.get(f"ret_{agent}"), errors="coerce").mean()) for agent in agents] or [float("nan")]
                )
                oracle_gap = float(oracle_series.mean() - regime_best_mean) if np.isfinite(regime_best_mean) else float("nan")
                quality_rows.append(
                    {
                        "split": split,
                        "regime_id": int(regime_id),
                        "occupancy": int(count),
                        "share": float(count / total),
                        "sample_count": int(count),
                        "mean_run_length": float(np.mean(run_lengths)) if run_lengths else 0.0,
                        "median_run_length": float(np.median(run_lengths)) if run_lengths else 0.0,
                        "fixed_best_gap": fixed_best_gap,
                        "oracle_gap": oracle_gap,
                        "winner_separation": winner_separation,
                        "rank_consistency": train_rank_consistency,
                        "confidence_mean": float(pd.to_numeric(regime_df.get("confidence_stable"), errors="coerce").mean()),
                        "margin_mean": float(pd.to_numeric(regime_df.get("top1_top2_margin"), errors="coerce").mean()),
                    }
                )
            bootstrap_rows.extend(self._bootstrap_regime_stability(split_df, split))
            for _, row in split_df.iterrows():
                row_dict = row.to_dict()
                row_dict.setdefault("assignment_uncertainty", 1.0 - float(row_dict.get("confidence_stable", 1.0) or 1.0))
                flags = self._boundary_flags(pd.Series(row_dict))
                if not flags:
                    continue
                boundary_records.append(
                    {
                        "date": pd.to_datetime(row_dict.get("Date")).strftime("%Y-%m-%d") if pd.notna(row_dict.get("Date")) else "",
                        "split": split,
                        "regime_id_stable": int(row_dict.get("regime_id", 0) or 0),
                        "cluster_id_raw20": int(row_dict.get("cluster_id_raw20", row_dict.get("regime_id", 0)) or 0),
                        "rare_flag_raw20": int(row_dict.get("rare_flag_raw20", 0) or 0),
                        "confidence_stable": float(row_dict.get("confidence_stable", 1.0) or 1.0),
                        "assignment_uncertainty": float(row_dict.get("assignment_uncertainty", 0.0) or 0.0),
                        "top1_expert": str(row_dict.get("top1_expert", "")),
                        "top2_expert": str(row_dict.get("top2_expert", "")),
                        "top1_top2_margin": float(row_dict.get("top1_top2_margin", 0.0) or 0.0),
                        "boundary_flags": "|".join(flags),
                    }
                )

        quality_df = pd.DataFrame(quality_rows)
        quality_df.to_csv(output_dir / "cluster_quality_by_regime.csv", index=False)
        pd.DataFrame(bootstrap_rows).to_csv(output_dir / "cluster_bootstrap_stability.csv", index=False)
        pd.DataFrame(boundary_records).to_csv(output_dir / "cluster_boundary_days.csv", index=False)
        summary = {
            "regime_count": int(cluster_context["regime_id"].nunique()) if not cluster_context.empty and "regime_id" in cluster_context.columns else 0,
            "transition_matrix": transition_matrix,
            "cluster_sample_count": {str(k): int(v) for k, v in cluster_context["regime_id"].astype(int).value_counts().sort_index().to_dict().items()} if "regime_id" in cluster_context.columns else {},
            "regime_winner_rank": self._regime_winner_rank(edge_table),
            "fixed_best_agent": fixed_best,
            "bootstrap_samples": int(self.config.bootstrap_samples),
            "boundary_day_count": int(len(boundary_records)),
        }
        (output_dir / "cluster_quality_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary

    def _write_cluster_usage(
        self,
        *,
        merged: pd.DataFrame,
        daily: pd.DataFrame,
        edge_table: pd.DataFrame,
        allowlist: pd.DataFrame,
        output_dir: Path,
        agents: Sequence[str],
    ) -> Dict[str, object]:
        allow_map = parse_allowlist(allowlist)
        rank_map = score_rank_map(edge_table)
        fixed_best = compute_fixed_best_agent(merged, agents, split="train")
        usage_rows: List[Dict[str, object]] = []
        attribution_counts = {
            "router": 0,
            "fallback": 0,
            "hard_routing": 0,
            "cluster_granularity": 0,
            "rare_policy": 0,
        }
        attribution_examples = {key: [] for key in attribution_counts}
        winner_mix = daily.groupby("regime_id")["winner_expert"].nunique().to_dict() if "winner_expert" in daily.columns else {}

        for _, row in daily.iterrows():
            split = str(row.get("Split", "test")).lower()
            regime_id = int(pd.to_numeric(pd.Series([row.get("regime_id")]), errors="coerce").fillna(0).iloc[0])
            allowed = allow_map.get(regime_id, allow_map.get(-1, list(agents)))
            winner = str(row.get("winner_expert", ""))
            chosen = str(row.get("selected_expert", row.get("chosen_expert", "")))
            winner_in_allowlist = winner in allowed if winner else False
            winner_rank = rank_map.get((regime_id, winner), {}).get("rank")
            chosen_scores = rank_map.get((regime_id, chosen), {})
            winner_scores = rank_map.get((regime_id, winner), {})
            chosen_ret = float(pd.to_numeric(pd.Series([row.get("ret_selected", row.get("ret"))]), errors="coerce").fillna(0.0).iloc[0])
            fixed_ret = float(pd.to_numeric(pd.Series([row.get(f"ret_{fixed_best}")]), errors="coerce").fillna(0.0).iloc[0]) if fixed_best else 0.0
            oracle_ret = float(pd.to_numeric(pd.Series([row.get("top1_ret")]), errors="coerce").fillna(0.0).iloc[0])
            margin = float(pd.to_numeric(pd.Series([row.get("top1_top2_margin")]), errors="coerce").fillna(0.0).iloc[0])
            confidence = float(pd.to_numeric(pd.Series([row.get("confidence_stable")]), errors="coerce").fillna(1.0).iloc[0])
            boundary_flag = bool(self._boundary_flags(pd.Series({**row.to_dict(), "assignment_uncertainty": 1.0 - confidence})))
            fallback_triggered = (not winner_in_allowlist) and winner != ""
            coverage = 1 if winner_in_allowlist else 0
            usage_rows.append(
                {
                    "Date": pd.to_datetime(row.get("Date")).strftime("%Y-%m-%d") if pd.notna(row.get("Date")) else "",
                    "split": split,
                    "regime_id": regime_id,
                    "cluster_id_stable": regime_id,
                    "cluster_id_raw20": int(pd.to_numeric(pd.Series([row.get("cluster_id_raw20")]), errors="coerce").fillna(regime_id).iloc[0]),
                    "rare_flag_raw20": int(pd.to_numeric(pd.Series([row.get("rare_flag_raw20")]), errors="coerce").fillna(0).iloc[0]),
                    "chosen_expert": chosen,
                    "chosen_ratio": float(pd.to_numeric(pd.Series([row.get("ratio")]), errors="coerce").fillna(0.0).iloc[0]),
                    "top1_expert": winner,
                    "top2_expert": str(row.get("top2_expert", "")),
                    "top1_top2_margin": margin,
                    "allowlist": "|".join(allowed),
                    "allowlist_rank_of_winner": int(winner_rank) if pd.notna(winner_rank) else -1,
                    "router_score": float(chosen_scores.get("router_score", np.nan)),
                    "edge_score": float(chosen_scores.get("edge_score", np.nan)),
                    "fallback_triggered": bool(fallback_triggered),
                    "candidate_coverage": coverage,
                    "dropped_winner_outside_candidates": bool(not winner_in_allowlist and winner != ""),
                    "ret": chosen_ret,
                    "regret_vs_fixed_best": float(fixed_ret - chosen_ret),
                    "regret_vs_oracle": float(oracle_ret - chosen_ret),
                }
            )
            label = None
            if winner and chosen != winner and winner_in_allowlist:
                label = "router"
            elif winner and not winner_in_allowlist:
                label = "fallback"
            elif winner and chosen != winner and boundary_flag and abs(margin) <= float(self.config.boundary_margin_threshold):
                label = "hard_routing"
            elif int(winner_mix.get(regime_id, 0)) >= max(3, len(agents) // 3):
                label = "cluster_granularity"
            elif int(pd.to_numeric(pd.Series([row.get("rare_flag_raw20")]), errors="coerce").fillna(0).iloc[0]) == 1:
                label = "rare_policy"
            if label:
                attribution_counts[label] += 1
                if len(attribution_examples[label]) < 3:
                    attribution_examples[label].append(
                        f"{pd.to_datetime(row.get('Date')).strftime('%Y-%m-%d')} regime={regime_id} chosen={chosen} winner={winner} margin={margin:.6f}"
                    )

        usage_df = pd.DataFrame(usage_rows)
        usage_df.to_csv(output_dir / "cluster_usage_audit_daily.csv", index=False)
        summary = {
            "regime_expert_adoption_count": usage_df.groupby(["regime_id", "chosen_expert"]).size().reset_index(name="count").to_dict(orient="records") if not usage_df.empty else [],
            "regime_winner_frequency": usage_df.groupby(["regime_id", "top1_expert"]).size().reset_index(name="count").to_dict(orient="records") if not usage_df.empty else [],
            "allowlist_contains_winner_count": int(usage_df["candidate_coverage"].sum()) if not usage_df.empty else 0,
            "fallback_induced_candidate_miss_rate": float((usage_df["dropped_winner_outside_candidates"].astype(int).mean())) if not usage_df.empty else 0.0,
            "candidate_set_coverage": candidate_coverage_rate(usage_df.get("candidate_coverage", [])),
            "boundary_day_performance": {
                "rows": int((usage_df["top1_top2_margin"].abs() <= float(self.config.boundary_margin_threshold)).sum()) if not usage_df.empty else 0,
                "mean_ret": float(usage_df.loc[usage_df["top1_top2_margin"].abs() <= float(self.config.boundary_margin_threshold), "ret"].mean()) if not usage_df.empty else 0.0,
            },
            "hard_routing_vs_soft_hypothesis": "soft-routing experiment output in soft_routing_ab_compare.csv",
            "rare_regime_candidate_drop_tendency": float(usage_df.loc[usage_df["rare_flag_raw20"] == 1, "dropped_winner_outside_candidates"].mean()) if not usage_df.empty and "rare_flag_raw20" in usage_df.columns else 0.0,
            "allowlist_outside_winner_frequency": float(usage_df["dropped_winner_outside_candidates"].mean()) if not usage_df.empty else 0.0,
            "fallback_trigger_performance_gap": float(usage_df.loc[usage_df["fallback_triggered"] == True, "regret_vs_oracle"].mean()) if not usage_df.empty else 0.0,
            "failure_attribution_counts": attribution_counts,
        }
        (output_dir / "cluster_usage_audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        self._write_failure_markdown(output_dir / "stepf_failure_attribution_summary.md", attribution_counts, attribution_examples)
        return summary

    def _write_focus_regime_audit(
        self,
        *,
        merged: pd.DataFrame,
        daily: pd.DataFrame,
        edge_table: pd.DataFrame,
        allowlist: pd.DataFrame,
        output_dir: Path,
    ) -> Dict[str, object]:
        focus = set(int(x) for x in self.config.focus_regimes)
        rank_map = score_rank_map(edge_table)
        allow_map = parse_allowlist(allowlist)
        focus_df = daily[daily["regime_id"].astype(int).isin(focus)].copy()
        rows: List[Dict[str, object]] = []
        for _, row in focus_df.iterrows():
            regime_id = int(row["regime_id"])
            chosen = str(row.get("selected_expert", ""))
            winner = str(row.get("winner_expert", ""))
            winner_rank = rank_map.get((regime_id, winner), {}).get("rank")
            chosen_scores = rank_map.get((regime_id, chosen), {})
            winner_scores = rank_map.get((regime_id, winner), {})
            allowed = allow_map.get(regime_id, allow_map.get(-1, []))
            confidence = float(pd.to_numeric(pd.Series([row.get("confidence_stable")]), errors="coerce").fillna(1.0).iloc[0])
            margin = float(pd.to_numeric(pd.Series([row.get("top1_top2_margin")]), errors="coerce").fillna(0.0).iloc[0])
            rows.append(
                {
                    "date": pd.to_datetime(row.get("Date")).strftime("%Y-%m-%d") if pd.notna(row.get("Date")) else "",
                    "split": str(row.get("Split", "test")).lower(),
                    "regime_id": regime_id,
                    "chosen_expert": chosen,
                    "chosen_ratio": float(pd.to_numeric(pd.Series([row.get("ratio")]), errors="coerce").fillna(0.0).iloc[0]),
                    "winner_expert": winner,
                    "ret": float(pd.to_numeric(pd.Series([row.get("ret")]), errors="coerce").fillna(0.0).iloc[0]),
                    "regret_vs_fixed_best": float(pd.to_numeric(pd.Series([row.get("regret_vs_fixed_best")]), errors="coerce").fillna(0.0).iloc[0]),
                    "regret_vs_oracle": float(pd.to_numeric(pd.Series([row.get("regret_vs_oracle")]), errors="coerce").fillna(0.0).iloc[0]),
                    "router_score_chosen": float(chosen_scores.get("router_score", np.nan)),
                    "router_score_winner": float(winner_scores.get("router_score", np.nan)),
                    "edge_score_chosen": float(chosen_scores.get("edge_score", np.nan)),
                    "edge_score_winner": float(winner_scores.get("edge_score", np.nan)),
                    "allowlist": "|".join(allowed),
                    "allowlist_rank_of_winner": int(winner_rank) if pd.notna(winner_rank) else -1,
                    "dropped_by_candidate_policy": bool(winner not in allowed and winner != ""),
                    "fallback_triggered": bool(winner not in allowed and winner != ""),
                    "top1_top2_margin": margin,
                    "boundary_flag": bool(abs(margin) <= float(self.config.boundary_margin_threshold) or confidence <= float(self.config.boundary_confidence_threshold)),
                }
            )
        deep_df = pd.DataFrame(rows)
        deep_df.to_csv(output_dir / "regime8_deep_audit_daily.csv", index=False)
        summary_lines = [
            "# Regime deep audit",
            "",
            f"focus_regimes: {sorted(focus)}",
            f"rows: {len(deep_df)}",
            f"allowlist外 winner 発生率: {float(deep_df['dropped_by_candidate_policy'].mean()) if not deep_df.empty else 0.0:.4f}",
            f"fallback 発火頻度: {float(deep_df['fallback_triggered'].mean()) if not deep_df.empty else 0.0:.4f}",
            f"boundary-day 集中度: {float(deep_df['boundary_flag'].mean()) if not deep_df.empty else 0.0:.4f}",
            "",
            "## 主要 expert 比較",
        ]
        if not deep_df.empty:
            expert_comp = deep_df.groupby(["chosen_expert", "winner_expert"]).size().reset_index(name="count")
            for rec in expert_comp.head(10).to_dict(orient="records"):
                summary_lines.append(f"- chosen={rec['chosen_expert']} winner={rec['winner_expert']} count={rec['count']}")
            close_loss_rate = float(((deep_df["winner_expert"] != deep_df["chosen_expert"]) & (deep_df["top1_top2_margin"].abs() <= float(self.config.boundary_margin_threshold))).mean())
            summary_lines.extend(
                [
                    "",
                    f"僅差負けの割合: {close_loss_rate:.4f}",
                    f"allowlist 内順位平均: {float(deep_df.loc[deep_df['allowlist_rank_of_winner'] > 0, 'allowlist_rank_of_winner'].mean()) if not deep_df.empty else 0.0:.4f}",
                    "",
                    "## 代表例",
                ]
            )
            for rec in deep_df.head(5).to_dict(orient="records"):
                summary_lines.append(
                    f"- {rec['date']} regime={rec['regime_id']} chosen={rec['chosen_expert']} winner={rec['winner_expert']} margin={rec['top1_top2_margin']:.6f} dropped={rec['dropped_by_candidate_policy']}"
                )
        summary_lines.extend(["", "failure attribution の暫定結論: router / allowlist / boundary-day を優先診断対象とする。"])
        (output_dir / "regime8_deep_audit_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
        return {"focus_regimes": sorted(focus), "rows": int(len(deep_df))}

    @staticmethod
    def _run_lengths(regimes: List[int], target: int) -> List[int]:
        lengths: List[int] = []
        current = 0
        for regime in regimes:
            if regime == target:
                current += 1
            elif current:
                lengths.append(current)
                current = 0
        if current:
            lengths.append(current)
        return lengths

    def _bootstrap_regime_stability(self, split_df: pd.DataFrame, split: str) -> List[Dict[str, object]]:
        if split_df.empty:
            return []
        rng = np.random.default_rng(int(self.config.bootstrap_seed))
        n = len(split_df)
        regimes = sorted(split_df["regime_id"].astype(int).unique().tolist())
        boot: Dict[int, List[float]] = {rid: [] for rid in regimes}
        co_membership: Dict[int, List[float]] = {rid: [] for rid in regimes}
        base_prev_same = split_df["regime_id"].astype(int).eq(split_df["regime_id"].astype(int).shift(1))
        for _ in range(int(self.config.bootstrap_samples)):
            idx = rng.integers(0, n, size=n)
            sample = split_df.iloc[idx].sort_values("Date")
            counts = sample["regime_id"].astype(int).value_counts(normalize=True)
            sample_prev_same = sample["regime_id"].astype(int).eq(sample["regime_id"].astype(int).shift(1))
            for rid in regimes:
                boot[rid].append(float(counts.get(rid, 0.0)))
                mask = sample["regime_id"].astype(int) == rid
                if mask.any():
                    co_membership[rid].append(float(sample_prev_same[mask].mean()))
        rows = []
        for rid in regimes:
            rows.append(
                {
                    "split": split,
                    "regime_id": int(rid),
                    "occupancy_mean": float(np.mean(boot[rid])) if boot[rid] else 0.0,
                    "occupancy_std": float(np.std(boot[rid])) if boot[rid] else 0.0,
                    "bootstrap_stability": float(1.0 / (1.0 + np.std(boot[rid]))) if boot[rid] else 0.0,
                    "co_membership_stability": float(np.mean(co_membership[rid])) if co_membership[rid] else 0.0,
                }
            )
        return rows

    @staticmethod
    def _rank_consistency(merged: pd.DataFrame, agents: Sequence[str], regime_id: int) -> float:
        train_df = merged[(merged["Split"].astype(str).str.lower() == "train") & (merged["regime_id"].astype(int) == regime_id)].copy()
        test_df = merged[(merged["Split"].astype(str).str.lower() == "test") & (merged["regime_id"].astype(int) == regime_id)].copy()
        if train_df.empty or test_df.empty:
            return float("nan")
        train_means = pd.Series({agent: float(pd.to_numeric(train_df.get(f"ret_{agent}"), errors="coerce").mean()) for agent in agents})
        test_means = pd.Series({agent: float(pd.to_numeric(test_df.get(f"ret_{agent}"), errors="coerce").mean()) for agent in agents})
        return float(train_means.rank(ascending=False).corr(test_means.rank(ascending=False), method="spearman"))

    @staticmethod
    def _regime_winner_rank(edge_table: pd.DataFrame) -> Dict[str, List[Dict[str, object]]]:
        out: Dict[str, List[Dict[str, object]]] = {}
        if edge_table.empty:
            return out
        score_col = "IR_shrink" if "IR_shrink" in edge_table.columns else "IR"
        for regime_id, regime_df in edge_table.groupby("regime_id"):
            ranked = regime_df.sort_values([score_col, "EV_shrink" if "EV_shrink" in regime_df.columns else "EV"], ascending=[False, False])
            out[str(int(regime_id))] = ranked[["agent", score_col]].rename(columns={score_col: "score"}).to_dict(orient="records")
        return out

    @staticmethod
    def _write_failure_markdown(path: Path, counts: Dict[str, int], examples: Dict[str, List[str]]) -> None:
        lines = ["# StepF failure attribution summary", ""]
        labels = {
            "router": "router起因",
            "fallback": "fallback起因",
            "hard_routing": "hard routing起因",
            "cluster_granularity": "cluster粒度起因",
            "rare_policy": "rare 専用 policy 起因",
        }
        for key, title in labels.items():
            lines.append(f"## {title}")
            lines.append(f"- 件数: {counts.get(key, 0)}")
            reps = examples.get(key, [])
            if reps:
                lines.append("- 代表例:")
                for rep in reps:
                    lines.append(f"  - {rep}")
            else:
                lines.append("- 代表例: なし")
            lines.append(f"- 要約: {title} の疑いがある日次事象を router-side 監査で集計。")
            lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")
