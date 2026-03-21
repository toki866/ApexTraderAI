from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ai_core.live.branch_materializer import BranchMaterializer
from ai_core.live.branch_specs import BRANCHES, DEFAULT_SAFE_BRANCHES
from ai_core.live.feature_store import FeatureStore
from ai_core.live.step_e_policy import StepEPolicy
from ai_core.services.step_f_service import StepFRouterConfig, StepFService


@dataclass
class TwoStageConfig:
    stage0_topk: int = 3
    fit_window_days: int = 504
    safe_branches: str = ",".join(DEFAULT_SAFE_BRANCHES)
    topk_branches_per_regime: int = 3
    min_samples_regime: int = 20
    topk_filter_ev_positive: bool = True
    softmax_beta: float = 1.0
    ema_alpha: float = 0.3
    pos_limit: float = 1.0
    eps_ir: float = 1e-8
    refresh_stepb: bool = False
    refresh_dprime: bool = False
    dry_run: bool = False
    stepdprime_pred_k: int = 20
    stepdprime_l_past: int = 63


class StepFTwoStageRouter:
    def __init__(self, output_root: str = "output") -> None:
        self.output_root = output_root

    def run_close_pre(self, symbol: str, mode: str, target_date: str, output_root: str, config: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        t0 = time.perf_counter()
        cfg = TwoStageConfig(**(config or {}))
        m = str(mode or "sim").strip().lower()
        if m in {"ops", "prod", "production", "real"}:
            m = "live"
        target_dt = pd.to_datetime(target_date, errors="coerce")
        if pd.isna(target_dt):
            raise ValueError(f"invalid target_date={target_date}")
        target_dt = target_dt.normalize()
        out_root = Path(output_root)
        svc = StepFService(app_config=SimpleNamespace(output_root=str(out_root)))

        all_agents = sorted(BRANCHES.keys())
        resolved_agents, _discovered_logs, _requested_agents, step_e_root = svc._resolve_agents(
            out_root=out_root,
            input_mode=m,
            symbol=symbol,
            requested_agents_raw="",
        )
        model_agents = [a for a in self._discover_agents(out_root, m, symbol) if a in resolved_agents]
        if not model_agents and resolved_agents:
            model_agents = list(resolved_agents)
        if not model_agents:
            raise RuntimeError(f"No StepE models found for {symbol} mode={m}")
        agents = [a for a in all_agents if a in model_agents]

        timings: Dict[str, float] = {}
        t = time.perf_counter()
        cluster_context, cluster_meta = svc._load_cluster_context(
            out_root=out_root,
            input_mode=m,
            mode=m,
            retrain="off",
            symbol=symbol,
            cfg=StepFRouterConfig(output_root=str(out_root)),
        )
        phase2_df, regime_dminus1, regime_d, fit_info = self._prepare_cluster_context_for_day(cluster_context, target_dt, cfg, cluster_meta)
        timings["cluster_context_sec"] = time.perf_counter() - t

        safe_set = [a.strip() for a in cfg.safe_branches.split(",") if a.strip() and a.strip() in agents]
        if not safe_set:
            safe_set = [a for a in DEFAULT_SAFE_BRANCHES if a in agents] or agents[: min(2, len(agents))]

        t = time.perf_counter()
        edge_table, allowlist_df, router_artifacts = self._load_router_artifacts(
            out_root=out_root,
            mode=m,
            symbol=symbol,
            agents=agents,
            safe_set=safe_set,
        )
        allow_map = {int(r.regime_id): [a for a in str(r.allowed_agents).split("|") if a] for r in allowlist_df.itertuples(index=False)}
        cached_safe = allow_map.get(-1, [])
        if cached_safe:
            safe_set = [a for a in cached_safe if a in agents]
        timings["edge_allowlist_sec"] = time.perf_counter() - t

        t = time.perf_counter()
        topk_candidates = self._stage0_candidates(series=phase2_df[["Date", "regime_id"]], target_date=target_dt, prev_regime_id=regime_dminus1, topk=cfg.stage0_topk)
        branches_stage0 = set(safe_set)
        for rid in topk_candidates:
            branches_stage0.update(allow_map.get(int(rid), []))
        branches_stage0 = sorted(a for a in branches_stage0 if a in agents)
        timings["stage0_sec"] = time.perf_counter() - t

        t = time.perf_counter()
        regime_final = int(regime_d)
        branches_final = [a for a in allow_map.get(regime_final, safe_set) if a in branches_stage0]
        if not branches_final:
            branches_final = list(branches_stage0)
        timings["stage1_select_sec"] = time.perf_counter() - t

        t = time.perf_counter()
        if bool(cfg.dry_run):
            mat = BranchMaterializer(output_root=str(out_root)).materialize(
                symbol=symbol,
                mode=m,
                target_date=str(target_dt.date()),
                branches_final=branches_final,
                refresh_stepb=False,
                refresh_dprime=False,
                pred_k=int(cfg.stepdprime_pred_k),
                l_past=int(cfg.stepdprime_l_past),
            )
        else:
            mat = BranchMaterializer(output_root=str(out_root)).materialize(
                symbol=symbol,
                mode=m,
                target_date=str(target_dt.date()),
                branches_final=branches_final,
                refresh_stepb=bool(cfg.refresh_stepb),
                refresh_dprime=bool(cfg.refresh_dprime),
                pred_k=int(cfg.stepdprime_pred_k),
                l_past=int(cfg.stepdprime_l_past),
            )
        timings["branch_materialize_sec"] = time.perf_counter() - t

        t = time.perf_counter()
        state_path = out_root / "stepF" / m / "live_close_pre" / f"state_{symbol}.json"
        state = self._load_state(state_path)
        pos_prev = float(state.get("last_ratio", 0.0) or 0.0)

        obs_cols_union = []
        policies: Dict[str, StepEPolicy] = {}
        for agent in branches_final:
            model_path = out_root / "stepE" / m / "models" / f"stepE_{agent}_{symbol}.pt"
            pol = StepEPolicy(model_path=model_path)
            policies[agent] = pol
            obs_cols_union.extend(pol.obs_cols)
        obs_cols_union = sorted(set(obs_cols_union))
        fs = FeatureStore(output_root=str(out_root), mode=m, symbol=symbol)
        obs = fs.get_row(target_dt, obs_cols_union)

        ratio_by_agent: Dict[str, float] = {}
        for agent in branches_final:
            ratio_by_agent[agent] = policies[agent].predict(obs, pos_prev=pos_prev, pos_limit=cfg.pos_limit)
        timings["agent_infer_sec"] = time.perf_counter() - t

        t = time.perf_counter()
        ir_map = {(int(r.regime_id), str(r.agent)): float(r.IR) for r in edge_table.itertuples(index=False)}
        scores = np.array([ir_map.get((regime_final, a), np.nan) for a in branches_final], dtype=float)
        if np.any(np.isnan(scores)):
            w_raw = np.ones(len(branches_final), dtype=float) / max(1, len(branches_final))
        else:
            z = cfg.softmax_beta * scores
            z = z - np.max(z)
            ex = np.exp(z)
            w_raw = ex / max(ex.sum(), 1e-12)

        ema_prev = {k: float(v) for k, v in (state.get("ema_weights", {}) or {}).items()}
        w_by_agent = {a: 0.0 for a in branches_final}
        for i, a in enumerate(branches_final):
            smoothed = cfg.ema_alpha * float(w_raw[i]) + (1.0 - cfg.ema_alpha) * ema_prev.get(a, 0.0)
            w_by_agent[a] = smoothed
        sw = sum(w_by_agent.values())
        if sw <= 0:
            for a in branches_final:
                w_by_agent[a] = 1.0 / max(1, len(branches_final))
        else:
            for a in list(w_by_agent.keys()):
                w_by_agent[a] = w_by_agent[a] / sw

        ratio_final = float(sum(w_by_agent.get(a, 0.0) * ratio_by_agent.get(a, 0.0) for a in branches_final))
        ratio_final = float(np.clip(ratio_final, -cfg.pos_limit, cfg.pos_limit))
        timings["stage1_aggregate_sec"] = time.perf_counter() - t

        out_dir = out_root / "stepF" / m / "live_close_pre"
        out_dir.mkdir(parents=True, exist_ok=True)
        decision = {
            "symbol": symbol,
            "mode": m,
            "target_date": str(target_dt.date()),
            "fit_end_date": fit_info.get("fit_end_date"),
            "fit_window_days": fit_info.get("fit_window_days"),
            "n_fit_rows": fit_info.get("n_train_rows"),
            "params": asdict(cfg),
            "cluster": {
                "input_cluster_source": cluster_meta.get("assignments_source", ""),
                "cluster_model_version": cluster_meta.get("cluster_model_version", ""),
                "cluster_refresh_mode": cluster_meta.get("cluster_refresh_mode", "monthly_reuse"),
                "cluster_training_performed": False,
            },
            "router_context": router_artifacts,
            "stage0": {
                "prev_regime_id": int(regime_dminus1),
                "topk_candidates": [int(x) for x in topk_candidates],
                "branches_stage0": branches_stage0,
            },
            "stage1": {
                "regime_id": int(regime_final),
                "branches_final": branches_final,
            },
            "stepB": {
                "executed": bool(mat.stepb_executed),
                "output_paths": mat.stepb_paths,
                "pred_k": int(mat.stepb_pred_k),
                "horizons": mat.stepb_horizons,
            },
            "dprime": {
                "executed_profiles": mat.dprime_executed_profiles,
                "output_paths": mat.dprime_paths,
            },
            "ratios": {k: float(v) for k, v in ratio_by_agent.items()},
            "weights": {k: float(v) for k, v in w_by_agent.items() if v > 0},
            "ratio_final": ratio_final,
            "state_in": {
                "pos_prev": pos_prev,
                "ema_prev": ema_prev,
            },
            "dry_run": bool(cfg.dry_run),
            "fit_info": fit_info,
            "timing": {**timings, "total_sec": time.perf_counter() - t0},
            "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        }

        dec_path = out_dir / f"decision_{symbol}_{target_dt.strftime('%Y%m%d')}.json"
        dec_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
        self._append_decision_csv(out_dir / f"decisions_{symbol}.csv", decision)

        if not bool(cfg.dry_run):
            next_state = {
                "last_ratio": ratio_final,
                "ema_weights": {k: float(v) for k, v in w_by_agent.items()},
                "last_regime_id": int(regime_final),
                "last_date": str(target_dt.date()),
                "updated_at_utc": pd.Timestamp.utcnow().isoformat(),
            }
            state_path.write_text(json.dumps(next_state, ensure_ascii=False, indent=2), encoding="utf-8")
        return decision

    def _discover_agents(self, out_root: Path, mode: str, symbol: str) -> List[str]:
        model_dir = out_root / "stepE" / mode / "models"
        out = []
        for p in sorted(model_dir.glob(f"stepE_*_{symbol}.pt")):
            stem = p.stem
            prefix = "stepE_"
            suffix = f"_{symbol}"
            if stem.startswith(prefix) and stem.endswith(suffix):
                out.append(stem[len(prefix) : -len(suffix)])
        return out

    def _load_router_artifacts(
        self,
        out_root: Path,
        mode: str,
        symbol: str,
        agents: List[str],
        safe_set: List[str],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
        candidates: List[Tuple[str, Path]] = []
        if mode == "live":
            candidates.append(("live_retrain_on", out_root / "stepF" / "live" / "retrain_on" / "router"))
            candidates.append(("sim", out_root / "stepF" / "sim" / "router"))
        else:
            candidates.append((mode, out_root / "stepF" / mode / "router"))

        attempted: List[str] = []
        for source_name, router_dir in candidates:
            edge_path = router_dir / f"regime_edge_table_{symbol}.csv"
            allow_path = router_dir / f"router_allowlist_{symbol}.csv"
            attempted.append(f"{source_name}:{edge_path}|{allow_path}")
            if not edge_path.exists() or not allow_path.exists():
                continue

            edge_table = pd.read_csv(edge_path)
            allowlist_df = pd.read_csv(allow_path)
            edge_table = edge_table[edge_table["agent"].astype(str).isin(agents)].copy()
            allowlist_df = allowlist_df.copy()
            if allowlist_df.empty:
                continue
            allowlist_df["allowed_agents"] = allowlist_df["allowed_agents"].map(
                lambda raw: "|".join(
                    [agent for agent in str(raw).split("|") if agent and agent in agents]
                )
            )
            if not (allowlist_df["regime_id"].astype(int) == -1).any():
                allowlist_df = pd.concat(
                    [
                        allowlist_df,
                        pd.DataFrame([{"regime_id": -1, "allowed_agents": "|".join(safe_set)}]),
                    ],
                    ignore_index=True,
                )
            return edge_table, allowlist_df, {
                "allowlist_source": "cached_router_allowlist",
                "router_artifact_source": source_name,
                "router_edge_path": str(edge_path),
                "router_allowlist_path": str(allow_path),
            }

        raise FileNotFoundError(
            "cached router artifacts are required for close-pre execution; "
            f"attempted={attempted}"
        )

    def _prepare_cluster_context_for_day(
        self,
        cluster_context: pd.DataFrame,
        target_dt: pd.Timestamp,
        cfg: TwoStageConfig,
        cluster_meta: Dict[str, object],
    ) -> Tuple[pd.DataFrame, int, int, Dict[str, object]]:
        phase2_df = cluster_context.copy()
        phase2_df["Date"] = pd.to_datetime(phase2_df["Date"], errors="coerce").dt.normalize()
        phase2_df = phase2_df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
        fit_end = (target_dt - pd.Timedelta(days=1)).normalize()
        fit_df = phase2_df[phase2_df["Date"] <= fit_end].copy()
        if int(cfg.fit_window_days) > 0 and len(fit_df) > int(cfg.fit_window_days):
            fit_df = fit_df.iloc[-int(cfg.fit_window_days):].copy()
        if fit_df.empty:
            raise RuntimeError(f"insufficient upstream cluster rows for phase2: {len(fit_df)}")

        def _cluster_at(dt: pd.Timestamp, *, required: bool) -> int:
            row = phase2_df.loc[phase2_df["Date"] == dt]
            if row.empty:
                if required:
                    raise RuntimeError(
                        "missing upstream cluster assignment for "
                        f"date={dt.date()} source={cluster_meta.get('assignments_source', '')}"
                    )
                return -1
            regime_id = int(pd.to_numeric(row["regime_id"], errors="coerce").fillna(-1).iloc[-1])
            if regime_id < 0 and required:
                raise RuntimeError(
                    "invalid upstream cluster assignment for "
                    f"date={dt.date()} regime_id={regime_id} source={cluster_meta.get('assignments_source', '')}"
                )
            return regime_id

        dminus1 = _cluster_at(fit_end, required=True)
        d = _cluster_at(target_dt, required=True)
        fit_info = {
            "fit_end_date": str(fit_end.date()),
            "fit_window_days": int(cfg.fit_window_days),
            "n_train_rows": int(len(fit_df)),
            "cluster_source": str(cluster_meta.get("assignments_source", "")),
            "cluster_model_version": str(cluster_meta.get("cluster_model_version", "")),
            "cluster_refresh_mode": str(cluster_meta.get("cluster_refresh_mode", "monthly_reuse")),
        }
        return phase2_df, dminus1, d, fit_info

    def _stage0_candidates(self, series: pd.DataFrame, target_date: pd.Timestamp, prev_regime_id: int, topk: int) -> List[int]:
        prev_date = (target_date - pd.Timedelta(days=1)).normalize()
        hist = series[series["Date"] <= prev_date].copy().sort_values("Date")
        r = hist["regime_id"].astype(int).to_numpy()
        if len(r) < 2:
            return [int(prev_regime_id)]
        trans: Dict[Tuple[int, int], int] = {}
        for a, b in zip(r[:-1], r[1:]):
            trans[(int(a), int(b))] = trans.get((int(a), int(b)), 0) + 1
        cands = [(nxt, cnt) for (prv, nxt), cnt in trans.items() if int(prv) == int(prev_regime_id)]
        if not cands:
            return [int(prev_regime_id)]
        cands.sort(key=lambda x: x[1], reverse=True)
        return [int(x[0]) for x in cands[: max(1, int(topk))]]

    def _load_state(self, path: Path) -> Dict[str, object]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _append_decision_csv(self, path: Path, decision: Dict[str, object]) -> None:
        row = {
            "timestamp_utc": decision.get("timestamp_utc", ""),
            "date": decision.get("target_date", ""),
            "symbol": decision.get("symbol", ""),
            "mode": decision.get("mode", ""),
            "prev_regime_id": decision.get("stage0", {}).get("prev_regime_id", -1),
            "stage0_candidates": "|".join(map(str, decision.get("stage0", {}).get("topk_candidates", []))),
            "regime_id_final": decision.get("stage1", {}).get("regime_id", -1),
            "n_agents_inferred": len(decision.get("ratios", {}) or {}),
            "n_agents_final": len(decision.get("stage1", {}).get("branches_final", []) or []),
            "ratio_final": decision.get("ratio_final", 0.0),
            "fit_end_date": decision.get("fit_info", {}).get("fit_end_date", ""),
            "fit_window_days": decision.get("fit_info", {}).get("fit_window_days", ""),
            "timing_total_sec": decision.get("timing", {}).get("total_sec", 0.0),
        }
        df = pd.DataFrame([row])
        if path.exists():
            prev = pd.read_csv(path)
            df = pd.concat([prev, df], ignore_index=True)
        df.to_csv(path, index=False)


def run_close_pre(symbol: str, mode: str, target_date: str, output_root: str = "output", config: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    return StepFTwoStageRouter(output_root=output_root).run_close_pre(
        symbol=symbol,
        mode=mode,
        target_date=target_date,
        output_root=output_root,
        config=config,
    )
