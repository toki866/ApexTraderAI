from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler

from ai_core.live.feature_store import FeatureStore
from ai_core.live.step_e_policy import StepEPolicy
from ai_core.services.step_dprime_service import _compute_base_features
from ai_core.services.step_f_service import StepFRouterConfig, StepFService

try:
    from hdbscan import HDBSCAN
    from hdbscan import prediction as hdbscan_prediction
except Exception:  # pragma: no cover
    HDBSCAN = None
    hdbscan_prediction = None


@dataclass
class TwoStageConfig:
    stage0_topk: int = 3
    fit_window_days: int = 504
    robust_scaler: bool = True
    pca_n_components: int = 30
    hdbscan_min_cluster_size: int = 30
    hdbscan_min_samples: int = 10
    past_window_days: int = 63
    past_resample_len: int = 20
    safe_set: str = "dprime_bnf_h01,dprime_all_features_h01"
    topK_agents_per_regime: int = 3
    min_samples_regime: int = 20
    topk_filter_ev_positive: bool = True
    softmax_beta: float = 1.0
    ema_alpha: float = 0.3
    pos_limit: float = 1.0
    eps_ir: float = 1e-8


class StepFTwoStageRouter:
    def __init__(self, output_root: str = "output") -> None:
        self.output_root = output_root

    def run_close_pre(self, symbol: str, mode: str, target_date: str, output_root: str, config: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        if HDBSCAN is None or hdbscan_prediction is None:
            raise ImportError("hdbscan is required for StepF two-stage router")
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

        agents = self._discover_agents(out_root, m, symbol)
        if not agents:
            raise RuntimeError(f"No StepE models found for {symbol} mode={m}")

        timings: Dict[str, float] = {}
        t = time.perf_counter()
        price_tech = StepFService(app_config=None)._load_stepa_price_tech(out_root=out_root, mode=m, symbol=symbol)
        phase2_df, regime_dminus1, regime_d, fit_info = self._build_phase2_for_day(price_tech, target_dt, cfg)
        timings["phase2_fit_predict_sec"] = time.perf_counter() - t

        t = time.perf_counter()
        logs_map = StepFService(app_config=None)._load_stepe_logs(out_root=out_root, mode=m, symbol=symbol, agents=agents)
        merged_for_stats = phase2_df[["Date", "regime_id"]].copy()
        for agent in agents:
            adf = logs_map[agent][["Date", "Split", "ratio", "stepE_ret_for_stats"]].copy()
            adf = adf.rename(columns={"ratio": f"ratio_{agent}", "stepE_ret_for_stats": f"ret_{agent}", "Split": f"Split_{agent}"})
            merged_for_stats = merged_for_stats.merge(adf, on="Date", how="left")
        merged_for_stats = merged_for_stats[merged_for_stats["Date"] <= (target_dt - pd.Timedelta(days=1))].copy()
        split_cols = [f"Split_{a}" for a in agents if f"Split_{a}" in merged_for_stats.columns]
        merged_for_stats["Split"] = merged_for_stats[split_cols].bfill(axis=1).iloc[:, 0].fillna("train") if split_cols else "train"

        fcfg = StepFRouterConfig(
            topK=cfg.topK_agents_per_regime,
            min_samples_regime=cfg.min_samples_regime,
            topk_filter_ev_positive=cfg.topk_filter_ev_positive,
            eps_ir=cfg.eps_ir,
        )
        safe_set = [a.strip() for a in cfg.safe_set.split(",") if a.strip() and a.strip() in agents]
        if not safe_set:
            safe_set = agents[: min(2, len(agents))]

        svc = StepFService(app_config=None)
        edge_table = svc._build_regime_edge_table(merged_for_stats, agents=agents, cfg=fcfg)
        allowlist_df = svc._build_allowlist(edge_table=edge_table, agents=agents, safe_set=safe_set, cfg=fcfg)
        allow_map = {int(r.regime_id): [a for a in str(r.allowed_agents).split("|") if a] for r in allowlist_df.itertuples(index=False)}
        timings["edge_allowlist_sec"] = time.perf_counter() - t

        t = time.perf_counter()
        topk_candidates = self._stage0_candidates(series=phase2_df[["Date", "regime_id"]], target_date=target_dt, prev_regime_id=regime_dminus1, topk=cfg.stage0_topk)
        candidate_agents = set(safe_set)
        for rid in topk_candidates:
            candidate_agents.update(allow_map.get(int(rid), []))
        candidate_agents = sorted(a for a in candidate_agents if a in agents)
        timings["stage0_sec"] = time.perf_counter() - t

        t = time.perf_counter()
        state_path = out_root / "stepF" / m / "live_close_pre" / f"state_{symbol}.json"
        state = self._load_state(state_path)
        pos_prev = float(state.get("last_ratio", 0.0) or 0.0)

        obs_cols_union = []
        policies: Dict[str, StepEPolicy] = {}
        for agent in candidate_agents:
            model_path = out_root / "stepE" / m / "models" / f"stepE_{agent}_{symbol}.pt"
            pol = StepEPolicy(model_path=model_path)
            policies[agent] = pol
            obs_cols_union.extend(pol.obs_cols)
        obs_cols_union = sorted(set(obs_cols_union))
        fs = FeatureStore(output_root=str(out_root), mode=m, symbol=symbol)
        obs = fs.get_row(target_dt, obs_cols_union)

        ratio_by_agent: Dict[str, float] = {}
        for agent in candidate_agents:
            ratio_by_agent[agent] = policies[agent].predict(obs, pos_prev=pos_prev, pos_limit=cfg.pos_limit)
        timings["agent_infer_sec"] = time.perf_counter() - t

        t = time.perf_counter()
        regime_final = int(regime_d)
        allowed_final = allow_map.get(regime_final, safe_set)
        allowed_final = [a for a in allowed_final if a in candidate_agents] or list(candidate_agents)

        ir_map = {(int(r.regime_id), str(r.agent)): float(r.IR) for r in edge_table.itertuples(index=False)}
        scores = np.array([ir_map.get((regime_final, a), np.nan) for a in allowed_final], dtype=float)
        if np.any(np.isnan(scores)):
            w_raw = np.ones(len(allowed_final), dtype=float) / max(1, len(allowed_final))
        else:
            z = cfg.softmax_beta * scores
            z = z - np.max(z)
            ex = np.exp(z)
            w_raw = ex / max(ex.sum(), 1e-12)

        ema_prev = {k: float(v) for k, v in (state.get("ema_weights", {}) or {}).items()}
        w_by_agent = {a: 0.0 for a in candidate_agents}
        for i, a in enumerate(allowed_final):
            smoothed = cfg.ema_alpha * float(w_raw[i]) + (1.0 - cfg.ema_alpha) * ema_prev.get(a, 0.0)
            w_by_agent[a] = smoothed
        sw = sum(w_by_agent.values())
        if sw <= 0:
            for a in allowed_final:
                w_by_agent[a] = 1.0 / max(1, len(allowed_final))
        else:
            for a in list(w_by_agent.keys()):
                w_by_agent[a] = w_by_agent[a] / sw

        ratio_final = float(sum(w_by_agent.get(a, 0.0) * ratio_by_agent.get(a, 0.0) for a in candidate_agents))
        ratio_final = float(np.clip(ratio_final, -cfg.pos_limit, cfg.pos_limit))
        timings["stage1_aggregate_sec"] = time.perf_counter() - t

        out_dir = out_root / "stepF" / m / "live_close_pre"
        out_dir.mkdir(parents=True, exist_ok=True)
        decision = {
            "symbol": symbol,
            "mode": m,
            "target_date": str(target_dt.date()),
            "stage0": {
                "prev_regime_id": int(regime_dminus1),
                "topk_candidates": [int(x) for x in topk_candidates],
                "allowed_agents_union": candidate_agents,
            },
            "stage1": {
                "regime_id_final": int(regime_final),
                "allowed_agents_final": allowed_final,
            },
            "ratios": {k: float(v) for k, v in ratio_by_agent.items()},
            "weights": {k: float(v) for k, v in w_by_agent.items() if v > 0},
            "ratio_final": ratio_final,
            "fit_info": fit_info,
            "timing": {**timings, "total_sec": time.perf_counter() - t0},
            "config": asdict(cfg),
            "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        }

        dec_path = out_dir / f"decision_{symbol}_{target_dt.strftime('%Y%m%d')}.json"
        dec_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
        self._append_decision_csv(out_dir / f"decisions_{symbol}.csv", decision)

        next_state = {
            "last_ratio": ratio_final,
            "ema_weights": {k: float(v) for k, v in w_by_agent.items()},
            "last_regime_id": int(regime_final),
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

    def _build_phase2_for_day(self, price_tech: pd.DataFrame, target_dt: pd.Timestamp, cfg: TwoStageConfig) -> Tuple[pd.DataFrame, int, int, Dict[str, object]]:
        tech_cols = [c for c in price_tech.columns if c.endswith("_tech")]
        tech = price_tech[["Date"] + tech_cols].copy()
        tech.columns = ["Date"] + [c.replace("_tech", "") for c in tech_cols]
        base_feat = _compute_base_features(price_tech, tech)
        features = [c for c in ["Gap", "ATR_norm", "gap_atr", "vol_log_ratio_20", "bnf_score", "ret_1", "ret_5", "ret_20", "range_atr", "body_ratio", "lower_wick_ratio", "upper_wick_ratio"] if c in base_feat.columns]
        base_feat = base_feat[["Date"] + features].copy().sort_values("Date").reset_index(drop=True)

        win, tgt = int(cfg.past_window_days), int(cfg.past_resample_len)
        target_idx = np.linspace(0, win - 1, tgt)
        z_rows, z_dates = [], []
        for i in range(len(base_feat)):
            if i + 1 < win:
                continue
            chunk = base_feat.iloc[i + 1 - win : i + 1]
            vec = []
            for c in features:
                s = chunk[c].to_numpy(dtype=float)
                vec.extend(np.interp(target_idx, np.arange(win), s).tolist())
            z_rows.append(vec)
            z_dates.append(base_feat.loc[i, "Date"])
        z = pd.DataFrame({"Date": z_dates})
        if z_rows:
            arr = np.asarray(z_rows, dtype=float)
            for i in range(arr.shape[1]):
                z[f"zp_{i:04d}"] = arr[:, i]
        X_df = z.merge(base_feat, on="Date", how="left").sort_values("Date").reset_index(drop=True)
        feat_cols = [c for c in X_df.columns if c != "Date"]
        X_df[feat_cols] = X_df[feat_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

        fit_end = (target_dt - pd.Timedelta(days=1)).normalize()
        fit_df = X_df[X_df["Date"] <= fit_end].copy()
        if int(cfg.fit_window_days) > 0 and len(fit_df) > int(cfg.fit_window_days):
            fit_df = fit_df.iloc[-int(cfg.fit_window_days) :].copy()
        if len(fit_df) < 40:
            raise RuntimeError(f"insufficient fit rows for phase2: {len(fit_df)}")

        scaler = RobustScaler() if cfg.robust_scaler else StandardScaler()
        x_train = scaler.fit_transform(fit_df[feat_cols].to_numpy(dtype=float))
        pca_n = max(1, min(int(cfg.pca_n_components), x_train.shape[0], x_train.shape[1]))
        pca = PCA(n_components=pca_n, random_state=42)
        x_train_p = pca.fit_transform(x_train)
        clusterer = HDBSCAN(min_cluster_size=int(cfg.hdbscan_min_cluster_size), min_samples=int(cfg.hdbscan_min_samples), prediction_data=True)
        clusterer.fit(x_train_p)
        labels_train = clusterer.labels_.astype(int)
        fit_df = fit_df.assign(regime_id=labels_train)

        def _predict_one(dt: pd.Timestamp) -> int:
            row = X_df[X_df["Date"] == dt]
            if row.empty:
                return -1
            x = row[feat_cols].to_numpy(dtype=float)
            x_s = scaler.transform(x)
            x_p = pca.transform(x_s)
            lab, _ = hdbscan_prediction.approximate_predict(clusterer, x_p)
            return int(lab[0]) if len(lab) else -1

        dminus1 = _predict_one(fit_end)
        d = _predict_one(target_dt)
        phase2_df = X_df[["Date"]].merge(fit_df[["Date", "regime_id"]], on="Date", how="left")
        phase2_df["regime_id"] = phase2_df["regime_id"].fillna(-1).astype(int)

        fit_info = {
            "fit_end_date": str(fit_end.date()),
            "fit_window_days": int(cfg.fit_window_days),
            "n_train_rows": int(len(fit_df)),
            "scaler": "RobustScaler" if cfg.robust_scaler else "StandardScaler",
            "pca_n_components": int(pca_n),
            "hdbscan_min_cluster_size": int(cfg.hdbscan_min_cluster_size),
            "hdbscan_min_samples": int(cfg.hdbscan_min_samples),
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
            "regime_id_final": decision.get("stage1", {}).get("regime_id_final", -1),
            "n_agents_inferred": len(decision.get("ratios", {}) or {}),
            "n_agents_final": len(decision.get("stage1", {}).get("allowed_agents_final", []) or []),
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
