from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from ai_core.clusterers.ticc_clusterer import TICCClusterer

import numpy as np
import pandas as pd


@dataclass
class ClusterRuntimeConfig:
    symbol: str
    mode: str = "sim"
    cluster_backend: str = "ticc"
    cluster_raw_k: int = 20
    cluster_k_eff_min: int = 12
    cluster_small_share_threshold: float = 0.01
    cluster_small_mean_run_threshold: float = 3.0
    cluster_short_window_days: int = 20
    cluster_mid_window_weeks: int = 8
    cluster_long_window_months: int = 6
    cluster_enable_8y_context: bool = True
    cluster_rare_flag_enabled: bool = True


class ClusterFeatureBuilder:
    """Build cluster features from StepA-derived inputs only.

    StepB/StepC predictions are explicitly out of scope to prevent leakage.
    """

    def build(self, data: pd.DataFrame, periodic: pd.DataFrame, cfg: ClusterRuntimeConfig) -> pd.DataFrame:
        cdf = data.copy()
        cdf["Date"] = pd.to_datetime(cdf["Date"], errors="coerce").dt.normalize()

        per = periodic.copy()
        if "Date" not in per.columns:
            per = per.reset_index().rename(columns={per.index.name or "index": "Date"})
        per["Date"] = pd.to_datetime(per["Date"], errors="coerce").dt.normalize()
        cdf = cdf.merge(per, on="Date", how="left", suffixes=("", "_periodic"))
        cdf = cdf.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)

        base = pd.to_numeric(cdf.get("bnf_score", 0.0), errors="coerce").fillna(0.0)
        cdf["cluster_short_signal"] = base.rolling(max(2, int(cfg.cluster_short_window_days)), min_periods=1).mean()
        cdf["cluster_mid_signal"] = base.rolling(max(2, int(cfg.cluster_mid_window_weeks) * 5), min_periods=1).mean()
        cdf["cluster_long_signal"] = base.rolling(max(2, int(cfg.cluster_long_window_months) * 21), min_periods=1).mean()

        if bool(cfg.cluster_enable_8y_context):
            span = 8 * 252
            rolling_high = pd.to_numeric(cdf.get("Close_anchor", 0.0), errors="coerce").rolling(span, min_periods=50).max()
            rolling_low = pd.to_numeric(cdf.get("Close_anchor", 0.0), errors="coerce").rolling(span, min_periods=50).min()
            close = pd.to_numeric(cdf.get("Close_anchor", 0.0), errors="coerce")
            range_8y = (rolling_high - rolling_low).replace(0, np.nan)
            cdf["ctx_8y_high_distance"] = (rolling_high - close) / range_8y
            cdf["ctx_8y_low_distance"] = (close - rolling_low) / range_8y
            cdf["ctx_8y_range_position"] = (close - rolling_low) / range_8y
            cdf["ctx_8y_drawdown"] = (close / rolling_high.replace(0, np.nan)) - 1.0
            cdf["ctx_8y_vol_percentile"] = pd.to_numeric(cdf.get("ATR_norm", 0.0), errors="coerce").rolling(span, min_periods=50).rank(pct=True)
            cdf["ctx_8y_return_rank"] = pd.to_numeric(cdf.get("ret_20", 0.0), errors="coerce").rolling(span, min_periods=50).rank(pct=True)
            cdf["ctx_8y_trend_strength_scalar"] = cdf["cluster_long_signal"].abs()

        for c in cdf.columns:
            if c != "Date" and pd.api.types.is_numeric_dtype(cdf[c]):
                cdf[c] = pd.to_numeric(cdf[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return cdf


class ClusterMonthlyTrainer:
    """Monthly refit facade using TICC backend for raw20 labels and stable map."""

    def train(self, features: pd.DataFrame, cfg: ClusterRuntimeConfig) -> Dict[str, object]:
        raw_k = int(cfg.cluster_raw_k)
        th_share = float(cfg.cluster_small_share_threshold)
        th_run = float(cfg.cluster_small_mean_run_threshold)
        k_eff_min = int(cfg.cluster_k_eff_min)

        ticc_feature_cols = ["ret_1"]
        if "ret_1" not in features.columns:
            raise ValueError("TICC required feature missing: ret_1")
        ticc_features = features[ticc_feature_cols].copy()
        ticc_features["ret_1"] = (
            pd.to_numeric(ticc_features["ret_1"], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        x_train = ticc_features.to_numpy(dtype=float)
        clusterer = TICCClusterer(
            num_clusters=max(2, raw_k),
            window_size=5,
            lambda_parameter=0.11,
            beta=600.0,
            max_iter=100,
            threshold=2e-5,
        )
        raw_id = pd.Series(clusterer.fit_predict_train(x_train), index=features.index, dtype=int)
        backend_diag = clusterer.get_diagnostics()

        train_df = features[["Date"]].copy()
        train_df["cluster_id_raw20"] = raw_id.astype(int)

        run_id = (train_df["cluster_id_raw20"] != train_df["cluster_id_raw20"].shift(1)).cumsum().rename("run_id")
        runs = train_df.groupby(["cluster_id_raw20", run_id], as_index=False).size().rename(columns={"size": "run_len"})
        mean_run = runs.groupby("cluster_id_raw20")["run_len"].mean()
        share = train_df["cluster_id_raw20"].value_counts(normalize=True).sort_index()

        small_clusters = [
            int(cid)
            for cid in share.index
            if float(share.loc[cid]) < th_share and float(mean_run.get(cid, 0.0)) < th_run
        ]
        valid_clusters = [int(cid) for cid in sorted(train_df["cluster_id_raw20"].unique().tolist()) if int(cid) not in set(small_clusters)]
        if not valid_clusters:
            valid_clusters = [0]
        k_eff = max(k_eff_min, len(valid_clusters))
        stable_map = {cid: i % k_eff for i, cid in enumerate(valid_clusters)}

        return {
            "train_df": train_df,
            "ticc_feature_cols": ticc_feature_cols,
            "ticc_feature_count": int(len(ticc_feature_cols)),
            "ticc_train_shape": [int(x_train.shape[0]), int(x_train.shape[1])],
            "small_clusters": small_clusters,
            "valid_clusters": valid_clusters,
            "k_valid": len(valid_clusters),
            "k_eff": int(k_eff),
            "stable_map": stable_map,
            "status": "live",
            "note": "TICC backend active for monthly raw20 fitting and stable mapping.",
            "backend_resolved_name": str(backend_diag.get("backend_resolved_name", "") or ""),
            "backend_entrypoint_name": str(backend_diag.get("backend_entrypoint_name", "") or ""),
            "backend_entrypoint_kind": str(backend_diag.get("backend_entrypoint_kind", "") or ""),
            "backend_api_candidates": list(backend_diag.get("backend_api_candidates", []) or []),
            "backend_predict_methods": list(backend_diag.get("backend_predict_methods", []) or []),
            "backend_methods": list(backend_diag.get("backend_methods", []) or []),
        }


class ClusterDailyAssigner:
    """Assign daily raw20/stable labels from latest windows."""

    def assign(self, features: pd.DataFrame, monthly: Dict[str, object], cfg: ClusterRuntimeConfig) -> pd.DataFrame:
        out = features[["Date"]].copy()
        out["cluster_id_raw20"] = monthly["train_df"]["cluster_id_raw20"].astype(int).to_numpy()
        stable_map = dict(monthly.get("stable_map", {}))
        fallback = int(next(iter(stable_map.values()))) if stable_map else 0
        out["cluster_id_stable"] = out["cluster_id_raw20"].map(lambda x: stable_map.get(int(x), fallback)).astype(int)
        out["rare_flag_raw20"] = out["cluster_id_raw20"].isin(set(monthly.get("small_clusters", []))).astype(int)
        out["year_month"] = pd.to_datetime(out["Date"], errors="coerce").dt.to_period("M").astype(str)
        if not bool(cfg.cluster_rare_flag_enabled):
            out["rare_flag_raw20"] = 0
        return out


class ClusterArtifactManager:
    """Manage cluster artifacts with canonical naming under stepDprime/cluster/<mode>/..."""

    def __init__(self, stepd_dir: Path, mode: str, symbol: str):
        self.stepd_dir = Path(stepd_dir)
        self.mode = mode
        self.symbol = symbol
        self.cluster_root = self.stepd_dir / "cluster" / mode

    def write(self, daily: pd.DataFrame, monthly: Dict[str, object], cfg: ClusterRuntimeConfig) -> Dict[str, str]:
        models_raw = self.cluster_root / "models" / "raw20"
        models_stable = self.cluster_root / "models" / "stable"
        models_raw.mkdir(parents=True, exist_ok=True)
        models_stable.mkdir(parents=True, exist_ok=True)

        assign_path = self.cluster_root / f"cluster_assignments_{self.symbol}.csv"
        summary_path = self.cluster_root / f"cluster_summary_{self.symbol}.json"
        mapping_path = self.cluster_root / f"cluster_mapping_raw20_to_stable_{self.symbol}.json"
        manifest_path = self.cluster_root / f"cluster_feature_manifest_{self.symbol}.json"

        daily.to_csv(assign_path, index=False)
        # legacy-compatible outputs under stepDprime/<mode>/ naming
        legacy_assign = self.stepd_dir / f"stepDprime_cluster_daily_assign_{self.symbol}.csv"
        daily.to_csv(legacy_assign, index=False)
        mapping_path.write_text(json.dumps({"mapping": monthly.get("stable_map", {})}, indent=2, ensure_ascii=False), encoding="utf-8")

        feature_manifest = {
            "source": "StepA-only",
            "multi_window": {
                "short_days": int(cfg.cluster_short_window_days),
                "mid_weeks": int(cfg.cluster_mid_window_weeks),
                "long_months": int(cfg.cluster_long_window_months),
                "enable_8y_context": bool(cfg.cluster_enable_8y_context),
            },
            "ticc_feature_cols": list(monthly.get("ticc_feature_cols", [])),
            "ticc_feature_count": int(monthly.get("ticc_feature_count", 0)),
            "ticc_train_shape": list(monthly.get("ticc_train_shape", [])),
            "status": "live",
        }
        manifest_path.write_text(json.dumps(feature_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        summary = {
            "symbol": self.symbol,
            "mode": self.mode,
            "cluster_backend": cfg.cluster_backend,
            "status": monthly.get("status", "live"),
            "note": monthly.get("note", ""),
            "k_raw": int(cfg.cluster_raw_k),
            "ticc_feature_cols": list(monthly.get("ticc_feature_cols", [])),
            "ticc_feature_count": int(monthly.get("ticc_feature_count", 0)),
            "ticc_train_shape": list(monthly.get("ticc_train_shape", [])),
            "k_valid": int(monthly.get("k_valid", 0)),
            "k_eff": int(monthly.get("k_eff", 0)),
            "small_clusters": list(monthly.get("small_clusters", [])),
            "paths": {
                "assignments": str(assign_path),
                "summary": str(summary_path),
                "mapping": str(mapping_path),
                "feature_manifest": str(manifest_path),
                "models_raw20": str(models_raw),
                "models_stable": str(models_stable),
            },
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        legacy_summary = self.stepd_dir / f"stepDprime_cluster_summary_{self.symbol}.json"
        legacy_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        summary["paths"]["legacy_daily_assign"] = str(legacy_assign)
        summary["paths"]["legacy_summary"] = str(legacy_summary)
        return summary["paths"]


class DPrimeClusterService:
    """Top-level API for DPrime cluster regime state generation."""

    def run(self, cfg: ClusterRuntimeConfig, data: pd.DataFrame, periodic: pd.DataFrame, stepd_dir: Path) -> Dict[str, object]:
        print(f"[DPrimeCluster] start backend={cfg.cluster_backend} mode={cfg.mode}")
        builder = ClusterFeatureBuilder()
        trainer = ClusterMonthlyTrainer()
        assigner = ClusterDailyAssigner()
        artifacts = ClusterArtifactManager(stepd_dir=stepd_dir, mode=cfg.mode, symbol=cfg.symbol)

        features = builder.build(data, periodic, cfg)
        print(f"[DPrimeCluster] feature_rows={len(features)}")
        print(f"[DPrimeCluster] unique_dates={int(features['Date'].nunique()) if 'Date' in features.columns else 0}")
        print(f"[DPrimeCluster] raw20 training start k_raw={cfg.cluster_raw_k}")
        monthly = trainer.train(features, cfg)
        print(f"[DPrimeCluster] ticc_feature_cols={monthly.get('ticc_feature_cols', [])}")
        shape = tuple(monthly.get("ticc_train_shape", []))
        print(f"[DPrimeCluster] ticc_train_shape={shape}")
        monthly_status = str(monthly.get("status", "")).strip().lower()
        monthly_note = str(monthly.get("note", ""))
        if monthly_status in {"placeholder", "not_wired", "planned"}:
            raise RuntimeError(
                "DPrime cluster path returned non-live status. "
                f"status={monthly_status} note={monthly_note}"
            )
        print(
            f"[DPrimeCluster] raw20 training end k_valid={monthly.get('k_valid')} small={monthly.get('small_clusters')}"
        )
        print(f"[DPrimeCluster] stable training start k_eff_min={cfg.cluster_k_eff_min}")
        daily = assigner.assign(features, monthly, cfg)
        print(f"[DPrimeCluster] stable training end k_eff={monthly.get('k_eff')}")

        paths = artifacts.write(daily=daily, monthly=monthly, cfg=cfg)
        print(f"[DPrimeCluster] assignments_written={paths.get('legacy_daily_assign', paths.get('assignments', ''))}")
        print(f"[DPrimeCluster] summary_written={paths.get('legacy_summary', paths.get('summary', ''))}")
        print(
            "[DPrimeCluster] end "
            f"cluster_id_raw20={int(daily['cluster_id_raw20'].iloc[-1]) if len(daily) else 0} "
            f"cluster_id_stable={int(daily['cluster_id_stable'].iloc[-1]) if len(daily) else 0} "
            f"rare_flag_raw20={int(daily['rare_flag_raw20'].iloc[-1]) if len(daily) else 0}"
        )
        print(f"[DPrimeCluster] artifact_paths={paths}")

        summary = {
            "cluster_backend": cfg.cluster_backend,
            "status": str(monthly.get("status", "live")),
            "note": str(monthly.get("note", "")),
            "k_raw": int(cfg.cluster_raw_k),
            "ticc_feature_cols": list(monthly.get("ticc_feature_cols", [])),
            "ticc_feature_count": int(monthly.get("ticc_feature_count", 0)),
            "ticc_train_shape": list(monthly.get("ticc_train_shape", [])),
            "k_valid": int(monthly.get("k_valid", 0)),
            "k_eff": int(monthly.get("k_eff", 0)),
            "small_clusters": list(monthly.get("small_clusters", [])),
            "backend_resolved_name": str(monthly.get("backend_resolved_name", "") or ""),
            "backend_entrypoint_name": str(monthly.get("backend_entrypoint_name", "") or ""),
            "backend_entrypoint_kind": str(monthly.get("backend_entrypoint_kind", "") or ""),
            "backend_api_candidates": list(monthly.get("backend_api_candidates", []) or []),
            "backend_predict_methods": list(monthly.get("backend_predict_methods", []) or []),
            "backend_methods": list(monthly.get("backend_methods", []) or []),
            "outputs": paths,
        }
        return {"daily": daily, "summary": summary}
