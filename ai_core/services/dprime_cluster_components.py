from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from ai_core.clusterers.ticc_clusterer import TICCClusterer
from ai_core.utils.cluster_stats import compute_cluster_stats, derive_valid_and_rare_clusters

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

    _FEATURE_SET_CORE8: List[str] = [
        "ret_1",
        "ret_5",
        "ret_20",
        "ATR_norm",
        "gap_atr",
        "vol_log_ratio_20",
        "dev_z_25",
        "body_ratio",
    ]
    _FEATURE_SET_CALENDAR10: List[str] = [
        "ret_1",
        "ret_5",
        "ret_20",
        "ATR_norm",
        "gap_atr",
        "vol_log_ratio_20",
        "dev_z_25",
        "body_ratio",
        "per_cal_year365_sin",
        "per_cal_year365_cos",
    ]
    _FEATURE_SETS: Dict[str, List[str]] = {
        "core8": _FEATURE_SET_CORE8,
        "calendar10": _FEATURE_SET_CALENDAR10,
    }

    @classmethod
    def _select_ticc_features(cls, features: pd.DataFrame, feature_set: str = "calendar10") -> List[str]:
        selected = cls._FEATURE_SETS.get(str(feature_set))
        if selected is None:
            supported = sorted(cls._FEATURE_SETS.keys())
            raise ValueError(f"Unsupported TICC feature_set={feature_set!r}. Supported: {supported}")

        missing = [col for col in selected if col not in features.columns]
        if missing:
            raise ValueError(
                f"Missing required TICC feature columns for feature_set={feature_set}: {missing}"
            )
        return list(selected)

    @staticmethod
    def _distribution(labels: pd.Series) -> Dict[str, int]:
        return {str(int(k)): int(v) for k, v in labels.value_counts().sort_index().items()}

    def train(self, features: pd.DataFrame, cfg: ClusterRuntimeConfig, feature_set: str = "calendar10") -> Dict[str, object]:
        raw_k = int(cfg.cluster_raw_k)
        th_share = float(cfg.cluster_small_share_threshold)
        th_run = float(cfg.cluster_small_mean_run_threshold)
        k_eff_min = int(cfg.cluster_k_eff_min)

        ticc_feature_cols = self._select_ticc_features(features, feature_set=feature_set)
        ticc_features = features[ticc_feature_cols].copy()
        for col in ticc_feature_cols:
            ticc_features[col] = pd.to_numeric(ticc_features[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        x_train = ticc_features.to_numpy(dtype=float)

        raw_clusterer = TICCClusterer(
            num_clusters=raw_k,
            window_size=5,
            lambda_parameter=0.11,
            beta=600.0,
            max_iter=100,
            threshold=2e-5,
        )
        raw_id = pd.Series(raw_clusterer.fit_predict_train(x_train), index=features.index, dtype=int)
        raw_backend_diag = raw_clusterer.get_diagnostics()

        train_df = features[["Date"]].copy()
        train_df["cluster_id_raw20"] = raw_id.astype(int)
        raw_distinct = int(train_df["cluster_id_raw20"].nunique())
        if raw_distinct <= 1:
            raise RuntimeError(f"raw20 clustering collapsed to {raw_distinct} distinct cluster(s)")

        raw_stats = compute_cluster_stats(train_df["cluster_id_raw20"].astype(int).tolist())
        valid_clusters, small_clusters = derive_valid_and_rare_clusters(
            raw_stats,
            share_min=th_share,
            mean_run_min=th_run,
        )
        k_valid = int(len(valid_clusters))
        collapse_after_small_filter = bool(k_valid <= 1)
        collapse_reason = ""
        if collapse_after_small_filter:
            collapse_reason = (
                f"raw20 valid clusters collapsed to {k_valid} after small-cluster filtering; "
                "continuing with k_eff fallback for stable re-fit"
            )

        k_eff = int(min(raw_k, max(k_eff_min, k_valid)))
        if not (k_eff_min <= k_eff <= raw_k):
            raise RuntimeError(f"invalid k_eff={k_eff} derived from k_valid={k_valid} raw_k={raw_k}")

        stable_clusterer = TICCClusterer(
            num_clusters=k_eff,
            window_size=5,
            lambda_parameter=0.11,
            beta=600.0,
            max_iter=100,
            threshold=2e-5,
        )
        stable_id = pd.Series(stable_clusterer.fit_predict_train(x_train), index=features.index, dtype=int)
        stable_backend_diag = stable_clusterer.get_diagnostics()
        train_df["cluster_id_stable"] = stable_id.astype(int)
        stable_distinct = int(train_df["cluster_id_stable"].nunique())
        if stable_distinct <= 1:
            raise RuntimeError(f"stable clustering collapsed to {stable_distinct} distinct cluster(s)")

        stable_map: Dict[int, int] = {}
        for cid in sorted(train_df["cluster_id_raw20"].unique().tolist()):
            rows = train_df.loc[train_df["cluster_id_raw20"] == int(cid), "cluster_id_stable"]
            if rows.empty:
                continue
            stable_map[int(cid)] = int(rows.mode(dropna=False).iloc[0])
        if len(set(stable_map.values())) <= 1:
            raise RuntimeError("raw20->stable mapping collapsed to a single stable cluster")

        raw_stats_records = raw_stats.to_dict(orient="records")
        small_details = []
        for rec in raw_stats_records:
            cid = int(rec["cluster_id"])
            is_small = cid in set(small_clusters)
            reason = ""
            if is_small:
                reason = (
                    f"share={float(rec['share']):.6f} < {th_share:.6f} and "
                    f"mean_run={float(rec['mean_run']):.6f} < {th_run:.6f}"
                )
            small_details.append(
                {
                    "cluster_id": cid,
                    "count": int(rec["count"]),
                    "share": float(rec["share"]),
                    "mean_run": float(rec["mean_run"]),
                    "is_small": bool(is_small),
                    "reason": reason,
                }
            )

        return {
            "train_df": train_df,
            "ticc_feature_set_name": str(feature_set),
            "ticc_feature_cols": ticc_feature_cols,
            "ticc_feature_count": int(len(ticc_feature_cols)),
            "ticc_train_shape": [int(x_train.shape[0]), int(x_train.shape[1])],
            "small_clusters": small_clusters,
            "valid_clusters": valid_clusters,
            "k_valid": k_valid,
            "k_eff": int(k_eff),
            "small_filtering_collapse_detected": bool(collapse_after_small_filter),
            "small_filtering_collapse_reason": collapse_reason,
            "stable_map": stable_map,
            "raw_label_distribution": self._distribution(train_df["cluster_id_raw20"]),
            "stable_label_distribution": self._distribution(train_df["cluster_id_stable"]),
            "raw_stats": raw_stats_records,
            "small_cluster_details": small_details,
            "stable_distinct": stable_distinct,
            "status": "live",
            "note": "TICC backend active for monthly raw20 fitting and stable mapping.",
            "backend_resolved_name": str(raw_backend_diag.get("backend_resolved_name", "") or ""),
            "backend_entrypoint_name": str(raw_backend_diag.get("backend_entrypoint_name", "") or ""),
            "backend_entrypoint_kind": str(raw_backend_diag.get("backend_entrypoint_kind", "") or ""),
            "backend_signature": str(raw_backend_diag.get("backend_signature", "") or ""),
            "backend_api_candidates": list(raw_backend_diag.get("backend_api_candidates", []) or []),
            "backend_predict_methods": list(raw_backend_diag.get("backend_predict_methods", []) or []),
            "backend_methods": list(raw_backend_diag.get("backend_methods", []) or []),
            "x_original_shape": list(raw_backend_diag.get("x_original_shape", []) or []),
            "x_sent_shape": list(raw_backend_diag.get("x_sent_shape", []) or []),
            "input_was_squeezed_univariate": bool(raw_backend_diag.get("input_was_squeezed_univariate", False)),
            "raw_backend_diagnostics": raw_backend_diag,
            "stable_backend_diagnostics": stable_backend_diag,
        }


class ClusterDailyAssigner:
    """Assign daily raw20/stable labels from latest windows."""

    def assign(self, features: pd.DataFrame, monthly: Dict[str, object], cfg: ClusterRuntimeConfig) -> pd.DataFrame:
        out = features[["Date"]].copy()
        out["cluster_id_raw20"] = monthly["train_df"]["cluster_id_raw20"].astype(int).to_numpy()
        out["cluster_id_stable"] = monthly["train_df"]["cluster_id_stable"].astype(int).to_numpy()
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

        pre_save_raw_dist = {str(int(k)): int(v) for k, v in daily["cluster_id_raw20"].value_counts().sort_index().items()}
        pre_save_stable_dist = {str(int(k)): int(v) for k, v in daily["cluster_id_stable"].value_counts().sort_index().items()}
        daily.to_csv(assign_path, index=False)
        # legacy-compatible outputs under stepDprime/<mode>/ naming
        legacy_assign = self.stepd_dir / f"stepDprime_cluster_daily_assign_{self.symbol}.csv"
        daily.to_csv(legacy_assign, index=False)

        reloaded = pd.read_csv(assign_path)
        post_save_raw_dist = {str(int(k)): int(v) for k, v in reloaded["cluster_id_raw20"].value_counts().sort_index().items()}
        post_save_stable_dist = {str(int(k)): int(v) for k, v in reloaded["cluster_id_stable"].value_counts().sort_index().items()}
        if pre_save_raw_dist != post_save_raw_dist or pre_save_stable_dist != post_save_stable_dist:
            raise RuntimeError("cluster assignments distribution changed after save/reload")

        mapping_payload = {
            "mapping": monthly.get("stable_map", {}),
            "raw_label_distribution": monthly.get("raw_label_distribution", {}),
            "stable_label_distribution": monthly.get("stable_label_distribution", {}),
        }
        mapping_path.write_text(json.dumps(mapping_payload, indent=2, ensure_ascii=False), encoding="utf-8")

        (models_raw / f"raw20_model_meta_{self.symbol}.json").write_text(
            json.dumps(monthly.get("raw_backend_diagnostics", {}), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (models_stable / f"stable_model_meta_{self.symbol}.json").write_text(
            json.dumps(monthly.get("stable_backend_diagnostics", {}), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        feature_manifest = {
            "source": "StepA-only",
            "multi_window": {
                "short_days": int(cfg.cluster_short_window_days),
                "mid_weeks": int(cfg.cluster_mid_window_weeks),
                "long_months": int(cfg.cluster_long_window_months),
                "enable_8y_context": bool(cfg.cluster_enable_8y_context),
            },
            "ticc_feature_set_name": str(monthly.get("ticc_feature_set_name", "calendar10")),
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
            "ticc_feature_set_name": str(monthly.get("ticc_feature_set_name", "calendar10")),
            "ticc_feature_cols": list(monthly.get("ticc_feature_cols", [])),
            "ticc_feature_count": int(monthly.get("ticc_feature_count", 0)),
            "ticc_train_shape": list(monthly.get("ticc_train_shape", [])),
            "k_valid": int(monthly.get("k_valid", 0)),
            "k_eff": int(monthly.get("k_eff", 0)),
            "small_filtering_collapse_detected": bool(monthly.get("small_filtering_collapse_detected", False)),
            "small_filtering_collapse_reason": str(monthly.get("small_filtering_collapse_reason", "")),
            "small_clusters": list(monthly.get("small_clusters", [])),
            "raw_label_distribution": monthly.get("raw_label_distribution", {}),
            "stable_label_distribution": monthly.get("stable_label_distribution", {}),
            "save_roundtrip_distribution": {
                "pre_save": {
                    "cluster_id_raw20": pre_save_raw_dist,
                    "cluster_id_stable": pre_save_stable_dist,
                },
                "post_save_reload": {
                    "cluster_id_raw20": post_save_raw_dist,
                    "cluster_id_stable": post_save_stable_dist,
                },
            },
            "small_cluster_details": list(monthly.get("small_cluster_details", [])),
            "raw_cluster_stats": list(monthly.get("raw_stats", [])),
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
        print(f"[DPrimeCluster] ticc_feature_set_name={monthly.get('ticc_feature_set_name', 'calendar10')}")
        print(f"[DPrimeCluster] ticc_feature_cols={monthly.get('ticc_feature_cols', [])}")
        shape = tuple(monthly.get("ticc_train_shape", []))
        print(f"[DPrimeCluster] ticc_train_shape={shape}")
        print(f"[TICC] backend_entrypoint={monthly.get('backend_entrypoint_name', '')}")
        print(f"[TICC] backend_signature={monthly.get('backend_signature', '')}")
        print(f"[TICC] x_original_shape={tuple(monthly.get('x_original_shape', []))}")
        print(f"[TICC] x_sent_shape={tuple(monthly.get('x_sent_shape', []))}")
        print(
            "[TICC] input_was_squeezed_univariate="
            f"{str(bool(monthly.get('input_was_squeezed_univariate', False))).lower()}"
        )
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
        stable_distinct = int(daily["cluster_id_stable"].nunique()) if len(daily) else 0
        if stable_distinct <= 1:
            raise RuntimeError(f"stable daily assignment collapsed to {stable_distinct} distinct cluster(s)")
        print(f"[DPrimeCluster] stable training end k_eff={monthly.get('k_eff')}")
        print(f"[DPrimeCluster] raw_label_distribution={monthly.get('raw_label_distribution', {})}")
        print(f"[DPrimeCluster] stable_label_distribution={monthly.get('stable_label_distribution', {})}")

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
            "ticc_feature_set_name": str(monthly.get("ticc_feature_set_name", "calendar10")),
            "ticc_feature_cols": list(monthly.get("ticc_feature_cols", [])),
            "ticc_feature_count": int(monthly.get("ticc_feature_count", 0)),
            "ticc_train_shape": list(monthly.get("ticc_train_shape", [])),
            "k_valid": int(monthly.get("k_valid", 0)),
            "k_eff": int(monthly.get("k_eff", 0)),
            "small_filtering_collapse_detected": bool(monthly.get("small_filtering_collapse_detected", False)),
            "small_filtering_collapse_reason": str(monthly.get("small_filtering_collapse_reason", "")),
            "small_clusters": list(monthly.get("small_clusters", [])),
            "backend_resolved_name": str(monthly.get("backend_resolved_name", "") or ""),
            "backend_entrypoint_name": str(monthly.get("backend_entrypoint_name", "") or ""),
            "backend_entrypoint_kind": str(monthly.get("backend_entrypoint_kind", "") or ""),
            "backend_signature": str(monthly.get("backend_signature", "") or ""),
            "backend_api_candidates": list(monthly.get("backend_api_candidates", []) or []),
            "backend_predict_methods": list(monthly.get("backend_predict_methods", []) or []),
            "backend_methods": list(monthly.get("backend_methods", []) or []),
            "x_original_shape": list(monthly.get("x_original_shape", []) or []),
            "x_sent_shape": list(monthly.get("x_sent_shape", []) or []),
            "input_was_squeezed_univariate": bool(monthly.get("input_was_squeezed_univariate", False)),
            "outputs": paths,
        }
        return {"daily": daily, "summary": summary}
