from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ai_core.services.step_dprime_service import _compute_base_features
from ai_core.utils.pipeline_artifact_utils import resolve_stepdprime_root

try:  # pragma: no cover - plotting is best-effort
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


@dataclass
class ClusterEvalPaths:
    output_root: Path
    mode: str
    symbol: str

    @property
    def out_dir(self) -> Path:
        return self.output_root / "audit" / "cluster_eval" / self.mode / self.symbol

    @property
    def stepd_root(self) -> Path:
        root, _, _ = resolve_stepdprime_root(self.output_root, self.mode)
        return root

    @property
    def cluster_root(self) -> Path:
        return self.stepd_root / "cluster" / self.mode


def _safe_read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    return df


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else str(value.date())
    return value


def _iqr(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return float("nan")
    return float(s.quantile(0.75) - s.quantile(0.25))


def _zscore_map(values: pd.Series) -> pd.Series:
    s = pd.to_numeric(values, errors="coerce")
    std = float(s.std(ddof=0)) if len(s) else 0.0
    if not np.isfinite(std) or std <= 1e-12:
        return pd.Series(0.0, index=s.index, dtype=float)
    return ((s - float(s.mean())) / std).astype(float)


def _max_drawdown_from_returns(ret: pd.Series) -> float:
    s = pd.to_numeric(ret, errors="coerce").dropna()
    if s.empty:
        return float("nan")
    eq = (1.0 + s).cumprod()
    dd = eq / eq.cummax() - 1.0
    return float(dd.min())


def _ret_metrics(ret: pd.Series) -> Dict[str, float]:
    s = pd.to_numeric(ret, errors="coerce").dropna()
    if s.empty:
        return {
            "count": 0.0,
            "mean_ret": float("nan"),
            "std_ret": float("nan"),
            "sharpe": float("nan"),
            "win_rate": float("nan"),
            "max_dd": float("nan"),
            "equity_multiple": float("nan"),
        }
    mean_ret = float(s.mean())
    std_ret = float(s.std(ddof=0))
    sharpe = float(mean_ret / std_ret * math.sqrt(252.0)) if std_ret > 1e-12 else float("nan")
    return {
        "count": float(len(s)),
        "mean_ret": mean_ret,
        "std_ret": std_ret,
        "sharpe": sharpe,
        "win_rate": float((s > 0).mean()),
        "max_dd": _max_drawdown_from_returns(s),
        "equity_multiple": float((1.0 + s).cumprod().iloc[-1]),
    }


def _normalize_numeric_frame(df: pd.DataFrame, keep: Sequence[str] = ("Date",)) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col in keep:
            continue
        if pd.api.types.is_numeric_dtype(out[col]) or out[col].dtype == object:
            out[col] = pd.to_numeric(out[col], errors="ignore")
    return out


def _run_lengths(labels: pd.Series) -> pd.DataFrame:
    s = pd.to_numeric(labels, errors="coerce")
    change = s.ne(s.shift()).fillna(True)
    run_id = change.cumsum()
    out = pd.DataFrame({"cluster_id": s, "run_id": run_id})
    out = out.dropna(subset=["cluster_id"]).copy()
    if out.empty:
        return pd.DataFrame(columns=["cluster_id", "run_len", "start_idx", "end_idx"])
    grp = out.groupby("run_id")
    runs = grp.agg(cluster_id=("cluster_id", "first"), run_len=("cluster_id", "size"))
    runs["start_idx"] = grp.indices and [int(min(v)) for v in grp.indices.values()] or []
    runs["end_idx"] = grp.indices and [int(max(v)) for v in grp.indices.values()] or []
    runs = runs.reset_index(drop=True)
    runs["cluster_id"] = runs["cluster_id"].astype(int)
    return runs


def _transition_entropy(labels: pd.Series, cid: int) -> float:
    s = pd.to_numeric(labels, errors="coerce").dropna().astype(int).reset_index(drop=True)
    if len(s) <= 1:
        return float("nan")
    mask = (s.iloc[:-1].to_numpy(dtype=int) == int(cid))
    nxt = s.iloc[1:].reset_index(drop=True).iloc[np.flatnonzero(mask)]
    if nxt.empty:
        return 0.0
    probs = nxt.value_counts(normalize=True).astype(float)
    vals = probs.to_numpy(dtype=float)
    return float(-(vals * np.log2(np.clip(vals, 1e-12, 1.0))).sum())


def _greedy_match(distance_df: pd.DataFrame) -> List[Dict[str, Any]]:
    if distance_df.empty:
        return []
    items: List[Tuple[int, int, float]] = []
    for prev_id in distance_df.index:
        for cur_id in distance_df.columns:
            items.append((int(prev_id), int(cur_id), float(distance_df.loc[prev_id, cur_id])))
    items.sort(key=lambda x: x[2])
    used_prev: set[int] = set()
    used_cur: set[int] = set()
    out: List[Dict[str, Any]] = []
    for prev_id, cur_id, dist in items:
        if prev_id in used_prev or cur_id in used_cur:
            continue
        used_prev.add(prev_id)
        used_cur.add(cur_id)
        out.append({"prev_cluster_id": prev_id, "curr_cluster_id": cur_id, "distance": dist})
    return out


def _pick_first(df: pd.DataFrame, candidates: Sequence[str], default: float = 0.0) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def _load_base_features(paths: ClusterEvalPaths) -> pd.DataFrame:
    base_path = paths.stepd_root / f"stepDprime_base_features_{paths.symbol}.csv"
    if base_path.exists():
        return _safe_read_csv(base_path)

    stepa_root = paths.output_root / "stepA" / paths.mode
    prices = pd.concat(
        [
            _safe_read_csv(stepa_root / f"stepA_prices_train_{paths.symbol}.csv"),
            _safe_read_csv(stepa_root / f"stepA_prices_test_{paths.symbol}.csv"),
        ],
        ignore_index=True,
    ).sort_values("Date").reset_index(drop=True)
    tech = pd.concat(
        [
            _safe_read_csv(stepa_root / f"stepA_tech_train_{paths.symbol}.csv"),
            _safe_read_csv(stepa_root / f"stepA_tech_test_{paths.symbol}.csv"),
        ],
        ignore_index=True,
    ).sort_values("Date").reset_index(drop=True)
    periodic = pd.concat(
        [
            _safe_read_csv(stepa_root / f"stepA_periodic_train_{paths.symbol}.csv"),
            _safe_read_csv(stepa_root / f"stepA_periodic_test_{paths.symbol}.csv"),
        ],
        ignore_index=True,
    ).sort_values("Date").reset_index(drop=True)
    base = _compute_base_features(prices, tech)
    all_df = base.merge(tech, on="Date", how="left", suffixes=("", "_tech")).merge(periodic, on="Date", how="left")
    if "Close_anchor" not in all_df.columns and "Close" in prices.columns:
        all_df["Close_anchor"] = pd.to_numeric(prices["Close"], errors="coerce")
    return all_df


def _build_eval_feature_frame(base_df: pd.DataFrame) -> pd.DataFrame:
    df = base_df.copy().sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    close = _pick_first(df, ["Close_anchor", "Close"], default=np.nan)
    volume = _pick_first(df, ["Volume", "volume"], default=np.nan)
    ret1 = _pick_first(df, ["ret_1"], default=np.nan)
    if ret1.isna().all():
        ret1 = close.pct_change(1)
    df["ret_1"] = ret1
    df["ret_5_eval"] = close.pct_change(5)
    df["ret_20_eval"] = close.pct_change(20)
    df["ret_60_eval"] = close.pct_change(60)
    ma20 = close.rolling(20, min_periods=5).mean()
    ma60 = close.rolling(60, min_periods=20).mean()
    df["ma_dev20"] = close / ma20.replace(0, np.nan) - 1.0
    df["ma_dev60"] = close / ma60.replace(0, np.nan) - 1.0
    df["ma_slope20"] = ma20.pct_change(5)
    df["ma_slope60"] = ma60.pct_change(10)
    df["vol20"] = ret1.rolling(20, min_periods=10).std(ddof=0) * math.sqrt(252.0)
    roll_high20 = close.rolling(20, min_periods=5).max()
    roll_high60 = close.rolling(60, min_periods=20).max()
    df["dd20"] = close / roll_high20.replace(0, np.nan) - 1.0
    df["dd60"] = close / roll_high60.replace(0, np.nan) - 1.0
    df["ATR_norm_eval"] = _pick_first(df, ["ATR_norm"], default=np.nan)
    df["panic_score_eval"] = _pick_first(df, ["BNF_PanicScore", "panic_score"], default=np.nan)
    df["energy_fade_eval"] = _pick_first(df, ["BNF_EnergyFade", "energy_fade"], default=np.nan)
    df["div_down_volup_eval"] = _pick_first(df, ["BNF_DivDownVolUp", "div_down_volup"], default=np.nan)
    df["volume_z_eval"] = (volume - volume.rolling(20, min_periods=10).mean()) / volume.rolling(20, min_periods=10).std(ddof=0).replace(0, np.nan)
    df["rvol20_eval"] = volume / volume.rolling(20, min_periods=10).mean().replace(0, np.nan)
    fwd1 = close.shift(-1) / close - 1.0
    fwd5 = close.shift(-5) / close - 1.0
    fwd5_vol = ret1.shift(-4).rolling(5, min_periods=3).std(ddof=0) * math.sqrt(252.0)
    future_window = pd.concat([close.shift(-i) for i in range(1, 6)], axis=1)
    future_min = future_window.min(axis=1)
    df["fwd_ret_1d"] = fwd1
    df["fwd_ret_5d"] = fwd5
    df["fwd_realized_vol_5d"] = fwd5_vol
    df["fwd_max_dd_5d"] = future_min / close.replace(0, np.nan) - 1.0
    return df


def _profile_table(merged: pd.DataFrame, cluster_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = [
        "ret_5_eval", "ret_20_eval", "ret_60_eval", "ma_dev20", "ma_dev60", "ma_slope20", "ma_slope60",
        "vol20", "ATR_norm_eval", "dd20", "dd60", "panic_score_eval", "energy_fade_eval",
        "div_down_volup_eval", "volume_z_eval", "rvol20_eval",
    ]
    rows: List[Dict[str, Any]] = []
    for cid, sub in merged.groupby(cluster_col, dropna=True):
        row: Dict[str, Any] = {"cluster_id": int(cid), "count": int(len(sub))}
        for col in feature_cols:
            s = pd.to_numeric(sub.get(col), errors="coerce")
            row[f"{col}_median"] = float(s.median()) if s.notna().any() else float("nan")
            row[f"{col}_iqr"] = _iqr(s)
            row[f"{col}_std"] = float(s.std(ddof=0)) if s.notna().any() else float("nan")
        rows.append(row)
    profile_df = pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True) if rows else pd.DataFrame(columns=["cluster_id", "count"])
    if profile_df.empty:
        return profile_df, pd.DataFrame(columns=["cluster_id", "label_candidate", "label_reason"])

    for raw_col, z_col in [
        ("ret_20_eval_median", "z_ret20"),
        ("ret_60_eval_median", "z_ret60"),
        ("ma_slope20_median", "z_ma20"),
        ("vol20_median", "z_vol20"),
        ("dd20_median", "z_dd20"),
        ("panic_score_eval_median", "z_panic"),
        ("volume_z_eval_median", "z_volume"),
        ("ma_dev20_median", "z_madev20"),
    ]:
        profile_df[z_col] = _zscore_map(profile_df[raw_col])

    profile_df["trend_score"] = profile_df["z_ret20"] + profile_df["z_ret60"] + profile_df["z_ma20"] - profile_df["z_dd20"] - 0.5 * profile_df["z_vol20"]
    profile_df["vol_score"] = profile_df["z_vol20"] + 0.5 * profile_df["z_volume"]
    profile_df["drawdown_score"] = -profile_df["z_dd20"]
    profile_df["panic_score"] = profile_df["z_panic"] + 0.5 * profile_df["z_volume"]
    score_cols = ["trend_score", "vol_score", "drawdown_score", "panic_score"]
    profile_df["semantic_separability_score"] = np.sqrt((profile_df[score_cols].astype(float) ** 2).sum(axis=1))

    labels: List[Dict[str, Any]] = []
    for row in profile_df.itertuples(index=False):
        reasons = {
            "trend": float(getattr(row, "trend_score", 0.0)),
            "vol": float(getattr(row, "vol_score", 0.0)),
            "drawdown": float(getattr(row, "drawdown_score", 0.0)),
            "panic": float(getattr(row, "panic_score", 0.0)),
            "dd20": float(getattr(row, "dd20_median", float("nan"))),
            "ret5": float(getattr(row, "ret_5_eval_median", float("nan"))),
            "ret20": float(getattr(row, "ret_20_eval_median", float("nan"))),
            "ret60": float(getattr(row, "ret_60_eval_median", float("nan"))),
            "vol20": float(getattr(row, "vol20_median", float("nan"))),
            "panic_med": float(getattr(row, "panic_score_eval_median", float("nan"))),
            "volume_z": float(getattr(row, "volume_z_eval_median", float("nan"))),
        }
        label = "mixed / unclear"
        if reasons["panic"] > 1.0 and reasons["vol"] > 0.75:
            label = "panic/volume spike"
        elif reasons["trend"] > 1.0 and reasons["vol"] < 0.5:
            label = "上昇トレンド"
        elif reasons["trend"] < -1.0 and reasons["vol"] >= 0.0:
            label = "下降トレンド"
        elif reasons["vol"] > 1.0 and reasons["ret20"] < 0:
            label = "高ボラ下落"
        elif reasons["vol"] < -0.5 and abs(reasons["trend"]) < 0.75:
            label = "低ボラ揉み合い"
        elif reasons["dd20"] < -0.08 and reasons["ret5"] > 0 and reasons["trend"] > 0:
            label = "深いDD後の反発初期"
        labels.append(
            {
                "cluster_id": int(row.cluster_id),
                "label_candidate": label,
                "label_reason": ", ".join(f"{k}={v:.4f}" for k, v in reasons.items() if np.isfinite(v)),
            }
        )
    return profile_df, pd.DataFrame(labels)


def _run_stats(merged: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    labels = pd.to_numeric(merged[cluster_col], errors="coerce").dropna().astype(int)
    total = len(labels)
    runs = _run_lengths(labels)
    rows: List[Dict[str, Any]] = []
    for cid in sorted(labels.unique().tolist()):
        sub_runs = runs[runs["cluster_id"] == int(cid)].copy()
        occ = int((labels == int(cid)).sum())
        exits = int(((labels == int(cid)).shift(0) & (labels.shift(-1) != int(cid))).fillna(False).sum()) if total else 0
        rows.append(
            {
                "cluster_source": cluster_col,
                "cluster_id": int(cid),
                "count": occ,
                "share": float(occ / total) if total else float("nan"),
                "mean_run": float(sub_runs["run_len"].mean()) if not sub_runs.empty else float("nan"),
                "median_run": float(sub_runs["run_len"].median()) if not sub_runs.empty else float("nan"),
                "switch_rate": float(exits / occ) if occ else float("nan"),
                "transition_entropy": _transition_entropy(labels, int(cid)),
                "run_count": int(len(sub_runs)),
            }
        )
    return pd.DataFrame(rows)


def _monthly_stability(merged: pd.DataFrame, cluster_col: str, label_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    work = merged[["Date", cluster_col, "ret_20_eval", "ret_60_eval", "vol20", "dd20", "panic_score_eval", "volume_z_eval"]].copy()
    work["year_month"] = pd.to_datetime(work["Date"], errors="coerce").dt.to_period("M").astype(str)
    label_map = {int(r.cluster_id): str(r.label_candidate) for r in label_df.itertuples(index=False)} if not label_df.empty else {}
    months = [m for m in work["year_month"].dropna().unique().tolist() if m]
    months = sorted(months)
    rows: List[Dict[str, Any]] = []
    heatmap: Dict[str, Any] = {}
    feat_cols = ["ret_20_eval", "ret_60_eval", "vol20", "dd20", "panic_score_eval", "volume_z_eval"]
    for prev_month, cur_month in zip(months[:-1], months[1:]):
        prev_df = work[work["year_month"] == prev_month]
        cur_df = work[work["year_month"] == cur_month]
        prev_prof = prev_df.groupby(cluster_col)[feat_cols].median().dropna(how="all")
        cur_prof = cur_df.groupby(cluster_col)[feat_cols].median().dropna(how="all")
        if prev_prof.empty or cur_prof.empty:
            continue
        prev_prof = prev_prof.fillna(0.0)
        cur_prof = cur_prof.fillna(0.0)
        dist = pd.DataFrame(index=prev_prof.index.astype(int), columns=cur_prof.index.astype(int), dtype=float)
        for pid in dist.index:
            for cid in dist.columns:
                dist.loc[pid, cid] = float(np.linalg.norm(prev_prof.loc[pid].to_numpy(dtype=float) - cur_prof.loc[cid].to_numpy(dtype=float)))
        matches = _greedy_match(dist)
        preserve = []
        for item in matches:
            prev_label = label_map.get(int(item["prev_cluster_id"]), "")
            cur_label = label_map.get(int(item["curr_cluster_id"]), "")
            keep = prev_label == cur_label and prev_label != ""
            preserve.append(1.0 if keep else 0.0)
            rows.append(
                {
                    "cluster_source": cluster_col,
                    "prev_month": prev_month,
                    "curr_month": cur_month,
                    "prev_cluster_id": int(item["prev_cluster_id"]),
                    "curr_cluster_id": int(item["curr_cluster_id"]),
                    "distance": float(item["distance"]),
                    "prev_label_candidate": prev_label,
                    "curr_label_candidate": cur_label,
                    "label_preserved": int(keep),
                }
            )
        heatmap[f"{prev_month}->{cur_month}"] = {
            "distance_matrix": {str(r): {str(c): float(dist.loc[r, c]) for c in dist.columns} for r in dist.index},
            "best_match": matches,
            "label_maintenance_rate": float(np.mean(preserve)) if preserve else float("nan"),
        }
    stability_df = pd.DataFrame(rows)
    return stability_df, heatmap


def _load_step_e_logs(output_root: Path, mode: str, symbol: str) -> List[Tuple[str, pd.DataFrame]]:
    root = output_root / "stepE" / mode
    logs: List[Tuple[str, pd.DataFrame]] = []
    for path in sorted(root.glob(f"stepE_daily_log_*_{symbol}.csv")):
        name = path.name[len("stepE_daily_log_") : -len(f"_{symbol}.csv")]
        df = _safe_read_csv(path)
        logs.append((name, df))
    return logs


def _step_e_effect(cluster_df: pd.DataFrame, output_root: Path, mode: str, symbol: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    agent_logs = _load_step_e_logs(output_root, mode, symbol)
    for agent, df in agent_logs:
        if "Date" not in df.columns or "ret" not in df.columns:
            continue
        merged = cluster_df.merge(df[[c for c in df.columns if c in {"Date", "Split", "ret", "ratio", "equity"}]], on="Date", how="inner")
        if merged.empty:
            continue
        test_mask = merged.get("Split", pd.Series("test", index=merged.index)).astype(str).str.lower().eq("test")
        merged = merged.loc[test_mask].copy()
        for cid, sub in merged.groupby("cluster_id_stable"):
            metrics = _ret_metrics(sub["ret"])
            rows.append(
                {
                    "cluster_id": int(cid),
                    "expert": agent,
                    **metrics,
                    "mean_ratio": float(pd.to_numeric(sub.get("ratio"), errors="coerce").mean()) if "ratio" in sub.columns else float("nan"),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    spread = out.groupby("cluster_id")["mean_ret"].agg(["max", "min"]).reset_index()
    spread["expert_mean_ret_spread"] = spread["max"] - spread["min"]
    out = out.merge(spread[["cluster_id", "expert_mean_ret_spread"]], on="cluster_id", how="left")
    return out.sort_values(["cluster_id", "expert"]).reset_index(drop=True)


def _load_stepf_router(output_root: Path, mode: str, symbol: str) -> pd.DataFrame:
    path = output_root / "stepF" / mode / f"stepF_daily_log_router_{symbol}.csv"
    return _safe_read_csv(path) if path.exists() else pd.DataFrame()


def _load_stepf_equity(output_root: Path, mode: str, symbol: str) -> pd.DataFrame:
    path = output_root / "stepF" / mode / f"stepF_equity_marl_{symbol}.csv"
    return _safe_read_csv(path) if path.exists() else pd.DataFrame()


def _stepf_effect(cluster_df: pd.DataFrame, output_root: Path, mode: str, symbol: str, step_e_effect: pd.DataFrame) -> pd.DataFrame:
    router = _load_stepf_router(output_root, mode, symbol)
    equity = _load_stepf_equity(output_root, mode, symbol)
    if router.empty and equity.empty:
        return pd.DataFrame()
    df = router.copy() if not router.empty else equity.copy()
    add_cols = [c for c in ["Date", "ret", "equity", "ratio", "Split", "turnover", "selected_expert"] if c in df.columns]
    if not equity.empty:
        eq_cols = [c for c in ["Date", "ret", "equity", "ratio", "Split", "turnover"] if c in equity.columns and c not in add_cols]
        if eq_cols:
            df = df.merge(equity[["Date"] + eq_cols], on="Date", how="outer")
    merged = cluster_df.merge(df, on="Date", how="inner", suffixes=("", "_stepf"))
    if "cluster_id_stable_stepf" in merged.columns and "cluster_id_stable" not in merged.columns:
        merged["cluster_id_stable"] = pd.to_numeric(merged["cluster_id_stable_stepf"], errors="coerce")
    if merged.empty:
        return pd.DataFrame()
    test_mask = merged.get("Split", pd.Series("test", index=merged.index)).astype(str).str.lower().eq("test")
    merged = merged.loc[test_mask].copy()
    if merged.empty:
        return pd.DataFrame()

    expert_baseline: Dict[int, Tuple[str, float]] = {}
    if not step_e_effect.empty:
        tmp = step_e_effect.sort_values(["cluster_id", "mean_ret"], ascending=[True, False]).drop_duplicates("cluster_id")
        expert_baseline = {int(r.cluster_id): (str(r.expert), float(r.mean_ret)) for r in tmp.itertuples(index=False)}

    rows: List[Dict[str, Any]] = []
    for cid, sub in merged.groupby("cluster_id_stable"):
        metrics = _ret_metrics(sub["ret"] if "ret" in sub.columns else pd.Series(dtype=float))
        chosen = sub["selected_expert"].value_counts(dropna=False).to_dict() if "selected_expert" in sub.columns else {}
        top_expert, top_count = (next(iter(chosen.items())) if chosen else ("", 0))
        baseline_name, baseline_mean_ret = expert_baseline.get(int(cid), ("", float("nan")))
        rows.append(
            {
                "cluster_id": int(cid),
                **metrics,
                "final_ratio_mean": float(pd.to_numeric(sub.get("ratio"), errors="coerce").mean()) if "ratio" in sub.columns else float("nan"),
                "final_ratio_std": float(pd.to_numeric(sub.get("ratio"), errors="coerce").std(ddof=0)) if "ratio" in sub.columns else float("nan"),
                "turnover_mean": float(pd.to_numeric(sub.get("turnover"), errors="coerce").mean()) if "turnover" in sub.columns else float("nan"),
                "selected_expert_top1": str(top_expert),
                "selected_expert_top1_share": float(top_count / len(sub)) if len(sub) else float("nan"),
                "selected_expert_distribution": json.dumps({str(k): int(v) for k, v in chosen.items()}, ensure_ascii=False),
                "baseline_best_expert": baseline_name,
                "baseline_best_expert_mean_ret": baseline_mean_ret,
                "delta_vs_best_expert_mean_ret": float(metrics["mean_ret"] - baseline_mean_ret) if np.isfinite(baseline_mean_ret) and np.isfinite(metrics["mean_ret"]) else float("nan"),
            }
        )
    out = pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)
    if not out.empty:
        out["downstream_utility_score"] = (
            _zscore_map(out["sharpe"].fillna(0.0))
            - 0.5 * _zscore_map((-out["max_dd"].fillna(0.0)).fillna(0.0))
            - 0.25 * _zscore_map(out["turnover_mean"].fillna(0.0))
            + 0.5 * _zscore_map(out["delta_vs_best_expert_mean_ret"].fillna(0.0))
        )
    return out


def _rare_eval(merged: pd.DataFrame, stepf_effect_df: pd.DataFrame, output_root: Path, mode: str, symbol: str) -> pd.DataFrame:
    router = _load_stepf_router(output_root, mode, symbol)
    joined = merged.copy()
    if not router.empty:
        use_cols = [c for c in ["Date", "ratio", "selected_expert", "Split"] if c in router.columns]
        joined = joined.merge(router[use_cols], on="Date", how="left", suffixes=("", "_stepf"))
    if "Split" in joined.columns:
        joined = joined[joined["Split"].astype(str).str.lower().eq("test") | joined["Split"].isna()].copy()
    joined["stable_raw_disagree"] = (pd.to_numeric(joined["cluster_id_stable"], errors="coerce") != pd.to_numeric(joined["cluster_id_raw20"], errors="coerce")).astype(int)

    rows: List[Dict[str, Any]] = []
    for name, sub in {
        "rare_flag_raw20=1": joined[joined["rare_flag_raw20"].astype(int) == 1],
        "rare_flag_raw20=0": joined[joined["rare_flag_raw20"].astype(int) == 0],
        "stable_raw_disagree=1": joined[joined["stable_raw_disagree"] == 1],
    }.items():
        metrics = _ret_metrics(sub["fwd_ret_1d"] if "fwd_ret_1d" in sub.columns else pd.Series(dtype=float))
        rows.append(
            {
                "segment": name,
                "days": int(len(sub)),
                "next_day_ret_mean": float(pd.to_numeric(sub.get("fwd_ret_1d"), errors="coerce").mean()) if not sub.empty else float("nan"),
                "forward_5d_cumret_mean": float(pd.to_numeric(sub.get("fwd_ret_5d"), errors="coerce").mean()) if not sub.empty else float("nan"),
                "forward_5d_maxdd_mean": float(pd.to_numeric(sub.get("fwd_max_dd_5d"), errors="coerce").mean()) if not sub.empty else float("nan"),
                "forward_5d_realized_vol_mean": float(pd.to_numeric(sub.get("fwd_realized_vol_5d"), errors="coerce").mean()) if not sub.empty else float("nan"),
                "current_dd20_mean": float(pd.to_numeric(sub.get("dd20"), errors="coerce").mean()) if not sub.empty else float("nan"),
                "current_vol20_mean": float(pd.to_numeric(sub.get("vol20"), errors="coerce").mean()) if not sub.empty else float("nan"),
                "stepf_abs_ratio_mean": float(pd.to_numeric(sub.get("ratio"), errors="coerce").abs().mean()) if "ratio" in sub.columns and not sub.empty else float("nan"),
                "selected_expert_distribution": json.dumps({str(k): int(v) for k, v in sub.get("selected_expert", pd.Series(dtype=object)).value_counts(dropna=False).to_dict().items()}, ensure_ascii=False),
                **{f"oracle_proxy_{k}": v for k, v in metrics.items()},
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        rare_row = out[out["segment"] == "rare_flag_raw20=1"]
        base_row = out[out["segment"] == "rare_flag_raw20=0"]
        if not rare_row.empty and not base_row.empty:
            rare_usefulness = (
                float(rare_row["forward_5d_realized_vol_mean"].iloc[0] - base_row["forward_5d_realized_vol_mean"].iloc[0])
                - float(rare_row["forward_5d_maxdd_mean"].iloc[0] - base_row["forward_5d_maxdd_mean"].iloc[0])
            )
            out["rare_usefulness_score"] = float(rare_usefulness)
        else:
            out["rare_usefulness_score"] = float("nan")
    return out


def _write_plot_box(profile_df: pd.DataFrame, out_path: Path) -> Optional[str]:
    if plt is None or profile_df.empty:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plot_cols = ["ret_20_eval_median", "ret_60_eval_median", "vol20_median", "dd20_median"]
    titles = ["ret20", "ret60", "vol20", "dd20"]
    for ax, col, title in zip(axes.ravel(), plot_cols, titles):
        ax.bar(profile_df["cluster_id"].astype(str), pd.to_numeric(profile_df[col], errors="coerce").fillna(0.0))
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return str(out_path)


def _write_plot_runs(run_df: pd.DataFrame, out_path: Path) -> Optional[str]:
    if plt is None or run_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(run_df["cluster_id"].astype(str), pd.to_numeric(run_df["mean_run"], errors="coerce").fillna(0.0), label="mean_run")
    ax.plot(run_df["cluster_id"].astype(str), pd.to_numeric(run_df["median_run"], errors="coerce").fillna(0.0), color="tab:orange", marker="o", label="median_run")
    ax.legend()
    ax.set_title("cluster run length")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return str(out_path)


def _write_plot_heatmap(matrix: Dict[str, Any], out_path: Path) -> Optional[str]:
    if plt is None or not matrix:
        return None
    key = sorted(matrix.keys())[0]
    dist = matrix[key].get("distance_matrix", {})
    if not dist:
        return None
    rows = sorted(dist.keys(), key=lambda x: int(x))
    cols = sorted({c for row in dist.values() for c in row.keys()}, key=lambda x: int(x))
    arr = np.array([[float(dist[r].get(c, np.nan)) for c in cols] for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(arr, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.set_title(f"monthly profile similarity {key}")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return str(out_path)


def _write_plot_expert_heatmap(effect_df: pd.DataFrame, out_path: Path) -> Optional[str]:
    if plt is None or effect_df.empty:
        return None
    pivot = effect_df.pivot_table(index="cluster_id", columns="expert", values="mean_ret", aggfunc="mean")
    if pivot.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.fillna(0.0).to_numpy(dtype=float), aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)), [str(x) for x in pivot.index])
    ax.set_title("cluster x best expert")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return str(out_path)


def _write_plot_rare_compare(rare_df: pd.DataFrame, out_path: Path) -> Optional[str]:
    if plt is None or rare_df.empty:
        return None
    sub = rare_df[rare_df["segment"].isin(["rare_flag_raw20=1", "rare_flag_raw20=0"])].copy()
    if sub.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = sub["segment"].tolist()
    vals1 = pd.to_numeric(sub["forward_5d_maxdd_mean"], errors="coerce").fillna(0.0).tolist()
    vals2 = pd.to_numeric(sub["forward_5d_realized_vol_mean"], errors="coerce").fillna(0.0).tolist()
    x = np.arange(len(labels))
    ax.bar(x - 0.15, vals1, width=0.3, label="fwd5d_maxdd")
    ax.bar(x + 0.15, vals2, width=0.3, label="fwd5d_vol")
    ax.set_xticks(x, labels, rotation=10)
    ax.legend()
    ax.set_title("rare vs non-rare forward risk")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return str(out_path)


def run_cluster_evaluation(*, output_root: str | Path, mode: str, symbol: str) -> Dict[str, Any]:
    paths = ClusterEvalPaths(output_root=Path(output_root), mode=str(mode), symbol=str(symbol))
    paths.out_dir.mkdir(parents=True, exist_ok=True)

    cluster_assign_path = paths.cluster_root / f"cluster_assignments_{symbol}.csv"
    if not cluster_assign_path.exists():
        cluster_assign_path = paths.stepd_root / f"stepDprime_cluster_daily_assign_{symbol}.csv"
    if not cluster_assign_path.exists():
        return {
            "status": "SKIP",
            "summary": "cluster assignments not found",
            "out_dir": str(paths.out_dir),
        }

    cluster_summary_path = paths.cluster_root / f"cluster_summary_{symbol}.json"
    cluster_summary = _read_json(cluster_summary_path) if cluster_summary_path.exists() else {}
    cluster_df = _safe_read_csv(cluster_assign_path)
    base_df = _load_base_features(paths)
    feat_df = _build_eval_feature_frame(base_df)
    merged = feat_df.merge(cluster_df[["Date", "cluster_id_stable", "cluster_id_raw20", "rare_flag_raw20"]], on="Date", how="inner")
    if merged.empty:
        return {
            "status": "WARN",
            "summary": "cluster eval merge is empty",
            "out_dir": str(paths.out_dir),
        }

    stable_profile, stable_labels = _profile_table(merged, "cluster_id_stable")
    raw20_profile, raw20_labels = _profile_table(merged, "cluster_id_raw20")
    stable_runs = _run_stats(merged, "cluster_id_stable")
    raw20_runs = _run_stats(merged, "cluster_id_raw20")
    stable_monthly, stable_heat = _monthly_stability(merged, "cluster_id_stable", stable_labels)
    raw20_monthly, raw20_heat = _monthly_stability(merged, "cluster_id_raw20", raw20_labels)
    step_e_effect = _step_e_effect(merged[["Date", "cluster_id_stable", "cluster_id_raw20", "rare_flag_raw20"]], paths.output_root, paths.mode, paths.symbol)
    stepf_effect = _stepf_effect(merged[["Date", "cluster_id_stable", "cluster_id_raw20", "rare_flag_raw20"]], paths.output_root, paths.mode, paths.symbol, step_e_effect)
    rare_eval = _rare_eval(merged, stepf_effect, paths.output_root, paths.mode, paths.symbol)

    run_stats_df = pd.concat([stable_runs, raw20_runs], ignore_index=True)
    monthly_df = pd.concat([stable_monthly, raw20_monthly], ignore_index=True)
    label_df = pd.concat(
        [stable_labels.assign(cluster_source="stable"), raw20_labels.assign(cluster_source="raw20")],
        ignore_index=True,
    )

    stable_total = stable_profile[["cluster_id", "semantic_separability_score"]].merge(
        stable_runs[["cluster_id", "mean_run", "median_run", "transition_entropy"]], on="cluster_id", how="left"
    )
    if not stable_total.empty:
        stable_total["temporal_stability_score"] = (
            _zscore_map(stable_total["mean_run"].fillna(0.0))
            + 0.5 * _zscore_map(stable_total["median_run"].fillna(0.0))
            - 0.5 * _zscore_map(stable_total["transition_entropy"].fillna(0.0))
        )
        if not stepf_effect.empty:
            stable_total = stable_total.merge(stepf_effect[["cluster_id", "downstream_utility_score"]], on="cluster_id", how="left")
        else:
            stable_total["downstream_utility_score"] = float("nan")
        stable_total["overall_score"] = stable_total[["semantic_separability_score", "temporal_stability_score", "downstream_utility_score"]].fillna(0.0).mean(axis=1)
    raw20_total = raw20_profile[["cluster_id", "semantic_separability_score"]].merge(
        raw20_runs[["cluster_id", "mean_run", "transition_entropy"]], on="cluster_id", how="left"
    )
    if not raw20_total.empty:
        raw20_total["temporal_stability_score"] = _zscore_map(raw20_total["mean_run"].fillna(0.0)) - 0.5 * _zscore_map(raw20_total["transition_entropy"].fillna(0.0))
        raw20_total["rare_usefulness_score"] = float(pd.to_numeric(rare_eval.get("rare_usefulness_score"), errors="coerce").dropna().iloc[0]) if "rare_usefulness_score" in rare_eval.columns and rare_eval["rare_usefulness_score"].notna().any() else float("nan")
        raw20_total["overall_score"] = raw20_total[["temporal_stability_score", "rare_usefulness_score"]].fillna(0.0).mean(axis=1)

    stable_profile_out = stable_profile.merge(stable_total[[c for c in stable_total.columns if c != "semantic_separability_score"]], on="cluster_id", how="left") if not stable_profile.empty else stable_profile
    raw20_profile_out = raw20_profile.merge(raw20_total[[c for c in raw20_total.columns if c != "semantic_separability_score"]], on="cluster_id", how="left") if not raw20_profile.empty else raw20_profile

    stable_profile_out.to_csv(paths.out_dir / "cluster_profile_stable.csv", index=False)
    raw20_profile_out.to_csv(paths.out_dir / "cluster_profile_raw20.csv", index=False)
    (paths.out_dir / "cluster_profile_stable.json").write_text(json.dumps(_json_safe(stable_profile_out.to_dict(orient="records")), ensure_ascii=False, indent=2), encoding="utf-8")
    (paths.out_dir / "cluster_profile_raw20.json").write_text(json.dumps(_json_safe(raw20_profile_out.to_dict(orient="records")), ensure_ascii=False, indent=2), encoding="utf-8")
    (paths.out_dir / "cluster_label_candidates.json").write_text(json.dumps(_json_safe(label_df.to_dict(orient="records")), ensure_ascii=False, indent=2), encoding="utf-8")
    run_stats_df.to_csv(paths.out_dir / "cluster_run_stats.csv", index=False)
    monthly_df.to_csv(paths.out_dir / "cluster_monthly_stability.csv", index=False)
    (paths.out_dir / "cluster_monthly_stability.json").write_text(json.dumps(_json_safe({"stable": stable_heat, "raw20": raw20_heat, "rows": monthly_df.to_dict(orient="records")}), ensure_ascii=False, indent=2), encoding="utf-8")
    step_e_effect.to_csv(paths.out_dir / "cluster_expert_effect.csv", index=False)
    stepf_effect.to_csv(paths.out_dir / "cluster_stepf_effect.csv", index=False)
    rare_eval.to_csv(paths.out_dir / "cluster_rare_eval_raw20.csv", index=False)
    (paths.out_dir / "cluster_rare_eval_raw20.json").write_text(json.dumps(_json_safe(rare_eval.to_dict(orient="records")), ensure_ascii=False, indent=2), encoding="utf-8")

    plots = {
        "cluster_boxplot": _write_plot_box(stable_profile_out, paths.out_dir / "cluster_boxplot_stable.png"),
        "cluster_run_length": _write_plot_runs(stable_runs, paths.out_dir / "cluster_run_stats_stable.png"),
        "monthly_profile_similarity": _write_plot_heatmap(stable_heat, paths.out_dir / "cluster_monthly_similarity_stable.png"),
        "cluster_best_expert": _write_plot_expert_heatmap(step_e_effect, paths.out_dir / "cluster_expert_heatmap.png"),
        "rare_flag_compare": _write_plot_rare_compare(rare_eval, paths.out_dir / "cluster_rare_forward_compare.png"),
    }
    plots = {k: v for k, v in plots.items() if v}

    stable_best = stable_profile_out.sort_values("overall_score", ascending=False).head(3) if not stable_profile_out.empty else pd.DataFrame()
    raw20_rare = raw20_profile_out.sort_values("overall_score", ascending=False).head(3) if not raw20_profile_out.empty else pd.DataFrame()
    step_e_best = step_e_effect.sort_values(["cluster_id", "mean_ret"], ascending=[True, False]).drop_duplicates("cluster_id") if not step_e_effect.empty else pd.DataFrame()

    summary_payload = {
        "status": "OK",
        "summary": "cluster evaluation artifacts generated",
        "out_dir": str(paths.out_dir),
        "stable_top_clusters": _json_safe(stable_best[[c for c in ["cluster_id", "overall_score", "label_candidate"] if c in stable_best.columns]].to_dict(orient="records") if not stable_best.empty else []),
        "raw20_top_clusters": _json_safe(raw20_rare[[c for c in ["cluster_id", "overall_score", "label_candidate"] if c in raw20_rare.columns]].to_dict(orient="records") if not raw20_rare.empty else []),
        "stepE_best_expert_by_cluster": _json_safe(step_e_best[[c for c in ["cluster_id", "expert", "mean_ret", "sharpe"] if c in step_e_best.columns]].to_dict(orient="records") if not step_e_best.empty else []),
        "rare_eval": _json_safe(rare_eval.to_dict(orient="records")),
        "small_clusters": cluster_summary.get("small_clusters", []),
        "plots": plots,
        "artifacts": {
            "cluster_profile_stable_csv": "cluster_profile_stable.csv",
            "cluster_profile_raw20_csv": "cluster_profile_raw20.csv",
            "cluster_label_candidates_json": "cluster_label_candidates.json",
            "cluster_run_stats_csv": "cluster_run_stats.csv",
            "cluster_monthly_stability_csv": "cluster_monthly_stability.csv",
            "cluster_expert_effect_csv": "cluster_expert_effect.csv",
            "cluster_stepf_effect_csv": "cluster_stepf_effect.csv",
            "cluster_rare_eval_raw20_csv": "cluster_rare_eval_raw20.csv",
            "cluster_eval_summary_md": "cluster_eval_summary.md",
            "cluster_eval_summary_json": "cluster_eval_summary.json",
        },
    }

    md_lines = [
        "# Cluster Evaluation Summary",
        "",
        f"- symbol: `{symbol}`",
        f"- mode: `{mode}`",
        f"- stable main source: `cluster_id_stable`",
        f"- raw20 auxiliary source: `cluster_id_raw20` + `rare_flag_raw20`",
        "",
        "## Highlights",
        f"- stable top clusters: {json.dumps(summary_payload['stable_top_clusters'], ensure_ascii=False)}",
        f"- raw20 top clusters: {json.dumps(summary_payload['raw20_top_clusters'], ensure_ascii=False)}",
        f"- stepE best expert by cluster: {json.dumps(summary_payload['stepE_best_expert_by_cluster'], ensure_ascii=False)}",
        f"- rare eval: {json.dumps(summary_payload['rare_eval'], ensure_ascii=False)}",
        "",
        "## Plots",
    ]
    for name, path in plots.items():
        md_lines.append(f"- {name}: `{Path(path).name}`")
    (paths.out_dir / "cluster_eval_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    (paths.out_dir / "cluster_eval_summary.json").write_text(json.dumps(_json_safe(summary_payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_payload
