#!/usr/bin/env python3
"""Best-effort evaluator for StepA/StepB/D'/StepE/StepF outputs.

Design goals:
- Never raise uncaught exceptions (workflow-safe).
- Always exit code 0.
- Write detailed markdown/json plus compact summary for issue posting.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import traceback
from typing import Any

import importlib
import importlib.util

import numpy as np
import pandas as pd

from ai_core.utils.cluster_eval import run_cluster_evaluation

_MPL_SPEC = importlib.util.find_spec("matplotlib")
MPL_AVAILABLE = _MPL_SPEC is not None
if MPL_AVAILABLE:
    matplotlib = importlib.import_module("matplotlib")
    matplotlib.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")
else:
    matplotlib = None
    plt = None

MAX_SUMMARY_LINES = 200
MAX_SUMMARY_CHARS = 15000
MAX_LIST_ITEMS = 12


def _find_first(pattern: str) -> str | None:
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else None


def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _parse_date(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    if date_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    return out




def _canonical_eval_roots(output_root: str, mode: str, symbol: str) -> list[str]:
    roots = [output_root]
    try:
        from tools.run_manifest import resolve_canonical_output_root

        norm = os.path.normpath(output_root)
        parts = norm.split(os.sep)
        test_start = ""
        if len(parts) >= 1:
            tail = parts[-1]
            if re.match(r"^\d{4}-\d{2}-\d{2}$", tail):
                test_start = tail
        canonical = str(resolve_canonical_output_root(mode, symbol, test_start or "unknown_test_start"))
        if canonical not in roots:
            roots.append(canonical)
    except Exception:
        pass
    return roots
def _to_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        f = float(v)
        if np.isnan(f):
            return None
        return f
    except Exception:
        return None


def _to_int(v: Any) -> int | None:
    f = _to_float(v)
    return int(f) if f is not None else None


def _fmt(v: Any, nd: int = 4) -> str:
    if v is None:
        return "NA"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _write_eval_tables(report: dict[str, Any], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    stepa = report.get("stepA", {})
    stepa_details = stepa.get("details", {}) if isinstance(stepa, dict) else {}
    stepa_row = {
        "status": stepa.get("status", "SKIP"),
        "summary": stepa.get("summary", "NA"),
        "test_rows": stepa_details.get("test_rows"),
        "test_date_start": stepa_details.get("test_date_start"),
        "test_date_end": stepa_details.get("test_date_end"),
        "missing_ohlcv_count": stepa_details.get("missing_ohlcv_count"),
        "path": stepa_details.get("path"),
    }
    pd.DataFrame([stepa_row]).to_csv(os.path.join(out_dir, "EVAL_TABLE_stepA.csv"), index=False)

    stepe_rows = report.get("stepE", {}).get("rows", []) if isinstance(report.get("stepE", {}), dict) else []
    pd.DataFrame(stepe_rows if stepe_rows else [{"status": report.get("stepE", {}).get("status", "SKIP"), "summary": report.get("stepE", {}).get("summary", "NA")}]).to_csv(
        os.path.join(out_dir, "EVAL_TABLE_stepE.csv"), index=False
    )

    stepf_rows = report.get("stepF", {}).get("rows", []) if isinstance(report.get("stepF", {}), dict) else []
    pd.DataFrame(stepf_rows if stepf_rows else [{"status": report.get("stepF", {}).get("status", "SKIP"), "summary": report.get("stepF", {}).get("summary", "NA")}]).to_csv(
        os.path.join(out_dir, "EVAL_TABLE_stepF.csv"), index=False
    )

    stepf_compare = report.get("stepF_compare", {}) if isinstance(report.get("stepF_compare", {}), dict) else {}
    stepf_compare_row = stepf_compare.get("row", {}) if isinstance(stepf_compare.get("row", {}), dict) else {}
    if stepf_compare_row:
        pd.DataFrame([stepf_compare_row]).to_csv(os.path.join(out_dir, "EVAL_TABLE_stepF_compare.csv"), index=False)
    else:
        pd.DataFrame([{"status": stepf_compare.get("status", "WARN"), "summary": stepf_compare.get("summary", "compare skipped"), "reason": stepf_compare.get("reason", "NA")}]).to_csv(
            os.path.join(out_dir, "EVAL_TABLE_stepF_compare.csv"), index=False
        )

    reward_cmp_rows = stepf_compare.get("stepF_reward_compare", {}).get("rows", []) if isinstance(stepf_compare.get("stepF_reward_compare", {}), dict) else []
    pd.DataFrame(reward_cmp_rows if reward_cmp_rows else [{"status": stepf_compare.get("stepF_reward_compare", {}).get("status", "WARN"), "summary": stepf_compare.get("stepF_reward_compare", {}).get("summary", "reward comparison unavailable")}]).to_csv(
        os.path.join(out_dir, "EVAL_TABLE_stepF_reward_compare.csv"), index=False
    )

    cluster_rows = stepf_compare.get("cluster", {}).get("rows", []) if isinstance(stepf_compare.get("cluster", {}), dict) else []
    pd.DataFrame(cluster_rows if cluster_rows else [{"status": stepf_compare.get("cluster", {}).get("status", "PENDING"), "reason": stepf_compare.get("cluster", {}).get("reason", "cluster comparison pending")}]).to_csv(
        os.path.join(out_dir, "EVAL_TABLE_stepF_compare_cluster.csv"), index=False
    )

    dprime = report.get("dprime", {}) if isinstance(report.get("dprime", {}), dict) else {}
    ddet = dprime.get("details", {}) if isinstance(dprime.get("details", {}), dict) else {}
    pd.DataFrame(
        [
            {
                "dprime_status": dprime.get("status", "SKIP"),
                "dprime_summary": dprime.get("summary", "NA"),
                "dprime_cluster_status": ddet.get("cluster_status"),
                "dprime_cluster_summary": ddet.get("cluster_summary"),
                "dprime_cluster_embeddings_count": ddet.get("cluster_embeddings_count", 0),
                "dprime_cluster_state_count": ddet.get("cluster_state_count", 0),
                "dprime_cluster_input_count": ddet.get("cluster_input_count", 0),
                "dprime_rl_status": ddet.get("rl_status"),
                "dprime_rl_summary": ddet.get("rl_summary"),
                "dprime_rl_state_count": ddet.get("rl_state_count", 0),
                "dprime_rl_profiles_count": ddet.get("rl_profiles_count", 0),
            }
        ]
    ).to_csv(os.path.join(out_dir, "EVAL_TABLE_dprime.csv"), index=False)


def _status_level(s: str) -> int:
    return {"OK": 0, "WARN": 1, "BAD": 2}.get(s, 1)


def _calc_metrics(true_s: pd.Series, pred_s: pd.Series) -> dict[str, Any]:
    pair = pd.DataFrame({"t": pd.to_numeric(true_s, errors="coerce"), "p": pd.to_numeric(pred_s, errors="coerce")}).dropna()
    if pair.empty:
        return {"mae": None, "corr": None, "n_eval": 0}

    err = pair["p"] - pair["t"]
    corr = pair["t"].corr(pair["p"]) if len(pair) >= 2 else None
    return {
        "mae": float(np.abs(err).mean()),
        "corr": (None if pd.isna(corr) else float(corr)),
        "n_eval": int(len(pair)),
    }


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _load_split_summary(output_root: str) -> dict[str, Any] | None:
    p = os.path.join(output_root, "split_summary.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else None
    except Exception:
        return None


def _resolve_split_series(
    df: pd.DataFrame,
    split_col: str | None,
    *,
    output_root: str,
    date_col: str | None = "Date",
) -> tuple[pd.Series, dict[str, Any], str | None]:
    work = df.copy()
    split_col_present = split_col is not None
    note: str | None = None
    source = "csv"

    if split_col_present:
        split_values = work[split_col].astype(str).str.strip().str.lower()  # type: ignore[index]
    else:
        split_values = pd.Series(["unknown"] * len(work), index=work.index, dtype="object")
        summary = _load_split_summary(output_root)
        date_series = None
        if date_col and date_col in work.columns:
            date_series = pd.to_datetime(work[date_col], errors="coerce")

        test_start = None
        if isinstance(summary, dict):
            ts = str(summary.get("test_start", "")).strip()
            if ts:
                parsed = pd.to_datetime(ts, errors="coerce")
                if pd.notna(parsed):
                    test_start = parsed.normalize()

        if test_start is not None and date_series is not None and date_series.notna().any():
            source = "summary"
            mask = date_series.dt.normalize() >= test_start
            split_values = pd.Series(np.where(mask, "test", "train"), index=work.index)
        elif date_series is not None and date_series.notna().any():
            source = "date_infer"
            norm = date_series.dt.normalize()
            uniq = sorted(norm.dropna().unique())
            if uniq:
                cutoff = uniq[-1]
                c = 1
                for i in range(len(uniq) - 2, -1, -1):
                    if (uniq[i + 1] - uniq[i]).days <= 3:
                        cutoff = uniq[i]
                        c += 1
                    else:
                        break
                if c <= 1 and len(uniq) > 1:
                    cutoff = uniq[max(0, len(uniq) // 2)]
                mask = norm >= cutoff
                split_values = pd.Series(np.where(mask, "test", "train"), index=work.index)
        else:
            source = "fallback_all_test"
            note = "Split missing: evaluated all rows as test"
            split_values = pd.Series(["test"] * len(work), index=work.index)

    train_rows = int((split_values == "train").sum())
    test_rows = int((split_values == "test").sum())
    note_emitted = bool(note)

    print(f"[STEPF_SPLIT] source={source}")
    print(f"[STEPF_SPLIT] split_col_present={str(split_col_present).lower()}")
    print(f"[STEPF_SPLIT] train_rows={train_rows}")
    print(f"[STEPF_SPLIT] test_rows={test_rows}")
    print(f"[STEPF_SPLIT] note_emitted={str(note_emitted).lower()}")

    meta = {
        "split_source": source,
        "split_col_present": split_col_present,
        "train_rows": train_rows,
        "test_rows": test_rows,
    }
    return split_values, meta, note


def _calc_equity_metrics(
    df: pd.DataFrame,
    equity_col: str,
    ret_col: str,
    split_col: str | None,
    *,
    output_root: str,
    date_col: str | None = "Date",
) -> tuple[dict[str, Any], str | None]:
    work = df.copy()
    split_values, split_meta, split_note = _resolve_split_series(work, split_col, output_root=output_root, date_col=date_col)
    df_test = work[split_values == "test"]

    if df_test.empty:
        return {
            "test_days": 0,
            "equity_multiple": None,
            "max_dd": None,
            "mean_ret": None,
            "std_ret": None,
            "sharpe": None,
            **split_meta,
        }, "no test rows after split filter"

    eq = pd.to_numeric(df_test[equity_col], errors="coerce")
    rets = pd.to_numeric(df_test[ret_col], errors="coerce")
    eq = eq[np.isfinite(eq)]
    rets = rets[np.isfinite(rets)]
    if len(eq) < 1:
        return {
            "test_days": int(len(df_test)),
            "equity_multiple": None,
            "max_dd": None,
            "mean_ret": None,
            "std_ret": None,
            "sharpe": None,
            **split_meta,
        }, "equity has no numeric rows"

    eq_start, eq_end = float(eq.iloc[0]), float(eq.iloc[-1])
    eq_multiple = None if eq_start == 0 else (eq_end / eq_start)

    peak = eq.cummax()
    dd = eq / peak - 1.0
    max_dd = float(dd.min()) if len(dd) else None

    mean_ret = float(rets.mean()) if len(rets) >= 1 else None
    std_ret = float(rets.std(ddof=1)) if len(rets) >= 2 else None

    sharpe = None
    sharpe_reason = None
    if len(rets) >= 2:
        if std_ret is not None and std_ret > 0:
            sharpe = float((mean_ret or 0.0) / std_ret * np.sqrt(252.0))
        else:
            sharpe_reason = "sharpe NA: ret std is 0"
    else:
        sharpe_reason = "sharpe NA: ret rows < 2"

    reason_out = split_note
    if sharpe_reason:
        reason_out = f"{reason_out}; {sharpe_reason}" if reason_out else sharpe_reason

    return {
        "test_days": int(len(df_test)),
        "equity_multiple": _to_float(eq_multiple),
        "max_dd": _to_float(max_dd),
        "mean_ret": _to_float(mean_ret),
        "std_ret": _to_float(std_ret),
        "sharpe": _to_float(sharpe),
        **split_meta,
    }, reason_out


def _calc_diversity(pos_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(pos_rows) < 2:
        return {
            "status": "SKIP",
            "summary": "need >=2 agents with numeric test position rows",
            "max_corr": None,
            "max_match_ratio": None,
            "pairs_over_0_9999": 0,
            "all_pairs": 0,
            "identical_all_agents": False,
        }

    pair_corrs: list[float] = []
    pair_match_ratios: list[float] = []
    over_09999 = 0
    valid_pairs = 0
    all_corr_one = True
    all_match_one = True

    for i in range(len(pos_rows)):
        for j in range(i + 1, len(pos_rows)):
            left = pos_rows[i]
            right = pos_rows[j]
            merged = left["series"].to_frame("a").join(right["series"].to_frame("b"), how="inner").dropna()
            if merged.empty:
                continue
            valid_pairs += 1
            if len(merged) < 2:
                corr = 1.0 if (merged["a"] == merged["b"]).all() else 0.0
            else:
                std_a = float(merged["a"].std(ddof=0))
                std_b = float(merged["b"].std(ddof=0))
                if std_a == 0.0 or std_b == 0.0:
                    corr = 1.0 if (merged["a"] == merged["b"]).all() else 0.0
                else:
                    corr = merged["a"].corr(merged["b"])
                    if corr is None or pd.isna(corr):
                        corr = 1.0 if (merged["a"] == merged["b"]).all() else 0.0
            corr = float(corr)
            pair_corrs.append(corr)
            if corr > 0.9999:
                over_09999 += 1
            if corr < 1.0:
                all_corr_one = False

            match_ratio = float((merged["a"] == merged["b"]).mean())
            pair_match_ratios.append(match_ratio)
            if match_ratio < 1.0:
                all_match_one = False

    if valid_pairs == 0:
        return {
            "status": "SKIP",
            "summary": "no overlapping numeric test position rows across agents",
            "max_corr": None,
            "max_match_ratio": None,
            "pairs_over_0_9999": 0,
            "all_pairs": 0,
            "identical_all_agents": False,
        }

    identical_all = all_corr_one and all_match_one
    high_corr_ratio = over_09999 / valid_pairs
    status = "OK"
    summary = "agent positions look diverse"
    if identical_all:
        status = "BAD"
        summary = "all agent test positions are identical"
    elif over_09999 >= 3 or high_corr_ratio >= 0.5:
        status = "WARN"
        summary = "many agent pairs are near-identical (corr>0.9999)"

    return {
        "status": status,
        "summary": summary,
        "max_corr": _to_float(max(pair_corrs) if pair_corrs else None),
        "max_match_ratio": _to_float(max(pair_match_ratios) if pair_match_ratios else None),
        "pairs_over_0_9999": int(over_09999),
        "all_pairs": int(valid_pairs),
        "identical_all_agents": bool(identical_all),
    }


def _metrics_from_ret_series(ret_s: pd.Series) -> dict[str, Any]:
    rets = pd.to_numeric(ret_s, errors="coerce").dropna().reset_index(drop=True)
    if rets.empty:
        return {
            "test_days": 0,
            "equity_multiple": None,
            "max_dd": None,
            "mean_ret": None,
            "std_ret": None,
            "sharpe": None,
        }
    eq = (1.0 + rets).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    mean_ret = float(rets.mean()) if len(rets) >= 1 else None
    std_ret = float(rets.std(ddof=1)) if len(rets) >= 2 else None
    sharpe = None
    if std_ret is not None and std_ret > 0:
        sharpe = float((mean_ret or 0.0) / std_ret * np.sqrt(252.0))
    return {
        "test_days": int(len(rets)),
        "equity_multiple": _to_float(float(eq.iloc[-1])),
        "max_dd": _to_float(float(dd.min())),
        "mean_ret": _to_float(mean_ret),
        "std_ret": _to_float(std_ret),
        "sharpe": _to_float(sharpe),
    }


def _build_test_ret_frame(df: pd.DataFrame, output_root: str, *, date_col: str = "Date") -> pd.DataFrame:
    work = df.copy()
    if date_col not in work.columns:
        return pd.DataFrame(columns=["Date", "ret"])
    ret_col = _pick_col(work, ["ret"])
    if ret_col is None:
        eq_col = _pick_col(work, ["equity", "Equity"])
        if eq_col is None:
            return pd.DataFrame(columns=["Date", "ret"])
        work["_ret_eval"] = pd.to_numeric(work[eq_col], errors="coerce").pct_change()
        ret_col = "_ret_eval"
    split_col = _pick_col(work, ["Split"])
    split_values, _, _ = _resolve_split_series(work, split_col, output_root=output_root, date_col=date_col)
    test_df = work[split_values == "test"].copy()
    test_df["Date"] = pd.to_datetime(test_df[date_col], errors="coerce").dt.normalize()
    test_df["ret"] = pd.to_numeric(test_df[ret_col], errors="coerce")
    extra_cols = [c for c in ["cluster_id_stable", "cluster_id_raw20", "rare_flag_raw20"] if c in test_df.columns]
    cols = ["Date", "ret", *extra_cols]
    return test_df[cols].dropna(subset=["Date", "ret"]).sort_values("Date").reset_index(drop=True)


def _load_stepf_mode_artifacts(output_root: str, mode: str, symbol: str, reward_mode: str) -> tuple[str, str] | None:
    base = os.path.join(output_root, "stepF", mode, f"reward_{reward_mode}")
    eq = os.path.join(base, f"stepF_equity_marl_{symbol}.csv")
    router = os.path.join(base, f"stepF_daily_log_router_{symbol}.csv")
    if os.path.exists(eq) and os.path.exists(router):
        return eq, router
    return None


def _mode_compare_row(name: str, test_df: pd.DataFrame, fixed_by_date: dict[Any, float], oracle_df: pd.DataFrame, pick_match_rate: float | None) -> dict[str, Any]:
    metrics = _metrics_from_ret_series(test_df["ret"] if "ret" in test_df.columns else pd.Series(dtype=float))
    cur_by_date = {d: float(r) for d, r in zip(test_df.get("Date", pd.Series(dtype='datetime64[ns]')), test_df.get("ret", pd.Series(dtype=float))) if np.isfinite(r)}
    win_days = 0
    common_days = 0
    for d, fxd in fixed_by_date.items():
        cur = cur_by_date.get(d)
        if cur is None:
            continue
        common_days += 1
        if cur > fxd:
            win_days += 1
    fixed_eq = _metrics_from_ret_series(pd.Series(list(fixed_by_date.values()), dtype=float)).get("equity_multiple")
    oracle_map = {d: float(r) for d, r in zip(oracle_df["Date"], oracle_df["oracle_ret"]) if np.isfinite(r)}
    cur_eq = metrics.get("equity_multiple")
    oracle_eq = _metrics_from_ret_series(oracle_df["oracle_ret"]).get("equity_multiple")
    return {
        "name": name,
        "equity_multiple": _to_float(cur_eq),
        "sharpe": _to_float(metrics.get("sharpe")),
        "max_dd": _to_float(metrics.get("max_dd")),
        "mean_ret": _to_float(metrics.get("mean_ret")),
        "std_ret": _to_float(metrics.get("std_ret")),
        "regret_vs_fixed_best": _to_float((fixed_eq - cur_eq) if fixed_eq is not None and cur_eq is not None else None),
        "regret_vs_oracle": _to_float((oracle_eq - cur_eq) if oracle_eq is not None and cur_eq is not None else None),
        "win_days_vs_fixed_best": _to_int(win_days),
        "pick_match_rate_vs_oracle": _to_float(pick_match_rate),
        "common_days_vs_fixed_best": _to_int(common_days),
    }


def _collect_stepf_compare(output_root: str, mode: str, symbol: str, report: dict[str, Any]) -> dict[str, Any]:
    stepf_rows = report.get("stepF", {}).get("rows", []) if isinstance(report.get("stepF", {}), dict) else []
    stepe_rows = report.get("stepE", {}).get("rows", []) if isinstance(report.get("stepE", {}), dict) else []
    current_row = next((r for r in stepf_rows if r.get("status") == "OK" and r.get("equity_multiple") is not None), None)
    ranked_stepe = [
        r
        for r in stepe_rows
        if r.get("status") == "OK" and r.get("equity_multiple") is not None and r.get("agent") and r.get("file")
    ]
    if current_row is None:
        return {"status": "WARN", "summary": "compare skipped", "reason": "current StepF row with numeric equity_multiple not found"}
    if len(ranked_stepe) < 1:
        return {"status": "WARN", "summary": "compare skipped", "reason": "StepE daily logs are missing or non-numeric"}

    ranked_stepe = sorted(
        ranked_stepe,
        key=lambda r: (float(r.get("equity_multiple") or -np.inf), float(r.get("sharpe") or -np.inf)),
        reverse=True,
    )
    fixed_best = ranked_stepe[0]

    expert_frames: dict[str, pd.DataFrame] = {}
    step_e_dir = os.path.join(output_root, "stepE", mode)
    for row in ranked_stepe:
        fpath = os.path.join(step_e_dir, str(row.get("file")))
        if not os.path.exists(fpath):
            continue
        try:
            expert_frames[str(row.get("agent"))] = _build_test_ret_frame(_read_csv(fpath), output_root)
        except Exception:
            continue
    if len(expert_frames) < 1:
        return {"status": "WARN", "summary": "compare skipped", "reason": "failed to load StepE test return frames"}

    all_dates = sorted({d for df in expert_frames.values() for d in pd.to_datetime(df["Date"], errors="coerce").dropna().tolist()})
    if not all_dates:
        return {"status": "WARN", "summary": "compare skipped", "reason": "StepE test date alignment is empty"}

    oracle_records: list[dict[str, Any]] = []
    for d in all_dates:
        picks: list[tuple[str, float]] = []
        for ag, df in expert_frames.items():
            s = df.loc[df["Date"] == d, "ret"]
            if not s.empty:
                rv = float(s.iloc[0])
                if np.isfinite(rv):
                    picks.append((ag, rv))
        if not picks:
            continue
        picks = sorted(picks, key=lambda x: x[1], reverse=True)
        oracle_records.append({"Date": d, "oracle_expert": picks[0][0], "oracle_ret": picks[0][1]})
    if not oracle_records:
        return {"status": "WARN", "summary": "compare skipped", "reason": "oracle series could not be constructed"}
    oracle_df = pd.DataFrame(oracle_records).sort_values("Date").reset_index(drop=True)
    oracle_metrics = _metrics_from_ret_series(oracle_df["oracle_ret"])

    stepf_eq_path = os.path.join(output_root, "stepF", mode, f"stepF_equity_marl_{symbol}.csv")
    stepf_test = _build_test_ret_frame(_read_csv(stepf_eq_path), output_root)
    current_by_date = {d: float(r) for d, r in zip(stepf_test["Date"], stepf_test["ret"]) if np.isfinite(r)}

    fixed_df = expert_frames.get(str(fixed_best.get("agent")), pd.DataFrame(columns=["Date", "ret"]))
    fixed_by_date = {d: float(r) for d, r in zip(fixed_df["Date"], fixed_df["ret"]) if np.isfinite(r)}

    by_date_rows: list[dict[str, Any]] = []
    cum_regret = 0.0
    win_days = 0
    common_days = 0
    for d in oracle_df["Date"].tolist():
        cur = current_by_date.get(d)
        fxd = fixed_by_date.get(d)
        orc = float(oracle_df.loc[oracle_df["Date"] == d, "oracle_ret"].iloc[0])
        if cur is not None:
            cum_regret += float(orc - cur)
        if cur is not None and fxd is not None:
            common_days += 1
            if cur > fxd:
                win_days += 1
        by_date_rows.append(
            {
                "Date": str(pd.Timestamp(d).date()),
                "current_stepf_ret": _to_float(cur),
                "fixed_best_ret": _to_float(fxd),
                "oracle_ret": _to_float(orc),
                "oracle_expert": str(oracle_df.loc[oracle_df["Date"] == d, "oracle_expert"].iloc[0]),
                "cumulative_regret_vs_oracle": _to_float(cum_regret),
            }
        )

    oracle_counts = oracle_df["oracle_expert"].value_counts().to_dict()

    # Optional StepF picked expert match-rate (via router weights)
    pick_match_rate = None
    stepf_selected_by_name: dict[str, int] = {}
    try:
        router_path = os.path.join(output_root, "stepF", mode, f"stepF_daily_log_router_{symbol}.csv")
        if os.path.exists(router_path):
            rdf = _read_csv(router_path)
            rdf_test = _build_test_ret_frame(rdf, output_root)
            rdf_test = rdf_test[["Date"]].copy()
            wcols = [c for c in rdf.columns if c.startswith("w_")]
            if wcols:
                rwork = rdf.copy()
                split_col = _pick_col(rwork, ["Split"])
                split_values, _, _ = _resolve_split_series(rwork, split_col, output_root=output_root, date_col="Date")
                rwork = rwork[split_values == "test"].copy()
                rwork["Date"] = pd.to_datetime(rwork["Date"], errors="coerce").dt.normalize()
                for c in wcols:
                    rwork[c] = pd.to_numeric(rwork[c], errors="coerce")
                rwork = rwork.dropna(subset=["Date"])
                if not rwork.empty:
                    argmax_cols = rwork[wcols].idxmax(axis=1)
                    rwork["stepf_expert"] = argmax_cols.str.replace("w_", "", regex=False)
                    stepf_selected_by_name = rwork["stepf_expert"].value_counts().astype(int).to_dict()
                    merged_pick = oracle_df[["Date", "oracle_expert"]].merge(rwork[["Date", "stepf_expert"]], on="Date", how="inner")
                    if not merged_pick.empty:
                        pick_match_rate = float((merged_pick["oracle_expert"] == merged_pick["stepf_expert"]).mean())
                        pick_map = {str(r.Date.date()): str(r.stepf_expert) for r in merged_pick.itertuples(index=False)}
                        for row in by_date_rows:
                            row["stepf_expert"] = pick_map.get(row["Date"])
    except Exception:
        pass

    cluster_rows: list[dict[str, Any]] = []
    cluster_status = "PENDING"
    cluster_reason = "cluster comparison pending"
    cluster_col = None
    cluster_map_df = None
    for ef in expert_frames.values():
        if "cluster_id_stable" in ef.columns:
            cluster_col = "cluster_id_stable"
            cluster_map_df = ef[["Date", cluster_col]].drop_duplicates(subset=["Date"])
            break
        if "cluster_id_raw20" in ef.columns:
            cluster_col = "cluster_id_raw20"
            cluster_map_df = ef[["Date", cluster_col]].drop_duplicates(subset=["Date"])
    if cluster_col is not None and cluster_map_df is not None:
        orc_cluster = oracle_df.merge(cluster_map_df, on="Date", how="left").dropna(subset=[cluster_col])
        if not orc_cluster.empty:
            cluster_status = "OK"
            cluster_reason = "computed from StepE daily logs"
            for cid, cdf in orc_cluster.groupby(cluster_col):
                cid_s = str(cid)
                top_oracle = cdf["oracle_expert"].value_counts(normalize=True)
                oracle_top_freq = float(top_oracle.iloc[0]) if not top_oracle.empty else 0.0
                oracle_top_name = str(top_oracle.index[0]) if not top_oracle.empty else "NA"
                mismatch_rate = None
                stepf_top_freq = None
                if any("stepf_expert" in x for x in by_date_rows):
                    c_dates = set(str(pd.Timestamp(x).date()) for x in cdf["Date"].tolist())
                    c_rows = [r for r in by_date_rows if r["Date"] in c_dates and r.get("stepf_expert")]
                    if c_rows:
                        stepf_names = pd.Series([r.get("stepf_expert") for r in c_rows], dtype="object")
                        stepf_top_freq = float(stepf_names.value_counts(normalize=True).iloc[0])
                        mismatch_rate = float(np.mean([r.get("stepf_expert") != r.get("oracle_expert") for r in c_rows]))
                # best expert by cluster from mean return
                best_name = "NA"
                best_ret = -np.inf
                for ag, ef in expert_frames.items():
                    if cluster_col not in ef.columns:
                        continue
                    tmp = ef[ef[cluster_col] == cid]
                    if tmp.empty:
                        continue
                    mr = float(pd.to_numeric(tmp["ret"], errors="coerce").dropna().mean())
                    if np.isfinite(mr) and mr > best_ret:
                        best_ret = mr
                        best_name = ag
                cluster_rows.append(
                    {
                        "cluster_id": cid_s,
                        "best_expert_by_cluster": best_name,
                        "oracle_top1_expert": oracle_top_name,
                        "top1_frequency": _to_float(oracle_top_freq),
                        "stepf_selected_frequency": _to_float(stepf_top_freq),
                        "mismatch_rate": _to_float(mismatch_rate),
                    }
                )

    reward_compare_rows: list[dict[str, Any]] = []
    reward_modes = ["legacy", "profit_basic", "profit_regret", "profit_light_risk"]
    reward_rows_by_mode: dict[str, dict[str, Any]] = {}
    current_test_df = stepf_test.copy()
    reward_legacy_row = _mode_compare_row("reward_legacy", current_test_df, fixed_by_date, oracle_df, pick_match_rate)
    reward_rows_by_mode["reward_legacy"] = reward_legacy_row
    reward_rows_by_mode["current_stepf"] = {**reward_legacy_row, "name": "current_stepf", "alias_of": "reward_legacy"}

    for rm in reward_modes:
        pair = _load_stepf_mode_artifacts(output_root, mode, symbol, rm)
        if pair is None:
            continue
        eq_path, router_path = pair
        try:
            rm_eq = _build_test_ret_frame(_read_csv(eq_path), output_root)
            rm_pick = None
            rdf = _read_csv(router_path)
            wcols = [c for c in rdf.columns if c.startswith("w_")]
            if wcols:
                rwork = rdf.copy()
                split_col = _pick_col(rwork, ["Split"])
                split_values, _, _ = _resolve_split_series(rwork, split_col, output_root=output_root, date_col="Date")
                rwork = rwork[split_values == "test"].copy()
                rwork["Date"] = pd.to_datetime(rwork["Date"], errors="coerce").dt.normalize()
                for c in wcols:
                    rwork[c] = pd.to_numeric(rwork[c], errors="coerce")
                rwork = rwork.dropna(subset=["Date"])
                if not rwork.empty:
                    rwork["stepf_expert"] = rwork[wcols].idxmax(axis=1).str.replace("w_", "", regex=False)
                    merged_pick = oracle_df[["Date", "oracle_expert"]].merge(rwork[["Date", "stepf_expert"]], on="Date", how="inner")
                    if not merged_pick.empty:
                        rm_pick = float((merged_pick["oracle_expert"] == merged_pick["stepf_expert"]).mean())
            reward_rows_by_mode[f"reward_{rm}"] = _mode_compare_row(f"reward_{rm}", rm_eq, fixed_by_date, oracle_df, rm_pick)
        except Exception:
            continue

    fixed_mode_row = {
        "name": "fixed_best",
        "equity_multiple": _to_float(fixed_best.get("equity_multiple")),
        "sharpe": _to_float(fixed_best.get("sharpe")),
        "max_dd": _to_float(fixed_best.get("max_dd")),
        "mean_ret": _to_float(fixed_best.get("mean_ret")),
        "std_ret": _to_float(fixed_best.get("std_ret")),
        "regret_vs_fixed_best": _to_float(0.0),
        "regret_vs_oracle": _to_float((oracle_metrics.get("equity_multiple") or np.nan) - (fixed_best.get("equity_multiple") or np.nan)),
        "win_days_vs_fixed_best": None,
        "pick_match_rate_vs_oracle": None,
        "common_days_vs_fixed_best": None,
    }
    oracle_mode_row = {
        "name": "daily_oracle",
        "equity_multiple": _to_float(oracle_metrics.get("equity_multiple")),
        "sharpe": _to_float(oracle_metrics.get("sharpe")),
        "max_dd": _to_float(oracle_metrics.get("max_dd")),
        "mean_ret": _to_float(oracle_metrics.get("mean_ret")),
        "std_ret": _to_float(oracle_metrics.get("std_ret")),
        "regret_vs_fixed_best": _to_float((fixed_best.get("equity_multiple") or np.nan) - (oracle_metrics.get("equity_multiple") or np.nan)),
        "regret_vs_oracle": _to_float(0.0),
        "win_days_vs_fixed_best": None,
        "pick_match_rate_vs_oracle": None,
        "common_days_vs_fixed_best": None,
    }

    ordered_names = ["reward_legacy", "current_stepf", "reward_profit_basic", "reward_profit_regret", "reward_profit_light_risk"]
    for n in ordered_names:
        if n in reward_rows_by_mode:
            reward_compare_rows.append(reward_rows_by_mode[n])
    reward_compare_rows.extend([fixed_mode_row, oracle_mode_row])

    current_em = float(current_row.get("equity_multiple") or np.nan)
    fixed_em = float(fixed_best.get("equity_multiple") or np.nan)
    oracle_em = float(oracle_metrics.get("equity_multiple") or np.nan)
    summary_row = {
        "current_stepf_equity_multiple": _to_float(current_em),
        "fixed_best_expert": str(fixed_best.get("agent")),
        "fixed_best_equity_multiple": _to_float(fixed_em),
        "oracle_equity_multiple": _to_float(oracle_em),
        "regret_vs_fixed_best": _to_float(fixed_em - current_em),
        "regret_vs_oracle": _to_float(oracle_em - current_em),
        "current_stepf_sharpe": _to_float(current_row.get("sharpe")),
        "fixed_best_sharpe": _to_float(fixed_best.get("sharpe")),
        "oracle_sharpe": _to_float(oracle_metrics.get("sharpe")),
        "current_stepf_max_dd": _to_float(current_row.get("max_dd")),
        "fixed_best_max_dd": _to_float(fixed_best.get("max_dd")),
        "oracle_max_dd": _to_float(oracle_metrics.get("max_dd")),
        "stepf_win_days_vs_fixed_best": _to_int(win_days),
        "stepf_common_days_vs_fixed_best": _to_int(common_days),
        "stepf_pick_match_rate_vs_oracle": _to_float(pick_match_rate),
        "oracle_unique_expert_count": _to_int(len(oracle_counts)),
    }
    return {
        "status": "OK",
        "summary": "current StepF vs fixed-best expert vs daily oracle",
        "row": summary_row,
        "best_expert_name": str(fixed_best.get("agent")),
        "best_expert_equity_multiple": _to_float(fixed_em),
        "best_expert_sharpe": _to_float(fixed_best.get("sharpe")),
        "best_expert_max_dd": _to_float(fixed_best.get("max_dd")),
        "oracle_equity_multiple": _to_float(oracle_em),
        "oracle_sharpe": _to_float(oracle_metrics.get("sharpe")),
        "oracle_max_dd": _to_float(oracle_metrics.get("max_dd")),
        "oracle_selected_expert_count_by_name": {str(k): int(v) for k, v in oracle_counts.items()},
        "oracle_unique_expert_count": int(len(oracle_counts)),
        "cumulative_regret_vs_oracle": _to_float(cum_regret),
        "cluster": {"status": cluster_status, "reason": cluster_reason, "rows": cluster_rows},
        "series": by_date_rows,
        "stepf_selected_expert_count_by_name": stepf_selected_by_name,
        "stepF_reward_compare": {
            "status": "OK" if len(reward_compare_rows) >= 3 else "WARN",
            "summary": "reward mode comparison across current/baseline/oracle",
            "rows": reward_compare_rows,
        },
    }


def _save_empty_plot_notice(path: str, title: str, message: str) -> None:
    if not MPL_AVAILABLE or plt is None:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _generate_plots(output_root: str, mode: str, symbol: str, report: dict[str, Any], out_dir: str) -> list[str]:
    notes: list[str] = []
    plots: list[dict[str, Any]] = []
    os.makedirs(out_dir, exist_ok=True)

    if not MPL_AVAILABLE or plt is None:
        notes.append("PLOT backend unavailable: matplotlib is not installed")
        report["plots"] = {"items": plots, "notes": notes}
        return notes

    def record_plot(name: str, reason: str | None = None) -> str:
        path = os.path.join(out_dir, name)
        plots.append({"name": name, "path": path, "exists": os.path.exists(path), "reason": reason})
        return path

    # StepE equity overlay
    try:
        step_e_logs = sorted(glob.glob(os.path.join(output_root, "stepE", mode, f"stepE_daily_log_*_{symbol}.csv")))
        if not step_e_logs:
            step_e_logs = sorted(glob.glob(os.path.join(output_root, "stepE", "*", f"stepE_daily_log_*_{symbol}.csv")))
        path = record_plot("equity_stepE_topN.png")
        if not step_e_logs:
            _save_empty_plot_notice(path, "StepE topN equity", "stepE_daily_log files were not found")
            notes.append("PLOT StepE topN: stepE_daily_log files were not found")
        else:
            rows = report.get("stepE", {}).get("rows", [])
            ranked = [r for r in rows if r.get("equity_multiple") is not None]
            ranked = sorted(ranked, key=lambda x: float(x.get("equity_multiple") or -1), reverse=True)[:10]
            if not ranked:
                _save_empty_plot_notice(path, "StepE topN equity", "No StepE rows with numeric equity_multiple")
                notes.append("PLOT StepE topN: no numeric StepE equity_multiple rows")
            else:
                fig, ax = plt.subplots(figsize=(12, 6))
                plotted = 0
                for row in ranked:
                    agent = row.get("agent")
                    if not agent:
                        continue
                    hit = [x for x in step_e_logs if f"_{agent}_{symbol}.csv" in os.path.basename(x)]
                    if not hit:
                        continue
                    df = _read_csv(hit[0])
                    split_col = _pick_col(df, ["Split"])
                    equity_col = _pick_col(df, ["equity", "Equity"])
                    if equity_col is None:
                        continue
                    if split_col and split_col in df.columns:
                        df = df[df[split_col] == "test"]
                    eq = pd.to_numeric(df[equity_col], errors="coerce").dropna().reset_index(drop=True)
                    if eq.empty:
                        continue
                    ax.plot(eq.values, linewidth=1.5, label=str(agent))
                    plotted += 1
                if plotted == 0:
                    _save_empty_plot_notice(path, "StepE topN equity", "No plottable StepE test equity series")
                    notes.append("PLOT StepE topN: no plottable StepE test equity series")
                else:
                    ax.set_title("StepE topN test equity")
                    ax.set_xlabel("Test step")
                    ax.set_ylabel("Equity")
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc="best", fontsize=8)
                    fig.tight_layout()
                    fig.savefig(path, dpi=120)
                    plt.close(fig)
    except Exception as exc:
        notes.append(f"PLOT StepE topN exception: {exc}")

    # StepF equity
    try:
        path = record_plot("equity_stepF.png")
        step_f_logs = sorted(glob.glob(os.path.join(output_root, "stepF", mode, f"stepF_equity_marl_{symbol}.csv")))
        if not step_f_logs:
            step_f_logs = sorted(glob.glob(os.path.join(output_root, "stepF", "*", f"stepF_equity_marl_{symbol}.csv")))
        if not step_f_logs:
            _save_empty_plot_notice(path, "StepF equity", "stepF_equity_marl file was not found")
            notes.append("PLOT StepF equity: stepF_equity_marl file not found")
        else:
            df = _read_csv(step_f_logs[0])
            split_col = _pick_col(df, ["Split"])
            equity_col = _pick_col(df, ["equity", "Equity"])
            if equity_col is None:
                _save_empty_plot_notice(path, "StepF equity", "equity column missing")
                notes.append("PLOT StepF equity: equity column missing")
            else:
                if split_col and split_col in df.columns:
                    df = df[df[split_col] == "test"]
                eq = pd.to_numeric(df[equity_col], errors="coerce").dropna().reset_index(drop=True)
                if eq.empty:
                    _save_empty_plot_notice(path, "StepF equity", "No numeric StepF test equity rows")
                    notes.append("PLOT StepF equity: no numeric test equity rows")
                else:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(eq.values, color="black", linewidth=2.0, label="StepF")
                    ax.set_title("StepF test equity")
                    ax.set_xlabel("Test step")
                    ax.set_ylabel("Equity")
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc="best")
                    fig.tight_layout()
                    fig.savefig(path, dpi=120)
                    plt.close(fig)
    except Exception as exc:
        notes.append(f"PLOT StepF equity exception: {exc}")

    # StepE returns bar
    try:
        path = record_plot("bar_stepE_return.png")
        rows = [r for r in report.get("stepE", {}).get("rows", []) if r.get("equity_multiple") is not None and r.get("agent")]
        if not rows:
            _save_empty_plot_notice(path, "StepE equity multiple", "No numeric StepE equity_multiple rows")
            notes.append("PLOT StepE return bar: no numeric StepE rows")
        else:
            rows = sorted(rows, key=lambda x: float(x.get("equity_multiple") or -1), reverse=True)[:10]
            fig, ax = plt.subplots(figsize=(12, 6))
            agents = [str(r.get("agent")) for r in rows]
            values = [float(r.get("equity_multiple") or 0.0) for r in rows]
            ax.bar(agents, values, color="tab:blue")
            ax.set_title("StepE equity multiple (topN)")
            ax.set_xlabel("Agent")
            ax.set_ylabel("Equity multiple")
            ax.tick_params(axis="x", rotation=30)
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(path, dpi=120)
            plt.close(fig)
    except Exception as exc:
        notes.append(f"PLOT StepE return bar exception: {exc}")

    # StepE DD vs return scatter and StepF comparison note
    try:
        path = record_plot("scatter_stepE_dd_vs_ret.png")
        rows = [
            r
            for r in report.get("stepE", {}).get("rows", [])
            if r.get("equity_multiple") is not None and r.get("max_dd") is not None and r.get("agent")
        ]
        if not rows:
            _save_empty_plot_notice(path, "StepE max_dd vs return", "No numeric StepE (max_dd, equity_multiple) pairs")
            notes.append("PLOT StepE scatter: no numeric StepE DD/return pairs")
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            x = [float(r.get("max_dd")) for r in rows]
            y = [float(r.get("equity_multiple")) for r in rows]
            ax.scatter(x, y, color="tab:green", alpha=0.8)
            for r in rows[:10]:
                ax.annotate(str(r.get("agent")), (float(r.get("max_dd")), float(r.get("equity_multiple"))), fontsize=8)
            ax.set_title("StepE max drawdown vs equity multiple")
            ax.set_xlabel("max_dd")
            ax.set_ylabel("equity_multiple")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(path, dpi=120)
            plt.close(fig)

        stepf_rows = [r for r in report.get("stepF", {}).get("rows", []) if r.get("equity_multiple") is not None]
        best_stepe = max((float(r.get("equity_multiple")) for r in report.get("stepE", {}).get("rows", []) if r.get("equity_multiple") is not None), default=None)
        stepf_best = max((float(r.get("equity_multiple")) for r in stepf_rows), default=None)
        if best_stepe is not None and stepf_best is not None:
            relation = "勝ってる" if stepf_best >= best_stepe else "負けてる"
            notes.append(f"StepF_vs_best_StepE: StepF({stepf_best:.4f}) は best StepE({best_stepe:.4f}) に{relation}")
        else:
            notes.append("StepF_vs_best_StepE: 比較に必要な equity_multiple が不足")
    except Exception as exc:
        notes.append(f"PLOT StepE scatter exception: {exc}")

    # StepF compare plots
    try:
        cmp = report.get("stepF_compare", {}) if isinstance(report.get("stepF_compare", {}), dict) else {}
        path = record_plot("equity_stepF_vs_fixed_best_vs_oracle.png")
        series = cmp.get("series", []) if isinstance(cmp.get("series", []), list) else []
        if not series:
            _save_empty_plot_notice(path, "StepF vs fixed-best vs oracle", "StepF comparison series is unavailable")
            notes.append("PLOT StepF compare equity: comparison series unavailable")
        else:
            s = pd.DataFrame(series)
            for c in ["current_stepf_ret", "fixed_best_ret", "oracle_ret"]:
                s[c] = pd.to_numeric(s.get(c), errors="coerce")
            s = s.dropna(subset=["Date"]).copy()
            if s.empty:
                _save_empty_plot_notice(path, "StepF vs fixed-best vs oracle", "No plottable compare rows")
                notes.append("PLOT StepF compare equity: no plottable rows")
            else:
                cur = (1.0 + s["current_stepf_ret"].fillna(0.0)).cumprod()
                fxd = (1.0 + s["fixed_best_ret"].fillna(0.0)).cumprod()
                orc = (1.0 + s["oracle_ret"].fillna(0.0)).cumprod()
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(cur.values, label="current_stepf", color="black", linewidth=2.0)
                ax.plot(fxd.values, label="fixed_best", color="tab:blue", linewidth=1.5)
                ax.plot(orc.values, label="daily_oracle", color="tab:orange", linewidth=1.5)
                ax.set_title("StepF vs Fixed Best vs Daily Oracle (test)")
                ax.set_xlabel("Test day")
                ax.set_ylabel("Equity")
                ax.grid(True, alpha=0.3)
                ax.legend(loc="best")
                fig.tight_layout()
                fig.savefig(path, dpi=120)
                plt.close(fig)

        path2 = record_plot("bar_stepF_regret.png")
        row = cmp.get("row", {}) if isinstance(cmp.get("row", {}), dict) else {}
        rv_fixed = row.get("regret_vs_fixed_best")
        rv_oracle = row.get("regret_vs_oracle")
        if rv_fixed is None and rv_oracle is None:
            _save_empty_plot_notice(path2, "StepF regret", "Regret values are unavailable")
            notes.append("PLOT StepF regret: regret values unavailable")
        else:
            vals = [0.0 if rv_fixed is None else float(rv_fixed), 0.0 if rv_oracle is None else float(rv_oracle)]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(["vs_fixed_best", "vs_oracle"], vals, color=["tab:blue", "tab:orange"])
            ax.axhline(0.0, color="black", linewidth=1.0)
            ax.set_title("StepF regret (equity multiple gap)")
            ax.set_ylabel("Regret")
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(path2, dpi=120)
            plt.close(fig)

        reward_cmp = cmp.get("stepF_reward_compare", {}) if isinstance(cmp.get("stepF_reward_compare", {}), dict) else {}
        rrows = reward_cmp.get("rows", []) if isinstance(reward_cmp.get("rows", []), list) else []
        path3 = record_plot("equity_stepF_reward_modes.png")
        if not rrows:
            _save_empty_plot_notice(path3, "StepF reward modes", "reward compare rows unavailable")
        else:
            names = [str(r.get("name")) for r in rrows]
            eqs = [0.0 if r.get("equity_multiple") is None else float(r.get("equity_multiple")) for r in rrows]
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(names, eqs, marker="o", linewidth=1.5)
            ax.set_title("StepF reward modes equity multiple")
            ax.set_ylabel("equity_multiple")
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(path3, dpi=120)
            plt.close(fig)

        path4 = record_plot("bar_stepF_reward_mode_regret.png")
        mode_rows = [r for r in rrows if str(r.get("name", "")).startswith("reward_") or str(r.get("name")) == "current_stepf"]
        if not mode_rows:
            _save_empty_plot_notice(path4, "StepF reward mode regret", "reward-mode regrets unavailable")
        else:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar([str(r.get("name")) for r in mode_rows], [0.0 if r.get("regret_vs_fixed_best") is None else float(r.get("regret_vs_fixed_best")) for r in mode_rows], color="tab:green")
            ax.axhline(0.0, color="black", linewidth=1.0)
            ax.set_title("StepF reward modes regret vs fixed best")
            ax.set_ylabel("regret_vs_fixed_best")
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(path4, dpi=120)
            plt.close(fig)
    except Exception as exc:
        notes.append(f"PLOT StepF compare exception: {exc}")

    for item in plots:
        item["exists"] = os.path.exists(item.get("path", ""))
    report["plots"] = {"items": plots, "notes": notes}
    return notes




def _collect_dprime_artifacts(output_root: str, mode: str, symbol: str) -> dict[str, Any]:
    bases = [os.path.join(output_root, "stepDprime", mode), os.path.join(output_root, "stepDPrime", mode)]

    rl_state_patterns: list[str] = []
    cluster_embeddings_patterns: list[str] = []
    cluster_state_patterns: list[str] = []
    cluster_input_patterns: list[str] = []
    traceback_patterns: list[str] = []
    failure_summary_patterns: list[str] = []
    for base in bases:
        rl_state_patterns.extend(
            [
                os.path.join(base, "stepDprime_state_test_*.csv"),
                os.path.join(base, f"stepDprime_state_*_{symbol}_test.csv"),
                os.path.join(base, f"stepDprime_state_test_*_{symbol}.csv"),
                os.path.join(base, f"stepDprime_state_*_{symbol}.csv"),
            ]
        )
        cluster_embeddings_patterns.extend(
            [
                os.path.join(base, "embeddings", f"stepDprime_*_{symbol}_embeddings*.csv"),
                os.path.join(base, "embeddings", "*.csv"),
            ]
        )
        cluster_state_patterns.extend(
            [
                os.path.join(base, "*cluster*state*.csv"),
                os.path.join(base, "*cluster_id*.csv"),
                os.path.join(base, "*rare_flag*.csv"),
                os.path.join(base, "*raw20*.csv"),
                os.path.join(base, "*stable*.csv"),
            ]
        )
        cluster_input_patterns.extend(
            [
                os.path.join(base, "*cluster*input*.csv"),
                os.path.join(base, "*cluster_features*.csv"),
            ]
        )
        traceback_patterns.append(os.path.join(base, f"stepDprime_traceback_{symbol}.log"))
        traceback_patterns.append(os.path.join(base, "stepDprime_traceback_*.log"))
        failure_summary_patterns.append(os.path.join(base, f"stepDprime_failure_summary_{symbol}.json"))
        failure_summary_patterns.append(os.path.join(base, "stepDprime_failure_summary_*.json"))

    def _collect(patterns: list[str]) -> list[str]:
        files: list[str] = []
        for pat in patterns:
            files.extend(glob.glob(pat))
        return sorted(set(files))

    rl_state_files = _collect(rl_state_patterns)
    cluster_embeddings_files = _collect(cluster_embeddings_patterns)
    cluster_state_files = _collect(cluster_state_patterns)
    cluster_input_files = _collect(cluster_input_patterns)
    traceback_files = _collect(traceback_patterns)
    failure_summary_files = _collect(failure_summary_patterns)

    rl_profiles = {
        m.group(1)
        for f in rl_state_files
        for m in [re.match(r"^stepDprime_state_test_(.+?)_[^_]+\.csv$", os.path.basename(f))]
        if m
    }

    rl_status = "OK" if rl_state_files else "WARN"
    cluster_status = "OK" if (cluster_embeddings_files or cluster_state_files or cluster_input_files) else "WARN"

    rl_summary = "RL state files found under stepDprime/sim" if rl_state_files else "missing RL state files"
    fail_reason = ""
    if failure_summary_files:
        try:
            with open(failure_summary_files[0], 'r', encoding='utf-8') as fh:
                payload = json.load(fh)
            fail_reason = str(payload.get('exception_repr') or payload.get('exception_type') or '')
        except Exception:
            fail_reason = "failure_summary_parse_error"
    cluster_parts = []
    if cluster_embeddings_files:
        cluster_parts.append("embeddings found")
    if cluster_state_files:
        cluster_parts.append("cluster state files found")
    if cluster_input_files:
        cluster_parts.append("cluster input files found")
    cluster_summary = ", ".join(cluster_parts) if cluster_parts else "missing cluster embeddings/state/input"

    print(f"[DPRIME_DIAG] cluster_embeddings_count={len(cluster_embeddings_files)}")
    print(f"[DPRIME_DIAG] cluster_state_count={len(cluster_state_files)}")
    print(f"[DPRIME_DIAG] rl_state_count={len(rl_state_files)}")
    print(f"[DPRIME_DIAG] rl_state_glob={rl_state_patterns[0] if rl_state_patterns else 'NA'}")
    print(f"[DPRIME_DIAG] cluster_glob={cluster_embeddings_patterns[0] if cluster_embeddings_patterns else 'NA'}")
    print(f"[DPRIME_DIAG] final_cluster_status={cluster_status}")
    print(f"[DPRIME_DIAG] final_rl_status={rl_status}")
    print(f"[DPRIME_DIAG] traceback_count={len(traceback_files)}")
    print(f"[DPRIME_DIAG] failure_summary_count={len(failure_summary_files)}")

    status = "OK" if rl_status == "OK" and cluster_status == "OK" else "WARN"
    summary = f"DPrimeCluster={cluster_status} ({cluster_summary}); DPrimeRL={rl_status} ({rl_summary})"
    if fail_reason:
        summary += f"; failure_reason={fail_reason}"

    return {
        "status": status,
        "summary": summary,
        "details": {
            "state_count": len(rl_state_files),
            "embeddings_count": len(cluster_embeddings_files),
            "cluster_status": cluster_status,
            "cluster_summary": cluster_summary,
            "cluster_embeddings_count": len(cluster_embeddings_files),
            "cluster_state_count": len(cluster_state_files),
            "cluster_input_count": len(cluster_input_files),
            "cluster_embeddings_files": [os.path.basename(x) for x in cluster_embeddings_files],
            "cluster_state_files": [os.path.basename(x) for x in cluster_state_files],
            "cluster_input_files": [os.path.basename(x) for x in cluster_input_files],
            "rl_status": rl_status,
            "rl_summary": rl_summary,
            "rl_state_count": len(rl_state_files),
            "rl_profiles_count": len(rl_profiles),
            "rl_state_files": [os.path.basename(x) for x in rl_state_files],
            "state_files": [os.path.basename(x) for x in rl_state_files],
            "embeddings_files": [os.path.basename(x) for x in cluster_embeddings_files],
            "searched": rl_state_patterns + cluster_embeddings_patterns + cluster_state_patterns + cluster_input_patterns,
            "rl_state_glob": rl_state_patterns,
            "cluster_glob": cluster_embeddings_patterns + cluster_state_patterns + cluster_input_patterns,
            "traceback_count": len(traceback_files),
            "traceback_files": [os.path.basename(x) for x in traceback_files],
            "failure_summary_count": len(failure_summary_files),
            "failure_summary_files": [os.path.basename(x) for x in failure_summary_files],
            "failure_reason": fail_reason,
            "traceback_glob": traceback_patterns,
            "failure_summary_glob": failure_summary_patterns,
        },
    }

def evaluate(output_root: str, mode: str, symbol: str) -> dict[str, Any]:
    report: dict[str, Any] = {
        "output_root": output_root,
        "mode": mode,
        "symbol": symbol,
        "stepA": {"status": "SKIP", "summary": "not evaluated", "details": {}},
        "stepB": {"status": "SKIP", "summary": "not evaluated", "rows": []},
        "dprime": {"status": "SKIP", "summary": "not evaluated", "details": {}},
        "stepE": {"status": "SKIP", "summary": "not evaluated", "rows": []},
        "stepF": {"status": "SKIP", "summary": "not evaluated", "rows": []},
        "diversity": {"status": "SKIP", "summary": "not evaluated"},
        "overall_status": "WARN",
    }

    stepa_prices: pd.DataFrame | None = None

    # StepA
    try:
        stepa_search_roots = _canonical_eval_roots(output_root, mode, symbol)
        prices_path = None
        stepa_patterns: list[str] = []
        for _root in stepa_search_roots:
            _pattern = os.path.join(_root, "stepA", "*", f"stepA_prices_test_{symbol}.csv")
            stepa_patterns.append(_pattern)
            prices_path = _find_first(_pattern)
            if prices_path:
                break

        if not prices_path:
            missing_path = stepa_patterns[0] if stepa_patterns else os.path.join(output_root, "stepA", "*", f"stepA_prices_test_{symbol}.csv")
            report["stepA"] = {
                "status": "SKIP",
                "summary": "stepA_prices_test file missing",
                "details": {
                    "missing_path": missing_path,
                    "searched_patterns": stepa_patterns,
                },
            }
        else:
            px = _parse_date(_read_csv(prices_path), "Date")
            stepa_prices = px
            d = {
                "path": prices_path,
                "test_rows": int(len(px)),
                "test_date_start": str(px["Date"].dropna().min().date()) if "Date" in px.columns and px["Date"].notna().any() else None,
                "test_date_end": str(px["Date"].dropna().max().date()) if "Date" in px.columns and px["Date"].notna().any() else None,
                "missing_ohlcv_count": int(
                    sum(int(px[c].isna().sum()) for c in ["Open", "High", "Low", "Close", "Volume"] if c in px.columns)
                ),
                "ohlcv_missing": {c: int(px[c].isna().sum()) for c in ["Open", "High", "Low", "Close", "Volume"] if c in px.columns},
                "searched_patterns": stepa_patterns,
            }
            report["stepA"] = {"status": "OK", "summary": "prices_test evaluated", "details": d}
    except Exception as exc:
        report["stepA"] = {"status": "SKIP", "summary": f"exception: {exc}", "details": {"traceback": traceback.format_exc(limit=2)}}

    # StepB
    try:
        patterns = [
            f"stepB_pred_time_all_{symbol}.csv",
            f"stepB_pred_close_mamba_{symbol}.csv",
            f"stepB_pred_path_mamba_{symbol}.csv",
            f"stepB_pred_close_wavelet_mamba_{symbol}.csv",
            f"stepB_pred_path_wavelet_mamba_{symbol}.csv",
            f"stepB_pred_close_mamba_periodic_{symbol}.csv",
        ]
        files: list[str] = []
        for p in patterns:
            files.extend(glob.glob(os.path.join(output_root, "stepB", mode, p)))
        files = sorted(set(files))

        if not files:
            report["stepB"] = {"status": "SKIP", "summary": "no stepB prediction files found", "rows": [], "files": []}
        else:
            rows: list[dict[str, Any]] = []
            key_cols = {"pred_close_mamba"}
            ignore_tokens = ("xsr", "fed")
            for fpath in files:
                try:
                    df = _read_csv(fpath)
                    date_col = "Date" if "Date" in df.columns else ("Date_anchor" if "Date_anchor" in df.columns else None)
                    if date_col:
                        df = _parse_date(df, date_col)

                    pred_cols = [
                        c
                        for c in df.columns
                        if pd.api.types.is_numeric_dtype(df[c])
                        and ("pred_" in c.lower() or c.lower().endswith("_pred"))
                        and not any(token in c.lower() for token in ignore_tokens)
                    ]
                    picked = [c for c in pred_cols if c.lower() in key_cols]
                    if not picked:
                        # StepB phase-1: evaluate only MAMBA-like prediction columns.
                        picked = [c for c in pred_cols if "mamba" in c.lower()]

                    for col in picked:
                        total_rows = len(df)
                        nn = int(df[col].notna().sum())
                        nn_ratio = float(nn / total_rows) if total_rows else 0.0
                        first_valid_date = None
                        if nn > 0 and date_col:
                            valid_dates = df.loc[df[col].notna(), date_col].dropna()
                            if not valid_dates.empty:
                                first_valid_date = str(valid_dates.iloc[0].date())

                        coverage_ratio = None
                        true_series = None
                        pred_series = None
                        if stepa_prices is not None and "Date" in stepa_prices.columns and date_col:
                            left = stepa_prices[["Date", "Close"]].copy()
                            right = df[[date_col, col]].rename(columns={date_col: "Date"}).copy()
                            left["Date"] = pd.to_datetime(left["Date"], errors="coerce").dt.normalize()
                            right["Date"] = pd.to_datetime(right["Date"], errors="coerce").dt.normalize()
                            merged = left.merge(right, on="Date", how="left")
                            coverage_ratio = float(merged[col].notna().mean()) if len(merged) else None
                            true_series = merged["Close"]
                            pred_series = merged[col]
                        elif "Close_true" in df.columns:
                            true_series = df["Close_true"]
                            pred_series = df[col]

                        mae = corr = None
                        if true_series is not None and pred_series is not None:
                            mm = _calc_metrics(true_series, pred_series)
                            mae, corr = mm["mae"], mm["corr"]

                        status = "OK"
                        if nn_ratio < 0.5:
                            status = "BAD"
                        elif nn_ratio < 0.9 or (coverage_ratio is not None and coverage_ratio < 0.9):
                            status = "WARN"

                        rows.append(
                            {
                                "file": os.path.basename(fpath),
                                "pred_col": col,
                                "non_null_ratio": nn_ratio,
                                "first_valid_date": first_valid_date,
                                "coverage_ratio_over_test": coverage_ratio,
                                "mae": mae,
                                "corr": corr,
                                "status": status,
                            }
                        )
                except Exception as exc:
                    rows.append({"file": os.path.basename(fpath), "pred_col": None, "status": "SKIP", "reason": str(exc)})

            stepb_status = "SKIP" if not rows else max((r.get("status", "WARN") for r in rows), key=_status_level)
            report["stepB"] = {
                "status": stepb_status,
                "summary": "stepB files evaluated",
                "rows": rows,
                "files": [os.path.basename(p) for p in files],
            }
    except Exception as exc:
        report["stepB"] = {"status": "SKIP", "summary": f"exception: {exc}", "rows": []}

    # D' (StepD prime artifacts only)
    try:
        report["dprime"] = _collect_dprime_artifacts(output_root=output_root, mode=mode, symbol=symbol)
    except Exception as exc:
        report["dprime"] = {"status": "WARN", "summary": f"exception: {exc}", "details": {}}

    # StepE
    try:
        step_e_logs = sorted(glob.glob(os.path.join(output_root, "stepE", mode, f"stepE_daily_log_*_{symbol}.csv")))
        if not step_e_logs:
            step_e_logs = sorted(glob.glob(os.path.join(output_root, "stepE", "*", f"stepE_daily_log_*_{symbol}.csv")))
        if not step_e_logs:
            report["stepE"] = {"status": "SKIP", "summary": "stepE_daily_log missing", "rows": []}
        else:
            rows = []
            pos_rows: list[dict[str, Any]] = []
            for fpath in step_e_logs:
                try:
                    df = _read_csv(fpath)
                    fname = os.path.basename(fpath)
                    m = re.match(rf"stepE_daily_log_(.+)_{re.escape(symbol)}\.csv$", fname)
                    agent = m.group(1) if m else fname
                    pos_col = _pick_col(df, ["pos", "Position"])
                    ret_col = _pick_col(df, ["ret"])
                    equity_col = _pick_col(df, ["equity"])
                    split_col = _pick_col(df, ["Split"])

                    missing_reasons = []
                    if pos_col is None:
                        missing_reasons.append("pos/Position missing")
                    if ret_col is None:
                        missing_reasons.append("ret missing")
                    if equity_col is None:
                        missing_reasons.append("equity missing")
                    if split_col is None:
                        missing_reasons.append("Split missing")
                    if missing_reasons:
                        rows.append({"file": fname, "agent": agent, "status": "SKIP", "reason": ", ".join(missing_reasons)})
                        continue

                    metrics, reason = _calc_equity_metrics(df, equity_col=equity_col, ret_col=ret_col, split_col=split_col, output_root=output_root, date_col="Date")
                    row = {
                        "file": fname,
                        "agent": agent,
                        **metrics,
                        "status": "OK" if metrics["test_days"] > 0 else "WARN",
                    }
                    if reason:
                        row["note"] = reason
                    rows.append(row)

                    test_pos = df[df[split_col] == "test"][pos_col] if split_col in df.columns else pd.Series(dtype=float)
                    test_pos = pd.to_numeric(test_pos, errors="coerce").dropna().reset_index(drop=True)
                    if not test_pos.empty:
                        pos_rows.append({"agent": agent, "series": test_pos})
                except Exception as exc:
                    rows.append({"file": os.path.basename(fpath), "status": "SKIP", "reason": str(exc)})

            report["diversity"] = _calc_diversity(pos_rows)
            rows = sorted(rows, key=lambda r: (0 if r.get("status") == "OK" else 1, str(r.get("agent", ""))))
            report["stepE"] = {
                "status": "OK" if any(r.get("status") == "OK" for r in rows) else "SKIP",
                "summary": "stepE daily logs evaluated",
                "rows": rows,
            }
    except Exception as exc:
        report["stepE"] = {"status": "SKIP", "summary": f"exception: {exc}", "rows": []}

    # StepF
    try:
        step_f_logs = sorted(glob.glob(os.path.join(output_root, "stepF", mode, f"stepF_equity_marl_{symbol}.csv")))
        if not step_f_logs:
            step_f_logs = sorted(glob.glob(os.path.join(output_root, "stepF", "*", f"stepF_equity_marl_{symbol}.csv")))
        if not step_f_logs:
            report["stepF"] = {"status": "SKIP", "summary": "stepF_equity_marl missing", "rows": []}
        else:
            rows = []
            for fpath in step_f_logs:
                try:
                    df = _read_csv(fpath)
                    equity_col = _pick_col(df, ["equity", "Equity"])
                    if equity_col is None:
                        rows.append({"file": os.path.basename(fpath), "status": "SKIP", "reason": "equity/Equity missing"})
                        continue
                    ret_col = _pick_col(df, ["ret"])
                    note = None
                    if ret_col is None:
                        work = df.copy()
                        work["_ret_eval"] = pd.to_numeric(work[equity_col], errors="coerce").pct_change()
                        df = work
                        ret_col = "_ret_eval"
                        note = "ret missing: computed from equity pct_change"
                    split_col = _pick_col(df, ["Split"])
                    metrics, reason = _calc_equity_metrics(df, equity_col=equity_col, ret_col=ret_col, split_col=split_col, output_root=output_root, date_col="Date")
                    row = {
                        "file": os.path.basename(fpath),
                        **metrics,
                        "status": "OK" if metrics["test_days"] > 0 else "WARN",
                    }
                    merged_note = "; ".join([x for x in [note, reason] if x])
                    if merged_note:
                        row["note"] = merged_note
                    rows.append(row)
                except Exception as exc:
                    rows.append({"file": os.path.basename(fpath), "status": "SKIP", "reason": str(exc)})
            report["stepF"] = {
                "status": "OK" if any(r.get("status") == "OK" for r in rows) else "SKIP",
                "summary": "stepF equity logs evaluated",
                "rows": rows,
            }
    except Exception as exc:
        report["stepF"] = {"status": "SKIP", "summary": f"exception: {exc}", "rows": []}

    # StepF comparison diagnostics (current vs fixed-best StepE vs daily oracle)
    try:
        report["stepF_compare"] = _collect_stepf_compare(output_root=output_root, mode=mode, symbol=symbol, report=report)
    except Exception as exc:
        report["stepF_compare"] = {"status": "WARN", "summary": "compare skipped", "reason": f"exception: {exc}"}

    # Cluster evaluation subsystem
    try:
        report["cluster_eval"] = run_cluster_evaluation(output_root=output_root, mode=mode, symbol=symbol)
    except Exception as exc:
        report["cluster_eval"] = {"status": "WARN", "summary": "cluster evaluation skipped", "reason": f"exception: {exc}"}

    statuses = [
        report["stepA"]["status"],
        report["stepB"]["status"],
        report["dprime"]["status"],
        report["stepE"]["status"],
        report["stepF"]["status"],
        report.get("diversity", {}).get("status", "SKIP"),
        report.get("stepF_compare", {}).get("status", "SKIP"),
    ]
    report["overall_status"] = "BAD" if "BAD" in statuses else ("WARN" if "WARN" in statuses or "SKIP" in statuses else "OK")
    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# EVAL_REPORT",
        "",
        f"- output_root: `{report.get('output_root')}`",
        f"- mode: `{report.get('mode')}`",
        f"- symbol: `{report.get('symbol')}`",
        f"- overall_status: **{report.get('overall_status')}**",
        "",
        "## DPrime diagnostics",
    ]
    dprime = report.get("dprime", {})
    ddet = dprime.get("details", {})
    lines.extend([
        f"- status: **{dprime.get('status', 'SKIP')}**",
        f"- summary: {dprime.get('summary', 'NA')}",
        "",
        "### DPrimeCluster",
        f"- status: **{_fmt(ddet.get('cluster_status'))}**",
        f"- summary: {_fmt(ddet.get('cluster_summary'))}",
        f"- cluster_embeddings_count: {_fmt(ddet.get('cluster_embeddings_count'))}",
        f"- cluster_state_count: {_fmt(ddet.get('cluster_state_count'))}",
        f"- cluster_input_count: {_fmt(ddet.get('cluster_input_count'))}",
        "",
        "### DPrimeRL",
        f"- status: **{_fmt(ddet.get('rl_status'))}**",
        f"- summary: {_fmt(ddet.get('rl_summary'))}",
        f"- rl_state_count: {_fmt(ddet.get('rl_state_count'))}",
        f"- rl_profiles_count: {_fmt(ddet.get('rl_profiles_count'))}",
    ])

    stepa = report.get("stepA", {})
    stepa_d = stepa.get("details", {}) if isinstance(stepa, dict) else {}
    lines.extend([
        "",
        "## StepA table",
        "| status | summary | test_rows | test_date_start | test_date_end | missing_ohlcv_count |",
        "|---|---|---:|---|---|---:|",
        f"| {stepa.get('status', 'SKIP')} | {stepa.get('summary', 'NA')} | {_fmt(stepa_d.get('test_rows'))} | {_fmt(stepa_d.get('test_date_start'))} | {_fmt(stepa_d.get('test_date_end'))} | {_fmt(stepa_d.get('missing_ohlcv_count'))} |",
    ])

    stepb_rows = report.get("stepB", {}).get("rows", [])
    lines.extend([
        "",
        "## StepB table",
    ])
    if stepb_rows:
        lines.extend([
            "| file | pred_col | non_null_ratio | coverage_ratio_over_test | mae | corr | status |",
            "|---|---|---:|---:|---:|---:|---|",
        ])
        for r in stepb_rows:
            lines.append(
                f"| {r.get('file', 'NA')} | {r.get('pred_col', 'NA')} | {_fmt(r.get('non_null_ratio'))} | {_fmt(r.get('coverage_ratio_over_test'))} | {_fmt(r.get('mae'))} | {_fmt(r.get('corr'))} | {r.get('status', 'NA')} |"
            )
    else:
        lines.append(f"- SKIP: {report.get('stepB', {}).get('summary')}")

    lines.extend([
        "",
        "## StepE table",
    ])

    stepe_rows = report.get("stepE", {}).get("rows", [])
    if stepe_rows:
        lines.extend([
            "| agent | file | test_days | equity_multiple | max_dd | mean_ret | std_ret | sharpe | note | status |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
        ])
        for r in stepe_rows:
            lines.append(
                f"| {r.get('agent', 'NA')} | {r.get('file', 'NA')} | {_fmt(r.get('test_days'))} | {_fmt(r.get('equity_multiple'))} | "
                f"{_fmt(r.get('max_dd'))} | {_fmt(r.get('mean_ret'))} | {_fmt(r.get('std_ret'))} | {_fmt(r.get('sharpe'))} | "
                f"{r.get('note', r.get('reason', ''))} | {r.get('status', 'NA')} |"
            )
    else:
        lines.append(f"- SKIP: {report.get('stepE', {}).get('summary')}")

    lines.extend([
        "",
        "## StepF table",
    ])
    stepf_rows = report.get("stepF", {}).get("rows", [])
    if stepf_rows:
        lines.extend([
            "| file | test_days | split_source | train_rows | equity_multiple | max_dd | mean_ret | std_ret | sharpe | note | status |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|",
        ])
        for r in stepf_rows:
            lines.append(
                f"| {r.get('file', 'NA')} | {_fmt(r.get('test_days'))} | {_fmt(r.get('split_source'))} | {_fmt(r.get('train_rows'))} | {_fmt(r.get('equity_multiple'))} | {_fmt(r.get('max_dd'))} | "
                f"{_fmt(r.get('mean_ret'))} | {_fmt(r.get('std_ret'))} | {_fmt(r.get('sharpe'))} | {r.get('note', r.get('reason', ''))} | {r.get('status', 'NA')} |"
            )
    else:
        lines.append(f"- SKIP: {report.get('stepF', {}).get('summary')}")

    stepf_cmp = report.get("stepF_compare", {}) if isinstance(report.get("stepF_compare", {}), dict) else {}
    cmp_row = stepf_cmp.get("row", {}) if isinstance(stepf_cmp.get("row", {}), dict) else {}
    lines.extend([
        "",
        "## StepFCompare",
        f"- status: **{stepf_cmp.get('status', 'WARN')}**",
        f"- summary: {stepf_cmp.get('summary', stepf_cmp.get('reason', 'NA'))}",
    ])
    if cmp_row:
        lines.extend([
            f"- current_stepf_equity_multiple: {_fmt(cmp_row.get('current_stepf_equity_multiple'))}",
            f"- fixed_best_expert: {_fmt(cmp_row.get('fixed_best_expert'))}",
            f"- fixed_best_equity_multiple: {_fmt(cmp_row.get('fixed_best_equity_multiple'))}",
            f"- oracle_equity_multiple: {_fmt(cmp_row.get('oracle_equity_multiple'))}",
            f"- regret_vs_fixed_best: {_fmt(cmp_row.get('regret_vs_fixed_best'))}",
            f"- regret_vs_oracle: {_fmt(cmp_row.get('regret_vs_oracle'))}",
            f"- stepf_win_days_vs_fixed_best: {_fmt(cmp_row.get('stepf_win_days_vs_fixed_best'))}",
            f"- stepf_pick_match_rate_vs_oracle: {_fmt(cmp_row.get('stepf_pick_match_rate_vs_oracle'))}",
        ])
    reward_cmp = stepf_cmp.get("stepF_reward_compare", {}) if isinstance(stepf_cmp.get("stepF_reward_compare", {}), dict) else {}
    lines.append(f"- StepFRewardCompare status: {reward_cmp.get('status', 'WARN')} ({reward_cmp.get('summary', 'NA')})")
    rrows = reward_cmp.get("rows", []) if isinstance(reward_cmp.get("rows", []), list) else []
    if rrows:
        lines.append("")
        lines.append("### StepFRewardCompare")
        lines.append("| name | equity_multiple | sharpe | max_dd | mean_ret | std_ret | regret_vs_fixed_best | regret_vs_oracle | win_days_vs_fixed_best | pick_match_rate_vs_oracle |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in rrows:
            lines.append(
                f"| {r.get('name', 'NA')} | {_fmt(r.get('equity_multiple'))} | {_fmt(r.get('sharpe'))} | {_fmt(r.get('max_dd'))} | {_fmt(r.get('mean_ret'))} | {_fmt(r.get('std_ret'))} | {_fmt(r.get('regret_vs_fixed_best'))} | {_fmt(r.get('regret_vs_oracle'))} | {_fmt(r.get('win_days_vs_fixed_best'))} | {_fmt(r.get('pick_match_rate_vs_oracle'))} |"
            )

    cl = stepf_cmp.get("cluster", {}) if isinstance(stepf_cmp.get("cluster", {}), dict) else {}
    lines.append(f"- cluster_status: {cl.get('status', 'PENDING')} ({cl.get('reason', 'cluster comparison pending')})")

    cluster_eval = report.get("cluster_eval", {}) if isinstance(report.get("cluster_eval", {}), dict) else {}
    lines.extend([
        "",
        "## ClusterEval",
        f"- status: **{cluster_eval.get('status', 'SKIP')}**",
        f"- summary: {cluster_eval.get('summary', cluster_eval.get('reason', 'NA'))}",
        f"- out_dir: {_fmt(cluster_eval.get('out_dir'))}",
        f"- stable_top_clusters: {_fmt(cluster_eval.get('stable_top_clusters'))}",
        f"- raw20_top_clusters: {_fmt(cluster_eval.get('raw20_top_clusters'))}",
    ])

    div = report.get("diversity", {})
    lines.extend([
        "",
        "## Diversity",
        f"- status: **{div.get('status', 'SKIP')}**",
        f"- summary: {div.get('summary', 'NA')}",
        f"- max_corr: {_fmt(div.get('max_corr'))}",
        f"- max_match_ratio: {_fmt(div.get('max_match_ratio'))}",
        f"- pairs_over_0_9999: {_fmt(div.get('pairs_over_0_9999'))} / {_fmt(div.get('all_pairs'))}",
        f"- identical_all_agents: {_fmt(div.get('identical_all_agents'))}",
    ])

    plots = report.get("plots", {})
    plot_items = plots.get("items", []) if isinstance(plots, dict) else []
    lines.extend([
        "",
        "## PLOTS",
    ])
    if plot_items:
        for item in plot_items:
            nm = item.get("name", "plot")
            lines.append(f"- [{nm}](./{nm})")
    else:
        lines.append("- (no plots)")
    for note in (plots.get("notes", []) if isinstance(plots, dict) else []):
        lines.append(f"  - note: {note}")

    lines.extend([
        "",
        "## Raw JSON",
        "```json",
        json.dumps(report, ensure_ascii=False, indent=2),
        "```",
        "",
        "Best-effort mode: this evaluator writes SKIP/notes and always exits 0.",
    ])
    return "\n".join(lines)


def render_summary(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"status={report.get('overall_status', 'WARN')}")

    stepa = report.get("stepA", {})
    ad = stepa.get("details", {})
    lines.append("StepA:")
    if stepa.get("status") == "SKIP":
        lines.append(f"  SKIP: {stepa.get('summary')}")
    else:
        lines.append(f"  test_rows={_fmt(ad.get('test_rows'))}")
        lines.append(f"  test_date_start={_fmt(ad.get('test_date_start'))} test_date_end={_fmt(ad.get('test_date_end'))}")
        lines.append(f"  missing_ohlcv_count={_fmt(ad.get('missing_ohlcv_count'))}")

    stepb = report.get("stepB", {})
    lines.append("StepB:")
    if stepb.get("status") == "SKIP":
        lines.append(f"  SKIP: {stepb.get('summary')}")
    else:
        files = stepb.get("files", [])
        show_files = files[:MAX_LIST_ITEMS]
        lines.append(f"  prediction_files_found={len(files)}")
        for fn in show_files:
            lines.append(f"    - {fn}")
        if len(files) > len(show_files):
            lines.append(f"    ... +{len(files) - len(show_files)} more")

        for r in stepb.get("rows", [])[:MAX_LIST_ITEMS * 2]:
            lines.append(
                "  "
                + f"{r.get('file')}::{r.get('pred_col')} nn_ratio={_fmt(r.get('non_null_ratio'))} "
                + f"first_valid_date={_fmt(r.get('first_valid_date'))} coverage_ratio_over_test={_fmt(r.get('coverage_ratio_over_test'))} "
                + f"mae={_fmt(r.get('mae'))} corr={_fmt(r.get('corr'))}"
            )

    dprime = report.get("dprime", {})
    ddet = dprime.get("details", {})
    lines.append("DPrimeCluster:")
    lines.append(f"  status={_fmt(ddet.get('cluster_status'))} summary={_fmt(ddet.get('cluster_summary'))}")
    lines.append(
        f"  cluster_embeddings_count={_fmt(ddet.get('cluster_embeddings_count'))} "
        f"cluster_state_count={_fmt(ddet.get('cluster_state_count'))} "
        f"cluster_input_count={_fmt(ddet.get('cluster_input_count'))}"
    )
    lines.append("DPrimeRL:")
    lines.append(f"  status={_fmt(ddet.get('rl_status'))} summary={_fmt(ddet.get('rl_summary'))}")
    lines.append(
        f"  rl_state_count={_fmt(ddet.get('rl_state_count'))} "
        f"rl_profiles_count={_fmt(ddet.get('rl_profiles_count'))}"
    )
    lines.append(
        f"  traceback_count={_fmt(ddet.get('traceback_count'))} "
        f"failure_summary_count={_fmt(ddet.get('failure_summary_count'))} "
        f"failure_reason={_fmt(ddet.get('failure_reason'))}"
    )

    stepe = report.get("stepE", {})
    lines.append("StepE:")
    if stepe.get("status") == "SKIP":
        lines.append(f"  SKIP: {stepe.get('summary')}")
    else:
        for r in stepe.get("rows", [])[:MAX_LIST_ITEMS]:
            lines.append(
                f"  {r.get('agent', 'NA')}::{r.get('file')} test_days={_fmt(r.get('test_days'))} equity_multiple={_fmt(r.get('equity_multiple'))} "
                f"max_dd={_fmt(r.get('max_dd'))} mean_ret={_fmt(r.get('mean_ret'))} std_ret={_fmt(r.get('std_ret'))} sharpe={_fmt(r.get('sharpe'))}"
            )
            if r.get("reason"):
                lines.append(f"    reason={r.get('reason')}")
            if r.get("note"):
                lines.append(f"    note={r.get('note')}")

    stepf = report.get("stepF", {})
    lines.append("StepF:")
    if stepf.get("status") == "SKIP":
        lines.append(f"  SKIP: {stepf.get('summary')}")
    else:
        for r in stepf.get("rows", [])[:MAX_LIST_ITEMS]:
            lines.append(
                f"  {r.get('file')} test_days={_fmt(r.get('test_days'))} split_source={_fmt(r.get('split_source'))} equity_multiple={_fmt(r.get('equity_multiple'))} "
                f"max_dd={_fmt(r.get('max_dd'))} mean_ret={_fmt(r.get('mean_ret'))} std_ret={_fmt(r.get('std_ret'))} sharpe={_fmt(r.get('sharpe'))}"
            )
            if r.get("reason"):
                lines.append(f"    reason={r.get('reason')}")
            if r.get("note"):
                lines.append(f"    note={r.get('note')}")

    stepf_cmp = report.get("stepF_compare", {}) if isinstance(report.get("stepF_compare", {}), dict) else {}
    cmp_row = stepf_cmp.get("row", {}) if isinstance(stepf_cmp.get("row", {}), dict) else {}
    lines.append("StepFCompare:")
    lines.append(f"  status={stepf_cmp.get('status', 'WARN')} summary={stepf_cmp.get('summary', stepf_cmp.get('reason', 'NA'))}")
    if cmp_row:
        lines.append(f"  current_stepf_equity_multiple={_fmt(cmp_row.get('current_stepf_equity_multiple'))}")
        lines.append(f"  fixed_best_expert={_fmt(cmp_row.get('fixed_best_expert'))}")
        lines.append(f"  fixed_best_equity_multiple={_fmt(cmp_row.get('fixed_best_equity_multiple'))}")
        lines.append(f"  oracle_equity_multiple={_fmt(cmp_row.get('oracle_equity_multiple'))}")
        lines.append(f"  regret_vs_fixed_best={_fmt(cmp_row.get('regret_vs_fixed_best'))}")
        lines.append(f"  regret_vs_oracle={_fmt(cmp_row.get('regret_vs_oracle'))}")
        lines.append(f"  stepf_win_days_vs_fixed_best={_fmt(cmp_row.get('stepf_win_days_vs_fixed_best'))}")
        lines.append(f"  stepf_pick_match_rate_vs_oracle={_fmt(cmp_row.get('stepf_pick_match_rate_vs_oracle'))}")
    reward_cmp = stepf_cmp.get("stepF_reward_compare", {}) if isinstance(stepf_cmp.get("stepF_reward_compare", {}), dict) else {}
    lines.append(f"  StepFRewardCompare status={reward_cmp.get('status', 'WARN')} summary={reward_cmp.get('summary', 'NA')}")
    for r in (reward_cmp.get("rows", [])[:8] if isinstance(reward_cmp.get("rows", []), list) else []):
        lines.append(
            f"    {r.get('name')} equity_multiple={_fmt(r.get('equity_multiple'))} regret_vs_fixed_best={_fmt(r.get('regret_vs_fixed_best'))} pick_match_rate_vs_oracle={_fmt(r.get('pick_match_rate_vs_oracle'))}"
        )

    cluster_eval = report.get("cluster_eval", {}) if isinstance(report.get("cluster_eval", {}), dict) else {}
    lines.append("ClusterEval:")
    lines.append(f"  status={cluster_eval.get('status', 'SKIP')} summary={cluster_eval.get('summary', cluster_eval.get('reason', 'NA'))}")
    lines.append(f"  out_dir={_fmt(cluster_eval.get('out_dir'))}")
    lines.append(f"  stable_top_clusters={_fmt(cluster_eval.get('stable_top_clusters'))}")
    lines.append(f"  raw20_top_clusters={_fmt(cluster_eval.get('raw20_top_clusters'))}")

    div = report.get("diversity", {})
    lines.append("Diversity:")
    lines.append(
        "  "
        + f"status={div.get('status', 'SKIP')} max_corr={_fmt(div.get('max_corr'))} "
        + f"max_match_ratio={_fmt(div.get('max_match_ratio'))} "
        + f"pairs_over_0_9999={_fmt(div.get('pairs_over_0_9999'))}/{_fmt(div.get('all_pairs'))} "
        + f"identical_all_agents={_fmt(div.get('identical_all_agents'))}"
    )
    lines.append(f"  summary={div.get('summary', 'NA')}")

    plots = report.get("plots", {}) if isinstance(report.get("plots", {}), dict) else {}
    lines.append("PLOTS:")
    for item in plots.get("items", []):
        exists = "yes" if item.get("exists") else "no"
        reason = item.get("reason") or ""
        lines.append(f"  {item.get('name')} exists={exists} {reason}".rstrip())
    for note in plots.get("notes", []):
        lines.append(f"  note={note}")

    text = "\n".join(lines)
    sliced = text[:MAX_SUMMARY_CHARS]
    out_lines = sliced.splitlines()[:MAX_SUMMARY_LINES]
    return "\n".join(out_lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--mode", default="sim")
    ap.add_argument("--symbol", default="SOXL")
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-summary", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_md)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_summary)), exist_ok=True)

    try:
        report = evaluate(args.output_root, args.mode, args.symbol)
        _generate_plots(
            output_root=args.output_root,
            mode=args.mode,
            symbol=args.symbol,
            report=report,
            out_dir=os.path.dirname(os.path.abspath(args.out_md)),
        )
        md = render_markdown(report)
        summary = render_summary(report)
    except Exception as exc:
        report = {
            "output_root": args.output_root,
            "mode": args.mode,
            "symbol": args.symbol,
            "overall_status": "WARN",
            "error": str(exc),
            "traceback": traceback.format_exc(limit=4),
        }
        md = "# EVAL_REPORT\n\nEvaluator failed but continued in best-effort mode.\n\n```\n" + report["traceback"] + "\n```\n"
        summary = f"status=WARN\nStepA:\n  SKIP: evaluator exception={exc}\nStepB:\n  SKIP: evaluator exception\nStepE:\n  SKIP: evaluator exception\nStepF:\n  SKIP: evaluator exception"

    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(md)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(args.out_summary, "w", encoding="utf-8") as f:
        f.write(summary)

    try:
        _write_eval_tables(report, os.path.dirname(os.path.abspath(args.out_md)))
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
    raise SystemExit(0)
