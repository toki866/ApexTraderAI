from __future__ import annotations

import json
import math
import os
import platform
import socket
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import pandas as pd


def _get_host() -> str:
    try:
        node = platform.node()
        if node:
            return node
    except Exception:
        pass

    try:
        hostname = socket.gethostname()
        if hostname:
            return hostname
    except Exception:
        pass

    return os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME") or "unknown"


class TimingLogger:
    CSV_COLUMNS = [
        "run_id",
        "mode",
        "retrain",
        "branch_id",
        "step",
        "section",
        "agent_id",
        "agent_kind",
        "run_type",
        "status",
        "symbol",
        "reward_mode",
        "profile_name",
        "expert_name",
        "fallback_used",
        "reused_steps",
        "skipped",
        "critical_path_sec",
        "started_at",
        "ended_at",
        "elapsed_sec",
        "host",
        "git_commit",
    ]

    def __init__(
        self,
        output_root: Path,
        mode: str,
        run_id: str,
        branch_id: str,
        execution_mode: str = "sequential",
        retrain: str = "",
        enabled: bool = False,
        clear: bool = False,
        run_type: str = "",
        symbol: str = "",
        reused_steps: Optional[list[str]] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.mode = str(mode)
        self.run_id = str(run_id)
        self.branch_id = str(branch_id)
        self.execution_mode = str(execution_mode or "sequential")
        self.retrain = str(retrain or "")
        self.events_path = Path(output_root) / "timing" / self.mode / "timing_events.jsonl"
        self.timings_csv_path = Path(output_root) / "timings.csv"
        self.timing_root = Path(output_root) / "timing"
        self.summary_step_csv_path = self.timing_root / "summary_step_elapsed.csv"
        self.summary_agent_csv_path = self.timing_root / "summary_agent_elapsed.csv"
        self.summary_branch_budget_csv_path = self.timing_root / "summary_branch_budget.csv"
        self.summary_live_start_budget_csv_path = self.timing_root / "summary_live_start_budget.csv"
        self.host = _get_host()
        self._base_meta: Dict[str, Any] = {
            "run_type": str(run_type or ""),
            "symbol": str(symbol or ""),
            "status": "success",
            "agent_kind": "none",
            "reward_mode": "",
            "profile_name": "",
            "expert_name": "",
            "fallback_used": False,
            "reused_steps": list(reused_steps or []),
            "skipped": False,
        }

        if not self.enabled:
            return

        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self.timing_root.mkdir(parents=True, exist_ok=True)
        if clear and self.events_path.exists():
            self.events_path.unlink()
        if clear and self.timings_csv_path.exists():
            self.timings_csv_path.unlink()
        if clear and self.summary_step_csv_path.exists():
            self.summary_step_csv_path.unlink()
        if clear and self.summary_agent_csv_path.exists():
            self.summary_agent_csv_path.unlink()
        if clear and self.summary_branch_budget_csv_path.exists():
            self.summary_branch_budget_csv_path.unlink()
        if clear and self.summary_live_start_budget_csv_path.exists():
            self.summary_live_start_budget_csv_path.unlink()

        if not self.events_path.exists():
            self.events_path.touch()
        if not self.timings_csv_path.exists():
            pd.DataFrame(columns=self.CSV_COLUMNS).to_csv(self.timings_csv_path, index=False)

    @staticmethod
    def _infer_step_and_section(stage: str) -> tuple[str, str]:
        section = str(stage or "")
        step = "StepUnknown"
        if "." in section:
            prefix, section = section.split(".", 1)
        else:
            prefix = section
            section = "total"
        pl = prefix.lower()
        if pl.startswith("stepdprime"):
            step = "StepDPrime"
        elif pl.startswith("step"):
            suffix = prefix[4:]
            if suffix:
                step = f"Step{suffix[0].upper()}{suffix[1:]}"
            else:
                step = "Step"
        elif pl.startswith("branch"):
            step = "Branch"
        elif pl.startswith("common"):
            step = "Common"
        elif pl.startswith("audit"):
            step = "Audit"
        return step, section

    @classmethod
    def disabled(cls) -> "TimingLogger":
        return cls(Path("."), "sim", "", "", enabled=False)

    def update_context(self, **kwargs: Any) -> None:
        if not kwargs:
            return
        for k, v in kwargs.items():
            if k == "reused_steps" and isinstance(v, str):
                self._base_meta[k] = [s.strip() for s in v.split(",") if s.strip()]
            else:
                self._base_meta[k] = v

    def mark_step_reused(self, step_key: str) -> None:
        vals = list(self._base_meta.get("reused_steps") or [])
        k = str(step_key or "").strip()
        if k and k not in vals:
            vals.append(k)
        self._base_meta["reused_steps"] = vals

    def _agent_meta(self, stage: str, agent_id: str, meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        merged = dict(self._base_meta)
        merged.update(meta or {})
        aid = str(agent_id or "")
        sl = str(stage or "").lower()
        if not merged.get("agent_kind") or str(merged.get("agent_kind")) == "none":
            if sl.startswith("stepdprimerl.profile"):
                merged["agent_kind"] = "profile"
                merged.setdefault("profile_name", aid)
            elif sl.startswith("stepe.agent"):
                merged["agent_kind"] = "expert"
                merged.setdefault("expert_name", aid)
            elif sl.startswith("stepf.reward_mode") or sl.startswith("stepf.router"):
                merged["agent_kind"] = "reward_mode"
                merged.setdefault("reward_mode", aid)
            else:
                merged["agent_kind"] = "none"
        if merged.get("agent_kind") == "profile" and aid and not merged.get("profile_name"):
            merged["profile_name"] = aid
        if merged.get("agent_kind") == "expert" and aid and not merged.get("expert_name"):
            merged["expert_name"] = aid
        if merged.get("agent_kind") == "reward_mode" and aid and not merged.get("reward_mode"):
            merged["reward_mode"] = aid
        merged["reused_steps"] = merged.get("reused_steps") or []
        merged["fallback_used"] = bool(merged.get("fallback_used", False))
        merged["skipped"] = bool(merged.get("skipped", False))
        merged["status"] = str(merged.get("status") or "success")
        return merged

    def emit(self, stage: str, elapsed_ms: float, agent_id: str = "", meta: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled:
            return
        merged = self._agent_meta(stage=stage, agent_id=agent_id, meta=meta)
        event = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "mode": self.mode,
            "branch_id": self.branch_id,
            "execution_mode": self.execution_mode,
            "stage": str(stage),
            "agent_id": str(agent_id or ""),
            "elapsed_ms": float(elapsed_ms),
            "meta": merged,
        }
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def emit_timing_row(
        self,
        *,
        stage: str,
        started_at: datetime,
        ended_at: datetime,
        elapsed_sec: float,
        agent_id: str = "",
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return
        step, section = self._infer_step_and_section(stage)
        merged = self._agent_meta(stage=stage, agent_id=agent_id, meta=meta)
        row = {
            "run_id": self.run_id,
            "mode": self.mode,
            "retrain": self.retrain,
            "branch_id": self.branch_id,
            "step": step,
            "section": section,
            "agent_id": str(agent_id or ""),
            "agent_kind": str(merged.get("agent_kind") or "none"),
            "run_type": str(merged.get("run_type") or ""),
            "status": str(merged.get("status") or "success"),
            "symbol": str(merged.get("symbol") or ""),
            "reward_mode": str(merged.get("reward_mode") or ""),
            "profile_name": str(merged.get("profile_name") or ""),
            "expert_name": str(merged.get("expert_name") or ""),
            "fallback_used": bool(merged.get("fallback_used", False)),
            "reused_steps": json.dumps(list(merged.get("reused_steps") or []), ensure_ascii=False),
            "skipped": bool(merged.get("skipped", False)),
            "critical_path_sec": float(merged.get("critical_path_sec") or elapsed_sec),
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "elapsed_sec": float(elapsed_sec),
            "host": self.host,
            "git_commit": "",
        }
        df = pd.DataFrame([row], columns=self.CSV_COLUMNS)
        write_header = not self.timings_csv_path.exists()
        df.to_csv(self.timings_csv_path, mode="a", header=write_header, index=False)

    def emit_instant(self, *, stage: str, status: str, agent_id: str = "", meta: Optional[Dict[str, Any]] = None) -> None:
        ended_at = datetime.now(timezone.utc)
        started_at = ended_at
        merged = dict(meta or {})
        merged["status"] = status
        if status == "skipped":
            merged["skipped"] = True
        self.emit(stage=stage, elapsed_ms=0.0, agent_id=agent_id, meta=merged)
        self.emit_timing_row(
            stage=stage,
            started_at=started_at,
            ended_at=ended_at,
            elapsed_sec=0.0,
            agent_id=agent_id,
            meta=merged,
        )

    def _pct(self, s: pd.Series, q: float) -> float:
        if s.empty:
            return 0.0
        return float(s.quantile(q))

    def write_summaries(self) -> None:
        if not self.enabled:
            return
        if not self.timings_csv_path.exists():
            return
        df = pd.read_csv(self.timings_csv_path)
        required_defaults: Dict[str, Any] = {
            "step": "",
            "section": "",
            "agent_id": "",
            "agent_kind": "none",
            "symbol": "",
            "run_type": "",
            "status": "success",
            "reward_mode": "",
            "profile_name": "",
            "expert_name": "",
            "fallback_used": False,
            "critical_path_sec": 0.0,
            "elapsed_sec": 0.0,
        }
        for col, default in required_defaults.items():
            if col not in df.columns:
                df[col] = default
        if df.empty:
            pd.DataFrame(
                columns=["step", "section", "count", "elapsed_sec_sum", "elapsed_sec_mean", "elapsed_sec_min", "elapsed_sec_p50", "elapsed_sec_p95", "elapsed_sec_max"]
            ).to_csv(self.summary_step_csv_path, index=False)
            pd.DataFrame(
                columns=["step", "section", "agent_id", "agent_kind", "reward_mode", "profile_name", "expert_name", "count", "elapsed_sec_sum", "elapsed_sec_mean", "elapsed_sec_min", "elapsed_sec_p50", "elapsed_sec_p95", "elapsed_sec_max"]
            ).to_csv(self.summary_agent_csv_path, index=False)
            pd.DataFrame(
                columns=["symbol", "run_type", "step", "section", "agent_kind", "agent_id", "n_runs", "elapsed_mean_sec", "elapsed_p50_sec", "elapsed_p95_sec", "elapsed_max_sec", "critical_path_p95_sec", "fallback_used_any"]
            ).to_csv(self.summary_branch_budget_csv_path, index=False)
            pd.DataFrame(
                columns=["symbol", "run_type", "branch_id", "target_stage", "elapsed_p95_sec", "recommended_buffer_sec", "recommended_start_offset_sec", "formula_note"]
            ).to_csv(self.summary_live_start_budget_csv_path, index=False)
            return

        step_summary = (
            df.groupby(["step", "section"], dropna=False)["elapsed_sec"]
            .agg(
                count="count",
                elapsed_sec_sum="sum",
                elapsed_sec_mean="mean",
                elapsed_sec_min="min",
                elapsed_sec_p50=lambda s: self._pct(s, 0.50),
                elapsed_sec_p95=lambda s: self._pct(s, 0.95),
                elapsed_sec_max="max",
            )
            .reset_index()
        )
        step_summary.to_csv(self.summary_step_csv_path, index=False)

        agent_df = df[df["agent_id"].fillna("").astype(str) != ""].copy()
        group_cols = ["step", "section", "agent_id", "agent_kind", "reward_mode", "profile_name", "expert_name"]
        agent_summary = (
            agent_df.groupby(group_cols, dropna=False)["elapsed_sec"]
            .agg(
                count="count",
                elapsed_sec_sum="sum",
                elapsed_sec_mean="mean",
                elapsed_sec_min="min",
                elapsed_sec_p50=lambda s: self._pct(s, 0.50),
                elapsed_sec_p95=lambda s: self._pct(s, 0.95),
                elapsed_sec_max="max",
            )
            .reset_index()
            if not agent_df.empty
            else pd.DataFrame(columns=[*group_cols, "count", "elapsed_sec_sum", "elapsed_sec_mean", "elapsed_sec_min", "elapsed_sec_p50", "elapsed_sec_p95", "elapsed_sec_max"])
        )
        agent_summary.to_csv(self.summary_agent_csv_path, index=False)

        budget_group_cols = ["symbol", "run_type", "step", "section", "agent_kind", "agent_id"]
        branch_budget = (
            df.groupby(budget_group_cols, dropna=False)
            .agg(
                n_runs=("elapsed_sec", "count"),
                elapsed_mean_sec=("elapsed_sec", "mean"),
                elapsed_p50_sec=("elapsed_sec", lambda s: self._pct(s, 0.50)),
                elapsed_p95_sec=("elapsed_sec", lambda s: self._pct(s, 0.95)),
                elapsed_max_sec=("elapsed_sec", "max"),
                critical_path_p95_sec=("critical_path_sec", lambda s: self._pct(s, 0.95)),
                fallback_used_any=("fallback_used", "max"),
            )
            .reset_index()
        )
        branch_budget["fallback_used_any"] = branch_budget["fallback_used_any"].astype(bool)
        branch_budget.to_csv(self.summary_branch_budget_csv_path, index=False)

        live_src = df[(df["status"].astype(str).str.lower() == "success") & (df["critical_path_sec"].astype(float) >= 0.0)].copy()
        live_group_cols = ["symbol", "run_type", "branch_id", "step", "section"]
        live_summary = (
            live_src.groupby(live_group_cols, dropna=False)["critical_path_sec"]
            .agg(elapsed_p95_sec=lambda s: self._pct(s, 0.95))
            .reset_index()
            if not live_src.empty
            else pd.DataFrame(columns=[*live_group_cols, "elapsed_p95_sec"])
        )
        if not live_summary.empty:
            live_summary["target_stage"] = live_summary["step"].astype(str) + "." + live_summary["section"].astype(str)
            live_summary["recommended_buffer_sec"] = live_summary["elapsed_p95_sec"].apply(
                lambda x: float(max(60, int(math.ceil(float(x) * 0.2))))
            )
            live_summary["recommended_start_offset_sec"] = (
                live_summary["elapsed_p95_sec"].astype(float) + live_summary["recommended_buffer_sec"].astype(float)
            )
            live_summary["formula_note"] = "recommended_buffer_sec=max(60,ceil(elapsed_p95_sec*0.2)); recommended_start_offset_sec=elapsed_p95_sec+recommended_buffer_sec"
            live_summary = live_summary[[
                "symbol",
                "run_type",
                "branch_id",
                "target_stage",
                "elapsed_p95_sec",
                "recommended_buffer_sec",
                "recommended_start_offset_sec",
                "formula_note",
            ]]
        else:
            live_summary = pd.DataFrame(
                columns=["symbol", "run_type", "branch_id", "target_stage", "elapsed_p95_sec", "recommended_buffer_sec", "recommended_start_offset_sec", "formula_note"]
            )
        live_summary.to_csv(self.summary_live_start_budget_csv_path, index=False)

    @contextmanager
    def stage(self, stage: str, agent_id: str = "", meta: Optional[Dict[str, Any]] = None) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        status = "success"
        try:
            yield
        except Exception:
            status = "fail"
            raise
        finally:
            ended_at = datetime.now(timezone.utc)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            merged = dict(meta or {})
            merged["status"] = str(merged.get("status") or status)
            self.emit(stage=stage, elapsed_ms=elapsed_ms, agent_id=agent_id, meta=merged)
            started_at = ended_at.timestamp() - (elapsed_ms / 1000.0)
            started_dt = datetime.fromtimestamp(started_at, tz=timezone.utc)
            self.emit_timing_row(
                stage=stage,
                started_at=started_dt,
                ended_at=ended_at,
                elapsed_sec=elapsed_ms / 1000.0,
                agent_id=agent_id,
                meta=merged,
            )
