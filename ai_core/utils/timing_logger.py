from __future__ import annotations

import json
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
        self.host = _get_host()

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

        if not self.events_path.exists():
            self.events_path.touch()
        if not self.timings_csv_path.exists():
            pd.DataFrame(
                columns=[
                    "run_id",
                    "mode",
                    "retrain",
                    "branch_id",
                    "step",
                    "section",
                    "agent_id",
                    "started_at",
                    "ended_at",
                    "elapsed_sec",
                    "host",
                    "git_commit",
                ]
            ).to_csv(self.timings_csv_path, index=False)

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

    def emit(self, stage: str, elapsed_ms: float, agent_id: str = "", meta: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled:
            return
        event = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "mode": self.mode,
            "branch_id": self.branch_id,
            "execution_mode": self.execution_mode,
            "stage": str(stage),
            "agent_id": str(agent_id or ""),
            "elapsed_ms": float(elapsed_ms),
            "meta": meta or {},
        }
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def emit_timing_row(self, *, stage: str, started_at: datetime, ended_at: datetime, elapsed_sec: float, agent_id: str = "") -> None:
        if not self.enabled:
            return
        step, section = self._infer_step_and_section(stage)
        row = {
            "run_id": self.run_id,
            "mode": self.mode,
            "retrain": self.retrain,
            "branch_id": self.branch_id,
            "step": step,
            "section": section,
            "agent_id": str(agent_id or ""),
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "elapsed_sec": float(elapsed_sec),
            "host": self.host,
            "git_commit": "",
        }
        df = pd.DataFrame([row])
        write_header = not self.timings_csv_path.exists()
        df.to_csv(self.timings_csv_path, mode="a", header=write_header, index=False)

    def write_summaries(self) -> None:
        if not self.enabled:
            return
        if not self.timings_csv_path.exists():
            return
        df = pd.read_csv(self.timings_csv_path)
        required = ["step", "section", "agent_id", "elapsed_sec"]
        for col in required:
            if col not in df.columns:
                df[col] = "" if col != "elapsed_sec" else 0.0
        if df.empty:
            pd.DataFrame(
                columns=["step", "section", "count", "elapsed_sec_sum", "elapsed_sec_mean", "elapsed_sec_max"]
            ).to_csv(self.summary_step_csv_path, index=False)
            pd.DataFrame(
                columns=["step", "section", "agent_id", "count", "elapsed_sec_sum", "elapsed_sec_mean", "elapsed_sec_max"]
            ).to_csv(self.summary_agent_csv_path, index=False)
            return

        step_summary = (
            df.groupby(["step", "section"], dropna=False)["elapsed_sec"]
            .agg(count="count", elapsed_sec_sum="sum", elapsed_sec_mean="mean", elapsed_sec_max="max")
            .reset_index()
        )
        step_summary.to_csv(self.summary_step_csv_path, index=False)

        agent_df = df[df["agent_id"].fillna("").astype(str) != ""].copy()
        agent_summary = (
            agent_df.groupby(["step", "section", "agent_id"], dropna=False)["elapsed_sec"]
            .agg(count="count", elapsed_sec_sum="sum", elapsed_sec_mean="mean", elapsed_sec_max="max")
            .reset_index()
        ) if not agent_df.empty else pd.DataFrame(
            columns=["step", "section", "agent_id", "count", "elapsed_sec_sum", "elapsed_sec_mean", "elapsed_sec_max"]
        )
        agent_summary.to_csv(self.summary_agent_csv_path, index=False)

    @contextmanager
    def stage(self, stage: str, agent_id: str = "", meta: Optional[Dict[str, Any]] = None) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            ended_at = datetime.now(timezone.utc)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.emit(stage=stage, elapsed_ms=elapsed_ms, agent_id=agent_id, meta=meta)
            started_at = ended_at.timestamp() - (elapsed_ms / 1000.0)
            started_dt = datetime.fromtimestamp(started_at, tz=timezone.utc)
            self.emit_timing_row(
                stage=stage,
                started_at=started_dt,
                ended_at=ended_at,
                elapsed_sec=elapsed_ms / 1000.0,
                agent_id=agent_id,
            )
