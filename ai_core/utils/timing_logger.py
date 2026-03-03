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
        self.host = _get_host()

        if not self.enabled:
            return

        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        if clear and self.events_path.exists():
            self.events_path.unlink()
        if clear and self.timings_csv_path.exists():
            self.timings_csv_path.unlink()

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

    def emit_timing_row(self, *, stage: str, started_at: datetime, ended_at: datetime, elapsed_sec: float) -> None:
        if not self.enabled:
            return
        step = "StepF"
        section = str(stage)
        if "." in section:
            prefix, section = section.split(".", 1)
            step = f"Step{prefix[-1].upper()}" if prefix.lower().startswith("step") else prefix
        row = {
            "run_id": self.run_id,
            "mode": self.mode,
            "retrain": self.retrain,
            "branch_id": self.branch_id,
            "step": step,
            "section": section,
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "elapsed_sec": float(elapsed_sec),
            "host": self.host,
            "git_commit": "",
        }
        df = pd.DataFrame([row])
        write_header = not self.timings_csv_path.exists()
        df.to_csv(self.timings_csv_path, mode="a", header=write_header, index=False)

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
            self.emit_timing_row(stage=stage, started_at=started_dt, ended_at=ended_at, elapsed_sec=elapsed_ms / 1000.0)
