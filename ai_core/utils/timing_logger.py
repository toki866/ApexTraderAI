from __future__ import annotations

import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


class TimingLogger:
    def __init__(
        self,
        output_root: Path,
        mode: str,
        run_id: str,
        branch_id: str,
        execution_mode: str = "sequential",
        enabled: bool = False,
        clear: bool = False,
    ) -> None:
        self.enabled = bool(enabled)
        self.mode = str(mode)
        self.run_id = str(run_id)
        self.branch_id = str(branch_id)
        self.execution_mode = str(execution_mode or "sequential")
        self.events_path = Path(output_root) / "timing" / self.mode / "timing_events.jsonl"

        if not self.enabled:
            return

        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        if clear and self.events_path.exists():
            self.events_path.unlink()

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

    @contextmanager
    def stage(self, stage: str, agent_id: str = "", meta: Optional[Dict[str, Any]] = None) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.emit(stage=stage, elapsed_ms=elapsed_ms, agent_id=agent_id, meta=meta)
