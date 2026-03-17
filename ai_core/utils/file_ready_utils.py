from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


_STATUSES = ("READY", "RUNNING", "FAILED")


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(text, encoding=encoding)
    os.replace(tmp, path)


def write_status_marker(marker_dir: Path, name: str, status: str, extra: Optional[Dict[str, Any]] = None) -> Path:
    status_norm = str(status or "").strip().upper()
    if status_norm not in _STATUSES:
        raise ValueError(f"invalid status={status}; expected one of {_STATUSES}")
    marker_dir.mkdir(parents=True, exist_ok=True)
    for st in _STATUSES:
        stale = marker_dir / f"{name}.{st}.json"
        if stale.exists():
            stale.unlink()
    payload: Dict[str, Any] = {"name": str(name), "status": status_norm, "ts": float(time.time())}
    if extra:
        payload.update(extra)
    out = marker_dir / f"{name}.{status_norm}.json"
    atomic_write_text(out, json.dumps(payload, ensure_ascii=False, indent=2))
    return out

