# ai_core/types/step_b_types.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional


@dataclass(init=False)
class StepBAgentResult:
    """Result for a single StepB agent (Mamba).

    This class is intentionally *keyword-flexible* for backward/forward compatibility.
    Runners in this repo have historically evolved and may pass additional keyword args
    (e.g., success/out_dir/pred paths/etc.).

    Policy:
    - Known fields are stored on the instance.
    - Unknown keyword args are stored into `info` (so they are not lost).
    """

    # --- Status ---
    success: bool = True
    message: str = ""

    # --- Identity ---
    agent: str = ""     # e.g., "xsr" / "mamba" / "fedformer"
    symbol: str = ""    # e.g., "SOXL"

    # --- Outputs ---
    out_dir: str = ""                      # agent output directory (if any)
    output_path: str = ""                  # primary CSV path (if any)
    artifacts: Dict[str, Any] = field(default_factory=dict)  # additional files (csv/model/logs/etc.)

    # --- Optional metadata ---
    metrics: Dict[str, Any] = field(default_factory=dict)    # training metrics, scores, etc.
    info: Dict[str, Any] = field(default_factory=dict)       # any extra info

    def __init__(
        self,
        *,
        success: bool = True,
        message: str = "",
        agent: str = "",
        symbol: str = "",
        out_dir: str = "",
        output_path: str = "",
        artifacts: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        # Set known fields
        self.success = bool(success)
        self.message = str(message) if message is not None else ""
        self.agent = str(agent) if agent is not None else ""
        self.symbol = str(symbol) if symbol is not None else ""
        self.out_dir = str(out_dir) if out_dir is not None else ""
        self.output_path = str(output_path) if output_path is not None else ""

        self.artifacts = dict(artifacts) if isinstance(artifacts, dict) else {}
        self.metrics = dict(metrics) if isinstance(metrics, dict) else {}
        self.info = dict(info) if isinstance(info, dict) else {}

        # Backward/forward compatibility: accept common alternative names
        # and merge into known fields if provided.
        if "output_dir" in kwargs and not self.out_dir:
            try:
                self.out_dir = str(kwargs.pop("output_dir"))
            except Exception:
                pass

        if "pred_path" in kwargs and not self.output_path:
            # Some older code used pred_path as primary output.
            try:
                self.output_path = str(kwargs.pop("pred_path"))
            except Exception:
                pass

        # Store remaining unknown kwargs so runners can evolve without breaking.
        if kwargs:
            # Prefer keeping file-like keys under artifacts if they look like paths
            # but keep everything else in info.
            for k, v in list(kwargs.items()):
                if isinstance(v, str) and (k.endswith("_path") or k.endswith("_csv") or k.endswith("_file")):
                    self.artifacts.setdefault(k, v)
                else:
                    self.info.setdefault(k, v)

    def to_dict(self) -> Dict[str, Any]:
        # asdict works with dataclasses; init=False still qualifies.
        return asdict(self)


@dataclass
class StepBResult:
    """Overall StepB result container."""

    success: bool = True
    message: str = ""

    out_dir: str = ""
    pred_time_all_path: str = ""
    split_summary_path: str = ""
    pred_time_full_path: str = ""
    pred_time_periodic_path: str = ""
    pred_future_periodic_path: str = ""
    pred_nextday_full_path: str = ""
    rollout63_full_path: str = ""

    # agent_results maps keys like "mamba" to StepBAgentResult
    agent_results: Dict[str, StepBAgentResult] = field(default_factory=dict)

    # Optional metadata
    info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        try:
            d["agent_results"] = {
                k: (v.to_dict() if hasattr(v, "to_dict") else dict(v))
                for k, v in self.agent_results.items()
            }
        except Exception:
            pass
        return d
