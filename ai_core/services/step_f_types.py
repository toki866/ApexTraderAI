from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Tuple

import yaml


@dataclass
class StepFAuditConfig:
    """Configuration for StepF audit and A/B diagnostic artifacts.

    StepF consumes upstream D'_cluster assignments; this config only controls
    how router-side audits and comparisons are generated from those consumed
    inputs.
    """

    stable_primary: bool = True
    use_raw20_aux: bool = True
    boundary_margin_threshold: float = 0.0015
    boundary_confidence_threshold: float = 0.55
    boundary_uncertainty_threshold: float = 0.45
    focus_regimes: Tuple[int, ...] = (8,)
    no_cluster_baseline_kind: str = "adaptive_ewma"
    enable_soft_routing_experiment: bool = True
    enable_fallback_experiment: bool = True
    rare_candidate_expansion: int = 2
    output_subdir: str = "audit"
    overwrite: bool = True
    save_policy: str = "overwrite"
    bootstrap_samples: int = 20
    bootstrap_seed: int = 42
    soft_routing_margin_threshold: float = 0.0015
    soft_routing_confidence_threshold: float = 0.55
    soft_routing_top2_min_weight: float = 0.35
    soft_routing_normalization: str = "softmax"
    soft_routing_max_candidates: int = 2
    soft_routing_churn_reference: float = 0.25
    fallback_penalty: float = 0.15
    fallback_confidence_expansion_threshold: float = 0.55
    fallback_margin_expansion_threshold: float = 0.0015
    fallback_rare_only_extra_candidates: int = 2
    adaptive_baseline_eta: float = 4.0
    adaptive_baseline_alpha: float = 0.35
    notes: str = ""

    @classmethod
    def from_mapping(cls, raw: dict | None) -> "StepFAuditConfig":
        raw = raw or {}
        init_keys = {name for name, f in cls.__dataclass_fields__.items() if f.init}  # type: ignore[attr-defined]
        payload = {k: v for k, v in raw.items() if k in init_keys}
        focus_regimes = payload.get("focus_regimes")
        if isinstance(focus_regimes, Iterable) and not isinstance(focus_regimes, (str, bytes, tuple)):
            payload["focus_regimes"] = tuple(int(x) for x in focus_regimes)
        return cls(**payload)

    @classmethod
    def from_yaml(cls, path: str | Path | None) -> "StepFAuditConfig":
        if not path:
            return cls()
        cfg_path = Path(path)
        if not cfg_path.exists():
            return cls()
        with cfg_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            return cls()
        nested = raw.get("stepf_audit") if isinstance(raw.get("stepf_audit"), dict) else raw
        return cls.from_mapping(nested)


@dataclass
class StepFDiagnosticBundle:
    merged: object
    daily: object
    cluster_context: object
    edge_table: object
    allowlist: object
    agents: Tuple[str, ...] = field(default_factory=tuple)
    symbol: str = ""
    mode: str = "sim"
    stepf_root: str = ""
    audit_root: str = ""
