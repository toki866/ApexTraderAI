# NOTE: This is a local headless runner (no OpenAI/API calls). It runs Steps A-F and prints progress.
# -*- coding: utf-8 -*-
"""
run_steps_a_f_headless_safe3_lazy_import.py

Headless runner for soxl_rl_gui (StepA〜StepF).

Purpose
-------
- Run StepA〜StepF sequentially from the repository root (no GUI).
- Be robust to minor API/signature differences by using introspection.
- Support autodebug_app by reading AUTODEBUG_* environment variables.

How to run (examples)
---------------------
# simplest (symbol from env or default "SOXL")
python run_steps_a_f_headless_safe3_lazy_import.py

# explicit symbol
python run_steps_a_f_headless_safe3_lazy_import.py --symbol SOXL

# choose steps
python run_steps_a_f_headless_safe3_lazy_import.py --steps A,B

Notes
-----
- This script does not "hide" failures. Any exception will stop execution and return a non-zero exit code.
- Heavy modules are imported lazily (inside functions) to reduce import-time issues.
"""

from __future__ import annotations
import argparse
import dataclasses
import inspect
import json
import os
import re
import shutil
import sys
import time
import traceback
import subprocess
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, List

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ai_core.utils.paths import get_repo_root
from ai_core.utils.leak_audit_utils import (
    audit_stepE_reward_alignment,
    audit_stepF_market_alignment,
    audit_stepe_agent_now,
    audit_stepf_now,
    write_audit_reports,
)
from ai_core.utils.manifest_path_utils import normalize_output_artifact_path
from ai_core.utils.step_contract_utils import (
    validate_step_a,
    validate_step_b,
    validate_step_c,
    validate_step_dprime,
    validate_step_e_agent,
    validate_step_f,
)
from ai_core.utils.timing_logger import TimingLogger
class _DateRangeShim:
    """Wrapper for DateRange that can carry extra attributes like `future_end` safely."""

    def __init__(self, base: Any, **extras: Any) -> None:
        self._base = base
        self._extras = dict(extras)

    def __getattr__(self, name: str) -> Any:
        if name in self._extras:
            return self._extras[name]
        return getattr(self._base, name)

    def __repr__(self) -> str:
        return f"_DateRangeShim(base={self._base!r}, extras={self._extras!r})"

class _ConfigShim:
    """Compatibility wrapper passed as `config`/`cfg` to services.

    Some services expect `config.symbol` and `config.date_range` even when the global
    AppConfig does not contain those fields. This shim provides those fields while
    delegating all other attribute access to the wrapped base object.
    """

    def __init__(self, base: Any, **overrides: Any) -> None:
        self._base = base
        for k, v in overrides.items():
            if v is not None:
                setattr(self, k, v)

    def __getattr__(self, name: str) -> Any:
        try:
            return getattr(self._base, name)
        except AttributeError:
            # Provide a soft default for missing fields expected by some services (e.g. StepE/StepF).
            # Returning None lets the service apply its own defaults.
            return None



def _try_set(obj: Any, attr: str, value: Any) -> bool:
    """Best-effort setattr; returns True on success."""
    if hasattr(obj, attr):
        try:
            setattr(obj, attr, value)
            return True
        except Exception:
            return False
    return False


def _parse_int_list(v: Optional[str]) -> Optional[List[int]]:
    """Parse comma/space-separated ints like '1,5,10,20' or '1 5 10 20'."""
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    # allow both comma and spaces
    parts = s.replace(',', ' ').split()
    out: List[int] = []
    for p in parts:
        try:
            out.append(int(p))
        except Exception:
            continue
    out = [x for x in out if x >= 1]
    if not out:
        return None
    return sorted(set(out))


def _apply_mamba_overrides_to_stepb_config(cfg: Any, mamba_lookback: Optional[int], mamba_horizons: Optional[List[int]]) -> Any:
    """Apply Mamba overrides (lookback/horizons) to a real StepBConfig instance.

    IMPORTANT:
      - StepBService.run() requires an actual StepBConfig (isinstance check).
      - Therefore this function must NEVER return a wrapper object.

    Strategy (best-effort):
      1) Try setattr on cfg and cfg.mamba (if mutable)
      2) If dataclass and frozen/slots, use dataclasses.replace to create a new instance

    This keeps the returned object as a StepBConfig (or subclass) instance.
    """
    if mamba_lookback is None and mamba_horizons is None:
        return cfg

    def _is_dc(obj: Any) -> bool:
        return hasattr(obj, '__dataclass_fields__')

    # Normalize values
    lb = int(mamba_lookback) if mamba_lookback is not None else None
    hz = tuple(int(x) for x in (mamba_horizons or [])) if mamba_horizons is not None else None

    # --- Prefer nested cfg.mamba ---
    try:
        m = getattr(cfg, 'mamba', None)
    except Exception:
        m = None

    if m is not None:
        # Try direct set
        if lb is not None:
            for nm in ('lookback_days', 'lookback', 'seq_len', 'window', 'lookback_bdays'):
                if hasattr(m, nm) and _try_set(m, nm, lb):
                    break
        if hz is not None:
            for nm in ('horizons', 'horizon_list', 'horizon_days', 'future_horizons'):
                if hasattr(m, nm) and _try_set(m, nm, hz):
                    break
            # Keep periodic snapshot horizons in sync when present (periodic-only variant)
            for nm in ('periodic_snapshot_horizons', 'periodic_endpoints'):
                if hasattr(m, nm):
                    _try_set(m, nm, hz)


        # If dataclass (possibly frozen), try replace
        if _is_dc(m):
            kwargs = {}
            if lb is not None:
                for nm in ('lookback_days', 'lookback', 'seq_len', 'window', 'lookback_bdays'):
                    if nm in getattr(m, '__dataclass_fields__', {}):
                        kwargs[nm] = lb
                        break
            if hz is not None:
                for nm in ('horizons', 'horizon_list', 'horizon_days', 'future_horizons'):
                    if nm in getattr(m, '__dataclass_fields__', {}):
                        kwargs[nm] = hz
                        break
                # Keep periodic snapshot horizons in sync when those fields exist
                if 'periodic_snapshot_horizons' in getattr(m, '__dataclass_fields__', {}):
                    kwargs['periodic_snapshot_horizons'] = hz
                if 'periodic_endpoints' in getattr(m, '__dataclass_fields__', {}):
                    kwargs['periodic_endpoints'] = hz
            if kwargs:
                try:
                    m2 = dataclasses.replace(m, **kwargs)
                    # set back into cfg
                    if _is_dc(cfg) and 'mamba' in getattr(cfg, '__dataclass_fields__', {}):
                        try:
                            cfg = dataclasses.replace(cfg, mamba=m2)
                        except Exception:
                            pass
                    else:
                        _try_set(cfg, 'mamba', m2)
                except Exception:
                    pass

    # --- Also try top-level cfg fields (in case the project uses them) ---
    if lb is not None:
        for nm in ('lookback_days', 'lookback'):
            if hasattr(cfg, nm):
                _try_set(cfg, nm, lb)
    if hz is not None:
        for nm in ('horizons',):
            if hasattr(cfg, nm):
                _try_set(cfg, nm, hz)

    # Dataclass replace on cfg itself if needed
    if _is_dc(cfg):
        kwargs = {}
        if lb is not None:
            for nm in ('lookback_days', 'lookback'):
                if nm in getattr(cfg, '__dataclass_fields__', {}):
                    kwargs[nm] = lb
                    break
        if hz is not None:
            if 'horizons' in getattr(cfg, '__dataclass_fields__', {}):
                kwargs['horizons'] = hz
        if kwargs:
            try:
                cfg = dataclasses.replace(cfg, **kwargs)
            except Exception:
                pass

    return cfg

def _ensure_date_range_aliases(date_range: Any) -> Any:
    """Ensure DateRange instances provide .start and .end attributes.

    Many services expect DateRange.start/end for the full span. In this project,
    DateRange is typically constructed with train_start/train_end/test_start/test_end.
    This function adds class-level @property aliases when missing:
      - start -> train_start (fallbacks: test_start, start_date, etc.)
      - end   -> test_end   (fallbacks: train_end, end_date, etc.)

    The aliasing is intentionally defensive to tolerate minor schema differences.
    """
    cls = date_range.__class__

    def _pick(self: Any, names: Sequence[str]) -> Any:
        for n in names:
            if hasattr(self, n):
                v = getattr(self, n)
                # prefer non-None values
                if v is not None:
                    return v
        # last resort: try dataclass fields
        fields = getattr(self, "__dataclass_fields__", None)
        if fields:
            for n in names:
                if n in fields:
                    v = getattr(self, n)
                    if v is not None:
                        return v
        raise AttributeError(f"DateRange is missing expected attributes. available={sorted(set(dir(self)))[:50]}...")

    if not hasattr(cls, "start"):
        setattr(cls, "start", property(lambda self: _pick(self, ("train_start", "test_start", "start_date", "date_start", "begin", "from_date"))))
    if not hasattr(cls, "end"):
        setattr(cls, "end", property(lambda self: _pick(self, ("test_end", "train_end", "end_date", "date_end", "finish", "to_date"))))
    return date_range


def _env_get(*keys: str) -> Optional[str]:
    for k in keys:
        v = os.environ.get(k)
        if v:
            return v
    return None



def _repo_root() -> Path:
    return get_repo_root()


def _ensure_run_log_file(run_id: str, output_root: Path) -> Optional[Path]:
    """Create run_<run_id>.log under sibling logs/ directory (best-effort)."""
    try:
        run_id = (run_id or '').strip()
        if not run_id:
            return None
        run_dir = output_root.parent
        log_dir = run_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        run_log = log_dir / f'run_{run_id}.log'
        run_log.touch(exist_ok=True)
        with run_log.open('a', encoding='utf-8') as fh:
            fh.write(f"[PIPELINE] run_log_initialized run_id={run_id}\n")
            fh.write(f"[PIPELINE] output_root={output_root}\n")
        return run_log
    except Exception as e:
        print(f"[PIPELINE] WARN run log initialization failed: {type(e).__name__}: {e}")
        return None


def _append_run_log(run_log_path: Optional[Path], text: str) -> None:
    try:
        if run_log_path is None:
            return
        run_log_path.parent.mkdir(parents=True, exist_ok=True)
        with run_log_path.open("a", encoding="utf-8") as fh:
            fh.write(str(text))
            if not str(text).endswith("\n"):
                fh.write("\n")
    except Exception as exc:
        print(f"[PIPELINE] WARN run log append failed: {type(exc).__name__}: {exc}", file=sys.stderr)


def _coerce_existing_path(raw: Any) -> Optional[Path]:
    try:
        s = str(raw or "").strip()
    except Exception:
        return None
    if not s:
        return None

    direct = Path(s).expanduser()
    if direct.exists():
        return direct.resolve()

    if os.name != "nt":
        m = re.match(r"^(?P<drive>[A-Za-z]):[\\/](?P<rest>.*)$", s)
        if m:
            rest = m.group("rest").replace("\\", "/")
            translated = Path("/mnt") / m.group("drive").lower() / rest
            if translated.exists():
                return translated.resolve()
    return None


def _iter_existing_paths(candidates: Sequence[Any]) -> List[Path]:
    seen: set[str] = set()
    out: List[Path] = []
    for raw in candidates:
        p = _coerce_existing_path(raw)
        if p is None:
            continue
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _build_logs_manifest_item(
    *,
    kind: str,
    label: str,
    expected: bool,
    source_path: Optional[Path],
    destination_path: Optional[Path],
    copy_performed: bool,
    notes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "kind": kind,
        "label": label,
        "expected": bool(expected),
        "found": source_path is not None and source_path.exists(),
        "source_path": str(source_path) if source_path is not None else None,
        "destination_path": str(destination_path) if destination_path is not None else None,
        "copy_performed": bool(copy_performed),
        "size_bytes": None,
        "mtime_utc": None,
        "notes": list(notes or []),
    }
    stat_target = destination_path if destination_path is not None and destination_path.exists() else source_path
    if stat_target is not None and stat_target.exists():
        st = stat_target.stat()
        item["size_bytes"] = int(st.st_size)
        item["mtime_utc"] = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return item


def _aggregate_logs_to_output_root(
    *,
    output_root: Path,
    run_id: str,
    pipeline_status: str,
    error_text: Optional[str] = None,
) -> Optional[Path]:
    try:
        logs_dir = Path(output_root) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = logs_dir / "logs_manifest.json"

        github_workspace = os.environ.get("GITHUB_WORKSPACE", "")
        runner_temp = os.environ.get("RUNNER_TEMP", "")
        apex_run_id = os.environ.get("APEX_RUN_ID", "") or run_id
        env_run_log = os.environ.get("RUN_LOG_PATH", "")
        env_console_log = os.environ.get("RUN_CONSOLE_LOG", "")
        env_one_tap = os.environ.get("ONE_TAP_ERROR_REPORT_PATH", "")
        env_diag_zip = os.environ.get("APEX_DIAGNOSTICS_ZIP", "")
        env_diag_dir = os.environ.get("APEX_DIAGNOSTICS_DIR", "")

        session_logs_root = r"C:\work\apex_work\session_logs"
        run_logs_root = r"C:\work\apex_work\runs"
        run_dir_candidates: List[Any] = []
        if apex_run_id:
            run_dir_candidates.append(Path(run_logs_root) / apex_run_id)
        run_dir_candidates.append(Path(output_root).parent)
        existing_run_dirs = [p for p in _iter_existing_paths(run_dir_candidates) if p.is_dir()]

        items: List[Dict[str, Any]] = []

        def _copy_first(kind: str, label: str, expected: bool, candidates: Sequence[Any], destination_name: Optional[str] = None) -> None:
            existing = _iter_existing_paths(candidates)
            source_path = existing[0] if existing else None
            destination_path = logs_dir / (destination_name or (source_path.name if source_path is not None else label))
            copy_performed = False
            notes: List[str] = []
            if source_path is None:
                notes.append("source_not_found")
                destination_path = destination_path if destination_name else None
            else:
                if source_path.resolve() != destination_path.resolve():
                    shutil.copy2(source_path, destination_path)
                    copy_performed = True
                else:
                    notes.append("source_already_in_logs_dir")
            items.append(
                _build_logs_manifest_item(
                    kind=kind,
                    label=label,
                    expected=expected,
                    source_path=source_path,
                    destination_path=destination_path if source_path is not None or destination_name else None,
                    copy_performed=copy_performed,
                    notes=notes,
                )
            )

        run_log_candidates: List[Any] = [env_run_log]
        for run_dir in existing_run_dirs:
            run_log_candidates.append(run_dir / "logs" / f"run_{apex_run_id}.log")
            run_log_candidates.extend(sorted((run_dir / "logs").glob("run_*.log")))
        session_log_dir = _coerce_existing_path(session_logs_root)
        if session_log_dir is not None and session_log_dir.is_dir():
            run_log_candidates.extend(sorted(session_log_dir.glob(f"run_{apex_run_id}.log")))
            run_log_candidates.extend(sorted(session_log_dir.glob("run_*.log"), key=lambda p: p.stat().st_mtime, reverse=True))
        _copy_first("run_log", f"run_{apex_run_id}.log", True, run_log_candidates)

        console_candidates: List[Any] = [
            env_console_log,
            Path(github_workspace) / "temp" / "run_all_local_then_copy_console.log" if github_workspace else None,
            Path(runner_temp) / "run_all_local_then_copy_console.log" if runner_temp else None,
        ]
        _copy_first("console_log", "run_all_local_then_copy_console.log", True, console_candidates)

        one_tap_candidates: List[Any] = [
            env_one_tap,
            Path(github_workspace) / "temp" / "ONE_TAP_ERROR_REPORT.txt" if github_workspace else None,
        ]
        for run_dir in existing_run_dirs:
            one_tap_candidates.append(run_dir / "logs" / "ONE_TAP_ERROR_REPORT.txt")
        _copy_first("one_tap_error_report", "ONE_TAP_ERROR_REPORT.txt", True, one_tap_candidates)

        exec_candidates: List[Path] = []
        for base in _iter_existing_paths([
            Path(github_workspace) / "temp" if github_workspace else None,
            Path(runner_temp) if runner_temp else None,
            *(run_dir / "logs" for run_dir in existing_run_dirs),
        ]):
            if not base.is_dir():
                continue
            exec_candidates.extend(sorted(base.glob("step*_exec.log")))
        seen_exec: set[str] = set()
        for source_path in exec_candidates:
            key = str(source_path.resolve())
            if key in seen_exec:
                continue
            seen_exec.add(key)
            destination_path = logs_dir / source_path.name
            copy_performed = False
            notes: List[str] = []
            if source_path.resolve() != destination_path.resolve():
                shutil.copy2(source_path, destination_path)
                copy_performed = True
            else:
                notes.append("source_already_in_logs_dir")
            items.append(
                _build_logs_manifest_item(
                    kind="step_exec_log",
                    label=source_path.name,
                    expected=False,
                    source_path=source_path,
                    destination_path=destination_path,
                    copy_performed=copy_performed,
                    notes=notes,
                )
            )

        diag_candidates: List[Path] = []
        for diag_base in _iter_existing_paths([
            env_diag_zip,
            env_diag_dir,
            Path(runner_temp) / "ApexTraderAI" / "diagnostics" if runner_temp else None,
        ]):
            if diag_base.is_file() and diag_base.suffix.lower() == ".zip":
                diag_candidates.append(diag_base)
            elif diag_base.is_dir():
                diag_candidates.extend(sorted(diag_base.glob("diag_*.zip"), key=lambda p: p.stat().st_mtime, reverse=True))
        if diag_candidates:
            source_path = diag_candidates[0]
            destination_path = logs_dir / source_path.name
            copy_performed = False
            if source_path.resolve() != destination_path.resolve():
                shutil.copy2(source_path, destination_path)
                copy_performed = True
            items.append(
                _build_logs_manifest_item(
                    kind="diagnostics_zip",
                    label=source_path.name,
                    expected=False,
                    source_path=source_path,
                    destination_path=destination_path,
                    copy_performed=copy_performed,
                    notes=[],
                )
            )
        else:
            items.append(
                _build_logs_manifest_item(
                    kind="diagnostics_zip",
                    label="diag_latest.zip",
                    expected=False,
                    source_path=None,
                    destination_path=None,
                    copy_performed=False,
                    notes=["source_not_found"],
                )
            )

        manifest = {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "pipeline_status": pipeline_status,
            "error": error_text,
            "run_id": apex_run_id,
            "output_root": str(output_root),
            "logs_dir": str(logs_dir),
            "items": items,
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest_path
    except Exception as exc:
        print(f"[PIPELINE] WARN log aggregation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None



def snap_prev_by_prices(date, available_dates_sorted):
    """Snap to previous available trading date (floor). If none, use min."""
    import pandas as pd

    if available_dates_sorted is None or len(available_dates_sorted) == 0:
        return pd.to_datetime(date).normalize()
    ad = pd.Series(pd.to_datetime(available_dates_sorted, errors="coerce")).dropna().sort_values().drop_duplicates().reset_index(drop=True)
    if len(ad) == 0:
        return pd.to_datetime(date).normalize()
    d = pd.to_datetime(date).normalize()
    idx = ad.searchsorted(d, side="right") - 1
    if idx < 0:
        return pd.to_datetime(ad.iloc[0]).normalize()
    return pd.to_datetime(ad.iloc[int(idx)]).normalize()
def _parse_steps(s: str) -> Tuple[str, ...]:
    default_steps: Tuple[str, ...] = ("A", "B", "C", "DPRIME", "E", "F")
    canonical_order: Tuple[str, ...] = ("A", "B", "C", "DPRIME", "E", "F")

    step_aliases = {
        "A": "A",
        "B": "B",
        "C": "C",
        "D": "DPRIME",
        "E": "E",
        "F": "F",
        "DPRIME": "DPRIME",
        "DPR": "DPRIME",
        "D_PRIME": "DPRIME",
        "D-PRIME": "DPRIME",
        "D'": "DPRIME",
    }

    s = (s or "").strip()
    if not s:
        return default_steps

    parts = [p.strip().upper() for p in s.split(",") if p.strip()]
    if "D" in parts:
        print("[headless] INFO: StepD is deprecated; treating 'D' as 'DPRIME'.")
    if any(p in ("ALL", "*") for p in parts):
        return canonical_order

    requested = {step_aliases[p] for p in parts if p in step_aliases}
    if not requested:
        return default_steps

    return tuple(step for step in canonical_order if step in requested)






_OFFICIAL_STEPE_AGENTS: Tuple[str, ...] = (
    "dprime_bnf_h01",
    "dprime_bnf_h02",
    "dprime_bnf_3scale",
    "dprime_mix_h01",
    "dprime_mix_h02",
    "dprime_mix_3scale",
    "dprime_all_features_h01",
    "dprime_all_features_h02",
    "dprime_all_features_h03",
    "dprime_all_features_3scale",
)


def _official_stepe_agent_specs() -> List[Dict[str, Any]]:
    """Return deterministic StepE expert specs for the official 10-agent set."""
    specs: List[Dict[str, Any]] = []
    for agent in _OFFICIAL_STEPE_AGENTS:
        fam = "all_features" if "all_features" in agent else ("mix" if "mix" in agent else "bnf")
        pred_type = "3scale" if agent.endswith("_3scale") else agent.rsplit("_", 1)[-1]
        specs.append(
            {
                "agent": agent,
                "obs_profile": "D",
                "dprime_profile": agent,
                "dprime_sources": fam,
                "dprime_horizons": pred_type,
            }
        )
    return specs


def _device_auto() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _inject_default_stepe_configs(app_config: Any, output_root: Path) -> None:
    from ai_core.services.step_e_service import StepEConfig

    cfg_list = []
    for idx, spec in enumerate(_official_stepe_agent_specs()):
        cfg = StepEConfig(agent=str(spec["agent"]))
        cfg.output_root = str(output_root)
        cfg.obs_profile = str(spec["obs_profile"])
        cfg.use_stepd_prime = True
        cfg.use_dprime_state = False
        cfg.dprime_profile = str(spec["dprime_profile"])
        cfg.dprime_sources = str(spec["dprime_sources"])
        cfg.dprime_horizons = str(spec["dprime_horizons"])
        cfg.seed = 42 + idx
        cfg.device = "auto"
        cfg.policy_kind = "ppo"
        cfg.ppo_total_timesteps = 80000
        cfg.ppo_n_steps = 2048
        cfg.ppo_batch_size = 512
        cfg.ppo_n_epochs = 2
        cfg.ppo_gamma = 0.99
        cfg.ppo_gae_lambda = 0.95
        cfg.ppo_ent_coef = 0.0
        cfg.ppo_clip_range = 0.2
        cfg.lr = 3e-4
        cfg.ppo_lr = 3e-4
        cfg.pos_l2 = 1e-3
        cfg.max_parallel_agents = 1
        cfg_list.append(cfg)

    if isinstance(app_config, dict):
        app_config["stepE"] = cfg_list
    else:
        setattr(app_config, "stepE", cfg_list)


def _iter_stepe_configs(app_config: Any) -> List[Any]:
    raw = app_config.get("stepE") if isinstance(app_config, dict) else getattr(app_config, "stepE", None)
    if raw is None:
        return []
    return list(raw) if isinstance(raw, (list, tuple)) else [raw]


def _apply_headless_stepe_overrides(app_config: Any, args: Any) -> None:
    cfgs = _iter_stepe_configs(app_config)
    if not cfgs:
        return

    override_policy_kind = getattr(args, "stepE_policy_kind", None)
    override_ppo_total_timesteps = getattr(args, "stepE_ppo_total_timesteps", None)
    override_ppo_n_epochs = getattr(args, "stepE_ppo_n_epochs", None)
    override_max_parallel_agents = getattr(args, "stepE_max_parallel_agents", None)

    for cfg in cfgs:
        current_policy_kind = str(getattr(cfg, "policy_kind", "ppo") or "ppo").strip().lower()
        final_policy_kind = str(override_policy_kind or current_policy_kind or "ppo").strip().lower()
        if final_policy_kind != "ppo":
            raise ValueError(f"Unsupported StepE policy_kind: {final_policy_kind}. StepE is PPO-only.")
        setattr(cfg, "policy_kind", final_policy_kind)
        if override_ppo_total_timesteps is not None:
            setattr(cfg, "ppo_total_timesteps", int(override_ppo_total_timesteps))
        elif getattr(cfg, "ppo_total_timesteps", None) is None:
            setattr(cfg, "ppo_total_timesteps", 80000)
        if override_ppo_n_epochs is not None:
            setattr(cfg, "ppo_n_epochs", int(override_ppo_n_epochs))
        elif getattr(cfg, "ppo_n_epochs", None) is None:
            setattr(cfg, "ppo_n_epochs", 2)
        if getattr(cfg, "ppo_batch_size", None) is None:
            setattr(cfg, "ppo_batch_size", 512)
        if getattr(cfg, "ppo_n_steps", None) is None:
            setattr(cfg, "ppo_n_steps", 2048)
        if getattr(cfg, "ppo_lr", None) is None:
            setattr(cfg, "ppo_lr", float(getattr(cfg, "lr", 3e-4) or 3e-4))
        if override_max_parallel_agents is not None:
            setattr(cfg, "max_parallel_agents", int(override_max_parallel_agents))
        elif getattr(cfg, "max_parallel_agents", None) is None:
            setattr(cfg, "max_parallel_agents", 1)
        print(
            f"[StepEConfig] agent={getattr(cfg, 'agent', '')} policy_kind={getattr(cfg, 'policy_kind', '')} "
            f"policy_role=ppo_only ppo_total_timesteps={int(getattr(cfg, 'ppo_total_timesteps', 0) or 0)} "
            f"ppo_n_epochs={int(getattr(cfg, 'ppo_n_epochs', 0) or 0)} "
            f"ppo_batch_size={int(getattr(cfg, 'ppo_batch_size', 0) or 0)} "
            f"max_parallel_agents={int(getattr(cfg, 'max_parallel_agents', 1) or 1)}"
        )


def _extract_stepe_agents_from_config(app_config: Any) -> List[str]:
    raw = app_config.get("stepE") if isinstance(app_config, dict) else getattr(app_config, "stepE", None)
    if raw is None:
        return []
    cfgs = list(raw) if isinstance(raw, (list, tuple)) else [raw]
    out = []
    for c in cfgs:
        a = getattr(c, "agent", None) if not isinstance(c, dict) else c.get("agent")
        if a:
            out.append(str(a))
    return sorted(set(out))


def _symbols_for_data_prep(primary_symbol: str) -> List[str]:
    symbol = (primary_symbol or "SOXL").upper()
    required = {symbol}
    if symbol in {"SOXL", "SOXS"}:
        required.update({"SOXL", "SOXS"})
    return sorted(required)


def _symbols_for_stepa_execution(primary_symbol: str) -> List[str]:
    # StepA heavy artifact generation is always limited to the primary symbol.
    symbol = (primary_symbol or "SOXL").upper()
    return [symbol]


def _prepare_missing_data_if_needed(
    *,
    repo_root: Path,
    data_root: Path,
    primary_symbol: str,
    auto_prepare_data: bool,
    data_start: str,
    data_end: str,
) -> None:
    required_symbols = _symbols_for_data_prep(primary_symbol)
    missing = [s for s in required_symbols if not (data_root / f"prices_{s}.csv").exists()]
    if not missing:
        print(f"[headless] data ready: {','.join(required_symbols)}")
        return
    if not auto_prepare_data:
        searched = ", ".join(str(data_root / f"prices_{s}.csv") for s in missing)
        raise FileNotFoundError(
            "Missing price CSV files and --auto-prepare-data=0 was set. "
            f"searched={searched}. "
            "hint: run tools/prepare_data.py with the SAME --data-dir passed to tools/run_pipeline.py"
        )

    from tools.prepare_data import ensure_price_csvs

    print(f"[headless] auto prepare data for symbols={','.join(missing)} start={data_start} end={data_end}")
    ensure_price_csvs(
        symbols=missing,
        start=data_start,
        end=data_end,
        data_dir=data_root,
        force=False,
    )

def _try_call(fn, /, *pos, **kw):
    return fn(*pos, **kw)


def _call_with_best_effort(fn, ctx: Dict[str, Any]):
    sig = inspect.signature(fn)
    kwargs: Dict[str, Any] = {}
    for name, p in sig.parameters.items():
        if name in ("self",):
            continue
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if name in ctx:
            kwargs[name] = ctx[name]
            continue
        # common aliases
        if name == "sym" and "symbol" in ctx:
            kwargs[name] = ctx["symbol"]
            continue
        if name == "symbol" and "sym" in ctx:
            kwargs[name] = ctx["sym"]
            continue
        if name in ("config", "cfg") and "app_config" in ctx:
            # Some services expect `config.symbol` / `config.date_range` even when the global AppConfig
            # does not carry those fields. Provide them via a shim.
            kwargs[name] = _ConfigShim(ctx["app_config"], symbol=ctx.get("symbol"), date_range=ctx.get("date_range"))
            continue
    return fn(**kwargs)

def _build_kwargs_from_signature(sig: inspect.Signature, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Build kwargs for a function/ctor call from ctx, matching only supported parameter names."""
    kwargs: Dict[str, Any] = {}
    for name, p in sig.parameters.items():
        if name in ("self",):
            continue
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        # direct match
        if name in ctx:
            kwargs[name] = ctx[name]
            continue
        # common aliases
        if name == "sym" and "symbol" in ctx:
            kwargs[name] = ctx["symbol"]
            continue
        if name == "symbol" and "sym" in ctx:
            kwargs[name] = ctx["sym"]
            continue
        if name in ("config", "cfg") and "app_config" in ctx:
            # Some code uses config=AppConfig instead of app_config=
            kwargs[name] = ctx["app_config"]
            continue
    return kwargs


def _instantiate_service(cls, ctx: Dict[str, Any]):
    """Instantiate service class robustly by passing only accepted __init__ kwargs."""
    try:
        sig = inspect.signature(cls.__init__)
        kwargs = _build_kwargs_from_signature(sig, ctx)
        return cls(**kwargs)
    except TypeError:
        # Fallbacks: try the most common ctor forms
        if "app_config" in ctx:
            try:
                return cls(ctx["app_config"])
            except TypeError:
                pass
        if "app_config" in ctx and "symbol" in ctx and "date_range" in ctx:
            try:
                return cls(ctx["app_config"], ctx["symbol"], ctx["date_range"])
            except TypeError:
                pass
        raise


def _filter_kwargs_for_ctor(cls: Any, **kwargs: Any) -> Dict[str, Any]:
    """Return ctor kwargs accepted by `cls`.

    This keeps call sites forward/backward compatible when service config dataclasses
    add/remove optional constructor parameters across versions.
    """
    try:
        sig = inspect.signature(cls)
        allowed = set(sig.parameters.keys())
    except Exception:
        allowed = set(getattr(cls, "__dataclass_fields__", {}).keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


def _load_prices_dates(
    symbol: str,
    repo_root: Path,
    output_root: Path,
    data_root: Path,
    mode: str,
) -> Tuple[Optional[Any], Optional[Any], List[Path], List[Path]]:
    import pandas as pd

    mode = (mode or "sim").strip().lower()
    if mode == "ops":
        mode = "live"

    split_train = output_root / "stepA" / mode / f"stepA_prices_train_{symbol}.csv"
    split_test = output_root / "stepA" / mode / f"stepA_prices_test_{symbol}.csv"

    candidates = [
        # (Priority 1) split outputs in the requested mode
        split_train,
        split_test,
        # (Priority 2) consolidated stepA prices (display and compatibility variants)
        output_root / "stepA" / mode / f"stepA_prices_{symbol}.csv",
        output_root / "stepA" / "display" / f"stepA_prices_{symbol}.csv",
        output_root / f"stepA_prices_{symbol}.csv",
        # (Priority 3) raw source fallback
        data_root / f"prices_{symbol}.csv",
    ]

    selected_dates = None
    selected_close = None
    selected_files: List[Path] = []

    # Split train/test pair has the highest priority and must be merged if both exist.
    if split_train.exists() and split_test.exists():
        merged_dates = []
        merged_close = []
        for p in (split_train, split_test):
            df = pd.read_csv(p)
            date_col = next((c for c in df.columns if c.lower() == "date"), None)
            if date_col is None:
                continue
            try:
                df[date_col] = pd.to_datetime(df[date_col])
            except Exception:
                pass
            close_col = next((c for c in df.columns if c.lower() == "close"), None)
            merged_dates.append(df[date_col])
            if close_col is not None:
                merged_close.append(df[close_col])
        if merged_dates:
            selected_dates = pd.concat(merged_dates, ignore_index=True)
            if merged_close:
                selected_close = pd.concat(merged_close, ignore_index=True)
            selected_files = [split_train, split_test]

    if selected_dates is None:
        for p in candidates:
            if p.exists():
                df = pd.read_csv(p)
                date_col = next((c for c in df.columns if c.lower() == "date"), None)
                if date_col is None:
                    continue
                try:
                    df[date_col] = pd.to_datetime(df[date_col])
                except Exception:
                    pass
                close_col = next((c for c in df.columns if c.lower() == "close"), None)
                selected_dates = df[date_col]
                selected_close = df[close_col] if close_col else None
                selected_files = [p]
                break

    return selected_dates, selected_close, candidates, selected_files


def _read_split_summary_json(summary_json: Path) -> Dict[str, Any]:
    if not summary_json.exists():
        raise FileNotFoundError(f"missing_split_summary_json={summary_json}")
    print(f"[SPLIT_SUMMARY] read path={summary_json}")
    raw_bytes = summary_json.read_bytes()
    normalized_bom = raw_bytes.startswith(b"\xef\xbb\xbf")
    raw_text = raw_bytes.decode("utf-8-sig")
    print(f"[SPLIT_SUMMARY] normalized_bom={'true' if normalized_bom else 'false'}")
    payload = json.loads(raw_text)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid_split_summary_json={summary_json}")
    print(f"[SPLIT_SUMMARY] validate=ok path={summary_json}")
    return payload


def _build_date_range(
    symbol: str,
    repo_root: Path,
    test_start: Optional[str],
    test_months: int,
    train_years: int,
    future_end: Optional[str] = None,
    mamba_mode: str = "sim",
    stepE_mode: str = "sim",
    mode: str = "sim",
    output_root: Optional[Path] = None,
    data_root: Optional[Path] = None,
    env_horizon_days: Optional[int] = None,
):
    import pandas as pd
    from ai_core.types.common import DateRange

    if output_root is None:
        raise ValueError("output_root is required for _build_date_range")
    out_root = Path(output_root)
    data_root = Path(data_root) if data_root is not None else (repo_root / "data")

    split_summary_json = out_root / "split_summary.json"
    split_summary_csv = out_root / "stepA" / str(mode or "sim").lower() / f"stepA_split_summary_{symbol}.csv"

    dates, _, searched_paths, selected_price_files = _load_prices_dates(
        symbol=symbol,
        repo_root=repo_root,
        output_root=out_root,
        data_root=data_root,
        mode=mode,
    )
    if dates is None or len(dates) == 0:
        searched = "\n".join(
            f"  - {'FOUND (invalid/empty?)' if p.exists() else 'MISSING'}: {p}" for p in searched_paths
        )
        raise RuntimeError(
            "Cannot build DateRange: no usable price dates found. Searched candidates:\n"
            f"{searched}"
        )
    dates_sorted = pd.to_datetime(dates, errors="coerce").dropna().sort_values().drop_duplicates().reset_index(drop=True)
    dmin = pd.to_datetime(dates_sorted.iloc[0]).normalize()
    dmax = pd.to_datetime(dates_sorted.iloc[-1]).normalize()

    print(
        f"[DATE_RANGE] prices_source_files={','.join(str(p) for p in selected_price_files) if selected_price_files else 'none'}"
    )
    print(f"[DATE_RANGE] dates_min={str(dmin.date())}")
    print(f"[DATE_RANGE] dates_max={str(dmax.date())}")
    print(f"[DATE_RANGE] cli_test_start={str(test_start) if test_start else ''}")

    split_source = ""
    split_payload: Dict[str, Any] = {}
    if split_summary_json.exists():
        split_payload = _read_split_summary_json(split_summary_json)
        split_source = "split_summary_json"
    elif split_summary_csv.exists():
        split_payload = _build_split_payload_from_stepa_summary(
            split_summary_csv,
            symbol=symbol,
            mode=str(mode or "sim").lower(),
            train_years=int(train_years),
            test_months=int(test_months),
        )
        split_source = "stepA_split_summary_csv"

    cli_test_start_dt = pd.to_datetime(test_start).normalize() if test_start else None

    if split_source:
        def _from_payload(name: str) -> Optional[Any]:
            raw = str(split_payload.get(name, "")).strip()
            if not raw:
                return None
            try:
                return pd.to_datetime(raw).normalize()
            except Exception:
                return None

        train_start = _from_payload("train_start")
        train_end = _from_payload("train_end")
        ts = _from_payload("test_start")
        test_end = _from_payload("test_end")

        if not (train_start and train_end and ts and test_end):
            split_source = ""
        elif cli_test_start_dt is not None and ts != cli_test_start_dt:
            mismatch = str((ts - cli_test_start_dt).days)
            print(f"[DATE_RANGE] cli_test_start={str(cli_test_start_dt.date())}")
            print(f"[DATE_RANGE] split_summary_test_start={str(ts.date())}")
            print(f"[DATE_RANGE] mismatch={mismatch}d")

    if not split_source:
        print("[DATE_RANGE] source=prices_csv_fallback")
        test_start_input = (
            cli_test_start_dt
            if cli_test_start_dt is not None
            else (pd.to_datetime(dates_sorted.iloc[-1]).normalize() - pd.DateOffset(months=test_months))
        )
        ts = snap_prev_by_prices(test_start_input, dates_sorted)

        train_start_raw = ts - pd.DateOffset(years=train_years)
        train_end_raw = ts - pd.Timedelta(days=1)
        test_end_raw = ts + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)

        train_start = snap_prev_by_prices(train_start_raw, dates_sorted)
        train_end = snap_prev_by_prices(train_end_raw, dates_sorted)
        test_end = snap_prev_by_prices(test_end_raw, dates_sorted)
    else:
        print(f"[DATE_RANGE] source={split_source}")

    train_start = min(max(train_start, dmin), dmax)
    train_end = min(max(train_end, dmin), dmax)
    ts = min(max(ts, dmin), dmax)
    test_end = min(max(test_end, dmin), dmax)
    print(f"[DATE_RANGE] resolved_train_start={str(train_start.date())}")
    print(f"[DATE_RANGE] resolved_train_end={str(train_end.date())}")
    print(f"[DATE_RANGE] resolved_test_start={str(ts.date())}")
    print(f"[DATE_RANGE] resolved_test_end={str(test_end.date())}")

    if str(mode or "sim").lower() == "sim":
        if ts == test_end:
            raise RuntimeError("date_range_invalid_before_stepB: sim_requires_test_span")
        if cli_test_start_dt is not None and ts < cli_test_start_dt:
            raise RuntimeError(
                "date_range_invalid_before_stepB: test_start_rounded_before_cli "
                f"cli={str(cli_test_start_dt.date())} resolved={str(ts.date())}"
            )
    kwargs = {"train_start": train_start, "train_end": train_end, "test_start": ts, "test_end": test_end}
    dr = DateRange(**kwargs)
    dr = _ensure_date_range_aliases(dr)
    # Attach extra attributes via shim so downstream steps can branch safely.
    # Some DateRange implementations are slots-like, so we wrap if direct setattr fails.
    fe = pd.to_datetime(future_end) if future_end else None
    extras = {
        "symbol": symbol,
        "mode": str(mode or "sim").lower(),
        "future_end": fe,
        "mamba_mode": str(mamba_mode or "sim").lower(),
        "stepE_mode": str(stepE_mode or "sim").lower(),
    }
    if env_horizon_days is not None:
        try:
            extras['env_horizon_days'] = int(env_horizon_days)
        except Exception:
            pass
    for k, v in list(extras.items()):
        if v is None:
            extras.pop(k, None)
            continue
        try:
            setattr(dr, k, v)
            extras.pop(k, None)
        except Exception:
            pass
    if extras:
        dr = _DateRangeShim(dr, **extras)
    return dr


def _ensure_stepb_pred_time_all(symbol: str, output_root: Path, mode: str) -> Path:
    """Ensure stepB_pred_time_all_<SYMBOL>.csv exists under output/stepB/<mode>/.

    Policy:
      - Target is always mode-separated (output/stepB/<mode>/stepB_pred_time_all_<SYMBOL>.csv).
      - We NEVER create or update legacy mixed paths like output/stepB/stepB_pred_time_all_<SYMBOL>.csv.
      - If a legacy file exists, we may copy it into the mode folder for compatibility.

    This prevents accidental training on mixed-mode artifacts.
    """

    mode = (mode or 'sim').strip().lower()
    if mode not in ('sim', 'live', 'ops', 'display'):
        # be permissive; create folder anyway
        pass

    import pandas as pd
    import shutil

    out_root = Path(output_root)
    stepb_mode_dir = out_root / 'stepB' / mode
    stepb_mode_dir.mkdir(parents=True, exist_ok=True)

    target = stepb_mode_dir / f'stepB_pred_time_all_{symbol}.csv'

    def _normalize_pred_time_all(path: Path) -> bool:
        if not path.exists() or path.stat().st_size <= 0:
            return False
        try:
            df = pd.read_csv(path)
        except Exception:
            return False
        if 'Date' not in df.columns:
            return False

        mamba_col = next((c for c in df.columns if c == 'Pred_Close_MAMBA'), None)
        if mamba_col is None:
            mamba_col = next((c for c in df.columns if c.lower() == 'pred_close_mamba'), None)
        if mamba_col is None:
            return False

        clean = df[['Date', mamba_col]].copy()
        clean = clean.rename(columns={mamba_col: 'Pred_Close_MAMBA'})
        clean.to_csv(path, index=False, encoding='utf-8')
        return True

    if _normalize_pred_time_all(target):
        return target

    # Build from mode-local pred_close first.
    pred_close = stepb_mode_dir / f'stepB_pred_close_mamba_{symbol}.csv'

    def _build_from_pred_close(pred_close_path: Path) -> bool:
        if not pred_close_path.exists() or pred_close_path.stat().st_size <= 0:
            return False
        try:
            df = pd.read_csv(pred_close_path)
        except Exception:
            return False
        date_col = next((c for c in df.columns if c.lower() == 'date'), None)
        if date_col is None:
            return False
        mamba_col = next((c for c in df.columns if c == 'Pred_Close_MAMBA'), None)
        if mamba_col is None:
            mamba_col = next((c for c in df.columns if c.lower() == 'pred_close_mamba'), None)
        if mamba_col is None:
            mamba_col = next((c for c in df.columns if c.lower().startswith('pred_close_mamba_')), None)
        if mamba_col is None:
            return False
        out_df = df[[date_col, mamba_col]].copy().rename(
            columns={date_col: 'Date', mamba_col: 'Pred_Close_MAMBA'}
        )
        out_df.to_csv(target, index=False, encoding='utf-8')
        return _normalize_pred_time_all(target)

    if _build_from_pred_close(pred_close):
        return target

    # If a legacy pred_time_all exists, copy it into the mode folder (do not delete legacy).
    legacy_candidates = [
        out_root / 'stepB' / f'stepB_pred_time_all_{symbol}.csv',
        out_root / f'stepB_pred_time_all_{symbol}.csv',
    ]
    for c in legacy_candidates:
        if c.exists() and c.stat().st_size > 0:
            shutil.copy2(c, target)
            if _normalize_pred_time_all(target):
                return target

    raise FileNotFoundError(f"Missing StepB pred_time_all/pred_close artifacts for {symbol} at {stepb_mode_dir}")
def _get_app_config(repo_root: Path):
    """Load AppConfig from YAML if available, otherwise fall back to a minimal default.

    Notes
    -----
    In this repository, some codepaths expect a module-level function
    `ai_core.config.app_config.load_from_yaml(path)`, while other codepaths
    provide `AppConfig.load_from_yaml(path)` as a classmethod.

    This loader supports both, and will also generate a minimal AppConfig when:
      - config/app_config.yaml does not exist, or
      - no compatible loader is available.

    The goal is to keep the headless runner robust so that downstream Step services
    can execute and surface their own errors.
    """
    try:
        from ai_core.config import app_config as app_config_mod
    except Exception as e:
        raise RuntimeError("Failed to import ai_core.config.app_config") from e

    path = repo_root / "config" / "app_config.yaml"

    # 1) Try to load from YAML using known loader entrypoints (module-level then class-level)
    if path.exists():
        # module-level loaders
        for name in ("load_from_yaml", "load_yaml", "from_yaml", "load"):
            fn = getattr(app_config_mod, name, None)
            if callable(fn):
                try:
                    return fn(path)
                except Exception:
                    pass

        # class-level loaders
        AppConfig = getattr(app_config_mod, "AppConfig", None)
        if AppConfig is not None:
            for name in ("load_from_yaml", "load_yaml", "from_yaml", "load"):
                fn = getattr(AppConfig, name, None)
                if callable(fn):
                    try:
                        return fn(path)
                    except Exception:
                        pass

    # 2) Fall back to a minimal default AppConfig (best-effort instantiation)
    AppConfig = getattr(app_config_mod, "AppConfig", None)
    if AppConfig is None:
        # As a last resort, return a minimal dict-like config; most services expect an object,
        # but this keeps the runner from crashing at the config stage.
        return {
            "repo_root": repo_root,
            "data_root": repo_root / "data",
            "output_root": repo_root / "output",
            "config_root": repo_root / "config",
            "artifacts_root": repo_root / "artifacts",
            "workspace_root": repo_root / "workspace",
            "log_root": repo_root / "logs",
        }

    import inspect

    candidates = {
        "repo_root": repo_root,
        "project_root": repo_root,
        "root": repo_root,
        "base_dir": repo_root,
        "data_root": repo_root / "data",
        "data_dir": repo_root / "data",
        "prices_dir": repo_root / "data",
        "output_root": repo_root / "output",
        "output_dir": repo_root / "output",
        "config_root": repo_root / "config",
        "config_dir": repo_root / "config",
        "artifacts_root": repo_root / "artifacts",
        "artifacts_dir": repo_root / "artifacts",
        "workspace_root": repo_root / "workspace",
        "workspace_dir": repo_root / "workspace",
        "log_root": repo_root / "logs",
        "log_dir": repo_root / "logs",
    }

    try:
        sig = inspect.signature(AppConfig)
        kwargs = {}
        for pname, p in sig.parameters.items():
            if pname == "self":
                continue
            if pname in candidates:
                kwargs[pname] = candidates[pname]
            elif p.default is not inspect._empty:
                # use default
                continue
            else:
                # required but unknown -> best-effort
                ann = p.annotation
                if ann is Path:
                    kwargs[pname] = repo_root
                else:
                    kwargs[pname] = None
        return AppConfig(**kwargs)  # type: ignore[arg-type]
    except Exception:
        # Best-effort: try no-arg constructor
        try:
            return AppConfig()  # type: ignore[call-arg]
        except Exception as e:
            raise RuntimeError("Failed to construct AppConfig (no YAML and incompatible signature)") from e


def _run_stepA(app_config, symbol: str, date_range):
    from ai_core.services.step_a_service import StepAService

    cfg_data_root = _read_config_data_dir(app_config)

    # NOTE:
    # StepAService.__init__ declares `app_config` as a positional-only argument.
    # Construct it explicitly to guarantee the runner-provided AppConfig (including
    # --data-dir propagation) is honored on every codepath.
    svc = StepAService(app_config)
    fn = getattr(svc, "run", None)
    if fn is None:
        raise RuntimeError("StepAService has no run()")

    # StepA must honor --data-dir on self-hosted runners where repo/data may not exist.
    if cfg_data_root is not None:
        resolved_data_dir = str(cfg_data_root)
        print(f"[StepA] data_dir={resolved_data_dir}")
        return fn(symbol=symbol, date_range=date_range, data_dir=resolved_data_dir, data_root=resolved_data_dir)
    return fn(symbol=symbol, date_range=date_range)


def _read_stepa_split_summary_csv(summary_csv: Path) -> Dict[str, str]:
    import pandas as pd

    if not summary_csv.exists():
        raise FileNotFoundError(f"missing_stepa_split_summary_csv={summary_csv}")

    df = pd.read_csv(summary_csv)
    if "key" not in df.columns or "value" not in df.columns:
        raise ValueError(f"invalid_stepa_split_summary_format={summary_csv}")

    out: Dict[str, str] = {}
    for _, row in df.iterrows():
        k = str(row.get("key", "")).strip()
        if not k:
            continue
        raw_v = row.get("value", "")
        out[k] = "" if raw_v is None else str(raw_v).strip()
    return out


def _build_split_payload_from_stepa_summary(
    summary_csv: Path,
    *,
    symbol: str,
    mode: str,
    train_years: int,
    test_months: int,
) -> Dict[str, Any]:
    summary_map = _read_stepa_split_summary_csv(summary_csv)

    def _pick_date(name: str) -> str:
        raw = str(summary_map.get(name, "")).strip()
        if not raw:
            return ""
        try:
            import pandas as pd

            return str(pd.to_datetime(raw).date())
        except Exception:
            return raw[:10]

    def _pick_int(name: str, fallback: int) -> int:
        raw = str(summary_map.get(name, "")).strip()
        if not raw:
            return int(fallback)
        try:
            return int(float(raw))
        except Exception:
            return int(fallback)

    return {
        "symbol": str(symbol).upper(),
        "mode": str(mode),
        "test_start": _pick_date("test_start"),
        "train_start": _pick_date("train_start"),
        "train_end": _pick_date("train_end"),
        "test_end": _pick_date("test_end"),
        "train_years": _pick_int("train_years", train_years),
        "test_months": _pick_int("test_months", test_months),
    }


def _write_split_summary_json_payload(dst: Path, payload: Dict[str, Any], log_prefix: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        _read_split_summary_json(dst)
    except Exception as e:
        raise RuntimeError(f"[SPLIT_SUMMARY] validate=fail path={dst} reason={type(e).__name__}:{e}") from e
    print(f"[{log_prefix}] split_summary_written={dst}")


def _sync_root_split_summary_from_stepa(
    *,
    resolved_output_root: Path,
    canonical_output_root: Path,
    symbol: str,
    mode: str,
    train_years: int,
    test_months: int,
    log_prefix: str,
) -> Dict[str, Any]:
    summary_csv = Path(canonical_output_root) / "stepA" / str(mode) / f"stepA_split_summary_{symbol}.csv"
    payload = _build_split_payload_from_stepa_summary(
        summary_csv,
        symbol=symbol,
        mode=mode,
        train_years=train_years,
        test_months=test_months,
    )
    _write_split_summary_json_payload(Path(canonical_output_root) / "split_summary.json", payload, log_prefix)
    if Path(canonical_output_root).resolve() != Path(resolved_output_root).resolve():
        _write_split_summary_json_payload(Path(resolved_output_root) / "split_summary.json", payload, log_prefix)
    return payload


def _check_and_repair_split_summary_before_stepb(
    *,
    resolved_output_root: Path,
    canonical_output_root: Path,
    symbol: str,
    mode: str,
    train_years: int,
    test_months: int,
) -> None:
    expected = _build_split_payload_from_stepa_summary(
        Path(canonical_output_root) / "stepA" / str(mode) / f"stepA_split_summary_{symbol}.csv",
        symbol=symbol,
        mode=mode,
        train_years=train_years,
        test_months=test_months,
    )

    root_split_path = Path(canonical_output_root) / "split_summary.json"
    current: Dict[str, Any] = {}
    if root_split_path.exists():
        try:
            current = _read_split_summary_json(root_split_path)
        except Exception as e:
            print(f"[SPLIT_CHECK] root_split_summary_read=fail path={root_split_path} reason={type(e).__name__}:{e}")

    check_keys = (
        "symbol",
        "mode",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "train_years",
        "test_months",
    )
    mismatch: List[str] = []
    for k in check_keys:
        if str(current.get(k, "")) != str(expected.get(k, "")):
            mismatch.append(f"{k}:root={current.get(k)!r}:stepA={expected.get(k)!r}")

    if mismatch:
        print(f"[SPLIT_CHECK] mismatch_detected count={len(mismatch)}")
        for item in mismatch:
            print(f"[SPLIT_CHECK] mismatch {item}")
        _sync_root_split_summary_from_stepa(
            resolved_output_root=resolved_output_root,
            canonical_output_root=canonical_output_root,
            symbol=symbol,
            mode=mode,
            train_years=train_years,
            test_months=test_months,
            log_prefix="SPLIT_REPAIR",
        )

        repaired = json.loads(root_split_path.read_text(encoding="utf-8")) if root_split_path.exists() else {}
        residual = [k for k in check_keys if str(repaired.get(k, "")) != str(expected.get(k, ""))]
        if residual:
            raise RuntimeError(f"split_summary_repair_failed keys={','.join(residual)}")
        print("[SPLIT_CHECK] repaired=ok")
        return

    print("[SPLIT_CHECK] split_summary_consistent=pass")


def _read_config_data_dir(app_config: Any) -> Optional[Path]:
    """Best-effort read of AppConfig data root (data_dir/data_root)."""

    def _cfg_get(obj: Any, name: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    cfg_data_root = _cfg_get(app_config, "data_dir", None)
    if cfg_data_root is None:
        cfg_data_root = _cfg_get(app_config, "data_root", None)
    if cfg_data_root is None:
        cfg_data = _cfg_get(app_config, "data", None)
        cfg_data_root = _cfg_get(cfg_data, "data_dir", None) if cfg_data is not None else None
    if cfg_data_root is None:
        cfg_data = _cfg_get(app_config, "data", None)
        cfg_data_root = _cfg_get(cfg_data, "data_root", None) if cfg_data is not None else None
    return Path(cfg_data_root).expanduser().resolve() if cfg_data_root is not None else None


def _resolve_cli_data_dir(repo_root: Path, cli_data_dir: Optional[str]) -> Path:
    """Resolve --data-dir with backward-compatible default (repo_root/data)."""
    if cli_data_dir:
        data_dir = Path(cli_data_dir).expanduser()
        if not data_dir.is_absolute():
            data_dir = repo_root / data_dir
        return data_dir.resolve()
    return (repo_root / "data").resolve()


def _apply_config_data_dir(app_config: Any, data_root: Path) -> Any:
    """Write data_dir/data_root onto AppConfig and nested data config (best-effort)."""
    if isinstance(app_config, dict):
        app_config["data_dir"] = str(data_root)
        app_config["data_root"] = str(data_root)
        data_cfg = app_config.get("data")
        if isinstance(data_cfg, dict):
            data_cfg["data_dir"] = str(data_root)
            data_cfg["data_root"] = str(data_root)
        return app_config

    try:
        setattr(app_config, "data_dir", str(data_root))
        setattr(app_config, "data_root", str(data_root))
        cfg_data = getattr(app_config, "data", None)
        if cfg_data is not None:
            setattr(cfg_data, "data_dir", data_root)
            setattr(cfg_data, "data_root", data_root)
        return app_config
    except Exception:
        return _ConfigShim(app_config, data_dir=str(data_root), data_root=str(data_root))


def _apply_config_output_root(app_config: Any, output_root: Path) -> Any:
    """Write output_root onto AppConfig top-level and nested data config (best-effort)."""
    output_root_str = str(output_root)

    def _safe_set_attr(obj: Any, name: str, value: Any) -> bool:
        try:
            setattr(obj, name, value)
            return True
        except Exception:
            return False

    def _replace_dc_attr(obj: Any, **kwargs: Any) -> Tuple[Any, bool]:
        if not dataclasses.is_dataclass(obj):
            return obj, False
        dc_fields = getattr(obj, "__dataclass_fields__", {})
        filtered = {k: v for k, v in kwargs.items() if k in dc_fields}
        if not filtered:
            return obj, False
        try:
            return dataclasses.replace(obj, **filtered), True
        except Exception:
            return obj, False

    def _ensure_data_cfg(obj: Any) -> Tuple[Any, Any]:
        cfg_data = getattr(obj, "data", None)
        if cfg_data is not None:
            return obj, cfg_data

        shim_data = _ConfigShim(object(), output_root=output_root_str, output_dir=output_root_str)
        if _safe_set_attr(obj, "data", shim_data):
            return obj, shim_data

        replaced_obj, ok = _replace_dc_attr(obj, data=shim_data)
        if ok:
            return replaced_obj, getattr(replaced_obj, "data", shim_data)

        wrapped = _ConfigShim(obj, data=shim_data)
        return wrapped, shim_data

    if isinstance(app_config, dict):
        app_config["output_root"] = output_root_str
        app_config["output_dir"] = output_root_str
        data_cfg = app_config.get("data")
        if isinstance(data_cfg, dict):
            data_cfg["output_root"] = output_root_str
            data_cfg["output_dir"] = output_root_str
        elif data_cfg is None:
            app_config["data"] = {"output_root": output_root_str, "output_dir": output_root_str}
        return app_config

    # top-level output_root and output_dir alias
    top_ok_root = _safe_set_attr(app_config, "output_root", output_root_str)
    top_ok_dir = _safe_set_attr(app_config, "output_dir", output_root_str)
    if not (top_ok_root and top_ok_dir):
        replaced_app, ok = _replace_dc_attr(app_config, output_root=output_root_str, output_dir=output_root_str)
        if ok:
            app_config = replaced_app
        else:
            app_config = _ConfigShim(app_config, output_root=output_root_str, output_dir=output_root_str)

    app_config, cfg_data = _ensure_data_cfg(app_config)

    data_ok_root = _safe_set_attr(cfg_data, "output_root", output_root_str)
    data_ok_dir = _safe_set_attr(cfg_data, "output_dir", output_root_str)
    if not (data_ok_root and data_ok_dir):
        replaced_data, ok = _replace_dc_attr(cfg_data, output_root=output_root_str, output_dir=output_root_str)
        if ok:
            if not _safe_set_attr(app_config, "data", replaced_data):
                replaced_app, ok_app = _replace_dc_attr(app_config, data=replaced_data)
                if ok_app:
                    app_config = replaced_app
                else:
                    app_config = _ConfigShim(app_config, data=_ConfigShim(replaced_data, output_root=output_root_str, output_dir=output_root_str))
            else:
                _safe_set_attr(app_config.data, "output_root", output_root_str)
                _safe_set_attr(app_config.data, "output_dir", output_root_str)
        else:
            data_shim = _ConfigShim(cfg_data, output_root=output_root_str, output_dir=output_root_str)
            if not _safe_set_attr(app_config, "data", data_shim):
                replaced_app, ok_app = _replace_dc_attr(app_config, data=data_shim)
                if ok_app:
                    app_config = replaced_app
                else:
                    app_config = _ConfigShim(app_config, data=data_shim)
    return app_config



def _enable_stepb_agents(cfg, enable_mamba: bool) -> None:
    """Best-effort enabling of StepB Mamba across differing config schemas."""
    def _try_set_in_mapping(obj, key: str, value):
        try:
            if isinstance(obj, dict):
                obj[key] = value
                return True
        except Exception:
            pass
        return False

    def _enable_subcfg(subcfg, value: bool) -> None:
        if isinstance(subcfg, bool):
            # handled by parent setter
            return
        # common flags
        for flag in ("enabled", "is_enabled", "enable", "train", "use", "on"):
            _try_set(subcfg, flag, value)
            _try_set_in_mapping(subcfg, flag, value)

    desired = {"mamba": bool(enable_mamba)}

    # 1) Direct boolean flags on cfg
    for name, value in desired.items():
        for prefix in ("train_", "enable_", "use_"):
            _try_set(cfg, f"{prefix}{name}", value)

    # 2) Nested sub-configs
    for name, value in desired.items():
        if hasattr(cfg, name):
            try:
                sub = getattr(cfg, name)
            except Exception:
                continue
            if isinstance(sub, bool):
                _try_set(cfg, name, value)
            else:
                _enable_subcfg(sub, value)

    # 3) Aliases sometimes used
    alias_map = {
        "mamba": ("wavelet_mamba", "mamba_cfg", "mamba_config", "mamba_train", "mamba_model"),
    }
    for name, aliases in alias_map.items():
        value = desired[name]
        for a in aliases:
            if hasattr(cfg, a):
                try:
                    sub = getattr(cfg, a)
                except Exception:
                    continue
                if isinstance(sub, bool):
                    _try_set(cfg, a, value)
                else:
                    _enable_subcfg(sub, value)

    # 4) Mapping/list styles
    if hasattr(cfg, "agents"):
        try:
            agents = getattr(cfg, "agents")
            if isinstance(agents, dict):
                for name, value in desired.items():
                    agents[name] = value
            elif isinstance(agents, (list, tuple, set)):
                # if list of enabled names
                enabled = {n for n, v in desired.items() if v}
                setattr(cfg, "agents", sorted(enabled))
        except Exception:
            pass

    for list_attr in ("enabled_agents", "enabled_models", "models", "model_names"):
        if hasattr(cfg, list_attr):
            try:
                enabled = [n for n, v in desired.items() if v]
                cur = getattr(cfg, list_attr)
                if isinstance(cur, (list, tuple, set)):
                    setattr(cfg, list_attr, enabled)
            except Exception:
                pass


def _run_stepB(app_config, symbol: str, date_range, enable_mamba: bool, enable_mamba_periodic: bool, mamba_lookback: Optional[int], mamba_horizons: Optional[List[int]]):
    from ai_core.services.step_b_service import StepBService
    from ai_core.config.step_b_config import StepBConfig
    ctx = {"app_config": app_config, "symbol": symbol, "sym": symbol, "date_range": date_range}
    svc = _instantiate_service(StepBService, ctx)

    # StepBConfig signature can vary; construct with best effort.
    try:
        cfg = StepBConfig(symbol=symbol, date_range=date_range)
    except TypeError:
        # fallback: try positional or reduced args
        try:
            cfg = StepBConfig(symbol=symbol)
        except TypeError:
            cfg = StepBConfig()

    _enable_stepb_agents(cfg, enable_mamba=enable_mamba)

    # Apply Mamba multi-horizon overrides (lookback/horizons) if provided
    cfg = _apply_mamba_overrides_to_stepb_config(cfg, mamba_lookback=mamba_lookback, mamba_horizons=mamba_horizons)
    # Periodic-only Mamba snapshots toggle (if supported by config/service)
    try:
        m = getattr(cfg, 'mamba', None)
        if m is not None and hasattr(m, 'enable_periodic_snapshots'):
            _try_set(m, 'enable_periodic_snapshots', bool(enable_mamba_periodic))
        elif m is not None and hasattr(m, 'enable_mamba_periodic'):
            _try_set(m, 'enable_mamba_periodic', bool(enable_mamba_periodic))
        elif hasattr(cfg, 'enable_mamba_periodic'):
            _try_set(cfg, 'enable_mamba_periodic', bool(enable_mamba_periodic))
        # If horizons were provided, keep periodic snapshot horizons aligned
        if enable_mamba_periodic and mamba_horizons:
            hz = tuple(int(x) for x in mamba_horizons)
            if m is not None and hasattr(m, 'periodic_snapshot_horizons'):
                _try_set(m, 'periodic_snapshot_horizons', hz)
            if m is not None and hasattr(m, 'periodic_endpoints'):
                _try_set(m, 'periodic_endpoints', hz)
    except Exception:
        pass


    fn = getattr(svc, "run", None)
    if fn is None:
        raise RuntimeError("StepBService has no run()")
    # Some StepBService.run takes cfg as positional; keep it simple.
    return fn(cfg)


def _build_stepdprime_config(app_config, symbol: str, mode: str, *, output_root: Optional[str] = None):
    from ai_core.services.step_dprime_service import StepDPrimeConfig

    out_root = str(output_root or getattr(app_config, "output_root", "output"))
    cluster_cfg = getattr(app_config, "cluster_regime", None)
    existing = app_config.get("stepDPrime") if isinstance(app_config, dict) else getattr(app_config, "stepDPrime", None)
    base_raw = {
        "symbol": symbol,
        "mode": (mode or "sim"),
        "output_root": out_root,
        "enable_cluster_regime": bool(getattr(cluster_cfg, "enable_cluster_regime", True)),
        "enable_cluster_monthly_refit": bool(getattr(cluster_cfg, "enable_cluster_monthly_refit", True)),
        "enable_cluster_daily_assign": bool(getattr(cluster_cfg, "enable_cluster_daily_assign", True)),
        "enable_cluster_in_rl_state": bool(getattr(cluster_cfg, "enable_cluster_in_rl_state", True)),
        "cluster_backend": str(getattr(cluster_cfg, "cluster_backend", "ticc")),
        "cluster_raw_k": int(getattr(cluster_cfg, "cluster_raw_k", 20)),
        "cluster_k_eff_min": int(getattr(cluster_cfg, "cluster_k_eff_min", 12)),
        "cluster_small_share_threshold": float(getattr(cluster_cfg, "cluster_small_share_threshold", 0.01)),
        "cluster_small_mean_run_threshold": float(getattr(cluster_cfg, "cluster_small_mean_run_threshold", 3.0)),
        "cluster_short_window_days": int(getattr(cluster_cfg, "cluster_short_window_days", 20)),
        "cluster_mid_window_weeks": int(getattr(cluster_cfg, "cluster_mid_window_weeks", 8)),
        "cluster_long_window_months": int(getattr(cluster_cfg, "cluster_long_window_months", 6)),
        "cluster_enable_8y_context": bool(getattr(cluster_cfg, "cluster_enable_8y_context", True)),
        "cluster_rare_flag_enabled": bool(getattr(cluster_cfg, "cluster_rare_flag_enabled", True)),
        "timing_logger": _get_timing_logger(app_config),
    }
    if existing is not None:
        if isinstance(existing, dict):
            for k, v in existing.items():
                base_raw[k] = v
        else:
            for k in getattr(StepDPrimeConfig, "__dataclass_fields__", {}).keys():
                if hasattr(existing, k):
                    base_raw[k] = getattr(existing, k)
    base_raw["symbol"] = symbol
    base_raw["mode"] = (mode or base_raw.get("mode") or "sim")
    base_raw["output_root"] = out_root
    return StepDPrimeConfig(**_filter_kwargs_for_ctor(StepDPrimeConfig, **base_raw))


def _build_stepe_profile_agent_map(app_config: Any, profiles: Sequence[str]) -> Dict[str, Any]:
    raw_cfgs = app_config.get("stepE") if isinstance(app_config, dict) else getattr(app_config, "stepE", None)
    cfgs = list(raw_cfgs) if isinstance(raw_cfgs, (list, tuple)) else ([raw_cfgs] if raw_cfgs else [])
    by_profile: Dict[str, Any] = {}
    by_agent: Dict[str, Any] = {}
    for cfg in cfgs:
        if cfg is None:
            continue
        agent = getattr(cfg, "agent", None) if not isinstance(cfg, dict) else cfg.get("agent")
        dprof = getattr(cfg, "dprime_profile", None) if not isinstance(cfg, dict) else cfg.get("dprime_profile")
        if dprof:
            by_profile[str(dprof)] = cfg
        if agent:
            by_agent[str(agent)] = cfg
    out: Dict[str, Any] = {}
    for p in profiles:
        if p in by_profile:
            out[p] = by_profile[p]
        elif p in by_agent:
            out[p] = by_agent[p]
    return out


def _run_stepDPrime(app_config, symbol: str, date_range, mode: str):
    from ai_core.services.step_dprime_service import StepDPrimeService

    out_root = str(getattr(app_config, "output_root", "output"))
    cfg = _build_stepdprime_config(app_config, symbol, mode, output_root=out_root)
    svc = StepDPrimeService()
    return svc.run(cfg)




def _validate_stepdprime_contract(output_root: Path, mode: str, symbol: str) -> None:
    base = Path(output_root) / "stepDprime" / str(mode)
    required = [
        base / f"stepDprime_state_test_{profile}_{symbol}.csv" for profile in _OFFICIAL_STEPE_AGENTS
    ]
    miss = [str(x.as_posix()) for x in required if not x.exists()]
    if miss:
        raise RuntimeError("StepDPrime contract missing required state files: " + ", ".join(miss))

def _run_step_generic(step_letter: str, app_config, symbol: str, date_range, prev_results: Dict[str, Any]):
    mod_map = {
        "C": ("ai_core.services.step_c_service", "StepCService"),
        "DPRIME": ("ai_core.services.step_dprime_service", "StepDPrimeService"),
        "E": ("ai_core.services.step_e_service", "StepEService"),
        "F": ("ai_core.services.step_f_service", "StepFService"),
    }
    module_name, cls_name = mod_map[step_letter]
    try:
        module = __import__(module_name, fromlist=[cls_name])
    except Exception as exc:
        if step_letter == "DPRIME":
            raise RuntimeError(
                "Requested step DPRIME but StepD-prime module is unavailable: "
                f"missing {module_name}.{cls_name} ({type(exc).__name__}: {exc})"
            ) from exc
        raise
    cls = getattr(module, cls_name)

    # ctx is shared for ctor/run best-effort calls
    ctx = {"app_config": app_config, "symbol": symbol, "sym": symbol, "date_range": date_range, **prev_results}
    svc = _instantiate_service(cls, ctx)

    # Prefer common entrypoints. Some services (notably StepE/StepF) may not expose run().
    if step_letter == "E":
        candidates = ["run", "run_all", "run_all_models", "run_all_agents", "run_agents", "train", "train_all", "train_single_model", "execute", "main"]
    elif step_letter == "F":
        candidates = ["run", "train_marl", "run_marl", "train", "train_all", "execute", "main"]
    else:
        candidates = ["run", "execute", "main"]

    for name in candidates:
        fn = getattr(svc, name, None)
        if callable(fn):
            if step_letter in {"E", "F"}:
                ctx2 = dict(ctx)
                ctx2.pop("agents", None)
                return _call_with_best_effort(fn, ctx2)
            return _call_with_best_effort(fn, ctx)

    # If nothing matched, provide a helpful error with available callables.
    available = []
    for name in dir(svc):
        if name.startswith("_"):
            continue
        v = getattr(svc, name, None)
        if callable(v):
            available.append(name)
    raise RuntimeError(f"{cls_name} has no supported entrypoint. Tried={candidates}. Available={sorted(set(available))}")


def _run_leak_audits(output_root: Path, mode: str, symbol: str, fail_on_audit: bool = False) -> None:
    """Aggregate per-step audit reports into a summary CSV.

    Per-step audits are now run immediately after each step completes
    (audit_stepe_agent_now / audit_stepf_now).  This function reads the
    already-written JSON report files and produces the final summary CSV.
    If a JSON report is missing (e.g. step was cancelled), the function
    falls back to re-running the audit from the daily_log CSV if available.
    """
    import pandas as pd

    audit_root = output_root / "audit" / mode
    rows: List[Dict[str, Any]] = []

    # StepE: read existing per-agent audit JSON reports; re-run if absent
    step_e_dir = output_root / "stepE" / mode
    for p in sorted(step_e_dir.glob(f"stepE_daily_log_*_{symbol}.csv")):
        agent = p.name[len("stepE_daily_log_") : -len(f"_{symbol}.csv")]
        prefix = f"audit_stepE_{agent}_{symbol}"
        json_path = audit_root / f"{prefix}.json"
        if json_path.exists():
            try:
                import json as _json
                audit = _json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                audit = {"status": "FAIL", "note": "failed to read existing report"}
        else:
            # Fallback: re-run audit (step may have been cancelled before audit ran)
            try:
                df = pd.read_csv(p)
                audit = audit_stepE_reward_alignment(df, split="test", tol=1e-12)
                write_audit_reports(audit_root, prefix, audit)
            except Exception as e:
                audit = {"status": "FAIL", "note": str(e)}
        rows.append({
            "step": "E",
            "name": agent,
            "split": audit.get("split", "test"),
            "status": audit.get("status", "FAIL"),
            "max_abs": audit.get("max_abs"),
            "max_date": audit.get("max_date"),
            "rel_err": audit.get("rel_err"),
            "note": audit.get("note", ""),
        })

    # StepF: read existing audit JSON reports; re-run if absent
    for name in ("router", "marl"):
        p = output_root / "stepF" / mode / f"stepF_daily_log_{name}_{symbol}.csv"
        if not p.exists():
            continue
        prefix = f"audit_stepF_{name}_{symbol}"
        json_path = audit_root / f"{prefix}.json"
        if json_path.exists():
            try:
                import json as _json
                audit = _json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                audit = {"status": "FAIL", "note": "failed to read existing report"}
        else:
            try:
                df = pd.read_csv(p)
                audit = audit_stepF_market_alignment(df, split="test", tol=1e-12)
                write_audit_reports(audit_root, prefix, audit)
            except Exception as e:
                audit = {"status": "FAIL", "note": str(e)}
        rows.append({
            "step": "F",
            "name": name,
            "split": audit.get("split", "test"),
            "status": audit.get("status", "FAIL"),
            "max_abs": audit.get("max_abs"),
            "max_date": audit.get("max_date"),
            "rel_err": audit.get("rel_err"),
            "note": audit.get("note", ""),
        })

    if rows:
        summary_path = audit_root / f"leak_audit_summary_{symbol}.csv"
        pd.DataFrame(rows, columns=["step", "name", "split", "status", "max_abs", "max_date", "rel_err", "note"]).to_csv(summary_path, index=False)
        failed = [r for r in rows if str(r.get("status", "")).upper() != "PASS"]
        print(f"[audit] wrote summary={summary_path} total={len(rows)} failed={len(failed)}")
        if failed and fail_on_audit:
            raise RuntimeError(f"Leak audit failed: {failed}")


def _auto_run_id() -> str:
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{now}_{uuid.uuid4().hex[:8]}"


def _auto_branch_id(steps: Sequence[str], stepe_agents_csv: Optional[str]) -> str:
    stepe = (stepe_agents_csv or "all").strip() or "all"
    return f"steps={','.join(steps)}|stepe={stepe}"


def _set_timing_logger(app_config: Any, timing: TimingLogger) -> None:
    if isinstance(app_config, dict):
        app_config["_timing_logger"] = timing
    else:
        setattr(app_config, "_timing_logger", timing)


def _get_timing_logger(app_config: Any) -> TimingLogger:
    if isinstance(app_config, dict):
        t = app_config.get("_timing_logger")
    else:
        t = getattr(app_config, "_timing_logger", None)
    return t if isinstance(t, TimingLogger) else TimingLogger.disabled()


def _cfg_value(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _timing_settings_from_config(app_config: Any) -> tuple[bool, bool]:
    raw = _cfg_value(app_config, "timing", {})
    if not isinstance(raw, dict):
        raw = {}
    enabled = bool(raw.get("enable", False))
    clear = bool(raw.get("clear_on_run_start", False))
    return enabled, clear


def _preflight_ticc_backend_if_needed(*, steps: Sequence[str]) -> None:
    if "F" not in tuple(str(s).upper() for s in steps):
        return
    cmd = [sys.executable, str(_REPO_ROOT / "tools" / "check_ticc_backend.py")]
    print(f"[headless] TICC preflight command={' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.stderr:
        print(proc.stderr.rstrip(), file=sys.stderr)
    if int(proc.returncode) != 0:
        raise RuntimeError(f"TICC backend preflight failed rc={proc.returncode}")


def _determine_run_type(*, steps: Sequence[str], reuse_output: bool, force_rebuild: bool, resolved_mode: str, retrain: str) -> str:
    steps_upper = tuple(str(s).upper() for s in steps)
    if str(resolved_mode).lower() == "live":
        return "live_retrain_on" if str(retrain).lower() == "on" else "live_retrain_off"
    if force_rebuild:
        return "full_rebuild"
    if not reuse_output:
        return "full_rebuild"
    if steps_upper == ("DPRIME", "E", "F"):
        return "dprime_to_F"
    if steps_upper == ("F",):
        return "branch_only"
    if "A" not in steps_upper:
        return "reuse_up_to_A"
    if "B" not in steps_upper:
        return "reuse_up_to_B"
    if "C" not in steps_upper:
        return "reuse_up_to_C"
    return "full_rebuild"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--steps", default="A,B,C,DPRIME,E,F", help="Comma-separated steps to run. Example: A,B,C,DPRIME,E,F")
    ap.add_argument("--skip-stepe", dest="skip_stepe", action="store_true", help="Debug escape hatch: skip StepE explicitly.")
    ap.add_argument("--test-start", dest="test_start", default=None, help="YYYY-MM-DD. If omitted, uses last-3-months start.")
    ap.add_argument("--train-years", type=int, default=8)
    ap.add_argument("--test-months", type=int, default=3)
    ap.add_argument("--output-root", dest="output_root", default=None, help="Output root directory. Defaults to config output_root or 'output'.")
    ap.add_argument("--data-dir", dest="data_dir", default=None, help="Data root directory. Defaults to config data_root or 'data'.")
    ap.add_argument("--mode", dest="mode", default=None, choices=["sim", "live", "display"], help="Pipeline mode (legacy-compatible). If set, it overrides default values for --mamba-mode / --stepE-mode.")
    ap.add_argument("--future-end", dest="future_end", default=None, help="YYYY-MM-DD. Future end date for periodic features (can exceed last real price date). If omitted, uses TEST_END.")
    ap.add_argument("--mamba-mode", dest="mamba_mode", default=None, choices=["sim", "live", "display"], help="MAMBA training/inference mode. If omitted, defaults to --mode (or sim).")
    ap.add_argument("--stepE-mode", dest="stepE_mode", default=None, choices=["sim", "live", "display"], help="StepE (RL) mode. If omitted, defaults to --mode (or sim).")
    ap.add_argument("--stepE-use-stepd-prime", dest="stepE_use_stepd_prime", action="store_true",
                   help="StepE: consume StepD' transformer embeddings as sequential features (instead of StepD envelopes).")
    ap.add_argument("--stepE-dprime-sources", dest="stepE_dprime_sources", default=None,
                   help="StepE: comma-separated StepD' sources (e.g. 'mamba_periodic,mamba').")
    ap.add_argument("--stepE-dprime-horizons", dest="stepE_dprime_horizons", default=None,
                   help="StepE: comma-separated horizons for StepD' embeddings (e.g. '1,5,10,20').")
    ap.add_argument("--stepE-dprime-join", dest="stepE_dprime_join", default="inner", choices=["inner", "left"],
                   help="StepE: join method when merging embeddings by Date (inner or left).")
    ap.add_argument("--stepE-policy-kind", dest="stepE_policy_kind", default=None, choices=["ppo"], help="StepE: override headless policy kind (PPO only).")
    ap.add_argument("--stepE-ppo-total-timesteps", dest="stepE_ppo_total_timesteps", type=int, default=None, help="StepE: override PPO total timesteps for headless runs.")
    ap.add_argument("--stepE-ppo-n-epochs", dest="stepE_ppo_n_epochs", type=int, default=None, help="StepE: override PPO n_epochs for headless runs.")
    ap.add_argument("--stepE-max-parallel-agents", dest="stepE_max_parallel_agents", type=int, default=None, help="StepE: maximum concurrent PPO agents (capped at 2; recommended sim=2, live=1 then 2 after headroom validation).")
    ap.add_argument("--mamba-lookback", dest="mamba_lookback", type=int, default=None, help="StepB(Mamba) lookback_days (sequence length).")
    ap.add_argument("--mamba-horizons", dest="mamba_horizons", default=None, help="StepB(Mamba) horizons as CSV (e.g., 1,5,10,20).")
    ap.add_argument(
        "--env-horizon-days",
        dest="env_horizon_days",
        default=None,
        help="StepD envelope horizon preference/base horizon. Accepts single int or CSV (e.g. 20 or 1,5,10,20).",
    )
    ap.add_argument(
        "--env-horizons",
        dest="env_horizons",
        default=None,
        help="StepD envelope horizons as CSV list (e.g. 1,5,10,20). Overrides --env-horizon-days list component.",
    )
    ap.add_argument("--enable-mamba", action="store_true", help="Enable Wavelet-Mamba training in StepB (best-effort).")
    ap.add_argument("--enable-mamba-periodic", action="store_true", help="Also generate periodic-only Wavelet-Mamba snapshot predictions (uses periodic features only).")
    ap.add_argument("--auto-prepare-data", type=int, default=1, choices=[0, 1], help="Automatically generate missing data/prices_<SYMBOL>.csv before StepA.")
    ap.add_argument("--fail-on-audit", type=int, default=0, choices=[0, 1], help="If 1, fail pipeline when leak audit reports FAIL.")
    ap.add_argument("--data-start", default="2010-01-01", help="Start date for auto data preparation (YYYY-MM-DD).")
    ap.add_argument("--data-end", default=None, help="End date for auto data preparation (YYYY-MM-DD, default=today).")
    ap.add_argument("--timing", type=int, default=None, choices=[0, 1], help="Enable timing event logging.")
    ap.add_argument("--run-id", dest="run_id", default=None, help="Optional run identifier for timing logs.")
    ap.add_argument("--branch-id", dest="branch_id", default=None, help="Optional branch identifier for timing logs.")
    ap.add_argument("--execution-mode", dest="execution_mode", default="sequential", help="Execution mode label for timing logs.")
    ap.add_argument("--clear-timing", type=int, default=None, choices=[0, 1], help="Clear timing events file before run when --timing=1.")
    ap.add_argument("--stepe-agents", dest="stepe_agents", default=None, help="Optional CSV subset of StepE agents to run.")
    ap.add_argument("--enable-dprime-stream", dest="enable_dprime_stream", type=int, default=0, help="If 1, run StepB->StepC and DPrime base+cluster in parallel, then stream per-profile StepE.")
    ap.add_argument("--force-cpu", dest="force_cpu", type=int, default=0, choices=[0,1], help="If 1, force StepD' final profile generation to CPU (avoid GPU contention with StepE).")
    ap.add_argument("--reuse-output", dest="reuse_output", type=int, default=0, choices=[0, 1],
                    help="If 1, reuse artifacts from prior runs with matching signature (skip complete steps).")
    ap.add_argument("--force-rebuild", dest="force_rebuild", type=int, default=0, choices=[0, 1],
                    help="If 1, ignore existing artifacts and rebuild all steps even if reuse-output=1.")
    ap.add_argument("--stepf-compare-reward-modes", dest="stepf_compare_reward_modes", type=int, default=1, choices=[0, 1],
                    help="If 1, StepF compare path runs multiple reward modes (legacy, profit_basic, profit_regret, profit_light_risk).")
    ap.add_argument("--stepf-reward-modes", dest="stepf_reward_modes", default="",
                    help="Optional CSV override for StepF reward modes (e.g. legacy,profit_basic,profit_regret,profit_light_risk).")
    args = ap.parse_args(argv)

    repo_root = _repo_root()

    symbol = args.symbol or _env_get("AUTODEBUG_SYMBOL", "AUTO_DEBUG_SYMBOL") or "SOXL"
    steps = _parse_steps(args.steps)
    _preflight_ticc_backend_if_needed(steps=steps)
    skip_stepe_env = str(_env_get("SKIP_STEPE") or "").strip().lower() in {"1", "true", "yes", "on"}
    skip_stepe = bool(args.skip_stepe or skip_stepe_env)
    if skip_stepe and "E" in steps:
        steps = tuple(step for step in steps if step != "E")

    enable_mamba = bool(args.enable_mamba)
    reuse_output = bool(int(args.reuse_output or _env_get("REUSE_OUTPUT") or 0))
    force_rebuild = bool(int(args.force_rebuild or _env_get("FORCE_REBUILD") or 0))
    if force_rebuild:
        reuse_output = True  # force_rebuild implies reuse_output context (manifest is written)

    # Ensure repo_root is on sys.path
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    app_config = _get_app_config(repo_root)

    resolved_mode = (args.mode or "sim").strip().lower()
    canonical_test_start = str(args.test_start or "unknown_test_start")
    cli_output_root: Optional[Path] = None
    if args.output_root:
        cli_output_root = Path(args.output_root).expanduser()
        if not cli_output_root.is_absolute():
            cli_output_root = (repo_root / cli_output_root).resolve()
        else:
            cli_output_root = cli_output_root.resolve()
    try:
        from tools.run_manifest import resolve_canonical_output_root as _resolve_canonical_output_root

        canonical_output_root = cli_output_root or _resolve_canonical_output_root(resolved_mode, symbol, canonical_test_start)
    except Exception:
        if cli_output_root is None:
            raise
        canonical_output_root = cli_output_root
    resolved_output_root = canonical_output_root
    resolved_mamba_mode = (args.mode or args.mamba_mode or "sim").strip().lower()
    resolved_stepE_mode = (args.mode or args.stepE_mode or "sim").strip().lower()

    resolved_output_root.mkdir(parents=True, exist_ok=True)
    (resolved_output_root / "logs").mkdir(parents=True, exist_ok=True)
    app_config = _apply_config_output_root(app_config, resolved_output_root)
    print(f"[PIPELINE] args_output_root={args.output_root}")
    print(f"[PIPELINE] resolved_output_root={resolved_output_root}")
    print(f"[PIPELINE] canonical_output_root={canonical_output_root}")
    print(f"[PIPELINE] cfg_output_root={getattr(app_config, 'output_root', None)}")
    print(f"[PIPELINE] cfg_data_output_root={getattr(getattr(app_config, 'data', None), 'output_root', None)}")
    reuse_signature_payload = {
        "mode": resolved_mode,
        "symbols": [symbol],
        "test_start_date": str(args.test_start or canonical_test_start or ""),
        "train_years": int(args.train_years),
        "test_months": int(args.test_months),
        "feature_signature": "canonical_stepA_stepF",
        "algorithm_signature": f"mamba={int(enable_mamba)};mamba_periodic={int(bool(args.enable_mamba_periodic))}",
        "parameter_signature": f"mamba_lookback={args.mamba_lookback};mamba_horizons={args.mamba_horizons};steps={','.join(steps)}",
    }
    reuse_signature_payload["reuse_key_hash"] = uuid.uuid5(uuid.NAMESPACE_URL, json.dumps(reuse_signature_payload, sort_keys=True)).hex[:16]
    (Path(resolved_output_root) / "reuse_signature.json").write_text(json.dumps(reuse_signature_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- Run-reuse manifest (initialised early; steps update it as they complete) ---
    _run_manifest = None
    if True:
        try:
            from tools.run_manifest import (  # lazy import to avoid hard dep
                RunManifest,
                build_run_signature,
                check_step_artifacts,
                check_stepe_agent_artifact,
            )
            _mamba_horizons_for_sig = tuple(_parse_int_list(args.mamba_horizons or "") or [])
            _stepe_agents_for_sig: Optional[Tuple[str, ...]] = None
            if args.stepe_agents:
                _stepe_agents_for_sig = tuple(a.strip() for a in args.stepe_agents.split(",") if a.strip())
            _run_sig = build_run_signature(
                symbol=symbol,
                mode=resolved_mode,
                test_start=args.test_start or "",
                train_years=int(args.train_years),
                test_months=int(args.test_months),
                steps=steps,
                enable_mamba=enable_mamba,
                enable_mamba_periodic=bool(args.enable_mamba_periodic),
                mamba_lookback=args.mamba_lookback,
                mamba_horizons=_mamba_horizons_for_sig,
                stepe_agents=_stepe_agents_for_sig,
            )
            _run_manifest = RunManifest.load_or_create(
                resolved_output_root, _run_sig, reuse_output, force_rebuild
            )
            print(
                f"[reuse] manifest loaded: hash={_run_sig.stable_hash()} "
                f"reuse={reuse_output} force_rebuild={force_rebuild} "
                f"output_root={resolved_output_root}"
            )
            print(f"[REUSE] source_output_root={resolved_output_root}")
            try:
                _run_manifest.mark_source_output_root(str(resolved_output_root))
            except Exception:
                pass
        except Exception as _e:
            print(f"[reuse] ERROR: failed to initialise run manifest: {type(_e).__name__}: {_e}", file=sys.stderr)
            try:
                import traceback as _traceback
                _traceback.print_exc()
            except Exception:
                pass
            _run_manifest = None

    def _stepf_required_reward_modes() -> Tuple[str, ...]:
        cfg = app_config.get("stepF") if isinstance(app_config, dict) else getattr(app_config, "stepF", None)
        if cfg is None:
            return tuple()
        compare_enabled = bool(getattr(cfg, "stepf_compare_reward_modes", False))
        raw_modes = str(getattr(cfg, "stepf_reward_modes", "") or "").strip()
        if raw_modes:
            return tuple(dict.fromkeys([m.strip().lower() for m in raw_modes.split(",") if m.strip()]))
        if compare_enabled:
            return ("legacy", "profit_basic", "profit_regret", "profit_light_risk")
        mode_name = str(getattr(cfg, "reward_mode", "legacy") or "legacy").strip().lower()
        return (mode_name,) if mode_name else ("legacy",)

    def _stepf_collect_manifest_fields() -> Dict[str, Any]:
        mode_root = Path(resolved_output_root) / "stepF" / resolved_mode
        summary_path = mode_root / f"stepF_multi_mode_summary_{symbol}.json"
        payload: Dict[str, Any] = {
            "reward_modes_requested": list(_stepf_required_reward_modes()),
            "reward_modes_completed": [],
            "reward_modes_failed": [],
            "reward_modes_reused": [],
            "publish_completed": False,
        }
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                summary = {}
            if isinstance(summary, dict):
                payload["reward_modes_requested"] = list(summary.get("reward_modes", payload["reward_modes_requested"]))
                payload["reward_modes_completed"] = list(summary.get("success_modes", []))
                payload["reward_modes_failed"] = list(summary.get("failed_modes", []))
                payload["reward_modes_reused"] = list(summary.get("reused_modes", []))
                payload["publish_completed"] = bool(summary.get("publish_completed", False))
                payload["multi_mode_summary_path"] = str(summary_path)
                return payload

        completed: List[str] = []
        failed: List[str] = []
        reused: List[str] = []
        requested = payload["reward_modes_requested"]
        for rm in requested:
            status_path = mode_root / f"reward_{rm}" / "status.json"
            if not status_path.exists():
                continue
            try:
                status_payload = json.loads(status_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            status_value = str(status_payload.get("status", "")).lower()
            if status_value == "complete":
                completed.append(rm)
                if bool(status_payload.get("reused", False)):
                    reused.append(rm)
            elif status_value in {"failed", "interrupted"}:
                failed.append(rm)
        payload["reward_modes_completed"] = completed
        payload["reward_modes_failed"] = failed
        payload["reward_modes_reused"] = reused
        payload["publish_completed"] = False
        return payload

    def _update_stepf_manifest_fields(status: str, *, error: BaseException | None = None) -> None:
        if _run_manifest is None:
            return
        try:
            step_data = _run_manifest._data.setdefault("steps", {}).setdefault("F", {})
            step_data.update(_stepf_collect_manifest_fields())
            step_data["status"] = str(status)
            if status == "completed":
                step_data["status"] = "complete"
            if status in {"failed", "interrupted"}:
                step_data["failed_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            if error is not None:
                step_data["error_type"] = type(error).__name__
                step_data["error_message"] = str(error)[:500]
            _run_manifest._data["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            _run_manifest.save()
        except Exception as _manifest_f_exc:
            print(f"[StepF] WARN manifest compare metadata update failed: {_manifest_f_exc}", file=sys.stderr)

    def _can_reuse_step(step_key: str) -> bool:
        """Return True iff step can be skipped due to reuse."""
        if not reuse_output or force_rebuild or _run_manifest is None:
            return False
        try:
            required_reward_modes = _stepf_required_reward_modes() if str(step_key).upper() == "F" else tuple()
            required_dprime_profiles = tuple(_extract_stepe_agents_from_config(app_config) or list(_OFFICIAL_STEPE_AGENTS)) if str(step_key).upper() == "DPRIME" else tuple()
            return (
                _run_manifest.can_reuse_step(step_key)
                and check_step_artifacts(
                    step_key,
                    resolved_output_root,
                    symbol,
                    resolved_mode,
                    required_stepf_reward_modes=required_reward_modes,
                    required_dprime_profiles=required_dprime_profiles,
                )
            )
        except Exception:
            return False

    def _mark_step(step_key: str, status: str) -> None:
        if _run_manifest is not None:
            try:
                _run_manifest.mark_step(step_key, status)
            except Exception:
                pass

    def _finalize_manifest_step(step_key: str, *, status: str, elapsed_sec: Optional[float], audit_status: Optional[str] = None) -> None:
        if _run_manifest is None:
            return
        try:
            _run_manifest.mark_step(step_key, status)
            if elapsed_sec is not None:
                _run_manifest.mark_step_elapsed(step_key, float(elapsed_sec))
            if audit_status is not None:
                _run_manifest.mark_step_audit(step_key, audit_status)
            try:
                step_data = _run_manifest._data.setdefault("steps", {}).setdefault(step_key.upper(), {})
                if status == "failed":
                    step_data["failed"] = True
                    step_data["completed_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                _run_manifest._data["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            except Exception:
                pass
            _run_manifest.save()
        except Exception as _manifest_step_e:
            print(
                f"[reuse] WARNING manifest finalize failed step={step_key} "
                f"reason={type(_manifest_step_e).__name__}: {_manifest_step_e}",
                file=sys.stderr,
            )

    def _mark_step_verified(step_key: str, status: str, *, audit_status: Optional[str] = None, invalid_status: str = "pending") -> bool:
        if _run_manifest is None:
            return True
        try:
            required_reward_modes = _stepf_required_reward_modes() if str(step_key).upper() == "F" else tuple()
            required_dprime_profiles = tuple(_extract_stepe_agents_from_config(app_config) or list(_OFFICIAL_STEPE_AGENTS)) if str(step_key).upper() == "DPRIME" else tuple()
            artifacts_ok = check_step_artifacts(
                step_key,
                resolved_output_root,
                symbol,
                resolved_mode,
                required_stepf_reward_modes=required_reward_modes,
                required_dprime_profiles=required_dprime_profiles,
            )
            return _run_manifest.mark_step_verified(
                step_key,
                status,
                artifacts_ok=artifacts_ok,
                audit_status=audit_status,
                invalid_status=invalid_status,
            )
        except Exception:
            return False

    def _mark_stepe_agent_verified(agent: str, status: str, *, audit_status: Optional[str] = None, invalid_status: str = "pending") -> bool:
        if _run_manifest is None:
            return True
        try:
            artifacts_ok = check_stepe_agent_artifact(agent, resolved_output_root, symbol, resolved_mode)
            return _run_manifest.mark_stepe_agent_verified(
                agent,
                status,
                artifacts_ok=artifacts_ok,
                audit_status=audit_status,
                invalid_status=invalid_status,
            )
        except Exception:
            return False

    def _normalize_manifest_reuse_statuses() -> None:
        if _run_manifest is None:
            return
        try:
            from tools.run_manifest import (
                read_stepe_quality_status as _read_stepe_quality_status,
                reconcile_stepf_manifest_from_artifacts as _reconcile_stepf_manifest_from_artifacts,
            )

            for _step in ("A", "B", "C", "DPRIME"):
                _required_reward_modes = _stepf_required_reward_modes() if _step == "F" else tuple()
                _required_dprime_profiles = tuple(_extract_stepe_agents_from_config(app_config) or list(_OFFICIAL_STEPE_AGENTS)) if _step == "DPRIME" else tuple()
                _artifact_ok = check_step_artifacts(
                    _step,
                    resolved_output_root,
                    symbol,
                    resolved_mode,
                    required_stepf_reward_modes=_required_reward_modes,
                    required_dprime_profiles=_required_dprime_profiles,
                )
                if not _artifact_ok and _run_manifest.step_status(_step) in ("complete", "reuse"):
                    _run_manifest.mark_step_verified(_step, "complete", artifacts_ok=False, audit_status="FAIL", invalid_status="pending")
                elif _artifact_ok and _run_manifest.step_status(_step) not in ("complete", "reuse"):
                    _existing_audit = _run_manifest._data.setdefault("steps", {}).setdefault(_step, {}).get("audit_status")
                    _run_manifest.mark_step_verified(_step, "complete", artifacts_ok=True, audit_status=str(_existing_audit or "PASS"), invalid_status="pending")
            for _agent in _extract_stepe_agents_from_config(app_config) or list(_OFFICIAL_STEPE_AGENTS):
                _agent_artifact_ok = check_stepe_agent_artifact(_agent, resolved_output_root, symbol, resolved_mode)
                if not _agent_artifact_ok and _run_manifest.stepe_agent_status(_agent) in ("complete", "reuse"):
                    _run_manifest.mark_stepe_agent_verified(_agent, "complete", artifacts_ok=False, audit_status="FAIL", invalid_status="pending")
                    continue
                if _agent_artifact_ok and _run_manifest.stepe_agent_status(_agent) not in ("complete", "reuse"):
                    _agent_audit_status = _read_stepe_quality_status(_agent, resolved_output_root, symbol, resolved_mode) or "PASS"
                    _run_manifest.mark_stepe_agent_verified(
                        _agent,
                        "complete" if _agent_audit_status == "PASS" else "failed",
                        artifacts_ok=True,
                        audit_status=_agent_audit_status,
                        invalid_status="pending",
                    )
            _stepe_agents = _extract_stepe_agents_from_config(app_config) or list(_OFFICIAL_STEPE_AGENTS)
            _stepe_complete = bool(_stepe_agents) and all(_run_manifest.can_reuse_stepe_agent(_agent) and check_stepe_agent_artifact(_agent, resolved_output_root, symbol, resolved_mode) for _agent in _stepe_agents)
            if _stepe_agents and not _stepe_complete:
                if _run_manifest.step_status("E") in ("complete", "reuse"):
                    _run_manifest.mark_step_verified("E", "complete", artifacts_ok=False, audit_status="FAIL", invalid_status="pending")
            elif _stepe_complete and _run_manifest.step_status("E") not in ("complete", "reuse"):
                _run_manifest.mark_step_verified("E", "complete", artifacts_ok=True, audit_status="PASS", invalid_status="pending")
            _reconcile_stepf_manifest_from_artifacts(
                output_root=Path(resolved_output_root),
                mode=resolved_mode,
                symbol=symbol,
                requested_reward_modes=list(_stepf_required_reward_modes()),
                manifest=_run_manifest,
            )
        except Exception as _manifest_norm_e:
            print(f"[reuse] WARNING manifest normalization failed: {type(_manifest_norm_e).__name__}: {_manifest_norm_e}", file=sys.stderr)

    _normalize_manifest_reuse_statuses()

    def _emit_step_status(step_key: str, *, status: str, started_at: float, ended_at: float, validated: Optional[bool], detail: str = "") -> None:
        start_ts = datetime.fromtimestamp(started_at, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        end_ts = datetime.fromtimestamp(ended_at, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        elapsed = max(0.0, ended_at - started_at)
        validated_str = "yes" if validated else ("no" if validated is False else "n/a")
        extra = f" detail={detail}" if detail else ""
        print(
            f"[STEP_METRIC] step={step_key} start={start_ts} end={end_ts} elapsed_sec={elapsed:.3f} "
            f"status={status} artifacts_validated={validated_str}{extra}"
        )

    def _evaluate_stepe_final_state(requested_agents: List[str]) -> Dict[str, Any]:
        """Recompute final StepE completion state from artifacts + persisted quality payloads."""
        from tools.run_manifest import reconcile_stepe_manifest_from_artifacts  # lazy import

        _result = reconcile_stepe_manifest_from_artifacts(
            requested_agents,
            output_root=Path(resolved_output_root),
            mode=resolved_mode,
            symbol=symbol,
            manifest=_run_manifest,
        )

        print(f"[STEPE_FINAL] requested_agents={','.join(_result['requested_agents'])}")
        print(f"[STEPE_FINAL] complete_agents={','.join(_result['complete_agents'])}")
        print(f"[STEPE_FINAL] missing_agents={','.join(_result['missing_agents'])}")
        print(f"[STEPE_FINAL] final_status={_result['final_status']}")
        print(f"[STEPE_FINAL] return_code={_result['return_code']}")
        if _result["all_complete"]:
            print("[STEPE_FINAL] all_requested_agents_complete=true")
            print("[STEPE_FINAL] stepE_status=complete")
            print("[STEPE_FINAL] return_code=0")
        else:
            for _agent in _result["missing_agents"]:
                print(f"[STEPE_FINAL] missing_agent={_agent} missing={'|'.join(_result['missing_detail'].get(_agent, []))}")

        return _result

    def _evaluate_stepf_final_state() -> Dict[str, Any]:
        from tools.run_manifest import reconcile_stepf_manifest_from_artifacts  # lazy import

        _result = reconcile_stepf_manifest_from_artifacts(
            output_root=Path(resolved_output_root),
            mode=resolved_mode,
            symbol=symbol,
            requested_reward_modes=list(_stepf_required_reward_modes()),
            manifest=_run_manifest,
        )
        print(f"[STEPF_FINAL] requested_reward_modes={','.join(_result['requested_reward_modes']) if _result['requested_reward_modes'] else '(none)'}")
        print(f"[STEPF_FINAL] success_modes={','.join(_result['success_modes']) if _result['success_modes'] else '(none)'}")
        print(f"[STEPF_FINAL] failed_modes={','.join(_result['failed_modes']) if _result['failed_modes'] else '(none)'}")
        print(f"[STEPF_FINAL] publish_completed={str(bool(_result['publish_completed'])).lower()}")
        print(f"[STEPF_FINAL] artifacts_ok={str(bool(_result['artifacts_ok'])).lower()}")
        print(f"[STEPF_FINAL] audit_status={_result['audit_status']}")
        for _issue in _result.get("structured_inconsistencies", []):
            print(f"[STEPF_FINAL] structured_inconsistency={json.dumps(_issue, ensure_ascii=False, sort_keys=True)}")
        return _result

    def _run_stepe_agents_incrementally(
        *,
        requested_agents: List[str],
        cfg_by_agent: Dict[str, Any],
        initial_reused_agents: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, Any], float, List[str]]:
        """Run StepE one agent at a time so partial completion survives interruptions."""
        _original_step_e = app_config.get("stepE") if isinstance(app_config, dict) else getattr(app_config, "stepE", None)
        _agent_results: Dict[str, Any] = {}
        _total_elapsed = 0.0
        _agent_failures: List[str] = []
        _audit_root_e = Path(resolved_output_root) / "audit" / resolved_mode
        _reused = list(initial_reused_agents or [])
        try:
            for _agent in requested_agents:
                _cfg = cfg_by_agent.get(_agent)
                if _cfg is None:
                    raise RuntimeError(f"StepE agent config not found: {_agent}")
                if isinstance(app_config, dict):
                    app_config["stepE"] = [_cfg]
                else:
                    setattr(app_config, "stepE", [_cfg])

                print(f"[STEPE_AGENT] agent={_agent} action=run")
                _mark_step("E", "running")
                if _run_manifest is not None:
                    _run_manifest.mark_stepe_agent(_agent, "running")
                _t0_agent = time.perf_counter()
                try:
                    with timing.stage(f"{stage_name}.{_agent}"):
                        _agent_result = _run_step_generic("E", app_config, symbol, date_range, results)
                except Exception:
                    _elapsed_agent = time.perf_counter() - _t0_agent
                    _total_elapsed += _elapsed_agent
                    if _run_manifest is not None:
                        _run_manifest.mark_stepe_agent_elapsed(_agent, _elapsed_agent)
                        _run_manifest.mark_stepe_agent(_agent, "failed")
                        _run_manifest.mark_stepe_agent_audit(_agent, "FAIL")
                    raise

                _elapsed_agent = time.perf_counter() - _t0_agent
                _total_elapsed += _elapsed_agent
                _agent_results[_agent] = _agent_result

                if isinstance(_agent_result, dict) and _agent_result.get("skipped"):
                    _reason = _agent_result.get("reason", "unknown")
                    if _run_manifest is not None:
                        _run_manifest.mark_stepe_agent_elapsed(_agent, _elapsed_agent)
                        _run_manifest.mark_stepe_agent(_agent, "failed")
                        _run_manifest.mark_stepe_agent_audit(_agent, "FAIL")
                    raise RuntimeError(f"StepE agent={_agent} reported skipped=True without explicit skip flag. reason={_reason}")

                _artifact_ok = check_stepe_agent_artifact(_agent, resolved_output_root, symbol, resolved_mode)
                _ag_status = "FAIL"
                if _artifact_ok:
                    try:
                        _ag_audit = audit_stepe_agent_now(
                            Path(resolved_output_root), resolved_mode, symbol, _agent, _audit_root_e
                        )
                        _ag_status = _ag_audit.get("status", "FAIL")
                    except Exception as _ae:
                        print(f"[StepE] WARN audit agent={_agent}: {_ae}", file=sys.stderr)
                else:
                    print(f"[STEPE_AGENT] agent={_agent} action=fail reason=artifact_missing")

                if _run_manifest is not None:
                    _run_manifest.mark_stepe_agent_elapsed(_agent, _elapsed_agent)
                    _mark_stepe_agent_verified(_agent, "complete", audit_status=_ag_status, invalid_status="pending")

                if _artifact_ok and _ag_status == "PASS":
                    print(f"[STEPE_AGENT] agent={_agent} action=complete elapsed_sec={round(_elapsed_agent, 3)}")
                else:
                    _agent_failures.append(_agent)
                    _reason = "artifact_missing" if not _artifact_ok else f"audit_{str(_ag_status).lower()}"
                    print(f"[STEPE_AGENT] agent={_agent} action=fail reason={_reason}")
        finally:
            if isinstance(app_config, dict):
                app_config["stepE"] = _original_step_e
            else:
                setattr(app_config, "stepE", _original_step_e)

        return {"reused_agents": _reused, "agent_results": _agent_results}, _total_elapsed, _agent_failures

    cfg_timing_enabled, cfg_timing_clear = _timing_settings_from_config(app_config)
    timing_enabled = bool(int(args.timing)) if args.timing is not None else cfg_timing_enabled
    run_id = args.run_id or _auto_run_id()
    branch_id = args.branch_id or _auto_branch_id(steps, args.stepe_agents)
    stepf_cfg_for_timing = app_config.get("stepF") if isinstance(app_config, dict) else getattr(app_config, "stepF", None)
    stepf_retrain = str(getattr(stepf_cfg_for_timing, "retrain", "off") or "off")
    run_type = _determine_run_type(
        steps=steps,
        reuse_output=reuse_output,
        force_rebuild=force_rebuild,
        resolved_mode=resolved_mode,
        retrain=stepf_retrain,
    )
    timing = TimingLogger(
        output_root=resolved_output_root,
        mode=resolved_mode,
        run_id=run_id,
        branch_id=branch_id,
        execution_mode=str(args.execution_mode or "sequential"),
        enabled=timing_enabled,
        clear=(bool(int(args.clear_timing)) if args.clear_timing is not None else cfg_timing_clear) and timing_enabled,
        run_type=run_type,
        symbol=symbol,
    )
    _set_timing_logger(app_config, timing)
    print(
        f"[PIPELINE] resolved_output_root={resolved_output_root} "
        f"cfg_output_root={getattr(app_config, 'output_root', None)} "
        f"cfg_data_output_root={getattr(getattr(app_config, 'data', None), 'output_root', None)}"
    )

    run_log_path = _ensure_run_log_file(run_id=run_id, output_root=resolved_output_root)
    if run_log_path is not None:
        print(f"[PIPELINE] run_log_path={run_log_path}")

    pipeline_status = "running"
    pipeline_error: Optional[str] = None

    resolved_data_root = _resolve_cli_data_dir(repo_root=repo_root, cli_data_dir=args.data_dir)
    app_config = _apply_config_data_dir(app_config, resolved_data_root)

    data_root = resolved_data_root
    env_horizon_list = _parse_int_list(args.env_horizons or args.env_horizon_days)
    env_horizon_base: Optional[int] = None
    if args.env_horizon_days is not None:
        parsed_base = _parse_int_list(args.env_horizon_days)
        if parsed_base and len(parsed_base) == 1:
            env_horizon_base = int(parsed_base[0])

    # Pass envelope horizon preference into services (best-effort)
    if env_horizon_base is not None or env_horizon_list:
        if isinstance(app_config, dict):
            if env_horizon_base is not None:
                app_config['env_horizon_days'] = int(env_horizon_base)
            app_config['env_horizons'] = list(env_horizon_list or [])
        else:
            try:
                if env_horizon_base is not None:
                    setattr(app_config, 'env_horizon_days', int(env_horizon_base))
                setattr(app_config, 'env_horizons', list(env_horizon_list or []))
            except Exception:
                app_config = _ConfigShim(app_config, env_horizon_days=env_horizon_base, env_horizons=list(env_horizon_list or []))
    auto_prepare_data = bool(int(args.auto_prepare_data))
    data_start = args.data_start
    data_end = args.data_end or date.today().isoformat()

    with timing.stage("common.prepare_data"):
        _prepare_missing_data_if_needed(
            repo_root=repo_root,
            data_root=data_root,
            primary_symbol=symbol,
            auto_prepare_data=auto_prepare_data,
            data_start=data_start,
            data_end=data_end,
        )

    future_end = args.future_end or _env_get("AUTODEBUG_FUTURE_END", "FUTURE_END", "FUTURE_END_DATE")
    with timing.stage("common.build_date_range"):
        date_range = _build_date_range(
        symbol,
        repo_root,
        args.test_start,
        args.test_months,
        args.train_years,
        future_end=future_end,
        mamba_mode=resolved_mamba_mode,
        stepE_mode=resolved_mode,
        mode=resolved_mode,
        output_root=resolved_output_root,
        data_root=data_root,
        env_horizon_days=env_horizon_base,
    )

    # Best-effort propagation: make date_range visible from app_config for downstream services.
    try:
        if isinstance(app_config, dict):
            app_config["date_range"] = date_range
        else:
            setattr(app_config, "date_range", date_range)
    except Exception:
        app_config = _ConfigShim(app_config, date_range=date_range)

    try:
        split_payload = {
            "symbol": str(symbol).upper(),
            "mode": str(resolved_mode),
            "test_start": str(getattr(date_range, "test_start", canonical_test_start))[:10],
            "train_start": str(getattr(date_range, "train_start", ""))[:10],
            "train_end": str(getattr(date_range, "train_end", ""))[:10],
            "test_end": str(getattr(date_range, "test_end", ""))[:10],
            "train_years": int(args.train_years),
            "test_months": int(args.test_months),
        }
        _write_split_summary_json_payload(Path(canonical_output_root) / "split_summary.json", split_payload, "reuse")
        if Path(canonical_output_root).resolve() != Path(resolved_output_root).resolve():
            _write_split_summary_json_payload(Path(resolved_output_root) / "split_summary.json", split_payload, "reuse")
    except Exception as _split_e:
        print(f"[reuse] WARNING split summary write failed: {type(_split_e).__name__}: {_split_e}", file=sys.stderr)

    results: Dict[str, Any] = {}

    # StepB enabled models list.
    enabled_agents: List[str] = ['mamba'] if enable_mamba else []
    # Store into results so downstream steps can read ctx['agents'] if needed.
    results['agents'] = enabled_agents
    # Optional: configure StepE to use StepD' transformer embeddings (compression-as-learning).
    # This lets StepE consume embeddings from:
    #   <output_root>/stepDprime/<mode>/embeddings/stepDprime_{source}_h{HH}_{SYMBOL}_embeddings.csv
    # rather than relying on StepD envelopes as sequential features.
    stepe_use_dprime = bool(getattr(args, "stepE_use_stepd_prime", False) or getattr(args, "stepE_dprime_sources", None) or getattr(args, "stepE_dprime_horizons", None))
    if stepe_use_dprime:
        try:
            from ai_core.services.step_e_service import StepEConfig  # lazy import
            dprime_sources = (args.stepE_dprime_sources or "mamba_periodic,mamba").strip()
            dprime_horizons = (args.stepE_dprime_horizons or "1,5,10,20").strip()
            dprime_join = (args.stepE_dprime_join or "inner").strip().lower()
            cfg_list = []
            for a in (enabled_agents or ["mamba"]):
                cfg = StepEConfig(agent=a)
                cfg.output_root = str(getattr(app_config, "output_root", "output"))
                cfg.use_stepd_prime = True
                cfg.dprime_sources = dprime_sources
                cfg.dprime_horizons = dprime_horizons
                cfg.dprime_join = dprime_join
                cfg.max_parallel_agents = int(getattr(args, "stepE_max_parallel_agents", None) or 1)
                cfg_list.append(cfg)
            setattr(app_config, "stepE", cfg_list if len(cfg_list) != 1 else cfg_list[0])
            print(f"[headless] StepE configured to use StepD' embeddings: sources={dprime_sources} horizons={dprime_horizons} join={dprime_join}")
        except Exception as e:
            print(f"[headless] WARNING: failed to configure StepE StepD' embeddings: {type(e).__name__}: {e}")

    # Ensure StepE has at least one runnable config in headless pipeline mode.
    if "E" in steps:
        raw_step_e_cfgs = getattr(app_config, "stepE", None)
        if raw_step_e_cfgs is None:
            current_step_e_cfgs = []
        elif isinstance(raw_step_e_cfgs, (list, tuple)):
            current_step_e_cfgs = list(raw_step_e_cfgs)
        else:
            current_step_e_cfgs = [raw_step_e_cfgs]

        if not current_step_e_cfgs:
            try:
                _inject_default_stepe_configs(app_config, resolved_output_root)
                print(f"[headless] StepE default config injected: agents={','.join(_OFFICIAL_STEPE_AGENTS)} seed=42+idx device=auto")
            except Exception as e:
                print(f"[headless] WARNING: failed to inject default StepE config: {type(e).__name__}: {e}")
        _apply_headless_stepe_overrides(app_config, args)

    if args.stepe_agents:
        requested_agents = {a.strip() for a in str(args.stepe_agents).split(",") if a.strip()}
        raw_step_e_cfgs = app_config.get("stepE") if isinstance(app_config, dict) else getattr(app_config, "stepE", None)
        if raw_step_e_cfgs is not None:
            cfg_list = list(raw_step_e_cfgs) if isinstance(raw_step_e_cfgs, (list, tuple)) else [raw_step_e_cfgs]
            cfg_list = [cfg for cfg in cfg_list if getattr(cfg, "agent", "") in requested_agents]
            if isinstance(app_config, dict):
                app_config["stepE"] = cfg_list
            else:
                setattr(app_config, "stepE", cfg_list)
            print(f"[headless] StepE filtered agents={','.join(sorted(requested_agents))} retained={len(cfg_list)}")

    if "F" in steps:
        step_f_cfg = app_config.get("stepF") if isinstance(app_config, dict) else getattr(app_config, "stepF", None)
        if step_f_cfg is None:
            try:
                from ai_core.services.step_f_service import StepFRouterConfig  # lazy import

                out_root = str(resolved_output_root)
                mode = resolved_mamba_mode or args.mode or "sim"
                step_e_daily_dir = Path(out_root) / "stepE" / str(mode)
                daily_log_pattern = f"stepE_daily_log_*_{symbol}.csv"

                unique_agents = {
                    str(p.name)[len("stepE_daily_log_"):-len(f"_{symbol}.csv")]
                    for p in step_e_daily_dir.glob(daily_log_pattern)
                    if p.name.startswith("stepE_daily_log_") and p.name.endswith(f"_{symbol}.csv")
                }
                if not unique_agents:
                    unique_agents = set(_extract_stepe_agents_from_config(app_config))
                if not unique_agents:
                    unique_agents = set(_OFFICIAL_STEPE_AGENTS)

                ordered_agents = [a for a in _OFFICIAL_STEPE_AGENTS if a in unique_agents]
                for a in sorted(unique_agents):
                    if a not in ordered_agents:
                        ordered_agents.append(a)
                safe_defaults = ["dprime_bnf_h01", "dprime_all_features_h01", "dprime_mix_3scale"]
                safe_defaults = [x for x in safe_defaults if x in ordered_agents]
                if len(safe_defaults) < 2:
                    safe_defaults = ordered_agents[: min(3, len(ordered_agents))]

                cfgF = StepFRouterConfig(
                    output_root=out_root,
                    agents=",".join(ordered_agents),
                    mode=str(mode),
                    safe_set=",".join(safe_defaults),
                    use_z_pred=False,
                    trade_cost_bps=15.0,
                )

                if isinstance(app_config, dict):
                    app_config["stepF"] = cfgF
                else:
                    setattr(app_config, "stepF", cfgF)

                print(
                    f"[headless] StepF default config injected: agents={cfgF.agents} safe_set={cfgF.safe_set} "
                    f"router(beta={cfgF.softmax_beta},ema={cfgF.ema_alpha},cost_bps={cfgF.trade_cost_bps})"
                )
            except Exception as e:
                print(f"[headless] WARNING: failed to inject default StepF config: {type(e).__name__}: {e}")

        step_f_cfg = app_config.get("stepF") if isinstance(app_config, dict) else getattr(app_config, "stepF", None)
        if step_f_cfg is not None:
            _compare_enabled = bool(int(args.stepf_compare_reward_modes))
            _raw_modes = str(args.stepf_reward_modes or "").strip()
            if _raw_modes:
                _modes = ",".join([m.strip().lower() for m in _raw_modes.split(",") if m.strip()])
            elif _compare_enabled:
                _modes = "legacy,profit_basic,profit_regret,profit_light_risk"
            else:
                _modes = str(getattr(step_f_cfg, "reward_mode", "legacy") or "legacy").strip().lower()
            setattr(step_f_cfg, "stepf_compare_reward_modes", _compare_enabled)
            setattr(step_f_cfg, "stepf_reward_modes", _modes)
            print(f"[headless] StepF reward compare enabled={int(_compare_enabled)} modes={_modes}")


    print(f"[headless] repo_root={repo_root}")
    data_prepare_symbols = _symbols_for_data_prep(symbol)
    stepa_execution_symbols = _symbols_for_stepa_execution(symbol)
    print(f"[headless] symbol={symbol}")
    print(f"[headless] primary_symbol={symbol}")
    print(f"[headless] data_prepare_symbols={','.join(data_prepare_symbols)}")
    print(f"[headless] stepa_execution_symbols={','.join(stepa_execution_symbols)}")
    if future_end:
        print(f"[headless] future_end={future_end}")
    print(f"[headless] steps={','.join(steps)}")
    print(f"[headless] reuse_output={int(reuse_output)} force_rebuild={int(force_rebuild)}")
    print(f"[headless] skip_stepe={int(skip_stepe)}")
    print(f"[headless] auto_prepare_data={int(auto_prepare_data)} data_start={data_start} data_end={data_end}")
    if args.mamba_lookback is not None:
        print(f"[headless] mamba_lookback={args.mamba_lookback}")
    if args.mamba_horizons:
        print(f"[headless] mamba_horizons={args.mamba_horizons}")
    if env_horizon_base is not None:
        print(f"[headless] env_horizon_days={env_horizon_base}")
    if env_horizon_list:
        print(f"[headless] env_horizons={','.join(str(x) for x in env_horizon_list)}")

    _base_reuse_key = (
        f"mode={resolved_mode}|primary_symbol={symbol}|test_start={args.test_start or ''}|"
        f"train_years={int(args.train_years)}|test_months={int(args.test_months)}|"
        f"enable_mamba={int(enable_mamba)}|enable_mamba_periodic={int(bool(args.enable_mamba_periodic))}|"
        f"mamba_lookback={args.mamba_lookback if args.mamba_lookback is not None else ''}|"
        f"mamba_horizons={args.mamba_horizons or ''}"
    )
    _stepe_agents_key = args.stepe_agents or ','.join(_extract_stepe_agents_from_config(app_config))
    print(f"[REUSE] stepA_reuse_key=mode={resolved_mode}|symbols={','.join(stepa_execution_symbols)}|test_start={args.test_start or ''}|train_years={int(args.train_years)}|test_months={int(args.test_months)}")
    print(f"[REUSE] stepB_reuse_key={_base_reuse_key}")
    print(f"[REUSE] stepC_reuse_key={_base_reuse_key}")
    print(f"[REUSE] stepDPrime_reuse_key={_base_reuse_key}|profile_set={_stepe_agents_key}")
    print(f"[REUSE] stepE_reuse_key={_base_reuse_key}|stepe_agents={_stepe_agents_key}")
    print(f"[REUSE] stepF_reuse_key={_base_reuse_key}|stepe_agents={_stepe_agents_key}|router_signature=minimal|reward_modes={','.join(_stepf_required_reward_modes())}")

    try:
        with timing.stage("branch.total"):
            if "A" in steps:
                if _can_reuse_step("A"):
                    print(f"[StepA] status=reuse signature={_run_sig.stable_hash()[:8] if '_run_sig' in dir() else 'n/a'}")
                    _mark_step("A", "reuse")
                    timing.mark_step_reused("A")
                    timing.emit_instant(stage="stepA.total", status="skipped", meta={"skipped": True})
                    if _run_manifest is not None:
                        _run_manifest.mark_step_elapsed("A", 0.0)
                    _emit_step_status("A", status="reuse", started_at=time.time(), ended_at=time.time(), validated=True)
                else:
                    stepa_symbols = _symbols_for_stepa_execution(symbol)
                    print(f"[StepA] start symbols={','.join(stepa_symbols)}")
                    _mark_step("A", "running")
                    _t0_a = time.perf_counter()
                    _t0_a_wall = time.time()
                    try:
                        for stepa_symbol in stepa_symbols:
                            with timing.stage("stepA.run"):
                                stepa_result = _run_stepA(app_config, stepa_symbol, date_range)
                            if stepa_symbol == symbol:
                                results["stepA_result"] = stepa_result
                    finally:
                        _elapsed_a = time.perf_counter() - _t0_a
                    _miss_a = validate_step_a(Path(resolved_output_root), symbol, resolved_mode)
                    if _miss_a:
                        if _run_manifest is not None:
                            _run_manifest.mark_step_elapsed("A", _elapsed_a)
                            _run_manifest.mark_step_audit("A", "FAIL")
                        _mark_step("A", "failed")
                        _emit_step_status("A", status="fail", started_at=_t0_a_wall, ended_at=time.time(), validated=False, detail="contract_missing")
                        raise RuntimeError("StepA contract missing required files: " + ", ".join(_miss_a))
                    _mark_step("A", "complete")
                    if _run_manifest is not None:
                        _run_manifest.mark_step_elapsed("A", _elapsed_a)
                        _run_manifest.mark_step_audit("A", "PASS")
                    _emit_step_status("A", status="run", started_at=_t0_a_wall, ended_at=time.time(), validated=True)

                    stepa_required = [
                        f"stepA_prices_train_{symbol}.csv",
                        f"stepA_prices_test_{symbol}.csv",
                        f"stepA_periodic_train_{symbol}.csv",
                        f"stepA_periodic_test_{symbol}.csv",
                        f"stepA_tech_train_{symbol}.csv",
                        f"stepA_tech_test_{symbol}.csv",
                        f"stepA_split_summary_{symbol}.csv",
                        f"stepA_periodic_future_{symbol}.csv",
                        f"stepA_daily_manifest_{symbol}.csv",
                    ]
                    stepa_dir_resolved = Path(resolved_output_root) / "stepA" / resolved_mode
                    stepa_dir_canonical = Path(canonical_output_root) / "stepA" / resolved_mode
                    stepa_daily_dir = stepa_dir_canonical / "daily"
                    print(f"[STEPA_VERIFY] canonical_stepa_dir={stepa_dir_canonical}")

                    if stepa_dir_resolved.resolve() != stepa_dir_canonical.resolve() and stepa_dir_resolved.exists():
                        stepa_dir_canonical.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copytree(stepa_dir_resolved, stepa_dir_canonical, dirs_exist_ok=True)
                        print(f"[STEPA_VERIFY] materialized src={stepa_dir_resolved} dst={stepa_dir_canonical}")

                        for _root_name in ("split_summary.json", "run_manifest.json", "reuse_signature.json"):
                            _root_src = Path(resolved_output_root) / _root_name
                            _root_dst = Path(canonical_output_root) / _root_name
                            if _root_src.exists():
                                _root_dst.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(_root_src, _root_dst)
                                print(f"[STEPA_VERIFY] materialized src={_root_src} dst={_root_dst}")

                    _manifest_path = stepa_dir_canonical / f"stepA_daily_manifest_{symbol}.csv"
                    if _manifest_path.exists():
                        try:
                            import pandas as _pd
                            _manifest_df = _pd.read_csv(_manifest_path)
                            for _col in ("prices_path", "periodic_path", "tech_path", "features_path", "window_features_path", "periodic_future_path"):
                                if _col in _manifest_df.columns:
                                    _manifest_df[_col] = _manifest_df[_col].fillna("").astype(str).map(
                                        lambda _raw: normalize_output_artifact_path(_raw, output_root=Path(canonical_output_root))
                                    )
                            _manifest_df.to_csv(_manifest_path, index=False)
                        except Exception as _manifest_e:
                            print(f"[STEPA_VERIFY] manifest_rewrite=fail reason={type(_manifest_e).__name__}:{_manifest_e}")

                    stepa_missing: List[str] = []
                    _status_map = {
                        f"stepA_prices_train_{symbol}.csv": "prices_train",
                        f"stepA_prices_test_{symbol}.csv": "prices_test",
                        f"stepA_periodic_train_{symbol}.csv": "periodic_train",
                        f"stepA_periodic_test_{symbol}.csv": "periodic_test",
                        f"stepA_tech_train_{symbol}.csv": "tech_train",
                        f"stepA_tech_test_{symbol}.csv": "tech_test",
                        f"stepA_split_summary_{symbol}.csv": "split_summary_csv",
                        f"stepA_periodic_future_{symbol}.csv": "periodic_future",
                        f"stepA_daily_manifest_{symbol}.csv": "daily_manifest",
                    }
                    for _name in stepa_required:
                        _p = stepa_dir_canonical / _name
                        _ok = _p.exists()
                        print(f"[STEPA_VERIFY] {_status_map.get(_name, _name)}={'pass' if _ok else 'fail'}")
                        if not _ok:
                            stepa_missing.append(_name)

                    _split_summary_json = Path(canonical_output_root) / "split_summary.json"
                    _split_ok = _split_summary_json.exists()
                    print(f"[STEPA_VERIFY] split_summary_json={'pass' if _split_ok else 'fail'}")
                    if not _split_ok:
                        stepa_missing.append("split_summary.json")

                    def _count_glob(_pat: str) -> int:
                        if not stepa_daily_dir.exists():
                            return 0
                        return len(list(stepa_daily_dir.glob(_pat)))

                    _daily_count_prices = _count_glob(f"stepA_prices_{symbol}_*.csv")
                    _daily_count_periodic = _count_glob(f"stepA_periodic_{symbol}_*.csv")
                    _daily_count_tech = _count_glob(f"stepA_tech_{symbol}_*.csv")
                    _daily_count_periodic_future = _count_glob(f"stepA_periodic_future_{symbol}_*_m*.csv")
                    print(f"[STEPA_VERIFY] daily_count_prices={_daily_count_prices}")
                    print(f"[STEPA_VERIFY] daily_count_periodic={_daily_count_periodic}")
                    print(f"[STEPA_VERIFY] daily_count_tech={_daily_count_tech}")
                    print(f"[STEPA_VERIFY] daily_count_periodic_future={_daily_count_periodic_future}")

                    if _manifest_path.exists():
                        try:
                            import pandas as _pd
                            _mani = _pd.read_csv(_manifest_path)
                            _manifest_missing: List[str] = []
                            for _col in ("prices_path", "periodic_path", "tech_path", "periodic_future_path"):
                                if _col not in _mani.columns:
                                    _manifest_missing.append(f"manifest_missing_col:{_col}")
                                    continue
                                for _raw in _mani[_col].fillna("").astype(str):
                                    if not _raw.strip() or not Path(_raw).exists():
                                        _manifest_missing.append(_raw or f"blank:{_col}")
                            if _manifest_missing:
                                stepa_missing.extend(_manifest_missing)
                        except Exception as _manifest_verify_e:
                            stepa_missing.append(f"daily_manifest_read_error:{type(_manifest_verify_e).__name__}")

                    if stepa_missing:
                        print(f"[STEPA_VERIFY] missing={','.join(stepa_missing)}")
                        print(f"[ONE_TAP][STEPA_VERIFY] missing={','.join(stepa_missing)}")
                    else:
                        print("[STEPA_VERIFY] missing=none")

                    print("[StepA] done")

            if "A" in steps:
                _sync_root_split_summary_from_stepa(
                    resolved_output_root=Path(resolved_output_root),
                    canonical_output_root=Path(canonical_output_root),
                    symbol=symbol,
                    mode=resolved_mode,
                    train_years=int(args.train_years),
                    test_months=int(args.test_months),
                    log_prefix="STEPA_SYNC",
                )

            _enable_dprime_stream = bool(int(getattr(args, "enable_dprime_stream", 0)))
            if _enable_dprime_stream and all(s in steps for s in ("B", "C", "DPRIME", "E")):
                from ai_core.services.dprime_pipeline_orchestrator import DPrimePipelineOrchestrator, DPrimePipelineOrchestratorConfig
                from ai_core.services.step_e_service import StepEService

                print("[DPRIME_STREAM] enabled=1 mode=safe")
                _d_cfg = _build_stepdprime_config(app_config, symbol, resolved_mode, output_root=str(resolved_output_root))
                _map = _build_stepe_profile_agent_map(app_config, _d_cfg.profiles)

                def _run_step_b_stream() -> None:
                    if _can_reuse_step("B"):
                        _mark_step("B", "reuse")
                        if _run_manifest is not None:
                            _run_manifest.mark_step_elapsed("B", 0.0)
                            _run_manifest.mark_step_audit("B", "PASS")
                        return
                    _mark_step("B", "running")
                    _t0_b_stream = time.perf_counter()
                    _t0_b_stream_wall = time.time()
                    try:
                        results["stepB_result"] = _run_step_generic("B", app_config, symbol, date_range, results)
                    finally:
                        _elapsed_b_stream = time.perf_counter() - _t0_b_stream
                    _miss_b_stream = validate_step_b(Path(resolved_output_root), symbol, resolved_mode, enabled_agents=tuple(results.get("agents", [])))
                    if _miss_b_stream:
                        if _run_manifest is not None:
                            _run_manifest.mark_step_elapsed("B", _elapsed_b_stream)
                            _run_manifest.mark_step_audit("B", "FAIL")
                        _mark_step("B", "failed")
                        _emit_step_status("B", status="fail", started_at=_t0_b_stream_wall, ended_at=time.time(), validated=False, detail="contract_or_coverage")
                        raise RuntimeError("StepB contract missing required files: " + ", ".join(_miss_b_stream))
                    _mark_step("B", "complete")
                    if _run_manifest is not None:
                        _run_manifest.mark_step_elapsed("B", _elapsed_b_stream)
                        _run_manifest.mark_step_audit("B", "PASS")
                    _emit_step_status("B", status="run", started_at=_t0_b_stream_wall, ended_at=time.time(), validated=True)

                def _run_step_c_stream() -> None:
                    if _can_reuse_step("C"):
                        _mark_step("C", "reuse")
                        if _run_manifest is not None:
                            _run_manifest.mark_step_elapsed("C", 0.0)
                            _run_manifest.mark_step_audit("C", "PASS")
                        return
                    _mark_step("C", "running")
                    _t0_c_stream = time.perf_counter()
                    _t0_c_stream_wall = time.time()
                    try:
                        results["stepC_result"] = _run_step_generic("C", app_config, symbol, date_range, results)
                    finally:
                        _elapsed_c_stream = time.perf_counter() - _t0_c_stream
                    _miss_c_stream = validate_step_c(Path(resolved_output_root), symbol, resolved_mode)
                    if _miss_c_stream:
                        if _run_manifest is not None:
                            _run_manifest.mark_step_elapsed("C", _elapsed_c_stream)
                            _run_manifest.mark_step_audit("C", "FAIL")
                        _mark_step("C", "failed")
                        _emit_step_status("C", status="fail", started_at=_t0_c_stream_wall, ended_at=time.time(), validated=False, detail="contract_missing")
                        raise RuntimeError("StepC contract missing required files: " + ", ".join(_miss_c_stream))
                    _mark_step("C", "complete")
                    if _run_manifest is not None:
                        _run_manifest.mark_step_elapsed("C", _elapsed_c_stream)
                        _run_manifest.mark_step_audit("C", "PASS")
                    _emit_step_status("C", status="run", started_at=_t0_c_stream_wall, ended_at=time.time(), validated=True)

                orch = DPrimePipelineOrchestrator(step_e_service=StepEService(app_config), date_range=date_range, stepe_cfg_by_profile=_map)
                _t0_dp_stream_wall = time.time()
                try:
                    _mark_step("DPRIME", "running")
                    _mark_step("E", "running")
                    _orch_result = orch.run(
                        DPrimePipelineOrchestratorConfig(
                            symbol=symbol,
                            mode=resolved_mode,
                            output_root=str(resolved_output_root),
                            stepd_cfg=_d_cfg,
                            force_cpu_dprime_final=bool(int(getattr(args, "force_cpu", 0))),
                        ),
                        run_step_b=_run_step_b_stream,
                        run_step_c=_run_step_c_stream,
                    )
                except Exception:
                    _mark_step("DPRIME", "failed")
                    _mark_step("E", "failed")
                    if _run_manifest is not None:
                        _run_manifest.mark_step_audit("DPRIME", "FAIL")
                        _run_manifest.mark_step_audit("E", "FAIL")
                    _emit_step_status("DPRIME", status="fail", started_at=_t0_dp_stream_wall, ended_at=time.time(), validated=False, detail="stream_exception")
                    _emit_step_status("E", status="fail", started_at=_t0_dp_stream_wall, ended_at=time.time(), validated=False, detail="stream_exception")
                    raise
                results["stepDPRIME_stream_result"] = _orch_result
                _stream_dprime_ok = check_step_artifacts(
                    "DPRIME",
                    resolved_output_root,
                    symbol,
                    resolved_mode,
                    required_dprime_profiles=tuple(_stream_agents),
                )
                _stream_agents = _extract_stepe_agents_from_config(app_config) or list(_OFFICIAL_STEPE_AGENTS)
                if _run_manifest is not None:
                    _run_manifest.mark_step_verified(
                        "DPRIME",
                        "complete",
                        artifacts_ok=_stream_dprime_ok,
                        audit_status="PASS" if _stream_dprime_ok else "FAIL",
                        invalid_status="pending",
                    )
                    _stream_step_e_dir = Path(resolved_output_root) / "stepE" / resolved_mode
                    _stream_model_dir = _stream_step_e_dir / "models"
                    _stream_audit_root = Path(resolved_output_root) / "audit" / resolved_mode
                    for _stream_agent in _stream_agents:
                        _core_ok = (
                            (_stream_step_e_dir / f"stepE_daily_log_{_stream_agent}_{symbol}.csv").exists()
                            and (_stream_step_e_dir / f"stepE_summary_{_stream_agent}_{symbol}.json").exists()
                            and (
                                (_stream_model_dir / f"stepE_{_stream_agent}_{symbol}.pt").exists()
                                or (_stream_model_dir / f"stepE_{_stream_agent}_{symbol}_ppo.zip").exists()
                            )
                        )
                        _stream_agent_audit = "FAIL"
                        if _core_ok:
                            try:
                                _stream_audit = audit_stepe_agent_now(
                                    Path(resolved_output_root), resolved_mode, symbol, _stream_agent, _stream_audit_root
                                )
                                _stream_agent_audit = _stream_audit.get("status", "FAIL")
                            except Exception as _stream_ae:
                                print(f"[StepE] WARN stream audit agent={_stream_agent}: {_stream_ae}", file=sys.stderr)
                        _run_manifest.mark_stepe_agent_verified(
                            _stream_agent,
                            "complete",
                            artifacts_ok=check_stepe_agent_artifact(_stream_agent, resolved_output_root, symbol, resolved_mode),
                            audit_status=_stream_agent_audit,
                            invalid_status="pending",
                        )
                _stream_stepe_final = _evaluate_stepe_final_state(_stream_agents)
                _stream_e_ok = _stream_stepe_final["all_complete"]
                _mark_step_verified("DPRIME", "complete", audit_status="PASS" if _stream_dprime_ok else "FAIL", invalid_status="pending")
                _mark_step_verified("E", "complete", audit_status="PASS" if _stream_e_ok else "FAIL", invalid_status="pending")
                _emit_step_status("DPRIME", status="run" if _stream_dprime_ok else "fail", started_at=_t0_dp_stream_wall, ended_at=time.time(), validated=_stream_dprime_ok)
                _emit_step_status("E", status="run" if _stream_e_ok else "fail", started_at=_t0_dp_stream_wall, ended_at=time.time(), validated=_stream_e_ok, detail="" if _stream_e_ok else "partial_agent_outputs")
                if not _stream_dprime_ok or not _stream_e_ok:
                    raise RuntimeError(
                        "Streaming DPrime/StepE manifest finalization failed: "
                        f"dprime_ok={_stream_dprime_ok} stepe_ok={_stream_e_ok}"
                    )
                steps = tuple(s for s in steps if s not in ("B", "C", "DPRIME", "E"))

            if "B" in steps:
                app_config = _apply_config_output_root(app_config, Path(canonical_output_root))
                try:
                    setattr(app_config, "effective_output_root", str(resolved_output_root))
                    setattr(app_config, "_effective_output_root", str(resolved_output_root))
                    setattr(app_config, "resolved_output_root", str(resolved_output_root))
                    setattr(app_config, "canonical_output_root", str(canonical_output_root))
                except Exception:
                    pass
                print(f"[PIPELINE] args_output_root={args.output_root}")
                print(f"[PIPELINE] cfg_output_root={getattr(app_config, 'output_root', None)}")
                print(f"[PIPELINE] cfg_data_output_root={getattr(getattr(app_config, 'data', None), 'output_root', None)}")
                print(f"[PIPELINE] effective_output_root={getattr(app_config, 'effective_output_root', None)}")
                print(f"[PIPELINE] canonical_output_root={getattr(app_config, 'canonical_output_root', None)}")
                _check_and_repair_split_summary_before_stepb(
                    resolved_output_root=Path(resolved_output_root),
                    canonical_output_root=Path(canonical_output_root),
                    symbol=symbol,
                    mode=resolved_mode,
                    train_years=int(args.train_years),
                    test_months=int(args.test_months),
                )
                _stepb_expected_pred_file = Path(resolved_output_root) / "stepB" / resolved_mamba_mode / f"stepB_pred_time_all_{symbol}.csv"
                print(f"[STEPB] output_root={resolved_output_root}")
                print(f"[STEPB] expected_pred_file={_stepb_expected_pred_file}")
                print(f"[STEPB_PRE] required=stepB/{resolved_mamba_mode}/stepB_pred_time_all_{symbol}.csv exists={'pass' if _stepb_expected_pred_file.exists() else 'fail'}")
                if _can_reuse_step("B"):
                    print(f"[StepB] status=reuse signature={_run_sig.stable_hash()[:8] if '_run_sig' in dir() else 'n/a'}")
                    _mark_step("B", "reuse")
                    timing.mark_step_reused("B")
                    timing.emit_instant(stage="stepB.total", status="skipped", meta={"skipped": True})
                    if _run_manifest is not None:
                        _run_manifest.mark_step_elapsed("B", 0.0)
                    _emit_step_status("B", status="reuse", started_at=time.time(), ended_at=time.time(), validated=True)
                else:
                    print("[StepB] start")
                    mamba_horizons_list = _parse_int_list(args.mamba_horizons)
                    _mark_step("B", "running")
                    _t0_b = time.perf_counter()
                    _t0_b_wall = time.time()
                    _stepb_status_emitted = False
                    try:
                        with timing.stage("stepB.run"):
                            results["stepB_result"] = _run_stepB(app_config, symbol, date_range, enable_mamba, args.enable_mamba_periodic, args.mamba_lookback, mamba_horizons_list)
                        _elapsed_b = time.perf_counter() - _t0_b
                        print(f"[StepB] agents: mamba={enable_mamba}")
                        # Ensure contract artifact exists
                        try:
                            p = _ensure_stepb_pred_time_all(symbol, resolved_output_root, mode=resolved_mamba_mode)
                            print(f"[StepB] ensured: {p}")
                        except Exception as e:
                            print(f"[StepB] WARN: failed to ensure stepB_pred_time_all: {e}", file=sys.stderr)
                        _post_stepb_missing: List[str] = []
                        if not _stepb_expected_pred_file.exists():
                            _post_stepb_missing.append(str(_stepb_expected_pred_file))
                            print(f"[STEPB_POST] required=stepB/{resolved_mamba_mode}/stepB_pred_time_all_{symbol}.csv exists=fail", file=sys.stderr)
                            print("[STEPB_POST] STEPB_FAIL_REASON=missing_pred_time_all", file=sys.stderr)
                        else:
                            print(f"[STEPB_POST] required=stepB/{resolved_mamba_mode}/stepB_pred_time_all_{symbol}.csv exists=pass")
                        _miss_b = validate_step_b(Path(resolved_output_root), symbol, resolved_mamba_mode)
                        if _post_stepb_missing:
                            _miss_b.extend(_post_stepb_missing)
                        if _miss_b:
                            _finalize_manifest_step("B", status="failed", elapsed_sec=_elapsed_b, audit_status="FAIL")
                            _emit_step_status("B", status="fail", started_at=_t0_b_wall, ended_at=time.time(), validated=False, detail="contract_or_coverage")
                            _stepb_status_emitted = True
                            raise RuntimeError("StepB contract missing/invalid: " + ", ".join(_miss_b))
                        _finalize_manifest_step("B", status="complete", elapsed_sec=_elapsed_b, audit_status="PASS")
                        _emit_step_status("B", status="run", started_at=_t0_b_wall, ended_at=time.time(), validated=True)
                        _stepb_status_emitted = True
                        print("[StepB] done")
                    except Exception:
                        _elapsed_b = time.perf_counter() - _t0_b
                        _tb_text = traceback.format_exc()
                        _append_run_log(run_log_path, "[PIPELINE][StepB] exception begin")
                        _append_run_log(run_log_path, _tb_text)
                        _append_run_log(run_log_path, "[PIPELINE][StepB] exception end")
                        _finalize_manifest_step("B", status="failed", elapsed_sec=_elapsed_b, audit_status="FAIL")
                        traceback.print_exc(file=sys.stderr)
                        # Make downstream skip-reason explicit in run log / ONE_TAP parsers.
                        for _downstream in ("C", "DPRIME", "E", "F"):
                            if _downstream in steps:
                                _emit_step_status(
                                    _downstream,
                                    status="skip",
                                    started_at=time.time(),
                                    ended_at=time.time(),
                                    validated=None,
                                    detail="skip_due_to_stepB_failure",
                                )
                        if not _stepb_status_emitted:
                            _emit_step_status("B", status="fail", started_at=_t0_b_wall, ended_at=time.time(), validated=False, detail="exception")
                        print("[PIPELINE] downstream_blocked_by=StepB reason=skip_due_to_stepB_failure", file=sys.stderr)
                        raise
                    finally:
                        if _run_manifest is not None:
                            try:
                                _run_manifest.save()
                            except Exception as _manifest_flush_e:
                                print(
                                    f"[reuse] WARNING manifest flush failed after StepB: "
                                    f"{type(_manifest_flush_e).__name__}: {_manifest_flush_e}",
                                    file=sys.stderr,
                                )

            if "C" in steps:
                if _can_reuse_step("C"):
                    print(f"[StepC] status=reuse signature={_run_sig.stable_hash()[:8] if '_run_sig' in dir() else 'n/a'}")
                    _mark_step("C", "reuse")
                    timing.mark_step_reused("C")
                    timing.emit_instant(stage="stepC.total", status="skipped", meta={"skipped": True})
                    if _run_manifest is not None:
                        _run_manifest.mark_step_elapsed("C", 0.0)
                    _emit_step_status("C", status="reuse", started_at=time.time(), ended_at=time.time(), validated=True)
                else:
                    print("[StepC] start")
                    _mark_step("C", "running")
                    _t0_c = time.perf_counter()
                    _t0_c_wall = time.time()
                    try:
                        with timing.stage("stepC.run"):
                            results["stepC_result"] = _run_step_generic("C", app_config, symbol, date_range, results)
                    finally:
                        _elapsed_c = time.perf_counter() - _t0_c
                    _miss_c = validate_step_c(Path(resolved_output_root), symbol, resolved_mode)
                    if _miss_c:
                        if _run_manifest is not None:
                            _run_manifest.mark_step_elapsed("C", _elapsed_c)
                            _run_manifest.mark_step_audit("C", "FAIL")
                        _mark_step("C", "failed")
                        _emit_step_status("C", status="fail", started_at=_t0_c_wall, ended_at=time.time(), validated=False, detail="contract_missing")
                        raise RuntimeError("StepC contract missing required files: " + ", ".join(_miss_c))
                    _mark_step("C", "complete")
                    if _run_manifest is not None:
                        _run_manifest.mark_step_elapsed("C", _elapsed_c)
                        _run_manifest.mark_step_audit("C", "PASS")
                    _emit_step_status("C", status="run", started_at=_t0_c_wall, ended_at=time.time(), validated=True)
                    print("[StepC] done")

            if "DPRIME" in steps:
                if _can_reuse_step("DPRIME"):
                    print(f"[StepDPrime] status=reuse signature={_run_sig.stable_hash()[:8] if '_run_sig' in dir() else 'n/a'}")
                    _mark_step("DPRIME", "reuse")
                    timing.mark_step_reused("DPRIME")
                    timing.emit_instant(stage="stepDPrime.total", status="skipped", meta={"skipped": True})
                    if _run_manifest is not None:
                        _run_manifest.mark_step_elapsed("DPRIME", 0.0)
                    _emit_step_status("DPRIME", status="reuse", started_at=time.time(), ended_at=time.time(), validated=True)
                else:
                    print("[StepDPrime] start")
                    _mark_step("DPRIME", "running")
                    _t0_dp = time.perf_counter()
                    _t0_dp_wall = time.time()
                    try:
                        try:
                            with timing.stage("stepDPrime.run"):
                                results["stepDPRIME_result"] = _run_stepDPrime(app_config, symbol, date_range, mode=resolved_mamba_mode)
                        except Exception:
                            if _run_manifest is not None:
                                _run_manifest.mark_step_audit("DPRIME", "FAIL")
                            _mark_step("DPRIME", "failed")
                            _emit_step_status(
                                "DPRIME",
                                status="fail",
                                started_at=_t0_dp_wall,
                                ended_at=time.time(),
                                validated=False,
                                detail="exception",
                            )
                            raise
                    finally:
                        _elapsed_dp = time.perf_counter() - _t0_dp
                    _dp_profiles = _OFFICIAL_STEPE_AGENTS
                    _dp_result = results.get("stepDPRIME_result")
                    if isinstance(_dp_result, dict):
                        _profiles_map = _dp_result.get("profiles")
                        if isinstance(_profiles_map, dict) and _profiles_map:
                            _dp_profiles = tuple(str(k) for k in _profiles_map.keys())
                    _miss_dp = validate_step_dprime(Path(resolved_output_root), resolved_mamba_mode, symbol, _dp_profiles)
                    if _miss_dp:
                        if _run_manifest is not None:
                            _run_manifest.mark_step_elapsed("DPRIME", _elapsed_dp)
                            _run_manifest.mark_step_audit("DPRIME", "FAIL")
                        _mark_step("DPRIME", "failed")
                        _emit_step_status("DPRIME", status="fail", started_at=_t0_dp_wall, ended_at=time.time(), validated=False, detail="contract_missing")
                        raise RuntimeError("StepDPrime contract missing required state files: " + ", ".join(_miss_dp))
                    _mark_step("DPRIME", "complete")
                    if _run_manifest is not None:
                        _run_manifest.mark_step_elapsed("DPRIME", _elapsed_dp)
                        _run_manifest.mark_step_audit("DPRIME", "PASS")
                    _emit_step_status("DPRIME", status="run", started_at=_t0_dp_wall, ended_at=time.time(), validated=True)
                    print("[StepDPrime] done")

            for step in ("D", "E", "F"):
                if step not in steps:
                    continue
                stage_name = "stepE.run" if step == "E" else ("stepF.run" if step == "F" else f"step{step}.run")
                if step == "E":
                    print("[StepE expert evaluation] candidate generation layer")
                if step == "F":
                    print("[StepF router / MARL final selection] candidate integration layer")

                # --- StepE: per-agent reuse/resume ---
                if step == "E" and reuse_output and _run_manifest is not None and not force_rebuild:
                    try:
                        from tools.run_manifest import check_stepe_agent_artifact as _check_agent_art  # noqa: F811

                        _raw_e_cfgs = app_config.get("stepE") if isinstance(app_config, dict) else getattr(app_config, "stepE", None)
                        _cfg_list_e = list(_raw_e_cfgs) if isinstance(_raw_e_cfgs, (list, tuple)) else ([_raw_e_cfgs] if _raw_e_cfgs is not None else [])
                        _cfg_by_agent = {
                            getattr(c, "agent", ""): c
                            for c in _cfg_list_e
                            if getattr(c, "agent", "")
                        }

                        if args.stepe_agents:
                            _requested_agents = [a.strip() for a in str(args.stepe_agents).split(",") if a.strip()]
                        else:
                            _requested_agents = list(_OFFICIAL_STEPE_AGENTS)
                        _all_agents = list(dict.fromkeys(_requested_agents))

                        _run_manifest.ensure_stepe_agents(_all_agents)

                        _artifact_complete_agents: List[str] = []
                        _manifest_complete_agents: List[str] = []
                        _normalized_reuse_agents: List[str] = []
                        _reusable_agents: List[str] = []
                        for _agent in _all_agents:
                            _manifest_status = _run_manifest.stepe_agent_status(_agent)
                            _manifest_ok = _run_manifest.can_reuse_stepe_agent(_agent)
                            _artifact_ok = _check_agent_art(_agent, resolved_output_root, symbol, resolved_mode)
                            print(
                                f"[STEPE_AGENT] agent={_agent} artifact_complete={'true' if _artifact_ok else 'false'} manifest_status={_manifest_status}"
                            )
                            if _artifact_ok:
                                _artifact_complete_agents.append(_agent)
                            if _manifest_ok:
                                _manifest_complete_agents.append(_agent)
                            if _manifest_ok and _artifact_ok:
                                _reusable_agents.append(_agent)

                        _pending_agents = [a for a in _all_agents if a not in _reusable_agents]

                        print(f"[STEPE_RESUME] all_agents={len(_all_agents)}")
                        print(f"[STEPE_RESUME] artifact_complete_agents={len(_artifact_complete_agents)}")
                        print(f"[STEPE_RESUME] manifest_complete_agents={len(_manifest_complete_agents)}")
                        print(f"[STEPE_RESUME] normalized_reuse_agents={len(_normalized_reuse_agents)}")
                        print(f"[STEPE_RESUME] pending_agents={len(_pending_agents)}")
                        print(f"[STEPE_RESUME] reuse_agent_list={','.join(_reusable_agents)}")
                        print(f"[STEPE_RESUME] run_agent_list={','.join(_pending_agents)}")

                        for _agent in _reusable_agents:
                            _mark_stepe_agent_verified(_agent, "reuse", audit_status="PASS", invalid_status="pending")
                            print(f"[STEPE_AGENT] agent={_agent} action=reuse")

                        if not _pending_agents:
                            _stepe_final = _evaluate_stepe_final_state(_all_agents)
                            if not _stepe_final["all_complete"]:
                                _mark_step("E", "pending")
                                _run_manifest.mark_step_elapsed("E", 0.0)
                                _run_manifest.mark_step_audit("E", "FAIL")
                                _emit_step_status("E", status="fail", started_at=time.time(), ended_at=time.time(), validated=False, detail="partial_agent_outputs")
                                raise RuntimeError("StepE partial completion; rerun will resume pending agents")
                            print(f"[StepE] status=reuse all {len(_all_agents)} agents complete signature={_run_sig.stable_hash()[:8] if '_run_sig' in dir() else 'n/a'}")
                            _mark_step_verified("E", "reuse", audit_status="PASS", invalid_status="pending")
                            timing.mark_step_reused("E")
                            timing.emit_instant(stage="stepE.total", status="skipped", meta={"skipped": True})
                            _run_manifest.mark_step_elapsed("E", 0.0)
                            _emit_step_status("E", status="reuse", started_at=time.time(), ended_at=time.time(), validated=True)
                            results["stepE_result"] = {"reused": True, "agents": _reusable_agents}
                            continue

                        if any(a not in _cfg_by_agent for a in _pending_agents):
                            raise RuntimeError(f"StepE pending agent config not found: {','.join(_pending_agents)}")
                        _mark_step("E", "running")
                        _t0_e = time.perf_counter()
                        _t0_e_wall = time.time()
                        step_result, _elapsed_e, _agent_failures = _run_stepe_agents_incrementally(
                            requested_agents=_pending_agents,
                            cfg_by_agent=_cfg_by_agent,
                            initial_reused_agents=_reusable_agents,
                        )
                        results[f"step{step}_result"] = step_result

                        _still_pending = [a for a in _all_agents if not (_run_manifest.can_reuse_stepe_agent(a) and _check_agent_art(a, resolved_output_root, symbol, resolved_mode))]
                        _run_manifest.mark_step_elapsed("E", _elapsed_e)

                        _miss_e = []
                        for _ea in _all_agents:
                            _miss_e.extend(validate_step_e_agent(Path(resolved_output_root), symbol, resolved_mode, _ea))
                        _stepe_final = _evaluate_stepe_final_state(_all_agents)
                        _step_e_audit_status = "PASS" if _stepe_final["all_complete"] else "FAIL"
                        if _miss_e:
                            print(f"[StepE] contract missing: {_miss_e}", file=sys.stderr)
                        if _agent_failures:
                            print(f"[StepE] WARN non-blocking agent failures={','.join(_agent_failures)}", file=sys.stderr)
                        if _still_pending:
                            print(f"[StepE] WARN non-blocking still_pending={','.join(_still_pending)}", file=sys.stderr)
                        _mark_step_verified("E", "complete", audit_status=_step_e_audit_status, invalid_status="pending")

                        if _stepe_final["all_complete"]:
                            _emit_step_status("E", status="run", started_at=_t0_e_wall, ended_at=time.time(), validated=True)
                            print(f"[StepE] done")
                        else:
                            _mark_step("E", "pending")
                            _emit_step_status("E", status="fail", started_at=_t0_e_wall, ended_at=time.time(), validated=False, detail="partial_or_contract_or_audit")
                            raise RuntimeError("StepE partial completion; rerun will resume pending agents")
                        continue
                    except RuntimeError:
                        raise
                    except Exception as _reuse_e:
                        print(f"[StepE] reuse logic error ({type(_reuse_e).__name__}: {_reuse_e}); running step normally", file=sys.stderr)

                # --- StepF: reuse ---
                if step == "F":
                    _stepf_agents = _extract_stepe_agents_from_config(app_config) or list(_OFFICIAL_STEPE_AGENTS)
                    _stepe_final_for_f = _evaluate_stepe_final_state(_stepf_agents)
                    if not _stepe_final_for_f["all_complete"]:
                        raise RuntimeError(f"StepF requires complete StepE artifacts for all agents. missing={_stepe_final_for_f['missing_detail']}")
                if step == "F" and _can_reuse_step("F"):
                    print(f"[StepF] status=reuse signature={_run_sig.stable_hash()[:8] if '_run_sig' in dir() else 'n/a'}")
                    _mark_step("F", "reuse")
                    _update_stepf_manifest_fields("reuse")
                    _evaluate_stepf_final_state()
                    timing.mark_step_reused("F")
                    timing.emit_instant(stage="stepF.total", status="skipped", meta={"skipped": True})
                    if _run_manifest is not None:
                        _run_manifest.mark_step_elapsed("F", 0.0)
                    _emit_step_status("F", status="reuse", started_at=time.time(), ended_at=time.time(), validated=True)
                    results["stepF_result"] = {"reused": True}
                    continue

                # --- Generic run (D, or E/F without reuse) ---
                print(f"[Step{step}] start")
                _mark_step(step if step != "D" else "D", "running")
                _t0_generic = time.perf_counter()
                _t0_generic_wall = time.time()
                try:
                    if step == "E":
                        _ge_raw_cfgs = app_config.get("stepE") if isinstance(app_config, dict) else getattr(app_config, "stepE", None)
                        _ge_cfg_list = list(_ge_raw_cfgs) if isinstance(_ge_raw_cfgs, (list, tuple)) else ([_ge_raw_cfgs] if _ge_raw_cfgs is not None else [])
                        _ge_cfg_by_agent = {getattr(c, "agent", ""): c for c in _ge_cfg_list if getattr(c, "agent", "")}
                        _ge_agents = [a for a in _extract_stepe_agents_from_config(app_config) if a in _ge_cfg_by_agent]
                        step_result, _elapsed_generic, _generic_agent_failures = _run_stepe_agents_incrementally(
                            requested_agents=_ge_agents or list(_ge_cfg_by_agent.keys()),
                            cfg_by_agent=_ge_cfg_by_agent,
                        )
                    else:
                        try:
                            with timing.stage(stage_name):
                                step_result = _run_step_generic(step, app_config, symbol, date_range, results)
                        finally:
                            _elapsed_generic = time.perf_counter() - _t0_generic
                    results[f"step{step}_result"] = step_result
                    if step == "E" and isinstance(step_result, dict) and step_result.get("skipped"):
                        reason = step_result.get("reason", "unknown")
                        raise RuntimeError(f"StepE reported skipped=True without explicit skip flag. reason={reason}")
                    if _run_manifest is not None:
                        _run_manifest.mark_step_elapsed(step if step != "D" else "D", _elapsed_generic)
                        if step == "E":
                            # StepE without per-agent reuse: audit all completed agents
                            _audit_root_ge = Path(resolved_output_root) / "audit" / resolved_mode
                            _ge_all_agents: List[str]
                            _ge_raw = app_config.get("stepE") if isinstance(app_config, dict) else getattr(app_config, "stepE", None)
                            if _ge_raw is not None:
                                _ge_cfg_list = list(_ge_raw) if isinstance(_ge_raw, (list, tuple)) else [_ge_raw]
                                _ge_all_agents = [getattr(c, "agent", "") for c in _ge_cfg_list if getattr(c, "agent", "")]
                            else:
                                _ge_all_agents = list(_OFFICIAL_STEPE_AGENTS)
                            for _ga in _ge_all_agents:
                                from tools.run_manifest import check_stepe_agent_artifact as _chk_ge
                                if _chk_ge(_ga, resolved_output_root, symbol, resolved_mode):
                                    try:
                                        _ga_audit = audit_stepe_agent_now(
                                            Path(resolved_output_root), resolved_mode, symbol, _ga, _audit_root_ge
                                        )
                                        _ga_status = _ga_audit.get("status", "FAIL")
                                    except Exception as _gae:
                                        _ga_status = "FAIL"
                                        print(f"[StepE] WARN audit agent={_ga}: {_gae}", file=sys.stderr)
                                    _mark_stepe_agent_verified(_ga, "complete", audit_status=_ga_status, invalid_status="pending")
                                    print(f"[StepE] agent={_ga} audit={_ga_status}")
                            _ge_audits = [_run_manifest.stepe_agent_audit_status(a) for a in _ge_all_agents]
                            _ge_step_status = "PASS" if _ge_audits and all(s == "PASS" for s in _ge_audits) else "FAIL"
                            _miss_e = []
                            for _ga in _ge_all_agents:
                                _miss_e.extend(validate_step_e_agent(Path(resolved_output_root), symbol, resolved_mode, _ga))
                            if _miss_e:
                                _ge_step_status = "FAIL"
                                print(f"[StepE] contract missing: {_miss_e}", file=sys.stderr)
                            _mark_step_verified("E", "complete", audit_status=_ge_step_status, invalid_status="pending")
                            if '_generic_agent_failures' in locals() and _generic_agent_failures:
                                print(f"[StepE] WARN non-blocking agent failures={','.join(_generic_agent_failures)}", file=sys.stderr)
                            if _ge_step_status != "PASS":
                                _emit_step_status("E", status="fail", started_at=_t0_generic_wall, ended_at=time.time(), validated=False, detail="contract_or_audit")
                                raise RuntimeError("StepE contract/audit failed")
                        elif step == "F":
                            from ai_core.services.step_f_service import StepFService

                            _audit_root_f = Path(resolved_output_root) / "audit" / resolved_mode
                            _sf_status = "PASS"
                            try:
                                _sf_audits = audit_stepf_now(Path(resolved_output_root), resolved_mode, symbol, _audit_root_f)
                                if _sf_audits and not all(v.get("status") == "PASS" for v in _sf_audits.values()):
                                    _sf_status = "FAIL"
                                    print("[StepF] WARN audit mismatch detected; manifest will record FAIL while preserving completed artifacts", file=sys.stderr)
                            except Exception as _sfe:
                                _sf_status = "WARN"
                                print(f"[StepF] WARN audit: {_sfe}", file=sys.stderr)

                            _final_eval_f = StepFService.evaluate_final_outputs(
                                output_root=Path(resolved_output_root),
                                mode=resolved_mode,
                                symbol=symbol,
                            )
                            _miss_f = validate_step_f(Path(resolved_output_root), symbol, resolved_mode)
                            if _miss_f:
                                print(f"[StepF] contract missing: {_miss_f}", file=sys.stderr)

                            if int(_final_eval_f.get("return_code", 1)) != 0:
                                _sf_status = "FAIL"

                            _run_manifest.mark_step_audit("F", _sf_status)
                            _update_stepf_manifest_fields("complete")
                            _stepf_final = _evaluate_stepf_final_state()
                            print(f"[StepF] audit={_sf_status}")
                            if _sf_status == "FAIL" or _stepf_final.get("final_status") != "complete":
                                _emit_step_status("F", status="fail", started_at=_t0_generic_wall, ended_at=time.time(), validated=False, detail="final_artifacts_invalid")
                                raise RuntimeError("StepF final artifacts invalid")
                        else:
                            _run_manifest.mark_step_audit(step if step != "D" else "D", "SKIP")
                    else:
                        if step == "E":
                            _ge_raw = app_config.get("stepE") if isinstance(app_config, dict) else getattr(app_config, "stepE", None)
                            _ge_cfg_list = list(_ge_raw) if isinstance(_ge_raw, (list, tuple)) else ([_ge_raw] if _ge_raw is not None else [])
                            _ge_all_agents = [getattr(c, "agent", "") for c in _ge_cfg_list if getattr(c, "agent", "")]
                            if not _ge_all_agents:
                                _ge_all_agents = list(_OFFICIAL_STEPE_AGENTS)
                            _miss_e = []
                            for _ga in _ge_all_agents:
                                _miss_e.extend(validate_step_e_agent(Path(resolved_output_root), symbol, resolved_mode, _ga))
                            if _miss_e:
                                _emit_step_status("E", status="fail", started_at=_t0_generic_wall, ended_at=time.time(), validated=False, detail="contract_missing")
                                raise RuntimeError("StepE contract missing required files: " + ", ".join(_miss_e))
                        if step == "F":
                            from ai_core.services.step_f_service import StepFService

                            _final_eval_f = StepFService.evaluate_final_outputs(
                                output_root=Path(resolved_output_root),
                                mode=resolved_mode,
                                symbol=symbol,
                            )
                            if int(_final_eval_f.get("return_code", 1)) != 0:
                                _emit_step_status("F", status="fail", started_at=_t0_generic_wall, ended_at=time.time(), validated=False, detail="final_artifacts_invalid")
                                raise RuntimeError("StepF final artifacts invalid: " + ", ".join(_final_eval_f.get("errors", [])))
                    if step == "DPRIME":
                        _mark_step_verified("DPRIME", "complete", invalid_status="pending")
                    elif step == "E":
                        _mark_step_verified("E", "complete", invalid_status="pending")
                    elif step == "F":
                        _update_stepf_manifest_fields("complete")
                        _stepf_final = _evaluate_stepf_final_state()
                        if _stepf_final.get("final_status") != "complete":
                            _emit_step_status("F", status="fail", started_at=_t0_generic_wall, ended_at=time.time(), validated=False, detail="manifest_reconciliation_incomplete")
                            raise RuntimeError("StepF finalize/manifest reconciliation incomplete")
                    else:
                        _mark_step(step if step != "D" else "D", "complete")
                    _emit_step_status(step if step != "D" else "D", status="run", started_at=_t0_generic_wall, ended_at=time.time(), validated=True)
                    print(f"[Step{step}] done")
                except (KeyboardInterrupt, SystemExit) as _interrupt:
                    _elapsed_generic = time.perf_counter() - _t0_generic
                    _finalize_manifest_step(step if step != "D" else "D", status="interrupted", elapsed_sec=_elapsed_generic, audit_status="FAIL")
                    if step == "F":
                        _update_stepf_manifest_fields("interrupted", error=_interrupt)
                    raise
                except Exception as _step_exc:
                    _elapsed_generic = time.perf_counter() - _t0_generic
                    _finalize_manifest_step(step if step != "D" else "D", status="failed", elapsed_sec=_elapsed_generic, audit_status="FAIL")
                    if step == "F":
                        _update_stepf_manifest_fields("failed", error=_step_exc)
                    raise

            with timing.stage("audit.leak"):
                _run_leak_audits(
                    output_root=Path(resolved_output_root),
                    mode=resolved_mode,
                    symbol=symbol,
                    fail_on_audit=bool(args.fail_on_audit),
                )

        if _run_manifest is not None and steps:
            _final_step = steps[-1]
            _final_status = _run_manifest.step_status(_final_step)
            if _final_status not in ("complete", "reuse"):
                raise RuntimeError(f"final step incomplete: step={_final_step} status={_final_status}")
        timing.write_summaries()
        timings_csv = Path(resolved_output_root) / "timings.csv"
        timing_events = Path(resolved_output_root) / "timing" / resolved_mode / "timing_events.jsonl"
        timing_summary_step = Path(resolved_output_root) / "timing" / "summary_step_elapsed.csv"
        timing_summary_agent = Path(resolved_output_root) / "timing" / "summary_agent_elapsed.csv"
        mandatory_root_files = [
            Path(resolved_output_root) / "run_manifest.json",
            Path(resolved_output_root) / "timings.csv",
            Path(resolved_output_root) / "reuse_signature.json",
        ]
        missing_root_files = [str(p) for p in mandatory_root_files if not p.exists()]
        if missing_root_files:
            raise RuntimeError(f"missing mandatory root artifacts: {missing_root_files}")
        print(f"[ONE_TAP][DIAG] timings_csv_exists={'yes' if timings_csv.exists() else 'no'}")
        print(f"[ONE_TAP][DIAG] timing_events_exists={'yes' if timing_events.exists() else 'no'}")
        print(f"[ONE_TAP][DIAG] timing_summary_step_exists={'yes' if timing_summary_step.exists() else 'no'}")
        print(f"[ONE_TAP][DIAG] timing_summary_agent_exists={'yes' if timing_summary_agent.exists() else 'no'}")
        print("[headless] ALL DONE")
        print(f"[PIPELINE] status=success steps={','.join(steps)} output_root={resolved_output_root}")
        pipeline_status = "success"
        return 0
    except Exception as exc:
        pipeline_status = "failed"
        pipeline_error = f"{type(exc).__name__}: {exc}"
        try:
            timing.write_summaries()
        except Exception:
            pass
        print(f"[PIPELINE] status=failed steps={','.join(steps)} output_root={resolved_output_root}")
        print(f"[PIPELINE] exception={type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise
    finally:
        _aggregate_logs_to_output_root(
            output_root=Path(resolved_output_root),
            run_id=run_id,
            pipeline_status=pipeline_status,
            error_text=pipeline_error,
        )


if __name__ == "__main__":
    raise SystemExit(main())
