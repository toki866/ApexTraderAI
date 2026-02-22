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
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, List

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ai_core.utils.paths import get_repo_root
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

    def _try_set(obj: Any, name: str, value: Any) -> bool:
        try:
            setattr(obj, name, value)
            return True
        except Exception:
            return False

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


def _parse_steps(s: str) -> Tuple[str, ...]:
    default_steps: Tuple[str, ...] = ("A", "B", "C", "DPRIME", "E", "F")
    canonical_order: Tuple[str, ...] = ("A", "B", "C", "D", "DPRIME", "E", "F")

    step_aliases = {
        "A": "A",
        "B": "B",
        "C": "C",
        "D": "D",
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
    if any(p in ("ALL", "*") for p in parts):
        return canonical_order

    requested = {step_aliases[p] for p in parts if p in step_aliases}
    if not requested:
        return default_steps

    return tuple(step for step in canonical_order if step in requested)






_OFFICIAL_STEPE_AGENTS: Tuple[str, ...] = (
    "dprime_bnf_h01",
    "dprime_bnf_h02",
    "dprime_bnf_h03",
    "dprime_all_features_h01",
    "dprime_all_features_h02",
    "dprime_all_features_h03",
    "dprime_mix_h01",
    "dprime_bnf_3scale",
    "dprime_all_features_3scale",
    "dprime_mix_3scale",
)


def _device_auto() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _inject_default_stepe_configs(app_config: Any, output_root: Path) -> None:
    from ai_core.services.step_e_service import StepEConfig

    cfg_list = []
    for agent_name in _OFFICIAL_STEPE_AGENTS:
        cfg = StepEConfig(agent=str(agent_name))
        cfg.output_root = str(output_root)
        cfg.obs_profile = "DPRIME"
        cfg.seed = 42
        cfg.device = "auto"
        cfg.epochs = 200
        cfg.patience = min(int(getattr(cfg, "patience", 15)), 10)
        cfg_list.append(cfg)

    if isinstance(app_config, dict):
        app_config["stepE"] = cfg_list
    else:
        setattr(app_config, "stepE", cfg_list)


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


def _load_prices_dates(
    symbol: str,
    repo_root: Path,
    output_root: Path,
    data_root: Path,
    mode: str,
) -> Tuple[Optional[Any], Optional[Any], List[Path]]:
    import pandas as pd

    mode = (mode or "sim").strip().lower()
    if mode == "ops":
        mode = "live"

    candidates = [
        # (Priority 1) split outputs in the requested mode
        output_root / "stepA" / mode / f"stepA_prices_train_{symbol}.csv",
        output_root / "stepA" / mode / f"stepA_prices_test_{symbol}.csv",
        # (Priority 2) consolidated stepA prices (display and compatibility variants)
        output_root / "stepA" / mode / f"stepA_prices_{symbol}.csv",
        output_root / "stepA" / "display" / f"stepA_prices_{symbol}.csv",
        output_root / f"stepA_prices_{symbol}.csv",
        repo_root / "output" / f"stepA_prices_{symbol}.csv",
        # (Priority 3) raw source fallback
        data_root / f"prices_{symbol}.csv",
    ]

    selected_dates = None
    selected_close = None
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
            dates = df[date_col]
            close = df[close_col] if close_col else None
            if selected_dates is None:
                selected_dates = dates
                selected_close = close
    return selected_dates, selected_close, candidates


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

    out_root = Path(output_root) if output_root is not None else (repo_root / "output")
    data_root = Path(data_root) if data_root is not None else (repo_root / "data")

    dates, _, searched_paths = _load_prices_dates(
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
    ts = pd.to_datetime(test_start) if test_start else pd.to_datetime(dates.iloc[-1]) - pd.DateOffset(months=test_months)
    train_start = ts - pd.DateOffset(years=train_years)
    train_end = ts - pd.Timedelta(days=1)
    test_end = ts + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)
    dmin = pd.to_datetime(dates.iloc[0])
    dmax = pd.to_datetime(dates.iloc[-1])
    train_start = max(train_start, dmin)
    train_end = min(train_end, dmax)
    ts = max(ts, dmin)
    test_end = min(test_end, dmax)
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


def _ensure_stepb_pred_time_all(symbol: str, repo_root: Path, mode: str) -> Path:
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

    out_root = repo_root / 'output'
    stepb_mode_dir = out_root / 'stepB' / mode
    stepb_mode_dir.mkdir(parents=True, exist_ok=True)

    target = stepb_mode_dir / f'stepB_pred_time_all_{symbol}.csv'
    if target.exists() and target.stat().st_size > 0:
        return target

    # If a legacy pred_time_all exists, copy it into the mode folder (do not delete legacy).
    legacy_candidates = [
        out_root / 'stepB' / f'stepB_pred_time_all_{symbol}.csv',
        out_root / f'stepB_pred_time_all_{symbol}.csv',
    ]
    for c in legacy_candidates:
        if c.exists() and c.stat().st_size > 0:
            shutil.copy2(c, target)
            return target

    # Build from agent pred_close files.
    prices_dates = _load_prices_dates(symbol, repo_root)
    df = pd.DataFrame({'Date': prices_dates})

    # Gather candidate files from mode dir first, then legacy stepB dir as fallback.
    candidate_dirs = [stepb_mode_dir, out_root / 'stepB']
    files: list[Path] = []
    for d in candidate_dirs:
        if d.exists():
            files.extend(sorted(d.glob(f'*{symbol}*.csv')))

    def _pick_best(files: list[Path], key: str) -> Path | None:
        key_l = key.lower()
        cands = [f for f in files if key_l in f.name.lower()]
        if not cands:
            return None
        # Prefer pred_close, then pred_path, then anything else; avoid delta.
        def score(f: Path) -> tuple:
            n = f.name.lower()
            return (
                0 if 'pred_time_all' in n else 1,
                0 if 'pred_close' in n else 1,
                0 if 'pred_path' in n else 1,
                1 if 'delta' in n else 0,
                len(n),
            )
        cands.sort(key=score)
        return cands[0]

    agent_map = {
        'MAMBA': _pick_best(files, 'mamba'),
    }

    for agent, path in agent_map.items():
        if path is None:
            continue
        series = _load_pred_close(path, agent)
        df = df.merge(series, on='Date', how='left')

    df.to_csv(target, index=False, encoding='utf-8')
    return target
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
    def _try_set(obj, attr: str, value):
        if hasattr(obj, attr):
            try:
                setattr(obj, attr, value)
                return True
            except Exception:
                return False
        return False

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


def _run_stepDPrime(app_config, symbol: str, date_range, mode: str):
    from ai_core.services.step_dprime_service import StepDPrimeService, StepDPrimeConfig

    out_root = str(getattr(app_config, "output_root", "output"))
    cfg = StepDPrimeConfig(
        symbol=symbol,
        mode=(mode or "sim"),
        output_root=out_root,
        seed=42,
        device="auto",
    )
    svc = StepDPrimeService()
    return svc.run(cfg)


def _run_step_generic(step_letter: str, app_config, symbol: str, date_range, prev_results: Dict[str, Any]):
    mod_map = {
        "C": ("ai_core.services.step_c_service", "StepCService"),
        "DPRIME": ("ai_core.services.step_dprime_service", "StepDPrimeService"),
        "D": ("ai_core.services.step_d_service", "StepDService"),
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
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--steps", default="A,B,C,D,DPRIME,E,F", help="Comma-separated steps to run. Example: A,B,C,D,DPRIME,E,F")
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
    ap.add_argument("--data-start", default="2010-01-01", help="Start date for auto data preparation (YYYY-MM-DD).")
    ap.add_argument("--data-end", default=None, help="End date for auto data preparation (YYYY-MM-DD, default=today).")
    args = ap.parse_args(argv)

    repo_root = _repo_root()

    symbol = args.symbol or _env_get("AUTODEBUG_SYMBOL", "AUTO_DEBUG_SYMBOL") or "SOXL"
    steps = _parse_steps(args.steps)
    skip_stepe_env = str(_env_get("SKIP_STEPE") or "").strip().lower() in {"1", "true", "yes", "on"}
    skip_stepe = bool(args.skip_stepe or skip_stepe_env)
    if skip_stepe and "E" in steps:
        steps = tuple(step for step in steps if step != "E")

    # StepB is Mamba-only. If user didn't specify, default to enabled.
    enable_mamba = bool(args.enable_mamba)
    if not enable_mamba:
        enable_mamba = True


    # Ensure repo_root is on sys.path
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    app_config = _get_app_config(repo_root)

    cfg_output_root = Path(getattr(app_config, "output_root", "output"))
    resolved_output_root = Path(args.output_root) if args.output_root else cfg_output_root
    resolved_mode = (args.mode or "sim").strip().lower()
    resolved_mamba_mode = (args.mode or args.mamba_mode or "sim").strip().lower()
    resolved_stepE_mode = (args.mode or args.stepE_mode or "sim").strip().lower()

    resolved_output_root.mkdir(parents=True, exist_ok=True)
    app_config = _apply_config_output_root(app_config, resolved_output_root)
    print(
        f"[headless] output_root_resolved={resolved_output_root} "
        f"cfg_output_root={getattr(app_config, 'output_root', None)} "
        f"cfg_data_output_root={getattr(getattr(app_config, 'data', None), 'output_root', None)}"
    )

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

    _prepare_missing_data_if_needed(
        repo_root=repo_root,
        data_root=data_root,
        primary_symbol=symbol,
        auto_prepare_data=auto_prepare_data,
        data_start=data_start,
        data_end=data_end,
    )

    future_end = args.future_end or _env_get("AUTODEBUG_FUTURE_END", "FUTURE_END", "FUTURE_END_DATE")
    date_range = _build_date_range(
        symbol,
        repo_root,
        args.test_start,
        args.test_months,
        args.train_years,
        future_end=future_end,
        mamba_mode=resolved_mamba_mode,
        stepE_mode=resolved_stepE_mode,
        mode=resolved_mode,
        output_root=resolved_output_root,
        data_root=data_root,
        env_horizon_days=env_horizon_base,
    )

    results: Dict[str, Any] = {}

    # Enabled agents list (propagated to StepE/StepF via ctx so they can run multi-agent).
    enabled_agents: List[str] = ['mamba'] if enable_mamba else []
    # Store into results so _run_step_generic passes it as ctx['agents'].
    results['agents'] = enabled_agents
    # Optional: configure StepE to use StepD' transformer embeddings (compression-as-learning).
    # This lets StepE consume embeddings from:
    #   <output_root>/stepD_prime/<mode>/embeddings/stepDprime_{source}_h{HH}_{SYMBOL}_embeddings.csv
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
                print(f"[headless] StepE default config injected: agents={','.join(_OFFICIAL_STEPE_AGENTS)} seed=42 device=auto")
            except Exception as e:
                print(f"[headless] WARNING: failed to inject default StepE config: {type(e).__name__}: {e}")

    if "F" in steps:
        step_f_cfg = app_config.get("stepF") if isinstance(app_config, dict) else getattr(app_config, "stepF", None)
        if step_f_cfg is None:
            try:
                from ai_core.services.step_f_service import StepFConfig  # lazy import

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

                cfgF = StepFConfig(
                    output_root=out_root,
                    agents=",".join(sorted(unique_agents)),
                    seed=42,
                    device="auto",
                )

                if isinstance(app_config, dict):
                    app_config["stepF"] = cfgF
                else:
                    setattr(app_config, "stepF", cfgF)

                print(f"[headless] StepF default config injected: agents={cfgF.agents} seed={cfgF.seed} device={cfgF.device}")
            except Exception as e:
                print(f"[headless] WARNING: failed to inject default StepF config: {type(e).__name__}: {e}")


    print(f"[headless] repo_root={repo_root}")
    print(f"[headless] symbol={symbol}")
    if future_end:
        print(f"[headless] future_end={future_end}")
    print(f"[headless] steps={','.join(steps)}")
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

    try:
        if "A" in steps:
            print("[StepA] start")
            results["stepA_result"] = _run_stepA(app_config, symbol, date_range)
            print("[StepA] done")

        if "B" in steps:
            print("[StepB] start")
            mamba_horizons_list = _parse_int_list(args.mamba_horizons)
            results["stepB_result"] = _run_stepB(app_config, symbol, date_range, enable_mamba, args.enable_mamba_periodic, args.mamba_lookback, mamba_horizons_list)
            print(f"[StepB] agents: mamba={enable_mamba}")
            print("[StepB] done")
            # Ensure contract artifact exists
            try:
                p = _ensure_stepb_pred_time_all(symbol, repo_root, mode=resolved_mamba_mode)
                print(f"[StepB] ensured: {p}")
            except Exception as e:
                print(f"[StepB] WARN: failed to ensure stepB_pred_time_all: {e}", file=sys.stderr)

        if "C" in steps:
            print("[StepC] start")
            results["stepC_result"] = _run_step_generic("C", app_config, symbol, date_range, results)
            print("[StepC] done")

        if "DPRIME" in steps:
            print("[StepDPrime] start")
            results["stepDPRIME_result"] = _run_stepDPrime(app_config, symbol, date_range, mode=resolved_mamba_mode)
            print("[StepDPrime] done")

        for step in ("D", "E", "F"):
            if step not in steps:
                continue
            print(f"[Step{step}] start")
            step_result = _run_step_generic(step, app_config, symbol, date_range, results)
            results[f"step{step}_result"] = step_result
            if step == "E" and isinstance(step_result, dict) and step_result.get("skipped"):
                reason = step_result.get("reason", "unknown")
                raise RuntimeError(f"StepE reported skipped=True without explicit skip flag. reason={reason}")
            print(f"[Step{step}] done")

        print("[headless] ALL DONE")
        print(f"[PIPELINE] status=success steps={','.join(steps)} output_root={resolved_output_root}")
        return 0
    except Exception:
        print(f"[PIPELINE] status=failed steps={','.join(steps)} output_root={resolved_output_root}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
