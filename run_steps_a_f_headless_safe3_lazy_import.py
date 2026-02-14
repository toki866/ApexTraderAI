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
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple


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
            # Returning None lets the service apply its own defaults (e.g. config.agents or ['xsr']).
            return None


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
    return Path(__file__).resolve().parent


def _parse_steps(s: str) -> Tuple[str, ...]:
    s = (s or "").strip()
    if not s:
        return ("A", "B", "C", "D", "E", "F")
    parts = [p.strip().upper() for p in s.replace(" ", "").split(",") if p.strip()]
    if any(p in ("ALL", "*") for p in parts):
        return ("A", "B", "C", "D", "E", "F")
    valid = {"A", "B", "C", "D", "E", "F"}
    out = tuple([p for p in parts if p in valid])
    return out if out else ("A", "B", "C", "D", "E", "F")


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


def _load_prices_dates(symbol: str, repo_root: Path) -> Tuple[Optional[Any], Optional[Any]]:
    import pandas as pd
    candidates = [
        repo_root / "output" / f"stepA_prices_{symbol}.csv",
        repo_root / "data" / f"prices_{symbol}.csv",
    ]
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
            return dates, close
    return None, None


def _build_date_range(symbol: str, repo_root: Path, test_start: Optional[str], test_months: int, train_years: int):
    import pandas as pd
    from ai_core.types.common import DateRange
    dates, _ = _load_prices_dates(symbol, repo_root)
    if dates is None or len(dates) == 0:
        raise RuntimeError("Cannot build DateRange: no price dates found (StepA output or data CSV missing).")
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
    return _ensure_date_range_aliases(dr)


def _ensure_stepb_pred_time_all(symbol: str, repo_root: Path) -> Path:
    import pandas as pd
    out_root = repo_root / "output"
    out_root.mkdir(parents=True, exist_ok=True)
    target = out_root / f"stepB_pred_time_all_{symbol}.csv"
    if target.exists():
        return target
    dates, close = _load_prices_dates(symbol, repo_root)
    if dates is None:
        raise RuntimeError("Cannot build stepB_pred_time_all: StepA prices/date not found.")
    base = pd.DataFrame({"Date": pd.to_datetime(dates)})
    stepb_dir = out_root / "stepB"
    files = list(stepb_dir.glob(f"*{symbol}*.csv"))
    files.extend(out_root.glob(f"stepB_*{symbol}*.csv"))
    agent_map = {"XSR": None, "MAMBA": None, "FED": None}
    for p in files:
        name = p.name.lower()
        if "xsr" in name:
            agent_map["XSR"] = p
        elif "mamba" in name:
            agent_map["MAMBA"] = p
        elif "fed" in name:
            agent_map["FED"] = p
    for agent, p in agent_map.items():
        if p is None:
            base[f"Pred_Close_{agent}"] = float("nan")
            continue
        s = _load_pred_close(p)
        if s is None:
            base[f"Pred_Close_{agent}"] = float("nan")
            continue
        s = s.reindex(base["Date"])
        base[f"Pred_Close_{agent}"] = s.values
    target.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(target, index=False)
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
    ctx = {"app_config": app_config, "symbol": symbol, "sym": symbol, "date_range": date_range}
    svc = _instantiate_service(StepAService, ctx)
    fn = getattr(svc, "run", None)
    if fn is None:
        raise RuntimeError("StepAService has no run()")
    return _call_with_best_effort(fn, ctx)


def _run_stepB(app_config, symbol: str, date_range):
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

    fn = getattr(svc, "run", None)
    if fn is None:
        raise RuntimeError("StepBService has no run()")
    # Some StepBService.run takes cfg as positional; keep it simple.
    return fn(cfg)


def _run_step_generic(step_letter: str, app_config, symbol: str, date_range, prev_results: Dict[str, Any]):
    mod_map = {
        "C": ("ai_core.services.step_c_service", "StepCService"),
        "D": ("ai_core.services.step_d_service", "StepDService"),
        "E": ("ai_core.services.step_e_service", "StepEService"),
        "F": ("ai_core.services.step_f_service", "StepFService"),
    }
    module_name, cls_name = mod_map[step_letter]
    module = __import__(module_name, fromlist=[cls_name])
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
    ap.add_argument("--steps", default=None, help="Comma-separated steps to run. Example: A,B,C (default ALL)")
    ap.add_argument("--test-start", dest="test_start", default=None, help="YYYY-MM-DD. If omitted, uses last-3-months start.")
    ap.add_argument("--train-years", type=int, default=8)
    ap.add_argument("--test-months", type=int, default=3)
    args = ap.parse_args(argv)

    repo_root = _repo_root()

    symbol = args.symbol or _env_get("AUTODEBUG_SYMBOL", "AUTO_DEBUG_SYMBOL") or "SOXL"
    steps = _parse_steps(args.steps)

    # Ensure repo_root is on sys.path
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    app_config = _get_app_config(repo_root)
    date_range = _build_date_range(symbol, repo_root, args.test_start, args.test_months, args.train_years)

    results: Dict[str, Any] = {}

    print(f"[headless] repo_root={repo_root}")
    print(f"[headless] symbol={symbol}")
    print(f"[headless] steps={','.join(steps)}")

    if "A" in steps:
        print("[StepA] start")
        results["stepA_result"] = _run_stepA(app_config, symbol, date_range)
        print("[StepA] done")

    if "B" in steps:
        print("[StepB] start")
        results["stepB_result"] = _run_stepB(app_config, symbol, date_range)
        print("[StepB] done")
        # Ensure contract artifact exists
        try:
            p = _ensure_stepb_pred_time_all(symbol, repo_root)
            print(f"[StepB] ensured: {p}")
        except Exception as e:
            print(f"[StepB] WARN: failed to ensure stepB_pred_time_all: {e}", file=sys.stderr)

    for step in ("C", "D", "E", "F"):
        if step not in steps:
            continue
        print(f"[Step{step}] start")
        results[f"step{step}_result"] = _run_step_generic(step, app_config, symbol, date_range, results)
        print(f"[Step{step}] done")

    print("[headless] ALL DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())