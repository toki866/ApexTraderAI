#!/usr/bin/env python3
from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_core.clusterers.ticc_clusterer import TICCClusterer


FUNCTION_CANDIDATES = ("ticc_labels", "fit_predict", "run_ticc", "ticc", "cluster")
CLASS_HINTS = ("TICC", "FastTICC")


def _short_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def main() -> int:
    print(f"python_executable={sys.executable}")

    fast_spec = importlib.util.find_spec("fast_ticc")
    solver_spec = importlib.util.find_spec("TICC_solver")
    print(f"find_spec_fast_ticc={fast_spec}")
    print(f"find_spec_TICC_solver={solver_spec}")

    fast_mod = None
    try:
        fast_mod = importlib.import_module("fast_ticc")
        print("import_fast_ticc=ok")
    except Exception as exc:
        print(f"import_fast_ticc=ng error={_short_error(exc)}")

    entries = []
    if fast_mod is not None:
        fast_version = str(getattr(fast_mod, "__version__", ""))
        print(f"fast_ticc_version={fast_version}")
        for name in FUNCTION_CANDIDATES:
            obj = getattr(fast_mod, name, None)
            if not callable(obj):
                continue
            try:
                sig = str(inspect.signature(obj))
            except Exception as exc:
                sig = f"<signature unavailable: {_short_error(exc)}>"
            entries.append({"name": f"fast_ticc.{name}", "kind": "function", "signature": sig})

        for name in sorted(n for n in dir(fast_mod) if not n.startswith("_")):
            obj = getattr(fast_mod, name, None)
            if not inspect.isclass(obj):
                continue
            lowered = name.lower()
            if name in CLASS_HINTS or "ticc" in lowered:
                try:
                    sig = str(inspect.signature(obj))
                except Exception as exc:
                    sig = f"<signature unavailable: {_short_error(exc)}>"
                entries.append({"name": f"fast_ticc.{name}", "kind": "class", "signature": sig})

    print("fast_ticc_entrypoints=" + json.dumps(entries, ensure_ascii=False))

    clusterer = TICCClusterer(
        num_clusters=2,
        window_size=2,
        lambda_parameter=0.11,
        beta=600.0,
        max_iter=1,
        threshold=2e-5,
    )
    try:
        diag = clusterer.get_diagnostics()
    except Exception as exc:
        print(f"ticc_clusterer_diagnostics=ng error={_short_error(exc)}")
        return 3

    print("ticc_clusterer_diagnostics=" + json.dumps(diag, ensure_ascii=False))
    print(f"resolved_backend_name={diag.get('backend_resolved_name', '')}")
    print(f"resolved_entrypoint={diag.get('backend_entrypoint_name', '')}")
    print(f"resolved_entrypoint_kind={diag.get('backend_entrypoint_kind', '')}")

    if not str(diag.get("backend_resolved_name", "") or ""):
        print("result=ng backend unresolved")
        return 3

    print("result=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
