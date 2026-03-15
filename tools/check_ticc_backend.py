#!/usr/bin/env python3
from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_core.clusterers.ticc_clusterer import TICCClusterer


def main() -> int:
    print(f"python_executable={sys.executable}")

    fast_spec = importlib.util.find_spec("fast_ticc")
    solver_spec = importlib.util.find_spec("TICC_solver")
    print(f"find_spec_fast_ticc={fast_spec}")
    print(f"find_spec_TICC_solver={solver_spec}")

    fast_version = ""
    fast_api = []
    try:
        fast_mod = importlib.import_module("fast_ticc")
        fast_version = str(getattr(fast_mod, "__version__", ""))
        fast_api = sorted([name for name in dir(fast_mod) if not name.startswith("_")])
        print("import_fast_ticc=ok")
    except Exception as exc:
        print(f"import_fast_ticc=ng error={type(exc).__name__}: {exc}")
    print(f"fast_ticc_version={fast_version}")
    print("fast_ticc_public_api=" + json.dumps(fast_api, ensure_ascii=False))

    clusterer = TICCClusterer(
        num_clusters=2,
        window_size=2,
        lambda_parameter=0.11,
        beta=600.0,
        max_iter=1,
        threshold=2e-5,
    )
    diag = clusterer.get_diagnostics()
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
