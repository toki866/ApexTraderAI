#!/usr/bin/env python3
from __future__ import annotations

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
    print(f"find_spec_fast_ticc={bool(fast_spec)}")
    print(f"find_spec_TICC_solver={bool(solver_spec)}")

    try:
        from fast_ticc import TICC as FastTICC  # type: ignore

        print(f"fast_ticc_import=ok class={FastTICC.__name__}")
    except Exception as exc:
        print(f"fast_ticc_import=ng error={type(exc).__name__}: {exc}")
        return 2

    try:
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
        backend_name = str(diag.get("backend_resolved_name", "") or "")
        if not backend_name:
            print("fast_ticc ng: backend unresolved")
            return 3
        print(f"fast_ticc ok backend={backend_name}")
        return 0
    except Exception as exc:
        print(f"ticc_clusterer_diag=ng error={type(exc).__name__}: {exc}")
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
