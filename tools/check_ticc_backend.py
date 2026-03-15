from __future__ import annotations

import json
import sys
from typing import Any, Dict, List

import numpy as np


def _safe_import(name: str):
    try:
        module = __import__(name)
        return True, module, ""
    except Exception as exc:
        return False, None, f"{type(exc).__name__}: {exc}"


def main() -> int:
    report: Dict[str, Any] = {
        "python_executable": sys.executable,
    }

    ok_fast, mod_fast, fast_err = _safe_import("fast_ticc")
    ok_solver, mod_solver, solver_err = _safe_import("TICC_solver")

    report["fast_ticc_import_ok"] = ok_fast
    report["fast_ticc_import_error"] = fast_err
    report["TICC_solver_import_ok"] = ok_solver
    report["TICC_solver_import_error"] = solver_err

    backend_name = ""
    ticc_cls = None
    if ok_fast:
        backend_name = "fast_ticc"
        ticc_cls = getattr(mod_fast, "TICC", None)
    elif ok_solver:
        backend_name = "TICC_solver"
        ticc_cls = getattr(mod_solver, "TICC", None)

    report["backend_selected"] = backend_name
    report["TICC_class_exists"] = ticc_cls is not None

    methods: List[str] = []
    usable = False
    fit_ok = False
    predict_ok = False
    instantiate_error = ""
    fit_error = ""
    predict_error = ""

    if ticc_cls is not None:
        kwargs = {
            "window_size": 3,
            "number_of_clusters": 2,
            "lambda_parameter": 0.11,
            "beta": 50.0,
            "maxIters": 5,
            "threshold": 1e-3,
            "write_out_file": False,
            "prefix_string": "",
        }
        try:
            import inspect

            sig = inspect.signature(ticc_cls)
            init_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
            obj = ticc_cls(**init_kwargs)
            methods = sorted([m for m in dir(obj) if not m.startswith("_")])
            report["init_kwargs_used"] = sorted(init_kwargs.keys())
            report["methods"] = methods
            x = np.random.RandomState(42).randn(40, 3).astype(float)
            for cand in ("fit_predict", "fit_transform", "fit"):
                fn = getattr(obj, cand, None)
                if callable(fn):
                    labels = np.asarray(fn(x)).reshape(-1)
                    fit_ok = labels.shape[0] == x.shape[0]
                    break
            if not fit_ok:
                fit_error = "no usable fit method or unexpected label shape"

            if fit_ok:
                x2 = np.random.RandomState(7).randn(8, 3).astype(float)
                for cand in ("predict", "predict_clusters", "transform"):
                    fn = getattr(obj, cand, None)
                    if callable(fn):
                        out = np.asarray(fn(x2)).reshape(-1)
                        predict_ok = out.shape[0] == x2.shape[0]
                        if predict_ok:
                            break
                if not predict_ok:
                    predict_error = "no direct predict method (adapter fallback may still work)"

            usable = fit_ok
        except Exception as exc:
            instantiate_error = f"{type(exc).__name__}: {exc}"

    report["instantiate_error"] = instantiate_error
    report["fit_ok"] = fit_ok
    report["fit_error"] = fit_error
    report["predict_ok"] = predict_ok
    report["predict_error"] = predict_error
    report["usable"] = usable

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"RESULT: {'usable' if usable else 'unusable'}")
    return 0 if usable else 1


if __name__ == "__main__":
    raise SystemExit(main())
