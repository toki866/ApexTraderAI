from __future__ import annotations

from dataclasses import dataclass
import importlib
import inspect
import types
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class TICCUnavailableError(RuntimeError):
    """Raised when TICC backend is unavailable in the current environment."""


@dataclass
class _BackendSpec:
    name: str
    entrypoint: Any
    entrypoint_name: str
    entrypoint_kind: str  # class | function


class TICCClusterer:
    """Adapter that uses a real TICC backend when installed.

    Supported backends (in priority order):
    - fast_ticc (class/function entrypoints discovered dynamically)
    - TICC_solver.TICC (legacy)
    """

    def __init__(
        self,
        *,
        num_clusters: int,
        window_size: int,
        lambda_parameter: float,
        beta: float,
        max_iter: int,
        threshold: float,
    ) -> None:
        self.num_clusters = int(num_clusters)
        self.window_size = int(window_size)
        self.lambda_parameter = float(lambda_parameter)
        self.beta = float(beta)
        self.max_iter = int(max_iter)
        self.threshold = float(threshold)
        self._model: Optional[Any] = None
        self._backend_name: str = ""
        self._backend_resolve_error: str = ""
        self._backend_methods: Tuple[str, ...] = tuple()
        self._backend_entrypoint_name: str = ""
        self._backend_entrypoint_kind: str = ""
        self._backend_api_candidates: Tuple[str, ...] = tuple()
        self._train_centroids: Optional[np.ndarray] = None
        self._centroid_cluster_ids: Optional[np.ndarray] = None

    def fit_predict_train(self, x_train: np.ndarray) -> np.ndarray:
        backend = self._resolve_backend()
        self._backend_name = backend.name
        self._backend_entrypoint_name = backend.entrypoint_name
        self._backend_entrypoint_kind = backend.entrypoint_kind
        self._backend_resolve_error = ""

        x_train = np.asarray(x_train, dtype=float)
        if x_train.ndim != 2 or x_train.shape[0] == 0:
            raise ValueError("x_train must be a non-empty 2D array")

        try:
            if backend.entrypoint_kind == "class":
                model = self._instantiate_model(backend)
                self._backend_methods = tuple(sorted(m for m in dir(model) if not m.startswith("_")))
                labels_raw = self._call_first_available(
                    model,
                    stage="fit_predict_train",
                    x=x_train,
                    candidates=("fit_predict", "fit_transform", "fit"),
                )
                self._model = model
            else:
                self._model = None
                self._backend_methods = tuple()
                labels_raw = self._call_function_backend_fit_predict(backend=backend, x=x_train)
        except Exception as exc:
            if isinstance(exc, TICCUnavailableError):
                raise
            raise TICCUnavailableError(
                "TICC backend execution failed: "
                f"backend={backend.name} entrypoint={backend.entrypoint_name} kind={backend.entrypoint_kind} "
                f"stage=fit_predict_train error={type(exc).__name__}: {exc}"
            ) from exc

        labels = self._normalize_labels(labels_raw, expected_len=x_train.shape[0])
        self._fit_centroids(x_train=x_train, labels=labels)
        return labels

    def predict_test(self, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x_test = np.asarray(x_test, dtype=float)
        if x_test.ndim != 2:
            raise ValueError("x_test must be a 2D array")
        if x_test.shape[0] == 0:
            return np.zeros((0,), dtype=int), np.zeros((0,), dtype=float)

        labels: Optional[np.ndarray] = None
        pred_error: Optional[Exception] = None
        if self._model is not None:
            try:
                labels_raw = self._call_first_available(
                    self._model,
                    stage="predict_test",
                    x=x_test,
                    candidates=("predict", "predict_clusters", "transform"),
                )
                labels = self._normalize_labels(labels_raw, expected_len=x_test.shape[0])
            except Exception as exc:
                pred_error = exc

        if labels is None:
            labels, confidence = self._assign_by_nearest_centroid(x_test)
            if labels is None:
                method_list = ",".join(self._backend_methods) if self._backend_methods else ""
                missing = "predict,predict_clusters,transform"
                cause = f"{type(pred_error).__name__}: {pred_error}" if pred_error is not None else "method_not_found"
                raise TICCUnavailableError(
                    "TICC backend test-time assignment failed "
                    f"backend={self._backend_name or 'unknown'} entrypoint={self._backend_entrypoint_name or 'unknown'} "
                    f"stage=predict_test missing_method={missing} available_methods=[{method_list}] cause={cause}"
                ) from pred_error
            return labels, confidence

        confidence = np.ones((labels.shape[0],), dtype=float)
        return labels, confidence

    def get_diagnostics(self) -> Dict[str, object]:
        backend_name = self._backend_name
        backend_error = self._backend_resolve_error
        entrypoint_name = self._backend_entrypoint_name
        entrypoint_kind = self._backend_entrypoint_kind
        if not backend_name and not backend_error:
            try:
                backend = self._resolve_backend()
                backend_name = backend.name
                entrypoint_name = backend.entrypoint_name
                entrypoint_kind = backend.entrypoint_kind
            except Exception as exc:
                backend_error = f"{type(exc).__name__}: {exc}"
                self._backend_resolve_error = backend_error
        return {
            "backend_resolved_name": backend_name,
            "backend_resolve_error": backend_error,
            "backend_entrypoint_name": entrypoint_name,
            "backend_entrypoint_kind": entrypoint_kind,
            "backend_predict_methods": [
                name for name in self._backend_methods if name in {"predict", "predict_clusters", "transform"}
            ],
            "backend_methods": list(self._backend_methods),
            "backend_api_candidates": list(self._backend_api_candidates),
        }

    def _instantiate_model(self, backend: _BackendSpec) -> Any:
        sig = inspect.signature(backend.entrypoint)
        kwargs = {k: v for k, v in self._ticc_common_kwargs().items() if k in sig.parameters}
        try:
            return backend.entrypoint(**kwargs)
        except Exception as exc:
            raise TICCUnavailableError(
                "Failed to initialize TICC backend "
                f"backend={backend.name} entrypoint={backend.entrypoint_name} "
                f"with kwargs={sorted(kwargs.keys())}: {type(exc).__name__}: {exc}"
            ) from exc

    def _call_function_backend_fit_predict(self, *, backend: _BackendSpec, x: np.ndarray) -> Any:
        fn = backend.entrypoint
        sig = inspect.signature(fn)
        params = sig.parameters
        data_arg_candidates = ("data_series", "time_series", "observations", "input_data", "x", "X", "data", "series")
        common_kwargs = self._ticc_common_kwargs()
        alias_kwargs = {
            "num_clusters": self.num_clusters,
            "sparsity_weight": self.lambda_parameter,
            "label_switching_cost": self.beta,
            "iteration_limit": self.max_iter,
        }
        kwargs_all = {**common_kwargs, **alias_kwargs}
        kwargs = {k: v for k, v in kwargs_all.items() if k in params}
        selected_data_kw: Optional[str] = None
        for key in data_arg_candidates:
            if key in params:
                kwargs[key] = x
                selected_data_kw = key
                break

        positional_args: List[Any] = []
        if selected_data_kw is None:
            required_positional = [
                p
                for p in params.values()
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                and p.default is inspect._empty
            ]
            if required_positional:
                positional_args = [x]

        if selected_data_kw is None and not positional_args:
            raise TICCUnavailableError(
                "TICC function backend API mismatch: no compatible data input argument was found "
                f"backend={backend.name} entrypoint={backend.entrypoint_name} signature={sig} "
                f"supported_data_arg_candidates={list(data_arg_candidates)} accepted_kwargs={sorted(kwargs.keys())}"
            )

        try:
            result = fn(*positional_args, **kwargs)
        except Exception as exc:
            raise TICCUnavailableError(
                "TICC function backend invocation failed "
                f"backend={backend.name} entrypoint={backend.entrypoint_name} signature={sig} "
                f"passed_positional={len(positional_args)} passed_keyword_names={sorted(kwargs.keys())} "
                f"error={type(exc).__name__}: {exc}"
            ) from exc

        if isinstance(result, dict):
            for key in ("labels", "cluster_assignment", "assignments", "y", "clusters", "point_labels"):
                if key in result:
                    return result[key]
        if isinstance(result, (tuple, list)):
            if self._is_label_like_array(result, expected_len=x.shape[0]):
                return result
            for item in result:
                if self._is_label_like_array(item, expected_len=x.shape[0]):
                    return item
        for attr_name in ("labels", "cluster_assignment", "assignments", "y", "clusters", "point_labels"):
            if hasattr(result, attr_name):
                attr = getattr(result, attr_name)
                if self._is_label_like_array(attr, expected_len=x.shape[0]):
                    return attr
        if not self._is_label_like_array(result, expected_len=x.shape[0]):
            raise TICCUnavailableError(
                "TICC function backend returned unsupported output for label extraction "
                f"backend={backend.name} entrypoint={backend.entrypoint_name} signature={sig} "
                f"passed_positional={len(positional_args)} passed_keyword_names={sorted(kwargs.keys())} "
                f"result_type={type(result).__name__}"
            )
        return result

    @staticmethod
    def _is_label_like_array(value: Any, *, expected_len: int) -> bool:
        try:
            arr = np.asarray(value)
        except Exception:
            return False
        if arr.ndim == 0:
            return False
        if arr.ndim == 1:
            return int(arr.shape[0]) == int(expected_len)
        if arr.ndim >= 2:
            return int(arr.shape[0]) == int(expected_len)
        return False

    @staticmethod
    def _call_first_available(model: Any, *, stage: str, x: np.ndarray, candidates: Tuple[str, ...]) -> Any:
        for method_name in candidates:
            method = getattr(model, method_name, None)
            if callable(method):
                return method(x)
        available = sorted(m for m in dir(model) if not m.startswith("_"))
        raise TICCUnavailableError(
            f"TICC backend API mismatch stage={stage} missing_method={','.join(candidates)} available_methods={available}"
        )

    @staticmethod
    def _normalize_labels(labels_raw: Any, expected_len: int) -> np.ndarray:
        labels = np.asarray(labels_raw)
        if labels.ndim == 2 and labels.shape[1] >= 1:
            labels = labels[:, 0]
        labels = labels.reshape(-1)
        if labels.shape[0] != expected_len:
            raise TICCUnavailableError(
                f"TICC returned unexpected label length: got={labels.shape[0]} expected={expected_len}"
            )
        return labels.astype(int)

    def _fit_centroids(self, *, x_train: np.ndarray, labels: np.ndarray) -> None:
        clusters = sorted(set(labels.astype(int).tolist()))
        if not clusters:
            self._train_centroids = None
            return
        centroids = []
        for cluster_id in clusters:
            centroids.append(np.mean(x_train[labels == cluster_id], axis=0))
        self._train_centroids = np.asarray(centroids, dtype=float)
        self._centroid_cluster_ids = np.asarray(clusters, dtype=int)

    def _assign_by_nearest_centroid(self, x_test: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self._train_centroids is None or self._train_centroids.size == 0:
            return None, None
        diff = x_test[:, None, :] - self._train_centroids[None, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        best_idx = np.argmin(dists, axis=1)
        if self._centroid_cluster_ids is None:
            return None, None
        labels = self._centroid_cluster_ids[best_idx].astype(int)
        best_dist = dists[np.arange(len(best_idx)), best_idx]
        return labels, (1.0 / (1.0 + best_dist)).astype(float)

    def _ticc_common_kwargs(self) -> Dict[str, Any]:
        return {
            "window_size": self.window_size,
            "number_of_clusters": self.num_clusters,
            "lambda_parameter": self.lambda_parameter,
            "beta": self.beta,
            "maxIters": self.max_iter,
            "threshold": self.threshold,
            "write_out_file": False,
            "prefix_string": "",
        }

    def _resolve_backend(self) -> _BackendSpec:
        details: List[str] = []
        self._backend_api_candidates = tuple()

        fast = self._resolve_fast_ticc_backend(details)
        if fast is not None:
            return fast

        try:
            from TICC_solver import TICC as SolverTICC  # type: ignore

            self._backend_api_candidates = tuple(sorted(set(self._backend_api_candidates + ("TICC_solver.TICC",))))
            return _BackendSpec(
                name="TICC_solver",
                entrypoint=SolverTICC,
                entrypoint_name="TICC_solver.TICC",
                entrypoint_kind="class",
            )
        except Exception as solver_exc:
            details.append(f"import TICC_solver.TICC failed: {type(solver_exc).__name__}: {solver_exc}")

        raise TICCUnavailableError(
            "No usable TICC backend found. Checked fast_ticc dynamic entrypoints and legacy TICC_solver.TICC. "
            + " | ".join(details)
        )

    def _resolve_fast_ticc_backend(self, details: List[str]) -> Optional[_BackendSpec]:
        try:
            mod = importlib.import_module("fast_ticc")
        except Exception as exc:
            details.append(f"import fast_ticc failed: {type(exc).__name__}: {exc}")
            return None

        candidates: List[str] = []
        class_candidates = self._discover_fast_ticc_classes(mod, candidates)
        function_candidates = self._discover_fast_ticc_functions(mod, candidates)
        self._backend_api_candidates = tuple(sorted(set(candidates)))

        if class_candidates:
            name, cls = class_candidates[0]
            return _BackendSpec(name="fast_ticc", entrypoint=cls, entrypoint_name=name, entrypoint_kind="class")
        if function_candidates:
            name, fn = function_candidates[0]
            return _BackendSpec(name="fast_ticc", entrypoint=fn, entrypoint_name=name, entrypoint_kind="function")

        details.append(
            "fast_ticc imported but no supported class/function entrypoint found. "
            f"discovered_candidates={list(self._backend_api_candidates)}"
        )
        return None

    @staticmethod
    def _discover_fast_ticc_classes(mod: types.ModuleType, candidates: List[str]) -> List[Tuple[str, Any]]:
        search_targets: List[Tuple[str, Any]] = [("fast_ticc", mod)]
        for sub_name in ("main", "ticc", "frontend", "solver"):
            try:
                sub_mod = importlib.import_module(f"fast_ticc.{sub_name}")
                search_targets.append((f"fast_ticc.{sub_name}", sub_mod))
            except Exception:
                continue

        matches: List[Tuple[str, Any]] = []
        for prefix, obj in search_targets:
            for name in dir(obj):
                attr = getattr(obj, name, None)
                if not inspect.isclass(attr):
                    continue
                full = f"{prefix}.{name}"
                candidates.append(full)
                has_fit = callable(getattr(attr, "fit", None)) or callable(getattr(attr, "fit_predict", None))
                if name in {"TICC", "FastTICC"} or ("ticc" in name.lower() and has_fit):
                    matches.append((full, attr))
        matches.sort(key=lambda x: (0 if x[0].endswith(".TICC") else 1, x[0]))
        return matches

    @staticmethod
    def _discover_fast_ticc_functions(mod: types.ModuleType, candidates: List[str]) -> List[Tuple[str, Any]]:
        names = ("ticc_labels", "fit_predict", "run_ticc", "ticc", "cluster")
        matches: List[Tuple[str, Any]] = []
        for name in names:
            fn = getattr(mod, name, None)
            if callable(fn):
                full = f"fast_ticc.{name}"
                candidates.append(full)
                matches.append((full, fn))
        return matches
