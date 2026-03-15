from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any, Dict, Optional, Tuple

import numpy as np


class TICCUnavailableError(RuntimeError):
    """Raised when TICC backend is unavailable in the current environment."""


@dataclass
class _BackendSpec:
    name: str
    cls: Any


class TICCClusterer:
    """Adapter that uses a real TICC backend when installed.

    Supported backends:
    - `fast_ticc.TICC` (primary)
    - `TICC_solver.TICC` (legacy)
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
        self._train_centroids: Optional[np.ndarray] = None
        self._centroid_cluster_ids: Optional[np.ndarray] = None

    def fit_predict_train(self, x_train: np.ndarray) -> np.ndarray:
        backend = self._resolve_backend()
        self._backend_name = backend.name
        self._backend_resolve_error = ""
        x_train = np.asarray(x_train, dtype=float)
        if x_train.ndim != 2 or x_train.shape[0] == 0:
            raise ValueError("x_train must be a non-empty 2D array")

        model = self._instantiate_model(backend)
        self._backend_methods = tuple(sorted(m for m in dir(model) if not m.startswith("_")))

        try:
            labels_raw = self._call_first_available(
                model,
                stage="fit_predict_train",
                x=x_train,
                candidates=("fit_predict", "fit_transform", "fit"),
            )
        except Exception as exc:
            if isinstance(exc, TICCUnavailableError):
                raise
            raise TICCUnavailableError(
                f"TICC backend execution failed: backend={backend.name} stage=fit_predict_train error={type(exc).__name__}: {exc}"
            ) from exc

        labels = self._normalize_labels(labels_raw, expected_len=x_train.shape[0])
        self._model = model
        self._fit_centroids(x_train=x_train, labels=labels)
        return labels

    def predict_test(self, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._model is None:
            raise RuntimeError("TICC model is not fitted")
        x_test = np.asarray(x_test, dtype=float)
        if x_test.ndim != 2:
            raise ValueError("x_test must be a 2D array")
        if x_test.shape[0] == 0:
            return np.zeros((0,), dtype=int), np.zeros((0,), dtype=float)

        labels: Optional[np.ndarray] = None
        pred_error: Optional[Exception] = None
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
                    f"backend={self._backend_name or 'unknown'} stage=predict_test missing_method={missing} "
                    f"available_methods=[{method_list}] cause={cause}"
                ) from pred_error
            return labels, confidence

        confidence = np.ones((labels.shape[0],), dtype=float)
        return labels, confidence

    def get_diagnostics(self) -> Dict[str, object]:
        backend_name = self._backend_name
        backend_error = self._backend_resolve_error
        if not backend_name and not backend_error:
            try:
                backend = self._resolve_backend()
                backend_name = backend.name
            except Exception as exc:
                backend_error = f"{type(exc).__name__}: {exc}"
                self._backend_resolve_error = backend_error
        return {
            "backend_resolved_name": backend_name,
            "backend_resolve_error": backend_error,
            "backend_predict_methods": [
                name for name in self._backend_methods if name in {"predict", "predict_clusters", "transform"}
            ],
            "backend_methods": list(self._backend_methods),
        }

    def _instantiate_model(self, backend: _BackendSpec) -> Any:
        args = {
            "window_size": self.window_size,
            "number_of_clusters": self.num_clusters,
            "lambda_parameter": self.lambda_parameter,
            "beta": self.beta,
            "maxIters": self.max_iter,
            "threshold": self.threshold,
            "write_out_file": False,
            "prefix_string": "",
        }
        sig = inspect.signature(backend.cls)
        kwargs = {k: v for k, v in args.items() if k in sig.parameters}
        try:
            return backend.cls(**kwargs)
        except Exception as exc:
            raise TICCUnavailableError(
                f"Failed to initialize TICC backend={backend.name} with kwargs={sorted(kwargs.keys())}: {type(exc).__name__}: {exc}"
            ) from exc

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
            mask = labels == cluster_id
            centroids.append(np.mean(x_train[mask], axis=0))
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
        confidence = 1.0 / (1.0 + best_dist)
        return labels, confidence.astype(float)

    @staticmethod
    def _resolve_backend() -> _BackendSpec:
        fast_exc: Optional[Exception] = None
        try:
            from fast_ticc import TICC as FastTICC  # type: ignore

            return _BackendSpec(name="fast_ticc", cls=FastTICC)
        except Exception as exc:
            fast_exc = exc

        try:
            from TICC_solver import TICC as SolverTICC  # type: ignore

            return _BackendSpec(name="TICC_solver", cls=SolverTICC)
        except Exception as solver_exc:
            raise TICCUnavailableError(
                "No usable TICC backend found. Install `fast-ticc` (primary) or `TICC_solver` (legacy). "
                f"fast_ticc_error={type(fast_exc).__name__ if fast_exc else 'None'}: {fast_exc}; "
                f"ticc_solver_error={type(solver_exc).__name__}: {solver_exc}"
            ) from solver_exc
