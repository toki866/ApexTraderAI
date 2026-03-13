from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

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
    - `fast_ticc.TICC`
    - `TICC_solver.TICC`
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

    def fit_predict_train(self, x_train: np.ndarray) -> np.ndarray:
        backend = self._resolve_backend()
        x_train = np.asarray(x_train, dtype=float)
        if x_train.ndim != 2 or x_train.shape[0] == 0:
            raise ValueError("x_train must be a non-empty 2D array")

        if backend.name == "fast_ticc":
            model = backend.cls(
                window_size=self.window_size,
                number_of_clusters=self.num_clusters,
                lambda_parameter=self.lambda_parameter,
                beta=self.beta,
                maxIters=self.max_iter,
                threshold=self.threshold,
            )
            labels = np.asarray(model.fit_predict(x_train), dtype=int)
            self._model = model
            return labels

        if backend.name == "TICC_solver":
            model = backend.cls(
                window_size=self.window_size,
                number_of_clusters=self.num_clusters,
                lambda_parameter=self.lambda_parameter,
                beta=self.beta,
                maxIters=self.max_iter,
                threshold=self.threshold,
                write_out_file=False,
                prefix_string="",
            )
            labels = np.asarray(model.fit(input_file=x_train), dtype=int)
            self._model = model
            return labels

        raise TICCUnavailableError("Unsupported TICC backend")

    def predict_test(self, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._model is None:
            raise RuntimeError("TICC model is not fitted")
        x_test = np.asarray(x_test, dtype=float)
        if x_test.ndim != 2:
            raise ValueError("x_test must be a 2D array")
        if x_test.shape[0] == 0:
            return np.zeros((0,), dtype=int), np.zeros((0,), dtype=float)

        if hasattr(self._model, "predict"):
            labels = np.asarray(self._model.predict(x_test), dtype=int)
        elif hasattr(self._model, "predict_clusters"):
            labels = np.asarray(self._model.predict_clusters(x_test), dtype=int)
        else:
            raise TICCUnavailableError("TICC backend does not expose test-time prediction")

        confidence = np.ones((labels.shape[0],), dtype=float)
        return labels, confidence

    @staticmethod
    def _resolve_backend() -> _BackendSpec:
        try:
            from fast_ticc import TICC as FastTICC  # type: ignore

            return _BackendSpec(name="fast_ticc", cls=FastTICC)
        except Exception:
            pass

        try:
            from TICC_solver import TICC as SolverTICC  # type: ignore

            return _BackendSpec(name="TICC_solver", cls=SolverTICC)
        except Exception as exc:
            raise TICCUnavailableError("No usable TICC backend found. Install `fast_ticc` or `TICC_solver`.") from exc
