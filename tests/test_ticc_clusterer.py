from __future__ import annotations

import numpy as np
import pytest

import ai_core.clusterers.ticc_clusterer as tc_mod
from ai_core.clusterers.ticc_clusterer import TICCClusterer, TICCUnavailableError


class _NoBackend:
    @staticmethod
    def _resolve_backend():
        raise TICCUnavailableError("No usable TICC backend found")


def test_backend_unavailable_has_clear_error(monkeypatch) -> None:
    monkeypatch.setattr(tc_mod.TICCClusterer, "_resolve_backend", staticmethod(_NoBackend._resolve_backend))
    c = TICCClusterer(num_clusters=2, window_size=3, lambda_parameter=0.1, beta=10.0, max_iter=5, threshold=1e-3)
    with pytest.raises(TICCUnavailableError, match="No usable TICC backend"):
        c.fit_predict_train(np.ones((8, 2), dtype=float))


class _FakeBackendNoPredict:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit_predict(self, x):
        n = len(x)
        return np.array([i % 2 for i in range(n)], dtype=int)


def test_predict_fallback_assignment_when_predict_missing(monkeypatch) -> None:
    monkeypatch.setattr(tc_mod.TICCClusterer, "_resolve_backend", staticmethod(lambda: tc_mod._BackendSpec(name="fast_ticc", cls=_FakeBackendNoPredict)))
    c = TICCClusterer(num_clusters=2, window_size=3, lambda_parameter=0.1, beta=10.0, max_iter=5, threshold=1e-3)
    train = np.array([[0.0, 0.0], [10.0, 10.0], [0.1, -0.2], [9.8, 10.2]], dtype=float)
    train_labels = c.fit_predict_train(train)
    assert train_labels.shape == (4,)

    test = np.array([[0.2, 0.1], [9.9, 10.1]], dtype=float)
    labels, conf = c.predict_test(test)
    assert labels.shape == (2,)
    assert conf.shape == (2,)
    assert labels.tolist() in ([0, 1], [1, 0])


class _FakeBackendWithPredict:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit_predict(self, x):
        return np.zeros((len(x),), dtype=int)

    def predict(self, x):
        return np.ones((len(x),), dtype=int)


def test_predict_path_uses_backend_predict(monkeypatch) -> None:
    monkeypatch.setattr(tc_mod.TICCClusterer, "_resolve_backend", staticmethod(lambda: tc_mod._BackendSpec(name="fast_ticc", cls=_FakeBackendWithPredict)))
    c = TICCClusterer(num_clusters=2, window_size=3, lambda_parameter=0.1, beta=10.0, max_iter=5, threshold=1e-3)
    c.fit_predict_train(np.ones((6, 2), dtype=float))
    labels, conf = c.predict_test(np.ones((3, 2), dtype=float))
    assert labels.tolist() == [1, 1, 1]
    assert conf.tolist() == [1.0, 1.0, 1.0]
