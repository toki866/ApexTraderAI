# -*- coding: utf-8 -*-
"""ai_core.config.train_config

Compatibility shim for training config classes.

Background
----------
Some code paths expect to import these names from ``ai_core.config.train_config``:
- XSRTrainConfig
- WaveletMambaTrainConfig
- FEDformerTrainConfig

If your project defines these elsewhere (e.g. ``ai_core.types.step_b_types``),
this module re-exports them when possible. Otherwise it provides permissive
fallback classes that accept **kwargs and store them as attributes.

This avoids import-time failures and lets the pipeline proceed to the next
runtime errors, where richer validation can occur.
"""

from __future__ import annotations

from typing import Any, Dict


def _mk_fallback(name: str):
    class _Cfg:
        __name__ = name

        def __init__(self, **kwargs: Any) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)

        def to_dict(self) -> Dict[str, Any]:
            return dict(self.__dict__)

        def __repr__(self) -> str:
            keys = ", ".join(sorted(self.__dict__.keys()))
            return f"{name}({keys})"

    _Cfg.__qualname__ = name
    return _Cfg


try:
    from ai_core.types.step_b_types import (  # type: ignore
        XSRTrainConfig as _XSRTrainConfig,
        WaveletMambaTrainConfig as _WaveletMambaTrainConfig,
        FEDformerTrainConfig as _FEDformerTrainConfig,
    )

    XSRTrainConfig = _XSRTrainConfig
    WaveletMambaTrainConfig = _WaveletMambaTrainConfig
    FEDformerTrainConfig = _FEDformerTrainConfig

except Exception:
    XSRTrainConfig = _mk_fallback("XSRTrainConfig")
    WaveletMambaTrainConfig = _mk_fallback("WaveletMambaTrainConfig")
    FEDformerTrainConfig = _mk_fallback("FEDformerTrainConfig")
