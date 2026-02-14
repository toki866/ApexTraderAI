from __future__ import annotations

"""
backend package (compatibility shim)

This package exists to keep backward compatibility with older import paths like:
    from backend.backend_controller import BackendController, FullPipelineConfig

Canonical modules live under:
    ai_core.backend.*

If you are editing code, prefer importing from ai_core.backend.* directly.
"""

from .backend_controller import BackendController, FullPipelineConfig

__all__ = ["BackendController", "FullPipelineConfig"]
