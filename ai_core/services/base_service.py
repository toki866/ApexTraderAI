# -*- coding: utf-8 -*-
"""ai_core.services.base_service

Minimal shared base class for service layer.

Why this file exists
--------------------
Some versions of Step*Service modules import BaseService from
``ai_core.services.base_service``. During automated refactors, this module
may go missing, causing ``ModuleNotFoundError`` and blocking the whole pipeline.

This BaseService intentionally stays lightweight:
- stores AppConfig (or config-like object)
- provides a simple logger helper
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import logging


@dataclass
class BaseService:
    """Base class for Step services."""

    app_config: Any

    def __post_init__(self) -> None:
        self._logger: logging.Logger = logging.getLogger(self.__class__.__name__)

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    def get(self, name: str, default: Any = None) -> Any:
        """Helper to read optional attributes from app_config safely."""
        return getattr(self.app_config, name, default)
