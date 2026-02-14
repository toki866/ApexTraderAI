# ai_core/gui/_utils_dates.py
"""ai_core.gui._utils_dates

Compatibility date utilities (legacy shim).

This project previously duplicated small date/pandas helpers across multiple
modules (GUI tabs, benchmark scripts, etc.). That makes global symbol
extraction noisy and can mask truly dangerous duplicates.

Canonical implementations now live in:
    ai_core.gui._tab_utils

Keep importing from this module if you have older code; it simply re-exports
the canonical helpers.

NOTE:
- These helpers are intentionally underscored (internal API).
- Do not add new implementations here; add to _tab_utils and re-export.
"""

from __future__ import annotations

from ._tab_utils import _ensure_datetime, _numeric_columns, _parse_ymd

__all__ = [
    "_parse_ymd",
    "_ensure_datetime",
    "_numeric_columns",
]
