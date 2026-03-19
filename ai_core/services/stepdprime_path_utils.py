from __future__ import annotations

from pathlib import Path
from typing import List, Optional


def _normalize_mode(mode: str) -> str:
    value = str(mode or "sim").strip().lower()
    if value in {"ops", "op", "prod", "production", "real"}:
        return "live"
    return value or "sim"


def stepdprime_read_candidates(out_root: Path, mode: str) -> List[Path]:
    base = Path(out_root)
    mode_name = _normalize_mode(mode)
    return [
        base / "stepDprime" / mode_name,
        base / "stepD_prime" / mode_name,
    ]


def resolve_stepdprime_dir(
    out_root: Path,
    mode: str,
    explicit_root: Optional[str | Path] = None,
    for_write: bool = False,
) -> Path:
    if explicit_root:
        return Path(explicit_root)

    candidates = stepdprime_read_candidates(out_root=Path(out_root), mode=mode)
    canonical = candidates[0]
    if for_write:
        return canonical

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return canonical
