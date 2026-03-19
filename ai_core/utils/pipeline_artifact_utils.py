from __future__ import annotations

from pathlib import Path
from typing import List, Tuple


def stepdprime_root_candidates(output_root: Path, mode: str) -> List[Path]:
    base = Path(output_root)
    mode_name = str(mode or "sim").strip().lower()
    return [
        base / "stepDprime" / mode_name,
        base / "stepD_prime" / mode_name,
    ]


def resolve_stepdprime_root(output_root: Path, mode: str) -> Tuple[Path, List[str], bool]:
    candidates = stepdprime_root_candidates(output_root=output_root, mode=mode)
    canonical = candidates[0]
    legacy = candidates[1]
    warnings: List[str] = []
    legacy_read = False

    canonical_exists = canonical.exists()
    legacy_exists = legacy.exists()
    if canonical_exists and legacy_exists:
        warnings.append(f"legacy_duplicate_detected:{legacy}")
        return canonical, warnings, False
    if canonical_exists:
        return canonical, warnings, False
    if legacy_exists:
        warnings.append(f"legacy_read:{legacy}")
        legacy_read = True
        return legacy, warnings, legacy_read
    return canonical, warnings, False
