from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import List, Optional, Tuple


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


def _normalized_posix(value: Path | str) -> str:
    return str(value).replace("\\", "/")


def normalize_output_artifact_path(
    raw_path: str | Path,
    *,
    canonical_output_root: Path,
    resolved_output_root: Optional[Path] = None,
    prefer_relative: bool = False,
) -> str:
    """Normalize artifact paths to canonical output-root-relative or canonical absolute form."""
    raw = str(raw_path or "").strip()
    if not raw:
        return ""

    canonical_root = Path(canonical_output_root)
    resolved_root = Path(resolved_output_root) if resolved_output_root else canonical_root
    raw_norm = raw.replace("\\", "/")

    candidate = Path(raw)
    mapped: Optional[Path] = None

    for root in (resolved_root, canonical_root):
        try:
            if candidate.is_absolute():
                rel = candidate.resolve().relative_to(root.resolve())
                mapped = canonical_root / rel
                break
        except Exception:
            pass

    if mapped is None and candidate.is_absolute():
        lower = raw_norm.lower()
        marker = "/output/"
        idx = lower.find(marker)
        if idx >= 0:
            suffix = raw_norm[idx + len(marker):].lstrip("/")
            if suffix:
                mapped = canonical_root / PurePosixPath(suffix)

    if mapped is None:
        if raw_norm.startswith("runs/"):
            mapped = canonical_root / PurePosixPath(raw_norm)
        else:
            mapped = canonical_root / PurePosixPath(raw_norm.lstrip("./"))

    mapped_posix = _normalized_posix(mapped)
    if prefer_relative:
        try:
            return _normalized_posix(mapped.relative_to(canonical_root))
        except Exception:
            return mapped_posix
    return mapped_posix


def resolve_output_artifact_path(raw_path: str | Path, *, output_root: Path) -> Path:
    raw = str(raw_path or "").strip()
    if not raw:
        return Path(output_root)
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    return Path(output_root) / PurePosixPath(raw.replace("\\", "/"))
