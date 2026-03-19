from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

_OUTPUT_ARTIFACT_ROOTS = (
    "stepA",
    "stepB",
    "stepC",
    "stepD",
    "stepDprime",
    "stepD_prime",
    "stepE",
    "stepF",
    "audit",
)
_OUTPUT_ROOT_FILES = ("split_summary.json", "run_manifest.json", "reuse_signature.json")


def _clean_path(raw_path: str | Path | None) -> str:
    return str(raw_path or "").strip().replace("\\", "/")


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = path.as_posix()
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _extract_output_artifact_suffix(path: Path) -> Optional[Path]:
    parts = path.parts
    for idx, part in enumerate(parts):
        if part in _OUTPUT_ARTIFACT_ROOTS:
            return Path(*parts[idx:])
        if part in _OUTPUT_ROOT_FILES:
            return Path(part)
    return None


def normalize_output_artifact_path(raw_path: str | Path | None, *, output_root: Path) -> str:
    cleaned = _clean_path(raw_path)
    if not cleaned:
        return ""

    path = Path(cleaned)
    output_root = Path(output_root)

    if not path.is_absolute():
        suffix = _extract_output_artifact_suffix(path)
        return (suffix or path).as_posix()

    try:
        return path.relative_to(output_root).as_posix()
    except Exception:
        suffix = _extract_output_artifact_suffix(path)
        return (suffix or path).as_posix()


def resolve_output_artifact_path(
    raw_path: str | Path | None,
    *,
    canonical_output_root: Path,
    effective_output_root: Path | None = None,
) -> Path:
    cleaned = _clean_path(raw_path)
    if not cleaned:
        return Path()

    canonical_output_root = Path(canonical_output_root)
    effective_output_root = Path(effective_output_root) if effective_output_root is not None else None
    path = Path(cleaned)

    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
        suffix = _extract_output_artifact_suffix(path)
        if suffix is not None:
            candidates.append(canonical_output_root / suffix)
            if effective_output_root is not None:
                candidates.append(effective_output_root / suffix)
        if effective_output_root is not None:
            try:
                rel = path.relative_to(effective_output_root)
            except Exception:
                rel = None
            if rel is not None:
                candidates.insert(0, canonical_output_root / rel)
                candidates.append(effective_output_root / rel)
    else:
        candidates.append(canonical_output_root / path)
        suffix = _extract_output_artifact_suffix(path)
        if suffix is not None:
            candidates.append(canonical_output_root / suffix)
        if effective_output_root is not None:
            candidates.append(effective_output_root / path)
            if suffix is not None:
                candidates.append(effective_output_root / suffix)

    deduped = _dedupe_paths(candidates)
    for candidate in deduped:
        if candidate.exists():
            return candidate
    return deduped[0] if deduped else Path(cleaned)
