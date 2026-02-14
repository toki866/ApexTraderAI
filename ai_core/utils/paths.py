from __future__ import annotations

from pathlib import Path


def get_repo_root(start: str | Path | None = None) -> Path:
    """Find repository root by traversing parents looking for `.git` or `pyproject.toml`."""
    current = Path(start).resolve() if start else Path(__file__).resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if (candidate / ".git").exists() or (candidate / "pyproject.toml").exists():
            return candidate

    raise FileNotFoundError("Could not find repo root (.git or pyproject.toml).")


def resolve_repo_path(rel: str | Path) -> Path:
    """Resolve relative path under repository root."""
    rel_path = Path(rel)
    if rel_path.is_absolute():
        return rel_path
    return get_repo_root() / rel_path


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
