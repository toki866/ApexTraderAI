"""Backward-compatible wrapper for the unified pipeline runner."""

from tools.run_pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())
