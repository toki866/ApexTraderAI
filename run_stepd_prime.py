"""Backward-compatible wrapper for tools/run_stepd_prime.py."""

from tools.run_stepd_prime import main


if __name__ == "__main__":
    raise SystemExit(main())
