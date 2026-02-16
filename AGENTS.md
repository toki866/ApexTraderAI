# AGENTS.md (Codex quick guide)

This repository's **single source of truth** for safe modifications is:

- `docs/PROJECT_GUIDE.md`

Use it first for:
- architecture and entrypoints
- GitHub Actions desktop pipeline flow
- local reproduction commands
- output/ZIP/artifact paths
- operational constraints (do-not-edit, naming, security, PowerShell policy)
- troubleshooting for Windows self-hosted runner issues

## Fast rules (must follow)
- Treat `output/`, `outputs/`, `logs/`, `artifacts/`, and generated `data/*.csv` as generated/runtime artifacts (do not commit them).
- Keep pipeline entrypoints unchanged unless intentionally modifying orchestration:
  - `.github/workflows/run_desktop_pipeline.yml`
  - `scripts/run_all_local_then_copy.bat`
  - `tools/run_pipeline.py`
- In workflows, prefer PowerShell invocation with:
  - `powershell -NoProfile -ExecutionPolicy Bypass ...`
- Never commit tokens/secrets (e.g., `.env`, key/pem/pfx files).

If anything in the repo conflicts with this quick guide, follow `docs/PROJECT_GUIDE.md` and update both docs in the same PR.
