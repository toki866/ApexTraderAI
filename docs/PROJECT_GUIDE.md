# ApexTraderAI Project Guide (Complete Understanding)

> Purpose: this document is the **single-source operational guide** to understand, run, and safely modify this repository without guesswork.

## 1) Repository summary

### What problem this repo solves
ApexTraderAI provides a reproducible, headless trading-research pipeline that runs standard Step A→F (A,B,C,DPRIME,E,F; D is retired) for SOXL/SOXS-oriented workflows, including data preparation, model/feature generation, RL/MARL stages, and export of run artifacts for local storage, GitHub artifacts, and OneDrive backup.

### High-level architecture
- **Pipeline orchestrator (Python):** `tools/run_pipeline.py` runs Step A–F with CLI control over symbol, date range, modes, and model toggles.
- **Data bootstrap (Python):** `tools/prepare_data.py` downloads/normalizes OHLCV into `prices_<SYMBOL>.csv`.
- **Desktop run wrapper (BAT):** `scripts/run_all_local_then_copy.bat` creates run folders, executes Python pipeline, zips outputs/logs, copies to OneDrive, and writes reproducibility logs.
- **Runner config defaults (BAT):** `scripts/bat_config.bat` centralizes symbols, date windows, work root, mode, and feature toggles.
- **Core services (Python):** `ai_core/services/step_*_service.py` implement each pipeline stage.
- **GitHub Actions integration:** `.github/workflows/run_desktop_pipeline.yml` dispatches the desktop pipeline on a self-hosted Windows runner and uploads resulting logs/zip as workflow artifacts.

### Main workflows
- **Run Desktop Pipeline (manual GitHub Actions):** `.github/workflows/run_desktop_pipeline.yml`
- **CI (push/PR checks):** `.github/workflows/ci.yml`
- **Local desktop batch run:** `scripts/run_all_local_then_copy.bat`

---

## 2) Real repository structure (auto-extracted)

The tree below was generated from the current repository state (depth-limited).

```text
.
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── run_desktop_pipeline.yml
├── ai_core/
│   ├── backend/
│   ├── config/
│   ├── features/
│   ├── gui/
│   ├── metrics/
│   ├── models/
│   ├── rl/
│   ├── services/
│   ├── types/
│   └── utils/
├── archive/
│   └── patch_notes/
├── config/
│   └── app_config.yaml
├── data/
│   └── .gitkeep
├── docs/
│   ├── pack_tools/
│   ├── bat_usage.md
│   ├── iphone_run.md
│   ├── README_ja.md
│   ├── self_hosted_runner_windows.md
│   └── unnecessary_files_audit_ja.md
├── engine/
├── scripts/
│   ├── bat_config.bat
│   ├── copy_run_to_onedrive.bat
│   ├── doctor.bat
│   └── run_all_local_then_copy.bat
├── specs/
├── tools/
│   ├── _payload_stepE/
│   ├── pack_tools/
│   ├── prepare_data.py
│   └── run_pipeline.py
├── .gitignore
├── README.md
└── requirements.txt
```

### Key files map

| File path | Role | Called by | Outputs produced |
|---|---|---|---|
| `.github/workflows/run_desktop_pipeline.yml` | Manual desktop pipeline workflow on self-hosted Windows runner | GitHub Actions `workflow_dispatch` | Workflow artifact `desktop-run-<run_id or github.run_id>` with console log, run logs, and canonical logs |
| `.github/workflows/ci.yml` | CI compile/import smoke checks on Ubuntu | GitHub push / pull_request triggers | CI job logs; no packaged runtime artifacts |
| `scripts/run_all_local_then_copy.bat` | End-to-end local run wrapper: setup run dirs, run prep + pipeline, export canonical output ZIP to OneDrive, snapshot repo | Workflow step “Run desktop BAT”; manual cmd/powershell execution | `C:\work\apex_work\runs\<run_id>\{data,logs}` + canonical output under `C:\work\apex_work\output\<mode>\<symbol>\<test_start_date>` + OneDrive `output_YYYYMMDD_NNN.zip` + `repo_<sha>_<run_id>.zip` snapshot |
| `scripts/bat_config.bat` | Default runtime variables (symbols, windows, mode, work root, flags) | Sourced by `run_all_local_then_copy.bat` / `doctor.bat` | Env vars only (no files directly) |
| `scripts/copy_run_to_onedrive.bat` | Re-export an existing canonical output (or run directory) to OneDrive | Manual cmd/powershell execution | OneDrive `runs\export\output_YYYYMMDD_NNN.zip` |
| `scripts/doctor.bat` | Preflight diagnostics (git/python/torch/data files) | Manual cmd/powershell execution | `doctor_<run_id>.log` in run logs folder |
| `tools/prepare_data.py` | Download + normalize OHLCV CSVs from yfinance | `run_all_local_then_copy.bat`; manual CLI | `prices_<SYMBOL>.csv` under selected `--data-dir` |
| `tools/run_pipeline.py` | Headless Step A→F orchestrator (standard: A,B,C,DPRIME,E,F) with mode/agent flags | `run_all_local_then_copy.bat`; manual CLI | Step outputs under `--output-root` (default `output/`) |
| `config/app_config.yaml` | Default app config roots and symbols | Loaded via `ai_core/config/app_config.py` | N/A (configuration source) |
| `ai_core/services/step_a_service.py` ... `step_f_service.py` | Step implementations used by headless orchestration | `tools/run_pipeline.py` | Step-specific CSV/model outputs under `output/` subtree |
| `ai_core/utils/paths.py` | Repo-root path resolution helpers | Imported by pipeline/config code | N/A (utility behavior) |

---

## 3) Execution flows (end-to-end)

## 3.1 GitHub Actions → self-hosted runner → desktop execution

### Workflow and trigger
- Workflow file: `.github/workflows/run_desktop_pipeline.yml`
- Trigger: `workflow_dispatch` (manual)
- Inputs:
  - `mode` (choice: `sim`/`live`/`display`)
  - `symbols` (comma-separated)
  - `start`, `end`, `test_start`
  - `train_years`, `test_months`
  - `copy_to_onedrive` (boolean; exports canonical output to OneDrive as `output_YYYYMMDD_NNN.zip` when enabled)

### Runner requirements
- `runs-on: [self-hosted, windows]`
- Must have:
  - Git + Python environment with `requirements.txt`
  - Windows PowerShell (`powershell`) available
  - OneDrive path available via `%OneDrive%` **or** explicit env overrides used by BAT (`ONE_DRIVE_RUNS_ROOT`, `ONE_DRIVE_SNAPSHOTS_ROOT`)
- GitHub Actions execution uses `%GITHUB_WORKSPACE%` as the repo root; the old `C:\work` / OneDrive-side repo clone is no longer required for workflow execution.

### Step-by-step pipeline behavior
1. **Checkout** repo into `%GITHUB_WORKSPACE%` (`actions/checkout@v4`).
2. **Prepare workspace repo** (PowerShell): sets `REPO_ROOT=%GITHUB_WORKSPACE%`, records `whoami`, explicitly runs `git config --global --add safe.directory "%GITHUB_WORKSPACE%"`, prints `git config --global --get-all safe.directory` / `git config --global --list`, and avoids cloning into a separate fixed path.
3. **Debug workspace context** (PowerShell): logs `whoami`, `%GITHUB_WORKSPACE%`, `PWD`, `REPO_ROOT`, `git config --global --get-all safe.directory`, `git config --global --list`, and `git rev-parse --show-toplevel` so the effective execution root is visible in Actions logs.
4. **Debug shells** (cmd): confirms `powershell`, optionally `pwsh`.
   - `pwsh` missing is treated as acceptable (`pwsh_not_found (OK)`).
5. **Run desktop BAT** in PowerShell (`-NoProfile -ExecutionPolicy Bypass`):
   - invokes `cmd /c scripts\run_all_local_then_copy.bat`
   - captures full console output to `%RUNNER_TEMP%\run_all_local_then_copy_console.log`
   - BAT logs non-fatal python diagnostics before execution (`where python`, `python --version`)
   - BAT normalizes malformed escaped quotes in `PYTHON_EXE` (e.g., `\"python\"` -> `python`)
   - **BAT internal execution** (`run_all_local_then_copy.bat`):
   - loads defaults from `scripts\bat_config.bat`
   - creates run folder structure under `%WORK_ROOT%\<run_id>`
   - logs commit and all executed commands
   - runs:
     - `python tools\run_with_python.py tools\prepare_data.py ... --data-dir <RUN_DIR>\data`
     - `python tools\run_with_python.py tools\run_pipeline.py ... --output-root <canonical_output_root> --data-dir <RUN_DIR>\data`
   - writes `DONE.txt` under canonical output
   - does **not** create a local run ZIP
   - creates only a canonical-output ZIP in OneDrive export destination (`output_YYYYMMDD_NNN.zip`)
   - creates repo snapshot zip in OneDrive snapshot root (`repo_<shortsha>_<run_id>.zip`)
   - emits `[OK] run_id=<run_id>` to stdout for workflow parsing
6. **Resolve latest run artifacts** (PowerShell):
   - extracts `run_id` from console log
   - resolves run folder under `%WORK_ROOT%` (workflow env default: `C:\work\apex_work\runs`)
   - sets outputs `log_glob` + `zip_path`
7. **Stage + upload artifacts** (`actions/upload-artifact@v4`):
   - artifact name: `desktop-run-${{ run_id || github.run_id }}`
   - workflow first copies required files to `%GITHUB_WORKSPACE%\temp\desktop_artifacts`
   - upload step uses a single rooted path (`%GITHUB_WORKSPACE%\temp\desktop_artifacts\**`) to avoid mixed-root `rootDirectory` failures

---

## Pipeline spec note (D retired, D-prime canonical)
- Standard pipeline steps are **A,B,C,DPRIME,E,F**.
- **Step D is retired** in standard operation and should not be selected for routine runs.
- `DPRIME` is the **StepD superior version** and independently performs chart-compression (Phase2) plus RL state generation.
- Internal responsibility separation for DPRIME is tracked in `specs/stepA_stepF_responsibility_reorg_ja.md` as **DPrimeCluster (cluster-only state)** and **DPrimeRL (RL input state)** while keeping external workflow/CLI compatibility on the single `DPRIME` step name.
- `StepE` consumes `DPRIME` state as its primary observation input (plus embeddings when required by the selected policy/model mode), and the standard pipeline now treats StepE as PPO-only.
- StepE agent parallelism remains operator-controlled (`--stepE-max-parallel-agents`, capped at 2). Recommended values are `sim=2` and `live=1` first, raising live to `2` only after confirming device headroom. Inspect `requested_parallel_agents`, `max_parallel_agents`, `effective_parallel_agents`, `parallelism_warning`, `merge_inputs_cache_hit`, and `merge_cache_key` in StepE summary/audit/log outputs.
- There are two distinct "3-month" windows:
  1. Fixed-length compression window in `DPRIME`/Phase2
  2. Lookback observation window in `StepB`

## 3.2 Local run (same pipeline, no GitHub Actions)

## Prerequisites
- Windows command prompt or PowerShell
- Local manual clone is assumed at `C:\work\apex-trader-ai` (the BAT/PS1 scripts themselves resolve the repo root relative to their own location, so another non-OneDrive path also works).
- Python with dependencies installed:

```bat
pip install -r requirements.txt
```

- Optional: set/override runtime defaults in `scripts\bat_config.bat` (or with env vars).

## Local commands (cmd)

```bat
scripts\doctor.bat
scripts\run_all_local_then_copy.bat
```

## Local commands (PowerShell)

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command "cmd /c scripts\doctor.bat"
powershell -NoProfile -ExecutionPolicy Bypass -Command "cmd /c scripts\run_all_local_then_copy.bat"
```

## Equivalent direct Python invocation (advanced, bypass BAT)

```bat
python tools\prepare_data.py --symbols SOXL,SOXS --start 2014-01-01 --end 2022-03-31 --force --data-dir data
python tools\run_pipeline.py --symbol SOXL --steps A,B,C,DPRIME,E,F --test-start 2022-01-03 --train-years 8 --test-months 3 --mode sim --output-root output --data-dir data --auto-prepare-data 0 --enable-mamba
```

> Note: GitHub Actions resolves the source repo from `%GITHUB_WORKSPACE%`, while local BAT/PS1 execution assumes a non-OneDrive clone such as `C:\work\apex-trader-ai`. BAT outputs still default to `C:\work\apex_work\runs\<run_id>\...`; direct Python defaults to repo `output/` and `data/` unless overridden.

---

## 4) Outputs and artifacts

## Local disk outputs (BAT default)
- Run root: `C:\work\apex_work\runs\<run_id>`
- Data: `...\data`
- Pipeline outputs: canonical output is written directly under `C:\work\apex_work\output\<mode>\<primarySymbol>\<test_start_date>`
- Canonical output root: `C:\work\apex_work\output\<mode>\<primarySymbol>\<test_start_date>`
- Canonical output logs: `C:\work\apex_work\output\<mode>\<primarySymbol>\<test_start_date>\logs\`
  - includes copied run/console/error/step-exec/diagnostics logs plus `logs_manifest.json`
- DPrime audit outputs now include:
  - `stepDprime_base_meta_<SYMBOL>.json`
  - `stepDprime_profile_summary_<PROFILE>_<SYMBOL>.json`
  - READY/FAILED markers under `stepDprime\<mode>\pipeline_markers\`
- StepE summaries now retain training/runtime config such as policy kind, PPO parameters, device, seed, and DPrime embedding usage.
- Logs: `...\logs\run_<run_id>.log`
- Completion marker: `<canonical output>\DONE.txt`
- Local ZIP archives are no longer generated; canonical output remains the only local source of truth.
- Date/sequence-based generation management belongs only to the OneDrive export artifact name, not to the local canonical output path.

## OneDrive outputs
Run destination resolution order:
1. `%ONE_DRIVE_RUNS_ROOT%\export`
2. `%OneDrive%\ApexTraderAI\runs\export`

Workflow export payload (when `copy_to_onedrive=true`):
- canonical-output ZIP only: `output_YYYYMMDD_NNN.zip`

The workflow no longer mirrors the raw canonical `output/` directory to OneDrive. Instead, it treats `C:\work\apex_work\output\<mode>\<symbol>\<test_start_date>` as the only local source of truth, scans the shared export directory for existing `output_YYYYMMDD_NNN.zip` files, assigns the next `NNN` for the local date, writes audit metadata locally (`latest_run_info.json` / `onedrive_zip_export_audit.json`), and stores only the canonical-output ZIP in OneDrive.

Snapshot destination resolution order:
1. `%ONE_DRIVE_SNAPSHOTS_ROOT%`
2. `%OneDrive%\ApexTraderAI\repo_snapshots`

Snapshot zip naming:
- `repo_<short_commit_sha>_<run_id>.zip`

## GitHub Actions artifacts
- Artifact name pattern: `desktop-run-<run_id>` (fallback `desktop-run-<github.run_id>`)
- Staging root: `%GITHUB_WORKSPACE%\temp\desktop_artifacts`
- Uploaded path: `%GITHUB_WORKSPACE%\temp\desktop_artifacts\**`
- The workflow now copies distributed logs into the canonical output root before staging/upload so the output tree alone is enough for failure triage.
- Staged files:
  - `run_all_local_then_copy_console.log`
  - `run_<run_id>.log`
  - `canonical_logs/` mirror from `output/<mode>/<primarySymbol>/<test_start_date>/logs/`
  - `logs_manifest.json`
- Additional publication branch for evaluation snapshots: `output-latest`
  - stores only the latest snapshot payload (`output_latest.zip`, checksum, run metadata, and lightweight eval/index/report files)
  - branch is force-pushed as latest-only distribution and is intentionally separated from `main` source history

---

## 5) Operational constraints and rules

## Do-not-edit / generated areas
Treat these as generated/runtime artifacts:
- `output/`, `outputs/`, `logs/`, `artifacts/`, `pdf/`
- `data/*.csv`
- Local run folders under `C:\work\apex_work\runs\<run_id>`
- OneDrive lightweight exports under `...\ApexTraderAI\runs\export`

## Naming conventions used by pipeline
- `run_id`: `yyyyMMdd_HHmmss`
- OneDrive export zip: `output_YYYYMMDD_NNN.zip`
- Run log: `logs\run_<run_id>.log`
- Doctor log: `logs\doctor_<run_id>.log`
- Repo snapshot zip: `repo_<short_sha>_<run_id>.zip`

## Security rules
- Never commit secrets/credentials (`.env`, `*.key`, `*.pem`, `*.pfx`, tokens).
- Keep self-hosted runner access restricted (repo permissions/branch protection).
- Avoid adding broad auto-triggers for self-hosted execution; this repo currently keeps desktop run manual (`workflow_dispatch`).

## PowerShell policy notes
- Current workflow uses:
  - `shell: powershell -NoProfile -ExecutionPolicy Bypass -Command "{0}"`
- Preferred hardened style for new/edited workflow steps:
  - `powershell -NoProfile -ExecutionPolicy Bypass -Command "..."`
  - (NoProfile reduces profile-side side effects.)

---

## 6) Troubleshooting (known issues)

## 6.1 Runner service installed vs interactive user session
**Symptom:** OneDrive paths unavailable, `%OneDrive%` empty, export warnings.

- If runner runs as a service account without OneDrive sign-in, the workflow should still complete local canonical output generation and only downgrade the OneDrive export to a warning.
- Mitigation:
  - Run under signed-in desktop user context, or
  - set explicit `ONE_DRIVE_RUNS_ROOT` / `ONE_DRIVE_SNAPSHOTS_ROOT` to accessible paths.

## 6.2 `pwsh` not found (PowerShell 7)
**Symptom:** `where pwsh` fails.

- Current workflow already tolerates this and continues with Windows PowerShell (`powershell`).
- No change required unless workflow is rewritten to require `pwsh`.

## 6.3 ExecutionPolicy errors
**Symptom:** script execution blocked by policy.

- Mitigation for workflow/local automation:
  - use `powershell -NoProfile -ExecutionPolicy Bypass -Command "..."`
- Existing workflow already applies `-NoProfile -ExecutionPolicy Bypass`.

## 6.4 OneDrive path/permission/encoding/long-path issues
**Symptoms:** OneDrive export warnings, `output_YYYYMMDD_NNN.zip` creation/copy failures, path encoding problems.

- Check destination existence and permissions for chosen user.
- Keep run roots short (`C:\work\apex_work\runs`) to reduce path-length risk.
- Ensure UTF-8 console mode (`chcp 65001`) remains in BAT (already present).
- Inspect `logs\run_<run_id>.log`, workflow summary, and `ONE_TAP_ERROR_REPORT.txt` for the recorded export warning reason.

## 6.5 Dirty git tree blocks local BAT run
**Symptom:** BAT fails before running pipeline when local tree has uncommitted changes.

- Outside GitHub Actions (`GITHUB_ACTIONS != true`), BAT checks `git status --porcelain` and exits if dirty.
- Commit or stash changes before rerunning.

## 6.6 `fatal: detected dubious ownership in repository`
**Symptom:** after `actions/checkout`, later `git` commands fail on the Windows self-hosted runner with `detected dubious ownership in repository`, often because the `_work\ApexTraderAI\ApexTraderAI` directory is still owned by an older service account such as `NETWORK SERVICE` (`S-1-5-20`) while the current runner service uses a different login such as `.\becky`.

- Current workflow mitigation:
  - immediately after checkout, log `whoami`
  - add `%GITHUB_WORKSPACE%` to global `safe.directory`
  - print the global `safe.directory` list and `git config --global --list`
  - persist current user / workspace / safe.directory diagnostics into `ONE_TAP_ERROR_REPORT.txt`
- Operational follow-up when the warning keeps recurring:
  - after changing the runner service account, stop the runner and delete the stale `_work` repository tree once so Git can recreate it under the new owner
  - if ownership mismatch persists, remove `C:\work\actions-runner\_work\ApexTraderAI\ApexTraderAI` and rerun checkout so the workspace is regenerated under the active service account
  - confirm the Actions logs now show the expected current user and the workspace path in `safe.directory`

---

## 7) TODO (needs confirmation)

1. **Workflow inputs are not currently wired into BAT variables.**
   - `run_desktop_pipeline.yml` sets `INPUT_*` env vars, but `scripts/run_all_local_then_copy.bat` / `scripts/bat_config.bat` do not consume them.
   - Confirm intended behavior:
     - Should `workflow_dispatch` inputs override `SYMBOLS`, dates, mode, etc. in BAT?
2. **Workflow output roots are centralized via env vars (`WORK_ROOT`, `CANONICAL_OUTPUT_ROOT`, `SESSION_LOG_ROOT`).**
   - Confirm whether multi-runner environments need different values from the current Windows defaults.

---

## 8) Safe-change checklist (for agents/developers)

Before changing orchestration:
- Verify changes in both:
  - `.github/workflows/run_desktop_pipeline.yml`
  - `scripts/run_all_local_then_copy.bat`
- Preserve artifact discoverability:
  - `[OK] run_id=...` line in console
  - `logs\run_*.log` and canonical logs in the staged artifact
- Keep OneDrive fallback behavior (`ONE_DRIVE_*` overrides then `%OneDrive%`).
- Prefer `-NoProfile -ExecutionPolicy Bypass` PowerShell style.
- Re-run at least:
  - `python -m py_compile $(git ls-files '*.py')` (or CI)
  - a local dry run (`scripts\doctor.bat`) on Windows host
