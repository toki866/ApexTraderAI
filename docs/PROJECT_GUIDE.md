# ApexTraderAI Project Guide (Complete Understanding)

> Purpose: this document is the **single-source operational guide** to understand, run, and safely modify this repository without guesswork.

## 1) Repository summary

### What problem this repo solves
ApexTraderAI provides a reproducible, headless trading-research pipeline that runs Step A→F for SOXL/SOXS-oriented workflows, including data preparation, model/feature generation, RL/MARL stages, and export of run artifacts for local storage, GitHub artifacts, and OneDrive backup.

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
| `.github/workflows/run_desktop_pipeline.yml` | Manual desktop pipeline workflow on self-hosted Windows runner | GitHub Actions `workflow_dispatch` | Workflow artifact `desktop-run-<run_id or github.run_id>` with console log, run logs, run zip |
| `.github/workflows/ci.yml` | CI compile/import smoke checks on Ubuntu | GitHub push / pull_request triggers | CI job logs; no packaged runtime artifacts |
| `scripts/run_all_local_then_copy.bat` | End-to-end local run wrapper: setup run dirs, run prep + pipeline, zip, OneDrive copy, snapshot zip | Workflow step “Run desktop BAT”; manual cmd/powershell execution | `C:\work\apex_work\runs\<run_id>\{data,output,logs}` + `run_<run_id>.zip` + OneDrive copy + `repo_<sha>_<run_id>.zip` snapshot |
| `scripts/bat_config.bat` | Default runtime variables (symbols, windows, mode, work root, flags) | Sourced by `run_all_local_then_copy.bat` / `doctor.bat` | Env vars only (no files directly) |
| `scripts/copy_run_to_onedrive.bat` | Re-copy existing local run directory to OneDrive | Manual cmd/powershell execution | OneDrive run mirror under `<runs_root>\<run_id>` |
| `scripts/doctor.bat` | Preflight diagnostics (git/python/torch/data files) | Manual cmd/powershell execution | `doctor_<run_id>.log` in run logs folder |
| `tools/prepare_data.py` | Download + normalize OHLCV CSVs from yfinance | `run_all_local_then_copy.bat`; manual CLI | `prices_<SYMBOL>.csv` under selected `--data-dir` |
| `tools/run_pipeline.py` | Headless Step A→F orchestrator with mode/agent flags | `run_all_local_then_copy.bat`; manual CLI | Step outputs under `--output-root` (default `output/`) |
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
  - `copy_to_onedrive` (boolean; currently reserved for future BAT integration)

### Runner requirements
- `runs-on: [self-hosted, windows]`
- Must have:
  - Git + Python environment with `requirements.txt`
  - Windows PowerShell (`powershell`) available
  - OneDrive path available via `%OneDrive%` **or** explicit env overrides used by BAT (`ONE_DRIVE_RUNS_ROOT`, `ONE_DRIVE_SNAPSHOTS_ROOT`)

### Step-by-step pipeline behavior
1. **Checkout** repo (`actions/checkout@v4`).
2. **Debug shells** (cmd): confirms `powershell`, optionally `pwsh`.
   - `pwsh` missing is treated as acceptable (`pwsh_not_found (OK)`).
3. **Run desktop BAT** in PowerShell with ExecutionPolicy Bypass:
   - invokes `cmd /c scripts\run_all_local_then_copy.bat`
   - captures full console output to `%RUNNER_TEMP%\run_all_local_then_copy_console.log`
4. **BAT internal execution** (`run_all_local_then_copy.bat`):
   - loads defaults from `scripts\bat_config.bat`
   - creates run folder structure under `%WORK_ROOT%\<run_id>`
   - logs commit and all executed commands
   - runs:
     - `python tools\prepare_data.py ... --data-dir <RUN_DIR>\data`
     - `python tools\run_pipeline.py ... --output-root <RUN_DIR>\output --data-dir <RUN_DIR>\data`
   - writes `DONE.txt` under output
   - creates zip `run_<run_id>.zip` containing output + logs (via `Compress-Archive`)
   - copies output/logs/zip to OneDrive run destination (`robocopy`)
   - creates repo snapshot zip in OneDrive snapshot root (`repo_<shortsha>_<run_id>.zip`)
   - emits `[OK] run_id=<run_id>` to stdout for workflow parsing
5. **Resolve latest run artifacts** (PowerShell):
   - extracts `run_id` from console log
   - resolves run folder under `C:\work\apex_work\runs`
   - sets outputs `log_glob` + `zip_path`
6. **Upload artifacts** (`actions/upload-artifact@v4`):
   - artifact name: `desktop-run-${{ run_id || github.run_id }}`
   - uploads:
     - runner console log
     - run logs glob
     - run zip

---

## 3.2 Local run (same pipeline, no GitHub Actions)

## Prerequisites
- Windows command prompt or PowerShell
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
python tools\run_pipeline.py --symbol SOXL --steps A,B,C,D,E,F --test-start 2022-01-03 --train-years 8 --test-months 3 --mode sim --output-root output --data-dir data --auto-prepare-data 0 --enable-mamba
```

> Note: BAT run writes outputs to per-run `C:\work\apex_work\runs\<run_id>\...`; direct Python defaults to repo `output/` and `data/` unless overridden.

---

## 4) Outputs and artifacts

## Local disk outputs (BAT default)
- Run root: `C:\work\apex_work\runs\<run_id>`
- Data: `...\data`
- Pipeline outputs: `...\output`
- Logs: `...\logs\run_<run_id>.log`
- Completion marker: `...\output\DONE.txt`
- Run zip: `...\run_<run_id>.zip` (contains `output/` and `logs/`)

## OneDrive outputs
Run destination resolution order:
1. `%ONE_DRIVE_RUNS_ROOT%\<run_id>`
2. `%OneDrive%\ApexTraderAI\runs\<run_id>`

Snapshot destination resolution order:
1. `%ONE_DRIVE_SNAPSHOTS_ROOT%`
2. `%OneDrive%\ApexTraderAI\repo_snapshots`

Snapshot zip naming:
- `repo_<short_commit_sha>_<run_id>.zip`

## GitHub Actions artifacts
- Artifact name pattern: `desktop-run-<run_id>` (fallback `desktop-run-<github.run_id>`)
- Uploaded paths:
  - `%RUNNER_TEMP%\run_all_local_then_copy_console.log`
  - `<run_dir>\logs\run_*.log`
  - `<run_dir>\run_<run_id>.zip`

---

## 5) Operational constraints and rules

## Do-not-edit / generated areas
Treat these as generated/runtime artifacts:
- `output/`, `outputs/`, `logs/`, `artifacts/`, `pdf/`
- `data/*.csv`
- Local run folders under `C:\work\apex_work\runs\<run_id>`
- OneDrive mirrors under `...\ApexTraderAI\runs\<run_id>`

## Naming conventions used by pipeline
- `run_id`: `yyyyMMdd_HHmmss`
- Run zip: `run_<run_id>.zip`
- Run log: `logs\run_<run_id>.log`
- Doctor log: `logs\doctor_<run_id>.log`
- Repo snapshot zip: `repo_<short_sha>_<run_id>.zip`

## Security rules
- Never commit secrets/credentials (`.env`, `*.key`, `*.pem`, `*.pfx`, tokens).
- Keep self-hosted runner access restricted (repo permissions/branch protection).
- Avoid adding broad auto-triggers for self-hosted execution; this repo currently keeps desktop run manual (`workflow_dispatch`).

## PowerShell policy notes
- Current workflow uses:
  - `shell: powershell -ExecutionPolicy Bypass -Command "{0}"`
- Preferred hardened style for new/edited workflow steps:
  - `powershell -NoProfile -ExecutionPolicy Bypass -Command "..."`
  - (NoProfile reduces profile-side side effects.)

---

## 6) Troubleshooting (known issues)

## 6.1 Runner service installed vs interactive user session
**Symptom:** OneDrive paths unavailable, `%OneDrive%` empty, copy failures.

- If runner runs as a service account without OneDrive sign-in, BAT fails resolving OneDrive destination.
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
- Existing workflow already applies `-ExecutionPolicy Bypass`; adding `-NoProfile` is recommended for future edits.

## 6.4 OneDrive path/permission/encoding/long-path issues
**Symptoms:** `robocopy` return code `>=8`, copy failures, path encoding problems.

- Check destination existence and permissions for chosen user.
- Keep run roots short (`C:\work\apex_work\runs`) to reduce path-length risk.
- Ensure UTF-8 console mode (`chcp 65001`) remains in BAT (already present).
- Inspect `logs\run_<run_id>.log` and failed `[CMD]`/`[RC]` pair.

## 6.5 Dirty git tree blocks local BAT run
**Symptom:** BAT fails before running pipeline when local tree has uncommitted changes.

- Outside GitHub Actions (`GITHUB_ACTIONS != true`), BAT checks `git status --porcelain` and exits if dirty.
- Commit or stash changes before rerunning.

---

## 7) TODO (needs confirmation)

1. **Workflow inputs are not currently wired into BAT variables.**
   - `run_desktop_pipeline.yml` sets `INPUT_*` env vars, but `scripts/run_all_local_then_copy.bat` / `scripts/bat_config.bat` do not consume them.
   - Confirm intended behavior:
     - Should `workflow_dispatch` inputs override `SYMBOLS`, dates, mode, etc. in BAT?
2. **`copy_to_onedrive` workflow input is marked reserved and appears unused.**
   - Confirm whether OneDrive copy should become optional via this input.
3. **Runner path assumptions are Windows-fixed (`C:\work\apex_work\runs`).**
   - Confirm whether path should be parameterized at workflow/job level for multi-runner environments.

---

## 8) Safe-change checklist (for agents/developers)

Before changing orchestration:
- Verify changes in both:
  - `.github/workflows/run_desktop_pipeline.yml`
  - `scripts/run_all_local_then_copy.bat`
- Preserve artifact discoverability:
  - `[OK] run_id=...` line in console
  - `logs\run_*.log` and `run_<run_id>.zip`
- Keep OneDrive fallback behavior (`ONE_DRIVE_*` overrides then `%OneDrive%`).
- Prefer `-NoProfile -ExecutionPolicy Bypass` PowerShell style.
- Re-run at least:
  - `python -m py_compile $(git ls-files '*.py')` (or CI)
  - a local dry run (`scripts\doctor.bat`) on Windows host

