# ONE_TAP Public Run Report

- run_id: 22810456589
- sha: ff026bb47be09ed5304d5c87c9c23faed5fef24c
- mode: sim
- symbols: SOXL,SOXS
- test_start: 2022-01-03
- train_years: 8
- test_months: 3
- output_root: C:/work/apex_work/runs/gh22810456589_att1_sim_20260308_093151_ff026bb/output

## StepE metrics
not found

## StepF metrics
| step  | mode | source_csv | status    |
| ----- | ---- | ---------- | --------- |
| StepF | sim  | not found  | not found |

## Key CSV files
- No target CSV files found

## Error summary
```text
﻿=== ONE TAP ERROR REPORT ===
timestamp=2026-03-08 09:31:45 +09:00
repo=toki866/ApexTraderAI
ref=refs/heads/main
sha=ff026bb47be09ed5304d5c87c9c23faed5fef24c
run_id=22810456589
run_attempt=1

[STEP] initialize_one_tap_report
[OK] report initialized
[PATH] C:\Users\becky\OneDrive\デスクトップ\Python\apex-trader-ai\actions-runner\_work\ApexTraderAI\ApexTraderAI\temp\ONE_TAP_ERROR_REPORT.txt
[INFO] prep_repo_outcome=success
[INFO] REPO_ROOT=C:\work\apex_repo_cache\ApexTraderAI

[INFO] HEAD=ff026bb

## GPU_DIAGNOSTICS
torch 2.10.0+cu126
cuda_available True
gpu NVIDIA GeForce RTX 3080 Ti
timestamp, utilization.gpu [%], utilization.memory [%], memory.used [MiB]
2026/03/08 09:31:48.762, 0 %, 7 %, 686 MiB

[STEP] compute_steps
[INFO] computed_steps=A,B,C,DPRIME,E,F
[INFO] steps_source=toggle
[INFO] effective_skip_stepe=0

[STEP] compute_derived_date_window
[INFO] test_start_date=2022-01-03
[INFO] data_start_date=2013-12-27
[INFO] data_end_date=2022-04-10

[STEP] prepare_environment
[PREPARE_ENV] begin
[PREPARE_ENV] REPO_ROOT=C:\work\apex_repo_cache\ApexTraderAI
[PREPARE_ENV] APEX_RUN_ID=gh22810456589_att1_sim_20260308_093151_ff026bb
[PREPARE_ENV] WSL_DISTRO=Ubuntu
[PREPARE_ENV] WSL_PYTHON=/home/becky/miniforge3/envs/mamba_cuda/bin/python
[PREPARE_ENV] WORK_ROOT=C:\work\apex_work\runs
[PREPARE_ENV] SYMBOLS=SOXL,SOXS
[PREPARE_ENV] winRunDir=C:\work\apex_work\runs\gh22810456589_att1_sim_20260308_093151_ff026bb winDataDir=C:\work\apex_work\runs\gh22810456589_att1_sim_20260308_093151_ff026bb\data winOutputDir=C:\work\apex_work\runs\gh22810456589_att1_sim_20260308_093151_ff026bb\output winLogDir=C:\work\apex_work\runs\gh22810456589_att1_sim_20260308_093151_ff026bb\logs
[PREPARE_ENV] create_dirs begin
[PREPARE_ENV] create_dirs ok
[PREPARE_ENV] console_log=C:\Users\becky\OneDrive\デスクトップ\Python\apex-trader-ai\actions-runner\_work\_temp\run_all_local_then_copy_console.log run_log=C:\work\apex_work\runs\gh22810456589_att1_sim_20260308_093151_ff026bb\logs\run_gh22810456589_att1_sim_20260308_093151_ff026bb.log
[PREPARE_ENV] wslpath_begin var=WSL_REPO_ROOT cmd='wsl.exe -d Ubuntu wslpath -u C:\work\apex_repo_cache\ApexTraderAI'
[PREPARE_ENV] wslpath_result var=WSL_REPO_ROOT rc=1 result='wslpath: C:workapex_repo_cacheApexTraderAI' raw='wslpath: C:workapex_repo_cacheApexTraderAI'
[NG] prepare_environment wslpath_invalid var=WSL_REPO_ROOT win='C:\work\apex_repo_cache\ApexTraderAI' wsl='wslpath: C:workapex_repo_cacheApexTraderAI' rc=1 distro='Ubuntu'

[STEP] append_error_tails
timestamp=2026-03-08 09:32:00 +09:00
[INFO] apex_run_id=gh22810456589_att1_sim_20260308_093151_ff026bb
---- python_env ----
C:\Users\becky\AppData\Local\Programs\Python\Python310\python.exe
Python 3.10.6
C:\Users\becky\AppData\Local\Programs\Python\Python310\python.exe
[INFO] console_log=C:\Users\becky\OneDrive\デスクトップ\Python\apex-trader-ai\actions-runner\_work\_temp\run_all_local_then_copy_console.log
---- console_log_tail (last 350 lines) ----
[INFO] work_root=C:\work\apex_work\runs
[INFO] latest_run_dir=C:\work\apex_work\runs\gh22810456589_att1_sim_20260308_093151_ff026bb
[INFO] run_log=C:\work\apex_work\runs\gh22810456589_att1_sim_20260308_093151_ff026bb\logs\run_gh22810456589_att1_sim_20260308_093151_ff026bb.log
---- run_log_tail (last 350 lines) ----
[STEP] package_diagnostics_to_onedrive
[INFO] diag_script=C:\work\apex_repo_cache\ApexTraderAI\scripts\package_diagnostics_to_onedrive.ps1
[PUBLISH] run_dir=C:\work\apex_work\runs\gh22810456589_att1_sim_20260308_093151_ff026bb
[PUBLISH] output_root=C:\work\apex_work\runs\gh22810456589_att1_sim_20260308_093151_ff026bb\output
[WARN] publish issue: D' state CSV missing under output/stepDprime/<mode> or stepD_prime/<mode>.
[WARN] publish issue: D' embeddings CSV missing under output/stepDprime/<mode>/embeddings or stepD_prime/<mode>/embeddings.
[OK] package_diagnostics_to_onedrive
[OK] diag_dir=C:\Users\becky\OneDrive\ApexTraderAI\diagnostics

[STEP] resolve_latest_run_artifacts
[INFO] work_root=C:\work\apex_work\runs
[OK] resolve_latest_run_artifacts
[OK] run_dir=C:\work\apex_work\runs\gh22810456589_att1_sim_20260308_093151_ff026bb
[OK] run_id=gh22810456589_att1_sim_20260308_093151_ff026bb
```
