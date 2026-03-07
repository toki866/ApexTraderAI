# ONE_TAP Public Run Report

- run_id: 22803384193
- sha: ac4f772146341530d734aabd8243d47757aa1be0
- mode: sim
- symbols: SOXL,SOXS
- test_start: 2022-01-03
- train_years: 8
- test_months: 3
- output_root: C:/work/apex_work/runs/gh22802585511_att1_sim_20260308_011950_3445e5f/output

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
timestamp=2026-03-08 02:09:22 +09:00
repo=toki866/ApexTraderAI
ref=refs/heads/main
sha=ac4f772146341530d734aabd8243d47757aa1be0
run_id=22803384193
run_attempt=1

[STEP] initialize_one_tap_report
[OK] report initialized
[PATH] C:\Users\becky\OneDrive\デスクトップ\Python\apex-trader-ai\actions-runner\_work\ApexTraderAI\ApexTraderAI\temp\ONE_TAP_ERROR_REPORT.txt
[INFO] prep_repo_outcome=success
[INFO] REPO_ROOT=C:\work\apex_repo_cache\ApexTraderAI

[INFO] HEAD=ac4f772

## GPU_DIAGNOSTICS
torch 2.10.0+cu126
cuda_available True
gpu NVIDIA GeForce RTX 3080 Ti
timestamp, utilization.gpu [%], utilization.memory [%], memory.used [MiB]
2026/03/08 02:09:25.650, 0 %, 3 %, 1994 MiB

[STEP] compute_steps
[INFO] computed_steps=A,B,C,DPRIME,E,F
[INFO] steps_source=toggle
[INFO] effective_skip_stepe=0

[STEP] compute_derived_date_window
[INFO] test_start_date=2022-01-03
[INFO] data_start_date=2013-12-27
[INFO] data_end_date=2022-04-10


[STEP] append_error_tails
timestamp=2026-03-08 02:09:30 +09:00
[INFO] apex_run_id=gh22803384193_att1_sim_20260308_020928_ac4f772
---- python_env ----
C:\Users\becky\AppData\Local\Programs\Python\Python310\python.exe
Python 3.10.6
C:\Users\becky\AppData\Local\Programs\Python\Python310\python.exe
[INFO] console_log=C:\Users\becky\OneDrive\デスクトップ\Python\apex-trader-ai\actions-runner\_work\_temp\run_all_local_then_copy_console.log
[WARN] console log not found
[INFO] bootstrap_log=
[WARN] bootstrap log not found
[INFO] work_root=C:\work\apex_work\runs
[INFO] latest_run_dir=C:\work\apex_work\runs\gh22802585511_att1_sim_20260308_011950_3445e5f
[WARN] run_*.log not found under latest_run_dir\logs
[STEP] package_diagnostics_to_onedrive
[INFO] diag_script=C:\work\apex_repo_cache\ApexTraderAI\scripts\package_diagnostics_to_onedrive.ps1
[PUBLISH] run_dir=C:\work\apex_work\runs\gh22802585511_att1_sim_20260308_011950_3445e5f
[PUBLISH] output_root=C:\work\apex_work\runs\gh22802585511_att1_sim_20260308_011950_3445e5f\output
[WARN] publish issue: D' state CSV missing under output/stepDprime/<mode> or stepD_prime/<mode>.
[WARN] publish issue: D' embeddings CSV missing under output/stepDprime/<mode>/embeddings or stepD_prime/<mode>/embeddings.
[OK] package_diagnostics_to_onedrive
[OK] diag_dir=C:\Users\becky\OneDrive\ApexTraderAI\diagnostics

[STEP] resolve_latest_run_artifacts
[INFO] work_root=C:\work\apex_work\runs
[OK] resolve_latest_run_artifacts
[OK] run_dir=C:\work\apex_work\runs\gh22802585511_att1_sim_20260308_011950_3445e5f
[OK] run_id=gh22803384193_att1_sim_20260308_020928_ac4f772
```
