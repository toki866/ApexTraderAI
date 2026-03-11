# ONE_TAP Public Run Report

- run_id: 22932705324
- sha: 50aea69b35cb15d58573e0785f63e995e9afe56e
- mode: sim
- symbols: SOXL,SOXS
- test_start: 2022-01-03
- train_years: 8
- test_months: 3
- output_root: C:/work/apex_work/output/sim/SOXL/2022-01-03

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
[REUSE] manifest_owner=run_pipeline_only
[STEP] resolve_reuse_stepa
[REUSE] primary_symbol=SOXL
[REUSE] data_prepare_symbols=SOXL,SOXS
[REUSE] stepa_execution_symbols=SOXL
[REUSE] stepa_reuse_symbols=SOXL
[REUSE] stepa_reuse_scope=stepA_simple
[REUSE] stepa_reuse_key=mode=sim|symbols=SOXL|test_start=2022-01-03|train_years=8|test_months=3
[REUSE] stepa_reuse_match_found=true
[REUSE] STEPA_REUSE_MATCH_FOUND=true
[REUSE] stepa_reuse_matched_output_root=C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03
[REUSE] stepa_reuse_matched_run_id=
[REUSE] stepa_reuse_selected_policy=canonical_output_root
[REUSE] stepa_reuse_reason=reuse
[REUSE] stepa_reuse_canonical_output_root=/mnt/c/work/apex_work/output/sim/SOXL/2022-01-03
[REUSE] stepa_reuse_eval_stepa_path=/mnt/c/work/apex_work/output/sim/SOXL/2022-01-03/stepA/sim
[REUSE] stepa_reuse_stepa_path=/mnt/c/work/apex_work/output/sim/SOXL/2022-01-03/stepA/sim
[REUSE][STEPA_REQUIRED] path=stepA/sim/stepA_prices_train_SOXL.csv required=true status=pass
[REUSE][STEPA_REQUIRED] path=stepA/sim/stepA_prices_test_SOXL.csv required=true status=pass
[REUSE][STEPA_REQUIRED] path=stepA/sim/stepA_periodic_train_SOXL.csv required=true status=pass
[REUSE][STEPA_REQUIRED] path=stepA/sim/stepA_periodic_test_SOXL.csv required=true status=pass
[REUSE][STEPA_REQUIRED] path=stepA/sim/stepA_tech_train_SOXL.csv required=true status=pass
[REUSE][STEPA_REQUIRED] path=stepA/sim/stepA_tech_test_SOXL.csv required=true status=pass
[REUSE][STEPA_REQUIRED] path=stepA/sim/stepA_split_summary_SOXL.csv required=true status=pass
[REUSE][STEPA_REQUIRED] path=stepA/sim/stepA_periodic_future_SOXL.csv required=true status=pass
[REUSE][STEPA_REQUIRED] path=stepA/sim/stepA_daily_manifest_SOXL.csv required=true status=pass
[REUSE][STEPA_REQUIRED] path=split_summary.json required=true status=pass
[STEP] resolve_reuse_pipeline
[REUSE] strict_reuse_decision=delegated_to_output_root_reuse
[REUSE] final_EFFECTIVE_WIN_OUTPUT='C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03'
[REUSE] final_EFFECTIVE_WSL_OUTPUT='/mnt/c/work/apex_work/output/sim/SOXL/2022-01-03'
[STEP] step_a_entry
[STEP_STATUS] requested_steps=A,B,C,DPRIME,E,F
[STEP_STATUS] prepare_environment_outcome=failure
[STEP_STATUS] resolve_reuse_outcome=success_via_stepa_simple
[STEP_STATUS] validate_context_outcome=not_defined
[STEP_STATUS] WIN_OUTPUT_DIR=
[STEP_STATUS] WSL_OUTPUT_DIR=
[STEP_STATUS] outRoot=C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03
[STEP_STATUS] wslOut=/mnt/c/work/apex_work/output/sim/SOXL/2022-01-03
[WARN] step_a skipped reason=prepare_environment_failure
[STEP_STATUS] step_a_decision_final=skip reason=prepare_environment_failure
[STEP_STATUS] step_a_entry_result=skip rc=0 last_exitcode=
[STEP_STATUS] step_a_entry_outRoot='C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03'
[STEP_STATUS] step_a_entry_wslOut='/mnt/c/work/apex_work/output/sim/SOXL/2022-01-03'
[STEP_STATUS] step_a_manifest_path=''
[STEP_STATUS] step=B manifest_reuse= artifact_reuse= reason= matched_run_id= manifest_reason= manifest_path= parse_error_type= fallback_to_artifact_check= reuse_key= reuse_match_found= reuse_matched_output_root= reuse_selected_policy=
[STEP_STATUS] step=B status= reason=execute
[STEPB] output_root=/mnt/c/work/apex_work/output/sim/SOXL/2022-01-03
[STEPB] expected_pred_file=/mnt/c/work/apex_work/output/sim/SOXL/2022-01-03/stepB/sim/stepB_pred_time_all_SOXL.csv
[NG] step_b invalid_wsl_repo_root=''

[STEP] append_error_tails
timestamp=2026-03-11 10:47:29 +09:00
[INFO] apex_run_id=gh22932705324_att1_sim_20260311_104710_50aea69
---- python_env ----
C:\Users\becky\AppData\Local\Programs\Python\Python310\python.exe
Python 3.10.6
C:\Users\becky\AppData\Local\Programs\Python\Python310\python.exe
[INFO] console_log=C:\Users\becky\OneDrive\デスクトップ\Python\apex-trader-ai\actions-runner\_work\_temp\run_all_local_then_copy_console.log
[WARN] console log not found
[INFO] work_root=C:\work\apex_work\runs
[INFO] session_log_root=C:\work\apex_work\session_logs
[INFO] latest_run_dir=C:\work\apex_work\runs\gh22931022882_att1_sim_20260311_094136_494d0ef
[WARN] run_*.log not found in RUN_LOG_PATH/session_logs/runs fallback
[STEP] package_diagnostics_to_onedrive
[INFO] diag_script=C:\work\apex_repo_cache\ApexTraderAI\scripts\package_diagnostics_to_onedrive.ps1
[PUBLISH] run_dir=C:\work\apex_work\runs\gh22931022882_att1_sim_20260311_094136_494d0ef
[PUBLISH] output_root=C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03
[WARN] publish issue: D' state CSV missing under output/stepDprime/<mode> or stepD_prime/<mode>.
[WARN] publish issue: D' embeddings CSV missing under output/stepDprime/<mode>/embeddings or stepD_prime/<mode>/embeddings.
[OK] package_diagnostics_to_onedrive
[OK] diag_dir=C:\Users\becky\OneDrive\ApexTraderAI\diagnostics

[STEP] resolve_latest_run_artifacts
[INFO] work_root=C:\work\apex_work\runs
[OK] resolve_latest_run_artifacts
[OK] run_dir=C:\work\apex_work\runs\gh22931022882_att1_sim_20260311_094136_494d0ef
[OK] run_id=gh22932705324_att1_sim_20260311_104710_50aea69
```
