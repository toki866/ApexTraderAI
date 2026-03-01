# ONE_TAP Public Run Report

- run_id: 22519795433
- sha: a7660ff40f41fc84c134b84efc9ea52a6ae9e9e8
- mode: sim
- symbols: SOXL,SOXS
- test_start: 2022-01-03
- train_years: 8
- test_months: 3
- output_root: C:/work/apex_work/runs/gh22519795433_att1_sim_20260228_202455_a7660ff/output

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
[RUN] data_dir=C:\work\apex_work\runs\gh22519795433_att1_sim_20260228_202455_a7660ff\data
[RUN] output_dir=C:\work\apex_work\runs\gh22519795433_att1_sim_20260228_202455_a7660ff\output

[CMD] git rev-parse --short HEAD
a7660ff

[CMD] git rev-parse --is-inside-work-tree
true

[CMD] where python
C:\Users\becky\AppData\Local\Programs\Python\Python310\python.exe

[CMD] python --version
Python 3.10.6

[CMD] "python" --version
Python 3.10.6

[CMD] "python" tools\run_with_python.py tools\prepare_data.py --symbols SOXL,SOXS --start 2013-12-27 --end 2022-04-10 --force --data-dir "C:\work\apex_work\runs\gh22519795433_att1_sim_20260228_202455_a7660ff\data"
[prepare_data] wrote C:\work\apex_work\runs\gh22519795433_att1_sim_20260228_202455_a7660ff\data\prices_SOXL.csv (2086 rows)
[prepare_data] wrote C:\work\apex_work\runs\gh22519795433_att1_sim_20260228_202455_a7660ff\data\prices_SOXS.csv (2086 rows)
[CMD] 'C:\Users\becky\AppData\Local\Programs\Python\Python310\python.exe' 'tools\prepare_data.py' --symbols SOXL,SOXS --start 2013-12-27 --end 2022-04-10 --force --data-dir 'C:\work\apex_work\runs\gh22519795433_att1_sim_20260228_202455_a7660ff\data'
[RC] 0

[CMD] "python" tools\run_with_python.py tools\run_pipeline.py --symbol SOXL --steps "A,B,C,DPRIME,E,F" --test-start 2022-01-03 --train-years 8 --test-months 3 --mode sim --output-root "C:\work\apex_work\runs\gh22519795433_att1_sim_20260228_202455_a7660ff\output" --data-dir "C:\work\apex_work\runs\gh22519795433_att1_sim_20260228_202455_a7660ff\data" --auto-prepare-data 0  --enable-mamba
[StepB] WARN: failed to ensure stepB_pred_time_all: _load_prices_dates() missing 3 required positional arguments: 'output_root', 'data_root', and 'mode'
[headless] output_root_resolved=C:\work\apex_work\runs\gh22519795433_att1_sim_20260228_202455_a7660ff\output cfg_output_root=C:\work\apex_work\runs\gh22519795433_att1_sim_20260228_202455_a7660ff\output cfg_data_output_root=C:\work\apex_work\runs\gh22519795433_att1_sim_20260228_202455_a7660ff\output
[headless] data ready: SOXL,SOXS
[headless] StepE default config injected: agents=dprime_bnf_h01,dprime_bnf_h02,dprime_bnf_3scale,dprime_mix_h01,dprime_mix_h02,dprime_mix_3scale,dprime_all_features_h01,dprime_all_features_h02,dprime_all_features_h03,dprime_all_features_3scale seed=42+idx device=auto
[headless] StepF default config injected: agents=dprime_all_features_3scale,dprime_all_features_h01,dprime_all_features_h02,dprime_all_features_h03,dprime_bnf_3scale,dprime_bnf_h01,dprime_bnf_h02,dprime_mix_3scale,dprime_mix_h01,dprime_mix_h02 seed=42 device=auto use_context=1 context_variant=mix context_profile=minimal
[headless] repo_root=C:\work\apex_repo_cache\ApexTraderAI
[headless] symbol=SOXL
[headless] steps=A,B,C,DPRIME,E,F
[headless] skip_stepe=0
[headless] auto_prepare_data=0 data_start=2010-01-01 data_end=2026-02-28
[StepA] start
[StepA] data_dir=C:\work\apex_work\runs\gh22519795433_att1_sim_20260228_202455_a7660ff\data
[StepA] data_dir=C:\work\apex_work\runs\gh22519795433_att1_sim_20260228_202455_a7660ff\data src_csv=C:\work\apex_work\runs\gh22519795433_att1_sim_20260228_202455_a7660ff\data\prices_SOXL.csv
[StepA] done
[StepB] start
[StepB:mamba] wrote daily manifest -> C:\work\apex_work\runs\gh22519795433_att1_sim_20260228_202455_a7660ff\output\stepB\sim\stepB_daily_manifest_SOXL.csv rows=63
[StepB:mamba] wrote daily manifest -> C:\work\apex_work\runs\gh22519795433_att1_sim_20260228_202455_a7660ff\output\stepB\sim\stepB_daily_manifest_periodic_SOXL.csv rows=63
[StepB] agents: mamba=True
[StepB] done
[StepC] start
[StepC] price_path=C:\work\apex_work\runs\gh22519795433_att1_sim_20260228_202455_a7660ff\output\stepA\sim\stepA_prices_train_SOXL.csv
[StepC] wrote daily snapshots: 63 -> C:\work\apex_work\runs\gh22519795433_att1_sim_20260228_202455_a7660ff\output\stepC\sim\stepC_daily_manifest_SOXL.csv
[StepC] done
[StepDPrime] start
[PIPELINE] status=failed steps=A,B,C,DPRIME,E,F output_root=C:\work\apex_work\runs\gh22519795433_att1_sim_20260228_202455_a7660ff\output
Traceback (most recent call last):
  File "C:\work\apex_repo_cache\ApexTraderAI\tools\run_pipeline.py", line 1529, in <module>
    raise SystemExit(main())
  File "C:\work\apex_repo_cache\ApexTraderAI\tools\run_pipeline.py", line 1505, in main
    results["stepDPRIME_result"] = _run_stepDPrime(app_config, symbol, date_range, mode=resolved_mamba_mode)
  File "C:\work\apex_repo_cache\ApexTraderAI\tools\run_pipeline.py", line 1168, in _run_stepDPrime
    cfg = StepDPrimeConfig(
TypeError: StepDPrimeConfig.__init__() got an unexpected keyword argument 'seed'
[CMD] 'C:\Users\becky\AppData\Local\Programs\Python\Python310\python.exe' 'tools\run_pipeline.py' --symbol SOXL --steps A,B,C,DPRIME,E,F --test-start 2022-01-03 --train-years 8 --test-months 3 --mode sim --output-root 'C:\work\apex_work\runs\gh22519795433_att1_sim_20260228_202455_a7660ff\output' --data-dir 'C:\work\apex_work\runs\gh22519795433_att1_sim_20260228_202455_a7660ff\data' --auto-prepare-data 0 --enable-mamba
[RC] 1
[RC] 
[FAILED] command="python" tools\run_with_python.py tools\run_pipeline.py --symbol SOXL --steps "A,B,C,DPRIME,E,F" --test-start 2022-01-03 --train-years 8 --test-months 3 --mode sim --output-root "C:\work\apex_work\runs\gh22519795433_att1_sim_20260228_202455_a7660ff\output" --data-dir "C:\work\apex_work\runs\gh22519795433_att1_sim_20260228_202455_a7660ff\data" --auto-prepare-data 0  --enable-mamba
[FAILED] exit_code=
[FAILED] run_id=gh22519795433_att1_sim_20260228_202455_a7660ff
[FAILED] log=C:\work\apex_work\runs\gh22519795433_att1_sim_20260228_202455_a7660ff\logs\run_gh22519795433_att1_sim_20260228_202455_a7660ff.log
[STEP] package_diagnostics_to_onedrive
[INFO] diag_script=C:\work\apex_repo_cache\ApexTraderAI\scripts\package_diagnostics_to_onedrive.ps1
[PUBLISH] run_dir=C:\work\apex_work\runs\gh22519795433_att1_sim_20260228_202455_a7660ff
[PUBLISH] output_root=C:\work\apex_work\runs\gh22519795433_att1_sim_20260228_202455_a7660ff\output
[WARN] publish issue: D' state CSV missing under output/stepD_prime/<mode>.
[WARN] publish issue: D' embeddings CSV missing under output/stepD_prime/<mode>/embeddings.
[OK] package_diagnostics_to_onedrive
[OK] diag_dir=C:\Users\becky\OneDrive\ApexTraderAI\diagnostics

[STEP] resolve_latest_run_artifacts
[INFO] work_root=C:\work\apex_work\runs
[OK] resolve_latest_run_artifacts
[OK] run_dir=C:\work\apex_work\runs\gh22519795433_att1_sim_20260228_202455_a7660ff
[OK] run_id=gh22519795433_att1_sim_20260228_202455_a7660ff
```
