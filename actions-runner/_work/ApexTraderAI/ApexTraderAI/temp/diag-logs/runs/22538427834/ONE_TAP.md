# ONE_TAP Public Run Report

- run_id: 22538427834
- sha: 72421a44d15aee35f5e759a51f14925d777e0477
- mode: sim
- symbols: SOXL,SOXS
- test_start: 2022-01-03
- train_years: 8
- test_months: 3
- output_root: C:/work/apex_work/runs/gh22538427834_att1_sim_20260301_162004_72421a4/output

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
[RUN] requirement=Never finish a workflow run without leaving a readable log artifact
[RUN] apex_run_id=gh22538427834_att1_sim_20260301_162004_72421a4
[RUN] run_id=gh22538427834_att1_sim_20260301_162004_72421a4
[RUN] repo=C:\work\apex_repo_cache\ApexTraderAI
[RUN] work_root=C:\work\apex_work\runs
[RUN] run_dir=C:\work\apex_work\runs\gh22538427834_att1_sim_20260301_162004_72421a4
[RUN] log=C:\work\apex_work\runs\gh22538427834_att1_sim_20260301_162004_72421a4\logs\run_gh22538427834_att1_sim_20260301_162004_72421a4.log
[RUN] timestamp=2026/03/01 16:20:05.88
==================================================
[RUN] commit=72421a4
[RUN] data_dir=C:\work\apex_work\runs\gh22538427834_att1_sim_20260301_162004_72421a4\data
[RUN] output_dir=C:\work\apex_work\runs\gh22538427834_att1_sim_20260301_162004_72421a4\output

[CMD] git rev-parse --short HEAD
72421a4

[CMD] git rev-parse --is-inside-work-tree
true

[CMD] where python
C:\Users\becky\AppData\Local\Programs\Python\Python310\python.exe

[CMD] python --version
Python 3.10.6

[CMD] "python" --version
Python 3.10.6

[CMD] "python" tools\run_with_python.py tools\prepare_data.py --symbols SOXL,SOXS --start 2013-12-27 --end 2022-04-10 --force --data-dir "C:\work\apex_work\runs\gh22538427834_att1_sim_20260301_162004_72421a4\data"
[prepare_data] wrote C:\work\apex_work\runs\gh22538427834_att1_sim_20260301_162004_72421a4\data\prices_SOXL.csv (2086 rows)
[prepare_data] wrote C:\work\apex_work\runs\gh22538427834_att1_sim_20260301_162004_72421a4\data\prices_SOXS.csv (2086 rows)
[CMD] 'C:\Users\becky\AppData\Local\Programs\Python\Python310\python.exe' 'tools\prepare_data.py' --symbols SOXL,SOXS --start 2013-12-27 --end 2022-04-10 --force --data-dir 'C:\work\apex_work\runs\gh22538427834_att1_sim_20260301_162004_72421a4\data'
[RC] 0

[CMD] "python" tools\run_with_python.py tools\run_pipeline.py --symbol SOXL --steps "DPRIME,E,F" --test-start 2022-01-03 --train-years 8 --test-months 3 --mode sim --output-root "C:\work\apex_work\runs\gh22538427834_att1_sim_20260301_162004_72421a4\output" --data-dir "C:\work\apex_work\runs\gh22538427834_att1_sim_20260301_162004_72421a4\data" --auto-prepare-data 0  --enable-mamba
[headless] output_root_resolved=C:\work\apex_work\runs\gh22538427834_att1_sim_20260301_162004_72421a4\output cfg_output_root=C:\work\apex_work\runs\gh22538427834_att1_sim_20260301_162004_72421a4\output cfg_data_output_root=C:\work\apex_work\runs\gh22538427834_att1_sim_20260301_162004_72421a4\output
[headless] data ready: SOXL,SOXS
[headless] StepE default config injected: agents=dprime_bnf_h01,dprime_bnf_h02,dprime_bnf_3scale,dprime_mix_h01,dprime_mix_h02,dprime_mix_3scale,dprime_all_features_h01,dprime_all_features_h02,dprime_all_features_h03,dprime_all_features_3scale seed=42+idx device=auto
[headless] StepF default config injected: agents=dprime_bnf_h01,dprime_bnf_h02,dprime_bnf_3scale,dprime_mix_h01,dprime_mix_h02,dprime_mix_3scale,dprime_all_features_h01,dprime_all_features_h02,dprime_all_features_h03,dprime_all_features_3scale safe_set=dprime_bnf_h01,dprime_all_features_h01 router(beta=1.0,ema=0.3,cost_bps=10.0)
[headless] repo_root=C:\work\apex_repo_cache\ApexTraderAI
[headless] symbol=SOXL
[headless] steps=DPRIME,E,F
[headless] skip_stepe=0
[headless] auto_prepare_data=0 data_start=2010-01-01 data_end=2026-03-01
[StepDPrime] start
[PIPELINE] status=failed steps=DPRIME,E,F output_root=C:\work\apex_work\runs\gh22538427834_att1_sim_20260301_162004_72421a4\output
Traceback (most recent call last):
  File "C:\work\apex_repo_cache\ApexTraderAI\tools\run_pipeline.py", line 1617, in <module>
    raise SystemExit(main())
  File "C:\work\apex_repo_cache\ApexTraderAI\tools\run_pipeline.py", line 1586, in main
    results["stepDPRIME_result"] = _run_stepDPrime(app_config, symbol, date_range, mode=resolved_mamba_mode)
  File "C:\work\apex_repo_cache\ApexTraderAI\tools\run_pipeline.py", line 1188, in _run_stepDPrime
    return svc.run(cfg)
  File "C:\work\apex_repo_cache\ApexTraderAI\ai_core\services\step_dprime_service.py", line 225, in run
    split = _read_split_summary(stepa_dir, cfg.symbol)
  File "C:\work\apex_repo_cache\ApexTraderAI\ai_core\services\step_dprime_service.py", line 56, in _read_split_summary
    raise FileNotFoundError(f"missing StepA split summary: {p}")
FileNotFoundError: missing StepA split summary: C:\work\apex_work\runs\gh22538427834_att1_sim_20260301_162004_72421a4\output\stepA\sim\stepA_split_summary_SOXL.csv
[CMD] 'C:\Users\becky\AppData\Local\Programs\Python\Python310\python.exe' 'tools\run_pipeline.py' --symbol SOXL --steps DPRIME,E,F --test-start 2022-01-03 --train-years 8 --test-months 3 --mode sim --output-root 'C:\work\apex_work\runs\gh22538427834_att1_sim_20260301_162004_72421a4\output' --data-dir 'C:\work\apex_work\runs\gh22538427834_att1_sim_20260301_162004_72421a4\data' --auto-prepare-data 0 --enable-mamba
[RC] 1
[RC] 
[FAILED] command="python" tools\run_with_python.py tools\run_pipeline.py --symbol SOXL --steps "DPRIME,E,F" --test-start 2022-01-03 --train-years 8 --test-months 3 --mode sim --output-root "C:\work\apex_work\runs\gh22538427834_att1_sim_20260301_162004_72421a4\output" --data-dir "C:\work\apex_work\runs\gh22538427834_att1_sim_20260301_162004_72421a4\data" --auto-prepare-data 0  --enable-mamba
[FAILED] exit_code=
[FAILED] run_id=gh22538427834_att1_sim_20260301_162004_72421a4
[FAILED] log=C:\work\apex_work\runs\gh22538427834_att1_sim_20260301_162004_72421a4\logs\run_gh22538427834_att1_sim_20260301_162004_72421a4.log
[STEP] package_diagnostics_to_onedrive
[INFO] diag_script=C:\work\apex_repo_cache\ApexTraderAI\scripts\package_diagnostics_to_onedrive.ps1
[PUBLISH] run_dir=C:\work\apex_work\runs\gh22538427834_att1_sim_20260301_162004_72421a4
[PUBLISH] output_root=C:\work\apex_work\runs\gh22538427834_att1_sim_20260301_162004_72421a4\output
[WARN] publish issue: D' state CSV missing under output/stepD_prime/<mode>.
[WARN] publish issue: D' embeddings CSV missing under output/stepD_prime/<mode>/embeddings.
[OK] package_diagnostics_to_onedrive
[OK] diag_dir=C:\Users\becky\OneDrive\ApexTraderAI\diagnostics

[STEP] resolve_latest_run_artifacts
[INFO] work_root=C:\work\apex_work\runs
[OK] resolve_latest_run_artifacts
[OK] run_dir=C:\work\apex_work\runs\gh22538427834_att1_sim_20260301_162004_72421a4
[OK] run_id=gh22538427834_att1_sim_20260301_162004_72421a4
```
