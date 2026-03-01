# ONE_TAP Public Run Report

- run_id: 22520779888
- sha: bcc26d16dce73b40664c1b5c3c401a28792b8863
- mode: sim
- symbols: SOXL,SOXS
- test_start: 2022-01-03
- train_years: 8
- test_months: 3
- output_root: C:/work/apex_work/runs/gh22520779888_att1_sim_20260228_213051_bcc26d1/output

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
[CMD] python --version
Python 3.10.6

[CMD] "python" --version
Python 3.10.6

[CMD] "python" tools\run_with_python.py tools\prepare_data.py --symbols SOXL,SOXS --start 2013-12-27 --end 2022-04-10 --force --data-dir "C:\work\apex_work\runs\gh22520779888_att1_sim_20260228_213051_bcc26d1\data"
[prepare_data] wrote C:\work\apex_work\runs\gh22520779888_att1_sim_20260228_213051_bcc26d1\data\prices_SOXL.csv (2086 rows)
[prepare_data] wrote C:\work\apex_work\runs\gh22520779888_att1_sim_20260228_213051_bcc26d1\data\prices_SOXS.csv (2086 rows)
[CMD] 'C:\Users\becky\AppData\Local\Programs\Python\Python310\python.exe' 'tools\prepare_data.py' --symbols SOXL,SOXS --start 2013-12-27 --end 2022-04-10 --force --data-dir 'C:\work\apex_work\runs\gh22520779888_att1_sim_20260228_213051_bcc26d1\data'
[RC] 0

[CMD] "python" tools\run_with_python.py tools\run_pipeline.py --symbol SOXL --steps "A,B,C,DPRIME,E,F" --test-start 2022-01-03 --train-years 8 --test-months 3 --mode sim --output-root "C:\work\apex_work\runs\gh22520779888_att1_sim_20260228_213051_bcc26d1\output" --data-dir "C:\work\apex_work\runs\gh22520779888_att1_sim_20260228_213051_bcc26d1\data" --auto-prepare-data 0  --enable-mamba
[headless] output_root_resolved=C:\work\apex_work\runs\gh22520779888_att1_sim_20260228_213051_bcc26d1\output cfg_output_root=C:\work\apex_work\runs\gh22520779888_att1_sim_20260228_213051_bcc26d1\output cfg_data_output_root=C:\work\apex_work\runs\gh22520779888_att1_sim_20260228_213051_bcc26d1\output
[headless] data ready: SOXL,SOXS
[headless] StepE default config injected: agents=dprime_bnf_h01,dprime_bnf_h02,dprime_bnf_3scale,dprime_mix_h01,dprime_mix_h02,dprime_mix_3scale,dprime_all_features_h01,dprime_all_features_h02,dprime_all_features_h03,dprime_all_features_3scale seed=42+idx device=auto
[headless] StepF default config injected: agents=dprime_all_features_3scale,dprime_all_features_h01,dprime_all_features_h02,dprime_all_features_h03,dprime_bnf_3scale,dprime_bnf_h01,dprime_bnf_h02,dprime_mix_3scale,dprime_mix_h01,dprime_mix_h02 seed=42 device=auto use_context=1 context_variant=mix context_profile=minimal
[headless] repo_root=C:\work\apex_repo_cache\ApexTraderAI
[headless] symbol=SOXL
[headless] steps=A,B,C,DPRIME,E,F
[headless] skip_stepe=0
[headless] auto_prepare_data=0 data_start=2010-01-01 data_end=2026-02-28
[StepA] start
[StepA] data_dir=C:\work\apex_work\runs\gh22520779888_att1_sim_20260228_213051_bcc26d1\data
[StepA] data_dir=C:\work\apex_work\runs\gh22520779888_att1_sim_20260228_213051_bcc26d1\data src_csv=C:\work\apex_work\runs\gh22520779888_att1_sim_20260228_213051_bcc26d1\data\prices_SOXL.csv
[StepA] done
[StepB] start
[StepB:mamba] wrote daily manifest -> C:\work\apex_work\runs\gh22520779888_att1_sim_20260228_213051_bcc26d1\output\stepB\sim\stepB_daily_manifest_SOXL.csv rows=63
[StepB:mamba] wrote daily manifest -> C:\work\apex_work\runs\gh22520779888_att1_sim_20260228_213051_bcc26d1\output\stepB\sim\stepB_daily_manifest_periodic_SOXL.csv rows=63
[StepB] agents: mamba=True
[StepB] done
[StepB] ensured: C:\work\apex_work\runs\gh22520779888_att1_sim_20260228_213051_bcc26d1\output\stepB\sim\stepB_pred_time_all_SOXL.csv
[StepC] start
[StepC] price_path=C:\work\apex_work\runs\gh22520779888_att1_sim_20260228_213051_bcc26d1\output\stepA\sim\stepA_prices_train_SOXL.csv
[StepC] wrote daily snapshots: 63 -> C:\work\apex_work\runs\gh22520779888_att1_sim_20260228_213051_bcc26d1\output\stepC\sim\stepC_daily_manifest_SOXL.csv
[StepC] done
[StepDPrime] start
[StepDPrime] done
[StepE] start
[StepE] agent=dprime_bnf_h01 mode=sim profile=D use_stepd_prime=True seed=42 device=auto
[PIPELINE] status=failed steps=A,B,C,DPRIME,E,F output_root=C:\work\apex_work\runs\gh22520779888_att1_sim_20260228_213051_bcc26d1\output
Traceback (most recent call last):
  File "C:\work\apex_repo_cache\ApexTraderAI\tools\run_pipeline.py", line 1532, in <module>
    raise SystemExit(main())
  File "C:\work\apex_repo_cache\ApexTraderAI\tools\run_pipeline.py", line 1516, in main
    step_result = _run_step_generic(step, app_config, symbol, date_range, results)
  File "C:\work\apex_repo_cache\ApexTraderAI\tools\run_pipeline.py", line 1230, in _run_step_generic
    return _call_with_best_effort(fn, ctx2)
  File "C:\work\apex_repo_cache\ApexTraderAI\tools\run_pipeline.py", line 465, in _call_with_best_effort
    return fn(**kwargs)
  File "C:\work\apex_repo_cache\ApexTraderAI\ai_core\services\step_e_service.py", line 146, in run
    self._run_one(cfg, date_range=date_range, symbol=symbol, mode=mode)
  File "C:\work\apex_repo_cache\ApexTraderAI\ai_core\services\step_e_service.py", line 173, in _run_one
    df_all, used_manifest = self._merge_inputs(cfg, out_root=out_root, mode=mode, symbol=symbol)
  File "C:\work\apex_repo_cache\ApexTraderAI\ai_core\services\step_e_service.py", line 427, in _merge_inputs
    dprime_df = self._load_stepD_prime_embeddings(cfg, out_root=out_root, mode=mode, symbol=symbol)
  File "C:\work\apex_repo_cache\ApexTraderAI\ai_core\services\step_e_service.py", line 531, in _load_stepD_prime_embeddings
    raise ValueError("dprime_sources is empty while use_stepd_prime=True")
ValueError: dprime_sources is empty while use_stepd_prime=True
[CMD] 'C:\Users\becky\AppData\Local\Programs\Python\Python310\python.exe' 'tools\run_pipeline.py' --symbol SOXL --steps A,B,C,DPRIME,E,F --test-start 2022-01-03 --train-years 8 --test-months 3 --mode sim --output-root 'C:\work\apex_work\runs\gh22520779888_att1_sim_20260228_213051_bcc26d1\output' --data-dir 'C:\work\apex_work\runs\gh22520779888_att1_sim_20260228_213051_bcc26d1\data' --auto-prepare-data 0 --enable-mamba
[RC] 1
[RC] 
[FAILED] command="python" tools\run_with_python.py tools\run_pipeline.py --symbol SOXL --steps "A,B,C,DPRIME,E,F" --test-start 2022-01-03 --train-years 8 --test-months 3 --mode sim --output-root "C:\work\apex_work\runs\gh22520779888_att1_sim_20260228_213051_bcc26d1\output" --data-dir "C:\work\apex_work\runs\gh22520779888_att1_sim_20260228_213051_bcc26d1\data" --auto-prepare-data 0  --enable-mamba
[FAILED] exit_code=
[FAILED] run_id=gh22520779888_att1_sim_20260228_213051_bcc26d1
[FAILED] log=C:\work\apex_work\runs\gh22520779888_att1_sim_20260228_213051_bcc26d1\logs\run_gh22520779888_att1_sim_20260228_213051_bcc26d1.log
[STEP] package_diagnostics_to_onedrive
[INFO] diag_script=C:\work\apex_repo_cache\ApexTraderAI\scripts\package_diagnostics_to_onedrive.ps1
[PUBLISH] run_dir=C:\work\apex_work\runs\gh22520779888_att1_sim_20260228_213051_bcc26d1
[PUBLISH] output_root=C:\work\apex_work\runs\gh22520779888_att1_sim_20260228_213051_bcc26d1\output
[WARN] publish issue: D' embeddings CSV missing under output/stepD_prime/<mode>/embeddings.
[OK] package_diagnostics_to_onedrive
[OK] diag_dir=C:\Users\becky\OneDrive\ApexTraderAI\diagnostics

[STEP] resolve_latest_run_artifacts
[INFO] work_root=C:\work\apex_work\runs
[OK] resolve_latest_run_artifacts
[OK] run_dir=C:\work\apex_work\runs\gh22520779888_att1_sim_20260228_213051_bcc26d1
[OK] run_id=gh22520779888_att1_sim_20260228_213051_bcc26d1
```
