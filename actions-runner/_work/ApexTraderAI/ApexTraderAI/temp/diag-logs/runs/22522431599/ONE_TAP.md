# ONE_TAP Public Run Report

- run_id: 22522431599
- sha: 448a5ffc4475c49d26cbcf012a2169030ae78bca
- mode: sim
- symbols: SOXL,SOXS
- test_start: 2022-01-03
- train_years: 8
- test_months: 3
- output_root: C:/work/apex_work/runs/gh22522431599_att1_sim_20260228_231736_448a5ff/output

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
[CMD] git rev-parse --short HEAD
448a5ff

[CMD] git rev-parse --is-inside-work-tree
true

[CMD] where python
C:\Users\becky\AppData\Local\Programs\Python\Python310\python.exe

[CMD] python --version
Python 3.10.6

[CMD] "python" --version
Python 3.10.6

[CMD] "python" tools\run_with_python.py tools\prepare_data.py --symbols SOXL,SOXS --start 2013-12-27 --end 2022-04-10 --force --data-dir "C:\work\apex_work\runs\gh22522431599_att1_sim_20260228_231736_448a5ff\data"
[prepare_data] wrote C:\work\apex_work\runs\gh22522431599_att1_sim_20260228_231736_448a5ff\data\prices_SOXL.csv (2086 rows)
[prepare_data] wrote C:\work\apex_work\runs\gh22522431599_att1_sim_20260228_231736_448a5ff\data\prices_SOXS.csv (2086 rows)
[CMD] 'C:\Users\becky\AppData\Local\Programs\Python\Python310\python.exe' 'tools\prepare_data.py' --symbols SOXL,SOXS --start 2013-12-27 --end 2022-04-10 --force --data-dir 'C:\work\apex_work\runs\gh22522431599_att1_sim_20260228_231736_448a5ff\data'
[RC] 0

[CMD] "python" tools\run_with_python.py tools\run_pipeline.py --symbol SOXL --steps "F" --test-start 2022-01-03 --train-years 8 --test-months 3 --mode sim --output-root "C:\work\apex_work\runs\gh22522431599_att1_sim_20260228_231736_448a5ff\output" --data-dir "C:\work\apex_work\runs\gh22522431599_att1_sim_20260228_231736_448a5ff\data" --auto-prepare-data 0  --enable-mamba --skip-stepe
[headless] output_root_resolved=C:\work\apex_work\runs\gh22522431599_att1_sim_20260228_231736_448a5ff\output cfg_output_root=C:\work\apex_work\runs\gh22522431599_att1_sim_20260228_231736_448a5ff\output cfg_data_output_root=C:\work\apex_work\runs\gh22522431599_att1_sim_20260228_231736_448a5ff\output
[headless] data ready: SOXL,SOXS
[headless] StepF default config injected: agents=dprime_bnf_h01,dprime_bnf_h02,dprime_bnf_3scale,dprime_mix_h01,dprime_mix_h02,dprime_mix_3scale,dprime_all_features_h01,dprime_all_features_h02,dprime_all_features_h03,dprime_all_features_3scale safe_set=dprime_bnf_h01,dprime_all_features_h01 router(beta=1.0,ema=0.3,cost_bps=10.0)
[headless] repo_root=C:\work\apex_repo_cache\ApexTraderAI
[headless] symbol=SOXL
[headless] steps=F
[headless] skip_stepe=1
[headless] auto_prepare_data=0 data_start=2010-01-01 data_end=2026-02-28
[StepF] start
[PIPELINE] status=failed steps=F output_root=C:\work\apex_work\runs\gh22522431599_att1_sim_20260228_231736_448a5ff\output
Traceback (most recent call last):
  File "C:\work\apex_repo_cache\ApexTraderAI\tools\run_pipeline.py", line 1544, in <module>
    raise SystemExit(main())
  File "C:\work\apex_repo_cache\ApexTraderAI\tools\run_pipeline.py", line 1528, in main
    step_result = _run_step_generic(step, app_config, symbol, date_range, results)
  File "C:\work\apex_repo_cache\ApexTraderAI\tools\run_pipeline.py", line 1234, in _run_step_generic
    return _call_with_best_effort(fn, ctx2)
  File "C:\work\apex_repo_cache\ApexTraderAI\tools\run_pipeline.py", line 469, in _call_with_best_effort
    return fn(**kwargs)
  File "C:\work\apex_repo_cache\ApexTraderAI\ai_core\services\step_f_service.py", line 72, in run
    return self._run_router(cfg, date_range, symbol=symbol, mode=resolved_mode)
  File "C:\work\apex_repo_cache\ApexTraderAI\ai_core\services\step_f_service.py", line 98, in _run_router
    prices_soxl = self._load_stepa_price_tech(out_root, mode, "SOXL")
  File "C:\work\apex_repo_cache\ApexTraderAI\ai_core\services\step_f_service.py", line 168, in _load_stepa_price_tech
    p_tr = pd.read_csv(base / f"stepA_prices_train_{symbol}.csv")
  File "C:\Users\becky\AppData\Roaming\Python\Python310\site-packages\pandas\io\parsers\readers.py", line 1026, in read_csv
    return _read(filepath_or_buffer, kwds)
  File "C:\Users\becky\AppData\Roaming\Python\Python310\site-packages\pandas\io\parsers\readers.py", line 620, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
  File "C:\Users\becky\AppData\Roaming\Python\Python310\site-packages\pandas\io\parsers\readers.py", line 1620, in __init__
    self._engine = self._make_engine(f, self.engine)
  File "C:\Users\becky\AppData\Roaming\Python\Python310\site-packages\pandas\io\parsers\readers.py", line 1880, in _make_engine
    self.handles = get_handle(
  File "C:\Users\becky\AppData\Roaming\Python\Python310\site-packages\pandas\io\common.py", line 873, in get_handle
    handle = open(
FileNotFoundError: [Errno 2] No such file or directory: 'C:\\work\\apex_work\\runs\\gh22522431599_att1_sim_20260228_231736_448a5ff\\output\\stepA\\sim\\stepA_prices_train_SOXL.csv'
[CMD] 'C:\Users\becky\AppData\Local\Programs\Python\Python310\python.exe' 'tools\run_pipeline.py' --symbol SOXL --steps F --test-start 2022-01-03 --train-years 8 --test-months 3 --mode sim --output-root 'C:\work\apex_work\runs\gh22522431599_att1_sim_20260228_231736_448a5ff\output' --data-dir 'C:\work\apex_work\runs\gh22522431599_att1_sim_20260228_231736_448a5ff\data' --auto-prepare-data 0 --enable-mamba --skip-stepe
[RC] 1
[RC] 
[FAILED] command="python" tools\run_with_python.py tools\run_pipeline.py --symbol SOXL --steps "F" --test-start 2022-01-03 --train-years 8 --test-months 3 --mode sim --output-root "C:\work\apex_work\runs\gh22522431599_att1_sim_20260228_231736_448a5ff\output" --data-dir "C:\work\apex_work\runs\gh22522431599_att1_sim_20260228_231736_448a5ff\data" --auto-prepare-data 0  --enable-mamba --skip-stepe
[FAILED] exit_code=
[FAILED] run_id=gh22522431599_att1_sim_20260228_231736_448a5ff
[FAILED] log=C:\work\apex_work\runs\gh22522431599_att1_sim_20260228_231736_448a5ff\logs\run_gh22522431599_att1_sim_20260228_231736_448a5ff.log
[STEP] package_diagnostics_to_onedrive
[INFO] diag_script=C:\work\apex_repo_cache\ApexTraderAI\scripts\package_diagnostics_to_onedrive.ps1
[PUBLISH] run_dir=C:\work\apex_work\runs\gh22522431599_att1_sim_20260228_231736_448a5ff
[PUBLISH] output_root=C:\work\apex_work\runs\gh22522431599_att1_sim_20260228_231736_448a5ff\output
[WARN] publish issue: D' state CSV missing under output/stepD_prime/<mode>.
[WARN] publish issue: D' embeddings CSV missing under output/stepD_prime/<mode>/embeddings.
[OK] package_diagnostics_to_onedrive
[OK] diag_dir=C:\Users\becky\OneDrive\ApexTraderAI\diagnostics

[STEP] resolve_latest_run_artifacts
[INFO] work_root=C:\work\apex_work\runs
[OK] resolve_latest_run_artifacts
[OK] run_dir=C:\work\apex_work\runs\gh22522431599_att1_sim_20260228_231736_448a5ff
[OK] run_id=gh22522431599_att1_sim_20260228_231736_448a5ff
```
