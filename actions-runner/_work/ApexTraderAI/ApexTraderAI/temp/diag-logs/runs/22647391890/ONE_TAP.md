# ONE_TAP Public Run Report

- run_id: 22647391890
- sha: ed934d026571797e648f0027a0f4f693d486213e
- mode: sim
- symbols: SOXL,SOXS
- test_start: 2022-01-03
- train_years: 8
- test_months: 3
- output_root: C:/work/apex_work/runs/gh22647391890_att1_sim_20260304_082244_ed934d0/output

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
Requirement already satisfied: pycparser in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from cffi>=1.12.0->curl_cffi<0.14,>=0.7->yfinance->-r requirements.txt (line 4)) (3.0)
Requirement already satisfied: six>=1.5 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from python-dateutil>=2.8.2->pandas->-r requirements.txt (line 2)) (1.16.0)
Requirement already satisfied: charset-normalizer<4,>=2 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from requests>=2.31->yfinance->-r requirements.txt (line 4)) (3.2.0)
Requirement already satisfied: idna<4,>=2.5 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from requests>=2.31->yfinance->-r requirements.txt (line 4)) (3.4)
Requirement already satisfied: urllib3<3,>=1.21.1 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from requests>=2.31->yfinance->-r requirements.txt (line 4)) (2.0.3)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from sympy>=1.13.3->torch<3.0,>=2.3->stable-baselines3->-r requirements.txt (line 9)) (1.3.0)
Requirement already satisfied: MarkupSafe>=2.0 in c:\users\becky\appdata\roaming\python\python310\site-packages (from jinja2->torch<3.0,>=2.3->stable-baselines3->-r requirements.txt (line 9)) (3.0.3)

[CMD] "python" tools\run_with_python.py tools\prepare_data.py --symbols SOXL,SOXS --start 2013-12-27 --end 2022-04-10 --force --data-dir "C:\work\apex_work\runs\gh22647391890_att1_sim_20260304_082244_ed934d0\data"
[prepare_data] wrote C:\work\apex_work\runs\gh22647391890_att1_sim_20260304_082244_ed934d0\data\prices_SOXL.csv (2086 rows)
[prepare_data] wrote C:\work\apex_work\runs\gh22647391890_att1_sim_20260304_082244_ed934d0\data\prices_SOXS.csv (2086 rows)
[CMD] 'C:\Users\becky\AppData\Local\Programs\Python\Python310\python.exe' 'tools\prepare_data.py' --symbols SOXL,SOXS --start 2013-12-27 --end 2022-04-10 --force --data-dir 'C:\work\apex_work\runs\gh22647391890_att1_sim_20260304_082244_ed934d0\data'
[RC] 0

[CMD] "python" tools\run_with_python.py tools\run_pipeline.py --symbol SOXL --steps "A,B,C,DPRIME,E,F" --test-start 2022-01-03 --train-years 8 --test-months 3 --mode sim --output-root "C:\work\apex_work\runs\gh22647391890_att1_sim_20260304_082244_ed934d0\output" --data-dir "C:\work\apex_work\runs\gh22647391890_att1_sim_20260304_082244_ed934d0\data" --auto-prepare-data 0  --enable-mamba
[headless] output_root_resolved=C:\work\apex_work\runs\gh22647391890_att1_sim_20260304_082244_ed934d0\output cfg_output_root=C:\work\apex_work\runs\gh22647391890_att1_sim_20260304_082244_ed934d0\output cfg_data_output_root=C:\work\apex_work\runs\gh22647391890_att1_sim_20260304_082244_ed934d0\output
[headless] data ready: SOXL,SOXS
[headless] StepE default config injected: agents=dprime_bnf_h01,dprime_bnf_h02,dprime_bnf_3scale,dprime_mix_h01,dprime_mix_h02,dprime_mix_3scale,dprime_all_features_h01,dprime_all_features_h02,dprime_all_features_h03,dprime_all_features_3scale seed=42+idx device=auto
[headless] StepF default config injected: agents=dprime_bnf_h01,dprime_bnf_h02,dprime_bnf_3scale,dprime_mix_h01,dprime_mix_h02,dprime_mix_3scale,dprime_all_features_h01,dprime_all_features_h02,dprime_all_features_h03,dprime_all_features_3scale safe_set=dprime_bnf_h01,dprime_all_features_h01,dprime_mix_3scale router(beta=1.0,ema=0.5,cost_bps=15.0)
[headless] repo_root=C:\work\apex_repo_cache\ApexTraderAI
[headless] symbol=SOXL
[headless] steps=A,B,C,DPRIME,E,F
[headless] skip_stepe=0
[headless] auto_prepare_data=0 data_start=2010-01-01 data_end=2026-03-04
[StepA] start symbols=SOXL,SOXS
[StepA] data_dir=C:\work\apex_work\runs\gh22647391890_att1_sim_20260304_082244_ed934d0\data
[StepA] data_dir=C:\work\apex_work\runs\gh22647391890_att1_sim_20260304_082244_ed934d0\data src_csv=C:\work\apex_work\runs\gh22647391890_att1_sim_20260304_082244_ed934d0\data\prices_SOXL.csv
[StepA] data_dir=C:\work\apex_work\runs\gh22647391890_att1_sim_20260304_082244_ed934d0\data
[StepA] data_dir=C:\work\apex_work\runs\gh22647391890_att1_sim_20260304_082244_ed934d0\data src_csv=C:\work\apex_work\runs\gh22647391890_att1_sim_20260304_082244_ed934d0\data\prices_SOXS.csv
[StepA] done
[StepB] start
[StepB:mamba] wrote daily manifest -> C:\work\apex_work\runs\gh22647391890_att1_sim_20260304_082244_ed934d0\output\stepB\sim\stepB_daily_manifest_SOXL.csv rows=63
[StepB:mamba] wrote daily manifest -> C:\work\apex_work\runs\gh22647391890_att1_sim_20260304_082244_ed934d0\output\stepB\sim\stepB_daily_manifest_periodic_SOXL.csv rows=63
[StepB] agents: mamba=True
[StepB] done
[StepB] ensured: C:\work\apex_work\runs\gh22647391890_att1_sim_20260304_082244_ed934d0\output\stepB\sim\stepB_pred_time_all_SOXL.csv
[StepC] start
[StepC] price_path=C:\work\apex_work\runs\gh22647391890_att1_sim_20260304_082244_ed934d0\output\stepA\sim\stepA_prices_train_SOXL.csv
[StepC] wrote daily snapshots: 63 -> C:\work\apex_work\runs\gh22647391890_att1_sim_20260304_082244_ed934d0\output\stepC\sim\stepC_daily_manifest_SOXL.csv
[StepC] done
[StepDPrime] start
[StepDPrime] done
[StepE] start
[PIPELINE] status=failed steps=A,B,C,DPRIME,E,F output_root=C:\work\apex_work\runs\gh22647391890_att1_sim_20260304_082244_ed934d0\output
Traceback (most recent call last):
  File "C:\work\apex_repo_cache\ApexTraderAI\tools\run_pipeline.py", line 1699, in <module>
    raise SystemExit(main())
  File "C:\work\apex_repo_cache\ApexTraderAI\tools\run_pipeline.py", line 1675, in main
    step_result = _run_step_generic(step, app_config, symbol, date_range, results)
  File "C:\work\apex_repo_cache\ApexTraderAI\tools\run_pipeline.py", line 1253, in _run_step_generic
    return _call_with_best_effort(fn, ctx2)
  File "C:\work\apex_repo_cache\ApexTraderAI\tools\run_pipeline.py", line 487, in _call_with_best_effort
    return fn(**kwargs)
  File "C:\work\apex_repo_cache\ApexTraderAI\ai_core\services\step_e_service.py", line 183, in run
    self._run_one(cfg, date_range=date_range, symbol=symbol, mode=mode)
  File "C:\work\apex_repo_cache\ApexTraderAI\ai_core\services\step_e_service.py", line 197, in _run_one
    timing = self._timing()
  File "C:\work\apex_repo_cache\ApexTraderAI\ai_core\services\step_e_service.py", line 156, in _timing
    return t if isinstance(t, TimingLogger) else TimingLogger.disabled()
NameError: name 'TimingLogger' is not defined
[CMD] 'C:\Users\becky\AppData\Local\Programs\Python\Python310\python.exe' 'tools\run_pipeline.py' --symbol SOXL --steps A,B,C,DPRIME,E,F --test-start 2022-01-03 --train-years 8 --test-months 3 --mode sim --output-root 'C:\work\apex_work\runs\gh22647391890_att1_sim_20260304_082244_ed934d0\output' --data-dir 'C:\work\apex_work\runs\gh22647391890_att1_sim_20260304_082244_ed934d0\data' --auto-prepare-data 0 --enable-mamba
[RC] 1
[RC] 
[FAILED] command="python" tools\run_with_python.py tools\run_pipeline.py --symbol SOXL --steps "A,B,C,DPRIME,E,F" --test-start 2022-01-03 --train-years 8 --test-months 3 --mode sim --output-root "C:\work\apex_work\runs\gh22647391890_att1_sim_20260304_082244_ed934d0\output" --data-dir "C:\work\apex_work\runs\gh22647391890_att1_sim_20260304_082244_ed934d0\data" --auto-prepare-data 0  --enable-mamba
[FAILED] exit_code=
[FAILED] run_id=gh22647391890_att1_sim_20260304_082244_ed934d0
[FAILED] log=C:\work\apex_work\runs\gh22647391890_att1_sim_20260304_082244_ed934d0\logs\run_gh22647391890_att1_sim_20260304_082244_ed934d0.log
[STEP] package_diagnostics_to_onedrive
[INFO] diag_script=C:\work\apex_repo_cache\ApexTraderAI\scripts\package_diagnostics_to_onedrive.ps1
[PUBLISH] run_dir=C:\work\apex_work\runs\gh22647391890_att1_sim_20260304_082244_ed934d0
[PUBLISH] output_root=C:\work\apex_work\runs\gh22647391890_att1_sim_20260304_082244_ed934d0\output
[OK] package_diagnostics_to_onedrive
[OK] diag_dir=C:\Users\becky\OneDrive\ApexTraderAI\diagnostics

[STEP] resolve_latest_run_artifacts
[INFO] work_root=C:\work\apex_work\runs
[OK] resolve_latest_run_artifacts
[OK] run_dir=C:\work\apex_work\runs\gh22647391890_att1_sim_20260304_082244_ed934d0
[OK] run_id=gh22647391890_att1_sim_20260304_082244_ed934d0
```
