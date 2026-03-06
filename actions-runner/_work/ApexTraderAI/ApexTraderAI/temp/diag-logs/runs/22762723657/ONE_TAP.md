# ONE_TAP Public Run Report

- run_id: 22762723657
- sha: 1379031f95faecc58ac730099b6d75e3dbea8a42
- mode: sim
- symbols: SOXL,SOXS
- test_start: 2022-01-03
- train_years: 8
- test_months: 3
- output_root: C:/work/apex_work/runs/gh22762723657_att1_sim_20260306_210607_1379031/output

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
Requirement already satisfied: packaging>=20.0 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from matplotlib->-r requirements.txt (line 5)) (26.0)
Requirement already satisfied: pillow>=6.2.0 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from matplotlib->-r requirements.txt (line 5)) (10.0.0)
Requirement already satisfied: pyparsing<3.1,>=2.3.1 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from matplotlib->-r requirements.txt (line 5)) (3.0.9)
Requirement already satisfied: scipy>=1.8.0 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from scikit-learn->-r requirements.txt (line 6)) (1.11.1)
Requirement already satisfied: joblib>=1.2.0 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from scikit-learn->-r requirements.txt (line 6)) (1.3.1)
Requirement already satisfied: threadpoolctl>=3.1.0 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from scikit-learn->-r requirements.txt (line 6)) (3.1.0)
Requirement already satisfied: cloudpickle>=1.2.0 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from gymnasium->-r requirements.txt (line 8)) (3.1.2)
Requirement already satisfied: typing-extensions>=4.3.0 in c:\users\becky\appdata\roaming\python\python310\site-packages (from gymnasium->-r requirements.txt (line 8)) (4.15.0)
Requirement already satisfied: farama-notifications>=0.0.1 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from gymnasium->-r requirements.txt (line 8)) (0.0.4)
Requirement already satisfied: torch<3.0,>=2.3 in c:\users\becky\appdata\roaming\python\python310\site-packages (from stable-baselines3->-r requirements.txt (line 9)) (2.10.0+cu126)
Requirement already satisfied: filelock in c:\users\becky\appdata\roaming\python\python310\site-packages (from torch<3.0,>=2.3->stable-baselines3->-r requirements.txt (line 9)) (3.24.0)
Requirement already satisfied: sympy>=1.13.3 in c:\users\becky\appdata\roaming\python\python310\site-packages (from torch<3.0,>=2.3->stable-baselines3->-r requirements.txt (line 9)) (1.14.0)
Requirement already satisfied: networkx>=2.5.1 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from torch<3.0,>=2.3->stable-baselines3->-r requirements.txt (line 9)) (3.1)
Requirement already satisfied: jinja2 in c:\users\becky\appdata\roaming\python\python310\site-packages (from torch<3.0,>=2.3->stable-baselines3->-r requirements.txt (line 9)) (3.1.6)
Requirement already satisfied: fsspec>=0.8.5 in c:\users\becky\appdata\roaming\python\python310\site-packages (from torch<3.0,>=2.3->stable-baselines3->-r requirements.txt (line 9)) (2026.2.0)
Requirement already satisfied: soupsieve>=1.6.1 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from beautifulsoup4>=4.11.1->yfinance->-r requirements.txt (line 4)) (2.8.3)
Requirement already satisfied: pycparser in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from cffi>=1.12.0->curl_cffi<0.14,>=0.7->yfinance->-r requirements.txt (line 4)) (3.0)
Requirement already satisfied: six>=1.5 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from python-dateutil>=2.8.2->pandas->-r requirements.txt (line 2)) (1.16.0)
Requirement already satisfied: charset-normalizer<4,>=2 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from requests>=2.31->yfinance->-r requirements.txt (line 4)) (3.2.0)
Requirement already satisfied: idna<4,>=2.5 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from requests>=2.31->yfinance->-r requirements.txt (line 4)) (3.4)
Requirement already satisfied: urllib3<3,>=1.21.1 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from requests>=2.31->yfinance->-r requirements.txt (line 4)) (2.0.3)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from sympy>=1.13.3->torch<3.0,>=2.3->stable-baselines3->-r requirements.txt (line 9)) (1.3.0)
Requirement already satisfied: MarkupSafe>=2.0 in c:\users\becky\appdata\roaming\python\python310\site-packages (from jinja2->torch<3.0,>=2.3->stable-baselines3->-r requirements.txt (line 9)) (3.0.3)
[RUN] mamba-ssm install step removed for Windows runner (Linux-only package)
[RUN] mamba_ssm unavailable on this runner; disable --enable-mamba

[CMD] "python" tools\run_with_python.py tools\prepare_data.py --symbols SOXL,SOXS --start 2013-12-27 --end 2022-04-10 --force --data-dir "C:\work\apex_work\runs\gh22762723657_att1_sim_20260306_210607_1379031\data"
[prepare_data] wrote C:\work\apex_work\runs\gh22762723657_att1_sim_20260306_210607_1379031\data\prices_SOXL.csv (2086 rows)
[prepare_data] wrote C:\work\apex_work\runs\gh22762723657_att1_sim_20260306_210607_1379031\data\prices_SOXS.csv (2086 rows)
[CMD] 'C:\Users\becky\AppData\Local\Programs\Python\Python310\python.exe' 'tools\prepare_data.py' --symbols SOXL,SOXS --start 2013-12-27 --end 2022-04-10 --force --data-dir 'C:\work\apex_work\runs\gh22762723657_att1_sim_20260306_210607_1379031\data'
[RC] 0

[headless] output_root_resolved=C:\work\apex_work\runs\gh22762723657_att1_sim_20260306_210607_1379031\output cfg_output_root=C:\work\apex_work\runs\gh22762723657_att1_sim_20260306_210607_1379031\output cfg_data_output_root=C:\work\apex_work\runs\gh22762723657_att1_sim_20260306_210607_1379031\output
[headless] data ready: SOXL,SOXS
[headless] StepE default config injected: agents=dprime_bnf_h01,dprime_bnf_h02,dprime_bnf_3scale,dprime_mix_h01,dprime_mix_h02,dprime_mix_3scale,dprime_all_features_h01,dprime_all_features_h02,dprime_all_features_h03,dprime_all_features_3scale seed=42+idx device=auto
[headless] StepF default config injected: agents=dprime_bnf_h01,dprime_bnf_h02,dprime_bnf_3scale,dprime_mix_h01,dprime_mix_h02,dprime_mix_3scale,dprime_all_features_h01,dprime_all_features_h02,dprime_all_features_h03,dprime_all_features_3scale safe_set=dprime_bnf_h01,dprime_all_features_h01,dprime_mix_3scale router(beta=1.0,ema=0.5,cost_bps=15.0)
[headless] repo_root=C:\work\apex_repo_cache\ApexTraderAI
[headless] symbol=SOXL
[headless] steps=A,B,C,DPRIME,E,F
[headless] skip_stepe=0
[headless] auto_prepare_data=0 data_start=2010-01-01 data_end=2026-03-06
[StepA] start symbols=SOXL,SOXS
[StepA] data_dir=C:\work\apex_work\runs\gh22762723657_att1_sim_20260306_210607_1379031\data
[StepA] data_dir=C:\work\apex_work\runs\gh22762723657_att1_sim_20260306_210607_1379031\data src_csv=C:\work\apex_work\runs\gh22762723657_att1_sim_20260306_210607_1379031\data\prices_SOXL.csv
[StepA] data_dir=C:\work\apex_work\runs\gh22762723657_att1_sim_20260306_210607_1379031\data
[StepA] data_dir=C:\work\apex_work\runs\gh22762723657_att1_sim_20260306_210607_1379031\data src_csv=C:\work\apex_work\runs\gh22762723657_att1_sim_20260306_210607_1379031\data\prices_SOXS.csv
[StepA] done
[StepB] start
[PIPELINE] status=failed steps=A,B,C,DPRIME,E,F output_root=C:\work\apex_work\runs\gh22762723657_att1_sim_20260306_210607_1379031\output
Traceback (most recent call last):
  File "C:\work\apex_repo_cache\ApexTraderAI\tools\run_pipeline.py", line 1696, in <module>
    raise SystemExit(main())
  File "C:\work\apex_repo_cache\ApexTraderAI\tools\run_pipeline.py", line 1643, in main
    results["stepB_result"] = _run_stepB(app_config, symbol, date_range, enable_mamba, args.enable_mamba_periodic, args.mamba_lookback, mamba_horizons_list)
  File "C:\work\apex_repo_cache\ApexTraderAI\tools\run_pipeline.py", line 1185, in _run_stepB
    return fn(cfg)
  File "C:\work\apex_repo_cache\ApexTraderAI\ai_core\services\step_b_service.py", line 211, in run
    raise ValueError("No agents enabled in StepBConfig (mamba disabled).")
ValueError: No agents enabled in StepBConfig (mamba disabled).
[CMD] 'C:\Users\becky\AppData\Local\Programs\Python\Python310\python.exe' 'tools\run_pipeline.py' --symbol SOXL --steps A,B,C,DPRIME,E,F --test-start 2022-01-03 --train-years 8 --test-months 3 --mode sim --output-root 'C:\work\apex_work\runs\gh22762723657_att1_sim_20260306_210607_1379031\output' --data-dir 'C:\work\apex_work\runs\gh22762723657_att1_sim_20260306_210607_1379031\data' --auto-prepare-data 0
[RC] 1
[RC] 
[FAILED] exit_code=
[FAILED] run_id=gh22762723657_att1_sim_20260306_210607_1379031
[FAILED] log=C:\work\apex_work\runs\gh22762723657_att1_sim_20260306_210607_1379031\logs\run_gh22762723657_att1_sim_20260306_210607_1379031.log
[STEP] package_diagnostics_to_onedrive
[INFO] diag_script=C:\work\apex_repo_cache\ApexTraderAI\scripts\package_diagnostics_to_onedrive.ps1
[PUBLISH] run_dir=C:\work\apex_work\runs\gh22762723657_att1_sim_20260306_210607_1379031
[PUBLISH] output_root=C:\work\apex_work\runs\gh22762723657_att1_sim_20260306_210607_1379031\output
[WARN] publish issue: D' state CSV missing under output/stepD_prime/<mode>.
[WARN] publish issue: D' embeddings CSV missing under output/stepD_prime/<mode>/embeddings.
[OK] package_diagnostics_to_onedrive
[OK] diag_dir=C:\Users\becky\OneDrive\ApexTraderAI\diagnostics

[STEP] resolve_latest_run_artifacts
[INFO] work_root=C:\work\apex_work\runs
[OK] resolve_latest_run_artifacts
[OK] run_dir=C:\work\apex_work\runs\gh22762723657_att1_sim_20260306_210607_1379031
[OK] run_id=gh22762723657_att1_sim_20260306_210607_1379031
```
