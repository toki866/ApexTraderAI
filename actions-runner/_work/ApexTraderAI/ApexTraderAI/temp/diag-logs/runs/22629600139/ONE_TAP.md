# ONE_TAP Public Run Report

- run_id: 22629600139
- sha: ecaafc907e36f3d19f7cc7dc8929aa0ae2d5613a
- mode: live
- symbols: SOXL,SOXS
- test_start: 2022-01-03
- train_years: 8
- test_months: 3
- output_root: C:/work/apex_work/runs/gh22629600139_att1_live_20260304_001842_ecaafc9/output

## StepE metrics
not found

## StepF metrics
| step  | mode | source_csv | status    |
| ----- | ---- | ---------- | --------- |
| StepF | live | not found  | not found |

## Key CSV files
- No target CSV files found

## Error summary
```text
Requirement already satisfied: stable-baselines3 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from -r requirements.txt (line 9)) (2.7.1)
Requirement already satisfied: python-dateutil>=2.8.2 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from pandas->-r requirements.txt (line 2)) (2.8.2)
Requirement already satisfied: pytz>=2020.1 in c:\users\becky\appdata\roaming\python\python310\site-packages (from pandas->-r requirements.txt (line 2)) (2025.2)
Requirement already satisfied: tzdata>=2022.7 in c:\users\becky\appdata\roaming\python\python310\site-packages (from pandas->-r requirements.txt (line 2)) (2025.3)
Requirement already satisfied: requests>=2.31 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from yfinance->-r requirements.txt (line 4)) (2.31.0)
Requirement already satisfied: multitasking>=0.0.7 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from yfinance->-r requirements.txt (line 4)) (0.0.12)
Requirement already satisfied: platformdirs>=2.0.0 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from yfinance->-r requirements.txt (line 4)) (4.9.1)
Requirement already satisfied: frozendict>=2.3.4 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from yfinance->-r requirements.txt (line 4)) (2.4.7)
Requirement already satisfied: peewee>=3.16.2 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from yfinance->-r requirements.txt (line 4)) (3.19.0)
Requirement already satisfied: beautifulsoup4>=4.11.1 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from yfinance->-r requirements.txt (line 4)) (4.14.3)
Requirement already satisfied: curl_cffi<0.14,>=0.7 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from yfinance->-r requirements.txt (line 4)) (0.13.0)
Requirement already satisfied: protobuf>=3.19.0 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from yfinance->-r requirements.txt (line 4)) (4.23.4)
Requirement already satisfied: websockets>=13.0 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from yfinance->-r requirements.txt (line 4)) (16.0)
Requirement already satisfied: cffi>=1.12.0 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from curl_cffi<0.14,>=0.7->yfinance->-r requirements.txt (line 4)) (2.0.0)
Requirement already satisfied: certifi>=2024.2.2 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from curl_cffi<0.14,>=0.7->yfinance->-r requirements.txt (line 4)) (2026.1.4)
Requirement already satisfied: contourpy>=1.0.1 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from matplotlib->-r requirements.txt (line 5)) (1.1.0)
Requirement already satisfied: cycler>=0.10 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from matplotlib->-r requirements.txt (line 5)) (0.11.0)
Requirement already satisfied: fonttools>=4.22.0 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from matplotlib->-r requirements.txt (line 5)) (4.40.0)
Requirement already satisfied: kiwisolver>=1.0.1 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from matplotlib->-r requirements.txt (line 5)) (1.4.4)
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

[CMD] "python" tools\run_with_python.py tools\prepare_data.py --symbols SOXL,SOXS --start 2013-12-27 --end 2022-04-10 --force --data-dir "C:\work\apex_work\runs\gh22629600139_att1_live_20260304_001842_ecaafc9\data"
[prepare_data] wrote C:\work\apex_work\runs\gh22629600139_att1_live_20260304_001842_ecaafc9\data\prices_SOXL.csv (2086 rows)
[prepare_data] wrote C:\work\apex_work\runs\gh22629600139_att1_live_20260304_001842_ecaafc9\data\prices_SOXS.csv (2086 rows)
[CMD] 'C:\Users\becky\AppData\Local\Programs\Python\Python310\python.exe' 'tools\prepare_data.py' --symbols SOXL,SOXS --start 2013-12-27 --end 2022-04-10 --force --data-dir 'C:\work\apex_work\runs\gh22629600139_att1_live_20260304_001842_ecaafc9\data'
[RC] 0

[CMD] "python" tools\run_with_python.py tools\run_pipeline.py --symbol SOXL --steps "A,B,C,DPRIME,E,F" --test-start 2022-01-03 --train-years 8 --test-months 3 --mode live --output-root "C:\work\apex_work\runs\gh22629600139_att1_live_20260304_001842_ecaafc9\output" --data-dir "C:\work\apex_work\runs\gh22629600139_att1_live_20260304_001842_ecaafc9\data" --auto-prepare-data 0  --enable-mamba
Traceback (most recent call last):
  File "C:\work\apex_repo_cache\ApexTraderAI\tools\run_pipeline.py", line 1699, in <module>
    raise SystemExit(main())
  File "C:\work\apex_repo_cache\ApexTraderAI\tools\run_pipeline.py", line 1429, in main
    timing = TimingLogger(
  File "C:\work\apex_repo_cache\ApexTraderAI\ai_core\utils\timing_logger.py", line 34, in __init__
    self.host = os.uname().nodename
AttributeError: module 'os' has no attribute 'uname'. Did you mean: 'name'?
[CMD] 'C:\Users\becky\AppData\Local\Programs\Python\Python310\python.exe' 'tools\run_pipeline.py' --symbol SOXL --steps A,B,C,DPRIME,E,F --test-start 2022-01-03 --train-years 8 --test-months 3 --mode live --output-root 'C:\work\apex_work\runs\gh22629600139_att1_live_20260304_001842_ecaafc9\output' --data-dir 'C:\work\apex_work\runs\gh22629600139_att1_live_20260304_001842_ecaafc9\data' --auto-prepare-data 0 --enable-mamba
[RC] 1
[RC] 
[FAILED] command="python" tools\run_with_python.py tools\run_pipeline.py --symbol SOXL --steps "A,B,C,DPRIME,E,F" --test-start 2022-01-03 --train-years 8 --test-months 3 --mode live --output-root "C:\work\apex_work\runs\gh22629600139_att1_live_20260304_001842_ecaafc9\output" --data-dir "C:\work\apex_work\runs\gh22629600139_att1_live_20260304_001842_ecaafc9\data" --auto-prepare-data 0  --enable-mamba
[FAILED] exit_code=
[FAILED] run_id=gh22629600139_att1_live_20260304_001842_ecaafc9
[FAILED] log=C:\work\apex_work\runs\gh22629600139_att1_live_20260304_001842_ecaafc9\logs\run_gh22629600139_att1_live_20260304_001842_ecaafc9.log
[STEP] package_diagnostics_to_onedrive
[INFO] diag_script=C:\work\apex_repo_cache\ApexTraderAI\scripts\package_diagnostics_to_onedrive.ps1
[PUBLISH] run_dir=C:\work\apex_work\runs\gh22629600139_att1_live_20260304_001842_ecaafc9
[PUBLISH] output_root=C:\work\apex_work\runs\gh22629600139_att1_live_20260304_001842_ecaafc9\output
[WARN] publish issue: D' state CSV missing under output/stepD_prime/<mode>.
[WARN] publish issue: D' embeddings CSV missing under output/stepD_prime/<mode>/embeddings.
[OK] package_diagnostics_to_onedrive
[OK] diag_dir=C:\Users\becky\OneDrive\ApexTraderAI\diagnostics

[STEP] resolve_latest_run_artifacts
[INFO] work_root=C:\work\apex_work\runs
[OK] resolve_latest_run_artifacts
[OK] run_dir=C:\work\apex_work\runs\gh22629600139_att1_live_20260304_001842_ecaafc9
[OK] run_id=gh22629600139_att1_live_20260304_001842_ecaafc9
```
