# ONE_TAP Public Run Report

- run_id: 22812206984
- sha: 05041c1222eb85337395e1f894a03dbef37b4f6f
- mode: sim
- symbols: SOXL,SOXS
- test_start: 2022-01-03
- train_years: 8
- test_months: 3
- output_root: C:/work/apex_work/runs/gh22812206984_att1_sim_20260308_113318_05041c1/output

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
Requirement already satisfied: wheel in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (0.46.3)
Requirement already satisfied: packaging>=24.0 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from wheel) (26.0)
Requirement already satisfied: numpy in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from -r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 1)) (1.25.1)
Requirement already satisfied: pandas in c:\users\becky\appdata\roaming\python\python310\site-packages (from -r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 2)) (2.3.3)
Requirement already satisfied: PyYAML in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from -r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 3)) (6.0)
Requirement already satisfied: yfinance in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from -r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 4)) (1.1.0)
Requirement already satisfied: matplotlib in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from -r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 5)) (3.7.2)
Requirement already satisfied: scikit-learn in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from -r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 6)) (1.7.2)
Requirement already satisfied: hdbscan in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from -r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 7)) (0.8.41)
Requirement already satisfied: gymnasium in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from -r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 8)) (1.2.3)
Requirement already satisfied: stable-baselines3 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from -r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 9)) (2.7.1)
Requirement already satisfied: python-dateutil>=2.8.2 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from pandas->-r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 2)) (2.8.2)
Requirement already satisfied: pytz>=2020.1 in c:\users\becky\appdata\roaming\python\python310\site-packages (from pandas->-r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 2)) (2025.2)
Requirement already satisfied: tzdata>=2022.7 in c:\users\becky\appdata\roaming\python\python310\site-packages (from pandas->-r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 2)) (2025.3)
Requirement already satisfied: requests>=2.31 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from yfinance->-r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 4)) (2.31.0)
Requirement already satisfied: multitasking>=0.0.7 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from yfinance->-r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 4)) (0.0.12)
Requirement already satisfied: platformdirs>=2.0.0 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from yfinance->-r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 4)) (4.9.1)
Requirement already satisfied: frozendict>=2.3.4 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from yfinance->-r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 4)) (2.4.7)
Requirement already satisfied: peewee>=3.16.2 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from yfinance->-r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 4)) (3.19.0)
Requirement already satisfied: beautifulsoup4>=4.11.1 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from yfinance->-r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 4)) (4.14.3)
Requirement already satisfied: curl_cffi<0.14,>=0.7 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from yfinance->-r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 4)) (0.13.0)
Requirement already satisfied: protobuf>=3.19.0 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from yfinance->-r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 4)) (4.23.4)
Requirement already satisfied: websockets>=13.0 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from yfinance->-r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 4)) (16.0)
Requirement already satisfied: cffi>=1.12.0 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from curl_cffi<0.14,>=0.7->yfinance->-r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 4)) (2.0.0)
Requirement already satisfied: certifi>=2024.2.2 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from curl_cffi<0.14,>=0.7->yfinance->-r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 4)) (2026.1.4)
Requirement already satisfied: contourpy>=1.0.1 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from matplotlib->-r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 5)) (1.1.0)
Requirement already satisfied: cycler>=0.10 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from matplotlib->-r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 5)) (0.11.0)
Requirement already satisfied: fonttools>=4.22.0 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from matplotlib->-r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 5)) (4.40.0)
Requirement already satisfied: kiwisolver>=1.0.1 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from matplotlib->-r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 5)) (1.4.4)
Requirement already satisfied: packaging>=20.0 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from matplotlib->-r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 5)) (26.0)
Requirement already satisfied: pillow>=6.2.0 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from matplotlib->-r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 5)) (10.0.0)
Requirement already satisfied: pyparsing<3.1,>=2.3.1 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from matplotlib->-r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 5)) (3.0.9)
Requirement already satisfied: scipy>=1.8.0 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from scikit-learn->-r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 6)) (1.11.1)
Requirement already satisfied: joblib>=1.2.0 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from scikit-learn->-r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 6)) (1.3.1)
Requirement already satisfied: threadpoolctl>=3.1.0 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from scikit-learn->-r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 6)) (3.1.0)
Requirement already satisfied: cloudpickle>=1.2.0 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from gymnasium->-r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 8)) (3.1.2)
Requirement already satisfied: typing-extensions>=4.3.0 in c:\users\becky\appdata\roaming\python\python310\site-packages (from gymnasium->-r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 8)) (4.15.0)
Requirement already satisfied: farama-notifications>=0.0.1 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from gymnasium->-r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 8)) (0.0.4)
Requirement already satisfied: torch<3.0,>=2.3 in c:\users\becky\appdata\roaming\python\python310\site-packages (from stable-baselines3->-r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 9)) (2.10.0+cu126)
Requirement already satisfied: filelock in c:\users\becky\appdata\roaming\python\python310\site-packages (from torch<3.0,>=2.3->stable-baselines3->-r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 9)) (3.24.0)
Requirement already satisfied: sympy>=1.13.3 in c:\users\becky\appdata\roaming\python\python310\site-packages (from torch<3.0,>=2.3->stable-baselines3->-r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 9)) (1.14.0)
Requirement already satisfied: networkx>=2.5.1 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from torch<3.0,>=2.3->stable-baselines3->-r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 9)) (3.1)
Requirement already satisfied: jinja2 in c:\users\becky\appdata\roaming\python\python310\site-packages (from torch<3.0,>=2.3->stable-baselines3->-r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 9)) (3.1.6)
Requirement already satisfied: fsspec>=0.8.5 in c:\users\becky\appdata\roaming\python\python310\site-packages (from torch<3.0,>=2.3->stable-baselines3->-r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 9)) (2026.2.0)
Requirement already satisfied: soupsieve>=1.6.1 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from beautifulsoup4>=4.11.1->yfinance->-r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 4)) (2.8.3)
Requirement already satisfied: pycparser in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from cffi>=1.12.0->curl_cffi<0.14,>=0.7->yfinance->-r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 4)) (3.0)
Requirement already satisfied: six>=1.5 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from python-dateutil>=2.8.2->pandas->-r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 2)) (1.16.0)
Requirement already satisfied: charset-normalizer<4,>=2 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from requests>=2.31->yfinance->-r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 4)) (3.2.0)
Requirement already satisfied: idna<4,>=2.5 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from requests>=2.31->yfinance->-r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 4)) (3.4)
Requirement already satisfied: urllib3<3,>=1.21.1 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from requests>=2.31->yfinance->-r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 4)) (2.0.3)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in c:\users\becky\appdata\local\programs\python\python310\lib\site-packages (from sympy>=1.13.3->torch<3.0,>=2.3->stable-baselines3->-r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 9)) (1.3.0)
Requirement already satisfied: MarkupSafe>=2.0 in c:\users\becky\appdata\roaming\python\python310\site-packages (from jinja2->torch<3.0,>=2.3->stable-baselines3->-r C:\work\apex_repo_cache\ApexTraderAI\requirements.txt (line 9)) (3.0.3)
[PREPARE_ENV] prepare_data begin prepCmd=cd '/mnt/c/work/apex_repo_cache/ApexTraderAI' && '/home/becky/miniforge3/envs/mamba_cuda/bin/python' tools/run_with_python.py tools/prepare_data.py --symbols SOXL,SOXS --start 2013-12-27 --end 2022-04-10 --force --data-dir '/mnt/c/work/apex_work/runs/gh22812206984_att1_sim_20260308_113318_05041c1/data'
[prepare_data] wrote /mnt/c/work/apex_work/runs/gh22812206984_att1_sim_20260308_113318_05041c1/data/prices_SOXL.csv (2086 rows)
[prepare_data] wrote /mnt/c/work/apex_work/runs/gh22812206984_att1_sim_20260308_113318_05041c1/data/prices_SOXS.csv (2086 rows)
[CMD] /home/becky/miniforge3/envs/mamba_cuda/bin/python tools/prepare_data.py --symbols SOXL,SOXS --start 2013-12-27 --end 2022-04-10 --force --data-dir /mnt/c/work/apex_work/runs/gh22812206984_att1_sim_20260308_113318_05041c1/data
[RC] 0
[INFO] work_root=C:\work\apex_work\runs
[INFO] latest_run_dir=C:\work\apex_work\runs\gh22812206984_att1_sim_20260308_113318_05041c1
[INFO] run_log=C:\work\apex_work\runs\gh22812206984_att1_sim_20260308_113318_05041c1\logs\run_gh22812206984_att1_sim_20260308_113318_05041c1.log
---- run_log_tail (last 350 lines) ----
[prepare_data] wrote /mnt/c/work/apex_work/runs/gh22812206984_att1_sim_20260308_113318_05041c1/data/prices_SOXL.csv (2086 rows)
[prepare_data] wrote /mnt/c/work/apex_work/runs/gh22812206984_att1_sim_20260308_113318_05041c1/data/prices_SOXS.csv (2086 rows)
[CMD] /home/becky/miniforge3/envs/mamba_cuda/bin/python tools/prepare_data.py --symbols SOXL,SOXS --start 2013-12-27 --end 2022-04-10 --force --data-dir /mnt/c/work/apex_work/runs/gh22812206984_att1_sim_20260308_113318_05041c1/data
[RC] 0
[STEP] package_diagnostics_to_onedrive
[INFO] diag_script=C:\work\apex_repo_cache\ApexTraderAI\scripts\package_diagnostics_to_onedrive.ps1
[PUBLISH] run_dir=C:\work\apex_work\runs\gh22812206984_att1_sim_20260308_113318_05041c1
[PUBLISH] output_root=C:\work\apex_work\runs\gh22812206984_att1_sim_20260308_113318_05041c1\output
[WARN] publish issue: D' state CSV missing under output/stepDprime/<mode> or stepD_prime/<mode>.
[WARN] publish issue: D' embeddings CSV missing under output/stepDprime/<mode>/embeddings or stepD_prime/<mode>/embeddings.
[OK] package_diagnostics_to_onedrive
[OK] diag_dir=C:\Users\becky\OneDrive\ApexTraderAI\diagnostics

[STEP] resolve_latest_run_artifacts
[INFO] work_root=C:\work\apex_work\runs
[OK] resolve_latest_run_artifacts
[OK] run_dir=C:\work\apex_work\runs\gh22812206984_att1_sim_20260308_113318_05041c1
[OK] run_id=gh22812206984_att1_sim_20260308_113318_05041c1
```
