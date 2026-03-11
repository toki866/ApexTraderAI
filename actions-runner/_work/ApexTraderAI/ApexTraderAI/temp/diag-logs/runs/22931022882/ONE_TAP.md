# ONE_TAP Public Run Report

- run_id: 22931022882
- sha: 494d0efef3c0170a431c40ac2099725549fb41e5
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
Collecting PyWavelets>=1.6 (from -r requirements.txt (line 10))
  Downloading pywavelets-1.8.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (9.0 kB)
Requirement already satisfied: python-dateutil>=2.8.2 in /home/becky/miniforge3/envs/mamba_cuda/lib/python3.10/site-packages (from pandas->-r requirements.txt (line 2)) (2.9.0.post0)
Requirement already satisfied: pytz>=2020.1 in /home/becky/miniforge3/envs/mamba_cuda/lib/python3.10/site-packages (from pandas->-r requirements.txt (line 2)) (2026.1.post1)
Requirement already satisfied: tzdata>=2022.7 in /home/becky/miniforge3/envs/mamba_cuda/lib/python3.10/site-packages (from pandas->-r requirements.txt (line 2)) (2025.3)
Requirement already satisfied: requests>=2.31 in /home/becky/miniforge3/envs/mamba_cuda/lib/python3.10/site-packages (from yfinance->-r requirements.txt (line 4)) (2.32.5)
Requirement already satisfied: multitasking>=0.0.7 in /home/becky/miniforge3/envs/mamba_cuda/lib/python3.10/site-packages (from yfinance->-r requirements.txt (line 4)) (0.0.12)
Requirement already satisfied: platformdirs>=2.0.0 in /home/becky/miniforge3/envs/mamba_cuda/lib/python3.10/site-packages (from yfinance->-r requirements.txt (line 4)) (4.9.4)
Requirement already satisfied: frozendict>=2.3.4 in /home/becky/miniforge3/envs/mamba_cuda/lib/python3.10/site-packages (from yfinance->-r requirements.txt (line 4)) (2.4.7)
Requirement already satisfied: peewee>=3.16.2 in /home/becky/miniforge3/envs/mamba_cuda/lib/python3.10/site-packages (from yfinance->-r requirements.txt (line 4)) (4.0.1)
Requirement already satisfied: beautifulsoup4>=4.11.1 in /home/becky/miniforge3/envs/mamba_cuda/lib/python3.10/site-packages (from yfinance->-r requirements.txt (line 4)) (4.14.3)
Requirement already satisfied: curl_cffi<0.14,>=0.7 in /home/becky/miniforge3/envs/mamba_cuda/lib/python3.10/site-packages (from yfinance->-r requirements.txt (line 4)) (0.13.0)
Requirement already satisfied: protobuf>=3.19.0 in /home/becky/miniforge3/envs/mamba_cuda/lib/python3.10/site-packages (from yfinance->-r requirements.txt (line 4)) (7.34.0)
Requirement already satisfied: websockets>=13.0 in /home/becky/miniforge3/envs/mamba_cuda/lib/python3.10/site-packages (from yfinance->-r requirements.txt (line 4)) (16.0)
Requirement already satisfied: cffi>=1.12.0 in /home/becky/miniforge3/envs/mamba_cuda/lib/python3.10/site-packages (from curl_cffi<0.14,>=0.7->yfinance->-r requirements.txt (line 4)) (2.0.0)
Requirement already satisfied: certifi>=2024.2.2 in /home/becky/miniforge3/envs/mamba_cuda/lib/python3.10/site-packages (from curl_cffi<0.14,>=0.7->yfinance->-r requirements.txt (line 4)) (2026.2.25)
Requirement already satisfied: contourpy>=1.0.1 in /home/becky/miniforge3/envs/mamba_cuda/lib/python3.10/site-packages (from matplotlib->-r requirements.txt (line 5)) (1.3.2)
Requirement already satisfied: cycler>=0.10 in /home/becky/miniforge3/envs/mamba_cuda/lib/python3.10/site-packages (from matplotlib->-r requirements.txt (line 5)) (0.12.1)
Requirement already satisfied: fonttools>=4.22.0 in /home/becky/miniforge3/envs/mamba_cuda/lib/python3.10/site-packages (from matplotlib->-r requirements.txt (line 5)) (4.61.1)
Requirement already satisfied: kiwisolver>=1.3.1 in /home/becky/miniforge3/envs/mamba_cuda/lib/python3.10/site-packages (from matplotlib->-r requirements.txt (line 5)) (1.4.9)
Requirement already satisfied: packaging>=20.0 in /home/becky/miniforge3/envs/mamba_cuda/lib/python3.10/site-packages (from matplotlib->-r requirements.txt (line 5)) (26.0)
Requirement already satisfied: pillow>=8 in /home/becky/miniforge3/envs/mamba_cuda/lib/python3.10/site-packages (from matplotlib->-r requirements.txt (line 5)) (12.1.1)
Requirement already satisfied: pyparsing>=3 in /home/becky/miniforge3/envs/mamba_cuda/lib/python3.10/site-packages (from matplotlib->-r requirements.txt (line 5)) (3.3.2)
Requirement already satisfied: scipy>=1.8.0 in /home/becky/miniforge3/envs/mamba_cuda/lib/python3.10/site-packages (from scikit-learn->-r requirements.txt (line 6)) (1.15.3)
Requirement already satisfied: joblib>=1.2.0 in /home/becky/miniforge3/envs/mamba_cuda/lib/python3.10/site-packages (from scikit-learn->-r requirements.txt (line 6)) (1.5.3)
Requirement already satisfied: threadpoolctl>=3.1.0 in /home/becky/miniforge3/envs/mamba_cuda/lib/python3.10/site-packages (from scikit-learn->-r requirements.txt (line 6)) (3.6.0)
Requirement already satisfied: cloudpickle>=1.2.0 in /home/becky/miniforge3/envs/mamba_cuda/lib/python3.10/site-packages (from gymnasium->-r requirements.txt (line 8)) (3.1.2)
Requirement already satisfied: typing-extensions>=4.3.0 in /home/becky/miniforge3/envs/mamba_cuda/lib/python3.10/site-packages (from gymnasium->-r requirements.txt (line 8)) (4.15.0)
Requirement already satisfied: farama-notifications>=0.0.1 in /home/becky/miniforge3/envs/mamba_cuda/lib/python3.10/site-packages (from gymnasium->-r requirements.txt (line 8)) (0.0.4)
Requirement already satisfied: torch<3.0,>=2.3 in /home/becky/miniforge3/envs/mamba_cuda/lib/python3.10/site-packages (from stable-baselines3->-r requirements.txt (line 9)) (2.5.1)
Requirement already satisfied: filelock in /home/becky/miniforge3/envs/mamba_cuda/lib/python3.10/site-packages (from torch<3.0,>=2.3->stable-baselines3->-r requirements.txt (line 9)) (3.25.0)
Requirement already satisfied: networkx in /home/becky/miniforge3/envs/mamba_cuda/lib/python3.10/site-packages (from torch<3.0,>=2.3->stable-baselines3->-r requirements.txt (line 9)) (3.4.2)
Requirement already satisfied: jinja2 in /home/becky/miniforge3/envs/mamba_cuda/lib/python3.10/site-packages (from torch<3.0,>=2.3->stable-baselines3->-r requirements.txt (line 9)) (3.1.6)
Requirement already satisfied: fsspec in /home/becky/miniforge3/envs/mamba_cuda/lib/python3.10/site-packages (from torch<3.0,>=2.3->stable-baselines3->-r requirements.txt (line 9)) (2026.2.0)
Requirement already satisfied: sympy==1.13.1 in /home/becky/miniforge3/envs/mamba_cuda/lib/python3.10/site-packages (from torch<3.0,>=2.3->stable-baselines3->-r requirements.txt (line 9)) (1.13.1)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /home/becky/miniforge3/envs/mamba_cuda/lib/python3.10/site-packages (from sympy==1.13.1->torch<3.0,>=2.3->stable-baselines3->-r requirements.txt (line 9)) (1.3.0)
Requirement already satisfied: soupsieve>=1.6.1 in /home/becky/miniforge3/envs/mamba_cuda/lib/python3.10/site-packages (from beautifulsoup4>=4.11.1->yfinance->-r requirements.txt (line 4)) (2.8.3)
Requirement already satisfied: pycparser in /home/becky/miniforge3/envs/mamba_cuda/lib/python3.10/site-packages (from cffi>=1.12.0->curl_cffi<0.14,>=0.7->yfinance->-r requirements.txt (line 4)) (3.0)
Requirement already satisfied: six>=1.5 in /home/becky/miniforge3/envs/mamba_cuda/lib/python3.10/site-packages (from python-dateutil>=2.8.2->pandas->-r requirements.txt (line 2)) (1.17.0)
Requirement already satisfied: charset_normalizer<4,>=2 in /home/becky/miniforge3/envs/mamba_cuda/lib/python3.10/site-packages (from requests>=2.31->yfinance->-r requirements.txt (line 4)) (3.4.5)
Requirement already satisfied: idna<4,>=2.5 in /home/becky/miniforge3/envs/mamba_cuda/lib/python3.10/site-packages (from requests>=2.31->yfinance->-r requirements.txt (line 4)) (3.11)
Requirement already satisfied: urllib3<3,>=1.21.1 in /home/becky/miniforge3/envs/mamba_cuda/lib/python3.10/site-packages (from requests>=2.31->yfinance->-r requirements.txt (line 4)) (2.6.3)
Requirement already satisfied: MarkupSafe>=2.0 in /home/becky/miniforge3/envs/mamba_cuda/lib/python3.10/site-packages (from jinja2->torch<3.0,>=2.3->stable-baselines3->-r requirements.txt (line 9)) (3.0.3)
Downloading pywavelets-1.8.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (4.5 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.5/4.5 MB 7.2 MB/s  0:00:00
Installing collected packages: PyWavelets
Successfully installed PyWavelets-1.8.0
wsl.exe : bash: line 4: warning: here-document at line 1 delimited by end-of-file (wanted `PY
At C:\Users\becky\OneDrive\デスクトップ\Python\apex-trader-ai\actions-runner\_work\_temp\34e94495-76a0-4025-b7b8-2fdb15e47eb9
.ps1:152 char:1
+ wsl.exe -d $env:WSL_DISTRO -- bash -c $wslPywtCheckCmd 2>&1 | Tee-Obj ...
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : NotSpecified: (bash: line 4: w...ile (wanted `PY:String) [], RemoteException
    + FullyQualifiedErrorId : NativeCommandError
 
')
pywt_available=1 1.8.0
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
NameError: name 'PY' is not defined
[INFO] work_root=C:\work\apex_work\runs
[INFO] session_log_root=C:\work\apex_work\session_logs
[INFO] latest_run_dir=C:\work\apex_work\runs\gh22931022882_att1_sim_20260311_094136_494d0ef
[INFO] run_log_selected=C:\work\apex_work\session_logs\22931022882_att1\run_gh22931022882_att1_sim_20260311_094136_494d0ef.log
---- run_log_tail (last 350 lines) ----
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
[OK] run_id=gh22931022882_att1_sim_20260311_094136_494d0ef
```
