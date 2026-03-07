# ONE_TAP Public Run Report

- run_id: 22796461551
- sha: e15a2095928ce120bce1f85d56ae8d48faa29344
- mode: sim
- symbols: SOXL,SOXS
- test_start: 2022-01-03
- train_years: 8
- test_months: 3
- output_root: C:/work/apex_work/runs/gh22796461551_att1_sim_20260307_182743_e15a209/output

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
|    learning_rate        | 0.0003   |
|    loss                 | -0.0237  |
|    n_updates            | 3520     |
|    policy_gradient_loss | 0.0136   |
|    std                  | 0.0284   |
|    value_loss           | 3.98e-06 |
--------------------------------------
---------------------------------------
| time/                   |           |
|    fps                  | 213       |
|    iterations           | 178       |
|    time_elapsed         | 1711      |
|    total_timesteps      | 364544    |
| train/                  |           |
|    approx_kl            | 3.6531696 |
|    clip_fraction        | 0.757     |
|    clip_range           | 0.2       |
|    entropy_loss         | 2.13      |
|    explained_variance   | 1         |
|    learning_rate        | 0.0003    |
|    loss                 | -0.0118   |
|    n_updates            | 3540      |
|    policy_gradient_loss | 0.0386    |
|    std                  | 0.0287    |
|    value_loss           | 3.54e-06  |
---------------------------------------
---------------------------------------
| time/                   |           |
|    fps                  | 213       |
|    iterations           | 179       |
|    time_elapsed         | 1719      |
|    total_timesteps      | 366592    |
| train/                  |           |
|    approx_kl            | 2.3712237 |
|    clip_fraction        | 0.721     |
|    clip_range           | 0.2       |
|    entropy_loss         | 2.13      |
|    explained_variance   | 1         |
|    learning_rate        | 0.0003    |
|    loss                 | -0.0545   |
|    n_updates            | 3560      |
|    policy_gradient_loss | 0.00573   |
|    std                  | 0.0286    |
|    value_loss           | 5.09e-06  |
---------------------------------------
---------------------------------------
| time/                   |           |
|    fps                  | 212       |
|    iterations           | 180       |
|    time_elapsed         | 1731      |
|    total_timesteps      | 368640    |
| train/                  |           |
|    approx_kl            | 2.2699764 |
|    clip_fraction        | 0.716     |
|    clip_range           | 0.2       |
|    entropy_loss         | 2.14      |
|    explained_variance   | 1         |
|    learning_rate        | 0.0003    |
|    loss                 | -0.0597   |
|    n_updates            | 3580      |
|    policy_gradient_loss | 0.00338   |
|    std                  | 0.0282    |
|    value_loss           | 4.41e-06  |
---------------------------------------
^C
[STEP] package_diagnostics_to_onedrive
[INFO] diag_script=C:\work\apex_repo_cache\ApexTraderAI\scripts\package_diagnostics_to_onedrive.ps1
[PUBLISH] run_dir=C:\work\apex_work\runs\gh22796461551_att1_sim_20260307_182743_e15a209
[PUBLISH] output_root=C:\work\apex_work\runs\gh22796461551_att1_sim_20260307_182743_e15a209\output
[WARN] publish issue: D' state CSV missing under output/stepDprime/<mode> or stepD_prime/<mode>.
[WARN] publish issue: D' embeddings CSV missing under output/stepDprime/<mode>/embeddings or stepD_prime/<mode>/embeddings.
[OK] package_diagnostics_to_onedrive
[OK] diag_dir=C:\Users\becky\OneDrive\ApexTraderAI\diagnostics

[STEP] resolve_latest_run_artifacts
[INFO] work_root=C:\work\apex_work\runs
[OK] resolve_latest_run_artifacts
[OK] run_dir=C:\work\apex_work\runs\gh22796461551_att1_sim_20260307_182743_e15a209
[OK] run_id=gh22796461551_att1_sim_20260307_182743_e15a209
```
