# ONE_TAP Public Run Report

- run_id: 22794075616
- sha: 04a93b06b617b1a895f9fa7df749a848511028cb
- mode: sim
- symbols: SOXL,SOXS
- test_start: 2022-01-03
- train_years: 8
- test_months: 3
- output_root: C:/work/apex_work/runs/gh22794075616_att1_sim_20260307_154454_04a93b0/output

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
|    explained_variance   | 1         |
|    learning_rate        | 0.0003    |
|    loss                 | -0.0863   |
|    n_updates            | 1680      |
|    policy_gradient_loss | -0.072    |
|    std                  | 0.073     |
|    value_loss           | 1.29e-05  |
---------------------------------------
----------------------------------------
| time/                   |            |
|    fps                  | 188        |
|    iterations           | 86         |
|    time_elapsed         | 934        |
|    total_timesteps      | 176128     |
| train/                  |            |
|    approx_kl            | 0.50594705 |
|    clip_fraction        | 0.573      |
|    clip_range           | 0.2        |
|    entropy_loss         | 1.21       |
|    explained_variance   | 1          |
|    learning_rate        | 0.0003     |
|    loss                 | -0.115     |
|    n_updates            | 1700       |
|    policy_gradient_loss | -0.0777    |
|    std                  | 0.0713     |
|    value_loss           | 9.43e-06   |
----------------------------------------
----------------------------------------
| time/                   |            |
|    fps                  | 188        |
|    iterations           | 87         |
|    time_elapsed         | 945        |
|    total_timesteps      | 178176     |
| train/                  |            |
|    approx_kl            | 0.49915066 |
|    clip_fraction        | 0.553      |
|    clip_range           | 0.2        |
|    entropy_loss         | 1.23       |
|    explained_variance   | 1          |
|    learning_rate        | 0.0003     |
|    loss                 | -0.106     |
|    n_updates            | 1720       |
|    policy_gradient_loss | -0.0717    |
|    std                  | 0.0695     |
|    value_loss           | 1.46e-05   |
----------------------------------------
---------------------------------------
| time/                   |           |
|    fps                  | 187       |
|    iterations           | 88        |
|    time_elapsed         | 960       |
|    total_timesteps      | 180224    |
| train/                  |           |
|    approx_kl            | 0.4814884 |
|    clip_fraction        | 0.553     |
|    clip_range           | 0.2       |
|    entropy_loss         | 1.26      |
|    explained_variance   | 1         |
|    learning_rate        | 0.0003    |
|    loss                 | -0.0781   |
|    n_updates            | 1740      |
|    policy_gradient_loss | -0.0524   |
|    std                  | 0.0682    |
|    value_loss           | 1.23e-05  |
---------------------------------------
[STEP] package_diagnostics_to_onedrive
[INFO] diag_script=C:\work\apex_repo_cache\ApexTraderAI\scripts\package_diagnostics_to_onedrive.ps1
[PUBLISH] run_dir=C:\work\apex_work\runs\gh22794075616_att1_sim_20260307_154454_04a93b0
[PUBLISH] output_root=C:\work\apex_work\runs\gh22794075616_att1_sim_20260307_154454_04a93b0\output
[WARN] publish issue: D' state CSV missing under output/stepD_prime/<mode>.
[WARN] publish issue: D' embeddings CSV missing under output/stepD_prime/<mode>/embeddings.
[OK] package_diagnostics_to_onedrive
[OK] diag_dir=C:\Users\becky\OneDrive\ApexTraderAI\diagnostics

[STEP] resolve_latest_run_artifacts
[INFO] work_root=C:\work\apex_work\runs
[OK] resolve_latest_run_artifacts
[OK] run_dir=C:\work\apex_work\runs\gh22794075616_att1_sim_20260307_154454_04a93b0
[OK] run_id=gh22794075616_att1_sim_20260307_154454_04a93b0
```
