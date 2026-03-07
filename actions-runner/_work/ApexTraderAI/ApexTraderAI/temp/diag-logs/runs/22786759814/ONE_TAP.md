# ONE_TAP Public Run Report

- run_id: 22786759814
- sha: 48997059576ca83a6b51ffd15b1fec7126bb6c00
- mode: sim
- symbols: SOXL,SOXS
- test_start: 2022-01-03
- train_years: 8
- test_months: 3
- output_root: C:/work/apex_work/runs/gh22786759814_att1_sim_20260307_084832_4899705/output

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
|    explained_variance   | 0.995       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.0953     |
|    n_updates            | 660         |
|    policy_gradient_loss | -0.0689     |
|    std                  | 0.385       |
|    value_loss           | 0.000592    |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 193         |
|    iterations           | 35          |
|    time_elapsed         | 370         |
|    total_timesteps      | 71680       |
| train/                  |             |
|    approx_kl            | 0.052360594 |
|    clip_fraction        | 0.296       |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.451      |
|    explained_variance   | 0.996       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.117      |
|    n_updates            | 680         |
|    policy_gradient_loss | -0.0719     |
|    std                  | 0.373       |
|    value_loss           | 0.000503    |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 193         |
|    iterations           | 36          |
|    time_elapsed         | 380         |
|    total_timesteps      | 73728       |
| train/                  |             |
|    approx_kl            | 0.053708173 |
|    clip_fraction        | 0.301       |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.419      |
|    explained_variance   | 0.996       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.0916     |
|    n_updates            | 700         |
|    policy_gradient_loss | -0.0698     |
|    std                  | 0.36        |
|    value_loss           | 0.000454    |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 194         |
|    iterations           | 37          |
|    time_elapsed         | 390         |
|    total_timesteps      | 75776       |
| train/                  |             |
|    approx_kl            | 0.060048833 |
|    clip_fraction        | 0.297       |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.383      |
|    explained_variance   | 0.995       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.111      |
|    n_updates            | 720         |
|    policy_gradient_loss | -0.0682     |
|    std                  | 0.347       |
|    value_loss           | 0.000646    |
-----------------------------------------
[STEP] package_diagnostics_to_onedrive
[INFO] diag_script=C:\work\apex_repo_cache\ApexTraderAI\scripts\package_diagnostics_to_onedrive.ps1
[PUBLISH] run_dir=C:\work\apex_work\runs\gh22786759814_att1_sim_20260307_084832_4899705
[PUBLISH] output_root=C:\work\apex_work\runs\gh22786759814_att1_sim_20260307_084832_4899705\output
[WARN] publish issue: D' state CSV missing under output/stepD_prime/<mode>.
[WARN] publish issue: D' embeddings CSV missing under output/stepD_prime/<mode>/embeddings.
[OK] package_diagnostics_to_onedrive
[OK] diag_dir=C:\Users\becky\OneDrive\ApexTraderAI\diagnostics

[STEP] resolve_latest_run_artifacts
[INFO] work_root=C:\work\apex_work\runs
[OK] resolve_latest_run_artifacts
[OK] run_dir=C:\work\apex_work\runs\gh22786759814_att1_sim_20260307_084832_4899705
[OK] run_id=gh22786759814_att1_sim_20260307_084832_4899705
```
