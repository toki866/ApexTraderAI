# ONE_TAP Public Run Report

- run_id: 22795926130
- sha: e15a2095928ce120bce1f85d56ae8d48faa29344
- mode: sim
- symbols: SOXL,SOXS
- test_start: 2022-01-03
- train_years: 8
- test_months: 3
- output_root: C:/work/apex_work/runs/gh22795926130_att1_sim_20260307_175055_e15a209/output

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
|    loss                 | -0.0379   |
|    n_updates            | 2200      |
|    policy_gradient_loss | -0.0139   |
|    std                  | 0.0387    |
|    value_loss           | 1.04e-05  |
---------------------------------------
---------------------------------------
| time/                   |           |
|    fps                  | 202       |
|    iterations           | 112       |
|    time_elapsed         | 1133      |
|    total_timesteps      | 229376    |
| train/                  |           |
|    approx_kl            | 1.1777494 |
|    clip_fraction        | 0.662     |
|    clip_range           | 0.2       |
|    entropy_loss         | 1.83      |
|    explained_variance   | 1         |
|    learning_rate        | 0.0003    |
|    loss                 | -0.0404   |
|    n_updates            | 2220      |
|    policy_gradient_loss | -0.0124   |
|    std                  | 0.0383    |
|    value_loss           | 1.11e-05  |
---------------------------------------
---------------------------------------
| time/                   |           |
|    fps                  | 202       |
|    iterations           | 113       |
|    time_elapsed         | 1140      |
|    total_timesteps      | 231424    |
| train/                  |           |
|    approx_kl            | 1.2848976 |
|    clip_fraction        | 0.646     |
|    clip_range           | 0.2       |
|    entropy_loss         | 1.85      |
|    explained_variance   | 1         |
|    learning_rate        | 0.0003    |
|    loss                 | -0.0529   |
|    n_updates            | 2240      |
|    policy_gradient_loss | -0.0258   |
|    std                  | 0.0376    |
|    value_loss           | 1.32e-05  |
---------------------------------------
---------------------------------------
| time/                   |           |
|    fps                  | 203       |
|    iterations           | 114       |
|    time_elapsed         | 1149      |
|    total_timesteps      | 233472    |
| train/                  |           |
|    approx_kl            | 1.4960734 |
|    clip_fraction        | 0.684     |
|    clip_range           | 0.2       |
|    entropy_loss         | 1.86      |
|    explained_variance   | 1         |
|    learning_rate        | 0.0003    |
|    loss                 | -0.0335   |
|    n_updates            | 2260      |
|    policy_gradient_loss | -0.00459  |
|    std                  | 0.0373    |
|    value_loss           | 1.25e-05  |
---------------------------------------
[STEP] package_diagnostics_to_onedrive
[INFO] diag_script=C:\work\apex_repo_cache\ApexTraderAI\scripts\package_diagnostics_to_onedrive.ps1
[PUBLISH] run_dir=C:\work\apex_work\runs\gh22795926130_att1_sim_20260307_175055_e15a209
[PUBLISH] output_root=C:\work\apex_work\runs\gh22795926130_att1_sim_20260307_175055_e15a209\output
[WARN] publish issue: D' state CSV missing under output/stepDprime/<mode> or stepD_prime/<mode>.
[WARN] publish issue: D' embeddings CSV missing under output/stepDprime/<mode>/embeddings or stepD_prime/<mode>/embeddings.
[OK] package_diagnostics_to_onedrive
[OK] diag_dir=C:\Users\becky\OneDrive\ApexTraderAI\diagnostics

[STEP] resolve_latest_run_artifacts
[INFO] work_root=C:\work\apex_work\runs
[OK] resolve_latest_run_artifacts
[OK] run_dir=C:\work\apex_work\runs\gh22795926130_att1_sim_20260307_175055_e15a209
[OK] run_id=gh22795926130_att1_sim_20260307_175055_e15a209
```
