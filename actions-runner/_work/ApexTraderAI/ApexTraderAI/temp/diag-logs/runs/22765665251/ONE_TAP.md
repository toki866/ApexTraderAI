# ONE_TAP Public Run Report

- run_id: 22765665251
- sha: 3560d51ecae39bc60d518565d7da29d17d1fc133
- mode: sim
- symbols: SOXL,SOXS
- test_start: 2022-01-03
- train_years: 8
- test_months: 3
- output_root: C:/work/apex_work/runs/gh22765665251_att1_sim_20260306_223344_3560d51/output

## StepE metrics
| step  | mode | source_csv                                                                                                                     | test_days | total_return_pct   | max_drawdown        | sharpe             | win_rate           | avg_ret             | std_ret            | status |
| ----- | ---- | ------------------------------------------------------------------------------------------------------------------------------ | --------- | ------------------ | ------------------- | ------------------ | ------------------ | ------------------- | ------------------ | ------ |
| StepE | sim  | C:/work/apex_work/runs/gh22765665251_att1_sim_20260306_223344_3560d51/output/stepE/sim/stepE_daily_log_dprime_bnf_h01_SOXL.csv | 63.0      | 0.5156694682584719 | -0.2754265220953953 | 1.7853119968003228 | 0.5555555555555556 | 0.00839657831703592 | 0.0746600867430533 |        |

## StepF metrics
| step  | mode | source_csv | test_days | total_return_pct | max_drawdown | sharpe | win_rate | avg_ret | std_ret | status    |
| ----- | ---- | ---------- | --------- | ---------------- | ------------ | ------ | -------- | ------- | ------- | --------- |
| StepF | sim  | not found  |           |                  |              |        |          |         |         | not found |

## Key CSV files
- StepE: C:/work/apex_work/runs/gh22765665251_att1_sim_20260306_223344_3560d51/output/stepE/sim/stepE_daily_log_dprime_bnf_h01_SOXL.csv

## Error summary
```text
|    n_updates            | 3060      |
|    policy_gradient_loss | -0.000822 |
|    std                  | 0.0303    |
|    value_loss           | 7.2e-06   |
---------------------------------------
---------------------------------------
| time/                   |           |
|    fps                  | 198       |
|    iterations           | 155       |
|    time_elapsed         | 1596      |
|    total_timesteps      | 317440    |
| train/                  |           |
|    approx_kl            | 1.6405795 |
|    clip_fraction        | 0.671     |
|    clip_range           | 0.2       |
|    entropy_loss         | 2.08      |
|    explained_variance   | 1         |
|    learning_rate        | 0.0003    |
|    loss                 | -0.0404   |
|    n_updates            | 3080      |
|    policy_gradient_loss | -0.00813  |
|    std                  | 0.0298    |
|    value_loss           | 1.02e-05  |
---------------------------------------
---------------------------------------
| time/                   |           |
|    fps                  | 198       |
|    iterations           | 156       |
|    time_elapsed         | 1609      |
|    total_timesteps      | 319488    |
| train/                  |           |
|    approx_kl            | 1.9564912 |
|    clip_fraction        | 0.7       |
|    clip_range           | 0.2       |
|    entropy_loss         | 2.09      |
|    explained_variance   | 1         |
|    learning_rate        | 0.0003    |
|    loss                 | -0.0181   |
|    n_updates            | 3100      |
|    policy_gradient_loss | -0.00121  |
|    std                  | 0.0293    |
|    value_loss           | 1.11e-05  |
---------------------------------------
---------------------------------------
| time/                   |           |
|    fps                  | 198       |
|    iterations           | 157       |
|    time_elapsed         | 1616      |
|    total_timesteps      | 321536    |
| train/                  |           |
|    approx_kl            | 1.7334368 |
|    clip_fraction        | 0.726     |
|    clip_range           | 0.2       |
|    entropy_loss         | 2.1       |
|    explained_variance   | 1         |
|    learning_rate        | 0.0003    |
|    loss                 | 0.00846   |
|    n_updates            | 3120      |
|    policy_gradient_loss | 0.0303    |
|    std                  | 0.0298    |
|    value_loss           | 7.85e-06  |
---------------------------------------
[RC] 
[FAILED] command=wsl run_pipeline
[FAILED] exit_code=
[FAILED] run_id=gh22765665251_att1_sim_20260306_223344_3560d51
[FAILED] log=C:\work\apex_work\runs\gh22765665251_att1_sim_20260306_223344_3560d51\logs\run_gh22765665251_att1_sim_20260306_223344_3560d51.log
[STEP] package_diagnostics_to_onedrive
[INFO] diag_script=C:\work\apex_repo_cache\ApexTraderAI\scripts\package_diagnostics_to_onedrive.ps1
[PUBLISH] run_dir=C:\work\apex_work\runs\gh22765665251_att1_sim_20260306_223344_3560d51
[PUBLISH] output_root=C:\work\apex_work\runs\gh22765665251_att1_sim_20260306_223344_3560d51\output
[OK] package_diagnostics_to_onedrive
[OK] diag_dir=C:\Users\becky\OneDrive\ApexTraderAI\diagnostics

[STEP] resolve_latest_run_artifacts
[INFO] work_root=C:\work\apex_work\runs
[OK] resolve_latest_run_artifacts
[OK] run_dir=C:\work\apex_work\runs\gh22765665251_att1_sim_20260306_223344_3560d51
[OK] run_id=gh22765665251_att1_sim_20260306_223344_3560d51
```
