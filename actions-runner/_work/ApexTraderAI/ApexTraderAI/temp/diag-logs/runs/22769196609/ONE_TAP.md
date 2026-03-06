# ONE_TAP Public Run Report

- run_id: 22769196609
- sha: 3560d51ecae39bc60d518565d7da29d17d1fc133
- mode: sim
- symbols: SOXL,SOXS
- test_start: 2022-01-03
- train_years: 8
- test_months: 3
- output_root: C:/work/apex_work/runs/gh22769196609_att1_sim_20260307_000806_3560d51/output

## StepE metrics
| step  | mode | source_csv                                                                                                                              | test_days | total_return_pct     | max_drawdown         | sharpe                | win_rate           | avg_ret                | std_ret              | status |
| ----- | ---- | --------------------------------------------------------------------------------------------------------------------------------------- | --------- | -------------------- | -------------------- | --------------------- | ------------------ | ---------------------- | -------------------- | ------ |
| StepE | sim  | C:/work/apex_work/runs/gh22769196609_att1_sim_20260307_000806_3560d51/output/stepE/sim/stepE_daily_log_dprime_all_features_h01_SOXL.csv | 63.0      | -0.2717102128865194  | -0.5110547789544784  | -0.6092891664140035   | 0.5396825396825397 | -0.002908318162483744  | 0.07577374109575971  |        |
| StepE | sim  | C:/work/apex_work/runs/gh22769196609_att1_sim_20260307_000806_3560d51/output/stepE/sim/stepE_daily_log_dprime_bnf_3scale_SOXL.csv       | 63.0      | 0.31551034794193367  | -0.3945505518765645  | 1.693171563984828     | 0.5396825396825397 | 0.008287317724920057   | 0.07769861791552933  |        |
| StepE | sim  | C:/work/apex_work/runs/gh22769196609_att1_sim_20260307_000806_3560d51/output/stepE/sim/stepE_daily_log_dprime_bnf_h01_SOXL.csv          | 63.0      | 0.3698821594500097   | -0.29588041998775316 | 1.4593742862066688    | 0.5238095238095238 | 0.00642958708737734    | 0.06993855638055367  |        |
| StepE | sim  | C:/work/apex_work/runs/gh22769196609_att1_sim_20260307_000806_3560d51/output/stepE/sim/stepE_daily_log_dprime_bnf_h02_SOXL.csv          | 63.0      | -0.19812718765011417 | -0.43829184832168233 | -0.008421905446951362 | 0.5079365079365079 | -4.039973870029873e-05 | 0.07614974708960269  |        |
| StepE | sim  | C:/work/apex_work/runs/gh22769196609_att1_sim_20260307_000806_3560d51/output/stepE/sim/stepE_daily_log_dprime_mix_3scale_SOXL.csv       | 63.0      | 0.07237827954755582  | -0.3625452188640177  | 0.9773589684263233    | 0.5396825396825397 | 0.004817442708438613   | 0.07824610469796389  |        |
| StepE | sim  | C:/work/apex_work/runs/gh22769196609_att1_sim_20260307_000806_3560d51/output/stepE/sim/stepE_daily_log_dprime_mix_h01_SOXL.csv          | 63.0      | -0.42849988144743867 | -0.5052833038143285  | -1.6148047651423523   | 0.5238095238095238 | -0.0062152452799874485 | 0.061099621587994965 |        |
| StepE | sim  | C:/work/apex_work/runs/gh22769196609_att1_sim_20260307_000806_3560d51/output/stepE/sim/stepE_daily_log_dprime_mix_h02_SOXL.csv          | 63.0      | 0.42111788749226187  | -0.3534847083322621  | 1.9399049133999686    | 0.5714285714285714 | 0.009476305152746941   | 0.07754590426183132  |        |

## StepF metrics
| step  | mode | source_csv | test_days | total_return_pct | max_drawdown | sharpe | win_rate | avg_ret | std_ret | status    |
| ----- | ---- | ---------- | --------- | ---------------- | ------------ | ------ | -------- | ------- | ------- | --------- |
| StepF | sim  | not found  |           |                  |              |        |          |         |         | not found |

## Key CSV files
- StepE: C:/work/apex_work/runs/gh22769196609_att1_sim_20260307_000806_3560d51/output/stepE/sim/stepE_daily_log_dprime_all_features_h01_SOXL.csv
- StepE: C:/work/apex_work/runs/gh22769196609_att1_sim_20260307_000806_3560d51/output/stepE/sim/stepE_daily_log_dprime_bnf_3scale_SOXL.csv
- StepE: C:/work/apex_work/runs/gh22769196609_att1_sim_20260307_000806_3560d51/output/stepE/sim/stepE_daily_log_dprime_bnf_h01_SOXL.csv
- StepE: C:/work/apex_work/runs/gh22769196609_att1_sim_20260307_000806_3560d51/output/stepE/sim/stepE_daily_log_dprime_bnf_h02_SOXL.csv
- StepE: C:/work/apex_work/runs/gh22769196609_att1_sim_20260307_000806_3560d51/output/stepE/sim/stepE_daily_log_dprime_mix_3scale_SOXL.csv
- StepE: C:/work/apex_work/runs/gh22769196609_att1_sim_20260307_000806_3560d51/output/stepE/sim/stepE_daily_log_dprime_mix_h01_SOXL.csv
- StepE: C:/work/apex_work/runs/gh22769196609_att1_sim_20260307_000806_3560d51/output/stepE/sim/stepE_daily_log_dprime_mix_h02_SOXL.csv

## Error summary
```text
|    clip_range           | 0.2         |
|    entropy_loss         | -0.766      |
|    explained_variance   | 0.982       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.0923     |
|    n_updates            | 480         |
|    policy_gradient_loss | -0.0603     |
|    std                  | 0.51        |
|    value_loss           | 0.00171     |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 217         |
|    iterations           | 26          |
|    time_elapsed         | 245         |
|    total_timesteps      | 53248       |
| train/                  |             |
|    approx_kl            | 0.033059962 |
|    clip_fraction        | 0.229       |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.731      |
|    explained_variance   | 0.984       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.0939     |
|    n_updates            | 500         |
|    policy_gradient_loss | -0.0597     |
|    std                  | 0.492       |
|    value_loss           | 0.00162     |
-----------------------------------------
----------------------------------------
| time/                   |            |
|    fps                  | 217        |
|    iterations           | 27         |
|    time_elapsed         | 254        |
|    total_timesteps      | 55296      |
| train/                  |            |
|    approx_kl            | 0.03571079 |
|    clip_fraction        | 0.239      |
|    clip_range           | 0.2        |
|    entropy_loss         | -0.697     |
|    explained_variance   | 0.987      |
|    learning_rate        | 0.0003     |
|    loss                 | -0.0961    |
|    n_updates            | 520        |
|    policy_gradient_loss | -0.0614    |
|    std                  | 0.477      |
|    value_loss           | 0.00135    |
----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 215         |
|    iterations           | 28          |
|    time_elapsed         | 265         |
|    total_timesteps      | 57344       |
| train/                  |             |
|    approx_kl            | 0.036218002 |
|    clip_fraction        | 0.255       |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.662      |
|    explained_variance   | 0.99        |
|    learning_rate        | 0.0003      |
|    loss                 | -0.0912     |
|    n_updates            | 540         |
|    policy_gradient_loss | -0.0641     |
|    std                  | 0.46        |
|    value_loss           | 0.00117     |
-----------------------------------------
[STEP] package_diagnostics_to_onedrive
[INFO] diag_script=C:\work\apex_repo_cache\ApexTraderAI\scripts\package_diagnostics_to_onedrive.ps1
[PUBLISH] run_dir=C:\work\apex_work\runs\gh22769196609_att1_sim_20260307_000806_3560d51
[PUBLISH] output_root=C:\work\apex_work\runs\gh22769196609_att1_sim_20260307_000806_3560d51\output
[OK] package_diagnostics_to_onedrive
[OK] diag_dir=C:\Users\becky\OneDrive\ApexTraderAI\diagnostics

[STEP] resolve_latest_run_artifacts
[INFO] work_root=C:\work\apex_work\runs
[OK] resolve_latest_run_artifacts
[OK] run_dir=C:\work\apex_work\runs\gh22769196609_att1_sim_20260307_000806_3560d51
[OK] run_id=gh22769196609_att1_sim_20260307_000806_3560d51
```
