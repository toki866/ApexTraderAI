# EVAL_REPORT

- output_root: `C:\work\apex_work\output\sim\SOXL\2022-01-03_20260322_001`
- mode: `sim`
- symbol: `SOXL`
- overall_status: **OK**

## DPrime diagnostics
- status: **OK**
- summary: DPrimeCluster=OK (embeddings found); DPrimeRL=OK (RL state files found under stepDprime/sim)

### DPrimeCluster
- status: **OK**
- summary: embeddings found
- cluster_embeddings_count: 80
- cluster_state_count: 0
- cluster_input_count: 0

### DPrimeRL
- status: **OK**
- summary: RL state files found under stepDprime/sim
- rl_state_count: 40
- rl_profiles_count: 10

## StepA table
| status | summary | test_rows | test_date_start | test_date_end | missing_ohlcv_count |
|---|---|---:|---|---|---:|
| OK | prices_test evaluated | 63 | 2022-01-03 | 2022-04-01 | 0 |

## StepB table
| file | pred_col | non_null_ratio | coverage_ratio_over_test | mae | corr | status |
|---|---|---:|---:|---:|---:|---|
| stepB_pred_close_mamba_SOXL.csv | Pred_Close_MAMBA | 1.0000 | 1.0000 | 4.7994 | 0.8839 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Pred_Close_MAMBA_PERIODIC_h01 | 1.0000 | 1.0000 | 3.3255 | 0.9252 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Delta_Close_pred_MAMBA_PERIODIC_h01 | 1.0000 | 1.0000 | 44.5081 | 0.0465 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Pred_Close_MAMBA_PERIODIC_h05 | 1.0000 | 1.0000 | 5.6956 | 0.8409 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Delta_Close_pred_MAMBA_PERIODIC_h05 | 1.0000 | 1.0000 | 42.5194 | 0.1642 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Pred_Close_MAMBA_PERIODIC_h10 | 1.0000 | 1.0000 | 6.9968 | 0.7095 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Delta_Close_pred_MAMBA_PERIODIC_h10 | 1.0000 | 1.0000 | 43.8223 | -0.2318 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Pred_Close_MAMBA_PERIODIC_h20 | 1.0000 | 1.0000 | 12.2753 | 0.5886 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Delta_Close_pred_MAMBA_PERIODIC_h20 | 1.0000 | 1.0000 | 40.6933 | 0.0702 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Pred_Close_MAMBA_PERIODIC | 1.0000 | 1.0000 | 3.3255 | 0.9252 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Delta_Close_pred_MAMBA_PERIODIC | 1.0000 | 1.0000 | 44.5081 | 0.0465 | OK |
| stepB_pred_time_all_SOXL.csv | Pred_Close_MAMBA | 1.0000 | 1.0000 | 4.7994 | 0.8839 | OK |

## StepE table
| agent | file | test_days | equity_multiple | max_dd | mean_ret | std_ret | sharpe | note | status |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| dprime_all_features_3scale | stepE_daily_log_dprime_all_features_3scale_SOXL.csv | 63 | 0.9918 | -0.2673 | 0.0006 | 0.0472 | 0.1929 |  | OK |
| dprime_all_features_h01 | stepE_daily_log_dprime_all_features_h01_SOXL.csv | 63 | 1.3285 | -0.1218 | 0.0045 | 0.0293 | 2.4621 |  | OK |
| dprime_all_features_h02 | stepE_daily_log_dprime_all_features_h02_SOXL.csv | 63 | 1.7106 | -0.0931 | 0.0083 | 0.0297 | 4.4533 |  | OK |
| dprime_all_features_h03 | stepE_daily_log_dprime_all_features_h03_SOXL.csv | 63 | 1.5497 | -0.1774 | 0.0075 | 0.0486 | 2.4432 |  | OK |
| dprime_bnf_3scale | stepE_daily_log_dprime_bnf_3scale_SOXL.csv | 63 | 1.1544 | -0.2097 | 0.0035 | 0.0547 | 1.0132 |  | OK |
| dprime_bnf_h01 | stepE_daily_log_dprime_bnf_h01_SOXL.csv | 63 | 1.0274 | -0.0941 | 0.0006 | 0.0157 | 0.5583 |  | OK |
| dprime_bnf_h02 | stepE_daily_log_dprime_bnf_h02_SOXL.csv | 63 | 0.8143 | -0.3262 | -0.0022 | 0.0457 | -0.7813 |  | OK |
| dprime_mix_3scale | stepE_daily_log_dprime_mix_3scale_SOXL.csv | 63 | 1.0464 | -0.3507 | 0.0022 | 0.0564 | 0.6166 |  | OK |
| dprime_mix_h01 | stepE_daily_log_dprime_mix_h01_SOXL.csv | 63 | 0.8720 | -0.2303 | -0.0016 | 0.0330 | -0.7532 |  | OK |
| dprime_mix_h02 | stepE_daily_log_dprime_mix_h02_SOXL.csv | 63 | 0.9444 | -0.1588 | -0.0004 | 0.0265 | -0.2617 |  | OK |

## StepF table
| file | test_days | split_source | train_rows | equity_multiple | max_dd | mean_ret | std_ret | sharpe | note | status |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| stepF_equity_marl_SOXL.csv | 63 | csv | 0 | 1.1916 | -0.1534 | 0.0032 | 0.0386 | 1.3222 |  | OK |

## StepFCompare
- status: **OK**
- summary: current StepF vs fixed-best expert vs daily oracle
- current_stepf_equity_multiple: 1.1916
- fixed_best_expert: dprime_all_features_h02
- fixed_best_equity_multiple: 1.7106
- oracle_equity_multiple: 8.3976
- regret_vs_fixed_best: 0.5190
- regret_vs_oracle: 7.2059
- stepf_win_days_vs_fixed_best: 27
- stepf_pick_match_rate_vs_oracle: 0.2381
- StepFRewardCompare status: OK (reward mode comparison across current/baseline/oracle)

### StepFRewardCompare
| name | equity_multiple | sharpe | max_dd | mean_ret | std_ret | regret_vs_fixed_best | regret_vs_oracle | win_days_vs_fixed_best | pick_match_rate_vs_oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reward_legacy | 1.1695 | 1.3222 | -0.1534 | 0.0032 | 0.0386 | 0.4721 | 7.2281 | 27 | 0.2381 |
| current_stepf | 1.1695 | 1.3222 | -0.1534 | 0.0032 | 0.0386 | 0.4721 | 7.2281 | 27 | 0.2381 |
| reward_profit_basic | 1.1538 | 1.2334 | -0.1545 | 0.0030 | 0.0386 | 0.4878 | 7.2438 | 26 | 0.2381 |
| reward_profit_regret | 0.7747 | -1.3104 | -0.3056 | -0.0033 | 0.0396 | 0.8668 | 7.6229 | 23 | 0.2381 |
| reward_profit_light_risk | 1.1081 | 0.9626 | -0.1636 | 0.0024 | 0.0393 | 0.5334 | 7.2894 | 26 | 0.2381 |
| fixed_best | 1.7106 | 4.4533 | -0.0931 | 0.0083 | 0.0297 | 0.0000 | 6.6870 | NA | NA |
| daily_oracle | 8.3976 | 13.2225 | -0.0677 | 0.0352 | 0.0422 | -6.6870 | 0.0000 | NA | NA |
- cluster_status: PENDING (cluster comparison pending)

## Diversity
- status: **OK**
- summary: agent positions look diverse
- max_corr: 0.8837
- max_match_ratio: 0.0952
- pairs_over_0_9999: 0 / 45
- identical_all_agents: False

## PLOTS
- [equity_stepE_topN.png](./equity_stepE_topN.png)
- [equity_stepF.png](./equity_stepF.png)
- [bar_stepE_return.png](./bar_stepE_return.png)
- [scatter_stepE_dd_vs_ret.png](./scatter_stepE_dd_vs_ret.png)
- [equity_stepF_vs_fixed_best_vs_oracle.png](./equity_stepF_vs_fixed_best_vs_oracle.png)
- [bar_stepF_regret.png](./bar_stepF_regret.png)
- [equity_stepF_reward_modes.png](./equity_stepF_reward_modes.png)
- [bar_stepF_reward_mode_regret.png](./bar_stepF_reward_mode_regret.png)
  - note: StepF_vs_best_StepE: StepF(1.1916) は best StepE(1.7106) に負けてる

## Raw JSON
```json
{
  "output_root": "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001",
  "mode": "sim",
  "symbol": "SOXL",
  "stepA": {
    "status": "OK",
    "summary": "prices_test evaluated",
    "details": {
      "path": "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepA\\sim\\stepA_prices_test_SOXL.csv",
      "test_rows": 63,
      "test_date_start": "2022-01-03",
      "test_date_end": "2022-04-01",
      "missing_ohlcv_count": 0,
      "ohlcv_missing": {
        "Open": 0,
        "High": 0,
        "Low": 0,
        "Close": 0,
        "Volume": 0
      },
      "searched_patterns": [
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepA\\*\\stepA_prices_test_SOXL.csv"
      ]
    }
  },
  "stepB": {
    "status": "OK",
    "summary": "stepB files evaluated",
    "rows": [
      {
        "file": "stepB_pred_close_mamba_SOXL.csv",
        "pred_col": "Pred_Close_MAMBA",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 4.799431028820219,
        "corr": 0.8839408493216454,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Pred_Close_MAMBA_PERIODIC_h01",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 3.325506770732039,
        "corr": 0.9252444098879546,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Delta_Close_pred_MAMBA_PERIODIC_h01",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 44.50807497689154,
        "corr": 0.046479128785460566,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Pred_Close_MAMBA_PERIODIC_h05",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 5.695563114143143,
        "corr": 0.8409380279562503,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Delta_Close_pred_MAMBA_PERIODIC_h05",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 42.519413888136526,
        "corr": 0.16424868786526187,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Pred_Close_MAMBA_PERIODIC_h10",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 6.996848689201731,
        "corr": 0.7094500491202888,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Delta_Close_pred_MAMBA_PERIODIC_h10",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 43.822281794149,
        "corr": -0.23183885836863236,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Pred_Close_MAMBA_PERIODIC_h20",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 12.275335048500953,
        "corr": 0.5885976626791866,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Delta_Close_pred_MAMBA_PERIODIC_h20",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 40.69327698683194,
        "corr": 0.07016659216158654,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Pred_Close_MAMBA_PERIODIC",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 3.325506770732039,
        "corr": 0.9252444098879546,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Delta_Close_pred_MAMBA_PERIODIC",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 44.50807497689154,
        "corr": 0.046479128785460566,
        "status": "OK"
      },
      {
        "file": "stepB_pred_time_all_SOXL.csv",
        "pred_col": "Pred_Close_MAMBA",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 4.799431028820219,
        "corr": 0.8839408493216454,
        "status": "OK"
      }
    ],
    "files": [
      "stepB_pred_close_mamba_SOXL.csv",
      "stepB_pred_close_mamba_periodic_SOXL.csv",
      "stepB_pred_path_mamba_SOXL.csv",
      "stepB_pred_time_all_SOXL.csv"
    ]
  },
  "dprime": {
    "status": "OK",
    "summary": "DPrimeCluster=OK (embeddings found); DPrimeRL=OK (RL state files found under stepDprime/sim)",
    "details": {
      "state_count": 40,
      "embeddings_count": 80,
      "cluster_status": "OK",
      "cluster_summary": "embeddings found",
      "cluster_embeddings_count": 80,
      "cluster_state_count": 0,
      "cluster_input_count": 0,
      "cluster_embeddings_files": [
        "stepDprime_all_features_3scale_SOXL_embeddings_all.csv",
        "stepDprime_all_features_h01_SOXL_embeddings_all.csv",
        "stepDprime_all_features_h02_SOXL_embeddings_all.csv",
        "stepDprime_all_features_h03_SOXL_embeddings_all.csv",
        "stepDprime_bnf_3scale_SOXL_embeddings_all.csv",
        "stepDprime_bnf_h01_SOXL_embeddings_all.csv",
        "stepDprime_bnf_h02_SOXL_embeddings_all.csv",
        "stepDprime_dprime_all_features_3scale_SOXL_embeddings_all.csv",
        "stepDprime_dprime_all_features_3scale_SOXL_embeddings_test.csv",
        "stepDprime_dprime_all_features_3scale_SOXL_embeddings_train.csv",
        "stepDprime_dprime_all_features_h01_SOXL_embeddings_all.csv",
        "stepDprime_dprime_all_features_h01_SOXL_embeddings_test.csv",
        "stepDprime_dprime_all_features_h01_SOXL_embeddings_train.csv",
        "stepDprime_dprime_all_features_h02_SOXL_embeddings_all.csv",
        "stepDprime_dprime_all_features_h02_SOXL_embeddings_test.csv",
        "stepDprime_dprime_all_features_h02_SOXL_embeddings_train.csv",
        "stepDprime_dprime_all_features_h03_SOXL_embeddings_all.csv",
        "stepDprime_dprime_all_features_h03_SOXL_embeddings_test.csv",
        "stepDprime_dprime_all_features_h03_SOXL_embeddings_train.csv",
        "stepDprime_dprime_bnf_3scale_SOXL_embeddings_all.csv",
        "stepDprime_dprime_bnf_3scale_SOXL_embeddings_test.csv",
        "stepDprime_dprime_bnf_3scale_SOXL_embeddings_train.csv",
        "stepDprime_dprime_bnf_h01_SOXL_embeddings_all.csv",
        "stepDprime_dprime_bnf_h01_SOXL_embeddings_test.csv",
        "stepDprime_dprime_bnf_h01_SOXL_embeddings_train.csv",
        "stepDprime_dprime_bnf_h02_SOXL_embeddings_all.csv",
        "stepDprime_dprime_bnf_h02_SOXL_embeddings_test.csv",
        "stepDprime_dprime_bnf_h02_SOXL_embeddings_train.csv",
        "stepDprime_dprime_mix_3scale_SOXL_embeddings_all.csv",
        "stepDprime_dprime_mix_3scale_SOXL_embeddings_test.csv",
        "stepDprime_dprime_mix_3scale_SOXL_embeddings_train.csv",
        "stepDprime_dprime_mix_h01_SOXL_embeddings_all.csv",
        "stepDprime_dprime_mix_h01_SOXL_embeddings_test.csv",
        "stepDprime_dprime_mix_h01_SOXL_embeddings_train.csv",
        "stepDprime_dprime_mix_h02_SOXL_embeddings_all.csv",
        "stepDprime_dprime_mix_h02_SOXL_embeddings_test.csv",
        "stepDprime_dprime_mix_h02_SOXL_embeddings_train.csv",
        "stepDprime_mix_3scale_SOXL_embeddings_all.csv",
        "stepDprime_mix_h01_SOXL_embeddings_all.csv",
        "stepDprime_mix_h02_SOXL_embeddings_all.csv",
        "stepDprime_all_features_3scale_SOXL_embeddings_all.csv",
        "stepDprime_all_features_h01_SOXL_embeddings_all.csv",
        "stepDprime_all_features_h02_SOXL_embeddings_all.csv",
        "stepDprime_all_features_h03_SOXL_embeddings_all.csv",
        "stepDprime_bnf_3scale_SOXL_embeddings_all.csv",
        "stepDprime_bnf_h01_SOXL_embeddings_all.csv",
        "stepDprime_bnf_h02_SOXL_embeddings_all.csv",
        "stepDprime_dprime_all_features_3scale_SOXL_embeddings_all.csv",
        "stepDprime_dprime_all_features_3scale_SOXL_embeddings_test.csv",
        "stepDprime_dprime_all_features_3scale_SOXL_embeddings_train.csv",
        "stepDprime_dprime_all_features_h01_SOXL_embeddings_all.csv",
        "stepDprime_dprime_all_features_h01_SOXL_embeddings_test.csv",
        "stepDprime_dprime_all_features_h01_SOXL_embeddings_train.csv",
        "stepDprime_dprime_all_features_h02_SOXL_embeddings_all.csv",
        "stepDprime_dprime_all_features_h02_SOXL_embeddings_test.csv",
        "stepDprime_dprime_all_features_h02_SOXL_embeddings_train.csv",
        "stepDprime_dprime_all_features_h03_SOXL_embeddings_all.csv",
        "stepDprime_dprime_all_features_h03_SOXL_embeddings_test.csv",
        "stepDprime_dprime_all_features_h03_SOXL_embeddings_train.csv",
        "stepDprime_dprime_bnf_3scale_SOXL_embeddings_all.csv",
        "stepDprime_dprime_bnf_3scale_SOXL_embeddings_test.csv",
        "stepDprime_dprime_bnf_3scale_SOXL_embeddings_train.csv",
        "stepDprime_dprime_bnf_h01_SOXL_embeddings_all.csv",
        "stepDprime_dprime_bnf_h01_SOXL_embeddings_test.csv",
        "stepDprime_dprime_bnf_h01_SOXL_embeddings_train.csv",
        "stepDprime_dprime_bnf_h02_SOXL_embeddings_all.csv",
        "stepDprime_dprime_bnf_h02_SOXL_embeddings_test.csv",
        "stepDprime_dprime_bnf_h02_SOXL_embeddings_train.csv",
        "stepDprime_dprime_mix_3scale_SOXL_embeddings_all.csv",
        "stepDprime_dprime_mix_3scale_SOXL_embeddings_test.csv",
        "stepDprime_dprime_mix_3scale_SOXL_embeddings_train.csv",
        "stepDprime_dprime_mix_h01_SOXL_embeddings_all.csv",
        "stepDprime_dprime_mix_h01_SOXL_embeddings_test.csv",
        "stepDprime_dprime_mix_h01_SOXL_embeddings_train.csv",
        "stepDprime_dprime_mix_h02_SOXL_embeddings_all.csv",
        "stepDprime_dprime_mix_h02_SOXL_embeddings_test.csv",
        "stepDprime_dprime_mix_h02_SOXL_embeddings_train.csv",
        "stepDprime_mix_3scale_SOXL_embeddings_all.csv",
        "stepDprime_mix_h01_SOXL_embeddings_all.csv",
        "stepDprime_mix_h02_SOXL_embeddings_all.csv"
      ],
      "cluster_state_files": [],
      "cluster_input_files": [],
      "rl_status": "OK",
      "rl_summary": "RL state files found under stepDprime/sim",
      "rl_state_count": 40,
      "rl_profiles_count": 10,
      "rl_state_files": [
        "stepDprime_state_test_dprime_all_features_3scale_SOXL.csv",
        "stepDprime_state_test_dprime_all_features_h01_SOXL.csv",
        "stepDprime_state_test_dprime_all_features_h02_SOXL.csv",
        "stepDprime_state_test_dprime_all_features_h03_SOXL.csv",
        "stepDprime_state_test_dprime_bnf_3scale_SOXL.csv",
        "stepDprime_state_test_dprime_bnf_h01_SOXL.csv",
        "stepDprime_state_test_dprime_bnf_h02_SOXL.csv",
        "stepDprime_state_test_dprime_mix_3scale_SOXL.csv",
        "stepDprime_state_test_dprime_mix_h01_SOXL.csv",
        "stepDprime_state_test_dprime_mix_h02_SOXL.csv",
        "stepDprime_state_train_dprime_all_features_3scale_SOXL.csv",
        "stepDprime_state_train_dprime_all_features_h01_SOXL.csv",
        "stepDprime_state_train_dprime_all_features_h02_SOXL.csv",
        "stepDprime_state_train_dprime_all_features_h03_SOXL.csv",
        "stepDprime_state_train_dprime_bnf_3scale_SOXL.csv",
        "stepDprime_state_train_dprime_bnf_h01_SOXL.csv",
        "stepDprime_state_train_dprime_bnf_h02_SOXL.csv",
        "stepDprime_state_train_dprime_mix_3scale_SOXL.csv",
        "stepDprime_state_train_dprime_mix_h01_SOXL.csv",
        "stepDprime_state_train_dprime_mix_h02_SOXL.csv",
        "stepDprime_state_test_dprime_all_features_3scale_SOXL.csv",
        "stepDprime_state_test_dprime_all_features_h01_SOXL.csv",
        "stepDprime_state_test_dprime_all_features_h02_SOXL.csv",
        "stepDprime_state_test_dprime_all_features_h03_SOXL.csv",
        "stepDprime_state_test_dprime_bnf_3scale_SOXL.csv",
        "stepDprime_state_test_dprime_bnf_h01_SOXL.csv",
        "stepDprime_state_test_dprime_bnf_h02_SOXL.csv",
        "stepDprime_state_test_dprime_mix_3scale_SOXL.csv",
        "stepDprime_state_test_dprime_mix_h01_SOXL.csv",
        "stepDprime_state_test_dprime_mix_h02_SOXL.csv",
        "stepDprime_state_train_dprime_all_features_3scale_SOXL.csv",
        "stepDprime_state_train_dprime_all_features_h01_SOXL.csv",
        "stepDprime_state_train_dprime_all_features_h02_SOXL.csv",
        "stepDprime_state_train_dprime_all_features_h03_SOXL.csv",
        "stepDprime_state_train_dprime_bnf_3scale_SOXL.csv",
        "stepDprime_state_train_dprime_bnf_h01_SOXL.csv",
        "stepDprime_state_train_dprime_bnf_h02_SOXL.csv",
        "stepDprime_state_train_dprime_mix_3scale_SOXL.csv",
        "stepDprime_state_train_dprime_mix_h01_SOXL.csv",
        "stepDprime_state_train_dprime_mix_h02_SOXL.csv"
      ],
      "state_files": [
        "stepDprime_state_test_dprime_all_features_3scale_SOXL.csv",
        "stepDprime_state_test_dprime_all_features_h01_SOXL.csv",
        "stepDprime_state_test_dprime_all_features_h02_SOXL.csv",
        "stepDprime_state_test_dprime_all_features_h03_SOXL.csv",
        "stepDprime_state_test_dprime_bnf_3scale_SOXL.csv",
        "stepDprime_state_test_dprime_bnf_h01_SOXL.csv",
        "stepDprime_state_test_dprime_bnf_h02_SOXL.csv",
        "stepDprime_state_test_dprime_mix_3scale_SOXL.csv",
        "stepDprime_state_test_dprime_mix_h01_SOXL.csv",
        "stepDprime_state_test_dprime_mix_h02_SOXL.csv",
        "stepDprime_state_train_dprime_all_features_3scale_SOXL.csv",
        "stepDprime_state_train_dprime_all_features_h01_SOXL.csv",
        "stepDprime_state_train_dprime_all_features_h02_SOXL.csv",
        "stepDprime_state_train_dprime_all_features_h03_SOXL.csv",
        "stepDprime_state_train_dprime_bnf_3scale_SOXL.csv",
        "stepDprime_state_train_dprime_bnf_h01_SOXL.csv",
        "stepDprime_state_train_dprime_bnf_h02_SOXL.csv",
        "stepDprime_state_train_dprime_mix_3scale_SOXL.csv",
        "stepDprime_state_train_dprime_mix_h01_SOXL.csv",
        "stepDprime_state_train_dprime_mix_h02_SOXL.csv",
        "stepDprime_state_test_dprime_all_features_3scale_SOXL.csv",
        "stepDprime_state_test_dprime_all_features_h01_SOXL.csv",
        "stepDprime_state_test_dprime_all_features_h02_SOXL.csv",
        "stepDprime_state_test_dprime_all_features_h03_SOXL.csv",
        "stepDprime_state_test_dprime_bnf_3scale_SOXL.csv",
        "stepDprime_state_test_dprime_bnf_h01_SOXL.csv",
        "stepDprime_state_test_dprime_bnf_h02_SOXL.csv",
        "stepDprime_state_test_dprime_mix_3scale_SOXL.csv",
        "stepDprime_state_test_dprime_mix_h01_SOXL.csv",
        "stepDprime_state_test_dprime_mix_h02_SOXL.csv",
        "stepDprime_state_train_dprime_all_features_3scale_SOXL.csv",
        "stepDprime_state_train_dprime_all_features_h01_SOXL.csv",
        "stepDprime_state_train_dprime_all_features_h02_SOXL.csv",
        "stepDprime_state_train_dprime_all_features_h03_SOXL.csv",
        "stepDprime_state_train_dprime_bnf_3scale_SOXL.csv",
        "stepDprime_state_train_dprime_bnf_h01_SOXL.csv",
        "stepDprime_state_train_dprime_bnf_h02_SOXL.csv",
        "stepDprime_state_train_dprime_mix_3scale_SOXL.csv",
        "stepDprime_state_train_dprime_mix_h01_SOXL.csv",
        "stepDprime_state_train_dprime_mix_h02_SOXL.csv"
      ],
      "embeddings_files": [
        "stepDprime_all_features_3scale_SOXL_embeddings_all.csv",
        "stepDprime_all_features_h01_SOXL_embeddings_all.csv",
        "stepDprime_all_features_h02_SOXL_embeddings_all.csv",
        "stepDprime_all_features_h03_SOXL_embeddings_all.csv",
        "stepDprime_bnf_3scale_SOXL_embeddings_all.csv",
        "stepDprime_bnf_h01_SOXL_embeddings_all.csv",
        "stepDprime_bnf_h02_SOXL_embeddings_all.csv",
        "stepDprime_dprime_all_features_3scale_SOXL_embeddings_all.csv",
        "stepDprime_dprime_all_features_3scale_SOXL_embeddings_test.csv",
        "stepDprime_dprime_all_features_3scale_SOXL_embeddings_train.csv",
        "stepDprime_dprime_all_features_h01_SOXL_embeddings_all.csv",
        "stepDprime_dprime_all_features_h01_SOXL_embeddings_test.csv",
        "stepDprime_dprime_all_features_h01_SOXL_embeddings_train.csv",
        "stepDprime_dprime_all_features_h02_SOXL_embeddings_all.csv",
        "stepDprime_dprime_all_features_h02_SOXL_embeddings_test.csv",
        "stepDprime_dprime_all_features_h02_SOXL_embeddings_train.csv",
        "stepDprime_dprime_all_features_h03_SOXL_embeddings_all.csv",
        "stepDprime_dprime_all_features_h03_SOXL_embeddings_test.csv",
        "stepDprime_dprime_all_features_h03_SOXL_embeddings_train.csv",
        "stepDprime_dprime_bnf_3scale_SOXL_embeddings_all.csv",
        "stepDprime_dprime_bnf_3scale_SOXL_embeddings_test.csv",
        "stepDprime_dprime_bnf_3scale_SOXL_embeddings_train.csv",
        "stepDprime_dprime_bnf_h01_SOXL_embeddings_all.csv",
        "stepDprime_dprime_bnf_h01_SOXL_embeddings_test.csv",
        "stepDprime_dprime_bnf_h01_SOXL_embeddings_train.csv",
        "stepDprime_dprime_bnf_h02_SOXL_embeddings_all.csv",
        "stepDprime_dprime_bnf_h02_SOXL_embeddings_test.csv",
        "stepDprime_dprime_bnf_h02_SOXL_embeddings_train.csv",
        "stepDprime_dprime_mix_3scale_SOXL_embeddings_all.csv",
        "stepDprime_dprime_mix_3scale_SOXL_embeddings_test.csv",
        "stepDprime_dprime_mix_3scale_SOXL_embeddings_train.csv",
        "stepDprime_dprime_mix_h01_SOXL_embeddings_all.csv",
        "stepDprime_dprime_mix_h01_SOXL_embeddings_test.csv",
        "stepDprime_dprime_mix_h01_SOXL_embeddings_train.csv",
        "stepDprime_dprime_mix_h02_SOXL_embeddings_all.csv",
        "stepDprime_dprime_mix_h02_SOXL_embeddings_test.csv",
        "stepDprime_dprime_mix_h02_SOXL_embeddings_train.csv",
        "stepDprime_mix_3scale_SOXL_embeddings_all.csv",
        "stepDprime_mix_h01_SOXL_embeddings_all.csv",
        "stepDprime_mix_h02_SOXL_embeddings_all.csv",
        "stepDprime_all_features_3scale_SOXL_embeddings_all.csv",
        "stepDprime_all_features_h01_SOXL_embeddings_all.csv",
        "stepDprime_all_features_h02_SOXL_embeddings_all.csv",
        "stepDprime_all_features_h03_SOXL_embeddings_all.csv",
        "stepDprime_bnf_3scale_SOXL_embeddings_all.csv",
        "stepDprime_bnf_h01_SOXL_embeddings_all.csv",
        "stepDprime_bnf_h02_SOXL_embeddings_all.csv",
        "stepDprime_dprime_all_features_3scale_SOXL_embeddings_all.csv",
        "stepDprime_dprime_all_features_3scale_SOXL_embeddings_test.csv",
        "stepDprime_dprime_all_features_3scale_SOXL_embeddings_train.csv",
        "stepDprime_dprime_all_features_h01_SOXL_embeddings_all.csv",
        "stepDprime_dprime_all_features_h01_SOXL_embeddings_test.csv",
        "stepDprime_dprime_all_features_h01_SOXL_embeddings_train.csv",
        "stepDprime_dprime_all_features_h02_SOXL_embeddings_all.csv",
        "stepDprime_dprime_all_features_h02_SOXL_embeddings_test.csv",
        "stepDprime_dprime_all_features_h02_SOXL_embeddings_train.csv",
        "stepDprime_dprime_all_features_h03_SOXL_embeddings_all.csv",
        "stepDprime_dprime_all_features_h03_SOXL_embeddings_test.csv",
        "stepDprime_dprime_all_features_h03_SOXL_embeddings_train.csv",
        "stepDprime_dprime_bnf_3scale_SOXL_embeddings_all.csv",
        "stepDprime_dprime_bnf_3scale_SOXL_embeddings_test.csv",
        "stepDprime_dprime_bnf_3scale_SOXL_embeddings_train.csv",
        "stepDprime_dprime_bnf_h01_SOXL_embeddings_all.csv",
        "stepDprime_dprime_bnf_h01_SOXL_embeddings_test.csv",
        "stepDprime_dprime_bnf_h01_SOXL_embeddings_train.csv",
        "stepDprime_dprime_bnf_h02_SOXL_embeddings_all.csv",
        "stepDprime_dprime_bnf_h02_SOXL_embeddings_test.csv",
        "stepDprime_dprime_bnf_h02_SOXL_embeddings_train.csv",
        "stepDprime_dprime_mix_3scale_SOXL_embeddings_all.csv",
        "stepDprime_dprime_mix_3scale_SOXL_embeddings_test.csv",
        "stepDprime_dprime_mix_3scale_SOXL_embeddings_train.csv",
        "stepDprime_dprime_mix_h01_SOXL_embeddings_all.csv",
        "stepDprime_dprime_mix_h01_SOXL_embeddings_test.csv",
        "stepDprime_dprime_mix_h01_SOXL_embeddings_train.csv",
        "stepDprime_dprime_mix_h02_SOXL_embeddings_all.csv",
        "stepDprime_dprime_mix_h02_SOXL_embeddings_test.csv",
        "stepDprime_dprime_mix_h02_SOXL_embeddings_train.csv",
        "stepDprime_mix_3scale_SOXL_embeddings_all.csv",
        "stepDprime_mix_h01_SOXL_embeddings_all.csv",
        "stepDprime_mix_h02_SOXL_embeddings_all.csv"
      ],
      "searched": [
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDprime\\sim\\stepDprime_state_test_*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDprime\\sim\\stepDprime_state_*_SOXL_test.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDprime\\sim\\stepDprime_state_test_*_SOXL.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDprime\\sim\\stepDprime_state_*_SOXL.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDPrime\\sim\\stepDprime_state_test_*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDPrime\\sim\\stepDprime_state_*_SOXL_test.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDPrime\\sim\\stepDprime_state_test_*_SOXL.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDPrime\\sim\\stepDprime_state_*_SOXL.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDprime\\sim\\embeddings\\stepDprime_*_SOXL_embeddings*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDprime\\sim\\embeddings\\*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDPrime\\sim\\embeddings\\stepDprime_*_SOXL_embeddings*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDPrime\\sim\\embeddings\\*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDprime\\sim\\*cluster*state*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDprime\\sim\\*cluster_id*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDprime\\sim\\*rare_flag*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDprime\\sim\\*raw20*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDprime\\sim\\*stable*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDPrime\\sim\\*cluster*state*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDPrime\\sim\\*cluster_id*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDPrime\\sim\\*rare_flag*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDPrime\\sim\\*raw20*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDPrime\\sim\\*stable*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDprime\\sim\\*cluster*input*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDprime\\sim\\*cluster_features*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDPrime\\sim\\*cluster*input*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDPrime\\sim\\*cluster_features*.csv"
      ],
      "rl_state_glob": [
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDprime\\sim\\stepDprime_state_test_*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDprime\\sim\\stepDprime_state_*_SOXL_test.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDprime\\sim\\stepDprime_state_test_*_SOXL.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDprime\\sim\\stepDprime_state_*_SOXL.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDPrime\\sim\\stepDprime_state_test_*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDPrime\\sim\\stepDprime_state_*_SOXL_test.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDPrime\\sim\\stepDprime_state_test_*_SOXL.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDPrime\\sim\\stepDprime_state_*_SOXL.csv"
      ],
      "cluster_glob": [
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDprime\\sim\\embeddings\\stepDprime_*_SOXL_embeddings*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDprime\\sim\\embeddings\\*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDPrime\\sim\\embeddings\\stepDprime_*_SOXL_embeddings*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDPrime\\sim\\embeddings\\*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDprime\\sim\\*cluster*state*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDprime\\sim\\*cluster_id*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDprime\\sim\\*rare_flag*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDprime\\sim\\*raw20*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDprime\\sim\\*stable*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDPrime\\sim\\*cluster*state*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDPrime\\sim\\*cluster_id*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDPrime\\sim\\*rare_flag*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDPrime\\sim\\*raw20*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDPrime\\sim\\*stable*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDprime\\sim\\*cluster*input*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDprime\\sim\\*cluster_features*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDPrime\\sim\\*cluster*input*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDPrime\\sim\\*cluster_features*.csv"
      ],
      "traceback_count": 0,
      "traceback_files": [],
      "failure_summary_count": 0,
      "failure_summary_files": [],
      "failure_reason": "",
      "traceback_glob": [
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDprime\\sim\\stepDprime_traceback_SOXL.log",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDprime\\sim\\stepDprime_traceback_*.log",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDPrime\\sim\\stepDprime_traceback_SOXL.log",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDPrime\\sim\\stepDprime_traceback_*.log"
      ],
      "failure_summary_glob": [
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDprime\\sim\\stepDprime_failure_summary_SOXL.json",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDprime\\sim\\stepDprime_failure_summary_*.json",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDPrime\\sim\\stepDprime_failure_summary_SOXL.json",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03_20260322_001\\stepDPrime\\sim\\stepDprime_failure_summary_*.json"
      ]
    }
  },
  "stepE": {
    "status": "OK",
    "summary": "stepE daily logs evaluated",
    "rows": [
      {
        "file": "stepE_daily_log_dprime_all_features_3scale_SOXL.csv",
        "agent": "dprime_all_features_3scale",
        "test_days": 63,
        "equity_multiple": 0.9917771170027627,
        "max_dd": -0.267250805230962,
        "mean_ret": 0.0005734422443127987,
        "std_ret": 0.04718220943785201,
        "sharpe": 0.19293529333875287,
        "split_source": "csv",
        "split_col_present": true,
        "train_rows": 2014,
        "test_rows": 63,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_all_features_h01_SOXL.csv",
        "agent": "dprime_all_features_h01",
        "test_days": 63,
        "equity_multiple": 1.3284890581220898,
        "max_dd": -0.1217670443779012,
        "mean_ret": 0.004544585153558857,
        "std_ret": 0.02930175816013007,
        "sharpe": 2.4620724932403806,
        "split_source": "csv",
        "split_col_present": true,
        "train_rows": 2014,
        "test_rows": 63,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_all_features_h02_SOXL.csv",
        "agent": "dprime_all_features_h02",
        "test_days": 63,
        "equity_multiple": 1.7106039731178828,
        "max_dd": -0.09311799492168216,
        "mean_ret": 0.00832821992510972,
        "std_ret": 0.02968709319635231,
        "sharpe": 4.453328988450917,
        "split_source": "csv",
        "split_col_present": true,
        "train_rows": 2014,
        "test_rows": 63,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_all_features_h03_SOXL.csv",
        "agent": "dprime_all_features_h03",
        "test_days": 63,
        "equity_multiple": 1.54969426890035,
        "max_dd": -0.1773917833545816,
        "mean_ret": 0.007481079443673135,
        "std_ret": 0.04860815675096674,
        "sharpe": 2.4431795487759866,
        "split_source": "csv",
        "split_col_present": true,
        "train_rows": 2014,
        "test_rows": 63,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_bnf_3scale_SOXL.csv",
        "agent": "dprime_bnf_3scale",
        "test_days": 63,
        "equity_multiple": 1.1544461431809934,
        "max_dd": -0.20974131625015147,
        "mean_ret": 0.0034924454529316196,
        "std_ret": 0.05471776324890957,
        "sharpe": 1.0132148962905065,
        "split_source": "csv",
        "split_col_present": true,
        "train_rows": 2014,
        "test_rows": 63,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_bnf_h01_SOXL.csv",
        "agent": "dprime_bnf_h01",
        "test_days": 63,
        "equity_multiple": 1.0273839705446226,
        "max_dd": -0.09413099651297452,
        "mean_ret": 0.0005520218320805343,
        "std_ret": 0.015697109099212448,
        "sharpe": 0.5582604325671512,
        "split_source": "csv",
        "split_col_present": true,
        "train_rows": 2014,
        "test_rows": 63,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_bnf_h02_SOXL.csv",
        "agent": "dprime_bnf_h02",
        "test_days": 63,
        "equity_multiple": 0.8143492257691701,
        "max_dd": -0.32616229692448173,
        "mean_ret": -0.002249981650088467,
        "std_ret": 0.045716817381913986,
        "sharpe": -0.7812737948308519,
        "split_source": "csv",
        "split_col_present": true,
        "train_rows": 2014,
        "test_rows": 63,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_mix_3scale_SOXL.csv",
        "agent": "dprime_mix_3scale",
        "test_days": 63,
        "equity_multiple": 1.0463878398130495,
        "max_dd": -0.3507039116034598,
        "mean_ret": 0.0021925559650068844,
        "std_ret": 0.056448969070326115,
        "sharpe": 0.616587822368102,
        "split_source": "csv",
        "split_col_present": true,
        "train_rows": 2014,
        "test_rows": 63,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_mix_h01_SOXL.csv",
        "agent": "dprime_mix_h01",
        "test_days": 63,
        "equity_multiple": 0.871991952041979,
        "max_dd": -0.2302726596942053,
        "mean_ret": -0.001563753353371534,
        "std_ret": 0.03295853786441211,
        "sharpe": -0.7531831360756606,
        "split_source": "csv",
        "split_col_present": true,
        "train_rows": 2014,
        "test_rows": 63,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_mix_h02_SOXL.csv",
        "agent": "dprime_mix_h02",
        "test_days": 63,
        "equity_multiple": 0.9444436522090549,
        "max_dd": -0.15876170340852402,
        "mean_ret": -0.0004370983063837665,
        "std_ret": 0.02651068536532792,
        "sharpe": -0.2617329732315635,
        "split_source": "csv",
        "split_col_present": true,
        "train_rows": 2014,
        "test_rows": 63,
        "status": "OK"
      }
    ]
  },
  "stepF": {
    "status": "OK",
    "summary": "stepF equity logs evaluated",
    "rows": [
      {
        "file": "stepF_equity_marl_SOXL.csv",
        "test_days": 63,
        "equity_multiple": 1.1916422066427026,
        "max_dd": -0.15342703146823766,
        "mean_ret": 0.0032114589262884456,
        "std_ret": 0.03855759553411437,
        "sharpe": 1.3221864403562424,
        "split_source": "csv",
        "split_col_present": true,
        "train_rows": 0,
        "test_rows": 63,
        "status": "OK"
      }
    ]
  },
  "diversity": {
    "status": "OK",
    "summary": "agent positions look diverse",
    "max_corr": 0.8837258050842104,
    "max_match_ratio": 0.09523809523809523,
    "pairs_over_0_9999": 0,
    "all_pairs": 45,
    "identical_all_agents": false
  },
  "overall_status": "OK",
  "stepF_compare": {
    "status": "OK",
    "summary": "current StepF vs fixed-best expert vs daily oracle",
    "row": {
      "current_stepf_equity_multiple": 1.1916422066427026,
      "fixed_best_expert": "dprime_all_features_h02",
      "fixed_best_equity_multiple": 1.7106039731178828,
      "oracle_equity_multiple": 8.397559012795073,
      "regret_vs_fixed_best": 0.5189617664751802,
      "regret_vs_oracle": 7.20591680615237,
      "current_stepf_sharpe": 1.3221864403562424,
      "fixed_best_sharpe": 4.453328988450917,
      "oracle_sharpe": 13.222531911384966,
      "current_stepf_max_dd": -0.15342703146823766,
      "fixed_best_max_dd": -0.09311799492168216,
      "oracle_max_dd": -0.06769869010821283,
      "stepf_win_days_vs_fixed_best": 27,
      "stepf_common_days_vs_fixed_best": 63,
      "stepf_pick_match_rate_vs_oracle": 0.23809523809523808,
      "oracle_unique_expert_count": 9
    },
    "best_expert_name": "dprime_all_features_h02",
    "best_expert_equity_multiple": 1.7106039731178828,
    "best_expert_sharpe": 4.453328988450917,
    "best_expert_max_dd": -0.09311799492168216,
    "oracle_equity_multiple": 8.397559012795073,
    "oracle_sharpe": 13.222531911384966,
    "oracle_max_dd": -0.06769869010821283,
    "oracle_selected_expert_count_by_name": {
      "dprime_all_features_h03": 15,
      "dprime_bnf_h02": 10,
      "dprime_bnf_3scale": 10,
      "dprime_mix_3scale": 7,
      "dprime_bnf_h01": 6,
      "dprime_all_features_h02": 6,
      "dprime_mix_h01": 5,
      "dprime_mix_h02": 3,
      "dprime_all_features_h01": 1
    },
    "oracle_unique_expert_count": 9,
    "cumulative_regret_vs_oracle": 2.0139452230352557,
    "cluster": {
      "status": "PENDING",
      "reason": "cluster comparison pending",
      "rows": []
    },
    "series": [
      {
        "Date": "2022-01-03",
        "current_stepf_ret": -0.0185990311278835,
        "fixed_best_ret": -0.0403676257101925,
        "oracle_ret": 0.0077325395413182,
        "oracle_expert": "dprime_mix_h02",
        "cumulative_regret_vs_oracle": 0.0263315706692017,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-01-04",
        "current_stepf_ret": 0.0079204195432874,
        "fixed_best_ret": 0.0090172583932146,
        "oracle_ret": 0.0109223604798316,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 0.0293335116057459,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-01-05",
        "current_stepf_ret": 0.0543357267839291,
        "fixed_best_ret": 0.0663425461320715,
        "oracle_ret": 0.0971595069169998,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 0.0721572917388166,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-01-06",
        "current_stepf_ret": -0.001941821535578,
        "fixed_best_ret": -0.0151112425988057,
        "oracle_ret": 0.0104967826009176,
        "oracle_expert": "dprime_mix_3scale",
        "cumulative_regret_vs_oracle": 0.0845958958753122,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-01-07",
        "current_stepf_ret": 0.043340287904443,
        "fixed_best_ret": 0.054106198180238,
        "oracle_ret": 0.087640749335289,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 0.1288963573061582,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-01-10",
        "current_stepf_ret": -0.000478561731095,
        "fixed_best_ret": -0.0019316131130531,
        "oracle_ret": 0.0013763528246332,
        "oracle_expert": "dprime_bnf_h02",
        "cumulative_regret_vs_oracle": 0.1307512718618864,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-01-11",
        "current_stepf_ret": -0.0203654434454907,
        "fixed_best_ret": -0.0274992006264574,
        "oracle_ret": 0.0118445789711676,
        "oracle_expert": "dprime_bnf_h02",
        "cumulative_regret_vs_oracle": 0.1629612942785447,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-01-12",
        "current_stepf_ret": -0.0138488919681017,
        "fixed_best_ret": -0.0094671428268036,
        "oracle_ret": 0.0006367329098951,
        "oracle_expert": "dprime_mix_h02",
        "cumulative_regret_vs_oracle": 0.17744691915654148,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-01-13",
        "current_stepf_ret": 0.0290342563768964,
        "fixed_best_ret": 0.036259858902013,
        "oracle_ret": 0.067571425974369,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 0.2159840887540141,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-01-14",
        "current_stepf_ret": -0.0022750530446925,
        "fixed_best_ret": 0.0107367831549354,
        "oracle_ret": 0.0448195816645268,
        "oracle_expert": "dprime_bnf_h02",
        "cumulative_regret_vs_oracle": 0.2630787234632334,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-01-18",
        "current_stepf_ret": 0.0848735435140623,
        "fixed_best_ret": 0.0656549234126779,
        "oracle_ret": 0.1280602540522813,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 0.3062654340014524,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-01-19",
        "current_stepf_ret": -0.0450822802023625,
        "fixed_best_ret": -0.0412732471701747,
        "oracle_ret": -0.0256792744051079,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 0.32566843979870697,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-01-20",
        "current_stepf_ret": -0.0577235346529443,
        "fixed_best_ret": -0.0262405486742407,
        "oracle_ret": -0.0262405486742407,
        "oracle_expert": "dprime_all_features_h02",
        "cumulative_regret_vs_oracle": 0.35715142577741055,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-01-21",
        "current_stepf_ret": -0.036213603973257,
        "fixed_best_ret": -0.028586237288472,
        "oracle_ret": -0.0173413814494923,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 0.37602364830117524,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-01-24",
        "current_stepf_ret": 0.0287124142939634,
        "fixed_best_ret": 0.0018165935056651,
        "oracle_ret": 0.0388078635334968,
        "oracle_expert": "dprime_bnf_3scale",
        "cumulative_regret_vs_oracle": 0.3861190975407086,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-01-25",
        "current_stepf_ret": -0.0055339714463736,
        "fixed_best_ret": -0.0001893144067312,
        "oracle_ret": 0.0418925829896036,
        "oracle_expert": "dprime_mix_h01",
        "cumulative_regret_vs_oracle": 0.4335456519766858,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-01-26",
        "current_stepf_ret": 0.0361518262669718,
        "fixed_best_ret": 0.0252329505934793,
        "oracle_ret": 0.0466809748113155,
        "oracle_expert": "dprime_bnf_3scale",
        "cumulative_regret_vs_oracle": 0.44407480052102954,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-01-27",
        "current_stepf_ret": -0.0749497764277354,
        "fixed_best_ret": -0.018755082137272,
        "oracle_ret": -0.0076474941646744,
        "oracle_expert": "dprime_mix_h01",
        "cumulative_regret_vs_oracle": 0.5113770827840906,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-01-28",
        "current_stepf_ret": 0.0474760341904246,
        "fixed_best_ret": 0.0394118238295018,
        "oracle_ret": 0.0553373741805553,
        "oracle_expert": "dprime_bnf_3scale",
        "cumulative_regret_vs_oracle": 0.5192384227742213,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-01-31",
        "current_stepf_ret": 0.0760450412194553,
        "fixed_best_ret": 0.0172575432147037,
        "oracle_ret": 0.1436193180464997,
        "oracle_expert": "dprime_bnf_3scale",
        "cumulative_regret_vs_oracle": 0.5868126996012657,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-02-01",
        "current_stepf_ret": -0.0048406847601008,
        "fixed_best_ret": -0.0038129469475688,
        "oracle_ret": 0.0084209685114022,
        "oracle_expert": "dprime_bnf_3scale",
        "cumulative_regret_vs_oracle": 0.6000743528727687,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-02-02",
        "current_stepf_ret": 0.0293259623011463,
        "fixed_best_ret": 0.0214617473120157,
        "oracle_ret": 0.0630551387794679,
        "oracle_expert": "dprime_bnf_3scale",
        "cumulative_regret_vs_oracle": 0.6338035293510903,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-02-03",
        "current_stepf_ret": -0.0148211611372306,
        "fixed_best_ret": 0.0145974331547183,
        "oracle_ret": 0.0299031778653266,
        "oracle_expert": "dprime_mix_3scale",
        "cumulative_regret_vs_oracle": 0.6785278683536475,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-02-04",
        "current_stepf_ret": 0.0124076463462944,
        "fixed_best_ret": 0.0109949492955282,
        "oracle_ret": 0.0181421108990907,
        "oracle_expert": "dprime_bnf_3scale",
        "cumulative_regret_vs_oracle": 0.6842623329064438,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-02-07",
        "current_stepf_ret": -0.0009042619604982,
        "fixed_best_ret": -0.0004167450448654,
        "oracle_ret": -0.0001592190733476,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 0.6850073757935944,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-02-08",
        "current_stepf_ret": 0.0292757452079157,
        "fixed_best_ret": 0.0247469966413464,
        "oracle_ret": 0.0667300794504554,
        "oracle_expert": "dprime_mix_3scale",
        "cumulative_regret_vs_oracle": 0.7224617100361341,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-02-09",
        "current_stepf_ret": -0.0090826259282773,
        "fixed_best_ret": 0.007448109151206,
        "oracle_ret": 0.031977717514586,
        "oracle_expert": "dprime_bnf_h02",
        "cumulative_regret_vs_oracle": 0.7635220534789975,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-02-10",
        "current_stepf_ret": 0.0164901420052462,
        "fixed_best_ret": 0.030573792054184,
        "oracle_ret": 0.0899027440899753,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 0.8369346555637266,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-02-11",
        "current_stepf_ret": -0.0603490344218116,
        "fixed_best_ret": -0.0661212825577289,
        "oracle_ret": 0.0366402008115141,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 0.9339238907970523,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-02-14",
        "current_stepf_ret": 0.0004239037483393,
        "fixed_best_ret": 0.0002602079472586,
        "oracle_ret": 0.0003840004101865,
        "oracle_expert": "dprime_mix_h01",
        "cumulative_regret_vs_oracle": 0.9338839874588994,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-02-15",
        "current_stepf_ret": 0.0764579655400334,
        "fixed_best_ret": 0.0673677345507703,
        "oracle_ret": 0.1351510094566195,
        "oracle_expert": "dprime_bnf_3scale",
        "cumulative_regret_vs_oracle": 0.9925770313754856,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-02-16",
        "current_stepf_ret": -0.0002061901939396,
        "fixed_best_ret": 0.0002687086037023,
        "oracle_ret": 0.000894434016466,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 0.9936776555858912,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-02-17",
        "current_stepf_ret": 0.0018035223229341,
        "fixed_best_ret": 0.0221213814332692,
        "oracle_ret": 0.0460891184120446,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 1.0379632516750017,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-02-18",
        "current_stepf_ret": -0.0125493278300289,
        "fixed_best_ret": -0.0184900471574067,
        "oracle_ret": -0.0054408958701065,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 1.045071683634924,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-02-22",
        "current_stepf_ret": -0.0097544622242357,
        "fixed_best_ret": -0.0083914514956442,
        "oracle_ret": -0.002139596659396,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 1.0526865491997637,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-02-23",
        "current_stepf_ret": -0.0139273068648424,
        "fixed_best_ret": -0.0262204700689469,
        "oracle_ret": 0.0168159930109396,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 1.0834298490755456,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-02-24",
        "current_stepf_ret": 0.0755795164808554,
        "fixed_best_ret": 0.0504161054355495,
        "oracle_ret": 0.10750636571209,
        "oracle_expert": "dprime_mix_3scale",
        "cumulative_regret_vs_oracle": 1.1153566983067802,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-02-25",
        "current_stepf_ret": -0.0303410958715631,
        "fixed_best_ret": -0.0262609815244532,
        "oracle_ret": 0.0044405162980349,
        "oracle_expert": "dprime_bnf_h02",
        "cumulative_regret_vs_oracle": 1.1501383104763783,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-02-28",
        "current_stepf_ret": 0.0035041934827964,
        "fixed_best_ret": 0.0091521280982443,
        "oracle_ret": 0.0105497967063235,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 1.1571839136999054,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-03-01",
        "current_stepf_ret": -0.0054569364669922,
        "fixed_best_ret": 0.0292247442815872,
        "oracle_ret": 0.0317754323897287,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 1.1944162825566262,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-03-02",
        "current_stepf_ret": 0.0472018642595821,
        "fixed_best_ret": 0.0300898775664017,
        "oracle_ret": 0.0914697338389207,
        "oracle_expert": "dprime_mix_3scale",
        "cumulative_regret_vs_oracle": 1.2386841521359648,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-03-03",
        "current_stepf_ret": 0.0113498327365532,
        "fixed_best_ret": 0.0526209379449917,
        "oracle_ret": 0.0526209379449917,
        "oracle_expert": "dprime_all_features_h02",
        "cumulative_regret_vs_oracle": 1.2799552573444033,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-03-04",
        "current_stepf_ret": -0.0367085759213971,
        "fixed_best_ret": 0.0089316502045578,
        "oracle_ret": 0.0089316502045578,
        "oracle_expert": "dprime_all_features_h02",
        "cumulative_regret_vs_oracle": 1.3255954834703583,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-03-07",
        "current_stepf_ret": -0.0700048183154692,
        "fixed_best_ret": 0.039097581102648,
        "oracle_ret": 0.039097581102648,
        "oracle_expert": "dprime_all_features_h02",
        "cumulative_regret_vs_oracle": 1.4346978828884756,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-03-08",
        "current_stepf_ret": 0.0436013018806743,
        "fixed_best_ret": 0.0205782128609647,
        "oracle_ret": 0.0524144677221775,
        "oracle_expert": "dprime_mix_3scale",
        "cumulative_regret_vs_oracle": 1.4435110487299787,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-03-09",
        "current_stepf_ret": -0.0009085886356134,
        "fixed_best_ret": -0.0432020542196397,
        "oracle_ret": 0.0232400531316571,
        "oracle_expert": "dprime_bnf_h02",
        "cumulative_regret_vs_oracle": 1.4676596904972492,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-03-10",
        "current_stepf_ret": -0.0028398561164278,
        "fixed_best_ret": 0.04170368415802,
        "oracle_ret": 0.04170368415802,
        "oracle_expert": "dprime_all_features_h02",
        "cumulative_regret_vs_oracle": 1.512203230771697,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-03-11",
        "current_stepf_ret": -0.0265850426771694,
        "fixed_best_ret": -0.0085392687892624,
        "oracle_ret": 0.0010443353257093,
        "oracle_expert": "dprime_mix_h01",
        "cumulative_regret_vs_oracle": 1.5398326087745757,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-03-14",
        "current_stepf_ret": -0.0662658557749315,
        "fixed_best_ret": -0.0252712269834404,
        "oracle_ret": -0.0095748980972156,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 1.5965235664522917,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-03-15",
        "current_stepf_ret": 0.0967967718103754,
        "fixed_best_ret": 0.0603637674115898,
        "oracle_ret": 0.127986045718193,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 1.6277128403601093,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-03-16",
        "current_stepf_ret": -0.0506070623904494,
        "fixed_best_ret": 0.0088569478987749,
        "oracle_ret": 0.0365538878926471,
        "oracle_expert": "dprime_all_features_h01",
        "cumulative_regret_vs_oracle": 1.7148737906432059,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-03-17",
        "current_stepf_ret": -0.0114779722224809,
        "fixed_best_ret": -0.00866007142311,
        "oracle_ret": 0.0034269472656699,
        "oracle_expert": "dprime_bnf_h02",
        "cumulative_regret_vs_oracle": 1.7297787101313566,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-03-18",
        "current_stepf_ret": -0.008639672800377,
        "fixed_best_ret": 0.0164481605853287,
        "oracle_ret": 0.0244757547511282,
        "oracle_expert": "dprime_bnf_h02",
        "cumulative_regret_vs_oracle": 1.7628941376828617,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-03-21",
        "current_stepf_ret": 0.0014789428427772,
        "fixed_best_ret": -0.0003586316211545,
        "oracle_ret": 0.0040790045633912,
        "oracle_expert": "dprime_mix_3scale",
        "cumulative_regret_vs_oracle": 1.7654941994034759,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-03-22",
        "current_stepf_ret": 0.0008134901910412,
        "fixed_best_ret": 0.0043834956075126,
        "oracle_ret": 0.009257703907692,
        "oracle_expert": "dprime_bnf_h02",
        "cumulative_regret_vs_oracle": 1.7739384131201266,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-03-23",
        "current_stepf_ret": -0.0014543459385431,
        "fixed_best_ret": -0.0110026766982205,
        "oracle_ret": 0.0247978425106092,
        "oracle_expert": "dprime_mix_h01",
        "cumulative_regret_vs_oracle": 1.8001906015692788,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-03-24",
        "current_stepf_ret": 0.0823267030814733,
        "fixed_best_ret": 0.0205193718318934,
        "oracle_ret": 0.1459064833223819,
        "oracle_expert": "dprime_bnf_3scale",
        "cumulative_regret_vs_oracle": 1.8637703818101874,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-03-25",
        "current_stepf_ret": -0.0034007631916043,
        "fixed_best_ret": -0.0016697968863019,
        "oracle_ret": -0.0002914409187459,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 1.8668797040830458,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-03-28",
        "current_stepf_ret": 4.709234647630781e-05,
        "fixed_best_ret": -0.0044591885419296,
        "oracle_ret": 0.0082799098016576,
        "oracle_expert": "dprime_bnf_3scale",
        "cumulative_regret_vs_oracle": 1.8751125215382272,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-03-29",
        "current_stepf_ret": -0.014871037161604,
        "fixed_best_ret": -0.0284684447885348,
        "oracle_ret": 0.0178802023935897,
        "oracle_expert": "dprime_bnf_h02",
        "cumulative_regret_vs_oracle": 1.9078637610934208,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-03-30",
        "current_stepf_ret": 0.0402733608339341,
        "fixed_best_ret": 0.0599819631269379,
        "oracle_ret": 0.0634216364256101,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 1.931012036685097,
        "stepf_expert": "dprime_all_features_h03"
      },
      {
        "Date": "2022-03-31",
        "current_stepf_ret": -0.0276049360990997,
        "fixed_best_ret": 0.0295969114334284,
        "oracle_ret": 0.0295969114334284,
        "oracle_expert": "dprime_all_features_h02",
        "cumulative_regret_vs_oracle": 1.988213884217625,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-04-01",
        "current_stepf_ret": -0.0141120106955081,
        "fixed_best_ret": 0.0078013155713934,
        "oracle_ret": 0.0116193281221225,
        "oracle_expert": "dprime_mix_h02",
        "cumulative_regret_vs_oracle": 2.0139452230352557,
        "stepf_expert": "dprime_mix_3scale"
      }
    ],
    "stepf_selected_expert_count_by_name": {
      "dprime_all_features_h03": 61,
      "dprime_mix_3scale": 2
    },
    "stepF_reward_compare": {
      "status": "OK",
      "summary": "reward mode comparison across current/baseline/oracle",
      "rows": [
        {
          "name": "reward_legacy",
          "equity_multiple": 1.1694788161480567,
          "sharpe": 1.3221864403562424,
          "max_dd": -0.15342703146823766,
          "mean_ret": 0.0032114589262884456,
          "std_ret": 0.03855759553411437,
          "regret_vs_fixed_best": 0.4720721360446347,
          "regret_vs_oracle": 7.228080196647016,
          "win_days_vs_fixed_best": 27,
          "pick_match_rate_vs_oracle": 0.23809523809523808,
          "common_days_vs_fixed_best": 63
        },
        {
          "name": "current_stepf",
          "equity_multiple": 1.1694788161480567,
          "sharpe": 1.3221864403562424,
          "max_dd": -0.15342703146823766,
          "mean_ret": 0.0032114589262884456,
          "std_ret": 0.03855759553411437,
          "regret_vs_fixed_best": 0.4720721360446347,
          "regret_vs_oracle": 7.228080196647016,
          "win_days_vs_fixed_best": 27,
          "pick_match_rate_vs_oracle": 0.23809523809523808,
          "common_days_vs_fixed_best": 63,
          "alias_of": "reward_legacy"
        },
        {
          "name": "reward_profit_basic",
          "equity_multiple": 1.1537752136305524,
          "sharpe": 1.2333563264881477,
          "max_dd": -0.15454395728267523,
          "mean_ret": 0.0029972351144138395,
          "std_ret": 0.03857736112373417,
          "regret_vs_fixed_best": 0.48777573856213907,
          "regret_vs_oracle": 7.24378379916452,
          "win_days_vs_fixed_best": 26,
          "pick_match_rate_vs_oracle": 0.23809523809523808,
          "common_days_vs_fixed_best": 63
        },
        {
          "name": "reward_profit_regret",
          "equity_multiple": 0.7747011412762524,
          "sharpe": -1.3103575647588779,
          "max_dd": -0.30557346323654777,
          "mean_ret": -0.0032692636352134566,
          "std_ret": 0.03960594626249258,
          "regret_vs_fixed_best": 0.866849810916439,
          "regret_vs_oracle": 7.62285787151882,
          "win_days_vs_fixed_best": 23,
          "pick_match_rate_vs_oracle": 0.23809523809523808,
          "common_days_vs_fixed_best": 63
        },
        {
          "name": "reward_profit_light_risk",
          "equity_multiple": 1.1081129508741117,
          "sharpe": 0.962591855387523,
          "max_dd": -0.16361629840503344,
          "mean_ret": 0.002385687936587243,
          "std_ret": 0.03934338494984834,
          "regret_vs_fixed_best": 0.5334380013185798,
          "regret_vs_oracle": 7.289446061920961,
          "win_days_vs_fixed_best": 26,
          "pick_match_rate_vs_oracle": 0.23809523809523808,
          "common_days_vs_fixed_best": 63
        },
        {
          "name": "fixed_best",
          "equity_multiple": 1.7106039731178828,
          "sharpe": 4.453328988450917,
          "max_dd": -0.09311799492168216,
          "mean_ret": 0.00832821992510972,
          "std_ret": 0.02968709319635231,
          "regret_vs_fixed_best": 0.0,
          "regret_vs_oracle": 6.68695503967719,
          "win_days_vs_fixed_best": null,
          "pick_match_rate_vs_oracle": null,
          "common_days_vs_fixed_best": null
        },
        {
          "name": "daily_oracle",
          "equity_multiple": 8.397559012795073,
          "sharpe": 13.222531911384966,
          "max_dd": -0.06769869010821283,
          "mean_ret": 0.035178843418911546,
          "std_ret": 0.042234485068861044,
          "regret_vs_fixed_best": -6.68695503967719,
          "regret_vs_oracle": 0.0,
          "win_days_vs_fixed_best": null,
          "pick_match_rate_vs_oracle": null,
          "common_days_vs_fixed_best": null
        }
      ]
    }
  },
  "plots": {
    "items": [
      {
        "name": "equity_stepE_topN.png",
        "path": "C:\\work\\actions-runner\\_work\\ApexTraderAI\\ApexTraderAI\\temp\\eval\\equity_stepE_topN.png",
        "exists": true,
        "reason": null
      },
      {
        "name": "equity_stepF.png",
        "path": "C:\\work\\actions-runner\\_work\\ApexTraderAI\\ApexTraderAI\\temp\\eval\\equity_stepF.png",
        "exists": true,
        "reason": null
      },
      {
        "name": "bar_stepE_return.png",
        "path": "C:\\work\\actions-runner\\_work\\ApexTraderAI\\ApexTraderAI\\temp\\eval\\bar_stepE_return.png",
        "exists": true,
        "reason": null
      },
      {
        "name": "scatter_stepE_dd_vs_ret.png",
        "path": "C:\\work\\actions-runner\\_work\\ApexTraderAI\\ApexTraderAI\\temp\\eval\\scatter_stepE_dd_vs_ret.png",
        "exists": true,
        "reason": null
      },
      {
        "name": "equity_stepF_vs_fixed_best_vs_oracle.png",
        "path": "C:\\work\\actions-runner\\_work\\ApexTraderAI\\ApexTraderAI\\temp\\eval\\equity_stepF_vs_fixed_best_vs_oracle.png",
        "exists": true,
        "reason": null
      },
      {
        "name": "bar_stepF_regret.png",
        "path": "C:\\work\\actions-runner\\_work\\ApexTraderAI\\ApexTraderAI\\temp\\eval\\bar_stepF_regret.png",
        "exists": true,
        "reason": null
      },
      {
        "name": "equity_stepF_reward_modes.png",
        "path": "C:\\work\\actions-runner\\_work\\ApexTraderAI\\ApexTraderAI\\temp\\eval\\equity_stepF_reward_modes.png",
        "exists": true,
        "reason": null
      },
      {
        "name": "bar_stepF_reward_mode_regret.png",
        "path": "C:\\work\\actions-runner\\_work\\ApexTraderAI\\ApexTraderAI\\temp\\eval\\bar_stepF_reward_mode_regret.png",
        "exists": true,
        "reason": null
      }
    ],
    "notes": [
      "StepF_vs_best_StepE: StepF(1.1916) は best StepE(1.7106) に負けてる"
    ]
  }
}
```

Best-effort mode: this evaluator writes SKIP/notes and always exits 0.