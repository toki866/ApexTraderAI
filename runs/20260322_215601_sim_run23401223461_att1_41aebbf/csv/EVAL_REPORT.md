# EVAL_REPORT

- output_root: `C:\work\apex_work\output\sim\SOXL\2022-01-03`
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
| stepB_pred_close_mamba_SOXL.csv | Pred_Close_MAMBA | 1.0000 | 1.0000 | 4.7783 | 0.8864 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Pred_Close_MAMBA_PERIODIC_h01 | 1.0000 | 1.0000 | 3.3612 | 0.9257 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Delta_Close_pred_MAMBA_PERIODIC_h01 | 1.0000 | 1.0000 | 44.4991 | 0.1049 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Pred_Close_MAMBA_PERIODIC_h05 | 1.0000 | 1.0000 | 5.8343 | 0.8340 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Delta_Close_pred_MAMBA_PERIODIC_h05 | 1.0000 | 1.0000 | 42.6230 | 0.1350 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Pred_Close_MAMBA_PERIODIC_h10 | 1.0000 | 1.0000 | 7.1167 | 0.7180 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Delta_Close_pred_MAMBA_PERIODIC_h10 | 1.0000 | 1.0000 | 43.2715 | -0.1676 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Pred_Close_MAMBA_PERIODIC_h20 | 1.0000 | 1.0000 | 12.6397 | 0.6217 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Delta_Close_pred_MAMBA_PERIODIC_h20 | 1.0000 | 1.0000 | 39.2528 | 0.1223 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Pred_Close_MAMBA_PERIODIC | 1.0000 | 1.0000 | 3.3612 | 0.9257 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Delta_Close_pred_MAMBA_PERIODIC | 1.0000 | 1.0000 | 44.4991 | 0.1049 | OK |
| stepB_pred_time_all_SOXL.csv | Pred_Close_MAMBA | 1.0000 | 1.0000 | 4.7783 | 0.8864 | OK |

## StepE table
| agent | file | test_days | equity_multiple | max_dd | mean_ret | std_ret | sharpe | note | status |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| dprime_all_features_3scale | stepE_daily_log_dprime_all_features_3scale_SOXL.csv | 63 | 1.3683 | -0.2506 | 0.0064 | 0.0634 | 1.6067 |  | OK |
| dprime_all_features_h01 | stepE_daily_log_dprime_all_features_h01_SOXL.csv | 63 | 1.4567 | -0.1624 | 0.0062 | 0.0354 | 2.7681 |  | OK |
| dprime_all_features_h02 | stepE_daily_log_dprime_all_features_h02_SOXL.csv | 63 | 1.2427 | -0.1080 | 0.0038 | 0.0275 | 2.1661 |  | OK |
| dprime_all_features_h03 | stepE_daily_log_dprime_all_features_h03_SOXL.csv | 63 | 1.5023 | -0.2904 | 0.0079 | 0.0579 | 2.1639 |  | OK |
| dprime_bnf_3scale | stepE_daily_log_dprime_bnf_3scale_SOXL.csv | 63 | 1.4583 | -0.1409 | 0.0066 | 0.0514 | 2.0326 |  | OK |
| dprime_bnf_h01 | stepE_daily_log_dprime_bnf_h01_SOXL.csv | 63 | 1.0274 | -0.0942 | 0.0006 | 0.0157 | 0.5580 |  | OK |
| dprime_bnf_h02 | stepE_daily_log_dprime_bnf_h02_SOXL.csv | 63 | 0.8141 | -0.3265 | -0.0023 | 0.0458 | -0.7816 |  | OK |
| dprime_mix_3scale | stepE_daily_log_dprime_mix_3scale_SOXL.csv | 63 | 1.0056 | -0.3048 | 0.0013 | 0.0539 | 0.3794 |  | OK |
| dprime_mix_h01 | stepE_daily_log_dprime_mix_h01_SOXL.csv | 63 | 0.8720 | -0.2303 | -0.0016 | 0.0330 | -0.7533 |  | OK |
| dprime_mix_h02 | stepE_daily_log_dprime_mix_h02_SOXL.csv | 63 | 0.9443 | -0.1589 | -0.0004 | 0.0265 | -0.2639 |  | OK |

## StepF table
| file | test_days | split_source | train_rows | equity_multiple | max_dd | mean_ret | std_ret | sharpe | note | status |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| stepF_equity_marl_SOXL.csv | 63 | csv | 0 | 1.2749 | -0.1387 | 0.0046 | 0.0363 | 2.0271 |  | OK |

## StepFCompare
- status: **OK**
- summary: current StepF vs fixed-best expert vs daily oracle
- current_stepf_equity_multiple: 1.2749
- fixed_best_expert: dprime_all_features_h03
- fixed_best_equity_multiple: 1.5023
- oracle_equity_multiple: 11.9940
- regret_vs_fixed_best: 0.2274
- regret_vs_oracle: 10.7191
- stepf_win_days_vs_fixed_best: 31
- stepf_pick_match_rate_vs_oracle: 0.1270
- StepFRewardCompare status: OK (reward mode comparison across current/baseline/oracle)

### StepFRewardCompare
| name | equity_multiple | sharpe | max_dd | mean_ret | std_ret | regret_vs_fixed_best | regret_vs_oracle | win_days_vs_fixed_best | pick_match_rate_vs_oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reward_legacy | 1.2849 | 2.0271 | -0.1387 | 0.0046 | 0.0363 | 0.1945 | 10.7091 | 31 | 0.1270 |
| current_stepf | 1.2849 | 2.0271 | -0.1387 | 0.0046 | 0.0363 | 0.1945 | 10.7091 | 31 | 0.1270 |
| reward_profit_basic | 1.2849 | 2.0271 | -0.1387 | 0.0046 | 0.0363 | 0.1945 | 10.7091 | 31 | 0.1270 |
| reward_profit_regret | 1.2849 | 2.0271 | -0.1387 | 0.0046 | 0.0363 | 0.1945 | 10.7091 | 31 | 0.1270 |
| reward_profit_light_risk | 1.2849 | 2.0271 | -0.1387 | 0.0046 | 0.0363 | 0.1945 | 10.7091 | 31 | 0.1270 |
| fixed_best | 1.5023 | 2.1639 | -0.2904 | 0.0079 | 0.0579 | 0.0000 | 10.4917 | NA | NA |
| daily_oracle | 11.9940 | 15.0288 | -0.0469 | 0.0411 | 0.0434 | -10.4917 | 0.0000 | NA | NA |
- cluster_status: PENDING (cluster comparison pending)

## Diversity
- status: **OK**
- summary: agent positions look diverse
- max_corr: 0.9244
- max_match_ratio: 0.1587
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
  - note: StepF_vs_best_StepE: StepF(1.2749) は best StepE(1.5023) に負けてる

## Raw JSON
```json
{
  "output_root": "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03",
  "mode": "sim",
  "symbol": "SOXL",
  "stepA": {
    "status": "OK",
    "summary": "prices_test evaluated",
    "details": {
      "path": "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepA\\sim\\stepA_prices_test_SOXL.csv",
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
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepA\\*\\stepA_prices_test_SOXL.csv"
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
        "mae": 4.7783077784947,
        "corr": 0.8863989010971224,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Pred_Close_MAMBA_PERIODIC_h01",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 3.3611987481600862,
        "corr": 0.9256555129036935,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Delta_Close_pred_MAMBA_PERIODIC_h01",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 44.49910138082418,
        "corr": 0.10494653366490629,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Pred_Close_MAMBA_PERIODIC_h05",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 5.8343208482139195,
        "corr": 0.834016009691451,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Delta_Close_pred_MAMBA_PERIODIC_h05",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 42.622962230072204,
        "corr": 0.135039341585629,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Pred_Close_MAMBA_PERIODIC_h10",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 7.116699306624954,
        "corr": 0.7180170396541666,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Delta_Close_pred_MAMBA_PERIODIC_h10",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 43.27150148462904,
        "corr": -0.16757472035871623,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Pred_Close_MAMBA_PERIODIC_h20",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 12.639721398899857,
        "corr": 0.6217131634279468,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Delta_Close_pred_MAMBA_PERIODIC_h20",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 39.25277124567539,
        "corr": 0.12232994922488637,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Pred_Close_MAMBA_PERIODIC",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 3.3611987481600862,
        "corr": 0.9256555129036935,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Delta_Close_pred_MAMBA_PERIODIC",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 44.49910138082418,
        "corr": 0.10494653366490629,
        "status": "OK"
      },
      {
        "file": "stepB_pred_time_all_SOXL.csv",
        "pred_col": "Pred_Close_MAMBA",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 4.7783077784947,
        "corr": 0.8863989010971224,
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
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDprime\\sim\\stepDprime_state_test_*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDprime\\sim\\stepDprime_state_*_SOXL_test.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDprime\\sim\\stepDprime_state_test_*_SOXL.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDprime\\sim\\stepDprime_state_*_SOXL.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDPrime\\sim\\stepDprime_state_test_*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDPrime\\sim\\stepDprime_state_*_SOXL_test.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDPrime\\sim\\stepDprime_state_test_*_SOXL.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDPrime\\sim\\stepDprime_state_*_SOXL.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDprime\\sim\\embeddings\\stepDprime_*_SOXL_embeddings*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDprime\\sim\\embeddings\\*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDPrime\\sim\\embeddings\\stepDprime_*_SOXL_embeddings*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDPrime\\sim\\embeddings\\*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDprime\\sim\\*cluster*state*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDprime\\sim\\*cluster_id*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDprime\\sim\\*rare_flag*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDprime\\sim\\*raw20*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDprime\\sim\\*stable*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDPrime\\sim\\*cluster*state*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDPrime\\sim\\*cluster_id*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDPrime\\sim\\*rare_flag*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDPrime\\sim\\*raw20*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDPrime\\sim\\*stable*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDprime\\sim\\*cluster*input*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDprime\\sim\\*cluster_features*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDPrime\\sim\\*cluster*input*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDPrime\\sim\\*cluster_features*.csv"
      ],
      "rl_state_glob": [
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDprime\\sim\\stepDprime_state_test_*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDprime\\sim\\stepDprime_state_*_SOXL_test.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDprime\\sim\\stepDprime_state_test_*_SOXL.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDprime\\sim\\stepDprime_state_*_SOXL.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDPrime\\sim\\stepDprime_state_test_*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDPrime\\sim\\stepDprime_state_*_SOXL_test.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDPrime\\sim\\stepDprime_state_test_*_SOXL.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDPrime\\sim\\stepDprime_state_*_SOXL.csv"
      ],
      "cluster_glob": [
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDprime\\sim\\embeddings\\stepDprime_*_SOXL_embeddings*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDprime\\sim\\embeddings\\*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDPrime\\sim\\embeddings\\stepDprime_*_SOXL_embeddings*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDPrime\\sim\\embeddings\\*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDprime\\sim\\*cluster*state*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDprime\\sim\\*cluster_id*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDprime\\sim\\*rare_flag*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDprime\\sim\\*raw20*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDprime\\sim\\*stable*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDPrime\\sim\\*cluster*state*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDPrime\\sim\\*cluster_id*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDPrime\\sim\\*rare_flag*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDPrime\\sim\\*raw20*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDPrime\\sim\\*stable*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDprime\\sim\\*cluster*input*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDprime\\sim\\*cluster_features*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDPrime\\sim\\*cluster*input*.csv",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDPrime\\sim\\*cluster_features*.csv"
      ],
      "traceback_count": 0,
      "traceback_files": [],
      "failure_summary_count": 0,
      "failure_summary_files": [],
      "failure_reason": "",
      "traceback_glob": [
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDprime\\sim\\stepDprime_traceback_SOXL.log",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDprime\\sim\\stepDprime_traceback_*.log",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDPrime\\sim\\stepDprime_traceback_SOXL.log",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDPrime\\sim\\stepDprime_traceback_*.log"
      ],
      "failure_summary_glob": [
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDprime\\sim\\stepDprime_failure_summary_SOXL.json",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDprime\\sim\\stepDprime_failure_summary_*.json",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDPrime\\sim\\stepDprime_failure_summary_SOXL.json",
        "C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03\\stepDPrime\\sim\\stepDprime_failure_summary_*.json"
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
        "equity_multiple": 1.368289155505722,
        "max_dd": -0.25064269949630325,
        "mean_ret": 0.0064165673854247665,
        "std_ret": 0.06339716934913343,
        "sharpe": 1.6066939656908124,
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
        "equity_multiple": 1.4566929302903182,
        "max_dd": -0.1624208772475435,
        "mean_ret": 0.0061661831781239075,
        "std_ret": 0.03536220312166247,
        "sharpe": 2.7680719730595986,
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
        "equity_multiple": 1.2426805939996777,
        "max_dd": -0.10795745802568679,
        "mean_ret": 0.0037534716389655033,
        "std_ret": 0.027507607162799746,
        "sharpe": 2.166110440154907,
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
        "equity_multiple": 1.5022582264695035,
        "max_dd": -0.2904218677001631,
        "mean_ret": 0.007896915325879909,
        "std_ret": 0.057933125230068445,
        "sharpe": 2.1638681490604763,
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
        "equity_multiple": 1.4583252453503066,
        "max_dd": -0.14092523094499243,
        "mean_ret": 0.006584762717084365,
        "std_ret": 0.051427120274935775,
        "sharpe": 2.032582555504207,
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
        "equity_multiple": 1.027364141167546,
        "max_dd": -0.09417621428500145,
        "mean_ret": 0.0005517097684585565,
        "std_ret": 0.015696764244469738,
        "sharpe": 0.5579571001357082,
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
        "equity_multiple": 0.8140692689300564,
        "max_dd": -0.3265411902639491,
        "mean_ret": -0.00225325321364868,
        "std_ret": 0.04576500595982499,
        "sharpe": -0.781585954482978,
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
        "equity_multiple": 1.0056454522963085,
        "max_dd": -0.3047948301869785,
        "mean_ret": 0.001289460789299448,
        "std_ret": 0.05394670766370988,
        "sharpe": 0.37944030932776107,
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
        "equity_multiple": 0.8719823223360793,
        "max_dd": -0.23026803205081037,
        "mean_ret": -0.0015639485821277313,
        "std_ret": 0.03295790170161868,
        "sharpe": -0.7532917081427238,
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
        "equity_multiple": 0.9442965342456254,
        "max_dd": -0.15888310756811608,
        "mean_ret": -0.0004403006303370175,
        "std_ret": 0.026482665359457595,
        "sharpe": -0.26392946952238105,
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
        "equity_multiple": 1.2748665057392659,
        "max_dd": -0.13874378215964156,
        "mean_ret": 0.004632382248376392,
        "std_ret": 0.03627623343091481,
        "sharpe": 2.027134062360967,
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
    "max_corr": 0.9244032580200539,
    "max_match_ratio": 0.15873015873015872,
    "pairs_over_0_9999": 0,
    "all_pairs": 45,
    "identical_all_agents": false
  },
  "overall_status": "OK",
  "stepF_compare": {
    "status": "OK",
    "summary": "current StepF vs fixed-best expert vs daily oracle",
    "row": {
      "current_stepf_equity_multiple": 1.2748665057392659,
      "fixed_best_expert": "dprime_all_features_h03",
      "fixed_best_equity_multiple": 1.5022582264695035,
      "oracle_equity_multiple": 11.993970325370881,
      "regret_vs_fixed_best": 0.2273917207302376,
      "regret_vs_oracle": 10.719103819631616,
      "current_stepf_sharpe": 2.027134062360967,
      "fixed_best_sharpe": 2.1638681490604763,
      "oracle_sharpe": 15.028781906704467,
      "current_stepf_max_dd": -0.13874378215964156,
      "fixed_best_max_dd": -0.2904218677001631,
      "oracle_max_dd": -0.046857713202065776,
      "stepf_win_days_vs_fixed_best": 31,
      "stepf_common_days_vs_fixed_best": 63,
      "stepf_pick_match_rate_vs_oracle": 0.12698412698412698,
      "oracle_unique_expert_count": 10
    },
    "best_expert_name": "dprime_all_features_h03",
    "best_expert_equity_multiple": 1.5022582264695035,
    "best_expert_sharpe": 2.1638681490604763,
    "best_expert_max_dd": -0.2904218677001631,
    "oracle_equity_multiple": 11.993970325370881,
    "oracle_sharpe": 15.028781906704467,
    "oracle_max_dd": -0.046857713202065776,
    "oracle_selected_expert_count_by_name": {
      "dprime_all_features_h03": 21,
      "dprime_bnf_h02": 12,
      "dprime_all_features_3scale": 8,
      "dprime_bnf_3scale": 6,
      "dprime_mix_3scale": 4,
      "dprime_bnf_h01": 3,
      "dprime_mix_h01": 3,
      "dprime_mix_h02": 2,
      "dprime_all_features_h02": 2,
      "dprime_all_features_h01": 2
    },
    "oracle_unique_expert_count": 10,
    "cumulative_regret_vs_oracle": 2.297129818349657,
    "cluster": {
      "status": "PENDING",
      "reason": "cluster comparison pending",
      "rows": []
    },
    "series": [
      {
        "Date": "2022-01-03",
        "current_stepf_ret": 0.007871369167055,
        "fixed_best_ret": -0.0152233481155042,
        "oracle_ret": 0.0077325395413182,
        "oracle_expert": "dprime_mix_h02",
        "cumulative_regret_vs_oracle": -0.00013882962573679922,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-01-04",
        "current_stepf_ret": 0.0654429835826315,
        "fixed_best_ret": 0.0102695132375753,
        "oracle_ret": 0.0109223604798316,
        "oracle_expert": "dprime_bnf_3scale",
        "cumulative_regret_vs_oracle": -0.0546594527285367,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-01-05",
        "current_stepf_ret": -0.0052911846857241,
        "fixed_best_ret": 0.0971268244385719,
        "oracle_ret": 0.0971595069169998,
        "oracle_expert": "dprime_bnf_3scale",
        "cumulative_regret_vs_oracle": 0.04779123887418721,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-01-06",
        "current_stepf_ret": 0.0556204554074437,
        "fixed_best_ret": -0.0123080339616793,
        "oracle_ret": 0.0099822506960753,
        "oracle_expert": "dprime_mix_3scale",
        "cumulative_regret_vs_oracle": 0.002153034162818805,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-01-07",
        "current_stepf_ret": -0.0011618902458647,
        "fixed_best_ret": 0.087560345262289,
        "oracle_ret": 0.087560345262289,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 0.09087526967097251,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-01-10",
        "current_stepf_ret": -0.0252201407860615,
        "fixed_best_ret": -0.0016970992293172,
        "oracle_ret": 0.00137774969723,
        "oracle_expert": "dprime_bnf_h02",
        "cumulative_regret_vs_oracle": 0.11747316015426401,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-01-11",
        "current_stepf_ret": -0.0140897233048464,
        "fixed_best_ret": -0.039803256880646,
        "oracle_ret": 0.0118394353907609,
        "oracle_expert": "dprime_bnf_h02",
        "cumulative_regret_vs_oracle": 0.1434023188498713,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-01-12",
        "current_stepf_ret": 0.0404788178658316,
        "fixed_best_ret": -0.0207561555057764,
        "oracle_ret": 0.0006466601923323,
        "oracle_expert": "dprime_mix_h02",
        "cumulative_regret_vs_oracle": 0.10357016117637202,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-01-13",
        "current_stepf_ret": -0.0196744903957564,
        "fixed_best_ret": 0.067571425974369,
        "oracle_ret": 0.067571425974369,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 0.19081607754649743,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-01-14",
        "current_stepf_ret": 0.0882707304524308,
        "fixed_best_ret": -0.0699667525915434,
        "oracle_ret": 0.0449159285675652,
        "oracle_expert": "dprime_bnf_h02",
        "cumulative_regret_vs_oracle": 0.1474612756616318,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-01-18",
        "current_stepf_ret": -0.0291597357290371,
        "fixed_best_ret": 0.1283064532279968,
        "oracle_ret": 0.1283064532279968,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 0.3049274646186657,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-01-19",
        "current_stepf_ret": -0.0512945949370039,
        "fixed_best_ret": -0.0132370244177213,
        "oracle_ret": 0.0090195556771849,
        "oracle_expert": "dprime_bnf_3scale",
        "cumulative_regret_vs_oracle": 0.3652416152328545,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-01-20",
        "current_stepf_ret": -0.0347645357002427,
        "fixed_best_ret": -0.051273109665168,
        "oracle_ret": -0.0300390990856975,
        "oracle_expert": "dprime_all_features_h02",
        "cumulative_regret_vs_oracle": 0.3699670518473997,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-01-21",
        "current_stepf_ret": 0.0253774005247187,
        "fixed_best_ret": -0.0374367785291107,
        "oracle_ret": -0.0173394763649903,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 0.3272501749576907,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-01-24",
        "current_stepf_ret": -0.0003746479365429,
        "fixed_best_ret": 0.0279428800546545,
        "oracle_ret": 0.0388078635334968,
        "oracle_expert": "dprime_all_features_3scale",
        "cumulative_regret_vs_oracle": 0.3664326864277304,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-01-25",
        "current_stepf_ret": 0.0374435929576901,
        "fixed_best_ret": 0.016784460533522,
        "oracle_ret": 0.0418955727969598,
        "oracle_expert": "dprime_mix_h01",
        "cumulative_regret_vs_oracle": 0.3708846662670001,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-01-26",
        "current_stepf_ret": -0.0889662853475536,
        "fixed_best_ret": 0.0464503992125392,
        "oracle_ret": 0.0465800539478659,
        "oracle_expert": "dprime_mix_3scale",
        "cumulative_regret_vs_oracle": 0.5064310055624196,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-01-27",
        "current_stepf_ret": 0.0451622714548629,
        "fixed_best_ret": -0.1350416826755435,
        "oracle_ret": -0.0076491920631041,
        "oracle_expert": "dprime_mix_h01",
        "cumulative_regret_vs_oracle": 0.4536195420444526,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-01-28",
        "current_stepf_ret": 0.0785841104887547,
        "fixed_best_ret": 0.0553708108067512,
        "oracle_ret": 0.0553708108067512,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 0.43040624236244907,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-01-31",
        "current_stepf_ret": -0.0059879666590926,
        "fixed_best_ret": 0.1088522152429295,
        "oracle_ret": 0.141298761141963,
        "oracle_expert": "dprime_all_features_3scale",
        "cumulative_regret_vs_oracle": 0.5776929701635046,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-02-01",
        "current_stepf_ret": 0.0237031741133306,
        "fixed_best_ret": -0.0088855163578031,
        "oracle_ret": 0.0083246798368993,
        "oracle_expert": "dprime_bnf_3scale",
        "cumulative_regret_vs_oracle": 0.5623144758870733,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-02-02",
        "current_stepf_ret": -0.0082898452933369,
        "fixed_best_ret": 0.0094771659480399,
        "oracle_ret": 0.0592263385948727,
        "oracle_expert": "dprime_bnf_3scale",
        "cumulative_regret_vs_oracle": 0.6298306597752829,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-02-03",
        "current_stepf_ret": 0.0125979709205084,
        "fixed_best_ret": -7.564159598317086e-05,
        "oracle_ret": 0.0159815040609465,
        "oracle_expert": "dprime_mix_3scale",
        "cumulative_regret_vs_oracle": 0.6332141929157209,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-02-04",
        "current_stepf_ret": -0.000925430499693,
        "fixed_best_ret": 0.0126436123407867,
        "oracle_ret": 0.0180367546081542,
        "oracle_expert": "dprime_all_features_3scale",
        "cumulative_regret_vs_oracle": 0.6521763780235681,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-02-07",
        "current_stepf_ret": 0.0350840652589725,
        "fixed_best_ret": -0.0003537797580924,
        "oracle_ret": -0.0001591962389483,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 0.6169331165256473,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-02-08",
        "current_stepf_ret": -0.0188647443016079,
        "fixed_best_ret": 0.0054156156803904,
        "oracle_ret": 0.0717704965324966,
        "oracle_expert": "dprime_mix_3scale",
        "cumulative_regret_vs_oracle": 0.7075683573597518,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-02-09",
        "current_stepf_ret": 0.0320628623369361,
        "fixed_best_ret": -0.0877443737069652,
        "oracle_ret": 0.0320128780761279,
        "oracle_expert": "dprime_bnf_h02",
        "cumulative_regret_vs_oracle": 0.7075183730989436,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-02-10",
        "current_stepf_ret": -0.0418889282334024,
        "fixed_best_ret": 0.0915252194106578,
        "oracle_ret": 0.0915252194106578,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 0.8409325207430038,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-02-11",
        "current_stepf_ret": 0.0002488375340348,
        "fixed_best_ret": 0.106493465598236,
        "oracle_ret": 0.106493465598236,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 0.947177148807205,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-02-14",
        "current_stepf_ret": 0.0600517215483437,
        "fixed_best_ret": -0.0003602950658538,
        "oracle_ret": 0.0003840016147004,
        "oracle_expert": "dprime_mix_h01",
        "cumulative_regret_vs_oracle": 0.8875094288735618,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-02-15",
        "current_stepf_ret": 4.2122826020664535e-05,
        "fixed_best_ret": -0.0204391725886185,
        "oracle_ret": 0.161162110209465,
        "oracle_expert": "dprime_bnf_3scale",
        "cumulative_regret_vs_oracle": 1.048629416257006,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-02-16",
        "current_stepf_ret": 0.0100837139160993,
        "fixed_best_ret": 0.0011036232933402,
        "oracle_ret": 0.0011036232933402,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 1.039649325634247,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-02-17",
        "current_stepf_ret": -0.0126159447183284,
        "fixed_best_ret": 0.087572045262869,
        "oracle_ret": 0.087572045262869,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 1.1398373156154444,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-02-18",
        "current_stepf_ret": -0.0090380150562269,
        "fixed_best_ret": 0.0072697462153141,
        "oracle_ret": 0.0072697462153141,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 1.1561450768869854,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-02-22",
        "current_stepf_ret": -0.0004904873553955,
        "fixed_best_ret": 0.0068741910453825,
        "oracle_ret": 0.0068741910453825,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 1.1635097552877633,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-02-23",
        "current_stepf_ret": 0.0420991306496625,
        "fixed_best_ret": 0.0512354291264741,
        "oracle_ret": 0.0512354291264741,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 1.1726460537645749,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-02-24",
        "current_stepf_ret": -0.032171526135854,
        "fixed_best_ret": 0.0060664002869422,
        "oracle_ret": 0.101733233797903,
        "oracle_expert": "dprime_all_features_3scale",
        "cumulative_regret_vs_oracle": 1.306550813698332,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-02-25",
        "current_stepf_ret": 0.0093748469745105,
        "fixed_best_ret": -0.0492560113929212,
        "oracle_ret": 0.0044241197917317,
        "oracle_expert": "dprime_bnf_h02",
        "cumulative_regret_vs_oracle": 1.301600086515553,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-02-28",
        "current_stepf_ret": 0.0250295878464097,
        "fixed_best_ret": 0.0180930794924497,
        "oracle_ret": 0.0180930794924497,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 1.294663578161593,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-03-01",
        "current_stepf_ret": 0.0284338359288309,
        "fixed_best_ret": 0.0969676407461371,
        "oracle_ret": 0.0969676407461371,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 1.3631973829788993,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-03-02",
        "current_stepf_ret": 0.034232954551638,
        "fixed_best_ret": -0.0314286167043784,
        "oracle_ret": 0.0745810036907745,
        "oracle_expert": "dprime_bnf_h02",
        "cumulative_regret_vs_oracle": 1.4035454321180358,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-03-03",
        "current_stepf_ret": -0.0163757495044322,
        "fixed_best_ret": 0.0621881750822067,
        "oracle_ret": 0.0621881750822067,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 1.4821093567046748,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-03-04",
        "current_stepf_ret": -0.0440166765568671,
        "fixed_best_ret": 0.0293092751243675,
        "oracle_ret": 0.0293092751243675,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 1.5554353083859094,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-03-07",
        "current_stepf_ret": 0.0327838231483116,
        "fixed_best_ret": 0.0072112883578734,
        "oracle_ret": 0.0279709534271791,
        "oracle_expert": "dprime_all_features_h02",
        "cumulative_regret_vs_oracle": 1.550622438664777,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-03-08",
        "current_stepf_ret": -0.0223246132896015,
        "fixed_best_ret": 0.0304304929482371,
        "oracle_ret": 0.0523601697683334,
        "oracle_expert": "dprime_all_features_3scale",
        "cumulative_regret_vs_oracle": 1.6253072217227118,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-03-09",
        "current_stepf_ret": 0.0208335384050787,
        "fixed_best_ret": -0.1027265306131752,
        "oracle_ret": 0.0365809345797591,
        "oracle_expert": "dprime_all_features_3scale",
        "cumulative_regret_vs_oracle": 1.6410546178973922,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-03-10",
        "current_stepf_ret": -0.0081580841424187,
        "fixed_best_ret": 0.0593867846727371,
        "oracle_ret": 0.0593867846727371,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 1.708599486712548,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-03-11",
        "current_stepf_ret": -0.0406583102540139,
        "fixed_best_ret": 0.0453507593279538,
        "oracle_ret": 0.0453507593279538,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 1.7946085562945158,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-03-14",
        "current_stepf_ret": 0.0708602088349275,
        "fixed_best_ret": -0.0013485904124811,
        "oracle_ret": -0.0013485904124811,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 1.7223997570471072,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-03-15",
        "current_stepf_ret": -0.0860829294716744,
        "fixed_best_ret": 0.0533813984050552,
        "oracle_ret": 0.127986045718193,
        "oracle_expert": "dprime_all_features_3scale",
        "cumulative_regret_vs_oracle": 1.9364687322369747,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-03-16",
        "current_stepf_ret": -0.0151697285707863,
        "fixed_best_ret": -0.1509019423276186,
        "oracle_ret": 0.0047371790134608,
        "oracle_expert": "dprime_bnf_h02",
        "cumulative_regret_vs_oracle": 1.9563756398212218,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-03-17",
        "current_stepf_ret": -0.0191228320443185,
        "fixed_best_ret": -0.0246966833472251,
        "oracle_ret": 0.0034334381878258,
        "oracle_expert": "dprime_bnf_h02",
        "cumulative_regret_vs_oracle": 1.978931910053366,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-03-18",
        "current_stepf_ret": 0.0021297546820806,
        "fixed_best_ret": -0.0641067976355552,
        "oracle_ret": 0.0245583338583539,
        "oracle_expert": "dprime_bnf_h02",
        "cumulative_regret_vs_oracle": 2.0013604892296395,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-03-21",
        "current_stepf_ret": -0.0020247642709794,
        "fixed_best_ret": 0.0041813470199704,
        "oracle_ret": 0.0041813470199704,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 2.0075666005205894,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-03-22",
        "current_stepf_ret": 0.0092599536651438,
        "fixed_best_ret": -0.0164639175161719,
        "oracle_ret": 0.0092714056738487,
        "oracle_expert": "dprime_bnf_h02",
        "cumulative_regret_vs_oracle": 2.0075780525292943,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-03-23",
        "current_stepf_ret": 0.0466644196496705,
        "fixed_best_ret": 0.0722984319329261,
        "oracle_ret": 0.0722984319329261,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 2.03321206481255,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-03-24",
        "current_stepf_ret": -0.0032596441375355,
        "fixed_best_ret": -0.0508982260781934,
        "oracle_ret": 0.1158741152657099,
        "oracle_expert": "dprime_all_features_3scale",
        "cumulative_regret_vs_oracle": 2.152345824215795,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-03-25",
        "current_stepf_ret": -0.0052706909545796,
        "fixed_best_ret": -0.0041847660150378,
        "oracle_ret": -0.0002919228040097,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 2.157324592366365,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-03-28",
        "current_stepf_ret": -0.0378253972586096,
        "fixed_best_ret": -0.0184529686810073,
        "oracle_ret": 0.0077648698436924,
        "oracle_expert": "dprime_bnf_h02",
        "cumulative_regret_vs_oracle": 2.2029148594686667,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-03-29",
        "current_stepf_ret": 0.0671056706839279,
        "fixed_best_ret": -0.0681294248700142,
        "oracle_ret": 0.0179214395194052,
        "oracle_expert": "dprime_bnf_h02",
        "cumulative_regret_vs_oracle": 2.153730628304144,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-03-30",
        "current_stepf_ret": -0.0077633225548128,
        "fixed_best_ret": 0.0937763547301292,
        "oracle_ret": 0.0937763547301292,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 2.2552703055890864,
        "stepf_expert": "dprime_all_features_3scale"
      },
      {
        "Date": "2022-03-31",
        "current_stepf_ret": -0.006704725431137,
        "fixed_best_ret": -0.0028046855849348,
        "oracle_ret": 0.0200413503752567,
        "oracle_expert": "dprime_all_features_h01",
        "cumulative_regret_vs_oracle": 2.28201638139548,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-04-01",
        "current_stepf_ret": -6.625796480715405e-05,
        "fixed_best_ret": -0.0069800226851999,
        "oracle_ret": 0.0150471789893695,
        "oracle_expert": "dprime_all_features_h01",
        "cumulative_regret_vs_oracle": 2.297129818349657,
        "stepf_expert": "dprime_mix_3scale"
      }
    ],
    "stepf_selected_expert_count_by_name": {
      "dprime_all_features_3scale": 61,
      "dprime_mix_3scale": 2
    },
    "stepF_reward_compare": {
      "status": "OK",
      "summary": "reward mode comparison across current/baseline/oracle",
      "rows": [
        {
          "name": "reward_legacy",
          "equity_multiple": 1.2849014506446539,
          "sharpe": 2.027134062360967,
          "max_dd": -0.13874378215964156,
          "mean_ret": 0.004632382248376392,
          "std_ret": 0.03627623343091481,
          "regret_vs_fixed_best": 0.19448737588392428,
          "regret_vs_oracle": 10.709068874726228,
          "win_days_vs_fixed_best": 31,
          "pick_match_rate_vs_oracle": 0.12698412698412698,
          "common_days_vs_fixed_best": 63
        },
        {
          "name": "current_stepf",
          "equity_multiple": 1.2849014506446539,
          "sharpe": 2.027134062360967,
          "max_dd": -0.13874378215964156,
          "mean_ret": 0.004632382248376392,
          "std_ret": 0.03627623343091481,
          "regret_vs_fixed_best": 0.19448737588392428,
          "regret_vs_oracle": 10.709068874726228,
          "win_days_vs_fixed_best": 31,
          "pick_match_rate_vs_oracle": 0.12698412698412698,
          "common_days_vs_fixed_best": 63,
          "alias_of": "reward_legacy"
        },
        {
          "name": "reward_profit_basic",
          "equity_multiple": 1.2849014506446539,
          "sharpe": 2.027134062360967,
          "max_dd": -0.13874378215964156,
          "mean_ret": 0.004632382248376392,
          "std_ret": 0.03627623343091481,
          "regret_vs_fixed_best": 0.19448737588392428,
          "regret_vs_oracle": 10.709068874726228,
          "win_days_vs_fixed_best": 31,
          "pick_match_rate_vs_oracle": 0.12698412698412698,
          "common_days_vs_fixed_best": 63
        },
        {
          "name": "reward_profit_regret",
          "equity_multiple": 1.2849014506446539,
          "sharpe": 2.027134062360967,
          "max_dd": -0.13874378215964156,
          "mean_ret": 0.004632382248376392,
          "std_ret": 0.03627623343091481,
          "regret_vs_fixed_best": 0.19448737588392428,
          "regret_vs_oracle": 10.709068874726228,
          "win_days_vs_fixed_best": 31,
          "pick_match_rate_vs_oracle": 0.12698412698412698,
          "common_days_vs_fixed_best": 63
        },
        {
          "name": "reward_profit_light_risk",
          "equity_multiple": 1.2849014506446539,
          "sharpe": 2.027134062360967,
          "max_dd": -0.13874378215964156,
          "mean_ret": 0.004632382248376392,
          "std_ret": 0.03627623343091481,
          "regret_vs_fixed_best": 0.19448737588392428,
          "regret_vs_oracle": 10.709068874726228,
          "win_days_vs_fixed_best": 31,
          "pick_match_rate_vs_oracle": 0.12698412698412698,
          "common_days_vs_fixed_best": 63
        },
        {
          "name": "fixed_best",
          "equity_multiple": 1.5022582264695035,
          "sharpe": 2.1638681490604763,
          "max_dd": -0.2904218677001631,
          "mean_ret": 0.007896915325879909,
          "std_ret": 0.057933125230068445,
          "regret_vs_fixed_best": 0.0,
          "regret_vs_oracle": 10.491712098901377,
          "win_days_vs_fixed_best": null,
          "pick_match_rate_vs_oracle": null,
          "common_days_vs_fixed_best": null
        },
        {
          "name": "daily_oracle",
          "equity_multiple": 11.993970325370881,
          "sharpe": 15.028781906704467,
          "max_dd": -0.046857713202065776,
          "mean_ret": 0.04109476031741856,
          "std_ret": 0.04340731670576367,
          "regret_vs_fixed_best": -10.491712098901377,
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
      "StepF_vs_best_StepE: StepF(1.2749) は best StepE(1.5023) に負けてる"
    ]
  }
}
```

Best-effort mode: this evaluator writes SKIP/notes and always exits 0.