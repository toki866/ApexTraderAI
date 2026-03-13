# EVAL_REPORT

- output_root: `C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03`
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
| stepB_pred_close_mamba_SOXL.csv | Pred_Close_MAMBA | 1.0000 | 1.0000 | 4.7622 | 0.8870 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Pred_Close_MAMBA_PERIODIC_h01 | 1.0000 | 1.0000 | 3.3699 | 0.9237 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Delta_Close_pred_MAMBA_PERIODIC_h01 | 1.0000 | 1.0000 | 44.4891 | 0.0531 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Pred_Close_MAMBA_PERIODIC_h05 | 1.0000 | 1.0000 | 5.8154 | 0.8360 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Delta_Close_pred_MAMBA_PERIODIC_h05 | 1.0000 | 1.0000 | 42.2799 | 0.1473 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Pred_Close_MAMBA_PERIODIC_h10 | 1.0000 | 1.0000 | 7.0332 | 0.7259 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Delta_Close_pred_MAMBA_PERIODIC_h10 | 1.0000 | 1.0000 | 43.8481 | -0.1671 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Pred_Close_MAMBA_PERIODIC_h20 | 1.0000 | 1.0000 | 13.1169 | 0.5994 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Delta_Close_pred_MAMBA_PERIODIC_h20 | 1.0000 | 1.0000 | 40.0181 | 0.1110 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Pred_Close_MAMBA_PERIODIC | 1.0000 | 1.0000 | 3.3699 | 0.9237 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Delta_Close_pred_MAMBA_PERIODIC | 1.0000 | 1.0000 | 44.4891 | 0.0531 | OK |
| stepB_pred_time_all_SOXL.csv | Pred_Close_MAMBA | 1.0000 | 1.0000 | 4.7622 | 0.8870 | OK |

## StepE table
| agent | file | test_days | equity_multiple | max_dd | mean_ret | std_ret | sharpe | note | status |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| dprime_all_features_3scale | stepE_daily_log_dprime_all_features_3scale_SOXL.csv | 63 | 0.9436 | -0.5004 | 0.0026 | 0.0718 | 0.5645 |  | OK |
| dprime_all_features_h01 | stepE_daily_log_dprime_all_features_h01_SOXL.csv | 63 | 0.4323 | -0.6638 | -0.0095 | 0.0761 | -1.9724 |  | OK |
| dprime_all_features_h02 | stepE_daily_log_dprime_all_features_h02_SOXL.csv | 63 | 0.6569 | -0.5836 | -0.0031 | 0.0725 | -0.6777 |  | OK |
| dprime_all_features_h03 | stepE_daily_log_dprime_all_features_h03_SOXL.csv | 63 | 0.9832 | -0.5098 | 0.0030 | 0.0695 | 0.6829 |  | OK |
| dprime_bnf_3scale | stepE_daily_log_dprime_bnf_3scale_SOXL.csv | 63 | 0.4754 | -0.6511 | -0.0083 | 0.0715 | -1.8451 |  | OK |
| dprime_bnf_h01 | stepE_daily_log_dprime_bnf_h01_SOXL.csv | 63 | 1.8305 | -0.3083 | 0.0109 | 0.0674 | 2.5554 |  | OK |
| dprime_bnf_h02 | stepE_daily_log_dprime_bnf_h02_SOXL.csv | 63 | 0.4496 | -0.5602 | -0.0088 | 0.0761 | -1.8444 |  | OK |
| dprime_mix_3scale | stepE_daily_log_dprime_mix_3scale_SOXL.csv | 63 | 0.6346 | -0.5198 | -0.0039 | 0.0693 | -0.8921 |  | OK |
| dprime_mix_h01 | stepE_daily_log_dprime_mix_h01_SOXL.csv | 63 | 1.2871 | -0.5213 | 0.0074 | 0.0719 | 1.6296 |  | OK |
| dprime_mix_h02 | stepE_daily_log_dprime_mix_h02_SOXL.csv | 63 | 0.7166 | -0.3748 | -0.0021 | 0.0679 | -0.4841 |  | OK |

## StepF table
| file | test_days | split_source | train_rows | equity_multiple | max_dd | mean_ret | std_ret | sharpe | note | status |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| stepF_equity_marl_SOXL.csv | 63 | summary | 0 | 0.8961 | -0.2879 | -0.0010 | 0.0380 | -0.4098 |  | OK |

## StepFCompare
- status: **OK**
- summary: current StepF vs fixed-best expert vs daily oracle
- current_stepf_equity_multiple: 0.8961
- fixed_best_expert: dprime_bnf_h01
- fixed_best_equity_multiple: 1.8305
- oracle_equity_multiple: 52.8450
- regret_vs_fixed_best: 0.9345
- regret_vs_oracle: 51.9489
- stepf_win_days_vs_fixed_best: 29
- stepf_pick_match_rate_vs_oracle: 0.0476
- StepFRewardCompare status: OK (reward mode comparison across current/baseline/oracle)

### StepFRewardCompare
| name | equity_multiple | sharpe | max_dd | mean_ret | std_ret | regret_vs_fixed_best | regret_vs_oracle | win_days_vs_fixed_best | pick_match_rate_vs_oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current_stepf | 0.8988 | -0.4098 | -0.2879 | -0.0010 | 0.0380 | 0.8214 | 51.9462 | 29 | 0.0476 |
| fixed_best | 1.8305 | 2.5554 | -0.3083 | 0.0109 | 0.0674 | 0.0000 | 51.0144 | NA | NA |
| daily_oracle | 52.8450 | 22.3411 | -0.0023 | 0.0660 | 0.0469 | -51.0144 | 0.0000 | NA | NA |
- cluster_status: PENDING (cluster comparison pending)

## Diversity
- status: **OK**
- summary: agent positions look diverse
- max_corr: 0.6029
- max_match_ratio: 0.3968
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
  - note: StepF_vs_best_StepE: StepF(0.8961) は best StepE(1.8305) に負けてる

## Raw JSON
```json
{
  "output_root": "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03",
  "mode": "sim",
  "symbol": "SOXL",
  "stepA": {
    "status": "OK",
    "summary": "prices_test evaluated",
    "details": {
      "path": "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepA\\sim\\stepA_prices_test_SOXL.csv",
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
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepA\\*\\stepA_prices_test_SOXL.csv"
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
        "mae": 4.762151142907521,
        "corr": 0.8870015163479097,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Pred_Close_MAMBA_PERIODIC_h01",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 3.3699079980394453,
        "corr": 0.9237236918156734,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Delta_Close_pred_MAMBA_PERIODIC_h01",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 44.4890944075322,
        "corr": 0.05309121089993201,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Pred_Close_MAMBA_PERIODIC_h05",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 5.8154264355098695,
        "corr": 0.8359965462414018,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Delta_Close_pred_MAMBA_PERIODIC_h05",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 42.279926888485,
        "corr": 0.14733217353480998,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Pred_Close_MAMBA_PERIODIC_h10",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 7.0331836425822605,
        "corr": 0.725865313089786,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Delta_Close_pred_MAMBA_PERIODIC_h10",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 43.848124296684794,
        "corr": -0.16706172197268024,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Pred_Close_MAMBA_PERIODIC_h20",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 13.116933817298655,
        "corr": 0.599352421956178,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Delta_Close_pred_MAMBA_PERIODIC_h20",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 40.01805456232278,
        "corr": 0.11101609228188113,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Pred_Close_MAMBA_PERIODIC",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 3.3699079980394453,
        "corr": 0.9237236918156734,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Delta_Close_pred_MAMBA_PERIODIC",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 44.4890944075322,
        "corr": 0.05309121089993201,
        "status": "OK"
      },
      {
        "file": "stepB_pred_time_all_SOXL.csv",
        "pred_col": "Pred_Close_MAMBA",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 4.762151142907521,
        "corr": 0.8870015163479097,
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
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDprime\\sim\\stepDprime_state_test_*.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDprime\\sim\\stepDprime_state_*_SOXL_test.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDprime\\sim\\stepDprime_state_test_*_SOXL.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDprime\\sim\\stepDprime_state_*_SOXL.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDPrime\\sim\\stepDprime_state_test_*.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDPrime\\sim\\stepDprime_state_*_SOXL_test.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDPrime\\sim\\stepDprime_state_test_*_SOXL.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDPrime\\sim\\stepDprime_state_*_SOXL.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDprime\\sim\\embeddings\\stepDprime_*_SOXL_embeddings*.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDprime\\sim\\embeddings\\*.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDPrime\\sim\\embeddings\\stepDprime_*_SOXL_embeddings*.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDPrime\\sim\\embeddings\\*.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDprime\\sim\\*cluster*state*.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDprime\\sim\\*cluster_id*.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDprime\\sim\\*rare_flag*.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDprime\\sim\\*raw20*.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDprime\\sim\\*stable*.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDPrime\\sim\\*cluster*state*.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDPrime\\sim\\*cluster_id*.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDPrime\\sim\\*rare_flag*.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDPrime\\sim\\*raw20*.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDPrime\\sim\\*stable*.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDprime\\sim\\*cluster*input*.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDprime\\sim\\*cluster_features*.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDPrime\\sim\\*cluster*input*.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDPrime\\sim\\*cluster_features*.csv"
      ],
      "rl_state_glob": [
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDprime\\sim\\stepDprime_state_test_*.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDprime\\sim\\stepDprime_state_*_SOXL_test.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDprime\\sim\\stepDprime_state_test_*_SOXL.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDprime\\sim\\stepDprime_state_*_SOXL.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDPrime\\sim\\stepDprime_state_test_*.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDPrime\\sim\\stepDprime_state_*_SOXL_test.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDPrime\\sim\\stepDprime_state_test_*_SOXL.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDPrime\\sim\\stepDprime_state_*_SOXL.csv"
      ],
      "cluster_glob": [
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDprime\\sim\\embeddings\\stepDprime_*_SOXL_embeddings*.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDprime\\sim\\embeddings\\*.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDPrime\\sim\\embeddings\\stepDprime_*_SOXL_embeddings*.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDPrime\\sim\\embeddings\\*.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDprime\\sim\\*cluster*state*.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDprime\\sim\\*cluster_id*.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDprime\\sim\\*rare_flag*.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDprime\\sim\\*raw20*.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDprime\\sim\\*stable*.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDPrime\\sim\\*cluster*state*.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDPrime\\sim\\*cluster_id*.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDPrime\\sim\\*rare_flag*.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDPrime\\sim\\*raw20*.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDPrime\\sim\\*stable*.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDprime\\sim\\*cluster*input*.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDprime\\sim\\*cluster_features*.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDPrime\\sim\\*cluster*input*.csv",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDPrime\\sim\\*cluster_features*.csv"
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
        "equity_multiple": 0.943572456478285,
        "max_dd": -0.5003949605610171,
        "mean_ret": 0.0025534487820706385,
        "std_ret": 0.0718032514603581,
        "sharpe": 0.5645251705596772,
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
        "equity_multiple": 0.4322772808337389,
        "max_dd": -0.6637513847474503,
        "mean_ret": -0.009455490104063302,
        "std_ret": 0.07610192415032502,
        "sharpe": -1.9723713127279898,
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
        "equity_multiple": 0.6569079628593278,
        "max_dd": -0.5835879958110313,
        "mean_ret": -0.0030949176276595237,
        "std_ret": 0.07249979728853868,
        "sharpe": -0.6776611254590291,
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
        "equity_multiple": 0.9831549520473212,
        "max_dd": -0.5098496036668981,
        "mean_ret": 0.00298847893629454,
        "std_ret": 0.0694697581785475,
        "sharpe": 0.6828961785185973,
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
        "equity_multiple": 0.475413182847401,
        "max_dd": -0.6511231420202932,
        "mean_ret": -0.008310839402233514,
        "std_ret": 0.07150486567946031,
        "sharpe": -1.845056056163411,
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
        "equity_multiple": 1.8305472764937787,
        "max_dd": -0.3082611898825852,
        "mean_ret": 0.010856195333468247,
        "std_ret": 0.06743917125224176,
        "sharpe": 2.555440036111843,
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
        "equity_multiple": 0.44959636819572785,
        "max_dd": -0.560161771433429,
        "mean_ret": -0.008846450216112826,
        "std_ret": 0.07614189782626514,
        "sharpe": -1.8443596436973295,
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
        "equity_multiple": 0.6346270619744051,
        "max_dd": -0.519763039308058,
        "mean_ret": -0.0038921126966288628,
        "std_ret": 0.06925973027310484,
        "sharpe": -0.8920822153922597,
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
        "equity_multiple": 1.287108677454538,
        "max_dd": -0.521263402625743,
        "mean_ret": 0.007382470251060133,
        "std_ret": 0.07191405249967057,
        "sharpe": 1.6296270061315616,
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
        "equity_multiple": 0.7165761669886196,
        "max_dd": -0.3748383392387905,
        "mean_ret": -0.002072267995737625,
        "std_ret": 0.06794922388485614,
        "sharpe": -0.48412965916056144,
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
        "equity_multiple": 0.8960882967337985,
        "max_dd": -0.2879106587653637,
        "mean_ret": -0.0009820299001523656,
        "std_ret": 0.03804567369849682,
        "sharpe": -0.40975069855609975,
        "split_source": "summary",
        "split_col_present": false,
        "train_rows": 0,
        "test_rows": 63,
        "status": "OK"
      }
    ]
  },
  "diversity": {
    "status": "OK",
    "summary": "agent positions look diverse",
    "max_corr": 0.6029388702618331,
    "max_match_ratio": 0.3968253968253968,
    "pairs_over_0_9999": 0,
    "all_pairs": 45,
    "identical_all_agents": false
  },
  "overall_status": "OK",
  "stepF_compare": {
    "status": "OK",
    "summary": "current StepF vs fixed-best expert vs daily oracle",
    "row": {
      "current_stepf_equity_multiple": 0.8960882967337985,
      "fixed_best_expert": "dprime_bnf_h01",
      "fixed_best_equity_multiple": 1.8305472764937787,
      "oracle_equity_multiple": 52.84499245760572,
      "regret_vs_fixed_best": 0.9344589797599802,
      "regret_vs_oracle": 51.94890416087192,
      "current_stepf_sharpe": -0.40975069855609975,
      "fixed_best_sharpe": 2.555440036111843,
      "oracle_sharpe": 22.34106474602189,
      "current_stepf_max_dd": -0.2879106587653637,
      "fixed_best_max_dd": -0.3082611898825852,
      "oracle_max_dd": -0.0023329560885888556,
      "stepf_win_days_vs_fixed_best": 29,
      "stepf_common_days_vs_fixed_best": 63,
      "stepf_pick_match_rate_vs_oracle": 0.047619047619047616,
      "oracle_unique_expert_count": 10
    },
    "best_expert_name": "dprime_bnf_h01",
    "best_expert_equity_multiple": 1.8305472764937787,
    "best_expert_sharpe": 2.555440036111843,
    "best_expert_max_dd": -0.3082611898825852,
    "oracle_equity_multiple": 52.84499245760572,
    "oracle_sharpe": 22.34106474602189,
    "oracle_max_dd": -0.0023329560885888556,
    "oracle_selected_expert_count_by_name": {
      "dprime_all_features_3scale": 12,
      "dprime_all_features_h03": 12,
      "dprime_bnf_h01": 10,
      "dprime_mix_h02": 8,
      "dprime_bnf_h02": 6,
      "dprime_mix_h01": 5,
      "dprime_all_features_h02": 4,
      "dprime_mix_3scale": 3,
      "dprime_bnf_3scale": 2,
      "dprime_all_features_h01": 1
    },
    "oracle_unique_expert_count": 10,
    "cumulative_regret_vs_oracle": 4.220163518258351,
    "cluster": {
      "status": "PENDING",
      "reason": "cluster comparison pending",
      "rows": []
    },
    "series": [
      {
        "Date": "2022-01-03",
        "current_stepf_ret": 0.0029779308379689,
        "fixed_best_ret": -0.0602793465852737,
        "oracle_ret": 0.0591381585001945,
        "oracle_expert": "dprime_all_features_3scale",
        "cumulative_regret_vs_oracle": 0.0561602276622256,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-01-04",
        "current_stepf_ret": 0.0602085428079396,
        "fixed_best_ret": 0.0109223604798316,
        "oracle_ret": 0.0109223604798316,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 0.006874045334117604,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-01-05",
        "current_stepf_ret": 0.0164960256616596,
        "fixed_best_ret": 0.0971595069169998,
        "oracle_ret": 0.0971595069169998,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 0.0875375265894578,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-01-06",
        "current_stepf_ret": 0.025951223171884,
        "fixed_best_ret": 0.0219024944603443,
        "oracle_ret": 0.0226612703502178,
        "oracle_expert": "dprime_mix_h02",
        "cumulative_regret_vs_oracle": 0.0842475737677916,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-01-07",
        "current_stepf_ret": 0.0023863574712555,
        "fixed_best_ret": 0.0845389275211133,
        "oracle_ret": 0.0878252183794975,
        "oracle_expert": "dprime_bnf_3scale",
        "cumulative_regret_vs_oracle": 0.16968643467603362,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-01-10",
        "current_stepf_ret": 0.0325479547239054,
        "fixed_best_ret": 0.0023654027059674,
        "oracle_ret": 0.0033521644547581,
        "oracle_expert": "dprime_mix_h02",
        "cumulative_regret_vs_oracle": 0.14049064440688633,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-01-11",
        "current_stepf_ret": -0.0058779508886877,
        "fixed_best_ret": 0.0528974863546817,
        "oracle_ret": 0.053833348840475,
        "oracle_expert": "dprime_mix_h02",
        "cumulative_regret_vs_oracle": 0.20020194413604903,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-01-12",
        "current_stepf_ret": -0.0266914810731585,
        "fixed_best_ret": -0.0091562754743062,
        "oracle_ret": 0.0193823819756507,
        "oracle_expert": "dprime_mix_h02",
        "cumulative_regret_vs_oracle": 0.24627580718485823,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-01-13",
        "current_stepf_ret": -0.0211369661042813,
        "fixed_best_ret": -0.0695339082479477,
        "oracle_ret": 0.067571425974369,
        "oracle_expert": "dprime_mix_h01",
        "cumulative_regret_vs_oracle": 0.33498419926350853,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-01-14",
        "current_stepf_ret": 0.0846209641921148,
        "fixed_best_ret": 0.0554239416092555,
        "oracle_ret": 0.0700963904857635,
        "oracle_expert": "dprime_all_features_h02",
        "cumulative_regret_vs_oracle": 0.32045962555715723,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-01-18",
        "current_stepf_ret": -0.059794593882027,
        "fixed_best_ret": -0.0061876988860676,
        "oracle_ret": 0.1283103396892547,
        "oracle_expert": "dprime_all_features_3scale",
        "cumulative_regret_vs_oracle": 0.5085645591284389,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-01-19",
        "current_stepf_ret": -0.0926119063460285,
        "fixed_best_ret": -0.0936537263114005,
        "oracle_ret": 0.0147349062866021,
        "oracle_expert": "dprime_all_features_3scale",
        "cumulative_regret_vs_oracle": 0.6159113717610695,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-01-20",
        "current_stepf_ret": -0.0350421348875655,
        "fixed_best_ret": -0.0960412824749946,
        "oracle_ret": 0.0023590057059562,
        "oracle_expert": "dprime_mix_h02",
        "cumulative_regret_vs_oracle": 0.6533125123545912,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-01-21",
        "current_stepf_ret": 0.0190181718621229,
        "fixed_best_ret": -0.0505760478079319,
        "oracle_ret": 0.0445470192891346,
        "oracle_expert": "dprime_all_features_3scale",
        "cumulative_regret_vs_oracle": 0.6788413597816029,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-01-24",
        "current_stepf_ret": 0.0422677974592314,
        "fixed_best_ret": -0.0265833512854248,
        "oracle_ret": 0.0388078635334968,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 0.6753814258558682,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-01-25",
        "current_stepf_ret": 0.0155638065137345,
        "fixed_best_ret": 0.0946514683999004,
        "oracle_ret": 0.1187777064442634,
        "oracle_expert": "dprime_bnf_3scale",
        "cumulative_regret_vs_oracle": 0.7785953257863971,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-01-26",
        "current_stepf_ret": 0.0319504547559992,
        "fixed_best_ret": -0.0539344159662723,
        "oracle_ret": 0.0470219051241874,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 0.7936667761545854,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-01-27",
        "current_stepf_ret": 0.0212831605435647,
        "fixed_best_ret": 0.1384422352313995,
        "oracle_ret": 0.1384422352313995,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 0.9108258508424202,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-01-28",
        "current_stepf_ret": -0.0306768558706613,
        "fixed_best_ret": -0.0485230667025491,
        "oracle_ret": 0.0553806146383285,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 0.99688332135141,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-01-31",
        "current_stepf_ret": -0.0143194493662975,
        "fixed_best_ret": -0.0898071721213225,
        "oracle_ret": 0.159635307431221,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 1.1708380781489285,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-02-01",
        "current_stepf_ret": -0.0270184694993933,
        "fixed_best_ret": -1.0091146185544766e-05,
        "oracle_ret": 0.0288339612782001,
        "oracle_expert": "dprime_bnf_h02",
        "cumulative_regret_vs_oracle": 1.2266905089265219,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-02-02",
        "current_stepf_ret": -0.0276076303339949,
        "fixed_best_ret": -0.0747309913020581,
        "oracle_ret": 0.0727113972902298,
        "oracle_expert": "dprime_all_features_h02",
        "cumulative_regret_vs_oracle": 1.3270095365507466,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-02-03",
        "current_stepf_ret": -0.0053264077276102,
        "fixed_best_ret": 0.1323333402872085,
        "oracle_ret": 0.1323333402872085,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 1.4646692845655653,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-02-04",
        "current_stepf_ret": -0.0001241410480789,
        "fixed_best_ret": -0.0116421427427473,
        "oracle_ret": 0.0184698106497526,
        "oracle_expert": "dprime_all_features_h02",
        "cumulative_regret_vs_oracle": 1.4832632362633968,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-02-07",
        "current_stepf_ret": 0.0057259003002477,
        "fixed_best_ret": -0.0012168333232402,
        "oracle_ret": -0.0006346779270184,
        "oracle_expert": "dprime_mix_3scale",
        "cumulative_regret_vs_oracle": 1.4769026580361309,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-02-08",
        "current_stepf_ret": 0.0219623928883224,
        "fixed_best_ret": 0.0331433432452116,
        "oracle_ret": 0.0592225281107706,
        "oracle_expert": "dprime_all_features_h01",
        "cumulative_regret_vs_oracle": 1.514162793258579,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-02-09",
        "current_stepf_ret": -0.0079303470481371,
        "fixed_best_ret": 0.0813280827314612,
        "oracle_ret": 0.093415010459721,
        "oracle_expert": "dprime_bnf_h02",
        "cumulative_regret_vs_oracle": 1.615508150766437,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-02-10",
        "current_stepf_ret": 0.0117612552665294,
        "fixed_best_ret": 0.0038236132120289,
        "oracle_ret": 0.0915925895571708,
        "oracle_expert": "dprime_all_features_3scale",
        "cumulative_regret_vs_oracle": 1.6953394850570784,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-02-11",
        "current_stepf_ret": -0.00035245311893,
        "fixed_best_ret": 0.0349460950920329,
        "oracle_ret": 0.1442784538269043,
        "oracle_expert": "dprime_all_features_3scale",
        "cumulative_regret_vs_oracle": 1.8399703920029127,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-02-14",
        "current_stepf_ret": 0.1149398869194238,
        "fixed_best_ret": -0.0003562870277091,
        "oracle_ret": 0.0002645228998735,
        "oracle_expert": "dprime_bnf_h02",
        "cumulative_regret_vs_oracle": 1.7252950279833623,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-02-15",
        "current_stepf_ret": -0.0005337645545822,
        "fixed_best_ret": 0.161162110209465,
        "oracle_ret": 0.161162110209465,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 1.8869909027474094,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-02-16",
        "current_stepf_ret": -0.082704823516297,
        "fixed_best_ret": 0.0004531890491838,
        "oracle_ret": 0.0014536305479705,
        "oracle_expert": "dprime_all_features_3scale",
        "cumulative_regret_vs_oracle": 1.971149356811677,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-02-17",
        "current_stepf_ret": 0.003769307398499,
        "fixed_best_ret": -0.114707453712821,
        "oracle_ret": 0.1106751283407211,
        "oracle_expert": "dprime_all_features_3scale",
        "cumulative_regret_vs_oracle": 2.078055177753899,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-02-18",
        "current_stepf_ret": -0.0149951478054546,
        "fixed_best_ret": 0.0142935563263813,
        "oracle_ret": 0.0258087461143732,
        "oracle_expert": "dprime_mix_3scale",
        "cumulative_regret_vs_oracle": 2.118859071673727,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-02-22",
        "current_stepf_ret": 0.010425803257385,
        "fixed_best_ret": -0.0078926340047803,
        "oracle_ret": 0.023173804461956,
        "oracle_expert": "dprime_all_features_3scale",
        "cumulative_regret_vs_oracle": 2.131607072878298,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-02-23",
        "current_stepf_ret": 0.0695915349905039,
        "fixed_best_ret": 0.0336574115839507,
        "oracle_ret": 0.0727527087330818,
        "oracle_expert": "dprime_all_features_3scale",
        "cumulative_regret_vs_oracle": 2.134768246620876,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-02-24",
        "current_stepf_ret": -0.0203451083496528,
        "fixed_best_ret": 0.0214114429507827,
        "oracle_ret": 0.1112249737381935,
        "oracle_expert": "dprime_all_features_h02",
        "cumulative_regret_vs_oracle": 2.266338328708722,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-02-25",
        "current_stepf_ret": 0.000695227598529,
        "fixed_best_ret": 0.0382636969197672,
        "oracle_ret": 0.0450691079497337,
        "oracle_expert": "dprime_mix_h02",
        "cumulative_regret_vs_oracle": 2.310712209059927,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-02-28",
        "current_stepf_ret": -0.0529212113516744,
        "fixed_best_ret": 0.0028828476766217,
        "oracle_ret": 0.0180930794924497,
        "oracle_expert": "dprime_all_features_3scale",
        "cumulative_regret_vs_oracle": 2.381726499904051,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-01",
        "current_stepf_ret": -0.0069121328612468,
        "fixed_best_ret": 0.0061618789363687,
        "oracle_ret": 0.0537839653571005,
        "oracle_expert": "dprime_all_features_3scale",
        "cumulative_regret_vs_oracle": 2.4424225981223984,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-02",
        "current_stepf_ret": -0.0227895334088597,
        "fixed_best_ret": -0.0991343361642211,
        "oracle_ret": 0.0957741087079048,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 2.560986240239163,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-03",
        "current_stepf_ret": -0.0325251308637853,
        "fixed_best_ret": 0.0625294094681739,
        "oracle_ret": 0.0625294094681739,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 2.6560407805711224,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-04",
        "current_stepf_ret": -0.0518125754159174,
        "fixed_best_ret": 0.0519642636511408,
        "oracle_ret": 0.0719099997580051,
        "oracle_expert": "dprime_mix_h02",
        "cumulative_regret_vs_oracle": 2.779763355745045,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-07",
        "current_stepf_ret": 0.0427151603354777,
        "fixed_best_ret": 0.1514377162456512,
        "oracle_ret": 0.1514377162456512,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 2.8884859116552186,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-08",
        "current_stepf_ret": -0.0168086513156986,
        "fixed_best_ret": 0.0219297359462822,
        "oracle_ret": 0.0524144677221775,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 2.9577090306930947,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-09",
        "current_stepf_ret": 0.0328311946642633,
        "fixed_best_ret": -0.0680381780754549,
        "oracle_ret": 0.1199884583950042,
        "oracle_expert": "dprime_mix_3scale",
        "cumulative_regret_vs_oracle": 3.0448662944238354,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-10",
        "current_stepf_ret": 0.0066976607265159,
        "fixed_best_ret": 0.0028373801165703,
        "oracle_ret": 0.0594751631617546,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 3.097643796859074,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-11",
        "current_stepf_ret": -0.017091106903718,
        "fixed_best_ret": 0.0616619863081723,
        "oracle_ret": 0.0621364583969116,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 3.176871362159704,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-14",
        "current_stepf_ret": 0.0347366306076129,
        "fixed_best_ret": 0.0909540226459503,
        "oracle_ret": 0.0909540226459503,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 3.2330887541980413,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-15",
        "current_stepf_ret": -0.0692495634615729,
        "fixed_best_ret": -0.1308245638608932,
        "oracle_ret": 0.127986045718193,
        "oracle_expert": "dprime_bnf_h02",
        "cumulative_regret_vs_oracle": 3.430324363377807,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-16",
        "current_stepf_ret": -0.0017282615179985,
        "fixed_best_ret": -0.0637204232923182,
        "oracle_ret": 0.1506216540113091,
        "oracle_expert": "dprime_mix_h01",
        "cumulative_regret_vs_oracle": 3.5826742789071147,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-17",
        "current_stepf_ret": 0.0188625987683503,
        "fixed_best_ret": 0.0077749990263057,
        "oracle_ret": 0.0169063789546489,
        "oracle_expert": "dprime_mix_h01",
        "cumulative_regret_vs_oracle": 3.5807180590934133,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-18",
        "current_stepf_ret": 0.0027006855590407,
        "fixed_best_ret": 0.0588169369548559,
        "oracle_ret": 0.0588169369548559,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 3.6368343104892285,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-21",
        "current_stepf_ret": -0.0008535918579183,
        "fixed_best_ret": 0.0019067313428497,
        "oracle_ret": 0.0041813470199704,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 3.641869249367117,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-22",
        "current_stepf_ret": 0.0061938871294005,
        "fixed_best_ret": 0.0037340104223245,
        "oracle_ret": 0.013667921539396,
        "oracle_expert": "dprime_mix_h02",
        "cumulative_regret_vs_oracle": 3.6493432837771125,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-23",
        "current_stepf_ret": -0.048194947303359,
        "fixed_best_ret": -0.024986571720764,
        "oracle_ret": 0.0722984319329261,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 3.7698366630133977,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-24",
        "current_stepf_ret": -0.003866833478722,
        "fixed_best_ret": 0.1458486822545528,
        "oracle_ret": 0.1461791118383407,
        "oracle_expert": "dprime_bnf_h02",
        "cumulative_regret_vs_oracle": 3.9198826083304605,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-25",
        "current_stepf_ret": -0.0053487204869737,
        "fixed_best_ret": -0.0048571428265422,
        "oracle_ret": -0.002332956088589,
        "oracle_expert": "dprime_mix_h01",
        "cumulative_regret_vs_oracle": 3.922898372728845,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-28",
        "current_stepf_ret": -0.0682653005424493,
        "fixed_best_ret": 0.0146627187281846,
        "oracle_ret": 0.0156627187281847,
        "oracle_expert": "dprime_bnf_h02",
        "cumulative_regret_vs_oracle": 4.006826391999479,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-29",
        "current_stepf_ret": 0.059142765528506,
        "fixed_best_ret": -0.0691140924096107,
        "oracle_ret": 0.0661974420547485,
        "oracle_expert": "dprime_mix_h01",
        "cumulative_regret_vs_oracle": 4.013881068525722,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-30",
        "current_stepf_ret": 0.0368231764407325,
        "fixed_best_ret": 0.0937763547301292,
        "oracle_ret": 0.0937763547301292,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 4.070834246815118,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-31",
        "current_stepf_ret": -0.0244998412632111,
        "fixed_best_ret": 0.0571636821586003,
        "oracle_ret": 0.0687142854332923,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 4.164048373511622,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-04-01",
        "current_stepf_ret": -0.0007583086363762,
        "fixed_best_ret": 0.0102852775816347,
        "oracle_ret": 0.0553568361103534,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 4.220163518258351,
        "stepf_expert": "dprime_mix_3scale"
      }
    ],
    "stepf_selected_expert_count_by_name": {
      "dprime_mix_3scale": 63
    },
    "stepF_reward_compare": {
      "status": "OK",
      "summary": "reward mode comparison across current/baseline/oracle",
      "rows": [
        {
          "name": "current_stepf",
          "equity_multiple": 0.8987567857061863,
          "sharpe": -0.40975069855609975,
          "max_dd": -0.2879106587653626,
          "mean_ret": -0.0009820299001523656,
          "std_ret": 0.03804567369849682,
          "regret_vs_fixed_best": 0.8214462970670912,
          "regret_vs_oracle": 51.94623567189954,
          "win_days_vs_fixed_best": 29,
          "pick_match_rate_vs_oracle": 0.047619047619047616,
          "common_days_vs_fixed_best": 63
        },
        {
          "name": "fixed_best",
          "equity_multiple": 1.8305472764937787,
          "sharpe": 2.555440036111843,
          "max_dd": -0.3082611898825852,
          "mean_ret": 0.010856195333468247,
          "std_ret": 0.06743917125224176,
          "regret_vs_fixed_best": 0.0,
          "regret_vs_oracle": 51.01444518111194,
          "win_days_vs_fixed_best": null,
          "pick_match_rate_vs_oracle": null,
          "common_days_vs_fixed_best": null
        },
        {
          "name": "daily_oracle",
          "equity_multiple": 52.84499245760572,
          "sharpe": 22.34106474602189,
          "max_dd": -0.0023329560885888556,
          "mean_ret": 0.06600469261188494,
          "std_ret": 0.046899824336816,
          "regret_vs_fixed_best": -51.01444518111194,
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
        "path": "C:\\Users\\becky\\OneDrive\\デスクトップ\\Python\\apex-trader-ai\\actions-runner\\_work\\ApexTraderAI\\ApexTraderAI\\temp\\eval\\equity_stepE_topN.png",
        "exists": true,
        "reason": null
      },
      {
        "name": "equity_stepF.png",
        "path": "C:\\Users\\becky\\OneDrive\\デスクトップ\\Python\\apex-trader-ai\\actions-runner\\_work\\ApexTraderAI\\ApexTraderAI\\temp\\eval\\equity_stepF.png",
        "exists": true,
        "reason": null
      },
      {
        "name": "bar_stepE_return.png",
        "path": "C:\\Users\\becky\\OneDrive\\デスクトップ\\Python\\apex-trader-ai\\actions-runner\\_work\\ApexTraderAI\\ApexTraderAI\\temp\\eval\\bar_stepE_return.png",
        "exists": true,
        "reason": null
      },
      {
        "name": "scatter_stepE_dd_vs_ret.png",
        "path": "C:\\Users\\becky\\OneDrive\\デスクトップ\\Python\\apex-trader-ai\\actions-runner\\_work\\ApexTraderAI\\ApexTraderAI\\temp\\eval\\scatter_stepE_dd_vs_ret.png",
        "exists": true,
        "reason": null
      },
      {
        "name": "equity_stepF_vs_fixed_best_vs_oracle.png",
        "path": "C:\\Users\\becky\\OneDrive\\デスクトップ\\Python\\apex-trader-ai\\actions-runner\\_work\\ApexTraderAI\\ApexTraderAI\\temp\\eval\\equity_stepF_vs_fixed_best_vs_oracle.png",
        "exists": true,
        "reason": null
      },
      {
        "name": "bar_stepF_regret.png",
        "path": "C:\\Users\\becky\\OneDrive\\デスクトップ\\Python\\apex-trader-ai\\actions-runner\\_work\\ApexTraderAI\\ApexTraderAI\\temp\\eval\\bar_stepF_regret.png",
        "exists": true,
        "reason": null
      },
      {
        "name": "equity_stepF_reward_modes.png",
        "path": "C:\\Users\\becky\\OneDrive\\デスクトップ\\Python\\apex-trader-ai\\actions-runner\\_work\\ApexTraderAI\\ApexTraderAI\\temp\\eval\\equity_stepF_reward_modes.png",
        "exists": true,
        "reason": null
      },
      {
        "name": "bar_stepF_reward_mode_regret.png",
        "path": "C:\\Users\\becky\\OneDrive\\デスクトップ\\Python\\apex-trader-ai\\actions-runner\\_work\\ApexTraderAI\\ApexTraderAI\\temp\\eval\\bar_stepF_reward_mode_regret.png",
        "exists": true,
        "reason": null
      }
    ],
    "notes": [
      "StepF_vs_best_StepE: StepF(0.8961) は best StepE(1.8305) に負けてる"
    ]
  }
}
```

Best-effort mode: this evaluator writes SKIP/notes and always exits 0.