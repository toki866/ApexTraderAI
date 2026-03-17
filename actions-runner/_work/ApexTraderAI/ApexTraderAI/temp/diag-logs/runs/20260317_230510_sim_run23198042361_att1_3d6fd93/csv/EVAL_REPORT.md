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
| stepB_pred_close_mamba_SOXL.csv | Pred_Close_MAMBA | 1.0000 | 1.0000 | 4.7566 | 0.8870 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Pred_Close_MAMBA_PERIODIC_h01 | 1.0000 | 1.0000 | 3.3778 | 0.9254 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Delta_Close_pred_MAMBA_PERIODIC_h01 | 1.0000 | 1.0000 | 44.3653 | 0.1249 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Pred_Close_MAMBA_PERIODIC_h05 | 1.0000 | 1.0000 | 5.7658 | 0.8392 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Delta_Close_pred_MAMBA_PERIODIC_h05 | 1.0000 | 1.0000 | 42.4367 | 0.1464 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Pred_Close_MAMBA_PERIODIC_h10 | 1.0000 | 1.0000 | 7.0230 | 0.7212 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Delta_Close_pred_MAMBA_PERIODIC_h10 | 1.0000 | 1.0000 | 43.4032 | -0.1741 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Pred_Close_MAMBA_PERIODIC_h20 | 1.0000 | 1.0000 | 12.6057 | 0.6183 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Delta_Close_pred_MAMBA_PERIODIC_h20 | 1.0000 | 1.0000 | 40.3018 | 0.1410 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Pred_Close_MAMBA_PERIODIC | 1.0000 | 1.0000 | 3.3778 | 0.9254 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Delta_Close_pred_MAMBA_PERIODIC | 1.0000 | 1.0000 | 44.3653 | 0.1249 | OK |
| stepB_pred_time_all_SOXL.csv | Pred_Close_MAMBA | 1.0000 | 1.0000 | 4.7566 | 0.8870 | OK |

## StepE table
| agent | file | test_days | equity_multiple | max_dd | mean_ret | std_ret | sharpe | note | status |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| dprime_all_features_3scale | stepE_daily_log_dprime_all_features_3scale_SOXL.csv | 63 | 0.9062 | -0.4048 | 0.0020 | 0.0729 | 0.4298 |  | OK |
| dprime_all_features_h01 | stepE_daily_log_dprime_all_features_h01_SOXL.csv | 63 | 1.1783 | -0.5046 | 0.0048 | 0.0799 | 0.9507 |  | OK |
| dprime_all_features_h02 | stepE_daily_log_dprime_all_features_h02_SOXL.csv | 63 | 1.1446 | -0.3901 | 0.0050 | 0.0712 | 1.1262 |  | OK |
| dprime_all_features_h03 | stepE_daily_log_dprime_all_features_h03_SOXL.csv | 63 | 0.8564 | -0.4013 | -0.0004 | 0.0787 | -0.0769 |  | OK |
| dprime_bnf_3scale | stepE_daily_log_dprime_bnf_3scale_SOXL.csv | 63 | 0.8376 | -0.4576 | 0.0011 | 0.0786 | 0.2316 |  | OK |
| dprime_bnf_h01 | stepE_daily_log_dprime_bnf_h01_SOXL.csv | 63 | 0.6697 | -0.5324 | -0.0029 | 0.0728 | -0.6396 |  | OK |
| dprime_bnf_h02 | stepE_daily_log_dprime_bnf_h02_SOXL.csv | 63 | 0.6919 | -0.4466 | -0.0025 | 0.0696 | -0.5728 |  | OK |
| dprime_mix_3scale | stepE_daily_log_dprime_mix_3scale_SOXL.csv | 63 | 0.8855 | -0.5160 | 0.0015 | 0.0720 | 0.3290 |  | OK |
| dprime_mix_h01 | stepE_daily_log_dprime_mix_h01_SOXL.csv | 63 | 0.5718 | -0.5931 | -0.0049 | 0.0782 | -0.9989 |  | OK |
| dprime_mix_h02 | stepE_daily_log_dprime_mix_h02_SOXL.csv | 63 | 0.7286 | -0.5414 | -0.0017 | 0.0753 | -0.3502 |  | OK |

## StepF table
| file | test_days | split_source | train_rows | equity_multiple | max_dd | mean_ret | std_ret | sharpe | note | status |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| stepF_equity_marl_SOXL.csv | 63 | csv | 0 | 0.9338 | -0.2660 | 0.0000 | 0.0472 | 0.0080 |  | OK |

## StepFCompare
- status: **OK**
- summary: current StepF vs fixed-best expert vs daily oracle
- current_stepf_equity_multiple: 0.9338
- fixed_best_expert: dprime_all_features_h01
- fixed_best_equity_multiple: 1.1783
- oracle_equity_multiple: 33.0482
- regret_vs_fixed_best: 0.2445
- regret_vs_oracle: 32.1144
- stepf_win_days_vs_fixed_best: 29
- stepf_pick_match_rate_vs_oracle: 0.2540
- StepFRewardCompare status: OK (reward mode comparison across current/baseline/oracle)

### StepFRewardCompare
| name | equity_multiple | sharpe | max_dd | mean_ret | std_ret | regret_vs_fixed_best | regret_vs_oracle | win_days_vs_fixed_best | pick_match_rate_vs_oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reward_legacy | 0.9352 | 0.0080 | -0.2660 | 0.0000 | 0.0472 | 0.1726 | 32.1130 | 29 | 0.2540 |
| current_stepf | 0.9352 | 0.0080 | -0.2660 | 0.0000 | 0.0472 | 0.1726 | 32.1130 | 29 | 0.2540 |
| fixed_best | 1.1783 | 0.9507 | -0.5046 | 0.0048 | 0.0799 | 0.0000 | 31.8699 | NA | NA |
| daily_oracle | 33.0482 | 17.6691 | -0.1054 | 0.0584 | 0.0525 | -31.8699 | 0.0000 | NA | NA |
- cluster_status: PENDING (cluster comparison pending)

## Diversity
- status: **OK**
- summary: agent positions look diverse
- max_corr: 0.7579
- max_match_ratio: 0.5079
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
  - note: StepF_vs_best_StepE: StepF(0.9338) は best StepE(1.1783) に負けてる

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
        "mae": 4.7566191960894875,
        "corr": 0.8869792069204155,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Pred_Close_MAMBA_PERIODIC_h01",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 3.3778021048422513,
        "corr": 0.9254321845599862,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Delta_Close_pred_MAMBA_PERIODIC_h01",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 44.36525929835751,
        "corr": 0.12492605012091476,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Pred_Close_MAMBA_PERIODIC_h05",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 5.765824163482412,
        "corr": 0.8392379017173771,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Delta_Close_pred_MAMBA_PERIODIC_h05",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 42.436692913272786,
        "corr": 0.14636641015917218,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Pred_Close_MAMBA_PERIODIC_h10",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 7.023025958376653,
        "corr": 0.7211696880066164,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Delta_Close_pred_MAMBA_PERIODIC_h10",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 43.40321126026137,
        "corr": -0.17407851328378718,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Pred_Close_MAMBA_PERIODIC_h20",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 12.605650750102722,
        "corr": 0.618309065027306,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Delta_Close_pred_MAMBA_PERIODIC_h20",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 40.30180096814214,
        "corr": 0.14099319943006153,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Pred_Close_MAMBA_PERIODIC",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 3.3778021048422513,
        "corr": 0.9254321845599862,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Delta_Close_pred_MAMBA_PERIODIC",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 44.36525929835751,
        "corr": 0.12492605012091476,
        "status": "OK"
      },
      {
        "file": "stepB_pred_time_all_SOXL.csv",
        "pred_col": "Pred_Close_MAMBA",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 4.7566191960894875,
        "corr": 0.8869792069204155,
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
      ],
      "traceback_count": 0,
      "traceback_files": [],
      "failure_summary_count": 0,
      "failure_summary_files": [],
      "failure_reason": "",
      "traceback_glob": [
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDprime\\sim\\stepDprime_traceback_SOXL.log",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDprime\\sim\\stepDprime_traceback_*.log",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDPrime\\sim\\stepDprime_traceback_SOXL.log",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDPrime\\sim\\stepDprime_traceback_*.log"
      ],
      "failure_summary_glob": [
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDprime\\sim\\stepDprime_failure_summary_SOXL.json",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDprime\\sim\\stepDprime_failure_summary_*.json",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDPrime\\sim\\stepDprime_failure_summary_SOXL.json",
        "C:\\\\work\\\\apex_work\\\\output\\\\sim\\\\SOXL\\\\2022-01-03\\stepDPrime\\sim\\stepDprime_failure_summary_*.json"
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
        "equity_multiple": 0.9061546961486919,
        "max_dd": -0.4047756526962659,
        "mean_ret": 0.0019734182657169326,
        "std_ret": 0.07289087649277155,
        "sharpe": 0.42978003956233063,
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
        "equity_multiple": 1.1783102188952348,
        "max_dd": -0.5045962348284485,
        "mean_ret": 0.004782860230448174,
        "std_ret": 0.07986609210638392,
        "sharpe": 0.9506606664934436,
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
        "equity_multiple": 1.1446282813426387,
        "max_dd": -0.39006937723802193,
        "mean_ret": 0.005049559086167494,
        "std_ret": 0.07117456531096382,
        "sharpe": 1.1262347031546498,
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
        "equity_multiple": 0.8564044319812378,
        "max_dd": -0.4012545633906417,
        "mean_ret": -0.0003812806459352313,
        "std_ret": 0.07869713456934516,
        "sharpe": -0.07691058443642287,
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
        "equity_multiple": 0.837590082779282,
        "max_dd": -0.4575779177280679,
        "mean_ret": 0.0011469546335499006,
        "std_ret": 0.07861166736687325,
        "sharpe": 0.23161117124899028,
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
        "equity_multiple": 0.6696737948867161,
        "max_dd": -0.5323966401871272,
        "mean_ret": -0.0029320839588648567,
        "std_ret": 0.07277182105866677,
        "sharpe": -0.6396073259233847,
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
        "equity_multiple": 0.6919216439021503,
        "max_dd": -0.44657656993414274,
        "mean_ret": -0.0025121998982760856,
        "std_ret": 0.06962751550457931,
        "sharpe": -0.5727611671638472,
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
        "equity_multiple": 0.8854618523157601,
        "max_dd": -0.5160480786588897,
        "mean_ret": 0.0014916465129488727,
        "std_ret": 0.07197379296688725,
        "sharpe": 0.3289968935577763,
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
        "equity_multiple": 0.5718167190267365,
        "max_dd": -0.5931148563174227,
        "mean_ret": -0.004921200154377474,
        "std_ret": 0.0782055633501677,
        "sharpe": -0.9989267670503301,
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
        "equity_multiple": 0.7286450570190401,
        "max_dd": -0.54137659160246,
        "mean_ret": -0.0016608799169637747,
        "std_ret": 0.07529118116538117,
        "sharpe": -0.3501824635883046,
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
        "equity_multiple": 0.9338094776943089,
        "max_dd": -0.2660013472553344,
        "mean_ret": 2.3803989136435945e-05,
        "std_ret": 0.04718899809049693,
        "sharpe": 0.008007726972144679,
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
    "max_corr": 0.7578514134034933,
    "max_match_ratio": 0.5079365079365079,
    "pairs_over_0_9999": 0,
    "all_pairs": 45,
    "identical_all_agents": false
  },
  "overall_status": "OK",
  "stepF_compare": {
    "status": "OK",
    "summary": "current StepF vs fixed-best expert vs daily oracle",
    "row": {
      "current_stepf_equity_multiple": 0.9338094776943089,
      "fixed_best_expert": "dprime_all_features_h01",
      "fixed_best_equity_multiple": 1.1783102188952348,
      "oracle_equity_multiple": 33.04819125164218,
      "regret_vs_fixed_best": 0.2445007412009259,
      "regret_vs_oracle": 32.11438177394787,
      "current_stepf_sharpe": 0.008007726972144679,
      "fixed_best_sharpe": 0.9506606664934436,
      "oracle_sharpe": 17.669145658017296,
      "current_stepf_max_dd": -0.2660013472553344,
      "fixed_best_max_dd": -0.5045962348284485,
      "oracle_max_dd": -0.10538635956049269,
      "stepf_win_days_vs_fixed_best": 29,
      "stepf_common_days_vs_fixed_best": 63,
      "stepf_pick_match_rate_vs_oracle": 0.25396825396825395,
      "oracle_unique_expert_count": 10
    },
    "best_expert_name": "dprime_all_features_h01",
    "best_expert_equity_multiple": 1.1783102188952348,
    "best_expert_sharpe": 0.9506606664934436,
    "best_expert_max_dd": -0.5045962348284485,
    "oracle_equity_multiple": 33.04819125164218,
    "oracle_sharpe": 17.669145658017296,
    "oracle_max_dd": -0.10538635956049269,
    "oracle_selected_expert_count_by_name": {
      "dprime_all_features_h01": 20,
      "dprime_bnf_h01": 14,
      "dprime_all_features_h03": 6,
      "dprime_bnf_h02": 6,
      "dprime_mix_3scale": 6,
      "dprime_mix_h02": 5,
      "dprime_all_features_3scale": 3,
      "dprime_mix_h01": 1,
      "dprime_bnf_3scale": 1,
      "dprime_all_features_h02": 1
    },
    "oracle_unique_expert_count": 10,
    "cumulative_regret_vs_oracle": 3.6768123685811522,
    "cluster": {
      "status": "PENDING",
      "reason": "cluster comparison pending",
      "rows": []
    },
    "series": [
      {
        "Date": "2022-01-03",
        "current_stepf_ret": 0.0014664512121143,
        "fixed_best_ret": -0.0598587031289935,
        "oracle_ret": 0.0591381585001945,
        "oracle_expert": "dprime_all_features_3scale",
        "cumulative_regret_vs_oracle": 0.0576717072880802,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-01-04",
        "current_stepf_ret": 0.0333972147095206,
        "fixed_best_ret": 0.0086713109136796,
        "oracle_ret": 0.0109223604798316,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 0.035196853058391195,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-01-05",
        "current_stepf_ret": 0.0108321873851256,
        "fixed_best_ret": 0.0970485550761222,
        "oracle_ret": 0.0971595069169998,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 0.12152417259026539,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-01-06",
        "current_stepf_ret": -0.0238081532647894,
        "fixed_best_ret": -0.0011162455623419,
        "oracle_ret": 0.0228315385878086,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 0.1681638644428634,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-01-07",
        "current_stepf_ret": 0.0024467730005887,
        "fixed_best_ret": 0.0460554441584377,
        "oracle_ret": 0.0878252183794975,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 0.2535423098217722,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-01-10",
        "current_stepf_ret": 0.006429907138052,
        "fixed_best_ret": 0.0025899632945656,
        "oracle_ret": 0.0033521644547581,
        "oracle_expert": "dprime_bnf_h02",
        "cumulative_regret_vs_oracle": 0.2504645671384783,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-01-11",
        "current_stepf_ret": 0.0031372415044808,
        "fixed_best_ret": 0.0307083412185872,
        "oracle_ret": 0.053833348840475,
        "oracle_expert": "dprime_bnf_h02",
        "cumulative_regret_vs_oracle": 0.3011606744744725,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-01-12",
        "current_stepf_ret": 0.009482557859712,
        "fixed_best_ret": 0.0100259590587898,
        "oracle_ret": 0.0193823819756507,
        "oracle_expert": "dprime_bnf_h02",
        "cumulative_regret_vs_oracle": 0.3110604985904112,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-01-13",
        "current_stepf_ret": -0.0173922534139707,
        "fixed_best_ret": 0.0668184147179126,
        "oracle_ret": 0.067571425974369,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 0.3960241779787509,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-01-14",
        "current_stepf_ret": -0.015871194597534,
        "fixed_best_ret": 0.0233495145425562,
        "oracle_ret": 0.0676704516620374,
        "oracle_expert": "dprime_mix_h01",
        "cumulative_regret_vs_oracle": 0.4795658242383223,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-01-18",
        "current_stepf_ret": -0.0353496315513727,
        "fixed_best_ret": 0.0437320914977891,
        "oracle_ret": 0.1283103396892547,
        "oracle_expert": "dprime_mix_3scale",
        "cumulative_regret_vs_oracle": 0.6432257954789496,
        "stepf_expert": "dprime_all_features_h02"
      },
      {
        "Date": "2022-01-19",
        "current_stepf_ret": -0.0745362302099618,
        "fixed_best_ret": -0.0938466402292251,
        "oracle_ret": 0.0906030555963516,
        "oracle_expert": "dprime_mix_3scale",
        "cumulative_regret_vs_oracle": 0.8083650812852631,
        "stepf_expert": "dprime_all_features_h02"
      },
      {
        "Date": "2022-01-20",
        "current_stepf_ret": -0.0433632385332868,
        "fixed_best_ret": -0.0960412824749946,
        "oracle_ret": 0.0969020968079567,
        "oracle_expert": "dprime_bnf_3scale",
        "cumulative_regret_vs_oracle": 0.9486304166265066,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-01-21",
        "current_stepf_ret": 0.011417324662642,
        "fixed_best_ret": -0.0505760478079319,
        "oracle_ret": -0.0284990014713638,
        "oracle_expert": "dprime_mix_3scale",
        "cumulative_regret_vs_oracle": 0.9087140904925008,
        "stepf_expert": "dprime_all_features_h01"
      },
      {
        "Date": "2022-01-24",
        "current_stepf_ret": 0.0512700904169355,
        "fixed_best_ret": 0.0388078635334968,
        "oracle_ret": 0.0388078635334968,
        "oracle_expert": "dprime_all_features_h01",
        "cumulative_regret_vs_oracle": 0.8962518636090622,
        "stepf_expert": "dprime_all_features_h01"
      },
      {
        "Date": "2022-01-25",
        "current_stepf_ret": 0.0293483381298108,
        "fixed_best_ret": -0.1167316192984581,
        "oracle_ret": 0.1195073967576026,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 0.986410922236854,
        "stepf_expert": "dprime_all_features_h01"
      },
      {
        "Date": "2022-01-26",
        "current_stepf_ret": -0.0665532804129568,
        "fixed_best_ret": 0.0470219051241874,
        "oracle_ret": 0.0470219051241874,
        "oracle_expert": "dprime_all_features_h01",
        "cumulative_regret_vs_oracle": 1.0999861077739983,
        "stepf_expert": "dprime_all_features_h01"
      },
      {
        "Date": "2022-01-27",
        "current_stepf_ret": 0.0525777421632607,
        "fixed_best_ret": -0.1377520987987518,
        "oracle_ret": 0.0826512279597087,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 1.1300595935704463,
        "stepf_expert": "dprime_all_features_h01"
      },
      {
        "Date": "2022-01-28",
        "current_stepf_ret": 0.0876587526898479,
        "fixed_best_ret": 0.0553806146383285,
        "oracle_ret": 0.0553806146383285,
        "oracle_expert": "dprime_all_features_h01",
        "cumulative_regret_vs_oracle": 1.097781455518927,
        "stepf_expert": "dprime_all_features_h01"
      },
      {
        "Date": "2022-01-31",
        "current_stepf_ret": 0.0150729956332813,
        "fixed_best_ret": 0.159635307431221,
        "oracle_ret": 0.159635307431221,
        "oracle_expert": "dprime_all_features_h01",
        "cumulative_regret_vs_oracle": 1.2423437673168667,
        "stepf_expert": "dprime_all_features_h01"
      },
      {
        "Date": "2022-02-01",
        "current_stepf_ret": 0.0055984538041565,
        "fixed_best_ret": 0.0288339612782001,
        "oracle_ret": 0.0288339612782001,
        "oracle_expert": "dprime_all_features_h01",
        "cumulative_regret_vs_oracle": 1.2655792747909103,
        "stepf_expert": "dprime_all_features_h01"
      },
      {
        "Date": "2022-02-02",
        "current_stepf_ret": -0.0383230692575546,
        "fixed_best_ret": 0.0737113972902298,
        "oracle_ret": 0.0737113972902298,
        "oracle_expert": "dprime_all_features_h01",
        "cumulative_regret_vs_oracle": 1.3776137413386946,
        "stepf_expert": "dprime_all_features_h01"
      },
      {
        "Date": "2022-02-03",
        "current_stepf_ret": 0.005513937068169,
        "fixed_best_ret": -0.1369951198101043,
        "oracle_ret": 0.1323333402872085,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 1.5044331445577341,
        "stepf_expert": "dprime_all_features_h01"
      },
      {
        "Date": "2022-02-04",
        "current_stepf_ret": -0.0006357495107883,
        "fixed_best_ret": 0.0184698106497526,
        "oracle_ret": 0.0184698106497526,
        "oracle_expert": "dprime_all_features_h01",
        "cumulative_regret_vs_oracle": 1.5235387047182751,
        "stepf_expert": "dprime_all_features_h01"
      },
      {
        "Date": "2022-02-07",
        "current_stepf_ret": 0.007760741186398,
        "fixed_best_ret": -0.0021504656868055,
        "oracle_ret": -0.0002685180773844,
        "oracle_expert": "dprime_mix_3scale",
        "cumulative_regret_vs_oracle": 1.5155094454544926,
        "stepf_expert": "dprime_all_features_h01"
      },
      {
        "Date": "2022-02-08",
        "current_stepf_ret": 0.0115474255159648,
        "fixed_best_ret": 0.0724853371977806,
        "oracle_ret": 0.0724853371977806,
        "oracle_expert": "dprime_all_features_h01",
        "cumulative_regret_vs_oracle": 1.5764473571363085,
        "stepf_expert": "dprime_all_features_h01"
      },
      {
        "Date": "2022-02-09",
        "current_stepf_ret": -0.0551087708190418,
        "fixed_best_ret": 0.0938498323559761,
        "oracle_ret": 0.0938498323559761,
        "oracle_expert": "dprime_all_features_h01",
        "cumulative_regret_vs_oracle": 1.7254059603113263,
        "stepf_expert": "dprime_all_features_h01"
      },
      {
        "Date": "2022-02-10",
        "current_stepf_ret": -0.0586198555507435,
        "fixed_best_ret": -0.0911607647538185,
        "oracle_ret": 0.0910420886427164,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 1.8750679045047862,
        "stepf_expert": "dprime_all_features_h01"
      },
      {
        "Date": "2022-02-11",
        "current_stepf_ret": 0.0004112894096445,
        "fixed_best_ret": -0.1492119262218475,
        "oracle_ret": 0.1439723804444074,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 2.018628995539549,
        "stepf_expert": "dprime_all_features_h01"
      },
      {
        "Date": "2022-02-14",
        "current_stepf_ret": 0.132364349564596,
        "fixed_best_ret": 0.0002645228998735,
        "oracle_ret": 0.0002645228998735,
        "oracle_expert": "dprime_all_features_h01",
        "cumulative_regret_vs_oracle": 1.8865291688748267,
        "stepf_expert": "dprime_all_features_h01"
      },
      {
        "Date": "2022-02-15",
        "current_stepf_ret": -0.0006727246660864,
        "fixed_best_ret": 0.161162110209465,
        "oracle_ret": 0.161162110209465,
        "oracle_expert": "dprime_all_features_h01",
        "cumulative_regret_vs_oracle": 2.048364003750378,
        "stepf_expert": "dprime_all_features_h01"
      },
      {
        "Date": "2022-02-16",
        "current_stepf_ret": -0.1129922190906444,
        "fixed_best_ret": 0.0009560998771339,
        "oracle_ret": 0.0010769737325608,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 2.1624331965735832,
        "stepf_expert": "dprime_all_features_h01"
      },
      {
        "Date": "2022-02-17",
        "current_stepf_ret": -0.021015276285385,
        "fixed_best_ret": -0.1144490147233009,
        "oracle_ret": -0.1053863595604927,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 2.0780621132984756,
        "stepf_expert": "dprime_bnf_h02"
      },
      {
        "Date": "2022-02-18",
        "current_stepf_ret": -0.0217634897484333,
        "fixed_best_ret": -0.0313400512784719,
        "oracle_ret": 0.0254406851828098,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 2.1252662882297186,
        "stepf_expert": "dprime_bnf_h02"
      },
      {
        "Date": "2022-02-22",
        "current_stepf_ret": -0.0503715686575665,
        "fixed_best_ret": -0.0267380884140729,
        "oracle_ret": 0.0007398527066379,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 2.176377709593923,
        "stepf_expert": "dprime_bnf_h02"
      },
      {
        "Date": "2022-02-23",
        "current_stepf_ret": 0.1096019399944799,
        "fixed_best_ret": -0.0709300915002822,
        "oracle_ret": 0.0066832116320359,
        "oracle_expert": "dprime_bnf_h02",
        "cumulative_regret_vs_oracle": 2.073458981231479,
        "stepf_expert": "dprime_bnf_h02"
      },
      {
        "Date": "2022-02-24",
        "current_stepf_ret": -0.0086615462885133,
        "fixed_best_ret": 0.1112249737381935,
        "oracle_ret": 0.1112249737381935,
        "oracle_expert": "dprime_all_features_h01",
        "cumulative_regret_vs_oracle": 2.193345501258186,
        "stepf_expert": "dprime_bnf_h02"
      },
      {
        "Date": "2022-02-25",
        "current_stepf_ret": -0.0090797002493416,
        "fixed_best_ret": 0.0450691079497337,
        "oracle_ret": 0.0450691079497337,
        "oracle_expert": "dprime_all_features_h01",
        "cumulative_regret_vs_oracle": 2.2474943094572613,
        "stepf_expert": "dprime_bnf_h02"
      },
      {
        "Date": "2022-02-28",
        "current_stepf_ret": 0.0009774523360379,
        "fixed_best_ret": -0.0194298828244209,
        "oracle_ret": 0.0131987407145489,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 2.2597155978357724,
        "stepf_expert": "dprime_bnf_h02"
      },
      {
        "Date": "2022-03-01",
        "current_stepf_ret": 0.0455890202390055,
        "fixed_best_ret": -0.109022354543209,
        "oracle_ret": 0.0829417436781181,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 2.297068321274885,
        "stepf_expert": "dprime_bnf_h02"
      },
      {
        "Date": "2022-03-02",
        "current_stepf_ret": 0.0061022875713426,
        "fixed_best_ret": 0.0957741087079048,
        "oracle_ret": 0.0957741087079048,
        "oracle_expert": "dprime_all_features_h01",
        "cumulative_regret_vs_oracle": 2.386740142411447,
        "stepf_expert": "dprime_bnf_h02"
      },
      {
        "Date": "2022-03-03",
        "current_stepf_ret": -0.0432706396380834,
        "fixed_best_ret": -0.0284605154902586,
        "oracle_ret": 0.0615294094681739,
        "oracle_expert": "dprime_all_features_h02",
        "cumulative_regret_vs_oracle": 2.4915401915177045,
        "stepf_expert": "dprime_bnf_h02"
      },
      {
        "Date": "2022-03-04",
        "current_stepf_ret": -0.1248327604986377,
        "fixed_best_ret": -0.0753161270618438,
        "oracle_ret": 0.0477309261520022,
        "oracle_expert": "dprime_bnf_h02",
        "cumulative_regret_vs_oracle": 2.664103878168344,
        "stepf_expert": "dprime_bnf_h02"
      },
      {
        "Date": "2022-03-07",
        "current_stepf_ret": 0.0531649634660528,
        "fixed_best_ret": -0.1502666155099868,
        "oracle_ret": -0.0385983020430745,
        "oracle_expert": "dprime_bnf_h02",
        "cumulative_regret_vs_oracle": 2.5723406126592168,
        "stepf_expert": "dprime_bnf_h02"
      },
      {
        "Date": "2022-03-08",
        "current_stepf_ret": 0.0020634900364351,
        "fixed_best_ret": 0.0524144677221775,
        "oracle_ret": 0.0524144677221775,
        "oracle_expert": "dprime_all_features_h01",
        "cumulative_regret_vs_oracle": 2.6226915903449592,
        "stepf_expert": "dprime_bnf_h02"
      },
      {
        "Date": "2022-03-09",
        "current_stepf_ret": -0.0039859473930454,
        "fixed_best_ret": 0.0290984800324942,
        "oracle_ret": 0.1199884583950042,
        "oracle_expert": "dprime_mix_3scale",
        "cumulative_regret_vs_oracle": 2.7466659961330087,
        "stepf_expert": "dprime_bnf_h02"
      },
      {
        "Date": "2022-03-10",
        "current_stepf_ret": -0.0345208756695269,
        "fixed_best_ret": 0.0150563232720269,
        "oracle_ret": 0.0594751631617546,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 2.84066203496429,
        "stepf_expert": "dprime_bnf_h02"
      },
      {
        "Date": "2022-03-11",
        "current_stepf_ret": -0.0435954267192932,
        "fixed_best_ret": -0.0645278218090534,
        "oracle_ret": 0.0230792915352029,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 2.9073367532187864,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-14",
        "current_stepf_ret": 0.1102346564554517,
        "fixed_best_ret": -0.0915832722783088,
        "oracle_ret": 0.0902367545515298,
        "oracle_expert": "dprime_mix_h02",
        "cumulative_regret_vs_oracle": 2.8873388513148646,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-15",
        "current_stepf_ret": 0.0049842883546772,
        "fixed_best_ret": 0.127986045718193,
        "oracle_ret": 0.127986045718193,
        "oracle_expert": "dprime_all_features_h01",
        "cumulative_regret_vs_oracle": 3.0103406086783804,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-16",
        "current_stepf_ret": 0.0007703048437863,
        "fixed_best_ret": 0.1510151942968368,
        "oracle_ret": 0.1510151942968368,
        "oracle_expert": "dprime_all_features_h01",
        "cumulative_regret_vs_oracle": 3.160585498131431,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-17",
        "current_stepf_ret": 0.0083158384908238,
        "fixed_best_ret": 0.0112361974710192,
        "oracle_ret": 0.0168708174824714,
        "oracle_expert": "dprime_mix_h02",
        "cumulative_regret_vs_oracle": 3.1691404771230784,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-03-18",
        "current_stepf_ret": -0.0002518247793941,
        "fixed_best_ret": -0.0649374879300594,
        "oracle_ret": 0.0581689836382865,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 3.227561285540759,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-03-21",
        "current_stepf_ret": 0.0066996885508513,
        "fixed_best_ret": -0.0027820839397835,
        "oracle_ret": 0.0041813470199704,
        "oracle_expert": "dprime_all_features_3scale",
        "cumulative_regret_vs_oracle": 3.2250429440098785,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-03-22",
        "current_stepf_ret": -0.0223091424299272,
        "fixed_best_ret": 0.0129365886540168,
        "oracle_ret": 0.0141282092556357,
        "oracle_expert": "dprime_mix_h02",
        "cumulative_regret_vs_oracle": 3.261480295695441,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-03-23",
        "current_stepf_ret": 0.110251411054689,
        "fixed_best_ret": 0.0713341816365718,
        "oracle_ret": 0.0722984319329261,
        "oracle_expert": "dprime_mix_3scale",
        "cumulative_regret_vs_oracle": 3.223527316573678,
        "stepf_expert": "dprime_bnf_h02"
      },
      {
        "Date": "2022-03-24",
        "current_stepf_ret": -0.0011752358868064,
        "fixed_best_ret": 0.1451791118383407,
        "oracle_ret": 0.1461791118383407,
        "oracle_expert": "dprime_mix_h02",
        "cumulative_regret_vs_oracle": 3.3708816642988255,
        "stepf_expert": "dprime_bnf_h02"
      },
      {
        "Date": "2022-03-25",
        "current_stepf_ret": 0.0096000398185761,
        "fixed_best_ret": -0.0048571428265422,
        "oracle_ret": -0.0007065907840349,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 3.3605750336962146,
        "stepf_expert": "dprime_bnf_h02"
      },
      {
        "Date": "2022-03-28",
        "current_stepf_ret": 0.03444818982489,
        "fixed_best_ret": -0.0058856958890858,
        "oracle_ret": 0.0156627187281847,
        "oracle_expert": "dprime_mix_h02",
        "cumulative_regret_vs_oracle": 3.341789562599509,
        "stepf_expert": "dprime_bnf_h02"
      },
      {
        "Date": "2022-03-29",
        "current_stepf_ret": 0.0146989500965913,
        "fixed_best_ret": -0.0684632599651813,
        "oracle_ret": 0.0661974420547485,
        "oracle_expert": "dprime_all_features_3scale",
        "cumulative_regret_vs_oracle": 3.3932880545576665,
        "stepf_expert": "dprime_bnf_h02"
      },
      {
        "Date": "2022-03-30",
        "current_stepf_ret": -0.0304601177407809,
        "fixed_best_ret": 0.0937763547301292,
        "oracle_ret": 0.0937763547301292,
        "oracle_expert": "dprime_all_features_h01",
        "cumulative_regret_vs_oracle": 3.5175245270285767,
        "stepf_expert": "dprime_bnf_h02"
      },
      {
        "Date": "2022-03-31",
        "current_stepf_ret": -0.0340346661458988,
        "fixed_best_ret": 0.0687142854332923,
        "oracle_ret": 0.0687142854332923,
        "oracle_expert": "dprime_all_features_h01",
        "cumulative_regret_vs_oracle": 3.6202734786077677,
        "stepf_expert": "dprime_bnf_h02"
      },
      {
        "Date": "2022-04-01",
        "current_stepf_ret": -0.0011820538630311,
        "fixed_best_ret": 0.0553568361103534,
        "oracle_ret": 0.0553568361103534,
        "oracle_expert": "dprime_all_features_h01",
        "cumulative_regret_vs_oracle": 3.6768123685811522,
        "stepf_expert": "dprime_bnf_h02"
      }
    ],
    "stepf_selected_expert_count_by_name": {
      "dprime_bnf_h02": 23,
      "dprime_all_features_h01": 19,
      "dprime_mix_3scale": 15,
      "dprime_mix_h02": 4,
      "dprime_all_features_h02": 2
    },
    "stepF_reward_compare": {
      "status": "OK",
      "summary": "reward mode comparison across current/baseline/oracle",
      "rows": [
        {
          "name": "reward_legacy",
          "equity_multiple": 0.9351788637347576,
          "sharpe": 0.008007726972144679,
          "max_dd": -0.2660013472553343,
          "mean_ret": 2.3803989136435945e-05,
          "std_ret": 0.04718899809049693,
          "regret_vs_fixed_best": 0.1725992335737645,
          "regret_vs_oracle": 32.113012387907425,
          "win_days_vs_fixed_best": 29,
          "pick_match_rate_vs_oracle": 0.25396825396825395,
          "common_days_vs_fixed_best": 63
        },
        {
          "name": "current_stepf",
          "equity_multiple": 0.9351788637347576,
          "sharpe": 0.008007726972144679,
          "max_dd": -0.2660013472553343,
          "mean_ret": 2.3803989136435945e-05,
          "std_ret": 0.04718899809049693,
          "regret_vs_fixed_best": 0.1725992335737645,
          "regret_vs_oracle": 32.113012387907425,
          "win_days_vs_fixed_best": 29,
          "pick_match_rate_vs_oracle": 0.25396825396825395,
          "common_days_vs_fixed_best": 63,
          "alias_of": "reward_legacy"
        },
        {
          "name": "fixed_best",
          "equity_multiple": 1.1783102188952348,
          "sharpe": 0.9506606664934436,
          "max_dd": -0.5045962348284485,
          "mean_ret": 0.004782860230448174,
          "std_ret": 0.07986609210638392,
          "regret_vs_fixed_best": 0.0,
          "regret_vs_oracle": 31.869881032746946,
          "win_days_vs_fixed_best": null,
          "pick_match_rate_vs_oracle": null,
          "common_days_vs_fixed_best": null
        },
        {
          "name": "daily_oracle",
          "equity_multiple": 33.04819125164218,
          "sharpe": 17.669145658017296,
          "max_dd": -0.10538635956049269,
          "mean_ret": 0.058385905077726154,
          "std_ret": 0.05245570597364833,
          "regret_vs_fixed_best": -31.869881032746946,
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
      "StepF_vs_best_StepE: StepF(0.9338) は best StepE(1.1783) に負けてる"
    ]
  }
}
```

Best-effort mode: this evaluator writes SKIP/notes and always exits 0.