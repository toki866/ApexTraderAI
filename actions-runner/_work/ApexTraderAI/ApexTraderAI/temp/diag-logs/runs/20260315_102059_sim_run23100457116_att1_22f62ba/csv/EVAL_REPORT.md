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
| dprime_all_features_3scale | stepE_daily_log_dprime_all_features_3scale_SOXL.csv | 63 | 0.8406 | -0.3863 | 0.0008 | 0.0735 | 0.1744 |  | OK |
| dprime_all_features_h01 | stepE_daily_log_dprime_all_features_h01_SOXL.csv | 63 | 0.8573 | -0.4106 | 0.0005 | 0.0725 | 0.1149 |  | OK |
| dprime_all_features_h02 | stepE_daily_log_dprime_all_features_h02_SOXL.csv | 63 | 1.2648 | -0.4286 | 0.0058 | 0.0787 | 1.1751 |  | OK |
| dprime_all_features_h03 | stepE_daily_log_dprime_all_features_h03_SOXL.csv | 63 | 1.3033 | -0.4664 | 0.0070 | 0.0705 | 1.5757 |  | OK |
| dprime_bnf_3scale | stepE_daily_log_dprime_bnf_3scale_SOXL.csv | 63 | 1.0963 | -0.2803 | 0.0051 | 0.0742 | 1.0854 |  | OK |
| dprime_bnf_h01 | stepE_daily_log_dprime_bnf_h01_SOXL.csv | 63 | 1.8305 | -0.3083 | 0.0109 | 0.0674 | 2.5553 |  | OK |
| dprime_bnf_h02 | stepE_daily_log_dprime_bnf_h02_SOXL.csv | 63 | 0.4434 | -0.5640 | -0.0090 | 0.0765 | -1.8766 |  | OK |
| dprime_mix_3scale | stepE_daily_log_dprime_mix_3scale_SOXL.csv | 63 | 1.5308 | -0.3574 | 0.0100 | 0.0685 | 2.3143 |  | OK |
| dprime_mix_h01 | stepE_daily_log_dprime_mix_h01_SOXL.csv | 63 | 1.2871 | -0.5213 | 0.0074 | 0.0719 | 1.6296 |  | OK |
| dprime_mix_h02 | stepE_daily_log_dprime_mix_h02_SOXL.csv | 63 | 0.6875 | -0.3783 | -0.0027 | 0.0685 | -0.6237 |  | OK |

## StepF table
| file | test_days | split_source | train_rows | equity_multiple | max_dd | mean_ret | std_ret | sharpe | note | status |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| stepF_equity_marl_SOXL.csv | 63 | csv | 0 | 1.2767 | -0.1405 | 0.0047 | 0.0376 | 1.9802 |  | OK |

## StepFCompare
- status: **OK**
- summary: current StepF vs fixed-best expert vs daily oracle
- current_stepf_equity_multiple: 1.2767
- fixed_best_expert: dprime_bnf_h01
- fixed_best_equity_multiple: 1.8305
- oracle_equity_multiple: 56.2963
- regret_vs_fixed_best: 0.5538
- regret_vs_oracle: 55.0196
- stepf_win_days_vs_fixed_best: 30
- stepf_pick_match_rate_vs_oracle: 0.1270
- StepFRewardCompare status: OK (reward mode comparison across current/baseline/oracle)

### StepFRewardCompare
| name | equity_multiple | sharpe | max_dd | mean_ret | std_ret | regret_vs_fixed_best | regret_vs_oracle | win_days_vs_fixed_best | pick_match_rate_vs_oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reward_legacy | 1.2864 | 1.9802 | -0.1405 | 0.0047 | 0.0376 | 0.4338 | 55.0099 | 30 | 0.1270 |
| current_stepf | 1.2864 | 1.9802 | -0.1405 | 0.0047 | 0.0376 | 0.4338 | 55.0099 | 30 | 0.1270 |
| reward_profit_basic | 1.3203 | 2.1579 | -0.1386 | 0.0051 | 0.0375 | 0.3999 | 54.9760 | 30 | 0.1270 |
| reward_profit_regret | 0.6007 | -2.5173 | -0.4604 | -0.0071 | 0.0446 | 1.1195 | 55.6956 | 25 | 0.1270 |
| reward_profit_light_risk | 0.9820 | 0.1766 | -0.1831 | 0.0004 | 0.0385 | 0.7381 | 55.3143 | 28 | 0.1270 |
| fixed_best | 1.8305 | 2.5553 | -0.3083 | 0.0109 | 0.0674 | 0.0000 | 54.4658 | NA | NA |
| daily_oracle | 56.2963 | 23.1654 | -0.0011 | 0.0670 | 0.0459 | -54.4658 | 0.0000 | NA | NA |
- cluster_status: PENDING (cluster comparison pending)

## Diversity
- status: **OK**
- summary: agent positions look diverse
- max_corr: 0.6136
- max_match_ratio: 0.4603
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
  - note: StepF_vs_best_StepE: StepF(1.2767) は best StepE(1.8305) に負けてる

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
        "equity_multiple": 0.840604224217135,
        "max_dd": -0.38633076553002843,
        "mean_ret": 0.0008072326576928656,
        "std_ret": 0.07345898463121332,
        "sharpe": 0.17444321125431628,
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
        "equity_multiple": 0.8572786110347426,
        "max_dd": -0.4106276751131329,
        "mean_ret": 0.0005243051265094705,
        "std_ret": 0.0724681399395677,
        "sharpe": 0.11485165566692696,
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
        "equity_multiple": 1.2647769454946856,
        "max_dd": -0.4285600769847464,
        "mean_ret": 0.005825833306120936,
        "std_ret": 0.07870427138181266,
        "sharpe": 1.175059943031892,
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
        "equity_multiple": 1.3032830960329322,
        "max_dd": -0.4664391989026695,
        "mean_ret": 0.0069969585632776825,
        "std_ret": 0.0704925857277292,
        "sharpe": 1.575673137917639,
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
        "equity_multiple": 1.0963071883903632,
        "max_dd": -0.2802665571501145,
        "mean_ret": 0.005072619495996682,
        "std_ret": 0.0741877278320806,
        "sharpe": 1.085426666181964,
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
        "equity_multiple": 1.8304948145571007,
        "max_dd": -0.3082823832395679,
        "mean_ret": 0.0108557392566539,
        "std_ret": 0.06743928796560703,
        "sharpe": 2.5553282578115257,
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
        "equity_multiple": 0.4434158484580179,
        "max_dd": -0.5640107492803142,
        "mean_ret": -0.009038096322606595,
        "std_ret": 0.07645589623099817,
        "sharpe": -1.8765764086644279,
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
        "equity_multiple": 1.5308434954656474,
        "max_dd": -0.3573966739320198,
        "mean_ret": 0.009989810433694445,
        "std_ret": 0.06852462898185724,
        "sharpe": 2.3142529433525247,
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
        "equity_multiple": 1.2871055005665757,
        "max_dd": -0.5212764765381807,
        "mean_ret": 0.007382414246198522,
        "std_ret": 0.07191381811345109,
        "sharpe": 1.629619954809349,
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
        "equity_multiple": 0.6874521453025502,
        "max_dd": -0.3782867321960728,
        "mean_ret": -0.0026904666867864037,
        "std_ret": 0.06847394257653056,
        "sharpe": -0.6237385051387884,
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
        "equity_multiple": 1.2766889840731983,
        "max_dd": -0.14046465945760123,
        "mean_ret": 0.00468487397818019,
        "std_ret": 0.03755596790983082,
        "sharpe": 1.9802463618621926,
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
    "max_corr": 0.6136287832972851,
    "max_match_ratio": 0.4603174603174603,
    "pairs_over_0_9999": 0,
    "all_pairs": 45,
    "identical_all_agents": false
  },
  "overall_status": "OK",
  "stepF_compare": {
    "status": "OK",
    "summary": "current StepF vs fixed-best expert vs daily oracle",
    "row": {
      "current_stepf_equity_multiple": 1.2766889840731983,
      "fixed_best_expert": "dprime_bnf_h01",
      "fixed_best_equity_multiple": 1.8304948145571007,
      "oracle_equity_multiple": 56.296287802174454,
      "regret_vs_fixed_best": 0.5538058304839024,
      "regret_vs_oracle": 55.01959881810126,
      "current_stepf_sharpe": 1.9802463618621926,
      "fixed_best_sharpe": 2.5553282578115257,
      "oracle_sharpe": 23.16540994899783,
      "current_stepf_max_dd": -0.14046465945760123,
      "fixed_best_max_dd": -0.3082823832395679,
      "oracle_max_dd": -0.0010686313641663503,
      "stepf_win_days_vs_fixed_best": 30,
      "stepf_common_days_vs_fixed_best": 63,
      "stepf_pick_match_rate_vs_oracle": 0.12698412698412698,
      "oracle_unique_expert_count": 10
    },
    "best_expert_name": "dprime_bnf_h01",
    "best_expert_equity_multiple": 1.8304948145571007,
    "best_expert_sharpe": 2.5553282578115257,
    "best_expert_max_dd": -0.3082823832395679,
    "oracle_equity_multiple": 56.296287802174454,
    "oracle_sharpe": 23.16540994899783,
    "oracle_max_dd": -0.0010686313641663503,
    "oracle_selected_expert_count_by_name": {
      "dprime_bnf_h01": 12,
      "dprime_mix_3scale": 11,
      "dprime_all_features_h02": 10,
      "dprime_mix_h02": 8,
      "dprime_all_features_h03": 5,
      "dprime_all_features_3scale": 5,
      "dprime_all_features_h01": 4,
      "dprime_mix_h01": 3,
      "dprime_bnf_h02": 3,
      "dprime_bnf_3scale": 2
    },
    "oracle_unique_expert_count": 10,
    "cumulative_regret_vs_oracle": 3.9279687710710327,
    "cluster": {
      "status": "PENDING",
      "reason": "cluster comparison pending",
      "rows": []
    },
    "series": [
      {
        "Date": "2022-01-03",
        "current_stepf_ret": 0.0075731713090054,
        "fixed_best_ret": -0.0602793465852737,
        "oracle_ret": 0.0591381585001945,
        "oracle_expert": "dprime_mix_h02",
        "cumulative_regret_vs_oracle": 0.0515649871911891,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-01-04",
        "current_stepf_ret": 0.0594519576077512,
        "fixed_best_ret": 0.0109223604798316,
        "oracle_ret": 0.0109223604798316,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 0.0030353900632694997,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-01-05",
        "current_stepf_ret": 0.0065827974156269,
        "fixed_best_ret": 0.0971595069169998,
        "oracle_ret": 0.0971595069169998,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 0.0936120995646424,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-01-06",
        "current_stepf_ret": 0.0016265237660049,
        "fixed_best_ret": 0.0219024944603443,
        "oracle_ret": 0.0226768637895584,
        "oracle_expert": "dprime_mix_h02",
        "cumulative_regret_vs_oracle": 0.11466243958819591,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-01-07",
        "current_stepf_ret": 0.002215838717339,
        "fixed_best_ret": 0.0845497291428608,
        "oracle_ret": 0.0878252183794975,
        "oracle_expert": "dprime_mix_3scale",
        "cumulative_regret_vs_oracle": 0.20027181925035442,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-01-10",
        "current_stepf_ret": 0.0542889154753869,
        "fixed_best_ret": 0.002365340180695,
        "oracle_ret": 0.0033521644547581,
        "oracle_expert": "dprime_mix_h02",
        "cumulative_regret_vs_oracle": 0.14933506822972561,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-01-11",
        "current_stepf_ret": 0.0026318613739486,
        "fixed_best_ret": 0.0528839924614355,
        "oracle_ret": 0.053833348840475,
        "oracle_expert": "dprime_all_features_h01",
        "cumulative_regret_vs_oracle": 0.200536555696252,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-01-12",
        "current_stepf_ret": -0.0011400438769813,
        "fixed_best_ret": -0.0091613804869866,
        "oracle_ret": 0.0193823819756507,
        "oracle_expert": "dprime_mix_h02",
        "cumulative_regret_vs_oracle": 0.221058981548884,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-01-13",
        "current_stepf_ret": 0.0048419871971032,
        "fixed_best_ret": -0.0695340330898761,
        "oracle_ret": 0.067571425974369,
        "oracle_expert": "dprime_mix_h01",
        "cumulative_regret_vs_oracle": 0.2837884203261498,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-01-14",
        "current_stepf_ret": 0.0212194834702806,
        "fixed_best_ret": 0.0554119683123119,
        "oracle_ret": 0.0554119683123119,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 0.3179809051681811,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-01-18",
        "current_stepf_ret": -0.0227990592423836,
        "fixed_best_ret": -0.0062069564282146,
        "oracle_ret": 0.1281114583313465,
        "oracle_expert": "dprime_bnf_3scale",
        "cumulative_regret_vs_oracle": 0.4688914227419112,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-01-19",
        "current_stepf_ret": -0.0531066544291798,
        "fixed_best_ret": -0.0936536527145654,
        "oracle_ret": 0.0904210950136184,
        "oracle_expert": "dprime_mix_3scale",
        "cumulative_regret_vs_oracle": 0.6124191721847094,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-01-20",
        "current_stepf_ret": -0.0502453003057856,
        "fixed_best_ret": -0.0960412824749946,
        "oracle_ret": 0.0459908679041608,
        "oracle_expert": "dprime_all_features_h02",
        "cumulative_regret_vs_oracle": 0.7086553403946558,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-01-21",
        "current_stepf_ret": 0.0143786501430808,
        "fixed_best_ret": -0.0505760478079319,
        "oracle_ret": 0.0326097856545059,
        "oracle_expert": "dprime_all_features_h02",
        "cumulative_regret_vs_oracle": 0.7268864759060809,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-01-24",
        "current_stepf_ret": -0.0204455915215781,
        "fixed_best_ret": -0.0265884556128114,
        "oracle_ret": 0.0388078635334968,
        "oracle_expert": "dprime_bnf_3scale",
        "cumulative_regret_vs_oracle": 0.7861399309611559,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-01-25",
        "current_stepf_ret": 0.0243311025679011,
        "fixed_best_ret": 0.0946630575784365,
        "oracle_ret": 0.1192513318955898,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 0.8810601602888446,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-01-26",
        "current_stepf_ret": -0.0350908010516419,
        "fixed_best_ret": -0.0539343672692775,
        "oracle_ret": 0.0470219051241874,
        "oracle_expert": "dprime_all_features_h02",
        "cumulative_regret_vs_oracle": 0.9631728664646739,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-01-27",
        "current_stepf_ret": 0.0307206530423335,
        "fixed_best_ret": 0.1384422352313995,
        "oracle_ret": 0.1384422352313995,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 1.07089444865374,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-01-28",
        "current_stepf_ret": 0.0188397496171908,
        "fixed_best_ret": -0.0485281218261482,
        "oracle_ret": 0.0553806146383285,
        "oracle_expert": "dprime_all_features_h02",
        "cumulative_regret_vs_oracle": 1.1074353136748776,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-01-31",
        "current_stepf_ret": 0.0073871950297777,
        "fixed_best_ret": -0.0898039728827779,
        "oracle_ret": 0.159635307431221,
        "oracle_expert": "dprime_mix_3scale",
        "cumulative_regret_vs_oracle": 1.2596834260763208,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-02-01",
        "current_stepf_ret": 0.000444095773848,
        "fixed_best_ret": -8.367165474495812e-06,
        "oracle_ret": 0.0288339612782001,
        "oracle_expert": "dprime_mix_3scale",
        "cumulative_regret_vs_oracle": 1.288073291580673,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-02-02",
        "current_stepf_ret": 0.028672332638706,
        "fixed_best_ret": -0.0747310205381363,
        "oracle_ret": 0.0737113972902298,
        "oracle_expert": "dprime_mix_3scale",
        "cumulative_regret_vs_oracle": 1.3331123562321967,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-02-03",
        "current_stepf_ret": 0.0091201251360823,
        "fixed_best_ret": 0.1323333402872085,
        "oracle_ret": 0.1323333402872085,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 1.456325571383323,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-02-04",
        "current_stepf_ret": -0.0007847483015183,
        "fixed_best_ret": -0.0116424600336729,
        "oracle_ret": 0.0184698106497526,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 1.475580130334594,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-02-07",
        "current_stepf_ret": 0.0333559477594855,
        "fixed_best_ret": -0.0012168254852294,
        "oracle_ret": -0.0010686313641664,
        "oracle_expert": "dprime_bnf_h02",
        "cumulative_regret_vs_oracle": 1.441155551210942,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-02-08",
        "current_stepf_ret": 0.0437684931600243,
        "fixed_best_ret": 0.0331420891157608,
        "oracle_ret": 0.0724853371977806,
        "oracle_expert": "dprime_mix_3scale",
        "cumulative_regret_vs_oracle": 1.4698723952486983,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-02-09",
        "current_stepf_ret": -0.0392493858137471,
        "fixed_best_ret": 0.0813325068068498,
        "oracle_ret": 0.0938498323559761,
        "oracle_expert": "dprime_mix_3scale",
        "cumulative_regret_vs_oracle": 1.6029716134184215,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-02-10",
        "current_stepf_ret": -0.0144620263566488,
        "fixed_best_ret": 0.0038240499058582,
        "oracle_ret": 0.0915925895571708,
        "oracle_expert": "dprime_all_features_3scale",
        "cumulative_regret_vs_oracle": 1.7090262293322411,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-02-11",
        "current_stepf_ret": -6.683912372818434e-05,
        "fixed_best_ret": 0.0349359918091713,
        "oracle_ret": 0.1320262795629473,
        "oracle_expert": "dprime_all_features_3scale",
        "cumulative_regret_vs_oracle": 1.8411193480189165,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-02-14",
        "current_stepf_ret": 0.1619872832108307,
        "fixed_best_ret": -0.0003562520099803,
        "oracle_ret": 0.0002645228998735,
        "oracle_expert": "dprime_all_features_h02",
        "cumulative_regret_vs_oracle": 1.6793965877079593,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-02-15",
        "current_stepf_ret": -0.0004765832455782,
        "fixed_best_ret": 0.161162110209465,
        "oracle_ret": 0.161162110209465,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 1.8410352811630024,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-02-16",
        "current_stepf_ret": -0.0977434903244557,
        "fixed_best_ret": 0.0004530845239223,
        "oracle_ret": 0.0009571819268167,
        "oracle_expert": "dprime_all_features_3scale",
        "cumulative_regret_vs_oracle": 1.9397359534142748,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-02-17",
        "current_stepf_ret": -0.0047569861609523,
        "fixed_best_ret": -0.1147074887901544,
        "oracle_ret": 0.1106751283407211,
        "oracle_expert": "dprime_all_features_3scale",
        "cumulative_regret_vs_oracle": 2.055168067915948,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-02-18",
        "current_stepf_ret": -0.0086193153823463,
        "fixed_best_ret": 0.0142900214245321,
        "oracle_ret": 0.0254609370158158,
        "oracle_expert": "dprime_all_features_h01",
        "cumulative_regret_vs_oracle": 2.0892483203141103,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-02-22",
        "current_stepf_ret": -0.0340132941233665,
        "fixed_best_ret": -0.0079020368176735,
        "oracle_ret": 0.0234399252384901,
        "oracle_expert": "dprime_all_features_h01",
        "cumulative_regret_vs_oracle": 2.1467015396759668,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-02-23",
        "current_stepf_ret": 0.0893023171554704,
        "fixed_best_ret": 0.0336658350804851,
        "oracle_ret": 0.0336658350804851,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 2.0910650576009813,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-02-24",
        "current_stepf_ret": 0.0171549755275215,
        "fixed_best_ret": 0.0214181689994746,
        "oracle_ret": 0.1112249737381935,
        "oracle_expert": "dprime_all_features_h02",
        "cumulative_regret_vs_oracle": 2.1851350558116533,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-02-25",
        "current_stepf_ret": 0.0016589959388298,
        "fixed_best_ret": 0.038263847654124,
        "oracle_ret": 0.0450691079497337,
        "oracle_expert": "dprime_all_features_h02",
        "cumulative_regret_vs_oracle": 2.228545167822557,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-02-28",
        "current_stepf_ret": -0.0092360249135986,
        "fixed_best_ret": 0.0028832133744359,
        "oracle_ret": 0.0139842637407912,
        "oracle_expert": "dprime_all_features_3scale",
        "cumulative_regret_vs_oracle": 2.251765456476947,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-03-01",
        "current_stepf_ret": 0.0301803514600986,
        "fixed_best_ret": 0.0061519871181657,
        "oracle_ret": 0.0647693986188738,
        "oracle_expert": "dprime_all_features_h01",
        "cumulative_regret_vs_oracle": 2.2863545036357222,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-03-02",
        "current_stepf_ret": 0.0074836499121203,
        "fixed_best_ret": -0.0991343839373439,
        "oracle_ret": 0.0957741087079048,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 2.374644962431507,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-03-03",
        "current_stepf_ret": -0.0060977127514916,
        "fixed_best_ret": 0.0625294094681739,
        "oracle_ret": 0.0625294094681739,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 2.4432720846511726,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-03-04",
        "current_stepf_ret": -0.0650396138467945,
        "fixed_best_ret": 0.0519727473388627,
        "oracle_ret": 0.0719156039059162,
        "oracle_expert": "dprime_mix_h02",
        "cumulative_regret_vs_oracle": 2.580227302403883,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-03-07",
        "current_stepf_ret": 0.0373802549421989,
        "fixed_best_ret": 0.1514377751052379,
        "oracle_ret": 0.1514377751052379,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 2.6942848225669223,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-03-08",
        "current_stepf_ret": -0.0368375942721472,
        "fixed_best_ret": 0.0219289167526054,
        "oracle_ret": 0.0524144677221775,
        "oracle_expert": "dprime_mix_3scale",
        "cumulative_regret_vs_oracle": 2.783536884561247,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-03-09",
        "current_stepf_ret": 0.0282478277215865,
        "fixed_best_ret": -0.0680306632655231,
        "oracle_ret": 0.1199884583950042,
        "oracle_expert": "dprime_mix_3scale",
        "cumulative_regret_vs_oracle": 2.875277515234665,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-03-10",
        "current_stepf_ret": 0.0162352630507237,
        "fixed_best_ret": 0.0028389667640887,
        "oracle_ret": 0.0594751631617546,
        "oracle_expert": "dprime_mix_h02",
        "cumulative_regret_vs_oracle": 2.9185174153456956,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-03-11",
        "current_stepf_ret": -0.025714910832738,
        "fixed_best_ret": 0.0616619990933686,
        "oracle_ret": 0.0621364583969116,
        "oracle_expert": "dprime_mix_h02",
        "cumulative_regret_vs_oracle": 3.0063687845753453,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-03-14",
        "current_stepf_ret": 0.024008847238192,
        "fixed_best_ret": 0.0909540226459503,
        "oracle_ret": 0.0909540226459503,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 3.0733139599831034,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-03-15",
        "current_stepf_ret": -0.0099705482346047,
        "fixed_best_ret": -0.1308245638608932,
        "oracle_ret": 0.127986045718193,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 3.211270553935901,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-03-16",
        "current_stepf_ret": -0.0003111494197816,
        "fixed_best_ret": -0.063706384748183,
        "oracle_ret": 0.1510151942968368,
        "oracle_expert": "dprime_all_features_h02",
        "cumulative_regret_vs_oracle": 3.3625968976525193,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-03-17",
        "current_stepf_ret": 0.028931333946937,
        "fixed_best_ret": 0.0077757956593043,
        "oracle_ret": 0.0169063789546489,
        "oracle_expert": "dprime_mix_h01",
        "cumulative_regret_vs_oracle": 3.350571942660231,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-03-18",
        "current_stepf_ret": 0.0014110559336437,
        "fixed_best_ret": 0.0588169597238302,
        "oracle_ret": 0.0588169597238302,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 3.4079778464504176,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-03-21",
        "current_stepf_ret": -0.0012418993437043,
        "fixed_best_ret": 0.0019060756532263,
        "oracle_ret": 0.0041813470199704,
        "oracle_expert": "dprime_mix_3scale",
        "cumulative_regret_vs_oracle": 3.4134010928140923,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-03-22",
        "current_stepf_ret": 0.01252427311497,
        "fixed_best_ret": 0.0037315651078909,
        "oracle_ret": 0.0136791328005492,
        "oracle_expert": "dprime_mix_h02",
        "cumulative_regret_vs_oracle": 3.4145559524996716,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-03-23",
        "current_stepf_ret": 0.044507069597436,
        "fixed_best_ret": -0.024986320365381,
        "oracle_ret": 0.0722984319329261,
        "oracle_expert": "dprime_mix_3scale",
        "cumulative_regret_vs_oracle": 3.4423473148351618,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-03-24",
        "current_stepf_ret": -0.0045724554018507,
        "fixed_best_ret": 0.145848679959774,
        "oracle_ret": 0.1461791118383407,
        "oracle_expert": "dprime_bnf_h02",
        "cumulative_regret_vs_oracle": 3.593098882075353,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-03-25",
        "current_stepf_ret": -0.0062620121231774,
        "fixed_best_ret": -0.0048571428265422,
        "oracle_ret": -0.0001330391105096,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 3.599227855088021,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-03-28",
        "current_stepf_ret": -0.0673110646665909,
        "fixed_best_ret": 0.0146627187281846,
        "oracle_ret": 0.0156627187281847,
        "oracle_expert": "dprime_bnf_h02",
        "cumulative_regret_vs_oracle": 3.6822016384827965,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-03-29",
        "current_stepf_ret": 0.0534610680018973,
        "fixed_best_ret": -0.0691140924096107,
        "oracle_ret": 0.0661974420547485,
        "oracle_expert": "dprime_mix_h01",
        "cumulative_regret_vs_oracle": 3.6949380125356477,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-03-30",
        "current_stepf_ret": -0.0015051517071119,
        "fixed_best_ret": 0.0937763547301292,
        "oracle_ret": 0.0937763547301292,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 3.7902195189728887,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-03-31",
        "current_stepf_ret": -0.0136670570382479,
        "fixed_best_ret": 0.057163543233784,
        "oracle_ret": 0.0687142854332923,
        "oracle_expert": "dprime_all_features_h02",
        "cumulative_regret_vs_oracle": 3.872600861444429,
        "stepf_expert": "dprime_mix_h02"
      },
      {
        "Date": "2022-04-01",
        "current_stepf_ret": -1.1073516250188015e-05,
        "fixed_best_ret": 0.0102747315636725,
        "oracle_ret": 0.0553568361103534,
        "oracle_expert": "dprime_all_features_h02",
        "cumulative_regret_vs_oracle": 3.9279687710710327,
        "stepf_expert": "dprime_mix_h02"
      }
    ],
    "stepf_selected_expert_count_by_name": {
      "dprime_mix_h02": 63
    },
    "stepF_reward_compare": {
      "status": "OK",
      "summary": "reward mode comparison across current/baseline/oracle",
      "rows": [
        {
          "name": "reward_legacy",
          "equity_multiple": 1.286357568457905,
          "sharpe": 1.9802463618621926,
          "max_dd": -0.140464659457601,
          "mean_ret": 0.00468487397818019,
          "std_ret": 0.03755596790983082,
          "regret_vs_fixed_best": 0.4337962147499592,
          "regret_vs_oracle": 55.00993023371655,
          "win_days_vs_fixed_best": 30,
          "pick_match_rate_vs_oracle": 0.12698412698412698,
          "common_days_vs_fixed_best": 63
        },
        {
          "name": "current_stepf",
          "equity_multiple": 1.286357568457905,
          "sharpe": 1.9802463618621926,
          "max_dd": -0.140464659457601,
          "mean_ret": 0.00468487397818019,
          "std_ret": 0.03755596790983082,
          "regret_vs_fixed_best": 0.4337962147499592,
          "regret_vs_oracle": 55.00993023371655,
          "win_days_vs_fixed_best": 30,
          "pick_match_rate_vs_oracle": 0.12698412698412698,
          "common_days_vs_fixed_best": 63,
          "alias_of": "reward_legacy"
        },
        {
          "name": "reward_profit_basic",
          "equity_multiple": 1.3202517485568437,
          "sharpe": 2.1578834268122424,
          "max_dd": -0.13857010944680404,
          "mean_ret": 0.005097104546455447,
          "std_ret": 0.03749694038757129,
          "regret_vs_fixed_best": 0.3999020346510205,
          "regret_vs_oracle": 54.97603605361761,
          "win_days_vs_fixed_best": 30,
          "pick_match_rate_vs_oracle": 0.12698412698412698,
          "common_days_vs_fixed_best": 63
        },
        {
          "name": "reward_profit_regret",
          "equity_multiple": 0.6006584446721613,
          "sharpe": -2.517324934983381,
          "max_dd": -0.46041684937102834,
          "mean_ret": -0.007073158852897414,
          "std_ret": 0.044604061355023275,
          "regret_vs_fixed_best": 1.1194953385357027,
          "regret_vs_oracle": 55.69562935750229,
          "win_days_vs_fixed_best": 25,
          "pick_match_rate_vs_oracle": 0.12698412698412698,
          "common_days_vs_fixed_best": 63
        },
        {
          "name": "reward_profit_light_risk",
          "equity_multiple": 0.9820372517341739,
          "sharpe": 0.1766482078083668,
          "max_dd": -0.18310362211617937,
          "mean_ret": 0.000428037022944083,
          "std_ret": 0.038465587464109656,
          "regret_vs_fixed_best": 0.7381165314736903,
          "regret_vs_oracle": 55.31425055044028,
          "win_days_vs_fixed_best": 28,
          "pick_match_rate_vs_oracle": 0.12698412698412698,
          "common_days_vs_fixed_best": 63
        },
        {
          "name": "fixed_best",
          "equity_multiple": 1.8304948145571007,
          "sharpe": 2.5553282578115257,
          "max_dd": -0.3082823832395679,
          "mean_ret": 0.0108557392566539,
          "std_ret": 0.06743928796560703,
          "regret_vs_fixed_best": 0.0,
          "regret_vs_oracle": 54.465792987617355,
          "win_days_vs_fixed_best": null,
          "pick_match_rate_vs_oracle": null,
          "common_days_vs_fixed_best": null
        },
        {
          "name": "daily_oracle",
          "equity_multiple": 56.296287802174454,
          "sharpe": 23.16540994899783,
          "max_dd": -0.0010686313641663503,
          "mean_ret": 0.06703358463010133,
          "std_ret": 0.045935952304126365,
          "regret_vs_fixed_best": -54.465792987617355,
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
      "StepF_vs_best_StepE: StepF(1.2767) は best StepE(1.8305) に負けてる"
    ]
  }
}
```

Best-effort mode: this evaluator writes SKIP/notes and always exits 0.