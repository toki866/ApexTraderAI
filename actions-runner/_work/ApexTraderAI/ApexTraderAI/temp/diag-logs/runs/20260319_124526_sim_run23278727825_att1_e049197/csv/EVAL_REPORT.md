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
| stepB_pred_close_mamba_SOXL.csv | Pred_Close_MAMBA | 1.0000 | 1.0000 | 4.7624 | 0.8869 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Pred_Close_MAMBA_PERIODIC_h01 | 1.0000 | 1.0000 | 3.3360 | 0.9251 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Delta_Close_pred_MAMBA_PERIODIC_h01 | 1.0000 | 1.0000 | 44.5305 | 0.0425 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Pred_Close_MAMBA_PERIODIC_h05 | 1.0000 | 1.0000 | 5.6725 | 0.8289 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Delta_Close_pred_MAMBA_PERIODIC_h05 | 1.0000 | 1.0000 | 42.9560 | 0.0940 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Pred_Close_MAMBA_PERIODIC_h10 | 1.0000 | 1.0000 | 7.7209 | 0.6831 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Delta_Close_pred_MAMBA_PERIODIC_h10 | 1.0000 | 1.0000 | 43.1753 | -0.2070 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Pred_Close_MAMBA_PERIODIC_h20 | 1.0000 | 1.0000 | 12.8284 | 0.5961 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Delta_Close_pred_MAMBA_PERIODIC_h20 | 1.0000 | 1.0000 | 40.7455 | 0.1083 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Pred_Close_MAMBA_PERIODIC | 1.0000 | 1.0000 | 3.3360 | 0.9251 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Delta_Close_pred_MAMBA_PERIODIC | 1.0000 | 1.0000 | 44.5305 | 0.0425 | OK |
| stepB_pred_time_all_SOXL.csv | Pred_Close_MAMBA | 1.0000 | 1.0000 | 4.7624 | 0.8869 | OK |

## StepE table
| agent | file | test_days | equity_multiple | max_dd | mean_ret | std_ret | sharpe | note | status |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| dprime_all_features_3scale | stepE_daily_log_dprime_all_features_3scale_SOXL.csv | 63 | 0.9786 | -0.4030 | 0.0034 | 0.0764 | 0.7105 |  | OK |
| dprime_all_features_h01 | stepE_daily_log_dprime_all_features_h01_SOXL.csv | 63 | 1.0080 | -0.3607 | 0.0016 | 0.0695 | 0.3587 |  | OK |
| dprime_all_features_h02 | stepE_daily_log_dprime_all_features_h02_SOXL.csv | 63 | 0.6355 | -0.3756 | -0.0056 | 0.0713 | -1.2452 |  | OK |
| dprime_all_features_h03 | stepE_daily_log_dprime_all_features_h03_SOXL.csv | 63 | 1.3253 | -0.3435 | 0.0084 | 0.0788 | 1.6931 |  | OK |
| dprime_bnf_3scale | stepE_daily_log_dprime_bnf_3scale_SOXL.csv | 63 | 0.8068 | -0.5156 | 0.0006 | 0.0798 | 0.1275 |  | OK |
| dprime_bnf_h01 | stepE_daily_log_dprime_bnf_h01_SOXL.csv | 63 | 0.5879 | -0.6082 | -0.0055 | 0.0636 | -1.3714 |  | OK |
| dprime_bnf_h02 | stepE_daily_log_dprime_bnf_h02_SOXL.csv | 63 | 0.5789 | -0.5521 | -0.0054 | 0.0692 | -1.2294 |  | OK |
| dprime_mix_3scale | stepE_daily_log_dprime_mix_3scale_SOXL.csv | 63 | 1.3251 | -0.3377 | 0.0081 | 0.0739 | 1.7332 |  | OK |
| dprime_mix_h01 | stepE_daily_log_dprime_mix_h01_SOXL.csv | 63 | 0.6824 | -0.5018 | -0.0028 | 0.0724 | -0.6211 |  | OK |
| dprime_mix_h02 | stepE_daily_log_dprime_mix_h02_SOXL.csv | 63 | 0.5405 | -0.5653 | -0.0063 | 0.0718 | -1.3909 |  | OK |

## StepF table
| file | test_days | split_source | train_rows | equity_multiple | max_dd | mean_ret | std_ret | sharpe | note | status |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| stepF_equity_marl_SOXL.csv | 63 | csv | 0 | 0.8976 | -0.2652 | -0.0008 | 0.0447 | -0.2744 |  | OK |

## StepFCompare
- status: **OK**
- summary: current StepF vs fixed-best expert vs daily oracle
- current_stepf_equity_multiple: 0.8976
- fixed_best_expert: dprime_all_features_h03
- fixed_best_equity_multiple: 1.3253
- oracle_equity_multiple: 45.5567
- regret_vs_fixed_best: 0.4276
- regret_vs_oracle: 44.6591
- stepf_win_days_vs_fixed_best: 28
- stepf_pick_match_rate_vs_oracle: 0.0476
- StepFRewardCompare status: OK (reward mode comparison across current/baseline/oracle)

### StepFRewardCompare
| name | equity_multiple | sharpe | max_dd | mean_ret | std_ret | regret_vs_fixed_best | regret_vs_oracle | win_days_vs_fixed_best | pick_match_rate_vs_oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reward_legacy | 0.8950 | -0.2744 | -0.2652 | -0.0008 | 0.0447 | 0.5064 | 44.6617 | 28 | 0.0476 |
| current_stepf | 0.8950 | -0.2744 | -0.2652 | -0.0008 | 0.0447 | 0.5064 | 44.6617 | 28 | 0.0476 |
| reward_profit_basic | 0.8938 | -0.1861 | -0.2759 | -0.0006 | 0.0493 | 0.5077 | 44.6630 | 30 | 0.0952 |
| fixed_best | 1.3253 | 1.6931 | -0.3435 | 0.0084 | 0.0788 | 0.0000 | 44.2315 | NA | NA |
| daily_oracle | 45.5567 | 21.1041 | -0.0506 | 0.0635 | 0.0478 | -44.2315 | 0.0000 | NA | NA |
- cluster_status: PENDING (cluster comparison pending)

## Diversity
- status: **OK**
- summary: agent positions look diverse
- max_corr: 0.7590
- max_match_ratio: 0.5397
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
  - note: StepF_vs_best_StepE: StepF(0.8976) は best StepE(1.3253) に負けてる

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
        "mae": 4.762436942448692,
        "corr": 0.8869496233618677,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Pred_Close_MAMBA_PERIODIC_h01",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 3.3360110420537223,
        "corr": 0.9251272331409109,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Delta_Close_pred_MAMBA_PERIODIC_h01",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 44.530541174848764,
        "corr": 0.0425101983129751,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Pred_Close_MAMBA_PERIODIC_h05",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 5.672530271738657,
        "corr": 0.8288763815683169,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Delta_Close_pred_MAMBA_PERIODIC_h05",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 42.955986260292796,
        "corr": 0.09401451270377191,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Pred_Close_MAMBA_PERIODIC_h10",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 7.720909853693362,
        "corr": 0.6831219029222829,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Delta_Close_pred_MAMBA_PERIODIC_h10",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 43.17525472301538,
        "corr": -0.2069576107547574,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Pred_Close_MAMBA_PERIODIC_h20",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 12.828428792988671,
        "corr": 0.5960616304442795,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Delta_Close_pred_MAMBA_PERIODIC_h20",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 40.74545550595762,
        "corr": 0.10832039492831294,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Pred_Close_MAMBA_PERIODIC",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 3.3360110420537223,
        "corr": 0.9251272331409109,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Delta_Close_pred_MAMBA_PERIODIC",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 44.530541174848764,
        "corr": 0.0425101983129751,
        "status": "OK"
      },
      {
        "file": "stepB_pred_time_all_SOXL.csv",
        "pred_col": "Pred_Close_MAMBA",
        "non_null_ratio": 1.0,
        "first_valid_date": "2022-01-03",
        "coverage_ratio_over_test": 1.0,
        "mae": 4.762436942448692,
        "corr": 0.8869496233618677,
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
        "equity_multiple": 0.9786239917241871,
        "max_dd": -0.40296008627297464,
        "mean_ret": 0.003419481090224736,
        "std_ret": 0.07639619288335503,
        "sharpe": 0.7105403740290696,
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
        "equity_multiple": 1.008028329390947,
        "max_dd": -0.3606711716993525,
        "mean_ret": 0.0015710896629560227,
        "std_ret": 0.06952221159530889,
        "sharpe": 0.35873823115084014,
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
        "equity_multiple": 0.6355033419619591,
        "max_dd": -0.37564801510678425,
        "mean_ret": -0.005595532932261191,
        "std_ret": 0.07133416833855531,
        "sharpe": -1.2452143708781591,
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
        "equity_multiple": 1.3252609192041407,
        "max_dd": -0.34345143738749584,
        "mean_ret": 0.008403393700044887,
        "std_ret": 0.07878905558201262,
        "sharpe": 1.6931252495704363,
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
        "equity_multiple": 0.8068093431959112,
        "max_dd": -0.5155887368050063,
        "mean_ret": 0.0006410354696969686,
        "std_ret": 0.07983432761596279,
        "sharpe": 0.1274655014981708,
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
        "equity_multiple": 0.5878562592937672,
        "max_dd": -0.608196048523329,
        "mean_ret": -0.0054976877344580875,
        "std_ret": 0.0636385224684288,
        "sharpe": -1.3713877035861979,
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
        "equity_multiple": 0.5789286389979496,
        "max_dd": -0.5521017632764691,
        "mean_ret": -0.005358172654116623,
        "std_ret": 0.06918728288710002,
        "sharpe": -1.2293928941541958,
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
        "equity_multiple": 1.3250745545342093,
        "max_dd": -0.33770752424204975,
        "mean_ret": 0.0080701919383334,
        "std_ret": 0.07391544090528084,
        "sharpe": 1.7332011260339808,
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
        "equity_multiple": 0.6823864410969654,
        "max_dd": -0.5017937165664266,
        "mean_ret": -0.002831885380943253,
        "std_ret": 0.07238006580045303,
        "sharpe": -0.6210934773177588,
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
        "equity_multiple": 0.5405434388057757,
        "max_dd": -0.5653473326082588,
        "mean_ret": -0.006287831199678783,
        "std_ret": 0.07176295320737539,
        "sharpe": -1.3909158051700599,
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
        "equity_multiple": 0.8976385955536075,
        "max_dd": -0.2651854979593703,
        "mean_ret": -0.0007732813976270487,
        "std_ret": 0.044727540201677816,
        "sharpe": -0.27444973665467215,
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
    "max_corr": 0.7589866353359872,
    "max_match_ratio": 0.5396825396825397,
    "pairs_over_0_9999": 0,
    "all_pairs": 45,
    "identical_all_agents": false
  },
  "overall_status": "OK",
  "stepF_compare": {
    "status": "OK",
    "summary": "current StepF vs fixed-best expert vs daily oracle",
    "row": {
      "current_stepf_equity_multiple": 0.8976385955536075,
      "fixed_best_expert": "dprime_all_features_h03",
      "fixed_best_equity_multiple": 1.3252609192041407,
      "oracle_equity_multiple": 45.55674579873577,
      "regret_vs_fixed_best": 0.4276223236505332,
      "regret_vs_oracle": 44.65910720318216,
      "current_stepf_sharpe": -0.27444973665467215,
      "fixed_best_sharpe": 1.6931252495704363,
      "oracle_sharpe": 21.104072277911513,
      "current_stepf_max_dd": -0.2651854979593703,
      "fixed_best_max_dd": -0.34345143738749584,
      "oracle_max_dd": -0.0505760478079319,
      "stepf_win_days_vs_fixed_best": 28,
      "stepf_common_days_vs_fixed_best": 63,
      "stepf_pick_match_rate_vs_oracle": 0.047619047619047616,
      "oracle_unique_expert_count": 10
    },
    "best_expert_name": "dprime_all_features_h03",
    "best_expert_equity_multiple": 1.3252609192041407,
    "best_expert_sharpe": 1.6931252495704363,
    "best_expert_max_dd": -0.34345143738749584,
    "oracle_equity_multiple": 45.55674579873577,
    "oracle_sharpe": 21.104072277911513,
    "oracle_max_dd": -0.0505760478079319,
    "oracle_selected_expert_count_by_name": {
      "dprime_all_features_h03": 15,
      "dprime_bnf_h02": 9,
      "dprime_bnf_h01": 9,
      "dprime_mix_3scale": 7,
      "dprime_mix_h01": 6,
      "dprime_mix_h02": 6,
      "dprime_all_features_h02": 4,
      "dprime_all_features_3scale": 3,
      "dprime_bnf_3scale": 2,
      "dprime_all_features_h01": 2
    },
    "oracle_unique_expert_count": 10,
    "cumulative_regret_vs_oracle": 4.052055630987842,
    "cluster": {
      "status": "PENDING",
      "reason": "cluster comparison pending",
      "rows": []
    },
    "series": [
      {
        "Date": "2022-01-03",
        "current_stepf_ret": -0.0029250949468578,
        "fixed_best_ret": 0.0574733502890915,
        "oracle_ret": 0.0591381585001945,
        "oracle_expert": "dprime_bnf_h02",
        "cumulative_regret_vs_oracle": 0.0620632534470523,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-01-04",
        "current_stepf_ret": 0.0258290384914343,
        "fixed_best_ret": 0.0109223604798316,
        "oracle_ret": 0.0109223604798316,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 0.0471565754354496,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-01-05",
        "current_stepf_ret": 0.0027624378298036,
        "fixed_best_ret": 0.0971595069169998,
        "oracle_ret": 0.0971595069169998,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 0.1415536445226458,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-01-06",
        "current_stepf_ret": 0.0099207070541932,
        "fixed_best_ret": -0.02265405831206,
        "oracle_ret": 0.0229024944603443,
        "oracle_expert": "dprime_all_features_h02",
        "cumulative_regret_vs_oracle": 0.1545354319287969,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-01-07",
        "current_stepf_ret": 0.0015386139572439,
        "fixed_best_ret": 0.0877594724893569,
        "oracle_ret": 0.0878252183794975,
        "oracle_expert": "dprime_bnf_3scale",
        "cumulative_regret_vs_oracle": 0.24082203635105048,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-01-10",
        "current_stepf_ret": 0.011493019025122,
        "fixed_best_ret": -0.0031431862278938,
        "oracle_ret": 0.0033521644547581,
        "oracle_expert": "dprime_mix_h01",
        "cumulative_regret_vs_oracle": 0.23268118178068659,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-01-11",
        "current_stepf_ret": 0.0019219955186136,
        "fixed_best_ret": 0.0237912276405502,
        "oracle_ret": 0.053833348840475,
        "oracle_expert": "dprime_mix_h01",
        "cumulative_regret_vs_oracle": 0.284592535102548,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-01-12",
        "current_stepf_ret": -0.0003862663679806,
        "fixed_best_ret": -0.0064644531367089,
        "oracle_ret": 0.0193823819756507,
        "oracle_expert": "dprime_bnf_h02",
        "cumulative_regret_vs_oracle": 0.3043611834461793,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-01-13",
        "current_stepf_ret": -0.0118380206639593,
        "fixed_best_ret": 0.0672242989838123,
        "oracle_ret": 0.067571425974369,
        "oracle_expert": "dprime_all_features_3scale",
        "cumulative_regret_vs_oracle": 0.3837706300845076,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-01-14",
        "current_stepf_ret": 0.0208753745041755,
        "fixed_best_ret": -0.0705187151432037,
        "oracle_ret": 0.0700963904857635,
        "oracle_expert": "dprime_all_features_h02",
        "cumulative_regret_vs_oracle": 0.43299164606609564,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-01-18",
        "current_stepf_ret": -0.0470743516856729,
        "fixed_best_ret": 0.1283103396892547,
        "oracle_ret": 0.1283103396892547,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 0.6083763374410233,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-01-19",
        "current_stepf_ret": -0.0850399832563726,
        "fixed_best_ret": 0.0810610551655028,
        "oracle_ret": 0.0906030555963516,
        "oracle_expert": "dprime_mix_3scale",
        "cumulative_regret_vs_oracle": 0.7840193762937475,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-01-20",
        "current_stepf_ret": -0.0497430980107497,
        "fixed_best_ret": 0.0038603507546904,
        "oracle_ret": 0.0498729793318851,
        "oracle_expert": "dprime_bnf_3scale",
        "cumulative_regret_vs_oracle": 0.8836354536363823,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-01-21",
        "current_stepf_ret": 0.0243886671272618,
        "fixed_best_ret": -0.0510979445986449,
        "oracle_ret": -0.0505760478079319,
        "oracle_expert": "dprime_mix_3scale",
        "cumulative_regret_vs_oracle": 0.8086707387011886,
        "stepf_expert": "dprime_bnf_h02"
      },
      {
        "Date": "2022-01-24",
        "current_stepf_ret": 0.0121757605276494,
        "fixed_best_ret": 0.0388078635334968,
        "oracle_ret": 0.0388078635334968,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 0.835302841707036,
        "stepf_expert": "dprime_bnf_h02"
      },
      {
        "Date": "2022-01-25",
        "current_stepf_ret": 0.0308286271251735,
        "fixed_best_ret": -0.0914840416584409,
        "oracle_ret": 0.105016319736792,
        "oracle_expert": "dprime_mix_h02",
        "cumulative_regret_vs_oracle": 0.9094905343186545,
        "stepf_expert": "dprime_bnf_h02"
      },
      {
        "Date": "2022-01-26",
        "current_stepf_ret": -0.1007225026340425,
        "fixed_best_ret": 0.0469140241146087,
        "oracle_ret": 0.0470219051241874,
        "oracle_expert": "dprime_all_features_3scale",
        "cumulative_regret_vs_oracle": 1.0572349420768843,
        "stepf_expert": "dprime_bnf_h02"
      },
      {
        "Date": "2022-01-27",
        "current_stepf_ret": 0.0482183618816382,
        "fixed_best_ret": -0.1377520987987518,
        "oracle_ret": 0.0252821314380601,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 1.0342987116333062,
        "stepf_expert": "dprime_bnf_h02"
      },
      {
        "Date": "2022-01-28",
        "current_stepf_ret": 0.0020037060810392,
        "fixed_best_ret": 0.0414507583148123,
        "oracle_ret": 0.0553806146383285,
        "oracle_expert": "dprime_mix_3scale",
        "cumulative_regret_vs_oracle": 1.0876756201905955,
        "stepf_expert": "dprime_bnf_h02"
      },
      {
        "Date": "2022-01-31",
        "current_stepf_ret": -0.0181698710025834,
        "fixed_best_ret": 0.1595089784860611,
        "oracle_ret": 0.159635307431221,
        "oracle_expert": "dprime_mix_3scale",
        "cumulative_regret_vs_oracle": 1.2654807986243999,
        "stepf_expert": "dprime_bnf_h02"
      },
      {
        "Date": "2022-02-01",
        "current_stepf_ret": -0.0141681246865062,
        "fixed_best_ret": 0.0114219599181795,
        "oracle_ret": 0.0288339612782001,
        "oracle_expert": "dprime_mix_h01",
        "cumulative_regret_vs_oracle": 1.308482884589106,
        "stepf_expert": "dprime_bnf_h02"
      },
      {
        "Date": "2022-02-02",
        "current_stepf_ret": -0.001557047721949,
        "fixed_best_ret": -0.0749256673455238,
        "oracle_ret": 0.073250125542283,
        "oracle_expert": "dprime_all_features_h02",
        "cumulative_regret_vs_oracle": 1.383290057853338,
        "stepf_expert": "dprime_bnf_h02"
      },
      {
        "Date": "2022-02-03",
        "current_stepf_ret": 0.0029382909447303,
        "fixed_best_ret": 0.0744205355424984,
        "oracle_ret": 0.1323333402872085,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 1.5126851071958163,
        "stepf_expert": "dprime_bnf_h02"
      },
      {
        "Date": "2022-02-04",
        "current_stepf_ret": -0.0002410653641504,
        "fixed_best_ret": -0.0046535707995817,
        "oracle_ret": 0.0184698106497526,
        "oracle_expert": "dprime_all_features_h01",
        "cumulative_regret_vs_oracle": 1.5313959832097193,
        "stepf_expert": "dprime_bnf_h02"
      },
      {
        "Date": "2022-02-07",
        "current_stepf_ret": 0.0479831197150722,
        "fixed_best_ret": -0.0027635433720424,
        "oracle_ret": -0.0004376116648379,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 1.4829752518298092,
        "stepf_expert": "dprime_bnf_h02"
      },
      {
        "Date": "2022-02-08",
        "current_stepf_ret": 0.0025098507794889,
        "fixed_best_ret": 0.0481998262801347,
        "oracle_ret": 0.0724853371977806,
        "oracle_expert": "dprime_mix_3scale",
        "cumulative_regret_vs_oracle": 1.552950738248101,
        "stepf_expert": "dprime_bnf_h02"
      },
      {
        "Date": "2022-02-09",
        "current_stepf_ret": -0.0144714304967338,
        "fixed_best_ret": -0.082589316095948,
        "oracle_ret": 0.0938498323559761,
        "oracle_expert": "dprime_mix_h01",
        "cumulative_regret_vs_oracle": 1.6612720011008109,
        "stepf_expert": "dprime_bnf_h02"
      },
      {
        "Date": "2022-02-10",
        "current_stepf_ret": 0.0001447995239415,
        "fixed_best_ret": 0.0914985497593879,
        "oracle_ret": 0.0915925895571708,
        "oracle_expert": "dprime_mix_h02",
        "cumulative_regret_vs_oracle": 1.7527197911340402,
        "stepf_expert": "dprime_bnf_h02"
      },
      {
        "Date": "2022-02-11",
        "current_stepf_ret": -0.0001283084073605,
        "fixed_best_ret": -0.1502119262218475,
        "oracle_ret": 0.1433428524434566,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 1.8961909519848572,
        "stepf_expert": "dprime_bnf_h02"
      },
      {
        "Date": "2022-02-14",
        "current_stepf_ret": 0.1189027448739113,
        "fixed_best_ret": 0.0002645228998735,
        "oracle_ret": 0.0002645228998735,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 1.7775527300108194,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-02-15",
        "current_stepf_ret": -0.0009748398904429,
        "fixed_best_ret": 0.161162110209465,
        "oracle_ret": 0.161162110209465,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 1.9396896801107273,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-02-16",
        "current_stepf_ret": -0.0263603161215717,
        "fixed_best_ret": 0.0003448607761106,
        "oracle_ret": 0.0009560998771339,
        "oracle_expert": "dprime_all_features_3scale",
        "cumulative_regret_vs_oracle": 1.9670060961094329,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-02-17",
        "current_stepf_ret": -0.0190851694855485,
        "fixed_best_ret": 0.110462562084198,
        "oracle_ret": 0.110462562084198,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 2.0965538276791795,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-02-18",
        "current_stepf_ret": -0.0184598256231812,
        "fixed_best_ret": -0.0323400512784719,
        "oracle_ret": 0.0253972600698471,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 2.140410913372208,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-02-22",
        "current_stepf_ret": -0.0633416522064851,
        "fixed_best_ret": -0.0267380884140729,
        "oracle_ret": 0.0118307571914282,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 2.215583322770121,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-02-23",
        "current_stepf_ret": 0.0828207027007732,
        "fixed_best_ret": -0.0709300915002822,
        "oracle_ret": 0.0376021604041383,
        "oracle_expert": "dprime_bnf_h02",
        "cumulative_regret_vs_oracle": 2.170364780473486,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-02-24",
        "current_stepf_ret": -0.0180069550962276,
        "fixed_best_ret": 0.1112249737381935,
        "oracle_ret": 0.1112249737381935,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 2.2995967093079073,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-02-25",
        "current_stepf_ret": -0.0137885843658684,
        "fixed_best_ret": 0.0227223139600756,
        "oracle_ret": 0.0383119849726241,
        "oracle_expert": "dprime_mix_h02",
        "cumulative_regret_vs_oracle": 2.3516972786464,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-02-28",
        "current_stepf_ret": -0.0112169055165141,
        "fixed_best_ret": -0.0196778224706649,
        "oracle_ret": 0.0180930794924497,
        "oracle_expert": "dprime_all_features_h02",
        "cumulative_regret_vs_oracle": 2.3810072636553636,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-01",
        "current_stepf_ret": 0.0768209555578373,
        "fixed_best_ret": -0.109022354543209,
        "oracle_ret": 0.0746713956807579,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 2.378857703778284,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-02",
        "current_stepf_ret": -0.022133088641764,
        "fixed_best_ret": 0.0957741087079048,
        "oracle_ret": 0.0957741087079048,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 2.496764901127953,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-03",
        "current_stepf_ret": -0.0535412003913224,
        "fixed_best_ret": -0.065057819545269,
        "oracle_ret": 0.0622231716811657,
        "oracle_expert": "dprime_mix_h02",
        "cumulative_regret_vs_oracle": 2.612529273200441,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-04",
        "current_stepf_ret": -0.1367160189806898,
        "fixed_best_ret": -0.0750345865488052,
        "oracle_ret": 0.0441084213662632,
        "oracle_expert": "dprime_bnf_h02",
        "cumulative_regret_vs_oracle": 2.793353713547394,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-07",
        "current_stepf_ret": 0.0413281272562384,
        "fixed_best_ret": -0.1502666155099868,
        "oracle_ret": 0.1513829671144485,
        "oracle_expert": "dprime_bnf_h02",
        "cumulative_regret_vs_oracle": 2.9034085534056038,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-08",
        "current_stepf_ret": 0.0692821591603483,
        "fixed_best_ret": 0.0524144677221775,
        "oracle_ret": 0.0524144677221775,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 2.886540861967433,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-09",
        "current_stepf_ret": 0.0122218067398914,
        "fixed_best_ret": 0.1199884583950042,
        "oracle_ret": 0.1199884583950042,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 2.994307513622546,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-10",
        "current_stepf_ret": -0.0343003385149351,
        "fixed_best_ret": -0.063410544604063,
        "oracle_ret": 0.0594751631617546,
        "oracle_expert": "dprime_bnf_h02",
        "cumulative_regret_vs_oracle": 3.088083015299236,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-11",
        "current_stepf_ret": -0.0495198479886177,
        "fixed_best_ret": -0.0639007443785667,
        "oracle_ret": 0.0620847799479961,
        "oracle_expert": "dprime_all_features_h01",
        "cumulative_regret_vs_oracle": 3.1996876432358494,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-14",
        "current_stepf_ret": 0.1148109131925798,
        "fixed_best_ret": -0.0915832722783088,
        "oracle_ret": 0.0901459375321865,
        "oracle_expert": "dprime_bnf_h02",
        "cumulative_regret_vs_oracle": 3.175022667575456,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-15",
        "current_stepf_ret": -0.0038060542768984,
        "fixed_best_ret": 0.127986045718193,
        "oracle_ret": 0.127986045718193,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 3.3068147675705473,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-16",
        "current_stepf_ret": -0.0187766674984338,
        "fixed_best_ret": 0.1510151942968368,
        "oracle_ret": 0.1510151942968368,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 3.476606629365818,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-17",
        "current_stepf_ret": -0.0278968225494892,
        "fixed_best_ret": -0.0194244384126603,
        "oracle_ret": 0.0082593288705759,
        "oracle_expert": "dprime_mix_h02",
        "cumulative_regret_vs_oracle": 3.512762780785883,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-18",
        "current_stepf_ret": 0.0042602793340159,
        "fixed_best_ret": -0.0642276267409324,
        "oracle_ret": 0.0585778201837092,
        "oracle_expert": "dprime_bnf_h02",
        "cumulative_regret_vs_oracle": 3.567080321635576,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-21",
        "current_stepf_ret": -0.0046987580900007,
        "fixed_best_ret": 0.0041813470199704,
        "oracle_ret": 0.0041813470199704,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 3.575960426745547,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-22",
        "current_stepf_ret": -0.014737673882238,
        "fixed_best_ret": 0.002584391326467,
        "oracle_ret": 0.0137656486853957,
        "oracle_expert": "dprime_bnf_h02",
        "cumulative_regret_vs_oracle": 3.604463749313181,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-23",
        "current_stepf_ret": 0.0829798658115357,
        "fixed_best_ret": -0.0746438687592744,
        "oracle_ret": 0.0722984319329261,
        "oracle_expert": "dprime_mix_3scale",
        "cumulative_regret_vs_oracle": 3.5937823154345714,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-24",
        "current_stepf_ret": -0.0010304807433722,
        "fixed_best_ret": 0.1461791118383407,
        "oracle_ret": 0.1461791118383407,
        "oracle_expert": "dprime_all_features_h03",
        "cumulative_regret_vs_oracle": 3.740991908016284,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-25",
        "current_stepf_ret": -0.0017642762947258,
        "fixed_best_ret": -0.0015374340669712,
        "oracle_ret": -0.0005727936848267,
        "oracle_expert": "dprime_mix_3scale",
        "cumulative_regret_vs_oracle": 3.742183390626183,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-28",
        "current_stepf_ret": 0.0420525589499739,
        "fixed_best_ret": 0.0064984760728568,
        "oracle_ret": 0.0156627187281847,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 3.715793550404394,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-29",
        "current_stepf_ret": 0.033557936573912,
        "fixed_best_ret": 0.0365250416672155,
        "oracle_ret": 0.0661974420547485,
        "oracle_expert": "dprime_bnf_h01",
        "cumulative_regret_vs_oracle": 3.7484330558852306,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-30",
        "current_stepf_ret": -0.0369754503542188,
        "fixed_best_ret": 0.0251137895096854,
        "oracle_ret": 0.0937763547301292,
        "oracle_expert": "dprime_mix_h01",
        "cumulative_regret_vs_oracle": 3.8791848609695787,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-03-31",
        "current_stepf_ret": -0.0490326346800164,
        "fixed_best_ret": -0.0232335178708262,
        "oracle_ret": 0.0687142854332923,
        "oracle_expert": "dprime_mix_h01",
        "cumulative_regret_vs_oracle": 3.9969317810828873,
        "stepf_expert": "dprime_mix_3scale"
      },
      {
        "Date": "2022-04-01",
        "current_stepf_ret": -0.0006584218506218,
        "fixed_best_ret": -0.0655250325649976,
        "oracle_ret": 0.0544654280543327,
        "oracle_expert": "dprime_mix_h02",
        "cumulative_regret_vs_oracle": 4.052055630987842,
        "stepf_expert": "dprime_mix_3scale"
      }
    ],
    "stepf_selected_expert_count_by_name": {
      "dprime_mix_3scale": 47,
      "dprime_bnf_h02": 16
    },
    "stepF_reward_compare": {
      "status": "OK",
      "summary": "reward mode comparison across current/baseline/oracle",
      "rows": [
        {
          "name": "reward_legacy",
          "equity_multiple": 0.8950129174336487,
          "sharpe": -0.27444973665467215,
          "max_dd": -0.2651854979593701,
          "mean_ret": -0.0007732813976270487,
          "std_ret": 0.044727540201677816,
          "regret_vs_fixed_best": 0.5064151868043503,
          "regret_vs_oracle": 44.66173288130212,
          "win_days_vs_fixed_best": 28,
          "pick_match_rate_vs_oracle": 0.047619047619047616,
          "common_days_vs_fixed_best": 63
        },
        {
          "name": "current_stepf",
          "equity_multiple": 0.8950129174336487,
          "sharpe": -0.27444973665467215,
          "max_dd": -0.2651854979593701,
          "mean_ret": -0.0007732813976270487,
          "std_ret": 0.044727540201677816,
          "regret_vs_fixed_best": 0.5064151868043503,
          "regret_vs_oracle": 44.66173288130212,
          "win_days_vs_fixed_best": 28,
          "pick_match_rate_vs_oracle": 0.047619047619047616,
          "common_days_vs_fixed_best": 63,
          "alias_of": "reward_legacy"
        },
        {
          "name": "reward_profit_basic",
          "equity_multiple": 0.8937598056708242,
          "sharpe": -0.18611321416088575,
          "max_dd": -0.2758771755841095,
          "mean_ret": -0.0005774614785006854,
          "std_ret": 0.04925451867738183,
          "regret_vs_fixed_best": 0.5076682985671748,
          "regret_vs_oracle": 44.66298599306494,
          "win_days_vs_fixed_best": 30,
          "pick_match_rate_vs_oracle": 0.09523809523809523,
          "common_days_vs_fixed_best": 63
        },
        {
          "name": "fixed_best",
          "equity_multiple": 1.3252609192041407,
          "sharpe": 1.6931252495704363,
          "max_dd": -0.34345143738749584,
          "mean_ret": 0.008403393700044887,
          "std_ret": 0.07878905558201262,
          "regret_vs_fixed_best": 0.0,
          "regret_vs_oracle": 44.23148487953163,
          "win_days_vs_fixed_best": null,
          "pick_match_rate_vs_oracle": null,
          "common_days_vs_fixed_best": null
        },
        {
          "name": "daily_oracle",
          "equity_multiple": 45.55674579873577,
          "sharpe": 21.104072277911513,
          "max_dd": -0.0505760478079319,
          "mean_ret": 0.06354506195138634,
          "std_ret": 0.04779866996916831,
          "regret_vs_fixed_best": -44.23148487953163,
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
      "StepF_vs_best_StepE: StepF(0.8976) は best StepE(1.3253) に負けてる"
    ]
  }
}
```

Best-effort mode: this evaluator writes SKIP/notes and always exits 0.