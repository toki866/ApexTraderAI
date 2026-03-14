# EVAL_REPORT

- output_root: `C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03`
- mode: `sim`
- symbol: `SOXL`
- overall_status: **WARN**

## DPrime diagnostics
- status: **OK**
- summary: DPrimeCluster=OK (embeddings found); DPrimeRL=OK (RL state files found under stepDprime/sim); failure_reason=LinAlgError('SVD did not converge')

### DPrimeCluster
- status: **OK**
- summary: embeddings found
- cluster_embeddings_count: 48
- cluster_state_count: 0
- cluster_input_count: 0

### DPrimeRL
- status: **OK**
- summary: RL state files found under stepDprime/sim
- rl_state_count: 24
- rl_profiles_count: 6

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
- SKIP: stepE_daily_log missing

## StepF table
- SKIP: stepF_equity_marl missing

## StepFCompare
- status: **WARN**
- summary: compare skipped
- StepFRewardCompare status: WARN (NA)
- cluster_status: PENDING (cluster comparison pending)

## Diversity
- status: **SKIP**
- summary: not evaluated
- max_corr: NA
- max_match_ratio: NA
- pairs_over_0_9999: NA / NA
- identical_all_agents: NA

## PLOTS
- [equity_stepE_topN.png](./equity_stepE_topN.png)
- [equity_stepF.png](./equity_stepF.png)
- [bar_stepE_return.png](./bar_stepE_return.png)
- [scatter_stepE_dd_vs_ret.png](./scatter_stepE_dd_vs_ret.png)
- [equity_stepF_vs_fixed_best_vs_oracle.png](./equity_stepF_vs_fixed_best_vs_oracle.png)
- [bar_stepF_regret.png](./bar_stepF_regret.png)
- [equity_stepF_reward_modes.png](./equity_stepF_reward_modes.png)
- [bar_stepF_reward_mode_regret.png](./bar_stepF_reward_mode_regret.png)
  - note: PLOT StepE topN: stepE_daily_log files were not found
  - note: PLOT StepF equity: stepF_equity_marl file not found
  - note: PLOT StepE return bar: no numeric StepE rows
  - note: PLOT StepE scatter: no numeric StepE DD/return pairs
  - note: StepF_vs_best_StepE: 比較に必要な equity_multiple が不足
  - note: PLOT StepF compare equity: comparison series unavailable
  - note: PLOT StepF regret: regret values unavailable

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
    "summary": "DPrimeCluster=OK (embeddings found); DPrimeRL=OK (RL state files found under stepDprime/sim); failure_reason=LinAlgError('SVD did not converge')",
    "details": {
      "state_count": 24,
      "embeddings_count": 48,
      "cluster_status": "OK",
      "cluster_summary": "embeddings found",
      "cluster_embeddings_count": 48,
      "cluster_state_count": 0,
      "cluster_input_count": 0,
      "cluster_embeddings_files": [
        "stepDprime_bnf_3scale_SOXL_embeddings_all.csv",
        "stepDprime_bnf_h01_SOXL_embeddings_all.csv",
        "stepDprime_bnf_h02_SOXL_embeddings_all.csv",
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
        "stepDprime_bnf_3scale_SOXL_embeddings_all.csv",
        "stepDprime_bnf_h01_SOXL_embeddings_all.csv",
        "stepDprime_bnf_h02_SOXL_embeddings_all.csv",
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
      "rl_state_count": 24,
      "rl_profiles_count": 6,
      "rl_state_files": [
        "stepDprime_state_test_dprime_bnf_3scale_SOXL.csv",
        "stepDprime_state_test_dprime_bnf_h01_SOXL.csv",
        "stepDprime_state_test_dprime_bnf_h02_SOXL.csv",
        "stepDprime_state_test_dprime_mix_3scale_SOXL.csv",
        "stepDprime_state_test_dprime_mix_h01_SOXL.csv",
        "stepDprime_state_test_dprime_mix_h02_SOXL.csv",
        "stepDprime_state_train_dprime_bnf_3scale_SOXL.csv",
        "stepDprime_state_train_dprime_bnf_h01_SOXL.csv",
        "stepDprime_state_train_dprime_bnf_h02_SOXL.csv",
        "stepDprime_state_train_dprime_mix_3scale_SOXL.csv",
        "stepDprime_state_train_dprime_mix_h01_SOXL.csv",
        "stepDprime_state_train_dprime_mix_h02_SOXL.csv",
        "stepDprime_state_test_dprime_bnf_3scale_SOXL.csv",
        "stepDprime_state_test_dprime_bnf_h01_SOXL.csv",
        "stepDprime_state_test_dprime_bnf_h02_SOXL.csv",
        "stepDprime_state_test_dprime_mix_3scale_SOXL.csv",
        "stepDprime_state_test_dprime_mix_h01_SOXL.csv",
        "stepDprime_state_test_dprime_mix_h02_SOXL.csv",
        "stepDprime_state_train_dprime_bnf_3scale_SOXL.csv",
        "stepDprime_state_train_dprime_bnf_h01_SOXL.csv",
        "stepDprime_state_train_dprime_bnf_h02_SOXL.csv",
        "stepDprime_state_train_dprime_mix_3scale_SOXL.csv",
        "stepDprime_state_train_dprime_mix_h01_SOXL.csv",
        "stepDprime_state_train_dprime_mix_h02_SOXL.csv"
      ],
      "state_files": [
        "stepDprime_state_test_dprime_bnf_3scale_SOXL.csv",
        "stepDprime_state_test_dprime_bnf_h01_SOXL.csv",
        "stepDprime_state_test_dprime_bnf_h02_SOXL.csv",
        "stepDprime_state_test_dprime_mix_3scale_SOXL.csv",
        "stepDprime_state_test_dprime_mix_h01_SOXL.csv",
        "stepDprime_state_test_dprime_mix_h02_SOXL.csv",
        "stepDprime_state_train_dprime_bnf_3scale_SOXL.csv",
        "stepDprime_state_train_dprime_bnf_h01_SOXL.csv",
        "stepDprime_state_train_dprime_bnf_h02_SOXL.csv",
        "stepDprime_state_train_dprime_mix_3scale_SOXL.csv",
        "stepDprime_state_train_dprime_mix_h01_SOXL.csv",
        "stepDprime_state_train_dprime_mix_h02_SOXL.csv",
        "stepDprime_state_test_dprime_bnf_3scale_SOXL.csv",
        "stepDprime_state_test_dprime_bnf_h01_SOXL.csv",
        "stepDprime_state_test_dprime_bnf_h02_SOXL.csv",
        "stepDprime_state_test_dprime_mix_3scale_SOXL.csv",
        "stepDprime_state_test_dprime_mix_h01_SOXL.csv",
        "stepDprime_state_test_dprime_mix_h02_SOXL.csv",
        "stepDprime_state_train_dprime_bnf_3scale_SOXL.csv",
        "stepDprime_state_train_dprime_bnf_h01_SOXL.csv",
        "stepDprime_state_train_dprime_bnf_h02_SOXL.csv",
        "stepDprime_state_train_dprime_mix_3scale_SOXL.csv",
        "stepDprime_state_train_dprime_mix_h01_SOXL.csv",
        "stepDprime_state_train_dprime_mix_h02_SOXL.csv"
      ],
      "embeddings_files": [
        "stepDprime_bnf_3scale_SOXL_embeddings_all.csv",
        "stepDprime_bnf_h01_SOXL_embeddings_all.csv",
        "stepDprime_bnf_h02_SOXL_embeddings_all.csv",
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
        "stepDprime_bnf_3scale_SOXL_embeddings_all.csv",
        "stepDprime_bnf_h01_SOXL_embeddings_all.csv",
        "stepDprime_bnf_h02_SOXL_embeddings_all.csv",
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
      "traceback_count": 2,
      "traceback_files": [
        "stepDprime_traceback_SOXL.log",
        "stepDprime_traceback_SOXL.log"
      ],
      "failure_summary_count": 2,
      "failure_summary_files": [
        "stepDprime_failure_summary_SOXL.json",
        "stepDprime_failure_summary_SOXL.json"
      ],
      "failure_reason": "LinAlgError('SVD did not converge')",
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
    "status": "SKIP",
    "summary": "stepE_daily_log missing",
    "rows": []
  },
  "stepF": {
    "status": "SKIP",
    "summary": "stepF_equity_marl missing",
    "rows": []
  },
  "diversity": {
    "status": "SKIP",
    "summary": "not evaluated"
  },
  "overall_status": "WARN",
  "stepF_compare": {
    "status": "WARN",
    "summary": "compare skipped",
    "reason": "current StepF row with numeric equity_multiple not found"
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
      "PLOT StepE topN: stepE_daily_log files were not found",
      "PLOT StepF equity: stepF_equity_marl file not found",
      "PLOT StepE return bar: no numeric StepE rows",
      "PLOT StepE scatter: no numeric StepE DD/return pairs",
      "StepF_vs_best_StepE: 比較に必要な equity_multiple が不足",
      "PLOT StepF compare equity: comparison series unavailable",
      "PLOT StepF regret: regret values unavailable"
    ]
  }
}
```

Best-effort mode: this evaluator writes SKIP/notes and always exits 0.