# EVAL_REPORT

- output_root: `C:\work\apex_work\runs\gh22522501429_att1_sim_20260228_232205_448a5ff\output`
- mode: `sim`
- symbol: `SOXL`
- overall_status: **WARN**

## D' (stepD_prime) artifacts
- status: **OK**
- summary: D' state/embeddings found
- state_count: 20
- embeddings_count: 40

## StepA table
| status | summary | test_rows | test_date_start | test_date_end | missing_ohlcv_count |
|---|---|---:|---|---|---:|
| OK | prices_test evaluated | 63 | 2022-01-03 | 2022-04-01 | 0 |

## StepB table
| file | pred_col | non_null_ratio | coverage_ratio_over_test | mae | corr | status |
|---|---|---:|---:|---:|---:|---|
| stepB_pred_close_mamba_SOXL.csv | Pred_Close_MAMBA | 0.9384 | 1.0000 | 5.2983 | 0.8719 | OK |
| stepB_pred_time_all_SOXL.csv | Pred_Close_MAMBA | 0.9384 | 1.0000 | 5.2983 | 0.8719 | OK |

## StepE table
| agent | file | test_days | equity_multiple | max_dd | mean_ret | std_ret | sharpe | note | status |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| dprime_all_features_3scale | stepE_daily_log_dprime_all_features_3scale_SOXL.csv | 63 | 1.4400 | -0.2806 | 0.0077 | 0.0629 | 1.9529 |  | OK |
| dprime_all_features_h01 | stepE_daily_log_dprime_all_features_h01_SOXL.csv | 63 | 1.0775 | -0.2349 | 0.0021 | 0.0442 | 0.7675 |  | OK |
| dprime_all_features_h02 | stepE_daily_log_dprime_all_features_h02_SOXL.csv | 63 | 0.9120 | -0.4237 | 0.0007 | 0.0667 | 0.1715 |  | OK |
| dprime_all_features_h03 | stepE_daily_log_dprime_all_features_h03_SOXL.csv | 63 | 1.2142 | -0.4582 | 0.0061 | 0.0776 | 1.2390 |  | OK |
| dprime_bnf_3scale | stepE_daily_log_dprime_bnf_3scale_SOXL.csv | 63 | 0.8304 | -0.4728 | 0.0002 | 0.0795 | 0.0332 |  | OK |
| dprime_bnf_h01 | stepE_daily_log_dprime_bnf_h01_SOXL.csv | 63 | 0.9966 | -0.2076 | 0.0009 | 0.0440 | 0.3200 |  | OK |
| dprime_bnf_h02 | stepE_daily_log_dprime_bnf_h02_SOXL.csv | 63 | 1.0601 | -0.3003 | 0.0025 | 0.0573 | 0.6983 |  | OK |
| dprime_mix_3scale | stepE_daily_log_dprime_mix_3scale_SOXL.csv | 63 | 1.2965 | -0.3657 | 0.0066 | 0.0712 | 1.4734 |  | OK |
| dprime_mix_h01 | stepE_daily_log_dprime_mix_h01_SOXL.csv | 63 | 1.1416 | -0.1663 | 0.0029 | 0.0399 | 1.1417 |  | OK |
| dprime_mix_h02 | stepE_daily_log_dprime_mix_h02_SOXL.csv | 63 | 1.0972 | -0.0555 | 0.0015 | 0.0112 | 2.1732 |  | OK |

## StepF table
- SKIP: stepF_equity_marl missing

## Diversity
- status: **OK**
- summary: agent positions look diverse
- max_corr: 0.9217
- max_match_ratio: 0.0000
- pairs_over_0_9999: 0 / 45
- identical_all_agents: False

## PLOTS
- [equity_stepE_topN.png](./equity_stepE_topN.png)
- [equity_stepF.png](./equity_stepF.png)
- [bar_stepE_return.png](./bar_stepE_return.png)
- [scatter_stepE_dd_vs_ret.png](./scatter_stepE_dd_vs_ret.png)
  - note: PLOT StepF equity: stepF_equity_marl file not found
  - note: StepF_vs_best_StepE: 比較に必要な equity_multiple が不足

## Raw JSON
```json
{
  "output_root": "C:\\work\\apex_work\\runs\\gh22522501429_att1_sim_20260228_232205_448a5ff\\output",
  "mode": "sim",
  "symbol": "SOXL",
  "stepA": {
    "status": "OK",
    "summary": "prices_test evaluated",
    "details": {
      "path": "C:\\work\\apex_work\\runs\\gh22522501429_att1_sim_20260228_232205_448a5ff\\output\\stepA\\sim\\stepA_prices_test_SOXL.csv",
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
      }
    }
  },
  "stepB": {
    "status": "OK",
    "summary": "stepB files evaluated",
    "rows": [
      {
        "file": "stepB_pred_close_mamba_SOXL.csv",
        "pred_col": "Pred_Close_MAMBA",
        "non_null_ratio": 0.9383726528647087,
        "first_valid_date": "2014-07-09",
        "coverage_ratio_over_test": 1.0,
        "mae": 5.298298457312206,
        "corr": 0.8718667623174237,
        "status": "OK"
      },
      {
        "file": "stepB_pred_time_all_SOXL.csv",
        "pred_col": "Pred_Close_MAMBA",
        "non_null_ratio": 0.9383726528647087,
        "first_valid_date": "2014-07-09",
        "coverage_ratio_over_test": 1.0,
        "mae": 5.298298457312206,
        "corr": 0.8718667623174237,
        "status": "OK"
      }
    ],
    "files": [
      "stepB_pred_close_mamba_SOXL.csv",
      "stepB_pred_path_mamba_SOXL.csv",
      "stepB_pred_time_all_SOXL.csv"
    ]
  },
  "dprime": {
    "status": "OK",
    "summary": "D' state/embeddings found",
    "details": {
      "state_count": 20,
      "embeddings_count": 40,
      "state_files": [
        "stepDprime_state_dprime_all_features_3scale_SOXL_test.csv",
        "stepDprime_state_dprime_all_features_3scale_SOXL_train.csv",
        "stepDprime_state_dprime_all_features_h01_SOXL_test.csv",
        "stepDprime_state_dprime_all_features_h01_SOXL_train.csv",
        "stepDprime_state_dprime_all_features_h02_SOXL_test.csv",
        "stepDprime_state_dprime_all_features_h02_SOXL_train.csv",
        "stepDprime_state_dprime_all_features_h03_SOXL_test.csv",
        "stepDprime_state_dprime_all_features_h03_SOXL_train.csv",
        "stepDprime_state_dprime_bnf_3scale_SOXL_test.csv",
        "stepDprime_state_dprime_bnf_3scale_SOXL_train.csv",
        "stepDprime_state_dprime_bnf_h01_SOXL_test.csv",
        "stepDprime_state_dprime_bnf_h01_SOXL_train.csv",
        "stepDprime_state_dprime_bnf_h02_SOXL_test.csv",
        "stepDprime_state_dprime_bnf_h02_SOXL_train.csv",
        "stepDprime_state_dprime_mix_3scale_SOXL_test.csv",
        "stepDprime_state_dprime_mix_3scale_SOXL_train.csv",
        "stepDprime_state_dprime_mix_h01_SOXL_test.csv",
        "stepDprime_state_dprime_mix_h01_SOXL_train.csv",
        "stepDprime_state_dprime_mix_h02_SOXL_test.csv",
        "stepDprime_state_dprime_mix_h02_SOXL_train.csv"
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
        "stepDprime_mix_h02_SOXL_embeddings_all.csv"
      ],
      "searched": [
        "C:\\work\\apex_work\\runs\\gh22522501429_att1_sim_20260228_232205_448a5ff\\output\\stepD_prime\\sim\\stepDprime_state_*_SOXL_train.csv",
        "C:\\work\\apex_work\\runs\\gh22522501429_att1_sim_20260228_232205_448a5ff\\output\\stepD_prime\\sim\\stepDprime_state_*_SOXL_test.csv",
        "C:\\work\\apex_work\\runs\\gh22522501429_att1_sim_20260228_232205_448a5ff\\output\\stepD_prime\\sim\\embeddings\\stepDprime_*_SOXL_embeddings*.csv"
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
        "equity_multiple": 1.4400209167406701,
        "max_dd": -0.2805548017481161,
        "mean_ret": 0.007739080336972306,
        "std_ret": 0.06290827661132302,
        "sharpe": 1.9529082388780494,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_all_features_h01_SOXL.csv",
        "agent": "dprime_all_features_h01",
        "test_days": 63,
        "equity_multiple": 1.0774916713101745,
        "max_dd": -0.23491112889848287,
        "mean_ret": 0.0021380102779302297,
        "std_ret": 0.044219988126672695,
        "sharpe": 0.7675230684865098,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_all_features_h02_SOXL.csv",
        "agent": "dprime_all_features_h02",
        "test_days": 63,
        "equity_multiple": 0.9120335842841772,
        "max_dd": -0.42369591781382543,
        "mean_ret": 0.0007199999065108824,
        "std_ret": 0.06666265193314476,
        "sharpe": 0.17145498788689,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_all_features_h03_SOXL.csv",
        "agent": "dprime_all_features_h03",
        "test_days": 63,
        "equity_multiple": 1.2142271143956678,
        "max_dd": -0.4581875182770594,
        "mean_ret": 0.006059686062924032,
        "std_ret": 0.07763588281286306,
        "sharpe": 1.239047339818342,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_bnf_3scale_SOXL.csv",
        "agent": "dprime_bnf_3scale",
        "test_days": 63,
        "equity_multiple": 0.8304404338379203,
        "max_dd": -0.4728328231484925,
        "mean_ret": 0.00016621882945949915,
        "std_ret": 0.07947015286066626,
        "sharpe": 0.03320293243203946,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_bnf_h01_SOXL.csv",
        "agent": "dprime_bnf_h01",
        "test_days": 63,
        "equity_multiple": 0.9966419789607223,
        "max_dd": -0.20756731338512424,
        "mean_ret": 0.0008871131350234644,
        "std_ret": 0.04401317127851223,
        "sharpe": 0.31996068520472515,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_bnf_h02_SOXL.csv",
        "agent": "dprime_bnf_h02",
        "test_days": 63,
        "equity_multiple": 1.0601466064510443,
        "max_dd": -0.3003151938367029,
        "mean_ret": 0.0025200409050624368,
        "std_ret": 0.05728793842641272,
        "sharpe": 0.6983042202228721,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_mix_3scale_SOXL.csv",
        "agent": "dprime_mix_3scale",
        "test_days": 63,
        "equity_multiple": 1.2964886288408761,
        "max_dd": -0.36565287194526086,
        "mean_ret": 0.006605295230641901,
        "std_ret": 0.07116702451063614,
        "sharpe": 1.4733763540017883,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_mix_h01_SOXL.csv",
        "agent": "dprime_mix_h01",
        "test_days": 63,
        "equity_multiple": 1.1415646012588712,
        "max_dd": -0.16633662585948616,
        "mean_ret": 0.002872486886363258,
        "std_ret": 0.03993990862801148,
        "sharpe": 1.1416980468925748,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_mix_h02_SOXL.csv",
        "agent": "dprime_mix_h02",
        "test_days": 63,
        "equity_multiple": 1.0971991222585995,
        "max_dd": -0.05552629730257164,
        "mean_ret": 0.0015347924709203551,
        "std_ret": 0.011211085216083943,
        "sharpe": 2.173212912336419,
        "status": "OK"
      }
    ]
  },
  "stepF": {
    "status": "SKIP",
    "summary": "stepF_equity_marl missing",
    "rows": []
  },
  "diversity": {
    "status": "OK",
    "summary": "agent positions look diverse",
    "max_corr": 0.9217494652615404,
    "max_match_ratio": 0.0,
    "pairs_over_0_9999": 0,
    "all_pairs": 45,
    "identical_all_agents": false
  },
  "overall_status": "WARN",
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
      }
    ],
    "notes": [
      "PLOT StepF equity: stepF_equity_marl file not found",
      "StepF_vs_best_StepE: 比較に必要な equity_multiple が不足"
    ]
  }
}
```

Best-effort mode: this evaluator writes SKIP/notes and always exits 0.