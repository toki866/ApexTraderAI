# EVAL_REPORT

- output_root: `C:\work\apex_work\runs\gh22521319788_att1_sim_20260228_220622_a550ea7\output`
- mode: `sim`
- symbol: `SOXL`
- overall_status: **WARN**

## D' (stepD_prime) artifacts
- status: **OK**
- summary: D' state/embeddings found
- state_count: 20
- embeddings_count: 30

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
| dprime_all_features_3scale | stepE_daily_log_dprime_all_features_3scale_SOXL.csv | 63 | 1.5853 | -0.3359 | 0.0101 | 0.0747 | 2.1383 |  | OK |
| dprime_all_features_h01 | stepE_daily_log_dprime_all_features_h01_SOXL.csv | 63 | 0.9004 | -0.5110 | 0.0001 | 0.0603 | 0.0371 |  | OK |
| dprime_all_features_h02 | stepE_daily_log_dprime_all_features_h02_SOXL.csv | 63 | 0.8195 | -0.4733 | -0.0009 | 0.0677 | -0.2103 |  | OK |
| dprime_all_features_h03 | stepE_daily_log_dprime_all_features_h03_SOXL.csv | 63 | 1.2208 | -0.4388 | 0.0059 | 0.0745 | 1.2556 |  | OK |
| dprime_bnf_3scale | stepE_daily_log_dprime_bnf_3scale_SOXL.csv | 63 | 0.8549 | -0.4448 | 0.0008 | 0.0807 | 0.1492 |  | OK |
| dprime_bnf_h01 | stepE_daily_log_dprime_bnf_h01_SOXL.csv | 63 | 0.4942 | -0.5452 | -0.0095 | 0.0570 | -2.6547 |  | OK |
| dprime_bnf_h02 | stepE_daily_log_dprime_bnf_h02_SOXL.csv | 63 | 0.6971 | -0.3981 | -0.0044 | 0.0512 | -1.3686 |  | OK |
| dprime_mix_3scale | stepE_daily_log_dprime_mix_3scale_SOXL.csv | 63 | 0.9265 | -0.4453 | 0.0020 | 0.0806 | 0.3971 |  | OK |
| dprime_mix_h01 | stepE_daily_log_dprime_mix_h01_SOXL.csv | 63 | 0.8300 | -0.1941 | -0.0028 | 0.0168 | -2.6670 |  | OK |
| dprime_mix_h02 | stepE_daily_log_dprime_mix_h02_SOXL.csv | 63 | 0.4318 | -0.5682 | -0.0106 | 0.0731 | -2.3050 |  | OK |

## StepF table
- SKIP: stepF_equity_marl missing

## Diversity
- status: **OK**
- summary: agent positions look diverse
- max_corr: 0.8337
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
  "output_root": "C:\\work\\apex_work\\runs\\gh22521319788_att1_sim_20260228_220622_a550ea7\\output",
  "mode": "sim",
  "symbol": "SOXL",
  "stepA": {
    "status": "OK",
    "summary": "prices_test evaluated",
    "details": {
      "path": "C:\\work\\apex_work\\runs\\gh22521319788_att1_sim_20260228_220622_a550ea7\\output\\stepA\\sim\\stepA_prices_test_SOXL.csv",
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
      "embeddings_count": 30,
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
        "stepDprime_dprime_mix_h02_SOXL_embeddings_train.csv"
      ],
      "searched": [
        "C:\\work\\apex_work\\runs\\gh22521319788_att1_sim_20260228_220622_a550ea7\\output\\stepD_prime\\sim\\stepDprime_state_*_SOXL_train.csv",
        "C:\\work\\apex_work\\runs\\gh22521319788_att1_sim_20260228_220622_a550ea7\\output\\stepD_prime\\sim\\stepDprime_state_*_SOXL_test.csv",
        "C:\\work\\apex_work\\runs\\gh22521319788_att1_sim_20260228_220622_a550ea7\\output\\stepD_prime\\sim\\embeddings\\stepDprime_*_SOXL_embeddings*.csv"
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
        "equity_multiple": 1.5853072087517779,
        "max_dd": -0.3359355907413403,
        "mean_ret": 0.010067093377796162,
        "std_ret": 0.07473616544033278,
        "sharpe": 2.1383242246362104,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_all_features_h01_SOXL.csv",
        "agent": "dprime_all_features_h01",
        "test_days": 63,
        "equity_multiple": 0.9004105170234729,
        "max_dd": -0.5109715095290324,
        "mean_ret": 0.00014091061420335086,
        "std_ret": 0.06032925612262992,
        "sharpe": 0.037077975055447805,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_all_features_h02_SOXL.csv",
        "agent": "dprime_all_features_h02",
        "test_days": 63,
        "equity_multiple": 0.8195000014793159,
        "max_dd": -0.47326701785423375,
        "mean_ret": -0.0008972869424421002,
        "std_ret": 0.06774148693638105,
        "sharpe": -0.21026979581332622,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_all_features_h03_SOXL.csv",
        "agent": "dprime_all_features_h03",
        "test_days": 63,
        "equity_multiple": 1.2208216365794882,
        "max_dd": -0.43875918158731986,
        "mean_ret": 0.0058919788256938015,
        "std_ret": 0.07449214357331485,
        "sharpe": 1.2555990434751156,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_bnf_3scale_SOXL.csv",
        "agent": "dprime_bnf_3scale",
        "test_days": 63,
        "equity_multiple": 0.8549443346325122,
        "max_dd": -0.4447626520166542,
        "mean_ret": 0.0007585311042411011,
        "std_ret": 0.08071619330369655,
        "sharpe": 0.14918082095211416,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_bnf_h01_SOXL.csv",
        "agent": "dprime_bnf_h01",
        "test_days": 63,
        "equity_multiple": 0.49417388238396837,
        "max_dd": -0.5451629183516613,
        "mean_ret": -0.009529229873603771,
        "std_ret": 0.056983710164082574,
        "sharpe": -2.654650498424135,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_bnf_h02_SOXL.csv",
        "agent": "dprime_bnf_h02",
        "test_days": 63,
        "equity_multiple": 0.6970884537908985,
        "max_dd": -0.39812436425119735,
        "mean_ret": -0.0044130084117770954,
        "std_ret": 0.051185663828549964,
        "sharpe": -1.368631986132717,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_mix_3scale_SOXL.csv",
        "agent": "dprime_mix_3scale",
        "test_days": 63,
        "equity_multiple": 0.9265186771513594,
        "max_dd": -0.4452773926964718,
        "mean_ret": 0.002016793772955208,
        "std_ret": 0.08063107698871481,
        "sharpe": 0.39706289199311817,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_mix_h01_SOXL.csv",
        "agent": "dprime_mix_h01",
        "test_days": 63,
        "equity_multiple": 0.8300200121129095,
        "max_dd": -0.1941286706941524,
        "mean_ret": -0.002816708236210495,
        "std_ret": 0.016765352728658353,
        "sharpe": -2.6670394459765303,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_mix_h02_SOXL.csv",
        "agent": "dprime_mix_h02",
        "test_days": 63,
        "equity_multiple": 0.43176377560811,
        "max_dd": -0.5682362243918899,
        "mean_ret": -0.010619439555624012,
        "std_ret": 0.0731361625045514,
        "sharpe": -2.3049934668350898,
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
    "max_corr": 0.8337385476449064,
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