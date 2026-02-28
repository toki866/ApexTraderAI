# EVAL_REPORT

- output_root: `C:\work\apex_work\runs\20260228_155646\output`
- mode: `sim`
- symbol: `SOXL`
- overall_status: **OK**

## D' (stepD_prime) artifacts
- status: **OK**
- summary: D' state/embeddings found
- state_count: 6
- embeddings_count: 4

## StepA table
| status | summary | test_rows | test_date_start | test_date_end | missing_ohlcv_count |
|---|---|---:|---|---|---:|
| OK | prices_test evaluated | 63 | 2022-01-03 | 2022-04-01 | 0 |

## StepB table
| file | pred_col | non_null_ratio | coverage_ratio_over_test | mae | corr | status |
|---|---|---:|---:|---:|---:|---|
| stepB_pred_close_mamba_SOXL.csv | Pred_Close_MAMBA | 0.9711 | 1.0000 | 5.0488 | 0.8823 | OK |
| stepB_pred_time_all_SOXL.csv | Pred_Close_MAMBA | 0.9711 | 1.0000 | 5.0488 | 0.8823 | OK |

## StepE table
| agent | file | test_days | equity_multiple | max_dd | mean_ret | std_ret | sharpe | note | status |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| dprime_all_features_3scale | stepE_daily_log_dprime_all_features_3scale_SOXL.csv | 63 | 0.5460 | -0.5808 | -0.0071 | 0.0727 | -1.5431 |  | OK |
| dprime_all_features_h01 | stepE_daily_log_dprime_all_features_h01_SOXL.csv | 63 | 0.7458 | -0.3550 | -0.0039 | 0.0423 | -1.4583 |  | OK |
| dprime_all_features_h02 | stepE_daily_log_dprime_all_features_h02_SOXL.csv | 63 | 0.3943 | -0.6114 | -0.0129 | 0.0631 | -3.2493 |  | OK |
| dprime_all_features_h03 | stepE_daily_log_dprime_all_features_h03_SOXL.csv | 63 | 0.3783 | -0.6424 | -0.0135 | 0.0635 | -3.3754 |  | OK |
| dprime_bnf_3scale | stepE_daily_log_dprime_bnf_3scale_SOXL.csv | 63 | 0.4238 | -0.5762 | -0.0116 | 0.0654 | -2.8255 |  | OK |
| dprime_bnf_h01 | stepE_daily_log_dprime_bnf_h01_SOXL.csv | 63 | 0.7295 | -0.3948 | -0.0040 | 0.0490 | -1.2829 |  | OK |
| dprime_bnf_h02 | stepE_daily_log_dprime_bnf_h02_SOXL.csv | 63 | 0.4040 | -0.6007 | -0.0125 | 0.0634 | -3.1332 |  | OK |
| dprime_bnf_h03 | stepE_daily_log_dprime_bnf_h03_SOXL.csv | 63 | 0.4397 | -0.5603 | -0.0113 | 0.0610 | -2.9488 |  | OK |
| dprime_mix_3scale | stepE_daily_log_dprime_mix_3scale_SOXL.csv | 63 | 0.4511 | -0.5489 | -0.0107 | 0.0657 | -2.5759 |  | OK |
| dprime_mix_h01 | stepE_daily_log_dprime_mix_h01_SOXL.csv | 63 | 0.3863 | -0.6137 | -0.0130 | 0.0675 | -3.0506 |  | OK |

## StepF table
| file | test_days | equity_multiple | max_dd | mean_ret | std_ret | sharpe | note | status |
|---|---:|---:|---:|---:|---:|---:|---|---|
| stepF_equity_marl_SOXL.csv | 63 | 0.4859 | -0.5151 | -0.0101 | 0.0554 | -2.8840 | Split missing: evaluated all rows as test | OK |

## Diversity
- status: **OK**
- summary: agent positions look diverse
- max_corr: 0.9957
- max_match_ratio: 0.0000
- pairs_over_0_9999: 0 / 45
- identical_all_agents: False

## PLOTS
- [equity_stepE_topN.png](./equity_stepE_topN.png)
- [equity_stepF.png](./equity_stepF.png)
- [bar_stepE_return.png](./bar_stepE_return.png)
- [scatter_stepE_dd_vs_ret.png](./scatter_stepE_dd_vs_ret.png)
  - note: StepF_vs_best_StepE: StepF(0.4859) は best StepE(0.7458) に負けてる

## Raw JSON
```json
{
  "output_root": "C:\\work\\apex_work\\runs\\20260228_155646\\output",
  "mode": "sim",
  "symbol": "SOXL",
  "stepA": {
    "status": "OK",
    "summary": "prices_test evaluated",
    "details": {
      "path": "C:\\work\\apex_work\\runs\\20260228_155646\\output\\stepA\\sim\\stepA_prices_test_SOXL.csv",
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
        "non_null_ratio": 0.9711121810303323,
        "first_valid_date": "2014-04-01",
        "coverage_ratio_over_test": 1.0,
        "mae": 5.048780471559555,
        "corr": 0.8822700440244475,
        "status": "OK"
      },
      {
        "file": "stepB_pred_time_all_SOXL.csv",
        "pred_col": "Pred_Close_MAMBA",
        "non_null_ratio": 0.9711121810303323,
        "first_valid_date": "2014-04-01",
        "coverage_ratio_over_test": 1.0,
        "mae": 5.048780471559555,
        "corr": 0.8822700440244475,
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
      "state_count": 6,
      "embeddings_count": 4,
      "state_files": [
        "stepDprime_state_all_features_SOXL_test.csv",
        "stepDprime_state_all_features_SOXL_train.csv",
        "stepDprime_state_bnf_SOXL_test.csv",
        "stepDprime_state_bnf_SOXL_train.csv",
        "stepDprime_state_mix_SOXL_test.csv",
        "stepDprime_state_mix_SOXL_train.csv"
      ],
      "embeddings_files": [
        "stepDprime_mamba_h01_SOXL_embeddings_all.csv",
        "stepDprime_mamba_h05_SOXL_embeddings_all.csv",
        "stepDprime_mamba_h10_SOXL_embeddings_all.csv",
        "stepDprime_mamba_h20_SOXL_embeddings_all.csv"
      ],
      "searched": [
        "C:\\work\\apex_work\\runs\\20260228_155646\\output\\stepD_prime\\sim\\stepDprime_state_*_SOXL_train.csv",
        "C:\\work\\apex_work\\runs\\20260228_155646\\output\\stepD_prime\\sim\\stepDprime_state_*_SOXL_test.csv",
        "C:\\work\\apex_work\\runs\\20260228_155646\\output\\stepD_prime\\sim\\embeddings\\stepDprime_*_SOXL_embeddings*.csv"
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
        "equity_multiple": 0.5460416463878625,
        "max_dd": -0.5808031635751338,
        "mean_ret": -0.007070804143228399,
        "std_ret": 0.07274054343997735,
        "sharpe": -1.543094547897975,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_all_features_h01_SOXL.csv",
        "agent": "dprime_all_features_h01",
        "test_days": 63,
        "equity_multiple": 0.7458387303636225,
        "max_dd": -0.35495616136307206,
        "mean_ret": -0.003883241018417126,
        "std_ret": 0.04227095800554086,
        "sharpe": -1.4583189736523385,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_all_features_h02_SOXL.csv",
        "agent": "dprime_all_features_h02",
        "test_days": 63,
        "equity_multiple": 0.3943217630974008,
        "max_dd": -0.6113687670946084,
        "mean_ret": -0.012922197212612941,
        "std_ret": 0.06313195359713651,
        "sharpe": -3.2492820135371137,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_all_features_h03_SOXL.csv",
        "agent": "dprime_all_features_h03",
        "test_days": 63,
        "equity_multiple": 0.3782626848843894,
        "max_dd": -0.6423502732463924,
        "mean_ret": -0.013500951724829121,
        "std_ret": 0.06349525604609442,
        "sharpe": -3.3753854650799724,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_bnf_3scale_SOXL.csv",
        "agent": "dprime_bnf_3scale",
        "test_days": 63,
        "equity_multiple": 0.4238259589228568,
        "max_dd": -0.5761740410771432,
        "mean_ret": -0.011646952406163246,
        "std_ret": 0.06543565934250103,
        "sharpe": -2.825518065361532,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_bnf_h01_SOXL.csv",
        "agent": "dprime_bnf_h01",
        "test_days": 63,
        "equity_multiple": 0.7295212321841182,
        "max_dd": -0.39476626081916233,
        "mean_ret": -0.00395753437150565,
        "std_ret": 0.04897189306955502,
        "sharpe": -1.2828564830593014,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_bnf_h02_SOXL.csv",
        "agent": "dprime_bnf_h02",
        "test_days": 63,
        "equity_multiple": 0.4040483895423722,
        "max_dd": -0.6007100075696925,
        "mean_ret": -0.01251888481460303,
        "std_ret": 0.06342796884671811,
        "sharpe": -3.133178297859023,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_bnf_h03_SOXL.csv",
        "agent": "dprime_bnf_h03",
        "test_days": 63,
        "equity_multiple": 0.43971160169360896,
        "max_dd": -0.5602883983063911,
        "mean_ret": -0.011331319868091546,
        "std_ret": 0.06100082597741211,
        "sharpe": -2.9487982088829394,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_mix_3scale_SOXL.csv",
        "agent": "dprime_mix_3scale",
        "test_days": 63,
        "equity_multiple": 0.4511025613997203,
        "max_dd": -0.5488974386002796,
        "mean_ret": -0.010653721129832162,
        "std_ret": 0.06565588075633727,
        "sharpe": -2.575893856476759,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_mix_h01_SOXL.csv",
        "agent": "dprime_mix_h01",
        "test_days": 63,
        "equity_multiple": 0.3862806064706211,
        "max_dd": -0.6137193935293789,
        "mean_ret": -0.012968023344116748,
        "std_ret": 0.06748149071539321,
        "sharpe": -3.05062894143682,
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
        "equity_multiple": 0.4859446388895488,
        "max_dd": -0.515078214690703,
        "mean_ret": -0.010071899633326128,
        "std_ret": 0.0554392135827994,
        "sharpe": -2.8839956346045263,
        "status": "OK",
        "note": "Split missing: evaluated all rows as test"
      }
    ]
  },
  "diversity": {
    "status": "OK",
    "summary": "agent positions look diverse",
    "max_corr": 0.9957432137181033,
    "max_match_ratio": 0.0,
    "pairs_over_0_9999": 0,
    "all_pairs": 45,
    "identical_all_agents": false
  },
  "overall_status": "OK",
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
      "StepF_vs_best_StepE: StepF(0.4859) は best StepE(0.7458) に負けてる"
    ]
  }
}
```

Best-effort mode: this evaluator writes SKIP/notes and always exits 0.