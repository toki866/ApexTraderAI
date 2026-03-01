# EVAL_REPORT

- output_root: `C:\work\apex_work\runs\gh22539025002_att1_sim_20260301_165903_5572072\output`
- mode: `sim`
- symbol: `SOXL`
- overall_status: **OK**

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
| dprime_all_features_3scale | stepE_daily_log_dprime_all_features_3scale_SOXL.csv | 63 | 1.3049 | -0.3180 | 0.0057 | 0.0671 | 1.3405 |  | OK |
| dprime_all_features_h01 | stepE_daily_log_dprime_all_features_h01_SOXL.csv | 63 | 1.1969 | -0.1841 | 0.0034 | 0.0439 | 1.2145 |  | OK |
| dprime_all_features_h02 | stepE_daily_log_dprime_all_features_h02_SOXL.csv | 63 | 0.9552 | -0.4460 | 0.0013 | 0.0684 | 0.2919 |  | OK |
| dprime_all_features_h03 | stepE_daily_log_dprime_all_features_h03_SOXL.csv | 63 | 1.8261 | -0.2568 | 0.0111 | 0.0709 | 2.4947 |  | OK |
| dprime_bnf_3scale | stepE_daily_log_dprime_bnf_3scale_SOXL.csv | 63 | 1.6120 | -0.3627 | 0.0099 | 0.0821 | 1.9213 |  | OK |
| dprime_bnf_h01 | stepE_daily_log_dprime_bnf_h01_SOXL.csv | 63 | 0.9451 | -0.2785 | 0.0003 | 0.0540 | 0.0748 |  | OK |
| dprime_bnf_h02 | stepE_daily_log_dprime_bnf_h02_SOXL.csv | 63 | 1.2223 | -0.3829 | 0.0046 | 0.0656 | 1.1180 |  | OK |
| dprime_mix_3scale | stepE_daily_log_dprime_mix_3scale_SOXL.csv | 63 | 1.3334 | -0.3835 | 0.0063 | 0.0734 | 1.3532 |  | OK |
| dprime_mix_h01 | stepE_daily_log_dprime_mix_h01_SOXL.csv | 63 | 1.1265 | -0.4772 | 0.0040 | 0.0778 | 0.8208 |  | OK |
| dprime_mix_h02 | stepE_daily_log_dprime_mix_h02_SOXL.csv | 63 | 1.0470 | -0.3894 | 0.0021 | 0.0684 | 0.4943 |  | OK |

## StepF table
| file | test_days | equity_multiple | max_dd | mean_ret | std_ret | sharpe | note | status |
|---|---:|---:|---:|---:|---:|---:|---|---|
| stepF_equity_marl_SOXL.csv | 63 | 1.0636 | -0.2089 | 0.0022 | 0.0474 | 0.7370 | Split missing: evaluated all rows as test | OK |

## Diversity
- status: **OK**
- summary: agent positions look diverse
- max_corr: 0.8963
- max_match_ratio: 0.0000
- pairs_over_0_9999: 0 / 45
- identical_all_agents: False

## PLOTS
- [equity_stepE_topN.png](./equity_stepE_topN.png)
- [equity_stepF.png](./equity_stepF.png)
- [bar_stepE_return.png](./bar_stepE_return.png)
- [scatter_stepE_dd_vs_ret.png](./scatter_stepE_dd_vs_ret.png)
  - note: StepF_vs_best_StepE: StepF(1.0636) は best StepE(1.8261) に負けてる

## Raw JSON
```json
{
  "output_root": "C:\\work\\apex_work\\runs\\gh22539025002_att1_sim_20260301_165903_5572072\\output",
  "mode": "sim",
  "symbol": "SOXL",
  "stepA": {
    "status": "OK",
    "summary": "prices_test evaluated",
    "details": {
      "path": "C:\\work\\apex_work\\runs\\gh22539025002_att1_sim_20260301_165903_5572072\\output\\stepA\\sim\\stepA_prices_test_SOXL.csv",
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
        "C:\\work\\apex_work\\runs\\gh22539025002_att1_sim_20260301_165903_5572072\\output\\stepD_prime\\sim\\stepDprime_state_*_SOXL_train.csv",
        "C:\\work\\apex_work\\runs\\gh22539025002_att1_sim_20260301_165903_5572072\\output\\stepD_prime\\sim\\stepDprime_state_*_SOXL_test.csv",
        "C:\\work\\apex_work\\runs\\gh22539025002_att1_sim_20260301_165903_5572072\\output\\stepD_prime\\sim\\embeddings\\stepDprime_*_SOXL_embeddings*.csv"
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
        "equity_multiple": 1.3048694618506491,
        "max_dd": -0.3179936103937364,
        "mean_ret": 0.005669116395958032,
        "std_ret": 0.06713340613720081,
        "sharpe": 1.3405313092438718,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_all_features_h01_SOXL.csv",
        "agent": "dprime_all_features_h01",
        "test_days": 63,
        "equity_multiple": 1.1968890886213277,
        "max_dd": -0.1840866087530898,
        "mean_ret": 0.0033565455814823448,
        "std_ret": 0.043873500428420524,
        "sharpe": 1.2144804658124262,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_all_features_h02_SOXL.csv",
        "agent": "dprime_all_features_h02",
        "test_days": 63,
        "equity_multiple": 0.9552165725485636,
        "max_dd": -0.4460269895013468,
        "mean_ret": 0.0012580950269549457,
        "std_ret": 0.06840943896468098,
        "sharpe": 0.29194274509940754,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_all_features_h03_SOXL.csv",
        "agent": "dprime_all_features_h03",
        "test_days": 63,
        "equity_multiple": 1.8261050717419622,
        "max_dd": -0.25683353041904755,
        "mean_ret": 0.011136035533531357,
        "std_ret": 0.07086319552432377,
        "sharpe": 2.494653005264706,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_bnf_3scale_SOXL.csv",
        "agent": "dprime_bnf_3scale",
        "test_days": 63,
        "equity_multiple": 1.611972483978526,
        "max_dd": -0.3626546548211921,
        "mean_ret": 0.009938982494604141,
        "std_ret": 0.0821179338555806,
        "sharpe": 1.9213398144181306,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_bnf_h01_SOXL.csv",
        "agent": "dprime_bnf_h01",
        "test_days": 63,
        "equity_multiple": 0.9450860202075305,
        "max_dd": -0.2784761071955667,
        "mean_ret": 0.0002542900056430411,
        "std_ret": 0.05397767265862545,
        "sharpe": 0.0747851564563359,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_bnf_h02_SOXL.csv",
        "agent": "dprime_bnf_h02",
        "test_days": 63,
        "equity_multiple": 1.2222549767696067,
        "max_dd": -0.3828942150031076,
        "mean_ret": 0.004620177553050106,
        "std_ret": 0.06559934922063256,
        "sharpe": 1.1180453126649759,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_mix_3scale_SOXL.csv",
        "agent": "dprime_mix_3scale",
        "test_days": 63,
        "equity_multiple": 1.3333599379054573,
        "max_dd": -0.3834621330309317,
        "mean_ret": 0.0062587486916325175,
        "std_ret": 0.07341930532383835,
        "sharpe": 1.3532483711311274,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_mix_h01_SOXL.csv",
        "agent": "dprime_mix_h01",
        "test_days": 63,
        "equity_multiple": 1.1264763582808088,
        "max_dd": -0.4772423104102992,
        "mean_ret": 0.0040242872459606985,
        "std_ret": 0.07783267921360877,
        "sharpe": 0.8207809391641288,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_mix_h02_SOXL.csv",
        "agent": "dprime_mix_h02",
        "test_days": 63,
        "equity_multiple": 1.04702457532063,
        "max_dd": -0.3893534327817587,
        "mean_ret": 0.0021314784359659925,
        "std_ret": 0.0684482814318052,
        "sharpe": 0.49433193194906483,
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
        "equity_multiple": 1.0635739504699853,
        "max_dd": -0.20891042445331887,
        "mean_ret": 0.0021987525464747834,
        "std_ret": 0.04736197943117102,
        "sharpe": 0.736964861149398,
        "status": "OK",
        "note": "Split missing: evaluated all rows as test"
      }
    ]
  },
  "diversity": {
    "status": "OK",
    "summary": "agent positions look diverse",
    "max_corr": 0.8962707826632739,
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
      "StepF_vs_best_StepE: StepF(1.0636) は best StepE(1.8261) に負けてる"
    ]
  }
}
```

Best-effort mode: this evaluator writes SKIP/notes and always exits 0.