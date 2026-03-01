# EVAL_REPORT

- output_root: `C:\work\apex_work\runs\20260228_172001\output`
- mode: `sim`
- symbol: `SOXL`
- overall_status: **OK**

## D' (stepD_prime) artifacts
- status: **OK**
- summary: D' state/embeddings found
- state_count: 6
- embeddings_count: 8

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
| dprime_all_features_3scale | stepE_daily_log_dprime_all_features_3scale_SOXL.csv | 63 | 0.5840 | -0.5417 | -0.0061 | 0.0699 | -1.3943 |  | OK |
| dprime_all_features_h01 | stepE_daily_log_dprime_all_features_h01_SOXL.csv | 63 | 0.7992 | -0.2854 | -0.0032 | 0.0304 | -1.6672 |  | OK |
| dprime_all_features_h02 | stepE_daily_log_dprime_all_features_h02_SOXL.csv | 63 | 0.3727 | -0.6339 | -0.0136 | 0.0662 | -3.2656 |  | OK |
| dprime_all_features_h03 | stepE_daily_log_dprime_all_features_h03_SOXL.csv | 63 | 0.3628 | -0.6521 | -0.0139 | 0.0673 | -3.2867 |  | OK |
| dprime_bnf_3scale | stepE_daily_log_dprime_bnf_3scale_SOXL.csv | 63 | 0.5793 | -0.4392 | -0.0069 | 0.0618 | -1.7845 |  | OK |
| dprime_bnf_h01 | stepE_daily_log_dprime_bnf_h01_SOXL.csv | 63 | 0.6033 | -0.5178 | -0.0058 | 0.0691 | -1.3435 |  | OK |
| dprime_bnf_h02 | stepE_daily_log_dprime_bnf_h02_SOXL.csv | 63 | 0.3852 | -0.6185 | -0.0131 | 0.0667 | -3.1073 |  | OK |
| dprime_bnf_h03 | stepE_daily_log_dprime_bnf_h03_SOXL.csv | 63 | 0.4974 | -0.5026 | -0.0094 | 0.0603 | -2.4809 |  | OK |
| dprime_mix_3scale | stepE_daily_log_dprime_mix_3scale_SOXL.csv | 63 | 0.3631 | -0.6369 | -0.0139 | 0.0681 | -3.2422 |  | OK |
| dprime_mix_h01 | stepE_daily_log_dprime_mix_h01_SOXL.csv | 63 | 0.3742 | -0.6258 | -0.0134 | 0.0690 | -3.0770 |  | OK |

## StepF table
| file | test_days | equity_multiple | max_dd | mean_ret | std_ret | sharpe | note | status |
|---|---:|---:|---:|---:|---:|---:|---|---|
| stepF_equity_marl_SOXL.csv | 63 | 0.4812 | -0.5216 | -0.0102 | 0.0552 | -2.9407 | Split missing: evaluated all rows as test | OK |

## Diversity
- status: **OK**
- summary: agent positions look diverse
- max_corr: 0.9936
- max_match_ratio: 0.0000
- pairs_over_0_9999: 0 / 45
- identical_all_agents: False

## PLOTS
- [equity_stepE_topN.png](./equity_stepE_topN.png)
- [equity_stepF.png](./equity_stepF.png)
- [bar_stepE_return.png](./bar_stepE_return.png)
- [scatter_stepE_dd_vs_ret.png](./scatter_stepE_dd_vs_ret.png)
  - note: StepF_vs_best_StepE: StepF(0.4812) は best StepE(0.7992) に負けてる

## Raw JSON
```json
{
  "output_root": "C:\\work\\apex_work\\runs\\20260228_172001\\output",
  "mode": "sim",
  "symbol": "SOXL",
  "stepA": {
    "status": "OK",
    "summary": "prices_test evaluated",
    "details": {
      "path": "C:\\work\\apex_work\\runs\\20260228_172001\\output\\stepA\\sim\\stepA_prices_test_SOXL.csv",
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
      "state_count": 6,
      "embeddings_count": 8,
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
        "stepDprime_mamba_h20_SOXL_embeddings_all.csv",
        "stepDprime_mamba_periodic_h01_SOXL_embeddings_all.csv",
        "stepDprime_mamba_periodic_h05_SOXL_embeddings_all.csv",
        "stepDprime_mamba_periodic_h10_SOXL_embeddings_all.csv",
        "stepDprime_mamba_periodic_h20_SOXL_embeddings_all.csv"
      ],
      "searched": [
        "C:\\work\\apex_work\\runs\\20260228_172001\\output\\stepD_prime\\sim\\stepDprime_state_*_SOXL_train.csv",
        "C:\\work\\apex_work\\runs\\20260228_172001\\output\\stepD_prime\\sim\\stepDprime_state_*_SOXL_test.csv",
        "C:\\work\\apex_work\\runs\\20260228_172001\\output\\stepD_prime\\sim\\embeddings\\stepDprime_*_SOXL_embeddings*.csv"
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
        "equity_multiple": 0.5839956012861601,
        "max_dd": -0.541726046756281,
        "mean_ret": -0.0061364169759187985,
        "std_ret": 0.06986552518814927,
        "sharpe": -1.3942870864181287,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_all_features_h01_SOXL.csv",
        "agent": "dprime_all_features_h01",
        "test_days": 63,
        "equity_multiple": 0.7991728122642803,
        "max_dd": -0.2853740837629083,
        "mean_ret": -0.0031881301705623914,
        "std_ret": 0.030355841163887214,
        "sharpe": -1.6672243473150157,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_all_features_h02_SOXL.csv",
        "agent": "dprime_all_features_h02",
        "test_days": 63,
        "equity_multiple": 0.37270933936978773,
        "max_dd": -0.6339002915066227,
        "mean_ret": -0.013612254566165593,
        "std_ret": 0.06617058530798504,
        "sharpe": -3.2656178146846373,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_all_features_h03_SOXL.csv",
        "agent": "dprime_all_features_h03",
        "test_days": 63,
        "equity_multiple": 0.36275948977956096,
        "max_dd": -0.6520829650961262,
        "mean_ret": -0.013936419455733675,
        "std_ret": 0.06731139680149525,
        "sharpe": -3.2867212803761956,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_bnf_3scale_SOXL.csv",
        "agent": "dprime_bnf_3scale",
        "test_days": 63,
        "equity_multiple": 0.5793084955040091,
        "max_dd": -0.4392300039136301,
        "mean_ret": -0.006942271946750154,
        "std_ret": 0.06175831360237784,
        "sharpe": -1.7844585481855473,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_bnf_h01_SOXL.csv",
        "agent": "dprime_bnf_h01",
        "test_days": 63,
        "equity_multiple": 0.6032593282613353,
        "max_dd": -0.5177777954568705,
        "mean_ret": -0.005848607178666583,
        "std_ret": 0.06910558998139851,
        "sharpe": -1.34350579584292,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_bnf_h02_SOXL.csv",
        "agent": "dprime_bnf_h02",
        "test_days": 63,
        "equity_multiple": 0.38520094121937054,
        "max_dd": -0.6184841887902549,
        "mean_ret": -0.013059742698845044,
        "std_ret": 0.06672022343750657,
        "sharpe": -3.1072586020338524,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_bnf_h03_SOXL.csv",
        "agent": "dprime_bnf_h03",
        "test_days": 63,
        "equity_multiple": 0.49738301791408773,
        "max_dd": -0.5026169820859123,
        "mean_ret": -0.009424610929157112,
        "std_ret": 0.06030545018214006,
        "sharpe": -2.480887877972514,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_mix_3scale_SOXL.csv",
        "agent": "dprime_mix_3scale",
        "test_days": 63,
        "equity_multiple": 0.36307797095682487,
        "max_dd": -0.6369220290431752,
        "mean_ret": -0.013900946553645,
        "std_ret": 0.06806124818731625,
        "sharpe": -3.2422368277576505,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_mix_h01_SOXL.csv",
        "agent": "dprime_mix_h01",
        "test_days": 63,
        "equity_multiple": 0.37421307393852904,
        "max_dd": -0.625786926061471,
        "mean_ret": -0.013365698923792914,
        "std_ret": 0.06895411789124055,
        "sharpe": -3.07703004830218,
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
        "equity_multiple": 0.4812070979915114,
        "max_dd": -0.5216495685579972,
        "mean_ret": -0.01022957779637109,
        "std_ret": 0.05522166001529234,
        "sharpe": -2.940685107136335,
        "status": "OK",
        "note": "Split missing: evaluated all rows as test"
      }
    ]
  },
  "diversity": {
    "status": "OK",
    "summary": "agent positions look diverse",
    "max_corr": 0.9936332224914407,
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
      "StepF_vs_best_StepE: StepF(0.4812) は best StepE(0.7992) に負けてる"
    ]
  }
}
```

Best-effort mode: this evaluator writes SKIP/notes and always exits 0.