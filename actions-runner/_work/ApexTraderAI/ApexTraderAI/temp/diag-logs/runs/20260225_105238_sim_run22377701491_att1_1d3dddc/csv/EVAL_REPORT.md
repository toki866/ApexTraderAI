# EVAL_REPORT

- output_root: `C:\work\apex_work\runs\20260225_101918\output`
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
| stepB_pred_close_mamba_SOXL.csv | Pred_Close_MAMBA | 0.9711 | 1.0000 | 3.6384 | 0.9239 | OK |
| stepB_pred_time_all_SOXL.csv | Pred_Close_MAMBA | 0.9711 | 1.0000 | 3.6384 | 0.9239 | OK |

## StepE table
| agent | file | test_days | equity_multiple | max_dd | mean_ret | std_ret | sharpe | note | status |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| dprime_all_features_3scale | stepE_daily_log_dprime_all_features_3scale_SOXL.csv | 63 | 0.5507 | -0.5683 | -0.0070 | 0.0720 | -1.5334 |  | OK |
| dprime_all_features_h01 | stepE_daily_log_dprime_all_features_h01_SOXL.csv | 63 | 0.7788 | -0.3171 | -0.0034 | 0.0363 | -1.4955 |  | OK |
| dprime_all_features_h02 | stepE_daily_log_dprime_all_features_h02_SOXL.csv | 63 | 0.3966 | -0.6057 | -0.0129 | 0.0628 | -3.2496 |  | OK |
| dprime_all_features_h03 | stepE_daily_log_dprime_all_features_h03_SOXL.csv | 63 | 0.4442 | -0.5586 | -0.0112 | 0.0596 | -2.9714 |  | OK |
| dprime_bnf_3scale | stepE_daily_log_dprime_bnf_3scale_SOXL.csv | 63 | 0.5821 | -0.4389 | -0.0069 | 0.0605 | -1.8219 |  | OK |
| dprime_bnf_h01 | stepE_daily_log_dprime_bnf_h01_SOXL.csv | 63 | 0.7714 | -0.3531 | -0.0033 | 0.0425 | -1.2492 |  | OK |
| dprime_bnf_h02 | stepE_daily_log_dprime_bnf_h02_SOXL.csv | 63 | 0.4980 | -0.5020 | -0.0095 | 0.0590 | -2.5484 |  | OK |
| dprime_bnf_h03 | stepE_daily_log_dprime_bnf_h03_SOXL.csv | 63 | 0.4238 | -0.5762 | -0.0119 | 0.0611 | -3.0957 |  | OK |
| dprime_mix_3scale | stepE_daily_log_dprime_mix_3scale_SOXL.csv | 63 | 0.4029 | -0.5971 | -0.0124 | 0.0667 | -2.9413 |  | OK |
| dprime_mix_h01 | stepE_daily_log_dprime_mix_h01_SOXL.csv | 63 | 0.4013 | -0.5987 | -0.0124 | 0.0667 | -2.9570 |  | OK |

## StepF table
| file | test_days | equity_multiple | max_dd | mean_ret | std_ret | sharpe | note | status |
|---|---:|---:|---:|---:|---:|---:|---|---|
| stepF_equity_marl_SOXL.csv | 63 | 0.5105 | -0.4895 | -0.0095 | 0.0517 | -2.9084 | Split missing: evaluated all rows as test | OK |

## Diversity
- status: **OK**
- summary: agent positions look diverse
- max_corr: 0.9905
- max_match_ratio: 0.0000
- pairs_over_0_9999: 0 / 45
- identical_all_agents: False

## PLOTS
- [equity_stepE_topN.png](./equity_stepE_topN.png)
- [equity_stepF.png](./equity_stepF.png)
- [bar_stepE_return.png](./bar_stepE_return.png)
- [scatter_stepE_dd_vs_ret.png](./scatter_stepE_dd_vs_ret.png)
  - note: StepF_vs_best_StepE: StepF(0.5105) は best StepE(0.7788) に負けてる

## Raw JSON
```json
{
  "output_root": "C:\\work\\apex_work\\runs\\20260225_101918\\output",
  "mode": "sim",
  "symbol": "SOXL",
  "stepA": {
    "status": "OK",
    "summary": "prices_test evaluated",
    "details": {
      "path": "C:\\work\\apex_work\\runs\\20260225_101918\\output\\stepA\\sim\\stepA_prices_test_SOXL.csv",
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
        "mae": 3.638424555460612,
        "corr": 0.9239117485706505,
        "status": "OK"
      },
      {
        "file": "stepB_pred_time_all_SOXL.csv",
        "pred_col": "Pred_Close_MAMBA",
        "non_null_ratio": 0.9711121810303323,
        "first_valid_date": "2014-04-01",
        "coverage_ratio_over_test": 1.0,
        "mae": 3.638424555460612,
        "corr": 0.9239117485706505,
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
        "C:\\work\\apex_work\\runs\\20260225_101918\\output\\stepD_prime\\sim\\stepDprime_state_*_SOXL_train.csv",
        "C:\\work\\apex_work\\runs\\20260225_101918\\output\\stepD_prime\\sim\\stepDprime_state_*_SOXL_test.csv",
        "C:\\work\\apex_work\\runs\\20260225_101918\\output\\stepD_prime\\sim\\embeddings\\stepDprime_*_SOXL_embeddings*.csv"
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
        "equity_multiple": 0.5507461881332713,
        "max_dd": -0.5683415148167794,
        "mean_ret": -0.006955689262545161,
        "std_ret": 0.07200666587020399,
        "sharpe": -1.5334433636110214,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_all_features_h01_SOXL.csv",
        "agent": "dprime_all_features_h01",
        "test_days": 63,
        "equity_multiple": 0.7787866823316999,
        "max_dd": -0.31708275052785395,
        "mean_ret": -0.0034162532360318804,
        "std_ret": 0.03626341867106053,
        "sharpe": -1.495483350890976,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_all_features_h02_SOXL.csv",
        "agent": "dprime_all_features_h02",
        "test_days": 63,
        "equity_multiple": 0.39664205074192377,
        "max_dd": -0.6057353914471546,
        "mean_ret": -0.012852183478470464,
        "std_ret": 0.06278460798038848,
        "sharpe": -3.2495558114014846,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_all_features_h03_SOXL.csv",
        "agent": "dprime_all_features_h03",
        "test_days": 63,
        "equity_multiple": 0.4441571592448994,
        "max_dd": -0.5585601777644225,
        "mean_ret": -0.011150848369912371,
        "std_ret": 0.05957167353004932,
        "sharpe": -2.971449678609056,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_bnf_3scale_SOXL.csv",
        "agent": "dprime_bnf_3scale",
        "test_days": 63,
        "equity_multiple": 0.5821081645108741,
        "max_dd": -0.43890358295181076,
        "mean_ret": -0.006941763884011412,
        "std_ret": 0.06048579964170913,
        "sharpe": -1.8218670503837668,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_bnf_h01_SOXL.csv",
        "agent": "dprime_bnf_h01",
        "test_days": 63,
        "equity_multiple": 0.7714008108731064,
        "max_dd": -0.3531278578617606,
        "mean_ret": -0.0033459524513870876,
        "std_ret": 0.042521047722798334,
        "sharpe": -1.2491542742871873,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_bnf_h02_SOXL.csv",
        "agent": "dprime_bnf_h02",
        "test_days": 63,
        "equity_multiple": 0.4980117283624477,
        "max_dd": -0.5019882716375523,
        "mean_ret": -0.009475552536998055,
        "std_ret": 0.05902612901524449,
        "sharpe": -2.5483584269619923,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_bnf_h03_SOXL.csv",
        "agent": "dprime_bnf_h03",
        "test_days": 63,
        "equity_multiple": 0.42379995993048863,
        "max_dd": -0.5762000400695113,
        "mean_ret": -0.011908164240017022,
        "std_ret": 0.06106328877537289,
        "sharpe": -3.0957429691966456,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_mix_3scale_SOXL.csv",
        "agent": "dprime_mix_3scale",
        "test_days": 63,
        "equity_multiple": 0.40291341250603385,
        "max_dd": -0.5970865874939661,
        "mean_ret": -0.012363292068512811,
        "std_ret": 0.06672656089683843,
        "sharpe": -2.9412751767542753,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_mix_h01_SOXL.csv",
        "agent": "dprime_mix_h01",
        "test_days": 63,
        "equity_multiple": 0.4013485634177499,
        "max_dd": -0.5986514365822502,
        "mean_ret": -0.012419926526952552,
        "std_ret": 0.06667549934721431,
        "sharpe": -2.9570115452056367,
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
        "equity_multiple": 0.5104937252291846,
        "max_dd": -0.48950627477081543,
        "mean_ret": -0.009476966648695176,
        "std_ret": 0.051727075396733435,
        "sharpe": -2.9083836745137996,
        "status": "OK",
        "note": "Split missing: evaluated all rows as test"
      }
    ]
  },
  "diversity": {
    "status": "OK",
    "summary": "agent positions look diverse",
    "max_corr": 0.9905489866698542,
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
      "StepF_vs_best_StepE: StepF(0.5105) は best StepE(0.7788) に負けてる"
    ]
  }
}
```

Best-effort mode: this evaluator writes SKIP/notes and always exits 0.