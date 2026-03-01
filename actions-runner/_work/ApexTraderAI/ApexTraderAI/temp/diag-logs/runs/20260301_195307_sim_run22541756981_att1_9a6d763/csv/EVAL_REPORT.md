# EVAL_REPORT

- output_root: `C:\work\apex_work\runs\gh22541756981_att1_sim_20260301_194521_9a6d763\output`
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
| stepB_pred_close_mamba_SOXL.csv | Pred_Close_MAMBA | 0.9384 | 1.0000 | 4.9469 | 0.8825 | OK |
| stepB_pred_time_all_SOXL.csv | Pred_Close_MAMBA | 0.9384 | 1.0000 | 4.9469 | 0.8825 | OK |

## StepE table
- SKIP: stepE_daily_log missing

## StepF table
- SKIP: stepF_equity_marl missing

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
  - note: PLOT StepE topN: stepE_daily_log files were not found
  - note: PLOT StepF equity: stepF_equity_marl file not found
  - note: PLOT StepE return bar: no numeric StepE rows
  - note: PLOT StepE scatter: no numeric StepE DD/return pairs
  - note: StepF_vs_best_StepE: 比較に必要な equity_multiple が不足

## Raw JSON
```json
{
  "output_root": "C:\\work\\apex_work\\runs\\gh22541756981_att1_sim_20260301_194521_9a6d763\\output",
  "mode": "sim",
  "symbol": "SOXL",
  "stepA": {
    "status": "OK",
    "summary": "prices_test evaluated",
    "details": {
      "path": "C:\\work\\apex_work\\runs\\gh22541756981_att1_sim_20260301_194521_9a6d763\\output\\stepA\\sim\\stepA_prices_test_SOXL.csv",
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
        "mae": 4.946948369344075,
        "corr": 0.8825267941155552,
        "status": "OK"
      },
      {
        "file": "stepB_pred_time_all_SOXL.csv",
        "pred_col": "Pred_Close_MAMBA",
        "non_null_ratio": 0.9383726528647087,
        "first_valid_date": "2014-07-09",
        "coverage_ratio_over_test": 1.0,
        "mae": 4.946948369344075,
        "corr": 0.8825267941155552,
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
        "C:\\work\\apex_work\\runs\\gh22541756981_att1_sim_20260301_194521_9a6d763\\output\\stepD_prime\\sim\\stepDprime_state_*_SOXL_train.csv",
        "C:\\work\\apex_work\\runs\\gh22541756981_att1_sim_20260301_194521_9a6d763\\output\\stepD_prime\\sim\\stepDprime_state_*_SOXL_test.csv",
        "C:\\work\\apex_work\\runs\\gh22541756981_att1_sim_20260301_194521_9a6d763\\output\\stepD_prime\\sim\\embeddings\\stepDprime_*_SOXL_embeddings*.csv"
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
      "PLOT StepE topN: stepE_daily_log files were not found",
      "PLOT StepF equity: stepF_equity_marl file not found",
      "PLOT StepE return bar: no numeric StepE rows",
      "PLOT StepE scatter: no numeric StepE DD/return pairs",
      "StepF_vs_best_StepE: 比較に必要な equity_multiple が不足"
    ]
  }
}
```

Best-effort mode: this evaluator writes SKIP/notes and always exits 0.