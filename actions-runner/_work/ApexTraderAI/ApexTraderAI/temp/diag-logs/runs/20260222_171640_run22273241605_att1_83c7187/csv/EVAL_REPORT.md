# EVAL_REPORT

- output_root: `C:\work\apex_work\runs\20260222_170025\output`
- mode: `sim`
- symbol: `SOXL`
- overall_status: **OK**

## StepE agents table
| agent | file | test_days | equity_multiple | max_drawdown | sharpe | note | status |
|---|---|---:|---:|---:|---:|---|---|
| dprime_all_features_3scale | stepE_daily_log_dprime_all_features_3scale_SOXL.csv | 62 | 1.1086 | -0.2982 | 0.9609 |  | OK |
| dprime_all_features_h01 | stepE_daily_log_dprime_all_features_h01_SOXL.csv | 62 | 1.1086 | -0.2982 | 0.9609 |  | OK |
| dprime_all_features_h02 | stepE_daily_log_dprime_all_features_h02_SOXL.csv | 62 | 1.1086 | -0.2982 | 0.9609 |  | OK |
| dprime_all_features_h03 | stepE_daily_log_dprime_all_features_h03_SOXL.csv | 62 | 1.1086 | -0.2982 | 0.9609 |  | OK |
| dprime_bnf_3scale | stepE_daily_log_dprime_bnf_3scale_SOXL.csv | 62 | 1.1086 | -0.2982 | 0.9609 |  | OK |
| dprime_bnf_h01 | stepE_daily_log_dprime_bnf_h01_SOXL.csv | 62 | 1.1086 | -0.2982 | 0.9609 |  | OK |
| dprime_bnf_h02 | stepE_daily_log_dprime_bnf_h02_SOXL.csv | 62 | 1.1086 | -0.2982 | 0.9609 |  | OK |
| dprime_bnf_h03 | stepE_daily_log_dprime_bnf_h03_SOXL.csv | 62 | 1.1086 | -0.2982 | 0.9609 |  | OK |
| dprime_mix_3scale | stepE_daily_log_dprime_mix_3scale_SOXL.csv | 62 | 1.1086 | -0.2982 | 0.9609 |  | OK |
| dprime_mix_h01 | stepE_daily_log_dprime_mix_h01_SOXL.csv | 62 | 1.1086 | -0.2982 | 0.9609 |  | OK |

## StepF router summary
| file | test_days | equity_multiple | max_drawdown | sharpe | note | status |
|---|---:|---:|---:|---:|---|---|
| stepF_equity_marl_SOXL.csv | 62 | 1.1086 | -0.2982 | 0.9609 | Split missing: evaluated all rows as test | OK |

## Raw JSON
```json
{
  "output_root": "C:\\work\\apex_work\\runs\\20260222_170025\\output",
  "mode": "sim",
  "symbol": "SOXL",
  "stepA": {
    "status": "OK",
    "summary": "prices_test evaluated",
    "details": {
      "path": "C:\\work\\apex_work\\runs\\20260222_170025\\output\\stepA\\sim\\stepA_prices_test_SOXL.csv",
      "test_rows": 62,
      "test_date_start": "2022-01-03",
      "test_date_end": "2022-03-31",
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
        "non_null_ratio": 0.9710982658959537,
        "first_valid_date": "2014-04-01",
        "coverage_ratio_over_test": 1.0,
        "mae": 4.506098778017106,
        "corr": 0.8972053476265942,
        "status": "OK"
      },
      {
        "file": "stepB_pred_time_all_SOXL.csv",
        "pred_col": "Pred_Close_MAMBA",
        "non_null_ratio": 0.9710982658959537,
        "first_valid_date": "2014-04-01",
        "coverage_ratio_over_test": 1.0,
        "mae": 4.506098778017106,
        "corr": 0.8972053476265942,
        "status": "OK"
      }
    ],
    "files": [
      "stepB_pred_close_mamba_SOXL.csv",
      "stepB_pred_path_mamba_SOXL.csv",
      "stepB_pred_time_all_SOXL.csv"
    ]
  },
  "stepE": {
    "status": "OK",
    "summary": "stepE daily logs evaluated",
    "rows": [
      {
        "file": "stepE_daily_log_dprime_all_features_3scale_SOXL.csv",
        "agent": "dprime_all_features_3scale",
        "test_days": 62,
        "equity_multiple": 1.1086063460568816,
        "max_drawdown": -0.2981575764173262,
        "sharpe": 0.960933731678113,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_all_features_h01_SOXL.csv",
        "agent": "dprime_all_features_h01",
        "test_days": 62,
        "equity_multiple": 1.1086063460568816,
        "max_drawdown": -0.2981575764173262,
        "sharpe": 0.960933731678113,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_all_features_h02_SOXL.csv",
        "agent": "dprime_all_features_h02",
        "test_days": 62,
        "equity_multiple": 1.1086063460568816,
        "max_drawdown": -0.2981575764173262,
        "sharpe": 0.960933731678113,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_all_features_h03_SOXL.csv",
        "agent": "dprime_all_features_h03",
        "test_days": 62,
        "equity_multiple": 1.1086063460568816,
        "max_drawdown": -0.2981575764173262,
        "sharpe": 0.960933731678113,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_bnf_3scale_SOXL.csv",
        "agent": "dprime_bnf_3scale",
        "test_days": 62,
        "equity_multiple": 1.1086063460568816,
        "max_drawdown": -0.2981575764173262,
        "sharpe": 0.960933731678113,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_bnf_h01_SOXL.csv",
        "agent": "dprime_bnf_h01",
        "test_days": 62,
        "equity_multiple": 1.1086063460568816,
        "max_drawdown": -0.2981575764173262,
        "sharpe": 0.960933731678113,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_bnf_h02_SOXL.csv",
        "agent": "dprime_bnf_h02",
        "test_days": 62,
        "equity_multiple": 1.1086063460568816,
        "max_drawdown": -0.2981575764173262,
        "sharpe": 0.960933731678113,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_bnf_h03_SOXL.csv",
        "agent": "dprime_bnf_h03",
        "test_days": 62,
        "equity_multiple": 1.1086063460568816,
        "max_drawdown": -0.2981575764173262,
        "sharpe": 0.960933731678113,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_mix_3scale_SOXL.csv",
        "agent": "dprime_mix_3scale",
        "test_days": 62,
        "equity_multiple": 1.1086063460568816,
        "max_drawdown": -0.2981575764173262,
        "sharpe": 0.960933731678113,
        "status": "OK"
      },
      {
        "file": "stepE_daily_log_dprime_mix_h01_SOXL.csv",
        "agent": "dprime_mix_h01",
        "test_days": 62,
        "equity_multiple": 1.1086063460568816,
        "max_drawdown": -0.2981575764173262,
        "sharpe": 0.960933731678113,
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
        "test_days": 62,
        "equity_multiple": 1.1086063575345915,
        "max_drawdown": -0.2981575680835453,
        "sharpe": 0.9609337765048558,
        "status": "OK",
        "note": "Split missing: evaluated all rows as test"
      }
    ]
  },
  "overall_status": "OK"
}
```

Best-effort mode: this evaluator writes SKIP/notes and always exits 0.