# EVAL_REPORT

- output_root: `C:\work\apex_work\runs\20260222_203050\output`
- mode: `sim`
- symbol: `SOXL`
- overall_status: **WARN**

## StepE agents table
- SKIP: stepE_daily_log missing

## StepF router summary
- SKIP: stepF_equity_marl missing

## Diversity
- status: **SKIP**
- summary: not evaluated
- max_corr: NA
- max_match_ratio: NA
- pairs_over_0_9999: NA / NA
- identical_all_agents: NA

## Raw JSON
```json
{
  "output_root": "C:\\work\\apex_work\\runs\\20260222_203050\\output",
  "mode": "sim",
  "symbol": "SOXL",
  "stepA": {
    "status": "OK",
    "summary": "prices_test evaluated",
    "details": {
      "path": "C:\\work\\apex_work\\runs\\20260222_203050\\output\\stepA\\sim\\stepA_prices_test_SOXL.csv",
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
  "overall_status": "WARN"
}
```

Best-effort mode: this evaluator writes SKIP/notes and always exits 0.