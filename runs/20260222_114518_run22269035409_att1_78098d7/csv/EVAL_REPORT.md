# EVAL_REPORT

- output_root: `C:\work\apex_work\runs\20260222_114212\output`
- mode: `sim`
- symbol: `SOXL`
- overall_status: **BAD**

## Raw JSON
```json
{
  "output_root": "C:\\work\\apex_work\\runs\\20260222_114212\\output",
  "mode": "sim",
  "symbol": "SOXL",
  "stepA": {
    "status": "OK",
    "summary": "prices_test evaluated",
    "details": {
      "path": "C:\\work\\apex_work\\runs\\20260222_114212\\output\\stepA\\sim\\stepA_prices_test_SOXL.csv",
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
    "status": "BAD",
    "summary": "stepB files evaluated",
    "rows": [
      {
        "file": "stepB_pred_close_mamba_SOXL.csv",
        "pred_col": "Pred_Close_MAMBA",
        "non_null_ratio": 0.9710982658959537,
        "first_valid_date": "2014-04-01",
        "coverage_ratio_over_test": 1.0,
        "mae": 4.360174086786086,
        "corr": 0.8982551856091188,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Pred_Close_MAMBA_PERIODIC_h20",
        "non_null_ratio": 0.9619460500963392,
        "first_valid_date": "2014-04-29",
        "coverage_ratio_over_test": 1.0,
        "mae": 11.614169846272814,
        "corr": 0.6217234630474424,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Delta_Close_pred_MAMBA_PERIODIC_h20",
        "non_null_ratio": 0.9619460500963392,
        "first_valid_date": "2014-04-29",
        "coverage_ratio_over_test": 1.0,
        "mae": 42.26501597980896,
        "corr": 0.08649802468581684,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Pred_Close_MAMBA_PERIODIC",
        "non_null_ratio": 0.9619460500963392,
        "first_valid_date": "2014-04-29",
        "coverage_ratio_over_test": 1.0,
        "mae": 11.614169846272814,
        "corr": 0.6217234630474424,
        "status": "OK"
      },
      {
        "file": "stepB_pred_close_mamba_periodic_SOXL.csv",
        "pred_col": "Delta_Close_pred_MAMBA_PERIODIC",
        "non_null_ratio": 0.9619460500963392,
        "first_valid_date": "2014-04-29",
        "coverage_ratio_over_test": 1.0,
        "mae": 42.26501597980896,
        "corr": 0.08649802468581684,
        "status": "OK"
      },
      {
        "file": "stepB_pred_path_mamba_SOXL.csv",
        "pred_col": "Pred_Close_t_plus_01",
        "non_null_ratio": 1.0,
        "first_valid_date": "2014-03-31",
        "coverage_ratio_over_test": 0.9838709677419355,
        "mae": 3.469225179953653,
        "corr": 0.936086436094378,
        "status": "OK"
      },
      {
        "file": "stepB_pred_path_mamba_SOXL.csv",
        "pred_col": "Pred_Close_t_plus_05",
        "non_null_ratio": 1.0,
        "first_valid_date": "2014-03-31",
        "coverage_ratio_over_test": 0.9838709677419355,
        "mae": 5.820147373637215,
        "corr": 0.8767652944263808,
        "status": "OK"
      },
      {
        "file": "stepB_pred_path_mamba_SOXL.csv",
        "pred_col": "Pred_Close_t_plus_10",
        "non_null_ratio": 1.0,
        "first_valid_date": "2014-03-31",
        "coverage_ratio_over_test": 0.9838709677419355,
        "mae": 12.262606542618549,
        "corr": 0.803235679015604,
        "status": "OK"
      },
      {
        "file": "stepB_pred_path_mamba_SOXL.csv",
        "pred_col": "Pred_Close_t_plus_20",
        "non_null_ratio": 1.0,
        "first_valid_date": "2014-03-31",
        "coverage_ratio_over_test": 0.9838709677419355,
        "mae": 5.912770849759461,
        "corr": 0.8369182046066634,
        "status": "OK"
      },
      {
        "file": "stepB_pred_path_mamba_periodic_SOXL.csv",
        "pred_col": "Pred_Close_t_plus_20",
        "non_null_ratio": 1.0,
        "first_valid_date": "2014-03-31",
        "coverage_ratio_over_test": 0.9838709677419355,
        "mae": 11.030998483063554,
        "corr": 0.6141074378249171,
        "status": "OK"
      },
      {
        "file": "stepB_pred_time_all_SOXL.csv",
        "pred_col": "Pred_Close_XSR",
        "non_null_ratio": 0.0,
        "first_valid_date": null,
        "coverage_ratio_over_test": 0.0,
        "mae": null,
        "corr": null,
        "status": "BAD"
      },
      {
        "file": "stepB_pred_time_all_SOXL.csv",
        "pred_col": "Pred_Close_MAMBA",
        "non_null_ratio": 0.9710982658959537,
        "first_valid_date": "2014-04-01",
        "coverage_ratio_over_test": 1.0,
        "mae": 4.360174086786086,
        "corr": 0.8982551856091188,
        "status": "OK"
      },
      {
        "file": "stepB_pred_time_all_SOXL.csv",
        "pred_col": "Pred_Close_FED",
        "non_null_ratio": 0.0,
        "first_valid_date": null,
        "coverage_ratio_over_test": 0.0,
        "mae": null,
        "corr": null,
        "status": "BAD"
      }
    ],
    "files": [
      "stepB_pred_close_mamba_SOXL.csv",
      "stepB_pred_close_mamba_periodic_SOXL.csv",
      "stepB_pred_path_mamba_SOXL.csv",
      "stepB_pred_path_mamba_periodic_SOXL.csv",
      "stepB_pred_time_all_SOXL.csv"
    ]
  },
  "stepE": {
    "status": "OK",
    "summary": "stepE daily logs evaluated",
    "rows": [
      {
        "file": "stepE_daily_log_mamba_SOXL.csv",
        "test_days": 2076,
        "equity_multiple": null,
        "max_dd": null,
        "sharpe": null,
        "mean_reward": null,
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
        "total_return": null,
        "max_drawdown": null,
        "sharpe": null,
        "status": "OK"
      }
    ]
  },
  "overall_status": "BAD"
}
```

Best-effort mode: this evaluator writes SKIP/notes and always exits 0.