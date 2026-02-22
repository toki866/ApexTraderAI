# EVAL_REPORT

- output_root: `C:\work\apex_work\runs\20260222_124318\output`
- mode: `sim`
- symbol: `SOXL`
- overall_status: **WARN**

## Raw JSON
```json
{
  "output_root": "C:\\work\\apex_work\\runs\\20260222_124318\\output",
  "mode": "sim",
  "symbol": "SOXL",
  "stepA": {
    "status": "OK",
    "summary": "prices_test evaluated",
    "details": {
      "path": "C:\\work\\apex_work\\runs\\20260222_124318\\output\\stepA\\sim\\stepA_prices_test_SOXL.csv",
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
    "status": "SKIP",
    "summary": "no stepB prediction files found",
    "rows": [],
    "files": []
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
  "overall_status": "WARN"
}
```

Best-effort mode: this evaluator writes SKIP/notes and always exits 0.