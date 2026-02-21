# EVAL_REPORT

- output_root: `C:\work\apex_work\runs\20260221_194910\output`
- mode: `sim`
- symbol: `SOXL`

## StepA summary
- status: **OK**
- summary: prices_test evaluated
- prices_test:
  - rows: 62
  - cols: 6
  - date_start: 2022-01-03
  - date_end: 2022-03-31
  - date_monotonic_increasing: True
  - date_duplicates: 0
  - ohlcv_missing:
    - Open: 0
    - High: 0
    - Low: 0
    - Close: 0
    - Volume: 0
- split_summary_path: C:\work\apex_work\runs\20260221_194910\output\stepA\sim\stepA_split_summary_SOXL.csv
- split_summary: {"mode": "sim", "train_start": "2014-01-03", "train_end": "2022-01-02", "test_start": "2022-01-03", "test_end": "2022-03-31"}

## StepB file-by-file
- status: **OK**
- summary: stepB files evaluated

| file | pred_col | rows | non_null_ratio | first_valid | last_valid | coverage_ratio | MAE | corr | dir_acc | status |
|---|---|---:|---:|---|---|---:|---:|---:|---:|---|
| stepB_pred_close_mamba_SOXL.csv | Pred_Close_MAMBA_h01 | 2076 | 0.9710982658959537 | 2014-04-01 | 2022-03-31 | 1.0 | 4.360174086786086 | 0.8982551856091188 | 0.36065573770491804 | OK |
| stepB_pred_close_mamba_SOXL.csv | Delta_Close_pred_MAMBA_h01 | 2076 | 0.9710982658959537 | 2014-04-01 | 2022-03-31 | 1.0 | None | None | None | OK |
| stepB_pred_close_mamba_SOXL.csv | Pred_Close_MAMBA_h05 | 2076 | 0.9691714836223507 | 2014-04-07 | 2022-03-31 | 1.0 | 8.69324274986021 | 0.7801136667555836 | 0.5737704918032787 | OK |
| stepB_pred_close_mamba_SOXL.csv | Delta_Close_pred_MAMBA_h05 | 2076 | 0.9691714836223507 | 2014-04-07 | 2022-03-31 | 1.0 | None | None | None | OK |
| stepB_pred_close_mamba_SOXL.csv | Pred_Close_MAMBA_h10 | 2076 | 0.9667630057803468 | 2014-04-14 | 2022-03-31 | 1.0 | 15.890213658732753 | 0.6203018887247069 | 0.5409836065573771 | OK |
| stepB_pred_close_mamba_SOXL.csv | Delta_Close_pred_MAMBA_h10 | 2076 | 0.9667630057803468 | 2014-04-14 | 2022-03-31 | 1.0 | None | None | None | OK |
| stepB_pred_close_mamba_SOXL.csv | Pred_Close_MAMBA_h20 | 2076 | 0.9619460500963392 | 2014-04-29 | 2022-03-31 | 1.0 | 14.89798641204834 | 0.5912410126691628 | 0.5737704918032787 | OK |
| stepB_pred_close_mamba_SOXL.csv | Delta_Close_pred_MAMBA_h20 | 2076 | 0.9619460500963392 | 2014-04-29 | 2022-03-31 | 1.0 | None | None | None | OK |
| stepB_pred_close_mamba_SOXL.csv | Pred_Close_MAMBA | 2076 | 0.9710982658959537 | 2014-04-01 | 2022-03-31 | 1.0 | 4.360174086786086 | 0.8982551856091188 | 0.36065573770491804 | OK |
| stepB_pred_close_mamba_SOXL.csv | Delta_Close_pred_MAMBA | 2076 | 0.9710982658959537 | 2014-04-01 | 2022-03-31 | 1.0 | None | None | None | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Pred_Close_MAMBA_PERIODIC_h20 | 2076 | 0.9619460500963392 | 2014-04-29 | 2022-03-31 | 1.0 | 11.614169846272814 | 0.6217234630474424 | 0.5081967213114754 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Delta_Close_pred_MAMBA_PERIODIC_h20 | 2076 | 0.9619460500963392 | 2014-04-29 | 2022-03-31 | 1.0 | None | None | None | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Pred_Close_MAMBA_PERIODIC | 2076 | 0.9619460500963392 | 2014-04-29 | 2022-03-31 | 1.0 | 11.614169846272814 | 0.6217234630474424 | 0.5081967213114754 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Delta_Close_pred_MAMBA_PERIODIC | 2076 | 0.9619460500963392 | 2014-04-29 | 2022-03-31 | 1.0 | None | None | None | OK |
| stepB_pred_path_mamba_SOXL.csv | Pred_Close_t_plus_01 | 2016 | 1.0 | 2014-03-31 | 2022-03-30 | 0.9838709677419355 | 3.469225179953653 | 0.936086436094378 | 0.6666666666666666 | OK |
| stepB_pred_path_mamba_SOXL.csv | Pred_Close_t_plus_05 | 2016 | 1.0 | 2014-03-31 | 2022-03-30 | 0.9838709677419355 | 5.820147373637215 | 0.8767652944263808 | 0.5666666666666667 | OK |
| stepB_pred_path_mamba_SOXL.csv | Pred_Close_t_plus_10 | 2016 | 1.0 | 2014-03-31 | 2022-03-30 | 0.9838709677419355 | 12.262606542618549 | 0.803235679015604 | 0.5166666666666667 | OK |
| stepB_pred_path_mamba_SOXL.csv | Pred_Close_t_plus_20 | 2016 | 1.0 | 2014-03-31 | 2022-03-30 | 0.9838709677419355 | 5.912770849759461 | 0.8369182046066634 | 0.48333333333333334 | OK |
| stepB_pred_path_mamba_periodic_SOXL.csv | Pred_Close_t_plus_20 | 2016 | 1.0 | 2014-03-31 | 2022-03-30 | 0.9838709677419355 | 11.030998483063554 | 0.6141074378249171 | 0.75 | OK |
| stepB_pred_time_all_SOXL.csv | Pred_Close_XSR | 2076 | 0.0 | None | None | 0.0 | None | None | None | BAD |
| stepB_pred_time_all_SOXL.csv | Pred_Close_MAMBA | 2076 | 0.9710982658959537 | 2014-04-01 | 2022-03-31 | 1.0 | 4.360174086786086 | 0.8982551856091188 | 0.36065573770491804 | OK |
| stepB_pred_time_all_SOXL.csv | Pred_Close_FED | 2076 | 0.0 | None | None | 0.0 | None | None | None | BAD |

## StepE daily logs
- status: **SKIP**
- summary: no stepE daily logs found

Status rule: OK if non_null_ratio>=0.90 and coverage_ratio>=0.90; WARN otherwise; BAD if non_null_ratio<0.50 or pred cols missing.
Best-effort mode: this evaluator writes SKIP/notes and always exits 0.
