# EVAL_REPORT

- output_root: `C:\work\apex_work\runs\20260222_095330\output`
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
- split_summary_path: C:\work\apex_work\runs\20260222_095330\output\stepA\sim\stepA_split_summary_SOXL.csv
- split_summary: {"mode": "sim", "train_start": "2017-01-03", "train_end": "2022-01-02", "test_start": "2022-01-03", "test_end": "2022-03-31"}

## StepB file-by-file
- status: **OK**
- summary: stepB files evaluated

| file | pred_col | rows | non_null_ratio | first_valid | last_valid | coverage_ratio | MAE | corr | dir_acc | status |
|---|---|---:|---:|---|---|---:|---:|---:|---:|---|
| stepB_pred_close_mamba_SOXL.csv | Pred_Close_MAMBA_h01 | 1321 | 0.9545798637395913 | 2017-03-30 | 2022-03-31 | 1.0 | 4.047740874751922 | 0.9174252645460718 | 0.4262295081967213 | OK |
| stepB_pred_close_mamba_SOXL.csv | Delta_Close_pred_MAMBA_h01 | 1321 | 0.9545798637395913 | 2017-03-30 | 2022-03-31 | 1.0 | None | None | None | OK |
| stepB_pred_close_mamba_SOXL.csv | Pred_Close_MAMBA_h05 | 1321 | 0.9515518546555639 | 2017-04-05 | 2022-03-31 | 1.0 | 10.551315153798749 | 0.7609621894366038 | 0.5081967213114754 | OK |
| stepB_pred_close_mamba_SOXL.csv | Delta_Close_pred_MAMBA_h05 | 1321 | 0.9515518546555639 | 2017-04-05 | 2022-03-31 | 1.0 | None | None | None | OK |
| stepB_pred_close_mamba_SOXL.csv | Pred_Close_MAMBA_h10 | 1321 | 0.9477668433005298 | 2017-04-12 | 2022-03-31 | 1.0 | 8.933659922692083 | 0.7481076301420394 | 0.5901639344262295 | OK |
| stepB_pred_close_mamba_SOXL.csv | Delta_Close_pred_MAMBA_h10 | 1321 | 0.9477668433005298 | 2017-04-12 | 2022-03-31 | 1.0 | None | None | None | OK |
| stepB_pred_close_mamba_SOXL.csv | Pred_Close_MAMBA_h20 | 1321 | 0.9401968205904617 | 2017-04-27 | 2022-03-31 | 1.0 | 13.137364510566957 | 0.6610654878851506 | 0.5081967213114754 | OK |
| stepB_pred_close_mamba_SOXL.csv | Delta_Close_pred_MAMBA_h20 | 1321 | 0.9401968205904617 | 2017-04-27 | 2022-03-31 | 1.0 | None | None | None | OK |
| stepB_pred_close_mamba_SOXL.csv | Pred_Close_MAMBA | 1321 | 0.9545798637395913 | 2017-03-30 | 2022-03-31 | 1.0 | 4.047740874751922 | 0.9174252645460718 | 0.4262295081967213 | OK |
| stepB_pred_close_mamba_SOXL.csv | Delta_Close_pred_MAMBA | 1321 | 0.9545798637395913 | 2017-03-30 | 2022-03-31 | 1.0 | None | None | None | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Pred_Close_MAMBA_PERIODIC_h20 | 1321 | 0.9401968205904617 | 2017-04-27 | 2022-03-31 | 1.0 | 20.947205318743716 | 0.4393351462911496 | 0.4426229508196721 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Delta_Close_pred_MAMBA_PERIODIC_h20 | 1321 | 0.9401968205904617 | 2017-04-27 | 2022-03-31 | 1.0 | None | None | None | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Pred_Close_MAMBA_PERIODIC | 1321 | 0.9401968205904617 | 2017-04-27 | 2022-03-31 | 1.0 | 20.947205318743716 | 0.4393351462911496 | 0.4426229508196721 | OK |
| stepB_pred_close_mamba_periodic_SOXL.csv | Delta_Close_pred_MAMBA_PERIODIC | 1321 | 0.9401968205904617 | 2017-04-27 | 2022-03-31 | 1.0 | None | None | None | OK |
| stepB_pred_path_mamba_SOXL.csv | Pred_Close_t_plus_01 | 1261 | 1.0 | 2017-03-29 | 2022-03-30 | 0.9838709677419355 | 2.9751618182072876 | 0.956632280215431 | 0.6666666666666666 | OK |
| stepB_pred_path_mamba_SOXL.csv | Pred_Close_t_plus_05 | 1261 | 1.0 | 2017-03-29 | 2022-03-30 | 0.9838709677419355 | 8.273618823192159 | 0.8473841599006254 | 0.7166666666666667 | OK |
| stepB_pred_path_mamba_SOXL.csv | Pred_Close_t_plus_10 | 1261 | 1.0 | 2017-03-29 | 2022-03-30 | 0.9838709677419355 | 6.00578980367692 | 0.9153863057706025 | 0.7333333333333333 | OK |
| stepB_pred_path_mamba_SOXL.csv | Pred_Close_t_plus_20 | 1261 | 1.0 | 2017-03-29 | 2022-03-30 | 0.9838709677419355 | 7.285471337740539 | 0.887117084933919 | 0.5666666666666667 | OK |
| stepB_pred_path_mamba_periodic_SOXL.csv | Pred_Close_t_plus_20 | 1261 | 1.0 | 2017-03-29 | 2022-03-30 | 0.9838709677419355 | 11.487521172928494 | 0.8537932637832665 | 0.7166666666666667 | OK |
| stepB_pred_time_all_SOXL.csv | Pred_Close_XSR | 1321 | 0.0 | None | None | 0.0 | None | None | None | BAD |
| stepB_pred_time_all_SOXL.csv | Pred_Close_MAMBA | 1321 | 0.9545798637395913 | 2017-03-30 | 2022-03-31 | 1.0 | 4.047740874751922 | 0.9174252645460718 | 0.4262295081967213 | OK |
| stepB_pred_time_all_SOXL.csv | Pred_Close_FED | 1321 | 0.0 | None | None | 0.0 | None | None | None | BAD |

## StepE daily logs
- status: **SKIP**
- summary: no stepE daily logs found

Status rule: OK if non_null_ratio>=0.90 and coverage_ratio>=0.90; WARN otherwise; BAD if non_null_ratio<0.50 or pred cols missing.
Best-effort mode: this evaluator writes SKIP/notes and always exits 0.
