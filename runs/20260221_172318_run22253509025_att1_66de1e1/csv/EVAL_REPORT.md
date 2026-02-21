# Run Output Evaluation Report

- output_root: `C:\work\apex_work\runs\20260221_172139\output`
- mode: `sim`
- symbol: `SOXL`

## StepA Evaluation
**Status: OK** - evaluated stepA

| Metric | Value |
|---|---|
| test_rows | 62 |
| test_cols | 6 |
| test_start | 2022-01-03 |
| test_end | 2022-03-31 |
| date_monotonic_increasing | True |
| date_duplicates | 0 |
| date_missing_days | 26 |
| split_vs_test_alignment_ok | None |

OHLCV missing counts:
- Open: 0
- High: 0
- Low: 0
- Close: 0
- Volume: 0

## StepB Evaluation
**Status: OK** - evaluated stepB prediction files

### StepB file 1
- path: `C:\work\apex_work\runs\20260221_172139\output\stepB\sim\stepB_pred_close_mamba_SOXL.csv`
- rows/cols: 2076/11
- date_monotonic: True, duplicates: 0
- pred_cols: Pred_Close_MAMBA_h01, Delta_Close_pred_MAMBA_h01, Pred_Close_MAMBA_h05, Delta_Close_pred_MAMBA_h05, Pred_Close_MAMBA_h10, Delta_Close_pred_MAMBA_h10, Pred_Close_MAMBA_h20, Delta_Close_pred_MAMBA_h20, Pred_Close_MAMBA, Delta_Close_pred_MAMBA
- pred NaN:
  - Pred_Close_MAMBA_h01: 60
  - Delta_Close_pred_MAMBA_h01: 60
  - Pred_Close_MAMBA_h05: 64
  - Delta_Close_pred_MAMBA_h05: 64
  - Pred_Close_MAMBA_h10: 69
  - Delta_Close_pred_MAMBA_h10: 69
  - Pred_Close_MAMBA_h20: 79
  - Delta_Close_pred_MAMBA_h20: 79
  - Pred_Close_MAMBA: 60
  - Delta_Close_pred_MAMBA: 60

| pred_col | n | MAE | RMSE | corr | dir_acc |
|---|---:|---:|---:|---:|---:|
| Pred_Close_MAMBA_h01 | 62 | 4.360174086786086 | 5.285104386720852 | 0.8982551856091188 | 0.36065573770491804 |
| Delta_Close_pred_MAMBA_h01 | 62 | 45.423632221837195 | 46.61324584787508 | 0.162493660123449 | 0.5409836065573771 |
| Pred_Close_MAMBA_h05 | 62 | 8.69324274986021 | 11.458469346048437 | 0.7801136667555836 | 0.5737704918032787 |
| Delta_Close_pred_MAMBA_h05 | 62 | 38.61341710244456 | 40.554814126689145 | 0.19187144287030244 | 0.5737704918032787 |
| Pred_Close_MAMBA_h10 | 62 | 15.890213658732753 | 17.839399304128435 | 0.6203018887247069 | 0.5409836065573771 |
| Delta_Close_pred_MAMBA_h10 | 62 | 29.1265804536881 | 33.53755755394503 | -0.5763486433265307 | 0.5573770491803278 |
| Pred_Close_MAMBA_h20 | 62 | 14.89798641204834 | 17.220689323061368 | 0.5912410126691628 | 0.5737704918032787 |
| Delta_Close_pred_MAMBA_h20 | 62 | 30.41659121359548 | 34.62253173501152 | -0.429200153690823 | 0.5901639344262295 |
| Pred_Close_MAMBA | 62 | 4.360174086786086 | 5.285104386720852 | 0.8982551856091188 | 0.36065573770491804 |
| Delta_Close_pred_MAMBA | 62 | 45.423632221837195 | 46.61324584787508 | 0.162493660123449 | 0.5409836065573771 |

### StepB file 2
- path: `C:\work\apex_work\runs\20260221_172139\output\stepB\sim\stepB_pred_close_mamba_periodic_SOXL.csv`
- rows/cols: 2076/5
- date_monotonic: True, duplicates: 0
- pred_cols: Pred_Close_MAMBA_PERIODIC_h20, Delta_Close_pred_MAMBA_PERIODIC_h20, Pred_Close_MAMBA_PERIODIC, Delta_Close_pred_MAMBA_PERIODIC
- pred NaN:
  - Pred_Close_MAMBA_PERIODIC_h20: 79
  - Delta_Close_pred_MAMBA_PERIODIC_h20: 79
  - Pred_Close_MAMBA_PERIODIC: 79
  - Delta_Close_pred_MAMBA_PERIODIC: 79

| pred_col | n | MAE | RMSE | corr | dir_acc |
|---|---:|---:|---:|---:|---:|
| Pred_Close_MAMBA_PERIODIC_h20 | 62 | 11.614169846272814 | 14.461448288719007 | 0.6217234630474424 | 0.5081967213114754 |
| Delta_Close_pred_MAMBA_PERIODIC_h20 | 62 | 42.26501597980896 | 45.42036289650913 | 0.08649802468581684 | 0.6065573770491803 |
| Pred_Close_MAMBA_PERIODIC | 62 | 11.614169846272814 | 14.461448288719007 | 0.6217234630474424 | 0.5081967213114754 |
| Delta_Close_pred_MAMBA_PERIODIC | 62 | 42.26501597980896 | 45.42036289650913 | 0.08649802468581684 | 0.6065573770491803 |

### StepB file 3
- path: `C:\work\apex_work\runs\20260221_172139\output\stepB\sim\stepB_pred_path_mamba_SOXL.csv`
- rows/cols: 2016/5
- date_monotonic: False, duplicates: None
- pred_cols: Pred_Close_t_plus_01, Pred_Close_t_plus_05, Pred_Close_t_plus_10, Pred_Close_t_plus_20
- pred NaN:
  - Pred_Close_t_plus_01: 0
  - Pred_Close_t_plus_05: 0
  - Pred_Close_t_plus_10: 0
  - Pred_Close_t_plus_20: 0

| pred_col | n | MAE | RMSE | corr | dir_acc |
|---|---:|---:|---:|---:|---:|
| Pred_Close_t_plus_01 | 0 | None | None | None | None |
| Pred_Close_t_plus_05 | 0 | None | None | None | None |
| Pred_Close_t_plus_10 | 0 | None | None | None | None |
| Pred_Close_t_plus_20 | 0 | None | None | None | None |

### StepB file 4
- path: `C:\work\apex_work\runs\20260221_172139\output\stepB\sim\stepB_pred_path_mamba_periodic_SOXL.csv`
- rows/cols: 2016/2
- date_monotonic: False, duplicates: None
- pred_cols: Pred_Close_t_plus_20
- pred NaN:
  - Pred_Close_t_plus_20: 0

| pred_col | n | MAE | RMSE | corr | dir_acc |
|---|---:|---:|---:|---:|---:|
| Pred_Close_t_plus_20 | 0 | None | None | None | None |

### StepB file 5
- path: `C:\work\apex_work\runs\20260221_172139\output\stepB\sim\stepB_pred_time_all_SOXL.csv`
- rows/cols: 2076/4
- date_monotonic: True, duplicates: 0
- pred_cols: Pred_Close_XSR, Pred_Close_MAMBA, Pred_Close_FED
- pred NaN:
  - Pred_Close_XSR: 2076
  - Pred_Close_MAMBA: 60
  - Pred_Close_FED: 2076

| pred_col | n | MAE | RMSE | corr | dir_acc |
|---|---:|---:|---:|---:|---:|
| Pred_Close_XSR | 0 | None | None | None | None |
| Pred_Close_MAMBA | 62 | 4.360174086786086 | 5.285104386720852 | 0.8982551856091188 | 0.36065573770491804 |
| Pred_Close_FED | 0 | None | None | None | None |

## StepE Reward/Return/Equity Evaluation
**Status: SKIP** - no stepE daily log files found

SKIP reason: no stepE daily log files found

---
best-effort evaluator: errors in individual sections are reported as SKIP; process exit code is always 0.
