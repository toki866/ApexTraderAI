# CSV Index

- source_run_dir: C:\work\apex_work\runs\20260222_114212
- source_output_root: C:\work\apex_work\runs\20260222_114212\output

- file: ./csv/stepA_prices_test_SOXL.csv | size_bytes: 5601 | head_only: no
  - preview:
    - Date,Open,High,Low,Close,Volume
    - 2022-01-03,69.05000305175781,72.11000061035156,68.38999938964844,72.0999984741211,15794800
- file: ./csv/stepA_split_summary_SOXL.csv | size_bytes: 448 | head_only: no
  - preview:
    - key,value
    - mode,sim
- file: ./csv/stepB_pred_close_mamba_periodic_SOXL.csv | size_bytes: 179433 | head_only: no
  - preview:
    - Date,Pred_Close_MAMBA_PERIODIC_h20,Delta_Close_pred_MAMBA_PERIODIC_h20,Pred_Close_MAMBA_PERIODIC,Delta_Close_pred_MAMBA_PERIODIC
    - 2014-01-03,,,,
- file: ./csv/stepB_pred_close_mamba_SOXL.csv | size_bytes: 411286 | head_only: no
  - preview:
    - Date,Pred_Close_MAMBA_h01,Delta_Close_pred_MAMBA_h01,Pred_Close_MAMBA_h05,Delta_Close_pred_MAMBA_h05,Pred_Close_MAMBA_h10,Delta_Close_pred_MAMBA_h10,Pred_Close_MAMBA_h20,Delta_Close_pred_MAMBA_h20,Pred_Close_MAMBA,Delta_Close_pred_MAMBA
    - 2014-01-03,,,,,,,,,,
- file: ./csv/stepB_pred_path_mamba_periodic_SOXL.csv | size_bytes: 61401 | head_only: no
  - preview:
    - Date_anchor,Pred_Close_t_plus_20
    - 2014-03-31,1.4129317456880628
- file: ./csv/stepB_pred_path_mamba_SOXL.csv | size_bytes: 172795 | head_only: no
  - preview:
    - Date_anchor,Pred_Close_t_plus_01,Pred_Close_t_plus_05,Pred_Close_t_plus_10,Pred_Close_t_plus_20
    - 2014-03-31,1.7613019943237305,2.0233371257781982,1.4943244457244873,1.540942668914795
- file: ./csv/stepB_pred_time_all_SOXL.csv | size_bytes: 66093 | head_only: no
  - preview:
    - Date,Pred_Close_XSR,Pred_Close_MAMBA,Pred_Close_FED
    - 2014-01-03,,,
- file: ./csv/stepE_daily_log_mamba_SOXL.csv | size_bytes: 206990 | head_only: no
  - preview:
    - Date,Split,pos,ret,equity,Position,Action
    - 2014-01-03,train,0.5350069403648376,-0.010654494166374207,0.9893455058336258,0.5350069403648376,1

- eval_report: ./csv/EVAL_REPORT.md
- eval_summary: ./csv/EVAL_SUMMARY.txt
