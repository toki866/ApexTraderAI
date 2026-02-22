# CSV Index

- source_run_dir: C:\work\apex_work\runs\20260222_095330
- source_output_root: C:\work\apex_work\runs\20260222_095330\output

- file: ./csv/stepA_prices_test_SOXL.csv | size_bytes: 5601 | head_only: no
  - preview:
    - Date,Open,High,Low,Close,Volume
    - 2022-01-03,69.05000305175781,72.11000061035156,68.38999938964844,72.0999984741211,15794800
- file: ./csv/stepA_split_summary_SOXL.csv | size_bytes: 448 | head_only: no
  - preview:
    - key,value
    - mode,sim
- file: ./csv/stepB_pred_close_mamba_periodic_SOXL.csv | size_bytes: 111067 | head_only: no
  - preview:
    - Date,Pred_Close_MAMBA_PERIODIC_h20,Delta_Close_pred_MAMBA_PERIODIC_h20,Pred_Close_MAMBA_PERIODIC,Delta_Close_pred_MAMBA_PERIODIC
    - 2017-01-03,,,,
- file: ./csv/stepB_pred_close_mamba_SOXL.csv | size_bytes: 254916 | head_only: no
  - preview:
    - Date,Pred_Close_MAMBA_h01,Delta_Close_pred_MAMBA_h01,Pred_Close_MAMBA_h05,Delta_Close_pred_MAMBA_h05,Pred_Close_MAMBA_h10,Delta_Close_pred_MAMBA_h10,Pred_Close_MAMBA_h20,Delta_Close_pred_MAMBA_h20,Pred_Close_MAMBA,Delta_Close_pred_MAMBA
    - 2017-01-03,,,,,,,,,,
- file: ./csv/stepB_pred_path_mamba_periodic_SOXL.csv | size_bytes: 38260 | head_only: no
  - preview:
    - Date_anchor,Pred_Close_t_plus_20
    - 2017-03-29,5.264058624308483
- file: ./csv/stepB_pred_path_mamba_SOXL.csv | size_bytes: 107575 | head_only: no
  - preview:
    - Date_anchor,Pred_Close_t_plus_01,Pred_Close_t_plus_05,Pred_Close_t_plus_10,Pred_Close_t_plus_20
    - 2017-03-29,6.294026851654053,5.211824893951416,5.301367282867432,6.070218086242676
- file: ./csv/stepB_pred_time_all_SOXL.csv | size_bytes: 41621 | head_only: no
  - preview:
    - Date,Pred_Close_XSR,Pred_Close_MAMBA,Pred_Close_FED
    - 2017-01-03,,,

- eval_report: ./csv/EVAL_REPORT.md
