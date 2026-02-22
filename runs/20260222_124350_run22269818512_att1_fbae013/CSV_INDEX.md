# CSV Index

- source_run_dir: C:\work\apex_work\runs\20260222_124318
- source_output_root: C:\work\apex_work\runs\20260222_124318\output

- file: ./csv/stepA_prices_test_SOXL.csv | size_bytes: 5601 | head_only: no
  - preview:
    - Date,Open,High,Low,Close,Volume
    - 2022-01-03,69.05000305175781,72.11000061035156,68.38999938964844,72.0999984741211,15794800
- file: ./csv/stepA_split_summary_SOXL.csv | size_bytes: 448 | head_only: no
  - preview:
    - key,value
    - mode,sim

- eval_report: ./csv/EVAL_REPORT.md
- eval_summary: ./csv/EVAL_SUMMARY.txt
