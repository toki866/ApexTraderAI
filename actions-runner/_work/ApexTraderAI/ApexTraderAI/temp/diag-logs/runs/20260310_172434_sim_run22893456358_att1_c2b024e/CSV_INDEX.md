# CSV Index

- run_folder: 20260310_172434_sim_run22893456358_att1_c2b024e
- source_run_dir: C:\work\apex_work\runs\gh22893456358_att1_sim_20260310_172204_c2b024e
- source_output_root: C:\\work\\apex_work\\output\\sim\\SOXL\\2022-01-03
- output_root_source: workflow_output
- found_count: 7
- published_count: 3

## Used globs
- StepA :: stepA/<mode>/**/*.csv
- StepB :: stepB/<mode>/**/*.csv
- StepDPrime :: stepD_prime/<mode>/**/*.csv
- StepE :: stepE/<mode>/**/stepE_daily_log_*.csv
- StepF :: stepF/<mode>/**/stepF_*.* (csv/json)
- Eval :: temp/eval/EVAL_SUMMARY.txt
- Eval :: temp/eval/EVAL_REPORT.md
- Eval :: temp/eval/EVAL_REPORT.json
- Eval :: temp/eval/EVAL_TABLE_stepA.csv
- Eval :: temp/eval/EVAL_TABLE_stepE.csv
- Eval :: temp/eval/EVAL_TABLE_stepF.csv

## Published files
- [EVAL_TABLE_stepA.csv](./csv/EVAL_TABLE_stepA.csv) | label=Eval | bytes=222 | source_pattern=workspace/temp/eval/EVAL_TABLE_stepA.csv | head_policy=original file copied
- [EVAL_TABLE_stepE.csv](./csv/EVAL_TABLE_stepE.csv) | label=Eval | bytes=46 | source_pattern=workspace/temp/eval/EVAL_TABLE_stepE.csv | head_policy=original file copied
- [EVAL_TABLE_stepF.csv](./csv/EVAL_TABLE_stepF.csv) | label=Eval | bytes=48 | source_pattern=workspace/temp/eval/EVAL_TABLE_stepF.csv | head_policy=original file copied

## Expanded search paths
- stepA/sim/**/*.csv
- stepA/live/**/*.csv
- stepA/display/**/*.csv
- stepB/sim/**/*.csv
- stepB/live/**/*.csv
- stepB/display/**/*.csv
- stepD_prime/sim/**/*.csv
- stepD_prime/live/**/*.csv
- stepD_prime/display/**/*.csv
- stepE/sim/**/stepE_daily_log_*.csv
- stepE/live/**/stepE_daily_log_*.csv
- stepE/display/**/stepE_daily_log_*.csv
- stepF/sim/**/stepF_*.csv
- stepF/sim/**/stepF_*.json
- stepF/live/**/stepF_*.csv
- stepF/live/**/stepF_*.json
- stepF/display/**/stepF_*.csv
- stepF/display/**/stepF_*.json

## Warnings
- [WARN] publish issue: failed to publish file: C:\work\apex_work\output\sim\SOXL\2022-01-03\stepA\sim\stepA_periodic_test_SOXL.csv
- [WARN] publish issue: failed to publish file: C:\work\apex_work\output\sim\SOXL\2022-01-03\stepA\sim\stepA_periodic_train_SOXL.csv
- [WARN] publish issue: failed to publish file: C:\work\apex_work\output\sim\SOXL\2022-01-03\stepA\sim\stepA_prices_test_SOXL.csv
- [WARN] publish issue: failed to publish file: C:\work\apex_work\output\sim\SOXL\2022-01-03\stepA\sim\stepA_prices_train_SOXL.csv
- [WARN] publish issue: failed to publish file: C:\work\apex_work\output\sim\SOXL\2022-01-03\stepA\sim\stepA_split_summary_SOXL.csv
- [WARN] publish issue: failed to publish file: C:\work\apex_work\output\sim\SOXL\2022-01-03\stepA\sim\stepA_tech_test_SOXL.csv
- [WARN] publish issue: failed to publish file: C:\work\apex_work\output\sim\SOXL\2022-01-03\stepA\sim\stepA_tech_train_SOXL.csv

## Final warnings
- [WARN] publish issue: failed to publish file: C:\work\apex_work\output\sim\SOXL\2022-01-03\stepA\sim\stepA_periodic_test_SOXL.csv
- [WARN] publish issue: failed to publish file: C:\work\apex_work\output\sim\SOXL\2022-01-03\stepA\sim\stepA_periodic_train_SOXL.csv
- [WARN] publish issue: failed to publish file: C:\work\apex_work\output\sim\SOXL\2022-01-03\stepA\sim\stepA_prices_test_SOXL.csv
- [WARN] publish issue: failed to publish file: C:\work\apex_work\output\sim\SOXL\2022-01-03\stepA\sim\stepA_prices_train_SOXL.csv
- [WARN] publish issue: failed to publish file: C:\work\apex_work\output\sim\SOXL\2022-01-03\stepA\sim\stepA_split_summary_SOXL.csv
- [WARN] publish issue: failed to publish file: C:\work\apex_work\output\sim\SOXL\2022-01-03\stepA\sim\stepA_tech_test_SOXL.csv
- [WARN] publish issue: failed to publish file: C:\work\apex_work\output\sim\SOXL\2022-01-03\stepA\sim\stepA_tech_train_SOXL.csv
- [WARN] publish issue: StepA files missing in diag publish output.
- [WARN] publish issue: StepB files missing in diag publish output.
- [WARN] publish issue: StepDPrime files missing in diag publish output.
- [WARN] publish issue: StepE files missing in diag publish output.
- [WARN] publish issue: StepF files missing in diag publish output.

- eval_report: ./csv/EVAL_REPORT.md
- eval_summary: ./csv/EVAL_SUMMARY.txt
