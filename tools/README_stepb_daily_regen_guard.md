# StepB Daily Regen Guard v2.2

This tool avoids stale StepB daily outputs when re-running with changed settings.

## Usage (cmd.exe)

```
python tools/run_steps_a_f_headless_safe4_force_stepb_agents_regen_daily.py ^
  --symbol SOXL ^
  --test-start 2022-01-03 ^
  --train-years 8 ^
  --test-months 3 ^
  --steps A,B ^
  --enable-mamba ^
  --enable-mamba-periodic ^
  --mamba-mode sim ^
  --mamba-lookback 30 ^
  --mamba-horizons 1,5,10,20
```

## What it does

- Purges `output/stepB/<mode>/daily*` and `stepB_daily_manifest_*` when signature changes.
- Runs the original headless runner.
- Repairs daily outputs to match `stepB_pred_path_mamba*.csv`.
- Writes a sanity report: `output/stepB/<mode>/stepB_daily_sanity_report_<SYMBOL>.txt`.

If sanity fails, exit code is 10.
