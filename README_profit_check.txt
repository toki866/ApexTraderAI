Profit check tool (StepE / StepF)

1) Run pipeline (example):
  python run_steps_a_f_headless_safe4_force_stepb_agents.py ^
    --symbol SOXL ^
    --test-start 2022-01-03 ^
    --train-years 8 ^
    --test-months 3 ^
    --steps A,B,C ^
    --mamba-mode sim ^
    --enable-mamba ^
    --enable-mamba-periodic

  python tools\run_stepd_prime_features.py ^
    --symbol SOXL ^
    --mode sim ^
    --sources bnf,all_features ^
    --scales 1,2,3 ^
    --fit-split train ^
    --export-split all

  python tools\run_stepE_compare_stepdprime10.py --symbol SOXL --mode sim
  python tools\run_stepF_marl_gate.py --symbol SOXL --mode sim --agents <10 agents...>

2) Check profit summary:
  python tools\check_profit_summary.py --symbol SOXL --mode sim
  python tools\check_profit_summary.py --symbol SOXL --mode sim --initial-capital 1000000

Outputs:
  output\stepE\sim\stepE_profit_summary_SOXL.csv
