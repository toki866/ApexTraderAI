router_bandit_agent_arg_helper_v2

Fixes vs v1
- No duplicate candidates (single glob pass + dedupe)
- No SyntaxWarning from backslashes (raw docstring / raw help strings)
- PRIMARY logs excluded by default (use --include-primary to include)

How to use
1) Extract this ZIP at your repo root (preserve folders)
2) Run:
   python tools\gen_agent_csv_args.py --dir output\stepE\sim --symbol SOXL --top 10

If you want to include PRIMARY logs:
   python tools\gen_agent_csv_args.py --dir output\stepE\sim --symbol SOXL --top 10 --include-primary

Then copy/paste the printed --agent-csv lines into your bandit backtest command.
