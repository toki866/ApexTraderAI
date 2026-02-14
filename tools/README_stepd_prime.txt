StepD' (StepD-prime) - Transformer summarizer (v1)

What it does:
  - Reads StepB daily predicted-path CSVs (mamba / mamba_periodic) for each horizon (h01/h05/h10/h20)
  - Builds fixed-length sequences (pad to max_len=20)
  - Trains a small Transformer encoder per (source, horizon) using supervised sign classification
  - Writes embeddings CSVs to output/stepD_prime/<mode>/embeddings/
  - Saves model checkpoints to output/stepD_prime/<mode>/models/

Quick start (Windows CMD):
  conda activate soxl_torch
  python tools\run_stepd_prime.py --symbol SOXL --mode sim --sources mamba_periodic,mamba --horizons 1,5,10,20 --fit-split available

Output:
  output/stepD_prime/<mode>/
    models/
      stepDprime_<source>_hXX_<SYMBOL>.pt
      stepDprime_<source>_hXX_<SYMBOL>.json
    embeddings/
      stepDprime_<source>_hXX_<SYMBOL>_embeddings.csv
    stepDprime_summary_<SYMBOL>.json

Notes:
  - fit-split=available is for first validation. Properly, train embeddings on training period only (fit-split=train).
    That requires StepB daily predicted paths for the training period as well.
