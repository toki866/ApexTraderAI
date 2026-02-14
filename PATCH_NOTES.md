# Patch: StepE/StepF reward alignment fix

## What changed
- StepE (dprime compare) reward target was incorrectly using same-day `oc_ret` (Open->Close) as the return.
  This can create a time-shift / leakage because the reward is available from same-day info.
- Now both StepE and StepF use **Close(Date[t+1]) / Close(Date[t]) - 1** as the return (`ret_cc_next`),
  computed **within each split** (train / test) and dropping the last row of each split to avoid boundary leakage.

## New columns in StepE daily log
- Date_next, ret_cc_next, ret_oc_same, ret_oc_next, gap_next
- cost, penalty, ret_gross
- implied_r = (ret + cost + penalty) / pos

## Verification
Run:
  python tools\check_stepE_reward_alignment.py --symbol SOXL --mode sim

You should see MAE ~ 0 and corr ~ 1.0 for `implied_r vs ret_cc_next` (when abs(pos) >= 0.2).
