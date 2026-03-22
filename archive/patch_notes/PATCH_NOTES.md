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


## First rerun watchpoints
After merge, prioritize checking the first full rerun for the following outputs:
- `run_manifest.steps.F.status` should no longer remain `running` when StepF artifacts and audits are already present.
- `run_manifest.steps.F.completed_at` should be populated once reconciliation promotes StepF to a terminal state.
- `run_manifest.steps.F.audit_status` should reflect the reconciled StepF quality result (`PASS` / `WARN` / `FAIL`).
- StepF market-alignment audit should be reviewed first to confirm the market-return semantics change removed the previous false FAIL pattern.
- StepE leak / reward-alignment audit should be reviewed first to confirm the zero-baseline relative-error hardening removed the previous false FAIL pattern.

This patch intentionally also narrows policy-compare scaling to the test window so the comparison magnitude remains interpretable during rerun triage.
