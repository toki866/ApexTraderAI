Patch v4c: StepE/StepF return alignment + obs leak hardening

What this patch does
- StepE: training return uses cc_ret_next (= Close_eff[t+1]/Close_eff[t]-1, fallback to Close) instead of oc_ret.
- StepE: removes oc_ret from profile B/C observation list (prevents same-day return leak).
- StepE: excludes label/target columns (label, label_available, etc.) from observations (prevents dprime label leakage).
- StepF: gating training/eval return uses cc_ret_next (Close shift(-1)).

How to apply
1) Extract this zip at repo root (so you get ./patch_files/ and ./tools/apply_patch_stepE_stepF_v4c.py)
2) Run:
   python tools/apply_patch_stepE_stepF_v4c.py

Then verify
- Re-run StepE compare + alignment check:
  python tools/check_stepE_reward_alignment.py --symbol SOXL --mode sim --all
