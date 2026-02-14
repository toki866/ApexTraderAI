Router smoke test tool (no trading)

1) Place your router_table yaml somewhere, e.g.
   output/stepF/sim/router_table_SOXL.yaml

2) Find your policy files (examples):
   dir /s /b output\*policy*SOXL*.zip
   dir /s /b output\*policy*SOXL*.npz

3) Run:
   python tools\run_router_decision_one_day.py --symbol SOXL --output-root output --mode sim --date 2022-01-03 ^
     --router-table output\stepF\sim\router_table_SOXL.yaml ^
     --router-log output\engine\router_log_SOXL.csv ^
     --agent dprime_all_features_h01=PATH_TO_POLICY ^
     --agent dprime_all_features_h02=PATH_TO_POLICY ^
     --agent dprime_all_features_h03=PATH_TO_POLICY

Tip: test a day where BNF_DivDownVolUp==1 or BNF_EnergyFade==1 to see switching.
