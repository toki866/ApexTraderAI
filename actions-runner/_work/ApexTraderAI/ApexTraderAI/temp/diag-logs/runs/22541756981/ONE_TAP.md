# ONE_TAP Public Run Report

- run_id: 22541756981
- sha: 9a6d763015950bfbb1bd43ef3d856e9ddcd049ee
- mode: sim
- symbols: SOXL,SOXS
- test_start: 2022-01-03
- train_years: 8
- test_months: 3
- output_root: C:/work/apex_work/runs/gh22541756981_att1_sim_20260301_194521_9a6d763/output

## StepE metrics
not found

## StepF metrics
| step  | mode | source_csv | status    |
| ----- | ---- | ---------- | --------- |
| StepF | sim  | not found  | not found |

## Key CSV files
- No target CSV files found

## Error summary
```text
[CMD] 'C:\Users\becky\AppData\Local\Programs\Python\Python310\python.exe' 'tools\prepare_data.py' --symbols SOXL,SOXS --start 2013-12-27 --end 2022-04-10 --force --data-dir 'C:\work\apex_work\runs\gh22541756981_att1_sim_20260301_194521_9a6d763\data'
[RC] 0

[CMD] "python" tools\run_with_python.py tools\run_pipeline.py --symbol SOXL --steps "A,B,C,DPRIME,E,F" --test-start 2022-01-03 --train-years 8 --test-months 3 --mode sim --output-root "C:\work\apex_work\runs\gh22541756981_att1_sim_20260301_194521_9a6d763\output" --data-dir "C:\work\apex_work\runs\gh22541756981_att1_sim_20260301_194521_9a6d763\data" --auto-prepare-data 0  --enable-mamba
[headless] output_root_resolved=C:\work\apex_work\runs\gh22541756981_att1_sim_20260301_194521_9a6d763\output cfg_output_root=C:\work\apex_work\runs\gh22541756981_att1_sim_20260301_194521_9a6d763\output cfg_data_output_root=C:\work\apex_work\runs\gh22541756981_att1_sim_20260301_194521_9a6d763\output
[headless] data ready: SOXL,SOXS
[headless] StepE default config injected: agents=dprime_bnf_h01,dprime_bnf_h02,dprime_bnf_3scale,dprime_mix_h01,dprime_mix_h02,dprime_mix_3scale,dprime_all_features_h01,dprime_all_features_h02,dprime_all_features_h03,dprime_all_features_3scale seed=42+idx device=auto
[headless] StepF default config injected: agents=dprime_bnf_h01,dprime_bnf_h02,dprime_bnf_3scale,dprime_mix_h01,dprime_mix_h02,dprime_mix_3scale,dprime_all_features_h01,dprime_all_features_h02,dprime_all_features_h03,dprime_all_features_3scale safe_set=dprime_bnf_h01,dprime_all_features_h01 router(beta=1.0,ema=0.3,cost_bps=10.0)
[headless] repo_root=C:\work\apex_repo_cache\ApexTraderAI
[headless] symbol=SOXL
[headless] steps=A,B,C,DPRIME,E,F
[headless] skip_stepe=0
[headless] auto_prepare_data=0 data_start=2010-01-01 data_end=2026-03-01
[StepA] start symbols=SOXL,SOXS
[StepA] data_dir=C:\work\apex_work\runs\gh22541756981_att1_sim_20260301_194521_9a6d763\data
[StepA] data_dir=C:\work\apex_work\runs\gh22541756981_att1_sim_20260301_194521_9a6d763\data src_csv=C:\work\apex_work\runs\gh22541756981_att1_sim_20260301_194521_9a6d763\data\prices_SOXL.csv
[StepA] data_dir=C:\work\apex_work\runs\gh22541756981_att1_sim_20260301_194521_9a6d763\data
[StepA] data_dir=C:\work\apex_work\runs\gh22541756981_att1_sim_20260301_194521_9a6d763\data src_csv=C:\work\apex_work\runs\gh22541756981_att1_sim_20260301_194521_9a6d763\data\prices_SOXS.csv
[StepA] done
[StepB] start
[StepB:mamba] wrote daily manifest -> C:\work\apex_work\runs\gh22541756981_att1_sim_20260301_194521_9a6d763\output\stepB\sim\stepB_daily_manifest_SOXL.csv rows=63
[StepB:mamba] wrote daily manifest -> C:\work\apex_work\runs\gh22541756981_att1_sim_20260301_194521_9a6d763\output\stepB\sim\stepB_daily_manifest_periodic_SOXL.csv rows=63
[StepB] agents: mamba=True
[StepB] done
[StepB] ensured: C:\work\apex_work\runs\gh22541756981_att1_sim_20260301_194521_9a6d763\output\stepB\sim\stepB_pred_time_all_SOXL.csv
[StepC] start
[StepC] price_path=C:\work\apex_work\runs\gh22541756981_att1_sim_20260301_194521_9a6d763\output\stepA\sim\stepA_prices_train_SOXL.csv
[StepC] wrote daily snapshots: 63 -> C:\work\apex_work\runs\gh22541756981_att1_sim_20260301_194521_9a6d763\output\stepC\sim\stepC_daily_manifest_SOXL.csv
[StepC] done
[StepDPrime] start
[StepDPrime] done
[StepE] start
[StepE] agent=dprime_bnf_h01 mode=sim profile=D use_stepd_prime=True seed=42 device=auto
[StepE] obs_cols_count=88
[StepE] obs_cols(first5)=['dprime_dprime_bnf_h01_emb_000', 'dprime_dprime_bnf_h01_emb_001', 'dprime_dprime_bnf_h01_emb_002', 'dprime_dprime_bnf_h01_emb_003', 'dprime_dprime_bnf_h01_emb_004']
[LEAK_GUARD] OK forbidden_hit=0
[PIPELINE] status=failed steps=A,B,C,DPRIME,E,F output_root=C:\work\apex_work\runs\gh22541756981_att1_sim_20260301_194521_9a6d763\output
Traceback (most recent call last):
  File "C:\work\apex_repo_cache\ApexTraderAI\ai_core\services\step_e_service.py", line 535, in _train_and_eval_ppo
    from stable_baselines3 import PPO
ModuleNotFoundError: No module named 'stable_baselines3'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\work\apex_repo_cache\ApexTraderAI\tools\run_pipeline.py", line 1628, in <module>
    raise SystemExit(main())
  File "C:\work\apex_repo_cache\ApexTraderAI\tools\run_pipeline.py", line 1605, in main
    step_result = _run_step_generic(step, app_config, symbol, date_range, results)
  File "C:\work\apex_repo_cache\ApexTraderAI\tools\run_pipeline.py", line 1250, in _run_step_generic
    return _call_with_best_effort(fn, ctx2)
  File "C:\work\apex_repo_cache\ApexTraderAI\tools\run_pipeline.py", line 485, in _call_with_best_effort
    return fn(**kwargs)
  File "C:\work\apex_repo_cache\ApexTraderAI\ai_core\services\step_e_service.py", line 179, in run
    self._run_one(cfg, date_range=date_range, symbol=symbol, mode=mode)
  File "C:\work\apex_repo_cache\ApexTraderAI\ai_core\services\step_e_service.py", line 276, in _run_one
    self._train_and_eval_ppo(
  File "C:\work\apex_repo_cache\ApexTraderAI\ai_core\services\step_e_service.py", line 538, in _train_and_eval_ppo
    raise RuntimeError("PPO policy_kind requires stable-baselines3 to be installed") from e
RuntimeError: PPO policy_kind requires stable-baselines3 to be installed
[CMD] 'C:\Users\becky\AppData\Local\Programs\Python\Python310\python.exe' 'tools\run_pipeline.py' --symbol SOXL --steps A,B,C,DPRIME,E,F --test-start 2022-01-03 --train-years 8 --test-months 3 --mode sim --output-root 'C:\work\apex_work\runs\gh22541756981_att1_sim_20260301_194521_9a6d763\output' --data-dir 'C:\work\apex_work\runs\gh22541756981_att1_sim_20260301_194521_9a6d763\data' --auto-prepare-data 0 --enable-mamba
[RC] 1
[RC] 
[FAILED] command="python" tools\run_with_python.py tools\run_pipeline.py --symbol SOXL --steps "A,B,C,DPRIME,E,F" --test-start 2022-01-03 --train-years 8 --test-months 3 --mode sim --output-root "C:\work\apex_work\runs\gh22541756981_att1_sim_20260301_194521_9a6d763\output" --data-dir "C:\work\apex_work\runs\gh22541756981_att1_sim_20260301_194521_9a6d763\data" --auto-prepare-data 0  --enable-mamba
[FAILED] exit_code=
[FAILED] run_id=gh22541756981_att1_sim_20260301_194521_9a6d763
[FAILED] log=C:\work\apex_work\runs\gh22541756981_att1_sim_20260301_194521_9a6d763\logs\run_gh22541756981_att1_sim_20260301_194521_9a6d763.log
[STEP] package_diagnostics_to_onedrive
[INFO] diag_script=C:\work\apex_repo_cache\ApexTraderAI\scripts\package_diagnostics_to_onedrive.ps1
[PUBLISH] run_dir=C:\work\apex_work\runs\gh22541756981_att1_sim_20260301_194521_9a6d763
[PUBLISH] output_root=C:\work\apex_work\runs\gh22541756981_att1_sim_20260301_194521_9a6d763\output
[OK] package_diagnostics_to_onedrive
[OK] diag_dir=C:\Users\becky\OneDrive\ApexTraderAI\diagnostics

[STEP] resolve_latest_run_artifacts
[INFO] work_root=C:\work\apex_work\runs
[OK] resolve_latest_run_artifacts
[OK] run_dir=C:\work\apex_work\runs\gh22541756981_att1_sim_20260301_194521_9a6d763
[OK] run_id=gh22541756981_att1_sim_20260301_194521_9a6d763
```
