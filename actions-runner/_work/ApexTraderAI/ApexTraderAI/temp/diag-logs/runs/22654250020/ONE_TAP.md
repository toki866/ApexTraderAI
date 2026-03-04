# ONE_TAP Public Run Report

- run_id: 22654250020
- sha: b596a13890e9912b8fb28c4dca3d1880977119d9
- mode: sim
- symbols: SOXL,SOXS
- test_start: 2022-01-03
- train_years: 8
- test_months: 3
- output_root: C:/work/apex_work/runs/gh22654250020_att1_sim_20260304_125212_b596a13/output

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
json_out["return_val"] = hook(**hook_input["kwargs"])
    File "C:\Users\becky\AppData\Local\Programs\Python\Python310\lib\site-packages\pip\_vendor\pyproject_hooks\_in_process\_in_process.py", line 143, in get_requires_for_build_wheel
      return hook(config_settings)
    File "C:\Users\becky\AppData\Local\Temp\pip-build-env-__svs83c\overlay\Lib\site-packages\setuptools\build_meta.py", line 333, in get_requires_for_build_wheel
      return self._get_build_requires(config_settings, requirements=[])
    File "C:\Users\becky\AppData\Local\Temp\pip-build-env-__svs83c\overlay\Lib\site-packages\setuptools\build_meta.py", line 301, in _get_build_requires
      self.run_setup()
    File "C:\Users\becky\AppData\Local\Temp\pip-build-env-__svs83c\overlay\Lib\site-packages\setuptools\build_meta.py", line 317, in run_setup
      exec(code, locals())
    File "<string>", line 175, in <module>
  NameError: name 'bare_metal_version' is not defined
  [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
ERROR: Failed to build 'mamba-ssm' when getting requirements to build wheel
[RC] 
[INFO] wrote pip_mamba_tail=C:\work\apex_work\runs\gh22654250020_att1_sim_20260304_125212_b596a13\logs\pip_install_mamba_ssm_tail_200.txt
[FAILED] pip_mamba_tail=C:\work\apex_work\runs\gh22654250020_att1_sim_20260304_125212_b596a13\logs\pip_install_mamba_ssm_tail_200.txt
[FAILED] command=python -m pip install mamba-ssm
[FAILED] exit_code=
[FAILED] run_id=gh22654250020_att1_sim_20260304_125212_b596a13
[FAILED] log=C:\work\apex_work\runs\gh22654250020_att1_sim_20260304_125212_b596a13\logs\run_gh22654250020_att1_sim_20260304_125212_b596a13.log
[STEP] package_diagnostics_to_onedrive
[INFO] diag_script=C:\work\apex_repo_cache\ApexTraderAI\scripts\package_diagnostics_to_onedrive.ps1
[PUBLISH] run_dir=C:\work\apex_work\runs\gh22654250020_att1_sim_20260304_125212_b596a13
[PUBLISH] output_root=C:\work\apex_work\runs\gh22654250020_att1_sim_20260304_125212_b596a13\output
[WARN] publish issue: D' state CSV missing under output/stepD_prime/<mode>.
[WARN] publish issue: D' embeddings CSV missing under output/stepD_prime/<mode>/embeddings.

---- pip_install_mamba_ssm_tail_200 ----
Collecting mamba-ssm
  Using cached mamba_ssm-2.3.0.tar.gz (121 kB)
  Installing build dependencies: started
  Installing build dependencies: finished with status 'done'
  Getting requirements to build wheel: started
  Getting requirements to build wheel: finished with status 'error'
  error: subprocess-exited-with-error
  
  Getting requirements to build wheel did not run successfully.
  exit code: 1
  
  [25 lines of output]
  C:\Users\becky\AppData\Local\Temp\pip-build-env-__svs83c\overlay\Lib\site-packages\wheel\bdist_wheel.py:4: FutureWarning: The 'wheel' package is no longer the canonical location of the 'bdist_wheel' command, and will be removed in a future release. Please update to setuptools v70.1 or later which contains an integrated version of this command.
    warn(
  C:\Users\becky\AppData\Local\Temp\pip-build-env-__svs83c\overlay\Lib\site-packages\torch\_subclasses\functional_tensor.py:283: UserWarning: Failed to initialize NumPy: No module named 'numpy' (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\pytorch\torch\csrc\utils\tensor_numpy.cpp:84.)
    cpu = _conversion_method_template(device=torch.device("cpu"))
  <string>:118: UserWarning: mamba_ssm was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, only images whose names contain 'devel' will provide nvcc.
  
  
  torch.__version__  = 2.10.0+cpu
  
  
  Traceback (most recent call last):
    File "C:\Users\becky\AppData\Local\Programs\Python\Python310\lib\site-packages\pip\_vendor\pyproject_hooks\_in_process\_in_process.py", line 389, in <module>
      main()
    File "C:\Users\becky\AppData\Local\Programs\Python\Python310\lib\site-packages\pip\_vendor\pyproject_hooks\_in_process\_in_process.py", line 373, in main
      json_out["return_val"] = hook(**hook_input["kwargs"])
    File "C:\Users\becky\AppData\Local\Programs\Python\Python310\lib\site-packages\pip\_vendor\pyproject_hooks\_in_process\_in_process.py", line 143, in get_requires_for_build_wheel
      return hook(config_settings)
    File "C:\Users\becky\AppData\Local\Temp\pip-build-env-__svs83c\overlay\Lib\site-packages\setuptools\build_meta.py", line 333, in get_requires_for_build_wheel
      return self._get_build_requires(config_settings, requirements=[])
    File "C:\Users\becky\AppData\Local\Temp\pip-build-env-__svs83c\overlay\Lib\site-packages\setuptools\build_meta.py", line 301, in _get_build_requires
      self.run_setup()
    File "C:\Users\becky\AppData\Local\Temp\pip-build-env-__svs83c\overlay\Lib\site-packages\setuptools\build_meta.py", line 317, in run_setup
      exec(code, locals())
    File "<string>", line 175, in <module>
  NameError: name 'bare_metal_version' is not defined
  [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
ERROR: Failed to build 'mamba-ssm' when getting requirements to build wheel
[OK] package_diagnostics_to_onedrive
[OK] diag_dir=C:\Users\becky\OneDrive\ApexTraderAI\diagnostics

[STEP] resolve_latest_run_artifacts
[INFO] work_root=C:\work\apex_work\runs
[OK] resolve_latest_run_artifacts
[OK] run_dir=C:\work\apex_work\runs\gh22654250020_att1_sim_20260304_125212_b596a13
[OK] run_id=gh22654250020_att1_sim_20260304_125212_b596a13
```
