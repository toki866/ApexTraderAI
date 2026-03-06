# ONE_TAP Public Run Report

- run_id: 22764074116
- sha: 8cfc00b332bc0619014cef1fa439be9825881ad6
- mode: sim
- symbols: SOXL,SOXS
- test_start: 2022-01-03
- train_years: 8
- test_months: 3
- output_root: C:/work/apex_work/runs/gh22764074116_att1_sim_20260306_214726_8cfc00b/output

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
Files : *.*
	    
  Options : *.* /S /E /DCOPY:DA /COPY:DAT /Z /R:2 /W:2 

------------------------------------------------------------------------------

	  New Dir          1	C:\work\apex_work\runs\gh22764074116_att1_sim_20260306_214726_8cfc00b\output\
	    New File  		     107	DONE.txt
100%  

------------------------------------------------------------------------------

               Total    Copied   Skipped  Mismatch    FAILED    Extras
    Dirs :         1         1         0         0         0         0
   Files :         1         1         0         0         0         0
   Bytes :       107       107         0         0         0         0
   Times :   0:00:00   0:00:00                       0:00:00   0:00:00
   Ended : 2026蟷ｴ3譛・譌･ 21:47:33

[RC] 

[CMD] robocopy "C:\work\apex_work\runs\gh22764074116_att1_sim_20260306_214726_8cfc00b\logs" "C:\Users\becky\OneDrive\ApexTraderAI\runs\gh22764074116_att1_sim_20260306_214726_8cfc00b\logs" /E /Z /R:2 /W:2

-------------------------------------------------------------------------------
   ROBOCOPY     ::     Robust File Copy for Windows                              
-------------------------------------------------------------------------------

  Started : 2026蟷ｴ3譛・譌･ 21:47:33
   Source : C:\work\apex_work\runs\gh22764074116_att1_sim_20260306_214726_8cfc00b\logs\
     Dest : C:\Users\becky\OneDrive\ApexTraderAI\runs\gh22764074116_att1_sim_20260306_214726_8cfc00b\logs\

    Files : *.*
	    
  Options : *.* /S /E /DCOPY:DA /COPY:DAT /Z /R:2 /W:2 

------------------------------------------------------------------------------

	  New Dir          2	C:\work\apex_work\runs\gh22764074116_att1_sim_20260306_214726_8cfc00b\logs\
	    New File  		    9111	pip_install_requirements.log
100%  
	    New File  		   15825	run_gh22764074116_att1_sim_20260306_214726_8cfc00b.log
100%  

------------------------------------------------------------------------------

               Total    Copied   Skipped  Mismatch    FAILED    Extras
    Dirs :         1         1         0         0         0         0
   Files :         2         2         0         0         0         0
   Bytes :    24.3 k    24.3 k         0         0         0         0
   Times :   0:00:00   0:00:00                       0:00:00   0:00:00


   Speed :            12468000 Bytes/sec.
   Speed :             713.424 MegaBytes/min.
   Ended : 2026蟷ｴ3譛・譌･ 21:47:33

[RC] 
[OK] repo_snapshots already exists: "C:\Users\becky\OneDrive\ApexTraderAI\repo_snapshots"

[CMD] git archive --format=zip --output="C:\Users\becky\OneDrive\ApexTraderAI\repo_snapshots\repo_8cfc00b_gh22764074116_att1_sim_20260306_214726_8cfc00b.zip" HEAD
[SUCCESS] Completed run_id=gh22764074116_att1_sim_20260306_214726_8cfc00b
[SUCCESS] local_run_dir=C:\work\apex_work\runs\gh22764074116_att1_sim_20260306_214726_8cfc00b
[SUCCESS] onedrive_dest=C:\Users\becky\OneDrive\ApexTraderAI\runs\gh22764074116_att1_sim_20260306_214726_8cfc00b
[SUCCESS] repo_snapshot=C:\Users\becky\OneDrive\ApexTraderAI\repo_snapshots\repo_8cfc00b_gh22764074116_att1_sim_20260306_214726_8cfc00b.zip
[SUCCESS] reproducible_command=run_all_local_then_copy.bat
[STEP] package_diagnostics_to_onedrive
[INFO] diag_script=C:\work\apex_repo_cache\ApexTraderAI\scripts\package_diagnostics_to_onedrive.ps1
[PUBLISH] run_dir=C:\work\apex_work\runs\gh22764074116_att1_sim_20260306_214726_8cfc00b
[PUBLISH] output_root=C:\work\apex_work\runs\gh22764074116_att1_sim_20260306_214726_8cfc00b\output
[WARN] publish issue: D' state CSV missing under output/stepD_prime/<mode>.
[WARN] publish issue: D' embeddings CSV missing under output/stepD_prime/<mode>/embeddings.
[OK] package_diagnostics_to_onedrive
[OK] diag_dir=C:\Users\becky\OneDrive\ApexTraderAI\diagnostics

[STEP] resolve_latest_run_artifacts
[INFO] work_root=C:\work\apex_work\runs
[OK] resolve_latest_run_artifacts
[OK] run_dir=C:\work\apex_work\runs\gh22764074116_att1_sim_20260306_214726_8cfc00b
[OK] run_id=gh22764074116_att1_sim_20260306_214726_8cfc00b
```
