@echo off
setlocal EnableExtensions

if "%~1"=="" (
  echo Usage: %~nx0 ^<local_run_dir^>
  echo Example: %~nx0 C:\work\apex_work\runs\20260115_073000
  exit /b 2
)

set "RUN_DIR=%~1"
if not exist "%RUN_DIR%" (
  echo [ERROR] run dir not found: %RUN_DIR%
  exit /b 3
)

for %%I in ("%RUN_DIR%") do set "RUN_ID=%%~nxI"

if defined ONE_DRIVE_RUNS_ROOT (
  set "DEST=%ONE_DRIVE_RUNS_ROOT%\%RUN_ID%"
) else if defined OneDrive (
  set "DEST=%OneDrive%\ApexTraderAI\runs\%RUN_ID%"
) else (
  echo [ERROR] OneDrive path is not set. Define OneDrive or ONE_DRIVE_RUNS_ROOT.
  exit /b 4
)

mkdir "%DEST%" >nul 2>&1
robocopy "%RUN_DIR%\output" "%DEST%\output" /E /Z /R:2 /W:2
set "RC=%ERRORLEVEL%"
if %RC% GEQ 8 exit /b %RC%

robocopy "%RUN_DIR%\logs" "%DEST%\logs" /E /Z /R:2 /W:2
set "RC=%ERRORLEVEL%"
if %RC% GEQ 8 exit /b %RC%

if exist "%RUN_DIR%\run_%RUN_ID%.zip" (
  robocopy "%RUN_DIR%" "%DEST%" "run_%RUN_ID%.zip" /R:2 /W:2
  set "RC=%ERRORLEVEL%"
  if %RC% GEQ 8 exit /b %RC%
)

echo [OK] copied run %RUN_ID% to %DEST%
exit /b 0
