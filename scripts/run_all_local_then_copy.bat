@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%.."
pushd "%REPO_ROOT%" >nul || (echo [ERROR] failed to enter repo root & exit /b 2)

if exist "%SCRIPT_DIR%bat_config.bat" call "%SCRIPT_DIR%bat_config.bat"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "RUN_ID=%%i"
if not defined RUN_ID (
  echo [ERROR] failed to generate run_id
  popd
  exit /b 2
)

set "RUN_DIR=%WORK_ROOT%\%RUN_ID%"
set "DATA_DIR=%RUN_DIR%\data"
set "OUTPUT_DIR=%RUN_DIR%\output"
set "LOG_DIR=%RUN_DIR%\logs"
set "LOG_FILE=%LOG_DIR%\run_%RUN_ID%.log"
set "ZIP_FILE=%RUN_DIR%\run_%RUN_ID%.zip"

mkdir "%DATA_DIR%" "%OUTPUT_DIR%" "%LOG_DIR%" >nul 2>&1

set "LAST_CMD=(not started)"
set "LAST_EXIT=0"

(
  echo ==================================================
  echo [RUN] ApexTraderAI local run started
  echo [RUN] run_id=%RUN_ID%
  echo [RUN] repo=%CD%
  echo [RUN] data_dir=%DATA_DIR%
  echo [RUN] output_dir=%OUTPUT_DIR%
  echo [RUN] log=%LOG_FILE%
  echo [RUN] timestamp=%DATE% %TIME%
  for /f %%h in ('git rev-parse --short HEAD 2^>nul') do echo [RUN] commit=%%h
  echo ==================================================
) >> "%LOG_FILE%" 2>&1

call :run "git rev-parse --short HEAD"
if errorlevel 1 goto :failed

call :run "git status --short"
if errorlevel 1 goto :failed

call :run "\"%PYTHON_EXE%\" tools\prepare_data.py --symbols %SYMBOLS% --start %DATA_START% --end %DATA_END% --force --data-dir \"%DATA_DIR%\""
if errorlevel 1 goto :failed

set "PIPELINE_CMD=\"%PYTHON_EXE%\" tools\run_pipeline.py --symbol %SYMBOL% --steps %STEPS% --test-start %TEST_START% --train-years %TRAIN_YEARS% --test-months %TEST_MONTHS% --mode %RUN_MODE% --output-root \"%OUTPUT_DIR%\" --data-dir \"%DATA_DIR%\" --auto-prepare-data %AUTO_PREPARE_DATA%"
if "%ENABLE_ALL%"=="1" set "PIPELINE_CMD=!PIPELINE_CMD! --enable-all"
if "%ENABLE_XSR%"=="1" set "PIPELINE_CMD=!PIPELINE_CMD! --enable-xsr"
if "%ENABLE_MAMBA%"=="1" set "PIPELINE_CMD=!PIPELINE_CMD! --enable-mamba"
if "%ENABLE_MAMBA_PERIODIC%"=="1" set "PIPELINE_CMD=!PIPELINE_CMD! --enable-mamba-periodic"
if "%ENABLE_FEDFORMER%"=="1" set "PIPELINE_CMD=!PIPELINE_CMD! --enable-fedformer"

call :run "!PIPELINE_CMD!"
if errorlevel 1 goto :failed

> "%OUTPUT_DIR%\DONE.txt" (
  echo run_id=%RUN_ID%
  echo finished_at=%DATE% %TIME%
  for /f %%h in ('git rev-parse --short HEAD 2^>nul') do echo commit=%%h
)

if "%ZIP_ON_SUCCESS%"=="1" (
  call :run "powershell -NoProfile -Command \"Compress-Archive -Path '%OUTPUT_DIR%','%LOG_DIR%' -DestinationPath '%ZIP_FILE%' -Force\""
  if errorlevel 1 goto :failed
)

set "ONE_DEST="
if defined ONE_DRIVE_RUNS_ROOT (
  set "ONE_DEST=%ONE_DRIVE_RUNS_ROOT%\%RUN_ID%"
) else if defined OneDrive (
  set "ONE_DEST=%OneDrive%\ApexTraderAI\runs\%RUN_ID%"
) else (
  echo [ERROR] OneDrive path is not set. Define OneDrive or ONE_DRIVE_RUNS_ROOT.>> "%LOG_FILE%"
  set "LAST_CMD=resolve OneDrive destination"
  set "LAST_EXIT=3"
  goto :failed
)

call :run "mkdir \"%ONE_DEST%\""
if errorlevel 1 goto :failed

call :run "robocopy \"%OUTPUT_DIR%\" \"%ONE_DEST%\output\" /E /Z /R:2 /W:2"
set "ROBO_RC=%LAST_EXIT%"
if %ROBO_RC% GEQ 8 (
  echo [ERROR] robocopy output failed with code %ROBO_RC%>> "%LOG_FILE%"
  set "LAST_CMD=robocopy output"
  set "LAST_EXIT=%ROBO_RC%"
  goto :failed
)

call :run "robocopy \"%LOG_DIR%\" \"%ONE_DEST%\logs\" /E /Z /R:2 /W:2"
set "ROBO_RC=%LAST_EXIT%"
if %ROBO_RC% GEQ 8 (
  echo [ERROR] robocopy logs failed with code %ROBO_RC%>> "%LOG_FILE%"
  set "LAST_CMD=robocopy logs"
  set "LAST_EXIT=%ROBO_RC%"
  goto :failed
)

if "%ZIP_ON_SUCCESS%"=="1" if exist "%ZIP_FILE%" (
  call :run "robocopy \"%RUN_DIR%\" \"%ONE_DEST%\" run_%RUN_ID%.zip /R:2 /W:2"
  set "ROBO_RC=%LAST_EXIT%"
  if %ROBO_RC% GEQ 8 (
    echo [ERROR] robocopy zip failed with code %ROBO_RC%>> "%LOG_FILE%"
    set "LAST_CMD=robocopy zip"
    set "LAST_EXIT=%ROBO_RC%"
    goto :failed
  )
)

(
  echo [SUCCESS] Completed run_id=%RUN_ID%
  echo [SUCCESS] local_run_dir=%RUN_DIR%
  echo [SUCCESS] onedrive_dest=%ONE_DEST%
  echo [SUCCESS] reproducible_command=%~nx0
) >> "%LOG_FILE%" 2>&1

echo [OK] run_id=%RUN_ID%
echo [OK] local=%RUN_DIR%
echo [OK] onedrive=%ONE_DEST%
popd >nul
exit /b 0

:run
set "LAST_CMD=%~1"
echo.>> "%LOG_FILE%"
echo [CMD] %~1>> "%LOG_FILE%"
cmd /d /v:off /c "%~1" >> "%LOG_FILE%" 2>&1
set "LAST_EXIT=%ERRORLEVEL%"
echo [RC] %LAST_EXIT%>> "%LOG_FILE%"
if %LAST_EXIT% GEQ 1 exit /b %LAST_EXIT%
exit /b 0

:failed
echo [FAILED] command=%LAST_CMD%>> "%LOG_FILE%"
echo [FAILED] exit_code=%LAST_EXIT%>> "%LOG_FILE%"
echo [FAILED] run_id=%RUN_ID%>> "%LOG_FILE%"
echo [FAILED] log=%LOG_FILE%>> "%LOG_FILE%"
echo [NG] failed. log=%LOG_FILE%
popd >nul
exit /b %LAST_EXIT%
