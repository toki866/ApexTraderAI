@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%.."
pushd "%REPO_ROOT%" >nul || (echo [ERROR] failed to enter repo root & exit /b 2)

if exist "%SCRIPT_DIR%bat_config.bat" call "%SCRIPT_DIR%bat_config.bat"

if not defined PYTHON_EXE set "PYTHON_EXE=python"
set "PYTHON_EXE=%PYTHON_EXE:\"=%"
set "PYTHON_EXE=%PYTHON_EXE:\\"=%"
if not defined PYTHON_EXE set "PYTHON_EXE=python"

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

call :run git rev-parse --short HEAD
if errorlevel 1 goto :failed

call :run git rev-parse --is-inside-work-tree
if errorlevel 1 goto :failed

rem --- diagnostics (non-fatal): python resolution/version for runner troubleshooting ---
call :run where python
call :run python --version
call :resolve_python
if errorlevel 1 goto :failed
call :run %PYTHON_CMD% %PYTHON_ARGS% --version
if errorlevel 1 goto :failed

if /i "%GITHUB_ACTIONS%"=="true" goto :skip_git_sync

set "HAS_DIRTY="
for /f %%s in ('git status --porcelain 2^>nul') do set "HAS_DIRTY=1"
if defined HAS_DIRTY (
  echo [ERROR] working tree is dirty. commit/stash changes before running.>> "%LOG_FILE%"
  set "LAST_CMD=git status --porcelain"
  set "LAST_EXIT=4"
  goto :failed
)

call :run git pull --ff-only
if errorlevel 1 goto :failed

:skip_git_sync

call :run %PYTHON_CMD% %PYTHON_ARGS% tools\run_with_python.py tools\prepare_data.py --symbols %SYMBOLS% --start %DATA_START% --end %DATA_END% --force --data-dir "%DATA_DIR%"
if errorlevel 1 goto :failed

set "PIPELINE_FLAGS="
if "%ENABLE_ALL%"=="1" set "PIPELINE_FLAGS=!PIPELINE_FLAGS! --enable-all"
if "%ENABLE_XSR%"=="1" set "PIPELINE_FLAGS=!PIPELINE_FLAGS! --enable-xsr"
if "%ENABLE_MAMBA%"=="1" set "PIPELINE_FLAGS=!PIPELINE_FLAGS! --enable-mamba"
if "%ENABLE_MAMBA_PERIODIC%"=="1" set "PIPELINE_FLAGS=!PIPELINE_FLAGS! --enable-mamba-periodic"
if "%ENABLE_FEDFORMER%"=="1" set "PIPELINE_FLAGS=!PIPELINE_FLAGS! --enable-fedformer"

call :run %PYTHON_CMD% %PYTHON_ARGS% tools\run_with_python.py tools\run_pipeline.py --symbol %SYMBOL% --steps %STEPS% --test-start %TEST_START% --train-years %TRAIN_YEARS% --test-months %TEST_MONTHS% --mode %RUN_MODE% --output-root "%OUTPUT_DIR%" --data-dir "%DATA_DIR%" --auto-prepare-data %AUTO_PREPARE_DATA% !PIPELINE_FLAGS!
if errorlevel 1 goto :failed

> "%OUTPUT_DIR%\DONE.txt" (
  echo run_id=%RUN_ID%
  echo finished_at=%DATE% %TIME%
  for /f %%h in ('git rev-parse --short HEAD 2^>nul') do echo commit=%%h
)

if "%ZIP_ON_SUCCESS%"=="1" (
  call :run powershell -NoProfile -Command "Compress-Archive -Path '%OUTPUT_DIR%','%LOG_DIR%' -DestinationPath '%ZIP_FILE%' -Force"
  if errorlevel 1 goto :failed
)

set "ONE_DEST="
if defined ONE_DRIVE_RUNS_ROOT (
  set "ONE_DEST=%ONE_DRIVE_RUNS_ROOT%\%RUN_ID%"
) else (
  if defined OneDrive (
    set "ONE_DEST=%OneDrive%\ApexTraderAI\runs\%RUN_ID%"
  ) else (
    echo [ERROR] OneDrive path is not set. Define OneDrive or ONE_DRIVE_RUNS_ROOT.>> "%LOG_FILE%"
    set "LAST_CMD=resolve OneDrive destination"
    set "LAST_EXIT=3"
    goto :failed
  )
)

call :run mkdir "%ONE_DEST%"
if errorlevel 1 goto :failed

call :run robocopy "%OUTPUT_DIR%" "%ONE_DEST%\output" /E /Z /R:2 /W:2
set "ROBO_RC=%LAST_EXIT%"
if %ROBO_RC% GEQ 8 (
  echo [ERROR] robocopy output failed with code %ROBO_RC%>> "%LOG_FILE%"
  set "LAST_CMD=robocopy output"
  set "LAST_EXIT=%ROBO_RC%"
  goto :failed
)

call :run robocopy "%LOG_DIR%" "%ONE_DEST%\logs" /E /Z /R:2 /W:2
set "ROBO_RC=%LAST_EXIT%"
if %ROBO_RC% GEQ 8 (
  echo [ERROR] robocopy logs failed with code %ROBO_RC%>> "%LOG_FILE%"
  set "LAST_CMD=robocopy logs"
  set "LAST_EXIT=%ROBO_RC%"
  goto :failed
)

if "%ZIP_ON_SUCCESS%"=="1" if exist "%ZIP_FILE%" (
  call :run robocopy "%RUN_DIR%" "%ONE_DEST%" run_%RUN_ID%.zip /R:2 /W:2
  set "ROBO_RC=%LAST_EXIT%"
  if %ROBO_RC% GEQ 8 (
    echo [ERROR] robocopy zip failed with code %ROBO_RC%>> "%LOG_FILE%"
    set "LAST_CMD=robocopy zip"
    set "LAST_EXIT=%ROBO_RC%"
    goto :failed
  )
)

set "SNAPSHOT_ROOT="
if defined ONE_DRIVE_SNAPSHOTS_ROOT (
  set "SNAPSHOT_ROOT=%ONE_DRIVE_SNAPSHOTS_ROOT%"
) else (
  if defined OneDrive (
    set "SNAPSHOT_ROOT=%OneDrive%\ApexTraderAI\repo_snapshots"
  ) else (
    echo [ERROR] OneDrive snapshot path is not set. Define OneDrive or ONE_DRIVE_SNAPSHOTS_ROOT.>> "%LOG_FILE%"
    set "LAST_CMD=resolve OneDrive snapshot destination"
    set "LAST_EXIT=3"
    goto :failed
  )
)

call :run mkdir "%SNAPSHOT_ROOT%"
if errorlevel 1 goto :failed

for /f %%h in ('git rev-parse --short HEAD 2^>nul') do set "SNAPSHOT_SHA=%%h"
if not defined SNAPSHOT_SHA (
  echo [ERROR] failed to resolve current short commit id for snapshot.>> "%LOG_FILE%"
  set "LAST_CMD=git rev-parse --short HEAD"
  set "LAST_EXIT=2"
  goto :failed
)

set "SNAPSHOT_ZIP=%SNAPSHOT_ROOT%\repo_%SNAPSHOT_SHA%_%RUN_ID%.zip"
if exist "%SNAPSHOT_ZIP%" del /f /q "%SNAPSHOT_ZIP%" >nul 2>&1

call :run git archive --format=zip --output="%SNAPSHOT_ZIP%" HEAD
if errorlevel 1 (
  call :run powershell -NoProfile -Command "Compress-Archive -Path '%REPO_ROOT%\*' -DestinationPath '%SNAPSHOT_ZIP%' -Force"
  if errorlevel 1 goto :failed
)

if exist "%SNAPSHOT_ZIP%" attrib +R "%SNAPSHOT_ZIP%" >nul 2>&1

(
  echo [SUCCESS] Completed run_id=%RUN_ID%
  echo [SUCCESS] local_run_dir=%RUN_DIR%
  echo [SUCCESS] onedrive_dest=%ONE_DEST%
  echo [SUCCESS] repo_snapshot=%SNAPSHOT_ZIP%
  echo [SUCCESS] reproducible_command=%~nx0
) >> "%LOG_FILE%" 2>&1

echo [OK] run_id=%RUN_ID%
echo [OK] local=%RUN_DIR%
echo [OK] onedrive=%ONE_DEST%
popd >nul
exit /b 0

:run
set "LAST_CMD=%*"
echo.>> "%LOG_FILE%"
echo [CMD] %*>> "%LOG_FILE%"
call %* >> "%LOG_FILE%" 2>&1
set "LAST_EXIT=!ERRORLEVEL!"
if not defined LAST_EXIT set "LAST_EXIT=-1"
echo [RC] %LAST_EXIT%>> "%LOG_FILE%"
if %LAST_EXIT% GEQ 1 exit /b %LAST_EXIT%
exit /b 0

:resolve_python
set "PYTHON_CMD="
set "PYTHON_ARGS="
set "_PYTHON_CANDIDATE=%PYTHON_EXE%"
if not defined _PYTHON_CANDIDATE set "_PYTHON_CANDIDATE=python"

if exist "%_PYTHON_CANDIDATE%" (
  set "PYTHON_CMD=""%_PYTHON_CANDIDATE%"""
  exit /b 0
)

where "%_PYTHON_CANDIDATE%" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
  set "PYTHON_CMD=%_PYTHON_CANDIDATE%"
  exit /b 0
)

where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
  set "PYTHON_CMD=python"
  exit /b 0
)

where py >nul 2>&1
if %ERRORLEVEL% EQU 0 (
  set "PYTHON_CMD=py"
  set "PYTHON_ARGS=-3"
  exit /b 0
)

echo [ERROR] python executable not found. tried "%_PYTHON_CANDIDATE%", "python", and "py -3".>> "%LOG_FILE%"
set "LAST_CMD=resolve python executable"
set "LAST_EXIT=127"
exit /b 127

:failed
set "_LAST_EXIT_NUM=%LAST_EXIT%"
for /f "delims=0123456789-" %%A in ("%_LAST_EXIT_NUM%") do set "_LAST_EXIT_NUM=-1"
if not defined _LAST_EXIT_NUM set "_LAST_EXIT_NUM=-1"
set "LAST_EXIT=%_LAST_EXIT_NUM%"
echo [FAILED] command=%LAST_CMD%>> "%LOG_FILE%"
echo [FAILED] exit_code=%LAST_EXIT%>> "%LOG_FILE%"
echo [FAILED] run_id=%RUN_ID%>> "%LOG_FILE%"
echo [FAILED] log=%LOG_FILE%>> "%LOG_FILE%"
echo [NG] failed. log=%LOG_FILE%
popd >nul
exit /b %LAST_EXIT%
