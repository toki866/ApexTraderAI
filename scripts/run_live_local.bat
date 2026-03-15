@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%.."
pushd "%REPO_ROOT%" >nul || (echo [ERROR] failed to enter repo root & exit /b 2)

if exist "%SCRIPT_DIR%bat_config.bat" call "%SCRIPT_DIR%bat_config.bat"
if not defined PYTHON_EXE set "PYTHON_EXE=python"
if not defined SYMBOL set "SYMBOL=SOXL"
if not defined TRADE_DATE for /f %%i in ('powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-Date -Format yyyy-MM-dd"') do set "TRADE_DATE=%%i"
if not defined LIVE_OUTPUT_ROOT set "LIVE_OUTPUT_ROOT=output\live\%SYMBOL%\%TRADE_DATE%"
if not defined DATA_DIR set "DATA_DIR=data"
if not defined LIVE_STEPS set "LIVE_STEPS=A,B,C,DPRIME,E,F"
if not defined REUSE_OUTPUT set "REUSE_OUTPUT=1"
if not defined FORCE_REBUILD set "FORCE_REBUILD=0"
if not defined TIMING set "TIMING=1"
if not defined ORDER_MODE set "ORDER_MODE=dry-run"
if not defined EXECUTE_ORDER set "EXECUTE_ORDER=0"
if not defined DECISION_WINDOW set "DECISION_WINDOW=close_pre"
if not defined TRAIN_YEARS set "TRAIN_YEARS=8"
if not defined TEST_MONTHS set "TEST_MONTHS=3"

set "LIVE_LOG_DIR=%LIVE_OUTPUT_ROOT%\logs\live"
set "LIVE_LOG_FILE=%LIVE_LOG_DIR%\run_live_local_%TRADE_DATE%.log"
mkdir "%LIVE_LOG_DIR%" >nul 2>&1

if "%~1"=="" goto :run
set "EXTRA_ARGS=%*"
goto :run

:run
echo [RUN] start run_live_once symbol=%SYMBOL% trade_date=%TRADE_DATE% > "%LIVE_LOG_FILE%" 2>&1
"%PYTHON_EXE%" tools\run_live_once.py ^
  --symbol "%SYMBOL%" ^
  --trade-date "%TRADE_DATE%" ^
  --output-root "%LIVE_OUTPUT_ROOT%" ^
  --data-dir "%DATA_DIR%" ^
  --steps "%LIVE_STEPS%" ^
  --reuse-output %REUSE_OUTPUT% ^
  --force-rebuild %FORCE_REBUILD% ^
  --timing %TIMING% ^
  --order-mode "%ORDER_MODE%" ^
  --execute-order %EXECUTE_ORDER% ^
  --decision-window "%DECISION_WINDOW%" ^
  --train-years %TRAIN_YEARS% ^
  --test-months %TEST_MONTHS% %EXTRA_ARGS% >> "%LIVE_LOG_FILE%" 2>&1
set "RC=%ERRORLEVEL%"
echo [RUN] finish rc=%RC% >> "%LIVE_LOG_FILE%"

type "%LIVE_LOG_FILE%"
popd >nul
exit /b %RC%
