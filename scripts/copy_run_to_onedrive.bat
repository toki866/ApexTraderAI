@echo off
setlocal EnableExtensions

if "%~1"=="" (
  echo Usage: %~nx0 ^<canonical_output_dir_or_run_dir^>
  echo Example: %~nx0 C:\work\apex_work\output\sim\SOXL\2022-01-03
  exit /b 2
)

set "INPUT_PATH=%~1"
if not exist "%INPUT_PATH%" (
  echo [ERROR] path not found: %INPUT_PATH%
  exit /b 3
)

set "OUTPUT_ROOT="
if exist "%INPUT_PATH%\run_manifest.json" (
  set "OUTPUT_ROOT=%INPUT_PATH%"
) else if exist "%INPUT_PATH%\output\run_manifest.json" (
  set "OUTPUT_ROOT=%INPUT_PATH%\output"
) else if exist "%INPUT_PATH%\output" (
  set "OUTPUT_ROOT=%INPUT_PATH%\output"
) else (
  set "OUTPUT_ROOT=%INPUT_PATH%"
)

if not defined APEX_RUN_ID (
  for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "APEX_RUN_ID=%%i"
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0export_output_zip_to_onedrive.ps1" -OutputRoot "%OUTPUT_ROOT%" -CopyToOneDrive 1 -OutputRootSource manual_copy_run_to_onedrive -EmitJson
set "RC=%ERRORLEVEL%"
if %RC% GEQ 1 exit /b %RC%

echo [OK] exported canonical output zip from %OUTPUT_ROOT%
exit /b 0
