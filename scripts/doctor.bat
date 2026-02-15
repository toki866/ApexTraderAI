@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%.."
pushd "%REPO_ROOT%" >nul || exit /b 2

if exist "%SCRIPT_DIR%bat_config.bat" call "%SCRIPT_DIR%bat_config.bat"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "RUN_ID=%%i"
set "RUN_DIR=%WORK_ROOT%\%RUN_ID%"
set "LOG_DIR=%RUN_DIR%\logs"
set "LOG_FILE=%LOG_DIR%\doctor_%RUN_ID%.log"
mkdir "%LOG_DIR%" >nul 2>&1

echo [doctor] repo=%CD% > "%LOG_FILE%"

echo [doctor] git hash:
echo [doctor] git hash:>> "%LOG_FILE%"
git rev-parse --short HEAD >> "%LOG_FILE%" 2>&1

echo [doctor] python version:
echo [doctor] python version:>> "%LOG_FILE%"
%PYTHON_EXE% --version >> "%LOG_FILE%" 2>&1

echo [doctor] torch/cuda:
echo [doctor] torch/cuda:>> "%LOG_FILE%"
%PYTHON_EXE% -c "import torch;print('torch',torch.__version__);print('cuda_available',torch.cuda.is_available());print('cuda_device_count',torch.cuda.device_count())" >> "%LOG_FILE%" 2>&1

echo [doctor] expected data files:
echo [doctor] expected data files:>> "%LOG_FILE%"
for %%S in (%SYMBOLS:,= %) do (
  if exist "data\prices_%%S.csv" (
    for %%F in ("data\prices_%%S.csv") do (
      echo OK data\prices_%%S.csv updated=%%~tF
      echo OK data\prices_%%S.csv updated=%%~tF>> "%LOG_FILE%"
    )
  ) else (
    echo MISSING data\prices_%%S.csv
    echo MISSING data\prices_%%S.csv>> "%LOG_FILE%"
  )
)

echo [doctor] latest output folders:
echo [doctor] latest output folders:>> "%LOG_FILE%"
dir /b /o:-d output >> "%LOG_FILE%" 2>&1

echo [doctor] log=%LOG_FILE%
popd >nul
exit /b 0
