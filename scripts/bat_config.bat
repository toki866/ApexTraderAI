@echo off
setlocal

rem ===== ApexTraderAI batch defaults =====

if not defined SYMBOL set "SYMBOL=SOXL"
if not defined SYMBOLS set "SYMBOLS=SOXL,SOXS"
if not defined DATA_START set "DATA_START=2014-01-01"
if not defined DATA_END set "DATA_END=2022-03-31"
if not defined TEST_START set "TEST_START=2022-01-03"
if not defined TRAIN_YEARS set "TRAIN_YEARS=8"
if not defined TEST_MONTHS set "TEST_MONTHS=3"
if not defined STEPS set "STEPS=A,B,C,D,DPRIME,E,F"
if not defined RUN_MODE set "RUN_MODE=sim"

if not defined ENABLE_ALL set "ENABLE_ALL=0"
if not defined ENABLE_XSR set "ENABLE_XSR=0"
if not defined ENABLE_MAMBA set "ENABLE_MAMBA=1"
if not defined ENABLE_MAMBA_PERIODIC set "ENABLE_MAMBA_PERIODIC=0"
if not defined ENABLE_FEDFORMER set "ENABLE_FEDFORMER=0"

if not defined WORK_ROOT set "WORK_ROOT=C:\work\apex_work\runs"
if not defined ZIP_ON_SUCCESS set "ZIP_ON_SUCCESS=1"
if not defined AUTO_PREPARE_DATA set "AUTO_PREPARE_DATA=0"
if not defined PYTHON_EXE set "PYTHON_EXE=python"

endlocal & (
  set "SYMBOL=%SYMBOL%"
  set "SYMBOLS=%SYMBOLS%"
  set "DATA_START=%DATA_START%"
  set "DATA_END=%DATA_END%"
  set "TEST_START=%TEST_START%"
  set "TRAIN_YEARS=%TRAIN_YEARS%"
  set "TEST_MONTHS=%TEST_MONTHS%"
  set "STEPS=%STEPS%"
  set "RUN_MODE=%RUN_MODE%"
  set "ENABLE_ALL=%ENABLE_ALL%"
  set "ENABLE_XSR=%ENABLE_XSR%"
  set "ENABLE_MAMBA=%ENABLE_MAMBA%"
  set "ENABLE_MAMBA_PERIODIC=%ENABLE_MAMBA_PERIODIC%"
  set "ENABLE_FEDFORMER=%ENABLE_FEDFORMER%"
  set "WORK_ROOT=%WORK_ROOT%"
  set "ZIP_ON_SUCCESS=%ZIP_ON_SUCCESS%"
  set "AUTO_PREPARE_DATA=%AUTO_PREPARE_DATA%"
  set "PYTHON_EXE=%PYTHON_EXE%"
)
