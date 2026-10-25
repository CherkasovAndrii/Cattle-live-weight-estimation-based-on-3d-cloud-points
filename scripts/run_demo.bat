@echo off
rem Start the defence demo: the farm web app on this computer and on phones
rem in the same Wi-Fi network.
rem   scripts\run_demo.bat C:\data\CowDatabase2 [C:\data\CowDatabase]
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0.."
if "%~1"=="" (
  echo usage: scripts\run_demo.bat PATH_TO_CowDatabase2 [PATH_TO_CowDatabase]
  exit /b 2
)
if not exist .venv\Scripts\cattle3d.exe (
  py -3 -m venv .venv || python -m venv .venv || exit /b 1
  .venv\Scripts\python -m pip install --upgrade pip
  .venv\Scripts\python -m pip install -e ".[api,demo]" || exit /b 1
)
if "%~2"=="" (
  .venv\Scripts\cattle3d demo --cowdb2 "%~1"
) else (
  .venv\Scripts\cattle3d demo --cowdb2 "%~1" --cowdb1 "%~2"
)
