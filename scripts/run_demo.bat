@echo off
rem Start the defence demo: the farm web app on this computer and on phones.
rem   scripts\run_demo.bat C:\data\CowDatabase2 [C:\data\CowDatabase]
rem With cloudflared installed (winget install Cloudflare.cloudflared) phones
rem on any network, mobile data included, get an internet address.
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
rem Phones on the same Wi-Fi need an inbound firewall rule for port 8000.
rem Adding it needs administrator rights, so Windows asks once.
netsh advfirewall firewall show rule name="cattle3d demo" >nul 2>&1
if errorlevel 1 (
  echo Allowing phones to connect: Windows will ask for administrator rights once.
  powershell -NoProfile -Command "Start-Process netsh -Verb RunAs -Wait -ArgumentList 'advfirewall firewall add rule name=\"cattle3d demo\" dir=in action=allow protocol=TCP localport=8000 profile=any'" >nul 2>&1
)
set TUNNEL=
where cloudflared >nul 2>&1 && set TUNNEL=--tunnel
if "%~2"=="" (
  .venv\Scripts\cattle3d demo --cowdb2 "%~1" %TUNNEL%
) else (
  .venv\Scripts\cattle3d demo --cowdb2 "%~1" --cowdb1 "%~2" %TUNNEL%
)
