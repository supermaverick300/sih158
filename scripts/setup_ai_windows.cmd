@echo off
setlocal
cd /d "%~dp0.."
if not exist venv\Scripts\python.exe (
  echo Run scripts\setup_windows.cmd first.
  exit /b 1
)
venv\Scripts\python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 exit /b 1
venv\Scripts\python -m pip install -r requirements-ai.txt
if errorlevel 1 exit /b 1
venv\Scripts\python scripts\download_depth_model.py
if errorlevel 1 exit /b 1
echo AI depth is ready. In the app choose Depth Anything V2 under Pipeline settings.
