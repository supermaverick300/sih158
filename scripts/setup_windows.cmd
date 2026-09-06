@echo off
setlocal
cd /d "%~dp0.."
py -3.12 -m venv venv
if errorlevel 1 exit /b 1
venv\Scripts\python -m pip install --upgrade pip
if errorlevel 1 exit /b 1
venv\Scripts\python -m pip install -r requirements-dev.txt
if errorlevel 1 exit /b 1
echo Setup complete. Run venv\Scripts\activate then python main.py
