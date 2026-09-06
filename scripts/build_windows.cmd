@echo off
setlocal
cd /d "%~dp0.."
venv\Scripts\python -m pip install pyinstaller
if errorlevel 1 exit /b 1
venv\Scripts\python -m PyInstaller --noconfirm --windowed --name Drone3DStudio --paths src --collect-all trimesh main.py
if errorlevel 1 exit /b 1
echo Output: dist\Drone3DStudio\Drone3DStudio.exe
