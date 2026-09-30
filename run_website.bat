@echo off
title Education Agent - AI Learning Dashboard
echo ========================================================
echo   Education Agent - AI-Powered Personalized Learning
echo ========================================================
echo.

cd /d "%~dp0"

echo [1/3] Checking Python virtual environment...
if not exist ".venv\Scripts\python.exe" (
    echo Error: .venv not found. Creating virtual environment...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    echo Virtual environment detected.
)

echo [2/3] Starting Education Agent Web Server on http://127.0.0.1:5000 ...
set PYTHONIOENCODING=utf-8

start "" "http://127.0.0.1:5000"

echo [3/3] Launching web server...
".venv\Scripts\python.exe" server.py

pause
