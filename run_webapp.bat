@echo off
title AI Career Mentor - Material 3 Web App
cd /d "%~dp0"
echo ====================================================
echo   AI Career Mentor - Material 3 & Glassmorphism App
echo ====================================================
echo.

if exist .venv\Scripts\uvicorn.exe (
    echo Starting FastAPI server via .venv on http://localhost:8000 ...
    .\.venv\Scripts\uvicorn.exe server:app --host 0.0.0.0 --port 8000 --reload
) else (
    echo Starting FastAPI server via default Python on http://localhost:8000 ...
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload
)

pause
