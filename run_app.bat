@echo off
title AI Career Mentor - Streamlit App
cd /d "%~dp0"
echo ====================================================
echo       AI Career Mentor Application Launcher
echo ====================================================
echo.

if exist .venv\Scripts\streamlit.exe (
    echo Starting Streamlit application via .venv...
    .\.venv\Scripts\streamlit.exe run app.py
) else (
    echo Starting Streamlit application via default Python...
    streamlit run app.py
)

pause
