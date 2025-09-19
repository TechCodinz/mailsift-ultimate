@echo off
title MailSift Ultra Desktop Application
echo Starting MailSift Ultra...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH.
    echo Please install Python from https://python.org
    pause
    exit /b 1
)

REM Install required packages if not already installed
echo Installing required packages...
pip install flask webview requests beautifulsoup4 lxml pandas openpyxl dnspython scikit-learn numpy >nul 2>&1

REM Run the application
echo Starting MailSift Ultra...
python desktop_app.py

pause
