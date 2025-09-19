@echo off
title MailSift Ultra Portable Setup
echo MailSift Ultra Portable Setup
echo =============================
echo.

REM Check if Python is already installed
python --version >nul 2>&1
if not errorlevel 1 (
    echo Python is already installed.
    goto :run_app
)

echo Python not found. Setting up portable Python...
echo.

REM Download and install Python (this is a simplified version)
echo Please download Python from https://python.org
echo Make sure to check "Add Python to PATH" during installation
echo.
echo After installing Python, run this script again.
pause
exit /b 0

:run_app
echo Installing MailSift Ultra...
pip install flask webview requests beautifulsoup4 lxml pandas openpyxl dnspython scikit-learn numpy

echo Starting MailSift Ultra...
python desktop_app.py

pause
