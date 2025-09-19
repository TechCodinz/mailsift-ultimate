# MailSift Ultra Desktop Application Launcher
Write-Host "Starting MailSift Ultra..." -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Python is not installed or not in PATH." -ForegroundColor Red
    Write-Host "Please install Python from https://python.org" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Install required packages
Write-Host "Installing required packages..." -ForegroundColor Yellow
pip install flask webview requests beautifulsoup4 lxml pandas openpyxl dnspython scikit-learn numpy --quiet

# Run the application
Write-Host "Starting MailSift Ultra..." -ForegroundColor Green
python desktop_app.py

Read-Host "Press Enter to exit"
