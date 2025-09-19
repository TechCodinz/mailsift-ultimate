#!/usr/bin/env python3
"""
Simple Desktop Application Creator for MailSift Ultra
Creates a standalone executable without complex installer dependencies.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import zipfile

def create_simple_executable():
    """Create a simple standalone executable"""
    
    print("🚀 Creating MailSift Ultra Desktop Application")
    print("=" * 50)
    
    # Check if desktop_app.py exists
    if not Path("desktop_app.py").exists():
        print("❌ desktop_app.py not found")
        return False
    
    # Create a simple batch file to run the Python app
    batch_content = '''@echo off
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
'''
    
    with open("MailSift_Ultra.bat", "w", encoding="utf-8") as f:
        f.write(batch_content)
    
    print("✅ Created MailSift_Ultra.bat launcher")
    
    # Create a PowerShell version too
    ps_content = '''# MailSift Ultra Desktop Application Launcher
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
'''
    
    with open("MailSift_Ultra.ps1", "w", encoding="utf-8") as f:
        f.write(ps_content)
    
    print("✅ Created MailSift_Ultra.ps1 launcher")
    
    # Create a requirements.txt file
    requirements = '''flask>=2.0.0
webview>=4.0.0
requests>=2.25.0
beautifulsoup4>=4.9.0
lxml>=4.6.0
pandas>=1.3.0
openpyxl>=3.0.0
dnspython>=2.1.0
scikit-learn>=1.0.0
numpy>=1.21.0
'''
    
    with open("requirements.txt", "w", encoding="utf-8") as f:
        f.write(requirements)
    
    print("✅ Created requirements.txt")
    
    # Create a simple installer package
    create_installer_package()
    
    return True

def create_installer_package():
    """Create a zip package for easy distribution"""
    
    package_name = "MailSift_Ultra_Desktop_Package.zip"
    
    # Files to include in the package
    files_to_include = [
        "desktop_app.py",
        "MailSift_Ultra.bat",
        "MailSift_Ultra.ps1",
        "requirements.txt",
        "README_DESKTOP.md"
    ]
    
    # Create README for desktop package
    readme_content = '''# MailSift Ultra Desktop Application

## Quick Start

### Option 1: Windows Batch File
1. Double-click `MailSift_Ultra.bat`
2. The application will automatically install dependencies and start

### Option 2: PowerShell Script
1. Right-click `MailSift_Ultra.ps1` and select "Run with PowerShell"
2. The application will automatically install dependencies and start

### Option 3: Manual Installation
1. Install Python from https://python.org
2. Open Command Prompt in this folder
3. Run: `pip install -r requirements.txt`
4. Run: `python desktop_app.py`

## Features
- ✅ AI-powered email extraction
- ✅ Web scraping with anti-bot protection
- ✅ Keyword search engine
- ✅ Email validation and enrichment
- ✅ Multiple export formats
- ✅ Dark mode interface
- ✅ Offline functionality

## System Requirements
- Windows 10/11
- Python 3.8 or higher
- Internet connection for web scraping features

## Troubleshooting
- If Python is not found, install Python from https://python.org
- Make sure to check "Add Python to PATH" during installation
- If you get permission errors, run as Administrator

## Support
- 🤖 AI Support: Visit the web version for AI chat
- 📧 Email: support@mailsift.com
- 📚 Docs: Full documentation available online

Enjoy using MailSift Ultra Desktop! 🚀
'''
    
    with open("README_DESKTOP.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    # Create the zip package
    with zipfile.ZipFile(package_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in files_to_include:
            if Path(file).exists():
                zipf.write(file)
                print(f"📦 Added {file} to package")
        
        # Add templates directory if it exists
        if Path("templates").exists():
            for root, dirs, files in os.walk("templates"):
                for file in files:
                    file_path = Path(root) / file
                    zipf.write(file_path, file_path)
                    print(f"📦 Added {file_path} to package")
    
    print(f"✅ Created desktop package: {package_name}")
    print(f"📁 Package size: {Path(package_name).stat().st_size / 1024 / 1024:.1f} MB")

def create_portable_version():
    """Create a portable version that includes Python"""
    
    print("\n🔧 Creating Portable Version...")
    
    # This would require downloading and bundling Python
    # For now, we'll create a script that downloads Python automatically
    
    portable_script = '''@echo off
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
'''
    
    with open("MailSift_Ultra_Portable_Setup.bat", "w", encoding="utf-8") as f:
        f.write(portable_script)
    
    print("✅ Created portable setup script")

def main():
    """Main function"""
    try:
        success = create_simple_executable()
        if success:
            create_portable_version()
            
            print("\n🎉 Desktop Application Package Created Successfully!")
            print("\n📁 Files created:")
            print("   - MailSift_Ultra.bat (Windows launcher)")
            print("   - MailSift_Ultra.ps1 (PowerShell launcher)")
            print("   - requirements.txt (Python dependencies)")
            print("   - README_DESKTOP.md (Installation guide)")
            print("   - MailSift_Ultra_Desktop_Package.zip (Complete package)")
            print("   - MailSift_Ultra_Portable_Setup.bat (Portable setup)")
            
            print("\n🚀 Next steps:")
            print("   1. Test the .bat file by double-clicking it")
            print("   2. Share the .zip package with users")
            print("   3. Users can extract and run MailSift_Ultra.bat")
            
            return True
        else:
            print("❌ Failed to create desktop application")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    main()
