#!/usr/bin/env python3
"""
Desktop Installer Builder for MailSift Ultra
Creates executable installers for Windows, macOS, and Linux.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def create_desktop_app():
    """Create the desktop application executable"""
    
    # Check if PyInstaller is available
    try:
        subprocess.run(["python", "-m", "PyInstaller", "--version"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        if not run_command("pip install pyinstaller", "Installing PyInstaller"):
            return False
    
    # Create build directory
    build_dir = Path("build")
    dist_dir = Path("dist")
    
    if build_dir.exists():
        shutil.rmtree(build_dir)
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    
    # PyInstaller command for desktop app
    pyinstaller_cmd = [
        "python", "-m", "PyInstaller",
        "--onefile",  # Create single executable
        "--windowed",  # No console window (GUI only)
        "--name=MailSift_Ultra",
        "--hidden-import=tkinter",
        "--hidden-import=PIL",
        "--hidden-import=requests",
        "--hidden-import=sqlite3",
        "desktop_app.py"
    ]
    
    # Add icon if it exists and is valid
    if Path("mail_icon.ico").exists():
        try:
            from PIL import Image
            with Image.open("mail_icon.ico") as img:
                pyinstaller_cmd.append("--icon=mail_icon.ico")
        except:
            print("⚠️  Icon file exists but is invalid, skipping icon")
    
    # Convert to string for subprocess
    cmd_str = " ".join(pyinstaller_cmd)
    
    if not run_command(cmd_str, "Building desktop application"):
        return False
    
    return True

def create_windows_installer():
    """Create Windows installer using NSIS"""
    
    # Check if NSIS is installed
    try:
        subprocess.run(["makensis", "/VERSION"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ NSIS not found. Please install NSIS to create Windows installer.")
        print("Download from: https://nsis.sourceforge.io/Download")
        return False
    
    # Create Windows installer script
    nsis_script = """
; MailSift Ultra Windows Installer Script
!define APPNAME "MailSift Ultra"
!define COMPANYNAME "MailSift"
!define DESCRIPTION "AI-Powered Email Extraction Software"
!define VERSIONMAJOR 1
!define VERSIONMINOR 0
!define VERSIONBUILD 0
!define HELPURL "https://mailsift.com/support"
!define UPDATEURL "https://mailsift.com/updates"
!define ABOUTURL "https://mailsift.com"
!define INSTALLSIZE 50000

RequestExecutionLevel admin
InstallDir "$PROGRAMFILES\\${APPNAME}"
Name "${APPNAME}"
outFile "MailSift_Ultra_Installer.exe"

!include LogicLib.nsh

page directory
page instfiles

!macro VerifyUserIsAdmin
UserInfo::GetAccountType
pop $0
${If} $0 != "admin"
    messageBox mb_iconstop "Administrator rights required!"
    setErrorLevel 740
    quit
${EndIf}
!macroend

function .onInit
    setShellVarContext all
    !insertmacro VerifyUserIsAdmin
functionEnd

section "install"
    setOutPath $INSTDIR
    file "dist\\MailSift_Ultra.exe"
    file "mail_icon.ico"
    
    writeUninstaller "$INSTDIR\\uninstall.exe"
    
    createDirectory "$SMPROGRAMS\\${APPNAME}"
    createShortCut "$SMPROGRAMS\\${APPNAME}\\${APPNAME}.lnk" "$INSTDIR\\MailSift_Ultra.exe" "" "$INSTDIR\\mail_icon.ico"
    createShortCut "$DESKTOP\\${APPNAME}.lnk" "$INSTDIR\\MailSift_Ultra.exe" "" "$INSTDIR\\mail_icon.ico"
    
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "DisplayName" "${APPNAME}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "UninstallString" "$\\"$INSTDIR\\uninstall.exe$\\""
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "QuietUninstallString" "$\\"$INSTDIR\\uninstall.exe$\\" /S"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "InstallLocation" "$\\"$INSTDIR$\\""
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "DisplayIcon" "$\\"$INSTDIR\\mail_icon.ico$\\""
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "Publisher" "${COMPANYNAME}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "HelpLink" "${HELPURL}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "URLUpdateInfo" "${UPDATEURL}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "URLInfoAbout" "${ABOUTURL}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "DisplayVersion" "${VERSIONMAJOR}.${VERSIONMINOR}.${VERSIONBUILD}"
    WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "VersionMajor" ${VERSIONMAJOR}
    WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "VersionMinor" ${VERSIONMINOR}
    WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "NoModify" 1
    WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "NoRepair" 1
    WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "EstimatedSize" ${INSTALLSIZE}
sectionEnd

section "uninstall"
    delete "$INSTDIR\\MailSift_Ultra.exe"
    delete "$INSTDIR\\mail_icon.ico"
    delete "$INSTDIR\\uninstall.exe"
    
    delete "$SMPROGRAMS\\${APPNAME}\\${APPNAME}.lnk"
    delete "$DESKTOP\\${APPNAME}.lnk"
    
    rmDir "$SMPROGRAMS\\${APPNAME}"
    rmDir "$INSTDIR"
    
    DeleteRegKey HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}"
sectionEnd
"""
    
    # Write NSIS script
    with open("MailSift_Ultra_Installer.nsi", "w") as f:
        f.write(nsis_script)
    
    # Build installer
    if not run_command("makensis MailSift_Ultra_Installer.nsi", "Creating Windows installer"):
        return False
    
    print("✅ Windows installer created: MailSift_Ultra_Installer.exe")
    return True

def create_macos_installer():
    """Create macOS installer using pkgbuild"""
    
    # Check if we're on macOS
    if sys.platform != "darwin":
        print("❌ macOS installer can only be created on macOS")
        return False
    
    # Create app bundle structure
    app_name = "MailSift Ultra.app"
    app_path = Path(app_name)
    
    if app_path.exists():
        shutil.rmtree(app_path)
    
    # Create app bundle directories
    (app_path / "Contents" / "MacOS").mkdir(parents=True)
    (app_path / "Contents" / "Resources").mkdir(parents=True)
    
    # Copy executable
    shutil.copy("dist/MailSift_Ultra", app_path / "Contents" / "MacOS" / "MailSift_Ultra")
    
    # Create Info.plist
    info_plist = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>MailSift_Ultra</string>
    <key>CFBundleIdentifier</key>
    <string>com.mailsift.ultra</string>
    <key>CFBundleName</key>
    <string>MailSift Ultra</string>
    <key>CFBundleVersion</key>
    <string>1.0.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleSignature</key>
    <string>????</string>
    <key>CFBundleIconFile</key>
    <string>mail_icon.icns</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.15</string>
</dict>
</plist>"""
    
    with open(app_path / "Contents" / "Info.plist", "w") as f:
        f.write(info_plist)
    
    # Copy icon (convert ico to icns if needed)
    if Path("mail_icon.ico").exists():
        # For now, just copy the ico file
        shutil.copy("mail_icon.ico", app_path / "Contents" / "Resources" / "mail_icon.ico")
    
    # Make executable
    os.chmod(app_path / "Contents" / "MacOS" / "MailSift_Ultra", 0o755)
    
    print("✅ macOS app bundle created: MailSift Ultra.app")
    return True

def create_linux_installer():
    """Create Linux AppImage"""
    
    # Check if we're on Linux
    if sys.platform not in ["linux", "linux2"]:
        print("❌ Linux installer can only be created on Linux")
        return False
    
    # Create AppDir structure
    appdir = Path("MailSift_Ultra.AppDir")
    if appdir.exists():
        shutil.rmtree(appdir)
    
    appdir.mkdir()
    (appdir / "usr" / "bin").mkdir(parents=True)
    (appdir / "usr" / "share" / "applications").mkdir(parents=True)
    (appdir / "usr" / "share" / "icons" / "hicolor" / "256x256" / "apps").mkdir(parents=True)
    
    # Copy executable
    shutil.copy("dist/MailSift_Ultra", appdir / "usr" / "bin" / "mailsift-ultra")
    os.chmod(appdir / "usr" / "bin" / "mailsift-ultra", 0o755)
    
    # Create desktop file
    desktop_file = """[Desktop Entry]
Version=1.0
Type=Application
Name=MailSift Ultra
Comment=AI-Powered Email Extraction Software
Exec=mailsift-ultra
Icon=mailsift-ultra
Terminal=false
Categories=Office;Utility;
StartupNotify=true
"""
    
    with open(appdir / "usr" / "share" / "applications" / "mailsift-ultra.desktop", "w") as f:
        f.write(desktop_file)
    
    # Create AppRun
    apprun = """#!/bin/bash
HERE="$(dirname "$(readlink -f "${0}")")"
exec "${HERE}"/usr/bin/mailsift-ultra "$@"
"""
    
    with open(appdir / "AppRun", "w") as f:
        f.write(apprun)
    os.chmod(appdir / "AppRun", 0o755)
    
    # Create AppImage metadata
    appstream = """<?xml version="1.0" encoding="UTF-8"?>
<component type="desktop-application">
  <id>com.mailsift.ultra</id>
  <metadata_license>MIT</metadata_license>
  <project_license>MIT</project_license>
  <name>MailSift Ultra</name>
  <summary>AI-Powered Email Extraction Software</summary>
  <description>
    <p>MailSift Ultra is a professional email extraction and validation tool powered by artificial intelligence.</p>
  </description>
  <launchable type="desktop-id">mailsift-ultra.desktop</launchable>
  <screenshots>
    <screenshot type="default">
      <caption>Main Application Window</caption>
    </screenshot>
  </screenshots>
  <url type="homepage">https://mailsift.com</url>
  <url type="bugtracker">https://mailsift.com/support</url>
  <url type="help">https://mailsift.com/docs</url>
</component>"""
    
    (appdir / "usr" / "share" / "metainfo").mkdir(parents=True)
    with open(appdir / "usr" / "share" / "metainfo" / "com.mailsift.ultra.appdata.xml", "w") as f:
        f.write(appstream)
    
    # Copy icon
    if Path("mail_icon.ico").exists():
        shutil.copy("mail_icon.ico", appdir / "usr" / "share" / "icons" / "hicolor" / "256x256" / "apps" / "mailsift-ultra.ico")
    
    print("✅ Linux AppDir created: MailSift_Ultra.AppDir")
    print("📝 To create AppImage, install appimagetool and run:")
    print("   appimagetool MailSift_Ultra.AppDir")
    
    return True

def create_icon():
    """Create application icon"""
    try:
        from PIL import Image, ImageDraw
        
        # Create a simple icon
        img = Image.new('RGBA', (64, 64), (0, 255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Draw a simple mail icon
        draw.rectangle([16, 16, 48, 48], outline=(0, 0, 0, 255), width=2)
        draw.polygon([(16, 16), (32, 28), (48, 16)], fill=(0, 255, 255, 255))
        
        # Save as ICO
        img.save("mail_icon.ico", format="ICO", sizes=[(64, 64), (32, 32), (16, 16)])
        print("✅ Created proper icon file")
        
    except ImportError:
        print("📝 PIL not available, creating placeholder icon")
        # Create a minimal ICO file header
        ico_data = b'\x00\x00\x01\x00\x01\x00\x10\x10\x00\x00\x01\x00\x20\x00\x68\x04\x00\x00\x16\x00\x00\x00'
        with open("mail_icon.ico", "wb") as f:
            f.write(ico_data)
        print("   Install Pillow for better icon support: pip install Pillow")

def main():
    """Main build process"""
    print("🚀 MailSift Ultra Desktop Installer Builder")
    print("=" * 50)
    
    # Check if desktop_app.py exists
    if not Path("desktop_app.py").exists():
        print("❌ desktop_app.py not found. Please ensure it exists.")
        return False
    
    # Create placeholder icon
    create_icon()
    
    # Build desktop application
    if not create_desktop_app():
        print("❌ Failed to build desktop application")
        return False
    
    # Create platform-specific installers
    success = True
    
    if sys.platform == "win32":
        print("\n🪟 Creating Windows installer...")
        if not create_windows_installer():
            success = False
    elif sys.platform == "darwin":
        print("\n🍎 Creating macOS installer...")
        if not create_macos_installer():
            success = False
    elif sys.platform in ["linux", "linux2"]:
        print("\n🐧 Creating Linux installer...")
        if not create_linux_installer():
            success = False
    else:
        print(f"❌ Unsupported platform: {sys.platform}")
        success = False
    
    if success:
        print("\n✅ Desktop installers created successfully!")
        print("\n📁 Output files:")
        if Path("dist").exists():
            for file in Path("dist").iterdir():
                print(f"   - {file}")
        if Path("MailSift_Ultra_Installer.exe").exists():
            print(f"   - MailSift_Ultra_Installer.exe")
        if Path("MailSift Ultra.app").exists():
            print(f"   - MailSift Ultra.app")
        if Path("MailSift_Ultra.AppDir").exists():
            print(f"   - MailSift_Ultra.AppDir")
    else:
        print("\n❌ Some installers failed to create")
    
    return success

if __name__ == "__main__":
    main()
