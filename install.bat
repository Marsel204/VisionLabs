@echo off
setlocal enabledelayedexpansion

echo =======================================================
echo     Traffic Annotator - Windows Installer Wrapper
echo =======================================================
echo.

cd /d "%~dp0"

where powershell >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] PowerShell is required to run the installer.
    pause
    exit /b 1
)

echo Launching PowerShell installation script...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Installation failed.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [DONE] Setup finished.
pause
