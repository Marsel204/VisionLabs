@echo off
setlocal

echo =======================================================
echo     Traffic Annotator - Windows Uninstaller Wrapper
echo =======================================================
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0uninstall.ps1" %*

pause
