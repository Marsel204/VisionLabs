@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

REM Check for uv
where uv >nul 2>nul
if %ERRORLEVEL% equ 0 (
    uv run visionlab %*
    exit /b %ERRORLEVEL%
)

REM Check for virtualenv python
if exist ".venv\Scripts\visionlab.exe" (
    ".venv\Scripts\visionlab.exe" %*
    exit /b %ERRORLEVEL%
)
if exist ".venv\Scripts\traffic-annotator.exe" (
    ".venv\Scripts\traffic-annotator.exe" %*
    exit /b %ERRORLEVEL%
)

REM Check for system python
where python >nul 2>nul
if %ERRORLEVEL% equ 0 (
    python -m app.main %*
    exit /b %ERRORLEVEL%
)

echo [ERROR] Neither uv nor Python was found.
echo Please run install.bat to set up VisionLab.
pause
exit /b 1
