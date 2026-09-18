@echo off
setlocal enabledelayedexpansion

echo =======================================================
echo   Traffic Annotator - Windows Executable Builder
echo =======================================================
echo.

cd /d "%~dp0\.."

REM Check for Python
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.12 or newer from https://www.python.org/
    pause
    exit /b 1
)

REM Check if uv is available
where uv >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [*] Using uv to run the build script...
    uv run python scripts/build_windows_exe.py %*
) else (
    echo [*] Using system Python to run the build script...
    python scripts/build_windows_exe.py %*
)

if %ERRORLEVEL% equ 0 (
    echo.
    echo [SUCCESS] Standalone executable and zip package generated in dist\
    echo.
) else (
    echo.
    echo [FAILED] Build failed with exit code %ERRORLEVEL%.
    echo.
)

pause
