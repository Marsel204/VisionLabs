: << "BATCH_BLOCK"
@echo off
setlocal enabledelayedexpansion
title VisionLab Studio

cd /d "%~dp0"

echo ========================================================
echo              VisionLab One-Click Launcher
echo ========================================================
echo.

REM 1. Check / Install uv
where uv >nul 2>nul
if !ERRORLEVEL! neq 0 (
    echo [SETUP] "uv" package manager not detected.
    echo Installing uv for Windows...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%USERPROFILE%\.local\bin;%LOCALAPPDATA%\bin;!PATH!"
    where uv >nul 2>nul
    if !ERRORLEVEL! neq 0 (
        echo [WARNING] Could not install uv automatically. Falling back to system Python...
    )
)

REM 2. Ensure virtual environment and dependencies exist
if not exist ".venv\Scripts\python.exe" (
    echo [SETUP] Initializing VisionLab environment (.venv)...
    where uv >nul 2>nul
    if !ERRORLEVEL! equ 0 (
        uv sync
    ) else (
        where python >nul 2>nul
        if !ERRORLEVEL! equ 0 (
            python -m venv .venv
            .venv\Scripts\python.exe -m pip install -e .
        ) else (
            echo [ERROR] Neither Python nor uv was found.
            echo Please install Python 3.12+ from https://python.org
            pause
            exit /b 1
        )
    )
)

REM 3. Resolve Python launcher
if exist ".venv\Scripts\python.exe" (
    set "PY_BIN=.venv\Scripts\python.exe"
) else (
    set "PY_BIN=uv run python"
)

REM 4. Mode Selection: --qt forces standalone Qt GUI; default launches Desktop Studio if npm is available
if "%1"=="--qt" (
    echo Launching VisionLab Standalone Qt GUI...
    shift
    %PY_BIN% -m app.main %*
    if !ERRORLEVEL! neq 0 pause
    exit /b !ERRORLEVEL!
)

where npm >nul 2>nul
if !ERRORLEVEL! equ 0 (
    if exist "desktop\package.json" (
        goto run_desktop_win
    )
)

REM Fallback to standalone Qt GUI if npm/desktop not available
echo Launching VisionLab Standalone Qt GUI...
%PY_BIN% -m app.main %*
if !ERRORLEVEL! neq 0 pause
exit /b !ERRORLEVEL!

:run_desktop_win
echo === VisionLab: Starting Desktop Studio ===
echo [1/2] Starting Python AI Engine on :8765...
start "VisionLab Backend" /B %PY_BIN% -m uvicorn app.api.server:app --port 8765 --host 127.0.0.1 --log-level warning

echo [2/2] Launching Tauri Desktop Studio...
cd desktop
if not exist "node_modules" (
    echo Installing desktop UI dependencies...
    call npm install
)
call npm run tauri dev
if !ERRORLEVEL! neq 0 (
    echo.
    echo [ERROR] VisionLab exited with an error.
    pause
)
exit /b !ERRORLEVEL!
BATCH_BLOCK

# ==========================================================
# LINUX / POSIX BASH SECTION
# ==========================================================
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "========================================================"
echo "             VisionLab One-Click Launcher"
echo "========================================================"
echo ""

# 1. Check / Install uv
if ! command -v uv >/dev/null 2>&1; then
    echo "[SETUP] 'uv' package manager not detected."
    echo "Installing uv for Linux..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# 2. Ensure virtual environment exists
if [ ! -f ".venv/bin/python" ]; then
    echo "[SETUP] Initializing VisionLab environment (.venv)..."
    if command -v uv >/dev/null 2>&1; then
        uv sync
    elif command -v python3 >/dev/null 2>&1; then
        python3 -m venv .venv
        .venv/bin/pip install -e .
    else
        echo "[ERROR] Python 3 or uv is required to run VisionLab."
        read -p "Press Enter to exit..."
        exit 1
    fi
fi

PY_BIN=".venv/bin/python"

# 3. Linux WebKit & Display fixes
export WEBKIT_DISABLE_DMABUF_RENDERER=1
export DISPLAY="${DISPLAY:-:1.0}"
if [ -z "$XAUTHORITY" ] || [ ! -f "$XAUTHORITY" ]; then
    LATEST_XAUTH=$(ls -t /run/user/$(id -u)/xauth_* 2>/dev/null | head -1)
    [ -n "$LATEST_XAUTH" ] && export XAUTHORITY="$LATEST_XAUTH"
fi

# 4. Mode Selection: --qt forces standalone Qt GUI; default launches Desktop Studio if npm is available
if [ "$1" = "--qt" ]; then
    shift
    echo "Launching VisionLab Standalone Qt GUI..."
    $PY_BIN -m app.main "$@" || {
        echo "[ERROR] VisionLab exited with an error."
        read -p "Press Enter to exit..."
    }
    exit 0
fi

if command -v npm >/dev/null 2>&1 && [ -f "desktop/package.json" ]; then
    echo "=== VisionLab: Starting Desktop Studio ==="
    if ! curl -s http://127.0.0.1:8765/api/health > /dev/null 2>&1; then
        echo "[1/2] Launching Python AI Engine in background..."
        $PY_BIN -m uvicorn app.api.server:app --port 8765 --host 127.0.0.1 --log-level warning &
        API_PID=$!
        trap "kill $API_PID 2>/dev/null || true" EXIT

        for i in {1..30}; do
            if curl -s http://127.0.0.1:8765/api/health > /dev/null 2>&1; then
                echo "      ✓ Python AI Engine online on :8765"
                break
            fi
            sleep 0.5
        done
    else
        echo "[1/2] ✓ Python AI Engine is already running on :8765"
    fi

    echo "[2/2] Launching Tauri v2 Desktop Studio..."
    cd desktop
    if [ ! -d "node_modules" ]; then
        echo "Installing desktop UI dependencies..."
        npm install
    fi
    ./node_modules/.bin/tauri dev || {
        echo ""
        echo "[ERROR] Desktop studio exited with an error."
        read -p "Press Enter to exit..."
    }
    exit 0
fi

# Fallback to standalone Qt GUI if npm/tauri is not available
echo "Launching VisionLab Standalone Qt GUI..."
$PY_BIN -m app.main "$@" || {
    echo "[ERROR] VisionLab exited with an error."
    read -p "Press Enter to exit..."
}
exit 0

