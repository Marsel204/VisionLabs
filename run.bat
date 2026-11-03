: << "BATCH_BLOCK"
@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    where uv >nul 2>nul
    if errorlevel 1 (
        echo Install uv, then run uv sync before launching Anotara.
        exit /b 1
    )
    call uv sync --locked
    if errorlevel 1 exit /b 1
)
".venv\Scripts\python.exe" -m app.main %*
exit /b %errorlevel%
BATCH_BLOCK

set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
if [ ! -x .venv/bin/python ]; then
    if ! command -v uv >/dev/null 2>&1; then
        printf 'Install uv, then run uv sync before launching Anotara.\n' >&2
        exit 1
    fi
    uv sync --locked
fi
exec .venv/bin/python -m app.main "$@"
