#!/usr/bin/env bash
# VisionLab - Modern Desktop Studio Launcher (Tauri v2 + React)

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# Linux NVIDIA WebKitGTK fix & Display export
export WEBKIT_DISABLE_DMABUF_RENDERER=1
export DISPLAY="${DISPLAY:-:1.0}"
if [ -z "$XAUTHORITY" ] || [ ! -f "$XAUTHORITY" ]; then
  LATEST_XAUTH=$(ls -t /run/user/$(id -u)/xauth_* 2>/dev/null | head -1)
  if [ -n "$LATEST_XAUTH" ]; then
    export XAUTHORITY="$LATEST_XAUTH"
  fi
fi

echo "=== VisionLab: Starting Desktop Studio ==="

# 1. Start Python AI API Server if not already running
if ! curl -s http://127.0.0.1:8765/api/health > /dev/null 2>&1; then
  echo "[1/2] Launching Python AI Engine (.venv)..."
  .venv/bin/python -m uvicorn app.api.server:app --port 8765 --host 127.0.0.1 --log-level warning &
  API_PID=$!
  trap "kill $API_PID 2>/dev/null || true" EXIT
  
  # Wait for healthcheck
  for i in {1..30}; do
    if curl -s http://127.0.0.1:8765/api/health > /dev/null 2>&1; then
      echo "      ✓ Python AI Engine is online on :8765"
      break
    fi
    sleep 0.5
  done
else
  echo "[1/2] ✓ Python AI Engine is already running on :8765"
fi

# 2. Launch Tauri v2 Desktop Application
echo "[2/2] Launching Tauri v2 Desktop Studio..."
cd desktop
./node_modules/.bin/tauri dev
