#!/usr/bin/env bash
set -Eeuo pipefail

APP_NAME="traffic-annotator"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${TRAFFIC_ANNOTATOR_INSTALL_DIR:-${HOME}/.local/share/${APP_NAME}}"
BIN_FILE="${HOME}/.local/bin/${APP_NAME}"
DESKTOP_FILE="${HOME}/.local/share/applications/${APP_NAME}.desktop"

if [[ "${SCRIPT_DIR}" != "${INSTALL_DIR}" ]]; then
    printf 'This is the source-tree uninstall script. Target: %s\n' "${INSTALL_DIR}"
fi
rm -f -- "${BIN_FILE}" "${DESKTOP_FILE}"
if [[ -d "${INSTALL_DIR}" && "${INSTALL_DIR}" != "${HOME}" && "${INSTALL_DIR}" != "/" ]]; then
    rm -rf -- "${INSTALL_DIR}"
fi
printf '%s uninstalled. User datasets, cache, and logs were preserved.\n' "${APP_NAME}"
