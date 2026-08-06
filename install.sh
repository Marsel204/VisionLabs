#!/usr/bin/env bash
set -Eeuo pipefail

APP_NAME="traffic-annotator"
VERSION="0.1.0"
SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${TRAFFIC_ANNOTATOR_INSTALL_DIR:-${HOME}/.local/share/${APP_NAME}}"
BIN_DIR="${HOME}/.local/bin"
APPLICATIONS_DIR="${HOME}/.local/share/applications"
DESKTOP_FILE="${APPLICATIONS_DIR}/${APP_NAME}.desktop"
USE_SYSTEM_TORCH=0
SKIP_SYSTEM_DEPS=0

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<EOF
Usage: ./install.sh [options]

Installs ${APP_NAME} for the current user.

Options:
  --cpu-only           Install the normal PyPI PyTorch package, even on Jetson.
  --no-system-deps     Do not install Ubuntu runtime packages with apt.
  --install-dir PATH   Override ${INSTALL_DIR}.
  -h, --help           Show this help.
EOF
}

while (($#)); do
    case "$1" in
        --cpu-only) USE_SYSTEM_TORCH=0; FORCE_CPU=1; shift ;;
        --no-system-deps) SKIP_SYSTEM_DEPS=1; shift ;;
        --install-dir)
            (($# >= 2)) || die "--install-dir requires a path"
            INSTALL_DIR="$2"
            shift 2
            ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

FORCE_CPU="${FORCE_CPU:-0}"
[[ "${EUID}" -ne 0 ]] || die "run this installer as your normal user, not root"
command -v python3 >/dev/null || die "python3 is required"
PYTHON_BIN="$(command -v python3)"
PYTHON_VERSION="$(${PYTHON_BIN} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PYTHON_MAJOR="${PYTHON_VERSION%%.*}"
PYTHON_MINOR="${PYTHON_VERSION##*.}"
(( PYTHON_MAJOR > 3 || (PYTHON_MAJOR == 3 && PYTHON_MINOR >= 12) )) \
    || die "Python 3.12 or newer is required (found ${PYTHON_VERSION})"

if [[ "${SKIP_SYSTEM_DEPS}" -eq 0 ]]; then
    command -v apt-get >/dev/null || die "Ubuntu/Debian apt-get is required"
    command -v sudo >/dev/null || die "sudo is required to install system packages"
    printf 'Installing desktop and Python runtime packages...\n'
    sudo apt-get update
    sudo apt-get install -y curl python3-venv python3-dev libgl1 libglib2.0-0 \
        libxkbcommon-x11-0 libxcb-cursor0
fi

if ! command -v uv >/dev/null; then
    command -v curl >/dev/null || die "curl is required to install uv (or install uv manually)"
    printf 'Installing uv in ~/.local/bin...\n'
    mkdir -p "${BIN_DIR}"
    curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="${BIN_DIR}" sh
    export PATH="${BIN_DIR}:${PATH}"
fi
command -v uv >/dev/null || die "uv could not be installed"

JETSON=0
if [[ "${FORCE_CPU}" -eq 0 ]] && command -v dpkg-query >/dev/null \
    && dpkg-query -W -f='${Status}' nvidia-l4t-core 2>/dev/null | grep -q 'install ok installed'; then
    JETSON=1
fi

if [[ "${INSTALL_DIR}" == "${HOME}" || "${INSTALL_DIR}" == "/" ]]; then
    die "refusing unsafe install directory: ${INSTALL_DIR}"
fi
if [[ "${SOURCE_DIR}" == "${INSTALL_DIR}" ]]; then
    die "run install.sh from the source checkout, not from the installed copy"
fi

printf 'Installing %s %s into %s...\n' "${APP_NAME}" "${VERSION}" "${INSTALL_DIR}"
rm -rf -- "${INSTALL_DIR}.new"
mkdir -p "${INSTALL_DIR}.new"
tar --exclude='./.venv' --exclude='./.git' --exclude='__pycache__' \
    -C "${SOURCE_DIR}" -cf - . | tar -C "${INSTALL_DIR}.new" -xf -
if [[ -e "${INSTALL_DIR}" ]]; then
    rm -rf -- "${INSTALL_DIR}.previous"
    mv -- "${INSTALL_DIR}" "${INSTALL_DIR}.previous"
fi
if ! mv -- "${INSTALL_DIR}.new" "${INSTALL_DIR}"; then
    if [[ -e "${INSTALL_DIR}.previous" ]]; then
        mv -- "${INSTALL_DIR}.previous" "${INSTALL_DIR}"
    fi
    die "could not replace ${INSTALL_DIR}"
fi
rm -rf -- "${INSTALL_DIR}.previous"

if [[ "${JETSON}" -eq 1 ]]; then
    printf 'Jetson/L4T detected; keeping the system CUDA-enabled PyTorch.\n'
    if ! "${PYTHON_BIN}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
        die "Jetson mode requires CUDA-enabled PyTorch importable by ${PYTHON_BIN}; install the NVIDIA Jetson PyTorch wheel for this JetPack/Python version or rerun with --cpu-only"
    fi
    uv venv --python "${PYTHON_BIN}" --system-site-packages "${INSTALL_DIR}/.venv"
    uv sync --directory "${INSTALL_DIR}" --locked --no-dev --no-install-package torch
else
    uv sync --directory "${INSTALL_DIR}" --locked --no-dev
fi

if [[ "${JETSON}" -eq 1 ]]; then
    TORCH_CHECK='import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'
else
    TORCH_CHECK='import torch'
fi
if ! "${INSTALL_DIR}/.venv/bin/python" -c "${TORCH_CHECK}" >/dev/null 2>&1; then
    die "PyTorch is not importable in ${INSTALL_DIR}/.venv; on Jetson, verify that the NVIDIA PyTorch wheel matches ${PYTHON_VERSION}"
fi
"${INSTALL_DIR}/.venv/bin/python" -c 'import PySide6, torch; print(f"PySide6 {PySide6.__version__}; PyTorch {torch.__version__}; CUDA {torch.cuda.is_available()}")'

mkdir -p "${BIN_DIR}" "${APPLICATIONS_DIR}"
cp -- "${SOURCE_DIR}/uninstall.sh" "${INSTALL_DIR}/uninstall.sh"
chmod 755 "${INSTALL_DIR}/uninstall.sh"
cat > "${BIN_DIR}/${APP_NAME}" <<EOF
#!/usr/bin/env bash
exec "${INSTALL_DIR}/.venv/bin/traffic-annotator" "\$@"
EOF
chmod 755 "${BIN_DIR}/${APP_NAME}"
cat > "${DESKTOP_FILE}" <<EOF
[Desktop Entry]
Type=Application
Name=Traffic Annotator
Comment=AI-assisted traffic image annotation
Exec=${BIN_DIR}/${APP_NAME}
Icon=applications-graphics
Terminal=false
Categories=Graphics;Education;
StartupWMClass=traffic-annotator
EOF
chmod 644 "${DESKTOP_FILE}"

printf '\nInstalled successfully. Launch with: %s\n' "${BIN_DIR}/${APP_NAME}"
printf 'The desktop entry is: %s\n' "${DESKTOP_FILE}"
if [[ "${JETSON}" -eq 1 ]]; then
    printf 'Jetson mode is active; CUDA availability was checked above.\n'
fi
