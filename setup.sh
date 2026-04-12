#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

print_banner() {
  echo "============================================================"
  echo " PRML Speech Denoising Project Setup"
  echo "============================================================"
}

check_python() {
  if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "[ERROR] ${PYTHON_BIN} not found. Install Python 3.10+ first."
    exit 1
  fi
}

create_venv() {
  if [[ ! -d "${VENV_DIR}" ]]; then
    echo "[INFO] Creating virtual environment at ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  else
    echo "[INFO] Reusing virtual environment at ${VENV_DIR}"
  fi
}

install_deps() {
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  echo "[INFO] Upgrading pip/setuptools/wheel"
  python -m pip install --upgrade pip setuptools wheel
  echo "[INFO] Installing project dependencies"
  python -m pip install -r "${PROJECT_ROOT}/requirements.txt"
  python -m pip install -e "${PROJECT_ROOT}"
}

print_next_steps() {
  cat <<'EOF'

[OK] Setup complete.

Run commands:
  source .venv/bin/activate
  python scripts/run_pca.py --help
  python scripts/run_resunet.py --help

Examples:
  python scripts/run_pca.py --num-samples 5 --mix-snr-db 0 --output-dir outputs/pca
  python scripts/run_resunet.py --num-samples 10 --epochs 8 --output-dir outputs/resunet

Notes:
  - The first run downloads LibriSpeech and UrbanSound8K through kagglehub.
  - If CUDA is available, PyTorch automatically uses GPU.
EOF
}

main() {
  print_banner
  check_python
  create_venv
  install_deps
  print_next_steps
}

main "$@"
