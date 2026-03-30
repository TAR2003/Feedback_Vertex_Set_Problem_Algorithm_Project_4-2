#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
  echo "Do not run with sudo."
  echo "Use your normal user so CMake uses your active Python environment (with pybind11 installed)."
  echo "Example: ./build_cpp.sh --clean --jobs \"\$(nproc)\""
  exit 1
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: ./build_cpp.sh [--clean] [--install] [--build-type TYPE] [--jobs N]

Examples:
  ./build_cpp.sh
  ./build_cpp.sh --clean --jobs "$(nproc)"
  ./build_cpp.sh --install
EOF
  exit 0
fi

python3 build_engine.py "$@"
