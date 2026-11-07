#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON=""

if [[ -f "$SCRIPT_DIR/.venv/bin/python" ]]; then
    VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"
elif [[ -f "$SCRIPT_DIR/.venv-ai/bin/python" ]]; then
    VENV_PYTHON="$SCRIPT_DIR/.venv-ai/bin/python"
elif [[ -f "/opt/wastenot-ai/venv/bin/python" ]]; then
    VENV_PYTHON="/opt/wastenot-ai/venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    VENV_PYTHON="$(command -v python3)"
fi

if [[ -z "$VENV_PYTHON" ]]; then
    echo "Error: Python environment tidak ditemukan." >&2
    exit 1
fi

export PYTHONPATH="$SCRIPT_DIR/src:${PYTHONPATH:-}"
echo "==> Menjalankan WasteNot AI Camera Demo..."
exec "$VENV_PYTHON" -m wastenot_ai.cli demo --demo-guides "$@"
