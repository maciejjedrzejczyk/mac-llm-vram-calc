#!/usr/bin/env bash
# Run the VRAM Calculator directly on the host.
# Requires: Python 3.10+, pip dependencies installed.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Create venv if it doesn't exist
if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$SCRIPT_DIR/.venv"
fi

# Activate and install deps
source "$SCRIPT_DIR/.venv/bin/activate"
pip install -q -r "$SCRIPT_DIR/requirements.txt"

echo "Starting VRAM Calculator at http://localhost:8501"
streamlit run "$SCRIPT_DIR/app.py"
