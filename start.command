#!/usr/bin/env bash
# macOS launcher for the labeling tool. Double-click this file in Finder,
# or run: ./start.command

set -e
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Prefer a project-local virtual environment when one exists.
if [[ -x ".venv/bin/python" ]]; then
  exec ".venv/bin/python" main.py "$@"
elif command -v python3 >/dev/null 2>&1; then
  exec python3 main.py "$@"
else
  exec python main.py "$@"
fi
