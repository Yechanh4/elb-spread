#!/usr/bin/env bash
# macOS / Linux: first run creates the venv and installs, then starts http://localhost:8780
set -e
cd "$(dirname "$0")"
PY=${PY:-python3.12}
command -v "$PY" >/dev/null 2>&1 || PY=python3
if [ ! -x .venv/bin/python ]; then
  "$PY" -m venv .venv
  .venv/bin/pip install -r requirements.txt
fi
exec .venv/bin/uvicorn app.main:app --port "${PORT:-8780}"
