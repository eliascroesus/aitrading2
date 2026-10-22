#!/bin/bash
# HIVEMIND launcher for macOS. Double-click to set up (first time) and start.
# Everything it creates stays inside this folder: delete the folder to remove it all.
cd "$(dirname "$0")" || exit 1
export PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

echo ""
echo "  HIVEMIND"
echo "  Everything lives inside this folder. To remove it all: close this window, then delete the folder."
echo ""

pause_exit() { echo ""; read -r -p "Press Enter to close this window."; exit "${1:-1}"; }

PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
    PY="$c"
    break
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3.10 or newer is needed. Opening the download page..."
  echo "Install Python, then double-click this file again."
  open "https://www.python.org/downloads/" >/dev/null 2>&1
  pause_exit 1
fi

if [ ! -f .env ]; then
  echo "First-time setup. Paste each key (Cmd+V) and press Enter."
  echo "Leave a key empty to skip it (without keys the hive runs in demo mode)."
  echo ""
  read -r -p "Anthropic API key (sk-ant-...): " AK
  read -r -p "Alpaca PAPER API key (PK...): " PK
  read -r -p "Alpaca PAPER secret key: " SK
  read -r -p "Stop Claude spending after how many dollars? [20]: " BUD
  AK="$(printf '%s' "$AK" | tr -d '[:space:]')"
  PK="$(printf '%s' "$PK" | tr -d '[:space:]')"
  SK="$(printf '%s' "$SK" | tr -d '[:space:]')"
  BUD="$(printf '%s' "${BUD:-20}" | tr -d '[:space:]$')"
  {
    echo "ANTHROPIC_API_KEY=$AK"
    echo "ALPACA_API_KEY=$PK"
    echo "ALPACA_SECRET_KEY=$SK"
    echo "HIVE_LLM_BUDGET_USD=$BUD"
  } > .env
  chmod 600 .env
  echo ""
  echo "Saved to the .env file in this folder (open it with TextEdit to change keys later)."
  echo ""
fi

if [ ! -x .venv/bin/python ]; then
  echo "Creating a private Python environment (.venv) inside this folder..."
  "$PY" -m venv .venv || { echo "Could not create the Python environment."; pause_exit 1; }
fi
if ! cmp -s requirements.txt .venv/requirements.installed; then
  echo "Installing what HIVEMIND needs (first run takes a few minutes)..."
  .venv/bin/python -m pip install --quiet --upgrade pip
  if .venv/bin/python -m pip install -r requirements.txt; then
    cp requirements.txt .venv/requirements.installed
  else
    echo "Install failed. Check your internet connection and double-click again."
    pause_exit 1
  fi
fi

echo ""
echo "Starting HIVEMIND. The dashboard opens in your browser (http://127.0.0.1:8000)."
echo "Keep this window open while it runs. Press Ctrl+C or close the window to stop it."
echo ""
HIVE_OPEN_BROWSER=1 .venv/bin/python -m hive
echo ""
echo "HIVEMIND stopped."
pause_exit 0
