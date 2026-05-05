#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Sybil Desktop App — Python Environment Setup
# Sybil supports Python 3.8–3.12. This script auto-detects a
# compatible version from common locations on macOS/Linux.
# ─────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   Sybil Risk Predictor — Environment Setup   ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ── Find a compatible Python (3.8–3.12) ─────────────────────────
find_compatible_python() {
  # Explicit version binaries first (most reliable)
  local candidates=(
    python3.10
    python3.9
    python3.8
    python3.11
    python3.12
    /usr/bin/python3.12
    /usr/bin/python3.11
    /usr/bin/python3.10
    /usr/bin/python3.9
    /usr/bin/python3.8
    /usr/local/bin/python3.12
    /usr/local/bin/python3.11
    /usr/local/bin/python3.10
    /usr/local/bin/python3.9
    /usr/local/bin/python3.8
    /opt/homebrew/bin/python3.10
    /opt/homebrew/bin/python3.9
    /opt/homebrew/bin/python3.8
    /opt/homebrew/bin/python3.11
    /opt/homebrew/bin/python3.12
    /opt/homebrew/bin/python3.9
    /opt/homebrew/bin/python3.8
    # pyenv shims
    "$HOME/.pyenv/shims/python3.10"
    "$HOME/.pyenv/shims/python3.9"
    "$HOME/.pyenv/shims/python3.8"
    # conda envs
    "$HOME/miniconda3/bin/python3.10"
    "$HOME/miniconda3/bin/python3.9"
    "$HOME/anaconda3/bin/python3.10"
    "$HOME/anaconda3/bin/python3.9"
  )

  for py in "${candidates[@]}"; do
    if command -v "$py" &>/dev/null 2>&1 || [ -x "$py" ]; then
      local ver
      ver=$("$py" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
      local major minor
      major=$(echo "$ver" | cut -d. -f1)
      minor=$(echo "$ver" | cut -d. -f2)
      if [ "$major" = "3" ] && [ "$minor" -ge 8 ] && [ "$minor" -le 12 ]; then
        echo "$py"
        return 0
      fi
    fi
  done
  return 1
}

PYTHON=$(find_compatible_python || true)

if [ -z "$PYTHON" ]; then
  echo "❌  Could not find Python 3.8–3.12."
  echo ""
  echo "  Sybil requires Python 3.8–3.12 (your system has a newer version)."
  echo ""
  echo "  Install a compatible version with one of:"
  echo "    brew install python@3.10        # Homebrew (recommended)"
  echo "    pyenv install 3.10.14           # pyenv"
  echo ""
  echo "  Then re-run:  bash setup.sh"
  exit 1
fi

PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
echo "✓  Found compatible Python $PY_VERSION at: $PYTHON"

# ── Virtual environment ───────────────────────────────────────
if [ -d ".venv" ]; then
  # Check if existing venv is compatible and not broken (e.g. moved from another path)
  EXISTING_VER=$(.venv/bin/python -c "import sys; print(f'{sys.version_info.minor}')" 2>/dev/null || echo "0")
  PIP_OK=$(.venv/bin/python -m pip --version &>/dev/null 2>&1 && echo "yes" || echo "no")
  if [ "$EXISTING_VER" -lt 8 ] || [ "$EXISTING_VER" -gt 12 ] || [ "$PIP_OK" = "no" ]; then
    echo "→  Existing .venv is broken or incompatible, recreating..."
    rm -rf .venv
  else
    echo "✓  Existing .venv is compatible (Python 3.$EXISTING_VER)"
  fi
fi

if [ ! -d ".venv" ]; then
  echo "→  Creating virtual environment with Python $PY_VERSION..."
  "$PYTHON" -m venv .venv
fi

source .venv/bin/activate
ACTIVE_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
echo "✓  Virtual environment activated (Python $ACTIVE_VER)"

# ── Dependencies ──────────────────────────────────────────────
echo "→  Upgrading pip..."
python3 -m pip install --upgrade pip -q

echo "→  Installing Flask dependencies..."
python3 -m pip install flask flask-cors pillow waitress -q

echo "→  Installing Sybil from bundled source..."
python3 -m pip install -e ./sybil-source/ -q

echo ""
echo "✅  Python environment ready!"
echo ""
echo "Next steps:"
echo "   cd electron_app && npm install && npm start"
echo ""
