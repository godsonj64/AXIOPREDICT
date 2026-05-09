#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# AXIO PREDICT — Python Environment Setup
# Sybil's setup.cfg pins python_requires = >=3.8,<3.11, so only
# Python 3.8, 3.9, or 3.10 work. This script auto-detects one of
# those and installs all dependencies into .venv/.
# All pip calls use .venv/bin/python directly — no reliance on
# shell activation or the system `python3` command.
# ─────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║      AXIO PREDICT — Environment Setup        ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ── Find a compatible Python (3.8–3.10 only) ─────────────────
# Sybil's setup.cfg: python_requires = >=3.8,<3.11
find_compatible_python() {
  local candidates=(
    python3.10
    python3.9
    python3.8
    /opt/homebrew/bin/python3.10
    /opt/homebrew/bin/python3.9
    /opt/homebrew/bin/python3.8
    /usr/local/bin/python3.10
    /usr/local/bin/python3.9
    /usr/local/bin/python3.8
    /usr/bin/python3.10
    /usr/bin/python3.9
    /usr/bin/python3.8
    "$HOME/.pyenv/shims/python3.10"
    "$HOME/.pyenv/shims/python3.9"
    "$HOME/.pyenv/shims/python3.8"
    "$HOME/miniconda3/bin/python3.10"
    "$HOME/miniconda3/bin/python3.9"
    "$HOME/miniconda3/bin/python3.8"
    "$HOME/anaconda3/bin/python3.10"
    "$HOME/anaconda3/bin/python3.9"
    "$HOME/anaconda3/bin/python3.8"
  )

  for py in "${candidates[@]}"; do
    if command -v "$py" &>/dev/null 2>&1 || [ -x "$py" ]; then
      local ver major minor
      ver=$("$py" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null) || continue
      major=$(echo "$ver" | cut -d. -f1)
      minor=$(echo "$ver" | cut -d. -f2)
      if [ "$major" = "3" ] && [ "$minor" -ge 8 ] && [ "$minor" -le 10 ]; then
        echo "$py"
        return 0
      fi
    fi
  done
  return 1
}

PYTHON=$(find_compatible_python || true)

if [ -z "$PYTHON" ]; then
  echo "❌  Could not find Python 3.8, 3.9, or 3.10."
  echo ""
  echo "  Sybil's setup.cfg pins python_requires = >=3.8,<3.11,"
  echo "  so 3.11+ will not work. Your system default python3"
  echo "  may be newer (3.11/3.12/3.13/3.14)."
  echo ""
  echo "  Install a compatible version:"
  echo "    brew install python@3.10        # Homebrew (recommended)"
  echo "    pyenv install 3.10.14           # pyenv"
  echo ""
  echo "  Then re-run:  bash setup.sh"
  exit 1
fi

PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
echo "✓  Found compatible Python $PY_VERSION at: $PYTHON"

# ── Virtual environment ───────────────────────────────────────
VENV_PYTHON=".venv/bin/python"

if [ -d ".venv" ]; then
  # Validate: check the venv's own python binary directly (not `python3`)
  EXISTING_MINOR=$("$VENV_PYTHON" -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "-1")
  EXISTING_MAJOR=$("$VENV_PYTHON" -c "import sys; print(sys.version_info.major)" 2>/dev/null || echo "0")
  PIP_OK=$("$VENV_PYTHON" -m pip --version &>/dev/null 2>&1 && echo "yes" || echo "no")

  if [ "$EXISTING_MAJOR" != "3" ] || \
     [ "$EXISTING_MINOR" -lt 8 ] || \
     [ "$EXISTING_MINOR" -gt 10 ] || \
     [ "$PIP_OK" = "no" ]; then
    echo "→  Existing .venv is broken or incompatible (Python $EXISTING_MAJOR.$EXISTING_MINOR), recreating..."
    rm -rf .venv
  else
    echo "✓  Existing .venv is compatible (Python $EXISTING_MAJOR.$EXISTING_MINOR)"
  fi
fi

if [ ! -d ".venv" ]; then
  echo "→  Creating virtual environment with Python $PY_VERSION..."
  "$PYTHON" -m venv .venv
fi

# Confirm the venv python — use .venv/bin/python directly, never the shell's python3
ACTIVE_VER=$("$VENV_PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
echo "✓  Using venv Python $ACTIVE_VER  ($VENV_PYTHON)"

# ── Dependencies ──────────────────────────────────────────────
echo "→  Upgrading pip..."
"$VENV_PYTHON" -m pip install --upgrade pip -q

echo "→  Installing Flask dependencies..."
"$VENV_PYTHON" -m pip install flask flask-cors pillow waitress -q

echo "→  Installing Sybil from bundled source..."
"$VENV_PYTHON" -m pip install -e ./sybil-source/ -q

echo ""
echo "✅  Python environment ready!"
echo ""
echo "Next steps:"
echo "   cd electron_app && npm install && npm start"
echo ""
