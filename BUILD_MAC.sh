#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
#  AXIO PREDICT v2.0.1 — macOS Universal Build Script
#  Produces: AXIO PREDICT-2.0.1-universal.dmg  (arm64 + x86_64)
#            AXIO PREDICT-2.0.1-universal-mac.zip
# ─────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ELECTRON_DIR="$SCRIPT_DIR/electron_app"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║     AXIO PREDICT — macOS Universal Builder      ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── 1. Check prerequisites ────────────────────────────────────────
echo "▶ Checking prerequisites..."

if ! command -v node &>/dev/null; then
  echo "✗ Node.js not found. Install from https://nodejs.org (v16+)"
  exit 1
fi
NODE_VER=$(node -e "process.stdout.write(process.versions.node)")
echo "  ✓ Node.js $NODE_VER"

if ! command -v npm &>/dev/null; then
  echo "✗ npm not found."
  exit 1
fi
echo "  ✓ npm $(npm --version)"

# ── 2. Install Electron dependencies ─────────────────────────────
echo ""
echo "▶ Installing Electron dependencies..."
cd "$ELECTRON_DIR"
npm install --prefer-offline 2>&1 | grep -v "^npm warn" || true
echo "  ✓ Dependencies installed"

# ── 3. Build Mac Universal ────────────────────────────────────────
echo ""
echo "▶ Building macOS Universal binary (arm64 + x86_64)..."
echo "  This may take 5–15 minutes depending on your machine."
echo ""
npm run dist:mac 2>&1

# ── 4. Report results ─────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║                 Build Complete!                  ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "Output files in: $ELECTRON_DIR/dist/"
echo ""
ls -lh "$ELECTRON_DIR/dist/"*.dmg "$ELECTRON_DIR/dist/"*.zip 2>/dev/null || ls -lh "$ELECTRON_DIR/dist/"
echo ""
echo "To install: open the .dmg and drag AXIO PREDICT to Applications."
echo ""
echo "NOTE: App is not code-signed. On first launch:"
echo "  → Right-click the app → Open → Open (to bypass Gatekeeper)"
echo "  Or: System Settings → Privacy & Security → Open Anyway"
