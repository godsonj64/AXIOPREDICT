#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
#  AXIO PREDICT v2.0.0 — Cross-platform build (Mac Universal + Win)
#  Run this on macOS to produce both Mac and Windows builds.
#  On Linux/Windows, use the platform-specific scripts instead.
# ─────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ELECTRON_DIR="$SCRIPT_DIR/electron_app"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  AXIO PREDICT — Full Cross-Platform Builder          ║"
echo "║  Mac Universal (arm64+x86_64) + Windows (x64+x86)   ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

if ! command -v node &>/dev/null; then
  echo "✗ Node.js not found. Install from https://nodejs.org"
  exit 1
fi

cd "$ELECTRON_DIR"
echo "▶ Installing Electron dependencies..."
npm install --prefer-offline 2>&1 | grep -v "^npm warn" || true

echo ""
echo "▶ Building all platforms..."
echo "  (Mac Universal + Windows — requires Wine for Win build on macOS)"
echo ""
npm run dist:all 2>&1

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║                   Build Complete!                    ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "Artifacts in: $ELECTRON_DIR/dist/"
ls -lh "$ELECTRON_DIR/dist/" 2>/dev/null
