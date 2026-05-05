# AXIO PREDICT v2.0.0 — Build Guide

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Node.js | ≥ 16 (LTS) | https://nodejs.org |
| npm | ≥ 8 | Included with Node.js |
| Python | 3.8–3.10 | https://python.org |

---

## Quick Start

### macOS — Universal DMG (Apple Silicon + Intel)

```bash
# 1. Set up Python environment (first time only)
bash setup.sh

# 2. Build the Universal DMG
bash BUILD_MAC.sh
```

**Output:** `electron_app/dist/AXIO PREDICT-2.0.0-universal.dmg`

Supports: macOS 10.15+ on both Apple Silicon (M1/M2/M3) and Intel Macs.

---

### Windows — Installer + Portable

```batch
REM 1. Set up Python environment (first time only)
setup.bat

REM 2. Build Windows installer
BUILD_WINDOWS.bat
```

**Output:**
- `electron_app/dist/AXIO PREDICT Setup 2.0.0.exe` — NSIS Installer (x64 + x86)
- `electron_app/dist/AXIO PREDICT 2.0.0.exe` — Portable (x64, no install needed)

---

### Build Both Platforms (from macOS)

```bash
bash BUILD_ALL.sh
```

> **Note:** Building Windows `.exe` on macOS requires [Wine](https://www.winehq.org/).  
> Easiest cross-platform workflow: build Mac on macOS, build Windows on Windows (or a Windows VM/CI).

---

## What Gets Bundled

The installer packages:
- **Electron app** — the UI shell
- **Python backend** — Flask server (`python_backend/server.py`)
- **Sybil source** — deep learning inference code (`sybil-source/`)

The bundled `.venv` virtual environment is **NOT** included — users run `setup.sh` / `setup.bat` once to create it locally.

---

## macOS Gatekeeper Note

The app is **not code-signed** (requires an Apple Developer ID certificate, $99/year).

First-launch workaround for users:
1. **Right-click** the app → **Open** → **Open**
2. Or: System Settings → **Privacy & Security** → **Open Anyway**

---

## Windows SmartScreen Note

The installer may trigger Windows SmartScreen (unsigned app).

Users click: **More info → Run anyway**

---

## Electron Builder Config Summary

| Platform | Format | Arch | Min OS |
|----------|--------|------|--------|
| macOS | DMG + ZIP | Universal (arm64 + x86_64) | macOS 10.15 |
| Windows | NSIS Installer | x64 + x86 | Windows 10 |
| Windows | Portable EXE | x64 | Windows 10 |

Config file: `electron_app/package.json` → `"build"` section.

---

## Troubleshooting

**`Python not found` during setup**
- macOS: `brew install python@3.10`
- Windows: Install from python.org, check "Add to PATH"

**`electron-builder` install fails**
```bash
cd electron_app && npm install
```

**macOS: `code sign` error during build**
The build config sets `gatekeeperAssess: false` which skips signing.  
If you have an Apple Developer cert, add `identity` to the `mac` config.

**Windows build on macOS fails**
Install Wine: `brew install --cask wine-stable`  
Or build Windows version natively on a Windows machine.
