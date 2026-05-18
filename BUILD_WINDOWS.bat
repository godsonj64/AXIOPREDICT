@echo off
REM ─────────────────────────────────────────────────────────────────
REM  AXIO PREDICT v2.0.1 — Windows Build Script
REM  Produces: AXIO PREDICT Setup 2.0.1.exe   (NSIS installer x64+x86)
REM            AXIO PREDICT 2.0.1.exe          (Portable x64)
REM ─────────────────────────────────────────────────────────────────
setlocal enabledelayedexpansion

echo.
echo ╔══════════════════════════════════════════════════╗
echo ║     AXIO PREDICT — Windows Builder              ║
echo ╚══════════════════════════════════════════════════╝
echo.

REM ── Check Node.js ─────────────────────────────────────────────
where node >nul 2>&1
if errorlevel 1 (
    echo ✗ Node.js not found.
    echo   Download from: https://nodejs.org/  ^(LTS version^)
    pause
    exit /b 1
)
for /f "delims=" %%v in ('node --version') do set NODE_VER=%%v
echo   ✓ Node.js !NODE_VER!

where npm >nul 2>&1
if errorlevel 1 (
    echo ✗ npm not found.
    pause
    exit /b 1
)
echo   ✓ npm found

REM ── Install dependencies ───────────────────────────────────────
echo.
echo ▶ Installing Electron dependencies...
cd /d "%~dp0electron_app"
call npm install --prefer-offline
if errorlevel 1 (
    echo ✗ npm install failed.
    pause
    exit /b 1
)
echo   ✓ Dependencies installed

REM ── Build Windows ──────────────────────────────────────────────
echo.
echo ▶ Building Windows installer (x64 + x86) and portable...
echo   This may take 5–15 minutes.
echo.
call npm run dist:win
if errorlevel 1 (
    echo ✗ Build failed. Check the output above.
    pause
    exit /b 1
)

REM ── Results ────────────────────────────────────────────────────
echo.
echo ╔══════════════════════════════════════════════════╗
echo ║                 Build Complete!                  ║
echo ╚══════════════════════════════════════════════════╝
echo.
echo Output files in: %~dp0electron_app\dist\
echo.
dir /b "%~dp0electron_app\dist\*.exe" 2>nul
echo.
echo "AXIO PREDICT Setup *.exe"  = Full installer (recommended)
echo "AXIO PREDICT *.exe"        = Portable (no install needed)
echo.
pause
