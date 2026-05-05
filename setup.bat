@echo off
REM ─────────────────────────────────────────────────────────────
REM Sybil Desktop App — Python Environment Setup (Windows)
REM ─────────────────────────────────────────────────────────────

echo.
echo  Sybil Risk Predictor — Environment Setup (Windows)
echo  ───────────────────────────────────────────────────
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8-3.12 from python.org
    pause
    exit /b 1
)

python --version

if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat
echo Virtual environment activated

pip install --upgrade pip -q
pip install flask flask-cors waitress -q

echo Installing Sybil from source...
pip install -e .\sybil-source\ -q

echo.
echo  Setup complete!
echo  Next: cd electron_app ^&^& npm install ^&^& npm start
echo.
pause
