@echo off
REM ───────────────────────────────────────────────────────────────────
REM  AXIO PREDICT — Python Environment Setup (Windows)
REM
REM  Creates a virtual environment at:
REM    %LOCALAPPDATA%\AXIO_PREDICT\.venv
REM
REM  This path is searched first by the installed application, so
REM  running this script once is enough to make the installer .exe
REM  work — no PATH changes, no shipping a venv inside the installer.
REM ───────────────────────────────────────────────────────────────────

setlocal enabledelayedexpansion
set "TARGET=%LOCALAPPDATA%\AXIO_PREDICT\.venv"
set "SCRIPT_DIR=%~dp0"

echo.
echo  AXIO PREDICT - Environment Setup (Windows)
echo  ------------------------------------------
echo  Target venv: %TARGET%
echo.

REM ── Locate a Python 3.8 / 3.9 / 3.10 (Sybil requires <3.11) ─────────
REM `py -3` would pick whatever 3.x is highest, which is often 3.13/3.14
REM and fails Sybil's `python_requires`. We probe each supported minor
REM explicitly via the py launcher, then fall back to per-user installs.
set "PY_CMD="
for %%V in (3.10 3.9 3.8) do (
    if "!PY_CMD!"=="" (
        py -%%V --version >nul 2>&1
        if not errorlevel 1 set "PY_CMD=py -%%V"
    )
)
if "%PY_CMD%"=="" (
    for %%V in (310 39 38) do (
        if "!PY_CMD!"=="" (
            if exist "%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe" (
                set "PY_CMD=%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe"
            )
        )
    )
)

if "%PY_CMD%"=="" (
    echo [ERROR] No compatible Python ^(3.8, 3.9, or 3.10^) was found.
    echo         Sybil's setup.cfg pins python_requires = ^>=3.8,^<3.11.
    echo         Newer versions ^(3.11+^) will not work.
    echo.
    echo         Install Python 3.10 from:
    echo           https://www.python.org/downloads/release/python-31011/
    echo         Tick "Add to PATH" during the installer, then re-run setup.bat.
    echo         Or just launch AXIO PREDICT - the in-app wizard will install
    echo         Python 3.10 automatically.
    echo.
    pause
    exit /b 1
)

%PY_CMD% --version
echo.

REM ── Create / reuse venv (validate Python version first) ─────────────
set "RECREATE_VENV=0"
if exist "%TARGET%\Scripts\python.exe" (
    for /f "tokens=2" %%V in ('"%TARGET%\Scripts\python.exe" --version 2^>^&1') do set "VENV_VER=%%V"
    for /f "tokens=1,2 delims=." %%A in ("!VENV_VER!") do (
        if not "%%A"=="3" set "RECREATE_VENV=1"
        if %%B LSS 8  set "RECREATE_VENV=1"
        if %%B GTR 10 set "RECREATE_VENV=1"
    )
    if "!RECREATE_VENV!"=="1" (
        echo Existing venv has incompatible Python !VENV_VER! ^(needs 3.8-3.10^); recreating...
        rmdir /s /q "%TARGET%"
    )
)

if not exist "%TARGET%\Scripts\python.exe" (
    echo Creating virtual environment...
    if not exist "%LOCALAPPDATA%\AXIO_PREDICT" mkdir "%LOCALAPPDATA%\AXIO_PREDICT"
    %PY_CMD% -m venv "%TARGET%"
    if errorlevel 1 (
        echo [ERROR] Failed to create venv at %TARGET%.
        pause
        exit /b 1
    )
) else (
    echo Reusing existing venv at %TARGET%.
)

set "VENV_PY=%TARGET%\Scripts\python.exe"

echo.
echo Upgrading pip...
"%VENV_PY%" -m pip install --upgrade pip -q
if errorlevel 1 (
    echo [ERROR] pip upgrade failed.
    pause
    exit /b 1
)

echo Installing Flask, flask-cors, waitress...
"%VENV_PY%" -m pip install flask flask-cors waitress -q
if errorlevel 1 (
    echo [ERROR] Dependency install failed.
    pause
    exit /b 1
)

echo Installing Sybil from source (this can take a few minutes)...
"%VENV_PY%" -m pip install -e "%SCRIPT_DIR%sybil-source" -q
if errorlevel 1 (
    echo [ERROR] Sybil install failed. Check the output above.
    pause
    exit /b 1
)

echo.
echo  Setup complete.
echo  Launch AXIO PREDICT from the Start menu - it will find the venv at:
echo    %TARGET%
echo.
pause
