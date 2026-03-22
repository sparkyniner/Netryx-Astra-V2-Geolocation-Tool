@echo off
echo.
echo ===================================================
echo   Netryx Astra V2 - Windows Setup
echo ===================================================
echo.

:: Remember where we started
set "NETRYX_DIR=%~dp0"

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Download from https://python.org
    echo         Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)
echo [OK] Python found

:: Check Git
git --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git not found. Download from https://git-scm.com
    pause
    exit /b 1
)
echo [OK] Git found

:: Create venv
cd /d "%NETRYX_DIR%"
if not exist "venv" (
    echo [SETUP] Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate.bat
echo [OK] Virtual environment activated

:: Install dependencies
echo.
echo [SETUP] Installing Python dependencies ^(this takes a few minutes^)...
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt -q
echo [OK] Dependencies installed

:: Clone MASt3R
echo.
if exist "%NETRYX_DIR%..\mast3r\mast3r\model.py" (
    echo [OK] MASt3R already cloned
) else (
    echo [SETUP] Cloning MASt3R ^(this may take a few minutes^)...
    cd /d "%NETRYX_DIR%.."
    git clone --recursive https://github.com/naver/mast3r.git
    cd /d "%NETRYX_DIR%..\mast3r"
    python -m pip install -r requirements.txt -q
    python -m pip install -r dust3r\requirements.txt -q
    cd /d "%NETRYX_DIR%"
    echo [OK] MASt3R cloned and dependencies installed
)

:: Return to Netryx directory
cd /d "%NETRYX_DIR%"

:: Pre-download MegaLoc weights
echo.
echo [SETUP] Downloading MegaLoc model weights ^(first time only^)...
python -c "import torch; model = torch.hub.load('gmberton/MegaLoc', 'get_trained_model'); print('[OK] MegaLoc ready')" 2>nul
if errorlevel 1 echo [WARN] MegaLoc download failed - will retry on first run

:: Pre-download MASt3R weights
echo.
echo [SETUP] Downloading MASt3R model weights ^(~1.2GB, first time only^)...
python -c "import sys,os; p=os.path.abspath(os.path.join('%NETRYX_DIR%','..','mast3r')); sys.path.insert(0,p); sys.path.insert(0,os.path.join(p,'dust3r')); from mast3r.model import AsymmetricMASt3R; m=AsymmetricMASt3R.from_pretrained('naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric'); print('[OK] MASt3R ready')" 2>nul
if errorlevel 1 echo [WARN] MASt3R download failed - will retry on first run

:: Create data dirs
cd /d "%NETRYX_DIR%"
mkdir netryx_data\megaloc_parts 2>nul
mkdir netryx_data\index 2>nul
echo [OK] Data directories created

:: Done
echo.
echo ===================================================
echo   Setup complete!
echo.
echo   To run Netryx:
echo     Double-click run.bat
echo   Or:
echo     venv\Scripts\activate
echo     python test_super.py
echo ===================================================
pause
