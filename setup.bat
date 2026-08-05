@echo off
echo.
echo ===================================================
echo   Netryx Astra V2 - Windows Setup
echo ===================================================
echo.

:: Remember where we started
set "NETRYX_DIR=%~dp0"

:: Check Python
python --version >nul 2>>installer.log
if errorlevel 1 (
    echo [ERROR] Python not found.
    echo         Download Python from https://python.org
    echo         Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

:: Verify Python version is compatible
python -c "import sys; exit(0 if sys.version_info >= (3,10) and sys.version_info < (3,13) else 1)"

if errorlevel 1 (
    echo [ERROR] Unsupported Python version detected.
    python --version
    echo.
    echo Astra requires Python 3.10 - 3.12.
    echo Please install a compatible Python version.
    pause
    exit /b 1
)

echo [OK] Compatible Python installation found.

:: Check Git
git --version >nul 2>>installer.log
if errorlevel 1 (
    echo [ERROR] Git not found. Download from https://git-scm.com
    pause
    exit /b 1
)
echo [OK] Git found

:: Check for CUDA Toolkit (requires Visual Studio Build Toools 2018-2022)
where nvcc >nul 2>>installer.log
if errorlevel 1 (
    echo [WARNING] CUDA Toolkit was not detected.
    echo [WARNING] Astra can still run using PyTorch CUDA, but MASt3R will use a slower RoPE2D fallback.
    echo [WARNING] Download CUDA Toolkit 12.4 from https://developer.nvidia.com/cuda-12-4-0-download-archive
) else (
    echo [OK] CUDA Toolkit detected
)

:: Create venv
cd /d "%NETRYX_DIR%"
if not exist "venv" (
    echo [SETUP] Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate.bat
echo [OK] Virtual environment activated

:: Install a version of PyTorch compatible with the available hardware
echo Checking for NVIDIA GPU...

nvidia-smi >nul 2>>installer.log
if errorlevel 1 (
    echo [INFO] No NVIDIA GPU detected
    echo Installing CPU PyTorch...
    python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 -q
) else (
    echo [OK] NVIDIA GPU detected
    echo Installing CUDA PyTorch...
    python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 -q
)

if errorlevel 1 (
    echo [ERROR] PyTorch installation failed.
    echo Review installer.log for details.
    pause
    exit /b 1
)

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
python -c "import torch; model = torch.hub.load('gmberton/MegaLoc', 'get_trained_model', trust_repo=True); print('[OK] MegaLoc ready')" 2>>installer.log
if errorlevel 1 echo [WARN] MegaLoc download failed - will retry on first run

:: Pre-download MASt3R weights
echo.
echo [SETUP] Downloading MASt3R model weights ^(~1.2GB, first time only^)...
python -c "import sys,os; p=os.path.abspath(os.path.join(r'%NETRYX_DIR%','..','mast3r')); sys.path.insert(0,p); sys.path.insert(0,os.path.join(p,'dust3r')); from mast3r.model import AsymmetricMASt3R; m=AsymmetricMASt3R.from_pretrained('naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric'); print('[OK] MASt3R ready')" 2>>installer.log
if errorlevel 1 echo [WARN] MASt3R download failed - will retry on first run

:: Install MixVPR
echo [SETUP] Installing MixVPR ^(this may take a few minutes^)...
cd "%USERPROFILE%\.cache\torch\hub"
git clone https://github.com/gmberton/VPR-methods-evaluation.git gmberton_VPR-methods-evaluation_master
echo [OK] MixVPR installed

:: Create data dirs
cd /d "%NETRYX_DIR%"
mkdir netryx_data\megaloc_parts 2>>installer.log
mkdir netryx_data\index 2>>installer.log
echo [OK] Data directories created

:: Disable Triton for windows beacause it is not supported
setx TORCH_COMPILE_DISABLE "1"
set TORCH_COMPILE_DISABLE=1

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
