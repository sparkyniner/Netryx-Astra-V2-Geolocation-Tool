#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Netryx Astra V2 — One-command setup
# Run: chmod +x setup.sh && ./setup.sh
# ═══════════════════════════════════════════════════════════════
set -e

# Save the directory this script lives in (works even if called from elsewhere)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo "🔱 Netryx Astra V2 — Setup"
echo "═══════════════════════════════════════"
echo ""

# ── Check Python ──
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Install Python 3.10+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅ Python $PYTHON_VERSION found"

# ── Create venv if not exists ──
cd "$SCRIPT_DIR"
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null
echo "✅ Virtual environment activated"

# ── Install Python dependencies ──
echo ""
echo "📦 Installing Python dependencies..."
pip install --upgrade pip -q
pip install torch torchvision -q
pip install numpy opencv-python pillow scikit-learn -q
pip install aiohttp -q
pip install tkintermapview -q
pip install huggingface_hub -q
pip install safetensors -q
echo "✅ Python dependencies installed"

# ── Clone MASt3R if not present ──
echo ""
MAST3R_DIR="$SCRIPT_DIR/../mast3r"
if [ -d "$MAST3R_DIR" ]; then
    echo "✅ MASt3R already cloned at $MAST3R_DIR"
else
    echo "📥 Cloning MASt3R (this may take a minute)..."
    cd "$SCRIPT_DIR/.."
    git clone --recursive https://github.com/naver/mast3r.git
    cd mast3r
    pip install -r requirements.txt -q
    pip install -r dust3r/requirements.txt -q
    cd "$SCRIPT_DIR"
    echo "✅ MASt3R cloned and dependencies installed"
fi

# ── Pre-download MegaLoc weights ──
echo ""
echo "📥 Pre-downloading MegaLoc weights (first run only)..."
cd "$SCRIPT_DIR"
python3 -c "
import torch
try:
    model = torch.hub.load('gmberton/MegaLoc', 'get_trained_model')
    print('✅ MegaLoc weights downloaded')
except Exception as e:
    print(f'⚠️  MegaLoc download failed: {e}')
    print('   Will retry automatically when you run the app.')
" 2>/dev/null

# ── Pre-download MASt3R weights ──
echo ""
echo "📥 Pre-downloading MASt3R weights (first run only, ~1.2GB)..."
python3 -c "
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join('$SCRIPT_DIR', '..', 'mast3r')))
try:
    from mast3r.model import AsymmetricMASt3R
    model = AsymmetricMASt3R.from_pretrained('naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric')
    print('✅ MASt3R weights downloaded')
except Exception as e:
    print(f'⚠️  MASt3R download failed: {e}')
    print('   Will retry automatically when you run the app.')
" 2>/dev/null

# ── Create data directories ──
echo ""
cd "$SCRIPT_DIR"
mkdir -p netryx_data/megaloc_parts
mkdir -p netryx_data/index
echo "✅ Data directories created"

# ── Done ──
echo ""
echo "═══════════════════════════════════════"
echo "🔱 Setup complete!"
echo ""
echo "To start Netryx:"
echo "  source venv/bin/activate"
echo "  python3 test_super.py"
echo ""
echo "Or to download a pre-built index:"
echo "  python3 netryx_hub.py download moscow-1km -o ./netryx_data/index"
echo "═══════════════════════════════════════"
