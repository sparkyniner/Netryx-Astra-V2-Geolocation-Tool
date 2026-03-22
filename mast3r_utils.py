"""
mast3r_utils.py — Auto-configuring MASt3R wrapper for Netryx

Automatically finds MASt3R installation in common locations:
  1. ../mast3r/          (recommended: cloned alongside this repo)
  2. ./mast3r/           (cloned inside this repo)
  3. ~/mast3r/           (cloned in home directory)
  4. Already on sys.path (pip installed or PYTHONPATH set)

No manual path configuration needed.
"""

import os
import sys
import numpy as np
import torch
import cv2
from PIL import Image

# ─────────────────────────────────────────────────────────────────
# Auto-detect MASt3R location
# ─────────────────────────────────────────────────────────────────

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

_SEARCH_PATHS = [
    os.path.join(_THIS_DIR, "..", "mast3r"),        # ../mast3r/ (recommended)
    os.path.join(_THIS_DIR, "mast3r"),              # ./mast3r/
    os.path.expanduser("~/mast3r"),                 # ~/mast3r/
    os.path.join(_THIS_DIR, "..", "..", "mast3r"),   # ../../mast3r/
]

_mast3r_found = False

for _path in _SEARCH_PATHS:
    _path = os.path.abspath(_path)
    if os.path.exists(os.path.join(_path, "mast3r", "model.py")):
        if _path not in sys.path:
            sys.path.insert(0, _path)
        # Also add dust3r submodule
        _dust3r_path = os.path.join(_path, "dust3r")
        if os.path.exists(_dust3r_path) and _dust3r_path not in sys.path:
            sys.path.insert(0, _dust3r_path)
        print(f"[MAST3R] Found MASt3R at: {_path}")
        _mast3r_found = True
        break

if not _mast3r_found:
    # Try importing anyway (maybe it's pip-installed)
    try:
        import mast3r
        _mast3r_found = True
        print("[MAST3R] Found MASt3R via Python path")
    except ImportError:
        print("[MAST3R] ⚠️  MASt3R not found. Run setup.sh or clone it:")
        print("[MAST3R]    cd .. && git clone --recursive https://github.com/naver/mast3r.git")


# ─────────────────────────────────────────────────────────────────
# Device detection
# ─────────────────────────────────────────────────────────────────

def _get_device():
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


# ─────────────────────────────────────────────────────────────────
# Model loading (singleton)
# ─────────────────────────────────────────────────────────────────

_model = None

def get_mast3r_model(device=None):
    """Load MASt3R model. Downloads weights automatically on first run (~1.2GB).
    
    Returns the model or None if MASt3R is not installed.
    """
    global _model
    if _model is not None:
        return _model
    
    if not _mast3r_found:
        print("[MAST3R] Cannot load model — MASt3R not installed")
        return None
    
    dev = device or _get_device()
    
    try:
        from mast3r.model import AsymmetricMASt3R
        
        print(f"[MAST3R] Loading model on {dev}...")
        _model = AsymmetricMASt3R.from_pretrained(
            "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
        ).to(dev).eval()
        
        # Freeze parameters
        for p in _model.parameters():
            p.requires_grad_(False)
        
        print(f"[MAST3R] Model loaded successfully on {dev}")
        return _model
    
    except Exception as e:
        print(f"[MAST3R] Error loading model: {e}")
        return None


# ─────────────────────────────────────────────────────────────────
# Image preprocessing
# ─────────────────────────────────────────────────────────────────

def _prepare_image(pil_img, size=512):
    """Convert PIL image to MASt3R input format.
    
    MASt3R expects images with max dimension 512px.
    Returns dict with 'img' tensor and 'true_shape' tensor.
    """
    img = pil_img.convert('RGB')
    
    # Resize so largest dimension is `size`
    w, h = img.size
    scale = size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    # Make dimensions multiples of 16 (required by ViT)
    new_w = (new_w // 16) * 16
    new_h = (new_h // 16) * 16
    if new_w == 0: new_w = 16
    if new_h == 0: new_h = 16
    
    img = img.resize((new_w, new_h), Image.BILINEAR)
    
    tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    true_shape = torch.tensor([[new_h, new_w]], dtype=torch.long)
    
    return {'img': tensor, 'true_shape': true_shape, 'instance': '0', 'idx': 0}


# ─────────────────────────────────────────────────────────────────
# Dense matching
# ─────────────────────────────────────────────────────────────────

def get_mast3r_matches(query_pil, db_pil, model=None, confidence_threshold=0.3):
    """Run MASt3R dense matching between two PIL images.
    
    Args:
        query_pil: PIL Image (query photo)
        db_pil: PIL Image (database/panorama crop)
        model: MASt3R model (if None, loads automatically)
        confidence_threshold: Minimum match confidence
        
    Returns:
        mkpts0: np.ndarray [N, 2] — matched keypoints in query image
        mkpts1: np.ndarray [N, 2] — matched keypoints in db image
        confidence: np.ndarray [N] — match confidence scores
    """
    if model is None:
        model = get_mast3r_model()
    if model is None:
        return np.array([]), np.array([]), np.array([])
    
    dev = next(model.parameters()).device
    
    # Prepare inputs
    img1 = _prepare_image(query_pil)
    img2 = _prepare_image(db_pil)
    img2['instance'] = '1'
    img2['idx'] = 1
    
    img1 = {k: v.to(dev) for k, v in img1.items()}
    img2 = {k: v.to(dev) for k, v in img2.items()}
    
    try:
        with torch.no_grad():
            result = model(img1, img2)
        
        pred1 = result['pred1']
        pred2 = result['pred2']
        
        # Get descriptors
        desc1 = pred1['desc'].squeeze(0)  # [H*W, D]
        desc2 = pred2['desc'].squeeze(0)  # [H*W, D]
        
        # Get confidence maps
        conf1 = pred1['conf'].squeeze(0).view(-1) if 'conf' in pred1 else torch.ones(desc1.shape[0], device=dev)
        conf2 = pred2['conf'].squeeze(0).view(-1) if 'conf' in pred2 else torch.ones(desc2.shape[0], device=dev)
        
        # Compute similarity and find mutual nearest neighbors
        desc1_norm = torch.nn.functional.normalize(desc1, dim=-1)
        desc2_norm = torch.nn.functional.normalize(desc2, dim=-1)
        
        sim = desc1_norm @ desc2_norm.T
        
        nn12 = sim.argmax(dim=1)
        nn21 = sim.argmax(dim=0)
        
        # Reciprocal check
        mutual = nn21[nn12] == torch.arange(len(nn12), device=dev)
        mutual_idx = torch.where(mutual)[0]
        
        if len(mutual_idx) < 4:
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2), np.array([])
        
        # Filter by confidence
        match_scores = sim[mutual_idx, nn12[mutual_idx]]
        good = match_scores > confidence_threshold
        mutual_idx = mutual_idx[good]
        match_scores = match_scores[good]
        
        if len(mutual_idx) < 4:
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2), np.array([])
        
        # Convert flat indices to 2D pixel coordinates
        H1 = img1['true_shape'][0, 0].item()
        W1 = img1['true_shape'][0, 1].item()
        H2 = img2['true_shape'][0, 0].item()
        W2 = img2['true_shape'][0, 1].item()
        
        # Patch grid dimensions (ViT patch size = 16 for MASt3R)
        patch_size = 16
        pH1, pW1 = H1 // patch_size, W1 // patch_size
        pH2, pW2 = H2 // patch_size, W2 // patch_size
        
        idx1 = mutual_idx.cpu().numpy()
        idx2 = nn12[mutual_idx].cpu().numpy()
        
        # Convert to pixel coordinates
        pts1_y = (idx1 // pW1) * patch_size + patch_size // 2
        pts1_x = (idx1 % pW1) * patch_size + patch_size // 2
        pts2_y = (idx2 // pW2) * patch_size + patch_size // 2
        pts2_x = (idx2 % pW2) * patch_size + patch_size // 2
        
        mkpts0 = np.stack([pts1_x, pts1_y], axis=1).astype(np.float32)
        mkpts1 = np.stack([pts2_x, pts2_y], axis=1).astype(np.float32)
        confidence = match_scores.cpu().numpy()
        
        # RANSAC geometric verification
        if len(mkpts0) >= 8:
            H_mat, mask = cv2.findFundamentalMat(
                mkpts0.astype(np.float64), mkpts1.astype(np.float64),
                cv2.FM_RANSAC, 3.0
            )
            if mask is not None:
                mask = mask.flatten().astype(bool)
                mkpts0 = mkpts0[mask]
                mkpts1 = mkpts1[mask]
                confidence = confidence[mask]
        
        return mkpts0, mkpts1, confidence
    
    except Exception as e:
        print(f"[MAST3R] Matching error: {e}")
        return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2), np.array([])
