"""
megaloc_utils.py — Drop-in replacement for cosplace_utils.py

Provides MegaLoc descriptor extraction with optional PCA dimensionality reduction
for compact indexing. The API mirrors cosplace_utils exactly so the main Netryx
code can switch with minimal changes.

Usage in main code:
    
    from megaloc_utils import (
        get_megaloc_model, extract_megaloc_descriptor,
        megaloc_similarity, batch_extract_megaloc,
        fit_pca, apply_pca, save_pca, load_pca,
        MEGALOC_RAW_DIM
    )
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tfm
from PIL import Image
import pickle
import threading



MEGALOC_RAW_DIM = 8448          # Native MegaLoc output dimension
MEGALOC_PCA_DIM = 1024          # Reduced dimension for indexing (tune as needed)
MEGALOC_INPUT_SIZE = 322        # Input resolution (multiple of 14)



_megaloc_model = None
_megaloc_lock = threading.Lock()
_pca_model = None

# Auto-detect device
if torch.backends.mps.is_available():
    _device = 'mps'
elif torch.cuda.is_available():
    _device = 'cuda'
else:
    _device = 'cpu'




def get_megaloc_model(device=None):
    """Load the MegaLoc model (singleton, thread-safe).
    
    First tries torch.hub (requires internet on first run).
    Falls back to local megaloc_model.py + manual weight loading.
    """
    global _megaloc_model
    if _megaloc_model is not None:
        return _megaloc_model

    with _megaloc_lock:
        if _megaloc_model is not None:
            return _megaloc_model

        dev = device or _device
        print(f"[MEGALOC] Loading MegaLoc model on {dev}...")

        try:
            model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
            print("[MEGALOC] Loaded via torch.hub")
        except Exception as e:
            print(f"[MEGALOC] torch.hub failed ({e}), trying local weights...")
            # Fallback: load from local file if user has downloaded weights
            from megaloc_model import MegaLoc
            model = MegaLoc()
            weights_path = os.path.join(os.path.dirname(__file__), "megaloc_weights.pth")
            if os.path.exists(weights_path):
                state = torch.load(weights_path, map_location='cpu')
                model.load_state_dict(state)
                print(f"[MEGALOC] Loaded local weights from {weights_path}")
            else:
                raise RuntimeError(
                    f"Could not load MegaLoc. Install internet for torch.hub "
                    f"or place weights at {weights_path}"
                )

        model = model.eval().to(dev)

        # Freeze all parameters
        for p in model.parameters():
            p.requires_grad_(False)

        # ── MPS FIX ──────────────────────────────────────────────────
        # The upstream MegaLoc code uses .view() in places where the
        # tensor is non-contiguous after permute/transpose (common on
        # MPS backend). We monkey-patch the backbone's forward and the
        # pos-encoding interpolation to use .reshape()/.contiguous().
        # This is safe on CUDA/CPU too.
        if hasattr(model, 'backbone'):
            _original_backbone_forward = model.backbone.forward
            _original_pos_interp = model.backbone.interpolate_pos_encoding

            def _patched_pos_interp(x, w, h):
                """Patched to use .reshape() instead of .view() for MPS."""
                import math as _math
                bb = model.backbone
                previous_dtype = x.dtype
                npatch = x.shape[1] - 1
                N = bb.pos_embed.shape[1] - 1
                if npatch == N and w == h:
                    return bb.pos_embed
                pos_embed = bb.pos_embed.float()
                class_pos_embed = pos_embed[:, 0]
                patch_pos_embed = pos_embed[:, 1:]
                dim = x.shape[-1]
                w0 = w // bb.patch_size
                h0 = h // bb.patch_size
                M = int(_math.sqrt(N))
                sx = float(w0 + bb.interpolate_offset) / M
                sy = float(h0 + bb.interpolate_offset) / M
                patch_pos_embed = F.interpolate(
                    patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
                    scale_factor=(sx, sy), mode="bicubic",
                    antialias=bb.interpolate_antialias)
                assert (w0, h0) == patch_pos_embed.shape[-2:]
                # KEY FIX: .reshape() instead of .view()
                patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
                return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

            def _patched_backbone_forward(images):
                """Patched to add .contiguous() calls for MPS."""
                bb = model.backbone
                B, _, H, W = images.shape
                x = bb.patch_embed(images)
                cls_tokens = bb.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
                x = x + _patched_pos_interp(x, H, W)
                for block in bb.blocks:
                    x = block(x)
                x = bb.norm(x)
                cls_token = x[:, 0]
                patch_tokens = x[:, 1:]
                # KEY FIX: .contiguous() before reshape
                patch_features = patch_tokens.contiguous().reshape(
                    B, H // bb.patch_size, W // bb.patch_size, bb.embed_dim
                ).permute(0, 3, 1, 2).contiguous()
                return patch_features, cls_token

            model.backbone.forward = _patched_backbone_forward
            print("[MEGALOC] Applied MPS-compatible patches (.view -> .reshape)")
        # ── END MPS FIX ──────────────────────────────────────────────

        _megaloc_model = model
        print(f"[MEGALOC] Model ready. Output dim: {MEGALOC_RAW_DIM}")
        return _megaloc_model


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _preprocess_pil(pil_img, target_size=MEGALOC_INPUT_SIZE):
    """Convert PIL image to normalized tensor for MegaLoc.
    
    MegaLoc expects [B, 3, H, W] with values in [0, 1].
    H, W must be multiples of 14. The model auto-resizes internally,
    but we pre-resize for consistency and to control memory.
    """
    # Resize to target (must be multiple of 14)
    size = target_size
    if size % 14 != 0:
        size = round(size / 14) * 14

    img = pil_img.convert('RGB').resize((size, size), Image.BILINEAR)
    tensor = tfm.to_tensor(img)  # [3, H, W] in [0, 1]
    return tensor


# ---------------------------------------------------------------------------
# Descriptor Extraction
# ---------------------------------------------------------------------------

def extract_megaloc_descriptor(pil_img, apply_pca_reduction=True):
    """Extract MegaLoc descriptor from a single PIL image.
    
    Args:
        pil_img: PIL Image (any size, will be resized)
        apply_pca_reduction: If True and PCA is fitted, reduce dimensions
        
    Returns:
        np.ndarray of shape (MEGALOC_PCA_DIM,) or (MEGALOC_RAW_DIM,)
    """
    model = get_megaloc_model()
    tensor = _preprocess_pil(pil_img).unsqueeze(0).to(_device)

    with torch.no_grad():
        desc = model(tensor)  # [1, 8448]

    desc = desc.cpu().numpy().squeeze()  # (8448,)

    if apply_pca_reduction and _pca_model is not None:
        desc = apply_pca(desc.reshape(1, -1)).squeeze()

    return desc


def batch_extract_megaloc(pil_images, batch_size=16, apply_pca_reduction=False):
    """Batch extract MegaLoc descriptors from a list of PIL images.
    
    Args:
        pil_images: List of PIL Images
        batch_size: Batch size for inference
        apply_pca_reduction: If True and PCA is fitted, reduce dimensions
        
    Returns:
        np.ndarray of shape (N, dim) where dim is PCA_DIM or RAW_DIM
    """
    model = get_megaloc_model()
    all_descs = []

    for i in range(0, len(pil_images), batch_size):
        batch = pil_images[i:i + batch_size]
        tensors = torch.stack([_preprocess_pil(img) for img in batch]).to(_device)

        with torch.no_grad():
            descs = model(tensors)  # [B, 8448]

        all_descs.append(descs.cpu().numpy())

        # Memory cleanup for MPS
        if _device == 'mps' and (i // batch_size) % 10 == 0:
            torch.mps.empty_cache()

    result = np.vstack(all_descs)

    if apply_pca_reduction and _pca_model is not None:
        result = apply_pca(result)

    return result


def megaloc_similarity(desc1, desc2):
    """Cosine similarity between two L2-normalized descriptors."""
    return float(np.dot(desc1, desc2))



# PCA Dimensionality reduce


def fit_pca(descriptors, n_components=MEGALOC_PCA_DIM, whiten=True):
    """Fit PCA model on a matrix of descriptors.
    
    Args:
        descriptors: np.ndarray of shape (N, MEGALOC_RAW_DIM)
        n_components: Target dimensionality
        whiten: Whether to whiten (recommended for retrieval)
        
    Returns:
        Fitted PCA object
    """
    global _pca_model
    from sklearn.decomposition import PCA

    print(f"[MEGALOC-PCA] Fitting PCA: {descriptors.shape[1]} -> {n_components} dims "
          f"on {descriptors.shape[0]} samples...")

    pca = PCA(n_components=n_components, whiten=whiten)
    pca.fit(descriptors)

    explained = pca.explained_variance_ratio_.sum()
    print(f"[MEGALOC-PCA] PCA fitted. Explained variance: {explained:.4f} "
          f"({explained*100:.1f}%)")

    _pca_model = pca
    return pca


def apply_pca(descriptors):
    """Apply fitted PCA to descriptors, then L2-normalize.
    
    Args:
        descriptors: np.ndarray of shape (N, MEGALOC_RAW_DIM) or (MEGALOC_RAW_DIM,)
        
    Returns:
        np.ndarray of shape (N, MEGALOC_PCA_DIM), L2-normalized
    """
    if _pca_model is None:
        raise RuntimeError("PCA not fitted. Call fit_pca() first or load_pca().")

    single = descriptors.ndim == 1
    if single:
        descriptors = descriptors.reshape(1, -1)

    reduced = _pca_model.transform(descriptors)

    # L2-normalize after PCA (critical for cosine similarity search)
    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    norms[norms == 0] = 1
    reduced = reduced / norms

    if single:
        reduced = reduced.squeeze(0)

    return reduced


def save_pca(path):
    """Save fitted PCA model to disk."""
    if _pca_model is None:
        raise RuntimeError("No PCA model to save.")
    with open(path, 'wb') as f:
        pickle.dump(_pca_model, f)
    print(f"[MEGALOC-PCA] Saved PCA model to {path}")


def load_pca(path):
    """Load PCA model from disk."""
    global _pca_model
    with open(path, 'rb') as f:
        _pca_model = pickle.load(f)
    print(f"[MEGALOC-PCA] Loaded PCA model from {path} "
          f"(components: {_pca_model.n_components_})")
    return _pca_model






# Quick test


if __name__ == "__main__":
    print("Testing MegaLoc utils...")

    # Test model loading
    model = get_megaloc_model()
    print(f"Model loaded. feat_dim = {model.feat_dim}")

    # Test with dummy image
    dummy = Image.new('RGB', (256, 256), color=(128, 64, 32))
    desc = extract_megaloc_descriptor(dummy, apply_pca_reduction=False)
    print(f"Raw descriptor shape: {desc.shape}")  # (8448,)
    print(f"L2 norm: {np.linalg.norm(desc):.4f}")  # Should be ~1.0

    # Test batch
    descs = batch_extract_megaloc([dummy, dummy], batch_size=2)
    print(f"Batch descriptor shape: {descs.shape}")  # (2, 8448)

    # Test PCA
    fake_data = np.random.randn(100, MEGALOC_RAW_DIM).astype(np.float32)
    fit_pca(fake_data, n_components=512)
    reduced = apply_pca(desc)
    print(f"PCA-reduced descriptor shape: {reduced.shape}")  # (512,)
    print(f"PCA-reduced L2 norm: {np.linalg.norm(reduced):.4f}")

    print("All tests passed!")
