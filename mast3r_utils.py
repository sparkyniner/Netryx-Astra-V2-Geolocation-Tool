import os
import sys
import torch
import numpy as np
from PIL import Image

# Correctly handle local clones of the mast3r dependency
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MAST3R_DIR = os.path.join(PROJECT_ROOT, "mast3r")

if os.path.exists(MAST3R_DIR):
    # Add MAST3R_DIR to path so 'import mast3r' works (as it contains the mast3r/ package folder)
    if MAST3R_DIR not in sys.path:
        sys.path.insert(0, MAST3R_DIR)
    
    # Add dust3r/croco/submodules to path for internal mast3r imports
    CROCO_DIR = os.path.join(MAST3R_DIR, "dust3r", "croco")
    if os.path.exists(CROCO_DIR) and CROCO_DIR not in sys.path:
        sys.path.insert(0, CROCO_DIR)

import mast3r.utils.path_to_dust3r
from mast3r.model import AsymmetricMASt3R
from dust3r.inference import inference
from mast3r.fast_nn import fast_reciprocal_NNs

_mast3r_model = None
device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

def get_mast3r_model():
    """Load MASt3R model. Cached after first call."""
    global _mast3r_model

    if _mast3r_model is not None:
        return _mast3r_model

    print(f"[MASt3R] Loading model on {device}...")
    try:
        weights_path = os.path.join(MAST3R_DIR, "checkpoints", "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
        _mast3r_model = AsymmetricMASt3R.from_pretrained(weights_path, img_size=(512, 512)).to(device)
        _mast3r_model.eval()
        print("[MASt3R] Model loaded successfully.")
    except Exception as e:
        print(f"[MASt3R] Error loading model: {e}")
        raise e
    
    return _mast3r_model

def get_mast3r_matches(img1_pil, img2_pil, model, image_size=512):
    """
    Run MASt3R dense matching between two PIL images.
    Returns: matches_im0, matches_im1, conf
    """
    from torchvision import transforms
    # MASt3R/Dust3R prep logic
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    def prep_img(pil_img):
        img_resized = pil_img.copy()
        img_resized.thumbnail((image_size, image_size))
        tensor = transform(img_resized).unsqueeze(0).to(device)
        return {"img": tensor, "true_shape": np.array([img_resized.size[::-1]]), "idx": 0, "instance": "1"}

    view1 = prep_img(img1_pil)
    view2 = prep_img(img2_pil)
    view2["idx"] = 1

    # Dust3R inference wrapper expects lists of pairs of dicts: [([view1, view2])]
    # Actually, inference takes `pairs` where each pair is a tuple of dicts
    # Wait, `load_images` returns a list of dicts. `inference` expects `[tuple(images)]`
    images = [view1, view2]
    
    with torch.no_grad():
        output = inference([tuple(images)], model, device, batch_size=1, verbose=False)
        
        v1, pred1 = output['view1'], output['pred1']
        v2, pred2 = output['view2'], output['pred2']
        
        desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()
        
        matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                       device=device, dist='dot', block_size=2**13)

        # Ignore small border around the edge
        H0, W0 = v1['true_shape'][0]
        valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
            matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

        H1, W1 = v2['true_shape'][0]
        valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
            matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

        valid_matches = valid_matches_im0 & valid_matches_im1
        matches_im0 = matches_im0[valid_matches].astype(np.float32)
        matches_im1 = matches_im1[valid_matches].astype(np.float32)

        # Return matches scaled to the original image coordinates?
        # MASt3R matched on thumbnail coordinates.
        # test_super.py visualize function expects original crop_size coordinates (400x400 usually).
        # We need to scale the matches back to the PIL image size if it was resized.
        # But wait, thumbnail keeps aspect ratio. Let's just calculate the scale.
        w1_o, h1_o = img1_pil.size
        scale1_x = w1_o / float(W0.cpu().item())
        scale1_y = h1_o / float(H0.cpu().item())
        
        w2_o, h2_o = img2_pil.size
        scale2_x = w2_o / float(W1.cpu().item())
        scale2_y = h2_o / float(H1.cpu().item())
        
        matches_im0[:, 0] *= scale1_x
        matches_im0[:, 1] *= scale1_y
        
        matches_im1[:, 0] *= scale2_x
        matches_im1[:, 1] *= scale2_y

    return matches_im0, matches_im1, None
