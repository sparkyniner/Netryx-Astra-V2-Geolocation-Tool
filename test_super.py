import os
import sys
# [MPS FIX] Enable CPU fallback for operators not implemented on MPS (like aten::kthvalue used by DISK)
# MUST be set before importing torch
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageOps, ImageFilter
import numpy as np
import torch
import re
import math
from collections import defaultdict
import concurrent.futures
import itertools
import threading
import queue
import time
import json
import uuid
import random
import glob
import cv2
try:
    import kornia.feature as KF
except ImportError:
    KF = None
import asyncio
import aiohttp
import tkintermapview
# tkintermapview 1.30 resizes overlay tiles with Image.ANTIALIAS, which Pillow 10
# removed; without this every map tile silently renders blank.
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.LANCZOS
import webbrowser
from megaloc_utils import (
    get_megaloc_model, extract_megaloc_descriptor,
    megaloc_similarity, batch_extract_megaloc,
    fit_pca, apply_pca, save_pca, load_pca,
    MEGALOC_RAW_DIM, MEGALOC_PCA_DIM
)
import gc
import torch._dynamo
torch._dynamo.config.suppress_errors = True

#PCA matching dimensions
INDEX_TARGET_DIM = 1024

# performance tuning
# performance tuning
MAX_PANOID_WORKERS = 128
MAX_HEADING_WORKERS = 4
MAX_DOWNLOAD_WORKERS = 120
MAX_MATCH_WORKERS = 16
EARLY_EXIT_INLIER_THRESHOLD = 300
# descriptors are identical at any batch size. On Apple GPUs (memory shared
# with the system) 16 is as fast as 64 and 64 pushed a 16 GB Mac into swap;
# dedicated CUDA/ROCm GPUs keep 64.
MEGALOC_BATCH_SIZE = 16 if torch.backends.mps.is_available() else 64
CROP_QUEUE_SIZE =  1024
# Create mode downloads and projects panoramas in chunks of this many, so the
# downloads overlap with encoding and only one chunk of tiles sits in RAM.
DOWNLOAD_CHUNK_PANOS = 256
# Threads stitching and projecting panoramas in Create mode. Projection runs on
# the GPU; on Apple GPUs (memory shared with the system) 128 threads each holding
# a full panorama took an 800-panorama run to 1.3 GB free, 8 keep 3.4 GB free.
PROJECTION_WORKERS = 8 if torch.backends.mps.is_available() else MAX_PANOID_WORKERS

# Each panoid API call already searches a radius around the query point
# (PANOID_SEARCH_RADIUS_M, sent in _panoids_request_body) and returns every pano
# Google finds inside it, not just the one nearest the exact coordinate.
# Grid points closer together than this radius search overlapping circles
# and mostly rediscover the same panos, which is pure wasted API calls.
PANOID_SEARCH_RADIUS_M = 50
# Grid spacing = radius * this factor. <1.0 leaves deliberate overlap so
# thin strips of coverage (e.g. a road running between two grid points)
# don't get missed; 1.0 is the "just barely touching" tiling. Don't push
# above ~1.0 or genuine gaps start opening up between search circles.
GRID_SPACING_OVERLAP_FACTOR = 0.85


#Device and model steup stuff

device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

extractor_lock = threading.Lock()

try:
    from mast3r_utils import get_mast3r_model, get_mast3r_matches
    MAST3R_AVAILABLE = True
except ImportError:
    MAST3R_AVAILABLE = False
    print("[MASt3R] mast3r_utils.py not found. MASt3R matching disabled.")


try:
    from netryx_hub import NetryxHub, create_bundle, extract_bundle
    HUB_AVAILABLE = True
except ImportError:
    HUB_AVAILABLE = False
    print("[HUB] netryx_hub.py not found. Community sharing disabled.")

mast3r_model_instance = None
mast3r_lock = threading.Lock()

def get_lazy_mast3r():
    global mast3r_model_instance
    with mast3r_lock:
        if mast3r_model_instance is None:
            mast3r_model_instance = get_mast3r_model()
    return mast3r_model_instance



# where we save all the data and stuff
# check if EXPANSION disk exists, otherwise use local folder
_potential_dir = "/Volumes/Expansion/netryx"
if os.path.exists(_potential_dir):
    DATA_DIR = _potential_dir
else:
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "netryx_data")

MEGALOC_PARTS_DIR = os.path.join(DATA_DIR, "megaloc_parts")
EMB_CSV = os.path.join(DATA_DIR, "embeddings_index.csv")
INDEXES_DIR = os.path.join(DATA_DIR, "indexes")
# COMPACT_INDEX_DIR and friends now point at whatever index load_index()/
# build_compact_index() last activated, under INDEXES_DIR/{uuid}/. There is
# no more single default "index" folder — an index must be built or loaded
# before these paths are meaningful.
COMPACT_INDEX_DIR = None
COMPACT_DESCS_PATH = None
COMPACT_META_PATH = None
COMPACT_INFO_PATH = None
COMPACT_PCA_PATH = None

def has_active_index():
    """True if an index has been built or loaded this session. Guard any
    COMPACT_* path access with this -- they are None until load_index()
    or build_compact_index() runs."""
    return COMPACT_INDEX_DIR is not None

# ── Encoder abstraction (MegaLoc | MixVPR) ──────────────────────────────────
# The retrieval encoder is switchable. Each encoder has its OWN parts + index
# dir (descriptors from different models are not comparable), so switching never
# corrupts or mixes indexes. set_encoder() reassigns the path globals above so
# the rest of the code keeps working unchanged.
ACTIVE_ENCODER = "megaloc"
ENCODER_USES_PCA = True          # MegaLoc: 8448-dim -> PCA. MixVPR: already compact.
_BATCH_ENCODE = batch_extract_megaloc

os.makedirs(INDEXES_DIR, exist_ok=True)

def set_encoder(name):
    """Switch the active retrieval encoder for the NEXT indexing run.

    This only controls where raw part files get written (each encoder has
    its own parts dir, since descriptors from different models aren't
    comparable) and which encoder build_compact_index() will use. It does
    NOT point at an index anymore -- indexes are UUID-keyed under
    INDEXES_DIR and are selected with load_index(), not set_encoder().
    Any COMPACT_* globals get cleared here so stale paths from a previously
    loaded index can't silently leak into a new build.
    """
    global ACTIVE_ENCODER, ENCODER_USES_PCA, _BATCH_ENCODE
    global MEGALOC_PARTS_DIR, COMPACT_INDEX_DIR, COMPACT_DESCS_PATH
    global COMPACT_META_PATH, COMPACT_INFO_PATH, COMPACT_PCA_PATH, _compact_cache
    name = (name or "megaloc").lower()

    # No-op if this encoder is already active AND an index is currently
    # loaded. Without this, any redundant call (re-clicking the same radio
    # button, a caller re-asserting the encoder defensively, etc.) silently
    # unloads a perfectly good index for no reason -- the caller wanted to
    # confirm the encoder, not wipe the loaded index.
    if name == ACTIVE_ENCODER and has_active_index():
        return

    if name == "mixvpr":
        import mixvpr_utils as _MX
        _MX.get_mixvpr_model  # ensure module import succeeds early
        MEGALOC_PARTS_DIR = os.path.join(DATA_DIR, "mixvpr_parts")
        ENCODER_USES_PCA = False
        _BATCH_ENCODE = _MX.batch_extract_mixvpr
        ACTIVE_ENCODER = "mixvpr"
    else:
        MEGALOC_PARTS_DIR = os.path.join(DATA_DIR, "megaloc_parts")
        ENCODER_USES_PCA = True
        _BATCH_ENCODE = batch_extract_megaloc
        ACTIVE_ENCODER = "megaloc"
    os.makedirs(MEGALOC_PARTS_DIR, exist_ok=True)
    os.makedirs(INDEXES_DIR, exist_ok=True)
    # Clear any previously loaded index's paths -- caller must build a new
    # index or load_index() an existing one before searching/building again.
    COMPACT_INDEX_DIR = None
    COMPACT_DESCS_PATH = None
    COMPACT_META_PATH = None
    COMPACT_INFO_PATH = None
    COMPACT_PCA_PATH = None
    _compact_cache = None
    print(f"[ENCODER] Active encoder: {ACTIVE_ENCODER} (parts: {MEGALOC_PARTS_DIR})")

def encode_query(pil_img):
    """Encode a query image to a search-ready descriptor for the active encoder."""
    if ACTIVE_ENCODER == "mixvpr":
        import mixvpr_utils as _MX
        return _MX.extract_mixvpr_descriptor(pil_img)
    return extract_megaloc_descriptor(pil_img, apply_pca_reduction=True)

def batch_encode(pil_images, batch_size=None):
    """Batch-encode crops for indexing with the active encoder."""
    if batch_size is None:
        return _BATCH_ENCODE(pil_images)
    return _BATCH_ENCODE(pil_images, batch_size=batch_size)

# Create dirs on startup. COMPACT_INDEX_DIR is intentionally excluded --
# it's None until an index is built or loaded via load_index().
for d in [DATA_DIR, MEGALOC_PARTS_DIR, INDEXES_DIR]:
    os.makedirs(d, exist_ok=True)

_mps_cleanup_counter = 0
_mps_cleanup_lock = threading.Lock()


def aggressive_mps_cleanup(force=False):
    global _mps_cleanup_counter
    with _mps_cleanup_lock:
        _mps_cleanup_counter += 1
        should_clean = force or (_mps_cleanup_counter % 100 == 0)
    if not should_clean:
        return
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()
    if force or (_mps_cleanup_counter % 50 == 0):
        import subprocess
        try:
            subprocess.run(
                ['find', '/private/var/folders', '-name', 'mpsgraph-*', '-type', 'f', '-mmin', '+1', '-delete'],
                capture_output=True, timeout=10
            )
        except Exception:
            pass

def pil_to_tensor(im):
    return torch.from_numpy(np.array(im.convert('RGB'))).float().permute(2, 0, 1).unsqueeze(0).div(255.0).to(device)

def tensor_to_pil(t):
    t = t.squeeze(0).cpu().clamp(0, 1).mul(255).add_(0.5).to(torch.uint8).permute(1, 2, 0).numpy()
    if t.shape[2] == 1:
        t = t.squeeze(2)
    return Image.fromarray(t)

def scan_indexes():
    indexes = []

    indexes_dir = os.path.join(DATA_DIR, "indexes")

    if not os.path.exists(indexes_dir):
        return indexes

    for index_id in os.listdir(indexes_dir):
        index_path = os.path.join(indexes_dir, index_id)
        manifest_path = os.path.join(index_path, "manifest.json")

        if not os.path.isfile(manifest_path):
            continue

        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)

            manifest["path"] = index_path
            manifest["index_id"] = index_id
            if "coverage_center" not in manifest:
                manifest["coverage_center"] = {
                    "lat": manifest.get("center_lat"),
                    "lon": manifest.get("center_lon"),
                }

            indexes.append(manifest)

        except Exception as e:
            print(f"[INDEX] Failed loading {index_id}: {e}")

    return indexes

def load_index(index_id):
    global COMPACT_INDEX_DIR
    global COMPACT_DESCS_PATH
    global COMPACT_META_PATH
    global COMPACT_INFO_PATH
    global COMPACT_PCA_PATH
    global ACTIVE_ENCODER
    global ENCODER_USES_PCA
    global _compact_cache

    index_path = os.path.join(INDEXES_DIR, index_id)

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index not found: {index_id}")

    # Read manifest to determine descriptor name
    manifest_path = os.path.join(index_path, "manifest.json")

    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"manifest.json missing for index: {index_id}")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Normalize coverage fields -- Hub-built bundles use flat center_lat/
    # center_lon/radius_km, while write_index_manifest() (used for indexes
    # built locally in this app) nests them under coverage_center. Callers
    # should always read manifest["coverage_center"]["lat"/"lon"] and
    # manifest["radius_km"] after this.
    if "coverage_center" not in manifest:
        manifest["coverage_center"] = {
            "lat": manifest.get("center_lat"),
            "lon": manifest.get("center_lon"),
        }

    encoder = manifest.get("descriptor_model", "MegaLoc").lower()

    # Don't blindly trust descriptor_model -- verify against what's actually
    # on disk. A manifest can end up self-contradictory (e.g. built while
    # ACTIVE_ENCODER said one thing but the real extraction ran as another),
    # and trusting the label alone means load_index() looks for a filename
    # that was never written, silently behaving as if no index exists.
    mixvpr_path = os.path.join(index_path, "mixvpr_descriptors.npy")
    megaloc_path = os.path.join(index_path, "megaloc_descriptors.npy")
    mixvpr_exists = os.path.isfile(mixvpr_path)
    megaloc_exists = os.path.isfile(megaloc_path)

    if encoder == "mixvpr" and not mixvpr_exists and megaloc_exists:
        print(f"[INDEX] WARNING: manifest for {index_id} claims 'mixvpr' but "
              f"only megaloc_descriptors.npy exists on disk. Using megaloc "
              f"instead (the manifest is stale/wrong, not the data).")
        encoder = "megaloc"
    elif encoder != "mixvpr" and not megaloc_exists and mixvpr_exists:
        print(f"[INDEX] WARNING: manifest for {index_id} claims '{encoder}' but "
              f"only mixvpr_descriptors.npy exists on disk. Using mixvpr "
              f"instead (the manifest is stale/wrong, not the data).")
        encoder = "mixvpr"

    if encoder == "mixvpr":
        desc_name = "mixvpr_descriptors.npy"
        ENCODER_USES_PCA = False
    else:
        desc_name = "megaloc_descriptors.npy"
        ENCODER_USES_PCA = True

    desc_path_check = os.path.join(index_path, desc_name)
    if not os.path.isfile(desc_path_check):
        raise FileNotFoundError(
            f"Index {index_id}: expected descriptor file not found at "
            f"{desc_path_check}, and no alternate-encoder descriptor file "
            f"was found either. This index looks incomplete or corrupted."
        )

    ACTIVE_ENCODER = encoder
    COMPACT_INDEX_DIR = index_path
    COMPACT_DESCS_PATH = os.path.join(index_path, desc_name)
    COMPACT_META_PATH = os.path.join(index_path, "metadata.npz")
    COMPACT_INFO_PATH = os.path.join(index_path, "index_info.txt")
    COMPACT_PCA_PATH = os.path.join(index_path, "megaloc_pca.pkl")

    _compact_cache = None

    print(
        f"[INDEX] Loaded {manifest.get('name', index_id)} "
        f"({encoder})"
    )

    return manifest


def write_index_manifest(index_dir, index_id, *, name=None, encoder="megaloc",
                          descriptor_dim=None, num_entries=None,
                          center_lat=None, center_lon=None, radius_km=None,
                          format_version=1):
    """Write manifest.json for an index dir. Called right after an index's
    data files are saved so scan_indexes()/load_index() can discover it."""
    manifest = {
        "index_id": index_id,
        "name": name or index_id,
        "descriptor_model": encoder,
        "descriptor_dim": descriptor_dim,
        "num_entries": num_entries,
        "coverage_center": {"lat": center_lat, "lon": center_lon},
        "radius_km": radius_km,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "format_version": format_version,
    }
    manifest_path = os.path.join(index_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest

def draw_matches(img1, img2, kp1, kp2, matches=None, color=(0, 255, 0)):
    w1, h1 = img1.size
    w2, h2 = img2.size
    new_h = max(h1, h2)
    result = Image.new("RGB", (w1 + w2, new_h), (255, 255, 255))
    result.paste(img1, (0, 0))
    result.paste(img2, (w1, 0))
    draw = ImageDraw.Draw(result)
    if matches is None:
        return result
    if isinstance(matches, np.ndarray) and matches.ndim == 2 and matches.shape[1] == 2:
        for i in range(len(matches)):
            idx0, idx1 = matches[i]
            p1, p2 = kp1[idx0], kp2[idx1]
            draw.line(((p1[0], p1[1]), (p2[0] + w1, p2[1])), fill=color, width=1)
    elif isinstance(matches, np.ndarray) and matches.ndim == 1:
        for idx, m in enumerate(matches):
            if m > -1:
                x1, y1 = kp1[idx]
                x2, y2 = kp2[m]
                draw.line(((x1, y1), (x2 + w1, y2)), fill=color, width=1)
    return result

# PANORAMA DOWNLOAD & STITCHING

IMGX = 4
IMGY = 2

# The old GET endpoint (maps/api/js/GeoPhotoService.SingleImageSearch) is
# decommissioned. This is the JSON+protobuf RPC the Maps JS API uses now; the
# response nests panoids and coordinates the same way, so panoids_from_response
# parses it unchanged.
PANOIDS_SEARCH_URL = "https://maps.googleapis.com/$rpc/google.internal.maps.mapsjs.v1.MapsJsInternalService/SingleImageSearch"

def _panoids_request_body(lat, lon, radius_m=PANOID_SEARCH_RADIUS_M):
    return json.dumps([
        ["apiv3", None, None, None, "US"],
        [[None, None, lat, lon], float(radius_m)],
        [None, ["en", "GB"], None, None, None, None, None, None, [2], None, [[[2, True, 2]]]],
        [[1, 2, 3, 4, 6, 8, 12], None, None, None, [], []],
    ])

def panoids_from_response(text):
    matches = re.findall(r'"([A-Za-z0-9_-]{22})"', text)
    out = []
    for panoid in matches:
        latlon = re.findall(r'"' + panoid + r'".+?\[null,null,(-?\d+\.\d+),(-?\d+\.\d+)', text)
        if latlon:
            lat, lon = map(float, latlon[0])
        else:
            lat, lon = None, None
        out.append({"panoid": panoid, "lat": lat, "lon": lon})
    filtered = []
    seen = set()
    for p in out:
        if p['panoid'] not in seen:
            seen.add(p['panoid'])
            filtered.append(p)
    return filtered

def tiles_info(panoid):
    # cbk0.google.com now returns 403 for all requests; use the current tile endpoint
    image_url = "https://streetviewpixels-pa.googleapis.com/v1/tile?cb_client=maps_sv.tactile&panoid={0:}&x={1:}&y={2:}&zoom=2&nbt=1&fover=2"
    coord = list(itertools.product(range(IMGX), range(IMGY)))
    tiles = [(x, y, "%s_%dx%d.jpg" % (panoid, x, y), image_url.format(panoid, x, y)) for x, y in coord]
    return tiles

async def download_tile_aiohttp(session, x, y, fname, url):
    for attempt in range(2):
        try:
            async with session.get(url.replace("http://", "https://"), timeout=10) as response:
                if response.status == 200:
                    data = await response.read()
                    return x, y, data
        except Exception:
            await asyncio.sleep(2)
    return x, y, None

def download_tiles(tiles, status_callback=None, max_workers=64):
    total = len(tiles)
    results = {}
    async def main():
        connector = aiohttp.TCPConnector(limit=max_workers)
        # Google endpoints 403 without browser-like headers
        _headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
            "Referer": "https://www.google.com/maps/",
        }
        async with aiohttp.ClientSession(connector=connector, headers=_headers) as session:
            tasks = []
            for i, (x, y, fname, url) in enumerate(tiles):
                tasks.append(download_tile_aiohttp(session, x, y, fname, url))
            for idx, coro in enumerate(asyncio.as_completed(tasks), 1):
                x, y, data = await coro
                if data:
                    results[(x, y)] = data
                if status_callback:
                    status_callback(idx, total)
    asyncio.run(main())
    return results

async def _download_tiles_multi(panoid_tile_lists, max_workers=120, status_callback=None):
    """Download tiles for MANY panoramas concurrently in ONE shared event loop
    and ONE shared connection pool, instead of spinning up a fresh asyncio
    loop (and a fresh TCPConnector) per panorama. Spinning up an event loop
    per-panoid inside an already-concurrent ThreadPoolExecutor was the real
    bottleneck here: e.g. 265 panoids meant 265 event loop start/teardown
    cycles happening across up to 128 threads at once, each with its own
    120-connection pool sized for just 8 tiles.

    panoid_tile_lists: dict {panoid_id: [(x, y, fname, url), ...]}
    Returns: dict {panoid_id: {(x, y): bytes}}
    """
    results = {pid: {} for pid in panoid_tile_lists}
    connector = aiohttp.TCPConnector(limit=max_workers)
    _headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
        "Referer": "https://www.google.com/maps/",
    }
    total_tiles = sum(len(t) for t in panoid_tile_lists.values())
    done = 0

    # The per-request timeout starts when session.get() is called, even if the
    # request is still waiting for a free pooled connection. With every tile
    # queued at once, tiles more than ~10 s back in the queue timed out without
    # ever being sent, and their panoramas were silently dropped (18% of 1,500
    # panoramas against a local server, 87% of 5,140 against Google). The
    # semaphore keeps queued tiles from starting their timeout early.
    in_flight = asyncio.Semaphore(max_workers)

    async with aiohttp.ClientSession(connector=connector, headers=_headers) as session:
        async def fetch_one(pid, x, y, fname, url):
            nonlocal done
            async with in_flight:
                xr, yr, data = await download_tile_aiohttp(session, x, y, fname, url)
            if data:
                results[pid][(xr, yr)] = data
            done += 1
            if status_callback:
                status_callback(done, total_tiles)

        tasks = [
            fetch_one(pid, x, y, fname, url)
            for pid, tiles in panoid_tile_lists.items()
            for (x, y, fname, url) in tiles
        ]
        await asyncio.gather(*tasks)

    return results

def download_tiles_for_panoids(panoid_ids, max_workers=120, status_callback=None):
    """Sync wrapper: fetch tiles for a whole batch of panoids in one shared
    event loop. Use this instead of calling download_tiles() once per
    panoid inside a thread pool."""
    panoid_tile_lists = {pid: tiles_info(pid) for pid in panoid_ids}
    return asyncio.run(_download_tiles_multi(panoid_tile_lists, max_workers=max_workers,
                                              status_callback=status_callback))

def stitch_tiles(tiles_data):
    tile_w, tile_h = 512, 512
    import io
    pano_np = np.zeros((IMGY * tile_h, IMGX * tile_w, 3), dtype=np.uint8)
    for (x, y), data in tiles_data.items():
        try:
            tile = Image.open(io.BytesIO(data))
            tile_np = np.array(tile)
            th, tw, _ = tile_np.shape
            pano_np[y*tile_h:y*tile_h+th, x*tile_w:x*tile_w+tw] = tile_np
            tile.close()
        except Exception:
            continue
    return Image.fromarray(pano_np)


# ═══════════════════════════════════════════════════════════════════
# GEO UTILITIES
# ═══════════════════════════════════════════════════════════════════

def haversine(p1, p2):
    R = 6371
    lat1, lon1 = map(math.radians, p1)
    lat2, lon2 = map(math.radians, p2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def haversine_vec(center_lat, center_lon, lats, lons):
    """Vectorized haversine: distance in km from one (lat, lon) point to
    every point in the lats/lons numpy arrays. Same formula as haversine()
    above, just array-based instead of scalar, for computing something like
    the max distance from a center to thousands of indexed points without
    a Python-level loop."""
    R = 6371
    lat1 = math.radians(center_lat)
    lon1 = math.radians(center_lon)
    lat2 = np.radians(lats)
    lon2 = np.radians(lons)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + math.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def coverage_polylines(latlons, max_link_km=0.03, bucket_deg=0.0002):
    """Turn panorama locations into polylines that trace the streets.

    Links each point only to nearby points (<= max_link_km), keeps the minimum
    spanning forest of those links, and splits it into chains at junctions and
    dead ends. Drawing every chain gives the street network; a single polyline
    through a whole component in visit order zigzags across it instead.

    Returns (chains, singles): chains is a list of [(lat, lon), ...] with at
    least two points each; singles are points with no neighbour in range.
    """
    grid = defaultdict(list)
    for loc in latlons:
        grid[(int(loc[0] / bucket_deg), int(loc[1] / bucket_deg))].append(loc)

    edges = []
    for (bx, by), bucket_locs in grid.items():
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for a in bucket_locs:
                    for b in grid.get((bx + dx, by + dy), ()):
                        if a < b:
                            d = haversine(a, b)
                            if d <= max_link_km:
                                edges.append((d, a, b))
    edges.sort()

    parent = {loc: loc for loc in latlons}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    adj = defaultdict(list)
    for _, a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            adj[a].append(b)
            adj[b].append(a)

    chains, used = [], set()
    for start in adj:
        if len(adj[start]) == 2:
            continue  # chains start and end at dead ends or junctions
        for nxt in adj[start]:
            if (start, nxt) in used:
                continue
            chain, prev, cur = [start], start, nxt
            used.add((start, nxt)); used.add((nxt, start))
            while True:
                chain.append(cur)
                if len(adj[cur]) != 2:
                    break
                prev, cur = cur, adj[cur][0] if adj[cur][0] != prev else adj[cur][1]
                used.add((prev, cur)); used.add((cur, prev))
            chains.append(chain)
    singles = [loc for loc in latlons if loc not in adj]
    return chains, singles

def grid_points(center, radius, spacing_m):
    """Build a grid of coordinates to probe for Street View panoramas.

    spacing_m: target distance in meters between adjacent grid points
    (this is what the GUI's "Grid Resolution" field actually documents
    itself as -- previously the code silently treated this number as a
    point-count instead, so a 300 they typed meaning "300m apart" became
    a 301x301 grid instead).

    Each panoid API call already searches PANOID_SEARCH_RADIUS_M around its
    query point, so spacing tighter than that just re-searches overlapping
    areas and rediscovers the same panos -- pure wasted API calls. spacing
    is floored here at PANOID_SEARCH_RADIUS_M * GRID_SPACING_OVERLAP_FACTOR
    for that reason; going below it buys duplicate coverage, not real
    coverage.
    """
    lat, lon = center
    min_spacing_m = PANOID_SEARCH_RADIUS_M * GRID_SPACING_OVERLAP_FACTOR
    if spacing_m < min_spacing_m:
        print(f"[GRID] Requested spacing {spacing_m}m is tighter than "
              f"{PANOID_SEARCH_RADIUS_M}m search radius can usefully cover "
              f"-- flooring to {min_spacing_m:.0f}m (tighter would just "
              f"re-search overlapping circles).")
        spacing_m = min_spacing_m

    top_left = (lat - radius / 70, lon + radius / 70)
    bottom_right = (lat + radius / 70, lon - radius / 70)
    lat_diff = top_left[0] - bottom_right[0]
    lon_diff = top_left[1] - bottom_right[1]

    # Convert the requested meter spacing into a point count across the
    # bounding box (diameter = 2 * radius, in km -> meters).
    diameter_m = radius * 2 * 1000
    resolution = max(1, round(diameter_m / spacing_m))

    test_points = list(itertools.product(range(resolution + 1), range(resolution + 1)))
    test_points = [
        (bottom_right[0] + x * lat_diff / resolution, bottom_right[1] + y * lon_diff / resolution)
        for (x, y) in test_points
    ]
    test_points = [p for p in test_points if haversine(p, center) <= radius]
    return test_points

def get_panoids(points, status_callback=None, max_workers=64):
    import csv
    async def fetch_one(session, idx, lat, lon, max_attempts=3):
        # 3 attempts + a short timeout: repeated failures are almost always
        # genuinely empty spots (water/no coverage), so extra retries and a long
        # timeout just add a slow tail with no extra panos found.
        body = _panoids_request_body(lat, lon)
        attempt = 0
        rate_limit_retries = 0  # 429s get their own budget so real points aren't dropped
        while attempt < max_attempts:
            try:
                async with session.post(PANOIDS_SEARCH_URL, data=body, timeout=15,
                                        headers={"Content-Type": "application/json+protobuf"}) as resp:
                    status = resp.status
                    text = await resp.text()
                    if status == 429:
                        # Transient throttling — back off and retry without
                        # consuming the fast-fail budget (up to a cap).
                        if rate_limit_retries < 5:
                            rate_limit_retries += 1
                            await asyncio.sleep(min(2 ** rate_limit_retries, 10))
                            continue
                        return []
                    elif status != 200:
                        attempt += 1
                        continue
                    pans = panoids_from_response(text)
                    if not pans:
                        return []
                    return pans
            except asyncio.TimeoutError:
                attempt += 1
                await asyncio.sleep(0.5)
            except Exception:
                attempt += 1
                await asyncio.sleep(0.5)
        return []

    async def main():
        connector = aiohttp.TCPConnector(limit=max_workers)
        # Google endpoints 403 without browser-like headers
        _headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
            "Referer": "https://www.google.com/maps/",
        }
        async with aiohttp.ClientSession(connector=connector, headers=_headers) as session:
            tasks = []
            for idx, (lat, lon) in enumerate(points):
                task = asyncio.create_task(fetch_one(session, idx, lat, lon))
                tasks.append(task)
            results = []
            for idx, task in enumerate(asyncio.as_completed(tasks), 1):
                pans = await task
                results.extend(pans)
                if status_callback:
                    status_callback(idx, len(points))
            return results

    panoids_raw = asyncio.run(main())
    already = set()
    filtered = []
    for pan in panoids_raw:
        if pan['panoid'] not in already:
            already.add(pan['panoid'])
            filtered.append(pan)
    print(f"[SUMMARY] Fetched {len(points)} grid points, found {len(filtered)} unique panoids.")
    return filtered

def generate_circle_points(center_lat, center_lon, radius_km, num_points=36):
    points = []
    R = 6371.0
    lat_rad = math.radians(center_lat)
    lon_rad = math.radians(center_lon)
    angular_dist = radius_km / R
    for i in range(num_points):
        bearing = math.radians(i * (360 / num_points))
        new_lat = math.asin(math.sin(lat_rad) * math.cos(angular_dist) +
                            math.cos(lat_rad) * math.sin(angular_dist) * math.cos(bearing))
        new_lon = lon_rad + math.atan2(math.sin(bearing) * math.sin(angular_dist) * math.cos(lat_rad),
                                       math.cos(angular_dist) - math.sin(lat_rad) * math.sin(new_lat))
        points.append((math.degrees(new_lat), math.degrees(new_lon)))
    return points



# EQUIRECTANGULAR PROJECTION 


def get_projection_base_dirs(fov_deg, out_hw):
    fov = math.radians(fov_deg)
    out_h, out_w = out_hw
    cx, cy = out_w / 2.0, out_h / 2.0
    fx = fy = (out_w / 2.0) / math.tan(fov / 2.0)
    xx, yy = torch.meshgrid(
        torch.arange(out_w, device=device, dtype=torch.float32),
        torch.arange(out_h, device=device, dtype=torch.float32),
        indexing='xy'
    )
    x = (xx - cx) / fx
    y = (yy - cy) / fy
    z = torch.ones_like(x)
    dirs = torch.stack([x, -y, z], dim=-1)
    dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
    return dirs.reshape(-1, 3).T

def equirectangular_to_rectilinear_torch(pano_tensor, fov_deg=90, out_hw=(400, 400), yaw_deg=0, pitch_deg=0, base_dirs=None):
    _, _, h, w = pano_tensor.shape
    out_h, out_w = out_hw
    if isinstance(yaw_deg, (float, int)):
        yaws = torch.tensor([yaw_deg], device=device, dtype=torch.float32)
    elif isinstance(yaw_deg, list):
        yaws = torch.tensor(yaw_deg, device=device, dtype=torch.float32)
    else:
        yaws = yaw_deg.to(device).float()
    B = len(yaws)
    yaws_rad = torch.deg2rad(yaws)
    cos_vals = torch.cos(yaws_rad)
    sin_vals = torch.sin(yaws_rad)
    zeros = torch.zeros_like(cos_vals)
    ones = torch.ones_like(cos_vals)
    row1 = torch.stack([cos_vals, zeros, sin_vals], dim=1)
    row2 = torch.stack([zeros, ones, zeros], dim=1)
    row3 = torch.stack([-sin_vals, zeros, cos_vals], dim=1)
    R = torch.stack([row1, row2, row3], dim=1)
    if base_dirs is None:
        base_dirs = get_projection_base_dirs(fov_deg, out_hw)
    dirs = torch.matmul(R, base_dirs.unsqueeze(0))
    dirs = dirs.permute(0, 2, 1)
    x = dirs[:, :, 0]
    y = dirs[:, :, 1]
    z = dirs[:, :, 2]
    lon = torch.atan2(x, z)
    lat = torch.asin(y.clamp(-1+1e-7, 1-1e-7))
    grid_x = lon / math.pi
    grid_y = -lat / (math.pi / 2.0)
    grid = torch.stack([grid_x, grid_y], dim=-1).reshape(B, out_h, out_w, 2)
    pano_batch = pano_tensor.expand(B, -1, -1, -1)
    out = torch.nn.functional.grid_sample(pano_batch, grid, mode='bilinear', align_corners=True)
    return out

def equirectangular_to_rectilinear(pano_img, fov_deg=90, out_hw=(400, 400), yaw_deg=0, pitch_deg=0):
    pano_tensor = pil_to_tensor(pano_img)
    out_tensor = equirectangular_to_rectilinear_torch(pano_tensor, fov_deg, out_hw, yaw_deg, pitch_deg)
    return tensor_to_pil(out_tensor)


# ═══════════════════════════════════════════════════════════════════
# COMPACT INDEX — BUILD, LOAD, SEARCH
# (Merged from compact_index.py)
# ═══════════════════════════════════════════════════════════════════

_compact_cache = None

def parse_emb_path(emb_path):
    """Extract panoid and heading from path like '/path/to/PANOID_HEADING.npz'."""
    filename = os.path.basename(emb_path)
    name = filename.replace('.npz', '')
    parts = name.rsplit('_', 1)
    if len(parts) == 2:
        try:
            return parts[0], int(parts[1])
        except ValueError:
            pass
    return None, None


def build_compact_index():
    """Build compact index from part files + CSV coordinates.
    
    Auto-applies PCA if descriptors are high-dimensional (e.g., 8448 from MegaLoc).
    """
    global COMPACT_INDEX_DIR
    global COMPACT_DESCS_PATH
    global COMPACT_META_PATH
    global COMPACT_INFO_PATH
    global COMPACT_PCA_PATH
    global _compact_cache
    import glob

    # Capture the encoder NOW, at the start of the build, and use this
    # captured value everywhere below -- including the final manifest write.
    # ACTIVE_ENCODER is a live global; if anything calls set_encoder() while
    # this build is running (a build can take a while), reading
    # ACTIVE_ENCODER again at the end would write a manifest describing a
    # DIFFERENT encoder than the one whose part files actually got indexed.
    # That exact mismatch previously produced an index whose manifest said
    # "mixvpr" while the descriptor file on disk was really MegaLoc's.
    build_encoder = ACTIVE_ENCODER

    index_id = str(uuid.uuid4())
    index_dir = os.path.join(INDEXES_DIR, index_id)
    os.makedirs(index_dir, exist_ok=True)

    megaloc_pattern = os.path.join(MEGALOC_PARTS_DIR, "megaloc_part_*.npz")
    part_files = sorted(glob.glob(megaloc_pattern))
    part_files = sorted(set(part_files))

    if not part_files:
        print(f"[INDEX] ERROR: No part files found")
        return False

    print(f"[INDEX] Found {len(part_files)} part files")

    # ── Pass 1: Count total entries and detect descriptor dimension ──
    total = 0
    raw_dim = None
    for pf in part_files:
        data = np.load(pf, allow_pickle=False)
        total += len(data['paths'])
        if raw_dim is None:
            raw_dim = data['descriptors'].shape[1]
        del data
    
    print(f"[INDEX] Total entries: {total}, raw descriptor dim: {raw_dim}")

    # ── Decide if PCA is needed ──
    needs_pca = raw_dim > INDEX_TARGET_DIM
    final_dim = INDEX_TARGET_DIM if needs_pca else raw_dim
    
    if needs_pca:
        print(f"[INDEX] Will apply PCA: {raw_dim} -> {final_dim}")
        
        # Fit PCA on a subsample (avoids 63GB RAM spike for 2M×8448)
        MAX_PCA_SAMPLES = 100_000
        COMPACT_PCA_PATH = os.path.join(index_dir, "megaloc_pca.pkl")
        pca_path = COMPACT_PCA_PATH

        # Collect subsample for PCA fitting
        print(f"[INDEX] Collecting subsample for PCA fitting (max {MAX_PCA_SAMPLES})...")
        pca_samples = []
        pca_count = 0
        for pf in part_files:
            if pca_count >= MAX_PCA_SAMPLES:
                break
            data = np.load(pf, allow_pickle=False)
            descs = data['descriptors']
            remaining = MAX_PCA_SAMPLES - pca_count
            pca_samples.append(descs[:remaining])
            pca_count += len(descs[:remaining])
            del data
        
        pca_matrix = np.vstack(pca_samples)
        del pca_samples
        print(f"[INDEX] Fitting PCA on {pca_matrix.shape[0]} samples...")
        
        from sklearn.decomposition import PCA
        pca = PCA(n_components=final_dim, whiten=True)
        pca.fit(pca_matrix)
        explained = pca.explained_variance_ratio_.sum()
        print(f"[INDEX] PCA fitted. Explained variance: {explained*100:.1f}%")
        del pca_matrix
        
        # Save PCA model for query-time use
        save_pca(pca_path, pca)
        print(f"[INDEX] Saved PCA model to {pca_path}")
        
        # Also load it into megaloc_utils global
        load_pca(pca_path)
    else:
        pca = None
        print(f"[INDEX] Descriptors already {raw_dim}-dim, no PCA needed")

    # ── Pass 2: Load descriptors and metadata ──
    print(f"[INDEX] Loading and merging {len(part_files)} files...")
    all_descs = np.zeros((total, final_dim), dtype=np.float32)
    all_paths = []
    all_embedded_lats = []
    all_embedded_lons = []

    import time
    idx = 0
    t0 = time.time()
    for i, pf in enumerate(part_files):
        data = np.load(pf, allow_pickle=False)
        n = len(data['paths'])
        
        descs = data['descriptors']
        
        # Apply PCA if needed
        if needs_pca and descs.shape[1] > final_dim:
            descs = pca.transform(descs).astype(np.float32)
            # L2-normalize after PCA
            norms = np.linalg.norm(descs, axis=1, keepdims=True)
            norms[norms == 0] = 1
            descs = descs / norms
        elif descs.shape[1] != final_dim:
            print(f"[INDEX] WARNING: Skipping {pf} — dim {descs.shape[1]} != expected {final_dim}")
            del data
            continue
        
        all_descs[idx:idx+n] = descs
        all_paths.extend(data['paths'].tolist())

        if 'lats' in data and 'lons' in data:
            all_embedded_lats.extend(data['lats'].tolist())
            all_embedded_lons.extend(data['lons'].tolist())
        else:
            all_embedded_lats.extend([0.0] * n)
            all_embedded_lons.extend([0.0] * n)

        idx += n
        del data, descs
        if (i+1) % 100 == 0:
            print(f"  Loaded {i+1}/{len(part_files)} ({idx} entries) [{time.time()-t0:.0f}s]")

    # Trim if we skipped any files
    if idx < total:
        all_descs = all_descs[:idx]
        print(f"[INDEX] Trimmed to {idx} entries (skipped some incompatible files)")
        total = idx

    print(f"[INDEX] Loaded all {idx} entries in {time.time()-t0:.1f}s")

    # ── Load lat/lon from CSV ──
    print(f"[INDEX] Loading coordinates from {EMB_CSV}...")
    csv_locations = {}
    csv_full_locations = {}
    if os.path.exists(EMB_CSV):
        with open(EMB_CSV, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    try:
                        lat, lon = float(parts[1]), float(parts[2])
                        csv_full_locations[parts[0]] = (lat, lon)
                        csv_locations[os.path.basename(parts[0])] = (lat, lon)
                    except ValueError:
                        pass
    print(f"[INDEX] CSV has {len(csv_locations)} location entries")

    # ── Match paths to coordinates ──
    lats = np.zeros(idx, dtype=np.float32)
    lons = np.zeros(idx, dtype=np.float32)
    headings = np.zeros(idx, dtype=np.int16)
    panoids = []
    valid_mask = np.zeros(idx, dtype=bool)
    matched = 0

    for i, path in enumerate(all_paths):
        filename = os.path.basename(path)
        name = filename.replace('.npz', '')
        parts_split = name.rsplit('_', 1)
        panoid = parts_split[0] if len(parts_split) == 2 else None
        try:
            heading = int(parts_split[1]) if len(parts_split) == 2 else 0
        except ValueError:
            heading = 0

        panoids.append(panoid or "")
        headings[i] = heading

        emb_lat = all_embedded_lats[i]
        emb_lon = all_embedded_lons[i]

        if emb_lat != 0 or emb_lon != 0:
            lats[i], lons[i] = emb_lat, emb_lon
            valid_mask[i] = True
            matched += 1
        else:
            loc = csv_full_locations.get(path) or csv_locations.get(filename)
            if loc:
                lats[i], lons[i] = loc
                valid_mask[i] = True
                matched += 1
        if (i + 1) % 200000 == 0:
            print(f"  Matching {i+1}/{idx}... ({matched} matched)")

    print(f"[INDEX] Matched {matched}/{idx} paths to coordinates")

    valid_idx = np.where(valid_mask)[0]
    print(f"[INDEX] Keeping {len(valid_idx)} entries with valid coordinates")

    # ── Filter and normalize ──
    # NOTE: fancy-index copy (all_descs[valid_idx].copy()) briefly doubles
    # peak RAM (~2x index size) and gets OOM-killed on multi-GB indexes.
    print("[INDEX] Filtering valid descriptors...")
    n_valid = len(valid_idx)
    if n_valid == idx:
        descs_valid = all_descs  # nothing filtered — no copy needed
    else:
        # valid_idx is sorted ascending, so row j's source index is >= j and
        # forward compaction never overwrites a row before it is read
        for j, src in enumerate(valid_idx):
            if j != src:
                all_descs[j] = all_descs[src]
        descs_valid = all_descs[:n_valid]

    print("[INDEX] Normalizing in-place (chunked)...")
    NORM_CHUNK = 200_000
    for s in range(0, n_valid, NORM_CHUNK):
        chunk = descs_valid[s:s + NORM_CHUNK]
        norms = np.sqrt(np.einsum('ij,ij->i', chunk, chunk))[:, None]
        norms[norms == 0] = 1
        chunk /= norms

    COMPACT_INDEX_DIR = index_dir
    COMPACT_DESCS_PATH = os.path.join(COMPACT_INDEX_DIR, "megaloc_descriptors.npy")
    COMPACT_META_PATH = os.path.join(COMPACT_INDEX_DIR, "metadata.npz")
    COMPACT_INFO_PATH = os.path.join(COMPACT_INDEX_DIR, "index_info.txt")
    np.save(COMPACT_DESCS_PATH, descs_valid)
    del descs_valid, all_descs

    print("[INDEX] Saving metadata...")
    np.savez_compressed(COMPACT_META_PATH,
        lats=lats[valid_idx], lons=lons[valid_idx], headings=headings[valid_idx],
        panoids=np.array([panoids[i] for i in valid_idx], dtype=str),
        paths=np.array([all_paths[i] for i in valid_idx], dtype=str)
    )

    COMPACT_INFO_PATH = os.path.join(COMPACT_INDEX_DIR, "index_info.txt")
    size_d = os.path.getsize(COMPACT_DESCS_PATH) / 1024 / 1024
    size_m = os.path.getsize(COMPACT_META_PATH) / 1024 / 1024
    with open(COMPACT_INFO_PATH, 'w') as f:
        f.write(f"Compact Index Info\n")
        f.write(f"Built: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Entries: {len(valid_idx)}\n")
        f.write(f"Descriptor dim: {final_dim}\n")
        f.write(f"Raw dim (pre-PCA): {raw_dim}\n")
        f.write(f"Total: {size_d + size_m:.1f} MB\n")

    # Write manifest.json LAST, once every data file for this index exists.
    # scan_indexes()/load_index() treat manifest.json's presence as the
    # signal that an index is complete and safe to use -- writing it first
    # (or not at all) would make a half-built index look ready, or make a
    # fully-built one invisible.
    try:
        idx_lats = lats[valid_idx]
        idx_lons = lons[valid_idx]
        if len(idx_lats):
            # Don't use a raw mean here -- for coastal/scattered coverage
            # (e.g. points along a curving shoreline, or split across a
            # fjord/strait), the arithmetic average of lat/lon can land in
            # open water or otherwise nowhere any real indexed point exists,
            # even though every actual point is on land. Instead, snap to
            # whichever real indexed point is closest to that average, so
            # "coverage center" is always a genuine indexed location.
            mean_lat = float(np.mean(idx_lats))
            mean_lon = float(np.mean(idx_lons))
            dists_sq = (idx_lats - mean_lat) ** 2 + (idx_lons - mean_lon) ** 2
            nearest_idx = int(np.argmin(dists_sq))
            center_lat = float(idx_lats[nearest_idx])
            center_lon = float(idx_lons[nearest_idx])

            # radius_km was previously never computed here at all -- every
            # manifest this function wrote had radius_km stuck at null, so
            # the GUI's "sync radius from the loaded index" logic silently
            # did nothing and left radius_var at whatever stale value it
            # already had (e.g. a leftover 0.09km from an earlier search),
            # producing a near-zero search radius around an otherwise
            # correct center. Compute it as the true max distance from the
            # chosen center to any indexed point, with small headroom.
            _dists_km = haversine_vec(center_lat, center_lon, idx_lats, idx_lons)
            radius_km = float(np.max(_dists_km)) * 1.05 if len(_dists_km) else None
        else:
            center_lat = center_lon = radius_km = None
    except Exception:
        center_lat = center_lon = radius_km = None

    write_index_manifest(
        index_dir,
        index_id,
        name=index_id,
        encoder=build_encoder,
        descriptor_dim=final_dim,
        num_entries=len(valid_idx),
        center_lat=center_lat,
        center_lon=center_lon,
        radius_km=radius_km,
    )

    print(f"\n[INDEX] ✅ Saved compact index:")
    print(f"  ID: {index_id}")
    print(f"  Descriptors: {COMPACT_DESCS_PATH} ({size_d:.1f} MB)")
    print(f"  Metadata: {COMPACT_META_PATH} ({size_m:.1f} MB)")
    print(f"  Descriptor dim: {final_dim} (from raw {raw_dim})")
    print(f"  Total: {size_d + size_m:.1f} MB")

    _compact_cache = None  # Force reload
    return True



def load_compact_index():
    """Load compact index into memory. Returns (descriptors, metadata_dict)."""
    global _compact_cache
    if _compact_cache is not None:
        return _compact_cache
    if not has_active_index() or not os.path.exists(COMPACT_DESCS_PATH) or not os.path.exists(COMPACT_META_PATH):
        print("[INDEX] ERROR: No active index. Build or load one first.")
        return None, None
    print("[INDEX] Loading compact index (memory-mapped)...")
    t0 = time.time()
    # Use mmap_mode='r' to keep the 7.4GB descriptors on disk and stream them into RAM
    descs = np.load(COMPACT_DESCS_PATH, mmap_mode='r')
    meta = np.load(COMPACT_META_PATH, allow_pickle=False)
    metadata = {
        'lats': meta['lats'].copy(), 'lons': meta['lons'].copy(),
        'headings': meta['headings'].copy(),
        'panoids': meta['panoids'], 'paths': meta['paths'],
    }
    del meta
    elapsed = time.time() - t0
    print(f"[INDEX] Loaded {len(descs)} entries ({descs.shape[1]}-dim) in {elapsed:.1f}s [mmap]")
    _compact_cache = (descs, metadata)
    return descs, metadata


def search_compact_index(query_desc, center, radius_km, top_k=100):
    """Search: radius filter → chunked dot-product → panoid dedup → top-K."""
    descs, metadata = load_compact_index()
    if descs is None:
        return []
    t0 = time.time()
    lat1 = np.radians(center[0])
    lon1 = np.radians(center[1])
    lat2 = np.radians(metadata['lats'])
    lon2 = np.radians(metadata['lons'])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    distances = 6371 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    radius_mask = distances <= radius_km
    radius_indices = np.where(radius_mask)[0]
    n_in_radius = len(radius_indices)
    print(f"[INDEX] Radius filter: {n_in_radius}/{len(descs)} in {radius_km}km ({time.time()-t0:.2f}s)")
    if n_in_radius == 0:
        return []

    t1 = time.time()
    query_norm = query_desc / (np.linalg.norm(query_desc) + 1e-8)
    query_norm = query_norm.astype(np.float32)

    # Guard against a stale/mismatched query descriptor -- e.g. the encoder
    # radio button was switched after an index was already loaded, so
    # ACTIVE_ENCODER (and therefore the query descriptor's dimension) no
    # longer matches the dimension of the currently loaded index's
    # descriptors. This used to crash deep inside the matmul with a cryptic
    # numpy ValueError; fail clearly here instead, and don't require a
    # rebuild -- the index on disk is fine, only the in-memory encoder
    # selection is out of sync with what's loaded.
    if query_norm.shape[-1] != descs.shape[-1]:
        print(f"[INDEX] ERROR: Query descriptor is {query_norm.shape[-1]}-dim "
              f"but the loaded index's descriptors are {descs.shape[-1]}-dim. "
              f"The active encoder ('{ACTIVE_ENCODER}') doesn't match the "
              f"encoder this index was built with. Re-select this index "
              f"from 'Select Index...' to resync, then search again.")
        return []

    # Chunked dot product — caps RAM at ~200MB per chunk
    CHUNK_SIZE = 100_000
    top_scores = np.full(top_k * 2, -np.inf, dtype=np.float32)  # keep 2x for panoid dedup
    top_indices = np.zeros(top_k * 2, dtype=np.int64)

    for chunk_start in range(0, n_in_radius, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, n_in_radius)
        chunk_idx = radius_indices[chunk_start:chunk_end]
        chunk_descs = np.array(descs[chunk_idx], dtype=np.float32)
        chunk_sims = chunk_descs @ query_norm
        del chunk_descs

        combined_scores = np.concatenate([top_scores, chunk_sims])
        combined_indices = np.concatenate([top_indices, chunk_idx])
        k = min(top_k * 2, len(combined_scores))
        best_k = np.argsort(combined_scores)[::-1][:k]
        top_scores = combined_scores[best_k]
        top_indices = combined_indices[best_k]

    # Panoid dedup: keep best heading per panoid, then top_k unique panoids
    seen_panoids = {}
    for gi, score in zip(top_indices, top_scores):
        if score == -np.inf:
            break
        pid = str(metadata['panoids'][gi])
        if pid not in seen_panoids or score > seen_panoids[pid]['score']:
            seen_panoids[pid] = {
                'panoid': pid,
                'heading': int(metadata['headings'][gi]),
                'lat': float(metadata['lats'][gi]),
                'lon': float(metadata['lons'][gi]),
                'score': float(score),
                'path': str(metadata['paths'][gi]),
            }

    results = sorted(seen_panoids.values(), key=lambda x: x['score'], reverse=True)[:top_k]
    print(f"[INDEX] Search: top-{len(results)} unique panoids in {time.time()-t1:.2f}s (best: {results[0]['score']:.3f})")
    return results


# ═══════════════════════════════════════════════════════════════════
# UI THEME
# ═══════════════════════════════════════════════════════════════════

# Each platform's own interface font (the fonts the UI used to name -- Inter,
# SF Pro Display, Avenir Next -- are missing on most machines, so Tk silently
# substituted different fallbacks per widget).
if sys.platform == "darwin":
    UI_FONT = ".AppleSystemUIFont"
elif sys.platform.startswith("win"):
    UI_FONT = "Segoe UI"
else:
    UI_FONT = "DejaVu Sans"

def fs(size):
    """Font size tuned on macOS (72 dpi); Windows/Linux render points ~1.33x
    larger, so scale down to keep the same visual size."""
    return size if sys.platform == "darwin" else max(7, round(size * 0.78))

BG = '#0a0a0f'            # window background
CARD = '#11111a'          # card surface
CARD_BORDER = '#23233a'
FIELD = '#1a1a2e'         # inputs and secondary buttons
FIELD_HOVER = '#252538'
FIELD_PRESSED = '#12121a'
ACCENT = '#8b5cf6'
ACCENT_DARK = '#7c3aed'
ACCENT_LIGHT = '#a78bfa'
TEXT = '#f3f4f6'
MUTED = '#9ca3af'
SUBTLE = '#6b7280'

SETTINGS_PATH = os.path.join(os.path.expanduser("~"), ".netryx_settings.json")

# "Try the demo": a small Community Hub index and a photo taken inside it
DEMO_INDEX_REPO = "netryx-hub/moscow-1km-1km"
DEMO_CENTER = (55.7569, 37.6151)
DEMO_PHOTO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "demo", "demo_photo.jpg")

def load_settings():
    try:
        with open(SETTINGS_PATH) as f:
            return json.load(f)
    except Exception:
        return {}

def set_app_icon(root):
    """The Dock icon on macOS (otherwise Python's rocket), the title bar /
    taskbar icon on Windows and Linux. Found next to this file, so it works
    whatever folder the app is started from."""
    if getattr(root, "_netryx_icon", None) is not None:
        return
    try:
        root._netryx_icon = tk.PhotoImage(
            master=root, file=os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "icon.png"))
        root.iconphoto(True, root._netryx_icon)
    except Exception as e:
        print(f"[UI] Could not set the app icon: {e}")

def save_settings(settings):
    try:
        with open(SETTINGS_PATH, "w") as f:
            json.dump(settings, f, indent=1)
    except Exception as e:
        print(f"[SETTINGS] Could not save settings: {e}")


class ProgressTracker:
    def __init__(self, total_items, estimate_storage=False, embeddings_per_item=4, avg_bytes_per_embedding=2560):
        self.total = total_items
        self.start_time = time.time()
        self.processed = 0
        self.estimate_storage = estimate_storage
        self.embeddings_per_item = embeddings_per_item
        self.avg_bytes_per_embedding = avg_bytes_per_embedding

    def update(self, current_count):
        self.processed = current_count

    def get_status(self):
        elapsed = time.time() - self.start_time
        if elapsed > 0.5 and self.processed > 0:
            speed = self.processed / elapsed
            remaining = self.total - self.processed
            # calc eta string
            eta_seconds = remaining / speed if speed > 0 else 0
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds)) if eta_seconds > 3600 else time.strftime("%M:%S", time.gmtime(eta_seconds))
            speed_fmt = f"{speed:.2f}"
        else:
            eta_str = "calculating..."
            speed_fmt = "--"
        percent = int((self.processed / self.total) * 100) if self.total > 0 else 0
        storage_str = ""
        if self.estimate_storage:
            total_bytes = self.total * self.embeddings_per_item * self.avg_bytes_per_embedding
            if total_bytes < 1024 * 1024:
                storage_str = f" | Storage: {total_bytes / 1024:.1f} KB"
            elif total_bytes < 1024 * 1024 * 1024:
                storage_str = f" | Storage: {total_bytes / (1024 * 1024):.1f} MB"
            else:
                storage_str = f" | Storage: {total_bytes / (1024 * 1024 * 1024):.2f} GB"
        return f"{self.processed}/{self.total} ({percent}%) | {speed_fmt} it/s | ETA: {eta_str}{storage_str}"


# GUI stuff for the app

class RoundedButton(tk.Canvas):
    def __init__(self, parent, text, command, width=200, height=44,
                 corner_radius=12, bg_color='#8b5cf6', hover_color='#a78bfa',
                 pressed_color='#7c3aed', text_color='#ffffff',
                 font=(UI_FONT, 11, 'bold'), debounce=0.3):
        try:
            parent_bg = parent.cget('bg')
        except:
            parent_bg = '#0a0a0f'
        super().__init__(parent, width=width, height=height,
                        highlightthickness=0, bg=parent_bg, cursor='hand2')
        self.command = command
        self.debounce = debounce
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.pressed_color = pressed_color
        self.text_color = text_color
        self.corner_radius = corner_radius
        self.width = width
        self.height = height
        self._text = text
        self._font = font
        self._draw_button(bg_color)
        self.bind('<Enter>', self._on_hover)
        self.bind('<Leave>', self._on_leave)
        self.bind('<Button-1>', self._on_press)
        self.bind('<ButtonRelease-1>', self._on_release)

    def _create_rounded_rect(self, x1, y1, x2, y2, r, **kwargs):
        points = [x1+r, y1, x2-r, y1, x2, y1, x2, y1+r, x2, y2-r, x2, y2,
                  x2-r, y2, x1+r, y2, x1, y2, x1, y2-r, x1, y1+r, x1, y1]
        return self.create_polygon(points, smooth=True, **kwargs)

    def _draw_button(self, color=None):
        if color is None:
            color = '#333333' if getattr(self, '_disabled', False) else self.bg_color
        self.delete('all')
        self._create_rounded_rect(2, 2, self.width-2, self.height-2,
                                  self.corner_radius, fill=color, outline='')
        self.create_text(self.width/2, self.height/2, text=self._text,
                        fill=self.text_color, font=self._font)

    def _on_hover(self, event):
        if not getattr(self, '_disabled', False): self._draw_button(self.hover_color)
    def _on_leave(self, event):
        if not getattr(self, '_disabled', False): self._draw_button(self.bg_color)
    def _on_press(self, event):
        if not getattr(self, '_disabled', False): self._draw_button(self.pressed_color)
    def _on_release(self, event):
        if not getattr(self, '_disabled', False):
            self._draw_button(self.hover_color)
            # A double-click is two press/release pairs; ignore the second
            # one so a double-click never runs the command twice. Kept short
            # (0.3 s) so deliberate quick clicks still register.
            now = time.monotonic()
            if now - getattr(self, '_last_fired', 0.0) < self.debounce:
                return
            self._last_fired = now
            if self.command: self.command()

    def configure(self, **kwargs):
        if 'text' in kwargs: self._text = kwargs['text']
        if 'command' in kwargs: self.command = kwargs['command']
        if 'state' in kwargs:
            self._disabled = kwargs['state'] == 'disabled'
        # Redraw in the colour for the current state: changing the text of a
        # disabled button used to repaint it as enabled.
        self._draw_button()
    config = configure


class RoundedEntry(tk.Canvas):
    def __init__(self, parent, textvariable=None, width=200, height=36, corner_radius=10,
                 bg_color='#1a1a2e', text_color='#ffffff', border_color='#2d2d3f',
                 focus_color='#8b5cf6', font=(UI_FONT, 10), **kwargs):
        try:
            parent_bg = parent.cget('bg')
        except:
            parent_bg = '#0a0a0f'
        super().__init__(parent, width=width, height=height, highlightthickness=0, bg=parent_bg)
        self.corner_radius = corner_radius
        self.bg_color = bg_color
        self.border_color = border_color
        self.focus_color = focus_color
        self.width = width
        self.height = height
        self.entry = tk.Entry(self, textvariable=textvariable, font=font,
                             bg=bg_color, fg=text_color, borderwidth=0,
                             insertbackground='white', highlightthickness=0)
        self._draw_background(self.border_color)
        self.create_window(width/2, height/2, window=self.entry, width=width-24, height=height-8)
        self.entry.bind('<FocusIn>', lambda e: self._draw_background(self.focus_color))
        self.entry.bind('<FocusOut>', lambda e: self._draw_background(self.border_color))

    def _draw_background(self, border_col):
        super().delete('bg')
        points = [1+self.corner_radius, 1, self.width-1-self.corner_radius, 1, self.width-1, 1,
                  self.width-1, 1+self.corner_radius, self.width-1, self.height-1-self.corner_radius,
                  self.width-1, self.height-1, self.width-1-self.corner_radius, self.height-1,
                  1+self.corner_radius, self.height-1, 1, self.height-1, 1, self.height-1-self.corner_radius,
                  1, 1+self.corner_radius, 1, 1]
        self.create_polygon(points, smooth=True, fill=self.bg_color,
                          outline=border_col, width=1, tags='bg')
        self.tag_lower('bg')

    def get(self): return self.entry.get()
    def insert(self, *args): return self.entry.insert(*args)
    def delete(self, *args): return self.entry.delete(*args)


class RoundedRadio(tk.Canvas):
    def __init__(self, parent, text, variable, value, width=120, height=30,
                 bg_color='#0a0a0f', active_color='#8b5cf6',
                 text_color='#ffffff', font=(UI_FONT, 10), command=None):
        super().__init__(parent, width=width, height=height, highlightthickness=0, bg=bg_color, cursor='hand2')
        self.variable = variable
        self.value = value
        self.command = command
        self.active_color = active_color
        self.text_color = text_color
        self._text = text
        self._font = font
        self.bind('<Button-1>', self._on_click)
        self.variable.trace_add("write", self._update_state)
        self._update_state()

    def _on_click(self, event):
        self.variable.set(self.value)
        if self.command: self.command()

    def _update_state(self, *args):
        self.delete('all')
        is_selected = (self.variable.get() == self.value)
        cy, r, x_circle = 15, 8, 15
        ring_color = self.active_color if is_selected else '#6b7280'
        self.create_oval(x_circle-r, cy-r, x_circle+r, cy+r, outline=ring_color, width=2)
        if is_selected:
            r_inner = 4
            self.create_oval(x_circle-r_inner, cy-r_inner, x_circle+r_inner, cy+r_inner, fill=self.active_color, outline='')
        self.create_text(x_circle + 20, cy, text=self._text, anchor='w', fill=self.text_color, font=self._font)


class SegmentedToggle(tk.Canvas):
    """Pill-shaped choice between a few options, bound to a tk variable."""
    def __init__(self, parent, options, variable, command=None, width=380, height=34):
        try:
            parent_bg = parent.cget('bg')
        except Exception:
            parent_bg = BG
        super().__init__(parent, width=width, height=height, highlightthickness=0, bg=parent_bg, cursor='hand2')
        self.options, self.variable, self.command = options, variable, command
        self.w, self.h = width, height
        self.bind('<Button-1>', self._on_click)
        self.variable.trace_add("write", lambda *a: self._draw())
        self._draw()

    def _pill(self, x1, y1, x2, y2, r, **kw):
        pts = [x1+r, y1, x2-r, y1, x2, y1, x2, y1+r, x2, y2-r, x2, y2,
               x2-r, y2, x1+r, y2, x1, y2, x1, y2-r, x1, y1+r, x1, y1]
        return self.create_polygon(pts, smooth=True, **kw)

    def _draw(self):
        self.delete('all')
        self._pill(1, 1, self.w - 1, self.h - 1, 12, fill=FIELD, outline=CARD_BORDER)
        seg = (self.w - 8) / len(self.options)
        for i, (label, value) in enumerate(self.options):
            x1 = 4 + i * seg
            selected = self.variable.get() == value
            if selected:
                self._pill(x1, 4, x1 + seg, self.h - 4, 10, fill=ACCENT_DARK, outline='')
            self.create_text(x1 + seg / 2, self.h / 2, text=label,
                             fill='#ffffff' if selected else MUTED,
                             font=(UI_FONT, fs(12), 'bold' if selected else 'normal'))

    def _on_click(self, event):
        seg = (self.w - 8) / len(self.options)
        i = min(len(self.options) - 1, max(0, int((event.x - 4) // seg)))
        value = self.options[i][1]
        if self.variable.get() != value:
            self.variable.set(value)
            if self.command:
                self.command()


# the main gui class for the app

class StreetViewMatcherGUI:
    def __init__(self, master):
        self.master = master
        master.title("Netryx Astra v2 | AI Geolocation")
        set_app_icon(master)
        master.configure(bg=BG)
        self.settings = load_settings()
        master.geometry(self.settings.get("geometry", "1440x920"))
        master.minsize(1100, 700)

        # vars for the gui (area and mode come back from the last session)
        s = self.settings
        self.lat_var = tk.DoubleVar(value=s.get("lat", 40.7132))   # NYC index center
        self.lon_var = tk.DoubleVar(value=s.get("lon", -74.0025))
        self.radius_var = tk.DoubleVar(value=s.get("radius_km", 13.0))
        self.res_var = tk.IntVar(value=s.get("grid_spacing_m", 300))
        self.match_threshold = tk.IntVar(value=50)
        self.crop_fov = tk.IntVar(value=90)
        self.crop_size = tk.IntVar(value=256)
        self.crop_step = tk.IntVar(value=90)
        self.query_img_path = None
        self.mode_var = tk.StringVar(value=s.get("mode", "search"))
        self.search_option_var = tk.StringVar(value="manual")
        self.encoder_var = tk.StringVar(value=ACTIVE_ENCODER)
        self.hf_token_var = tk.StringVar(value=os.getenv("HF_TOKEN", ""))
        # Index selector state. selected_index_ids can hold multiple picks
        # in the UI, but only the first is actually activated for now --
        # true multi-index search is future work (after MixVPR is solid).
        self.selected_index_ids = []
        self.index_selector_var = tk.StringVar(value="No index selected")

        # theme and styles
        style = ttk.Style(master)
        style.theme_use('clam')
        style.configure('TFrame', background=BG)
        style.configure('TLabel', background=BG, foreground=TEXT, font=(UI_FONT, fs(12)))
        style.configure('Horizontal.TProgressbar', background=ACCENT, troughcolor=FIELD,
                        bordercolor=CARD, lightcolor=ACCENT, darkcolor=ACCENT, thickness=8)
        style.configure('TButton', background=FIELD, foreground=TEXT, font=(UI_FONT, fs(12)), borderwidth=0)
        style.map('TButton', background=[('active', FIELD_HOVER)])
        style.configure('Vertical.TScrollbar', background=FIELD, troughcolor=BG, bordercolor=BG,
                        arrowcolor=MUTED, lightcolor=FIELD, darkcolor=FIELD)
        style.configure("Treeview", background=CARD, foreground=TEXT, fieldbackground=CARD,
                        rowheight=30, font=(UI_FONT, fs(12)), borderwidth=0)
        style.map("Treeview", background=[('selected', '#5b21b6')], foreground=[('selected', '#ffffff')])
        style.configure("Treeview.Heading", background=FIELD, foreground=ACCENT_LIGHT,
                        font=(UI_FONT, fs(11), 'bold'), padding=6, borderwidth=0)
        style.map("Treeview.Heading", background=[('active', FIELD_HOVER)])

        # ── Layout: fixed-width scrolling sidebar on the left, the map fills the rest ──
        SIDEBAR_W = 440
        self._cards = []                 # step 1, step 2, step 3, results, sharing
        INNER_W = SIDEBAR_W - 56          # card content width (card padding + scrollbar)
        self._inner_w = INNER_W
        secondary = dict(bg_color=FIELD, hover_color=FIELD_HOVER, pressed_color=FIELD_PRESSED)

        frm = tk.Frame(master, bg=BG)
        frm.pack(fill='both', expand=True, padx=16, pady=16)
        frm.columnconfigure(0, weight=0, minsize=SIDEBAR_W)
        frm.columnconfigure(1, weight=1)
        frm.rowconfigure(0, weight=1)

        sidebar_container = tk.Frame(frm, bg=BG)
        sidebar_container.grid(row=0, column=0, sticky='nsew')
        self.sidebar_container = sidebar_container
        self.sidebar_canvas = tk.Canvas(sidebar_container, bg=BG, highlightthickness=0, width=SIDEBAR_W - 16)
        scrollbar = ttk.Scrollbar(sidebar_container, orient="vertical", command=self.sidebar_canvas.yview)
        self.sidebar_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.sidebar_canvas.configure(yscrollcommand=scrollbar.set)

        left_ctrl = tk.Frame(self.sidebar_canvas, bg=BG)
        self.left_ctrl = left_ctrl
        self._tour_left_ctrl = left_ctrl
        self.canvas_window = self.sidebar_canvas.create_window((0, 0), window=left_ctrl, anchor="nw")
        left_ctrl.bind("<Configure>", lambda e: self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all")))
        self.sidebar_canvas.bind("<Configure>", lambda e: self.sidebar_canvas.itemconfig(self.canvas_window, width=e.width - 4))
        self.sidebar_canvas.bind_all("<MouseWheel>", self._on_sidebar_wheel)

        # Header
        header = tk.Frame(left_ctrl, bg=BG)
        header.pack(fill='x', pady=(2, 2))
        tk.Label(header, text="Netryx Astra", font=(UI_FONT, fs(22), 'bold'), bg=BG, fg=TEXT).pack(side='left')
        tk.Label(header, text="v2", font=(UI_FONT, fs(13), 'bold'), bg=BG, fg=ACCENT_LIGHT).pack(side='left', padx=(6, 0), pady=(7, 0))
        self.help_btn = RoundedButton(header, text="?  Help & tour", command=lambda: self.show_tutorial(force=True),
                                      width=120, height=30, font=(UI_FONT, fs(12), 'bold'), **secondary)
        self.help_btn.pack(side='right')
        tk.Label(left_ctrl, text="Find where a street-level photo was taken.", font=(UI_FONT, fs(12)),
                 bg=BG, fg=MUTED).pack(anchor='w', pady=(0, 12))

        # One-click demo: sample index + sample photo, so a first search works right away
        demo = tk.Frame(left_ctrl, bg=CARD, highlightbackground=ACCENT_DARK, highlightthickness=1)
        demo.pack(fill='x', pady=(0, 12))
        self.demo_frame = demo
        tk.Label(demo, text="New here? Try a search in one click.", font=(UI_FONT, fs(12), 'bold'),
                 bg=CARD, fg=TEXT).pack(anchor='w', padx=16, pady=(12, 2))
        tk.Label(demo, text="Loads a small sample index (central Moscow) and a sample photo. "
                            "Then just press Run Search.", font=(UI_FONT, fs(11)), bg=CARD, fg=MUTED,
                 wraplength=INNER_W, justify='left').pack(anchor='w', padx=16)
        self.demo_btn = RoundedButton(demo, text="Try the demo", command=self.try_demo,
                                      width=INNER_W, height=36, font=(UI_FONT, fs(12), 'bold'))
        self.demo_btn.pack(padx=16, pady=(10, 14))
        self.demo_status = tk.Label(demo, text="", font=(UI_FONT, fs(11)), bg=CARD, fg=MUTED,
                                    wraplength=INNER_W, justify='left', anchor='w')

        # ── Step 1: what to do ──
        card, body, self._badge1 = self._card(left_ctrl, "Choose what to do", step=1)
        self._field_label(body, "Mode")
        self.mode_frame = SegmentedToggle(body, [("Search a photo", "search"), ("Create an index", "create")],
                                          self.mode_var, command=self._update_mode, width=INNER_W)
        self.mode_frame.pack(anchor='w', pady=(0, 12))
        self._field_label(body, "Encoder")
        self.encoder_frame = SegmentedToggle(body, [("MegaLoc", "megaloc"), ("MixVPR", "mixvpr")],
                                             self.encoder_var, command=self._on_encoder_change, width=INNER_W)
        self.encoder_frame.pack(anchor='w')
        self._hint(body, "MegaLoc is more accurate; MixVPR builds indexes faster. Each has its own indexes.")
        self._field_label(body, "Index", top=12)
        self.index_selector_label = tk.Label(body, textvariable=self.index_selector_var, font=(UI_FONT, fs(12)),
                                             bg=CARD, fg=TEXT, wraplength=INNER_W, justify='left', anchor='w')
        self.index_selector_label.pack(fill='x', pady=(0, 8))
        idx_row = tk.Frame(body, bg=CARD)
        idx_row.pack(fill='x')
        half = (INNER_W - 8) // 2
        self.index_selector_btn = RoundedButton(idx_row, text="Select index", command=self.show_index_selector,
                                                width=half, height=36, font=(UI_FONT, fs(12), 'bold'), **secondary)
        self.index_selector_btn.pack(side='left')
        self.hub_btn = RoundedButton(idx_row, text="Community Hub", command=self.show_community_hub,
                                     width=half, height=36, font=(UI_FONT, fs(12), 'bold'), **secondary)
        self.hub_btn.pack(side='right')
        self.coverage_btn = RoundedButton(body, text="Show coverage on the map", command=self.show_coverage_map,
                                          width=INNER_W, height=36, font=(UI_FONT, fs(12), 'bold'), **secondary)
        self.coverage_btn.pack(pady=(8, 0))
        self._mode_hint = self._hint(body, "")

        # ── Step 2: the area ──
        card, body, self._badge2 = self._card(left_ctrl, "Set the area", step=2)
        grid = tk.Frame(body, bg=CARD)
        grid.pack(fill='x')
        grid.columnconfigure((0, 1), weight=1)
        self._coord_labels = []
        fields = [("Latitude", self.lat_var, 0, 0), ("Longitude", self.lon_var, 0, 1),
                  ("Radius (km)", self.radius_var, 1, 0), ("Grid spacing (m)", self.res_var, 1, 1)]
        for txt, var, r, c in fields:
            cell = tk.Frame(grid, bg=CARD)
            cell.grid(row=r, column=c, sticky='w', padx=(0 if c == 0 else 8, 0), pady=(0, 10))
            lbl = self._field_label(cell, txt)
            RoundedEntry(cell, textvariable=var, width=half, height=34, font=(UI_FONT, fs(12)),
                         bg_color=FIELD, border_color=CARD_BORDER).pack(anchor='w')
            self._coord_labels.append(lbl)
        self._area_frame = grid
        self._hint(body, "The search only looks inside this circle, so it must overlap your index. "
                         "Grid spacing is only used when creating an index.", top=0)

        # ── Step 3: the photo ──
        card, body, self._badge3 = self._card(left_ctrl, "Choose a photo", step=3)
        self._photo_card = card
        self.photo_zone = tk.Frame(body, bg=FIELD, highlightbackground=CARD_BORDER, highlightthickness=1,
                                   cursor='hand2', height=150)
        self.photo_zone.pack(fill='x')
        self.photo_zone.pack_propagate(False)
        self.query_img_label = tk.Label(self.photo_zone, text="Click to choose a photo\nor drag one here",
                                        font=(UI_FONT, fs(12)), bg=FIELD, fg=MUTED, cursor='hand2')
        self.query_img_label.pack(expand=True)
        for w in (self.photo_zone, self.query_img_label):
            w.bind("<Button-1>", lambda e: self.choose_photo())
        self.photo_name_label = tk.Label(body, text="", font=(UI_FONT, fs(11)), bg=CARD, fg=MUTED, anchor='w')
        self.photo_name_label.pack(fill='x', pady=(6, 0))
        self._hint(body, "Best: daytime, eye level, buildings or storefronts in view. Avoid mostly sky, trees or night shots.", top=2)

        # ── Run ──
        run_frame = tk.Frame(left_ctrl, bg=BG)
        run_frame.pack(fill='x', pady=(4, 12))
        self._run_frame = run_frame
        self.query_btn = RoundedButton(run_frame, text="▶  Run Search", command=self.run,
                                       width=INNER_W + 32, height=50, font=(UI_FONT, fs(14), 'bold'))
        self.query_btn.pack()
        self.run_hint = tk.Label(run_frame, text="", font=(UI_FONT, fs(12)), bg=BG, fg=ACCENT_LIGHT,
                                 wraplength=INNER_W + 32, justify='left', anchor='w')
        self.run_hint.pack(fill='x', pady=(8, 0))
        self.cancel_btn = RoundedButton(run_frame, text="■  Cancel Search", command=self.cancel_current_search,
            width=INNER_W + 32, height=40, bg_color='#7f1d1d', hover_color='#991b1b', pressed_color='#5f1515',
            font=(UI_FONT, fs(12), 'bold'))
        # hidden until a search is running (shown by start_full_search)
        self.status_label = tk.Label(run_frame, text="Ready.", font=(UI_FONT, fs(12)), bg=BG, fg=MUTED,
                                     wraplength=INNER_W + 32, justify='left', anchor='w')
        self.status_label.pack(fill='x', pady=(8, 6))
        self.progress = ttk.Progressbar(run_frame, orient="horizontal", mode="determinate")
        self.progress.pack(fill='x')

        # ── Results ──
        card, body, _ = self._card(left_ctrl, "Results")
        self.results_card = card
        self.result_title = tk.Label(body, text="No results yet. Run a search and the best match appears here.",
                                     font=(UI_FONT, fs(12)), bg=CARD, fg=MUTED, wraplength=INNER_W, justify='left', anchor='w')
        self.result_title.pack(fill='x')
        self.result_detail = tk.Label(body, text="", font=(UI_FONT, fs(11)), bg=CARD, fg=MUTED,
                                      wraplength=INNER_W, justify='left', anchor='w')
        self.result_detail.pack(fill='x', pady=(2, 0))
        pair = tk.Frame(body, bg=CARD)
        pair.pack(fill='x', pady=(10, 0))
        pair.columnconfigure((0, 1), weight=1)
        self.result_query_img = tk.Label(pair, bg=CARD)
        self.result_query_img.grid(row=0, column=0)
        self.canvas = tk.Label(pair, bg=CARD)   # the matched Street View crop
        self.canvas.grid(row=0, column=1)
        self.result_caption_q = tk.Label(pair, text="", font=(UI_FONT, fs(11)), bg=CARD, fg=MUTED)
        self.result_caption_q.grid(row=1, column=0, pady=(4, 0))
        self.result_caption_m = tk.Label(pair, text="", font=(UI_FONT, fs(11)), bg=CARD, fg=MUTED)
        self.result_caption_m.grid(row=1, column=1, pady=(4, 0))
        self.match_lines_var = tk.BooleanVar(value=True)
        self.match_lines_toggle = tk.Checkbutton(body, text="Show matching points", variable=self.match_lines_var,
            command=self._render_result_images, bg=CARD, fg=MUTED, selectcolor=FIELD, activebackground=CARD,
            activeforeground=TEXT, font=(UI_FONT, fs(11)), highlightthickness=0, borderwidth=0)

        res_btns = tk.Frame(body, bg=CARD)
        res_btns.pack(fill='x', pady=(12, 0))
        self._res_btns = res_btns
        self.maps_btn = RoundedButton(res_btns, text="Open in Google Maps", command=self.open_res_gmaps,
                                      width=half, height=38, font=(UI_FONT, fs(12), 'bold'))
        self.maps_btn.pack(side='left')
        self.copy_btn = RoundedButton(res_btns, text="Copy coordinates", command=self.copy_res_coords,
                                      width=half, height=38, font=(UI_FONT, fs(12), 'bold'), **secondary)
        self.copy_btn.pack(side='right')

        self._field_label(body, "Other likely spots (click one to show it on the map)", top=14)
        tree_frame = tk.Frame(body, bg=CARD)
        tree_frame.pack(fill='x')
        cols = ("Rank", "Score", "Coordinates")
        self.res_tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=5, style="Treeview")
        self.res_tree.heading("Rank", text="#")
        self.res_tree.heading("Score", text="Match points")
        self.res_tree.heading("Coordinates", text="Coordinates")
        self.res_tree.column("Rank", width=34, anchor='center', stretch=False)
        self.res_tree.column("Score", width=110, anchor='center', stretch=False)
        self.res_tree.column("Coordinates", width=INNER_W - 160, anchor='w')
        res_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.res_tree.yview)
        self.res_tree.configure(yscrollcommand=res_scroll.set)
        self.res_tree.pack(side='left', fill='x', expand=True)
        res_scroll.pack(side='right', fill='y')
        self.res_menu = tk.Menu(master, tearoff=0, bg=FIELD, fg='white', activebackground=ACCENT)
        self.res_menu.add_command(label="Copy coordinates", command=self.copy_res_coords)
        self.res_menu.add_command(label="Open in Google Maps", command=self.open_res_gmaps)
        self.res_tree.bind("<Button-2>" if "darwin" in sys.platform else "<Button-3>", self.show_res_menu)
        self.res_tree.bind("<<TreeviewSelect>>", self._on_res_select)

        # ── Coverage & sharing ──
        card, body, _ = self._card(left_ctrl, "Sharing")
        io_row = tk.Frame(body, bg=CARD)
        io_row.pack(fill='x')
        self.export_btn = RoundedButton(io_row, text="Export index file", command=self.export_index,
                                        width=half, height=36, font=(UI_FONT, fs(12), 'bold'), **secondary)
        self.export_btn.pack(side='left')
        self.import_btn = RoundedButton(io_row, text="Import index file", command=self.import_index,
                                        width=half, height=36, font=(UI_FONT, fs(12), 'bold'), **secondary)
        self.import_btn.pack(side='right')
        self._hint(body, "Share an index as a .netryx file, or open one someone sent you. No account needed.", top=4)
        self._field_label(body, "Hugging Face token (only for uploading to the Community Hub)", top=12)
        self.hf_token_entry = RoundedEntry(body, textvariable=self.hf_token_var, width=INNER_W, height=34,
                                           font=(UI_FONT, fs(12)), bg_color=FIELD, border_color=CARD_BORDER)
        self.hf_token_entry.pack(anchor='w')
        self.hf_token_entry.entry.config(show="*")

        def open_hf_tokens():
            webbrowser.open("https://huggingface.co/settings/tokens")
        self.get_token_btn = tk.Label(body, text="Get a free token →", font=(UI_FONT, fs(11), 'underline'),
                                      bg=CARD, fg=ACCENT_LIGHT, cursor='hand2')
        self.get_token_btn.pack(anchor='w', pady=(6, 0))
        self.get_token_btn.bind("<Button-1>", lambda e: open_hf_tokens())

        tk.Label(left_ctrl, text="Made by Sairaj Balaji", font=(UI_FONT, fs(11)), bg=BG, fg=SUBTLE).pack(pady=(4, 12))

        # Map
        self.map_frame = tk.Frame(frm, bg=BG)
        self.map_frame.grid(row=0, column=1, sticky='nsew', padx=(16, 0))
        self.map_widget = tkintermapview.TkinterMapView(self.map_frame, corner_radius=14, bg_color=BG)
        self.map_widget.pack(fill="both", expand=True)
        # CARTO basemaps now need an API key (every tile is an "API KEY REQUIRED"
        # image) and OSM blocks tkintermapview's user agent. Esri's satellite
        # imagery and label overlay are keyless and go to zoom 19.
        self.map_widget.set_tile_server(
            "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", max_zoom=19)
        self.map_widget.set_overlay_tile_server(
            "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}")
        self.map_widget.set_position(self.lat_var.get(), self.lon_var.get())
        self.map_widget.set_zoom(15)

        # Live preview of the candidates being checked during a search (shown by _handle_scan_blip)
        self.monitor_label = tk.Label(self.map_frame, text="", bg=CARD, fg=ACCENT_LIGHT,
                                      font=(UI_FONT, fs(11), 'bold'), compound='top', padx=6, pady=6)
        tk.Label(self.map_frame, text="Imagery © Esri, Maxar, Earthstar Geographics", fg=MUTED,
                 bg=BG, font=(UI_FONT, fs(10))).place(relx=0.99, rely=0.99, anchor="se")

        # Drag a photo anywhere onto the window (needs the optional tkinterdnd2 package)
        if hasattr(master, 'drop_target_register'):
            try:
                from tkinterdnd2 import DND_FILES
                master.drop_target_register(DND_FILES)
                master.dnd_bind('<<Drop>>', self._on_drop)
            except Exception as e:
                print(f"[UI] Drag and drop unavailable: {e}")

        for var in (self.lat_var, self.lon_var, self.radius_var, self.index_selector_var):
            var.trace_add("write", lambda *a: self._refresh_steps())
        self.index_selector_var.trace_add("write", lambda *a: self.master.after(100, self._center_map_on_area))
        master.protocol("WM_DELETE_WINDOW", self._on_close)

        # State
        self.coverage_markers, self.result_elements, self.search_nets = [], [], []

        self.match_queue, self.results_queue = queue.Queue(), queue.Queue()
        self.cancel_search = threading.Event()
        self.thumbnail_pool = []
        self._thumbnail_pool_lock = threading.Lock()
        self._job = None  # name of the running long job, see _begin_job
        self._update_mode()
        self.poll_match_queue()

        # First-run onboarding tour (non-blocking; only if never seen)
        self.master.after(600, lambda: self.show_tutorial(force=False))

        # Reflect whatever index (if any) __main__ already auto-loaded
        # before this GUI was constructed.
        if has_active_index():
            try:
                info_manifest = None
                for m in scan_indexes():
                    if COMPACT_INDEX_DIR and m.get('index_id') == os.path.basename(COMPACT_INDEX_DIR):
                        info_manifest = m
                        break
                name = info_manifest.get('name') if info_manifest else os.path.basename(COMPACT_INDEX_DIR)
                self.selected_index_ids = [info_manifest['index_id']] if info_manifest else []
                self.index_selector_var.set(f"Active: {name}")
                if info_manifest:
                    cov = info_manifest.get("coverage_center", {})
                    if cov.get("lat") is not None and cov.get("lon") is not None:
                        self.lat_var.set(cov["lat"])
                        self.lon_var.set(cov["lon"])
                    if info_manifest.get("radius_km") is not None:
                        self.radius_var.set(info_manifest["radius_km"])
            except Exception:
                pass

    def _on_encoder_change(self):
        """Switch retrieval encoder. Each encoder has its own separate index."""
        if self._job_running():
            # set_encoder() swaps the parts directory and unloads the index,
            # which a running search or index build is still using.
            self.encoder_var.set(ACTIVE_ENCODER)
            return
        name = self.encoder_var.get()
        # No-op if this encoder is already active -- set_encoder() always
        # clears COMPACT_INDEX_DIR (and the loaded index with it), even when
        # nothing actually changed. Without this guard, re-clicking the
        # already-selected radio button (or any redundant call) silently
        # unloads a working index for no reason, and the next search fails
        # with "No active index" even though one was just loaded.
        if name == ACTIVE_ENCODER and has_active_index():
            self._set_status(f"Encoder: {name.upper()} — index ready.")
            return
        try:
            set_encoder(name)
        except Exception as e:
            self._set_status(f"Encoder '{name}' unavailable: {e}")
            self.encoder_var.set("megaloc")
            set_encoder("megaloc")
            return
        # set_encoder() only prepares indexing for this encoder -- it doesn't
        # select an index anymore. An index must be picked via load_index().
        if has_active_index() and os.path.exists(COMPACT_DESCS_PATH):
            self._set_status(f"Encoder: {name.upper()} — index ready.")
        else:
            self._set_status(f"Encoder: {name.upper()} — no index selected. "
                             f"Load an existing {name.upper()} index or build a new one.")
            # The "Select Index..." status label is a separate widget from
            # the main status bar -- without clearing it too, it keeps
            # showing "Active: <old index name>" even though set_encoder()
            # just cleared COMPACT_INDEX_DIR, and searching produces a
            # confusing "no candidates" result instead of an obvious
            # "you have no index loaded" signal.
            self.selected_index_ids = []
            self.index_selector_var.set(f"No {name.upper()} index selected")

    def show_index_selector(self):
        """Popup listing every index found under INDEXES_DIR (manifest.json
        present = complete). Selection is multi-pick (Ctrl/Shift/drag), but
        only the first selected index is actually activated for search right
        now -- true multi-index search is future work, once MixVPR is solid.
        """
        if self._focus_existing('_index_win'):
            return
        sel_win = tk.Toplevel(self.master)
        self._index_win = sel_win
        sel_win.title("Select Index")
        sel_win.configure(bg='#0a0a0f')
        sel_win.geometry("560x480")
        sel_win.transient(self.master)

        header = tk.Frame(sel_win, bg='#0a0a0f')
        header.pack(fill='x', padx=20, pady=(20, 10))
        tk.Label(header, text="📍 Select Index", font=(UI_FONT, 18, 'bold'),
                 bg='#0a0a0f', fg='#ffffff').pack(anchor='w')
        tk.Label(header, text="Ctrl/Cmd-click or drag to select multiple. "
                              "Only the first pick is used for search right now --\n"
                              "combining multiple indexes in one search is coming later.",
                 font=(UI_FONT, 9), bg='#0a0a0f', fg='#8b5cf6', justify='left').pack(anchor='w', pady=(4, 0))

        list_frame = tk.Frame(sel_win, bg='#12121a')
        list_frame.pack(fill='both', expand=True, padx=20, pady=10)

        listbox = tk.Listbox(list_frame, font=(UI_FONT, 10),
                              bg='#12121a', fg='#f3f4f6', selectbackground='#8b5cf6',
                              selectforeground='white', borderwidth=0,
                              highlightthickness=0, activestyle='none',
                              selectmode=tk.EXTENDED)
        listbox.pack(fill='both', expand=True, side='left')

        scrollbar = tk.Scrollbar(list_frame, command=listbox.yview)
        scrollbar.pack(side='right', fill='y')
        listbox.config(yscrollcommand=scrollbar.set)

        status_lbl = tk.Label(sel_win, text="", font=(UI_FONT, 9),
                               bg='#0a0a0f', fg='#6b7280')
        status_lbl.pack(anchor='w', padx=20)

        indexes = scan_indexes()
        if not indexes:
            status_lbl.config(text="No indexes found. Build one in Create mode, "
                                    "or download one from the Community Hub.")
        for m in indexes:
            entries = m.get('num_entries', '?')
            enc = m.get('descriptor_model', '?')
            created = m.get('created', '')
            listbox.insert(tk.END, f"{m.get('name', m.get('index_id'))}  "
                                    f"[{enc}, {entries} entries]  {created}")
        listbox._index_ids = [m.get('index_id') for m in indexes]
        listbox._manifests = indexes

        bottom = tk.Frame(sel_win, bg='#0a0a0f')
        bottom.pack(fill='x', padx=20, pady=(0, 20))

        def do_select():
            if self._job_running():
                status_lbl.config(text=f"Wait for {self._job} to finish first.")
                return
            sel = listbox.curselection()
            if not sel:
                status_lbl.config(text="Select at least one index first.")
                return
            picked_ids = [listbox._index_ids[i] for i in sel]
            picked_manifests = [listbox._manifests[i] for i in sel]
            self.selected_index_ids = picked_ids

            # Activate the first pick now. Later, once MixVPR support and
            # multi-index search land, this is where we'd fan out search
            # across all of picked_ids instead of just loading one.
            try:
                loaded_manifest = load_index(picked_ids[0])
            except Exception as e:
                status_lbl.config(text=f"Failed to load index: {e}")
                return

            # Sync search center/radius to the loaded index's coverage --
            # otherwise the lat/lon/radius fields stay at whatever was
            # typed for a *different* index and every search returns 0
            # results in the radius filter.
            cov = loaded_manifest.get("coverage_center", {})
            if cov.get("lat") is not None and cov.get("lon") is not None:
                self.lat_var.set(cov["lat"])
                self.lon_var.set(cov["lon"])
            if loaded_manifest.get("radius_km") is not None:
                self.radius_var.set(loaded_manifest["radius_km"])

            # Clear any leftover search area from a PREVIOUS index -- the
            # coverage map draws this as a yellow circle, and leaving it in
            # place after switching to an unrelated index makes it look
            # like the new index's coverage is somewhere it was never
            # actually built (e.g. a stale circle out in open water).
            self.search_nets = []

            if len(picked_ids) == 1:
                self.index_selector_var.set(f"Active: {picked_manifests[0].get('name', picked_ids[0])}")
            else:
                self.index_selector_var.set(
                    f"Active: {picked_manifests[0].get('name', picked_ids[0])}  "
                    f"(+{len(picked_ids) - 1} more selected, not yet combined in search)"
                )
            self._set_status(f"Index loaded: {picked_manifests[0].get('name', picked_ids[0])}")
            sel_win.destroy()

        select_btn = RoundedButton(bottom, text="Use Selected", command=do_select,
                                    width=160, height=40)
        select_btn.pack(side='left')

        refresh_btn = RoundedButton(bottom, text="⟳ Refresh", command=lambda: (sel_win.destroy(), self.show_index_selector()),
                                     width=120, height=40, bg_color='#1a1a2e',
                                     hover_color='#252538', pressed_color='#12121a')
        refresh_btn.pack(side='left', padx=(10, 0))

    def show_tutorial(self, force=False):
        """Guided tour: walks through every part of the window, scrolling each
        one into view and outlining it. Shown once on first launch; reopen it
        with the Help & tour button."""
        flag = os.path.join(os.path.expanduser("~"), ".netryx_tour_v2_seen")
        if not force and os.path.exists(flag):
            return
        if self._focus_existing('_tutorial_win'):
            return

        # (title, explanation, widget to outline or None)
        steps = [
            ("Welcome to Netryx Astra",
             "Netryx finds where a street-level photo was taken by comparing it with "
             "Street View panoramas of an area.\n\n"
             "This tour shows you every part of the screen. The part being explained is "
             "outlined in purple. Use Next and Back, or close the tour any time.",
             None),
            ("Try the demo",
             "The quickest way to see Netryx work: this loads a small sample index of "
             "central Moscow and a sample photo taken there. Then press Run Search and "
             "watch it find the spot.\n\n"
             "The first time, the sample index (about 180 MB) downloads from the Community Hub.",
             lambda: self.demo_frame),
            ("Step 1 · Mode",
             "Search a photo: find where your photo was taken, inside an area that "
             "already has an index.\n\n"
             "Create an index: build an index for a new area yourself. Netryx downloads "
             "that area's Street View, which can take a while for big areas.",
             lambda: self.mode_frame),
            ("Step 1 · Encoder",
             "The encoder is the model that turns pictures into searchable fingerprints.\n\n"
             "MegaLoc (recommended) is the most accurate. MixVPR builds indexes several "
             "times faster but is a little less accurate. Each encoder has its own indexes.",
             lambda: self.encoder_frame),
            ("Step 1 · Index",
             "An index is a prepared database of one area's Street View. A photo can only "
             "be found if it was taken inside the active index's area.\n\n"
             "This line shows the active index. “Select index” lists the indexes on this computer.",
             lambda: self.index_selector_btn.master.master),
            ("Step 1 · Community Hub",
             "The quickest way to start: download a ready-made index that someone else "
             "built, such as New York or Moscow.\n\n"
             "Once downloaded it becomes the active index, and its area fills in step 2.",
             lambda: self.hub_btn),
            ("Step 1 · Coverage",
             "Shows every street the active index covers, as lines on the map. If a search "
             "finds nothing, check here first: if your photo's location isn't covered, no "
             "match is possible.",
             lambda: self.coverage_btn),
            ("Step 2 · The area",
             "Latitude and longitude are the centre of the area, and the radius is how far "
             "around it to look, in kilometres. The search only looks inside this circle, "
             "so it must overlap your index.\n\n"
             "Grid spacing is only used when creating an index.",
             lambda: self._cards[1]),
            ("Step 3 · Your photo",
             "Click here to choose the photo you want to locate, or drag a photo file onto "
             "the window.\n\n"
             "Best results: daytime, eye level, with buildings, shop fronts or street "
             "furniture in view. Mostly sky, trees or night shots rarely work.",
             lambda: self.photo_zone),
            ("Run",
             "When steps 1 to 3 are ticked, press this to start. A search first shortlists "
             "likely spots, then checks each one in detail, which usually takes a few "
             "minutes. A red Cancel button appears while it runs.\n\n"
             "In Create mode the same button builds the index.",
             lambda: self.query_btn),
            ("What's missing, and progress",
             "The purple line under the button says what still needs doing before you can "
             "run. While a search or index build runs, the status line and progress bar "
             "show what's happening and how long is left.",
             lambda: self._run_frame),
            ("The map",
             "Results appear here: a pin at the best match and a circle around the search "
             "area. While searching, the candidates being checked flash on the map and a "
             "preview shows in the top corner.\n\n"
             "Scroll to zoom and drag to move around.",
             lambda: self.map_frame),
            ("Results",
             "The coordinates of the best match, your photo next to the matching Street View "
             "picture, and how many points matched. More match points means a more certain "
             "match, but always check the two pictures show the same place.\n\n"
             "Open the spot in Google Maps, copy the coordinates, or click another likely spot in the list.",
             lambda: self.results_card),
            ("Sharing indexes",
             "Export an index as a .netryx file to send to someone, or import one you were "
             "sent. No account needed.\n\n"
             "Uploading to the Community Hub needs a free Hugging Face token, entered here.",
             lambda: self._cards[4]),
            ("You're ready",
             "In short: get an index (step 1), check the area (step 2), choose a photo "
             "(step 3), then Run Search.\n\n"
             "Reopen this tour any time with “Help & tour”. The full guide has the technical details.",
             lambda: self.help_btn),
        ]

        win = tk.Toplevel(self.master)
        self._tutorial_win = win
        win.title("Netryx tour")
        win.configure(bg=CARD)
        win.resizable(False, False)
        win.transient(self.master)
        try:
            win.attributes('-topmost', True)
        except Exception:
            pass
        W, H = 470, 360
        try:  # over the map's top-right corner, so the sidebar stays visible
            self.master.update_idletasks()
            x = self.map_frame.winfo_rootx() + self.map_frame.winfo_width() - W - 24
            y = self.map_frame.winfo_rooty() + 24
            win.geometry(f"{W}x{H}+{max(x, 0)}+{max(y, 0)}")
        except Exception:
            win.geometry(f"{W}x{H}")

        # Outline = four thin strips on the main window, so it can surround
        # anything (sidebar widgets or the map) without covering it.
        T = 3
        strips = [tk.Frame(self.master, bg=ACCENT_LIGHT, highlightthickness=0) for _ in range(4)]
        state = {"i": 0, "target": None, "pulse": False}

        def scroll_into_view(target):
            w = target
            while w is not None and w is not self.left_ctrl:
                w = w.master
            if w is None:
                return
            self.master.update_idletasks()
            y = target.winfo_rooty() - self.left_ctrl.winfo_rooty()
            total = max(self.left_ctrl.winfo_height(), 1)
            self.sidebar_canvas.yview_moveto(max(0.0, (y - 70) / total))

        def place_outline():
            for s_ in strips:
                s_.place_forget()
            target = state["target"]
            if not win.winfo_exists() or target is None or not target.winfo_exists():
                return
            self.master.update_idletasks()
            x = target.winfo_rootx() - self.master.winfo_rootx() - 2 * T
            y = target.winfo_rooty() - self.master.winfo_rooty() - 2 * T
            w = target.winfo_width() + 4 * T
            h = target.winfo_height() + 4 * T
            # keep it inside the visible part of the sidebar for tall cards
            if target is not self.map_frame:
                top = self.sidebar_canvas.winfo_rooty() - self.master.winfo_rooty()
                bottom = top + self.sidebar_canvas.winfo_height()
                if y < top:
                    h -= top - y
                    y = top
                if y + h > bottom:
                    h = bottom - y
            if w < 8 or h < 8:
                return
            strips[0].place(x=x, y=y, width=w, height=T)
            strips[1].place(x=x, y=y + h - T, width=w, height=T)
            strips[2].place(x=x, y=y, width=T, height=h)
            strips[3].place(x=x + w - T, y=y, width=T, height=h)
            for s_ in strips:
                s_.lift()

        def pulse():
            if not win.winfo_exists():
                return
            state["pulse"] = not state["pulse"]
            for s_ in strips:
                s_.config(bg=ACCENT_LIGHT if state["pulse"] else ACCENT_DARK)
            win.after(500, pulse)

        counter = tk.Label(win, text="", bg=CARD, fg=SUBTLE, font=(UI_FONT, fs(11)))
        counter.pack(anchor='w', padx=24, pady=(20, 0))
        title_lbl = tk.Label(win, text="", bg=CARD, fg=TEXT, font=(UI_FONT, fs(17), 'bold'),
                             wraplength=W - 48, justify='left')
        title_lbl.pack(anchor='w', padx=24, pady=(4, 8))
        body_lbl = tk.Label(win, text="", bg=CARD, fg=MUTED, font=(UI_FONT, fs(12)),
                            wraplength=W - 48, justify='left')
        body_lbl.pack(anchor='w', padx=24)

        nav = tk.Frame(win, bg=CARD)
        nav.pack(side='bottom', fill='x', padx=20, pady=(0, 18))

        def finish():
            try:
                with open(flag, 'w') as f:
                    f.write("seen")
            except Exception:
                pass
            for s_ in strips:
                try:
                    s_.destroy()
                except Exception:
                    pass
            win.destroy()

        def render():
            i = state["i"]
            t, b, target = steps[i]
            counter.config(text=f"{i + 1} of {len(steps)}")
            title_lbl.config(text=t)
            body_lbl.config(text=b)
            back_btn.configure(state='normal' if i > 0 else 'disabled')
            next_btn.configure(text="Finish" if i == len(steps) - 1 else "Next  →")
            state["target"] = target() if callable(target) else None
            if state["target"] is not None:
                scroll_into_view(state["target"])
            win.after(60, place_outline)

        def go_next():
            if state["i"] == len(steps) - 1:
                finish()
            else:
                state["i"] += 1
                render()

        def go_back():
            if state["i"] > 0:
                state["i"] -= 1
                render()

        secondary = dict(bg_color=FIELD, hover_color=FIELD_HOVER, pressed_color=FIELD_PRESSED)
        RoundedButton(nav, text="Skip", command=finish, width=64, height=34, text_color=MUTED,
                      font=(UI_FONT, fs(12)), bg_color=CARD, hover_color=FIELD, pressed_color=CARD).pack(side='left')
        RoundedButton(nav, text="Full guide", command=self.show_help, width=96, height=34,
                      font=(UI_FONT, fs(12)), **secondary).pack(side='left', padx=(6, 0))
        next_btn = RoundedButton(nav, text="Next  →", command=go_next, width=110, height=36,
                                 font=(UI_FONT, fs(12), 'bold'), debounce=0)
        next_btn.pack(side='right')
        back_btn = RoundedButton(nav, text="←  Back", command=go_back, width=90, height=36,
                                 font=(UI_FONT, fs(12), 'bold'), debounce=0, **secondary)
        back_btn.pack(side='right', padx=(0, 8))

        # keep the outline on its target when the main window is resized (only
        # the window's own event: children's events would re-trigger on the strips)
        def on_resize(e):
            if e.widget is self.master and win.winfo_exists():
                win.after(50, place_outline)
        self.master.bind("<Configure>", on_resize, add="+")
        win.protocol("WM_DELETE_WINDOW", finish)
        render()
        pulse()

    def _update_mode(self):
        if getattr(self, '_job', None):
            return  # the button keeps showing the running job until it ends
        mode = self.mode_var.get()
        if mode == "search":
            self.query_btn.config(text="▶  Run Search", command=self.run)
            self._mode_hint.config(text="Search compares your photo with the active index.")
            if not self.query_img_path:
                self.query_img_label.config(text="Click to choose a photo\nor drag one here", fg=MUTED)
            self.photo_zone.config(cursor='hand2')
        else:
            self.query_btn.config(text="▶  Create Index", command=self.run)
            self._mode_hint.config(text="Create builds a new index for the area in step 2 by downloading "
                                        "its Street View. The index above is not used.")
            if not self.query_img_path:
                self.query_img_label.config(text="Not needed when creating an index", fg=SUBTLE)
            self.photo_zone.config(cursor='arrow')
        self._refresh_steps()

    def _area_or_warn(self):
        """(lat, lon, radius_km) from step 2, or None after telling the user what's wrong."""
        try:
            lat, lon, radius = float(self.lat_var.get()), float(self.lon_var.get()), float(self.radius_var.get())
            if -90 <= lat <= 90 and -180 <= lon <= 180 and radius > 0:
                return lat, lon, radius
        except (tk.TclError, ValueError):
            pass
        self._set_status("⚠ Enter a valid latitude (-90 to 90), longitude (-180 to 180) and radius in step 2.")
        return None

    def run(self):
        if self._job_running():
            return
        area = self._area_or_warn()
        if area is None:
            return
        lat, lon, radius = area
        if self.mode_var.get() == "create":
            try:
                res = int(self.res_var.get())
            except (tk.TclError, ValueError):
                self._set_status("⚠ Grid spacing must be a whole number of metres.")
                return
            fov, size, step = self.crop_fov.get(), self.crop_size.get(), self.crop_step.get()
            if self._run_job("Index creation",
                             lambda: self._create_embeddings((lat, lon), radius, res, fov, size, step)):
                self.query_btn.config(text="⏳  Creating index...")
                self.run_hint.config(text="Building the index. You can keep using the map meanwhile.")
                self._set_status("Finding Street View panoramas in the area...")
            return

        if not has_active_index():
            self._set_status("⚠ No index selected. Select one in step 1 or download one from the Community Hub.")
            # A status line alone is easy to miss, so flash the button too.
            self.query_btn.config(text="⚠ Select an index first")
            self.master.after(1800, self._update_mode)
            return
        if not self.query_img_path:
            self.choose_photo()
            if not self.query_img_path:
                return
        self.search_nets = [(lat, lon, radius, "UNKNOWN", self.res_var.get(), self.crop_fov.get(),
                             'full', 'Manual search')]
        self.start_full_search()

    def _center_map_on_area(self):
        """Move the map to the area in step 2, zoomed so the radius fits."""
        area = None
        try:
            area = float(self.lat_var.get()), float(self.lon_var.get()), float(self.radius_var.get())
        except (tk.TclError, ValueError):
            return
        lat, lon, radius = area
        self.map_widget.set_position(lat, lon)
        self.map_widget.set_zoom(max(3, min(17, round(14 - math.log2(max(radius, 0.25))))))

    # make emdbedings for the grid

    def _create_embeddings(self, center, radius, res, crop_fov, crop_size, crop_step):
        q = self.match_queue
        q.put(('status', "Getting grid points..."))
        points = grid_points(center, radius, res)
        # scan for nodes
        q.put(('status', f"Generated {len(points)} grid points. Downloading scan nodes..."))

        panoids = get_panoids(
            points,
            status_callback=lambda idx, total: q.put(('status', f"Scan node fetch {idx}/{total}...")),
            max_workers=MAX_PANOID_WORKERS
        )
        # Rough speed for a first estimate (the live ETA replaces it once running)
        rate = 3.0 if torch.backends.mps.is_available() else 6.0
        minutes = len(panoids) / rate / 60
        estimate = f"about {minutes:.0f} min" if minutes >= 1 else "under a minute"
        q.put(('status', f"Found {len(panoids)} panoramas. Rough estimate: {estimate}."))

        headings_all = sorted(list(set(((h // crop_step) * crop_step) % 360 for h in range(0, 360, crop_step))))
        embeddings_per_panoid = len(headings_all)

        os.makedirs(MEGALOC_PARTS_DIR, exist_ok=True)

        # Load existing embeddings for skip logic (from part files, not CSV — crash-safe)
        existing_files = set()
        try:
            existing_parts = glob.glob(os.path.join(MEGALOC_PARTS_DIR, "megaloc_part_*.npz"))
            for ep in existing_parts:
                data = np.load(ep, allow_pickle=False)
                for p in data['paths']:
                    existing_files.add(os.path.basename(str(p)))
                del data
            if existing_files:
                q.put(('status', f"Loaded {len(existing_files)} existing entries from part files. Starting..."))
        except Exception as e:
            q.put(('status', f"Warning: Could not load existing parts: {e}"))

        crop_queue = queue.Queue(maxsize=CROP_QUEUE_SIZE)
        tracker = ProgressTracker(len(panoids), estimate_storage=True,
                                 embeddings_per_item=embeddings_per_panoid, avg_bytes_per_embedding=2560)
        total_extracted = 0

        # thread to extract features in batch
        def batch_extractor():
            nonlocal total_extracted
            target_batch_size = MEGALOC_BATCH_SIZE
            batch_buffer = []
            megaloc_buffer_descs = []
            megaloc_buffer_paths = []
            megaloc_buffer_lats = []
            megaloc_buffer_lons = []

            def save_megaloc_chunk():
                if not megaloc_buffer_descs:
                    return
                try:
                    timestamp = int(time.time() * 1000)
                    part_filename = os.path.join(MEGALOC_PARTS_DIR, f"megaloc_part_{timestamp}.npz")
                    all_descs = np.vstack(megaloc_buffer_descs)
                    # uncompressed savez: float32 descriptors barely compress, and
                    # np.load reads both formats identically
                    np.savez(
                        part_filename,
                        descriptors=all_descs,
                        paths=np.array(megaloc_buffer_paths, dtype=str),
                        lats=np.array(megaloc_buffer_lats, dtype=np.float32),
                        lons=np.array(megaloc_buffer_lons, dtype=np.float32),
                    )
                    q.put(('status', f"Saved index chunk: {len(megaloc_buffer_paths)} items"))
                    megaloc_buffer_descs.clear()
                    megaloc_buffer_paths.clear()
                    megaloc_buffer_lats.clear()
                    megaloc_buffer_lons.clear()
                except Exception as e:
                    print(f"Error saving EigenPlace chunk: {e}")

            def process_batch(buffer):
                nonlocal total_extracted
                crops = [b[0] for b in buffer]
                meta = [b[1] for b in buffer]
                try:
                    total_extracted += len(meta)

                    crops_pil = [tensor_to_pil(c) for c in crops]
                    cos_descs = batch_encode(crops_pil, batch_size=len(crops))
                    megaloc_buffer_descs.append(cos_descs)
                    megaloc_buffer_paths.extend([m['path'] for m in meta])
                    megaloc_buffer_lats.extend([m['lat'] for m in meta])
                    megaloc_buffer_lons.extend([m['lon'] for m in meta])

                    if len(megaloc_buffer_paths) >= 5000:
                        save_megaloc_chunk()
                except Exception as e:
                    print(f"Batch processing error: {e}")

            while True:
                item = crop_queue.get()
                if item == "DONE":
                    if batch_buffer:
                        process_batch(batch_buffer)
                    save_megaloc_chunk()
                    crop_queue.task_done()
                    break
                batch_buffer.append(item)
                if len(batch_buffer) >= target_batch_size:
                    process_batch(batch_buffer)
                    batch_buffer = []
                crop_queue.task_done()

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()

        extractor_thread = threading.Thread(target=batch_extractor)
        extractor_thread.start()

        base_dirs = get_projection_base_dirs(crop_fov, (crop_size, crop_size))

        # Figure out which panoids actually need downloading BEFORE touching
        # the network at all -- previously this skip check happened inside
        # process_one_panoid, so already-indexed panoids still paid for a
        # ThreadPoolExecutor slot even though they did nothing.
        panoids_needing_download = []
        missing_yaws_by_id = {}
        for panoid in panoids:
            panoid_id = panoid['panoid']
            missing = [y for y in headings_all if f"{panoid_id}_{y}.npz" not in existing_files]
            if missing:
                panoids_needing_download.append(panoid)
                missing_yaws_by_id[panoid_id] = missing

        skipped = len(panoids) - len(panoids_needing_download)

        def process_one_panoid(panoid, tiles_data):
            panoid_id = panoid['panoid']
            missing_yaws = missing_yaws_by_id.get(panoid_id)
            if not missing_yaws:
                return True
            # A panorama with missing tiles would be indexed with black holes.
            if not tiles_data or len(tiles_data) < IMGX * IMGY:
                return False
            try:
                pano_img = stitch_tiles(tiles_data)
            except Exception:
                return False
            maxw = 2048
            if pano_img.size[0] > maxw:
                pano_img = pano_img.resize((maxw, int(pano_img.size[1] * (maxw / pano_img.size[0]))), Image.BILINEAR)

            pano_t = pil_to_tensor(pano_img)

            crops_batch = equirectangular_to_rectilinear_torch(
                pano_t, fov_deg=crop_fov, out_hw=(crop_size, crop_size),
                yaw_deg=missing_yaws, pitch_deg=0, base_dirs=base_dirs
            )
            for i, yaw in enumerate(missing_yaws):
                crop_t = crops_batch[i].unsqueeze(0)
                emb_path = f"{panoid_id}_{yaw}.npz"
                meta = {'path': emb_path, 'lat': panoid['lat'], 'lon': panoid['lon'], 'yaw': yaw}
                crop_queue.put((crop_t, meta))

            pano_img.close()
            del pano_t
            return True

        # Download and project in chunks: each chunk's tiles are fetched in one
        # shared event loop / connection pool, then stitched and projected
        # while the extractor thread keeps encoding the crops already queued.
        # Downloading the whole area first left the encoder idle for the
        # entire download and held every tile of the area in RAM at once.
        tracker.update(skipped)
        done_count, incomplete = skipped, 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=PROJECTION_WORKERS) as executor:
            for start in range(0, len(panoids_needing_download), DOWNLOAD_CHUNK_PANOS):
                chunk = panoids_needing_download[start:start + DOWNLOAD_CHUNK_PANOS]
                def _dl_progress(done, total, start=start):
                    q.put(('status', f"Downloading tiles: panoramas {start + 1}-{start + len(chunk)} "
                                     f"of {len(panoids_needing_download)} ({done}/{total} tiles)..."))
                chunk_tiles = download_tiles_for_panoids(
                    [p['panoid'] for p in chunk],
                    max_workers=MAX_DOWNLOAD_WORKERS,
                    status_callback=_dl_progress,
                )
                results = executor.map(lambda p: process_one_panoid(p, chunk_tiles.get(p['panoid'])), chunk)
                for ok in results:
                    done_count += 1
                    incomplete += not ok
                    tracker.update(done_count)
                    q.put(('progress', done_count, len(panoids)))
                    q.put(('status', f"Stitching & projecting: {tracker.get_status()}"))
                del chunk_tiles
        if incomplete:
            q.put(('status', f"Skipped {incomplete} panoramas with missing tiles."))
            print(f"[INDEX] Skipped {incomplete} panoramas with missing tiles")

        crop_queue.put("DONE")
        extractor_thread.join()

        # PCA is fitted inside build_compact_index() on a 100k subsample —
        # fitting here on ALL raw descriptors loads every part file into RAM
        # at once and OOMs on large indexes, and its result was overwritten
        # by build_compact_index()'s own fit anyway.
        q.put(('status', f"All embeddings saved ({total_extracted} new). Building index (fits PCA)..."))

        build_compact_index()
        q.put(('status', f"Done! Index ready. {total_extracted} new entries added."))

        global _compact_cache
        _compact_cache = None

    def query(self):
        self.choose_photo()

    def start_full_search(self):
        if not self.query_img_path or not self.search_nets:
            self._set_status("⚠ Choose a photo in step 3 first.")
            return
        if self._job_running():
            return

        self.query_btn.config(state='disabled', text="Searching...")
        self.cancel_search.clear()
        self.cancel_btn.config(text="■  Cancel Search", state='normal', command=self.cancel_current_search)
        self.cancel_btn.pack(pady=(8, 0), after=self.query_btn)  # reveal cancel while searching
        self.run_hint.config(text="Searching. This usually takes a few minutes; you can cancel anytime.")
        self._reset_results("Searching…", "Candidates being checked appear on the map.")
        self.stop_animation = False
        self.thumbnail_pool = []

        fov = self.crop_fov.get()
        size = self.crop_size.get()
        step = self.crop_step.get()
        threshold = self.match_threshold.get()
        res = self.res_var.get()

        def run_search_background():
            threads = []
            for net in self.search_nets:
                if len(net) == 8:
                    lat, lon, radius, direction, grid_res, net_fov, dir_precision, rationale = net
                elif len(net) == 4:
                    lat, lon, radius, direction = net
                    grid_res, net_fov = res, fov
                else:
                    continue
                center = (lat, lon)
                t = threading.Thread(target=self._run_search,
                    args=(center, radius, grid_res, threshold, net_fov, size, step, direction))
                threads.append(t)
                t.start()
            for t in threads:
                t.join()

            all_bests = []
            while not self.results_queue.empty():
                try:
                    all_bests.append(self.results_queue.get_nowait())
                except queue.Empty:
                    break

            if self.cancel_search.is_set():
                self.master.after(0, lambda: self._set_status("Search cancelled."))
            elif all_bests:
                global_best = max(all_bests, key=lambda b: b['inliers'])
                query_img_resized = Image.open(self.query_img_path).convert('RGB').resize((size, size), Image.BILINEAR)
                self.master.after(0, lambda: self._handle_match_done(
                    global_best, query_img_resized, fov, size,
                    (global_best.get('lat', self.lat_var.get()), global_best.get('lon', self.lon_var.get())),
                    max(net[2] for net in self.search_nets)))
            else:
                self.master.after(0, lambda: self._set_status("No good matches found."))

            self.stop_animation = True
            self.master.after(0, lambda: self.cancel_btn.pack_forget())
            self.master.after(0, lambda: self.query_btn.config(text="▶  Run Search", command=self.run))
        def search_job():
            try:
                run_search_background()
            finally:  # also reset the buttons if the search raised part-way
                self.stop_animation = True
                self.master.after(0, self.monitor_label.place_forget)
                self.master.after(0, lambda: (self.cancel_btn.pack_forget(),
                                              self.query_btn.config(text="▶  Run Search", command=self.run)))
        self._run_job("Search", search_job)
        self.query_btn.config(text="Searching...")

    def cancel_current_search(self):
        """Signal the running search to stop; the Stage-2 loop checks this flag."""
        self.cancel_search.set()
        self.stop_animation = True
        self._set_status("Cancelling search...")
        self.cancel_btn.config(text="Cancelling...", state='disabled')

    # the main search pipeline thingy


    def _run_search(self, center, radius, res, threshold, crop_fov, crop_size, crop_step, direction="UNKNOWN"):
        q = self.match_queue
        q.put(('status', "Starting search..."))
        early_exit_event = threading.Event()

        try:
            # Load PCA only for encoders that use it (MegaLoc). MixVPR descriptors
            # are already compact and searched directly.
            if ENCODER_USES_PCA:
                pca_path = COMPACT_PCA_PATH or (
                    os.path.join(COMPACT_INDEX_DIR, "megaloc_pca.pkl")
                    if COMPACT_INDEX_DIR else None
                )
                if pca_path and os.path.exists(pca_path):
                    from megaloc_utils import load_pca, _pca_model
                    if _pca_model is None:
                        load_pca(pca_path)

            # Step 1: Load query image
            query_img = Image.open(self.query_img_path).convert("RGB")
            query_img_resize = query_img.resize((crop_size, crop_size), Image.BILINEAR)
            self.current_search_context = (query_img_resize, crop_fov, crop_size, center, radius)

            # Step 2: get the query descriptor (multiscale) via the active encoder
            q.put(('status', f"Extracting query descriptor ({ACTIVE_ENCODER}, multi-scale)..."))
            query_for_megaloc = query_img_resize
            desc_original = encode_query(query_for_megaloc)

            # Slight zoom in (center crop 80%) — matches closer viewpoints
            w, h = query_img_resize.size
            margin_x, margin_y = int(w * 0.1), int(h * 0.1)
            cropped = query_img_resize.crop((margin_x, margin_y, w - margin_x, h - margin_y))
            cropped = cropped.resize((crop_size, crop_size), Image.BILINEAR)
            desc_zoom = encode_query(cropped)
            cropped.close()

            # Average: original + zoom (weighted toward original)
            query_megaloc_desc = 0.65 * desc_original + 0.35 * desc_zoom
            query_megaloc_desc = query_megaloc_desc / (np.linalg.norm(query_megaloc_desc) + 1e-8)

            # Also extract flipped descriptor
            query_img_flipped = query_img_resize.transpose(Image.FLIP_LEFT_RIGHT)
            desc_flipped = encode_query(query_img_flipped)
            desc_flipped_zoom = encode_query(
                query_img_flipped.crop((margin_x, margin_y, w - margin_x, h - margin_y)).resize((crop_size, crop_size), Image.BILINEAR)
            )
            desc_flipped = 0.65 * desc_flipped + 0.35 * desc_flipped_zoom
            desc_flipped = desc_flipped / (np.linalg.norm(desc_flipped) + 1e-8)



            # Step 3: Search compact index (original + flipped)
            q.put(('status', "Searching index (original + flipped)..."))
            K_MEGALOC = 1000
            # Detect an encoder/index dimension mismatch before searching so
            # the status message is specific ("wrong encoder selected") in
            # the UI, not just printed to console -- search_compact_index()
            # itself also guards this and returns [], but its message only
            # reaches the terminal.
            _loaded_descs, _ = load_compact_index()
            if _loaded_descs is not None and query_megaloc_desc.shape[-1] != _loaded_descs.shape[-1]:
                q.put(('status', f"Encoder mismatch: query is {query_megaloc_desc.shape[-1]}-dim "
                                  f"but the loaded index is {_loaded_descs.shape[-1]}-dim. "
                                  f"Re-select the index from 'Select Index...' to resync."))
                self.results_queue.put({'inliers': 0, 'panoid': None, 'heading': None,
                                       'lat': None, 'lon': None, 'matches': None,
                                       'kp1': None, 'kp2': None, 'emb_path': None, 'confidence': 'none'})
                return
            results_original = search_compact_index(query_desc=query_megaloc_desc, center=center, radius_km=radius, top_k=100)
            results_flipped = search_compact_index(query_desc=desc_flipped, center=center, radius_km=radius, top_k=100)
            
            # Merge and deduplicate by panoid, keep higher score
            seen = {}
            for r in results_original + results_flipped:
                key = r['panoid']
                if key not in seen or r['score'] > seen[key]['score']:
                    seen[key] = r
            compact_results = sorted(seen.values(), key=lambda x: x['score'], reverse=True)[:K_MEGALOC]

            if not compact_results:
                q.put(('status', "No candidates found in radius."))
                self.results_queue.put({'inliers': 0, 'panoid': None, 'heading': None,
                                       'lat': None, 'lon': None, 'matches': None,
                                       'kp1': None, 'kp2': None, 'emb_path': None, 'confidence': 'none'})
                return

            if True: # MASt3R is now the default and only Stage 2
                MAST3R_STAGE2_TOP_N = 100
                candidates_to_check = compact_results[:MAST3R_STAGE2_TOP_N]
                q.put(('status', f"Stage 2: Running MASt3R directly on top {len(candidates_to_check)} candidates..."))
                
                all_mast3r_matches = []
                best = {'inliers': 0, 'panoid': None, 'heading': None, 'lat': None, 'lon': None,
                        'matches': None, 'kp1': None, 'kp2': None, 'emb_path': None}
                
                try:
                    mast3r = get_lazy_mast3r()
                    if mast3r is not None:
                        # Prefetch upcoming candidates' tiles in the background
                        # while MASt3R matches the current one, instead of
                        # blocking on a fresh download for every candidate in
                        # sequence. Bounded lookahead (not the whole batch)
                        # because best['inliers'] >= 450 can break out of
                        # this loop early -- downloading all 100 candidates
                        # up front would waste bandwidth on ones the early
                        # exit ends up skipping.
                        PREFETCH_LOOKAHEAD = 4
                        prefetch_pool = concurrent.futures.ThreadPoolExecutor(max_workers=PREFETCH_LOOKAHEAD)
                        prefetch_futures = {}

                        def _fetch_pano_tiles(cand):
                            cpid = cand.get('panoid')
                            if not cpid:
                                return None
                            try:
                                return download_tiles(tiles_info(cpid), max_workers=16)
                            except Exception:
                                return None

                        for j in range(min(PREFETCH_LOOKAHEAD, len(candidates_to_check))):
                            prefetch_futures[j] = prefetch_pool.submit(_fetch_pano_tiles, candidates_to_check[j])

                        for i, match in enumerate(candidates_to_check):
                            if self.cancel_search.is_set():
                                q.put(('status', "Search cancelled."))
                                break
                            q.put(('progress', i, len(candidates_to_check)))
                            q.put(('status', f"MASt3R Match: {i+1}/{len(candidates_to_check)}"))
                            pid = match.get('panoid')
                            hdg = match.get('heading')
                            if not pid or hdg is None: continue

                            # Keep the lookahead window full: queue the next
                            # not-yet-queued candidate now that we're
                            # consuming this one.
                            next_to_queue = i + PREFETCH_LOOKAHEAD
                            if next_to_queue < len(candidates_to_check) and next_to_queue not in prefetch_futures:
                                prefetch_futures[next_to_queue] = prefetch_pool.submit(
                                    _fetch_pano_tiles, candidates_to_check[next_to_queue])

                            pano_img = None
                            try:
                                fut = prefetch_futures.pop(i, None)
                                td = fut.result() if fut is not None else download_tiles(tiles_info(pid), max_workers=16)
                                if td:
                                    pano_img = stitch_tiles(td)
                                    maxw = 2048
                                    if pano_img.size[0] > maxw:
                                        pano_img = pano_img.resize((maxw, int(pano_img.size[1] * (maxw / pano_img.size[0]))), Image.BILINEAR)
                            except: continue
                            
                            if pano_img:
                                pano_t = pil_to_tensor(pano_img)
                                base_dirs_m3 = get_projection_base_dirs(crop_fov, (crop_size, crop_size))
                                crop_t_m3 = equirectangular_to_rectilinear_torch(
                                    pano_t, fov_deg=crop_fov, out_hw=(crop_size, crop_size),
                                    yaw_deg=[hdg], pitch_deg=0, base_dirs=base_dirs_m3)[0].unsqueeze(0)
                                
                                crop_pil = tensor_to_pil(crop_t_m3)
                                m3_matches0, m3_matches1, m3_conf = get_mast3r_matches(query_img_resize, crop_pil, mast3r)
                                m3_score = len(m3_matches0)
                                if m3_score > 50:
                                    print(f"[Stage 2 MASt3R] Candidate got {m3_score} dense matches")
                                
                                match_res = {
                                    'inliers': m3_score, 
                                    'panoid': pid, 'heading': hdg, 'lat': match.get('lat'), 'lon': match.get('lon'),
                                    'kp1': m3_matches0, 'kp2': m3_matches1, 'matches': np.array([[k, k] for k in range(m3_score)]),
                                    'emb_path': match.get('emb_path', '')
                                }
                                
                                if m3_score > 50:
                                    all_mast3r_matches.append(match_res)
                                    q.put(('scan_blip', match.get('lat'), match.get('lon'), m3_score, None))

                                if m3_score > best['inliers']:
                                    best = match_res.copy()
                                    if best['inliers'] >= 150:
                                        q.put(('match_update', best))
                                        q.put(('status', f"New Top Match: {best['inliers']} dense patches!"))
                                
                                del pano_t, crop_t_m3
                                pano_img.close()
                                if torch.backends.mps.is_available(): torch.mps.empty_cache()
                                
                                if best['inliers'] >= 450: # Slightly higher early exit with consensus
                                    q.put(('status', f"Ultra-Strong MASt3R match! {best['inliers']} points — stopping early"))
                                    break

                        # Stop prefetching -- cancel any in-flight lookahead
                        # downloads for candidates we no longer need (early
                        # exit, cancel, or just finished the batch).
                        prefetch_pool.shutdown(wait=False, cancel_futures=True)

                        # ── NEW: Rank Top 5 Geographic Clusters for Sidebar ──
                        if len(all_mast3r_matches) >= 3:
                            CELL_SIZE = 0.00045 # ~50m
                            cells = defaultdict(list)
                            for m in all_mast3r_matches:
                                cell = (round(m['lat'] / CELL_SIZE), round(m['lon'] / CELL_SIZE))
                                cells[cell].append(m)
                            
                            scored_clusters = []
                            for cell_key, cell_matches in cells.items():
                                neighborhood = []
                                for dlat in [-1, 0, 1]:
                                    for dlon in [-1, 0, 1]:
                                        neighbor = (cell_key[0] + dlat, cell_key[1] + dlon)
                                        neighborhood.extend(cells.get(neighbor, []))
                                
                                cell_score = sum(math.sqrt(m['inliers']) for m in neighborhood)
                                cluster_best = max(neighborhood, key=lambda m: m['inliers'])
                                scored_clusters.append({'score': cell_score, 'match': cluster_best})

                            # Sort by consensus score and pick top 10 unique representatives
                            scored_clusters.sort(key=lambda x: x['score'], reverse=True)
                            
                            top_results_for_sidebar = []
                            seen_pids = set()
                            for sc in scored_clusters:
                                r = sc['match']
                                if r['panoid'] not in seen_pids:
                                    top_results_for_sidebar.append(r)
                                    seen_pids.add(r['panoid'])
                                if len(top_results_for_sidebar) >= 10: break
                            
                            if top_results_for_sidebar:
                                best = top_results_for_sidebar[0].copy()
                                best['all_top_clusters'] = top_results_for_sidebar # For sidebar display
                        elif best['inliers'] > 0:
                             best['all_top_clusters'] = [best]
                            
                except Exception as e:
                    print(f"Stage 2 MASt3R error: {e}")
                
                if best['inliers'] > 0:
                    best['inliers'] = 200 + best['inliers'] // 10
                    self.results_queue.put(best)
                else:
                    self.results_queue.put({'inliers': 0, 'panoid': None, 'heading': None,
                                           'lat': None, 'lon': None, 'matches': None,
                                           'kp1': None, 'kp2': None, 'emb_path': None, 'confidence': 'none'})
                return

        except Exception as e:
            import traceback
            traceback.print_exc()
            q.put(('status', f"Search error: {e}"))
            self.results_queue.put({'inliers': 0, 'panoid': None, 'heading': None,
                                   'lat': None, 'lon': None, 'matches': None,
                                   'kp1': None, 'kp2': None, 'emb_path': None, 'confidence': 'none'})

    def show_coverage_map(self):
        from collections import defaultdict
        self._set_status("Loading coverage data...")
        locations = set()

        # Load only metadata for coverage (skip descriptors entirely)
        if has_active_index() and os.path.exists(COMPACT_META_PATH):
            try:
                meta = np.load(COMPACT_META_PATH, allow_pickle=False)
                lats = meta['lats']
                lons = meta['lons']
                for i in range(len(lats)):
                    locations.add((round(float(lats[i]), 6), round(float(lons[i]), 6)))
                del meta
            except Exception as e:
                print(f"[COVERAGE] Error loading metadata: {e}")

        self._clear_coverage_markers()
        self._clear_result_elements()

        if locations:
            latlons = list(locations)
            center_lat = sum(lat for lat, lon in latlons) / len(latlons)
            center_lon = sum(lon for lat, lon in latlons) / len(latlons)
            self.map_widget.set_position(center_lat, center_lon)
            self.map_widget.set_zoom(13)

            MAX_CONNECT_DIST_KM = 0.03
            chains, singles = coverage_polylines(latlons, MAX_CONNECT_DIST_KM)
            line_count = len(chains)
            for chain in chains:
                path = self.map_widget.set_path(chain, color="#3b82f6", width=3)
                self.coverage_markers.append(path)
            for lat, lon in singles:
                marker = self.map_widget.set_marker(lat, lon, text="",
                    marker_color_circle="#3b82f6", marker_color_outside="#1e3a8a")
                self.coverage_markers.append(marker)

            status_msg = f"Coverage: {len(locations)} points, {line_count} segments."
        else:
            if self.search_nets:
                self.map_widget.set_position(self.search_nets[0][0], self.search_nets[0][1])
            else:
                self.map_widget.set_position(self.lat_var.get(), self.lon_var.get())
            self.map_widget.set_zoom(14)
            status_msg = "No index found — only showing search area(s)."

        # Draw yellow search circles -- these mark the area(s) from your
        # MOST RECENT search (self.search_nets), NOT the current index's
        # coverage. They're independent: self.search_nets only updates when
        # you actually run a search, so switching indexes without searching
        # again leaves an old, unrelated circle on the map. Note that in the
        # status text so it doesn't read as "this is where the index is."
        if self.search_nets:
            for net in self.search_nets:
                net_lat, net_lon, net_radius = net[0], net[1], net[2]
                circle_points = generate_circle_points(net_lat, net_lon, net_radius)
                poly = self.map_widget.set_polygon(circle_points, outline_color="yellow", border_width=6, fill_color="")
                self.result_elements.append(poly)
            status_msg += " (Yellow = your last search area, not the index's coverage.)"

        self._set_status(status_msg)
        self.master.update_idletasks()

    

    def _handle_match_done(self, best, query_img_resize, crop_fov, crop_size, center, radius):
        self.stop_animation = True
        self._clear_coverage_markers()

        if best['inliers'] > self.match_threshold.get() and best['panoid'] is not None:
            confidence = best.get('confidence', 'UNKNOWN')
            pano_img = None
            best_crop = None

            cached_tensor = best.pop('_cached_pano_tensor', None)
            if cached_tensor is not None:
                try:
                    crop_tensor = equirectangular_to_rectilinear_torch(
                        cached_tensor, fov_deg=crop_fov, out_hw=(crop_size, crop_size),
                        yaw_deg=best['heading'], pitch_deg=0)
                    best_crop = tensor_to_pil(crop_tensor)
                    del cached_tensor, crop_tensor
                except Exception:
                    best_crop = None
                    del cached_tensor

            if best_crop is None:
                tiles = tiles_info(best['panoid'])
                tiles_data = download_tiles(tiles, max_workers=MAX_DOWNLOAD_WORKERS)
                try:
                    pano_img = stitch_tiles(tiles_data)
                except Exception:
                    self._set_status("Failed to download visualization.")
                    return
                maxw = 2048
                if pano_img.size[0] > maxw:
                    pano_img = pano_img.resize((maxw, int(pano_img.size[1] * (maxw / pano_img.size[0]))), Image.BILINEAR)
                best_crop = equirectangular_to_rectilinear(
                    pano_img, fov_deg=crop_fov, out_hw=(crop_size, crop_size),
                    yaw_deg=best['heading'], pitch_deg=0)

            self._clear_result_elements()
            self.map_widget.set_position(center[0], center[1])
            self.map_widget.set_zoom(16)

            circle_points = generate_circle_points(center[0], center[1], radius)
            circle_poly = self.map_widget.set_polygon(circle_points, outline_color="red", border_width=2, fill_color=None)
            self.result_elements.append(circle_poly)

            if best['lat'] is not None and best['lon'] is not None:
                marker = self.map_widget.set_marker(best['lat'], best['lon'],
                    text=f"📍 {best['lat']:.6f}, {best['lon']:.6f}\n{best['inliers']} inliers | {best['heading']}°")
                self.result_elements.append(marker)

            kp1_scaled = kp2_scaled = None
            if best['kp1'] is not None and best['kp2'] is not None and best['matches'] is not None:
                scale1 = np.array([crop_size / query_img_resize.size[0], crop_size / query_img_resize.size[1]])
                scale2 = np.array([crop_size / best_crop.size[0], crop_size / best_crop.size[1]])
                kp1_scaled = best['kp1'] * scale1
                kp2_scaled = best['kp2'] * scale2
            self._last_result = (query_img_resize.copy(), best_crop.copy(), kp1_scaled, kp2_scaled, best['matches'])
            self._render_result_images()
            if kp1_scaled is not None:
                self.match_lines_toggle.pack(anchor='w', pady=(8, 0), before=self._res_btns)

            try:
                if pano_img: pano_img.close()
            except: pass
            del pano_img, best_crop
            gc.collect()
            if torch.backends.mps.is_available(): torch.mps.empty_cache()

            self._set_status("Best match found. Check that the two pictures show the same place.")
            if best['lat'] is not None:
                self.result_title.config(text=f"{best['lat']:.6f}, {best['lon']:.6f}", fg=TEXT,
                                         font=(UI_FONT, fs(17), 'bold'))
            self.result_detail.config(text=f"{best['inliers']} match points  ·  camera facing {best['heading']}°  ·  "
                                           f"more match points means a more certain match")

            # Update results list
            self.res_tree.delete(*self.res_tree.get_children())
            # If result list is provided (from consensus), use it, otherwise just use best
            results_to_show = best.get('all_top_clusters', [best])[:10]
            for i, r in enumerate(results_to_show, 1):
                coords = f"{r['lat']:.6f}, {r['lon']:.6f}"
                self.res_tree.insert("", "end", values=(i, r['inliers'], coords))
        else:
            self._set_status("No match found.")
            self._reset_results("No confident match found.",
                                "Check that the photo's area is inside the index (Show coverage on the map), "
                                "or try a clearer photo with more buildings in view.")

    # Context menu actions
    def show_res_menu(self, event):
        item = self.res_tree.identify_row(event.y)
        if item:
            self.res_tree.selection_set(item)
            self.res_menu.post(event.x_root, event.y_root)

    def _on_res_select(self, event):
        item = self.res_tree.selection()
        if item:
            val = self.res_tree.item(item[0])['values']
            lat, lon = map(float, val[2].split(", "))
            self.map_widget.set_position(lat, lon)
            self.map_widget.set_zoom(18)

    def copy_res_coords(self):
        coords = self._selected_or_best_coords()
        if coords:
            self.master.clipboard_clear()
            self.master.clipboard_append(coords)
            self._set_status(f"Copied: {coords}")

    def open_res_gmaps(self):
        coords = self._selected_or_best_coords()
        if coords:
            url = f"https://www.google.com/maps/search/?api=1&query={coords.replace(' ', '')}"
            webbrowser.open(url)

    # ═══════════════════════════════════════════════════════════════
    # UI HELPERS
    # ═══════════════════════════════════════════════════════════════

    def _clear_coverage_markers(self):
        for m in self.coverage_markers: m.delete()
        self.coverage_markers = []

    def _clear_result_elements(self):
        for e in self.result_elements: e.delete()
        self.result_elements = []

    def _set_status(self, text):
        self.match_queue.put(('status', text))

    # ── One long job at a time ───────────────────────────────────────
    # Search, index creation, import/export and Community Hub transfers all
    # run in background threads. Two at once compete for the GPU and memory,
    # and some swap the active index under each other, so a second one is
    # refused (with a status message) while the first runs. Main thread only.

    def _job_buttons(self):
        return [self.query_btn, self.coverage_btn, self.index_selector_btn,
                self.export_btn, self.import_btn, self.demo_btn]

    def _job_running(self):
        """True (and tells the user) if a long job is already running."""
        if self._job:
            self._set_status(f"⚠ {self._job} is still running. Please wait for it to finish.")
            return True
        return False

    def _begin_job(self, name):
        if self._job_running():
            return False
        self._job = name
        for btn in self._job_buttons():
            btn.config(state='disabled')
        return True

    def _end_job(self):
        self._job = None
        for btn in self._job_buttons():
            btn.config(state='normal')
        if getattr(self, '_demo_pending', False):
            # on success activate() runs right after this and sets "Demo loaded"
            self.demo_btn.config(text="Try the demo")
        self.progress['value'] = 0
        self._update_mode()

    def _run_job(self, name, target, on_success=None, on_error=None):
        """Run target in a background thread as the one long job. The buttons
        come back when it ends, even if it raises. on_success(result) or
        on_error(message) then runs on the main thread, after the job has ended."""
        if not self._begin_job(name):
            return False
        def wrapper():
            result, failed = None, None
            try:
                result = target()
            except Exception as e:
                failed = str(e) or type(e).__name__
                import traceback
                traceback.print_exc()
                self.master.after(0, lambda m=str(e): self._set_status(f"{name} failed: {m}"))
            finally:
                self.master.after(0, self._end_job)
                if on_success is not None and failed is None:
                    self.master.after(0, lambda: on_success(result))
                if on_error is not None and failed is not None:
                    self.master.after(0, lambda: on_error(failed))
        threading.Thread(target=wrapper, daemon=True).start()
        return True

    def _focus_existing(self, attr):
        """If the window stored in self.<attr> is open, bring it to the front."""
        win = getattr(self, attr, None)
        if win is not None and win.winfo_exists():
            win.deiconify()
            win.lift()
            win.focus_force()
            return True
        return False

    # ── Sidebar building blocks ──────────────────────────────────────

    def _card(self, parent, title, step=None):
        """A bordered section with a title (and a numbered step badge)."""
        card = tk.Frame(parent, bg=CARD, highlightbackground=CARD_BORDER, highlightthickness=1)
        card.pack(fill='x', pady=(0, 12))
        self._cards.append(card)
        head = tk.Frame(card, bg=CARD)
        head.pack(fill='x', padx=16, pady=(14, 10))
        badge = None
        if step is not None:
            badge = tk.Canvas(head, width=24, height=24, bg=CARD, highlightthickness=0)
            badge.pack(side='left', padx=(0, 10))
            badge.step = step
            self._draw_badge(badge, done=False)
        tk.Label(head, text=title, font=(UI_FONT, fs(14), 'bold'), bg=CARD, fg=TEXT).pack(side='left')
        body = tk.Frame(card, bg=CARD)
        body.pack(fill='x', padx=16, pady=(0, 16))
        return card, body, badge

    def _draw_badge(self, badge, done, dim=False):
        badge.delete('all')
        if done:
            badge.create_oval(1, 1, 23, 23, fill=ACCENT, outline=ACCENT)
            badge.create_text(12, 12, text="✓", fill='#ffffff', font=(UI_FONT, fs(12), 'bold'))
        else:
            color = SUBTLE if dim else ACCENT_LIGHT
            badge.create_oval(2, 2, 22, 22, outline=color, width=2)
            badge.create_text(12, 12, text=str(badge.step), fill=color, font=(UI_FONT, fs(11), 'bold'))

    def _field_label(self, parent, text, top=0):
        lbl = tk.Label(parent, text=text, font=(UI_FONT, fs(11), 'bold'), bg=parent.cget('bg'), fg=MUTED, anchor='w')
        lbl.pack(anchor='w', pady=(top, 4))
        return lbl

    def _hint(self, parent, text, top=6):
        lbl = tk.Label(parent, text=text, font=(UI_FONT, fs(11)), bg=parent.cget('bg'), fg=SUBTLE,
                       wraplength=self._inner_w, justify='left', anchor='w')
        lbl.pack(fill='x', pady=(top, 0))
        return lbl

    def _on_sidebar_wheel(self, event):
        """Scroll the sidebar only when the pointer is over it (the map zooms on its own)."""
        try:
            w = self.master.winfo_containing(event.x_root, event.y_root)
        except Exception:
            return
        while w is not None and w is not self.sidebar_container:
            w = w.master
        if w is None:
            return
        step = -event.delta if sys.platform == "darwin" else -int(event.delta / 120)
        self.sidebar_canvas.yview_scroll(step, "units")

    def _refresh_steps(self):
        """Tick off the numbered steps and say what is still missing."""
        if not hasattr(self, '_badge3'):
            return
        searching = self.mode_var.get() == "search"
        has_index = has_active_index()
        try:
            lat, lon, rad = float(self.lat_var.get()), float(self.lon_var.get()), float(self.radius_var.get())
            area_ok = -90 <= lat <= 90 and -180 <= lon <= 180 and rad > 0
        except (tk.TclError, ValueError):
            area_ok, rad = False, 0
        photo_ok = bool(self.query_img_path)
        self._draw_badge(self._badge1, done=has_index or not searching)
        self._draw_badge(self._badge2, done=area_ok)
        self._draw_badge(self._badge3, done=photo_ok and searching, dim=not searching)
        if getattr(self, '_job', None):
            return
        if searching and not has_index:
            hint = "Step 1: select an index or download one from the Community Hub, or press Try the demo."
        elif not area_ok:
            hint = "Step 2: enter a valid latitude, longitude and radius."
        elif searching and not photo_ok:
            hint = "Step 3: choose the photo you want to locate."
        elif searching:
            hint = "Ready. Press Run Search to locate your photo."
        else:
            hint = f"Ready to build an index within {rad:g} km of this point. Large areas can take hours."
        self.run_hint.config(text=hint)

    def choose_photo(self, path=None):
        """Pick the photo to locate (file dialog, or a path dropped on the window)."""
        if self.mode_var.get() != "search":
            self._set_status("Switch the mode to “Search a photo” to choose a photo.")
            return
        if self._job_running():
            return
        if path is None:
            path = filedialog.askopenfilename(
                filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.bmp"), ("All files", "*.*")])
            if not path:
                return
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            self._set_status(f"Couldn't open that image: {e}")
            return
        self.query_img_path = path
        imgtk = ImageTk.PhotoImage(ImageOps.contain(img, (self._inner_w - 4, 146)))
        self.query_img_label.configure(image=imgtk, text="")
        self.query_img_label.image = imgtk
        self.photo_name_label.config(text=f"{os.path.basename(path)}  ·  click the photo to change it")
        self.search_nets = []
        self._clear_result_elements()
        self._refresh_steps()

    def _find_demo_index(self):
        """A local MegaLoc index covering the demo area, if one is installed."""
        for m in scan_indexes():
            cov = m.get("coverage_center", {})
            if cov.get("lat") is None or str(m.get("descriptor_model", "MegaLoc")).lower() == "mixvpr":
                continue
            if haversine((cov["lat"], cov["lon"]), DEMO_CENTER) <= 1.0:
                return m
        return None

    def try_demo(self):
        """Activate the sample index (downloading it from the Community Hub
        the first time) and load the sample photo, ready to Run Search. Every
        step shows in the demo box, and the photo and Run button are brought
        into view when it's ready."""
        if self._job_running():
            return
        self._demo_pending = True
        self.demo_btn.config(text="Loading the demo…")
        self._demo_status("Getting the sample index ready…")

        def activate(manifest):
            global _compact_cache
            self._demo_status("Loading the sample index…")
            self.master.update_idletasks()
            loaded = load_index(manifest["index_id"]) or manifest
            _compact_cache = None
            self.encoder_var.set(ACTIVE_ENCODER)
            self.index_selector_var.set(f"Active: {loaded.get('name', 'demo index')}")
            cov = loaded.get("coverage_center") or {"lat": loaded.get("center_lat"), "lon": loaded.get("center_lon")}
            if cov.get("lat") is not None:
                self.lat_var.set(cov["lat"])
                self.lon_var.set(cov["lon"])
            if loaded.get("radius_km") is not None:
                self.radius_var.set(loaded["radius_km"])
            self.mode_var.set("search")
            self._update_mode()
            self.choose_photo(DEMO_PHOTO)
            if self.query_img_path != DEMO_PHOTO:
                self._demo_failed("the sample photo could not be loaded")
                return
            self._demo_pending = False
            self.demo_btn.config(text="✓  Demo loaded")
            self._demo_status("✓ Sample index active    ✓ Sample photo loaded\n"
                              "Now press Run Search below (it usually takes a minute or two).", ACCENT_LIGHT)
            self._set_status("Demo ready: press Run Search.")
            # bring the loaded photo and the Run button into view, and point them out
            self._scroll_into_view(self._cards[2])
            self._flash_frame(self.photo_zone)
            self._flash_button(self.query_btn)

        local = self._find_demo_index()
        if local:
            # let the "Loading…" feedback paint before the (blocking) index load
            self.master.after(60, lambda: activate(local))
            return
        if ACTIVE_ENCODER != "megaloc":
            set_encoder("megaloc")

        def progress(msg):
            self.master.after(0, lambda m=msg: (self._set_status(m), self._demo_status(m)))

        def download():
            manifest = NetryxHub().download(DEMO_INDEX_REPO, DATA_DIR, progress_callback=progress)
            if not manifest or not manifest.get("index_id"):
                raise RuntimeError("the demo index could not be downloaded")
            return manifest
        if self._run_job("Demo download", download, on_success=activate, on_error=self._demo_failed):
            self.demo_btn.config(text="Downloading the sample index…")
            self._demo_status("Downloading the sample index (about 180 MB, only the first time)…")
            self._set_status("Downloading the demo index (about 180 MB, once)...")

    def _demo_status(self, text, color=MUTED):
        self.demo_status.config(text=text, fg=color)
        if text:
            self.demo_status.pack(anchor='w', padx=16, pady=(0, 12), before=self.demo_btn)
        else:
            self.demo_status.pack_forget()

    def _demo_failed(self, reason):
        self._demo_pending = False
        self.demo_btn.config(text="Try the demo")
        self._demo_status(f"⚠ The demo couldn't start: {reason}. Please try again.", ACCENT_LIGHT)

    def _scroll_into_view(self, widget):
        """Scroll the sidebar so widget is near the top of what's visible."""
        self.master.update_idletasks()
        y = widget.winfo_rooty() - self.left_ctrl.winfo_rooty()
        total = max(self.left_ctrl.winfo_height(), 1)
        self.sidebar_canvas.yview_moveto(max(0.0, (y - 20) / total))

    def _flash_frame(self, frame, times=6):
        """Blink a frame's border in the accent colour to draw the eye to it."""
        def tick(n):
            if not frame.winfo_exists():
                return
            on = n % 2 == 1
            frame.config(highlightbackground=ACCENT_LIGHT if on else CARD_BORDER, highlightthickness=2 if on else 1)
            if n > 0:
                frame.after(280, tick, n - 1)
        tick(times)

    def _flash_button(self, btn, times=6):
        def tick(n):
            if not btn.winfo_exists():
                return
            btn._draw_button(ACCENT_LIGHT if n % 2 == 1 else None)
            if n > 0:
                btn.after(280, tick, n - 1)
        tick(times)

    def _on_drop(self, event):
        paths = self.master.tk.splitlist(event.data)
        if not paths or self._job_running():
            return
        if self.mode_var.get() != "search":
            self.mode_var.set("search")
            self._update_mode()
        self.choose_photo(paths[0])

    def _on_close(self):
        if self._job and not messagebox.askyesno(
                "Quit Netryx?", f"{self._job} is still running and will be stopped. Quit anyway?"):
            return
        s = self.settings
        try:
            s.update(lat=float(self.lat_var.get()), lon=float(self.lon_var.get()),
                     radius_km=float(self.radius_var.get()), grid_spacing_m=int(self.res_var.get()))
        except (tk.TclError, ValueError):
            pass
        s["mode"] = self.mode_var.get()
        s["geometry"] = self.master.geometry()
        if has_active_index():
            s["last_index_id"] = os.path.basename(COMPACT_INDEX_DIR)
        save_settings(s)
        self.master.destroy()

    def _reset_results(self, title, detail=""):
        self._last_result = None
        self.result_title.config(text=title, fg=MUTED, font=(UI_FONT, fs(12)))
        self.result_detail.config(text=detail)
        for lbl in (self.result_query_img, self.canvas):
            lbl.config(image='')
            lbl.image = None
        self.result_caption_q.config(text="")
        self.result_caption_m.config(text="")
        self.match_lines_toggle.pack_forget()
        self.res_tree.delete(*self.res_tree.get_children())

    def _selected_or_best_coords(self):
        """The selected result row, or the best one when nothing is selected."""
        items = self.res_tree.selection() or self.res_tree.get_children()[:1]
        if not items:
            self._set_status("No result yet. Run a search first.")
            return None
        return str(self.res_tree.item(items[0])['values'][2])

    def _render_result_images(self):
        """Your photo next to the matched Street View view, or both joined by
        the matching points when "Show matching points" is ticked."""
        r = getattr(self, '_last_result', None)
        if not r:
            return
        q_img, m_img, kp1, kp2, matches = r
        size = (self._inner_w - 12) // 2
        if self.match_lines_var.get() and kp1 is not None:
            joined = ImageOps.contain(draw_matches(q_img.copy(), m_img.copy(), kp1, kp2, matches), (self._inner_w, size))
            tkimg = ImageTk.PhotoImage(joined)
            self.result_query_img.config(image=tkimg)
            self.result_query_img.image = tkimg
            self.result_query_img.grid(row=0, column=0, columnspan=2)
            self.canvas.grid_remove()
            self.result_caption_q.config(text="Your photo  ·  lines join matching points  ·  Street View")
            self.result_caption_q.grid(row=1, column=0, columnspan=2)
            self.result_caption_m.grid_remove()
        else:
            qi = ImageTk.PhotoImage(ImageOps.fit(q_img, (size, size)))
            mi = ImageTk.PhotoImage(ImageOps.fit(m_img, (size, size)))
            self.result_query_img.config(image=qi)
            self.result_query_img.image = qi
            self.result_query_img.grid(row=0, column=0, columnspan=1)
            self.canvas.config(image=mi)
            self.canvas.image = mi
            self.canvas.grid()
            self.result_caption_q.config(text="Your photo")
            self.result_caption_q.grid(row=1, column=0, columnspan=1)
            self.result_caption_m.config(text="Street View match")
            self.result_caption_m.grid()

    def _set_progress(self, value, maximum):
        self.match_queue.put(('progress', value, maximum))

    def _set_canvas_img(self, imgtk):
        self.canvas.configure(image=imgtk)
        self.canvas.image = imgtk

    def _add_to_thumbnail_pool(self, thumb):
        if thumb is None: return
        with self._thumbnail_pool_lock:
            self.thumbnail_pool.append(thumb)
            if len(self.thumbnail_pool) > 50:
                self.thumbnail_pool.pop(0)

    def _handle_scan_blip(self, lat, lon, inliers, thumb=None):
        if getattr(self, 'stop_animation', False): return
        if inliers > 50 and self.coverage_markers:
            self._clear_coverage_markers()
        if thumb is not None:
            self._add_to_thumbnail_pool(thumb)

        img_to_show = thumb
        if img_to_show is None and self.thumbnail_pool:
            with self._thumbnail_pool_lock:
                if self.thumbnail_pool:
                    img_to_show = random.choice(self.thumbnail_pool)

        if img_to_show:
            try:
                img_filled = ImageOps.fit(img_to_show, (128, 128), method=Image.Resampling.LANCZOS)
                border_color = (0, 255, 157) if inliers > 50 else (255, 215, 0) if inliers > 20 else (255, 68, 68)
                border = Image.new('RGB', (132, 132), border_color)
                border.paste(img_filled, (2, 2))
                photo = ImageTk.PhotoImage(border)
                self.monitor_label.config(image=photo, text=f"Checking candidates\n{inliers} match points")
                self.monitor_label.place(relx=0.985, rely=0.015, anchor="ne")
                self.monitor_label.image = photo
            except Exception: pass

        try:
            current_pos = self.map_widget.get_position()
            if haversine(current_pos, (lat, lon)) > 0.05:
                self.map_widget.set_position(lat, lon)
        except Exception: pass

        color = "#00ff9d" if inliers > 50 else "#ffd700" if inliers > 20 else "#ff4444"
        try:
            marker = self.map_widget.set_marker(lat, lon, marker_color_circle=color, marker_color_outside=color)
            # kill the marker after a bit
            self.master.after(1200, marker.delete)
        except Exception: pass

    # ═══════════════════════════════════════════════════════════════
    # COMMUNITY HUB METHODS
    # ═══════════════════════════════════════════════════════════════

    def show_community_hub(self):
        if self._focus_existing('_hub_win'):
            return
        hub_win = tk.Toplevel(self.master)
        self._hub_win = hub_win
        hub_win.title("Netryx Community Hub")
        hub_win.configure(bg='#0a0a0f')
        hub_win.geometry("700x550")
        hub_win.transient(self.master)

        # Header
        header = tk.Frame(hub_win, bg='#0a0a0f')
        header.pack(fill='x', padx=20, pady=(20, 10))
        tk.Label(header, text="🌐 Community Hub", font=(UI_FONT, 20, 'bold'),
                 bg='#0a0a0f', fg='#ffffff').pack(anchor='w')
        tk.Label(header, text="Download pre-built indexes from the community",
                 font=(UI_FONT, 11), bg='#0a0a0f', fg='#8b5cf6').pack(anchor='w', pady=(4, 0))

        # Search bar
        search_frame = tk.Frame(hub_win, bg='#0a0a0f')
        search_frame.pack(fill='x', padx=20, pady=(10, 5))

        self._hub_search_var = tk.StringVar()
        search_entry = tk.Entry(search_frame, textvariable=self._hub_search_var,
                                font=(UI_FONT, 11), bg='#1a1a2e', fg='#ffffff',
                                insertbackground='white', borderwidth=0, highlightthickness=1,
                                highlightcolor='#8b5cf6', highlightbackground='#2d2d3f')
        search_entry.pack(side='left', fill='x', expand=True, ipady=8, padx=(0, 10))
        search_entry.insert(0, "Search by city name...")
        search_entry.bind('<FocusIn>', lambda e: search_entry.delete(0, 'end') if search_entry.get() == "Search by city name..." else None)

        self.search_btn = RoundedButton(search_frame, text="Search",
                                       command=lambda: self._hub_search(hub_win),
                                       width=100, height=36, corner_radius=10)
        self.search_btn.pack(side='right')

        # Results list
        list_frame = tk.Frame(hub_win, bg='#12121a')
        list_frame.pack(fill='both', expand=True, padx=20, pady=10)

        self._hub_listbox = tk.Listbox(list_frame, font=(UI_FONT, 10),
                                        bg='#12121a', fg='#f3f4f6', selectbackground='#8b5cf6',
                                        selectforeground='white', borderwidth=0,
                                        highlightthickness=0, activestyle='none')
        self._hub_listbox.pack(fill='both', expand=True, side='left')

        scrollbar = tk.Scrollbar(list_frame, command=self._hub_listbox.yview)
        scrollbar.pack(side='right', fill='y')
        self._hub_listbox.config(yscrollcommand=scrollbar.set)

        self._hub_indexes = []

        # Bottom buttons
        bottom = tk.Frame(hub_win, bg='#0a0a0f')
        bottom.pack(fill='x', padx=20, pady=(0, 20))

        self.dl_btn = RoundedButton(bottom, text="⬇ Download",
                                   command=lambda: self._hub_download(hub_win),
                                   width=160, height=40)
        self.dl_btn.pack(side='left')

        self.up_btn = RoundedButton(bottom, text="⬆ Upload Index",
                                   command=lambda: self._hub_upload(hub_win),
                                   width=165, height=40, bg_color='#1e3a5f',
                                   hover_color='#2d4a6f', pressed_color='#0f2a4f')
        self.up_btn.pack(side='right')

        self._hub_status = tk.Label(bottom, text="", font=(UI_FONT, 9),
                                     bg='#0a0a0f', fg='#6b7280')
        self._hub_status.pack(side='left', padx=20)

        # Auto-load on open
        self._hub_refresh(hub_win)

    def _hub_refresh(self, hub_win):
        self._hub_status.config(text="Loading indexes...")
        hub_win.update_idletasks()

        def do_refresh():
            try:
                if not HUB_AVAILABLE:
                    self.master.after(0, lambda: self._hub_status.config(
                        text="Hub unavailable. Install: pip install huggingface_hub"))
                    return
                hub = NetryxHub()
                indexes = hub.list_indexes()
                self._hub_indexes = indexes

                def update_ui():
                    self._hub_listbox.delete(0, 'end')
                    for idx in indexes:
                        size_mb = idx.get('file_size_bytes', 0) / 1024 / 1024
                        author = idx.get('author', '?')
                        badge = "🟣 Official" if idx.get('is_official') else "🟢 Community"
                        enc = idx.get('encoder') or ('mixvpr' if 'MixVPR' in str(idx.get('descriptor_model','')) else 'megaloc')
                        enc_tag = "MixVPR" if enc == 'mixvpr' else "MegaLoc"

                        line = f"📦 {idx['name']:<16} | {enc_tag:<7} | {idx['radius_km']:>3}km | {idx['num_entries']:>6,} pts | {size_mb:>4.0f}MB | {badge} by @{author}"
                        self._hub_listbox.insert('end', line)
                    self._hub_status.config(text=f"Found {len(indexes)} indexes")

                self.master.after(0, update_ui)
            except Exception as e:
                err_msg = str(e)
                self.master.after(0, lambda m=err_msg: self._hub_status.config(text=f"Error: {m}"))

        threading.Thread(target=do_refresh, daemon=True).start()

    def _hub_search(self, hub_win):
        query = self._hub_search_var.get().strip()
        if not query or query == "Search by city name...":
            self._hub_refresh(hub_win)
            return

        self._hub_status.config(text=f"Searching for '{query}'...")
        hub_win.update_idletasks()

        def do_search():
            try:
                hub = NetryxHub()
                results = hub.search(city=query)
                self._hub_indexes = results

                def update_ui():
                    self._hub_listbox.delete(0, 'end')
                    for idx in results:
                        size_mb = idx.get('file_size_bytes', 0) / 1024 / 1024
                        author = idx.get('author', '?')
                        badge = "🟣 Official" if idx.get('is_official') else "🟢 Community"
                        enc = idx.get('encoder') or ('mixvpr' if 'MixVPR' in str(idx.get('descriptor_model','')) else 'megaloc')
                        enc_tag = "MixVPR" if enc == 'mixvpr' else "MegaLoc"

                        line = f"📦 {idx['name']:<16} | {enc_tag:<7} | {idx['radius_km']:>3}km | {idx['num_entries']:>6,} pts | {size_mb:>4.0f}MB | {badge} by @{author}"
                        self._hub_listbox.insert('end', line)
                    self._hub_status.config(text=f"Found {len(results)} indexes for '{query}'")

                    self.master.after(0, update_ui)
            except Exception as e:
                err_msg = str(e)
                self.master.after(0, lambda m=err_msg: self._hub_status.config(text=f"Search error: {m}"))

        threading.Thread(target=do_search, daemon=True).start()

    def _hub_download(self, hub_win):
        if self._job_running():
            return
        sel = self._hub_listbox.curselection()
        if not sel or not self._hub_indexes:
            self._hub_status.config(text="Select an index first")
            return

        idx = self._hub_indexes[sel[0]]
        repo_id = idx.get('repo_id', '')
        name = idx.get('name', 'Unknown')

        # Refuse to re-download an index already present locally. Older
        # bundles predate index_id and won't have one -- for those we can't
        # be sure, so we don't block (better a rare duplicate than blocking
        # a legitimate download).
        remote_id = idx.get('index_id')
        if remote_id:
            local_ids = {m.get('index_id') for m in scan_indexes()}
            if remote_id in local_ids:
                self._hub_status.config(text=f"'{name}' is already downloaded.")
                return

        # A downloaded index must land in its own encoder's directory. Switch the
        # active encoder to match the index type before downloading.
        target_enc = idx.get('encoder') or ('mixvpr' if 'MixVPR' in str(idx.get('descriptor_model', '')) else 'megaloc')
        if target_enc != ACTIVE_ENCODER:
            try:
                set_encoder(target_enc)
                self.encoder_var.set(target_enc)
                self._set_status(f"Switched encoder to {target_enc.upper()} for this index.")
            except Exception as e:
                self._hub_status.config(text=f"Cannot use {target_enc} index: {e}")
                return
        dest_dir = DATA_DIR  # download() appends indexes/{index_id} itself

        self._hub_status.config(text=f"Downloading {name} ({target_enc.upper()})...")
        hub_win.update_idletasks()

        def do_download():
            try:
                hub = NetryxHub()
                manifest = hub.download(
                    repo_id, dest_dir,
                    progress_callback=lambda msg: self.master.after(0, lambda m=msg: self._hub_status.config(text=m))
                )

                global _compact_cache
                loaded_manifest = None
                if manifest and manifest.get("index_id"):
                    loaded_manifest = load_index(manifest["index_id"])
                _compact_cache = None

                def on_done():
                    self._hub_status.config(text=f"✅ Downloaded {name}! Ready to search.")
                    self._set_status(f"Index loaded: {name}")
                    if loaded_manifest:
                        cov = loaded_manifest.get("coverage_center", {})
                        if cov.get("lat") is not None and cov.get("lon") is not None:
                            self.lat_var.set(cov["lat"])
                            self.lon_var.set(cov["lon"])
                            self.map_widget.set_position(cov["lat"], cov["lon"])
                            self.map_widget.set_zoom(13)
                        if loaded_manifest.get("radius_km") is not None:
                            self.radius_var.set(loaded_manifest["radius_km"])

                self.master.after(0, on_done)
            except Exception as e:
                err_msg = str(e)
                self.master.after(0, lambda m=err_msg: self._hub_status.config(text=f"Download error: {m}"))

        self._run_job("Community Hub download", do_download)

    def _hub_upload(self, hub_win):
        if self._job_running():
            return
        if not has_active_index() or not os.path.exists(COMPACT_DESCS_PATH):
            self._hub_status.config(text="No index to upload. Create one first.")
            return

        # Simple dialog for metadata
        if self._focus_existing('_upload_win'):
            return
        upload_win = tk.Toplevel(hub_win)
        self._upload_win = upload_win
        upload_win.title("Upload Index")
        upload_win.configure(bg='#0a0a0f')
        upload_win.geometry("400x350")
        upload_win.transient(hub_win)

        tk.Label(upload_win, text="Upload to Community Hub", font=(UI_FONT, 16, 'bold'),
                 bg='#0a0a0f', fg='#ffffff').pack(pady=(20, 5))
        # Show which encoder this index was built with (uploaded as that type)
        tk.Label(upload_win, text=f"Encoder: {ACTIVE_ENCODER.upper()}  (index type is fixed to how it was built)",
                 font=(UI_FONT, 9), bg='#0a0a0f', fg='#a78bfa').pack(pady=(0, 12))

        fields_frame = tk.Frame(upload_win, bg='#0a0a0f')
        fields_frame.pack(padx=20, fill='x')

        city_var = tk.StringVar(value="")
        radius_var = tk.StringVar(value=str(self.radius_var.get()))
        lat_var = tk.StringVar(value=str(self.lat_var.get()))
        lon_var = tk.StringVar(value=str(self.lon_var.get()))
        tags_var = tk.StringVar(value="")

        for label, var in [("City name:", city_var), ("Radius (km):", radius_var),
                           ("Center Lat:", lat_var), ("Center Lon:", lon_var),
                           ("Tags (comma-sep):", tags_var)]:
            row = tk.Frame(fields_frame, bg='#0a0a0f')
            row.pack(fill='x', pady=4)
            tk.Label(row, text=label, font=(UI_FONT, 10), bg='#0a0a0f', fg='#9ca3af',
                     width=16, anchor='w').pack(side='left')
            tk.Entry(row, textvariable=var, font=(UI_FONT, 10), bg='#1a1a2e', fg='#ffffff',
                     insertbackground='white', borderwidth=0, highlightthickness=1,
                     highlightcolor='#8b5cf6', highlightbackground='#2d2d3f').pack(side='left', fill='x', expand=True, ipady=4)

        status_lbl = tk.Label(upload_win, text="", font=(UI_FONT, 9), bg='#0a0a0f', fg='#6b7280')
        status_lbl.pack(pady=(10, 5))

        def do_upload():
            city = city_var.get().strip()
            if not city:
                status_lbl.config(text="City name required")
                return

            status_lbl.config(text="Uploading...")
            upload_win.update_idletasks()

            def upload_thread():
                try:
                    token = self.hf_token_var.get().strip()
                    if not token:
                        self.master.after(0, lambda: tk.messagebox.showerror("Hugging Face Help", 
                            "Please connect Hugging Face to upload.\n\n"
                            "1. Click 'Get Hugging Face Token'\n"
                            "2. Generate a WRITE token\n"
                            "3. Paste it in the token field\n"
                            "4. Try uploading again"))
                        self.master.after(0, lambda: status_lbl.config(text="Token missing"))
                        return

                    os.environ["HF_TOKEN"] = token
                    hub = NetryxHub(token=token)
                    
                    tags = [t.strip() for t in tags_var.get().split(',') if t.strip()]
                    url = hub.upload(
                        index_dir=COMPACT_INDEX_DIR,
                        city=city,
                        radius_km=float(radius_var.get()),
                        center_lat=float(lat_var.get()),
                        center_lon=float(lon_var.get()),
                        tags=tags,
                        encoder=ACTIVE_ENCODER,
                    )
                    self.master.after(0, lambda: status_lbl.config(text=f"✅ Uploaded! {url}"))
                    self.master.after(0, lambda: self._hub_status.config(text=f"Uploaded {city}!"))
                except Exception as e:
                    err_msg = str(e)
                    self.master.after(0, lambda m=err_msg: status_lbl.config(text=f"Error: {m}"))

            self._run_job("Community Hub upload", upload_thread)

        self.final_up_btn = RoundedButton(upload_win, text="⬆  Start Upload",
                                         command=do_upload,
                                         width=180, height=42)
        self.final_up_btn.pack(pady=(10, 20))

    def export_index(self):
        if self._job_running():
            return
        if not has_active_index() or not os.path.exists(COMPACT_DESCS_PATH):
            self._set_status("No index to export. Create one first.")
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".netryx",
            filetypes=[("Netryx Index", "*.netryx")],
            title="Export Index As",
            initialfile=f"netryx_index_{int(self.radius_var.get())}km.netryx"
        )
        if not save_path:
            return

        self._set_status("Exporting index...")

        def do_export():
            try:
                from netryx_hub import create_bundle
                path, manifest = create_bundle(
                    index_dir=COMPACT_INDEX_DIR,
                    output_path=save_path,
                    name=f"Netryx Index {self.radius_var.get()}km",
                    description="Exported from Netryx Drishti",
                    center_lat=self.lat_var.get(),
                    center_lon=self.lon_var.get(),
                    radius_km=self.radius_var.get(),
                )
                size_mb = os.path.getsize(save_path) / 1024 / 1024
                self.master.after(0, lambda: self._set_status(
                    f"✅ Exported: {save_path} ({size_mb:.0f} MB)"))
            except Exception as e:
                err_msg = str(e)
                self.master.after(0, lambda m=err_msg: self._set_status(f"Export error: {m}"))

        self._run_job("Index export", do_export)

    def import_index(self):
        if self._job_running():
            return
        file_path = filedialog.askopenfilename(
            filetypes=[("Netryx Index", "*.netryx"), ("All files", "*.*")],
            title="Import Netryx Index"
        )
        if not file_path:
            return

        self._set_status("Importing index...")

        def do_import():
            try:
                import zipfile
                # Peek at the bundle's manifest before extracting, so we can
                # refuse a re-import without ever touching disk.
                with zipfile.ZipFile(file_path, 'r') as zf:
                    peek_manifest = json.loads(zf.read("manifest.json"))
                bundle_id = peek_manifest.get("index_id")
                if bundle_id:
                    local_ids = {m.get('index_id') for m in scan_indexes()}
                    if bundle_id in local_ids:
                        self.master.after(0, lambda: self._set_status(
                            f"'{peek_manifest.get('name', bundle_id)}' is already imported."))
                        return

                from netryx_hub import extract_bundle
                # extract_bundle wants the BASE data dir -- it appends
                # indexes/{uuid} itself and writes manifest.json there.
                manifest = extract_bundle(file_path, DATA_DIR)

                global _compact_cache
                imported_id = manifest["index_id"]
                load_index(imported_id)  # repoints all COMPACT_* globals
                _compact_cache = None

                # Load PCA if present (load_index already set COMPACT_PCA_PATH)
                if ENCODER_USES_PCA and COMPACT_PCA_PATH and os.path.exists(COMPACT_PCA_PATH):
                    try:
                        from megaloc_utils import load_pca
                        load_pca(COMPACT_PCA_PATH)
                    except Exception:
                        pass

                def on_done():
                    self._set_status(f"✅ Imported: {manifest.get('name', 'Unknown')} — Ready to search!")
                    if manifest:
                        self.lat_var.set(manifest.get('center_lat', self.lat_var.get()))
                        self.lon_var.set(manifest.get('center_lon', self.lon_var.get()))
                        self.radius_var.set(manifest.get('radius_km', self.radius_var.get()))
                        self.map_widget.set_position(
                            manifest.get('center_lat', self.lat_var.get()),
                            manifest.get('center_lon', self.lon_var.get()))
                        self.map_widget.set_zoom(13)

                self.master.after(0, on_done)
            except Exception as e:
                err_msg = str(e)
                self.master.after(0, lambda m=err_msg: self._set_status(f"Import error: {m}"))

        self._run_job("Index import", do_import)

    def show_help(self):
        if self._focus_existing('_help_win'):
            return
        help_win = tk.Toplevel(self.master)
        self._help_win = help_win
        help_win.title("Netryx Drishti - Technical User Guide")
        help_win.geometry("850x750")
        help_win.configure(bg='#0a0a0f')
        help_win.transient(self.master)

        main_frame = tk.Frame(help_win, bg='#0a0a0f')
        main_frame.pack(fill='both', expand=True, padx=30, pady=30)

        title_lbl = tk.Label(main_frame, text="Netryx Astra v2 Engine Reference", 
                            font=(UI_FONT, 24, 'bold'), bg='#0a0a0f', fg='#ffffff')
        title_lbl.pack(anchor='w', pady=(0, 25))

        content_canvas = tk.Canvas(main_frame, bg='#0a0a0f', highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=content_canvas.yview)
        scrollable_frame = tk.Frame(content_canvas, bg='#0a0a0f')

        scrollable_frame.bind(
            "<Configure>",
            lambda e: content_canvas.configure(scrollregion=content_canvas.bbox("all"))
        )

        content_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=770)
        content_canvas.configure(yscrollcommand=scrollbar.set)

        content_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        sections = [
            ("Core Logic and Technical Overview", 
             "Netryx is built on a global-to-local visual search pipeline. It is designed to find a specific location in an urban environment by comparing your query image against a vast database of pre-indexed street-level views. The system does not rely on GPS metadata from your photo; instead, it looks at the actual architecture, textures, and spatial relationships in the scene."),

            ("The Astra Search Pipeline", 
             "When you run a search, Netryx goes through three distinct stages to ensure millimetric accuracy:\n\n"
             "1. Global Retrieval (MegaLoc): The system extracts a high-level visual signature from your photo and scans the entire city index. It identifies the top 100 most similar locations based on broad visual features.\n\n"
             "2. Dense Geometric Matching (MASt3R): For the top candidates found in Stage 1, we pull the original panoramas and perform an extremely detailed point-to-point comparison. This stage finds thousands of tiny matching 'patches' between the images to confirm they are the same spot.\n\n"
             "3. Spatial Consensus: To prevent errors caused by repetitive architecture (like identical-looking chain stores), candidates are clustered into geographic groups. A location is only confirmed if multiple nearby images also match well, ensuring that isolated false positives are ignored."),

            ("Working with City Indexes", 
             "A city index is a collection of mathematical descriptors for every street-level view in a given radius. You can manage these in several ways:\n\n"
             "• **Building a New Index**: Switch the main mode to 'Create Index', set your center point and radius on the map, and click Run. The system will download panoramas and build the database locally.\n\n"
             "• **Community Hub**: Browse and download pre-built city indexes directly into your local database. This saves you hours of processing time.\n\n"
             "• **Contributing**: Have a GPU and the time to index your neighborhood? Use the 'Upload' button in the Community Hub to share your index with the world.\n\n"
             "### Hugging Face Tokens\n"
             "To upload and contribute city indexes, you need a Hugging Face Access Token:\n"
             "1. Create a free account at [huggingface.co](https://huggingface.co).\n"
             "2. Go to **Settings > Access Tokens**.\n"
             "3. Create a new token with **'Write'** permissions.\n"
             "4. Paste it into the token field in the sidebar.\n\n"
             "### Coordinate Search\n"
             "You can manually enter coordinates or paste them into the Lat/Lon fields. Use the 'Power Actions' (Copy/Open Maps) on search results to extract coordinates for external use.\n"),

            ("Search Parameters and Calibration", 
             "For the best results, you should fine-tune your search based on the city's density:\n\n"
             "• Grid Resolution: This is the gap between scan points, in meters. For broad coverage and general mapping, a 300-meter resolution is highly recommended. For extreme precision in dense urban areas, you can use 25-50 meters (note: values below ~43m are floored, since Street View's own search radius per point already covers that tightly).\n\n"
             "• Match Threshold: This controls how picky the Stage 1 retrieval is. A higher threshold (0.80+) is faster but might miss subtle matches. Lowering it (0.60) can help in difficult conditions like light or weather changes."),

            ("Practical Tips for Researchers", 
             "• Orientation Matters: If the AI-assigned heading seems off, try rotating your query image or adjusting the step size during indexing.\n\n"
             "• Local Storage: Indexes are stored on your Expansion drive whenever possible to save space on your primary disk. Large city indexes can exceed several gigabytes.")
        ]

        for sec_title, sec_text in sections:
            s_frame = tk.Frame(scrollable_frame, bg='#0a0a0f', pady=20)
            s_frame.pack(fill='x')
            
            tk.Label(s_frame, text=sec_title, font=(UI_FONT, 15, 'bold'), 
                     bg='#0a0a0f', fg='#8b5cf6').pack(anchor='w')
            
            tk.Label(s_frame, text=sec_text, font=(UI_FONT, 12), 
                     bg='#0a0a0f', fg='#f3f4f6', justify='left', wraplength=730).pack(anchor='w', pady=(8, 0))
            
            tk.Frame(s_frame, bg='#1a1a2e', height=1).pack(fill='x', pady=(20, 0))

        close_btn = tk.Button(help_win, text="Return to Console", font=(UI_FONT, 11, 'bold'),
                              bg='#1e3a5f', fg='white', borderwidth=0, padx=40, pady=12,
                              command=help_win.destroy)
        close_btn.pack(pady=25)

    def poll_match_queue(self):
        try:
            for _ in range(20):
                msg = self.match_queue.get_nowait()
                if msg[0] == 'status':
                    if hasattr(self, 'status_label') and self.status_label.winfo_exists():
                        self.status_label.config(text=msg[1],
                                                 fg=ACCENT_LIGHT if str(msg[1]).startswith("⚠") else MUTED)
                        self.master.update_idletasks()
                elif msg[0] == 'progress':
                    if hasattr(self, 'progress') and self.progress.winfo_exists():
                        self.progress['maximum'] = msg[2]
                        self.progress['value'] = msg[1]
                elif msg[0] == 'match_update':
                    match_res = msg[1]
                    if hasattr(self, 'current_search_context'):
                        self._clear_coverage_markers()
                        self._handle_match_done(match_res, *self.current_search_context)
                elif msg[0] == 'scan_blip':
                    self._handle_scan_blip(msg[1], msg[2], msg[3], msg[4] if len(msg) > 4 else None)
        except queue.Empty:
            pass
        self.master.after(100, self.poll_match_queue)
        # check queue again soon


# start the app here god please fucking work i wanna kms

if __name__ == "__main__":
    # Ensure data dirs exist. COMPACT_INDEX_DIR is intentionally excluded --
    # it's None until an index is built or loaded via load_index().
    for d in [DATA_DIR, MEGALOC_PARTS_DIR, INDEXES_DIR]:
        os.makedirs(d, exist_ok=True)

    available = scan_indexes()
    print(f"[INDEX] Found {len(available)} index(es): "
          f"{[m.get('name', m.get('index_id')) for m in available]}")
    if available:
        # Reopen the index used last time if it still exists, otherwise the
        # most recently built one, so the app opens usable.
        last_id = load_settings().get("last_index_id")
        ids = [m["index_id"] for m in available]
        load_index(last_id if last_id in ids else ids[-1])

    # tkinterdnd2 (optional) lets a photo be dragged onto the window
    try:
        from tkinterdnd2 import TkinterDnD
        root = TkinterDnD.Tk()
    except Exception:
        root = tk.Tk()
    set_app_icon(root)  # straight away, so the Dock never shows Python's rocket
    app = StreetViewMatcherGUI(root)
    root.mainloop()
