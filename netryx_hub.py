"""
netryx_hub.py — Community Index Sharing via Hugging Face Hub

Share and download pre-computed MegaLoc indexes so users don't have to
re-index cities that someone else already indexed.

Requirements:
    pip install huggingface_hub

Setup (one time):
    1. Create a free Hugging Face account at https://huggingface.co
    2. Create an access token at https://huggingface.co/settings/tokens
    3. Run: huggingface-cli login
    
    OR set the environment variable:
    export HF_TOKEN=

Usage:
    from netryx_hub import NetryxHub
    
    hub = NetryxHub()
    
    # Browse available indexes
    indexes = hub.list_indexes()
    
    # Download an index
    hub.download("moscow-10km", output_dir="/path/to/netryx/index")
    
    # Upload your index
    hub.upload(
        index_dir="/path/to/netryx/index",
        city="moscow",
        radius_km=10,
        center_lat=55.7539,
        center_lon=37.6208,
    )
"""


import os
import json
import re
import hashlib
import io
import zipfile
import tempfile
import shutil
import time, uuid
import numpy as np
from pathlib import Path

try:
    from huggingface_hub import (
        HfApi, hf_hub_download, upload_file, create_repo,
        list_repo_files, repo_exists, list_models
    )
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("[HUB] WARNING: huggingface_hub not installed. Run: pip install huggingface_hub")


# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────

HF_ORG = None  # No default org — use tags for discovery!
BUNDLE_FORMAT_VERSION = "3.0"
BUNDLE_EXTENSION = ".netryx"


# ─────────────────────────────────────────────────────────────────
# Bundle creation and extraction
# ─────────────────────────────────────────────────────────────────

def _haversine_np(lat1, lon1, lats, lons):
    """Vectorized haversine distance in km from a single point to arrays of points."""
    R = 6371.0
    lat1_r = np.radians(lat1)
    lon1_r = np.radians(lon1)
    lat2_r = np.radians(lats)
    lon2_r = np.radians(lons)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def create_bundle(index_dir, output_path, name, description, center_lat, center_lon,
                  radius_km, tags=None, heading_step=90, crop_fov=90, crop_size=256,
                  creator="anonymous"):
    """Package a local Netryx index into a .netryx bundle.
    
    Only includes entries within the specified radius of the center point.
    This lets you upload just "Moscow 1km" even if your index also contains
    Paris, Tokyo, etc.
    
    Args:
        index_dir: Directory containing megaloc_descriptors.npy, metadata.npz, etc.
        output_path: Where to save the .netryx file
        name: Human-readable name (e.g., "Moscow Central")
        description: Coverage description
        center_lat, center_lon: Center of indexed area
        radius_km: Radius covered
        tags: List of search tags
        
    Returns:
        Path to created bundle, manifest dict
    """
    # Try both possible descriptor filenames
    descs_path = os.path.join(index_dir, "megaloc_descriptors.npy")
    if not os.path.exists(descs_path):
        descs_path = os.path.join(index_dir, "cosplace_descriptors.npy")
    meta_path = os.path.join(index_dir, "metadata.npz")
    pca_path = os.path.join(index_dir, "megaloc_pca.pkl")
    info_path = os.path.join(index_dir, "index_info.txt")

    if not os.path.exists(descs_path):
        raise FileNotFoundError(f"Descriptors not found in {index_dir}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata not found at {meta_path}")
    if os.path.exists(pca_path):
        _validate_index_arrays(pca_path)

    # Read full index
    descs = np.load(descs_path, mmap_mode='r')
    meta = np.load(meta_path, allow_pickle=False)
    lats = meta['lats']
    lons = meta['lons']

    # ── Geographic filter: only keep entries within radius ──
    distances = _haversine_np(center_lat, center_lon, lats, lons)
    # Add 10% margin so edges aren't clipped
    mask = distances <= (radius_km * 1.1)
    valid_idx = np.where(mask)[0]

    if len(valid_idx) == 0:
        raise ValueError(
            f"No entries found within {radius_km}km of ({center_lat}, {center_lon}). "
            f"Your index may cover a different area."
        )

    print(f"[HUB] Geographic filter: {len(valid_idx)}/{len(descs)} entries within {radius_km}km")

    # Extract filtered metadata subset. Descriptors are NOT materialized here —
    # np.array(descs[valid_idx]) would copy the whole multi-GB subset into RAM
    # and get the process OOM-killed on large areas; they are streamed to disk
    # in chunks below instead.
    filtered_lats = lats[valid_idx]
    filtered_lons = lons[valid_idx]
    filtered_headings = meta['headings'][valid_idx]
    filtered_panoids = meta['panoids'][valid_idx]
    filtered_paths = meta['paths'][valid_idx]

    num_entries = len(valid_idx)
    desc_dim = descs.shape[1]
    panoid_set = set(str(p) for p in filtered_panoids)
    num_panoids = len(panoid_set)

    del meta

    # Build manifest
    manifest = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "index_id": str(uuid.uuid4()),
        "name": name,
        "description": description,
        "creator": creator,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "center_lat": float(center_lat),
        "center_lon": float(center_lon),
        "radius_km": float(radius_km),
        "num_entries": int(num_entries),
        "num_panoids": int(num_panoids),
        "descriptor_dim": int(desc_dim),
        "raw_descriptor_dim": 8448,
        "descriptor_model": "MegaLoc" if os.path.exists(
            os.path.join(index_dir, "megaloc_descriptors.npy")
        ) else "MixVPR",
        "pca_components": int(desc_dim) if os.path.exists(
            os.path.join(index_dir, "megaloc_pca.pkl")
        ) else None,
        "heading_step_deg": int(heading_step),
        "crop_fov_deg": int(crop_fov),
        "crop_size_px": int(crop_size),
        "tags": tags or [],
    }

    # Create ZIP bundle with FILTERED data
    print(f"[HUB] Creating bundle: {name} ({num_entries} entries, {num_panoids} panoids)")
    
    # Save filtered arrays to temp files, then zip them. Temp lives next to the
    # output so multi-GB staging doesn't fill the system disk.
    tmp_dir = tempfile.mkdtemp(prefix="netryx_bundle_",
                               dir=os.path.dirname(os.path.abspath(output_path)) or None)
    try:
        tmp_descs = os.path.join(tmp_dir, "descriptors.npy")
        tmp_meta = os.path.join(tmp_dir, "metadata.npz")

        # Stream descriptors mmap -> disk in chunks (bounded RAM)
        out_descs = np.lib.format.open_memmap(
            tmp_descs, mode='w+', dtype=np.float32, shape=(num_entries, desc_dim))
        CHUNK = 100_000
        for s in range(0, num_entries, CHUNK):
            idx_chunk = valid_idx[s:s + CHUNK]
            out_descs[s:s + len(idx_chunk)] = descs[idx_chunk]
        out_descs.flush()
        del out_descs, descs

        np.savez_compressed(tmp_meta,
            lats=filtered_lats, lons=filtered_lons,
            headings=filtered_headings, panoids=filtered_panoids,
            paths=filtered_paths)

        del filtered_lats, filtered_lons
        del filtered_headings, filtered_panoids, filtered_paths
        
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
            zf.write(tmp_descs, "descriptors.npy")
            zf.write(tmp_meta, "metadata.npz")
            if os.path.exists(pca_path):
                zf.write(pca_path, "pca_model.pkl")
            if os.path.exists(info_path):
                zf.write(info_path, "index_info.txt")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Compute SHA256 (streamed — reading a multi-GB bundle whole would OOM)
    _h = hashlib.sha256()
    with open(output_path, 'rb') as _f:
        for _blk in iter(lambda: _f.read(1 << 24), b''):
            _h.update(_blk)
    sha = _h.hexdigest()
    manifest["sha256"] = sha
    manifest["file_size_bytes"] = os.path.getsize(output_path)

    size_mb = manifest["file_size_bytes"] / 1024 / 1024
    print(f"[HUB] Bundle created: {output_path} ({size_mb:.0f} MB)")
    print(f"[HUB]   {num_entries} entries, {num_panoids} panoids, {desc_dim}-dim")

    return output_path, manifest


def _validate_index_arrays(source):
    try:
        with np.load(source, allow_pickle=False) as arrays:
            for name in arrays.files:
                if arrays[name].dtype.hasobject:
                    raise ValueError("Object arrays are not supported.")
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Unsafe or unsupported index arrays. Rebuild and export the index.") from exc


def _index_destination(base_dir, manifest):
    if not isinstance(manifest, dict) or manifest.get("format_version") != BUNDLE_FORMAT_VERSION:
        raise ValueError("Unsafe or unsupported index format. Rebuild and export the index with this version of Netryx.")
    index_id = manifest.get("index_id", str(uuid.uuid4()))
    reserved = {"CON", "PRN", "AUX", "NUL"} | {f"{prefix}{n}" for prefix in ("COM", "LPT") for n in range(1, 10)}
    if (not isinstance(index_id, str) or
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", index_id) is None or
            index_id.upper() in reserved):
        raise ValueError("Invalid index ID: use one bounded name containing letters, digits, underscores or hyphens.")
    indexes_dir = Path(base_dir).resolve() / "indexes"
    destination = indexes_dir / index_id
    if destination.resolve().parent != indexes_dir:
        raise ValueError("Index destination escapes the index directory.")
    for name in ("manifest.json", "metadata.npz", "megaloc_pca.pkl", "index_info.txt",
                 "megaloc_descriptors.npy", "mixvpr_descriptors.npy"):
        if (destination / name).resolve().parent != destination:
            raise ValueError("Index file destination escapes the selected index directory.")
    manifest["index_id"] = index_id
    return str(destination)


def extract_bundle(bundle_path, index_dir):
    """Extract a .netryx bundle into Netryx's index directory.
    
    Args:
        bundle_path: Path to .netryx file
        index_dir: Target directory (e.g., COMPACT_INDEX_DIR)
        
    Returns:
        manifest dict
    """

    with zipfile.ZipFile(bundle_path, 'r') as zf:
        # Read manifest first
        manifest = json.loads(zf.read("manifest.json"))
        index_dir = _index_destination(index_dir, manifest)
        for name in ("metadata.npz", "pca_model.pkl"):
            if name in zf.namelist():
                _validate_index_arrays(io.BytesIO(zf.read(name)))
        os.makedirs(index_dir, exist_ok=True)
        print(f"[HUB] Extracting: {manifest['name']}")
        print(f"[HUB]   Coverage: ({manifest['center_lat']:.4f}, {manifest['center_lon']:.4f}) "
              f"r={manifest['radius_km']}km")
        print(f"[HUB]   Entries: {manifest['num_entries']}, dim: {manifest['descriptor_dim']}")

        # Extract with correct filenames for Netryx
        # Choose descriptor filename based on encoder
        if manifest.get("descriptor_model", "").lower() == "mixvpr":
            descriptor_name = "mixvpr_descriptors.npy"
        else:
            descriptor_name = "megaloc_descriptors.npy"

        file_mapping = {
            "descriptors.npy": descriptor_name,
            "metadata.npz": "metadata.npz",
            "pca_model.pkl": "megaloc_pca.pkl",
            "index_info.txt": "index_info.txt",
            "manifest.json": "manifest.json",
        }

        for src_name, dst_name in file_mapping.items():
            if src_name in zf.namelist():
                data = zf.read(src_name)
                dst_path = os.path.join(index_dir, dst_name)
                with open(dst_path, 'wb') as f:
                    f.write(data)
                print(f"[HUB]   Extracted {src_name} -> {dst_name}")

    print(f"[HUB] Bundle loaded. Ready to search.")
    return manifest


# ─────────────────────────────────────────────────────────────────
# Hugging Face Hub integration
# ─────────────────────────────────────────────────────────────────

def _make_repo_id(city, radius_km, org=None):
    """Generate a HF repo ID from city name and radius."""
    slug = city.lower().strip().replace(" ", "-").replace(",", "")
    name = f"netryx-{slug}-{int(radius_km)}km"
    return f"{org}/{name}" if org else name


def _make_readme(manifest):
    """Generate a README.md for the HF dataset repo."""
    tags_yaml = "\n".join([f"- {t}" for t in manifest.get("tags", [])])
    
    return f"""---
tags:
{tags_yaml}
license: cc-by-4.0
---

# {manifest['name']}

Pre-computed MegaLoc index for **Netryx Drishti** geolocation.

## Coverage
- **Center:** {manifest['center_lat']:.6f}, {manifest['center_lon']:.6f}
- **Radius:** {manifest['radius_km']} km
- **Panoramas:** {manifest['num_panoids']:,}
- **Index entries:** {manifest['num_entries']:,}
- **Descriptor model:** {manifest['descriptor_model']}
- **Descriptor dim:** {manifest['descriptor_dim']} (PCA from {manifest['raw_descriptor_dim']})

## Usage

```python
from netryx_hub import NetryxHub

hub = NetryxHub()
hub.download("{manifest['name'].lower().replace(' ', '-')}", output_dir="./netryx_data")
# The index is registered under netryx_data/indexes/{{uuid}}/ -- it'll show
# up in Netryx's index list automatically. Select it there to search.
```

Or download manually and use **Import Index** in the Netryx GUI.

## Details
- **Heading step:** {manifest['heading_step_deg']}°
- **Crop FOV:** {manifest['crop_fov_deg']}°
- **Crop size:** {manifest['crop_size_px']}px
- **Created:** {manifest['created_at']}
- **Created by:** {manifest['creator']}
- **Tags:** {', '.join(manifest.get('tags', []))}
"""


class NetryxHub:
    """Client for sharing Netryx indexes via Hugging Face Hub.
    
    Setup:
        pip install huggingface_hub
        huggingface-cli login
    """

    def __init__(self, token=None, org=None):
        if not HF_AVAILABLE:
            raise ImportError("Install huggingface_hub: pip install huggingface_hub")
        self.token = token
        self.org = org
        self.api = HfApi(token=token)

    def list_indexes(self):
        """List all available Netryx indexes on the hub.
        
        Returns:
            List of dicts with index metadata
        """
        print(f"[HUB] Searching globally for Netryx indexes...")
        try:
            # 1. Global search (may have indexing delay)
            datasets_global = list(self.api.list_datasets(search="netryx"))
            
            # 2. Direct author search (bypasses indexing lag for YOUR own uploads)
            datasets_mine = []
            try:
                user_info = self.api.whoami()
                datasets_mine = list(self.api.list_datasets(author=user_info["name"]))
            except:
                pass
            
            # Merge and deduplicate
            seen_ids = set()
            datasets = []
            for ds in datasets_global + datasets_mine:
                if ds.id not in seen_ids:
                    datasets.append(ds)
                    seen_ids.add(ds.id)
                    
        except Exception as e:
            print(f"[HUB] Error listing datasets: {e}")
            return []

        indexes = []
        for ds in datasets:
            try:
                # Filter: strictly require manifest.json (ground truth)
                files = list_repo_files(repo_id=ds.id, repo_type="dataset")
                if "manifest.json" not in files:
                    continue
                
                # Download just the manifest to get metadata
                manifest_path = hf_hub_download(
                    repo_id=ds.id,
                    filename="manifest.json",
                    repo_type="dataset",
                )
                with open(manifest_path) as f:
                    manifest = json.load(f)
                manifest["repo_id"] = ds.id
                manifest["hf_url"] = f"https://huggingface.co/datasets/{ds.id}"
                
                # Author/Badge info
                author = ds.id.split("/")[0]
                manifest["author"] = author
                manifest["is_official"] = (author == "netryx-hub")

                indexes.append(manifest)
            except Exception:
                # Skip repos that don't have a manifest or aren't accessible
                continue

        print(f"[HUB] Found {len(indexes)} indexes")
        for idx in indexes:
            size_mb = idx.get('file_size_bytes', 0) / 1024 / 1024
            print(f"  📦 {idx['name']} — {idx['radius_km']}km — "
                  f"{idx['num_entries']:,} entries — {size_mb:.0f}MB")

        return indexes

    def search(self, lat=None, lon=None, max_distance_km=100, city=None):
        """Search for indexes near a location.
        
        Args:
            lat, lon: Search center coordinates
            max_distance_km: Maximum distance from search point to index center
            city: Search by city name (searches tags and name)
            
        Returns:
            List of matching index manifests, sorted by distance
        """
        all_indexes = self.list_indexes()

        if city:
            city_lower = city.lower()
            all_indexes = [
                idx for idx in all_indexes
                if city_lower in idx.get('name', '').lower()
                or city_lower in ' '.join(idx.get('tags', [])).lower()
            ]

        if lat is not None and lon is not None:
            import math
            def haversine(p1, p2):
                R = 6371
                lat1, lon1 = map(math.radians, p1)
                lat2, lon2 = map(math.radians, p2)
                dlat, dlon = lat2 - lat1, lon2 - lon1
                a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
                return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

            for idx in all_indexes:
                idx['_distance'] = haversine(
                    (lat, lon),
                    (idx['center_lat'], idx['center_lon'])
                )
            all_indexes = [
                idx for idx in all_indexes
                if idx['_distance'] <= max_distance_km + idx.get('radius_km', 0)
            ]
            all_indexes.sort(key=lambda x: x['_distance'])

        return all_indexes

    def download(self, repo_name, output_dir, progress_callback=None):
        """Download an index from the hub and install it.

        Args:
            repo_name: Either full repo ID ("netryx-community/moscow-10km")
                       or just the short name ("moscow-10km")
            output_dir: Netryx DATA_DIR (base dir) -- extract_bundle appends
                        indexes/{uuid} itself.
            progress_callback: fn(message) for status updates

        Returns:
            manifest dict (includes index_id -- pass to load_index() after
            this returns to actually activate the downloaded index)
        """
        # Normalize repo ID
        if "/" not in repo_name:
            # Assume it's the user's own repo
            try:
                user_info = self.api.whoami()
                username = user_info["name"]
                repo_id = f"{username}/{repo_name}"
            except Exception:
                repo_id = repo_name # Fallback
        else:
            repo_id = repo_name

        if progress_callback:
            progress_callback(f"Downloading index from {repo_id}...")
        print(f"[HUB] Downloading from {repo_id}...")

        # Download the bundle file
        try:
            bundle_path = hf_hub_download(
                repo_id=repo_id,
                filename="index.netryx",
                repo_type="dataset",
            )
        except Exception as e:
            # Maybe files are stored individually, not as a bundle
            print(f"[HUB] Bundle not found, trying individual files...")
            return self._download_individual(repo_id, output_dir, progress_callback)

        if progress_callback:
            progress_callback("Extracting index...")

        # Extract bundle (nests under output_dir/indexes/{index_id}/)
        manifest = extract_bundle(bundle_path, output_dir)

        if progress_callback:
            progress_callback(f"Index ready: {manifest['name']}")

        return manifest

    def _download_individual(self, repo_id, output_dir, progress_callback=None):
        """Download index files individually (fallback if no bundle).

        output_dir: Netryx DATA_DIR (base dir) -- files land under
        output_dir/indexes/{index_id}/, matching extract_bundle's layout.
        """
        # Need the manifest first to know the index_id, so fetch it before
        # anything else.
        try:
            manifest_tmp = hf_hub_download(
                repo_id=repo_id, filename="manifest.json", repo_type="dataset",
            )
            with open(manifest_tmp) as f:
                manifest = json.load(f)
        except Exception as e:
            print(f"[HUB]   Could not fetch manifest.json: {e}")
            return None

        index_dir = _index_destination(output_dir, manifest)
        os.makedirs(index_dir, exist_ok=True)

        descriptor_name = ("mixvpr_descriptors.npy"
                            if manifest.get("descriptor_model", "").lower() == "mixvpr"
                            else "megaloc_descriptors.npy")
        file_mapping = {
            "descriptors.npy": descriptor_name,
            "metadata.npz": "metadata.npz",
            "pca_model.pkl": "megaloc_pca.pkl",
            "index_info.txt": "index_info.txt",
        }

        for src_name, dst_name in file_mapping.items():
            try:
                downloaded = hf_hub_download(
                    repo_id=repo_id,
                    filename=src_name,
                    repo_type="dataset",
                )
                if src_name in ("metadata.npz", "pca_model.pkl"):
                    _validate_index_arrays(downloaded)
                dst_path = os.path.join(index_dir, dst_name)
                shutil.copy2(downloaded, dst_path)
                print(f"[HUB]   Downloaded {src_name} -> {dst_name}")
            except ValueError:
                raise
            except Exception as e:
                print(f"[HUB]   Skipping {src_name}: {e}")

        # Write manifest.json last, once every other file is in place --
        # its presence is what tells scan_indexes()/load_index() the index
        # is complete and safe to use.
        with open(os.path.join(index_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        if progress_callback:
            progress_callback(f"Index ready: {manifest.get('name', repo_id)}")

        return manifest

    def upload(self, index_dir, city, radius_km, center_lat, center_lon,
               description=None, tags=None, heading_step=90, crop_fov=90,
               crop_size=256, private=False, token=None):
        """Upload a local index to Hugging Face Hub.
        
        Args:
            index_dir: Directory containing your computed index
            city: City name (used for repo name)
            radius_km: Coverage radius
            center_lat, center_lon: Center coordinates
            description: Optional description
            tags: Optional list of tags
            private: Whether to make the repo private
            
        Returns:
            URL of the created dataset
        """
        # Get HF username dynamically
        try:
            user_info = self.api.whoami()
            username = user_info["name"]
            creator = username
        except Exception:
            username = None
            creator = "anonymous"

        repo_id = _make_repo_id(city, radius_km, org=username)
        name = f"{city.title()} {int(radius_km)}km"
        
        # Inject discovery tags
        base_tags = ["netryx", "geolocation", f"netryx-{city.lower().replace(' ', '-')}"]
        if tags:
            tags = list(set(tags + base_tags))
        else:
            tags = base_tags

        print(f"[HUB] Preparing upload: {name}")
        print(f"[HUB] Repo: {repo_id}")

        # Create bundle — staged next to the index, not in system temp: bundles
        # can be many GB and the OS disk may not have room (staging inherits
        # this location too, see create_bundle)
        bundle_path = os.path.join(
            os.path.dirname(os.path.abspath(index_dir)) or tempfile.gettempdir(),
            f"{city.lower()}_{int(radius_km)}km.netryx")
        bundle_path, manifest = create_bundle(
            index_dir=index_dir,
            output_path=bundle_path,
            name=name,
            description=description,
            center_lat=center_lat,
            center_lon=center_lon,
            radius_km=radius_km,
            tags=tags,
            heading_step=heading_step,
            crop_fov=crop_fov,
            crop_size=crop_size,
            creator=creator,
        )

        # Create HF dataset repo
        print(f"[HUB] Creating repo {repo_id}...")
        try:
            create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True, token=token or self.token)
        except Exception as e:
            print(f"[HUB] Note: {e}")

        # Upload bundle
        print(f"[HUB] Uploading bundle ({manifest['file_size_bytes'] / 1e6:.0f} MB)...")
        upload_file(
            path_or_fileobj=bundle_path,
            path_in_repo="index.netryx",
            repo_id=repo_id,
            repo_type="dataset",
            token=token or self.token,
        )

        # Upload manifest separately (so list_indexes can read it without downloading the full bundle)
        manifest_path = os.path.join(tempfile.gettempdir(), "manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        upload_file(
            path_or_fileobj=manifest_path,
            path_in_repo="manifest.json",
            repo_id=repo_id,
            repo_type="dataset",
            token=token or self.token,
        )

        # Upload README
        readme_path = os.path.join(tempfile.gettempdir(), "README.md")
        with open(readme_path, 'w') as f:
            f.write(_make_readme(manifest))
        upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )

        # Cleanup temp files
        for p in [bundle_path, manifest_path, readme_path]:
            try:
                os.remove(p)
            except Exception:
                pass

        url = f"https://huggingface.co/datasets/{repo_id}"
        print(f"[HUB] ✅ Upload complete!")
        print(f"[HUB] URL: {url}")
        return url

    def delete(self, repo_name):
        """Delete an index from the hub (only works if you own it)."""
        if "/" not in repo_name:
            repo_id = f"{HF_ORG}/{repo_name}"
        else:
            repo_id = repo_name
        self.api.delete_repo(repo_id=repo_id, repo_type="dataset")
        print(f"[HUB] Deleted {repo_id}")


# ─────────────────────────────────────────────────────────────────
# CLI interface
# ─────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Netryx Community Index Hub")
    sub = parser.add_subparsers(dest="command")

    # List
    list_cmd = sub.add_parser("list", help="List available indexes")

    # Search
    search_cmd = sub.add_parser("search", help="Search indexes by city or coordinates")
    search_cmd.add_argument("--city", type=str, help="City name")
    search_cmd.add_argument("--lat", type=float, help="Latitude")
    search_cmd.add_argument("--lon", type=float, help="Longitude")

    # Download
    dl_cmd = sub.add_parser("download", help="Download an index")
    dl_cmd.add_argument("repo", help="Repo name (e.g., moscow-10km)")
    dl_cmd.add_argument("--output", "-o", required=True, help="Output directory")

    # Upload
    up_cmd = sub.add_parser("upload", help="Upload your index")
    up_cmd.add_argument("--index-dir", required=True, help="Path to index directory")
    up_cmd.add_argument("--city", required=True, help="City name")
    up_cmd.add_argument("--radius", type=float, required=True, help="Radius in km")
    up_cmd.add_argument("--lat", type=float, required=True, help="Center latitude")
    up_cmd.add_argument("--lon", type=float, required=True, help="Center longitude")
    up_cmd.add_argument("--tags", nargs="+", help="Tags")

    # Export (local, no upload)
    exp_cmd = sub.add_parser("export", help="Export index as .netryx file (no upload)")
    exp_cmd.add_argument("--index-dir", required=True, help="Path to index directory")
    exp_cmd.add_argument("--output", "-o", required=True, help="Output .netryx file path")
    exp_cmd.add_argument("--city", required=True, help="City name")
    exp_cmd.add_argument("--radius", type=float, required=True, help="Radius in km")
    exp_cmd.add_argument("--lat", type=float, required=True, help="Center latitude")
    exp_cmd.add_argument("--lon", type=float, required=True, help="Center longitude")

    # Import (local, from file)
    imp_cmd = sub.add_parser("import", help="Import a .netryx file")
    imp_cmd.add_argument("bundle", help="Path to .netryx file")
    imp_cmd.add_argument("--output", "-o", required=True, help="Output index directory")

    args = parser.parse_args()

    if args.command == "list":
        hub = NetryxHub()
        hub.list_indexes()

    elif args.command == "search":
        hub = NetryxHub()
        results = hub.search(lat=args.lat, lon=args.lon, city=args.city)
        for r in results:
            dist = r.get('_distance', '?')
            print(f"  📦 {r['name']} — {r['radius_km']}km — "
                  f"{r['num_entries']:,} entries — dist: {dist:.1f}km")

    elif args.command == "download":
        hub = NetryxHub()
        hub.download(args.repo, args.output)

    elif args.command == "upload":
        hub = NetryxHub()
        hub.upload(
            index_dir=args.index_dir,
            city=args.city,
            radius_km=args.radius,
            center_lat=args.lat,
            center_lon=args.lon,
            tags=args.tags,
        )

    elif args.command == "export":
        create_bundle(
            index_dir=args.index_dir,
            output_path=args.output,
            name=f"{args.city.title()} {int(args.radius)}km",
            description=f"Netryx index for {args.city}",
            center_lat=args.lat,
            center_lon=args.lon,
            radius_km=args.radius,
        )

    elif args.command == "import":
        extract_bundle(args.bundle, args.output)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
