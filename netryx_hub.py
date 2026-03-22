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
import hashlib
import zipfile
import tempfile
import shutil
import time
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

HF_ORG = "netryx-hub"  # Hugging Face organization name
BUNDLE_FORMAT_VERSION = "2.0"
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

    # Read full index
    descs = np.load(descs_path, mmap_mode='r')
    meta = np.load(meta_path, allow_pickle=True)
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

    # Extract filtered subset
    filtered_descs = np.array(descs[valid_idx], dtype=np.float32)
    filtered_lats = lats[valid_idx]
    filtered_lons = lons[valid_idx]
    filtered_headings = meta['headings'][valid_idx]
    filtered_panoids = meta['panoids'][valid_idx]
    filtered_paths = meta['paths'][valid_idx]
    
    num_entries = len(filtered_descs)
    desc_dim = filtered_descs.shape[1]
    panoid_set = set(str(p) for p in filtered_panoids)
    num_panoids = len(panoid_set)
    
    del descs, meta

    # Build manifest
    manifest = {
        "format_version": BUNDLE_FORMAT_VERSION,
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
        "descriptor_model": "MegaLoc",
        "pca_components": int(desc_dim),
        "heading_step_deg": int(heading_step),
        "crop_fov_deg": int(crop_fov),
        "crop_size_px": int(crop_size),
        "tags": tags or [],
    }

    # Create ZIP bundle with FILTERED data
    print(f"[HUB] Creating bundle: {name} ({num_entries} entries, {num_panoids} panoids)")
    
    # Save filtered arrays to temp files, then zip them
    tmp_dir = tempfile.mkdtemp(prefix="netryx_bundle_")
    try:
        tmp_descs = os.path.join(tmp_dir, "descriptors.npy")
        tmp_meta = os.path.join(tmp_dir, "metadata.npz")
        
        np.save(tmp_descs, filtered_descs)
        np.savez_compressed(tmp_meta,
            lats=filtered_lats, lons=filtered_lons,
            headings=filtered_headings, panoids=filtered_panoids,
            paths=filtered_paths)
        
        del filtered_descs, filtered_lats, filtered_lons
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

    # Compute SHA256
    sha = hashlib.sha256(open(output_path, 'rb').read()).hexdigest()
    manifest["sha256"] = sha
    manifest["file_size_bytes"] = os.path.getsize(output_path)

    size_mb = manifest["file_size_bytes"] / 1024 / 1024
    print(f"[HUB] Bundle created: {output_path} ({size_mb:.0f} MB)")
    print(f"[HUB]   {num_entries} entries, {num_panoids} panoids, {desc_dim}-dim")

    return output_path, manifest


def extract_bundle(bundle_path, index_dir):
    """Extract a .netryx bundle into Netryx's index directory.
    
    Args:
        bundle_path: Path to .netryx file
        index_dir: Target directory (e.g., COMPACT_INDEX_DIR)
        
    Returns:
        manifest dict
    """
    os.makedirs(index_dir, exist_ok=True)

    with zipfile.ZipFile(bundle_path, 'r') as zf:
        # Read manifest first
        manifest = json.loads(zf.read("manifest.json"))
        print(f"[HUB] Extracting: {manifest['name']}")
        print(f"[HUB]   Coverage: ({manifest['center_lat']:.4f}, {manifest['center_lon']:.4f}) "
              f"r={manifest['radius_km']}km")
        print(f"[HUB]   Entries: {manifest['num_entries']}, dim: {manifest['descriptor_dim']}")

        # Extract with correct filenames for Netryx
        file_mapping = {
            "descriptors.npy": "megaloc_descriptors.npy",
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

def _make_repo_id(city, radius_km):
    """Generate a HF repo ID from city name and radius."""
    slug = city.lower().strip().replace(" ", "-").replace(",", "")
    return f"{HF_ORG}/{slug}-{int(radius_km)}km"


def _make_readme(manifest):
    """Generate a README.md for the HF dataset repo."""
    return f"""---
tags:
- netryx
- geolocation
- visual-place-recognition
- megaloc
- osint
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
hub.download("{manifest['name'].lower().replace(' ', '-')}", output_dir="./netryx_data/index")
# Now open Netryx and search!
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

    def __init__(self, token=None):
        if not HF_AVAILABLE:
            raise ImportError("Install huggingface_hub: pip install huggingface_hub")
        self.token = token
        self.api = HfApi(token=token)

    def list_indexes(self):
        """List all available Netryx indexes on the hub.
        
        Returns:
            List of dicts with index metadata
        """
        print(f"[HUB] Searching for indexes in {HF_ORG}...")
        try:
            datasets = list(self.api.list_datasets(author=HF_ORG))
        except Exception as e:
            print(f"[HUB] Error listing datasets: {e}")
            return []

        indexes = []
        for ds in datasets:
            try:
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
                indexes.append(manifest)
            except Exception:
                # Skip repos that don't have a manifest (not a netryx index)
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
            output_dir: Netryx COMPACT_INDEX_DIR to install into
            progress_callback: fn(message) for status updates
            
        Returns:
            manifest dict
        """
        # Normalize repo ID
        if "/" not in repo_name:
            repo_id = f"{HF_ORG}/{repo_name}"
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

        # Extract bundle
        manifest = extract_bundle(bundle_path, output_dir)

        # Load PCA model if present
        pca_path = os.path.join(output_dir, "megaloc_pca.pkl")
        if os.path.exists(pca_path):
            try:
                from megaloc_utils import load_pca
                load_pca(pca_path)
            except Exception:
                pass

        if progress_callback:
            progress_callback(f"Index ready: {manifest['name']}")

        return manifest

    def _download_individual(self, repo_id, output_dir, progress_callback=None):
        """Download index files individually (fallback if no bundle)."""
        os.makedirs(output_dir, exist_ok=True)

        file_mapping = {
            "descriptors.npy": "megaloc_descriptors.npy",
            "metadata.npz": "metadata.npz",
            "pca_model.pkl": "megaloc_pca.pkl",
            "manifest.json": "manifest.json",
            "index_info.txt": "index_info.txt",
        }

        manifest = None
        for src_name, dst_name in file_mapping.items():
            try:
                downloaded = hf_hub_download(
                    repo_id=repo_id,
                    filename=src_name,
                    repo_type="dataset",
                )
                dst_path = os.path.join(output_dir, dst_name)
                shutil.copy2(downloaded, dst_path)
                print(f"[HUB]   Downloaded {src_name} -> {dst_name}")

                if src_name == "manifest.json":
                    with open(downloaded) as f:
                        manifest = json.load(f)
            except Exception as e:
                print(f"[HUB]   Skipping {src_name}: {e}")

        if manifest and progress_callback:
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
        repo_id = _make_repo_id(city, radius_km)
        name = f"{city.title()} {int(radius_km)}km"
        if description is None:
            description = f"Netryx MegaLoc index for {city.title()}, {radius_km}km radius"

        if tags is None:
            tags = [city.lower(), "urban"]

        # Get HF username
        try:
            user_info = self.api.whoami()
            creator = user_info["name"]
        except Exception:
            creator = "anonymous"

        print(f"[HUB] Preparing upload: {name}")
        print(f"[HUB] Repo: {repo_id}")

        # Create bundle
        bundle_path = os.path.join(tempfile.gettempdir(), f"{city.lower()}_{int(radius_km)}km.netryx")
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
