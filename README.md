<p align="center">
  <h1 align="center">🔱 Netryx Astra V2</h1>
  <p align="center"><strong>State-of-the-art AI geolocation from a single image.</strong></p>
  <p align="center">
    <a href="#the-idea">The Idea</a> •
    <a href="#how-it-works">How It Works</a> •
    <a href="#getting-started">Getting Started</a> •
    <a href="#community-hub">Community Hub</a> •
    <a href="#installation">Installation</a>
  </p>
  <p align="center">
    <a href="https://www.linkedin.com/in/sairaj-balaji-7295b2246/"><img src="https://img.shields.io/badge/LinkedIn-Sairaj%20Balaji-0A66C2?logo=linkedin" alt="LinkedIn"></a>
    <img src="https://img.shields.io/badge/MegaLoc-CVPR%202025-blue" alt="MegaLoc">
    <img src="https://img.shields.io/badge/MASt3R-ECCV%202024-green" alt="MASt3R">
    <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
    <img src="https://img.shields.io/badge/Python-3.10%2B-orange" alt="Python">
  </p>
</p>

<p align="center">
  <img src="assets/demo.gif" width="800" alt="Netryx Astra V2 in action">
</p>

---

## The Idea

You have a photograph. Maybe it's a screenshot from a video. Maybe it's a cropped, blurry phone photo someone posted online. Maybe it shows just a storefront, a stretch of road, or the corner of a building. You want to know *exactly* where it was taken.

Netryx Astra V2 answers that question.

It's an open-source geolocation system that takes a single image and finds the precise GPS coordinates by matching it against a database of street-view panoramas. Upload your photo, and within minutes it tells you the street, the city, the coordinates — down to a few meters.

What makes V2 different from the original Netryx (and from other tools out there) is the matching pipeline. We rebuilt everything from the ground up using two models that didn't exist when we started this project:

- **MegaLoc** (CVPR 2025) — the most accurate image retrieval model for place recognition, trained across six datasets covering indoor, outdoor, day, night, and seasonal variations. It finds the right neighborhood.

- **MASt3R** (ECCV 2024) — a 3D-aware dense matcher that understands the geometry of scenes, not just pixel patterns. It confirms the exact location, even from partial or heavily cropped photos that would break traditional matchers.

The result is a three-step pipeline that's both simpler and more accurate than the nine-stage system it replaced.

## What Changed from V1

The original Netryx used CosPlace for retrieval and a stack of DISK + LightGlue + LoFTR + RANSAC + descriptor hopping + neighborhood expansion for verification. It worked, but it was fragile — lots of heuristics layered on top of each other, each one a workaround for a limitation in the previous stage.

V2 threw all of that away. Here's what replaced what:

| | V1 (Original) | V2 (Astra) |
|---|---|---|
| **Finding candidates** | CosPlace (ResNet-50, 512-dim) | MegaLoc (DINOv2 ViT-B/14, 8448-dim → PCA 1024) |
| **Confirming matches** | DISK + LightGlue + RANSAC | MASt3R dense 3D matching |
| **Handling edge cases** | LoFTR fallback, descriptor hopping, neighborhood expansion, Ultra Mode | Spatial consensus — that's it |
| **Total pipeline stages** | 9+ | 3 |
| **Partial image matching** | Weak — sparse keypoints fail on small overlaps | Strong — MASt3R finds dense correspondences in tiny regions |
| **Sharing indexes** | Not possible | Community Hub via Hugging Face + offline `.netryx` bundles |

The simplification isn't just aesthetic. Fewer stages means fewer places for things to go wrong, faster searches, and code that's actually maintainable.

## How It Works

The pipeline has three stages. That's not an oversimplification — it's genuinely just three stages.

```
Query Image
     │
     ▼
┌─────────────┐
│   MegaLoc   │  "Where in the city could this be?"
│  Retrieval  │  
└─────┬───────┘
      │  Top 500 candidates
      ▼
┌─────────────┐
│   MASt3R    │  "Is this actually the same place?"
│   Matching  │  
└─────┬───────┘
      │  Scored candidates
      ▼
┌─────────────┐
│   Spatial   │  "Which cluster of matches is most trustworthy?"
│  Consensus  │  
└─────┬───────┘
      │
      ▼
  📍 GPS Coordinates
```

### Stage 1: MegaLoc Retrieval

Your query image gets converted into a compact descriptor — a 8448-dimensional vector that captures the visual essence of the scene. This gets PCA-reduced to 1024 dimensions, then compared against every indexed location via dot-product similarity.

We also extract a descriptor for a slightly zoomed-in center crop and for a horizontally flipped version of the query, then merge the results. This handles cases where the query is at a different zoom level or facing the opposite direction from the indexed view.

The output is the top 500 candidate locations from the index, ranked by visual similarity.

MegaLoc is from Gabriele Berton's lab (the same group that made CosPlace and EigenPlaces). It's the latest in their line of work, trained on SF-XL, GSV-Cities, MSLS, and landmark retrieval data simultaneously. No other retrieval model consistently beats it across every benchmark — indoor, outdoor, urban, rural, day, night.

### Stage 2: MASt3R Dense Matching

For each of those 500 candidates, we download the corresponding street-view panorama, crop it at the indexed heading angle, and run MASt3R to find dense pixel correspondences between the query and the crop.

This is where the magic happens for difficult queries. Traditional matchers like SuperPoint + LightGlue extract maybe 500-2000 sparse keypoints and try to match them. If your query image only overlaps 20% with the database image, there might only be 50 co-visible keypoints — not enough for a reliable match.

MASt3R works completely differently. It treats matching as a 3D reconstruction problem, predicting dense point maps and local feature descriptors for every pixel. Even a small overlapping region produces hundreds of reliable correspondences, because it understands the 3D structure of the scene, not just 2D pixel patterns.

On the Map-free localization benchmark (single reference image, viewpoint changes up to 180°), MASt3R beats previous methods by 30%. That's not an incremental improvement — it's a generational leap.

### Stage 3: Spatial Consensus

Here's the problem with just picking the candidate with the highest match score: false positives exist. Two identical chain restaurants 5km apart will both produce high MASt3R scores. A row of Soviet-era apartment blocks all look the same.

Spatial consensus solves this. We divide the search area into ~50-meter grid cells and cluster all the good matches geographically. Each cell gets a score based on the combined evidence from all matches in that cell and its neighbors.

A single outlier with 200 inliers at the wrong location gets outscored by a cluster of 5 matches with 80-150 inliers each at the right location. The winning cluster's best match becomes the final answer.

This is why accuracy holds up even at larger search radii where there are more look-alike locations.

## Getting Started

### Option A: Download an existing index and start searching

The fastest way. Someone else already did the indexing work — you just download their pre-built index.

```bash
# Download a pre-built index (no Hugging Face account needed)
python netryx_hub.py download moscow-1km -o ./netryx_data/index

# Launch the app
python test_super.py
```

Set mode to **Search**, click **Run Search**, and select your query image. The map coordinates and search radius auto-populate from the index metadata.

### Option B: Build your own index from scratch

Want to index a city or area that nobody's done yet? The app handles everything — downloading panoramas, extracting descriptors, building the search index.

```bash
python test_super.py
```

1. Set mode to **Create**
2. Enter the center coordinates (latitude, longitude) and radius in km
3. Click **Create Index**
4. Wait. For 1km radius, expect ~20-30 minutes. For 5km, a few hours. For 10km, overnight.

What's happening under the hood: the app generates a grid of points within the radius, finds all available panorama locations, downloads each panorama as tiles and stitches them, crops each panorama at multiple heading angles, extracts MegaLoc descriptors for each crop, fits PCA on all descriptors, and builds a compact search index.

### Option C: Import an index file from someone

Got a `.netryx` file from a friend, a Discord server, or a download link? Just click **📥 Import Index** in the app, select the file, and you're ready to search. No account needed, no internet needed — it's a fully offline workflow.

## Community Hub

This is the part we're most excited about.

Indexing a city takes hours of compute time. It's wasteful for every user to independently index the same city. So we built a sharing system: one person indexes Moscow, uploads the result, and everyone else downloads it in minutes.

### How it works

Indexes are hosted on [Hugging Face Hub](https://huggingface.co/netryx-community) as public datasets. Anyone can download without an account. Contributing (uploading) requires a free Hugging Face account.

**From the GUI:** Click the **🌐 Community Hub** button to browse, search, and download available indexes. Click **⬆ Upload Current Index** to share yours.

**From the command line:**

```bash
# See what's available
python netryx_hub.py list

# Search for a specific city
python netryx_hub.py search --city paris

# Search by coordinates (finds all indexes covering that area)
python netryx_hub.py search --lat 48.8566 --lon 2.3522

# Download an index
python netryx_hub.py download paris-10km -o ./netryx_data/index

# Upload your index (requires HF login: huggingface-cli login)
python netryx_hub.py upload \
  --index-dir ./netryx_data/index \
  --city tokyo \
  --radius 5 \
  --lat 35.6762 \
  --lon 139.6503 \
  --tags tokyo japan urban shibuya
```

### The .netryx format

Index bundles use the `.netryx` format — a ZIP archive containing:
- PCA-reduced MegaLoc descriptors (the actual search vectors)
- Coordinate metadata (lat/lon/heading/panoid for every entry)
- The fitted PCA model (needed to transform query descriptors at search time)
- A manifest with coverage info (center, radius, entry count, creator, etc.)

When you export, geographic filtering happens automatically. If your index contains Moscow + Paris + Tokyo but you export "Moscow 1km", only the Moscow entries get included. You can slice specific regions from a larger index without any manual work.

### Offline sharing

Don't want to use Hugging Face? Just export and share the file however you want:

```bash
# Export
python netryx_hub.py export \
  --index-dir ./netryx_data/index \
  -o moscow_1km.netryx \
  --city moscow --radius 1 --lat 55.75 --lon 37.62

# Send the file via Discord, email, Google Drive, whatever

# Other person imports
python netryx_hub.py import moscow_1km.netryx -o ./netryx_data/index
```

## Installation

### Quick setup (recommended)

**Mac / Linux:**
```bash
git clone https://github.com/yourusername/netryx-astra-v2.git
cd netryx-astra-v2
chmod +x setup.sh && ./setup.sh
source venv/bin/activate
python3 test_super.py
```

**Windows:**
```
git clone https://github.com/yourusername/netryx-astra-v2.git
cd netryx-astra-v2
```
Then double-click **`setup.bat`** to install everything. When it finishes, double-click **`run.bat`** to launch.

That's it. The setup script creates a virtual environment, installs all dependencies, clones MASt3R alongside the repo, and pre-downloads the model weights. No manual configuration needed.

### What you need

- **Python 3.10+**
- **macOS** (Apple Silicon via MPS), **Linux** (NVIDIA GPU via CUDA), or **Windows** (NVIDIA GPU via CUDA)
- **8GB+ RAM** for searching (16GB+ recommended for indexing large areas)
- **~2GB disk** per city index



MASt3R weights download automatically from Hugging Face on first run.

Your folder structure should look like this:

```
some_folder/
├── netryx-astra-v2/       # This repo
│   ├── test_super.py
│   ├── megaloc_utils.py
│   ├── mast3r_utils.py
│   ├── netryx_hub.py
│   └── ...
└── mast3r/                # Cloned separately
    ├── mast3r/
    ├── dust3r/
    └── ...
```

`mast3r_utils.py` automatically finds and imports the adjacent `mast3r/` directory at runtime. No path configuration needed.

### Apple Silicon (MPS) notes

Everything runs on Apple Silicon out of the box. The code handles MPS quirks automatically — CPU fallback for unimplemented ops, monkey-patching of `.view()` → `.reshape()` for non-contiguous tensors, and MPS cache cleanup during long indexing runs.

If you have an M1/M2/M3/M4 Mac, it'll use GPU acceleration automatically. No configuration needed.

## Project Structure

```
netryx-astra-v2/
├── test_super.py          # Main application — GUI, pipeline, everything
├── megaloc_utils.py       # MegaLoc model loading, descriptor extraction, PCA
├── megaloc_model.py       # Self-contained MegaLoc architecture (fallback if torch.hub fails)
├── mast3r_utils.py        # MASt3R loading and dense matching
├── netryx_hub.py          # Community Hub — upload, download, export, import
├── README.md
├── LICENSE
└── requirements.txt

# Created at runtime (not committed to git):
netryx_data/
├── megaloc_parts/         # Raw 8448-dim descriptor chunks (created during indexing)
└── index/                 # The compact search index
    ├── megaloc_descriptors.npy   # PCA-reduced descriptors
    ├── metadata.npz              # Coordinates, headings, panoid IDs
    ├── megaloc_pca.pkl           # PCA model for query-time transformation
    └── manifest.json             # Present if downloaded from Community Hub

# External dependency (cloned separately, NOT inside this repo):
../mast3r/                 # https://github.com/naver/mast3r
```

## Configuration

You can tune these parameters in `test_super.py` if needed. The defaults work well for most use cases.

| Parameter | Default | What it does |
|---|---|---|
| `INDEX_TARGET_DIM` | 1024 | PCA output dimension. 512 = smaller index, slightly less accurate. 1024 = good balance. |
| `MAX_PANOID_WORKERS` | 16 | How many panoramas to download in parallel during indexing |
| `MAX_DOWNLOAD_WORKERS` | 100 | Concurrent tile download connections (each pano = 8 tiles) |
| `EARLY_EXIT_INLIER_THRESHOLD` | 450 | If MASt3R finds this many dense matches, stop searching early — it's a confident hit |
| `MAST3R_STAGE2_TOP_N` | 500 | How many MegaLoc candidates to run through MASt3R |

The main thing you might want to adjust is `INDEX_TARGET_DIM` — dropping to 512 halves your index size with only ~2-3% accuracy loss.

## Limitations

We believe in being upfront about what this tool can and can't do.

**It only finds places that are in the index.** If a location hasn't been indexed by you or downloaded from the community, it won't be found. No model, no matter how advanced, can match against data that doesn't exist. The accuracy ceiling is set by coverage, not by the models.

**Repetitive architecture causes false positives.** Chain stores, suburban housing developments, rows of identical apartment blocks — these genuinely look the same from street level. Spatial consensus helps (a cluster of nearby matches beats a single isolated outlier), but it's not bulletproof, especially at large search radii (10km+) where there are more look-alike candidates. This is an unsolved problem in the field, not a limitation specific to Netryx.

**Coverage gaps exist in some areas.** Rural areas, developing countries, newly built neighborhoods, and indoor spaces may have limited or no street-level imagery available for indexing.

**This is not a real-time system.** Running MASt3R on 500 candidates takes several minutes on a consumer GPU. This is designed for forensic analysis — investigative journalism, human rights documentation, OSINT research, insurance fraud investigation — not for navigation or live tracking.

**Indexing requires meaningful compute.** MegaLoc's DINOv2 backbone is ~5x heavier than the CosPlace backbone used in V1. Indexing a 1km radius takes ~20-30 minutes. A full city (10km+) is an overnight job. That's why the Community Hub exists — so this cost is paid once and shared.



## Citation

If you use Netryx Astra V2 in your research or work, we'd appreciate a citation:

```bibtex
@software{netryx_astra_v2,
  title={Netryx Astra V2: State-of-the-Art AI Geolocation},
  author={Sairaj Balaji},
  year={2026},
  url={https://github.com/yourusername/netryx-astra-v2}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

MegaLoc weights are MIT licensed. MASt3R is Apache 2.0 licensed. DINOv2 is Apache 2.0 licensed. Community-shared indexes are CC-BY-4.0.

---

<p align="center">
  <strong>Built by <a href="https://github.com/yourusername">Sairaj Balaji</a></strong><br>
  <em>Founder, <a href="https://avtaar.ai">Avtaar.ai</a></em><br><br>
  As seen in <a href="#">Fast Company</a> · <a href="#">404 Media</a> · <a href="#">Deutsche Welle</a>
</p>
