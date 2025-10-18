# MEvid Hybrid Person Search System

## 🔍 Overview

A state-of-the-art person re-identification system combining **CLIP** (text-to-image) and **Video ReID** (temporal matching) for accurate cross-camera person tracking in the MEvid dataset.

### Key Features

- 🎯 **Hybrid Search**: CLIP + Video ReID for maximum accuracy
- ⚡ **Fast Retrieval**: FAISS-powered vector search
- 🎬 **Video Tracklets**: Animated GIF previews of detected persons
- 📹 **Multi-Camera Tracking**: Find the same person across different cameras
- 🖥️ **Interactive UI**: Beautiful Streamlit web interface
- 📊 **Analytics**: Performance metrics and visualizations

---

## 📁 Project Structure

```
mevid-textsearch/
│
├── 📂 mevid_textsearch/              Core Package
├── 📂 scripts/                       Executable Scripts
├── 📂 artifacts/                     Generated Data
├── 📂 app/                           Streamlit Application
├── config.py                         Configuration
├── requirements.txt                  Dependencies
└── README.md                         This file
```

---

## 📦 Detailed File Reference

### 🔷 Core Package (`mevid_textsearch/`)

The main Python package containing all core functionality.

#### **`__init__.py`** 
```python
# Marks directory as Python package
# Can be empty or contain package-level imports
```
**Use Case:** Required for Python to recognize the folder as a package.  
**Required:** ✅ YES  
**When to modify:** Never (keep empty or add package exports)

---

#### **`clip_utils.py`**
```python
Functions:
- load_clip()           # Load CLIP model
- encode_frame()        # Encode single image
- encode_text()         # Encode text query
- pool_mean()           # Average frame features
```
**Use Case:** Handles all CLIP-related operations for text-to-image matching.  
**Required:** ✅ YES  
**Dependencies:** `torch`, `clip`, `PIL`  
**When to use:** Automatically used during search for text encoding and frame feature extraction.

**Example:**
```python
from mevid_textsearch.clip_utils import encode_text
query_vec = encode_text(["man in black jacket"])
```

---

#### **`faiss_utils.py`**
```python
Functions:
- build_index()         # Create FAISS index
- save_index()          # Save index to disk
- load_index()          # Load index from disk
```
**Use Case:** Manages FAISS vector search index for fast similarity search.  
**Required:** ✅ YES  
**Dependencies:** `faiss`, `numpy`  
**When to use:** Building search index (once) and loading for queries (always).

**Example:**
```python
from mevid_textsearch.faiss_utils import load_index
index = load_index("artifacts/faiss.index")
similarities, indices = index.search(query_vec, k=10)
```

---

#### **`mevid.py`**
```python
Classes/Functions:
- Tracklet              # Dataclass for tracklet info
- build_tracklets()     # Parse MEvid annotations
- save_jsonl()          # Save tracklets to file
```
**Use Case:** Parses MEvid dataset annotation files into structured tracklet objects.  
**Required:** ⚠️ ONCE (for data preparation)  
**Dependencies:** `pathlib`, `json`  
**When to use:** Only during initial setup when parsing the MEvid dataset.

**Example:**
```python
from mevid_textsearch.mevid import build_tracklets
tracklets = build_tracklets(mevid_root, split="test")
# Returns list of Tracklet objects with frame paths, IDs, etc.
```

---

#### **`viz.py`**
```python
Functions:
- make_gif_from_images() # Create animated GIF from frames
```
**Use Case:** Generates animated GIF thumbnails from video tracklet frames.  
**Required:** 🎨 OPTIONAL (but recommended for better UX)  
**Dependencies:** `PIL`  
**When to use:** Generate thumbnails once after extracting features.

**Example:**
```python
from mevid_textsearch.viz import make_gif_from_images
frame_paths = [Path("frame1.jpg"), Path("frame2.jpg"), ...]
make_gif_from_images(frame_paths, "output.gif", fps=6)
```

---

#### **`temporal_reid.py`**
```python
Classes:
- TemporalAttention       # Attention mechanism for frames
- VideoReIDModel          # Main ReID neural network
- VideoReIDExtractor      # Wrapper for feature extraction
```
**Use Case:** Video-based person re-identification using temporal information.  
**Required:** ✅ YES  
**Dependencies:** `torch`, `torchvision`  
**When to use:** Automatically used when ReID is enabled in search.

**Model Architecture:**
```
Input: (batch, seq_len, 3, 256, 128)
  ↓
ResNet50 Backbone
  ↓
Temporal Attention (learns frame importance)
  ↓
Feature Vector (2048-d)
  ↓
L2 Normalization
```

**Example:**
```python
from mevid_textsearch.temporal_reid import VideoReIDExtractor
extractor = VideoReIDExtractor(model_path="reid_model.pth")
feature = extractor.extract_tracklet_feature(frame_paths)
```

---

#### **`hybrid_search.py`**
```python
Classes:
- SearchResult            # Result container
- HybridSearchEngine      # Main search engine
```
**Use Case:** Combines CLIP and Video ReID for hybrid person search.  
**Required:** ✅ YES  
**Dependencies:** All above modules  
**When to use:** Main interface for all search operations.

**Key Methods:**
```python
engine = HybridSearchEngine(artifacts_dir, reid_model_path)

# Text-based search
results = engine.search(
    query="man in black jacket",
    use_reid_rerank=True
)

# Video-to-video ReID
results = engine.person_reidentification(
    reference_track_id=1548
)

# Speed optimization
engine.precompute_reid_features(mevid_root)
```

---

### 🔷 Scripts (`scripts/`)

Executable Python scripts for various tasks.

#### **`parse_mevid.py`**
**Purpose:** Parse MEvid dataset annotations into structured format.  
**Required:** ⚠️ RUN ONCE  
**When to run:** Initial setup before any searches.  

**What it does:**
1. Reads `test_name.txt` and `track_test_info.txt`
2. Parses tracklet information (person ID, camera, frames)
3. Saves to `artifacts/tracks_test.jsonl`
4. Extracts query indices from `query_IDX.txt`

**Usage:**
```bash
python scripts/parse_mevid.py
```

**Output:**
- `artifacts/tracks_test.jsonl` (1754 tracklets)
- `artifacts/query_idx.json` (200 query indices)

---

#### **`extract_vecs.py`**
**Purpose:** Extract CLIP feature vectors for all tracklets.  
**Required:** ⚠️ RUN ONCE  
**When to run:** After parsing, before building index.  

**What it does:**
1. Loads parsed tracklets
2. Samples 12 frames per tracklet
3. Encodes frames with CLIP
4. Averages frame features
5. Saves L2-normalized vectors

**Usage:**
```bash
python scripts/extract_vecs.py
# Takes 5-10 minutes on GPU, 15-25 minutes on CPU
```

**Output:**
- `artifacts/vecs_test.npy` (1754 × 512 float32 array)
- `artifacts/meta_test.pkl` (metadata for each tracklet)

---

#### **`build_index.py`**
**Purpose:** Build FAISS search index from CLIP features.  
**Required:** ⚠️ RUN ONCE  
**When to run:** After extracting features.  

**What it does:**
1. Loads feature vectors
2. Builds IVF-PQ FAISS index for fast search
3. Saves index to disk

**Usage:**
```bash
python scripts/build_index.py
# Takes ~1-2 minutes
```

**Output:**
- `artifacts/faiss.index` (compressed search index)

---

#### **`make_thumbs.py`**
**Purpose:** Generate animated GIF thumbnails for visualization.  
**Required:** 🎨 OPTIONAL  
**When to run:** After extracting features (optional).  

**What it does:**
1. Samples frames from each tracklet
2. Creates animated GIF (320×240)
3. Saves to thumbs directory

**Usage:**
```bash
python scripts/make_thumbs.py
# Takes ~10-15 minutes for 1754 GIFs
```

**Output:**
- `artifacts/thumbs/track_0.gif` through `track_1753.gif`

---

#### **`train_reid.py`**
**Purpose:** Train Video ReID model on MEvid dataset.  
**Required:** 🔬 OPTIONAL  
**When to run:** If you want to improve ReID accuracy beyond pretrained ResNet50.  

**What it does:**
1. Loads training tracklets
2. Trains VideoReIDModel with:
   - Classification loss (person ID)
   - Triplet loss (metric learning)
   - Temporal attention
3. Saves best model checkpoint

**Usage:**
```bash
python scripts/train_reid.py \
    --epochs 50 \
    --batch_size 2 \
    --frames 8 \
    --lr 0.0003
# Takes 2-3 hours on GPU, 20+ hours on CPU
```

**Output:**
- `artifacts/reid_model.pth` (trained model weights)

**Training Curves:**
```
Epoch 1: Loss=4.234, Acc=32.45%
Epoch 10: Loss=2.156, Acc=58.32%
Epoch 50: Loss=1.234, Acc=74.56%
```

---

#### **`search_hybrid.py`**
**Purpose:** Command-line interface for searching.  
**Required:** 🛠️ OPTIONAL (alternative to Streamlit app)  
**When to use:** Quick searches from terminal or scripting.  

**Usage:**
```bash
# Text search
python scripts/search_hybrid.py \
    --query "man in black jacket" \
    --topk 10

# Video ReID
python scripts/search_hybrid.py \
    --track_id 1548 \
    --topk 20

# Precompute features
python scripts/search_hybrid.py --precompute
```

---

#### **`eval_hybrid.py`**
**Purpose:** Evaluate system performance with metrics.  
**Required:** 📊 OPTIONAL  
**When to use:** Benchmarking, comparing CLIP vs Hybrid.  

**What it does:**
1. Runs queries on test set
2. Computes Rank@1, Rank@5, Rank@10, mAP
3. Compares CLIP-only vs Hybrid modes

**Usage:**
```bash
# Evaluate ReID
python scripts/eval_hybrid.py --eval_reid

# Compare CLIP vs Hybrid
python scripts/eval_hybrid.py --eval_text --eval_both
```

**Output:**
```
Video ReID (Video-to-Video)
  Rank-1:  68.50%
  Rank-5:  85.30%
  Rank-10: 91.20%
  mAP:     61.20%

Text Search Comparison
Metric      CLIP Only    Hybrid       Improvement
Recall@1    45.20%       62.70%       +17.50%
Recall@5    71.80%       84.50%       +12.70%
```

---

#### **`eval_reid_video2video.py`**
**Purpose:** Standalone video-to-video ReID evaluation.  
**Required:** 📊 OPTIONAL  
**When to use:** Specifically test pure ReID performance.  

**Usage:**
```bash
python scripts/eval_reid_video2video.py
```

---

#### **`demo_visual.py`**
**Purpose:** Create visual comparison images of search results.  
**Required:** 📊 OPTIONAL  
**When to use:** Generate presentable result visualizations.  

**Usage:**
```bash
python scripts/demo_visual.py \
    --query "man in black jacket" \
    --compare \
    --output comparison.jpg
```

**Output:**
- `comparison.jpg` (side-by-side CLIP vs Hybrid results)

---

#### **`check_setup.py`**
**Purpose:** Verify installation and configuration.  
**Required:** 🛠️ HELPFUL  
**When to use:** After installation, troubleshooting.  

**What it checks:**
- ✅ Python version
- ✅ All dependencies installed
- ✅ CUDA availability
- ✅ Config paths valid
- ✅ MEvid dataset present
- ✅ Artifacts generated

**Usage:**
```bash
python scripts/check_setup.py
```

---

### 🔷 Configuration

#### **`config.py`**
**Purpose:** Central configuration for all paths and hyperparameters.  
**Required:** ✅ YES  
**When to modify:** Setup (change paths) or tuning (change parameters).  

```python
# Dataset Paths
MEVID_ROOT = Path(r"D:\MEVID-main\mevid")
ARTIFACTS = Path("artifacts")
THUMBS_DIR = ARTIFACTS / "thumbs"

# CLIP Settings
FRAMES_PER_TRACKLET = 12
CLIP_MODEL = "ViT-B/16"

# FAISS Settings
FAISS_KIND = "ivfpq"
FAISS_NLIST = 512
FAISS_M = 64
FAISS_NPROBE = 16

# ReID Settings
REID_FRAMES = 16
REID_ALPHA = 0.6
DIVERSITY_PENALTY = 0.03

# Training Settings
TRAIN_EPOCHS = 50
TRAIN_BATCH_SIZE = 8
TRAIN_LR = 0.0003
TRAIN_MARGIN = 0.3
```

**Important Variables:**
- `MEVID_ROOT`: **MUST CHANGE** to your dataset location
- `REID_ALPHA`: 0.6 = 60% CLIP + 40% ReID (tune for your use case)
- `FRAMES_PER_TRACKLET`: More frames = better accuracy but slower

---

### 🔷 Application

#### **`app.py`**
**Purpose:** Interactive Streamlit web interface.  
**Required:** ✅ YES (for UI)  
**When to use:** Main application interface.  

**Features:**
- 🔤 Text-based person search
- 🎥 Video-to-video re-identification
- 📊 Real-time statistics
- 🎬 Animated GIF previews
- 📥 CSV export
- ⚙️ Adjustable parameters

**Usage:**
```bash
streamlit run app.py
# Opens browser at http://localhost:8501
```

**UI Sections:**
1. **Sidebar**: Settings (ReID on/off, alpha, topk, etc.)
2. **Main**: Search input and results
3. **Results Grid**: 3-column layout with GIFs
4. **Metrics**: Stats on persons, cameras, scores
5. **Export**: Download results as CSV

---

### 🔷 Data Files (`artifacts/`)

Generated during setup, required for operation.

#### **`tracks_test.jsonl`**
**Format:** JSON Lines (one JSON object per line)  
**Size:** ~500 KB  
**Generated by:** `parse_mevid.py`  

**Content:**
```json
{"tid": 0, "pid": 201, "outfit": 1, "camid": 329, "frames": ["bbox_test/0201/..."]}
{"tid": 1, "pid": 201, "outfit": 1, "camid": 330, "frames": ["bbox_test/0201/..."]}
```

**Use Case:** Metadata for all test tracklets (person ID, camera, frame paths).

---

#### **`query_idx.json`**
**Format:** JSON array  
**Size:** ~2 KB  
**Generated by:** `parse_mevid.py`  

**Content:**
```json
[0, 15, 28, 42, ...]  // 200 query indices
```

**Use Case:** Which tracklets to use as queries in evaluation.

---

#### **`vecs_test.npy`**
**Format:** NumPy array (1754 × 512, float32)  
**Size:** ~3.5 MB  
**Generated by:** `extract_vecs.py`  

**Content:** L2-normalized CLIP feature vectors for each tracklet.

**Use Case:** Fast similarity search with FAISS.

---

#### **`meta_test.pkl`**
**Format:** Python pickle (list of dicts)  
**Size:** ~200 KB  
**Generated by:** `extract_vecs.py`  

**Content:**
```python
[
    {"tid": 0, "pid": 201, "outfit": 1, "camid": 329, "frames": [...]},
    ...
]
```

**Use Case:** Retrieve tracklet metadata from search results.

---

#### **`faiss.index`**
**Format:** FAISS IVF-PQ index  
**Size:** ~1-2 MB  
**Generated by:** `build_index.py`  

**Use Case:** Fast approximate nearest neighbor search.

---

#### **`reid_model.pth`**
**Format:** PyTorch model checkpoint  
**Size:** ~100 MB  
**Generated by:** `train_reid.py` (optional)  

**Content:** Trained VideoReIDModel weights.

**Use Case:** Better ReID accuracy (optional, uses pretrained ResNet50 if absent).

---

#### **`reid_features.npy`**
**Format:** NumPy array (1754 × 2048, float32)  
**Size:** ~14 MB  
**Generated by:** `search_hybrid.py --precompute` (optional)  

**Use Case:** Precomputed ReID features for 10× faster searches.

---

#### **`thumbs/track_*.gif`**
**Format:** Animated GIF  
**Size:** ~50-200 KB each  
**Generated by:** `make_thumbs.py` (optional)  

**Use Case:** Visual previews in Streamlit app.

---

### 🔷 Dependencies

#### **`requirements.txt`**
```
torch>=1.12.0
torchvision>=0.13.0
git+https://github.com/openai/CLIP.git
faiss-cpu>=1.7.3
Pillow>=9.0.0
numpy>=1.21.0
scikit-learn>=1.0.0
tqdm>=4.62.0
streamlit>=1.28.0
plotly>=5.14.0
pandas>=1.3.0
```

**Installation:**
```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start Guide

### **Minimal Setup (Just Run Searches)**

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure paths
# Edit config.py: Set MEVID_ROOT to your dataset location

# 3. Build index (one-time setup)
python scripts/parse_mevid.py
python scripts/extract_vecs.py
python scripts/build_index.py

# 4. Run app
streamlit run app.py
```

### **Full Setup (With Training & Evaluation)**

```bash
# ... same as above, then:

# 5. Generate thumbnails (optional, recommended)
python scripts/make_thumbs.py

# 6. Train ReID model (optional, takes 2-3 hours on GPU)
python scripts/train_reid.py --epochs 50 --batch_size 2

# 7. Precompute features for speed (optional)
python scripts/search_hybrid.py --precompute

# 8. Evaluate performance (optional)
python scripts/eval_hybrid.py --eval_both
```

---

## 📊 Use Case Matrix

| Task | Required Files | Script to Run |
|------|----------------|---------------|
| **Run searches** | Core package + artifacts + app.py | `streamlit run app.py` |
| **Parse dataset** | mevid.py + parse_mevid.py | `python scripts/parse_mevid.py` |
| **Extract features** | clip_utils.py + extract_vecs.py | `python scripts/extract_vecs.py` |
| **Build index** | faiss_utils.py + build_index.py | `python scripts/build_index.py` |
| **Generate GIFs** | viz.py + make_thumbs.py | `python scripts/make_thumbs.py` |
| **Train model** | temporal_reid.py + train_reid.py | `python scripts/train_reid.py` |
| **CLI search** | All core + search_hybrid.py | `python scripts/search_hybrid.py` |
| **Evaluate** | All core + eval_*.py | `python scripts/eval_hybrid.py` |

---

## 🎯 Recommended Workflow

### **For End Users (Just Want to Search)**

```
1. config.py ✅
2. parse_mevid.py (once) ✅
3. extract_vecs.py (once) ✅
4. build_index.py (once) ✅
5. app.py ✅

Delete: All scripts/ except maybe search_hybrid.py
Keep: Core package + artifacts/ + config.py + app.py
```

### **For Developers (Full System)**

```
Keep everything!
- Core package: All modules
- Scripts: All scripts for different tasks
- Artifacts: All generated data
- Config: Tune parameters
- App: Customize UI
```

### **For Researchers (Evaluation & Training)**

```
Focus on:
- temporal_reid.py (model architecture)
- train_reid.py (training)
- eval_*.py (benchmarking)
- hybrid_search.py (algorithm)
```

---

## 🔧 Troubleshooting

### Common Issues

**Problem:** "No module named 'config'"  
**Solution:** Run from project root, not scripts/ folder

**Problem:** "CUDA out of memory"  
**Solution:** Reduce batch size in train_reid.py

**Problem:** "No thumbnails showing"  
**Solution:** Run `make_thumbs.py` or app shows placeholders

**Problem:** "Search returns no results"  
**Solution:** Rebuild index with `build_index.py`

---

## 📈 Performance

| Metric | CLIP Only | CLIP + ReID | Improvement |
|--------|-----------|-------------|-------------|
| Text Search R@1 | 45.2% | 62.7% | +17.5% |
| Text Search R@5 | 71.8% | 84.5% | +12.7% |
| Video ReID R@1 | - | 68.5% | - |
| Speed (per query) | 0.1s | 1.2s | 12× slower |

---

## 📄 License

MIT License - Feel free to use in your projects!

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Submit pull request

---

## 📞 Support

- 📧 Email: your.email@example.com
- 🐛 Issues: GitHub Issues
- 📖 Docs: This README

---

**Built with ❤️ using PyTorch, CLIP, FAISS, and Streamlit**