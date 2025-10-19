# config.py
from pathlib import Path

# >>>>> CHANGE THIS to your MEVID path <<<<<
MEVID_ROOT = Path(r"D:/MEVID-main/mevid")  # folder that contains bbox_train, bbox_test, and the annotation txt files

# Where to write outputs (vectors/index/thumbs)
ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# Frame sampling per tracklet (evenly spaced)
FRAMES_PER_TRACKLET = 8

# Choose FAISS index: "flat" (exact) or "ivfpq" (ANN)
FAISS_KIND = "flat"   # "flat" or "ivfpq"
FAISS_NLIST = 512      # inverted lists (if ivfpq)
FAISS_M = 64           # PQ bytes (if ivfpq)
FAISS_NPROBE = 16

# Streamlit thumbs directory (small GIFs/webm)
THUMBS_DIR = ARTIFACTS / "thumbs"
THUMBS_DIR.mkdir(exist_ok=True)


# ReID Model Settings
REID_MODEL_TYPE = 'ap3d'  # Options: 'temporal', 'ap3d', 'fastreid', 'transreid'
REID_FRAMES = 16           # Number of frames to sample
REID_BATCH_SIZE = 8        # Batch size for feature extraction