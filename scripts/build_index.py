
import sys, os
from pathlib import Path
import json

# ensure we can import config + package when running as: python scripts/parse_mevid.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# scripts/build_index.py
from pathlib import Path
import numpy as np
import pickle
from config import ARTIFACTS, FAISS_KIND, FAISS_NLIST, FAISS_M, FAISS_NPROBE
from mevid_textsearch.faiss_utils import build_index, save_index

def main():
    X_path = ARTIFACTS / "vecs_test.npy"
    meta_path = ARTIFACTS / "meta_test.pkl"
    assert X_path.exists() and meta_path.exists(), "Run extract_vecs.py first"
    X = np.load(X_path)
    index = build_index(X, FAISS_KIND, FAISS_NLIST, FAISS_M, FAISS_NPROBE)
    save_index(index, ARTIFACTS / "faiss.index")
    print("[index] saved index at artifacts/faiss.index")

if __name__ == "__main__":
    main()
