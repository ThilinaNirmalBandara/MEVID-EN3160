# mevid_textsearch/faiss_utils.py
import faiss
import numpy as np
from pathlib import Path
from typing import Literal

def build_index(
    X: np.ndarray,
    kind: Literal["flat","ivfpq"]="flat",
    nlist: int = 512,
    m: int = 64,
    nprobe: int = 16,
):
    d = X.shape[1]
    if kind == "flat":
        index = faiss.IndexFlatIP(d)
        index.add(X.astype("float32"))
        return index
    # IVF-PQ
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
    index.train(X.astype("float32"))
    index.add(X.astype("float32"))
    index.nprobe = nprobe
    return index

def save_index(index, path: Path):
    faiss.write_index(index, str(path))

def load_index(path: Path):
    return faiss.read_index(str(path))
