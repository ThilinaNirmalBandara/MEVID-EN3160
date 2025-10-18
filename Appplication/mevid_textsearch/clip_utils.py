# mevid_textsearch/clip_utils.py
import torch
import clip
from PIL import Image
import numpy as np
from typing import List
from pathlib import Path
from tqdm import tqdm

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL = None
_PREP = None

def load_clip(model_name: str = "ViT-B/16"):
    global _MODEL, _PREP
    if _MODEL is None:
        _MODEL, _PREP = clip.load(model_name, device=_DEVICE)
        _MODEL.eval()
    return _MODEL, _PREP, _DEVICE

@torch.no_grad()
def encode_frame(path: Path):
    model, preprocess, device = load_clip()
    im = Image.open(path).convert("RGB")
    t = preprocess(im).unsqueeze(0).to(device)
    v = model.encode_image(t)  # (1, D)
    v = torch.nn.functional.normalize(v, dim=-1)
    return v.squeeze(0).cpu().numpy()  # (D,)

@torch.no_grad()
def encode_text(texts: List[str]) -> np.ndarray:
    model, _, device = load_clip()
    tok = clip.tokenize(texts).to(device)
    v = model.encode_text(tok)
    v = torch.nn.functional.normalize(v, dim=-1)
    return v.cpu().numpy()  # (N, D)

def pool_mean(vecs: List[np.ndarray]) -> np.ndarray:
    V = np.stack(vecs, axis=0)  # (T, D)
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    m = V.mean(axis=0)
    m = m / (np.linalg.norm(m) + 1e-12)
    return m
