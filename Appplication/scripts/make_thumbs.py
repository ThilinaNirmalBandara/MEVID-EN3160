import sys, os
from pathlib import Path
import json

# ensure we can import config + package when running as: python scripts/parse_mevid.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# scripts/make_thumbs.py
from pathlib import Path
import pickle
import numpy as np
from tqdm import tqdm
from config import MEVID_ROOT, ARTIFACTS, THUMBS_DIR, FRAMES_PER_TRACKLET
from mevid_textsearch.viz import make_gif_from_images

def evenly_sample(lst, k):
    if len(lst) <= k:
        return lst
    idxs = np.linspace(0, len(lst)-1, num=k, dtype=int)
    return [lst[i] for i in idxs]

def resolve_rel_path(rel: str) -> Path:
    """
    Try multiple layouts:
    1) <MEVID_ROOT>/<rel>
    2) <MEVID_ROOT>/bbox_<split>/<pid4>/<filename>
    """
    rel = rel.replace("\\", "/")
    p1 = MEVID_ROOT / rel
    if p1.exists():
        return p1

    parts = rel.split("/")
    if parts and parts[0].startswith("bbox_"):
        split_dir = parts[0]            # e.g., 'bbox_test'
        fname = parts[-1]               # e.g., '0205O003C639T024F00000.jpg'
        if len(fname) >= 4:
            pid4 = fname[:4]            # '0205'
            p2 = MEVID_ROOT / split_dir / pid4 / fname
            if p2.exists():
                return p2

    return p1  # fallback

def main():
    meta = pickle.load(open(ARTIFACTS / "meta_test.pkl", "rb"))
    out_dir = THUMBS_DIR
    out_dir.mkdir(exist_ok=True)

    made = 0
    skipped_empty = 0

    for i, m in enumerate(tqdm(meta, desc="thumbs")):
        gif_out = out_dir / f"track_{i}.gif"
        if gif_out.exists():
            continue

        frames = [resolve_rel_path(p) for p in m["frames"]]
        frames = [p for p in frames if p.exists()]
        frames = evenly_sample(frames, FRAMES_PER_TRACKLET)

        if not frames:
            skipped_empty += 1
            continue

        ok = make_gif_from_images(frames, gif_out, max_frames=FRAMES_PER_TRACKLET, fps=6, max_side=320)
        if ok:
            made += 1

    print(f"[thumbs] created: {made}, skipped (no frames): {skipped_empty}")

if __name__ == "__main__":
    main()
