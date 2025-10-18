
import sys, os
from pathlib import Path
import json

# ensure we can import config + package when running as: python scripts/parse_mevid.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# scripts/extract_vecs.py
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json
import pickle
from config import MEVID_ROOT, ARTIFACTS, FRAMES_PER_TRACKLET
from mevid_textsearch.clip_utils import encode_frame, pool_mean

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
       where pid4 = first 4 chars of the filename (e.g., '0232')
    """
    rel = rel.replace("\\", "/")
    p1 = MEVID_ROOT / rel
    if p1.exists():
        return p1

    parts = rel.split("/")
    if parts and parts[0].startswith("bbox_"):
        split_dir = parts[0]                   # e.g., 'bbox_test'
        fname = parts[-1]                      # e.g., '0232O004C...jpg'
        if len(fname) >= 4:
            pid4 = fname[:4]                   # '0232'
            p2 = MEVID_ROOT / split_dir / pid4 / fname
            if p2.exists():
                return p2

    # fallback (likely missing, but return something)
    return p1

def main():
    tracks_jsonl = ARTIFACTS / "tracks_test.jsonl"
    assert tracks_jsonl.exists(), "Run parse_mevid.py first"
    lines = [json.loads(x) for x in tracks_jsonl.read_text().splitlines()]
    print(f"[extract] loaded {len(lines)} test tracks")

    vecs = []
    meta = []
    total_frames = 0
    found_frames = 0

    for i, t in enumerate(tqdm(lines, desc="encoding tracks")):
        frames_rel = t["frames"]
        frames = [resolve_rel_path(p) for p in frames_rel]
        frames = evenly_sample(frames, FRAMES_PER_TRACKLET)

        fvecs = []
        for fp in frames:
            total_frames += 1
            if fp.exists():
                found_frames += 1
                fvecs.append(encode_frame(fp))
        if fvecs:
            v = pool_mean(fvecs).astype(np.float32)
        else:
            v = np.zeros((512,), dtype=np.float32)

        vecs.append(v)
        meta.append({
            "tid": t["tid"],
            "pid": t["pid"],
            "outfit": t["outfit"],
            "camid": t["camid"],
            "frames": frames_rel[:20]  # keep the original rels for display
        })

    X = np.stack(vecs).astype(np.float32)
    np.save(ARTIFACTS / "vecs_test.npy", X)
    with open(ARTIFACTS / "meta_test.pkl", "wb") as f:
        pickle.dump(meta, f)

    zero_rows = int((np.linalg.norm(X, axis=1) == 0).sum())
    print(f"[extract] saved vecs: {X.shape}, meta: {len(meta)}")
    print(f"[extract] frames found: {found_frames}/{total_frames}")
    print(f"[extract] zero vectors: {zero_rows}/{X.shape[0]}")

if __name__ == "__main__":
    main()
