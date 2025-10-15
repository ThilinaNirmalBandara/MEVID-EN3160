
import sys, os
from pathlib import Path
import json

# ensure we can import config + package when running as: python scripts/parse_mevid.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pickle
from config import ARTIFACTS
from mevid_textsearch.faiss_utils import load_index
from mevid_textsearch.clip_utils import encode_text

def rerank_diversity(ids, sims, meta, penalty=0.03):
    """Penalize repeated cameras to increase top-k variety."""
    seen = set()
    scored = []
    for idx, s in zip(ids, sims):
        cam = meta[idx]["camid"]
        s2 = s - (penalty if cam in seen else 0.0)
        scored.append((idx, s2))
        seen.add(cam)
    scored.sort(key=lambda x: x[1], reverse=True)
    new_ids  = [i for i, _ in scored]
    new_sims = [s for _, s in scored]
    return new_ids, new_sims

def search(query: str, topk=10, diversify=True):
    index = load_index(ARTIFACTS / "faiss.index")
    meta = pickle.load(open(ARTIFACTS / "meta_test.pkl", "rb"))
    qvec = encode_text([query])  # (1,512) L2-normalized
    sims, ids = index.search(qvec, topk)
    ids, sims = ids[0].tolist(), sims[0].tolist()
    if diversify:
        ids, sims = rerank_diversity(ids, sims, meta, penalty=0.03)
    return ids, sims, meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True, help="text query")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--no_diversify", action="store_true", help="disable camera diversity re-rank")
    args = ap.parse_args()

    ids, sims, meta = search(args.q, args.k, diversify=(not args.no_diversify))
    print(f"Query: {args.q}\n")
    for rank, (i, s) in enumerate(zip(ids, sims), 1):
        m = meta[i]
        print(f"#{rank}  score={s:.3f}  tid={m['tid']}  pid={m['pid']}  outfit={m['outfit']}  cam={m['camid']}")
        print(f"     sample frames: {m['frames'][:3]}")
    print()

if __name__ == "__main__":
    main()
