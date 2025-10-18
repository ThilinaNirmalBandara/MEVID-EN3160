# scripts/eval_reid_video2video.py
# This evaluates tracklet→tracklet retrieval (no text) as a sanity check.
import json, pickle
import numpy as np
from pathlib import Path
from config import ARTIFACTS
from sklearn.metrics import average_precision_score
from tqdm import tqdm

def main():
    X = np.load(ARTIFACTS / "vecs_test.npy")                # (N,512) L2-normalized
    with open(ARTIFACTS / "meta_test.pkl", "rb") as f:
        meta = pickle.load(f)
    qidx = json.loads((ARTIFACTS / "query_idx.json").read_text())
    N = X.shape[0]
    sims = X @ X.T   # cosine similarities

    # query set from qidx, gallery = everyone else
    pids = np.array([m["pid"] for m in meta])

    recalls = {1:0,5:0,10:0}
    aps = []
    for qi in tqdm(qidx, desc="eval"):
        qpid = pids[qi]
        # build labels for all gallery (exclude self)
        mask = np.ones(N, dtype=bool); mask[qi] = False
        gal_idx = np.where(mask)[0]
        y_true = (pids[gal_idx] == qpid).astype(np.int32)
        y_score = sims[qi, gal_idx]

        # ranks
        order = np.argsort(-y_score)
        y_true_sorted = y_true[order]
        # recall@k
        for k in recalls.keys():
            recalls[k] += int(y_true_sorted[:k].sum() > 0)

        # AP
        aps.append(average_precision_score(y_true, y_score))

    M = len(qidx)
    print("\nVideo→Video retrieval:")
    for k in sorted(recalls.keys()):
        print(f"R@{k}: {recalls[k]/M:.4f}")
    print(f"mAP: {np.mean(aps):.4f}")

if __name__ == "__main__":
    main()
