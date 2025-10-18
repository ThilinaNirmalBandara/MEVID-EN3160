# scripts/parse_mevid.py
import sys, os
from pathlib import Path
import json

# ensure we can import config + package when running as: python scripts/parse_mevid.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MEVID_ROOT, ARTIFACTS
from mevid_textsearch.mevid import build_tracklets, save_jsonl

def f2i(x: str) -> int:
    """Robust float->int for lines like '0.000000000000000000e+00' etc."""
    return int(float(x))

def main():
    assert MEVID_ROOT.exists(), f"MEVID_ROOT not found: {MEVID_ROOT}"

    # 1) Build test tracklets
    test_tracks = build_tracklets(MEVID_ROOT, split="test")
    out_jsonl = ARTIFACTS / "tracks_test.jsonl"
    print(f"[parse] saving {len(test_tracks)} test tracks -> {out_jsonl}")
    save_jsonl([t.__dict__ for t in test_tracks], out_jsonl)

    # 2) Parse query indices (robust to scientific notation / commas / tabs)
    query_idx_path = MEVID_ROOT / "query_IDX.txt"
    assert query_idx_path.exists(), f"Missing: {query_idx_path}"
    raw_lines = [ln.strip() for ln in query_idx_path.read_text().splitlines() if ln.strip()]

    qidx = []
    for ln in raw_lines:
        # take first token, tolerate commas/tabs/multiple spaces
        tok = ln.replace(",", " ").split()[0]
        qidx.append(f2i(tok))

    (ARTIFACTS / "query_idx.json").write_text(json.dumps(qidx))
    print(f"[parse] saved {len(qidx)} query indices -> {ARTIFACTS / 'query_idx.json'}")

    # (optional) quick range sanity check
    n_tracks = len(test_tracks)
    bad = [i for i in qidx if not (0 <= i < n_tracks)]
    if bad:
        print(f"[warn] {len(bad)} query indices out of range [0,{n_tracks-1}]. First few: {bad[:10]}")

    print("[parse] done.")

if __name__ == "__main__":
    main()
